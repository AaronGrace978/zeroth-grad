"""A small pre-norm transformer in an explicit state dict, and its weight-space symmetries.

The residual stream is one coordinate system shared by every layer: the
embeddings write into it, and every attention head and MLP reads from it and
writes back. A change of residual basis therefore has to be applied to every
read and every write at once. The symmetries that leave the function unchanged
are:

- residual permutation P, for both norms (LayerNorm and RMSNorm commute with
  a permutation of features when their gains and biases are permuted too);
- residual scale c > 0 on every write, since the norms divide it out;
- residual rotation Q, for RMSNorm once its gains are folded into the next
  read (RMSNorm(Qr) = Q RMSNorm(r) for orthogonal Q; LayerNorm's mean
  subtraction breaks this);
- head order within a layer;
- a shared orthogonal map R on the query and key dimensions of a head
  (q.k = (Rq).(Rk)), and another on its value dimensions, undone on the
  output projection; permutations are the special case;
- MLP hidden units, exactly as in the two-layer MLP.

Every function here takes and returns a dict of tensors; `forward` runs the
model from one, so the edits can be checked against the function directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TransformerConfig:
    vocab: int = 18
    n_ctx: int = 9
    d_model: int = 32
    n_layers: int = 2
    n_heads: int = 4
    d_head: int = 8
    d_mlp: int = 64
    d_out: int = 16
    norm: str = "layer"

    def to_dict(self) -> dict:
        return asdict(self)


def _norm_keys(prefix: str, cfg: TransformerConfig) -> list[str]:
    return [f"{prefix}.g", f"{prefix}.b"] if cfg.norm == "layer" else [f"{prefix}.g"]


def init_transformer(cfg: TransformerConfig, seed: int = 0) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)

    def normal(*shape: int, std: float) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * std

    d, h, dh, m = cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_mlp
    w = {"embed": normal(cfg.vocab, d, std=1.0), "pos": normal(cfg.n_ctx, d, std=0.5)}
    for layer in range(cfg.n_layers):
        p = f"l{layer}"
        for prefix in (f"{p}.ln1", f"{p}.ln2"):
            w[f"{prefix}.g"] = torch.ones(d)
            if cfg.norm == "layer":
                w[f"{prefix}.b"] = torch.zeros(d)
        for name in ("wq", "wk", "wv"):
            w[f"{p}.{name}"] = normal(h, dh, d, std=d**-0.5)
        for name in ("bq", "bk", "bv"):
            w[f"{p}.{name}"] = torch.zeros(h, dh)
        w[f"{p}.wo"] = normal(h, d, dh, std=(h * dh) ** -0.5)
        w[f"{p}.bo"] = torch.zeros(d)
        w[f"{p}.w_in"] = normal(m, d, std=d**-0.5)
        w[f"{p}.b_in"] = torch.zeros(m)
        w[f"{p}.w_out"] = normal(d, m, std=m**-0.5)
        w[f"{p}.b_out"] = torch.zeros(d)
    w["lnf.g"] = torch.ones(d)
    if cfg.norm == "layer":
        w["lnf.b"] = torch.zeros(d)
    w["unembed"] = normal(cfg.d_out, d, std=d**-0.5)
    w["b_unembed"] = torch.zeros(cfg.d_out)
    return w


def _normalize(x: torch.Tensor, w: dict[str, torch.Tensor], prefix: str, cfg: TransformerConfig) -> torch.Tensor:
    if cfg.norm == "layer":
        return F.layer_norm(x, (x.shape[-1],), w[f"{prefix}.g"], w[f"{prefix}.b"], eps=1e-5)
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * w[f"{prefix}.g"]


def forward(weights: dict[str, torch.Tensor], cfg: TransformerConfig, tokens: torch.Tensor) -> torch.Tensor:
    """Causal forward. tokens (B, T) -> logits (B, T, d_out)."""
    t = tokens.shape[1]
    x = weights["embed"][tokens] + weights["pos"][:t]
    mask = torch.ones(t, t, dtype=torch.bool, device=tokens.device).tril()
    for layer in range(cfg.n_layers):
        p = f"l{layer}"
        a = _normalize(x, weights, f"{p}.ln1", cfg)
        q = torch.einsum("btd,hed->bhte", a, weights[f"{p}.wq"]) + weights[f"{p}.bq"][None, :, None, :]
        k = torch.einsum("btd,hed->bhte", a, weights[f"{p}.wk"]) + weights[f"{p}.bk"][None, :, None, :]
        v = torch.einsum("btd,hed->bhte", a, weights[f"{p}.wv"]) + weights[f"{p}.bv"][None, :, None, :]
        scores = torch.einsum("bhte,bhse->bhts", q, k) / cfg.d_head**0.5
        scores = scores.masked_fill(~mask, float("-inf"))
        mixed = torch.einsum("bhts,bhse->bhte", scores.softmax(dim=-1), v)
        x = x + torch.einsum("bhte,hde->btd", mixed, weights[f"{p}.wo"]) + weights[f"{p}.bo"]
        b = _normalize(x, weights, f"{p}.ln2", cfg)
        hidden = F.gelu(b @ weights[f"{p}.w_in"].T + weights[f"{p}.b_in"])
        x = x + hidden @ weights[f"{p}.w_out"].T + weights[f"{p}.b_out"]
    x = _normalize(x, weights, "lnf", cfg)
    return x @ weights["unembed"].T + weights["b_unembed"]


def clone(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in weights.items()}


def _reads(cfg: TransformerConfig, layer: int) -> list[str]:
    p = f"l{layer}"
    return [f"{p}.wq", f"{p}.wk", f"{p}.wv", f"{p}.w_in"]


def fold_norm_gains(weights: dict[str, torch.Tensor], cfg: TransformerConfig) -> dict[str, torch.Tensor]:
    """Absorb each norm's gain (and LayerNorm bias) into the weights that read its output.

    Afterward every gain is 1 and every norm bias is 0, and the function is
    unchanged. For RMSNorm this makes the residual stream rotation-equivariant.
    """
    w = clone(weights)
    for layer in range(cfg.n_layers):
        p = f"l{layer}"
        g1 = w[f"{p}.ln1.g"]
        for name, bias in (("wq", "bq"), ("wk", "bk"), ("wv", "bv")):
            if cfg.norm == "layer":
                w[f"{p}.{bias}"] = w[f"{p}.{bias}"] + torch.einsum("hed,d->he", w[f"{p}.{name}"], w[f"{p}.ln1.b"])
            w[f"{p}.{name}"] = w[f"{p}.{name}"] * g1
        g2 = w[f"{p}.ln2.g"]
        if cfg.norm == "layer":
            w[f"{p}.b_in"] = w[f"{p}.b_in"] + w[f"{p}.w_in"] @ w[f"{p}.ln2.b"]
            w[f"{p}.ln1.b"] = torch.zeros_like(w[f"{p}.ln1.b"])
            w[f"{p}.ln2.b"] = torch.zeros_like(w[f"{p}.ln2.b"])
        w[f"{p}.w_in"] = w[f"{p}.w_in"] * g2
        w[f"{p}.ln1.g"] = torch.ones_like(g1)
        w[f"{p}.ln2.g"] = torch.ones_like(g2)
    if cfg.norm == "layer":
        w["b_unembed"] = w["b_unembed"] + w["unembed"] @ w["lnf.b"]
        w["lnf.b"] = torch.zeros_like(w["lnf.b"])
    w["unembed"] = w["unembed"] * w["lnf.g"]
    w["lnf.g"] = torch.ones_like(w["lnf.g"])
    return w


def permute_residual(weights: dict[str, torch.Tensor], cfg: TransformerConfig, perm: torch.Tensor) -> dict[str, torch.Tensor]:
    """New residual coordinate i is old coordinate perm[i], in every read, write, and norm."""
    w = clone(weights)
    w["embed"] = w["embed"][:, perm]
    w["pos"] = w["pos"][:, perm]
    for layer in range(cfg.n_layers):
        p = f"l{layer}"
        for key in _norm_keys(f"{p}.ln1", cfg) + _norm_keys(f"{p}.ln2", cfg):
            w[key] = w[key][perm]
        for name in ("wq", "wk", "wv"):
            w[f"{p}.{name}"] = w[f"{p}.{name}"][:, :, perm]
        w[f"{p}.wo"] = w[f"{p}.wo"][:, perm, :]
        w[f"{p}.bo"] = w[f"{p}.bo"][perm]
        w[f"{p}.w_in"] = w[f"{p}.w_in"][:, perm]
        w[f"{p}.w_out"] = w[f"{p}.w_out"][perm]
        w[f"{p}.b_out"] = w[f"{p}.b_out"][perm]
    for key in _norm_keys("lnf", cfg):
        w[key] = w[key][perm]
    w["unembed"] = w["unembed"][:, perm]
    return w


def _check_rotatable(weights: dict[str, torch.Tensor], cfg: TransformerConfig) -> None:
    if cfg.norm != "rms":
        raise ValueError("residual rotations are a symmetry only with RMSNorm")
    for key, value in weights.items():
        if key.endswith(".g") and not torch.allclose(value, torch.ones_like(value)):
            raise ValueError("fold_norm_gains before rotating the residual stream")


def rotate_residual(weights: dict[str, torch.Tensor], cfg: TransformerConfig, q: torch.Tensor) -> dict[str, torch.Tensor]:
    """Map the residual stream r -> Q r for orthogonal Q. Needs RMSNorm with folded gains."""
    _check_rotatable(weights, cfg)
    w = clone(weights)
    w["embed"] = w["embed"] @ q.T
    w["pos"] = w["pos"] @ q.T
    for layer in range(cfg.n_layers):
        p = f"l{layer}"
        for name in ("wq", "wk", "wv"):
            w[f"{p}.{name}"] = w[f"{p}.{name}"] @ q.T
        w[f"{p}.wo"] = torch.einsum("ij,hje->hie", q, w[f"{p}.wo"])
        w[f"{p}.bo"] = q @ w[f"{p}.bo"]
        w[f"{p}.w_in"] = w[f"{p}.w_in"] @ q.T
        w[f"{p}.w_out"] = q @ w[f"{p}.w_out"]
        w[f"{p}.b_out"] = q @ w[f"{p}.b_out"]
    w["unembed"] = w["unembed"] @ q.T
    return w


def scale_residual(weights: dict[str, torch.Tensor], cfg: TransformerConfig, c: float) -> dict[str, torch.Tensor]:
    """Multiply every write into the residual stream by c > 0. Reads go through a norm, so they do not change."""
    if c <= 0:
        raise ValueError("residual scale must be positive")
    w = clone(weights)
    for key in ("embed", "pos"):
        w[key] = w[key] * c
    for layer in range(cfg.n_layers):
        p = f"l{layer}"
        for key in ("wo", "bo", "w_out", "b_out"):
            w[f"{p}.{key}"] = w[f"{p}.{key}"] * c
    return w


def permute_heads(weights: dict[str, torch.Tensor], cfg: TransformerConfig, layer: int, perm: torch.Tensor) -> dict[str, torch.Tensor]:
    w = clone(weights)
    p = f"l{layer}"
    for name in ("wq", "wk", "wv", "bq", "bk", "bv", "wo"):
        w[f"{p}.{name}"] = w[f"{p}.{name}"][perm]
    return w


def transform_head(
    weights: dict[str, torch.Tensor],
    cfg: TransformerConfig,
    layer: int,
    head: int,
    qk: torch.Tensor | None = None,
    vo: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Apply orthogonal maps inside one head: q, k -> R_qk q, R_qk k and v -> R_vo v with W_O R_vo^T.

    Scores q.k and the head's output W_O v are unchanged. Permutation matrices
    are the special case used by permutation alignment.
    """
    w = clone(weights)
    p = f"l{layer}"
    if qk is not None:
        for name, bias in (("wq", "bq"), ("wk", "bk")):
            w[f"{p}.{name}"][head] = qk @ w[f"{p}.{name}"][head]
            w[f"{p}.{bias}"][head] = qk @ w[f"{p}.{bias}"][head]
    if vo is not None:
        w[f"{p}.wv"][head] = vo @ w[f"{p}.wv"][head]
        w[f"{p}.bv"][head] = vo @ w[f"{p}.bv"][head]
        w[f"{p}.wo"][head] = w[f"{p}.wo"][head] @ vo.T
    return w


def permute_mlp(weights: dict[str, torch.Tensor], cfg: TransformerConfig, layer: int, perm: torch.Tensor) -> dict[str, torch.Tensor]:
    w = clone(weights)
    p = f"l{layer}"
    w[f"{p}.w_in"] = w[f"{p}.w_in"][perm]
    w[f"{p}.b_in"] = w[f"{p}.b_in"][perm]
    w[f"{p}.w_out"] = w[f"{p}.w_out"][:, perm]
    return w


def permutation_matrix(perm: torch.Tensor) -> torch.Tensor:
    """Matrix M with (M x)[i] = x[perm[i]]."""
    n = perm.numel()
    m = torch.zeros(n, n)
    m[torch.arange(n), perm] = 1.0
    return m


def random_symmetry(
    weights: dict[str, torch.Tensor],
    cfg: TransformerConfig,
    seed: int = 0,
    rotate: bool = False,
    scale: float | None = None,
) -> dict[str, torch.Tensor]:
    """Apply a random element of the symmetry group. The function does not change.

    Residual permutation, head order, query/key and value/output bases (random
    rotations if `rotate`, permutations otherwise), MLP order, and, with RMSNorm
    and `rotate`, a random residual rotation. `scale` multiplies residual writes.
    """
    generator = torch.Generator().manual_seed(seed)
    w = clone(weights)
    if rotate and cfg.norm == "rms":
        w = fold_norm_gains(w, cfg)
        w = rotate_residual(w, cfg, random_orthogonal(cfg.d_model, generator))
    else:
        w = permute_residual(w, cfg, torch.randperm(cfg.d_model, generator=generator))
    for layer in range(cfg.n_layers):
        w = permute_heads(w, cfg, layer, torch.randperm(cfg.n_heads, generator=generator))
        for head in range(cfg.n_heads):
            if rotate:
                qk = random_orthogonal(cfg.d_head, generator)
                vo = random_orthogonal(cfg.d_head, generator)
            else:
                qk = permutation_matrix(torch.randperm(cfg.d_head, generator=generator))
                vo = permutation_matrix(torch.randperm(cfg.d_head, generator=generator))
            w = transform_head(w, cfg, layer, head, qk, vo)
        w = permute_mlp(w, cfg, layer, torch.randperm(cfg.d_mlp, generator=generator))
    if scale is not None:
        w = scale_residual(w, cfg, scale)
    return w


def random_orthogonal(n: int, generator: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(n, n, generator=generator))
    return q * torch.sign(torch.diagonal(r))[None, :]
