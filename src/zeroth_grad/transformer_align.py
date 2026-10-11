"""Align and merge transformers through their weight-space symmetries.

Alignment is weight matching by coordinate descent, in the style of Git
Re-Basin, extended to the attention and residual symmetries in
`transformer.py`:

1. Residual stream. One basis for the whole network. Each residual coordinate
   is described by everything that reads or writes it (embeddings, positions,
   unembedding, norm parameters, and, once the blocks are aligned, every
   query/key/value/MLP read and every output/MLP write). The source basis is
   matched to the reference with one assignment (permutation) or one
   orthogonal Procrustes solve (rotation, RMSNorm with folded gains only).
   Applying it everywhere keeps the residual stream a single global
   coordinate system: no layer is moved without the others. The residual
   scale is matched as well, as the least-squares ratio of the two models'
   writes, since the norms make every read scale-free.
2. Heads. With the residual fixed, each head is described by its QK circuit
   W_Q^T W_K and its OV circuit W_O W_V, d x d matrices that do not depend on
   the head's internal basis. Heads are matched on the cosine of both.
3. Head internals. Inside a matched pair, the query/key dimensions are matched
   on [W_Q | b_Q | W_K | b_K] and the value dimensions on [W_V | b_V | W_O^T],
   by permutation or by Procrustes rotation. Attention scores and the head's
   output are unchanged by either.
4. MLP units, matched on [W_in | b_in | W_out^T].

Steps 1-4 repeat, so the residual estimate uses the aligned blocks.
"""

from __future__ import annotations

import torch

from zeroth_grad.solvers import solve_assignment
from zeroth_grad.transformer import (
    TransformerConfig,
    clone,
    fold_norm_gains,
    permutation_matrix,
    permute_heads,
    permute_mlp,
    permute_residual,
    rotate_residual,
    scale_residual,
    transform_head,
)


def _procrustes(reference: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    """Orthogonal R minimizing ||reference - R source||_F (rows are the rotated axis)."""
    u, _, vh = torch.linalg.svd(reference.double() @ source.double().T)
    return (u @ vh).float()


def _match(reference: torch.Tensor, source: torch.Tensor, solver: str) -> torch.Tensor:
    """perm with source[perm[i]] best matching reference[i], by maximum inner product."""
    cost = -(reference.double() @ source.double().T)
    return solve_assignment(cost.float(), solver)


def residual_features(
    w: dict[str, torch.Tensor],
    cfg: TransformerConfig,
    blocks: bool,
    norms: bool = True,
) -> torch.Tensor:
    """(d_model, features): every quantity indexed by a residual coordinate, as columns.

    Norm gains and biases permute with the stream but do not rotate with it,
    so `norms=False` leaves them out for Procrustes.
    """
    cols = [w["embed"].T, w["pos"].T, w["unembed"].T]
    if norms:
        for key in sorted(k for k in w if k.endswith(".g") or k.endswith(".b")):
            if w[key].shape == (cfg.d_model,):
                cols.append(w[key][:, None])
    if blocks:
        for layer in range(cfg.n_layers):
            p = f"l{layer}"
            for name in ("wq", "wk", "wv"):
                cols.append(w[f"{p}.{name}"].reshape(-1, cfg.d_model).T)
            cols.append(w[f"{p}.wo"].permute(1, 0, 2).reshape(cfg.d_model, -1))
            cols.append(w[f"{p}.bo"][:, None])
            cols.append(w[f"{p}.w_in"].T)
            cols.append(w[f"{p}.w_out"])
            cols.append(w[f"{p}.b_out"][:, None])
    return torch.cat(cols, dim=1)


def write_features(w: dict[str, torch.Tensor], cfg: TransformerConfig, blocks: bool) -> torch.Tensor:
    """(d_model, features): everything that writes into the residual stream."""
    cols = [w["embed"].T, w["pos"].T]
    if blocks:
        for layer in range(cfg.n_layers):
            p = f"l{layer}"
            cols += [w[f"{p}.wo"].permute(1, 0, 2).reshape(cfg.d_model, -1), w[f"{p}.bo"][:, None], w[f"{p}.w_out"], w[f"{p}.b_out"][:, None]]
    return torch.cat(cols, dim=1)


def _circuits(w: dict[str, torch.Tensor], layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    p = f"l{layer}"
    qk = torch.einsum("hed,hef->hdf", w[f"{p}.wq"], w[f"{p}.wk"]).flatten(1)
    ov = torch.einsum("hde,hef->hdf", w[f"{p}.wo"], w[f"{p}.wv"]).flatten(1)
    return qk, ov


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)


def align_transformer(
    reference: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    cfg: TransformerConfig,
    rotations: bool = False,
    solver: str = "hungarian",
    rounds: int = 3,
    scale: bool = True,
) -> dict[str, torch.Tensor]:
    """Move `source` along its symmetry group to line up with `reference`. Its function is unchanged.

    With `rotations`, head internals are aligned by orthogonal Procrustes, and
    so is the residual stream when the model uses RMSNorm (both models are
    gain-folded first). Otherwise every map is a permutation. With `scale`,
    the residual scale is matched too: every write is multiplied by the
    least-squares ratio of the reference's writes to the source's.
    """
    rotate_stream = rotations and cfg.norm == "rms"
    ref = fold_norm_gains(reference, cfg) if rotate_stream else clone(reference)
    out = fold_norm_gains(source, cfg) if rotate_stream else clone(source)
    for round_index in range(rounds):
        blocks = round_index > 0
        f_ref = residual_features(ref, cfg, blocks, norms=not rotate_stream)
        f_src = residual_features(out, cfg, blocks, norms=not rotate_stream)
        if rotate_stream:
            out = rotate_residual(out, cfg, _procrustes(f_ref, f_src))
        else:
            out = permute_residual(out, cfg, _match(f_ref, f_src, solver))
        if scale:
            w_ref = write_features(ref, cfg, blocks)
            w_src = write_features(out, cfg, blocks)
            ratio = float((w_ref * w_src).sum() / w_src.pow(2).sum().clamp_min(1e-12))
            if ratio > 0:
                out = scale_residual(out, cfg, ratio)
        for layer in range(cfg.n_layers):
            out = _align_heads(ref, out, cfg, layer, rotations, solver)
            out = _align_mlp(ref, out, cfg, layer, solver)
    return out


def _align_heads(ref, out, cfg, layer, rotations, solver):
    qk_r, ov_r = _circuits(ref, layer)
    qk_s, ov_s = _circuits(out, layer)
    similarity = _unit(qk_r) @ _unit(qk_s).T + _unit(ov_r) @ _unit(ov_s).T
    out = permute_heads(out, cfg, layer, solve_assignment(-similarity, solver))
    p = f"l{layer}"
    for head in range(cfg.n_heads):
        a_ref = torch.cat([ref[f"{p}.wq"][head], ref[f"{p}.bq"][head][:, None], ref[f"{p}.wk"][head], ref[f"{p}.bk"][head][:, None]], dim=1)
        a_src = torch.cat([out[f"{p}.wq"][head], out[f"{p}.bq"][head][:, None], out[f"{p}.wk"][head], out[f"{p}.bk"][head][:, None]], dim=1)
        b_ref = torch.cat([ref[f"{p}.wv"][head], ref[f"{p}.bv"][head][:, None], ref[f"{p}.wo"][head].T], dim=1)
        b_src = torch.cat([out[f"{p}.wv"][head], out[f"{p}.bv"][head][:, None], out[f"{p}.wo"][head].T], dim=1)
        if rotations:
            qk, vo = _procrustes(a_ref, a_src), _procrustes(b_ref, b_src)
        else:
            qk = permutation_matrix(_match(a_ref, a_src, solver))
            vo = permutation_matrix(_match(b_ref, b_src, solver))
        out = transform_head(out, cfg, layer, head, qk, vo)
    return out


def _align_mlp(ref, out, cfg, layer, solver):
    p = f"l{layer}"
    f_ref = torch.cat([ref[f"{p}.w_in"], ref[f"{p}.b_in"][:, None], ref[f"{p}.w_out"].T], dim=1)
    f_src = torch.cat([out[f"{p}.w_in"], out[f"{p}.b_in"][:, None], out[f"{p}.w_out"].T], dim=1)
    return permute_mlp(out, cfg, layer, _match(f_ref, f_src, solver))


def weight_distance(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    """Relative L2 distance between two state dicts with the same keys."""
    num = sum(float((a[k] - b[k]).pow(2).sum()) for k in a)
    den = sum(float(a[k].pow(2).sum()) for k in a)
    return (num / max(den, 1e-12)) ** 0.5


def merge_transformers(
    models: list[dict[str, torch.Tensor]],
    cfg: TransformerConfig,
    base: dict[str, torch.Tensor] | None = None,
    scale: float = 1.0,
    align: bool = True,
    rotations: bool = False,
    solver: str = "hungarian",
) -> dict[str, torch.Tensor]:
    """Merge transformers in the residual frame of `models[0]`.

    Without `base`, the merge is the mean of the aligned models. With `base`
    (a common ancestor), it is task arithmetic, base + scale * sum_m (m - base),
    with every model and the base aligned to models[0] first.
    """
    if not models:
        raise ValueError("merge_transformers needs at least one model")
    rotate_stream = rotations and cfg.norm == "rms"
    prepare = (lambda w: fold_norm_gains(w, cfg)) if rotate_stream else clone
    reference = prepare(models[0])
    if align:
        aligned = [reference] + [align_transformer(reference, m, cfg, rotations=rotations, solver=solver) for m in models[1:]]
        anchor = align_transformer(reference, base, cfg, rotations=rotations, solver=solver) if base is not None else None
    else:
        aligned = [prepare(m) for m in models]
        anchor = prepare(base) if base is not None else None
    if anchor is None:
        return {k: torch.stack([m[k] for m in aligned]).mean(dim=0) for k in reference}
    return {k: anchor[k] + scale * sum(m[k] - anchor[k] for m in aligned) for k in reference}
