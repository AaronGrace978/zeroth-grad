"""Fit the compiler with a functional loss. The checkpoint is an input, not an optimizer state."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from zeroth_grad.align import permute_h1, permute_h2
from zeroth_grad.compiler import WeightCompiler
from zeroth_grad.data import skill_labels
from zeroth_grad.edit import expand_h1, expand_h2, mix_checkpoints
from zeroth_grad.mlp import functional_mlp
from zeroth_grad.zoo import Record, load_concepts, load_zoo


def drop_inactive(weights: dict[str, torch.Tensor], eps: float = 1e-8) -> dict[str, torch.Tensor]:
    """Remove neurons whose outgoing weights are zero. The function is unchanged."""
    keep1 = (weights["w2"].norm(dim=0) > eps).nonzero(as_tuple=False).flatten()
    keep2 = (weights["w3"].norm(dim=0) > eps).nonzero(as_tuple=False).flatten()
    if keep1.numel() == 0:
        keep1 = torch.arange(weights["w1"].shape[0])
    if keep2.numel() == 0:
        keep2 = torch.arange(weights["w2"].shape[0])
    return {
        "w1": weights["w1"][keep1],
        "b1": weights["b1"][keep1],
        "w2": weights["w2"][keep2][:, keep1],
        "b2": weights["b2"][keep2],
        "w3": weights["w3"][:, keep2],
        "b3": weights["b3"],
    }


def _bce_on(weights: dict[str, torch.Tensor], x: torch.Tensor, concepts: torch.Tensor, head: int) -> torch.Tensor:
    logits = functional_mlp(weights, x)[:, head]
    y = skill_labels(x, concepts[head])
    return F.binary_cross_entropy_with_logits(logits, y)


def duplication_task(
    weights: dict[str, torch.Tensor],
    generator: torch.Generator,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """Split every live neuron into four copies and shuffle them.

    The keep-budget is the live width. Magnitude pruning keeps the largest
    fragments and drops the rest. Cluster collapse restores each survivor.
    """
    compact = drop_inactive(weights)
    h1 = int(compact["w1"].shape[0])
    h2 = int(compact["w2"].shape[0])
    # Four-way split of every live neuron. Equal fragments of a large neuron
    # outrank the fragments of a small one, so magnitude pruning keeps the
    # large neuron and drops the rest.
    wide = expand_h1(compact, [index for index in range(h1) for _ in range(3)])
    wide = expand_h2(wide, [index for index in range(h2) for _ in range(3)])
    perm1 = torch.randperm(wide["w1"].shape[0], generator=generator)
    perm2 = torch.randperm(wide["w2"].shape[0], generator=generator)
    return permute_h2(permute_h1(wide, perm1), perm2), h1, h2


def _residual_norm(module: torch.nn.Linear, features: torch.Tensor) -> torch.Tensor:
    return module(features).squeeze(-1).pow(2).mean()


def train_compiler(
    compiler: WeightCompiler,
    records: list[Record],
    concepts: torch.Tensor,
    steps: int = 600,
    batch_size: int = 256,
    seed: int = 0,
    lr: float = 1e-3,
    scale_lr: float = 2e-2,
) -> list[float]:
    train_records = [r for r in records if r.split == "train" and r.kind == "single"]
    if len(train_records) < 2:
        raise ValueError("need at least two train checkpoints")
    scale_ids = {id(p) for p in compiler.scale_parameters()}
    other = [p for p in compiler.parameters() if id(p) not in scale_ids]
    opt = torch.optim.Adam(
        [
            {"params": other, "lr": lr},
            {"params": compiler.scale_parameters(), "lr": scale_lr},
        ]
    )
    generator = torch.Generator().manual_seed(seed)
    losses: list[float] = []
    compiler.train()
    for step in range(steps):
        op_draw = torch.rand(1, generator=generator).item()
        if op_draw < 0.5:
            op = "inject"
        elif op_draw < 0.75:
            op = "compress"
        else:
            op = "merge"
        x = torch.randn(batch_size, concepts.shape[1], generator=generator)
        opt.zero_grad(set_to_none=True)
        if op == "inject":
            record = train_records[int(torch.randint(len(train_records), (1,), generator=generator).item())]
            src = record.skills[0]
            choices = [s for s in range(6) if s != src]
            tgt = choices[int(torch.randint(len(choices), (1,), generator=generator).item())]
            edited = compiler.inject(record.weights, concepts[tgt], tgt)
            f1, f2 = compiler.encode(record.weights, task=1, phi=concepts[tgt], target_head=tgt)
            reg = (
                _residual_norm(compiler.a_head, f1)
                + _residual_norm(compiler.s_head, f1)
                + _residual_norm(compiler.c_head, f2)
                + _residual_norm(compiler.r_head, f2)
            )
            loss = _bce_on(edited, x, concepts, tgt) + _bce_on(edited, x, concepts, src) + 1e-3 * reg
        elif op == "compress":
            record = train_records[int(torch.randint(len(train_records), (1,), generator=generator).item())]
            wide, k1, k2 = duplication_task(record.weights, generator)
            edited = compiler.compress(wide, k1, k2)
            f1, f2 = compiler.encode(wide, task=0)
            reg = _residual_norm(compiler.keep1, f1) + _residual_norm(compiler.keep2, f2)
            loss = _bce_on(edited, x, concepts, record.skills[0]) + 1e-2 * reg
        else:
            left = train_records[int(torch.randint(len(train_records), (1,), generator=generator).item())]
            right_choices = [r for r in train_records if r.skills[0] != left.skills[0]]
            right = right_choices[int(torch.randint(len(right_choices), (1,), generator=generator).item())]
            lam1, lam2, lam3, aligned = compiler.merge_lambdas(left.weights, right.weights)
            edited = mix_checkpoints(left.weights, aligned, lam1, lam2, lam3)
            # Keep the residual heads small so the geometric prior stays the default.
            pair1, pair2, pair3 = compiler.merge_features(left.weights, aligned)
            reg = (
                _residual_norm(compiler.merge1, pair1)
                + _residual_norm(compiler.merge2, pair2)
                + _residual_norm(compiler.merge3, pair3)
            )
            loss = 0.5 * (
                _bce_on(edited, x, concepts, left.skills[0]) + _bce_on(edited, x, concepts, right.skills[0])
            )
            loss = loss + 1e-3 * reg
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step} op={op}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(compiler.parameters(), 5.0)
        opt.step()
        losses.append(float(loss.detach()))
        if step % 50 == 0 or step == steps - 1:
            scales = (
                float(F.softplus(compiler.alpha_raw).detach()),
                float(F.softplus(compiler.r_raw).detach()),
                float(compiler.db3_raw.detach()),
                float(F.softplus(compiler.head_temp_raw).detach()),
            )
            print(
                f"step {step:4d} op={op:8s} loss={losses[-1]:.4f} alpha={scales[0]:.2f} "
                f"readout={scales[1]:.2f} db3={scales[2]:.2f} head_temp={scales[3]:.2f}"
            )
    return losses


def save_compiler(path: str | Path, compiler: WeightCompiler) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": compiler.state_dict(),
            "d_in": compiler.d_in,
            "d_out": compiler.d_out,
            "d_model": compiler.d_model,
        },
        path,
    )


def load_compiler(path: str | Path) -> WeightCompiler:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    compiler = WeightCompiler(d_in=blob["d_in"], d_out=blob["d_out"], d_model=blob["d_model"])
    # Compilers saved before head routing lack these; their defaults are the geometric prior.
    optional = {"merge3.weight", "merge3.bias", "head_temp_raw"}
    result = compiler.load_state_dict(blob["state_dict"], strict=False)
    missing = set(result.missing_keys) - optional
    if missing or result.unexpected_keys:
        raise RuntimeError(f"compiler state mismatch: missing={sorted(missing)} unexpected={result.unexpected_keys}")
    compiler.eval()
    return compiler


def train_from_dir(checkpoint_dir: str | Path, out: str | Path, steps: int = 600, seed: int = 0) -> WeightCompiler:
    concepts = load_concepts(checkpoint_dir)
    records = load_zoo(checkpoint_dir)
    compiler = WeightCompiler(d_in=concepts.shape[1], d_out=concepts.shape[0])
    train_compiler(compiler, records, concepts, steps=steps, seed=seed)
    save_compiler(out, compiler)
    return compiler
