"""Held-out comparison against geometric baselines. Targets are never trained here, except the labeled oracle."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from zeroth_grad.align import align_to, permute_h1, permute_h2
from zeroth_grad.compiler import WeightCompiler
from zeroth_grad.data import skill_labels
from zeroth_grad.edit import (
    geometric_collapse,
    magnitude_prune,
    mix_checkpoints,
    neuron_norm_h1,
    neuron_norm_h2,
    norm_prior,
    uniform_average,
)
from zeroth_grad.mlp import functional_mlp
from zeroth_grad.train import duplication_task, load_compiler
from zeroth_grad.zoo import Record, head_accuracy, load_concepts, load_zoo


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _pair_accuracy(weights: dict[str, torch.Tensor], concepts: torch.Tensor, heads: list[int], n: int = 3000, seed: int = 0) -> float:
    return _mean([head_accuracy(weights, concepts, head, n=n, seed=seed) for head in heads])


@torch.no_grad()
def _inject_case(
    compiler: WeightCompiler,
    weights: dict[str, torch.Tensor],
    concepts: torch.Tensor,
    src: int,
    tgt: int,
    n: int,
) -> dict[str, float]:
    before_old = head_accuracy(weights, concepts, src, n=n, seed=1)
    before_new = head_accuracy(weights, concepts, tgt, n=n, seed=2)
    edited = compiler.inject(weights, concepts[tgt], tgt)
    return {
        "before_old": before_old,
        "before_new": before_new,
        "after_old": head_accuracy(edited, concepts, src, n=n, seed=1),
        "after_new": head_accuracy(edited, concepts, tgt, n=n, seed=2),
    }


def finetune_oracle(
    weights: dict[str, torch.Tensor],
    concepts: torch.Tensor,
    src: int,
    tgt: int,
    steps: int = 200,
    lr: float = 0.05,
    batch_size: int = 256,
) -> dict[str, torch.Tensor]:
    """Adam on the target checkpoint for `steps` steps. A ceiling, not the compiler."""
    params = {key: value.clone().detach().requires_grad_(True) for key, value in weights.items()}
    opt = torch.optim.Adam(params.values(), lr=lr)
    for _ in range(steps):
        x = torch.randn(batch_size, concepts.shape[1])
        logits = functional_mlp(params, x)
        loss = F.binary_cross_entropy_with_logits(logits[:, tgt], skill_labels(x, concepts[tgt]))
        loss = loss + F.binary_cross_entropy_with_logits(logits[:, src], skill_labels(x, concepts[src]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return {key: value.detach() for key, value in params.items()}


def evaluate_inject(
    compiler: WeightCompiler,
    records: list[Record],
    concepts: torch.Tensor,
    targets: tuple[int, ...] = (6, 7),
    n: int = 3000,
    oracle_limit: int | None = None,
) -> dict[str, float]:
    compiler.eval()
    fresh = WeightCompiler(d_in=compiler.d_in, d_out=compiler.d_out, d_model=compiler.d_model)
    fresh.eval()
    heldout = [r for r in records if r.split == "heldout" and r.kind == "single"]
    trained_rows = []
    fresh_rows = []
    oracle_new = []
    seen = 0
    for record in heldout:
        src = record.skills[0]
        for tgt in targets:
            trained_rows.append(_inject_case(compiler, record.weights, concepts, src, tgt, n))
            fresh_rows.append(_inject_case(fresh, record.weights, concepts, src, tgt, n))
            if oracle_limit is None or seen < oracle_limit:
                edited = finetune_oracle(record.weights, concepts, src, tgt)
                oracle_new.append(head_accuracy(edited, concepts, tgt, n=n, seed=2))
                seen += 1
    return {
        "n": float(len(trained_rows)),
        "trained_new_acc": _mean([row["after_new"] for row in trained_rows]),
        "trained_old_acc": _mean([row["after_old"] for row in trained_rows]),
        "untrained_new_acc": _mean([row["after_new"] for row in fresh_rows]),
        "untrained_old_acc": _mean([row["after_old"] for row in fresh_rows]),
        "before_new_acc": _mean([row["before_new"] for row in trained_rows]),
        "before_old_acc": _mean([row["before_old"] for row in trained_rows]),
        "oracle_new_acc": _mean(oracle_new),
    }


def evaluate_compress(
    compiler: WeightCompiler,
    records: list[Record],
    concepts: torch.Tensor,
    n: int = 3000,
) -> dict[str, float]:
    compiler.eval()
    heldout = [r for r in records if r.split == "heldout" and r.kind == "single"]
    dense, compiled, magnitude, geometric = [], [], [], []
    for index, record in enumerate(heldout):
        generator = torch.Generator().manual_seed(1000 + index)
        wide, k1, k2 = duplication_task(record.weights, generator)
        skill = record.skills[0]
        dense.append(head_accuracy(record.weights, concepts, skill, n=n, seed=3))
        compiled.append(head_accuracy(compiler.compress(wide, k1, k2), concepts, skill, n=n, seed=3))
        magnitude.append(head_accuracy(magnitude_prune(wide, k1, k2), concepts, skill, n=n, seed=3))
        geometric.append(head_accuracy(geometric_collapse(wide), concepts, skill, n=n, seed=3))
    return {
        "n": float(len(heldout)),
        "dense_acc": _mean(dense),
        "compiler_acc": _mean(compiled),
        "magnitude_acc": _mean(magnitude),
        "geometric_acc": _mean(geometric),
    }


def _merge_accuracies(
    weights: dict[str, torch.Tensor],
    concepts: torch.Tensor,
    heads: list[int],
    n: int,
) -> float:
    return _pair_accuracy(weights, concepts, heads, n=n, seed=4)


def evaluate_merge(
    compiler: WeightCompiler,
    records: list[Record],
    concepts: torch.Tensor,
    n: int = 2500,
) -> dict[str, float]:
    compiler.eval()
    heldout = [r for r in records if r.split == "heldout" and r.kind == "single"]
    by_seed: dict[str, list[Record]] = {}
    for record in heldout:
        by_seed.setdefault(record.metadata["seed"], []).append(record)
    compiler_scores = []
    unaligned_scores = []
    aligned_soup_scores = []
    norm_gate_scores = []
    perm_compiler = []
    perm_unaligned = []
    pairs = 0
    for seed, group in sorted(by_seed.items()):
        for i, left in enumerate(group):
            for right in group[i + 1 :]:
                if left.skills[0] == right.skills[0]:
                    continue
                heads = [left.skills[0], right.skills[0]]
                merged = compiler.merge(left.weights, right.weights)
                compiler_scores.append(_merge_accuracies(merged, concepts, heads, n))
                unaligned_scores.append(_merge_accuracies(uniform_average(left.weights, right.weights), concepts, heads, n))
                aligned = align_to(left.weights, right.weights)
                aligned_soup_scores.append(_merge_accuracies(uniform_average(left.weights, aligned), concepts, heads, n))
                lam1 = norm_prior(neuron_norm_h1(left.weights), neuron_norm_h1(aligned))
                lam2 = norm_prior(neuron_norm_h2(left.weights), neuron_norm_h2(aligned))
                norm_gate_scores.append(_merge_accuracies(mix_checkpoints(left.weights, aligned, lam1, lam2), concepts, heads, n))
                generator = torch.Generator().manual_seed(2000 + pairs)
                perm1 = torch.randperm(right.weights["w1"].shape[0], generator=generator)
                perm2 = torch.randperm(right.weights["w2"].shape[0], generator=generator)
                permuted = permute_h2(permute_h1(right.weights, perm1), perm2)
                perm_compiler.append(_merge_accuracies(compiler.merge(left.weights, permuted), concepts, heads, n))
                perm_unaligned.append(_merge_accuracies(uniform_average(left.weights, permuted), concepts, heads, n))
                pairs += 1
    return {
        "n": float(pairs),
        "compiler_acc": _mean(compiler_scores),
        "unaligned_acc": _mean(unaligned_scores),
        "aligned_soup_acc": _mean(aligned_soup_scores),
        "norm_gate_acc": _mean(norm_gate_scores),
        "permuted_compiler_acc": _mean(perm_compiler),
        "permuted_unaligned_acc": _mean(perm_unaligned),
    }


def evaluate_all(
    compiler: WeightCompiler,
    checkpoint_dir: str | Path,
    out: str | Path | None = None,
) -> dict[str, dict[str, float]]:
    concepts = load_concepts(checkpoint_dir)
    records = load_zoo(checkpoint_dir)
    metrics = {
        "inject": evaluate_inject(compiler, records, concepts),
        "compress": evaluate_compress(compiler, records, concepts),
        "merge": evaluate_merge(compiler, records, concepts),
    }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
