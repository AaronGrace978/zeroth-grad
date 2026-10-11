"""Transformer merge benchmark on two synthetic sequence tasks.

Each sequence starts with a task token and continues with 8 digits in [0, 16).
Task "prev" (token 16) predicts, at positions 2..8, the previous digit, which
needs an attention head that looks one step back. Task "max" (token 17)
predicts, at positions 1..8, the running maximum, which needs attention over
the whole prefix. One readout over the 16 digits serves both.

Two experiments, each for LayerNorm (permutation symmetry) and RMSNorm
(rotation symmetry):

- Fine-tune merge. A base trained on both tasks is fine-tuned on each task
  alone, and the two fine-tunes are merged by averaging and by task
  arithmetic. The partner is then moved by a random element of its symmetry
  group (residual basis, head order, head-internal bases, MLP order): the
  function is unchanged, the unaligned merge collapses, and the aligned
  merge returns to the unpermuted one.
- Independent initializations. Two models trained from different seeds on
  both tasks are averaged at the midpoint, with and without alignment.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn.functional as F

from zeroth_grad.transformer import TransformerConfig, clone, forward, init_transformer, random_symmetry
from zeroth_grad.transformer_align import align_transformer, merge_transformers, weight_distance

N_DIGITS = 16
TASK_TOKENS = {"prev": 16, "max": 17}
SEQ = 8


def sample_batch(task: str, n: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (tokens (n, 9), targets (n, 9), mask (n, 9)) for one task."""
    digits = torch.randint(0, N_DIGITS, (n, SEQ), generator=generator)
    tokens = torch.cat([torch.full((n, 1), TASK_TOKENS[task]), digits], dim=1)
    targets = torch.zeros_like(tokens)
    mask = torch.zeros_like(tokens, dtype=torch.bool)
    if task == "prev":
        targets[:, 2:] = digits[:, :-1]
        mask[:, 2:] = True
    else:
        targets[:, 1:] = torch.cummax(digits, dim=1).values
        mask[:, 1:] = True
    return tokens, targets, mask


def task_loss(weights, cfg, task, n, generator):
    tokens, targets, mask = sample_batch(task, n, generator)
    logits = forward(weights, cfg, tokens)
    return F.cross_entropy(logits[mask], targets[mask])


@torch.no_grad()
def task_accuracy(weights: dict[str, torch.Tensor], cfg: TransformerConfig, task: str, n: int = 2000, seed: int = 123) -> float:
    generator = torch.Generator().manual_seed(seed)
    tokens, targets, mask = sample_batch(task, n, generator)
    predicted = forward(weights, cfg, tokens).argmax(dim=-1)
    return float((predicted[mask] == targets[mask]).float().mean())


def train(
    weights: dict[str, torch.Tensor],
    cfg: TransformerConfig,
    tasks: tuple[str, ...],
    steps: int,
    seed: int,
    lr: float = 3e-3,
    batch_size: int = 128,
) -> dict[str, torch.Tensor]:
    params = {k: v.clone().requires_grad_(True) for k, v in weights.items()}
    opt = torch.optim.AdamW(params.values(), lr=lr, weight_decay=0.0)
    generator = torch.Generator().manual_seed(seed)
    for step in range(steps):
        loss = sum(task_loss(params, cfg, task, batch_size, generator) for task in tasks) / len(tasks)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params.values(), 1.0)
        opt.step()
    return {k: v.detach() for k, v in params.items()}


def _scores(weights, cfg) -> dict[str, float]:
    prev = task_accuracy(weights, cfg, "prev")
    mx = task_accuracy(weights, cfg, "max")
    return {"prev": prev, "max": mx, "mean": 0.5 * (prev + mx)}


def run_norm(norm: str, base_steps: int = 200, tune_steps: int = 400, train_steps: int = 600, seed: int = 0) -> dict:
    cfg = replace(TransformerConfig(), norm=norm)
    rotations = norm == "rms"
    init = init_transformer(cfg, seed=seed)
    base = train(init, cfg, ("prev", "max"), base_steps, seed=seed + 1)
    tuned_prev = train(base, cfg, ("prev",), tune_steps, seed=seed + 2)
    tuned_max = train(base, cfg, ("max",), tune_steps, seed=seed + 3)
    moved = random_symmetry(tuned_max, cfg, seed=seed + 4, rotate=rotations, scale=1.7 if rotations else None)

    results: dict[str, dict] = {
        "base": _scores(base, cfg),
        "tuned_prev": _scores(tuned_prev, cfg),
        "tuned_max": _scores(tuned_max, cfg),
        "partner_moved_alone": _scores(moved, cfg),
    }
    pair = [tuned_prev, tuned_max]
    moved_pair = [tuned_prev, moved]
    results["average"] = _scores(merge_transformers(pair, cfg, align=False), cfg)
    results["task_arithmetic"] = _scores(merge_transformers(pair, cfg, base=base, align=False), cfg)
    results["moved_average_unaligned"] = _scores(merge_transformers(moved_pair, cfg, align=False), cfg)
    results["moved_task_arithmetic_unaligned"] = _scores(merge_transformers(moved_pair, cfg, base=base, align=False), cfg)
    aligned_avg = merge_transformers(moved_pair, cfg, align=True, rotations=rotations)
    aligned_ta = merge_transformers(moved_pair, cfg, base=base, align=True, rotations=rotations)
    results["moved_average_aligned"] = _scores(aligned_avg, cfg)
    results["moved_task_arithmetic_aligned"] = _scores(aligned_ta, cfg)
    recovered = align_transformer(tuned_prev, moved, cfg, rotations=rotations)
    reference = align_transformer(tuned_prev, tuned_max, cfg, rotations=rotations)
    results["partner_distance_before"] = weight_distance(reference, moved)
    results["partner_distance_after"] = weight_distance(reference, recovered)

    independent = []
    for pair_seed in (10, 20):
        m_a = train(init_transformer(cfg, seed=pair_seed), cfg, ("prev", "max"), train_steps, seed=pair_seed + 1)
        m_b = train(init_transformer(cfg, seed=pair_seed + 5), cfg, ("prev", "max"), train_steps, seed=pair_seed + 6)
        independent.append(
            {
                "endpoints": 0.5 * (_scores(m_a, cfg)["mean"] + _scores(m_b, cfg)["mean"]),
                "midpoint_unaligned": _scores(merge_transformers([m_a, m_b], cfg, align=False), cfg)["mean"],
                "midpoint_aligned": _scores(merge_transformers([m_a, m_b], cfg, align=True, rotations=rotations), cfg)["mean"],
                "midpoint_aligned_permutations_only": _scores(merge_transformers([m_a, m_b], cfg, align=True, rotations=False), cfg)["mean"],
            }
        )
    results["independent"] = {k: sum(r[k] for r in independent) / len(independent) for k in independent[0]}
    results["config"] = cfg.to_dict()
    return results


def run_all(out: str | Path | None = None, base_steps: int = 200, tune_steps: int = 400, train_steps: int = 600) -> dict:
    torch.manual_seed(0)
    metrics = {
        norm: run_norm(norm, base_steps=base_steps, tune_steps=tune_steps, train_steps=train_steps) for norm in ("layer", "rms")
    }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
