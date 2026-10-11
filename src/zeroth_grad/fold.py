"""Activation folding: free slots in dense checkpoints without a sparsity penalty.

A dense checkpoint uses many hidden units, but what its heads read from a
layer usually lives in a much smaller span of activations. Folding keeps the
fewest units of each layer whose activations, on an unlabeled calibration
batch, reproduce the owned heads, and re-solves the next layer's weights by
least squares onto the kept units. The dropped units become dead slots that a
merge or an inject can use.

Units are ranked by greedy forward selection (orthogonal matching pursuit
with a matrix target): each step takes the unit whose activation column,
after projecting out the units already taken, explains the most of what the
next layer reads. The kept prefix is the shortest one whose refit keeps every
owned head's balanced sign agreement with the original at or above
`agreement`. Fits and agreement weight the inputs the original calls
positive and negative equally, so a rare class is not traded away.

Layers are folded from the readout down: hidden-2 against the owned heads'
logits, then hidden-1 against the kept hidden-2 pre-activations, each column
weighted by how strongly the heads read that unit. Readout rows of heads the
checkpoint does not own are zeroed, since they were never trained and a merge
only needs the owned heads.

The singular values of each target give a floor on the units any selection
needs (a refit through r units has rank at most r); `fold_report` returns it
next to the count actually kept.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from zeroth_grad.mlp import clone_weights


def _hidden(weights: dict[str, torch.Tensor], x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h1 = F.relu(x @ weights["w1"].T + weights["b1"])
    h2 = F.relu(h1 @ weights["w2"].T + weights["b2"])
    return h1, h2


def greedy_order(
    features: torch.Tensor,
    target: torch.Tensor,
    candidates: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
) -> list[int]:
    """Candidate columns of `features` in greedy order of how much of `target` each newly explains.

    A constant column is always in the span, so the next layer's bias is free.
    """
    n = features.shape[0]
    root = torch.ones(n, 1, dtype=torch.float64) if sample_weight is None else sample_weight.double().sqrt()[:, None]
    feats = features.double() * root
    ones = root / root.norm()
    target = target.double() * root
    resid = target - ones @ (ones.T @ target)
    basis = [ones]
    pool = [int(i) for i in candidates.tolist()]
    order: list[int] = []
    while pool:
        q = torch.cat(basis, dim=1)
        block = feats[:, pool]
        perp = block - q @ (q.T @ block)
        length = perp.norm(dim=0)
        usable = length > 1e-9 * (block.norm(dim=0) + 1e-12)
        if not bool(usable.any()):
            break
        gain = (perp.T @ resid).norm(dim=1) / length.clamp_min(1e-12)
        gain[~usable] = -1.0
        best = int(gain.argmax())
        direction = perp[:, best : best + 1] / length[best]
        resid = resid - direction @ (direction.T @ resid)
        basis.append(direction)
        order.append(pool.pop(best))
    return order


def refit(
    features: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor | None = None,
    ridge: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted least squares target ~ features @ W.T + b. Returns (W, b)."""
    n = features.shape[0]
    weight = torch.ones(n, dtype=torch.float64) if sample_weight is None else sample_weight.double()
    design = torch.cat([features.double(), torch.ones(n, 1, dtype=torch.float64)], dim=1)
    gram = design.T @ (weight[:, None] * design)
    gram = gram + ridge * torch.eye(gram.shape[0], dtype=torch.float64) * gram.diagonal().mean().clamp_min(1e-12)
    coef = torch.linalg.solve(gram, design.T @ (weight[:, None] * target.double()))
    return coef[:-1].T.float(), coef[-1].float()


def numerical_rank(target: torch.Tensor, tol: float = 0.05) -> int:
    """Fewest singular directions that reconstruct the centered target to relative error `tol`."""
    centered = (target - target.mean(dim=0, keepdim=True)).double()
    if centered.numel() == 0:
        return 0
    energy = torch.linalg.svdvals(centered).square()
    total = float(energy.sum())
    if total <= 1e-24:
        return 0
    kept = torch.cumsum(energy, 0)
    return int((kept < (1 - tol**2) * total).sum()) + 1


def _agreement(logits: torch.Tensor, reference: torch.Tensor) -> float:
    """Worst head's balanced sign agreement with the reference.

    Agreement is averaged over the inputs the reference calls positive and
    those it calls negative, so a rare positive class counts as much as the
    common negative one.
    """
    positive = reference > 0
    match = ((logits > 0) == positive).float()
    rates = []
    for side in (positive, ~positive):
        count = side.float().sum(dim=0)
        rate = (match * side.float()).sum(dim=0) / count.clamp_min(1)
        rates.append(torch.where(count > 0, rate, torch.ones_like(rate)))
    return float((0.5 * (rates[0] + rates[1])).min())


def balanced_weights(reference: torch.Tensor) -> torch.Tensor:
    """Per-input weights that give each head's positive and negative sides equal total mass."""
    positive = (reference > 0).float()
    n = reference.shape[0]
    pos = positive.sum(dim=0).clamp_min(1)
    neg = (1 - positive).sum(dim=0).clamp_min(1)
    per_head = positive * (n / (2 * pos)) + (1 - positive) * (n / (2 * neg))
    return per_head.mean(dim=1)


def _set_readout(out, heads_t, keep2_t, w3_new, b3_new):
    out["w3"][heads_t] = 0
    if keep2_t.numel():
        out["w3"][heads_t[:, None], keep2_t[None, :]] = w3_new
    out["b3"][heads_t] = b3_new


@torch.no_grad()
def fold_report(
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
    heads: list[int],
    agreement: float = 0.99,
    eps: float = 1e-6,
) -> dict:
    """Fold both hidden layers onto the fewest units that keep the owned heads. Width is unchanged."""
    out = clone_weights(weights)
    heads_t = torch.tensor(sorted(set(int(h) for h in heads)), dtype=torch.long)
    others = [k for k in range(out["w3"].shape[0]) if k not in set(heads_t.tolist())]
    out["w3"][others] = 0
    h1, h2 = _hidden(out, x)
    reference = h2 @ out["w3"][heads_t].T + out["b3"][heads_t]
    weight = balanced_weights(reference)

    live2 = (out["w3"][heads_t].norm(dim=0) > eps).nonzero().flatten()
    order2 = greedy_order(h2, reference, live2, weight)
    keep2_t = torch.tensor(order2, dtype=torch.long)
    w3_new, b3_new = refit(h2[:, keep2_t], reference, weight)
    for r in range(1, len(order2) + 1):
        trial = torch.tensor(order2[:r], dtype=torch.long)
        w_r, b_r = refit(h2[:, trial], reference, weight)
        if _agreement(h2[:, trial] @ w_r.T + b_r, reference) >= agreement:
            keep2_t, w3_new, b3_new = trial, w_r, b_r
            break
    _set_readout(out, heads_t, keep2_t, w3_new, b3_new)
    drop2 = torch.ones(out["w2"].shape[0], dtype=torch.bool)
    drop2[keep2_t] = False
    out["w2"][drop2] = 0
    out["b2"][drop2] = 0

    pre2 = h1 @ out["w2"][keep2_t].T + out["b2"][keep2_t]
    importance = out["w3"][heads_t][:, keep2_t].norm(dim=0)
    live1 = (out["w2"][keep2_t].norm(dim=0) > eps).nonzero().flatten()
    order1 = greedy_order(h1, pre2 * importance[None, :], live1, weight)

    def with_h1(kept: list[int]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        trial = clone_weights(out)
        kept_t = torch.tensor(kept, dtype=torch.long)
        w2_new, b2_new = refit(h1[:, kept_t], pre2, weight)
        rows = torch.zeros(keep2_t.numel(), trial["w2"].shape[1])
        if kept:
            rows[:, kept_t] = w2_new
        trial["w2"][keep2_t] = rows
        trial["b2"][keep2_t] = b2_new
        drop1 = torch.ones(trial["w1"].shape[0], dtype=torch.bool)
        drop1[kept_t] = False
        trial["w2"][:, drop1] = 0
        trial["w1"][drop1] = 0
        trial["b1"][drop1] = 0
        _, h2_trial = _hidden(trial, x)
        return trial, h2_trial @ trial["w3"][heads_t].T + trial["b3"][heads_t]

    keep1 = order1
    folded, _ = with_h1(order1)
    for r in range(1, len(order1) + 1):
        trial, logits = with_h1(order1[:r])
        if _agreement(logits, reference) >= agreement:
            keep1, folded = order1[:r], trial
            break
    _, final_h2 = _hidden(folded, x)
    final = final_h2 @ folded["w3"][heads_t].T + folded["b3"][heads_t]
    return {
        "weights": folded,
        "live_h1_before": int((weights["w2"].norm(dim=0) > eps).sum()),
        "live_h2_before": int((weights["w3"].norm(dim=0) > eps).sum()),
        "kept_h1": len(keep1),
        "kept_h2": int(keep2_t.numel()),
        "target_rank_h1": numerical_rank(pre2) if keep2_t.numel() else 0,
        "target_rank_h2": numerical_rank(reference),
        "agreement": _agreement(final, reference),
    }


def fold(
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
    heads: list[int],
    agreement: float = 0.99,
) -> dict[str, torch.Tensor]:
    return fold_report(weights, x, heads, agreement=agreement)["weights"]
