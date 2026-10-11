"""Assignment solvers for hidden-unit alignment.

`hungarian` is exact and costs O(N^3). `sinkhorn` solves the entropy-regularized
transport problem with matrix-vector scaling, O(N^2) per iteration, and rounds
the plan to a permutation. Rounding is greedy over each row's top candidates,
so it is O(N^2) as well; whatever the greedy pass cannot place goes to an exact
solve on the leftover rows, which is small when the plan is sharp.
"""

from __future__ import annotations

import math

import torch
from scipy.optimize import linear_sum_assignment

SOLVERS = ("hungarian", "sinkhorn", "auto")
AUTO_SINKHORN_WIDTH = 2048


def hungarian(cost: torch.Tensor) -> torch.Tensor:
    """Exact minimum-cost permutation: perm[i] is the source column for reference row i."""
    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().double().numpy())
    perm = torch.empty(cost.shape[0], dtype=torch.long)
    perm[torch.as_tensor(row_ind, dtype=torch.long)] = torch.as_tensor(col_ind, dtype=torch.long)
    return perm


def _gibbs(f: torch.Tensor, g: torch.Tensor, c: torch.Tensor, eps: float) -> torch.Tensor:
    """exp((f + g - C) / eps), with the exponent floored so no entry is a subnormal float.

    Subnormals make CPU matrix products many times slower. The floor, e^-87,
    is the smallest normal float32 and is negligible against the plan's mass.
    """
    z = (f[:, None] + g[None, :] - c) / eps
    return torch.exp(z.clamp_min_(-87.0))


def sinkhorn_potentials(
    cost: torch.Tensor,
    eps_start: float = 1.0,
    eps_end: float = 1e-5,
    stages: int = 16,
    iters: int = 30,
    keep: int = 1,
    tol: float = 1e-2,
) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
    """Dual potentials (f, g) of entropic OT with uniform marginals, by epsilon scaling.

    The cost is divided by its mean, so the epsilons are relative. Each stage
    builds the Gibbs kernel exp((f + g - C) / eps) once and runs plain
    multiplicative Sinkhorn updates on it, then folds the scalings back into
    the potentials. Folding keeps the kernel near 1 where the plan has mass,
    so float32 does not overflow as eps shrinks. A stage stops early once
    every row sum is within `tol` of 1. Returns (f, g, eps) for the
    last `keep` stages, coarsest first.
    """
    c = cost.detach().to(dtype=torch.float32)
    scale = float(c.abs().mean())
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    c = c / scale
    n, m = c.shape
    f = torch.zeros(n, dtype=c.dtype, device=c.device)
    g = -c.min(dim=0).values
    ratio = (eps_end / eps_start) ** (1.0 / max(stages - 1, 1))
    eps = eps_start
    tiny = torch.finfo(c.dtype).tiny
    history: list[tuple[torch.Tensor, torch.Tensor, float]] = []
    for _ in range(stages):
        kernel = _gibbs(f, g, c, eps)
        u = torch.ones(n, dtype=c.dtype, device=c.device)
        v = torch.ones(m, dtype=c.dtype, device=c.device)
        for step in range(iters):
            row = (kernel @ v).clamp_min(tiny)
            # After a column update the columns are exact, so the row sums
            # measure how far the plan is from both marginals.
            if step > 0 and float((u * row - 1).abs().max()) < tol:
                break
            u = 1.0 / row
            v = 1.0 / (kernel.T @ u).clamp_min(tiny)
        f = f + eps * torch.log(u)
        g = g + eps * torch.log(v)
        # A row with no mass left in the kernel gets its potential reset from
        # the exact soft-min, so the next stage starts from a finite point.
        bad = ~torch.isfinite(f)
        if bool(bad.any()):
            f[bad] = -eps * torch.logsumexp((g[None, :] - c[bad]) / eps, dim=1)
        bad = ~torch.isfinite(g)
        if bool(bad.any()):
            g[bad] = -eps * torch.logsumexp((f[:, None] - c[:, bad]) / eps, dim=0)
        history.append((f.clone(), g.clone(), eps))
        eps *= ratio
    return history[-keep:]


def _plans(cost: torch.Tensor, **kwargs):
    c = cost.detach().to(dtype=torch.float32)
    scale = float(c.abs().mean())
    if not math.isfinite(scale) or scale <= 0:
        scale = 1.0
    c = c / scale
    for f, g, eps in sinkhorn_potentials(cost, **kwargs):
        yield _gibbs(f, g, c, eps)


def sinkhorn_plan(cost: torch.Tensor, **kwargs) -> torch.Tensor:
    """Entropic transport plan of the final stage, rows and columns each summing to about 1."""
    return next(_plans(cost, keep=1, **kwargs))


def _greedy_round(score: torch.Tensor, candidates: int, generator: torch.Generator) -> torch.Tensor:
    """Greedy matching on each row's top-`candidates` columns. Unplaced rows get -1."""
    n, m = score.shape
    k = min(candidates, m)
    # Ties are common (every dead unit pairs with every free slot at zero
    # cost). A tiny multiplicative jitter spreads tied rows over different
    # columns, so the greedy pass can place them in one sweep.
    jitter = 1.0 + 1e-4 * torch.rand(score.shape, generator=generator, dtype=score.dtype).to(score.device)
    values, cols = torch.topk(score * jitter, k, dim=1)
    rows = torch.arange(n, device=score.device)[:, None].expand(n, k)
    order = torch.argsort(values.flatten(), descending=True, stable=True).cpu()
    rows = rows.flatten()[order].cpu().tolist()
    cols = cols.flatten()[order].cpu().tolist()
    perm = [-1] * n
    taken = [False] * m
    placed = 0
    for r, c in zip(rows, cols):
        if perm[r] == -1 and not taken[c]:
            perm[r] = c
            taken[c] = True
            placed += 1
            if placed == n:
                break
    return torch.tensor(perm, dtype=torch.long)


def sinkhorn(
    cost: torch.Tensor,
    candidates: int = 8,
    exact_leftover: int = 512,
    rounds: int = 3,
    seed: int = 0,
    **kwargs,
) -> torch.Tensor:
    """Approximate minimum-cost permutation from rounded Sinkhorn plans.

    The plans of the last `rounds` epsilon stages are each rounded, and the
    cheapest permutation is kept. Rounding is greedy and a slightly blurrier
    plan sometimes rounds better, so this costs a few O(N^2) passes and
    removes most of the sensitivity to the epsilon schedule.
    """
    n = cost.shape[0]
    if cost.shape[0] != cost.shape[1]:
        raise ValueError("sinkhorn alignment needs a square cost")
    if n <= 1:
        return torch.arange(n, dtype=torch.long)
    best, best_cost = None, float("inf")
    for plan in _plans(cost, keep=rounds, **kwargs):
        perm = _round_plan(plan, cost, candidates, exact_leftover, torch.Generator().manual_seed(seed))
        total = assignment_cost(cost, perm)
        if total < best_cost:
            best, best_cost = perm, total
    return two_opt(cost, best)


def two_opt(cost: torch.Tensor, perm: torch.Tensor, passes: int = 20, batch: int = 256, chunk: int = 1024) -> torch.Tensor:
    """Improve an assignment by pairwise swaps, applying disjoint improving swaps each pass.

    Swapping the columns of rows i and j changes the cost by
    C[i, p_j] + C[j, p_i] - C[i, p_i] - C[j, p_j]. If both rows already hold
    their row minimum, neither term can fall, so only rows above their
    minimum are scored, against every partner. A pass is O(S N) for S such
    rows, and at most O(N^2).
    """
    c = cost.detach().to(dtype=torch.float32)
    n = c.shape[0]
    perm = perm.clone().to(c.device)
    tol = 1e-7 * float(c.abs().mean())
    row_min = c.min(dim=1).values
    for _ in range(passes):
        held = c[torch.arange(n, device=c.device), perm]
        suspects = (held > row_min + tol).nonzero().flatten()
        if suspects.numel() == 0:
            break
        best_vals, best_i, best_j = [], [], []
        for lo in range(0, suspects.numel(), chunk):
            rows = suspects[lo : lo + chunk]
            # gain[r, j] for suspect row r = rows[r] and partner j.
            gain = c[rows][:, perm] + c[:, perm[rows]].T - held[rows][:, None] - held[None, :]
            gain[torch.arange(rows.numel(), device=c.device), rows] = 0
            k = min(batch, gain.numel())
            values, flat = torch.topk(-gain.flatten(), k)
            best_vals.append(values)
            best_i.append(rows[flat // n])
            best_j.append(flat % n)
        values = torch.cat(best_vals)
        order = torch.argsort(values, descending=True)
        improving = values[order] > tol
        if not bool(improving.any()):
            break
        used = set()
        swaps = []
        for i, j in zip(torch.cat(best_i)[order][improving].tolist(), torch.cat(best_j)[order][improving].tolist()):
            if i in used or j in used or i == j:
                continue
            used.update((i, j))
            swaps.append((i, j))
        for i, j in swaps:
            perm[i], perm[j] = perm[j].clone(), perm[i].clone()
    return perm.cpu()


def _round_plan(
    plan: torch.Tensor,
    cost: torch.Tensor,
    candidates: int,
    exact_leftover: int,
    generator: torch.Generator,
) -> torch.Tensor:
    n = cost.shape[0]
    perm = _greedy_round(plan, candidates, generator)
    rows = (perm < 0).nonzero().flatten()
    while rows.numel() > 0:
        used = torch.zeros(n, dtype=torch.bool)
        used[perm[perm >= 0]] = True
        cols = (~used).nonzero().flatten()
        sub_cost = cost.detach().cpu()[rows][:, cols]
        if rows.numel() <= exact_leftover:
            perm[rows] = cols[hungarian(sub_cost)]
            break
        sub_perm = _greedy_round(plan.cpu()[rows][:, cols], candidates, generator)
        placed = sub_perm >= 0
        perm[rows[placed]] = cols[sub_perm[placed]]
        rows = rows[~placed]
    return perm


def solve_assignment(cost: torch.Tensor, solver: str = "hungarian") -> torch.Tensor:
    """Return perm such that source[perm[i]] aligns with reference[i]."""
    if solver not in SOLVERS:
        raise ValueError(f"unknown solver {solver!r}; choose from {SOLVERS}")
    if solver == "auto":
        solver = "sinkhorn" if cost.shape[0] > AUTO_SINKHORN_WIDTH else "hungarian"
    if solver == "sinkhorn":
        return sinkhorn(cost)
    return hungarian(cost)


def assignment_cost(cost: torch.Tensor, perm: torch.Tensor) -> float:
    rows = torch.arange(cost.shape[0])
    return float(cost.detach().cpu()[rows, perm].double().sum())
