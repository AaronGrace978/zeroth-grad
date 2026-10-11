"""Rank-k inject: install a non-linear skill from labeled examples, in closed form.

Rank-1 inject writes relu(alpha * phi . x) onto spare units, which can only
express a halfspace. Rank-k inject writes a small sub-network instead:

    Delta W1[spares1] = Theta U,    Theta in R^{m x k},  U in R^{k x d}

so the hidden-1 update has rank k, as in a LoRA adapter. U spans the concept
subspace, estimated from the examples by second-order inverse regression: the
top eigenvectors of the whitened class-mean and class-covariance differences.
That spans directions a mean difference misses, such as the two axes of an XOR
or the normal of a slab, whose class means coincide.

Theta (unit directions inside the subspace) and the hidden-1 biases are chosen
by greedy forward selection from a dictionary of ReLU ridge features
relu(theta . U x - beta). The readout of the selected features starts from a
weighted least-squares solve against the +-1 labels and is refined by a few
Newton steps of ridge-regularized logistic regression: a convex problem in at
most m + 1 readout weights, solved exactly at each step. Two spare hidden-2
units carry the signed readout through the second ReLU, relu(t) - relu(-t) = t,
so the target head's logit is t. The network itself is never optimized: units
that are not spare, and heads other than the target, are not touched. If the
checkpoint has too few spares, it is widened with dead units first.
"""

from __future__ import annotations

import math

import torch

from zeroth_grad.capacity import ensure_spares
from zeroth_grad.edit import neuron_norm_h1, neuron_norm_h2
from zeroth_grad.fold import balanced_weights, greedy_order, refit
from zeroth_grad.mlp import clone_weights


def estimate_subspace(x: torch.Tensor, y: torch.Tensor, rank: int, ridge: float = 1e-3) -> torch.Tensor:
    """Orthonormal rows (rank, d) spanning the directions along which the label depends on x.

    In whitened coordinates z, the kernel is
        M = sum_c p_c [ (mu_c)(mu_c)^T + (Sigma_c - I)^2 ],
    the sliced-average-variance form of inverse regression. Its top
    eigenvectors are mapped back to input space through the whitening.
    """
    x = x.double()
    y = y.view(-1) > 0.5
    if bool(y.all()) or bool((~y).all()):
        raise ValueError("estimate_subspace needs both classes")
    d = x.shape[1]
    mean = x.mean(dim=0)
    centered = x - mean
    cov = centered.T @ centered / x.shape[0]
    cov = cov + ridge * torch.eye(d, dtype=x.dtype) * cov.diagonal().mean()
    evals, evecs = torch.linalg.eigh(cov)
    whiten = evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.T
    z = centered @ whiten
    kernel = torch.zeros(d, d, dtype=x.dtype)
    eye = torch.eye(d, dtype=x.dtype)
    for side in (y, ~y):
        p = float(side.float().mean())
        zc = z[side]
        mu = zc.mean(dim=0)
        sigma = (zc - mu).T @ (zc - mu) / max(zc.shape[0] - 1, 1)
        diff = sigma - eye
        kernel = kernel + p * (torch.outer(mu, mu) + diff @ diff)
    _, vecs = torch.linalg.eigh(kernel)
    top = vecs[:, -rank:].flip(1)
    directions = (whiten @ top).T
    q, _ = torch.linalg.qr(directions.T)
    return q.T.float()


def _unit_directions(k: int, extra: int, generator: torch.Generator) -> torch.Tensor:
    """Axes, signed pairwise diagonals, and `extra` random unit vectors in R^k."""
    dirs = []
    eye = torch.eye(k)
    for i in range(k):
        dirs.extend([eye[i], -eye[i]])
        for j in range(i + 1, k):
            for si in (1.0, -1.0):
                for sj in (1.0, -1.0):
                    dirs.append((si * eye[i] + sj * eye[j]) / math.sqrt(2))
    if extra > 0 and k > 1:
        rand = torch.randn(extra, k, generator=generator)
        dirs.extend(list(rand / rand.norm(dim=1, keepdim=True)))
    return torch.stack(dirs)


def ridge_dictionary(
    projected: torch.Tensor,
    extra_directions: int = 32,
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9),
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Candidate ReLU features relu(theta . p - beta) on projected inputs p = U x.

    Returns (features (n, c), thetas (c, k), betas (c,)). Offsets are 0 and
    quantiles of each direction's projection.
    """
    generator = torch.Generator().manual_seed(seed)
    thetas = _unit_directions(projected.shape[1], extra_directions, generator)
    t = projected @ thetas.T
    q = torch.tensor((0.0, *quantiles))
    levels = torch.quantile(t, q[1:], dim=0)
    betas = torch.cat([torch.zeros(1, t.shape[1]), levels], dim=0)
    all_theta = thetas.repeat(betas.shape[0], 1)
    all_beta = betas.reshape(-1)
    features = torch.relu(t.repeat(1, betas.shape[0]) - all_beta[None, :])
    return features, all_theta, all_beta


def fit_circuit(
    x: torch.Tensor,
    y: torch.Tensor,
    rank: int,
    units: int,
    subspace: torch.Tensor | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Closed-form rank-k circuit for labels y on inputs x.

    Returns U (k, d), the chosen hidden-1 rows (units, d) and biases, and the
    readout (beta, beta0) such that t(x) = beta . relu(rows x + bias) + beta0
    approximates 2y - 1.
    """
    u = estimate_subspace(x, y, rank) if subspace is None else subspace.float()
    projected = x.float() @ u.T
    features, thetas, betas = ridge_dictionary(projected, seed=seed)
    target = (2.0 * (y.view(-1, 1).float() > 0.5).float() - 1.0)
    weight = balanced_weights(target)
    usable = (features.std(dim=0) > 1e-6).nonzero().flatten()
    order = greedy_order(features, target, usable, weight)[:units]
    chosen = torch.tensor(order, dtype=torch.long)
    beta, beta0 = refit(features[:, chosen], target, weight)
    beta, beta0 = logistic_readout(features[:, chosen], target.view(-1) > 0, weight, beta.view(-1), beta0.view(()))
    rows = thetas[chosen] @ u
    return {
        "subspace": u,
        "rows": rows,
        "bias": -betas[chosen],
        "beta": beta.view(-1),
        "beta0": beta0.view(()),
    }


def logistic_readout(
    features: torch.Tensor,
    positive: torch.Tensor,
    weight: torch.Tensor,
    beta: torch.Tensor,
    beta0: torch.Tensor,
    steps: int = 15,
    ridge: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Newton steps (IRLS) for weighted, ridge-regularized logistic regression on fixed features.

    Starts from the least-squares readout. The problem is convex in the m + 1
    readout weights, and each step is one (m + 1) x (m + 1) solve.
    """
    n = features.shape[0]
    design = torch.cat([features.double(), torch.ones(n, 1, dtype=torch.float64)], dim=1)
    target = positive.double()
    w = weight.double() / weight.double().mean()
    theta = torch.cat([beta.double(), beta0.double().view(1)])
    scale = design[:, :-1].pow(2).mean(dim=0)
    penalty = torch.cat([ridge * scale, torch.zeros(1, dtype=torch.float64)])
    for _ in range(steps):
        z = design @ theta
        p = torch.sigmoid(z)
        grad = design.T @ (w * (p - target)) / n + penalty * theta
        hess = design.T @ (design * (w * p * (1 - p))[:, None]) / n + torch.diag(penalty + 1e-9)
        step = torch.linalg.solve(hess, grad)
        theta = theta - step
        if float(step.abs().max()) < 1e-8:
            break
    return theta[:-1].float(), theta[-1].float()


def apply_rank_k_inject(
    weights: dict[str, torch.Tensor],
    circuit: dict[str, torch.Tensor],
    target_head: int,
    gain: float = 1.0,
    eps: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Write `circuit` onto spare units and point `target_head` at it. Other heads are unchanged."""
    m = circuit["rows"].shape[0]
    out, _, _ = ensure_spares(weights, m, 2)
    spare1 = (neuron_norm_h1(out) <= eps).nonzero().flatten()[:m]
    spare2 = (neuron_norm_h2(out) <= eps).nonzero().flatten()[:2]
    out = clone_weights(out)
    out["w1"][spare1] = circuit["rows"].to(out["w1"])
    out["b1"][spare1] = circuit["bias"].to(out["b1"])
    pos, neg = int(spare2[0]), int(spare2[1])
    out["w2"][pos] = 0
    out["w2"][neg] = 0
    out["w2"][pos, spare1] = circuit["beta"].to(out["w2"])
    out["w2"][neg, spare1] = -circuit["beta"].to(out["w2"])
    out["b2"][pos] = circuit["beta0"]
    out["b2"][neg] = -circuit["beta0"]
    out["w3"][target_head] = 0
    out["w3"][target_head, pos] = gain
    out["w3"][target_head, neg] = -gain
    out["b3"][target_head] = 0
    return out


def inject_rank_k(
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    target_head: int,
    rank: int = 2,
    units: int | None = None,
    subspace: torch.Tensor | None = None,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Estimate a rank-`rank` subspace from (x, y), fit a circuit on `units` ReLUs, and write it.

    `units` defaults to the number of spare hidden-1 units, at least 2 * rank.
    """
    if units is None:
        units = max(int((neuron_norm_h1(weights) <= 1e-6).sum()), 2 * rank)
    circuit = fit_circuit(x, y, rank, units, subspace=subspace, seed=seed)
    return apply_rank_k_inject(weights, circuit, target_head)
