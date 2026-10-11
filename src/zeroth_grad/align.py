"""Layerwise Hungarian alignment of hidden units.

Cost is magnitude-weighted cosine distance on incoming weights (row plus bias),
plus a small penalty for pairing two large orthogonal neurons. That keeps a
trained detector matched to a spare rather than to an unrelated detector, and
still matches two copies of the same direction exactly.
"""

from __future__ import annotations

import torch
from scipy.optimize import linear_sum_assignment

from zeroth_grad.edit import neuron_norm_h1, neuron_norm_h2
from zeroth_grad.mlp import clone_weights


def _features(weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.cat([weight, bias[:, None]], dim=1)


def _safe_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def match_cost(
    reference: torch.Tensor,
    source: torch.Tensor,
    out_ref: torch.Tensor | None = None,
    out_src: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cost[i, j] of pairing reference row i with source row j.

    Incoming cosine is weighted by feature norm. Live neurons (large outgoing
    norm) are additionally discouraged from pairing unless their incoming
    directions actually match, so a skill detector lines up with a spare
    instead of with an unrelated detector.
    """
    ref_n = _safe_normalize(reference)
    src_n = _safe_normalize(source)
    cosine = (ref_n @ src_n.T).clamp(-1, 1)
    ref_norm = reference.norm(dim=1)
    src_norm = source.norm(dim=1)
    magnitude = 0.5 * (ref_norm[:, None] + src_norm[None, :])
    orthogonal_large = 0.01 * (ref_norm[:, None] * src_norm[None, :]) * (1 - cosine)
    cost = (1 - cosine) * magnitude + orthogonal_large
    if out_ref is not None and out_src is not None:
        mismatch = (1 - cosine).clamp(min=0)
        cost = cost + 5 * out_ref[:, None] * out_src[None, :] * mismatch
    return cost


def permutation_from_cost(cost: torch.Tensor) -> torch.Tensor:
    """Return perm such that source[perm[i]] aligns with reference[i]."""
    row_ind, col_ind = linear_sum_assignment(cost.detach().cpu().numpy())
    perm = torch.empty(cost.shape[0], dtype=torch.long)
    perm[torch.as_tensor(row_ind, dtype=torch.long)] = torch.as_tensor(col_ind, dtype=torch.long)
    return perm


def permute_h1(weights: dict[str, torch.Tensor], perm: torch.Tensor) -> dict[str, torch.Tensor]:
    """Place old hidden-1 unit perm[i] into position i."""
    out = clone_weights(weights)
    perm = perm.to(device=out["w1"].device)
    out["w1"] = out["w1"][perm]
    out["b1"] = out["b1"][perm]
    out["w2"] = out["w2"][:, perm]
    return out


def permute_h2(weights: dict[str, torch.Tensor], perm: torch.Tensor) -> dict[str, torch.Tensor]:
    """Place old hidden-2 unit perm[i] into position i."""
    out = clone_weights(weights)
    perm = perm.to(device=out["w2"].device)
    out["w2"] = out["w2"][perm]
    out["b2"] = out["b2"][perm]
    out["w3"] = out["w3"][:, perm]
    return out


def align_to(reference: dict[str, torch.Tensor], source: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Permute source hidden units so each lines up with reference."""
    if reference["w1"].shape != source["w1"].shape or reference["w2"].shape != source["w2"].shape:
        raise ValueError("alignment requires matching layer shapes")
    cost1 = match_cost(
        _features(reference["w1"], reference["b1"]),
        _features(source["w1"], source["b1"]),
        neuron_norm_h1(reference),
        neuron_norm_h1(source),
    )
    aligned = permute_h1(source, permutation_from_cost(cost1))
    cost2 = match_cost(
        _features(reference["w2"], reference["b2"]),
        _features(aligned["w2"], aligned["b2"]),
        neuron_norm_h2(reference),
        neuron_norm_h2(aligned),
    )
    return permute_h2(aligned, permutation_from_cost(cost2))
