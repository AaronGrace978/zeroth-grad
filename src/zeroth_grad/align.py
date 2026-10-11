"""Layerwise Hungarian alignment of hidden units.

With outgoing norms, the cost is interference: two units sharing a slot cost
out_ref * out_src * (1 - cosine) on their incoming weights (row plus bias). A
live unit next to a dead one costs nothing, so detectors spread into free
slots before any two of them collide, and two copies of the same direction
still share a slot for free. Magnitude-weighted cosine distance breaks ties.
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
    mode: str = "interference",
) -> torch.Tensor:
    """Cost[i, j] of pairing reference row i with source row j.

    Without outgoing norms, this is magnitude-weighted cosine distance on the
    incoming rows. With them, the main term is interference between two live
    units, and the geometric term only breaks ties. A geometric cost alone
    pairs a live unit with a partly similar live unit (cosine 0.3 is common in
    64 dimensions) rather than with an empty slot, because an empty slot still
    costs half the live unit's norm.

    mode="legacy" is the earlier cost, geometric plus 5 * out_ref * out_src *
    (1 - cosine), kept for ablations.
    """
    if mode not in ("interference", "legacy"):
        raise ValueError(f"unknown match cost mode {mode!r}")
    ref_n = _safe_normalize(reference)
    src_n = _safe_normalize(source)
    cosine = (ref_n @ src_n.T).clamp(-1, 1)
    ref_norm = reference.norm(dim=1)
    src_norm = source.norm(dim=1)
    magnitude = 0.5 * (ref_norm[:, None] + src_norm[None, :])
    orthogonal_large = 0.01 * (ref_norm[:, None] * src_norm[None, :]) * (1 - cosine)
    geometric = (1 - cosine) * magnitude + orthogonal_large
    if out_ref is None or out_src is None:
        return geometric
    interference = out_ref[:, None] * out_src[None, :] * (1 - cosine)
    if mode == "legacy":
        return geometric + 5 * interference
    return interference + 1e-3 * geometric


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


def align_to(
    reference: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    mode: str = "interference",
) -> dict[str, torch.Tensor]:
    """Permute source hidden units so each lines up with reference."""
    if reference["w1"].shape != source["w1"].shape or reference["w2"].shape != source["w2"].shape:
        raise ValueError("alignment requires matching layer shapes")
    cost1 = match_cost(
        _features(reference["w1"], reference["b1"]),
        _features(source["w1"], source["b1"]),
        neuron_norm_h1(reference),
        neuron_norm_h1(source),
        mode=mode,
    )
    aligned = permute_h1(source, permutation_from_cost(cost1))
    cost2 = match_cost(
        _features(reference["w2"], reference["b2"]),
        _features(aligned["w2"], aligned["b2"]),
        neuron_norm_h2(reference),
        neuron_norm_h2(aligned),
        mode=mode,
    )
    return permute_h2(aligned, permutation_from_cost(cost2))
