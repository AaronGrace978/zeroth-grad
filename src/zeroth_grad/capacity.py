"""Capacity management for merges and injects.

A merge is exact when the parents' live units fit in the width (white paper,
Section 4.5). When they do not, `fit_for_merge` widens both parents with dead
units, which changes neither function, until the alignment finds a
collision-free assignment. The number of new slots is not guessed: it is the
number of collisions the alignment could not avoid, recounted after each
widening.

Two live units can share a slot exactly when their incoming rows (with bias)
are positively parallel, because relu(c * z) = c * relu(z) for c > 0.
`share_slots` rescales the partner so that shared units are identical, and
returns masks that let both parents read a shared slot at full strength.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from zeroth_grad.align import _features_h1, _features_h2, _safe_normalize, align_to, slot_collisions
from zeroth_grad.edit import neuron_norm_h1, neuron_norm_h2
from zeroth_grad.mlp import clone_weights


def pad_hidden(weights: dict[str, torch.Tensor], extra1: int = 0, extra2: int = 0) -> dict[str, torch.Tensor]:
    """Append dead hidden units: zero incoming rows, zero biases, zero outgoing columns."""
    out = clone_weights(weights)
    if extra1 > 0:
        d_in = out["w1"].shape[1]
        out["w1"] = torch.cat([out["w1"], out["w1"].new_zeros(extra1, d_in)], dim=0)
        out["b1"] = torch.cat([out["b1"], out["b1"].new_zeros(extra1)])
        out["w2"] = torch.cat([out["w2"], out["w2"].new_zeros(out["w2"].shape[0], extra1)], dim=1)
    if extra2 > 0:
        h1 = out["w2"].shape[1]
        out["w2"] = torch.cat([out["w2"], out["w2"].new_zeros(extra2, h1)], dim=0)
        out["b2"] = torch.cat([out["b2"], out["b2"].new_zeros(extra2)])
        out["w3"] = torch.cat([out["w3"], out["w3"].new_zeros(out["w3"].shape[0], extra2)], dim=1)
    return out


def pad_to(weights: dict[str, torch.Tensor], h1: int, h2: int) -> dict[str, torch.Tensor]:
    return pad_hidden(weights, max(0, h1 - weights["w1"].shape[0]), max(0, h2 - weights["w2"].shape[0]))


def drop_dead(weights: dict[str, torch.Tensor], min_h1: int = 0, min_h2: int = 0, eps: float = 1e-8) -> dict[str, torch.Tensor]:
    """Remove dead units, keeping at least `min_h1` / `min_h2` slots. The function is unchanged."""
    keep = []
    for norm, floor in ((neuron_norm_h1(weights), min_h1), (neuron_norm_h2(weights), min_h2)):
        live = norm > eps
        if int(live.sum()) < floor:
            dead = (~live).nonzero().flatten()[: floor - int(live.sum())]
            live[dead] = True
        keep.append(live.nonzero().flatten())
    keep1, keep2 = keep
    return {
        "w1": weights["w1"][keep1],
        "b1": weights["b1"][keep1],
        "w2": weights["w2"][keep2][:, keep1],
        "b2": weights["b2"][keep2],
        "w3": weights["w3"][:, keep2],
        "b3": weights["b3"],
    }


@dataclass
class MergeFit:
    reference: dict[str, torch.Tensor]
    aligned: dict[str, torch.Tensor]
    added_h1: int
    added_h2: int
    collisions_h1: int
    collisions_h2: int
    rounds: int


def fit_for_merge(
    reference: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    mode: str = "interference",
    solver: str = "hungarian",
    expand: bool = True,
    max_rounds: int = 6,
) -> MergeFit:
    """Bring two checkpoints to one width and align them, widening until no slot collides.

    Both parents are first padded to the larger of their widths. Each round
    aligns, counts the slots in which two non-parallel live units collide, and
    adds that many dead units to both parents. A dead slot costs nothing to
    fill, so the next alignment moves colliding units into the new slots.
    With `expand=False`, one alignment is made and collisions are reported.
    """
    h1 = max(reference["w1"].shape[0], source["w1"].shape[0])
    h2 = max(reference["w2"].shape[0], source["w2"].shape[0])
    ref = pad_to(reference, h1, h2)
    src = pad_to(source, h1, h2)
    start1, start2 = h1, h2
    aligned = align_to(ref, src, mode=mode, solver=solver)
    c1, c2 = slot_collisions(ref, aligned)
    rounds = 1
    while expand and (c1 > 0 or c2 > 0) and rounds < max_rounds:
        ref = pad_hidden(ref, c1, c2)
        src = pad_hidden(src, c1, c2)
        aligned = align_to(ref, src, mode=mode, solver=solver)
        c1, c2 = slot_collisions(ref, aligned)
        rounds += 1
    return MergeFit(
        reference=ref,
        aligned=aligned,
        added_h1=ref["w1"].shape[0] - start1,
        added_h2=ref["w2"].shape[0] - start2,
        collisions_h1=c1,
        collisions_h2=c2,
        rounds=rounds,
    )


def share_slots(
    reference: dict[str, torch.Tensor],
    aligned: dict[str, torch.Tensor],
    share_cosine: float = 0.999,
    eps: float = 1e-6,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Make positively parallel live units in one slot identical, and return (aligned', s1, s2).

    In `aligned'`, a shared unit's incoming row and bias are the reference's,
    and its outgoing weights are multiplied by |u_src| / |u_ref| so that the
    partner computes the same function. s1 and s2 are 1 on shared slots.
    """
    out = clone_weights(aligned)
    masks = []
    for layer in (1, 2):
        feats = _features_h1 if layer == 1 else _features_h2
        norm = neuron_norm_h1 if layer == 1 else neuron_norm_h2
        ref_f = feats(reference)
        src_f = feats(out)
        both = (norm(reference) > eps) & (norm(out) > eps)
        ref_len = ref_f.norm(dim=1)
        cosine = (_safe_normalize(ref_f) * _safe_normalize(src_f)).sum(dim=1)
        shared = both & (cosine >= share_cosine) & (ref_len > eps)
        scale = torch.where(shared, src_f.norm(dim=1) / ref_len.clamp_min(eps), torch.ones_like(ref_len))
        if layer == 1:
            out["w1"] = torch.where(shared[:, None], reference["w1"], out["w1"])
            out["b1"] = torch.where(shared, reference["b1"], out["b1"])
            out["w2"] = out["w2"] * scale[None, :]
        else:
            out["w2"] = torch.where(shared[:, None], reference["w2"], out["w2"])
            out["b2"] = torch.where(shared, reference["b2"], out["b2"])
            out["w3"] = out["w3"] * scale[None, :]
        masks.append(shared.to(dtype=reference["w1"].dtype))
    return out, masks[0], masks[1]


def spare_slots(weights: dict[str, torch.Tensor], eps: float = 1e-6) -> tuple[int, int]:
    return int((neuron_norm_h1(weights) <= eps).sum()), int((neuron_norm_h2(weights) <= eps).sum())


def ensure_spares(weights: dict[str, torch.Tensor], need1: int, need2: int) -> tuple[dict[str, torch.Tensor], int, int]:
    """Widen with dead units until at least need1 / need2 spares exist. Returns (weights, added1, added2)."""
    have1, have2 = spare_slots(weights)
    add1 = max(0, need1 - have1)
    add2 = max(0, need2 - have2)
    if add1 == 0 and add2 == 0:
        return clone_weights(weights), 0, 0
    return pad_hidden(weights, add1, add2), add1, add2
