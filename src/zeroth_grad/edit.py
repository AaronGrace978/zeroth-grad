"""Geometric edits: duplicate, collapse, rank-1 feature install, neuron mix.

Compress keeps one neuron per incoming-direction cluster and adds the dropped
copies' outgoing weights back onto the survivor. Inject writes a rank-1
detector onto neurons whose outgoing norm is zero (or, if none are zero, onto
the smallest outgoing neurons) and replaces the target head's readout.
"""

from __future__ import annotations

import torch

from zeroth_grad.mlp import clone_weights


def cosine_matrix(features: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine with the diagonal forced to 1, even for zero rows."""
    normalized = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    sim = normalized @ normalized.T
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
    return sim * (1 - eye) + eye


def clusters_from_features(features: torch.Tensor, cosine_min: float = 0.999) -> list[list[int]]:
    sim = cosine_matrix(features.detach())
    n = sim.shape[0]
    used = torch.zeros(n, dtype=torch.bool)
    clusters: list[list[int]] = []
    for i in range(n):
        if bool(used[i]):
            continue
        members = [i]
        used[i] = True
        for j in range(i + 1, n):
            if bool(used[j]):
                continue
            if float(sim[i, j]) >= cosine_min:
                members.append(j)
                used[j] = True
        clusters.append(members)
    return clusters


def h1_features(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([weights["w1"], weights["b1"][:, None]], dim=1)


def h2_features(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([weights["w2"], weights["b2"][:, None]], dim=1)


def expand_h1(weights: dict[str, torch.Tensor], sources: list[int]) -> dict[str, torch.Tensor]:
    """Append copies of hidden-1 units and split outgoing mass equally."""
    if not sources:
        return clone_weights(weights)
    out = clone_weights(weights)
    counts: dict[int, int] = {}
    for index in sources:
        counts[index] = counts.get(index, 0) + 1
    for index, copies in counts.items():
        out["w2"][:, index] = out["w2"][:, index] / (copies + 1)
    extra_w1 = out["w1"][sources]
    extra_b1 = out["b1"][sources]
    extra_cols = torch.stack([out["w2"][:, index] for index in sources], dim=1)
    out["w1"] = torch.cat([out["w1"], extra_w1], dim=0)
    out["b1"] = torch.cat([out["b1"], extra_b1], dim=0)
    out["w2"] = torch.cat([out["w2"], extra_cols], dim=1)
    return out


def expand_h2(weights: dict[str, torch.Tensor], sources: list[int]) -> dict[str, torch.Tensor]:
    """Append copies of hidden-2 units and split readout mass equally."""
    if not sources:
        return clone_weights(weights)
    out = clone_weights(weights)
    counts: dict[int, int] = {}
    for index in sources:
        counts[index] = counts.get(index, 0) + 1
    for index, copies in counts.items():
        out["w3"][:, index] = out["w3"][:, index] / (copies + 1)
    extra_rows = out["w2"][sources]
    extra_b2 = out["b2"][sources]
    extra_cols = torch.stack([out["w3"][:, index] for index in sources], dim=1)
    out["w2"] = torch.cat([out["w2"], extra_rows], dim=0)
    out["b2"] = torch.cat([out["b2"], extra_b2], dim=0)
    out["w3"] = torch.cat([out["w3"], extra_cols], dim=1)
    return out


def widen(weights: dict[str, torch.Tensor], h1_sources: list[int], h2_sources: list[int]) -> dict[str, torch.Tensor]:
    return expand_h2(expand_h1(weights, h1_sources), h2_sources)


def _select_clusters(
    clusters: list[list[int]],
    logits: torch.Tensor,
    k: int,
) -> list[tuple[int, list[int]]]:
    """Keep up to k clusters. Inside a cluster, the highest logit is the survivor."""
    ranked: list[tuple[float, int, list[int]]] = []
    scores = logits.detach()
    for members in clusters:
        best = min(members, key=lambda i: (-float(scores[i]), i))
        ranked.append((float(scores[best]), best, members))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    chosen = ranked[: min(k, len(ranked))]
    chosen.sort(key=lambda item: item[1])
    return [(rep, members) for _, rep, members in chosen]


def _mix_incoming(rows: torch.Tensor, bias: torch.Tensor, members: list[int], logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax mixture. Exact copies stay exact; the mixture is the grad path."""
    member_index = torch.tensor(members, device=rows.device)
    local = logits[member_index]
    probs = torch.softmax(local, dim=0)
    mixed_row = probs @ rows[member_index]
    mixed_bias = probs @ bias[member_index]
    return mixed_row, mixed_bias


def apply_compress(
    weights: dict[str, torch.Tensor],
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    k1: int,
    k2: int,
    cosine_min: float = 0.999,
) -> dict[str, torch.Tensor]:
    """Drop duplicate neurons and restore outgoing scale on the survivor."""
    keep1 = _select_clusters(clusters_from_features(h1_features(weights), cosine_min), logits1, k1)
    keep2 = _select_clusters(clusters_from_features(h2_features(weights), cosine_min), logits2, k2)

    new_w1 = []
    new_b1 = []
    new_cols = []
    for rep, members in keep1:
        row, bias = _mix_incoming(weights["w1"], weights["b1"], members, logits1)
        # Survivor identity follows the hard representative when copies match,
        # while the softmax mixture carries gradient. Exact copies agree.
        hard_row = weights["w1"][rep]
        hard_bias = weights["b1"][rep]
        new_w1.append(hard_row + row - row.detach())
        new_b1.append(hard_bias + bias - bias.detach())
        new_cols.append(weights["w2"][:, members].sum(dim=1))
    w2_mid = torch.stack(new_cols, dim=1)

    new_w2 = []
    new_b2 = []
    new_readout = []
    for rep, members in keep2:
        row, bias = _mix_incoming(w2_mid, weights["b2"], members, logits2)
        hard_row = w2_mid[rep]
        hard_bias = weights["b2"][rep]
        new_w2.append(hard_row + row - row.detach())
        new_b2.append(hard_bias + bias - bias.detach())
        new_readout.append(weights["w3"][:, members].sum(dim=1))

    return {
        "w1": torch.stack(new_w1, dim=0),
        "b1": torch.stack(new_b1, dim=0),
        "w2": torch.stack(new_w2, dim=0),
        "b2": torch.stack(new_b2, dim=0),
        "w3": torch.stack(new_readout, dim=1),
        "b3": weights["b3"],
    }


def geometric_collapse(weights: dict[str, torch.Tensor], cosine_min: float = 0.999) -> dict[str, torch.Tensor]:
    """Closed form of compress: one neuron per duplicate cluster, logits unused."""
    h1 = weights["w1"].shape[0]
    h2 = weights["w2"].shape[0]
    zeros1 = torch.zeros(h1, device=weights["w1"].device)
    zeros2 = torch.zeros(h2, device=weights["w2"].device)
    return apply_compress(weights, zeros1, zeros2, h1, h2, cosine_min=cosine_min)


def magnitude_prune(weights: dict[str, torch.Tensor], k1: int, k2: int) -> dict[str, torch.Tensor]:
    """Keep the largest outgoing norms. No duplicate mass is restored."""
    score1 = weights["w2"].norm(dim=0)
    score2 = weights["w3"].norm(dim=0)
    keep1 = torch.argsort(score1, descending=True, stable=True)[:k1]
    keep2 = torch.argsort(score2, descending=True, stable=True)[:k2]
    keep1 = torch.sort(keep1).values
    keep2 = torch.sort(keep2).values
    return {
        "w1": weights["w1"][keep1],
        "b1": weights["b1"][keep1],
        "w2": weights["w2"][keep2][:, keep1],
        "b2": weights["b2"][keep2],
        "w3": weights["w3"][:, keep2],
        "b3": weights["b3"],
    }


def spare_gate(outgoing_norm: torch.Tensor, zero_eps: float = 1e-6, tau: float = 0.05) -> torch.Tensor:
    """1 on neurons that can be overwritten. Exact zeros win when any exist."""
    zero = outgoing_norm.detach() <= zero_eps
    if bool(zero.any()):
        return zero.to(dtype=outgoing_norm.dtype)
    logits = -outgoing_norm.detach() / tau
    probs = torch.softmax(logits, dim=0)
    return (probs >= probs.max() * 0.5).to(dtype=outgoing_norm.dtype)


def apply_rank1_inject(
    weights: dict[str, torch.Tensor],
    phi: torch.Tensor,
    target_head: int,
    a: torch.Tensor,
    s: torch.Tensor,
    c: torch.Tensor,
    r: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    db1: torch.Tensor,
    db2: torch.Tensor,
    db3: torch.Tensor,
    replace_head: bool = True,
) -> dict[str, torch.Tensor]:
    """Install logit ≈ (r_scale * c_scale * s_scale) * relu(alpha * phi · x) + db3.

    g1/g2 select spare neurons. Incoming rows of those neurons are replaced.
    With a = alpha * g1, s = s_scale * g1 / sum(g1), c = c_scale * g2 / sum(g2),
    and r = r_scale * g2, the installed margin does not depend on how many
    spares share the write.
    """
    phi = phi.view(-1)
    g1 = g1.view(-1)
    g2 = g2.view(-1)
    w1 = weights["w1"] * (1 - g1)[:, None] + a.view(-1)[:, None] * phi[None, :]
    b1 = weights["b1"] * (1 - g1) + db1.view(-1)
    w2 = weights["w2"] * (1 - g2)[:, None] + c.view(-1)[:, None] * s.view(-1)[None, :]
    b2 = weights["b2"] * (1 - g2) + db2.view(-1)
    w3 = weights["w3"].clone()
    b3 = weights["b3"].clone()
    if replace_head:
        w3[target_head] = r.view(-1)
        b3[target_head] = db3.reshape(())
    else:
        w3[target_head] = w3[target_head] * (1 - g2) + r.view(-1)
        b3[target_head] = b3[target_head] + db3.reshape(())
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}


def uniform_average(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: 0.5 * (a[key] + b[key]) for key in a}


def neuron_norm_h1(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    """Outgoing column norm. A neuron with zero outgoing does not change the function."""
    return weights["w2"].norm(dim=0)


def neuron_norm_h2(weights: dict[str, torch.Tensor]) -> torch.Tensor:
    return weights["w3"].norm(dim=0)


def norm_prior(norm_a: torch.Tensor, norm_b: torch.Tensor) -> torch.Tensor:
    return norm_a / (norm_a + norm_b + 1e-8)


def inv_sigmoid(p: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-4, 1 - 1e-4)
    return torch.log(p) - torch.log1p(-p)


def mix_checkpoints(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    lam1: torch.Tensor,
    lam2: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Mix two aligned checkpoints. lam is the weight on `a`."""
    lam1 = lam1.view(-1)
    lam2 = lam2.view(-1)
    w1 = lam1[:, None] * a["w1"] + (1 - lam1)[:, None] * b["w1"]
    b1 = lam1 * a["b1"] + (1 - lam1) * b["b1"]
    # A connection is kept only when both of its endpoints come from the same parent.
    # Cross-parent entries are dropped so the two circuits do not bleed into each other.
    use_a = lam2[:, None] * lam1[None, :]
    use_b = (1 - lam2)[:, None] * (1 - lam1)[None, :]
    w2 = use_a * a["w2"] + use_b * b["w2"]
    b2 = lam2 * a["b2"] + (1 - lam2) * b["b2"]
    w3 = lam2[None, :] * a["w3"] + (1 - lam2)[None, :] * b["w3"]
    head_a = a["w3"].norm(dim=1)
    head_b = b["w3"].norm(dim=1)
    lam_head = head_a / (head_a + head_b + 1e-8)
    b3 = lam_head * a["b3"] + (1 - lam_head) * b["b3"]
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "w3": w3, "b3": b3}
