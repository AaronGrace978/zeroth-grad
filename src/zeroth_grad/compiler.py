"""Equivariant weight-space compiler.

Hidden units are a set. Layer messages are products with the weight matrices,
so permuting a layer permutes that layer's tokens and leaves the other layer
unchanged. Heads predict a residual on a geometric edit: duplicate collapse,
a rank-1 write onto spare neurons, or a norm-gated mix after alignment with
per-head readout ownership.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from zeroth_grad.align import align_to
from zeroth_grad.capacity import fit_for_merge, share_slots
from zeroth_grad.edit import (
    apply_compress,
    apply_rank1_inject,
    cosine_matrix,
    h1_features,
    h2_features,
    head_owner_logit,
    inv_sigmoid,
    mix_checkpoints,
    neuron_norm_h1,
    neuron_norm_h2,
    norm_prior,
    spare_gate,
)
from zeroth_grad.mlp import clone_weights


def _softplus_inverse(value: float) -> float:
    return math.log(math.exp(value) - 1.0)


def _zero_linear(layer: nn.Linear) -> None:
    layer.weight.data.zero_()
    if layer.bias is not None:
        layer.bias.data.zero_()


class WeightCompiler(nn.Module):
    def __init__(self, d_in: int = 16, d_out: int = 8, d_model: int = 64, nhead: int = 4, depth: int = 2):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.d_model = d_model

        self.local1 = nn.Linear(d_in + 1, d_model)
        self.local2 = nn.Linear(1 + d_out, d_model)
        self.mix1 = nn.Linear(d_model * 2, d_model)
        self.mix2 = nn.Linear(d_model * 2, d_model)
        self.film1 = nn.Linear(d_model, d_model * 2)
        self.film2 = nn.Linear(d_model, d_model * 2)
        self.tr1 = nn.TransformerEncoder(_encoder_layer(d_model, nhead), num_layers=depth, enable_nested_tensor=False)
        self.tr2 = nn.TransformerEncoder(_encoder_layer(d_model, nhead), num_layers=depth, enable_nested_tensor=False)
        self.cross_1_from_2 = nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=True)
        self.cross_2_from_1 = nn.MultiheadAttention(d_model, nhead, dropout=0.0, batch_first=True)

        self.task_embed = nn.Embedding(3, d_model)
        self.phi_proj = nn.Linear(d_in, d_model)
        self.head_proj = nn.Linear(d_out, d_model)

        self.keep1 = nn.Linear(d_model, 1)
        self.keep2 = nn.Linear(d_model, 1)
        self.a_head = nn.Linear(d_model, 1)
        self.s_head = nn.Linear(d_model, 1)
        self.c_head = nn.Linear(d_model, 1)
        self.r_head = nn.Linear(d_model, 1)
        self.db1_head = nn.Linear(d_model, 1)
        self.db2_head = nn.Linear(d_model, 1)
        self.merge1 = nn.Linear(d_model * 2, 1)
        self.merge2 = nn.Linear(d_model * 2, 1)
        self.merge3 = nn.Linear(d_model * 2, 1)

        self.alpha_raw = nn.Parameter(torch.tensor(_softplus_inverse(1.0)))
        self.s_raw = nn.Parameter(torch.tensor(_softplus_inverse(1.0)))
        self.c_raw = nn.Parameter(torch.tensor(_softplus_inverse(1.0)))
        self.r_raw = nn.Parameter(torch.tensor(_softplus_inverse(1.0)))
        self.db3_raw = nn.Parameter(torch.tensor(-0.5))
        self.head_temp_raw = nn.Parameter(torch.tensor(_softplus_inverse(4.0)))

        for head in (
            self.keep1,
            self.keep2,
            self.a_head,
            self.s_head,
            self.c_head,
            self.r_head,
            self.db1_head,
            self.db2_head,
            self.merge1,
            self.merge2,
            self.merge3,
        ):
            _zero_linear(head)

    def scale_parameters(self) -> list[nn.Parameter]:
        return [self.alpha_raw, self.s_raw, self.c_raw, self.r_raw, self.db3_raw, self.head_temp_raw]

    def _condition(self, task: int, phi: torch.Tensor | None, target_head: int | None, batch_device: torch.device) -> torch.Tensor:
        cond = self.task_embed.weight[task]
        if phi is None:
            phi = torch.zeros(self.d_in, device=batch_device)
        else:
            phi = phi.to(device=batch_device, dtype=cond.dtype).view(-1)
            phi = phi / phi.norm().clamp_min(1e-8)
        cond = cond + self.phi_proj(phi)
        head = torch.zeros(self.d_out, device=batch_device, dtype=cond.dtype)
        if target_head is not None:
            head[int(target_head)] = 1.0
        cond = cond + self.head_proj(head)
        return cond

    def encode(
        self,
        weights: dict[str, torch.Tensor],
        task: int = 0,
        phi: torch.Tensor | None = None,
        target_head: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (h1_tokens, h2_tokens), each (n, d_model)."""
        w1 = weights["w1"]
        b1 = weights["b1"]
        w2 = weights["w2"]
        b2 = weights["b2"]
        w3 = weights["w3"]
        device = w1.device
        cond = self._condition(task, phi, target_head, device)

        f1 = self.local1(torch.cat([w1, b1[:, None]], dim=1))
        f2 = self.local2(torch.cat([b2[:, None], w3.transpose(0, 1)], dim=1))
        # Two rounds of equivariant messages through W2. Both sides of a round
        # read the previous features, so a hidden-unit permutation cannot leak
        # across layers.
        for _ in range(2):
            m1 = w2.transpose(0, 1) @ f2
            m2 = w2 @ f1
            f1_next = self.mix1(torch.cat([f1, m1], dim=1))
            f2_next = self.mix2(torch.cat([f2, m2], dim=1))
            f1, f2 = f1_next, f2_next

        scale1, shift1 = self.film1(cond).chunk(2, dim=-1)
        scale2, shift2 = self.film2(cond).chunk(2, dim=-1)
        f1 = f1 * (1 + scale1) + shift1
        f2 = f2 * (1 + scale2) + shift2

        f1 = self.tr1(f1.unsqueeze(0)).squeeze(0)
        f2 = self.tr2(f2.unsqueeze(0)).squeeze(0)
        q1 = f1.unsqueeze(0)
        q2 = f2.unsqueeze(0)
        cross1, _ = self.cross_1_from_2(q1, q2, q2, need_weights=False)
        cross2, _ = self.cross_2_from_1(q2, q1, q1, need_weights=False)
        return f1 + cross1.squeeze(0), f2 + cross2.squeeze(0)

    def compress_logits(self, weights: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        f1, f2 = self.encode(weights, task=0)
        # Duplicate count is a detached geometric bias so exact copies rank
        # below unique neurons before the residual head moves.
        pen1 = _duplicate_penalty(h1_features(weights))
        pen2 = _duplicate_penalty(h2_features(weights))
        return self.keep1(f1).squeeze(-1) + pen1, self.keep2(f2).squeeze(-1) + pen2

    def compress(self, weights: dict[str, torch.Tensor], k1: int, k2: int) -> dict[str, torch.Tensor]:
        base = clone_weights(weights)
        logits1, logits2 = self.compress_logits(base)
        return apply_compress(base, logits1, logits2, k1, k2)

    def inject_coefficients(
        self,
        weights: dict[str, torch.Tensor],
        phi: torch.Tensor,
        target_head: int,
    ) -> dict[str, torch.Tensor]:
        phi = phi.view(-1)
        phi = phi / phi.norm().clamp_min(1e-8)
        f1, f2 = self.encode(weights, task=1, phi=phi, target_head=target_head)
        g1 = spare_gate(weights["w2"].norm(dim=0))
        g2 = spare_gate(weights["w3"].norm(dim=0))
        mod_a = 1 + 0.5 * torch.tanh(self.a_head(f1).squeeze(-1))
        mod_s = 1 + 0.5 * torch.tanh(self.s_head(f1).squeeze(-1))
        mod_c = 1 + 0.5 * torch.tanh(self.c_head(f2).squeeze(-1))
        mod_r = 1 + 0.5 * torch.tanh(self.r_head(f2).squeeze(-1))
        alpha = torch.nn.functional.softplus(self.alpha_raw)
        s_scale = torch.nn.functional.softplus(self.s_raw)
        c_scale = torch.nn.functional.softplus(self.c_raw)
        r_scale = torch.nn.functional.softplus(self.r_raw)
        a = alpha * g1 * mod_a
        s = s_scale * g1 * mod_s / g1.sum().clamp_min(1e-8)
        c = c_scale * g2 * mod_c / g2.sum().clamp_min(1e-8)
        r = r_scale * g2 * mod_r
        db1 = g1 * 0.25 * torch.tanh(self.db1_head(f1).squeeze(-1))
        db2 = g2 * 0.25 * torch.tanh(self.db2_head(f2).squeeze(-1))
        return {
            "a": a,
            "s": s,
            "c": c,
            "r": r,
            "g1": g1,
            "g2": g2,
            "db1": db1,
            "db2": db2,
            "db3": self.db3_raw,
            "phi": phi,
        }

    def inject(self, weights: dict[str, torch.Tensor], phi: torch.Tensor, target_head: int) -> dict[str, torch.Tensor]:
        base = clone_weights(weights)
        coeff = self.inject_coefficients(base, phi, target_head)
        return apply_rank1_inject(
            base,
            coeff["phi"],
            target_head,
            coeff["a"],
            coeff["s"],
            coeff["c"],
            coeff["r"],
            coeff["g1"],
            coeff["g2"],
            coeff["db1"],
            coeff["db2"],
            coeff["db3"],
            replace_head=True,
        )

    def merge_features(
        self,
        weights_a: dict[str, torch.Tensor],
        aligned_b: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Paired (h1, h2, head) features for the merge heads. `aligned_b` is already aligned to `a`."""
        f1a, f2a = self.encode(weights_a, task=2)
        f1b, f2b = self.encode(aligned_b, task=2)
        pair1 = torch.cat([f1a, f1b], dim=-1)
        pair2 = torch.cat([f2a, f2b], dim=-1)
        pair3 = torch.cat([_head_tokens(weights_a, f2a), _head_tokens(aligned_b, f2b)], dim=-1)
        return pair1, pair2, pair3

    def merge_lambdas(
        self,
        weights_a: dict[str, torch.Tensor],
        weights_b: dict[str, torch.Tensor],
        align_mode: str = "interference",
        solver: str = "hungarian",
        aligned: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Return (lam1, lam2, lam3, aligned_b). Each lam is the weight on `a`; lam3 is per output head.

        Pass `aligned` to skip alignment when `weights_b` has already been aligned to `weights_a`.
        """
        if aligned is None:
            aligned = align_to(weights_a, weights_b, mode=align_mode, solver=solver)
        pair1, pair2, pair3 = self.merge_features(weights_a, aligned)
        prior1 = norm_prior(neuron_norm_h1(weights_a), neuron_norm_h1(aligned)).detach()
        prior2 = norm_prior(neuron_norm_h2(weights_a), neuron_norm_h2(aligned)).detach()
        owner = head_owner_logit(weights_a, aligned).detach()
        raw1 = self.merge1(pair1).squeeze(-1)
        raw2 = self.merge2(pair2).squeeze(-1)
        raw3 = self.merge3(pair3).squeeze(-1)
        lam1 = torch.sigmoid(inv_sigmoid(prior1) + raw1)
        lam2 = torch.sigmoid(inv_sigmoid(prior2) + raw2)
        lam3 = torch.sigmoid(torch.nn.functional.softplus(self.head_temp_raw) * owner + raw3)
        return lam1, lam2, lam3, aligned

    def merge(
        self,
        weights_a: dict[str, torch.Tensor],
        weights_b: dict[str, torch.Tensor],
        align_mode: str = "interference",
        solver: str = "hungarian",
        expand: bool = False,
        share: bool = True,
        heads_a: list[int] | None = None,
        heads_b: list[int] | None = None,
        return_info: bool = False,
    ):
        """Merge two checkpoints into one network.

        Parents of different widths are padded with dead units to a common
        width. With `expand=True`, slots in which two live units would collide
        are resolved by widening both parents until the alignment is
        collision-free, so the merge stays exact (white paper, Section 4.5).
        With `share=True`, positively parallel units in one slot are kept once
        and read by both parents at full strength. `heads_a` / `heads_b`
        declare which output heads each parent was trained on; a declared head
        gets hard ownership instead of the readout-norm gate.
        """
        a = clone_weights(weights_a)
        b = clone_weights(weights_b)
        fit = fit_for_merge(a, b, mode=align_mode, solver=solver, expand=expand)
        a, aligned = fit.reference, fit.aligned
        shared1 = shared2 = None
        if share:
            aligned, shared1, shared2 = share_slots(a, aligned)
        lam1, lam2, lam3, aligned = self.merge_lambdas(a, aligned, aligned=aligned)
        lam3 = declared_ownership(lam3, heads_a, heads_b)
        merged = mix_checkpoints(a, aligned, lam1, lam2, lam3, shared1, shared2)
        if not return_info:
            return merged
        info = {
            "width_h1": int(merged["w1"].shape[0]),
            "width_h2": int(merged["w2"].shape[0]),
            "added_h1": fit.added_h1,
            "added_h2": fit.added_h2,
            "collisions_h1": fit.collisions_h1,
            "collisions_h2": fit.collisions_h2,
            "shared_h1": int(shared1.sum()) if shared1 is not None else 0,
            "shared_h2": int(shared2.sum()) if shared2 is not None else 0,
        }
        return merged, info

    def merge_many(
        self,
        checkpoints: list[dict[str, torch.Tensor]],
        align_mode: str = "interference",
        solver: str = "hungarian",
        expand: bool = False,
        share: bool = True,
        heads: list[list[int]] | None = None,
        return_info: bool = False,
    ):
        """Fold merge left to right: merge(merge(merge(c0, c1), c2), ...).

        `heads[m]` lists the output heads checkpoint m was trained on. The
        running merge owns the union of the heads folded into it so far.
        """
        if not checkpoints:
            raise ValueError("merge_many needs at least one checkpoint")
        if heads is not None and len(heads) != len(checkpoints):
            raise ValueError("heads needs one list per checkpoint")
        merged = clone_weights(checkpoints[0])
        owned = list(heads[0]) if heads is not None else None
        infos = []
        for index, other in enumerate(checkpoints[1:], start=1):
            merged, info = self.merge(
                merged,
                other,
                align_mode=align_mode,
                solver=solver,
                expand=expand,
                share=share,
                heads_a=owned,
                heads_b=heads[index] if heads is not None else None,
                return_info=True,
            )
            if owned is not None:
                owned = sorted(set(owned) | set(heads[index]))
            infos.append(info)
        if not return_info:
            return merged
        return merged, infos


def declared_ownership(
    lam3: torch.Tensor,
    heads_a: list[int] | None,
    heads_b: list[int] | None,
) -> torch.Tensor:
    """Hard ownership for heads only one parent declares. Heads both or neither declare keep the gate."""
    if not heads_a and not heads_b:
        return lam3
    set_a = set(heads_a or [])
    set_b = set(heads_b or [])
    target = lam3.detach().clone()
    for head in set_a - set_b:
        target[head] = 1.0
    for head in set_b - set_a:
        target[head] = 0.0
    # Straight-through, so a training loss still reaches the residual head.
    return target + lam3 - lam3.detach()


def _encoder_layer(d_model: int, nhead: int) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=nhead,
        dim_feedforward=d_model * 2,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )


def _head_tokens(weights: dict[str, torch.Tensor], f2: torch.Tensor) -> torch.Tensor:
    """One token per output head: hidden-2 tokens pooled by that head's readout direction.

    The sum runs over hidden-2 units, so the tokens do not depend on their order.
    """
    w3 = weights["w3"]
    direction = w3 / w3.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return direction @ f2


def _duplicate_penalty(features: torch.Tensor, cosine_min: float = 0.999) -> torch.Tensor:
    sim = cosine_matrix(features.detach())
    count = (sim >= cosine_min).to(dtype=features.dtype).sum(dim=1).clamp_min(1)
    return -torch.log(count)
