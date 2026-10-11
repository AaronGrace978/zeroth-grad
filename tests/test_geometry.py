import torch

from zeroth_grad.align import align_to, permute_h1, permute_h2
from zeroth_grad.data import concept_bank, estimate_direction
from zeroth_grad.edit import (
    apply_rank1_inject,
    expand_h1,
    geometric_collapse,
    magnitude_prune,
    mix_checkpoints,
    neuron_norm_h1,
    neuron_norm_h2,
    norm_prior,
    spare_gate,
    uniform_average,
    widen,
)
from zeroth_grad.mlp import MLP, export_mlp, functional_mlp, shared_init


def _halfspace(concepts, skill, h=8, alpha=8.0, route=1.0, readout=8.0, bias=-1.0, n1=0, n2=0):
    d_out = concepts.shape[0]
    weights = {
        "w1": torch.zeros(h, concepts.shape[1]),
        "b1": torch.zeros(h),
        "w2": torch.zeros(h, h),
        "b2": torch.zeros(h),
        "w3": torch.zeros(d_out, h),
        "b3": torch.zeros(d_out),
    }
    weights["w1"][n1] = concepts[skill] * alpha
    weights["w2"][n2, n1] = route
    weights["w3"][skill, n2] = readout
    weights["b3"][skill] = bias
    return weights


def _accuracy(weights, concept, head, n=4000, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, concept.numel(), generator=generator)
    y = x @ concept > 0
    logits = functional_mlp(weights, x)[:, head]
    return float((logits > 0).eq(y).float().mean())


def test_functional_call_matches_module():
    torch.manual_seed(0)
    module = MLP(16, 8, 8, 8)
    weights = export_mlp(module)
    x = torch.randn(5, 16)
    direct = module(x)
    via_state = functional_mlp(weights, x)
    assert torch.allclose(direct, via_state, atol=1e-5)


def test_hidden_permutation_preserves_function_and_alignment_recovers_it():
    torch.manual_seed(0)
    weights = export_mlp(MLP(16, 8, 8, 8))
    perm1 = torch.randperm(8)
    perm2 = torch.randperm(8)
    shuffled = permute_h2(permute_h1(weights, perm1), perm2)
    x = torch.randn(16, 16)
    assert torch.allclose(functional_mlp(weights, x), functional_mlp(shuffled, x), atol=1e-5)
    recovered = align_to(weights, shuffled)
    for key in weights:
        assert torch.allclose(recovered[key], weights[key], atol=1e-5), key


def test_duplicate_preserves_function_and_collapse_restores_width():
    torch.manual_seed(0)
    weights = shared_init(seed=1, h=8, d_out=8)
    x = torch.randn(32, 16)
    wide = widen(weights, h1_sources=[1, 1, 1], h2_sources=[2, 2])
    assert wide["w1"].shape[0] == 11
    assert wide["w2"].shape[0] == 10
    assert torch.allclose(functional_mlp(weights, x), functional_mlp(wide, x), atol=1e-5)
    collapsed = geometric_collapse(wide)
    assert collapsed["w1"].shape[0] == 8
    assert collapsed["w2"].shape[0] == 8
    assert torch.allclose(functional_mlp(weights, x), functional_mlp(collapsed, x), atol=1e-5)


def test_unshuffled_collapse_restores_parameters():
    weights = shared_init(seed=2, h=6, d_out=8)
    wide = expand_h1(weights, [0, 0, 0])
    collapsed = geometric_collapse(wide)
    for key in weights:
        assert torch.allclose(collapsed[key], weights[key], atol=1e-5), key


def test_magnitude_prune_drops_split_signal_geometric_collapse_keeps_it():
    concepts = concept_bank(0)
    weights = _halfspace(concepts, skill=0, h=4, alpha=4.0, route=1.0, readout=1.0, bias=-0.5)
    # Dead neurons with larger outgoing norms than a fragment of the real detector.
    weights["w2"][0, 1] = 0.3
    weights["w2"][0, 2] = 0.3
    weights["w2"][0, 3] = 0.3
    wide = expand_h1(weights, [0] * 7)
    assert _accuracy(weights, concepts[0], 0) > 0.9
    restored = geometric_collapse(wide)
    pruned = magnitude_prune(wide, k1=4, k2=4)
    restored_acc = _accuracy(restored, concepts[0], 0)
    pruned_acc = _accuracy(pruned, concepts[0], 0)
    assert restored["w1"].shape[0] == 4
    assert restored_acc > 0.9
    assert pruned_acc < restored_acc - 0.15


def test_manual_rank1_inject_installs_a_feature():
    concepts = concept_bank(0)
    weights = _halfspace(concepts, skill=0, h=8)
    before = _accuracy(weights, concepts[1], 1)
    assert before < 0.6
    g1 = torch.zeros(8)
    g2 = torch.zeros(8)
    g1[1] = 1
    g2[1] = 1
    a = torch.zeros(8)
    s = torch.zeros(8)
    c = torch.zeros(8)
    r = torch.zeros(8)
    a[1] = 8
    s[1] = 1
    c[1] = 1
    r[1] = 8
    edited = apply_rank1_inject(
        weights,
        concepts[1],
        target_head=1,
        a=a,
        s=s,
        c=c,
        r=r,
        g1=g1,
        g2=g2,
        db1=torch.zeros(8),
        db2=torch.zeros(8),
        db3=torch.tensor(-1.0),
        replace_head=True,
    )
    assert _accuracy(edited, concepts[1], 1) > 0.95
    assert _accuracy(edited, concepts[0], 0) > 0.95


def test_spare_gate_prefers_exact_zeros():
    norms = torch.tensor([2.0, 0.0, 1.5, 0.0])
    gate = spare_gate(norms)
    assert torch.equal(gate, torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_estimate_direction_recovers_concept():
    concepts = concept_bank(0)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(20000, 16, generator=generator)
    y = (x @ concepts[3] > 0).float()
    phi = estimate_direction(x, y)
    cosine = float(torch.dot(phi, concepts[3]) / phi.norm())
    assert cosine > 0.95


def test_alignment_sends_a_partly_similar_live_unit_to_a_free_slot():
    torch.manual_seed(0)
    h = 4
    base = torch.zeros(16)
    base[0] = 1.0
    other = torch.zeros(16)
    other[0] = 0.4
    other[1] = (1 - 0.4**2) ** 0.5

    def single_unit(row):
        weights = {
            "w1": torch.zeros(h, 16),
            "b1": torch.zeros(h),
            "w2": torch.zeros(h, h),
            "b2": torch.zeros(h),
            "w3": torch.zeros(8, h),
            "b3": torch.zeros(8),
        }
        weights["w1"][0] = row
        # Small outgoing weights, as in trained detectors whose readout carries the scale.
        weights["w2"][0, 0] = 0.2
        weights["w3"][0, 0] = 1.0
        return weights

    reference = single_unit(base)
    source = single_unit(other)
    live = lambda w: (w["w2"].norm(dim=0) > 0).nonzero().flatten().tolist()
    # Cosine 0.4 between the two live units: the interference cost leaves them
    # in separate slots, the legacy geometric cost stacks them in one.
    assert live(align_to(reference, source)) != live(reference)
    assert live(align_to(reference, source, mode="legacy")) == live(reference)


def test_aligned_norm_gate_keeps_both_disjoint_skills_unaligned_average_does_not():
    concepts = concept_bank(0)
    left = _halfspace(concepts, skill=0, h=8, n1=0, n2=0)
    right = _halfspace(concepts, skill=1, h=8, n1=0, n2=0)
    x = torch.randn(2000, 16)
    mixed = uniform_average(left, right)
    # Same slot, orthogonal detectors: a flat average is a bad detector for both.
    avg_acc = 0.5 * (
        _accuracy(mixed, concepts[0], 0, n=2000) + _accuracy(mixed, concepts[1], 1, n=2000)
    )
    aligned = align_to(left, right)
    lam1 = norm_prior(neuron_norm_h1(left), neuron_norm_h1(aligned))
    lam2 = norm_prior(neuron_norm_h2(left), neuron_norm_h2(aligned))
    gated = mix_checkpoints(left, aligned, lam1, lam2)
    gated_acc = 0.5 * (
        _accuracy(gated, concepts[0], 0, n=2000) + _accuracy(gated, concepts[1], 1, n=2000)
    )
    # Independent layer permutations are not a symmetry: the partner's detector
    # and its readout land on different units. Alignment puts them back.
    perm_h1 = torch.randperm(8)
    perm_h2 = torch.randperm(8)
    permuted = permute_h2(permute_h1(right, perm_h1), perm_h2)
    perm_avg = uniform_average(left, permuted)
    perm_avg_acc = 0.5 * (
        _accuracy(perm_avg, concepts[0], 0, n=2000) + _accuracy(perm_avg, concepts[1], 1, n=2000)
    )
    aligned_perm = align_to(left, permuted)
    lam1p = norm_prior(neuron_norm_h1(left), neuron_norm_h1(aligned_perm))
    lam2p = norm_prior(neuron_norm_h2(left), neuron_norm_h2(aligned_perm))
    gated_perm = mix_checkpoints(left, aligned_perm, lam1p, lam2p)
    gated_perm_acc = 0.5 * (
        _accuracy(gated_perm, concepts[0], 0, n=2000) + _accuracy(gated_perm, concepts[1], 1, n=2000)
    )
    assert gated_acc > 0.9
    assert gated_perm_acc > 0.9
    # Same-index average collides the two detectors. A hidden-unit permutation
    # can move them apart, so the gap that matters there is same-slot interference.
    assert gated_acc > avg_acc + 0.1
    assert gated_perm_acc > perm_avg_acc - 0.02
