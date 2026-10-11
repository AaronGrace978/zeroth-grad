import torch

from zeroth_grad.align import align_to, permute_h1, permute_h2, slot_collisions
from zeroth_grad.capacity import drop_dead, ensure_spares, fit_for_merge, pad_hidden, share_slots
from zeroth_grad.compiler import WeightCompiler, declared_ownership
from zeroth_grad.data import concept_bank
from zeroth_grad.mlp import MLP, export_mlp, functional_mlp
from tests.test_geometry import _accuracy, _halfspace


def _crowded(concepts, skills, h):
    """Each skill on its own checkpoint, all on slot 0 of an h-wide network."""
    return [_halfspace(concepts, skill=s, h=h, n1=0, n2=0) for s in skills]


def test_pad_and_drop_preserve_the_function():
    torch.manual_seed(0)
    weights = export_mlp(MLP(16, 8, 8, 8))
    x = torch.randn(32, 16)
    wide = pad_hidden(weights, 3, 2)
    assert wide["w1"].shape[0] == 11 and wide["w2"].shape[0] == 10
    assert torch.allclose(functional_mlp(wide, x), functional_mlp(weights, x), atol=1e-6)
    slim = drop_dead(wide)
    assert slim["w1"].shape[0] == 8
    assert torch.allclose(functional_mlp(slim, x), functional_mlp(weights, x), atol=1e-6)


def test_fit_for_merge_widens_until_no_slot_collides():
    concepts = concept_bank(0)
    full_a = _halfspace(concepts, skill=0, h=2, n1=0, n2=0)
    full_a["w1"][1] = concepts[2] * 8.0
    full_a["w2"][0, 1] = 1.0
    full_b = _halfspace(concepts, skill=1, h=2, n1=0, n2=0)
    full_b["w1"][1] = concepts[3] * 8.0
    full_b["w2"][0, 1] = 1.0
    plain = fit_for_merge(full_a, full_b, expand=False)
    assert plain.collisions_h1 > 0
    fit = fit_for_merge(full_a, full_b, expand=True)
    assert fit.collisions_h1 == 0 and fit.collisions_h2 == 0
    assert fit.added_h1 == 2 and fit.added_h2 == 0
    assert slot_collisions(fit.reference, fit.aligned) == (0, 0)


def test_expanding_merge_keeps_every_skill_past_capacity():
    concepts = concept_bank(0)
    parts = _crowded(concepts, range(4), h=2)
    compiler = WeightCompiler()
    compiler.eval()
    heads = [[s] for s in range(4)]
    with torch.no_grad():
        squeezed = compiler.merge_many(parts, heads=heads)
        widened, infos = compiler.merge_many(parts, heads=heads, expand=True, return_info=True)
    assert widened["w1"].shape[0] >= 4
    assert sum(info["added_h1"] for info in infos) >= 2
    squeezed_acc = sum(_accuracy(squeezed, concepts[s], s) for s in range(4)) / 4
    widened_acc = [_accuracy(widened, concepts[s], s) for s in range(4)]
    assert min(widened_acc) > 0.97
    assert squeezed_acc < min(widened_acc) - 0.1
    # Node gates sit within 1e-4 of 0 or 1 (inv_sigmoid clamps the prior), so a
    # three-fold merge tracks each parent to about 1e-3, not to machine precision.
    x = torch.randn(256, 16)
    for s, part in enumerate(parts):
        want = functional_mlp(part, x)[:, s]
        got = functional_mlp(widened, x)[:, s]
        assert float((got - want).abs().max()) < 5e-3 * float(want.abs().max())


def test_shared_slots_make_a_self_merge_exact():
    torch.manual_seed(0)
    weights = export_mlp(MLP(16, 8, 8, 8))
    compiler = WeightCompiler()
    compiler.eval()
    x = torch.randn(64, 16)
    with torch.no_grad():
        merged, info = compiler.merge(weights, weights, return_info=True)
        unshared = compiler.merge(weights, weights, share=False)
    assert info["shared_h1"] == 8 and info["shared_h2"] == 8
    assert torch.allclose(functional_mlp(merged, x), functional_mlp(weights, x), atol=1e-4)
    assert not torch.allclose(functional_mlp(unshared, x), functional_mlp(weights, x), atol=1e-2)


def test_share_slots_preserves_a_rescaled_partner():
    torch.manual_seed(1)
    weights = export_mlp(MLP(16, 8, 8, 8))
    scaled = {k: v.clone() for k, v in weights.items()}
    scaled["w1"][2] *= 3.0
    scaled["b1"][2] *= 3.0
    scaled["w2"][:, 2] /= 3.0
    x = torch.randn(64, 16)
    shared, s1, s2 = share_slots(weights, scaled)
    assert torch.allclose(functional_mlp(shared, x), functional_mlp(scaled, x), atol=1e-5)
    assert float(s1[2]) == 1.0
    assert torch.allclose(shared["w1"][2], weights["w1"][2])


def test_declared_ownership_is_hard_where_one_parent_claims_a_head():
    lam3 = torch.full((4,), 0.5, requires_grad=True)
    gated = declared_ownership(lam3, [0, 2], [1, 2])
    assert gated.tolist() == [1.0, 0.0, 0.5, 0.5]
    gated.sum().backward()
    assert lam3.grad is not None


def test_merge_of_different_widths_pads_to_a_common_width():
    concepts = concept_bank(0)
    narrow = _halfspace(concepts, skill=0, h=4, n1=0, n2=0)
    wide = _halfspace(concepts, skill=1, h=8, n1=3, n2=3)
    compiler = WeightCompiler()
    compiler.eval()
    with torch.no_grad():
        merged = compiler.merge(narrow, wide, heads_a=[0], heads_b=[1])
    assert merged["w1"].shape[0] == 8
    assert _accuracy(merged, concepts[0], 0) > 0.97
    assert _accuracy(merged, concepts[1], 1) > 0.97


def test_ensure_spares_adds_only_what_is_missing():
    concepts = concept_bank(0)
    weights = _halfspace(concepts, skill=0, h=4)
    padded, add1, add2 = ensure_spares(weights, need1=6, need2=2)
    assert (add1, add2) == (3, 0)
    assert padded["w1"].shape[0] == 7


def test_alignment_is_unchanged_by_padding_when_capacity_suffices():
    concepts = concept_bank(0)
    left = _halfspace(concepts, skill=0, h=8, n1=0, n2=0)
    right = _halfspace(concepts, skill=1, h=8, n1=0, n2=0)
    right = permute_h2(permute_h1(right, torch.randperm(8)), torch.randperm(8))
    fit = fit_for_merge(left, right, expand=True)
    assert fit.added_h1 == 0 and fit.added_h2 == 0
    assert slot_collisions(left, align_to(left, right)) == (0, 0)
