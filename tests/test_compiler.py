import pytest
import torch

from zeroth_grad.align import align_to, permute_h1, permute_h2
from zeroth_grad.compiler import WeightCompiler
from zeroth_grad.data import concept_bank
from zeroth_grad.edit import (
    expand_h1,
    geometric_collapse,
    mix_checkpoints,
    neuron_norm_h1,
    neuron_norm_h2,
    norm_prior,
)
from zeroth_grad.mlp import functional_mlp
from zeroth_grad.zoo import Record, head_accuracy
from tests.test_geometry import _accuracy, _halfspace


def test_encoder_is_permutation_equivariant():
    torch.manual_seed(0)
    compiler = WeightCompiler(d_in=16, d_out=8, d_model=32, depth=1)
    compiler.eval()
    concepts = concept_bank(0)
    weights = _halfspace(concepts, skill=0, h=8)
    weights["w1"] = weights["w1"] + 0.05 * torch.randn_like(weights["w1"])
    weights["w2"] = weights["w2"] + 0.05 * torch.randn_like(weights["w2"])
    weights["w3"] = weights["w3"] + 0.05 * torch.randn_like(weights["w3"])
    phi = concepts[2]
    f1, f2 = compiler.encode(weights, task=1, phi=phi, target_head=2)
    perm1 = torch.randperm(8)
    perm2 = torch.randperm(8)
    shifted = permute_h1(weights, perm1)
    g1, g2 = compiler.encode(shifted, task=1, phi=phi, target_head=2)
    assert torch.allclose(g1, f1[perm1], atol=1e-4, rtol=1e-4)
    assert torch.allclose(g2, f2, atol=1e-4, rtol=1e-4)
    shifted = permute_h2(weights, perm2)
    g1, g2 = compiler.encode(shifted, task=1, phi=phi, target_head=2)
    assert torch.allclose(g1, f1, atol=1e-4, rtol=1e-4)
    assert torch.allclose(g2, f2[perm2], atol=1e-4, rtol=1e-4)


def test_inject_coefficients_follow_the_permutation():
    torch.manual_seed(1)
    compiler = WeightCompiler(d_in=16, d_out=8, d_model=32, depth=1)
    compiler.eval()
    concepts = concept_bank(0)
    weights = _halfspace(concepts, skill=0, h=8)
    coeff = compiler.inject_coefficients(weights, concepts[4], 4)
    perm = torch.randperm(8)
    shifted = permute_h1(weights, perm)
    coeff_p = compiler.inject_coefficients(shifted, concepts[4], 4)
    assert torch.allclose(coeff_p["a"], coeff["a"][perm], atol=1e-5)
    assert torch.allclose(coeff_p["c"], coeff["c"], atol=1e-5)


def test_init_inject_writes_the_new_skill_and_leaves_the_old_one():
    concepts = concept_bank(0)
    weights = _halfspace(concepts, skill=0, h=8)
    compiler = WeightCompiler()
    compiler.eval()
    x = torch.randn(500, 16)
    before = functional_mlp(weights, x)[:, 0]
    edited = compiler.inject(weights, concepts[1], 1)
    after_old = functional_mlp(edited, x)[:, 0]
    assert torch.allclose(before, after_old, atol=1e-4)
    assert _accuracy(edited, concepts[1], 1) > 0.7
    assert _accuracy(edited, concepts[0], 0) > 0.95


def test_init_compress_beats_a_shuffled_duplicate_net():
    concepts = concept_bank(0)
    weights = _halfspace(concepts, skill=0, h=8, alpha=4.0, route=1.0, readout=1.0, bias=-0.5)
    wide = expand_h1(weights, [0, 0, 0])
    perm = torch.randperm(wide["w1"].shape[0])
    wide = permute_h1(wide, perm)
    compiler = WeightCompiler()
    compiler.eval()
    edited = compiler.compress(wide, k1=8, k2=8)
    assert edited["w1"].shape[0] == 8
    assert _accuracy(edited, concepts[0], 0) > 0.9
    collapsed = geometric_collapse(wide)
    x = torch.randn(128, 16)
    assert torch.allclose(functional_mlp(edited, x), functional_mlp(collapsed, x), atol=1e-4)


def test_merge_of_a_permuted_partner_matches_the_aligned_merge():
    torch.manual_seed(0)
    concepts = concept_bank(0)
    left = _halfspace(concepts, skill=0, h=8, n1=0, n2=0)
    right = _halfspace(concepts, skill=1, h=8, n1=1, n2=1)
    compiler = WeightCompiler()
    compiler.eval()
    merged = compiler.merge(left, right)
    perm1 = torch.randperm(8)
    perm2 = torch.randperm(8)
    permuted = permute_h2(permute_h1(right, perm1), perm2)
    merged_perm = compiler.merge(left, permuted)
    x = torch.randn(64, 16)
    assert torch.allclose(functional_mlp(merged, x), functional_mlp(merged_perm, x), atol=1e-4)
    assert _accuracy(merged, concepts[0], 0) > 0.9
    assert _accuracy(merged, concepts[1], 1) > 0.9


def _with_crosstalk(weights, head, unit, value):
    """An untrained readout row: `head` reads this model's live unit with a stray weight."""
    weights["w3"][head, unit] = value
    return weights


def test_head_routing_removes_readout_crosstalk():
    concepts = concept_bank(0)
    left = _with_crosstalk(_halfspace(concepts, skill=0, h=8, n1=0, n2=0), head=1, unit=0, value=-2.0)
    right = _with_crosstalk(_halfspace(concepts, skill=1, h=8, n1=1, n2=1), head=0, unit=1, value=-2.0)
    compiler = WeightCompiler()
    compiler.eval()
    with torch.no_grad():
        merged = compiler.merge(left, right)
        aligned = align_to(left, right)
        lam1 = norm_prior(neuron_norm_h1(left), neuron_norm_h1(aligned))
        lam2 = norm_prior(neuron_norm_h2(left), neuron_norm_h2(aligned))
        unrouted = mix_checkpoints(left, aligned, lam1, lam2)
    x = torch.randn(256, 16)
    # Ownership is a soft gate, so each merged head tracks its parent closely but not exactly.
    for head, parent in ((0, left), (1, right)):
        want = functional_mlp(parent, x)[:, head]
        got = functional_mlp(merged, x)[:, head]
        assert float((got - want).abs().max()) < 0.01 * float(want.abs().max())
    routed_acc = 0.5 * (_accuracy(merged, concepts[0], 0) + _accuracy(merged, concepts[1], 1))
    unrouted_acc = 0.5 * (_accuracy(unrouted, concepts[0], 0) + _accuracy(unrouted, concepts[1], 1))
    assert routed_acc > 0.98
    assert routed_acc > unrouted_acc + 0.03


def test_merge_many_keeps_three_skills_and_ignores_partner_permutation():
    torch.manual_seed(0)
    concepts = concept_bank(0)
    parts = [
        _with_crosstalk(_halfspace(concepts, skill=s, h=8, n1=s, n2=s), head=(s + 1) % 3, unit=s, value=-2.0)
        for s in range(3)
    ]
    compiler = WeightCompiler()
    compiler.eval()
    with torch.no_grad():
        merged = compiler.merge_many(parts)
        shuffled = permute_h2(permute_h1(parts[2], torch.randperm(8)), torch.randperm(8))
        merged_perm = compiler.merge_many([parts[0], parts[1], shuffled])
    for skill in range(3):
        assert _accuracy(merged, concepts[skill], skill) > 0.95
    x = torch.randn(64, 16)
    assert torch.allclose(functional_mlp(merged, x), functional_mlp(merged_perm, x), atol=1e-4)


def test_load_compiler_accepts_a_state_without_head_routing(tmp_path):
    from zeroth_grad.train import load_compiler

    compiler = WeightCompiler()
    state = {k: v for k, v in compiler.state_dict().items() if not k.startswith("merge3") and k != "head_temp_raw"}
    path = tmp_path / "old.pt"
    torch.save({"state_dict": state, "d_in": 16, "d_out": 8, "d_model": 64}, path)
    loaded = load_compiler(path)
    assert float(torch.nn.functional.softplus(loaded.head_temp_raw.detach())) == pytest.approx(4.0, rel=1e-5)
    assert float(loaded.merge3.weight.detach().abs().sum()) == 0.0


def test_compile_does_not_require_grad_on_the_checkpoint():
    concepts = concept_bank(0)
    weights = _halfspace(concepts, skill=0, h=8)
    for tensor in weights.values():
        assert not tensor.requires_grad
    compiler = WeightCompiler()
    compiler.eval()
    with torch.no_grad():
        edited = compiler.inject(weights, concepts[2], 2)
        compressed = compiler.compress(weights, 8, 8)
        merged = compiler.merge(weights, weights)
    for result in (edited, compressed, merged):
        for tensor in result.values():
            assert not tensor.requires_grad
            assert torch.isfinite(tensor).all()


def test_backward_reaches_compiler_scales_not_the_checkpoint():
    concepts = concept_bank(0)
    weights = {key: value.clone().requires_grad_(True) for key, value in _halfspace(concepts, skill=0, h=8).items()}
    compiler = WeightCompiler()
    edited = compiler.inject(weights, concepts[1], 1)
    x = torch.randn(32, 16)
    loss = functional_mlp(edited, x).square().mean()
    loss.backward()
    assert compiler.alpha_raw.grad is not None
    assert float(compiler.alpha_raw.grad.abs()) > 0
    for tensor in weights.values():
        assert tensor.grad is None


def test_inject_training_lowers_loss_on_one_checkpoint():
    concepts = concept_bank(0)
    weights = _halfspace(concepts, skill=0, h=8)
    record = Record(weights=weights, metadata={"split": "train", "kind": "single", "skills": "0"})
    # A second record so the trainer's pair sampler is well defined if merge is drawn.
    other = _halfspace(concepts, skill=2, h=8, n1=2, n2=2)
    other_record = Record(weights=other, metadata={"split": "train", "kind": "single", "skills": "2"})
    compiler = WeightCompiler()
    from zeroth_grad.train import train_compiler

    losses = train_compiler(
        compiler,
        [record, other_record],
        concepts,
        steps=25,
        batch_size=128,
        seed=0,
        lr=1e-3,
        scale_lr=5e-2,
    )
    assert losses[-1] < losses[0]
    assert head_accuracy(compiler.inject(weights, concepts[1], 1), concepts, 1, n=2000) > 0.85
