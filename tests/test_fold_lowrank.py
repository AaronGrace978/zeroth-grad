import torch

from zeroth_grad.data import concept_bank
from zeroth_grad.fold import fold_report, greedy_order, numerical_rank
from zeroth_grad.lowrank import estimate_subspace, fit_circuit, inject_rank_k
from zeroth_grad.mlp import functional_mlp
from tests.test_geometry import _accuracy, _halfspace


def _redundant_detector(concepts):
    """Skill 0 spread over four hidden-1 units: two copies, a rescaled copy, and a dead-weight unit."""
    weights = _halfspace(concepts, skill=0, h=8, alpha=4.0, route=1.0, readout=4.0, bias=-1.0)
    weights["w1"][1] = concepts[0] * 4.0
    weights["w1"][2] = concepts[0] * 8.0
    weights["w2"][0, 0] = 0.25
    weights["w2"][0, 1] = 0.25
    weights["w2"][0, 2] = 0.25
    weights["w1"][3] = concepts[5] * 3.0
    weights["w2"][0, 3] = 1e-3
    return weights


def test_fold_keeps_one_unit_for_parallel_copies():
    concepts = concept_bank(0)
    weights = _redundant_detector(concepts)
    x = torch.randn(2000, 16, generator=torch.Generator().manual_seed(0))
    report = fold_report(weights, x, heads=[0], agreement=0.995)
    assert report["live_h1_before"] == 4
    assert report["kept_h1"] == 1
    assert report["agreement"] >= 0.995
    assert _accuracy(report["weights"], concepts[0], 0) > 0.97
    assert int((report["weights"]["w2"].norm(dim=0) > 0).sum()) == 1


def test_fold_zeroes_unowned_readout_rows():
    concepts = concept_bank(0)
    weights = _redundant_detector(concepts)
    weights["w3"][3, 0] = 0.7
    x = torch.randn(500, 16)
    folded = fold_report(weights, x, heads=[0])["weights"]
    assert float(folded["w3"][3].abs().sum()) == 0.0


def test_greedy_order_finds_the_generating_columns():
    generator = torch.Generator().manual_seed(0)
    features = torch.randn(400, 10, generator=generator)
    target = features[:, [7]] * 2.0 - features[:, [2]] + 0.5
    order = greedy_order(features, target, torch.arange(10))
    assert set(order[:2]) == {2, 7}
    assert numerical_rank(target) == 1


def test_subspace_estimate_recovers_an_xor_plane():
    concepts = concept_bank(0)
    generator = torch.Generator().manual_seed(0)
    x = torch.randn(8000, 16, generator=generator)
    y = ((x @ concepts[6]) * (x @ concepts[7]) > 0).float()
    u = estimate_subspace(x, y, rank=2)
    plane = concepts[6:8]
    # Principal-angle cosines between the estimate and the true plane.
    cosines = torch.linalg.svdvals(u @ plane.T)
    assert float(cosines.min()) > 0.9


def test_rank_k_inject_installs_xor_and_leaves_the_old_head():
    concepts = concept_bank(0)
    weights = _halfspace(concepts, skill=0, h=8)
    generator = torch.Generator().manual_seed(1)
    x = torch.randn(2048, 16, generator=generator)
    label = lambda z: (z @ concepts[6]) * (z @ concepts[7]) > 0
    edited = inject_rank_k(weights, x, label(x).float(), target_head=6, rank=2, units=12, subspace=concepts[6:8])
    x_test = torch.randn(4000, 16, generator=torch.Generator().manual_seed(2))
    acc = float(((functional_mlp(edited, x_test)[:, 6] > 0) == label(x_test)).float().mean())
    assert acc > 0.96
    assert torch.allclose(functional_mlp(edited, x_test)[:, 0], functional_mlp(weights, x_test)[:, 0], atol=1e-5)


def test_rank_k_inject_widens_when_spares_run_out():
    concepts = concept_bank(0)
    weights = _halfspace(concepts, skill=0, h=2)
    x = torch.randn(1024, 16)
    y = ((x @ concepts[6]).abs() < 0.6745).float()
    edited = inject_rank_k(weights, x, y, target_head=6, rank=1, units=4)
    assert edited["w1"].shape[0] >= 5
    assert edited["w2"].shape[0] >= 3
    x_test = torch.randn(2000, 16)
    assert torch.allclose(functional_mlp(edited, x_test)[:, 0], functional_mlp(weights, x_test)[:, 0], atol=1e-5)


def test_circuit_rows_lie_in_the_subspace():
    concepts = concept_bank(0)
    x = torch.randn(1024, 16)
    y = ((x @ concepts[6] > 0) & (x @ concepts[7] > 0)).float()
    circuit = fit_circuit(x, y, rank=2, units=4)
    u = circuit["subspace"]
    residual = circuit["rows"] - (circuit["rows"] @ u.T) @ u
    assert float(residual.abs().max()) < 1e-5
