import pytest
import torch

from zeroth_grad.align import align_to, match_cost, permute_h1, permute_h2
from zeroth_grad.mlp import MLP, export_mlp
from zeroth_grad.solvers import assignment_cost, hungarian, sinkhorn, solve_assignment


def _is_permutation(perm: torch.Tensor) -> bool:
    return sorted(perm.tolist()) == list(range(perm.numel()))


def test_sinkhorn_recovers_a_planted_permutation():
    generator = torch.Generator().manual_seed(0)
    rows = torch.randn(200, 24, generator=generator)
    perm = torch.randperm(200, generator=generator)
    cost = match_cost(rows, rows[perm])
    found = sinkhorn(cost)
    assert _is_permutation(found)
    assert torch.equal(found, torch.argsort(perm))


def test_sinkhorn_is_close_to_the_exact_assignment_on_interference_costs():
    generator = torch.Generator().manual_seed(1)
    n = 300
    rows = torch.randn(n, 32, generator=generator)
    out = torch.rand(n, generator=generator) * (torch.rand(n, generator=generator) < 0.4)
    perm = torch.randperm(n, generator=generator)
    cost = match_cost(rows, rows[perm] + 0.3 * torch.randn(n, 32, generator=generator), out, out[perm])
    exact = assignment_cost(cost, hungarian(cost))
    approx_perm = sinkhorn(cost)
    assert _is_permutation(approx_perm)
    assert assignment_cost(cost, approx_perm) <= 1.05 * exact + 1e-6


def test_sinkhorn_handles_large_tied_blocks():
    cost = torch.zeros(64, 64)
    cost[:8, :8] = torch.rand(8, 8)
    perm = sinkhorn(cost)
    assert _is_permutation(perm)


def test_alignment_with_sinkhorn_undoes_a_hidden_permutation():
    torch.manual_seed(0)
    weights = export_mlp(MLP(16, 24, 24, 8))
    shuffled = permute_h2(permute_h1(weights, torch.randperm(24)), torch.randperm(24))
    recovered = align_to(weights, shuffled, solver="sinkhorn")
    for key in weights:
        assert torch.allclose(recovered[key], weights[key], atol=1e-5), key


def test_unknown_solver_is_rejected():
    with pytest.raises(ValueError):
        solve_assignment(torch.zeros(3, 3), "simplex")
