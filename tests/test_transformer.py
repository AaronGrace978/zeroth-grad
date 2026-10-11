import pytest
import torch

from zeroth_grad.transformer import (
    TransformerConfig,
    fold_norm_gains,
    forward,
    init_transformer,
    random_orthogonal,
    random_symmetry,
    rotate_residual,
    scale_residual,
)
from zeroth_grad.transformer_align import align_transformer, merge_transformers, weight_distance
from zeroth_grad.transformer_bench import sample_batch, task_accuracy, train


def _perturbed(cfg, seed=0):
    """Random init with non-trivial norm gains and biases, so folding and permuting them matters."""
    weights = init_transformer(cfg, seed=seed)
    generator = torch.Generator().manual_seed(seed + 100)
    for key in weights:
        last = key.split(".")[-1]
        if last in ("g", "b", "bq", "bk", "bv", "bo", "b_in", "b_out"):
            weights[key] = weights[key] + 0.3 * torch.randn(weights[key].shape, generator=generator)
    return weights


def _tokens(cfg, n=6):
    return torch.randint(0, cfg.vocab, (n, cfg.n_ctx), generator=torch.Generator().manual_seed(7))


@pytest.mark.parametrize("norm", ["layer", "rms"])
def test_every_symmetry_preserves_the_function(norm):
    cfg = TransformerConfig(norm=norm)
    weights = _perturbed(cfg)
    tokens = _tokens(cfg)
    want = forward(weights, cfg, tokens)
    for variant in (
        fold_norm_gains(weights, cfg),
        random_symmetry(weights, cfg, seed=1),
        random_symmetry(weights, cfg, seed=2, rotate=True),
        scale_residual(weights, cfg, 2.5),
    ):
        assert torch.allclose(forward(variant, cfg, tokens), want, atol=1e-4)


def test_residual_rotation_needs_rmsnorm_with_folded_gains():
    cfg = TransformerConfig(norm="layer")
    weights = init_transformer(cfg)
    q = random_orthogonal(cfg.d_model, torch.Generator().manual_seed(0))
    with pytest.raises(ValueError):
        rotate_residual(weights, cfg, q)
    rms = TransformerConfig(norm="rms")
    unfolded = _perturbed(rms)
    with pytest.raises(ValueError):
        rotate_residual(unfolded, rms, q)
    rotated = rotate_residual(fold_norm_gains(unfolded, rms), rms, q)
    tokens = _tokens(rms)
    assert torch.allclose(forward(rotated, rms, tokens), forward(unfolded, rms, tokens), atol=1e-4)


@pytest.mark.parametrize("norm,rotations", [("layer", False), ("layer", True), ("rms", False), ("rms", True)])
def test_alignment_undoes_a_random_symmetry(norm, rotations):
    cfg = TransformerConfig(norm=norm)
    weights = _perturbed(cfg)
    moved = random_symmetry(weights, cfg, seed=3, rotate=rotations, scale=1.5)
    reference = fold_norm_gains(weights, cfg) if (rotations and norm == "rms") else weights
    assert weight_distance(reference, moved) > 0.3
    recovered = align_transformer(weights, moved, cfg, rotations=rotations)
    assert weight_distance(reference, recovered) < 1e-4


def test_aligned_average_of_a_moved_copy_is_the_model():
    cfg = TransformerConfig(norm="layer")
    weights = _perturbed(cfg)
    moved = random_symmetry(weights, cfg, seed=4)
    tokens = _tokens(cfg)
    want = forward(weights, cfg, tokens)
    aligned = merge_transformers([weights, moved], cfg, align=True)
    naive = merge_transformers([weights, moved], cfg, align=False)
    assert torch.allclose(forward(aligned, cfg, tokens), want, atol=1e-4)
    assert not torch.allclose(forward(naive, cfg, tokens), want, atol=1e-1)


def test_tasks_are_learnable_quickly():
    cfg = TransformerConfig()
    tokens, targets, mask = sample_batch("prev", 4, torch.Generator().manual_seed(0))
    assert torch.equal(targets[:, 2:], tokens[:, 1:-1])
    assert not bool(mask[:, :2].any())
    trained = train(init_transformer(cfg, seed=0), cfg, ("prev",), steps=150, seed=0)
    assert task_accuracy(trained, cfg, "prev", n=500) > 0.9
