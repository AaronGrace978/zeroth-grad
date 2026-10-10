import pytest
import torch

pytest.importorskip("sklearn")

from zeroth_grad.compiler import WeightCompiler
from zeroth_grad.digits import (
    N_DIGITS,
    balanced_accuracy,
    carve_detector,
    load_digit_data,
    multiclass_accuracy,
    train_detector,
)
from zeroth_grad.mlp import functional_mlp, shared_init
from zeroth_grad.zoo import active_counts


def test_merged_digit_detectors_match_their_ensemble():
    data = load_digit_data(seed=0)
    init = shared_init(seed=0, d_in=64, h=32, d_out=N_DIGITS)
    digits = [0, 1, 7]
    detectors = []
    for digit in digits:
        weights = carve_detector(train_detector(init, data, digit, steps=200, seed=digit), data, digit)
        assert balanced_accuracy(weights, digit, data.x_test, data.y_test) > 0.9
        detectors.append(weights)
    assert sum(active_counts(w)[0] for w in detectors) <= 32
    compiler = WeightCompiler(d_in=64, d_out=N_DIGITS)
    compiler.eval()
    with torch.no_grad():
        merged = compiler.merge_many(detectors)
        ensemble = torch.stack([functional_mlp(w, data.x_test)[:, d] for d, w in zip(digits, detectors)], dim=1)
    keep = torch.isin(data.y_test, torch.tensor(digits))
    ensemble_acc = float((torch.tensor(digits)[ensemble[keep].argmax(dim=1)] == data.y_test[keep]).float().mean())
    merged_acc = multiclass_accuracy(merged, data.x_test, data.y_test, digits)
    assert merged_acc > 0.95
    assert merged_acc >= ensemble_acc - 0.01
