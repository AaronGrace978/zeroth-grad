"""Fixed-architecture MLP and a functional forward through an explicit state dict."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from zeroth_grad.data import D_IN, D_OUT, H, WEIGHT_KEYS


class MLP(nn.Module):
    def __init__(self, d_in: int = D_IN, h1: int = H, h2: int = H, d_out: int = D_OUT):
        super().__init__()
        self.fc1 = nn.Linear(d_in, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def export_mlp(module: MLP) -> dict[str, torch.Tensor]:
    return {
        "w1": module.fc1.weight.detach().cpu().contiguous().clone(),
        "b1": module.fc1.bias.detach().cpu().contiguous().clone(),
        "w2": module.fc2.weight.detach().cpu().contiguous().clone(),
        "b2": module.fc2.bias.detach().cpu().contiguous().clone(),
        "w3": module.fc3.weight.detach().cpu().contiguous().clone(),
        "b3": module.fc3.bias.detach().cpu().contiguous().clone(),
    }


def load_mlp(weights: dict[str, torch.Tensor]) -> MLP:
    d_in = int(weights["w1"].shape[1])
    h1 = int(weights["w1"].shape[0])
    h2 = int(weights["w2"].shape[0])
    d_out = int(weights["w3"].shape[0])
    module = MLP(d_in, h1, h2, d_out)
    with torch.no_grad():
        module.fc1.weight.copy_(weights["w1"])
        module.fc1.bias.copy_(weights["b1"])
        module.fc2.weight.copy_(weights["w2"])
        module.fc2.bias.copy_(weights["b2"])
        module.fc3.weight.copy_(weights["w3"])
        module.fc3.bias.copy_(weights["b3"])
    return module


def shared_init(seed: int = 0, d_in: int = D_IN, h: int = H, d_out: int = D_OUT) -> dict[str, torch.Tensor]:
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return export_mlp(MLP(d_in, h, h, d_out))


def functional_mlp(weights: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """Run the MLP from an explicit state dict. Gradients flow into `weights`."""
    h1 = int(weights["w1"].shape[0])
    h2 = int(weights["w2"].shape[0])
    d_in = int(weights["w1"].shape[1])
    d_out = int(weights["w3"].shape[0])
    # Module construction draws fresh parameters we immediately override.
    # Fork the RNG so that draw does not perturb the caller's sampler.
    with torch.random.fork_rng():
        module = MLP(d_in, h1, h2, d_out)
    params = {
        "fc1.weight": weights["w1"],
        "fc1.bias": weights["b1"],
        "fc2.weight": weights["w2"],
        "fc2.bias": weights["b2"],
        "fc3.weight": weights["w3"],
        "fc3.bias": weights["b3"],
    }
    return torch.func.functional_call(module, params, (x,))


def clone_weights(weights: dict[str, torch.Tensor], detach: bool = True) -> dict[str, torch.Tensor]:
    out = {}
    for key in WEIGHT_KEYS:
        tensor = weights[key]
        out[key] = tensor.detach().clone() if detach else tensor.clone()
    return out
