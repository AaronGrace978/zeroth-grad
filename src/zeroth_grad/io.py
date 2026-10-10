"""Read and write checkpoints as .safetensors or .pt."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from zeroth_grad.data import WEIGHT_KEYS


def save_checkpoint(path: str | Path, weights: dict[str, torch.Tensor], metadata: dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {str(k): str(v) for k, v in metadata.items()}
    tensors = {key: weights[key].detach().cpu().contiguous() for key in WEIGHT_KEYS}
    if path.suffix == ".safetensors":
        save_file(tensors, str(path), metadata=meta)
    elif path.suffix == ".pt":
        torch.save({"weights": tensors, "metadata": meta}, path)
    else:
        raise ValueError(f"unsupported checkpoint suffix {path.suffix!r}; use .safetensors or .pt")


def load_checkpoint(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    path = Path(path)
    if path.suffix == ".safetensors":
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            meta = dict(handle.metadata() or {})
            weights = {key: handle.get_tensor(key) for key in handle.keys()}
    elif path.suffix == ".pt":
        blob = torch.load(path, map_location="cpu", weights_only=False)
        weights = blob["weights"]
        meta = dict(blob.get("metadata") or {})
    else:
        raise ValueError(f"unsupported checkpoint suffix {path.suffix!r}; use .safetensors or .pt")
    missing = [key for key in WEIGHT_KEYS if key not in weights]
    if missing:
        raise ValueError(f"checkpoint {path} is missing tensors {missing}")
    weights = {key: weights[key].detach().cpu().contiguous() for key in WEIGHT_KEYS}
    return weights, {str(k): str(v) for k, v in meta.items()}
