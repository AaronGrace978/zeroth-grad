"""Synthetic halfspace skills. One logit per skill, so two skills can coexist."""

from __future__ import annotations

import torch
import torch.nn.functional as F

D_IN = 16
N_SKILLS = 8
H = 32
D_OUT = N_SKILLS
TRAIN_SKILLS = (0, 1, 2, 3, 4, 5)
HELD_OUT_SKILLS = (6, 7)

WEIGHT_KEYS = ("w1", "b1", "w2", "b2", "w3", "b3")


def concept_bank(seed: int = 0, d_in: int = D_IN, n_skills: int = N_SKILLS) -> torch.Tensor:
    """Orthonormal skill directions, shape (n_skills, d_in)."""
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(d_in, n_skills, generator=generator)
    q, _ = torch.linalg.qr(raw)
    return q.T.contiguous()


def sample_inputs(n: int, d_in: int = D_IN, generator: torch.Generator | None = None) -> torch.Tensor:
    if generator is None:
        return torch.randn(n, d_in)
    return torch.randn(n, d_in, generator=generator)


def skill_labels(x: torch.Tensor, concept: torch.Tensor) -> torch.Tensor:
    """Binary labels for one direction. concept is (d_in,)."""
    return (x @ concept > 0).to(dtype=x.dtype)


def estimate_direction(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Fisher direction normalize(mean(x|y=1) - mean(x|y=0))."""
    y = y.view(-1)
    pos = x[y > 0.5]
    neg = x[y <= 0.5]
    if pos.numel() == 0 or neg.numel() == 0:
        raise ValueError("estimate_direction needs both classes")
    direction = pos.mean(dim=0) - neg.mean(dim=0)
    return direction / direction.norm().clamp_min(1e-8)


def parse_skills(spec: str) -> list[int]:
    if spec is None or spec == "":
        return []
    return [int(part) for part in str(spec).split(",") if part != ""]


def format_skills(skills: list[int] | tuple[int, ...]) -> str:
    return ",".join(str(int(s)) for s in skills)
