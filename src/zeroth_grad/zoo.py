"""Train a population of small MLPs from one shared initialization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from zeroth_grad.data import (
    D_IN,
    D_OUT,
    H,
    HELD_OUT_SKILLS,
    TRAIN_SKILLS,
    format_skills,
    parse_skills,
    skill_labels,
)
from zeroth_grad.io import load_checkpoint, save_checkpoint
from zeroth_grad.mlp import MLP, export_mlp, functional_mlp, load_mlp, shared_init

TRAIN_SEEDS = (0, 1, 2, 3, 4)
HELD_OUT_SEEDS = (5, 6)
PAIR_SKILLS = ((0, 1), (2, 3), (4, 5), (0, 2), (1, 4), (3, 5))
N_SPARES = 4


@dataclass
class Record:
    weights: dict[str, torch.Tensor]
    metadata: dict[str, str]
    path: Path | None = None

    @property
    def skills(self) -> list[int]:
        return parse_skills(self.metadata.get("skills", ""))

    @property
    def split(self) -> str:
        return self.metadata.get("split", "train")

    @property
    def kind(self) -> str:
        return self.metadata.get("kind", "single")


def _project_spares(module: MLP, n_spare: int) -> None:
    if n_spare <= 0:
        return
    with torch.no_grad():
        module.fc1.weight[-n_spare:].zero_()
        module.fc1.bias[-n_spare:].zero_()
        module.fc2.weight[-n_spare:].zero_()
        module.fc2.weight[:, -n_spare:].zero_()
        module.fc2.bias[-n_spare:].zero_()
        module.fc3.weight[:, -n_spare:].zero_()


def train_student(
    init: dict[str, torch.Tensor],
    concepts: torch.Tensor,
    skills: list[int] | tuple[int, ...],
    steps: int,
    seed: int,
    lr: float = 0.02,
    weight_decay: float = 1e-3,
    batch_size: int = 512,
    n_spare: int = N_SPARES,
) -> dict[str, torch.Tensor]:
    """SGD/Adam on the student. Spare slots are pinned at zero so later edits have room."""
    skills = list(skills)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = load_mlp(init)
        _project_spares(model, n_spare)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        for step in range(steps):
            skill = skills[step % len(skills)]
            x = torch.randn(batch_size, concepts.shape[1])
            y = skill_labels(x, concepts[skill])
            loss = F.binary_cross_entropy_with_logits(model(x)[:, skill], y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            _project_spares(model, n_spare)
        return export_mlp(model)


@torch.no_grad()
def head_accuracy(
    weights: dict[str, torch.Tensor],
    concepts: torch.Tensor,
    head: int,
    n: int = 4000,
    seed: int = 0,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, concepts.shape[1], generator=generator)
    y = skill_labels(x, concepts[head]) > 0.5
    logits = functional_mlp(weights, x)[:, head]
    return float((logits > 0).eq(y).float().mean())


@torch.no_grad()
def carve_spares(
    weights: dict[str, torch.Tensor],
    concepts: torch.Tensor,
    skills: list[int],
    tol: float = 0.01,
    n: int = 2500,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Zero outgoing weights that the trained skills do not need."""
    from zeroth_grad.mlp import clone_weights

    current = clone_weights(weights)
    original = [head_accuracy(current, concepts, skill, n=n, seed=seed) for skill in skills]

    order1 = torch.argsort(current["w2"].norm(dim=0)).tolist()
    for index in order1:
        trial = clone_weights(current)
        trial["w2"][:, index] = 0
        trial["w1"][index] = 0
        trial["b1"][index] = 0
        trial_acc = [head_accuracy(trial, concepts, skill, n=n, seed=seed) for skill in skills]
        if all(original[i] - trial_acc[i] <= tol for i in range(len(skills))):
            current = trial

    order2 = torch.argsort(current["w3"].norm(dim=0)).tolist()
    for index in order2:
        trial = clone_weights(current)
        trial["w3"][:, index] = 0
        trial["w2"][index] = 0
        trial["b2"][index] = 0
        trial_acc = [head_accuracy(trial, concepts, skill, n=n, seed=seed) for skill in skills]
        if all(original[i] - trial_acc[i] <= tol for i in range(len(skills))):
            current = trial
    return current


def active_counts(weights: dict[str, torch.Tensor], eps: float = 1e-6) -> tuple[int, int]:
    h1 = int((weights["w2"].norm(dim=0) > eps).sum())
    h2 = int((weights["w3"].norm(dim=0) > eps).sum())
    return h1, h2


def populate(
    out_dir: str | Path,
    steps: int = 300,
    h: int = H,
    n_spare: int = N_SPARES,
    seed: int = 0,
) -> list[Record]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from zeroth_grad.data import concept_bank

    concepts = concept_bank(seed)
    torch.save({"concepts": concepts}, out_dir / "concepts.pt")
    init = shared_init(seed=seed, d_in=D_IN, h=h, d_out=D_OUT)
    save_checkpoint(
        out_dir / "init.safetensors",
        init,
        {"kind": "init", "split": "none", "skills": "", "seed": str(seed), "base_h1": str(h), "base_h2": str(h), "h1": str(h), "h2": str(h), "d_in": str(D_IN), "d_out": str(D_OUT), "carved": "0"},
    )

    records: list[Record] = []
    jobs: list[tuple[str, str, tuple[int, ...], int]] = []
    for skill in TRAIN_SKILLS:
        for model_seed in TRAIN_SEEDS:
            jobs.append(("train", "single", (skill,), model_seed))
        for model_seed in HELD_OUT_SEEDS:
            jobs.append(("heldout", "single", (skill,), model_seed))
    for pair in PAIR_SKILLS:
        jobs.append(("train", "pair", pair, 0))

    for split, kind, skills, model_seed in jobs:
        weights = train_student(init, concepts, skills, steps=steps, seed=model_seed + 17 * skills[0], n_spare=n_spare)
        weights = carve_spares(weights, concepts, list(skills))
        acc = [head_accuracy(weights, concepts, skill) for skill in skills]
        meta = {
            "d_in": str(D_IN),
            "d_out": str(D_OUT),
            "h1": str(weights["w1"].shape[0]),
            "h2": str(weights["w2"].shape[0]),
            "base_h1": str(weights["w1"].shape[0]),
            "base_h2": str(weights["w2"].shape[0]),
            "skills": format_skills(skills),
            "seed": str(model_seed),
            "split": split,
            "kind": kind,
            "carved": "1",
        }
        name = f"{split}_{kind}_s{meta['skills'].replace(',', '-')}_seed{model_seed}.safetensors"
        path = out_dir / name
        save_checkpoint(path, weights, meta)
        record = Record(weights=weights, metadata=meta, path=path)
        records.append(record)
        used = active_counts(weights)
        print(f"{name} acc={[round(a, 3) for a in acc]} active_h1={used[0]} active_h2={used[1]}")
    return records


def load_concepts(checkpoint_dir: str | Path) -> torch.Tensor:
    blob = torch.load(Path(checkpoint_dir) / "concepts.pt", map_location="cpu", weights_only=False)
    return blob["concepts"]


def load_zoo(checkpoint_dir: str | Path) -> list[Record]:
    root = Path(checkpoint_dir)
    records = []
    for path in sorted(root.glob("*.safetensors")):
        if path.name == "init.safetensors":
            continue
        weights, meta = load_checkpoint(path)
        records.append(Record(weights=weights, metadata=meta, path=path))
    return records
