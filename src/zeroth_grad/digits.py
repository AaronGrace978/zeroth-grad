"""Real-data merge benchmark: one-vs-rest detectors on scikit-learn's 8x8 digits.

Each checkpoint detects one digit on its own logit. The detectors share an
initialization, are trained with a group-sparsity penalty on outgoing weights
so that each one uses few hidden units, and are carved like the halfspace zoo.
Merging all ten gives a 10-way classifier: the prediction is the argmax over
heads. No checkpoint is trained during the merge.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import torch
import torch.nn.functional as F

from zeroth_grad.compiler import WeightCompiler
from zeroth_grad.evaluate import aligned_soup_many
from zeroth_grad.io import load_checkpoint, save_checkpoint
from zeroth_grad.mlp import clone_weights, export_mlp, functional_mlp, load_mlp, shared_init
from zeroth_grad.zoo import N_SPARES, _project_spares, active_counts

N_DIGITS = 10
D_PIXELS = 64


@dataclass
class DigitData:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor


def load_digit_data(seed: int = 0, n_train: int = 1200) -> DigitData:
    """Shuffle the 1,797 digits, split train/test, and standardize with train statistics."""
    try:
        from sklearn.datasets import load_digits
    except ImportError as exc:
        raise ImportError('the digits benchmark needs scikit-learn: pip install -e ".[digits]"') from exc
    raw = load_digits()
    x = torch.tensor(raw.data, dtype=torch.float32) / 16.0
    y = torch.tensor(raw.target, dtype=torch.long)
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))
    x, y = x[order], y[order]
    mean = x[:n_train].mean(dim=0)
    std = x[:n_train].std(dim=0).clamp_min(0.1)
    x = (x - mean) / std
    return DigitData(x[:n_train], y[:n_train], x[n_train:], y[n_train:])


@torch.no_grad()
def balanced_accuracy(weights: dict[str, torch.Tensor], head: int, x: torch.Tensor, y: torch.Tensor) -> float:
    """Mean of the true-positive and true-negative rates of one detector head."""
    predicted = functional_mlp(weights, x)[:, head] > 0
    target = y == head
    tpr = (predicted & target).float().sum() / target.sum().clamp_min(1)
    tnr = (~predicted & ~target).float().sum() / (~target).sum().clamp_min(1)
    return float(0.5 * (tpr + tnr))


@torch.no_grad()
def multiclass_accuracy(
    weights: dict[str, torch.Tensor],
    x: torch.Tensor,
    y: torch.Tensor,
    digits: list[int] | None = None,
) -> float:
    """Argmax accuracy over `digits` heads, on the test samples of those digits."""
    logits = functional_mlp(weights, x)
    return _restricted_accuracy(logits, y, digits)


def _restricted_accuracy(logits: torch.Tensor, y: torch.Tensor, digits: list[int] | None) -> float:
    if digits is None:
        return float((logits.argmax(dim=1) == y).float().mean())
    heads = torch.tensor(digits)
    keep = torch.isin(y, heads)
    predicted = heads[logits[keep][:, heads].argmax(dim=1)]
    return float((predicted == y[keep]).float().mean())


def train_detector(
    init: dict[str, torch.Tensor],
    data: DigitData,
    digit: int,
    steps: int = 400,
    seed: int = 0,
    sparsity: float = 3e-3,
    n_spare: int = N_SPARES,
    lr: float = 0.01,
    batch_size: int = 64,
) -> dict[str, torch.Tensor]:
    """Balanced one-vs-rest BCE on head `digit`, plus a group-sparsity penalty.

    The penalty is the sum of hidden-1 outgoing column norms and the absolute
    readout of hidden-2 units into this head, so the detector concentrates on
    few units and leaves the rest of the width for other detectors.
    """
    generator = torch.Generator().manual_seed(seed)
    positives = (data.y_train == digit).nonzero().flatten()
    negatives = (data.y_train != digit).nonzero().flatten()
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = load_mlp(init)
        _project_spares(model, n_spare)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        for _ in range(steps):
            index = torch.cat(
                [
                    positives[torch.randint(len(positives), (batch_size,), generator=generator)],
                    negatives[torch.randint(len(negatives), (batch_size,), generator=generator)],
                ]
            )
            logits = model(data.x_train[index])[:, digit]
            loss = F.binary_cross_entropy_with_logits(logits, (data.y_train[index] == digit).float())
            if sparsity > 0:
                group = model.fc2.weight.norm(dim=0).sum() + model.fc3.weight[digit].abs().sum()
                loss = loss + sparsity * group
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            _project_spares(model, n_spare)
    return export_mlp(model)


@torch.no_grad()
def carve_detector(
    weights: dict[str, torch.Tensor],
    data: DigitData,
    digit: int,
    tol: float = 0.005,
) -> dict[str, torch.Tensor]:
    """Zero every hidden unit whose removal costs at most `tol` training balanced accuracy."""
    current = clone_weights(weights)
    base = balanced_accuracy(current, digit, data.x_train, data.y_train)
    for index in torch.argsort(current["w2"].norm(dim=0)).tolist():
        trial = clone_weights(current)
        trial["w2"][:, index] = 0
        trial["w1"][index] = 0
        trial["b1"][index] = 0
        if base - balanced_accuracy(trial, digit, data.x_train, data.y_train) <= tol:
            current = trial
    for index in torch.argsort(current["w3"].norm(dim=0)).tolist():
        trial = clone_weights(current)
        trial["w3"][:, index] = 0
        trial["w2"][index] = 0
        trial["b2"][index] = 0
        if base - balanced_accuracy(trial, digit, data.x_train, data.y_train) <= tol:
            current = trial
    return current


def train_joint(
    init: dict[str, torch.Tensor],
    data: DigitData,
    steps: int = 800,
    seed: int = 0,
    lr: float = 0.01,
    batch_size: int = 128,
) -> dict[str, torch.Tensor]:
    """A 10-way softmax MLP trained on all labels at once. A reference, not a merge."""
    generator = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        model = load_mlp(init)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        for _ in range(steps):
            index = torch.randint(len(data.x_train), (batch_size,), generator=generator)
            loss = F.cross_entropy(model(data.x_train[index]), data.y_train[index])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return export_mlp(model)


def populate_digits(
    out_dir: str | Path,
    hidden: int = 64,
    steps: int = 400,
    sparsity: float = 3e-3,
    seeds: tuple[int, ...] = (0, 1, 2),
    data_seed: int = 0,
    init_seed: int = 0,
) -> None:
    """Write ten carved detectors per seed, plus a jointly trained reference per seed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = load_digit_data(data_seed)
    init = shared_init(seed=init_seed, d_in=D_PIXELS, h=hidden, d_out=N_DIGITS)
    common = {
        "dataset": "digits",
        "data_seed": str(data_seed),
        "d_in": str(D_PIXELS),
        "d_out": str(N_DIGITS),
        "h1": str(hidden),
        "h2": str(hidden),
        "base_h1": str(hidden),
        "base_h2": str(hidden),
        "sparsity": str(sparsity),
    }
    save_checkpoint(out_dir / "init.safetensors", init, {**common, "kind": "init", "skills": ""})
    for seed in seeds:
        totals = [0, 0]
        for digit in range(N_DIGITS):
            weights = train_detector(init, data, digit, steps=steps, seed=1000 * seed + digit, sparsity=sparsity)
            weights = carve_detector(weights, data, digit)
            live = active_counts(weights)
            totals[0] += live[0]
            totals[1] += live[1]
            acc = balanced_accuracy(weights, digit, data.x_test, data.y_test)
            meta = {**common, "kind": "single", "skills": str(digit), "seed": str(seed), "live_h1": str(live[0]), "live_h2": str(live[1])}
            save_checkpoint(out_dir / f"digit{digit}_seed{seed}.safetensors", weights, meta)
            print(f"digit{digit}_seed{seed} balanced_acc={acc:.3f} live_h1={live[0]} live_h2={live[1]}")
        joint = train_joint(init, data, steps=2 * steps, seed=seed)
        meta = {**common, "kind": "joint", "skills": ",".join(str(d) for d in range(N_DIGITS)), "seed": str(seed)}
        save_checkpoint(out_dir / f"joint_seed{seed}.safetensors", joint, meta)
        print(
            f"seed {seed}: detectors use {totals[0]} hidden-1 and {totals[1]} hidden-2 units in total "
            f"(width {hidden}); joint 10-way acc={multiclass_accuracy(joint, data.x_test, data.y_test):.3f}"
        )


def _load_groups(checkpoint_dir: Path) -> tuple[dict[str, list[dict[str, torch.Tensor]]], dict[str, dict[str, torch.Tensor]], dict[str, str]]:
    detectors: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    joints: dict[str, dict[str, torch.Tensor]] = {}
    meta_any: dict[str, str] = {}
    for path in sorted(checkpoint_dir.glob("*.safetensors")):
        weights, meta = load_checkpoint(path)
        if meta.get("dataset") != "digits":
            continue
        meta_any = meta
        if meta.get("kind") == "single":
            detectors.setdefault(meta["seed"], {})[int(meta["skills"])] = weights
        elif meta.get("kind") == "joint":
            joints[meta["seed"]] = weights
    groups = {}
    for seed, by_digit in detectors.items():
        if sorted(by_digit) != list(range(N_DIGITS)):
            raise ValueError(f"seed {seed} is missing detectors: has {sorted(by_digit)}")
        groups[seed] = [by_digit[d] for d in range(N_DIGITS)]
    if not groups:
        raise ValueError(f"no digit detectors in {checkpoint_dir}; run digits-populate first")
    return groups, joints, meta_any


def _subsets(k: int, limit: int, generator: torch.Generator) -> list[tuple[int, ...]]:
    if math.comb(N_DIGITS, k) <= limit:
        return list(combinations(range(N_DIGITS), k))
    chosen: set[tuple[int, ...]] = set()
    while len(chosen) < limit:
        chosen.add(tuple(sorted(torch.randperm(N_DIGITS, generator=generator)[:k].tolist())))
    return sorted(chosen)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


@torch.no_grad()
def evaluate_digits(
    checkpoint_dir: str | Path,
    compiler: WeightCompiler | None = None,
    out: str | Path | None = None,
    subsets_per_size: int = 20,
    seed: int = 0,
) -> dict:
    """Merge the detectors of each seed and compare the result to the ensemble ceiling and baselines.

    With no compiler, an untrained WeightCompiler is used: alignment, norm
    gates, and readout-norm head ownership, with no learned residual.
    """
    groups, joints, meta = _load_groups(Path(checkpoint_dir))
    data = load_digit_data(int(meta.get("data_seed", "0")))
    if compiler is None:
        compiler = WeightCompiler(d_in=D_PIXELS, d_out=N_DIGITS)
    compiler.eval()
    rows: dict[str, list[float]] = {
        "compiler_acc": [],
        "legacy_alignment_acc": [],
        "aligned_soup_acc": [],
        "ensemble_acc": [],
        "joint_acc": [],
        "live_h1": [],
        "live_h2": [],
    }
    per_size: dict[int, dict[str, list[float]]] = {}
    generator = torch.Generator().manual_seed(seed)
    for group_seed, models in sorted(groups.items()):
        ensemble = torch.stack([functional_mlp(m, data.x_test)[:, d] for d, m in enumerate(models)], dim=1)
        merged = compiler.merge_many(models)
        rows["compiler_acc"].append(multiclass_accuracy(merged, data.x_test, data.y_test))
        rows["legacy_alignment_acc"].append(
            multiclass_accuracy(compiler.merge_many(models, align_mode="legacy"), data.x_test, data.y_test)
        )
        rows["aligned_soup_acc"].append(multiclass_accuracy(aligned_soup_many(models), data.x_test, data.y_test))
        rows["ensemble_acc"].append(_restricted_accuracy(ensemble, data.y_test, None))
        if group_seed in joints:
            rows["joint_acc"].append(multiclass_accuracy(joints[group_seed], data.x_test, data.y_test))
        rows["live_h1"].append(float(sum(active_counts(m)[0] for m in models)))
        rows["live_h2"].append(float(sum(active_counts(m)[1] for m in models)))
        for k in range(2, N_DIGITS + 1):
            bucket = per_size.setdefault(k, {"compiler_acc": [], "legacy_alignment_acc": [], "aligned_soup_acc": [], "ensemble_acc": []})
            for subset in _subsets(k, subsets_per_size, generator):
                digits = list(subset)
                chosen = [models[d] for d in digits]
                bucket["compiler_acc"].append(multiclass_accuracy(compiler.merge_many(chosen), data.x_test, data.y_test, digits))
                bucket["legacy_alignment_acc"].append(
                    multiclass_accuracy(compiler.merge_many(chosen, align_mode="legacy"), data.x_test, data.y_test, digits)
                )
                bucket["aligned_soup_acc"].append(multiclass_accuracy(aligned_soup_many(chosen), data.x_test, data.y_test, digits))
                bucket["ensemble_acc"].append(_restricted_accuracy(ensemble, data.y_test, digits))
    width = int(next(iter(groups.values()))[0]["w1"].shape[0])
    metrics = {
        "width": width,
        "seeds": len(groups),
        "ten_way": {key: _mean(values) for key, values in rows.items()},
        "ten_way_per_seed": {key: values for key, values in rows.items()},
        "merge_k": {
            str(k): {"n": float(len(b["compiler_acc"])), **{key: _mean(values) for key, values in b.items()}}
            for k, b in sorted(per_size.items())
        },
    }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics
