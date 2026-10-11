"""Benchmarks for the scaling extensions. Every number lands in runs/extensions_metrics.json.

- `alignment_scaling`: Hungarian against Sinkhorn on wide interference costs,
  wall clock and assignment cost, and merge accuracy with each solver.
- `capacity`: digits merges over capacity, with declared head ownership,
  collision-driven widening, and activation folding of dense detectors.
- `rank_k_inject`: non-linear skills installed from labeled examples, rank-1
  against rank-k.
- `transformer`: alignment and merging of small transformers.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from zeroth_grad.align import match_cost
from zeroth_grad.compiler import WeightCompiler
from zeroth_grad.data import estimate_direction
from zeroth_grad.mlp import functional_mlp
from zeroth_grad.solvers import assignment_cost, hungarian, sinkhorn


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _wide_layer_cost(width: int, d_in: int = 64, live: float = 0.4, noise: float = 0.3, seed: int = 0) -> torch.Tensor:
    """Interference cost between a sparse wide layer and a noisy, permuted copy of it."""
    generator = torch.Generator().manual_seed(seed)
    rows = torch.randn(width, d_in, generator=generator)
    out = torch.rand(width, generator=generator) * (torch.rand(width, generator=generator) < live)
    perm = torch.randperm(width, generator=generator)
    partner = rows[perm] + noise * torch.randn(width, d_in, generator=generator)
    return match_cost(rows, partner, out, out[perm])


def alignment_scaling(widths: tuple[int, ...] = (256, 512, 1024, 2048, 4096)) -> dict:
    rows = {}
    for width in widths:
        cost = _wide_layer_cost(width)
        start = time.perf_counter()
        exact = hungarian(cost)
        t_h = time.perf_counter() - start
        start = time.perf_counter()
        approx = sinkhorn(cost)
        t_s = time.perf_counter() - start
        best = assignment_cost(cost, exact)
        rows[str(width)] = {
            "hungarian_s": t_h,
            "sinkhorn_s": t_s,
            "speedup": t_h / max(t_s, 1e-9),
            "cost_ratio": assignment_cost(cost, approx) / max(best, 1e-12),
            "same_assignment": float((exact == approx).float().mean()),
        }
        print(f"width {width}: hungarian {t_h:.3f}s sinkhorn {t_s:.3f}s cost ratio {rows[str(width)]['cost_ratio']:.4f}")
    return rows


@torch.no_grad()
def solver_merge_accuracy(checkpoint_dir: str | Path, digits_dir: str | Path | None) -> dict:
    """Merge accuracy with each solver on the held-out zoo pairs and, if present, ten-digit merges."""
    from zeroth_grad.evaluate import _heldout_by_seed, _merge_accuracies
    from zeroth_grad.zoo import load_concepts, load_zoo

    compiler = WeightCompiler().eval()
    out: dict[str, dict[str, float]] = {}
    concepts = load_concepts(checkpoint_dir)
    records = load_zoo(checkpoint_dir)
    for solver in ("hungarian", "sinkhorn"):
        scores = []
        for _, group in sorted(_heldout_by_seed(records).items()):
            for i, left in enumerate(group):
                for right in group[i + 1 :]:
                    if left.skills[0] == right.skills[0]:
                        continue
                    merged = compiler.merge(left.weights, right.weights, solver=solver)
                    scores.append(_merge_accuracies(merged, concepts, [left.skills[0], right.skills[0]], 2500))
        out[solver] = {"pairs_acc": _mean(scores)}
    if digits_dir is not None and Path(digits_dir).exists():
        from zeroth_grad.digits import D_PIXELS, N_DIGITS, _load_groups, load_digit_data, multiclass_accuracy

        groups, _, meta = _load_groups(Path(digits_dir))
        data = load_digit_data(int(meta.get("data_seed", "0")))
        prior = WeightCompiler(d_in=D_PIXELS, d_out=N_DIGITS).eval()
        for solver in ("hungarian", "sinkhorn"):
            accs = [multiclass_accuracy(prior.merge_many(models, solver=solver), data.x_test, data.y_test) for models in groups.values()]
            out[solver]["digits_ten_way_acc"] = _mean(accs)
    return out


@torch.no_grad()
def capacity(digit_dirs: dict[str, str | Path], agreements: tuple[float, ...] = (0.995, 0.99, 0.98)) -> dict:
    """Ten-digit merges on each zoo: plain, declared ownership, widened, and folded."""
    from zeroth_grad.digits import D_PIXELS, N_DIGITS, _load_groups, _restricted_accuracy, load_digit_data, multiclass_accuracy
    from zeroth_grad.fold import fold_report
    from zeroth_grad.zoo import active_counts

    compiler = WeightCompiler(d_in=D_PIXELS, d_out=N_DIGITS).eval()
    heads = [[d] for d in range(N_DIGITS)]
    out = {}
    for name, directory in digit_dirs.items():
        if not Path(directory).exists():
            continue
        groups, _, meta = _load_groups(Path(directory))
        data = load_digit_data(int(meta.get("data_seed", "0")))
        rows: dict[str, list[float]] = {}

        def add(key: str, value: float) -> None:
            rows.setdefault(key, []).append(float(value))

        for models in groups.values():
            ensemble = torch.stack([functional_mlp(m, data.x_test)[:, d] for d, m in enumerate(models)], dim=1)
            add("ensemble_acc", _restricted_accuracy(ensemble, data.y_test, None))
            add("width", models[0]["w1"].shape[0])
            add("live_h1", sum(active_counts(m)[0] for m in models))
            add("merge_acc", multiclass_accuracy(compiler.merge_many(models), data.x_test, data.y_test))
            add("declared_acc", multiclass_accuracy(compiler.merge_many(models, heads=heads), data.x_test, data.y_test))
            widened = compiler.merge_many(models, heads=heads, expand=True)
            add("expand_acc", multiclass_accuracy(widened, data.x_test, data.y_test))
            add("expand_width_h1", widened["w1"].shape[0])
            add("expand_width_h2", widened["w2"].shape[0])
            for agreement in agreements:
                tag = f"fold{agreement}"
                reports = [fold_report(m, data.x_train, [d], agreement=agreement) for d, m in enumerate(models)]
                folded = [r["weights"] for r in reports]
                add(f"{tag}_kept_h1", sum(r["kept_h1"] for r in reports))
                add(f"{tag}_kept_h2", sum(r["kept_h2"] for r in reports))
                folded_ensemble = torch.stack([functional_mlp(m, data.x_test)[:, d] for d, m in enumerate(folded)], dim=1)
                add(f"{tag}_ensemble_acc", _restricted_accuracy(folded_ensemble, data.y_test, None))
                add(f"{tag}_merge_acc", multiclass_accuracy(compiler.merge_many(folded, heads=heads), data.x_test, data.y_test))
                folded_wide = compiler.merge_many(folded, heads=heads, expand=True)
                add(f"{tag}_expand_acc", multiclass_accuracy(folded_wide, data.x_test, data.y_test))
                add(f"{tag}_expand_width_h1", folded_wide["w1"].shape[0])
        out[name] = {key: _mean(values) for key, values in rows.items()}
        print(name, json.dumps({k: round(v, 3) for k, v in out[name].items()}))
    return out


def nonlinear_tasks(concepts: torch.Tensor) -> dict[str, tuple[int, callable]]:
    """(rank, labeler) for skills on held-out directions v6, v7, each with a positive rate of about 1/2."""
    v6, v7 = concepts[6], concepts[7]
    corner = -0.5449  # P(v6.x > c and v7.x > c) = 1/2 for x ~ N(0, I)
    return {
        "halfspace": (1, lambda x: x @ v6 > 0),
        "slab": (1, lambda x: (x @ v6).abs() < 0.6745),
        "wedge": (2, lambda x: (x @ v6 > corner) & (x @ v7 > corner)),
        "xor": (2, lambda x: (x @ v6) * (x @ v7) > 0),
    }


def _finetune_examples(weights, x, y, new_head, old_head, old_concept, steps=200, lr=0.05):
    params = {k: v.clone().requires_grad_(True) for k, v in weights.items()}
    opt = torch.optim.Adam(params.values(), lr=lr)
    old_y = (x @ old_concept > 0).float()
    for _ in range(steps):
        logits = functional_mlp(params, x)
        loss = F.binary_cross_entropy_with_logits(logits[:, new_head], y) + F.binary_cross_entropy_with_logits(logits[:, old_head], old_y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return {k: v.detach() for k, v in params.items()}


def rank_k_inject(
    checkpoint_dir: str | Path,
    compiler_path: str | Path | None,
    sizes: tuple[int, ...] = (512, 2048),
    oracle_limit: int = 6,
) -> dict:
    from zeroth_grad.lowrank import inject_rank_k
    from zeroth_grad.train import load_compiler
    from zeroth_grad.zoo import load_concepts, load_zoo

    concepts = load_concepts(checkpoint_dir)
    heldout = [r for r in load_zoo(checkpoint_dir) if r.split == "heldout" and r.kind == "single"]
    rank1 = load_compiler(compiler_path) if compiler_path is not None and Path(compiler_path).exists() else WeightCompiler()
    rank1.eval()
    x_test = torch.randn(4000, concepts.shape[1], generator=torch.Generator().manual_seed(99))
    target = 6
    out: dict[str, dict] = {}
    for name, (rank, labeler) in nonlinear_tasks(concepts).items():
        y_test = labeler(x_test)
        span = concepts[6:6 + rank]
        for n in sizes:
            rows: dict[str, list[float]] = {}
            generator = torch.Generator().manual_seed(1000 + n)
            for index, record in enumerate(heldout):
                x = torch.randn(n, concepts.shape[1], generator=generator)
                y = labeler(x).float()
                src = record.skills[0]

                def score(w, key):
                    rows.setdefault(f"{key}_new", []).append(float(((functional_mlp(w, x_test)[:, target] > 0) == y_test).float().mean()))
                    old = (x_test @ concepts[src] > 0)
                    rows.setdefault(f"{key}_old", []).append(float(((functional_mlp(w, x_test)[:, src] > 0) == old).float().mean()))

                with torch.no_grad():
                    score(record.weights, "before")
                    score(rank1.inject(record.weights, estimate_direction(x, y), target), "rank1")
                    score(inject_rank_k(record.weights, x, y, target, rank=rank), "rankk")
                    score(inject_rank_k(record.weights, x, y, target, rank=rank, subspace=span), "rankk_known_subspace")
                    score(inject_rank_k(record.weights, x, y, target, rank=rank, units=16), "rankk_16_units")
                if index < oracle_limit:
                    score(_finetune_examples(record.weights, x, y, target, src, concepts[src]), "finetune")
            out[f"{name}_n{n}"] = {key: _mean(values) for key, values in rows.items()}
            print(name, n, json.dumps({k: round(v, 3) for k, v in out[f"{name}_n{n}"].items() if k.endswith("_new")}))
    return out


def run_extensions(
    checkpoints: str | Path = "checkpoints",
    compiler: str | Path | None = "runs/compiler.pt",
    digits: str | Path = "digits",
    digits_w32: str | Path = "digits_w32",
    digits_dense: str | Path = "digits_dense",
    widths: tuple[int, ...] = (256, 512, 1024, 2048, 4096),
    out: str | Path | None = "runs/extensions_metrics.json",
    sections: tuple[str, ...] = ("alignment", "capacity", "inject", "transformer"),
) -> dict:
    metrics: dict[str, dict] = {}
    if out is not None and Path(out).exists():
        metrics = json.loads(Path(out).read_text())

    def save() -> None:
        if out is not None:
            path = Path(out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(metrics, indent=2) + "\n")

    if "alignment" in sections:
        metrics["alignment_scaling"] = alignment_scaling(widths)
        metrics["solver_merge"] = solver_merge_accuracy(checkpoints, digits)
        save()
    if "capacity" in sections:
        metrics["capacity"] = capacity({"sparse_w64": digits, "sparse_w32": digits_w32, "dense_w64": digits_dense})
        save()
    if "inject" in sections:
        metrics["rank_k_inject"] = rank_k_inject(checkpoints, compiler)
        save()
    if "transformer" in sections:
        from zeroth_grad.transformer_bench import run_all

        metrics["transformer"] = run_all()
        save()
    return metrics
