"""Draw the held-out compiler results from runs/metrics.json, and the digits results if present."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

INK = "#1c1917"
MUTED = "#78716c"
PAPER = "#faf7f2"
TEAL = "#0f766e"
AMBER = "#c2410c"
STONE = "#a8a29e"


def _series(metrics: dict) -> tuple[dict, dict, dict]:
    inject = metrics["inject"]
    compress = metrics["compress"]
    merge = metrics["merge"]
    return (
        {
            "Before": inject["before_new_acc"],
            "Untrained": inject["untrained_new_acc"],
            "Trained": inject["trained_new_acc"],
            "Adam oracle": inject["oracle_new_acc"],
        },
        {
            "Dense": compress["dense_acc"],
            "Compiler": compress["compiler_acc"],
            "Collapse": compress["geometric_acc"],
            "Magnitude": compress["magnitude_acc"],
        },
        {
            "Compiler": merge["compiler_acc"],
            "Compiler, permuted": merge["permuted_compiler_acc"],
            "Unrouted stitch": merge["norm_gate_acc"],
            "Aligned soup": merge["aligned_soup_acc"],
            "Unaligned": merge["unaligned_acc"],
            "Unaligned, permuted": merge["permuted_unaligned_acc"],
        },
    )


def _style(ax) -> None:
    ax.set_facecolor(PAPER)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy", color=MUTED)
    ax.axhline(0.5, color="#e7e5e4", linewidth=1, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e7e5e4")
    ax.spines["bottom"].set_color("#e7e5e4")
    ax.tick_params(colors=MUTED, length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#f0eeea", linewidth=1)


def _panel(ax, title, series, highlight):
    names = list(series)
    values = [series[name] for name in names]
    colors = [TEAL if name in highlight else STONE for name in names]
    positions = np.arange(len(names))
    bars = ax.bar(positions, values, color=colors, width=0.72, zorder=2)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12, fontweight="medium")
    ax.set_xticks(positions, names, rotation=28, ha="right", fontsize=9, color=INK)
    _style(ax)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.02,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK,
        )


def _curve_panel(ax, merge_many: dict) -> None:
    sizes = sorted(merge_many, key=int)
    ks = [int(k) for k in sizes]
    lines = (
        ("Compiler", "compiler_acc", TEAL, "-"),
        ("Untrained compiler", "routed_prior_acc", TEAL, ":"),
        ("Aligned soup", "aligned_soup_acc", AMBER, "-"),
        ("Source models", "source_acc", STONE, "--"),
    )
    for label, key, color, style in lines:
        ax.plot(ks, [merge_many[k][key] for k in sizes], style, color=color, marker="o", markersize=4, linewidth=1.8, label=label, zorder=2)
    ax.set_title("Merge k checkpoints into one", loc="left", fontsize=13, color=INK, pad=12, fontweight="medium")
    ax.set_xticks(ks, [str(k) for k in ks], fontsize=9, color=INK)
    ax.set_xlabel("Skills merged", color=MUTED)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)


def _digits_figure(metrics: dict, out: Path) -> None:
    ten = metrics["ten_way"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=160, gridspec_kw={"width_ratios": [1, 1.25]})
    fig.patch.set_facecolor(PAPER)
    bars = {
        "Merged (compiler)": ten["compiler_acc"],
        "Ensemble of 10": ten["ensemble_acc"],
        "Joint training": ten["joint_acc"],
        "Legacy alignment": ten["legacy_alignment_acc"],
        "Aligned soup": ten["aligned_soup_acc"],
    }
    _panel(axes[0], "Ten detectors, one classifier", bars, {"Merged (compiler)", "Ensemble of 10", "Joint training"})
    merge_k = metrics["merge_k"]
    sizes = sorted(merge_k, key=int)
    ks = [int(k) for k in sizes]
    lines = (
        ("Merged (compiler)", "compiler_acc", TEAL, "-"),
        ("Ensemble of separate detectors", "ensemble_acc", STONE, "--"),
        ("Legacy alignment", "legacy_alignment_acc", MUTED, ":"),
        ("Aligned soup", "aligned_soup_acc", AMBER, "-"),
    )
    ax = axes[1]
    for label, key, color, style in lines:
        ax.plot(ks, [merge_k[k][key] for k in sizes], style, color=color, marker="o", markersize=4, linewidth=1.8, label=label, zorder=2)
    ax.set_title("Merge k digit detectors", loc="left", fontsize=13, color=INK, pad=12, fontweight="medium")
    ax.set_xticks(ks, [str(k) for k in ks], fontsize=9, color=INK)
    ax.set_xlabel("Digits merged (k-way test accuracy)", color=MUTED)
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower left", labelcolor=INK)
    fig.suptitle(
        f"8x8 digits, width {metrics['width']}, mean of {metrics['seeds']} detector seeds",
        x=0.02,
        ha="left",
        fontsize=16,
        color=INK,
        fontweight="medium",
    )
    fig.text(0.02, 0.02, "No checkpoint is trained during the merge. Joint training is a reference network trained on all ten labels.", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=ROOT / "runs" / "metrics.json")
    parser.add_argument("--out", type=Path, default=ROOT / "docs" / "heldout_results.png")
    parser.add_argument("--digits-metrics", type=Path, default=ROOT / "runs" / "digits_metrics.json")
    parser.add_argument("--digits-out", type=Path, default=ROOT / "docs" / "digits_results.png")
    args = parser.parse_args()
    if not args.metrics.exists():
        raise SystemExit(f"{args.metrics} not found; run `python -m zeroth_grad eval` first")
    metrics = json.loads(args.metrics.read_text())
    inject, compress, merge = _series(metrics)

    plt.rcParams["font.family"] = "DejaVu Sans"
    if args.digits_metrics.exists():
        _digits_figure(json.loads(args.digits_metrics.read_text()), args.digits_out)
    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.8), dpi=160)
    fig.patch.set_facecolor(PAPER)
    _panel(axes[0], "Inject an unseen skill", inject, {"Trained"})
    _panel(axes[1], "Compress duplicated neurons", compress, {"Compiler", "Dense", "Collapse"})
    _panel(axes[2], "Merge two checkpoints", merge, {"Compiler", "Compiler, permuted"})
    _curve_panel(axes[3], metrics["merge_many"])
    fig.suptitle(
        "Weight compiler, held-out checkpoints",
        x=0.02,
        ha="left",
        fontsize=16,
        color=INK,
        fontweight="medium",
    )
    fig.text(
        0.02,
        0.02,
        "Teal is the compiler or the original model. Gray and amber are baselines.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor=fig.get_facecolor())
    print(args.out)


if __name__ == "__main__":
    main()
