"""Draw the held-out compiler results. Numbers match runs/metrics.json."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Held-out accuracies from runs/metrics.json, rounded for the figure.
INJECT = {
    "Before": 0.503,
    "Untrained": 0.807,
    "Trained": 0.998,
    "Adam oracle": 0.904,
}
COMPRESS = {
    "Dense": 0.984,
    "Compiler": 0.984,
    "Collapse": 0.984,
    "Magnitude": 0.643,
}
MERGE = {
    "Compiler": 0.874,
    "Norm gate": 0.874,
    "Unaligned": 0.846,
    "Aligned soup": 0.886,
    "Compiler, permuted": 0.874,
    "Unaligned, permuted": 0.861,
}

INK = "#1c1917"
MUTED = "#78716c"
PAPER = "#faf7f2"
TEAL = "#0f766e"
AMBER = "#c2410c"
STONE = "#a8a29e"


def _panel(ax, title, series, highlight):
    names = list(series)
    values = [series[name] for name in names]
    colors = [TEAL if name in highlight else STONE for name in names]
    positions = np.arange(len(names))
    bars = ax.bar(positions, values, color=colors, width=0.72, zorder=2)
    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12, fontweight="medium")
    ax.set_xticks(positions, names, rotation=28, ha="right", fontsize=9, color=INK)
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


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.6), dpi=160)
    fig.patch.set_facecolor(PAPER)
    for ax in axes:
        ax.set_facecolor(PAPER)
    _panel(axes[0], "Inject an unseen skill", INJECT, {"Trained"})
    _panel(axes[1], "Compress duplicated neurons", COMPRESS, {"Compiler", "Dense", "Collapse"})
    _panel(axes[2], "Merge two checkpoints", MERGE, {"Compiler", "Compiler, permuted"})
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
        "Teal bars are the compiler or the original model. Gray bars are baselines.",
        color=MUTED,
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    out = Path(__file__).resolve().parents[1] / "docs" / "heldout_results.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=fig.get_facecolor())
    print(out)


if __name__ == "__main__":
    main()
