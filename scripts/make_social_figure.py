"""Draw the share figure for talking about this work outside the paper.

    python -m scripts.make_social_figure

One picture, two panels. The left panel is the architecture the reference
papers share, drawn the way they draw it: two frozen encoders, a projection
each, and a fusion step. The right panel is what this project found when that
fusion was run across four encoders -- the benefit changes sign.

It reads the per-seed records like everything else here, so the bars cannot
drift from the paper. It is a social-media figure, not a paper figure: the
paper's own figure keeps the error bars, the chance line and the axis labels
a reviewer needs, and `scripts/alignment_benefit.py --figure` draws that one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NUMBERS = Path("runs/paper_numbers.json")
OUT = Path("paper/social.png")

INK = "#1b1f2a"
MUTED = "#8b93a7"
POSITIVE = "#2f6f4e"
NEGATIVE = "#b4452f"
PANEL = "#f4f5f8"
LINE = "#c9ced9"

SHORT = {
    "CLIP ViT-L/14": "CLIP ViT-L/14",
    "CLIP ViT-B/32": "CLIP ViT-B/32",
    "jina-clip-v2": "jina-clip-v2",
    "CLIPfa": "CLIPfa",
}


def box(ax, x, y, w, h, label, *, face="white", edge=LINE, size=10, bold=False):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            linewidth=1.3,
            facecolor=face,
            edgecolor=edge,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=size,
        color=INK,
        fontweight="bold" if bold else "normal",
        linespacing=1.45,
    )


def arrow(ax, x1, y1, x2, y2, color=MUTED):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={"arrowstyle": "-|>", "color": color, "linewidth": 1.4},
    )


def architecture(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.0,
        1.0,
        "The architecture everyone shares",
        fontsize=13,
        fontweight="bold",
        color=INK,
        va="top",
    )
    ax.text(
        0.0,
        0.935,
        "Both encoders stay frozen. Only the small head is trained.",
        fontsize=9.5,
        color=MUTED,
        va="top",
    )

    box(ax, 0.02, 0.655, 0.27, 0.125, "image", face=PANEL)
    box(ax, 0.02, 0.345, 0.27, 0.125, "text on the image", face=PANEL)

    box(ax, 0.36, 0.645, 0.24, 0.145, "frozen\nimage encoder")
    box(ax, 0.36, 0.335, 0.24, 0.145, "frozen\ntext encoder")

    box(ax, 0.655, 0.665, 0.115, 0.105, "proj.", size=9)
    box(ax, 0.655, 0.355, 0.115, 0.105, "proj.", size=9)

    arrow(ax, 0.295, 0.7175, 0.352, 0.7175)
    arrow(ax, 0.295, 0.4075, 0.352, 0.4075)
    arrow(ax, 0.605, 0.7175, 0.650, 0.7175)
    arrow(ax, 0.605, 0.4075, 0.650, 0.4075)

    box(ax, 0.835, 0.508, 0.135, 0.11, "fuse", face="#e7ecf6", bold=True, size=11)
    arrow(ax, 0.775, 0.7175, 0.885, 0.623)
    arrow(ax, 0.775, 0.4075, 0.885, 0.503)

    arrow(ax, 0.9025, 0.503, 0.9025, 0.425)
    ax.text(
        0.9025,
        0.385,
        "hateful / not",
        ha="center",
        va="center",
        fontsize=9.5,
        color=INK,
    )

    ax.text(
        0.0,
        0.195,
        "Fusing by element-wise product is the step under test.",
        fontsize=10.5,
        color=INK,
        va="top",
    )
    ax.text(
        0.0,
        0.125,
        "It is reported to help. Nobody had checked whether that\n"
        "depends on which encoder sits in front of it.",
        fontsize=9.5,
        color=MUTED,
        va="top",
        linespacing=1.6,
    )


def finding(ax, benefit: dict) -> None:
    order = ["CLIP ViT-L/14", "CLIP ViT-B/32", "jina-clip-v2", "CLIPfa"]
    rows = [(SHORT[k], benefit[k]["mean"]) for k in order if k in benefit]

    ax.text(
        0.0,
        1.0,
        "What four encoders actually did",
        fontsize=13,
        fontweight="bold",
        color=INK,
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.935,
        "Same fusion, same data, same settings.",
        fontsize=9.5,
        color=MUTED,
        va="top",
        transform=ax.transAxes,
    )

    ys = list(range(len(rows)))
    values = [v for _, v in rows]
    colours = [POSITIVE if v >= 0 else NEGATIVE for v in values]
    ax.barh(ys, values, height=0.40, color=colours, zorder=3)

    for y, (name, value) in zip(ys, rows, strict=True):
        ax.text(
            -0.10 if value >= 0 else 0.10,
            y,
            name,
            ha="right" if value >= 0 else "left",
            va="center",
            fontsize=10.5,
            color=INK,
            zorder=4,
        )
        ax.text(
            value + (0.11 if value >= 0 else -0.11),
            y,
            "helps" if value >= 0 else "hurts",
            ha="left" if value >= 0 else "right",
            va="center",
            fontsize=10,
            fontweight="bold",
            color=POSITIVE if value >= 0 else NEGATIVE,
            zorder=4,
        )

    ax.axvline(0, color=INK, linewidth=1.2, zorder=2)
    ax.set_xlim(-2.75, 2.2)
    ax.set_ylim(-0.6, len(rows) - 0.25)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xticks([])
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    ax.text(
        0.0,
        -0.09,
        "Only the encoder changes — and the sign flips.",
        fontsize=10.5,
        color=INK,
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        -0.165,
        "The usual explanation for why did not survive being tested.",
        fontsize=9.5,
        color=MUTED,
        va="top",
        transform=ax.transAxes,
    )


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not NUMBERS.exists():
        raise SystemExit(f"{NUMBERS} not found; run python -m scripts.paper_numbers")
    benefit = json.loads(NUMBERS.read_text(encoding="utf-8"))["fusion_benefit"]

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.28), dpi=100)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.04, right=0.97, top=0.80, bottom=0.215, wspace=0.22)

    architecture(axes[0])
    finding(axes[1], benefit)

    fig.text(
        0.04,
        0.935,
        "Does cross-modal fusion help? It depends on the encoder.",
        fontsize=16.5,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.04,
        0.038,
        "Four frozen encoders · 180 runs · 15 seeds per cell · "
        "bars to scale · all four significant",
        fontsize=9,
        color=MUTED,
    )
    fig.text(
        0.96,
        0.038,
        "github.com/zhrzkrneng/Persian-hate-speech-detection",
        fontsize=9,
        color=MUTED,
        ha="right",
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, facecolor="white")
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
