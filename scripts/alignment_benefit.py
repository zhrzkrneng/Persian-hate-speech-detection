"""Does an encoder's cross-modal alignment predict how much fusion buys?

    python -m scripts.alignment_benefit
    python -m scripts.alignment_benefit --figure docs/paper/figures/alignment.pdf

This is the relationship the project is built to measure. Hate-CLIPper's
Limitations section reports that on Tamil its fusion variants gave no advantage
over plain concatenation, and attributes that to a feature space CLIP was never
trained to align for that language. If the attribution is right, the benefit of
fusion is a property of the encoder's feature space rather than of the
architecture, and it should therefore be predictable from a measurement of that
space taken without training anything.

So each encoder contributes two independent numbers:

- **alignment**, zero-shot retrieval R@1 on Crossmodal-3600. No classifier,
  no labels, no training. Reported against chance, which on a 1000-image
  slice is 0.1% -- a recall is meaningless without it.
- **fusion benefit**, the paired per-seed difference in test AUROC between
  element-wise-product fusion and plain concatenation, over shared seeds.

text -> image R@1 is the primary alignment measure, chosen before the numbers
were in: it is the strictest direction, it saturates least, and it is the one
whose failure mode -- a caption that lands nowhere near its image -- is what
the downstream classifier would suffer from. image -> text R@1 is reported
beside it, and the ordering under both is printed, so a conclusion that holds
only under the direction that happened to suit it cannot pass unnoticed.

**On what this can and cannot establish.** Four encoders is four points. A
Spearman correlation at n = 4 reaches p = 0.083 at its most extreme possible
value, so a perfect ordering here is not a significant one, and this script
prints n beside every coefficient to keep that in view. Four points also cannot
distinguish a monotone relationship from a non-monotone one, which matters
because the two make different predictions: a monotone reading says less
alignment always means more to gain from an explicit interaction, while
Hate-CLIPper's Tamil result is a case of alignment so poor that nothing helps
at all. An encoder whose English alignment sits at chance -- CLIPfa, on an
English corpus -- is the point that separates those readings, which is why it
is in the comparison. The multilingual extension exists to turn four points
into twenty.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RETRIEVAL = Path("runs/retrieval")
HMC = Path("runs/hmc")

# preset in runs/retrieval -> directory in runs/hmc. Both halves of a point
# must name the same model; tests/test_retrieval_presets.py pins the preset to
# the training config, and this mapping is checked against it by
# tests/test_alignment_benefit.py.
ENCODERS = {
    "clip-l14": "openai__clip-vit-large-patch14",
    "clip-b32": "openai__clip-vit-base-patch32",
    "jina-v2": "jinaai__jina-clip-v2",
    "clipfa": "CLIPfa",
}
LABELS = {
    "clip-l14": "CLIP ViT-L/14",
    "clip-b32": "CLIP ViT-B/32",
    "jina-v2": "jina-clip-v2",
    "clipfa": "CLIPfa",
}


class Missing(Exception):
    """A point cannot be formed because one of its two halves is absent."""


@dataclass(frozen=True)
class Point:
    encoder: str
    label: str
    language: str
    t2i_r1: float
    i2t_r1: float
    images: int
    benefit: float
    benefit_sd: float
    benefit_p: float
    seeds: list[int]
    align_auroc: float
    concat_auroc: float

    @property
    def chance(self) -> float:
        """R@1 a random ranking would reach, in percent."""
        return 100.0 / self.images


def read_alignment(preset: str, language: str) -> tuple[float, float, int]:
    path = RETRIEVAL / f"{preset}_{language}.json"
    if not path.exists():
        raise Missing(
            f"{path} missing; run scripts/eval_retrieval.py --encoder {preset}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return (
        payload["results"]["text_to_image"]["R@1"],
        payload["results"]["image_to_text"]["R@1"],
        payload["images"],
    )


def read_cell(directory: Path) -> dict[int, float]:
    files = sorted(directory.glob("seed*.json"))
    if not files:
        raise Missing(f"no seed*.json under {directory}")
    out: dict[int, float] = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[payload["seed"]] = payload["test"]["auroc"] * 100
    return out


def read_benefit(slug: str, architecture: str = "hateclipper") -> dict:
    """Paired align - concat on test AUROC, over the seeds both cells ran."""
    align = read_cell(HMC / slug / architecture / "align")
    concat = read_cell(HMC / slug / architecture / "concat")
    seeds = sorted(set(align) & set(concat))
    if len(seeds) < 2:
        raise Missing(f"{slug}: fewer than two shared seeds ({seeds})")

    diffs = [align[s] - concat[s] for s in seeds]
    from scipy import stats

    return {
        "benefit": st.mean(diffs),
        "benefit_sd": st.stdev(diffs),
        "benefit_p": float(stats.ttest_1samp(diffs, 0.0).pvalue),
        "seeds": seeds,
        "align_auroc": st.mean(align[s] for s in seeds),
        "concat_auroc": st.mean(concat[s] for s in seeds),
    }


def collect(language: str, architecture: str) -> tuple[list[Point], list[str]]:
    points: list[Point] = []
    skipped: list[str] = []
    for preset, slug in ENCODERS.items():
        try:
            t2i, i2t, images = read_alignment(preset, language)
            benefit = read_benefit(slug, architecture)
        except Missing as problem:
            skipped.append(f"{preset}: {problem}")
            continue
        points.append(
            Point(
                encoder=preset,
                label=LABELS[preset],
                language=language,
                t2i_r1=t2i,
                i2t_r1=i2t,
                images=images,
                **benefit,
            )
        )
    return points, skipped


def correlate(xs: list[float], ys: list[float]) -> dict:
    from scipy import stats

    pearson = stats.pearsonr(xs, ys)
    spearman = stats.spearmanr(xs, ys)
    return {
        "n": len(xs),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def print_table(points: list[Point], architecture: str, language: str) -> None:
    print(
        f"alignment on Crossmodal-3600 {language} vs fusion benefit on "
        f"Hateful Memes, {architecture}\n"
    )
    chance = points[0].chance if points else 0.0
    print(
        f"{'encoder':<16} {'t2i R@1':>8} {'i2t R@1':>8} "
        f"{'concat':>8} {'align':>8} {'benefit':>9} {'sd':>6} {'p':>7}"
    )
    for point in sorted(points, key=lambda p: p.t2i_r1):
        stars = (
            "***"
            if point.benefit_p < 0.001
            else "**"
            if point.benefit_p < 0.01
            else "*"
            if point.benefit_p < 0.05
            else ""
        )
        print(
            f"{point.label:<16} {point.t2i_r1:8.2f} {point.i2t_r1:8.2f} "
            f"{point.concat_auroc:8.2f} {point.align_auroc:8.2f} "
            f"{point.benefit:+9.2f} {point.benefit_sd:6.2f} "
            f"{point.benefit_p:7.3f} {stars}"
        )
    print(f"\nchance R@1 on {points[0].images} images: {chance:.2f}%")
    print(f"seeds per cell: {points[0].seeds}")


def print_relationship(points: list[Point]) -> dict:
    out: dict = {}
    benefits = [p.benefit for p in points]
    print()

    # Two points always lie on a line, so a coefficient computed from them is
    # +/-1 by construction and carries no information at all. Printing one would
    # be worse than printing nothing, so it is refused rather than annotated.
    if len(points) < 3:
        print(
            f"  {len(points)} points: no coefficient is reported. Any two points "
            "are collinear, so r\n  would be +/-1 whatever the numbers are."
        )
        return out

    for name, xs in (
        ("text -> image R@1", [p.t2i_r1 for p in points]),
        ("image -> text R@1", [p.i2t_r1 for p in points]),
    ):
        stats_ = correlate(xs, benefits)
        out[name] = stats_
        print(
            f"benefit vs {name:<18} n = {stats_['n']}  "
            f"Pearson r = {stats_['pearson_r']:+.3f} (p = {stats_['pearson_p']:.3f})  "
            f"Spearman rho = {stats_['spearman_rho']:+.3f} "
            f"(p = {stats_['spearman_p']:.3f})"
        )

    # Pearson and Spearman disagreeing is not a curiosity, it is the finding.
    # A high r with a low rho means one extreme point is carrying the line
    # while the ordering of the rest does not follow it -- which is what a
    # non-monotone relationship looks like, and exactly the situation in which
    # quoting r as support for a linear trend would be wrong. So it is said
    # here rather than left for a reader to notice.
    linear = out.get("text -> image R@1", {})
    if linear:
        gap = abs(linear["pearson_r"]) - abs(linear["spearman_rho"])
        if gap > 0.3:
            print(
                f"\n  Pearson {linear['pearson_r']:+.2f} against Spearman "
                f"{linear['spearman_rho']:+.2f}: the line is held up by an "
                "extreme point\n  while the ordering does not follow it. Do not "
                "quote r as evidence of a\n  linear trend; the shape is not "
                "monotone."
            )

    if len(points) < 5:
        print(
            f"\n  n = {len(points)}. A Spearman coefficient on {len(points)} points "
            "cannot reach p < 0.05 even\n  when the ordering is perfect, and cannot "
            "tell a monotone relationship from a\n  non-monotone one. Read the "
            "ordering, not the p-value."
        )
    return out


def draw(points: list[Point], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(5.4, 3.4))
    ordered = sorted(points, key=lambda p: p.t2i_r1)
    span = ordered[-1].t2i_r1 - ordered[0].t2i_r1
    for index, point in enumerate(ordered):
        axes.errorbar(
            point.t2i_r1,
            point.benefit,
            yerr=point.benefit_sd / len(point.seeds) ** 0.5,
            fmt="o",
            capsize=3,
            color="#222222",
        )
        # Two of these encoders sit 1.3 points apart on this axis, which is
        # itself part of the result, so their labels have to be moved off
        # their points. A label moved off its point can end up beside its
        # neighbour's -- it did, and the figure then named the wrong encoder --
        # so every label is tied to its own point by a leader line and the
        # placement cannot mislead however crowded it gets.
        crowded = index + 1 < len(ordered) and (
            ordered[index + 1].t2i_r1 - point.t2i_r1 < 0.1 * span
        )
        left = crowded or index == len(ordered) - 1
        axes.annotate(
            point.label,
            (point.t2i_r1, point.benefit),
            textcoords="offset points",
            xytext=(-22 if left else 22, 14 if crowded else 6),
            ha="right" if left else "left",
            fontsize=8,
            arrowprops={
                "arrowstyle": "-",
                "linewidth": 0.5,
                "color": "#999999",
                "shrinkA": 1,
                "shrinkB": 5,
            },
        )
    axes.axhline(0.0, linewidth=0.8, linestyle="--", color="#888888")
    axes.axvline(
        points[0].chance,
        linewidth=0.8,
        linestyle=":",
        color="#888888",
    )
    axes.set_xlabel(
        f"zero-shot alignment: text -> image R@1 on Crossmodal-3600 "
        f"{points[0].language} (%)"
    )
    axes.set_ylabel("fusion benefit:\nalign - concat test AUROC")
    axes.set_title(
        "The dotted vertical line is chance; the dashed horizontal line is no benefit",
        fontsize=7,
        color="#555555",
    )
    axes.set_xlim(-0.08 * span, max(p.t2i_r1 for p in points) + 0.18 * span)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language", default="en", help="the language alignment is measured in"
    )
    parser.add_argument("--architecture", default="hateclipper")
    parser.add_argument("--figure", help="also write the figure here")
    parser.add_argument("--out", default="runs/alignment_benefit.json")
    args = parser.parse_args()

    points, skipped = collect(args.language, args.architecture)
    for note in skipped:
        print(f"skipped {note}")
    if len(points) < 2:
        raise SystemExit("fewer than two complete points; nothing to relate")
    if skipped:
        print()

    print_table(points, args.architecture, args.language)
    relationship = print_relationship(points)

    if args.figure:
        draw(points, Path(args.figure))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "language": args.language,
                "architecture": args.architecture,
                "points": [vars(p) for p in points],
                "relationship": relationship,
                "skipped": skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
