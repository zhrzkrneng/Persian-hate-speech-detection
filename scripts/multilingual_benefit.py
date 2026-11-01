"""Does measured alignment predict the fusion benefit across languages?

    python -m scripts.multilingual_benefit
    python -m scripts.multilingual_benefit --figure docs/paper/figures/multilingual.pdf

This is the test the four-encoder result asked for, and it is reported here
because it **failed**. On the Hateful Memes table the benefit of element-wise
fusion changed sign with the encoder's measured alignment. Extending that to
twenty encoder-language cells -- four encoders read zero-shot on Multi3Hate's
300 parallel memes in five languages, with each cell's alignment measured on
Crossmodal-3600 in the same language -- the relationship is absent.

What the script reports, in this order, because the order matters:

1. the twenty cells, with each one's alignment, both fusions, and the paired
   benefit over five shared seeds;
2. the correlation, over all cells and over the subsets above chance, so that
   a floor effect cannot be offered as the explanation without being checked;
3. the benefit averaged per encoder, which is what did survive.

Two reasons the quantity here is not the quantity there, both of which belong
in the paper rather than in a footnote:

- These cells are **zero-shot transfer**. The head trains on English Hateful
  Memes and reads a different corpus in a different language, where the
  Hateful Memes benefit is measured in distribution. Many cells sit near an
  AUROC of 50, and a difference between two near-random classifiers is not the
  same measurement as a difference between two working ones.
- The axis is **cross-language**, and Crossmodal-3600's captions are written
  per language rather than translated. German supplies more and longer
  captions than English for the same images, so part of any cross-language gap
  is annotation style. Comparing encoders within a language is clean;
  comparing languages is not.

Reporting a prediction that did not hold is worth more than leaving it
untested, so this script exists to make that reporting reproducible rather
than to rescue the claim.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

M3H = Path("runs/multi3hate")
RETRIEVAL = Path("runs/retrieval")

ENCODERS = {
    "openai/clip-vit-large-patch14": ("CLIP ViT-L/14", "clip-l14"),
    "openai/clip-vit-base-patch32": ("CLIP ViT-B/32", "clip-b32"),
    "jinaai/jina-clip-v2": ("jina-clip-v2", "jina-v2"),
    "CLIPfa": ("CLIPfa", "clipfa"),
}
LANGUAGES = ("en", "de", "es", "hi", "zh")
# Above this, a cell is doing something; below it, a difference between two
# fusions is a difference between two nearly random rankings.
ABOVE_CHANCE = (55.0, 60.0)


def read_cells() -> dict:
    """{(encoder, language, fusion): {seed: AUROC}} from the committed records."""
    out: dict = {}
    for path in sorted(M3H.rglob("seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        label = ENCODERS[payload["encoder"]][0]
        for language, metrics in payload["multi3hate"].items():
            key = (label, language, payload["fusion"])
            out.setdefault(key, {})[payload["seed"]] = metrics["auroc"] * 100
    return out


def alignment(preset: str, language: str) -> float | None:
    path = RETRIEVAL / f"{preset}_{language}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["results"]["text_to_image"][
        "R@1"
    ]


def collect() -> list[dict]:
    from scipy import stats

    cells = read_cells()
    rows = []
    for label, preset in ENCODERS.values():
        for language in LANGUAGES:
            align = cells.get((label, language, "align"))
            concat = cells.get((label, language, "concat"))
            score = alignment(preset, language)
            if not (align and concat and score is not None):
                continue
            seeds = sorted(set(align) & set(concat))
            if len(seeds) < 2:
                continue
            diffs = [align[s] - concat[s] for s in seeds]
            result = stats.ttest_1samp(diffs, 0.0)
            rows.append(
                {
                    "encoder": label,
                    "language": language,
                    "alignment": score,
                    "concat": st.mean(concat[s] for s in seeds),
                    "align": st.mean(align[s] for s in seeds),
                    "benefit": st.mean(diffs),
                    "benefit_sd": st.stdev(diffs),
                    "benefit_p": float(result.pvalue),
                    "seeds": seeds,
                }
            )
    return rows


def correlate(rows: list[dict]) -> dict:
    from scipy import stats

    xs = [r["alignment"] for r in rows]
    ys = [r["benefit"] for r in rows]
    pearson = stats.pearsonr(xs, ys)
    spearman = stats.spearmanr(xs, ys)
    return {
        "n": len(rows),
        "pearson_r": round(float(pearson.statistic), 3),
        "pearson_p": round(float(pearson.pvalue), 4),
        "spearman_rho": round(float(spearman.statistic), 3),
        "spearman_p": round(float(spearman.pvalue), 4),
    }


def per_encoder(rows: list[dict]) -> dict:
    from scipy import stats

    out = {}
    for label, _ in ENCODERS.values():
        sub = [r["benefit"] for r in rows if r["encoder"] == label]
        if len(sub) < 2:
            continue
        result = stats.ttest_1samp(sub, 0.0)
        out[label] = {
            "mean": round(st.mean(sub), 2),
            "sd": round(st.stdev(sub), 2),
            "p": round(float(result.pvalue), 3),
            "n": len(sub),
            "negative_in": sum(1 for b in sub if b < 0),
        }
    return out


def stars(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def report(rows: list[dict]) -> dict:
    print("zero-shot on Multi3Hate, 300 parallel memes per language, 5 seeds\n")
    print(
        f"{'encoder':<15} {'lang':<5} {'align R@1':>9} {'concat':>7} {'align':>7} "
        f"{'benefit':>8} {'sd':>5} {'p':>7}"
    )
    for row in rows:
        print(
            f"{row['encoder']:<15} {row['language']:<5} {row['alignment']:9.2f} "
            f"{row['concat']:7.2f} {row['align']:7.2f} {row['benefit']:+8.2f} "
            f"{row['benefit_sd']:5.2f} {row['benefit_p']:7.3f} "
            f"{stars(row['benefit_p'])}"
        )

    print("\nbenefit against alignment:")
    overall = correlate(rows)
    print(
        f"  all cells                  n = {overall['n']:2d}  "
        f"r = {overall['pearson_r']:+.3f} (p = {overall['pearson_p']:.3f})  "
        f"rho = {overall['spearman_rho']:+.3f} (p = {overall['spearman_p']:.3f})"
    )
    subsets = {}
    for floor in ABOVE_CHANCE:
        sub = [r for r in rows if r["concat"] >= floor]
        if len(sub) >= 3:
            subsets[floor] = correlate(sub)
            s = subsets[floor]
            print(
                f"  concat AUROC >= {floor:4.0f}          n = {s['n']:2d}  "
                f"r = {s['pearson_r']:+.3f} (p = {s['pearson_p']:.3f})  "
                f"rho = {s['spearman_rho']:+.3f} (p = {s['spearman_p']:.3f})"
            )
    print(
        "\n  A floor effect is the obvious objection, so it is checked rather "
        "than\n  argued: dropping the cells near chance does not produce a "
        "relationship."
    )

    encoders = per_encoder(rows)
    print("\nbenefit per encoder, across the five languages:")
    for label, value in encoders.items():
        print(
            f"  {label:<15} {value['mean']:+6.2f}  sd {value['sd']:4.2f}  "
            f"p = {value['p']:.3f} {stars(value['p']):<4}  "
            f"negative in {value['negative_in']}/{value['n']} languages"
        )
    print(
        "\n  This is what survived. jina-clip-v2 loses from element-wise fusion "
        "in every\n  language while holding the highest alignment in every one "
        "of them, which is\n  consistent with its Hateful Memes result and "
        "inconsistent with alignment\n  being what decides."
    )
    return {"overall": overall, "subsets": subsets, "per_encoder": encoders}


def draw(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = {
        "CLIP ViT-L/14": "o",
        "CLIP ViT-B/32": "s",
        "jina-clip-v2": "^",
        "CLIPfa": "D",
    }
    figure, axes = plt.subplots(figsize=(5.4, 3.4))
    for label in markers:
        sub = [r for r in rows if r["encoder"] == label]
        if not sub:
            continue
        axes.errorbar(
            [r["alignment"] for r in sub],
            [r["benefit"] for r in sub],
            yerr=[r["benefit_sd"] / len(r["seeds"]) ** 0.5 for r in sub],
            fmt=markers[label],
            capsize=2,
            markersize=5,
            linestyle="none",
            label=label,
            color="#222222",
            markerfacecolor="none" if label.startswith("CLIP ViT-B") else None,
        )
    axes.axhline(0.0, linewidth=0.8, linestyle="--", color="#888888")
    axes.set_xlabel("zero-shot alignment in that language: text -> image R@1 (%)")
    axes.set_ylabel("fusion benefit:\nalign - concat zero-shot AUROC")
    axes.legend(fontsize=7, frameon=False)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200)
    print(f"wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure")
    parser.add_argument("--out", default="runs/multilingual_benefit.json")
    args = parser.parse_args()

    rows = collect()
    if len(rows) < 3:
        raise SystemExit(f"only {len(rows)} complete cells; run the evaluation first")
    summary = report(rows)

    if args.figure:
        draw(rows, Path(args.figure))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cells": rows, **summary}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
