"""Does a non-significant difference actually exclude an effect?

    python -m scripts.equivalence
    python -m scripts.equivalence --margin 0.5

This project's central claim is that the fusion benefit tracks the encoder's
feature space and not its capacity, and the evidence offered for the second
half is a comparison that came out *non-significant*: the two CLIP variants
differ by 3.45 AUROC under concatenation yet their fusion benefits test at
p = 0.576. A non-significant paired test is not evidence of equivalence. At
five seeds it means only that four degrees of freedom could not resolve
whatever is there, and a reviewer is right to say so.

What answers the question is a confidence interval and an equivalence test.

- The interval says what sizes remain plausible. If it spans -2 to +2 AUROC,
  nothing has been excluded and the claim is unsupported.
- Two one-sided tests (TOST) against a margin asks the opposite question from
  the usual one: not "is the difference non-zero?" but "is it smaller than
  something that would matter?". It passes only when **both** one-sided tests
  reject, which is the same as the interval lying inside the margin.

The margin has to be chosen for a reason, not fitted after the fact. The
default here is 1.0 AUROC: the largest fusion benefit this study measures is
1.93 and the largest architecture effect 1.33, so an interaction below one
point is smaller than any effect the paper reports as real. State the margin
before looking, and report the interval whatever it says.

TOST on five observations has very little power -- it will usually fail to
establish equivalence even when the effect is genuinely tiny. A failure here
means "these runs establish neither a difference nor its absence", and that is
the honest reading to print.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HMC = Path("runs/hmc")
ENCODERS = {
    "CLIP ViT-L/14": "openai__clip-vit-large-patch14",
    "CLIP ViT-B/32": "openai__clip-vit-base-patch32",
    "jina-clip-v2": "jinaai__jina-clip-v2",
    "CLIPfa": "CLIPfa",
}
# Chosen before the numbers were looked at: smaller than any effect this study
# reports as real (largest fusion benefit 1.93, largest architecture effect
# 1.33), so a difference under it cannot carry an argument in this paper.
DEFAULT_MARGIN = 1.0


def read_cell(slug: str, architecture: str, fusion: str) -> dict[int, float]:
    directory = HMC / slug / architecture / fusion
    out: dict[int, float] = {}
    for path in sorted(directory.glob("seed*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[payload["seed"]] = payload["test"]["auroc"] * 100
    return out


def benefit(slug: str) -> dict[int, float]:
    """align - concat per seed, over the seeds both cells ran."""
    align = read_cell(slug, "hateclipper", "align")
    concat = read_cell(slug, "hateclipper", "concat")
    return {s: align[s] - concat[s] for s in sorted(set(align) & set(concat))}


def interval(values: list[float], confidence: float = 0.95) -> tuple[float, float]:
    from scipy import stats

    n = len(values)
    sem = st.stdev(values) / n**0.5
    half = stats.t.ppf(0.5 + confidence / 2, n - 1) * sem
    return st.mean(values) - half, st.mean(values) + half


def tost(values: list[float], margin: float) -> dict:
    """Two one-sided tests. Equivalence holds only if both reject."""
    from scipy import stats

    n = len(values)
    mean = st.mean(values)
    sem = st.stdev(values) / n**0.5
    # H0 for each side is that the true difference is at or beyond the margin.
    lower = stats.t.sf((mean + margin) / sem, n - 1)  # tests mean <= -margin
    upper = stats.t.cdf((mean - margin) / sem, n - 1)  # tests mean >= +margin
    p = max(float(lower), float(upper))
    low, high = interval(values, 0.90)  # the interval TOST corresponds to
    return {
        "mean": round(mean, 2),
        "sd": round(st.stdev(values), 2),
        "n": n,
        "p_tost": round(p, 4),
        "equivalent": bool(p < 0.05),
        "ci90_low": round(low, 2),
        "ci90_high": round(high, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--out", default="runs/equivalence.json")
    args = parser.parse_args()

    benefits = {label: benefit(slug) for label, slug in ENCODERS.items()}
    from scipy import stats

    print(
        f"interaction in the fusion benefit between two encoders\n"
        f"equivalence margin: +/- {args.margin:.2f} AUROC\n"
    )
    print(
        f"{'pair':<34} {'diff':>7} {'95% CI':>16} {'p(diff)':>8} "
        f"{'p(TOST)':>8}  verdict"
    )
    results = {}
    for left, right in combinations(ENCODERS, 2):
        seeds = sorted(set(benefits[left]) & set(benefits[right]))
        if len(seeds) < 3:
            continue
        diffs = [benefits[left][s] - benefits[right][s] for s in seeds]
        low, high = interval(diffs)
        equivalence = tost(diffs, args.margin)
        difference = float(stats.ttest_1samp(diffs, 0.0).pvalue)

        if difference < 0.05:
            verdict = "differs"
        elif equivalence["equivalent"]:
            verdict = "equivalent"
        else:
            verdict = "undecided"
        results[f"{left} - {right}"] = {
            **equivalence,
            "p_difference": round(difference, 4),
            "ci95_low": round(low, 2),
            "ci95_high": round(high, 2),
            "verdict": verdict,
            "seeds": seeds,
        }
        print(
            f"{left + ' - ' + right:<34} {st.mean(diffs):+7.2f} "
            f"[{low:+6.2f},{high:+6.2f}] {difference:8.3f} "
            f"{equivalence['p_tost']:8.3f}  {verdict}"
        )

    print(
        "\n  'differs'    the paired test rejects no difference.\n"
        "  'equivalent' the difference is inside the margin: an effect that\n"
        "               size or larger is excluded.\n"
        "  'undecided'  neither. These runs establish neither a difference nor\n"
        "               its absence, and a non-significant p is not the second."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"margin": args.margin, "pairs": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
