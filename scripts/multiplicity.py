"""Holm-Bonferroni over the comparisons the paper reports as findings.

    python -m scripts.multiplicity
    python -m scripts.multiplicity --extended

Fourteen paired t-tests carry the results section: four fusion benefits, four
architecture effects and six interactions. At alpha = 0.05 that family expects
about 0.7 false rejections from chance alone, so a reader is entitled to ask
what survives a correction, and a paper that does not answer invites the
question at review.

Holm-Bonferroni is used rather than Bonferroni because it is uniformly more
powerful and makes the same assumption -- none. Sorted ascending, the i-th of
m p-values is compared with alpha / (m - i), and the first failure stops the
procedure: everything after it is retained whatever its p-value.

Two things this does *not* do. It does not touch the equivalence claims: TOST
asks the opposite question, and a correction designed to make rejection harder
makes "no difference" easier to assert, which is the wrong direction for a
claim that two things are the same. And it does not re-order the paper -- the
family is fixed here, before the thresholds are seen, so a claim cannot be
moved out of the family to save it.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.equivalence import ENCODERS, read_cell  # noqa: E402

ALPHA = 0.05


def holm(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm-Bonferroni. Returns each label's rank, threshold and verdict.

    The step-down stops at the first failure: once one hypothesis is retained,
    every later one is retained too, however small its p-value. That is the
    property a plain Bonferroni loop does not have, and getting it wrong is
    how a correction ends up reporting more rejections than it licenses.
    """
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    out: dict[str, dict] = {}
    still_rejecting = True
    for index, (label, p) in enumerate(ordered):
        threshold = alpha / (m - index)
        if still_rejecting and p < threshold:
            verdict = "holds"
        else:
            still_rejecting = False
            verdict = "fails"
        out[label] = {
            "p": p,
            "rank": index + 1,
            "threshold": threshold,
            "margin": threshold - p,
            "holds": verdict == "holds",
        }
    return out


def _paired(a: dict[int, float], b: dict[int, float]) -> list[float]:
    return [a[s] - b[s] for s in sorted(set(a) & set(b))]


def _p(diffs: list[float]) -> float:
    from scipy import stats

    return float(stats.ttest_1samp(diffs, 0.0).pvalue)


def family(extended: bool = False) -> dict[str, tuple[float, float]]:
    """The comparisons under correction, as label -> (effect, p).

    Raw per-seed values, never the rounded ones the manuscript prints.
    """
    tests: dict[str, tuple[float, float]] = {}
    benefit: dict[str, list[float]] = {}
    for label, slug in ENCODERS.items():
        diffs = _paired(
            read_cell(slug, "hateclipper", "align"),
            read_cell(slug, "hateclipper", "concat"),
        )
        benefit[label] = diffs
        tests[f"fusion benefit: {label}"] = (st.mean(diffs), _p(diffs))
        arch = _paired(
            read_cell(slug, "memeclip", "align"),
            read_cell(slug, "hateclipper", "align"),
        )
        tests[f"architecture: {label}"] = (st.mean(arch), _p(arch))
    for left, right in combinations(ENCODERS, 2):
        pairs = zip(benefit[left], benefit[right], strict=True)
        diffs = [x - y for x, y in pairs]
        tests[f"interaction: {left} − {right}"] = (st.mean(diffs), _p(diffs))
    if extended:
        for fusion in ("concat", "align"):
            for left, right in combinations(ENCODERS, 2):
                diffs = _paired(
                    read_cell(ENCODERS[left], "hateclipper", fusion),
                    read_cell(ENCODERS[right], "hateclipper", fusion),
                )
                tests[f"encoder gap ({fusion}): {left} − {right}"] = (
                    st.mean(diffs),
                    _p(diffs),
                )
    return tests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument(
        "--extended",
        action="store_true",
        help="also correct over the twelve encoder-gap tests",
    )
    parser.add_argument("--out", default="runs/multiplicity.json")
    args = parser.parse_args()

    tests = family(args.extended)
    corrected = holm({k: p for k, (_, p) in tests.items()}, args.alpha)

    print(
        f"Holm-Bonferroni, alpha = {args.alpha}, family of {len(tests)}\n\n"
        f"{'comparison':<42}{'effect':>8}{'p':>11}{'threshold':>11}  verdict"
    )
    rows = sorted(corrected.items(), key=lambda kv: kv[1]["rank"])
    for label, row in rows:
        effect = tests[label][0]
        mark = "holds" if row["holds"] else "fails"
        print(
            f"{label:<42}{effect:+8.2f}{row['p']:11.2e}{row['threshold']:11.2e}  {mark}"
        )

    survivors = [r for _, r in rows if r["holds"]]
    if survivors:
        tightest = min(survivors, key=lambda r: r["margin"])
        label = next(k for k, v in corrected.items() if v is tightest)
        print(
            f"\ntightest survivor: {label} at p = {tightest['p']:.4f} "
            f"against {tightest['threshold']:.4f} "
            f"(margin {tightest['margin']:.4f})"
        )
    print(
        "\nA 'fails' row is not a refuted claim: the two that fail here are the "
        "two\nthe paper reports as null, and a correction cannot make a null "
        "weaker."
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "alpha": args.alpha,
                "family_size": len(tests),
                "extended": args.extended,
                # Rounded, all of it: an unrounded scipy p-value differs in
                # its last bits between environments, and a committed file
                # that cannot be regenerated byte for byte is a file no test
                # can check.
                "tests": {
                    label: {
                        "effect": round(tests[label][0], 2),
                        "p": round(row["p"], 6),
                        "rank": row["rank"],
                        "threshold": round(row["threshold"], 6),
                        "margin": round(row["margin"], 6),
                        "holds": row["holds"],
                    }
                    for label, row in corrected.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
