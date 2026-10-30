"""Compare two result cells seed by seed, with a paired test.

    python -m scripts.compare_cells \
        --a runs/hmc/openai__clip-vit-large-patch14/hateclipper/align \
        --b runs/hmc/openai__clip-vit-large-patch14/hateclipper/concat

    python -m scripts.compare_cells --table

Two independent means with their own standard deviations are the wrong way to
compare variants here. Both ran the same seeds, so the seed noise they share
can be removed by differencing per seed, which is what a paired comparison
does -- and the differences this project is about are well under a point of
AUROC, so that noise is most of what two separate error bars contain.

The reason this script exists rather than a heredoc: mean - sd > 0 was used as
a verdict once, and it is a rule of thumb, not a test. At five seeds it called
a difference of +0.72 +/- 0.62 settled when a paired t-test puts it at
p = 0.061. Every comparison quoted in the paper should come from one command
that reports the test, so the claim and the evidence cannot drift apart.

With five seeds the test has four degrees of freedom and very little power:
it can confirm a large effect and cannot rule out a small one. "n.s." here
means "these five runs do not establish it", never "there is no difference".
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
CLIP = "openai__clip-vit-large-patch14"
B32 = "openai__clip-vit-base-patch32"
JINA = "jinaai__jina-clip-v2"
CLIPFA = "CLIPfa"

# Every encoder on the axis, in the order the paper's table uses.
ENCODERS = {
    "CLIP ViT-L/14": CLIP,
    "CLIP ViT-B/32": B32,
    "jina-clip-v2": JINA,
    "CLIPfa": CLIPFA,
}

# The comparisons the paper makes, so the table and the prose cannot diverge.
COMPARISONS = [
    *[
        (
            f"fusion: align - concat, {label}",
            f"{slug}/hateclipper/align",
            f"{slug}/hateclipper/concat",
        )
        for label, slug in ENCODERS.items()
    ],
    *[
        (
            f"architecture: MemeCLIP - HateCLIPper, {label}",
            f"{slug}/memeclip/align",
            f"{slug}/hateclipper/align",
        )
        for label, slug in ENCODERS.items()
    ],
    # Which encoder reaches the higher score, which is a different question
    # from how much fusion buys it -- and the two answers disagree, which is
    # the point: CLIP ViT-B/32 scores below jina-clip-v2 and gains more from
    # fusion than jina does.
    *[
        (
            f"encoder: {left} - {right}, under {fusion}",
            f"{ENCODERS[left]}/hateclipper/{fusion}",
            f"{ENCODERS[right]}/hateclipper/{fusion}",
        )
        for fusion in ("align", "concat")
        for left, right in combinations(ENCODERS, 2)
    ],
]


def read_cell(directory: Path, split: str) -> dict[int, float]:
    files = sorted(directory.glob("seed*.json"))
    if not files:
        raise SystemExit(f"no seed*.json under {directory}")
    out: dict[int, float] = {}
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[payload["seed"]] = payload[split]["auroc"] * 100
    return out


def paired(a: dict[int, float], b: dict[int, float]) -> tuple[list[int], list[float]]:
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2:
        raise SystemExit(f"need at least two shared seeds, found {seeds}")
    return seeds, [a[s] - b[s] for s in seeds]


def report(label: str, diffs: list[float], seeds: list[int]) -> None:
    from scipy import stats

    mean = st.mean(diffs)
    sd = st.stdev(diffs)
    sem = sd / len(diffs) ** 0.5
    result = stats.ttest_1samp(diffs, 0.0)
    p = float(result.pvalue)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."

    print(f"{label}")
    print(f"  seeds {seeds}")
    print("  per seed: " + "  ".join(f"{d:+.2f}" for d in diffs))
    print(
        f"  mean {mean:+.2f}  sd {sd:.2f}  sem {sem:.2f}  "
        f"t({len(diffs) - 1}) = {float(result.statistic):.2f}  p = {p:.3f}  {stars}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", help="result directory, the left side of the difference")
    parser.add_argument("--b", help="result directory, the right side")
    parser.add_argument(
        "--split", default="test", choices=("test", "val"), help="default test"
    )
    parser.add_argument(
        "--table",
        action="store_true",
        help="run every comparison the paper makes, plus the interaction",
    )
    args = parser.parse_args()

    if args.table:
        print(f"paired comparison of {args.split} AUROC, two-tailed t-test\n")
        fusion: dict[str, list[float]] = {}
        for label, a, b in COMPARISONS:
            seeds, diffs = paired(
                read_cell(HMC / a, args.split), read_cell(HMC / b, args.split)
            )
            report(label, diffs, seeds)
            print()
            if label.startswith("fusion"):
                fusion[label.split(", ")[-1]] = diffs

        # The claim is not any one fusion benefit but whether the benefit
        # depends on the encoder. Differencing two differences cancels the seed
        # noise they share, which is why an interaction can resolve while
        # neither of its halves does.
        #
        # The pair that matters most is the two CLIP variants. They differ by
        # about three and a half points of absolute AUROC, so if the fusion
        # benefit tracked model capacity they would differ here too. Finding
        # that they do not, while both differ from the other two encoders, is
        # what separates "the feature space decides" from "the stronger model
        # decides".
        print("INTERACTIONS: does the fusion benefit differ between encoders?\n")
        for left, right in combinations(ENCODERS, 2):
            if not {left, right} <= fusion.keys():
                continue
            interaction = [
                a - b for a, b in zip(fusion[left], fusion[right], strict=True)
            ]
            report(
                f"{left} - {right}",
                interaction,
                sorted(read_cell(HMC / COMPARISONS[0][1], args.split)),
            )
            print()
        return

    if not (args.a and args.b):
        parser.error("pass --a and --b, or --table")
    seeds, diffs = paired(
        read_cell(Path(args.a), args.split), read_cell(Path(args.b), args.split)
    )
    report(f"{args.a}\n  minus {args.b}", diffs, seeds)


if __name__ == "__main__":
    main()
