"""Verify a local copy of the Hateful Memes data before trusting it.

    python -m scripts.check_hmc --root data/hmc
    python -m scripts.check_hmc --root data/hmc --skip-images   # faster

Run this on whatever copy you obtained. The official distribution has become
hard to reach and copies circulate through mirrors; a mirror that is a subset,
or that re-splits the data, yields numbers comparable with nothing, and
training on it would simply succeed.

Exits non-zero if anything is wrong, so it can gate a pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.hmc_check import (  # noqa: E402
    EXPECTED_SIZES,
    leakage,
    read_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/hmc")
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="do not check that every referenced image exists",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"not a folder: {root}")
        print("See scripts/download_hmc.sh for the expected layout.")
        raise SystemExit(2)

    print(f"checking {root}\n")
    print(f"{'split':<12} {'rows':>6} {'expected':>9} {'labelled':>9}  status")
    print("-" * 56)

    reports = [
        read_split(root, split, check_images=not args.skip_images)
        for split in EXPECTED_SIZES
    ]

    for report in reports:
        status = "ok" if not report.problems else "PROBLEM"
        rows = str(report.count) if report.present else "-"
        print(
            f"{report.split:<12} {rows:>6} {report.expected:>9} "
            f"{report.labelled:>9}  {status}"
        )

    problems = [(r.split, p) for r in reports for p in r.problems]

    # Label balance, for the splits that ship labelled.
    print()
    for report in reports:
        if report.label_counts:
            total = sum(report.label_counts.values())
            share = 100 * report.label_counts.get(1, 0) / total
            print(
                f"{report.split:<12} hateful {report.label_counts.get(1, 0):>5} "
                f"({share:4.1f}%)   non-hateful {report.label_counts.get(0, 0):>5}"
            )

    overlaps = leakage(root)
    for first, second, count in overlaps:
        problems.append((first, f"{count} id(s) shared with {second}"))

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for split, problem in problems:
            print(f"  {split}: {problem}")
        print(
            "\nA copy that does not match the published splits produces numbers\n"
            "that cannot be compared with any paper. Fix the copy, or state the\n"
            "difference explicitly wherever the numbers appear."
        )
        raise SystemExit(1)

    print("All splits match the published sizes, and train shares no ids with")
    print("any evaluation split. Note that dev_seen and dev_unseen overlapping")
    print("is expected: it is a property of the original release.")


if __name__ == "__main__":
    main()
