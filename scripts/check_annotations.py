"""Check the Persian annotation file and report agreement.

    python -m scripts.check_annotations --file data/persian/annotations.jsonl

Run it after the pilot round, and again whenever annotations are added. It
validates the schema, finds duplicate images, reports the label distribution,
and computes Krippendorff's alpha and raw pairwise agreement.

Nothing here decides anything. It reports, so that a problem is found while
it is still cheap to fix.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.persian import LABELS, AnnotationError, PersianMemeSet  # noqa: E402
from src.utils.agreement import krippendorff_alpha, pairwise_agreement  # noqa: E402

# Multi3Hate's five annotator groups scored between these on the same task.
# Printed as context so a normal alpha is not read as a failure.
REFERENCE_ALPHA = (0.39, 0.47)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default="data/persian/annotations.jsonl")
    parser.add_argument(
        "--min-annotators",
        type=int,
        default=3,
        help="the guideline's floor for majority voting (default 3)",
    )
    args = parser.parse_args()

    try:
        dataset = PersianMemeSet(args.file)
    except (AnnotationError, FileNotFoundError) as exc:
        print(f"FAILED: {exc}")
        raise SystemExit(1) from exc

    annotators = dataset.annotators()
    print(f"{len(dataset)} memes, {len(annotators)} annotators: {annotators}\n")

    # Coverage
    thin = [
        m.uid
        for m in dataset.memes
        if sum(label is not None for label in m.labels) < args.min_annotators
    ]
    if thin:
        print(
            f"UNDER-ANNOTATED  {len(thin)} meme(s) have fewer than "
            f"{args.min_annotators} labels, first: {thin[:5]}"
        )

    ties = [m.uid for m in dataset.memes if m.majority() is None]
    if ties:
        print(
            f"UNRESOLVED       {len(ties)} meme(s) have no majority. The "
            f"guideline resolves these by collecting more annotations, "
            f"first: {ties[:5]}"
        )

    duplicates = dataset.duplicate_images()
    if duplicates:
        print(f"DUPLICATE IMAGE  {len(duplicates)} hash(es) appear more than once")
        for digest, uids in list(duplicates.items())[:5]:
            print(f"                 {digest[:12]}… {uids}")

    # Distribution
    majorities = Counter(m.majority() for m in dataset.memes)
    print("\nmajority label")
    for value, count in sorted(majorities.items(), key=lambda kv: str(kv[0])):
        name = LABELS.get(value, "no majority")
        share = 100 * count / len(dataset)
        print(f"  {name:<12} {count:>5}  {share:5.1f}%")

    targets = Counter(
        a.target for m in dataset.memes for a in m.annotations if a.target
    )
    if targets:
        print("\ntarget (over hate annotations)")
        for name, count in targets.most_common():
            print(f"  {name:<22} {count:>5}")

    explicit = Counter(
        a.explicit for m in dataset.memes for a in m.annotations if a.label == 1
    )
    if explicit:
        total = sum(explicit.values())
        print("\nexplicitness")
        for value, count in explicit.most_common():
            name = {True: "explicit", False: "implicit"}.get(value, str(value))
            print(f"  {name:<12} {count:>5}  {100 * count / total:5.1f}%")

    # Agreement
    print("\nagreement")
    matrix = dataset.label_matrix()
    try:
        alpha = krippendorff_alpha(matrix)
        raw = pairwise_agreement(matrix)
    except ValueError as exc:
        print(f"  not computable: {exc}")
        return

    low, high = REFERENCE_ALPHA
    print(f"  Krippendorff's alpha   {alpha:.3f}")
    print(f"  pairwise agreement     {100 * raw:.1f}%")
    print(
        f"\n  Multi3Hate reports alpha between {low} and {high} across its five\n"
        "  annotator groups on this same task, and calls those comparable to or\n"
        "  higher than earlier hate speech research. An alpha in that range is a\n"
        "  normal result. A much higher one usually means annotators were trained\n"
        "  into agreement, or the sample was too easy."
    )
    if alpha < 0:
        print(
            "\n  A negative alpha is worse than chance, which usually means two\n"
            "  annotators read the guideline differently. Read the disagreements\n"
            "  before annotating further."
        )


if __name__ == "__main__":
    main()
