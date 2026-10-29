"""Collect retrieval results into one table.

    python -m scripts.summarise_retrieval --dir runs/retrieval

Prints chance R@1 alongside the scores, because a recall number means nothing
without it: 2% looks respectable until you notice chance was 2%.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="runs/retrieval")
    args = parser.parse_args()

    files = sorted(Path(args.dir).glob("*.json"))
    if not files:
        print(f"no results under {args.dir}")
        return

    rows = [json.loads(p.read_text(encoding="utf-8")) for p in files]

    header = (
        f"{'encoder':<12} {'lang':<5} {'imgs':>5} {'chance':>7} "
        f"{'t2i R@1':>8} {'R@5':>7} {'R@10':>7} "
        f"{'i2t R@1':>8} {'R@5':>7} {'R@10':>7}"
    )
    print(header)
    print("-" * len(header))

    for row in sorted(rows, key=lambda r: (r["language"], r["encoder"])):
        t2i = row["results"]["text_to_image"]
        i2t = row["results"]["image_to_text"]
        chance = 100.0 / row["images"]
        print(
            f"{row['encoder']:<12} {row['language']:<5} {row['images']:>5} "
            f"{chance:>6.2f}% "
            f"{t2i['R@1']:>8.2f} {t2i['R@5']:>7.2f} {t2i['R@10']:>7.2f} "
            f"{i2t['R@1']:>8.2f} {i2t['R@5']:>7.2f} {i2t['R@10']:>7.2f}"
        )

    print("\nt2i = caption retrieves its image; i2t = image retrieves a caption.")
    print("A score near chance means the encoder does not represent that language.")


if __name__ == "__main__":
    main()
