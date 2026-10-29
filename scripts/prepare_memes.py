"""Turn a folder of collected memes into an annotation file.

    python -m scripts.prepare_memes --images data/persian/raw --dry-run
    python -m scripts.prepare_memes --images data/persian/raw \
        --out data/persian/annotations.jsonl

Reports duplicates and possible identifying detail, then writes the skeleton
annotators work from: one line per meme, valid against the schema in
src/data/persian.py, with an empty `annotations` list and an empty `text` for
each.

The text is left empty on purpose. The guideline says meme text is transcribed
by hand, because Persian OCR on meme typography is unreliable and a wrong
transcription is a label error no model can recover from.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.collection import (  # noqa: E402
    exact_duplicates,
    find_identifiers,
    near_duplicates,
    scan,
)

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="folder of collected memes")
    parser.add_argument("--out", help="where to write the annotation skeleton")
    parser.add_argument(
        "--dry-run", action="store_true", help="report only, write nothing"
    )
    parser.add_argument(
        "--near-threshold",
        type=int,
        default=5,
        help="perceptual-hash distance counted as a near duplicate (default 5)",
    )
    parser.add_argument("--prefix", default="fa", help="id prefix (default fa)")
    args = parser.parse_args()

    if not args.dry_run and not args.out:
        parser.error("pass --out, or --dry-run to report without writing")

    folder = Path(args.images)
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")

    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUFFIXES)
    if not paths:
        raise SystemExit(f"no images under {folder} (looked for {sorted(SUFFIXES)})")

    memes = scan(paths)
    print(f"{len(memes)} image(s) under {folder}\n")

    exact = exact_duplicates(memes)
    if exact:
        print(f"EXACT DUPLICATES  {len(exact)} group(s)")
        for digest, group in list(exact.items())[:10]:
            print(f"  {digest[:12]}…  {[p.name for p in group]}")
        print("  Keep one of each; the rest are not independent judgements.\n")

    near = near_duplicates(memes, args.near_threshold)
    if near:
        print(
            f"NEAR DUPLICATES   {len(near)} pair(s) within {args.near_threshold} bits"
        )
        for first, second, distance in near[:10]:
            print(f"  d={distance}  {first.name}  ~  {second.name}")
        print(
            "  Check these by eye. Two memes on one template with different\n"
            "  text are different memes, and that difference is the task.\n"
        )

    # Identifier scan runs over filenames here, since no text exists yet. It is
    # worth running again over the transcriptions before annotation starts.
    flagged = {
        meme.path.name: found
        for meme in memes
        if (found := find_identifiers(meme.path.name))
    }
    if flagged:
        print(f"POSSIBLE IDENTIFIERS in {len(flagged)} filename(s)")
        for name, found in list(flagged.items())[:10]:
            print(f"  {name}: {found}")
        print()

    print(
        "Reminder: anonymise before annotators see these. Usernames, handles,\n"
        "profile pictures and phone numbers belonging to private individuals\n"
        "are removed or blurred at collection time, and no pattern list\n"
        "catches a face or a logo.\n"
    )

    if args.dry_run:
        print("dry run: nothing written")
        return

    # One line per unique image; exact duplicates collapse to their first path.
    seen: set[str] = set()
    rows = []
    for index, meme in enumerate(memes):
        if meme.sha256 in seen:
            continue
        seen.add(meme.sha256)
        rows.append(
            {
                "id": f"{args.prefix}_{index:04d}",
                "image_sha256": meme.sha256,
                "source": {"platform": "", "url": "", "collected": ""},
                "text": "",
                "annotations": [],
            }
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out}: {len(rows)} unique meme(s)")
    print(
        "\nNext: fill in each source, transcribe each text by hand, then add\n"
        "annotations. scripts/check_annotations.py validates the result."
    )


if __name__ == "__main__":
    main()
