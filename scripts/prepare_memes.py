"""Turn a folder of collected memes into an annotation file.

    # after each collecting session: list the new files for provenance
    python -m scripts.prepare_memes --images data/persian/raw --init-manifest

    # report duplicates and identifying detail, write nothing
    python -m scripts.prepare_memes --images data/persian/raw --dry-run

    # build or extend the annotation skeleton
    python -m scripts.prepare_memes --images data/persian/raw \
        --out data/persian/annotations.jsonl

Reports duplicates and possible identifying detail, then writes the skeleton
annotators work from: one line per meme, valid against the schema in
src/data/persian.py, with an empty `annotations` list and an empty `text` for
each.

**Running this twice is safe, and it has to be**, because collection happens
in sessions: save some memes, prepare, annotate, save some more. An earlier
version rebuilt the file from scratch every time. Re-running it after one more
meme had been collected destroyed nine annotations and three hand
transcriptions without a word, exited 0, and renumbered every id, so `fa_0000`
afterwards named a different meme than the one the annotators had discussed.
So: existing rows are matched by image hash and kept exactly as they are, ids
are never reassigned, new memes are appended with fresh ids, and a row whose
image has gone missing is reported rather than dropped.

The text is left empty on purpose. The guideline says meme text is transcribed
by hand, because Persian OCR on meme typography is unreliable and a wrong
transcription is a label error no model can recover from.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.collection import (  # noqa: E402
    ManifestError,
    exact_duplicates,
    extend_manifest,
    find_identifiers,
    manifest_gaps,
    near_duplicates,
    read_manifest,
    scan,
)
from src.data.persian import AnnotationError, PersianMemeSet  # noqa: E402

SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_MANIFEST = "data/persian/sources.csv"


def load_existing(path: Path) -> list[dict]:
    """Read an annotation file as raw JSON, keeping every field.

    Raw rather than through PersianMemeSet: this file is rewritten from what is
    read here, so anything the reader does not model -- a field added later, a
    note someone put on a row -- has to survive the round trip rather than be
    dropped to the schema.
    """
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: invalid JSON ({exc})") from exc
            if not isinstance(row, dict) or "image_sha256" not in row:
                raise SystemExit(f"{path}:{number}: not an annotation row")
            rows.append(row)
    return rows


def next_index(rows: list[dict], prefix: str) -> int:
    """One past the highest id already used, so ids are never reused."""
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    used = [
        int(match.group(1))
        for row in rows
        if (match := pattern.match(str(row.get("id", ""))))
    ]
    return max(used) + 1 if used else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, help="folder of collected memes")
    parser.add_argument("--out", help="annotation file to create or extend")
    parser.add_argument(
        "--dry-run", action="store_true", help="report only, write nothing"
    )
    parser.add_argument(
        "--manifest",
        help=f"provenance CSV (default {DEFAULT_MANIFEST} when it exists)",
    )
    parser.add_argument(
        "--init-manifest",
        action="store_true",
        help="add a blank row to the manifest for each new image, then stop",
    )
    parser.add_argument(
        "--near-threshold",
        type=int,
        default=5,
        help="perceptual-hash distance counted as a near duplicate (default 5)",
    )
    parser.add_argument("--prefix", default="fa", help="id prefix (default fa)")
    args = parser.parse_args()

    if not (args.dry_run or args.out or args.init_manifest):
        parser.error("pass --out, --init-manifest, or --dry-run")

    folder = Path(args.images)
    if not folder.is_dir():
        raise SystemExit(f"not a folder: {folder}")

    paths = sorted(p for p in folder.rglob("*") if p.suffix.lower() in SUFFIXES)
    if not paths:
        raise SystemExit(f"no images under {folder} (looked for {sorted(SUFFIXES)})")

    manifest_path = Path(args.manifest or DEFAULT_MANIFEST)

    if args.init_manifest:
        added = extend_manifest(manifest_path, [p.name for p in paths])
        print(f"{manifest_path}: {len(paths)} image(s), {len(added)} row(s) added")
        if added:
            print(
                "\nFill in platform, url and collected (YYYY-MM-DD) for the new\n"
                "rows. A spreadsheet opens this file; save it as CSV again."
            )
        return

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

    # Provenance. Explicit --manifest makes a missing file an error; the
    # default path is a convenience, so its absence is a warning.
    sources = {}
    if args.manifest or manifest_path.exists():
        try:
            sources = read_manifest(manifest_path)
        except ManifestError as exc:
            raise SystemExit(f"FAILED: {exc}") from exc
        gaps = manifest_gaps([p.name for p in paths], sources)
        if gaps:
            print(
                f"PROVENANCE        {len(gaps)} image(s) incomplete in {manifest_path}"
            )
            for name, issues in list(gaps.items())[:10]:
                print(f"  {name}: {'; '.join(issues)}")
            print(
                "  The platform and date are knowable now and not later. Fill\n"
                "  these in before the memes reach an annotator.\n"
            )
        without_url = sum(
            1 for p in paths if p.name in sources and not sources[p.name].url
        )
        if without_url:
            print(
                f"  {without_url} image(s) have no source URL. Allowed -- a "
                "forwarded meme\n  has none -- but the paper should say how "
                "many.\n"
            )
    else:
        print(
            f"NO MANIFEST       {manifest_path} does not exist, so `source` is "
            "left blank.\n  Create it with --init-manifest: the platform and "
            "the collection date\n  cannot be recovered from a folder of "
            "images a week later.\n"
        )

    print(
        "Reminder: anonymise before annotators see these. Usernames, handles,\n"
        "profile pictures and phone numbers belonging to private individuals\n"
        "are removed or blurred at collection time, and no pattern list\n"
        "catches a face or a logo.\n"
    )

    if args.dry_run:
        print("dry run: nothing written")
        return

    out = Path(args.out)
    existing = load_existing(out) if out.exists() else []
    by_hash = {row["image_sha256"]: row for row in existing}

    # Exact duplicates collapse to the first path, so each image hash is one
    # row and one filename.
    name_of: dict[str, str] = {}
    for meme in memes:
        name_of.setdefault(meme.sha256, meme.path.name)

    index = next_index(existing, args.prefix)
    rows = list(existing)
    added = 0
    for digest, name in name_of.items():
        if digest in by_hash:
            continue
        rows.append(
            {
                "id": f"{args.prefix}_{index:04d}",
                "image_sha256": digest,
                "source": sources[name].as_dict()
                if name in sources
                else {"platform": "", "url": "", "collected": ""},
                "text": "",
                "annotations": [],
            }
        )
        index += 1
        added += 1

    # Fill provenance that was unknown when a row was first written, without
    # touching a source already recorded: the manifest is edited by hand and
    # the annotation file may have been too.
    filled = 0
    for row in rows:
        name = name_of.get(row["image_sha256"])
        source = row.get("source") or {}
        if name in sources and not any(source.get(k) for k in source):
            row["source"] = sources[name].as_dict()
            filled += 1

    missing = [row["id"] for row in existing if row["image_sha256"] not in name_of]

    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(out.suffix + ".tmp")
    temporary.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    # Validate before replacing the file annotation work lives in: a skeleton
    # that fails the reader is found here, not after a round of annotation.
    try:
        PersianMemeSet(temporary)
    except AnnotationError as exc:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"FAILED: would have written an invalid file: {exc}") from exc
    temporary.replace(out)

    kept = len(existing)
    annotated = sum(1 for row in rows if row.get("annotations"))
    print(f"wrote {out}: {len(rows)} meme(s) -- {kept} kept, {added} added")
    if filled:
        print(f"  filled in {filled} source(s) from {manifest_path}")
    if annotated:
        print(f"  {annotated} row(s) already carry annotations; untouched")
    if missing:
        print(
            f"  {len(missing)} row(s) reference an image not under {folder}, "
            f"first: {missing[:5]}\n"
            "  Kept, not dropped: a renamed or moved file must not take its\n"
            "  annotations with it. Restore the image or remove the row by hand."
        )
    print(
        "\nNext: transcribe each text by hand, then annotate with "
        "tools/annotate.html.\n"
        "scripts/check_annotations.py validates the result."
    )


if __name__ == "__main__":
    main()
