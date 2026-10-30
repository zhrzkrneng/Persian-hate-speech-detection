"""Fetch Multi3Hate and convert it to the layout this project reads.

    python -m scripts.download_multi3hate

Multi3Hate [Bui, von der Wense & Lauscher, NAACL 2025] is published by its
authors under CC BY-NC-ND 4.0: research use, no commercial use, and no
distribution of adapted material. So this script downloads it from the authors'
own repository rather than this project redistributing it, writes it where the
reader expects, and nothing derived from it is ever committed or uploaded --
/data/ and /features/ are gitignored, and the feature cache built from these
images stays on whichever machine computes it.

Produces:

    data/multi3hate/
      en.jsonl ... zh.jsonl      one line per meme, with every country's label
      img/<lang>/<id>.jpg

JSONL and plain JPEGs, not the published parquet, so that reading the result
needs nothing beyond the standard library and Pillow. The reader is imported
under CI's deliberately small dependency set, where neither `datasets` nor
`pyarrow` is installed.

Resumable: anything already on disk is left alone.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.multi3hate import (  # noqa: E402
    COUNTRIES,
    LANGUAGES,
    NATIVE_COUNTRY,
    SEPARATOR,
)

REPO = "https://github.com/MinhDucBui/Multi3Hate"
CITATION = (
    "Bui, von der Wense & Lauscher. Multi3Hate: Multimodal, Multilingual, and "
    "Multicultural Hate Speech Detection with Vision-Language Models. "
    "NAACL 2025. arXiv:2411.03888"
)


def clone(destination: Path) -> Path:
    """Shallow-clone the authors' repository, or reuse an existing checkout."""
    if (destination / ".git").exists():
        print(f"reusing {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {REPO}")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO, str(destination)],
        check=True,
        env={"GIT_LFS_SKIP_SMUDGE": "1", "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    return destination


def read_labels(source: Path) -> dict[str, dict[str, int]]:
    path = source / "data" / "final_annotations.csv"
    labels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            uid = row["Meme ID"].strip()
            labels[uid] = {
                country: int(row[country])
                for country in COUNTRIES
                if row.get(country, "").strip() != ""
            }
    return labels


def read_captions(source: Path, language: str) -> dict[str, tuple[str, str]]:
    """{id: (text, template)} with the dataset's <sep> rendered as a newline.

    The separator marks the break between a meme's upper and lower text. A
    newline is how that reads on the image, and it keeps a meme's text one
    string, as HatefulMemesSplit presents it.
    """
    path = source / "data" / "captions" / f"{language}.csv"
    out: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            uid = row["Meme ID"].strip()
            text = (row.get("Translation") or "").replace(SEPARATOR, "\n")
            text = "\n".join(part.strip() for part in text.split("\n")).strip()
            out[uid] = (text, (row.get("Template Name") or "").strip())
    return out


def convert(source: Path, root: Path) -> None:
    labels = read_labels(source)
    print(f"{len(labels)} labelled memes, countries {list(COUNTRIES)}")

    for language in LANGUAGES:
        captions = read_captions(source, language)
        images = {
            path.stem: path
            for path in (source / "data" / "memes" / language).rglob("*.jpg")
        }

        missing_caption = sorted(set(images) - set(captions))
        missing_image = sorted(set(captions) - set(images))
        if missing_caption or missing_image:
            raise SystemExit(
                f"{language}: {len(missing_caption)} image(s) without a caption "
                f"and {len(missing_image)} caption(s) without an image. "
                "The copy is incomplete; do not convert it."
            )

        target = root / "img" / language
        target.mkdir(parents=True, exist_ok=True)
        rows = []
        for uid in sorted(images, key=int):
            text, template = captions[uid]
            destination = target / f"{uid}.jpg"
            if not destination.exists():
                shutil.copyfile(images[uid], destination)
            rows.append(
                {
                    "id": uid,
                    "img": f"img/{language}/{uid}.jpg",
                    "text": text,
                    "template": template,
                    "labels": labels.get(uid, {}),
                }
            )

        out = root / f"{language}.jsonl"
        out.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        native = NATIVE_COUNTRY[language]
        hateful = sum(1 for r in rows if r["labels"].get(native) == 1)
        print(
            f"{language}: {len(rows):>4} memes -> {out.name}  "
            f"native label {native}: {hateful} hateful "
            f"({100 * hateful / max(len(rows), 1):.1f}%)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/multi3hate")
    parser.add_argument(
        "--source",
        help="an existing checkout of the authors' repository; cloned if absent",
    )
    args = parser.parse_args()

    root = Path(args.root)
    source = Path(args.source) if args.source else root / "_upstream"
    convert(clone(source), root)

    print(
        "\nCite it:\n  "
        + CITATION
        + "\n\nCC BY-NC-ND 4.0: research use, non-commercial, and no "
        "distribution of\nadapted material. Neither these images nor any "
        "feature cache built from\nthem may be published or uploaded."
    )


if __name__ == "__main__":
    main()
