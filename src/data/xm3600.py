"""Crossmodal-3600 (Thapliyal et al., EMNLP 2022), used here for Persian.

3600 images, each captioned in 36 languages by a native speaker looking at the
photograph rather than translating an English caption. Persian ("fa") is one of
them, which makes this the one public benchmark that can say, with numbers,
whether a given encoder understands Persian well enough to be worth building
the rest of the project on.

Captions and images ship separately: captions from the project page, images
from Open Images. `scripts/download_xm3600.py` assembles both into:

    data/xm3600/
      captions.jsonl
      img/<image_key>.jpg
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaptionedImage:
    key: str
    image_path: Path
    captions: tuple[str, ...]


class Crossmodal3600:
    """One language's slice of the benchmark."""

    def __init__(self, root: str | Path, language: str = "fa") -> None:
        self.root = Path(root)
        self.language = language
        self.records = self._read()

    def _read(self) -> list[CaptionedImage]:
        path = self.root / "captions.jsonl"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run scripts/download_xm3600.py first."
            )

        records: list[CaptionedImage] = []
        missing_language = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                entry = row.get(self.language)
                if not entry or not entry.get("caption"):
                    missing_language += 1
                    continue
                key = row["image/key"]
                records.append(
                    CaptionedImage(
                        key=key,
                        image_path=self.root / "img" / f"{key}.jpg",
                        captions=tuple(entry["caption"]),
                    )
                )

        if not records:
            raise ValueError(
                f"no captions for language {self.language!r} in {path}; "
                "check the language code against the file's keys"
            )
        if missing_language:
            print(
                f"warning: {missing_language} image(s) have no {self.language} "
                "caption and were skipped"
            )
        return records

    def with_images(self) -> list[CaptionedImage]:
        """Only the records whose image was actually downloaded."""
        return [r for r in self.records if r.image_path.exists()]

    def missing_images(self) -> list[str]:
        return [r.key for r in self.records if not r.image_path.exists()]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> CaptionedImage:
        return self.records[index]
