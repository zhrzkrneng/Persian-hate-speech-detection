"""The Persian meme annotation set.

Schema and procedure are defined in docs/annotation-guideline.md. This module
reads the file and refuses anything that does not match, because a schema
error found at training time is a week of annotation already spent.

One JSONL line per meme. Images are never committed; the file carries hashes
and source URLs so a meme can be traced without redistributing it.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LABELS = {0: "non-hate", 1: "hate"}
TARGETS = {
    "gender",
    "religion",
    "ethnicity_nationality",
    "political",
    "disability",
    "sexual_orientation",
    "other",
}


@dataclass(frozen=True)
class Annotation:
    annotator: str
    label: int | None  # None is the guideline's "I don't know"
    target: str | None
    explicit: bool | None


@dataclass(frozen=True)
class AnnotatedMeme:
    uid: str
    image_sha256: str
    text: str
    annotations: tuple[Annotation, ...]
    label: int | None
    source: dict

    @property
    def labels(self) -> tuple[int | None, ...]:
        return tuple(a.label for a in self.annotations)

    def majority(self) -> int | None:
        """Majority label over non-abstaining annotators; None on a tie.

        The guideline resolves ties by collecting more annotations, so a tie
        here means the meme is not finished, not that it is borderline.
        """
        votes = Counter(label for label in self.labels if label is not None)
        if not votes:
            return None
        ranked = votes.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]


class AnnotationError(ValueError):
    """A line that does not match the guideline's schema."""


def _parse_annotation(raw: dict, where: str) -> Annotation:
    label = raw.get("label")
    if label not in (0, 1, None):
        raise AnnotationError(f"{where}: label must be 0, 1 or null, got {label!r}")

    target = raw.get("target")
    if label == 1:
        if target not in TARGETS:
            raise AnnotationError(
                f"{where}: a hate label needs a target from {sorted(TARGETS)}, "
                f"got {target!r}"
            )
        if not isinstance(raw.get("explicit"), bool):
            raise AnnotationError(f"{where}: a hate label needs explicit true/false")
    elif target is not None:
        raise AnnotationError(
            f"{where}: target {target!r} set on a non-hate label; the guideline "
            "only asks for a target when the label is hate"
        )

    annotator = raw.get("annotator")
    if not isinstance(annotator, str) or not annotator:
        raise AnnotationError(f"{where}: annotator must be a non-empty string")

    return Annotation(
        annotator=annotator,
        label=label,
        target=target,
        explicit=raw.get("explicit"),
    )


def _parse_meme(row: dict, where: str) -> AnnotatedMeme:
    for field in ("id", "image_sha256", "text", "annotations"):
        if field not in row:
            raise AnnotationError(f"{where}: missing field {field!r}")

    digest = row["image_sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise AnnotationError(f"{where}: image_sha256 must be 64 hex characters")

    raw_annotations = row["annotations"]
    # An empty list is allowed: it is a meme prepared but not yet annotated,
    # which is the normal state of most of the file while annotation is in
    # progress, and what scripts/prepare_memes.py writes. Coverage is reported
    # by scripts/check_annotations.py rather than enforced here, so progress
    # can be checked on a half-finished file.
    if not isinstance(raw_annotations, list):
        raise AnnotationError(f"{where}: annotations must be a list")

    annotations = tuple(
        _parse_annotation(a, f"{where} annotation {i}")
        for i, a in enumerate(raw_annotations)
    )

    seen = Counter(a.annotator for a in annotations)
    repeated = [name for name, count in seen.items() if count > 1]
    if repeated:
        raise AnnotationError(f"{where}: annotator(s) {repeated} appear twice")

    return AnnotatedMeme(
        uid=str(row["id"]),
        image_sha256=digest,
        text=row["text"],
        annotations=annotations,
        label=row.get("label"),
        source=row.get("source", {}),
    )


class PersianMemeSet:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.memes = self._read()

    def _read(self) -> list[AnnotatedMeme]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. See docs/annotation-guideline.md for "
                "the format."
            )

        memes: list[AnnotatedMeme] = []
        with self.path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                where = f"{self.path}:{number}"
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AnnotationError(f"{where}: invalid JSON") from exc
                memes.append(_parse_meme(row, where))

        duplicates = [
            uid for uid, count in Counter(m.uid for m in memes).items() if count > 1
        ]
        if duplicates:
            raise AnnotationError(f"{self.path}: duplicate id(s) {duplicates}")
        return memes

    def duplicate_images(self) -> dict[str, list[str]]:
        """Memes sharing an image hash. Templates recur; the same meme should not."""
        by_hash: dict[str, list[str]] = {}
        for meme in self.memes:
            by_hash.setdefault(meme.image_sha256, []).append(meme.uid)
        return {h: uids for h, uids in by_hash.items() if len(uids) > 1}

    def annotators(self) -> list[str]:
        return sorted({a.annotator for m in self.memes for a in m.annotations})

    def label_matrix(self) -> list[list[int | None]]:
        """One row per meme, one column per annotator, for agreement scoring."""
        names = self.annotators()
        rows = []
        for meme in self.memes:
            by_name = {a.annotator: a.label for a in meme.annotations}
            rows.append([by_name.get(name) for name in names])
        return rows

    def __len__(self) -> int:
        return len(self.memes)

    def __getitem__(self, index: int) -> AnnotatedMeme:
        return self.memes[index]
