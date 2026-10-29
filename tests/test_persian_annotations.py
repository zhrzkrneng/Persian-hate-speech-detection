"""Schema validation for the Persian annotation file.

A schema error found at training time is a week of annotation already spent,
so every rule the guideline states is enforced at read time and tested here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.persian import AnnotationError, PersianMemeSet  # noqa: E402

DIGEST = "a" * 64


def meme(**overrides) -> dict:
    row = {
        "id": "fa_0001",
        "image_sha256": DIGEST,
        "text": "متن روی تصویر",
        "annotations": [
            {"annotator": "A1", "label": 1, "target": "gender", "explicit": False},
            {"annotator": "A2", "label": 1, "target": "gender", "explicit": True},
            {"annotator": "A3", "label": 0},
        ],
    }
    row.update(overrides)
    return row


def write(tmp_path: Path, *rows: dict) -> Path:
    path = tmp_path / "annotations.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    return path


def test_reads_a_valid_file(tmp_path: Path) -> None:
    dataset = PersianMemeSet(write(tmp_path, meme()))
    assert len(dataset) == 1
    assert dataset.annotators() == ["A1", "A2", "A3"]
    assert dataset[0].majority() == 1


def test_persian_text_survives_the_round_trip(tmp_path: Path) -> None:
    dataset = PersianMemeSet(write(tmp_path, meme(text="خروس و مرغ در باغ")))
    assert dataset[0].text == "خروس و مرغ در باغ"


def test_tie_has_no_majority(tmp_path: Path) -> None:
    """The guideline resolves ties with more annotations, so a tie is unfinished."""
    row = meme(
        annotations=[
            {"annotator": "A1", "label": 1, "target": "gender", "explicit": True},
            {"annotator": "A2", "label": 0},
        ]
    )
    assert PersianMemeSet(write(tmp_path, row))[0].majority() is None


def test_abstentions_do_not_count_toward_the_majority(tmp_path: Path) -> None:
    row = meme(
        annotations=[
            {"annotator": "A1", "label": 1, "target": "other", "explicit": True},
            {"annotator": "A2", "label": None},
            {"annotator": "A3", "label": None},
        ]
    )
    assert PersianMemeSet(write(tmp_path, row))[0].majority() == 1


def test_hate_label_requires_a_target(tmp_path: Path) -> None:
    row = meme(annotations=[{"annotator": "A1", "label": 1, "explicit": True}])
    with pytest.raises(AnnotationError, match="needs a target"):
        PersianMemeSet(write(tmp_path, row))


def test_hate_label_requires_explicitness(tmp_path: Path) -> None:
    row = meme(annotations=[{"annotator": "A1", "label": 1, "target": "gender"}])
    with pytest.raises(AnnotationError, match="needs explicit"):
        PersianMemeSet(write(tmp_path, row))


def test_unknown_target_is_rejected(tmp_path: Path) -> None:
    row = meme(
        annotations=[
            {"annotator": "A1", "label": 1, "target": "football", "explicit": True}
        ]
    )
    with pytest.raises(AnnotationError, match="needs a target from"):
        PersianMemeSet(write(tmp_path, row))


def test_target_on_a_non_hate_label_is_rejected(tmp_path: Path) -> None:
    row = meme(annotations=[{"annotator": "A1", "label": 0, "target": "gender"}])
    with pytest.raises(AnnotationError, match="on a non-hate label"):
        PersianMemeSet(write(tmp_path, row))


def test_out_of_range_label_is_rejected(tmp_path: Path) -> None:
    row = meme(annotations=[{"annotator": "A1", "label": 2}])
    with pytest.raises(AnnotationError, match="must be 0, 1 or null"):
        PersianMemeSet(write(tmp_path, row))


def test_same_annotator_twice_is_rejected(tmp_path: Path) -> None:
    row = meme(
        annotations=[
            {"annotator": "A1", "label": 0},
            {"annotator": "A1", "label": 1, "target": "other", "explicit": True},
        ]
    )
    with pytest.raises(AnnotationError, match="appear twice"):
        PersianMemeSet(write(tmp_path, row))


def test_duplicate_id_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(AnnotationError, match="duplicate id"):
        PersianMemeSet(write(tmp_path, meme(), meme()))


def test_short_hash_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(AnnotationError, match="64 hex characters"):
        PersianMemeSet(write(tmp_path, meme(image_sha256="abc")))


def test_duplicate_images_are_reported_not_rejected(tmp_path: Path) -> None:
    """Two ids, one image: worth seeing, but not a schema error."""
    dataset = PersianMemeSet(write(tmp_path, meme(), meme(id="fa_0002")))
    assert dataset.duplicate_images() == {DIGEST: ["fa_0001", "fa_0002"]}


def test_label_matrix_aligns_annotators_across_memes(tmp_path: Path) -> None:
    """Each column must be one annotator, or the agreement score is nonsense."""
    first = meme(
        id="fa_0001",
        annotations=[
            {"annotator": "A1", "label": 0},
            {"annotator": "A2", "label": 1, "target": "other", "explicit": True},
        ],
    )
    second = meme(
        id="fa_0002",
        annotations=[
            {"annotator": "A2", "label": 0},
            {"annotator": "A1", "label": 1, "target": "other", "explicit": True},
        ],
    )
    dataset = PersianMemeSet(write(tmp_path, first, second))
    assert dataset.annotators() == ["A1", "A2"]
    assert dataset.label_matrix() == [[0, 1], [1, 0]]


def test_missing_file_names_the_guideline(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="annotation-guideline"):
        PersianMemeSet(tmp_path / "absent.jsonl")
