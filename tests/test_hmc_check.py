"""The verifier has to catch the ways a mirror actually differs.

Each test builds a copy with one specific defect and asserts it is reported.
A verifier that passes everything is worse than none, because it is trusted.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.hmc_check import (  # noqa: E402
    EXPECTED_SIZES,
    leakage,
    read_split,
    split_ids,
)

# Scaled-down stand-ins, so a fixture is not 8500 rows. read_split compares
# against EXPECTED_SIZES, which the tests patch where the size is the point.
SMALL = {"train": 6, "dev_seen": 3, "dev_unseen": 3, "test_seen": 2, "test_unseen": 2}


def write_split(
    root: Path,
    split: str,
    count: int,
    *,
    labelled: bool,
    start: int = 0,
    make_images: bool = True,
    drop_field: str | None = None,
    bad_label: bool = False,
    duplicate_id: bool = False,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "img").mkdir(exist_ok=True)
    rows = []
    for i in range(count):
        uid = start + (0 if duplicate_id and i == count - 1 else i)
        row = {"id": uid, "img": f"img/{uid:05d}.png", "text": f"caption {uid}"}
        if labelled:
            row["label"] = 2 if (bad_label and i == 0) else i % 2
        if drop_field:
            row.pop(drop_field, None)
        if make_images:
            (root / f"img/{uid:05d}.png").write_bytes(b"x")
        rows.append(row)
    (root / f"{split}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
    )


def build_good(root: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setitem_many = None  # documentation: we patch sizes below
    start = 0
    for split, count in SMALL.items():
        write_split(
            root,
            split,
            count,
            labelled=split in {"train", "dev_seen", "dev_unseen"},
            start=start,
        )
        start += 100
    return root


@pytest.fixture
def small_sizes(monkeypatch: pytest.MonkeyPatch):
    for split, count in SMALL.items():
        monkeypatch.setitem(EXPECTED_SIZES, split, count)


def test_a_good_copy_has_no_problems(tmp_path: Path, small_sizes, monkeypatch) -> None:
    build_good(tmp_path, monkeypatch)
    for split in EXPECTED_SIZES:
        assert read_split(tmp_path, split).problems == [], split
    assert leakage(tmp_path) == []


def test_missing_split_is_reported(tmp_path: Path, small_sizes, monkeypatch) -> None:
    build_good(tmp_path, monkeypatch)
    (tmp_path / "dev_unseen.jsonl").unlink()
    report = read_split(tmp_path, "dev_unseen")
    assert not report.present
    assert "missing" in report.problems[0]


def test_wrong_size_is_reported_with_the_delta(
    tmp_path: Path, small_sizes, monkeypatch
) -> None:
    """The defect that matters most: a subset trains fine and compares to nothing."""
    build_good(tmp_path, monkeypatch)
    write_split(tmp_path, "train", SMALL["train"] - 2, labelled=True)
    report = read_split(tmp_path, "train")
    assert not report.size_ok
    assert "-2" in " ".join(report.problems)


def test_missing_field_is_reported(tmp_path: Path, small_sizes, monkeypatch) -> None:
    build_good(tmp_path, monkeypatch)
    write_split(
        tmp_path, "dev_seen", SMALL["dev_seen"], labelled=True, drop_field="text"
    )
    assert "text" in " ".join(read_split(tmp_path, "dev_seen").problems)


def test_out_of_range_label_is_reported(
    tmp_path: Path, small_sizes, monkeypatch
) -> None:
    build_good(tmp_path, monkeypatch)
    write_split(tmp_path, "dev_seen", SMALL["dev_seen"], labelled=True, bad_label=True)
    assert "outside 0/1" in " ".join(read_split(tmp_path, "dev_seen").problems)


def test_duplicate_id_is_reported(tmp_path: Path, small_sizes, monkeypatch) -> None:
    build_good(tmp_path, monkeypatch)
    write_split(tmp_path, "train", SMALL["train"], labelled=True, duplicate_id=True)
    assert "duplicate" in " ".join(read_split(tmp_path, "train").problems)


def test_missing_image_is_reported(tmp_path: Path, small_sizes, monkeypatch) -> None:
    build_good(tmp_path, monkeypatch)
    # start=900 so these ids have no image from any other split; reusing the
    # default start would silently find train's images and report nothing.
    write_split(
        tmp_path,
        "test_seen",
        SMALL["test_seen"],
        labelled=False,
        start=900,
        make_images=False,
    )
    report = read_split(tmp_path, "test_seen")
    assert report.missing_images == SMALL["test_seen"]
    assert "referenced but absent" in " ".join(report.problems)


def test_skipping_image_checks_hides_only_that(
    tmp_path: Path, small_sizes, monkeypatch
) -> None:
    build_good(tmp_path, monkeypatch)
    write_split(
        tmp_path,
        "test_seen",
        SMALL["test_seen"],
        labelled=False,
        start=900,
        make_images=False,
    )
    assert read_split(tmp_path, "test_seen", check_images=False).problems == []


def test_labelled_test_split_is_reported(
    tmp_path: Path, small_sizes, monkeypatch
) -> None:
    """The original release withholds test labels; a labelled one is not it."""
    build_good(tmp_path, monkeypatch)
    write_split(tmp_path, "test_seen", SMALL["test_seen"], labelled=True)
    assert "not the original data" in " ".join(
        read_split(tmp_path, "test_seen").problems
    )


def test_train_test_leakage_is_reported(
    tmp_path: Path, small_sizes, monkeypatch
) -> None:
    build_good(tmp_path, monkeypatch)
    # Re-write test_seen using train's ids.
    write_split(tmp_path, "test_seen", SMALL["test_seen"], labelled=False, start=0)
    found = leakage(tmp_path)
    assert found and found[0][1] == "test_seen"


def test_dev_overlap_is_not_called_leakage(
    tmp_path: Path, small_sizes, monkeypatch
) -> None:
    """The real release overlaps dev_seen and dev_unseen; that is the data."""
    build_good(tmp_path, monkeypatch)
    write_split(tmp_path, "dev_unseen", SMALL["dev_unseen"], labelled=True, start=100)
    assert split_ids(tmp_path, "dev_seen") & split_ids(tmp_path, "dev_unseen")
    assert leakage(tmp_path) == []


def test_published_sizes_are_the_ones_in_the_paper() -> None:
    """Hate-CLIPper, Section 4.1. Changing these invalidates the comparison."""
    assert EXPECTED_SIZES == {
        "train": 8500,
        "dev_seen": 500,
        "dev_unseen": 540,
        "test_seen": 1000,
        "test_unseen": 2000,
    }
