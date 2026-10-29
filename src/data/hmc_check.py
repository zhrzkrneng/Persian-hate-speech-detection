"""Checks on a local copy of the Hateful Memes Challenge data.

The official distribution has become hard to reach, so copies now circulate
through third-party mirrors. Mirrors differ: some are subsets, some re-split
the data, some drop a field. A copy whose splits do not match the published
ones produces numbers that cannot be compared with any paper, and nothing
downstream would reveal it — training would simply succeed on the wrong data.

So the split sizes are checked against the published ones before anything
reads them.

Counts as reported in the Hate-CLIPper paper (Kumar & Nandakumar, WOAH 2022),
Section 4.1.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

EXPECTED_SIZES = {
    "train": 8500,
    "dev_seen": 500,
    "dev_unseen": 540,
    "test_seen": 1000,
    "test_unseen": 2000,
}

# Every split should carry labels.
#
# The original competition withheld the test labels and scored submissions on
# a server, so an earlier version of this file treated a labelled test split
# as evidence of a tampered copy. That was wrong twice over: the labels were
# released after the competition closed, which is why Hate-CLIPper can report
# a test-seen AUROC at all, and src/train.py needs them to evaluate anything.
# A copy whose test splits are unlabelled is the one that cannot be used.
LABELLED = {"train", "dev_seen", "dev_unseen", "test_seen", "test_unseen"}

REQUIRED_FIELDS = {"id", "img", "text"}


@dataclass
class SplitReport:
    split: str
    present: bool
    count: int = 0
    expected: int = 0
    labelled: int = 0
    label_counts: Counter | None = None
    missing_fields: list[str] | None = None
    bad_labels: list[object] | None = None
    duplicate_ids: int = 0
    missing_images: int = 0
    first_missing_image: str | None = None

    @property
    def size_ok(self) -> bool:
        return self.present and self.count == self.expected

    @property
    def problems(self) -> list[str]:
        issues: list[str] = []
        if not self.present:
            return [f"{self.split}.jsonl is missing"]
        if not self.size_ok:
            issues.append(
                f"{self.count} rows, expected {self.expected} "
                f"({self.count - self.expected:+d})"
            )
        if self.missing_fields:
            issues.append(f"rows missing fields: {sorted(set(self.missing_fields))}")
        if self.bad_labels:
            issues.append(
                f"labels outside 0/1: {sorted(set(map(str, self.bad_labels)))}"
            )
        if self.duplicate_ids:
            issues.append(f"{self.duplicate_ids} duplicate id(s)")
        if self.labelled != self.count:
            issues.append(
                f"{self.count - self.labelled} row(s) unlabelled; the test "
                "labels were released after the competition closed and "
                "src/train.py needs them to evaluate"
            )
        if self.missing_images:
            issues.append(
                f"{self.missing_images} image(s) referenced but absent, "
                f"first: {self.first_missing_image}"
            )
        return issues


def read_split(root: Path, split: str, check_images: bool = True) -> SplitReport:
    path = root / f"{split}.jsonl"
    expected = EXPECTED_SIZES[split]
    if not path.exists():
        return SplitReport(split=split, present=False, expected=expected)

    report = SplitReport(
        split=split,
        present=True,
        expected=expected,
        label_counts=Counter(),
        missing_fields=[],
        bad_labels=[],
    )
    seen_ids: Counter = Counter()

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            report.count += 1

            for field in REQUIRED_FIELDS - set(row):
                report.missing_fields.append(field)

            uid = row.get("id")
            if uid is not None:
                seen_ids[str(uid)] += 1

            if "label" in row and row["label"] is not None:
                report.labelled += 1
                if row["label"] in (0, 1):
                    report.label_counts[row["label"]] += 1
                else:
                    report.bad_labels.append(row["label"])

            if check_images and "img" in row:
                if not (root / row["img"]).exists():
                    report.missing_images += 1
                    if report.first_missing_image is None:
                        report.first_missing_image = row["img"]

    report.duplicate_ids = sum(1 for n in seen_ids.values() if n > 1)
    return report


def split_ids(root: Path, split: str) -> set[str]:
    path = root / f"{split}.jsonl"
    if not path.exists():
        return set()
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.add(str(json.loads(line)["id"]))
    return ids


def leakage(root: Path) -> list[tuple[str, str, int]]:
    """Overlaps between splits that should not share examples.

    dev_seen and dev_unseen are deliberately excluded: the original release
    overlaps them, and Hate-CLIPper notes that 400 of the 540 dev_unseen memes
    also appear in dev_seen. That overlap is a property of the data, not a
    defect in a copy of it. Train sharing examples with any evaluation split
    is a defect, and one that silently inflates every result.
    """
    train = split_ids(root, "train")
    found = []
    for split in ("dev_seen", "dev_unseen", "test_seen", "test_unseen"):
        shared = train & split_ids(root, split)
        if shared:
            found.append(("train", split, len(shared)))
    return found
