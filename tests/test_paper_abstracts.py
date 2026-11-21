"""The abstracts must not outlive the seed count they quote.

The Persian abstract was updated to fifteen seeds when the sweep landed. The
English one was not, and said "over five seeds" through every rebuild after
that -- in the single paragraph a foreign reader is most likely to read, and
in the file no Persian-speaking proofreader looks at twice.

The English abstract now interpolates the count from the run records. These
tests pin that it stays interpolated, and that the hand-written Persian word
agrees with the records too.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RECORDS = ROOT / "runs" / "paper_numbers.json"

# The main-table cells, not Multi3Hate's: section 6 runs five seeds on purpose
# and says so, so a blanket search for a seed word would flag correct prose.
PERSIAN_NUMERALS = {
    "یک": 1,
    "دو": 2,
    "سه": 3,
    "چهار": 4,
    "پنج": 5,
    "شش": 6,
    "هفت": 7,
    "هشت": 8,
    "نه": 9,
    "ده": 10,
    "پانزده": 15,
    "بیست": 20,
    "بیست‌وپنج": 25,
}


@pytest.fixture(scope="module")
def seeds() -> int:
    if not RECORDS.exists():
        pytest.skip("runs/paper_numbers.json not committed")
    numbers = json.loads(RECORDS.read_text(encoding="utf-8"))
    return numbers["fusion_benefit"]["CLIP ViT-L/14"]["n"]


def test_english_abstract_interpolates_the_seed_count() -> None:
    source = (PAPER / "content_d.js").read_text(encoding="utf-8")
    assert "${SEED_WORD} seeds" in source, (
        "the English abstract must take its seed count from the records"
    )
    # Comments are stripped first: the comment above SEED_WORDS quotes the old
    # "five seeds" to say what went wrong, and a guard that flags its own
    # explanation teaches people to delete the explanation.
    code = "\n".join(re.sub(r"//.*$", "", line) for line in source.split("\n"))
    hardcoded = re.findall(r'"[^"]*\b(five|ten|fifteen|twenty) seeds\b', code)
    assert not hardcoded, f"hard-coded seed count in the English abstract: {hardcoded}"


def test_persian_abstract_agrees_with_the_records(seeds: int) -> None:
    """The Persian abstract's seed word is written by hand; check it."""
    source = (PAPER / "content_a.js").read_text(encoding="utf-8")
    words = re.findall(r"([؀-ۿ‌]+)\s+بذر", source)
    assert words, "no seed count found in the Persian abstract"
    for word in words:
        if word in PERSIAN_NUMERALS:
            assert PERSIAN_NUMERALS[word] == seeds, (
                f"the Persian abstract says {word} ({PERSIAN_NUMERALS[word]}) "
                f"where the records hold {seeds}"
            )


def test_the_two_abstracts_cannot_disagree(seeds: int) -> None:
    """Whatever the count becomes, both abstracts have to follow it."""
    english = (PAPER / "content_d.js").read_text(encoding="utf-8")
    match = re.search(r"SEED_WORDS\s*=\s*\{([^}]*)\}", english)
    assert match, "the English abstract's numeral map is gone"
    mapped = {int(k) for k in re.findall(r"(\d+)\s*:", match.group(1))}
    assert seeds in mapped, (
        f"{seeds} seeds has no English word in SEED_WORDS; the abstract would "
        f"print the digits"
    )
