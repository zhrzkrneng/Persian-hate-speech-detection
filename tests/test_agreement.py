"""Krippendorff's alpha, checked against the published worked example.

A wrong alpha is a plausible number that nothing downstream contradicts, so
the implementation is pinned to a value someone else computed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.agreement import krippendorff_alpha, pairwise_agreement  # noqa: E402

# Krippendorff, "Computing Krippendorff's Alpha-Reliability" (2011), the
# three-observer example. Columns are units, rows observers, "*" is missing.
#
#   A: * * * * * 3 4 1 2 1 1 3 3 * 3
#   B: 1 * 2 1 3 3 4 3 * * * * * * *
#   C: * * 2 1 3 4 4 * 2 1 1 3 3 * 4
#
# Published nominal alpha: 0.691
KRIPPENDORFF_EXAMPLE = [
    [None, 1, None],
    [None, None, None],
    [None, 2, 2],
    [None, 1, 1],
    [None, 3, 3],
    [3, 3, 4],
    [4, 4, 4],
    [1, 3, None],
    [2, None, 2],
    [1, None, 1],
    [1, None, 1],
    [3, None, 3],
    [3, None, 3],
    [None, None, None],
    [3, None, 4],
]


def test_matches_published_example() -> None:
    assert krippendorff_alpha(KRIPPENDORFF_EXAMPLE) == pytest.approx(0.691, abs=0.001)


def test_perfect_agreement_is_one() -> None:
    units = [[0, 0, 0], [1, 1, 1], [0, 0, 0], [1, 1, 1]]
    assert krippendorff_alpha(units) == pytest.approx(1.0)


def test_systematic_disagreement_is_negative() -> None:
    """Two annotators who always disagree do worse than chance."""
    units = [[0, 1], [1, 0], [0, 1], [1, 0]]
    assert krippendorff_alpha(units) < 0


def test_single_label_everywhere_returns_one() -> None:
    """Expected disagreement is zero; alpha carries no information."""
    assert krippendorff_alpha([[1, 1], [1, 1]]) == 1.0


def test_units_with_one_label_are_excluded() -> None:
    """A unit only one annotator saw says nothing about agreement."""
    with_singles = [[0, 0], [1, 1], [0, None], [1, None]]
    without = [[0, 0], [1, 1]]
    assert krippendorff_alpha(with_singles) == pytest.approx(
        krippendorff_alpha(without)
    )


def test_no_usable_unit_raises() -> None:
    with pytest.raises(ValueError, match="undefined, not zero"):
        krippendorff_alpha([[1, None], [None, None]])


def test_label_values_are_opaque() -> None:
    """Nominal alpha must not depend on how labels are named."""
    numeric = [[0, 0, 1], [1, 1, 0], [0, 1, 1]]
    textual = [["no", "no", "yes"], ["yes", "yes", "no"], ["no", "yes", "yes"]]
    assert krippendorff_alpha(numeric) == pytest.approx(krippendorff_alpha(textual))


def test_pairwise_agreement_counts_pairs() -> None:
    # unit 1: 3 pairs, all agree; unit 2: 3 pairs, one agrees (0,0)
    units = [[1, 1, 1], [0, 0, 1]]
    assert pairwise_agreement(units) == pytest.approx(4 / 6)


def test_pairwise_agreement_is_higher_than_alpha() -> None:
    """Raw agreement ignores chance, so it flatters; both get reported."""
    units = [[0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 0]]
    assert pairwise_agreement(units) > krippendorff_alpha(units)
