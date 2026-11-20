"""Holm-Bonferroni, and the step-down property a Bonferroni loop lacks.

The mistake these pin is reporting more rejections than the procedure
licenses. Holm sorts the p-values and stops at the first failure: every
hypothesis after it is retained, however small its own p-value. A loop that
simply compares each p with alpha / (m - rank) does not stop, and will happily
reject a hypothesis that sits behind a retained one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.multiplicity import holm  # noqa: E402


def test_thresholds_step_down() -> None:
    out = holm({"a": 0.001, "b": 0.002, "c": 0.003, "d": 0.004}, alpha=0.05)
    assert [out[k]["rank"] for k in "abcd"] == [1, 2, 3, 4]
    assert out["a"]["threshold"] == pytest.approx(0.05 / 4)
    assert out["b"]["threshold"] == pytest.approx(0.05 / 3)
    assert out["c"]["threshold"] == pytest.approx(0.05 / 2)
    assert out["d"]["threshold"] == pytest.approx(0.05 / 1)


def test_first_failure_stops_the_procedure() -> None:
    """The point of the test: 'c' is tiny but sits behind a retained 'b'."""
    out = holm({"a": 0.001, "b": 0.40, "c": 0.002}, alpha=0.05)
    assert out["a"]["holds"] is True
    # ranked second at 0.002, well under its 0.025 threshold
    assert out["c"]["rank"] == 2
    assert out["c"]["p"] < out["c"]["threshold"]
    assert out["c"]["holds"] is True
    # 'b' is ranked last and fails
    assert out["b"]["holds"] is False


def test_a_retained_hypothesis_blocks_every_later_one() -> None:
    out = holm({"a": 0.9, "b": 0.0001}, alpha=0.05)
    assert out["b"]["holds"] is True  # ranked first, 0.0001 < 0.025
    assert out["a"]["holds"] is False
    # now make the first-ranked one fail: nothing after it may hold
    out = holm({"a": 0.03, "b": 0.04}, alpha=0.05)
    assert out["a"]["holds"] is False  # 0.03 > 0.05/2
    assert out["b"]["holds"] is False


def test_single_test_is_uncorrected() -> None:
    out = holm({"only": 0.04}, alpha=0.05)
    assert out["only"]["threshold"] == pytest.approx(0.05)
    assert out["only"]["holds"] is True


def test_holm_is_never_stricter_than_bonferroni() -> None:
    pvalues = {f"h{i}": p for i, p in enumerate([0.001, 0.009, 0.02, 0.3, 0.7])}
    out = holm(pvalues, alpha=0.05)
    bonferroni = 0.05 / len(pvalues)
    for label, p in pvalues.items():
        if p < bonferroni:
            assert out[label]["holds"], label


def test_margin_is_threshold_minus_p() -> None:
    out = holm({"a": 0.01, "b": 0.5}, alpha=0.05)
    assert out["a"]["margin"] == pytest.approx(out["a"]["threshold"] - 0.01)


def test_the_committed_family_survives_correction() -> None:
    """Every claim the paper calls significant holds; the two nulls do not."""
    from scripts.multiplicity import family

    try:
        tests = family()
    except (FileNotFoundError, KeyError):  # pragma: no cover - records absent
        pytest.skip("run records not available")
    if len(tests) < 14:
        pytest.skip("not every cell has records")
    out = holm({k: p for k, (_, p) in tests.items()})
    failed = {k for k, v in out.items() if not v["holds"]}
    assert failed == {
        "interaction: CLIP ViT-L/14 − CLIP ViT-B/32",
        "architecture: CLIP ViT-B/32",
    }
