"""The interaction statistic has to be built from the raw per-seed differences.

`paper_numbers.py` reports each comparison with its per-seed differences
rounded to two decimals, because that is what the manuscript prints. The
interaction -- the difference of two encoders' fusion benefits, which is the
statistic behind the central equivalence claim -- was being computed from that
rounded list, and positionally rather than by seed. Over fifteen runs it moved
the 95% interval from [-0.31, +0.81] to [-0.32, +0.81], so the paper quoted a
bound that `scripts/equivalence.py` and `scripts/compare_cells.py` both
disagreed with in their last digit.

The drift is small and the cause is not: three scripts computing one number
three ways is how a paper and the statistic behind it come apart. These tests
pin the interaction to the unrounded, seed-matched differences, and pin the
agreement between the two scripts on the committed records.
"""

from __future__ import annotations

import json
import statistics as st
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.equivalence as eq  # noqa: E402

RECORDS = ROOT / "runs" / "paper_numbers.json"


@pytest.fixture(scope="module")
def numbers() -> dict:
    if not RECORDS.exists():
        pytest.skip(f"{RECORDS} not committed")
    return json.loads(RECORDS.read_text(encoding="utf-8"))


def raw_benefit(label: str) -> dict[int, float]:
    """align - concat per seed, unrounded, straight from the run records."""
    return eq.benefit(eq.ENCODERS[label])


def test_interaction_matches_the_unrounded_differences(numbers: dict) -> None:
    for left, right in combinations(eq.ENCODERS, 2):
        key = f"{left} - {right}"
        if key not in numbers["interaction"]:
            continue
        a, b = raw_benefit(left), raw_benefit(right)
        shared = sorted(set(a) & set(b))
        diffs = [a[s] - b[s] for s in shared]
        stored = numbers["interaction"][key]

        assert stored["n"] == len(diffs), key
        assert stored["mean"] == pytest.approx(round(st.mean(diffs), 2)), key
        assert stored["sd"] == pytest.approx(round(st.stdev(diffs), 2)), key
        low, high = eq.interval(diffs)
        assert stored["ci_low"] == pytest.approx(round(low, 2)), key
        assert stored["ci_high"] == pytest.approx(round(high, 2)), key


def test_interaction_interval_agrees_with_the_equivalence_script(
    numbers: dict,
) -> None:
    """The equivalence claim's interval must read the same in both scripts."""
    for left, right in combinations(eq.ENCODERS, 2):
        key = f"{left} - {right}"
        if key not in numbers["interaction"]:
            continue
        a, b = raw_benefit(left), raw_benefit(right)
        shared = sorted(set(a) & set(b))
        if len(shared) < 3:
            continue
        diffs = [a[s] - b[s] for s in shared]
        low, high = eq.interval(diffs)
        stored = numbers["interaction"][key]
        assert (stored["ci_low"], stored["ci_high"]) == (
            round(low, 2),
            round(high, 2),
        ), key


def test_rounded_differences_would_have_moved_an_interval() -> None:
    """Why the raw values are kept: rounding the inputs moves the bound.

    Not a hypothetical. These are the fifteen per-seed differences of the
    central comparison, and rounding them to the two decimals the manuscript
    prints shifts the lower bound by a hundredth -- which is the whole gap
    between the two numbers that were in the paper.
    """
    a, b = raw_benefit("CLIP ViT-L/14"), raw_benefit("CLIP ViT-B/32")
    shared = sorted(set(a) & set(b))
    if len(shared) < 15:
        pytest.skip("needs the fifteen-seed records")
    raw = [a[s] - b[s] for s in shared]
    rounded = [round(a[s], 2) - round(b[s], 2) for s in shared]
    assert round(eq.interval(raw)[0], 2) != round(eq.interval(rounded)[0], 2)


def test_script_runs_and_is_idempotent(tmp_path: Path) -> None:
    """Regenerating must not change the committed file."""
    if not RECORDS.exists():
        pytest.skip(f"{RECORDS} not committed")
    out = tmp_path / "paper_numbers.json"
    result = subprocess.run(
        [sys.executable, "-m", "scripts.paper_numbers", "--out", str(out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(out.read_text(encoding="utf-8")) == json.loads(
        RECORDS.read_text(encoding="utf-8")
    ), "runs/paper_numbers.json is stale; run python -m scripts.paper_numbers"
