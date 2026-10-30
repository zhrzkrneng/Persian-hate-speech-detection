"""The paired comparison, and that it refuses what it cannot compare.

This script decides what the paper is allowed to claim, so its arithmetic is
pinned here. It exists because a rule of thumb was used as a verdict once:
`mean - sd > 0` called a difference of +0.72 +/- 0.62 over five seeds settled,
where a paired t-test puts it at p = 0.061.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.compare_cells import paired, read_cell  # noqa: E402


def cell(directory: Path, values: dict[int, float], split: str = "test") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for seed, auroc in values.items():
        other = "val" if split == "test" else "test"
        (directory / f"seed{seed}.json").write_text(
            json.dumps(
                {
                    "encoder": "e",
                    "architecture": "a",
                    "fusion": "f",
                    "seed": seed,
                    split: {"auroc": auroc, "accuracy": 0.7, "macro_f1": 0.6},
                    other: {"auroc": 0.5, "accuracy": 0.5, "macro_f1": 0.5},
                }
            ),
            encoding="utf-8",
        )
    return directory


def test_read_cell_returns_percentages_by_seed(tmp_path: Path) -> None:
    cell(tmp_path, {0: 0.8, 1: 0.82})
    assert read_cell(tmp_path, "test") == {0: 80.0, 1: 82.0}


def test_differences_are_taken_per_seed(tmp_path: Path) -> None:
    """Pairing is the point: the seed noise the variants share cancels."""
    a = cell(tmp_path / "a", {0: 0.80, 1: 0.90, 2: 0.70})
    b = cell(tmp_path / "b", {0: 0.79, 1: 0.89, 2: 0.69})

    seeds, diffs = paired(read_cell(a, "test"), read_cell(b, "test"))
    assert seeds == [0, 1, 2]
    assert diffs == pytest.approx([1.0, 1.0, 1.0])


def test_only_shared_seeds_are_compared(tmp_path: Path) -> None:
    a = cell(tmp_path / "a", {0: 0.80, 1: 0.90, 7: 0.99})
    b = cell(tmp_path / "b", {0: 0.70, 1: 0.80})

    seeds, diffs = paired(read_cell(a, "test"), read_cell(b, "test"))
    assert seeds == [0, 1]
    assert diffs == pytest.approx([10.0, 10.0])


def test_one_shared_seed_is_refused(tmp_path: Path) -> None:
    """A single pair has no spread, so there is nothing to test against zero."""
    a = cell(tmp_path / "a", {0: 0.80})
    b = cell(tmp_path / "b", {0: 0.70})
    with pytest.raises(SystemExit, match="at least two shared seeds"):
        paired(read_cell(a, "test"), read_cell(b, "test"))


def test_an_empty_cell_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no seed"):
        read_cell(tmp_path, "test")


def test_the_val_split_can_be_read_separately(tmp_path: Path) -> None:
    """Selection numbers and comparison numbers must stay distinguishable."""
    directory = cell(tmp_path, {0: 0.77}, split="val")
    assert read_cell(directory, "val") == {0: 77.0}
    assert read_cell(directory, "test") == {0: 50.0}


def test_a_known_difference_reproduces_its_t_statistic() -> None:
    """Pins the test itself, not just the differencing.

    The five CLIP align-minus-concat differences, which the paper reports as
    +0.72 +/- 0.62, t(4) = 2.58, p = 0.061 -- not significant, which is the
    whole reason this script replaced a rule of thumb.
    """
    import statistics as st

    from scipy import stats

    diffs = [0.65, 0.90, 0.19, 1.69, 0.17]
    result = stats.ttest_1samp(diffs, 0.0)

    assert st.mean(diffs) == pytest.approx(0.72, abs=0.005)
    assert st.stdev(diffs) == pytest.approx(0.62, abs=0.005)
    assert float(result.statistic) == pytest.approx(2.58, abs=0.01)
    assert float(result.pvalue) == pytest.approx(0.061, abs=0.001)
    assert float(result.pvalue) > 0.05
