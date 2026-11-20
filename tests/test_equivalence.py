"""An equivalence test, and the mistake it exists to prevent.

The mistake is reading a non-significant paired test as evidence that two
things are the same. It is not: it can equally mean the sample was too small
to resolve whatever is there. These tests pin the three verdicts apart, and
pin the one that matters most -- a difference that is small but measured on so
few runs that nothing is excluded comes back "undecided", not "equivalent".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.equivalence as eq  # noqa: E402


def write(root: Path, slug: str, fusion: str, by_seed: dict[int, float]) -> None:
    directory = root / slug / "hateclipper" / fusion
    directory.mkdir(parents=True, exist_ok=True)
    for seed, auroc in by_seed.items():
        (directory / f"seed{seed}.json").write_text(
            json.dumps({"seed": seed, "test": {"auroc": auroc / 100}}),
            encoding="utf-8",
        )


@pytest.fixture
def hmc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(eq, "HMC", tmp_path)
    return tmp_path


def test_a_tight_small_difference_is_equivalent() -> None:
    """Many observations, all near zero: the margin excludes a real effect."""
    result = eq.tost([0.05, -0.05, 0.04, -0.04, 0.06, -0.06, 0.03, -0.03], margin=1.0)
    assert result["equivalent"]
    assert -0.5 < result["ci90_low"] and result["ci90_high"] < 0.5


def test_a_small_difference_on_few_noisy_runs_is_not_equivalent() -> None:
    """The real case this script was written for.

    A mean near zero with a spread this wide over five runs leaves effects as
    large as the ones the study reports entirely plausible. Calling that
    "equivalent" would be the error; TOST must refuse.
    """
    result = eq.tost([-0.27, 1.2, -1.4, 0.6, -1.5], margin=1.0)
    assert not result["equivalent"]
    assert result["ci90_low"] < -0.8


def test_a_real_difference_is_not_equivalent() -> None:
    result = eq.tost([2.5, 2.7, 2.4, 2.6, 2.8], margin=1.0)
    assert not result["equivalent"]
    assert result["ci90_low"] > 1.0


def test_widening_the_margin_can_only_help_equivalence() -> None:
    """A margin is a claim about what size matters; it must be stated, not fitted."""
    values = [0.3, -0.2, 0.4, -0.1, 0.2]
    assert (
        eq.tost(values, margin=0.5)["p_tost"] >= eq.tost(values, margin=2.0)["p_tost"]
    )


def test_the_benefit_is_paired_per_seed(hmc: Path) -> None:
    write(hmc, "enc", "align", {0: 80.0, 1: 81.0})
    write(hmc, "enc", "concat", {0: 79.0, 1: 79.5})
    assert eq.benefit("enc") == {0: pytest.approx(1.0), 1: pytest.approx(1.5)}


def test_a_seed_present_in_only_one_cell_is_dropped(hmc: Path) -> None:
    """Differencing unpaired seeds would mix a seed effect into the result."""
    write(hmc, "enc", "align", {0: 80.0, 1: 81.0, 2: 90.0})
    write(hmc, "enc", "concat", {0: 79.0, 1: 79.5})
    assert sorted(eq.benefit("enc")) == [0, 1]


def test_the_interval_narrows_as_runs_are_added() -> None:
    """Why fifteen seeds were run: the verdict turns on the interval's width."""
    few = eq.interval([0.5, -0.5, 0.4, -0.4, 0.3])
    many = eq.interval([0.5, -0.5, 0.4, -0.4, 0.3] * 3)
    assert (many[1] - many[0]) < (few[1] - few[0])
