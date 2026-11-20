"""The analysis of a prediction that failed, and the ways it could be rescued.

A null result is easy to turn into a positive one by accident: drop the
inconvenient cells, average over the wrong thing, or let an incomplete cell
through. These tests pin the arithmetic and the two guards that matter -- that
the floor-effect subsets are computed rather than asserted, and that a cell
with unshared seeds is not differenced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.multilingual_benefit as mb  # noqa: E402

ENC = "openai/clip-vit-base-patch32"
LABEL = "CLIP ViT-B/32"


def write_eval(root: Path, fusion: str, seed: int, scores: dict[str, float]) -> None:
    path = root / "openai__clip-vit-base-patch32" / "hateclipper" / fusion
    path.mkdir(parents=True, exist_ok=True)
    (path / f"seed{seed}.json").write_text(
        json.dumps(
            {
                "encoder": ENC,
                "architecture": "hateclipper",
                "fusion": fusion,
                "seed": seed,
                "multi3hate": {
                    lang: {"auroc": v / 100, "memes": 300} for lang, v in scores.items()
                },
            }
        ),
        encoding="utf-8",
    )


def write_retrieval(root: Path, language: str, r1: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"clip-b32_{language}.json").write_text(
        json.dumps(
            {
                "encoder": "clip-b32",
                "language": language,
                "images": 1000,
                "results": {
                    "text_to_image": {"R@1": r1},
                    "image_to_text": {"R@1": r1},
                },
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(mb, "M3H", tmp_path / "m3h")
    monkeypatch.setattr(mb, "RETRIEVAL", tmp_path / "retrieval")
    return tmp_path


def test_the_benefit_is_paired_per_seed(world: Path) -> None:
    for seed, (a, c) in enumerate([(70.0, 68.0), (60.0, 59.0)]):
        write_eval(world / "m3h", "align", seed, {"en": a})
        write_eval(world / "m3h", "concat", seed, {"en": c})
    write_retrieval(world / "retrieval", "en", 58.75)

    rows = mb.collect()
    assert len(rows) == 1
    assert rows[0]["benefit"] == pytest.approx(1.5)
    assert rows[0]["encoder"] == LABEL and rows[0]["alignment"] == 58.75


def test_a_cell_with_one_shared_seed_is_dropped(world: Path) -> None:
    """Differencing one pair is a single-seed number, which this project bans."""
    write_eval(world / "m3h", "align", 0, {"en": 70.0})
    write_eval(world / "m3h", "concat", 0, {"en": 68.0})
    write_retrieval(world / "retrieval", "en", 58.75)
    assert mb.collect() == []


def test_a_cell_with_no_alignment_is_dropped(world: Path) -> None:
    """Without a retrieval score there is no point to place on the axis."""
    for seed in (0, 1):
        write_eval(world / "m3h", "align", seed, {"en": 70.0})
        write_eval(world / "m3h", "concat", seed, {"en": 68.0})
    assert mb.collect() == []


def test_the_floor_subsets_are_computed_not_asserted(world: Path, capsys) -> None:
    """The obvious objection to a null result is a floor effect, so the
    subsets above chance have to be reported, not waved at."""
    for seed in (0, 1):
        write_eval(
            world / "m3h",
            "align",
            seed,
            {"en": 70.0, "de": 51.0, "es": 66.0, "hi": 64.0},
        )
        write_eval(
            world / "m3h",
            "concat",
            seed,
            {"en": 68.0, "de": 52.0, "es": 61.0, "hi": 62.0},
        )
    for lang, r1 in (("en", 58.75), ("de", 19.25), ("es", 26.87), ("hi", 0.16)):
        write_retrieval(world / "retrieval", lang, r1)

    rows = mb.collect()
    summary = mb.report(rows)
    assert summary["overall"]["n"] == 4
    # de sits below both floors and drops out of each subset. Below three
    # points no coefficient is reported at all, for the same reason it is not
    # reported for two encoders: any two points are collinear.
    assert summary["subsets"][55.0]["n"] == 3
    printed = capsys.readouterr().out
    assert "concat AUROC >=" in printed
    assert "floor effect" in printed


def test_the_per_encoder_summary_counts_the_negative_languages(world: Path) -> None:
    """One encoder losing in every language is the finding that survived."""
    for seed in (0, 1):
        write_eval(world / "m3h", "align", seed, {"en": 60.0, "de": 60.0, "es": 60.0})
        write_eval(world / "m3h", "concat", seed, {"en": 63.0, "de": 62.0, "es": 64.0})
    for lang, r1 in (("en", 70.0), ("de", 84.0), ("es", 77.0)):
        write_retrieval(world / "retrieval", lang, r1)

    summary = mb.per_encoder(mb.collect())
    assert summary[LABEL]["negative_in"] == 3
    assert summary[LABEL]["n"] == 3
    assert summary[LABEL]["mean"] < 0


def test_reliability_is_high_when_cells_are_measured_precisely(world: Path) -> None:
    """Four cells far apart, each estimated tightly: little attenuation."""
    rows = [
        {"benefit": b, "benefit_sd": 0.1, "seeds": [0, 1, 2, 3, 4]}
        for b in (-3.0, -1.0, 1.0, 3.0)
    ]
    assert mb.reliability(rows)["reliability"] > 0.95


def test_reliability_collapses_when_the_noise_swamps_the_spread(world: Path) -> None:
    """The case that would make a null meaningless, and must be visible.

    If each cell's standard error is as large as the variation between cells,
    nothing can be concluded from a correlation, and the reliability has to say
    so rather than quietly reporting a coefficient.
    """
    rows = [
        {"benefit": b, "benefit_sd": 5.0, "seeds": [0, 1, 2, 3, 4]}
        for b in (-0.3, -0.1, 0.1, 0.3)
    ]
    assert mb.reliability(rows)["reliability"] < 0.2


def test_within_language_fits_use_only_that_language(world: Path) -> None:
    """Inside one language the four encoders share captions and memes, so the
    annotation-style confound of the pooled axis is gone."""
    rows = []
    for language, offset in (("en", 0.0), ("de", 10.0)):
        for i, encoder in enumerate(("A", "B", "C", "D")):
            rows.append(
                {
                    "encoder": encoder,
                    "language": language,
                    "alignment": 10.0 * i + offset,
                    "benefit": -1.0 * i,  # perfectly decreasing inside each language
                    "benefit_sd": 0.1,
                    "seeds": [0, 1],
                }
            )
    out = mb.within_language(rows)
    assert set(out) == {"en", "de"}
    for value in out.values():
        assert value["n"] == 4
        assert value["pearson_r"] == pytest.approx(-1.0, abs=1e-6)


def test_a_language_with_too_few_cells_is_skipped(world: Path) -> None:
    """Two points are collinear; no coefficient is reported from them."""
    rows = [
        {
            "encoder": e,
            "language": "en",
            "alignment": a,
            "benefit": b,
            "benefit_sd": 0.1,
            "seeds": [0, 1],
        }
        for e, a, b in (("A", 10.0, 1.0), ("B", 20.0, 2.0))
    ]
    assert mb.within_language(rows) == {}
