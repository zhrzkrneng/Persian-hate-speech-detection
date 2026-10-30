"""The relationship the paper argues, and the ways it could be drawn wrongly.

Each point on the figure is built from two files written by two different
scripts, so the failures worth testing are about pairing and about restraint:
reading one encoder's alignment beside another's AUROC, differencing cells that
did not run the same seeds, and reporting a correlation coefficient from so few
points that it is +/-1 by construction.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.alignment_benefit as ab  # noqa: E402
from scripts.eval_retrieval import PRESETS  # noqa: E402


def write_retrieval(root: Path, preset: str, language: str, t2i: float, i2t: float):
    path = root / f"{preset}_{language}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "encoder": preset,
                "language": language,
                "images": 1000,
                "captions": 2000,
                "results": {
                    "text_to_image": {"R@1": t2i, "R@5": t2i, "R@10": t2i},
                    "image_to_text": {"R@1": i2t, "R@5": i2t, "R@10": i2t},
                },
            }
        ),
        encoding="utf-8",
    )


def write_cell(root: Path, slug: str, fusion: str, aurocs: dict[int, float]):
    directory = root / slug / "hateclipper" / fusion
    directory.mkdir(parents=True, exist_ok=True)
    for seed, auroc in aurocs.items():
        (directory / f"seed{seed}.json").write_text(
            json.dumps({"seed": seed, "test": {"auroc": auroc / 100}}),
            encoding="utf-8",
        )


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ab, "RETRIEVAL", tmp_path / "retrieval")
    monkeypatch.setattr(ab, "HMC", tmp_path / "hmc")
    return tmp_path


def test_a_points_two_halves_come_from_the_same_encoder(world: Path) -> None:
    """The failure: one encoder's alignment plotted against another's AUROC."""
    write_retrieval(world / "retrieval", "clip-l14", "en", 60.0, 72.0)
    write_retrieval(world / "retrieval", "jina-v2", "en", 70.0, 78.0)
    write_cell(world / "hmc", ab.ENCODERS["clip-l14"], "align", {0: 82.0, 1: 82.0})
    write_cell(world / "hmc", ab.ENCODERS["clip-l14"], "concat", {0: 81.0, 1: 81.0})
    write_cell(world / "hmc", ab.ENCODERS["jina-v2"], "align", {0: 79.0, 1: 79.0})
    write_cell(world / "hmc", ab.ENCODERS["jina-v2"], "concat", {0: 79.5, 1: 79.5})

    points, skipped = ab.collect("en", "hateclipper")
    # The other two encoders have no files here, so they are skipped; what
    # matters is that the two complete ones paired with their own halves.
    assert {p.encoder for p in points} == {"clip-l14", "jina-v2"}
    assert all("clip-l14" not in note and "jina-v2" not in note for note in skipped)
    by_encoder = {p.encoder: p for p in points}
    assert by_encoder["clip-l14"].t2i_r1 == 60.0
    assert by_encoder["clip-l14"].benefit == pytest.approx(1.0)
    assert by_encoder["jina-v2"].t2i_r1 == 70.0
    assert by_encoder["jina-v2"].benefit == pytest.approx(-0.5)


def test_an_encoder_missing_either_half_is_skipped_not_guessed(world: Path) -> None:
    """A trained encoder with no retrieval point has no place on the axis."""
    write_cell(world / "hmc", ab.ENCODERS["clip-b32"], "align", {0: 79.0, 1: 79.0})
    write_cell(world / "hmc", ab.ENCODERS["clip-b32"], "concat", {0: 78.0, 1: 78.0})
    points, skipped = ab.collect("en", "hateclipper")
    assert points == []
    assert any("clip-b32" in note for note in skipped)


def test_the_benefit_is_paired_over_shared_seeds_only(world: Path) -> None:
    """Unshared seeds would make the difference partly a difference of seeds."""
    write_retrieval(world / "retrieval", "clip-l14", "en", 60.0, 72.0)
    write_cell(
        world / "hmc", ab.ENCODERS["clip-l14"], "align", {0: 80.0, 1: 90.0, 2: 70.0}
    )
    write_cell(world / "hmc", ab.ENCODERS["clip-l14"], "concat", {0: 79.0, 1: 89.0})

    benefit = ab.read_benefit(ab.ENCODERS["clip-l14"])
    assert benefit["seeds"] == [0, 1]
    assert benefit["benefit"] == pytest.approx(1.0)


def test_one_shared_seed_is_refused(world: Path) -> None:
    write_cell(world / "hmc", ab.ENCODERS["clip-l14"], "align", {0: 80.0})
    write_cell(world / "hmc", ab.ENCODERS["clip-l14"], "concat", {0: 79.0})
    with pytest.raises(ab.Missing, match="fewer than two shared seeds"):
        ab.read_benefit(ab.ENCODERS["clip-l14"])


def test_no_coefficient_is_reported_from_two_points(capsys) -> None:
    """Two points are collinear, so r is +/-1 whatever the numbers are."""
    points = [
        ab.Point("a", "A", "en", 10.0, 10.0, 1000, 1.0, 0.1, 0.01, [0, 1], 81.0, 80.0),
        ab.Point("b", "B", "en", 20.0, 20.0, 1000, 0.0, 0.1, 0.90, [0, 1], 80.0, 80.0),
    ]
    assert ab.print_relationship(points) == {}
    printed = capsys.readouterr().out
    assert "no coefficient is reported" in printed
    assert "Pearson" not in printed


def test_a_coefficient_is_reported_from_three(capsys) -> None:
    points = [
        ab.Point("a", "A", "en", 10.0, 10.0, 1000, 2.0, 0.1, 0.01, [0, 1], 82.0, 80.0),
        ab.Point("b", "B", "en", 20.0, 20.0, 1000, 1.0, 0.1, 0.02, [0, 1], 81.0, 80.0),
        ab.Point("c", "C", "en", 30.0, 30.0, 1000, 0.0, 0.1, 0.90, [0, 1], 80.0, 80.0),
    ]
    out = ab.print_relationship(points)
    assert out["text -> image R@1"]["n"] == 3
    assert out["text -> image R@1"]["spearman_rho"] == pytest.approx(-1.0)
    assert "Read the ordering, not the p-value" in capsys.readouterr().out


def test_chance_is_carried_with_every_point() -> None:
    """A recall without its chance level has already misled once."""
    point = ab.Point("a", "A", "fa", 2.0, 2.0, 50, 0.0, 0.1, 0.5, [0, 1], 80.0, 80.0)
    assert point.chance == pytest.approx(2.0)


def test_every_encoder_on_the_axis_has_a_retrieval_preset() -> None:
    """Otherwise its alignment would be read from a file nothing writes."""
    assert set(ab.ENCODERS) == set(PRESETS)
    assert set(ab.LABELS) == set(ab.ENCODERS)


def test_a_line_held_up_by_one_extreme_point_is_called_out(capsys) -> None:
    """The real shape of these four encoders, in miniature.

    One point far from the rest gives a high Pearson r while the ordering of
    the others does not follow it, so r looks like evidence for a linear trend
    and is not. Quoting it would be the mistake; the script says so instead.
    """
    points = [
        ab.Point("a", "A", "en", 0.2, 0.4, 1000, -1.5, 1.0, 0.03, [0, 1], 61.0, 62.6),
        ab.Point("b", "B", "en", 58.8, 70.0, 1000, 0.8, 0.6, 0.03, [0, 1], 78.8, 78.0),
        ab.Point("c", "C", "en", 60.1, 72.2, 1000, 0.7, 0.6, 0.06, [0, 1], 82.1, 81.4),
        ab.Point("d", "D", "en", 70.3, 78.1, 1000, -0.3, 0.4, 0.20, [0, 1], 79.2, 79.5),
    ]
    out = ab.print_relationship(points)
    linear = out["text -> image R@1"]
    assert linear["pearson_r"] > 0.7
    assert linear["spearman_rho"] < 0.5
    printed = capsys.readouterr().out
    assert "held up by an extreme point" in printed
    assert "the shape is not monotone" in printed


def test_an_honestly_linear_set_is_not_called_out(capsys) -> None:
    """The warning has to stay quiet when the ordering does follow the line."""
    points = [
        ab.Point(c, c.upper(), "en", x, x, 1000, x / 10, 0.1, 0.03, [0, 1], 80.0, 79.0)
        for c, x in (("a", 10.0), ("b", 30.0), ("c", 50.0), ("d", 70.0))
    ]
    ab.print_relationship(points)
    assert "held up by an extreme point" not in capsys.readouterr().out


def test_every_label_in_the_figure_is_tied_to_its_own_point(tmp_path: Path) -> None:
    """A label moved aside once ended up beside its neighbour's point, so the
    figure named the wrong encoder. Each label now carries a leader line."""
    points = [
        ab.Point("a", "A", "en", 58.8, 70.0, 1000, 0.8, 0.6, 0.03, [0, 1], 78.8, 78.0),
        ab.Point("b", "B", "en", 60.1, 72.2, 1000, 0.7, 0.6, 0.06, [0, 1], 82.1, 81.4),
        ab.Point("c", "C", "en", 70.3, 78.1, 1000, -0.3, 0.4, 0.20, [0, 1], 79.2, 79.5),
    ]
    path = tmp_path / "figure.png"
    ab.draw(points, path)
    assert path.exists() and path.stat().st_size > 0

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure = plt.gcf()
    annotations = [
        child
        for axes in figure.axes
        for child in axes.get_children()
        if isinstance(child, matplotlib.text.Annotation)
    ]
    assert {a.get_text() for a in annotations} == {"A", "B", "C"}
    for annotation in annotations:
        assert annotation.arrow_patch is not None, (
            f"{annotation.get_text()} has no leader line to its point"
        )
    plt.close("all")
