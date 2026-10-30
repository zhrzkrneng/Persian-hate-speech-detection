"""Aggregation must refuse to average runs that are not the same experiment.

The result path names the dataset, the encoder, the architecture and the
fusion, so a directory holding two different settings should be impossible.
"Should be impossible" is what the result path already said when it was keyed
by fusion alone, and memeclip overwrote hateclipper anyway. A mean over two
encoders is not a result about either, and it does not look wrong: it looks
like one number with a wide standard deviation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.summarise import collect  # noqa: E402

METRICS = {"auroc": 0.8, "accuracy": 0.7, "macro_f1": 0.69}


def write(
    directory: Path,
    seed: int,
    encoder: str = "jinaai/jina-clip-v2",
    architecture: str = "hateclipper",
    fusion: str = "align",
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"seed{seed}.json").write_text(
        json.dumps(
            {
                "encoder": encoder,
                "architecture": architecture,
                "fusion": fusion,
                "seed": seed,
                "val": METRICS,
                "test": METRICS,
            }
        ),
        encoding="utf-8",
    )


def test_five_seeds_of_one_setting_collect(tmp_path: Path) -> None:
    for seed in range(5):
        write(tmp_path, seed)
    val, test, seeds = collect(tmp_path)
    assert seeds == [0, 1, 2, 3, 4]
    assert len(val) == len(test) == 5


def test_two_encoders_in_one_directory_are_refused(tmp_path: Path) -> None:
    write(tmp_path, 0, encoder="jinaai/jina-clip-v2")
    write(tmp_path, 1, encoder="openai/clip-vit-large-patch14")
    with pytest.raises(SystemExit, match="more than one setting"):
        collect(tmp_path)


def test_two_architectures_in_one_directory_are_refused(tmp_path: Path) -> None:
    write(tmp_path, 0, architecture="hateclipper")
    write(tmp_path, 1, architecture="memeclip")
    with pytest.raises(SystemExit, match="more than one setting"):
        collect(tmp_path)


def test_the_error_names_both_settings(tmp_path: Path) -> None:
    """Naming them is the difference between a fix and a hunt."""
    write(tmp_path, 0, encoder="jinaai/jina-clip-v2")
    write(tmp_path, 1, encoder="openai/clip-vit-large-patch14")
    with pytest.raises(SystemExit) as caught:
        collect(tmp_path)
    assert "jinaai/jina-clip-v2" in str(caught.value)
    assert "openai/clip-vit-large-patch14" in str(caught.value)


def test_an_empty_directory_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no seed"):
        collect(tmp_path)
