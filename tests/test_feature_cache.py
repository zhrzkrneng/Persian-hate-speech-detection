"""A cache file must say what produced it, and be refused when it disagrees.

The feature directory is keyed by encoder name, which is protection only for
as long as nobody moves, copies or re-downloads a file. In this project the
caches are built in one place -- a Kaggle GPU for jina-clip-v2 -- and trained
on in another, so "this directory holds that encoder's features" has been an
assumption about a path rather than a fact about a file. Written inside the
file it is checkable, and the check is what these tests are for.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.features import (  # noqa: E402
    CacheMismatch,
    FeatureCache,
    FeatureDataset,
)

ENCODER = "jinaai/jina-clip-v2"


def write(directory: Path, split: str = "train", encoder: str = ENCODER, n: int = 4):
    cache = FeatureCache(directory, split)
    cache.save(
        image_features=torch.randn(n, 8),
        text_features=torch.randn(n, 8),
        labels=torch.zeros(n, dtype=torch.long),
        uids=[f"id{i}" for i in range(n)],
        encoder=encoder,
    )
    return cache


def test_save_records_what_produced_it(tmp_path: Path) -> None:
    cache = write(tmp_path)
    meta = cache.load()["meta"]

    assert meta["encoder"] == ENCODER
    assert meta["split"] == "train"
    assert meta["count"] == 4
    assert meta["feature_dim"] == 8
    assert meta["written"].startswith("20")
    # torch is always installed where this runs; transformers is not, and its
    # absence is recorded as absence rather than as an error.
    assert "torch" in meta["versions"]


def test_the_expected_encoder_is_accepted(tmp_path: Path) -> None:
    write(tmp_path)
    data = FeatureDataset(tmp_path, "train", ENCODER)
    assert len(data) == 4
    assert data.meta["encoder"] == ENCODER


def test_a_different_encoder_is_refused(tmp_path: Path) -> None:
    """The failure this exists for: training on the wrong features silently."""
    write(tmp_path, encoder="openai/clip-vit-large-patch14")
    with pytest.raises(CacheMismatch, match="openai/clip-vit-large-patch14"):
        FeatureDataset(tmp_path, "train", ENCODER)


def test_the_refusal_names_both_encoders(tmp_path: Path) -> None:
    write(tmp_path, encoder="openai/clip-vit-large-patch14")
    with pytest.raises(CacheMismatch) as caught:
        FeatureDataset(tmp_path, "train", ENCODER)
    assert ENCODER in str(caught.value)
    assert "openai/clip-vit-large-patch14" in str(caught.value)


def test_a_cache_without_provenance_is_still_read(tmp_path: Path) -> None:
    """The first real results came from a cache written before this field.

    Refusing it would discard verified work to enforce a field that did not
    exist when it was produced.
    """
    path = tmp_path / "train.pt"
    torch.save(
        {
            "image_features": torch.randn(3, 8),
            "text_features": torch.randn(3, 8),
            "labels": torch.zeros(3, dtype=torch.long),
            "uids": ["a", "b", "c"],
        },
        path,
    )
    data = FeatureDataset(tmp_path, "train", ENCODER)
    assert len(data) == 3
    assert data.meta == {}


def test_no_expectation_means_no_check(tmp_path: Path) -> None:
    write(tmp_path, encoder="openai/clip-vit-large-patch14")
    assert len(FeatureDataset(tmp_path, "train")) == 4


def test_mismatched_lengths_are_still_caught(tmp_path: Path) -> None:
    """The older guard must survive the new one being added above it."""
    path = tmp_path / "train.pt"
    torch.save(
        {
            "image_features": torch.randn(3, 8),
            "text_features": torch.randn(2, 8),
            "labels": torch.zeros(3, dtype=torch.long),
            "uids": ["a", "b", "c"],
            "meta": {"encoder": ENCODER},
        },
        path,
    )
    with pytest.raises(ValueError, match="disagree in length"):
        FeatureDataset(tmp_path, "train", ENCODER)


def test_a_missing_cache_says_how_to_build_one(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="extract_features"):
        FeatureDataset(tmp_path, "train", ENCODER)
