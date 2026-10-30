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


# ── resumable shards ─────────────────────────────────────────────────────


def shard(directory: Path, start: int, n: int, encoder: str = ENCODER, split="train"):
    from src.data.features import save_shard

    return save_shard(
        directory, split, start, torch.randn(n, 8), torch.randn(n, 8), encoder
    )


def test_shards_reload_in_order(tmp_path: Path) -> None:
    from src.data.features import load_shards

    shard(tmp_path, 1000, 500)  # written out of order on purpose
    shard(tmp_path, 0, 1000)

    found = load_shards(tmp_path, "train", ENCODER)
    assert [s["start"] for s in found] == [0, 1000]
    assert sum(s["count"] for s in found) == 1500


def test_no_shards_means_start_from_zero(tmp_path: Path) -> None:
    from src.data.features import load_shards

    assert load_shards(tmp_path, "train", ENCODER) == []


def test_a_shard_from_another_encoder_is_refused(tmp_path: Path) -> None:
    """Half a split from each encoder would produce a plausible number."""
    from src.data.features import ShardError, load_shards

    shard(tmp_path, 0, 100)
    shard(tmp_path, 100, 100, encoder="openai/clip-vit-large-patch14")
    with pytest.raises(ShardError, match="not 'jinaai/jina-clip-v2'"):
        load_shards(tmp_path, "train", ENCODER)


def test_a_gap_between_shards_is_refused(tmp_path: Path) -> None:
    """Happens when the shard size changes between runs."""
    from src.data.features import ShardError, load_shards

    shard(tmp_path, 0, 1000)
    shard(tmp_path, 2000, 1000)
    with pytest.raises(ShardError, match="do not tile from 0"):
        load_shards(tmp_path, "train", ENCODER)


def test_shards_of_one_split_do_not_disturb_another(tmp_path: Path) -> None:
    from src.data.features import load_shards

    shard(tmp_path, 0, 100, split="train")
    shard(tmp_path, 0, 50, split="test_seen")

    assert sum(s["count"] for s in load_shards(tmp_path, "train", ENCODER)) == 100
    assert sum(s["count"] for s in load_shards(tmp_path, "test_seen", ENCODER)) == 50


def test_clear_shards_removes_only_that_split(tmp_path: Path) -> None:
    from src.data.features import clear_shards, load_shards

    shard(tmp_path, 0, 100, split="train")
    shard(tmp_path, 0, 50, split="test_seen")

    assert clear_shards(tmp_path, "train") == 1
    assert load_shards(tmp_path, "train", ENCODER) == []
    assert len(load_shards(tmp_path, "test_seen", ENCODER)) == 1


def test_a_finished_cache_is_not_a_shard(tmp_path: Path) -> None:
    """clear_shards must never touch the cache the shards were built for."""
    from src.data.features import clear_shards

    write(tmp_path)
    assert clear_shards(tmp_path, "train") == 0
    assert (tmp_path / "train.pt").exists()


def test_a_cache_from_another_transformers_major_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Measured, not assumed: image features differ across that boundary.

    Encoding the same 64 memes under 4.57.6 and comparing with a cache written
    under 5.22.0 gives a max absolute difference of 7.1e-02 on CLIP ViT-L/14,
    against a feature scale of 11.4 -- the default image processor changed from
    the slow PIL one to the fast torchvision one. Text features are identical.
    Mixing the two means comparing preprocessing pipelines as well as encoders.
    """
    from src.data import features as module

    monkeypatch.setattr(module, "_versions", lambda: {"transformers": "5.22.0"})
    FeatureCache(tmp_path, "train").save(
        image_features=torch.randn(2, 8),
        text_features=torch.randn(2, 8),
        labels=torch.zeros(2, dtype=torch.long),
        uids=["a", "b"],
        encoder="openai/clip-vit-large-patch14",
    )

    monkeypatch.setattr(module, "_versions", lambda: {"transformers": "4.57.6"})
    with pytest.raises(CacheMismatch, match="transformers 5.x"):
        FeatureDataset(tmp_path, "train", "openai/clip-vit-large-patch14")


def test_the_same_major_with_a_different_minor_is_fine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the major boundary was measured to matter; do not refuse more."""
    from src.data import features as module

    monkeypatch.setattr(module, "_versions", lambda: {"transformers": "4.40.0"})
    FeatureCache(tmp_path, "train").save(
        image_features=torch.randn(2, 8),
        text_features=torch.randn(2, 8),
        labels=torch.zeros(2, dtype=torch.long),
        uids=["a", "b"],
        encoder="openai/clip-vit-large-patch14",
    )

    monkeypatch.setattr(module, "_versions", lambda: {"transformers": "4.57.6"})
    assert len(FeatureDataset(tmp_path, "train", "openai/clip-vit-large-patch14")) == 2
