"""Sharing one image pass between languages, and refusing to when it is unsafe.

Crossmodal-3600 captions the same images in every language, so the expensive
image pass can be computed once per encoder. The saving is real -- on CPU the
images are minutes and the captions are seconds -- and so is the failure it
risks: features are matched to captions by row, so a cache built from a
different set of images, or the same set in a different order, would compare
every caption against the wrong image and still produce a plausible-looking
recall.

`Crossmodal3600` skips any image with no caption in the requested language, so
a language with a gap really would produce a different list. That is the case
these tests are about.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.retrieval_cache import (  # noqa: E402
    RetrievalCacheError,
    cache_path,
    clear_shards,
    load,
    load_shards,
    save,
    save_shard,
)

ENCODER = "openai/clip-vit-base-patch32"


def test_nothing_cached_yet_is_not_an_error(tmp_path: Path) -> None:
    assert load(tmp_path / "absent.pt", ["a", "b"], ENCODER) is None


def test_a_cache_round_trips(tmp_path: Path) -> None:
    path = cache_path(tmp_path, "clip-b32", 2)
    features = torch.randn(2, 8)
    save(path, features, ["a", "b"], ENCODER)
    loaded = load(path, ["a", "b"], ENCODER)
    assert loaded is not None
    assert torch.equal(loaded, features)


def test_a_second_language_reuses_the_same_file(tmp_path: Path) -> None:
    """The point of the whole module: the path does not mention the language."""
    assert cache_path(tmp_path, "clip-b32", 1000) == cache_path(
        tmp_path, "clip-b32", 1000
    )


def test_another_encoders_cache_is_refused(tmp_path: Path) -> None:
    path = cache_path(tmp_path, "clip-b32", 2)
    save(path, torch.randn(2, 8), ["a", "b"], ENCODER)
    with pytest.raises(RetrievalCacheError, match="cannot stand in for"):
        load(path, ["a", "b"], "jinaai/jina-clip-v2")


def test_the_same_images_in_another_order_are_refused(tmp_path: Path) -> None:
    """Reordering is the dangerous case: the count and the set both match."""
    path = cache_path(tmp_path, "clip-b32", 3)
    save(path, torch.randn(3, 8), ["a", "b", "c"], ENCODER)
    with pytest.raises(RetrievalCacheError, match="different order"):
        load(path, ["c", "b", "a"], ENCODER)


def test_a_different_set_of_images_is_refused(tmp_path: Path) -> None:
    path = cache_path(tmp_path, "clip-b32", 3)
    save(path, torch.randn(3, 8), ["a", "b", "c"], ENCODER)
    with pytest.raises(RetrievalCacheError, match="not the same set"):
        load(path, ["a", "b", "d"], ENCODER)


def test_a_different_count_is_a_different_file(tmp_path: Path) -> None:
    """--limit 50 is how a quick check is run; it must not shadow the real one."""
    small = cache_path(tmp_path, "clip-b32", 50)
    full = cache_path(tmp_path, "clip-b32", 1000)
    assert small != full
    save(small, torch.randn(50, 8), [str(i) for i in range(50)], ENCODER)
    assert load(full, [str(i) for i in range(1000)], ENCODER) is None


def test_rows_and_keys_must_correspond_on_write(tmp_path: Path) -> None:
    with pytest.raises(RetrievalCacheError, match="row for row"):
        save(
            cache_path(tmp_path, "clip-b32", 2), torch.randn(3, 8), ["a", "b"], ENCODER
        )


def test_a_partial_write_leaves_no_usable_cache(tmp_path: Path) -> None:
    """Written to a temporary name and renamed, so a kill cannot leave half a file.

    This container restarts without notice, which is why the HMC feature cache
    writes shards; the same reasoning applies to a file written in one go.
    """
    path = cache_path(tmp_path, "clip-b32", 2)
    save(path, torch.randn(2, 8), ["a", "b"], ENCODER)
    assert path.exists()
    assert not path.with_suffix(".tmp").exists()


def test_an_image_cache_from_another_transformers_major_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cached half is the image half, which is exactly the half that moved.

    Measured at up to 7.1e-02 on a feature scale of 11.4 between 4.57.6 and
    5.22.0: the default image processor changed from slow PIL to fast
    torchvision. Reusing such a cache would put two preprocessing pipelines
    into one retrieval table.
    """
    from src.data import features

    path = cache_path(tmp_path, "clip-b32", 2)
    monkeypatch.setattr(features, "_versions", lambda: {"transformers": "5.22.0"})
    save(path, torch.randn(2, 8), ["a", "b"], ENCODER)

    monkeypatch.setattr(features, "_versions", lambda: {"transformers": "4.57.6"})
    with pytest.raises(RetrievalCacheError, match="transformers 5.x"):
        load(path, ["a", "b"], ENCODER)


def test_the_same_transformers_major_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.data import features

    path = cache_path(tmp_path, "clip-b32", 2)
    monkeypatch.setattr(features, "_versions", lambda: {"transformers": "4.40.0"})
    features_tensor = torch.randn(2, 8)
    save(path, features_tensor, ["a", "b"], ENCODER)

    monkeypatch.setattr(features, "_versions", lambda: {"transformers": "4.57.6"})
    loaded = load(path, ["a", "b"], ENCODER)
    assert loaded is not None and torch.equal(loaded, features_tensor)


# --- shards -------------------------------------------------------------


def test_nothing_to_resume_from_is_not_an_error(tmp_path: Path) -> None:
    features, count = load_shards(cache_path(tmp_path, "jina-v2", 4), ["a"], ENCODER)
    assert count == 0 and features.numel() == 0


def test_shards_resume_where_they_stopped(tmp_path: Path) -> None:
    """The point of the whole thing: ninety-five minutes should not be lost."""
    path = cache_path(tmp_path, "jina-v2", 4)
    keys = ["a", "b", "c", "d"]
    first = torch.randn(2, 8)
    save_shard(path, 0, first, keys[:2], ENCODER)

    done, count = load_shards(path, keys, ENCODER)
    assert count == 2 and torch.equal(done, first)

    second = torch.randn(2, 8)
    save_shard(path, 2, second, keys[2:], ENCODER)
    done, count = load_shards(path, keys, ENCODER)
    assert count == 4
    assert torch.equal(done, torch.cat([first, second]))


def test_a_gap_in_the_shards_is_refused(tmp_path: Path) -> None:
    """Concatenating across a gap would misalign every caption after it."""
    path = cache_path(tmp_path, "jina-v2", 4)
    keys = ["a", "b", "c", "d"]
    save_shard(path, 0, torch.randn(2, 8), keys[:2], ENCODER)
    save_shard(path, 3, torch.randn(1, 8), keys[3:], ENCODER)
    with pytest.raises(RetrievalCacheError, match="do not tile from 0"):
        load_shards(path, keys, ENCODER)


def test_shards_from_another_encoder_are_refused(tmp_path: Path) -> None:
    path = cache_path(tmp_path, "jina-v2", 2)
    save_shard(path, 0, torch.randn(2, 8), ["a", "b"], ENCODER)
    with pytest.raises(RetrievalCacheError, match="not 'jinaai/jina-clip-v2'"):
        load_shards(path, ["a", "b"], "jinaai/jina-clip-v2")


def test_shards_from_another_library_major_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.data import features

    path = cache_path(tmp_path, "jina-v2", 2)
    monkeypatch.setattr(features, "_versions", lambda: {"transformers": "5.22.0"})
    save_shard(path, 0, torch.randn(2, 8), ["a", "b"], ENCODER)
    monkeypatch.setattr(features, "_versions", lambda: {"transformers": "4.57.6"})
    with pytest.raises(RetrievalCacheError, match="transformers 5.x"):
        load_shards(path, ["a", "b"], ENCODER)


def test_shards_holding_other_images_are_refused(tmp_path: Path) -> None:
    """Same count, same encoder, different images: the dangerous case."""
    path = cache_path(tmp_path, "jina-v2", 2)
    save_shard(path, 0, torch.randn(2, 8), ["a", "b"], ENCODER)
    with pytest.raises(RetrievalCacheError, match="different images"):
        load_shards(path, ["x", "y"], ENCODER)


def test_clearing_shards_leaves_the_finished_cache(tmp_path: Path) -> None:
    path = cache_path(tmp_path, "jina-v2", 2)
    save_shard(path, 0, torch.randn(2, 8), ["a", "b"], ENCODER)
    features = torch.randn(2, 8)
    save(path, features, ["a", "b"], ENCODER)
    assert clear_shards(path) == 1
    loaded = load(path, ["a", "b"], ENCODER)
    assert loaded is not None and torch.equal(loaded, features)
