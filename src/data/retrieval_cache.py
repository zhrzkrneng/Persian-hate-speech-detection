"""Image features for Crossmodal-3600, computed once and shared across languages.

Crossmodal-3600 captions the same 3600 images in every language it covers, so
retrieval in six languages re-encodes the same images six times. On CPU that is
most of the cost: encoding 1000 images takes minutes per encoder, encoding
their captions takes seconds. Measured across the six languages this project
reads, all of them list the identical 3600 keys in the identical order, which is
what makes one cache per encoder correct.

That "which is what makes it correct" is the whole reason this module exists
rather than a dictionary in the caller. Sharing image features between two
languages is only valid while the two read the same images in the same order,
and nothing in the dataset guarantees that: `Crossmodal3600` skips any image
with no caption in the requested language, so a language with a gap would
produce a shorter, differently aligned list and every retrieval score computed
against the cache would be silently wrong -- a caption compared with another
image's features. So the keys are stored with the features and checked on
every read, and a mismatch is refused rather than reconciled.

The cache lives under `features/`, which is gitignored: it is a derived
tensor, not a result.
"""

from __future__ import annotations

from pathlib import Path

import torch


class RetrievalCacheError(RuntimeError):
    """A cache that cannot be used for the request being made of it."""


def cache_path(root: Path, preset: str, count: int) -> Path:
    """One file per encoder and image count.

    The count is in the name because `--limit` is how a quick check is run:
    a 50-image trial must not be mistaken for the 1000-image measurement, and
    keying on the encoder alone would let it be.
    """
    return Path(root) / "xm3600" / preset / f"images_{count}.pt"


def save(path: Path, features: torch.Tensor, keys: list[str], encoder: str) -> None:
    if features.shape[0] != len(keys):
        raise RetrievalCacheError(
            f"{features.shape[0]} feature rows for {len(keys)} images; "
            "the two must correspond row for row"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    from .features import _versions

    torch.save(
        {
            "features": features,
            "keys": list(keys),
            "encoder": encoder,
            "versions": _versions(),
        },
        temporary,
    )
    temporary.replace(path)


def load(path: Path, keys: list[str], encoder: str) -> torch.Tensor | None:
    """The cached features, or None if there is no cache yet.

    Raises rather than returns None when a cache exists but does not match:
    silently re-encoding would hide a real disagreement, and using it would
    compare captions against the wrong images.
    """
    if not path.exists():
        return None

    from .features import _major, _versions

    payload = torch.load(path, map_location="cpu", weights_only=False)

    # Image features differ across the transformers major boundary, measured at
    # up to 7.1e-02 on a feature scale of 11.4, because the default image
    # processor changed. A cached image pass from the other side of it would
    # silently mix two preprocessing pipelines into one retrieval table.
    cached_major = _major((payload.get("versions") or {}).get("transformers"))
    running_major = _major(_versions().get("transformers"))
    if cached_major and running_major and cached_major != running_major:
        raise RetrievalCacheError(
            f"{path} was encoded under transformers {cached_major}.x and this "
            f"run has {running_major}.x; the default image processor differs, "
            "so delete it and re-encode rather than reusing it."
        )

    recorded = payload.get("encoder")
    if recorded != encoder:
        raise RetrievalCacheError(
            f"{path} was written by {recorded!r}, not {encoder!r}. Image "
            "features from one encoder cannot stand in for another's."
        )

    cached_keys = list(payload.get("keys", []))
    if cached_keys != list(keys):
        if set(cached_keys) == set(keys):
            raise RetrievalCacheError(
                f"{path} holds the same {len(keys)} images in a different order. "
                "Features are matched to captions by row, so the order is part "
                "of the data."
            )
        raise RetrievalCacheError(
            f"{path} holds {len(cached_keys)} images, the request has "
            f"{len(keys)}, and they are not the same set. A language missing "
            "captions for some image produces a different list; re-encode "
            "rather than mix them."
        )

    features = payload["features"]
    if features.shape[0] != len(keys):
        raise RetrievalCacheError(
            f"{path}: {features.shape[0]} rows for {len(keys)} keys"
        )
    return features


# --- shards -------------------------------------------------------------
#
# The image pass is written in pieces for the same reason the Hateful Memes
# cache is: this container restarts without notice, and a two-hour cap kills a
# background job outright. jina-clip-v2 works at 512x512 at batch 8, and its
# pass over 1000 images ran for ninety-five minutes without finishing; when the
# job was stopped, the whole ninety-five minutes was gone, because the cache is
# written once, at the end. A shard every `shard_size` images costs the shard
# in flight instead.
#
# Shards carry the same checks as the finished file -- encoder, library major
# version, and the keys themselves -- and must tile the range from 0 with no
# gap. Half a pass from each of two encoders is refused rather than joined.


def shard_dir(path: Path) -> Path:
    """Where the pieces of `path` live while it is being built."""
    return path.with_suffix("")


def save_shard(
    path: Path, start: int, features: torch.Tensor, keys: list[str], encoder: str
) -> Path:
    from .features import _versions

    if features.shape[0] != len(keys):
        raise RetrievalCacheError(
            f"shard at {start}: {features.shape[0]} rows for {len(keys)} keys"
        )
    directory = shard_dir(path)
    directory.mkdir(parents=True, exist_ok=True)
    out = directory / f"shard_{start:06d}.pt"
    temporary = out.with_suffix(".tmp")
    torch.save(
        {
            "features": features,
            "keys": list(keys),
            "encoder": encoder,
            "start": start,
            "versions": _versions(),
        },
        temporary,
    )
    temporary.replace(out)
    return out


def load_shards(path: Path, keys: list[str], encoder: str) -> tuple[torch.Tensor, int]:
    """Whatever has been encoded so far, and how many images that is.

    Returns an empty tensor and 0 when there is nothing to resume from. A shard
    that disagrees about the encoder, the library version or the keys is
    refused: resuming across those boundaries would join two different passes.
    """
    from .features import _major, _versions

    directory = shard_dir(path)
    if not directory.exists():
        return torch.empty(0), 0

    running = _major(_versions().get("transformers"))
    pieces = []
    for shard in sorted(directory.glob("shard_*.pt")):
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        if payload.get("encoder") != encoder:
            raise RetrievalCacheError(
                f"{shard} was written by {payload.get('encoder')!r}, not "
                f"{encoder!r}; delete {directory} and start again."
            )
        cached = _major((payload.get("versions") or {}).get("transformers"))
        if cached and running and cached != running:
            raise RetrievalCacheError(
                f"{shard} was encoded under transformers {cached}.x and this "
                f"run has {running}.x; delete {directory} and start again."
            )
        pieces.append(payload)

    if not pieces:
        return torch.empty(0), 0

    # They must tile the range from 0 with no gap and no overlap, in order.
    position = 0
    for payload in pieces:
        if payload["start"] != position:
            raise RetrievalCacheError(
                f"{directory}: shards do not tile from 0 -- expected a shard at "
                f"{position}, found one at {payload['start']}. Delete the "
                "directory and start again."
            )
        width = payload["features"].shape[0]
        if list(payload["keys"]) != list(keys[position : position + width]):
            raise RetrievalCacheError(
                f"{directory}: the shard at {position} holds different images "
                "than this run asks for. Delete the directory and start again."
            )
        position += width

    return torch.cat([p["features"] for p in pieces]), position


def clear_shards(path: Path) -> int:
    directory = shard_dir(path)
    if not directory.exists():
        return 0
    removed = 0
    for shard in directory.glob("shard_*.pt"):
        shard.unlink()
        removed += 1
    directory.rmdir()
    return removed
