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
    torch.save(
        {"features": features, "keys": list(keys), "encoder": encoder},
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

    payload = torch.load(path, map_location="cpu", weights_only=False)
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
