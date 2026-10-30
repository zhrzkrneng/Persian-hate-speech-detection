"""Cached encoder features.

Encoding every meme with CLIP is the slow part and it only depends on the
encoder, never on the fusion or the classifier. So it happens once, is written
to disk, and every later experiment reads tensors.

A cache file therefore has to say what produced it. Two reasons, both real
here:

- The directory is keyed by encoder name, which protects against reading one
  encoder's features as another's only for as long as nobody moves, copies or
  re-downloads a file. This project's caches have been built in one place and
  trained on in another, so the claim that a file holds a given encoder's
  features rests on the path alone. Written inside the file, it can be checked.
- The transformers version changes the features, which was an assumption
  until it was measured. Encoding the same 64 memes under 4.57.6 and comparing
  with a cache written under 5.22.0: the **text** features are bit-identical,
  and the **image** features are not -- max |delta| of 7.1e-02 on CLIP
  ViT-L/14 against a feature scale of 11.4, 1.9e-02 on ViT-B/32, 1.3e-02 on
  CLIPfa. The cause is in the library's own warning: 5.x defaults to the fast
  torchvision image processor where 4.x uses the slow PIL one for these
  checkpoints, so the image is resized and normalised slightly differently.
  Neither is wrong; they are simply not the same pipeline, and an encoder
  comparison needs one. A cache whose recorded transformers major version
  disagrees with the running one is therefore refused.

Nothing here imports transformers. Versions are read from installed package
metadata, which does not execute the package: training reads this module, and
training must not pull in an encoder.
"""

from __future__ import annotations

from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import torch
from torch.utils.data import Dataset

PROVENANCE_PACKAGES = ("torch", "transformers")


class CacheMismatch(ValueError):
    """The cache was built by something other than what is being asked for."""


def _major(version: str | None) -> str | None:
    """The major component of a version string, or None if there isn't one."""
    if not version:
        return None
    head = version.split(".")[0]
    return head if head.isdigit() else None


def _versions() -> dict[str, str]:
    found = {}
    for package in PROVENANCE_PACKAGES:
        try:
            found[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            continue  # CI installs a subset; absence is not an error
    return found


class FeatureCache:
    """A split's image features, text features and labels on disk."""

    def __init__(self, directory: str | Path, split: str) -> None:
        self.split = split
        self.path = Path(directory) / f"{split}.pt"

    def exists(self) -> bool:
        return self.path.exists()

    def save(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: torch.Tensor,
        uids: list[str],
        encoder: str,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "image_features": image_features,
                "text_features": text_features,
                "labels": labels,
                "uids": uids,
                "meta": {
                    "encoder": encoder,
                    "split": self.split,
                    "count": len(uids),
                    "feature_dim": int(image_features.shape[1]),
                    "written": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "versions": _versions(),
                },
            },
            self.path,
        )

    def load(self) -> dict:
        if not self.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run scripts/extract_features.py first."
            )
        return torch.load(self.path, map_location="cpu", weights_only=False)


class ShardError(ValueError):
    """Shards on disk do not describe the split being encoded."""


def shard_path(directory: str | Path, split: str, start: int) -> Path:
    return Path(directory) / f"{split}.shard{start:06d}.pt"


def save_shard(
    directory: str | Path,
    split: str,
    start: int,
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    encoder: str,
) -> Path:
    """Write one contiguous piece of a split's features.

    Extraction on CPU takes hours, and this machine is restarted without
    notice. The first CLIP ViT-L/14 attempt died at batch 15 of 133 after 37
    minutes and left nothing at all, because a split's cache was only written
    once the whole split had been encoded. Shards make the cost of a restart
    the current shard rather than everything.
    """
    path = shard_path(directory, split, start)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "image_features": image_features,
            "text_features": text_features,
            "start": start,
            "count": len(image_features),
            "encoder": encoder,
            "split": split,
        },
        path,
    )
    return path


def load_shards(directory: str | Path, split: str, encoder: str) -> list[dict]:
    """Every shard on disk for this split, in order, validated.

    A shard recorded against another encoder is refused rather than ignored:
    silently encoding half a split with one encoder and half with another is
    the kind of mistake that produces a plausible number.
    """
    found = []
    for path in sorted(Path(directory).glob(f"{split}.shard*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("encoder") != encoder:
            raise ShardError(
                f"{path} was written from {payload.get('encoder')!r}, not "
                f"{encoder!r}. Delete the shards for this split and re-encode."
            )
        if payload.get("split") != split:
            raise ShardError(f"{path} holds split {payload.get('split')!r}")
        found.append(payload)

    found.sort(key=lambda p: p["start"])
    expected = 0
    for payload in found:
        if payload["start"] != expected:
            raise ShardError(
                f"{split}: shards do not tile from 0 -- expected a shard at "
                f"{expected}, found one at {payload['start']}. This happens "
                "when the shard size changes between runs; delete them and "
                "re-encode."
            )
        expected += payload["count"]
    return found


def clear_shards(directory: str | Path, split: str) -> int:
    paths = list(Path(directory).glob(f"{split}.shard*.pt"))
    for path in paths:
        path.unlink()
    return len(paths)


class FeatureDataset(Dataset):
    def __init__(
        self,
        directory: str | Path,
        split: str,
        expect_encoder: str | None = None,
    ) -> None:
        cache = FeatureCache(directory, split)
        payload = cache.load()
        self.meta: dict = payload.get("meta") or {}

        # A cache written before provenance was recorded is read, not refused:
        # the jina-clip-v2 features this project's first results came from
        # predate this field, and rejecting them would discard verified work.
        # An encoder that is recorded and disagrees is a different matter.
        recorded = self.meta.get("encoder")
        if expect_encoder and recorded and recorded != expect_encoder:
            raise CacheMismatch(
                f"{cache.path} was written from {recorded!r}, but this run "
                f"expects {expect_encoder!r}. The directory name is not "
                "evidence: check which features this file really holds."
            )
        if expect_encoder and not recorded:
            print(
                f"note: {cache.path} carries no provenance, so the encoder "
                f"cannot be checked; its path says {expect_encoder}"
            )

        # The transformers major version is checked for the same reason as the
        # encoder, and it is not a formality: measured on 64 memes, image
        # features differ between 4.57.6 and 5.22.0 by up to 7.1e-02 on a
        # feature scale of 11.4, because 5.x resizes images with the fast
        # torchvision processor where 4.x uses the slow PIL one. Text features
        # are identical. A table whose rows were encoded under both is
        # comparing two preprocessing pipelines as well as two encoders.
        #
        # Absent provenance is still read with a note, as above: the first real
        # results came from a cache written before this field existed.
        cached_major = _major((self.meta.get("versions") or {}).get("transformers"))
        running_major = _major(_versions().get("transformers"))
        if cached_major and running_major and cached_major != running_major:
            raise CacheMismatch(
                f"{cache.path} was encoded under transformers {cached_major}.x "
                f"and this run has {running_major}.x. Image features are not "
                "the same across that boundary -- the default image processor "
                "changed -- so re-extract this cache under the running version "
                "rather than mixing the two."
            )

        self.image_features: torch.Tensor = payload["image_features"]
        self.text_features: torch.Tensor = payload["text_features"]
        self.labels: torch.Tensor = payload["labels"]
        self.uids: list[str] = payload["uids"]

        counts = {
            "image_features": len(self.image_features),
            "text_features": len(self.text_features),
            "labels": len(self.labels),
            "uids": len(self.uids),
        }
        if len(set(counts.values())) != 1:
            raise ValueError(f"cached tensors disagree in length: {counts}")

    @property
    def feature_dim(self) -> int:
        return int(self.image_features.shape[1])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            self.image_features[index],
            self.text_features[index],
            self.labels[index],
        )
