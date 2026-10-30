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
- The two halves of the comparison were encoded under different library
  versions: jina-clip-v2 on a Kaggle GPU, CLIP ViT-L/14 on this machine. For a
  fixed checkpoint the values should not depend on the transformers version,
  but "should not" is an assumption, and transformers 5 did change the output
  objects enough that `_as_features()` exists to absorb it. A number in a paper
  should be traceable to the versions that produced it.

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
