"""Cached encoder features.

Encoding every meme with CLIP is the slow part and it only depends on the
encoder, never on the fusion or the classifier. So it happens once, is written
to disk, and every later experiment reads tensors.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class FeatureCache:
    """A split's image features, text features and labels on disk."""

    def __init__(self, directory: str | Path, split: str) -> None:
        self.path = Path(directory) / f"{split}.pt"

    def exists(self) -> bool:
        return self.path.exists()

    def save(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        labels: torch.Tensor,
        uids: list[str],
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "image_features": image_features,
                "text_features": text_features,
                "labels": labels,
                "uids": uids,
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
    def __init__(self, directory: str | Path, split: str) -> None:
        payload = FeatureCache(directory, split).load()
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
