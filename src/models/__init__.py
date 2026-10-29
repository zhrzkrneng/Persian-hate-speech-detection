from __future__ import annotations

import torch.nn as nn

from ..config import ModelConfig
from .classifier import MemeClassifier
from .fusion import FUSIONS, CrossModalFusion, fused_dim
from .memeclip import Adapter, CosineClassifier, MemeClipClassifier

ARCHITECTURES = ("hateclipper", "memeclip")


def build_model(config: ModelConfig, feature_dim: int) -> nn.Module:
    """Pick the architecture named in the config.

    Both read the same cached features, so a comparison between them changes
    only the model and never the encoder or the data.
    """
    if config.architecture == "hateclipper":
        return MemeClassifier(config, feature_dim)
    if config.architecture == "memeclip":
        return MemeClipClassifier(config, feature_dim)
    raise ValueError(
        f"unknown architecture {config.architecture!r}, expected one of {ARCHITECTURES}"
    )


__all__ = [
    "ARCHITECTURES",
    "Adapter",
    "CosineClassifier",
    "CrossModalFusion",
    "FUSIONS",
    "MemeClassifier",
    "MemeClipClassifier",
    "build_model",
    "fused_dim",
]
