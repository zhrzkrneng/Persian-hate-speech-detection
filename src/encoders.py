"""Frozen vision-language encoders.

The encoder is deliberately the only place that knows about a specific model.
Swapping English CLIP for a multilingual one is a config change, which is what
the planned Persian experiments need.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .config import EncoderConfig
from .utils.device import resolve_device

__all__ = ["FrozenEncoder", "resolve_device"]


class FrozenEncoder:
    """Wraps a CLIP-style model and returns unnormalised pooled features."""

    def __init__(self, config: EncoderConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        self.model = AutoModel.from_pretrained(
            config.name, trust_remote_code=config.trust_remote_code
        )
        self.processor = AutoProcessor.from_pretrained(
            config.name, trust_remote_code=config.trust_remote_code
        )
        self.model.to(device)
        self.model.eval()
        self.model.requires_grad_(False)

        for attr in ("get_image_features", "get_text_features"):
            if not hasattr(self.model, attr):
                raise TypeError(
                    f"{config.name} does not expose {attr}(); this pipeline "
                    "expects a CLIP-style dual encoder"
                )

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model.get_image_features(**inputs).float().cpu()

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.processor(
            text=list(texts),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.config.max_text_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model.get_text_features(**inputs).float().cpu()

    @property
    def feature_dim(self) -> int:
        config = self.model.config
        for attr in ("projection_dim", "hidden_size"):
            if hasattr(config, attr):
                return int(getattr(config, attr))
        raise AttributeError(f"cannot infer feature dim for {self.config.name}")
