"""Frozen vision-language encoders.

The encoder is deliberately the only place that knows about a specific model.
Swapping English CLIP for a multilingual one is a config change, which is what
the Persian experiments need.

Two calling conventions, chosen by `EncoderConfig.kind`:

- "clip": one model exposing `get_image_features` / `get_text_features`.
  Covers OpenAI CLIP, SigLIP and jina-clip-v2.
- "dual_tower": two separately published towers read through `pooler_output`.
  CLIPfa ships this way — a RoBERTa text tower and a CLIP vision tower,
  trained together but released as two models — and its model card specifies
  exactly this pooling.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from .config import EncoderConfig
from .utils.device import resolve_device

__all__ = [
    "DualTowerEncoder",
    "Encoder",
    "FrozenEncoder",
    "build_encoder",
    "resolve_device",
]


class Encoder(Protocol):
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor: ...

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor: ...


def _freeze(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def _as_features(output) -> torch.Tensor:
    """Take the joint-space embedding out of whatever the model returned.

    transformers 4.x returns a plain tensor from get_image_features /
    get_text_features; 5.x returns a BaseModelOutputWithPooling whose
    pooler_output holds the same projected embedding. Supporting both keeps
    one requirements range working.
    """
    if torch.is_tensor(output):
        return output.float().cpu()
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(output, attr, None)
        if torch.is_tensor(value):
            return value.float().cpu()
    raise TypeError(
        f"cannot find an embedding tensor on a {type(output).__name__}; "
        "the installed transformers returns something this code does not know"
    )


class FrozenEncoder:
    """A CLIP-style dual encoder published as a single model."""

    def __init__(self, config: EncoderConfig, device: torch.device) -> None:
        self.config = config
        self.device = device
        kwargs = {"trust_remote_code": config.trust_remote_code}
        self.model = _freeze(AutoModel.from_pretrained(config.name, **kwargs), device)
        self.processor = AutoProcessor.from_pretrained(config.name, **kwargs)

        for attr in ("get_image_features", "get_text_features"):
            if not hasattr(self.model, attr):
                raise TypeError(
                    f"{config.name} does not expose {attr}(); set encoder.kind "
                    'to "dual_tower" if it ships as two separate models'
                )

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return _as_features(self.model.get_image_features(**inputs))

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
        return _as_features(self.model.get_text_features(**inputs))

    @property
    def feature_dim(self) -> int:
        config = self.model.config
        for attr in ("projection_dim", "hidden_size"):
            if hasattr(config, attr):
                return int(getattr(config, attr))
        raise AttributeError(f"cannot infer feature dim for {self.config.name}")


class DualTowerEncoder:
    """Two separately published towers, pooled via `pooler_output`."""

    def __init__(self, config: EncoderConfig, device: torch.device) -> None:
        if not (config.text_model and config.vision_model):
            raise ValueError(
                'encoder.kind "dual_tower" needs both text_model and vision_model'
            )
        self.config = config
        self.device = device
        kwargs = {"trust_remote_code": config.trust_remote_code}

        self.text_model = _freeze(
            AutoModel.from_pretrained(config.text_model, **kwargs), device
        )
        self.vision_model = _freeze(
            AutoModel.from_pretrained(config.vision_model, **kwargs), device
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model, **kwargs)
        self.processor = AutoProcessor.from_pretrained(config.vision_model, **kwargs)

    @staticmethod
    def _pool(output) -> torch.Tensor:
        pooled = getattr(output, "pooler_output", None)
        if pooled is None:
            raise TypeError(
                "tower has no pooler_output; this encoder reads the pooled "
                "representation the model card specifies"
            )
        return pooled.float().cpu()

    @torch.no_grad()
    def encode_images(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._pool(self.vision_model(**inputs))

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        inputs = self.tokenizer(
            list(texts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.config.max_text_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self._pool(self.text_model(**inputs))

    @property
    def feature_dim(self) -> int:
        return int(self.text_model.config.hidden_size)


def build_encoder(config: EncoderConfig, device: torch.device) -> Encoder:
    if config.kind == "clip":
        return FrozenEncoder(config, device)
    if config.kind == "dual_tower":
        return DualTowerEncoder(config, device)
    raise ValueError(
        f'unknown encoder kind {config.kind!r}, expected "clip" or "dual_tower"'
    )
