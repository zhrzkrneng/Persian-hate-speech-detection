"""MemeCLIP (Shah et al., EMNLP 2024) as a second baseline.

Reimplemented from the authors' released code at
https://github.com/SiddhantBikram/MemeCLIP, against the architecture described
in https://aclanthology.org/2024.emnlp-main.959/.

It shares Hate-CLIPper's core: both project frozen CLIP features and fuse them
with an element-wise product. MemeCLIP adds three things on top, and the
`MemeClipClassifier` below keeps each one separable so an ablation can say
which of them is doing the work:

1. A residual feature adapter per modality, mixed back by `adapter_ratio`:
   `ratio * adapter(p) + (1 - ratio) * p`.
2. L2 normalisation of both projections before the product, which Hate-CLIPper
   does not do. The product of two unit vectors is bounded, so the fused
   feature cannot be dominated by a few large dimensions.
3. A cosine classifier whose weights can be initialised from CLIP's own text
   embeddings of the class names ("a photo of a {class}"), giving the head a
   semantic starting point rather than a random one.

Deliberate differences from the released code, both about cost rather than
architecture:

- The reference collator re-encodes CLIP features for every batch of every
  epoch, one image at a time in a Python loop. We read features from the cache
  instead, which is the same computation done once.
- Text-feature initialisation of the head is optional here. It needs the CLIP
  text tower, which the training path does not otherwise load; without it the
  cosine classifier starts from random weights and `init_head_from_text` can
  supply them later.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig


class Adapter(nn.Module):
    """Bottleneck MLP, mixed residually with its input by the caller."""

    def __init__(self, dim: int, reduction: int = 4) -> None:
        super().__init__()
        if reduction < 1 or dim // reduction < 1:
            raise ValueError(
                f"adapter reduction {reduction} is too large for width {dim}"
            )
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class CosineClassifier(nn.Module):
    """Scaled cosine similarity between the feature and a weight per class.

    Normalising both sides means the logit is an angle, not a magnitude, so the
    head can be initialised directly from CLIP text embeddings, which live on
    the unit sphere too.
    """

    def __init__(self, feat_dim: int, num_classes: int, scale: float = 30.0) -> None:
        super().__init__()
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, feat_dim))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (
            F.linear(F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1))
            * self.scale
        )

    @torch.no_grad()
    def init_from_text(self, text_features: torch.Tensor) -> None:
        """Seed the weights with one CLIP text embedding per class."""
        if text_features.shape != self.weight.shape:
            raise ValueError(
                f"expected text features of shape {tuple(self.weight.shape)}, "
                f"got {tuple(text_features.shape)}"
            )
        self.weight.copy_(text_features)


class MemeClipClassifier(nn.Module):
    def __init__(self, config: ModelConfig, feature_dim: int) -> None:
        super().__init__()
        self.config = config
        dim = config.projection_dim

        self.image_projection = nn.Sequential(
            nn.Linear(feature_dim, dim), nn.Dropout(config.dropout)
        )
        self.text_projection = nn.Sequential(
            nn.Linear(feature_dim, dim), nn.Dropout(config.dropout)
        )

        self.use_adapters = config.adapter_ratio > 0.0
        if self.use_adapters:
            self.image_adapter = Adapter(dim, config.adapter_reduction)
            self.text_adapter = Adapter(dim, config.adapter_reduction)

        layers: list[nn.Module] = [nn.Dropout(config.dropout)]
        in_dim = dim
        for _ in range(config.num_pre_output_layers):
            layers += [
                nn.Linear(in_dim, config.pre_output_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ]
            in_dim = config.pre_output_dim
        self.pre_output = nn.Sequential(*layers)

        self.classifier = CosineClassifier(
            in_dim, config.num_classes, config.cosine_scale
        )

    def _adapt(
        self, projection: torch.Tensor, adapter: nn.Module | None
    ) -> torch.Tensor:
        if adapter is None:
            return projection
        ratio = self.config.adapter_ratio
        return ratio * adapter(projection) + (1.0 - ratio) * projection

    def forward(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        image = self._adapt(
            self.image_projection(image_features),
            self.image_adapter if self.use_adapters else None,
        )
        text = self._adapt(
            self.text_projection(text_features),
            self.text_adapter if self.use_adapters else None,
        )

        if self.config.normalise_before_fusion:
            image = F.normalize(image, dim=-1)
            text = F.normalize(text, dim=-1)

        fused = image * text
        return self.classifier(self.pre_output(fused))

    def init_head_from_text(self, text_features: torch.Tensor) -> None:
        self.classifier.init_from_text(text_features)

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
