"""Classification head operating on cached frozen encoder features.

The encoders stay frozen, so the trainable part is small: one projection per
modality, the fusion, and a shallow MLP. That is what makes a full sweep cheap
enough for a free Colab or Kaggle GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig
from .fusion import CrossModalFusion, fused_dim


class MemeClassifier(nn.Module):
    def __init__(self, config: ModelConfig, feature_dim: int) -> None:
        super().__init__()
        self.config = config

        self.image_projection = nn.Linear(feature_dim, config.projection_dim)
        self.text_projection = nn.Linear(feature_dim, config.projection_dim)
        self.fusion = CrossModalFusion(config.fusion)

        layers: list[nn.Module] = []
        in_dim = fused_dim(config.fusion, config.projection_dim)
        for _ in range(config.num_pre_output_layers):
            layers += [
                nn.Linear(in_dim, config.pre_output_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ]
            in_dim = config.pre_output_dim
        self.pre_output = nn.Sequential(*layers)
        self.output = nn.Linear(in_dim, config.num_classes)

    def forward(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        image = self.image_projection(image_features)
        text = self.text_projection(text_features)
        fused = self.fusion(image, text)
        return self.output(self.pre_output(fused))

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
