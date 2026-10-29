"""The fusion maths is the one place a silent bug would invalidate every result."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ModelConfig  # noqa: E402
from src.models.classifier import MemeClassifier  # noqa: E402
from src.models.fusion import CrossModalFusion, fused_dim  # noqa: E402


@pytest.mark.parametrize(
    ("fusion", "expected"),
    [("concat", 16), ("align", 8), ("cross", 64)],
)
def test_fused_dim(fusion: str, expected: int) -> None:
    assert fused_dim(fusion, 8) == expected


@pytest.mark.parametrize("fusion", ["concat", "align", "cross"])
def test_fusion_output_shape(fusion: str) -> None:
    image = torch.randn(4, 8)
    text = torch.randn(4, 8)
    out = CrossModalFusion(fusion)(image, text)
    assert out.shape == (4, fused_dim(fusion, 8))


def test_align_is_the_diagonal_of_cross() -> None:
    """align keeps diag(p_i (x) p_t); its sum is the dot product."""
    image = torch.randn(3, 8)
    text = torch.randn(3, 8)

    align = CrossModalFusion("align")(image, text)
    cross = CrossModalFusion("cross")(image, text).reshape(3, 8, 8)

    torch.testing.assert_close(align, torch.diagonal(cross, dim1=1, dim2=2))
    torch.testing.assert_close(align.sum(-1), (image * text).sum(-1))


def test_mismatched_widths_are_rejected() -> None:
    with pytest.raises(ValueError, match="must match"):
        CrossModalFusion("align")(torch.randn(2, 8), torch.randn(2, 4))


def test_unknown_fusion_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown fusion"):
        CrossModalFusion("bilinear")


def test_classifier_forward() -> None:
    config = ModelConfig(fusion="align", projection_dim=16, pre_output_dim=8)
    model = MemeClassifier(config, feature_dim=32)
    logits = model(torch.randn(5, 32), torch.randn(5, 32))
    assert logits.shape == (5, 2)
    assert model.trainable_parameters() > 0
