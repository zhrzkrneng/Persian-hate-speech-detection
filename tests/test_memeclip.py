"""MemeCLIP's components, each testable on its own so an ablation is meaningful."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ModelConfig  # noqa: E402
from src.models import build_model  # noqa: E402
from src.models.memeclip import (  # noqa: E402
    Adapter,
    CosineClassifier,
    MemeClipClassifier,
)


def config(**overrides) -> ModelConfig:
    base = {
        "architecture": "memeclip",
        "projection_dim": 16,
        "pre_output_dim": 16,
        "adapter_reduction": 4,
    }
    return ModelConfig(**{**base, **overrides})


def test_forward_shape() -> None:
    model = MemeClipClassifier(config(), feature_dim=32)
    assert model(torch.randn(5, 32), torch.randn(5, 32)).shape == (5, 2)


def test_adapter_preserves_width() -> None:
    assert Adapter(16, reduction=4)(torch.randn(3, 16)).shape == (3, 16)


def test_adapter_rejects_impossible_reduction() -> None:
    with pytest.raises(ValueError, match="too large"):
        Adapter(2, reduction=8)


def test_zero_ratio_disables_adapters() -> None:
    """ratio 0 must leave no adapter parameters, so the ablation is real."""
    with_adapters = MemeClipClassifier(config(adapter_ratio=0.2), feature_dim=32)
    without = MemeClipClassifier(config(adapter_ratio=0.0), feature_dim=32)

    assert with_adapters.use_adapters
    assert not without.use_adapters
    assert without.trainable_parameters() < with_adapters.trainable_parameters()
    assert not any("adapter" in name for name, _ in without.named_parameters())


def test_cosine_logits_are_bounded_by_scale() -> None:
    """Both sides are normalised, so a logit cannot exceed the scale."""
    head = CosineClassifier(feat_dim=16, num_classes=2, scale=30.0)
    logits = head(torch.randn(8, 16) * 100)
    assert logits.abs().max() <= 30.0 + 1e-4


def test_cosine_is_invariant_to_input_magnitude() -> None:
    head = CosineClassifier(feat_dim=16, num_classes=2)
    x = torch.randn(4, 16)
    torch.testing.assert_close(head(x), head(x * 7.5), rtol=1e-4, atol=1e-4)


def test_head_init_from_text() -> None:
    head = CosineClassifier(feat_dim=8, num_classes=2)
    text = torch.randn(2, 8)
    head.init_from_text(text)
    torch.testing.assert_close(head.weight.data, text)


def test_head_init_rejects_wrong_shape() -> None:
    head = CosineClassifier(feat_dim=8, num_classes=2)
    with pytest.raises(ValueError, match="expected text features"):
        head.init_from_text(torch.randn(3, 8))


def test_normalisation_toggle_changes_output() -> None:
    torch.manual_seed(0)
    image, text = torch.randn(4, 32), torch.randn(4, 32)

    torch.manual_seed(1)
    normalised = MemeClipClassifier(config(normalise_before_fusion=True), 32).eval()
    torch.manual_seed(1)
    raw = MemeClipClassifier(config(normalise_before_fusion=False), 32).eval()

    with torch.no_grad():
        assert not torch.allclose(normalised(image, text), raw(image, text))


@pytest.mark.parametrize("architecture", ["hateclipper", "memeclip"])
def test_build_model_dispatches(architecture: str) -> None:
    model = build_model(config(architecture=architecture), feature_dim=32)
    assert model(torch.randn(2, 32), torch.randn(2, 32)).shape == (2, 2)


def test_build_model_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown architecture"):
        build_model(config(architecture="clipbert"), feature_dim=32)
