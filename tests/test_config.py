from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    "path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name
)
def test_shipped_configs_load(path: Path) -> None:
    config = load_config(path)
    assert config.model.fusion in {"concat", "align", "cross"}
    assert config.train.epochs > 0


def test_feature_dir_is_keyed_by_encoder(tmp_path: Path) -> None:
    """Two encoders must never share a feature cache."""
    base = """
encoder:
  name: {name}
data:
  name: hmc
  feature_root: features
"""
    first = tmp_path / "a.yaml"
    first.write_text(base.format(name="openai/clip-vit-large-patch14"))
    second = tmp_path / "b.yaml"
    second.write_text(base.format(name="jinaai/jina-clip-v2"))

    assert load_config(first).feature_dir != load_config(second).feature_dir


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  fusionn: align\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)
