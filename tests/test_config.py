from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize(
    "path", sorted(CONFIG_DIR.rglob("*.yaml")), ids=lambda p: p.name
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


def test_result_dir_separates_architectures(tmp_path: Path) -> None:
    """memeclip also fuses by product, so fusion alone must not key results."""
    from src.train import result_dir

    base = """
model:
  architecture: {arch}
  fusion: align
"""
    first = tmp_path / "a.yaml"
    first.write_text(base.format(arch="hateclipper"))
    second = tmp_path / "b.yaml"
    second.write_text(base.format(arch="memeclip"))

    assert result_dir(load_config(first)) != result_dir(load_config(second))


def test_result_dir_separates_encoders(tmp_path: Path) -> None:
    """The encoder comparison is the contribution; it cannot share a path.

    Without this, running CLIP's align variant would write its five seed
    files over jina's, and summarise would average whatever was left with
    nothing in its output saying two encoders had been mixed.
    """
    from src.train import result_dir

    base = """
encoder:
  name: {name}
model:
  architecture: hateclipper
  fusion: align
"""
    first = tmp_path / "a.yaml"
    first.write_text(base.format(name="openai/clip-vit-large-patch14"))
    second = tmp_path / "b.yaml"
    second.write_text(base.format(name="jinaai/jina-clip-v2"))

    assert result_dir(load_config(first)) != result_dir(load_config(second))


def test_result_dir_and_feature_dir_agree_on_the_encoder(tmp_path: Path) -> None:
    """One slug function, so the two paths cannot disagree."""
    from src.train import result_dir

    path = tmp_path / "a.yaml"
    path.write_text("encoder:\n  name: jinaai/jina-clip-v2\n")
    config = load_config(path)

    assert config.encoder_slug == "jinaai__jina-clip-v2"
    assert config.encoder_slug in result_dir(config).parts
    assert config.encoder_slug in config.feature_dir.parts


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  fusionn: align\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)
