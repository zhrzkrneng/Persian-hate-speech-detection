"""Experiment configuration loaded from YAML."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EncoderConfig:
    """Which vision-language encoder to extract frozen features from.

    `name` is a HuggingFace model id. The pipeline is written against models
    that expose `get_image_features` / `get_text_features`, which covers CLIP
    and the multilingual CLIP variants we intend to compare against.
    """

    name: str = "openai/clip-vit-large-patch14"
    trust_remote_code: bool = False
    max_text_length: int = 77
    batch_size: int = 64


@dataclass
class ModelConfig:
    architecture: str = "hateclipper"  # hateclipper | memeclip
    fusion: str = "align"  # concat | align | cross; hateclipper only
    projection_dim: int = 1024
    pre_output_dim: int = 1024
    num_pre_output_layers: int = 1
    dropout: float = 0.1
    num_classes: int = 2

    # memeclip only. Each is separable so an ablation can attribute the gap
    # between the two architectures to a specific component.
    adapter_ratio: float = 0.2  # 0 disables the adapters entirely
    adapter_reduction: int = 4
    cosine_scale: float = 30.0
    normalise_before_fusion: bool = True


@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    grad_clip: float = 0.1
    seed: int = 0
    num_workers: int = 2
    device: str = "auto"  # auto | cuda | cpu


@dataclass
class DataConfig:
    name: str = "hmc"
    root: Path = Path("data/hmc")
    feature_root: Path = Path("features")
    train_split: str = "train"
    val_split: str = "dev_seen"
    test_split: str = "test_seen"


@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    output_dir: Path = Path("runs")

    @property
    def feature_dir(self) -> Path:
        """Cache location, keyed by encoder so variants never collide."""
        slug = self.encoder.name.replace("/", "__")
        return self.data.feature_root / self.data.name / slug


def _build(cls: type, values: dict[str, Any]) -> Any:
    known = {f.name: f for f in fields(cls)}
    unknown = set(values) - set(known)
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    kwargs = {}
    for key, value in values.items():
        if known[key].type is Path or known[key].type == "Path":
            value = Path(value)
        kwargs[key] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}

    sections = {
        "encoder": EncoderConfig,
        "model": ModelConfig,
        "train": TrainConfig,
        "data": DataConfig,
    }
    kwargs: dict[str, Any] = {}
    for key, cls in sections.items():
        kwargs[key] = _build(cls, raw.pop(key, {}) or {})
    if "output_dir" in raw:
        kwargs["output_dir"] = Path(raw.pop("output_dir"))
    if raw:
        raise ValueError(f"unknown top-level config keys: {sorted(raw)}")

    return Config(**kwargs)
