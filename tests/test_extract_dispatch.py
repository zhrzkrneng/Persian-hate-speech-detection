"""Feature extraction must pick the encoder class from the config's `kind`.

It did not. `scripts/extract_features.py` constructed `FrozenEncoder`
directly, so a `kind: dual_tower` config was loaded as if it were a single
CLIP-style model. CLIPfa is two separately published towers -- a Persian
RoBERTa and a CLIP vision tower -- so its `name` is only a label, and loading
that label as a model id fails with "CLIPfa is not a local folder and is not a
valid model identifier".

The bug survived because `scripts/eval_retrieval.py` had always gone through
`build_encoder`: CLIPfa could be measured for retrieval and never encoded for
classification, and nothing connected the two until an encoder of that kind
first entered the classification path.

tests/test_configs_build.py could not catch it -- it builds the classifier
head, not the encoder, and deliberately never downloads a model. So this test
checks the dispatch without loading anything: the encoder is replaced, and
every split's cache already exists, so `run` hands the config to the factory
and then has nothing left to encode.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.extract_features as extract  # noqa: E402
from src.data.features import FeatureCache  # noqa: E402

CONFIG = """
encoder:
  name: {name}
  kind: {kind}
  text_model: SajjadAyoubi/clip-fa-text
  vision_model: SajjadAyoubi/clip-fa-vision
data:
  name: hmc
  root: {root}
  feature_root: {features}
train:
  device: cpu
"""


def prepare(tmp_path: Path, name: str, kind: str) -> Path:
    """A config whose three split caches already exist, so nothing is encoded."""
    config = tmp_path / "config.yaml"
    config.write_text(
        CONFIG.format(
            name=name,
            kind=kind,
            root=tmp_path / "data",
            features=tmp_path / "features",
        ),
        encoding="utf-8",
    )

    from src.config import load_config

    directory = load_config(config).feature_dir
    for split in ("train", "dev_seen", "test_seen"):
        FeatureCache(directory, split).save(
            image_features=torch.randn(2, 8),
            text_features=torch.randn(2, 8),
            labels=torch.zeros(2, dtype=torch.long),
            uids=["a", "b"],
            encoder=name,
        )
    return config


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("openai/clip-vit-large-patch14", "clip"),
        ("CLIPfa", "dual_tower"),
    ],
)
def test_the_configs_kind_reaches_the_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, kind: str
) -> None:
    from src.config import load_config

    seen = {}

    def fake_build_encoder(encoder_config, device):
        seen["name"] = encoder_config.name
        seen["kind"] = encoder_config.kind
        return object()

    monkeypatch.setattr(extract, "build_encoder", fake_build_encoder)
    extract.run(
        load_config(prepare(tmp_path, name, kind)), overwrite=False, shard_size=1000
    )

    assert seen == {"name": name, "kind": kind}


def test_extraction_does_not_name_an_encoder_class_itself() -> None:
    """The factory is the only place that may choose; belt and braces.

    A direct reference would compile and pass the test above if someone wired
    it back in beside the factory call, so the source is checked too.
    """
    source = Path(extract.__file__).read_text(encoding="utf-8")
    for class_name in ("FrozenEncoder(", "DualTowerEncoder("):
        assert class_name not in source, (
            f"{class_name} is constructed directly in extract_features; the "
            "config's `kind` must choose, through build_encoder"
        )
