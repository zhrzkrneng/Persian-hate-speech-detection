"""Every shipped config must build a model that runs.

A config is easy to get subtly wrong — a fusion name that does not exist, a
projection width the architecture cannot take, an encoder kind nothing
handles. Caught here, it costs nothing; caught after feature extraction, it
costs however long that took, which for jina-clip-v2 on CPU was over an hour.

This builds each config's model and pushes a batch through it. It does not
load an encoder: that needs the network and a multi-gigabyte download, and
the encoder is exercised by the retrieval study instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.models import ARCHITECTURES, build_model  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
# rglob, not glob: an ablation group lives in configs/ablation/, and a glob
# that stops at the top level would silently stop covering it.
CONFIGS = sorted(CONFIG_DIR.rglob("*.yaml"))

# Encoder embedding widths, so each config is built at the width its own
# encoder actually produces rather than at an arbitrary one.
# Read from each model's published config.json, not guessed: CLIP ViT-B/32
# projects to 512, and CLIPfa pools pooler_output from a RoBERTa text tower and
# a CLIPVisionModel that are both 768 wide.
FEATURE_DIM = {
    "openai/clip-vit-large-patch14": 768,
    "openai/clip-vit-base-patch32": 512,
    "jinaai/jina-clip-v2": 1024,
    "CLIPfa": 768,
}


def test_configs_exist() -> None:
    """A glob that matched nothing would make every test below vacuous."""
    assert CONFIGS, f"no configs under {CONFIG_DIR}"


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_config_builds_and_runs(path: Path) -> None:
    config = load_config(path)
    assert config.model.architecture in ARCHITECTURES

    dim = FEATURE_DIM.get(config.encoder.name)
    assert dim is not None, (
        f"{path.name} names encoder {config.encoder.name!r}, whose embedding "
        "width is not recorded in FEATURE_DIM; add it so this config is "
        "built at the width it will really see"
    )

    model = build_model(config.model, dim)
    logits = model(torch.randn(2, dim), torch.randn(2, dim))
    assert logits.shape == (2, config.model.num_classes)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_feature_cache_is_not_shared_across_encoders(path: Path) -> None:
    """Two encoders sharing a cache would train on the wrong features."""
    config = load_config(path)
    assert config.encoder.name.replace("/", "__") in str(config.feature_dir)


def test_jina_config_matches_the_clip_one_except_for_the_encoder() -> None:
    """The comparison is only meaningful if nothing else differs."""
    clip = load_config(CONFIG_DIR / "hmc_align.yaml")
    jina = load_config(CONFIG_DIR / "hmc_align_jina.yaml")

    assert clip.model == jina.model
    assert clip.train == jina.train
    assert clip.data == jina.data
    assert clip.encoder.name != jina.encoder.name


def test_jina_config_carries_its_requirements() -> None:
    """jina-clip-v2 does not load without these, and the OOM was real."""
    jina = load_config(CONFIG_DIR / "hmc_align_jina.yaml")
    assert jina.encoder.trust_remote_code is True
    assert jina.encoder.batch_size <= 8, (
        "batch 32 was killed by the OOM killer at 512x512 on a 15 GB machine"
    )
