"""A retrieval preset and its classification config must name one model.

The whole argument of this project is a relationship between two numbers per
encoder: the alignment measured by zero-shot retrieval on Crossmodal-3600, and
the AUROC reached by a classifier trained on its cached features. The two are
produced by different scripts reading different files -- `scripts/eval_retrieval.py`
has its own PRESETS table, `scripts/train.py` reads `configs/`. Nothing
connects them, so a preset that drifted from its config would put a point on
the alignment axis for one model and on the performance axis for another, and
every figure would still render.

`max_text_length` is deliberately not compared. It is a truncation limit, not
part of the model's identity, and the two scripts read texts of very different
lengths: meme text in classification, image captions here. What it does on
this benchmark was measured rather than assumed -- see
`scripts/measure_tokenisation.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.eval_retrieval import PRESETS  # noqa: E402
from src.config import load_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Which preset each encoder's configs are expected to correspond to. An encoder
# appearing in configs/ and in no preset is the failure this maps against: it
# would have an AUROC and no place on the alignment axis.
PRESET_FOR_MODEL = {
    "openai/clip-vit-large-patch14": "clip-l14",
    "openai/clip-vit-base-patch32": "clip-b32",
    "jinaai/jina-clip-v2": "jina-v2",
    "CLIPfa": "clipfa",
}

IDENTITY = ("name", "kind", "text_model", "vision_model", "trust_remote_code")


def configs() -> list[Path]:
    return sorted((ROOT / "configs").glob("hmc_*.yaml"))


def test_there_are_configs_to_check() -> None:
    assert configs(), "no hmc_*.yaml configs found; this test would pass vacuously"


@pytest.mark.parametrize("path", configs(), ids=lambda p: p.name)
def test_every_config_encoder_has_a_matching_preset(path: Path) -> None:
    # Through load_config, so a field the YAML omits is compared as the
    # default training would actually use, not as absent.
    encoder = load_config(path).encoder
    name = encoder.name

    assert name in PRESET_FOR_MODEL, (
        f"{path.name} trains on {name!r}, which has no retrieval preset, so it "
        "can have an AUROC and no alignment score"
    )
    preset = PRESETS[PRESET_FOR_MODEL[name]]

    for field in IDENTITY:
        expected = getattr(preset, field)
        actual = getattr(encoder, field)
        assert actual == expected, (
            f"{path.name} sets encoder.{field}={actual!r} but preset "
            f"{PRESET_FOR_MODEL[name]!r} uses {expected!r}; the alignment and "
            "AUROC axes would describe different models"
        )


def test_every_preset_is_reachable_from_the_map() -> None:
    """A preset nothing maps to is a model measured for alignment and never trained."""
    assert set(PRESET_FOR_MODEL.values()) == set(PRESETS)
