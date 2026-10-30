"""Reading a second corpus with a head trained on the first.

Nothing here trains a real model: `train_and_keep` is replaced by a stub whose
output is known, so what is tested is the script's own reasoning -- that a
head is never applied to features of the wrong width, that accuracy is always
accompanied by the baseline it has to beat, and that a collapsed classifier is
visible rather than reported as a language effect.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.eval_multi3hate as evaluation  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.features import FeatureCache  # noqa: E402
from src.utils.metrics import Metrics  # noqa: E402

CONFIG = """
encoder:
  name: openai/clip-vit-base-patch32
  kind: clip
model:
  architecture: hateclipper
  fusion: align
data:
  name: hmc
  root: {root}
  feature_root: {features}
train:
  device: cpu
  batch_size: 8
output_dir: {out}
"""


class Separable(torch.nn.Module):
    """Predicts from the first feature value, so the fixtures decide the score."""

    def forward(self, image_features, text_features):
        signal = image_features[:, 0]
        return torch.stack([-signal, signal], dim=-1)


class Collapsed(torch.nn.Module):
    """Always answers "hateful", which is the degenerate case worth seeing."""

    def forward(self, image_features, text_features):
        n = image_features.shape[0]
        return torch.stack([torch.full((n,), -5.0), torch.full((n,), 5.0)], dim=-1)


def cache(directory: Path, split: str, labels: list[int], dim: int = 8, flip=False):
    directory.mkdir(parents=True, exist_ok=True)
    signal = torch.tensor(
        [(1.0 if bool(v) != flip else -1.0) for v in labels]
    ).unsqueeze(1)
    features = torch.cat([signal, torch.zeros(len(labels), dim - 1)], dim=1)
    FeatureCache(directory, split).save(
        image_features=features,
        text_features=features.clone(),
        labels=torch.tensor(labels, dtype=torch.long),
        uids=[str(i) for i in range(len(labels))],
        encoder="openai/clip-vit-base-patch32",
    )


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    path = tmp_path / "c.yaml"
    path.write_text(
        CONFIG.format(
            root=tmp_path / "data",
            features=tmp_path / "features",
            out=tmp_path / "runs",
        ),
        encoding="utf-8",
    )
    config = load_config(path)

    for split in ("train", "dev_seen", "test_seen"):
        cache(config.feature_dir, split, [0, 1, 0, 1])

    m3h = tmp_path / "features" / "multi3hate" / config.encoder_slug
    monkeypatch.setattr(evaluation, "feature_dir", lambda _c, d=m3h: d)

    metrics = {
        "val": Metrics(auroc=0.8, accuracy=0.7, macro_f1=0.7),
        "test": Metrics(auroc=0.81, accuracy=0.71, macro_f1=0.71),
    }
    monkeypatch.setattr(
        evaluation, "train_and_keep", lambda _c: (Separable(), dict(metrics))
    )
    return config, m3h, metrics


def test_a_perfectly_separable_language_scores_one(world) -> None:
    config, m3h, _ = world
    cache(m3h, "en", [0, 1, 0, 1])
    payload = evaluation.run(config, seed=0, languages=["en"])
    row = payload["multi3hate"]["en"]
    assert row["auroc"] == pytest.approx(1.0)
    assert row["memes"] == 4
    assert row["country"] == "US"


def test_a_language_the_head_has_backwards_scores_zero(world) -> None:
    """The signal is inverted, so the ordering is exactly wrong -- a real
    failure mode for an encoder whose space does not transfer."""
    config, m3h, _ = world
    cache(m3h, "de", [0, 1, 0, 1], flip=True)
    row = evaluation.run(config, seed=0, languages=["de"])["multi3hate"]["de"]
    assert row["auroc"] == pytest.approx(0.0)


def test_accuracy_always_arrives_with_its_baseline(world) -> None:
    """On a 3-to-1 split, 75% accuracy is what answering one label gives."""
    config, m3h, _ = world
    cache(m3h, "es", [0, 0, 0, 1])
    row = evaluation.run(config, seed=0, languages=["es"])["multi3hate"]["es"]
    assert row["majority_baseline"] == pytest.approx(0.75)


def test_a_collapsed_head_is_visible(world, monkeypatch) -> None:
    config, m3h, metrics = world
    monkeypatch.setattr(
        evaluation, "train_and_keep", lambda _c: (Collapsed(), dict(metrics))
    )
    cache(m3h, "hi", [0, 1, 0, 1])
    row = evaluation.run(config, seed=0, languages=["hi"])["multi3hate"]["hi"]
    assert row["predicted_positive_rate"] == pytest.approx(1.0)
    assert row["accuracy"] == pytest.approx(0.5)


def test_features_of_the_wrong_width_are_refused(world) -> None:
    """The head was built for the training split's width; a mismatch means the
    two caches came from different encoders."""
    config, m3h, _ = world
    cache(m3h, "zh", [0, 1, 0, 1], dim=16)
    with pytest.raises(SystemExit, match="different encoders"):
        evaluation.run(config, seed=0, languages=["zh"])


def test_the_hmc_metrics_travel_with_the_zero_shot_ones(world) -> None:
    """So a drift against runs/hmc shows up as a disagreement, not a silence."""
    config, m3h, metrics = world
    cache(m3h, "en", [0, 1, 0, 1])
    payload = evaluation.run(config, seed=0, languages=["en"])
    assert payload["hmc"]["test"]["auroc"] == pytest.approx(metrics["test"].auroc)
    assert payload["seed"] == 0


def test_the_result_path_separates_encoder_architecture_and_fusion(world) -> None:
    """Keying on fusion alone already let one architecture overwrite another's
    results, and leaving the encoder out would do it across encoders."""
    config, _, _ = world
    path = evaluation.result_path(config, 3)
    assert path.parts[-4:] == (
        "openai__clip-vit-base-patch32",
        "hateclipper",
        "align",
        "seed3.json",
    )
    assert "multi3hate" in path.parts


def test_a_missing_multi3hate_cache_says_what_to_run(world, monkeypatch) -> None:
    config, _, _ = world
    monkeypatch.setattr(evaluation, "feature_dir", lambda _c: Path("/nowhere/at/all"))
    with pytest.raises(SystemExit, match="extract_multi3hate"):
        evaluation.run(config, seed=0, languages=["en"])


def test_the_written_file_is_json(world, tmp_path) -> None:
    config, m3h, _ = world
    cache(m3h, "en", [0, 1, 0, 1])
    payload = evaluation.run(config, seed=0, languages=["en"])
    path = tmp_path / "out.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8"))["fusion"] == "align"
