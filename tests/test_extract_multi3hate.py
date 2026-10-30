"""Encoding Multi3Hate, and the two ways it could silently produce nonsense.

The dataset's value is that the memes are parallel, so the checks worth having
are about keeping them that way and about never letting an unlabelled meme
reach a metric. Nothing here loads a model: the encoder is replaced, so what is
tested is the script's own decisions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.extract_multi3hate as extract  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data.features import FeatureCache  # noqa: E402
from src.data.multi3hate import LANGUAGES, Multi3HateError  # noqa: E402

CONFIG = """
encoder:
  name: openai/clip-vit-base-patch32
  kind: clip
  batch_size: 4
data:
  name: hmc
train:
  device: cpu
"""
LABELS = {"US": 1, "DE": 0, "MX": 1, "CN": 0, "IN": 1}


class FakeEncoder:
    """Returns one row per input, so a shape error cannot hide behind a stub."""

    def __init__(self) -> None:
        self.images = 0
        self.texts = 0

    def encode_images(self, images):
        self.images += len(images)
        return torch.ones(len(images), 8)

    def encode_texts(self, texts):
        self.texts += len(texts)
        return torch.ones(len(texts), 8) * 2


def write(root: Path, language: str, ids, labels=None) -> None:
    (root / "img" / language).mkdir(parents=True, exist_ok=True)
    rows = []
    for uid in ids:
        (root / "img" / language / f"{uid}.jpg").write_bytes(b"x")
        rows.append(
            {
                "id": uid,
                "img": f"img/{language}/{uid}.jpg",
                "text": f"{language} {uid}",
                "template": "t",
                "labels": dict(LABELS if labels is None else labels),
            }
        )
    (root / f"{language}.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    config_path = tmp_path / "c.yaml"
    config_path.write_text(CONFIG, encoding="utf-8")
    config = load_config(config_path)

    root = tmp_path / "m3h"
    for language in LANGUAGES:
        write(root, language, ("0", "1", "2"))

    monkeypatch.setattr(extract, "ROOT", tmp_path / "features" / "multi3hate")
    monkeypatch.setattr(extract, "resolve_device", lambda _: torch.device("cpu"))
    fake = FakeEncoder()
    monkeypatch.setattr(extract, "build_encoder", lambda *_: fake)
    monkeypatch.setattr(
        extract, "Image", type("I", (), {"open": staticmethod(lambda p: _Img())})
    )
    return config, root, fake


class _Img:
    def convert(self, _mode):
        return self

    def close(self):
        pass


def test_every_language_is_cached_under_the_encoder(setup) -> None:
    config, root, fake = setup
    extract.run(config, root, overwrite=False, languages=list(LANGUAGES))

    directory = extract.feature_dir(config)
    assert directory.name == "openai__clip-vit-base-patch32"
    for language in LANGUAGES:
        cache = FeatureCache(directory, language)
        assert cache.exists()
    assert fake.images == 3 * len(LANGUAGES)
    assert fake.texts == 3 * len(LANGUAGES)


def test_the_cache_is_not_written_where_hateful_memes_is_looked_for(setup) -> None:
    """A Multi3Hate cache under the config's dataset would be found as HMC's."""
    config, root, _ = setup
    assert "multi3hate" in str(extract.feature_dir(config))
    assert config.data.name == "hmc"
    assert config.feature_dir != extract.feature_dir(config)


def test_each_language_gets_its_native_countrys_label(setup) -> None:
    config, root, _ = setup
    extract.run(config, root, overwrite=False, languages=["de", "hi"])
    directory = extract.feature_dir(config)
    # DE says 0, IN says 1 for every meme in the fixture.
    assert FeatureCache(directory, "de").load()["labels"].tolist() == [0, 0, 0]
    assert FeatureCache(directory, "hi").load()["labels"].tolist() == [1, 1, 1]


def test_a_ragged_copy_is_refused_before_any_encoding(setup) -> None:
    """Comparing languages over different memes is not a language comparison."""
    config, root, fake = setup
    write(root, "zh", ("0", "1", "2", "99"))
    with pytest.raises(Multi3HateError, match="same memes"):
        extract.run(config, root, overwrite=False, languages=list(LANGUAGES))
    assert fake.images == 0, "nothing should be encoded before the check"


def test_an_unlabelled_meme_is_refused_not_given_minus_one(setup) -> None:
    """-1 in a label column enters a metric quietly; the HMC cache uses it for
    deliberately unlabelled splits, and copying that habit here would mean
    scoring a meme nobody judged."""
    config, root, _ = setup
    write(root, "es", ("0", "1", "2"), labels={"US": 1, "DE": 0, "CN": 0, "IN": 1})
    with pytest.raises(Multi3HateError, match="no label from MX"):
        extract.run(config, root, overwrite=False, languages=["es"])


def test_a_missing_image_is_refused(setup) -> None:
    config, root, _ = setup
    for image in (root / "img" / "en").glob("*.jpg"):
        image.unlink()
    with pytest.raises(FileNotFoundError, match="download_multi3hate"):
        extract.run(config, root, overwrite=False, languages=["en"])


def test_a_cached_language_is_not_re_encoded(setup) -> None:
    config, root, fake = setup
    extract.run(config, root, overwrite=False, languages=["en"])
    first = fake.images
    extract.run(config, root, overwrite=False, languages=["en"])
    assert fake.images == first


def test_overwrite_re_encodes(setup) -> None:
    config, root, fake = setup
    extract.run(config, root, overwrite=False, languages=["en"])
    extract.run(config, root, overwrite=True, languages=["en"])
    assert fake.images == 6
