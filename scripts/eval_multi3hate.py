"""Read Multi3Hate's five languages with a head trained on English memes.

    python -m scripts.eval_multi3hate --config configs/hmc_align_clipb32.yaml --seed 0

This is the step that turns the project's claim into something testable on more
than two points. A head is trained on the Hateful Memes Challenge -- English --
and then applied, without any further training, to 300 parallel memes in
English, German, Spanish, Hindi and Mandarin. Because the memes are parallel,
the drop from English to another language is caused by the language, and
Crossmodal-3600 gives each encoder a measured alignment in that same language.
Four encoders over five languages is twenty (alignment, score) pairs where the
Hateful Memes table alone gives four.

Persian never appears here, and that is the point. It has an alignment score
and no labelled memes anywhere, so a relationship fitted over the languages
that do have labels makes Persian a prediction rather than a claim about data
nobody has.

The model is trained in this process rather than loaded from a checkpoint.
A run over cached features takes about two minutes, so re-training is cheaper
than keeping thirty checkpoints correct across two machines, and the Hateful
Memes metrics this script recomputes are written beside the zero-shot ones so
they can be compared with runs/hmc as a determinism check -- a drift there
would mean the two tables describe different models.

Accuracy is reported against the majority-class rate, never alone: Multi3Hate's
label balance differs by country, so an accuracy that looks respectable can be
below the baseline of answering "not hateful" every time.

Nothing from Multi3Hate is written outside runs/, and its features stay on this
machine: CC BY-NC-ND 4.0 forbids distributing adapted material, and an encoded
copy of its images is adapted material. A measurement is not an adaptation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.extract_multi3hate import feature_dir  # noqa: E402
from src.config import Config, load_config  # noqa: E402
from src.data.features import FeatureDataset  # noqa: E402
from src.data.multi3hate import LANGUAGES, NATIVE_COUNTRY  # noqa: E402
from src.evaluate import predict  # noqa: E402
from src.train import train_and_keep  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402


def result_path(config: Config, seed: int) -> Path:
    return (
        Path("runs/multi3hate")
        / config.encoder_slug
        / config.model.architecture
        / config.model.fusion
        / f"seed{seed}.json"
    )


def score_language(
    model: torch.nn.Module,
    directory: Path,
    language: str,
    encoder: str,
    batch_size: int,
    device: torch.device,
    expect_dim: int,
) -> dict:
    dataset = FeatureDataset(directory, language, encoder)
    if dataset.feature_dim != expect_dim:
        raise SystemExit(
            f"{language}: features are {dataset.feature_dim}-d but the head was "
            f"trained on {expect_dim}-d. The two caches came from different "
            "encoders; re-extract one of them."
        )

    labels, probabilities = predict(
        model, DataLoader(dataset, batch_size=batch_size, shuffle=False), device
    )
    metrics = compute_metrics(labels, probabilities)

    positives = int(labels.sum())
    count = len(labels)
    return {
        **metrics.as_dict(),
        "memes": count,
        "hateful": positives,
        # The baseline for accuracy: answering the more common label every
        # time. Multi3Hate's balance differs by country, so an accuracy above
        # 50% can still be below this.
        "majority_baseline": max(positives, count - positives) / count,
        # Visible so a head that collapsed to one class cannot look like a
        # language effect.
        "predicted_positive_rate": float((probabilities >= 0.5).mean()),
        "country": NATIVE_COUNTRY[language],
    }


def run(config: Config, seed: int, languages: list[str]) -> dict:
    config.train.seed = seed
    device = resolve_device(config.train.device)

    model, hmc = train_and_keep(config)

    directory = feature_dir(config)
    if not directory.exists():
        raise SystemExit(
            f"{directory} missing. Run scripts/extract_multi3hate.py --config "
            "<this config> first."
        )

    # The dimension the head was actually built for, taken from the split it
    # trained on rather than guessed from the model's parameters.
    trained_on = FeatureDataset(
        config.feature_dir, config.data.train_split, config.encoder.name
    ).feature_dim

    per_language = {}
    for language in languages:
        per_language[language] = score_language(
            model,
            directory,
            language,
            config.encoder.name,
            config.train.batch_size,
            device,
            trained_on,
        )
        row = per_language[language]
        print(
            f"{language}  AUROC {row['auroc'] * 100:6.2f}  "
            f"acc {row['accuracy'] * 100:6.2f} (majority "
            f"{row['majority_baseline'] * 100:5.2f})  "
            f"predicted positive {row['predicted_positive_rate'] * 100:5.1f}%"
        )

    return {
        "encoder": config.encoder.name,
        "architecture": config.model.architecture,
        "fusion": config.model.fusion,
        "seed": seed,
        # Recomputed here, for comparison with runs/hmc: a difference would
        # mean these zero-shot numbers came from a different model than the
        # table they sit beside.
        "hmc": {name: metrics.as_dict() for name, metrics in hmc.items()},
        "multi3hate": per_language,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    path = result_path(config, args.seed)
    if path.exists() and not args.overwrite:
        print(f"{path} exists; pass --overwrite to redo it")
        return

    payload = run(config, args.seed, args.languages)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
