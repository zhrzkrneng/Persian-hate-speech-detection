"""Encode every split once with the frozen encoder and cache the result.

    python -m scripts.extract_features --config configs/hmc_align.yaml

This is the only expensive step. Everything downstream reads the cache, so a
training run costs minutes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, load_config  # noqa: E402
from src.data.features import FeatureCache  # noqa: E402
from src.data.hmc import HatefulMemesSplit  # noqa: E402
from src.encoders import FrozenEncoder, resolve_device  # noqa: E402


def encode_split(
    encoder: FrozenEncoder, split: HatefulMemesSplit, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    missing = split.missing_images()
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) referenced by {split.split}.jsonl are "
            f"missing, first: {missing[0]}"
        )

    image_batches: list[torch.Tensor] = []
    text_batches: list[torch.Tensor] = []

    for start in tqdm(range(0, len(split), batch_size), desc=f"encoding {split.split}"):
        records = split.records[start : start + batch_size]
        images = [Image.open(r.image_path).convert("RGB") for r in records]
        image_batches.append(encoder.encode_images(images))
        text_batches.append(encoder.encode_texts([r.text for r in records]))
        for image in images:
            image.close()

    # Unlabelled splits (test_unseen as shipped) get -1 so the cache shape is
    # uniform; metrics must not be computed over them.
    labels = torch.tensor(
        [-1 if r.label is None else int(r.label) for r in split.records],
        dtype=torch.long,
    )
    return (
        torch.cat(image_batches),
        torch.cat(text_batches),
        labels,
        [r.uid for r in split.records],
    )


def run(config: Config, overwrite: bool) -> None:
    device = resolve_device(config.train.device)
    print(f"loading {config.encoder.name} on {device}")
    encoder = FrozenEncoder(config.encoder, device)

    splits = [
        config.data.train_split,
        config.data.val_split,
        config.data.test_split,
    ]
    for name in dict.fromkeys(splits):  # de-duplicate, keep order
        cache = FeatureCache(config.feature_dir, name)
        if cache.exists() and not overwrite:
            print(f"{name}: cached at {cache.path}, skipping")
            continue
        split = HatefulMemesSplit(config.data.root, name)
        image_features, text_features, labels, uids = encode_split(
            encoder, split, config.encoder.batch_size
        )
        cache.save(image_features, text_features, labels, uids, config.encoder.name)
        print(f"{name}: {len(uids)} examples -> {cache.path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--overwrite", action="store_true", help="re-encode splits already cached"
    )
    args = parser.parse_args()
    run(load_config(args.config), args.overwrite)


if __name__ == "__main__":
    main()
