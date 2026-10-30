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
from src.data.features import (  # noqa: E402
    FeatureCache,
    clear_shards,
    load_shards,
    save_shard,
)
from src.data.hmc import HatefulMemesSplit  # noqa: E402
from src.encoders import Encoder, build_encoder, resolve_device  # noqa: E402


def encode_shard(
    encoder: Encoder,
    split: HatefulMemesSplit,
    begin: int,
    end: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_batches: list[torch.Tensor] = []
    text_batches: list[torch.Tensor] = []
    for start in tqdm(
        range(begin, end, batch_size),
        desc=f"{split.split} {begin}-{end}",
        leave=False,
    ):
        records = split.records[start : min(start + batch_size, end)]
        images = [Image.open(r.image_path).convert("RGB") for r in records]
        image_batches.append(encoder.encode_images(images))
        text_batches.append(encoder.encode_texts([r.text for r in records]))
        for image in images:
            image.close()
    return torch.cat(image_batches), torch.cat(text_batches)


def encode_split(
    encoder: Encoder,
    split: HatefulMemesSplit,
    batch_size: int,
    directory: Path,
    encoder_name: str,
    shard_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Encode a split in shards, resuming from whatever is already on disk.

    Hours of CPU encoding on a machine that restarts without notice used to
    cost everything: the cache was written once, after the whole split, so a
    restart at batch 15 of 133 left nothing. Now each shard lands as it is
    finished and a restart costs the shard in flight.
    """
    missing = split.missing_images()
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) referenced by {split.split}.jsonl are "
            f"missing, first: {missing[0]}"
        )

    done = load_shards(directory, split.split, encoder_name)
    encoded = sum(shard["count"] for shard in done)
    if encoded:
        print(
            f"{split.split}: resuming, {encoded}/{len(split)} already encoded "
            f"in {len(done)} shard(s)"
        )

    while encoded < len(split):
        end = min(encoded + shard_size, len(split))
        images, texts = encode_shard(encoder, split, encoded, end, batch_size)
        save_shard(directory, split.split, encoded, images, texts, encoder_name)
        encoded = end
        print(f"{split.split}: {encoded}/{len(split)}", flush=True)

    done = load_shards(directory, split.split, encoder_name)
    total = sum(shard["count"] for shard in done)
    if total != len(split):
        raise ValueError(
            f"{split.split}: shards hold {total} examples, the split has "
            f"{len(split)}. Delete the shards for this split and re-encode."
        )
    image_batches = [shard["image_features"] for shard in done]
    text_batches = [shard["text_features"] for shard in done]

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


def run(config: Config, overwrite: bool, shard_size: int) -> None:
    device = resolve_device(config.train.device)
    print(f"loading {config.encoder.name} ({config.encoder.kind}) on {device}")
    # build_encoder, not FrozenEncoder: the config's `kind` decides which
    # encoder class to use, and this script ignored it. CLIPfa ships as two
    # separately published towers, so its `name` is only a label and loading
    # that label as a model id fails with "CLIPfa is not a local folder and is
    # not a valid model identifier". scripts/eval_retrieval.py had always gone
    # through build_encoder, which is why CLIPfa could be measured for
    # retrieval but never encoded for classification.
    encoder = build_encoder(config.encoder, device)

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
            encoder,
            split,
            config.encoder.batch_size,
            config.feature_dir,
            config.encoder.name,
            shard_size,
        )
        cache.save(image_features, text_features, labels, uids, config.encoder.name)
        print(f"{name}: {len(uids)} examples -> {cache.path}")
        # The shards were scaffolding for the cache that now exists.
        removed = clear_shards(config.feature_dir, name)
        if removed:
            print(f"{name}: removed {removed} shard file(s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--overwrite", action="store_true", help="re-encode splits already cached"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=1000,
        help="examples per resumable shard (default 1000)",
    )
    args = parser.parse_args()
    if args.shard_size < 1:
        parser.error("--shard-size must be at least 1")
    run(load_config(args.config), args.overwrite, args.shard_size)


if __name__ == "__main__":
    main()
