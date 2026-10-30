"""Train a classifier over cached encoder features.

python -m src.train --config configs/hmc_align.yaml [--seed 1]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, load_config
from .data.features import FeatureDataset
from .evaluate import evaluate
from .models import build_model
from .utils.device import resolve_device
from .utils.metrics import Metrics
from .utils.seed import set_seed


def build_loaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    directory = config.feature_dir
    # The encoder name is passed so a cache that records a different one is
    # refused rather than trained on. The path alone is not evidence: these
    # files are built on one machine and trained on on another.
    expect = config.encoder.name
    train_set = FeatureDataset(directory, config.data.train_split, expect)
    val_set = FeatureDataset(directory, config.data.val_split, expect)
    test_set = FeatureDataset(directory, config.data.test_split, expect)

    common = {
        "batch_size": config.train.batch_size,
        "num_workers": config.train.num_workers,
    }
    return (
        DataLoader(train_set, shuffle=True, drop_last=False, **common),
        DataLoader(val_set, shuffle=False, **common),
        DataLoader(test_set, shuffle=False, **common),
        train_set.feature_dim,
    )


def train(config: Config) -> dict[str, Metrics]:
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)

    train_loader, val_loader, test_loader, feature_dim = build_loaders(config)
    model = build_model(config.model, feature_dim).to(device)
    print(
        f"architecture={config.model.architecture} "
        f"fusion={config.model.fusion} "
        f"trainable params={model.trainable_parameters():,} "
        f"device={device}"
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    # Model selection is on validation AUROC; the test split is read once, at
    # the end, with the selected weights.
    best_auroc = -float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_val: Metrics | None = None

    for epoch in range(1, config.train.epochs + 1):
        model.train()
        running = 0.0
        for image_features, text_features, labels in tqdm(
            train_loader, desc=f"epoch {epoch}/{config.train.epochs}", leave=False
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(image_features.to(device), text_features.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            if config.train.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimizer.step()
            running += loss.item() * labels.size(0)

        train_loss = running / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device)
        print(f"epoch {epoch:>3} loss {train_loss:.4f} | val {val_metrics}")

        if val_metrics.auroc > best_auroc:
            best_auroc = val_metrics.auroc
            best_val = val_metrics
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training produced no checkpoint; was epochs set to 0?")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    print(f"best val {best_val}")
    print(f"test     {test_metrics}")

    assert best_val is not None
    return {"val": best_val, "test": test_metrics}


def result_dir(config: Config) -> Path:
    """Where this run's per-seed files live.

    Keyed by the encoder as well as the architecture and the fusion. Each of
    the three has already been learned the hard way:

    - fusion alone let memeclip, which also fuses by element-wise product,
      overwrite hateclipper's align results;
    - leaving the encoder out would do the same across encoders, and the
      encoder comparison is this project's contribution. CLIP's align run
      would land on jina's five seed files and summarise would average
      whichever happened to be there, with nothing in the output saying so.
    """
    return (
        config.output_dir
        / config.data.name
        / config.encoder_slug
        / config.model.architecture
        / config.model.fusion
    )


def write_result(config: Config, results: dict[str, Metrics]) -> Path:
    output = result_dir(config)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"seed{config.train.seed}.json"
    path.write_text(
        json.dumps(
            {
                "encoder": config.encoder.name,
                "architecture": config.model.architecture,
                "fusion": config.model.fusion,
                "seed": config.train.seed,
                "val": results["val"].as_dict(),
                "test": results["test"].as_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, help="override the config seed")
    parser.add_argument("--fusion", help="override the config fusion variant")
    parser.add_argument("--architecture", help="override the config architecture")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.train.seed = args.seed
    if args.fusion is not None:
        config.model.fusion = args.fusion
    if args.architecture is not None:
        config.model.architecture = args.architecture

    results = train(config)
    print(f"wrote {write_result(config, results)}")


if __name__ == "__main__":
    main()
