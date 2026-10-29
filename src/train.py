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
from .encoders import resolve_device
from .evaluate import evaluate
from .models.classifier import MemeClassifier
from .utils.metrics import Metrics
from .utils.seed import set_seed


def build_loaders(config: Config) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    directory = config.feature_dir
    train_set = FeatureDataset(directory, config.data.train_split)
    val_set = FeatureDataset(directory, config.data.val_split)
    test_set = FeatureDataset(directory, config.data.test_split)

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
    model = MemeClassifier(config.model, feature_dim).to(device)
    print(
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("training produced no checkpoint; was epochs set to 0?")

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, device)
    print(f"best val {best_val}")
    print(f"test     {test_metrics}")

    assert best_val is not None
    return {"val": best_val, "test": test_metrics}


def write_result(config: Config, results: dict[str, Metrics]) -> Path:
    output = config.output_dir / config.data.name / config.model.fusion
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"seed{config.train.seed}.json"
    path.write_text(
        json.dumps(
            {
                "encoder": config.encoder.name,
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
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.train.seed = args.seed
    if args.fusion is not None:
        config.model.fusion = args.fusion

    results = train(config)
    print(f"wrote {write_result(config, results)}")


if __name__ == "__main__":
    main()
