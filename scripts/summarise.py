"""Aggregate per-seed result files into a mean +/- std table.

python -m scripts.summarise --config configs/hmc_align.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config  # noqa: E402
from src.utils.metrics import Metrics, summarise  # noqa: E402


def collect(directory: Path) -> tuple[list[Metrics], list[Metrics], list[int]]:
    files = sorted(directory.glob("seed*.json"))
    if not files:
        raise FileNotFoundError(f"no seed*.json under {directory}")

    val, test, seeds = [], [], []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        val.append(Metrics(**payload["val"]))
        test.append(Metrics(**payload["test"]))
        seeds.append(payload["seed"])
    return val, test, seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    directory = config.output_dir / config.data.name / config.model.fusion
    val, test, seeds = collect(directory)

    print(f"{config.data.name} / {config.model.fusion} / {config.encoder.name}")
    print(f"{len(seeds)} seeds: {seeds}\n")
    for name, runs in (("val", val), ("test", test)):
        summary = summarise(runs)
        parts = [
            f"{key} {mean * 100:.2f} +/- {std * 100:.2f}"
            for key, (mean, std) in summary.items()
        ]
        print(f"  {name:<5} " + " | ".join(parts))


if __name__ == "__main__":
    main()
