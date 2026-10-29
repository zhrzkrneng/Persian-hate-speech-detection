#!/usr/bin/env bash
# Train one config over several seeds and summarise.
#
#   bash scripts/run_seeds.sh configs/hmc_align.yaml [n_seeds]
#
# Reported gaps between fusion variants are under a point of AUROC, which a
# single run cannot resolve. Report mean and standard deviation, always.
set -euo pipefail

CONFIG="${1:?usage: run_seeds.sh CONFIG [N_SEEDS]}"
N="${2:-5}"

for ((seed = 0; seed < N; seed++)); do
  echo "=== seed $seed ==="
  python -m src.train --config "$CONFIG" --seed "$seed"
done

python -m scripts.summarise --config "$CONFIG"
