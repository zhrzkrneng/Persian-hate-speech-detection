#!/usr/bin/env bash
# Compare encoders on Crossmodal-3600 retrieval, Persian against an English
# control.
#
#   bash scripts/run_encoder_study.sh [N_IMAGES]
#
# Every encoder sees the same images, so the comparison is fair even on a
# subset. Chance R@1 is 100/N percent, so read any score against that: at
# N=1000 chance is 0.1%, and an encoder scoring near it does not represent
# that language at all.
#
# English is the control. An encoder that scores well on English and badly on
# Persian has a language problem; one that scores badly on both has a
# different problem, and the two must not be confused.
#
# On CPU this takes hours. On a GPU (free Colab or Kaggle) the full 3600 runs
# in minutes, which is how the final numbers should be produced.
set -uo pipefail

N="${1:-1000}"
OUT="runs/retrieval"

for encoder in clip-l14 jina-v2 clipfa; do
  for language in fa en; do
    echo "=== ${encoder} / ${language} (${N} images) ==="
    # Keep going if one encoder fails: a missing or gated model should not
    # cost the runs that would have succeeded.
    python -m scripts.eval_retrieval \
      --encoder "$encoder" \
      --language "$language" \
      --limit "$N" \
      --out "$OUT" || echo "FAILED: ${encoder}/${language}"
  done
done

echo
echo "=== summary ==="
python -m scripts.summarise_retrieval --dir "$OUT"
