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

# Batch size per encoder, because they do not cost the same.
#
# jina-clip-v2 processes images at 512x512 — over five times the pixels of
# CLIP's 224 — so its activations are correspondingly larger. At batch 32 on
# a 15 GB CPU machine it reached 13.6 GB resident and the kernel's OOM killer
# took it; the run left no result and no error anyone could see, because the
# process was killed rather than failing. At batch 8 it peaks near 7.2 GB.
#
# Raise these on a GPU, where the limit is VRAM and the arithmetic is
# different.
batch_for() {
  case "$1" in
    jina-v2) echo 8 ;;
    *) echo 32 ;;
  esac
}

for encoder in clip-l14 jina-v2 clipfa; do
  batch="$(batch_for "$encoder")"
  for language in fa en; do
    echo "=== ${encoder} / ${language} (${N} images, batch ${batch}) ==="
    # Keep going if one encoder fails: a missing or gated model should not
    # cost the runs that would have succeeded.
    python -m scripts.eval_retrieval \
      --encoder "$encoder" \
      --language "$language" \
      --limit "$N" \
      --batch-size "$batch" \
      --out "$OUT" || echo "FAILED: ${encoder}/${language}"
  done
done

echo
echo "=== summary ==="
python -m scripts.summarise_retrieval --dir "$OUT"
