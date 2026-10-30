#!/usr/bin/env bash
# The multilingual extension: twenty (alignment, score) pairs instead of four.
#
#   bash scripts/run_multilingual_study.sh
#
# Three stages, each resumable, each skipping work already on disk, so an
# interrupted run costs only the step in flight. This container restarts
# without notice, and the whole thing is hours on CPU.
#
#   1. retrieval alignment for each encoder in each Multi3Hate language
#   2. Multi3Hate feature extraction, one cache per encoder per language
#   3. zero-shot evaluation, five seeds per cell
#
# Order matters only in that stage 3 needs stage 2. Stage 1 is independent and
# comes first because it is the one whose cost the image cache changed: every
# language of Crossmodal-3600 captions the same images, so the image pass runs
# once per encoder and the four languages after the first pay for their
# captions alone.
#
# On a GPU this is minutes per encoder rather than tens of minutes. Develop and
# verify here; produce the final numbers where there is a GPU.
set -uo pipefail

# Multi3Hate's languages. Persian is deliberately absent: it has an alignment
# score and no labelled memes anywhere, so it is a prediction from the fitted
# relationship, not a row in the table.
LANGUAGES=(en de es hi zh)

# One config per encoder for the feature work; the fusion it names is read only
# for stage 3, where both fusions are needed per encoder.
declare -A CONFIG=(
  [clip-l14]=configs/hmc_align.yaml
  [clip-b32]=configs/hmc_align_clipb32.yaml
  [jina-v2]=configs/hmc_align_jina.yaml
  [clipfa]=configs/hmc_align_clipfa.yaml
)
# jina-clip-v2 works at 512x512, over five times CLIP's pixel count; at batch
# 32 it reached 13.6 GB resident here and was killed by the OOM killer with no
# error anyone could see. See scripts/run_encoder_study.sh.
declare -A BATCH=([jina-v2]=8)

echo "=== stage 1: retrieval alignment ==="
for encoder in clip-l14 clip-b32 jina-v2 clipfa; do
  batch="${BATCH[$encoder]:-32}"
  for language in "${LANGUAGES[@]}"; do
    if [ -f "runs/retrieval/${encoder}_${language}.json" ]; then
      echo "skip ${encoder}/${language} (done)"
      continue
    fi
    echo "--- ${encoder} / ${language} (batch ${batch})"
    python -m scripts.eval_retrieval \
      --encoder "$encoder" --language "$language" \
      --limit 1000 --batch-size "$batch" \
      || echo "FAILED retrieval ${encoder}/${language}"
  done
done

echo
echo "=== stage 2: Multi3Hate features ==="
for encoder in clip-l14 clip-b32 jina-v2 clipfa; do
  echo "--- ${encoder}"
  python -m scripts.extract_multi3hate --config "${CONFIG[$encoder]}" \
    || echo "FAILED features ${encoder}"
done

echo
echo "=== stage 3: zero-shot evaluation, 5 seeds ==="
# Both fusions per encoder: the quantity the paper reports is the difference
# between them, so a cell with only one is useless.
for encoder in clip-l14 clip-b32 jina-v2 clipfa; do
  for fusion in align concat; do
    config="${CONFIG[$encoder]/align/$fusion}"
    [ -f "$config" ] || { echo "no config $config"; continue; }
    for seed in 0 1 2 3 4; do
      python -m scripts.eval_multi3hate --config "$config" --seed "$seed" \
        || echo "FAILED eval ${encoder}/${fusion}/seed${seed}"
    done
  done
done

echo
echo "=== relationship ==="
python -m scripts.alignment_benefit --language en
