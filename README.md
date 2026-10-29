# Persian Hateful Meme Detection

Cross-modal hateful meme classification with CLIP features, extended to Persian.

This repository hosts the code for an ongoing study. It builds on
[Hate-CLIPper](https://arxiv.org/abs/2210.05916) (Kumar & Nandakumar, WOAH 2022)
as its base architecture and extends it in two directions: a stronger fusion /
training pipeline evaluated against published baselines, and a Persian evaluation
set that probes how these models behave on a language CLIP was never trained for.

## Status

Early. The pipeline runs end to end on the Hateful Memes Challenge (HMC) data,
but no result in this repository has been validated yet. Nothing here should be
cited.

## Motivation

Hate-CLIPper reports that on TamilMemes its cross- and align-fusion variants gave
no advantage over plain concatenation, and attributes this to CLIP's text and
image spaces being misaligned for a language absent from its pretraining. Its
Limitations section asks for "more thorough ablation studies" on exactly this
point. Persian has the same problem and no public hateful-meme dataset at all,
which is the gap this work targets.

## Planned components

| Role | Source |
| --- | --- |
| Base model | Hate-CLIPper, align-fusion variant |
| Second baseline | [MemeCLIP](https://aclanthology.org/2024.emnlp-main.959) (EMNLP 2024) |
| Technical contribution | One of: multilingual text encoder, or retrieval-guided contrastive loss ([RGCL](https://arxiv.org/abs/2311.08110)) |
| Data contribution | Persian meme evaluation set, annotated under a [Multi3Hate](https://aclanthology.org/2025.naacl-long.490/)-style cross-cultural protocol |

Only one technical contribution will be carried through to completion. A single
idea evaluated properly over multiple seeds is worth more than three partial ones.

## Datasets

**No dataset files are stored in this repository.**

The Hateful Memes Challenge data is distributed by Meta under a click-through
licence that does not permit redistribution. Download it yourself:

```bash
bash scripts/download_hmc.sh
```

The script does not fetch anything. It prints the official URL, states the
licence condition, and checks that the files you placed under `data/hmc/` have
the expected layout.

The Persian evaluation set is not yet collected. When it exists, this repository
will carry the annotation guideline and the labels, not the images.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

## Usage

Feature extraction runs once and caches to disk. Every later experiment reads the
cache, so a full training run costs minutes on a free Colab or Kaggle GPU.

```bash
python -m scripts.extract_features --config configs/hmc_align.yaml
```

```bash
python -m src.train --config configs/hmc_align.yaml
```

```bash
bash scripts/run_seeds.sh configs/hmc_align.yaml
```

The last command trains over five seeds and reports mean and standard deviation.
Differences between fusion variants in the published tables are small enough that
single-run numbers do not separate them.

## Layout

```
configs/      experiment configs, one YAML per run
scripts/      feature extraction, data checks, seed sweeps
src/
  encoders.py   pluggable vision-language encoders
  data/         dataset readers and the cached-feature dataset
  models/       fusion variants and the classification head
  train.py      training loop
  evaluate.py   evaluation and metric reporting
  utils/        seeding, metrics
tests/        unit tests for the fusion maths
```

## Licence

Code is released under the MIT Licence. The licence covers this repository's
code only, never any dataset it is pointed at.
