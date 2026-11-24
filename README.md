# Persian Hateful Meme Detection

Cross-modal hateful meme classification with CLIP features, extended to Persian.

This repository hosts the code for an ongoing study. It builds on
[Hate-CLIPper](https://arxiv.org/abs/2210.05916) (Kumar & Nandakumar, WOAH 2022)
as its base architecture and extends it in two directions: a stronger fusion /
training pipeline evaluated against published baselines, and a Persian evaluation
set that probes how these models behave on a language CLIP was never trained for.

## Status

The study is complete. Every cell in the results table runs fifteen seeds,
comparisons are corrected for multiplicity, and every number in the manuscript
is generated from the per-seed records committed under `runs/` rather than
typed. `scripts/` holds the code that produces each of them.

**The manuscript itself is not in this repository while it is under review.**
`paper/` holds the generator, the figures and the reference material, so the
document can be rebuilt from source with `bash paper/build.sh --all`, but the
built `.docx` files are withheld until the paper is accepted. They will be
added back, or replaced by a link to the published version, once that is
settled.

There is still no public Persian hateful-meme dataset, and this repository
reports no Persian classification result. What it does hold for Persian is the
collection protocol, the annotation guideline in both languages, and the
tooling: see `docs/collection-protocol.md` and `docs/annotation-guideline.fa.md`.
Help with annotation is welcome.

## Motivation

Hate-CLIPper reports that on TamilMemes its cross- and align-fusion variants gave
no advantage over plain concatenation, and attributes this to CLIP's text and
image spaces being misaligned for a language absent from its pretraining. Its
Limitations section asks for "more thorough ablation studies" on exactly this
point. Persian has the same problem and no public hateful-meme dataset at all,
which is the gap this work targets.

## Planned components

| Role | Source | Status |
| --- | --- | --- |
| Base model | Hate-CLIPper, align-fusion variant | implemented |
| Second baseline | [MemeCLIP](https://aclanthology.org/2024.emnlp-main.959) (EMNLP 2024) | implemented |
| Technical contribution | One of: multilingual text encoder, or retrieval-guided contrastive loss ([RGCL](https://arxiv.org/abs/2311.08110)) | undecided |
| Data contribution | Persian meme evaluation set, annotated under a [Multi3Hate](https://aclanthology.org/2025.naacl-long.490/)-style cross-cultural protocol | not started |

Only one technical contribution will be carried through to completion. A single
idea evaluated properly over multiple seeds is worth more than three partial ones.

### Baselines

Both architectures read the same cached features, so a comparison between them
changes the model and nothing else.

```bash
python -m src.train --config configs/hmc_align.yaml      # Hate-CLIPper
```

```bash
python -m src.train --config configs/hmc_memeclip.yaml   # MemeCLIP
```

They share more than the papers suggest: both fuse by element-wise product of
the projected features. MemeCLIP adds residual adapters, L2 normalisation
before the product, and a cosine classifier. Each is separately switchable
(`adapter_ratio: 0`, `normalise_before_fusion: false`) so an ablation can say
which one earns its place.

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

### Building the Persian set

`docs/collection-protocol.md` has the scope decision and the commands for a
collecting session; `docs/annotation-guideline.md` is the labelling protocol and
`docs/annotation-guideline.fa.md` is the Persian version annotators read.

```bash
python -m scripts.prepare_memes --images data/persian/raw --init-manifest
python -m scripts.prepare_memes --images data/persian/raw --dry-run
python -m scripts.prepare_memes --images data/persian/raw \
    --out data/persian/annotations.jsonl
python -m scripts.check_annotations --file data/persian/annotations.jsonl
```

Annotation itself happens in `tools/annotate.html`, a single local page that
loads nothing from the network. Running `prepare_memes` again after a later
collecting session keeps every id, transcription and label already in the file.

## Layout

```
configs/      experiment configs, one YAML per run
docs/         collection protocol, annotation guideline (en + fa), related work
scripts/      feature extraction, data checks, seed sweeps, meme preparation
src/
  encoders.py   pluggable vision-language encoders
  data/         dataset readers, the cached-feature dataset, collection tooling
  models/       fusion variants and the classification head
  train.py      training loop
  evaluate.py   evaluation and metric reporting
  utils/        seeding, metrics, annotator agreement
tests/        unit tests
tools/        annotate.html, the offline annotation page
```

## Licence

Code is released under the MIT Licence. The licence covers this repository's
code only, never any dataset it is pointed at.
