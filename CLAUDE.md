# Working notes for this repository

Conventions and hard-won facts, so they do not have to be rediscovered. Every
rule here exists because breaking it already cost something.

## What this project is

A study of cross-modal hateful meme classification, extended to Persian. The
output is a paper and a reproducible repository, not a deployed system. That
shapes the priorities: a claim has to be defensible, and a number has to be
reproducible, before either is interesting.

Hate-CLIPper's own Limitations section asks for exactly the ablation this
project runs — it reports that on Tamil its fusion variants gave no advantage
over plain concatenation, and attributes that to misaligned feature spaces for
a language CLIP never saw. Persian has the same problem and no public hateful
meme dataset at all.

## Rules about results

**Never report a single-seed number.** Published gaps between fusion variants
are under a point of AUROC, which one run cannot resolve. Use
`scripts/run_seeds.sh` and report mean ± standard deviation.

**Never report a recall without its chance level.** On a 50-image slice, 2%
looks like a result until you notice chance was 2%. `summarise_retrieval.py`
prints chance beside every row for this reason.

**Keep synthetic-data numbers out of anything that looks like a result.** The
smoke runs use features that are separable by construction and score near 100.
They prove the plumbing works and say nothing about any model.

**Select on validation, touch test once.** `src/train.py` keeps the checkpoint
with the best validation AUROC and evaluates the test split only at the end.

## Rules about data

**No dataset file is ever committed.** `/data/` and `/features/` are
gitignored, anchored with a leading slash so `src/data/` is not swept up with
them — that exact bug already silently dropped three source files from a
commit.

The Hateful Memes corpus is under a Meta click-through licence that forbids
redistribution. `scripts/download_hmc.sh` downloads nothing; it prints
instructions and validates a local copy. Crossmodal-3600 captions are CC BY 4.0
from Google; its images come from a separate tarball with their own
attributions, which `scripts/download_xm3600.py` fetches alongside.

**Read the official download instructions; do not guess URLs.** Guessing the
Open Images URL pattern for Crossmodal-3600 images produced 1746 spurious 404s.
The project page links a single tarball that takes twelve seconds.

## Rules about the code

**Encoding is cached, never repeated.** Frozen encoder features depend on the
encoder alone, not on the fusion or the classifier, so they are computed once
and written to disk. The feature cache is keyed by encoder name so two
encoders can never share one. The reference MemeCLIP implementation re-encodes
every batch of every epoch, one image at a time; that is the cost this design
exists to avoid.

**Result directories are keyed by architecture *and* fusion.** MemeCLIP also
fuses by element-wise product, so keying on fusion alone let one architecture
silently overwrite the other's results.

**Training must not import an encoder.** `src/train.py` reads cached tensors
and nothing else. Importing `src/encoders.py` for a helper pulled in
`transformers` and broke training on a machine that had torch without it.

**CI installs a subset of `requirements.txt`.** A third-party import added to
`src/` that is not in the workflow's install list fails only in CI.
`tests/test_imports.py` exists to make that failure name itself.

## Environment facts

- No GPU in the cloud session. A full Crossmodal-3600 run takes hours on CPU
  here and minutes on a free Colab or Kaggle GPU. Develop and verify here; run
  the final numbers there.
- `huggingface.co` had to be added to the session's allowed network domains.
  `google.github.io` is reachable by `curl` but not by the fetch tool.
- `transformers` is pinned below 5. jina-clip-v2's remote code imports
  `clip_loss`, which 5.x removed. Our own code handles both via
  `_as_features()`, but the comparison needs one version all three encoders
  load under.
- jina-clip-v2 needs `trust_remote_code=True`, which executes code downloaded
  from its model repo. It is also CC BY-NC 4.0 — fine for research, and it has
  to be stated in the paper.
- **Do not switch branches in the main checkout while a background job is
  running there.** It already pulled the running script out from under a job
  mid-flight. Use `git worktree` instead.

## Before pushing

```bash
ruff check . && ruff format --check . && python -m pytest tests/ -q
```

A push that turns CI red costs a cycle. For a fix to a failure, reproduce the
failure first, then show the same check passing — the tqdm CI break was
reproduced locally by blocking the import before the fix went up.

## Reading list

The papers this work sits against, and what each is for:

| Paper | Role |
| --- | --- |
| [Hate-CLIPper](https://arxiv.org/abs/2210.05916) (WOAH 2022) | base architecture; its Limitations section motivates the Persian work |
| [MemeCLIP](https://aclanthology.org/2024.emnlp-main.959) (EMNLP 2024) | second baseline; same fusion, three additions |
| [RGCL](https://arxiv.org/abs/2311.08110) | stronger method, 87.0 AUROC; cite, do not claim SOTA over it |
| [Multi3Hate](https://aclanthology.org/2025.naacl-long.490/) (NAACL 2025) | annotation protocol for the Persian set; cross-cultural disagreement angle |
| [Crossmodal-3600](https://aclanthology.org/2022.emnlp-main.45) (EMNLP 2022) | how encoder choice is decided on evidence |
