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

**Preparing the annotation file must be idempotent.** Collection happens in
sessions, so `scripts/prepare_memes.py` runs many times over a file that holds
annotation work. It rebuilt that file from scratch once: re-running it after a
single further meme had been collected destroyed nine labels and three hand
transcriptions, exited 0, and renumbered every id, so `fa_0000` afterwards
named a different meme than the annotators had discussed. Rows are matched by
image hash, ids are never reassigned, and a row whose image has gone missing
is reported rather than dropped.

**Provenance is recorded at collection time, not after.** The platform and the
collection date are not recoverable from a folder of images, and no one
reconstructs three hundred of them from memory. `data/persian/sources.csv`
holds them, `--init-manifest` extends it without touching a filled-in row, and
the scope decision behind all of it is in `docs/collection-protocol.md`.

## Rules about the code

**Encoding is cached, never repeated.** Frozen encoder features depend on the
encoder alone, not on the fusion or the classifier, so they are computed once
and written to disk. The feature cache is keyed by encoder name so two
encoders can never share one. The reference MemeCLIP implementation re-encodes
every batch of every epoch, one image at a time; that is the cost this design
exists to avoid.

**A cache file says what produced it.** Keying the directory by encoder is
protection only until a file is moved, copied or re-downloaded, and these
caches are built on one machine and trained on on another. `FeatureCache`
records the encoder, the split, the count, the feature width, the time and the
torch and transformers versions, and training refuses a cache whose recorded
encoder disagrees with the config. A cache from before this field is read with
a note, not refused: the first real results came from one.

**Split sizes matching is not a verified copy.** Three HMC mirrors were
checked against the published split sizes and passed. The images were never
counted: `neuralcatcher` and `luyuxin2026` each ship 9664 images while their
own JSONL files reference 12140 ids, so 2043 of the 10000 that
train/dev_seen/test_seen need are simply absent, and `snapshot_download`
reports success. Training on it would have succeeded on 7957 of 10000 memes
and produced a number comparable with nothing. `check_hmc` catches it because
it checks referenced images, not just row counts. The gap is fillable from
`limjiayi/hateful_memes_expanded`, whose files differ byte for byte -- PNG
compression -- but decode to identical pixels, which is what makes mixing the
two safe and is checked on a sample before any fetch.

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
- The cloud session's checkout arrives with **no `remote.origin.fetch`
  refspec**, so `origin/main` is frozen at whatever it was when the container
  started and never moves, however often you fetch. `git log origin/main` then
  shows a stale tip while `git ls-remote origin main` shows the real one — two
  merged pull requests read as unmerged that way. Fix it once per session:
  `git config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'`.

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
