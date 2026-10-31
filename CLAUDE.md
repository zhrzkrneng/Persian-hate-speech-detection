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

**Crossmodal-3600's captions are written per language, not translated, so
alignment is not cleanly comparable across languages.** Each language has its
own annotators and its own style: German supplies 2571 captions averaging 11.1
whitespace tokens for the same 1000 images where English supplies 2000
averaging 9.3, and jina-clip-v2 scores *higher* on German (84.29) than on
English (70.25). A longer, more specific caption is an easier query, so part
of a cross-language gap is annotation style rather than encoder ability.
Comparing encoders *within* a language is clean; comparing languages is not,
and the twenty-point analysis rests on exactly that comparison, so it has to
be stated rather than assumed away. (Whitespace token counts are also
meaningless for Chinese, which does not space its words -- that measure read
1.0 and was my own instrument failing, not the data.)

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

**Not every cache may be backed up, so a backup names datasets and never a
directory.** The Hateful Memes feature caches live on a private Kaggle dataset,
which that corpus's click-through licence permits as storage. Multi3Hate is
CC BY-NC-ND 4.0 -- no distribution of adapted material, and an encoded copy of
its images is adapted material -- so its caches may not be copied off this
machine at all, not even privately. Both sit under `features/`, which is
gitignored, so the risk is not a commit: it is a backup command written to
sweep `features/`, which would have taken both. `features/multi3hate/` is
separate from `features/hmc/` for that reason. Reporting a measurement made
from those features is not distributing them.

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

**Long encoding runs write shards, because this container restarts without
notice.** The cloud session was restarted mid-session and killed a CLIP
ViT-L/14 extraction at batch 15 of 133, 37 minutes in, which left *nothing*:
the cache was written once, after a whole split, and `train` is one split of
8500. `extract_features` now writes a shard every `--shard-size` examples
(default 1000) and resumes from whatever is on disk, so a restart costs the
shard in flight. Shards record their encoder and must tile the split from 0
with no gap -- half a split from each of two encoders, or shards left over
from a different shard size, are refused rather than concatenated. A restart
also kills every background job, so check for one before assuming a job is
still running: `ps -o etime -p 1` shows the container's age.

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

It covers `scripts/` as well as `src/`, and did not at first: when
`scripts/compare_cells.py` began importing `scipy`, CI had never installed it
and it only arrived because scikit-learn happens to depend on it. A dependency
that reaches you transitively is not declared -- `scipy` is now named in
`requirements.txt` and in the workflow, and the walk covers both packages.

**A rule of thumb is not a test.** `mean - sd > 0` was used as a verdict on a
difference of +0.72 +/- 0.62 over five seeds and called it settled; the paired
t-test puts it at p = 0.061. Comparisons between cells go through
`scripts/compare_cells.py`, which reports the paired t-test, so a claim in the
paper and the statistic behind it cannot drift apart. At five seeds the test
has four degrees of freedom: "n.s." means these runs do not establish it, not
that there is no difference.

**The validation column cannot compare variants.** The validation AUROC each
run records is the maximum over 20 epochs that selection picked, so it is
biased upward, and the bias need not be equal across variants -- align feeds
the head n inputs where concat feeds 2n. On jina the validation difference is
significant (+0.90, p = 0.002) in the direction the test split does not
support. Selection is what the validation split is for; `test_seen` is the
only column that compares.

**The floor is Python 3.10, and this machine cannot check that.** CI runs 3.10
and 3.12; the cloud session runs 3.13. A 3.11-or-later standard library API
therefore passes `ruff` and the whole test suite locally and fails only in CI,
where it breaks *collection* rather than one test: `from datetime import UTC`
in `src/data/features.py` took out every test module that imports it, however
indirectly. `ruff`'s `target-version = "py310"` does not catch this -- its
pyupgrade rules suggest modernising up to the target, they do not flag
reaching past it. So prefer the older spelling (`datetime.now(timezone.utc)`)
and, for anything newer, check the version it was added in before pushing.

## Environment facts

- **Run one heavy job at a time: two is slower than serial, not twice as
  fast.** There are four cores, and torch defaults its thread pool to all of
  them, so two python jobs ask for eight threads on four cores. A training run
  that takes 1.5 minutes alone took 13 minutes beside a retrieval job -- an
  8.7x slowdown for 2x the work, with load average at 8 -- and the retrieval
  job was slower too. Measured on the same CLIPfa cells before and after
  stopping the other job. Either serialise, or set `OMP_NUM_THREADS` so the
  two together ask for four.
- **Three of this session's own measurement tools have returned false
  negatives, each costing real time.** `| head -3` truncated a listing into "the
  Kaggle token is gone" and bought two hours of CPU instead of GPU. `pgrep -q`
  is unsupported here and printed "PROCESS GONE" for a live process. `pgrep -f`
  matched nothing for a `python -m src.train` that `ps aux | grep` found
  immediately, which read as a dead sweep. Before reporting that something has
  stopped, confirm it a second way -- `ps aux | grep -E '[p]attern'` and the
  job's own output file -- because "it died" and "my check is wrong" look
  identical from one command.

- No GPU in the cloud session. A full Crossmodal-3600 run takes hours on CPU
  here and minutes on a free Colab or Kaggle GPU. Develop and verify here; run
  the final numbers there.
- `huggingface.co` had to be added to the session's allowed network domains.
  `google.github.io` is reachable by `curl` but not by the fetch tool.
- `transformers` is pinned below 5. jina-clip-v2's remote code imports
  `clip_loss`, which 5.x removed. Our own code handles both via
  `_as_features()`, but the comparison needs one version all three encoders
  load under.

- **A container restart reverts pip to the image's versions, so the pin has to
  be reapplied every session.** `requirements.txt` is a declaration, not a
  state: a restart brought back `transformers` 5.22.0 and jina then failed on
  four languages in a row with `cannot import name 'clip_loss'`. Check the
  installed version before any encoding run, not after:
  `python -c "import transformers; print(transformers.__version__)"`.

- **The transformers major version changes image features, so caches from
  either side of it cannot be mixed.** This was an assumption written down as
  one -- "for a fixed checkpoint the values should not depend on the version"
  -- until it was measured on 64 memes against the 5.22.0 caches:

  | encoder | image max abs diff | text | feature scale |
  | --- | --- | --- | --- |
  | CLIP ViT-L/14 | 7.1e-02 | 0 | 11.37 |
  | CLIP ViT-B/32 | 1.9e-02 | 0 | 8.11 |
  | CLIPfa | 1.3e-02 | 0 | 3.87 |

  Text is bit-identical; image is not. The cause is in the library's own
  warning: 5.x defaults to the fast torchvision image processor where 4.x uses
  the slow PIL one for these checkpoints. `FeatureDataset` and the retrieval
  image cache now refuse a cache from the other side of that boundary. Note
  what this does *not* touch: a fusion benefit is a difference between two
  cells that read the same cache, so preprocessing cancels in it. What moves is
  the absolute level.
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
