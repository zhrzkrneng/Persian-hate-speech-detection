# Paper draft

What is written, what is not, and what each missing piece is waiting for.

Markdown rather than LaTeX for now: the content is what needs reviewing, and a
diff of it is readable in a pull request. Converting to a venue's template is
mechanical and happens once the content settles.

## Written

| section | file | basis |
| --- | --- | --- |
| Method, experimental setup, reproducibility | `method.md` | the code — every number in it is computed from the repository |
| Results on HMC | `results.md` | `runs/`, with every comparison from `scripts/compare_cells.py` |
| Related work | `related-work.md` | reading, with five claims still ⚠️ unverified (below) |

## The result, in one paragraph

The encoder choice dominates: CLIP ViT-L/14 beats jina-clip-v2 by 1.9 to 2.9
AUROC on English, on all ten paired runs, *p* < 0.001. The benefit of align
fusion over concatenation **depends on the encoder** — it is +0.98 AUROC larger
on CLIP than on jina-clip-v2, *p* = 0.006 — and that interaction is the claim,
because neither fusion comparison reaches significance on its own. The encoder
gap is wider under align (+2.89) than under concat (+1.91), so the multilingual
encoder loses most where the fusion mechanism is supposed to contribute, which
is the mechanism Hate-CLIPper's Limitations section proposes for Tamil,
measured here on English. MemeCLIP's additions help over CLIP (+0.47) and hurt
over jina (−1.33): the sign flips with the encoder.

Bounding all of it: this pipeline reaches 82.09 where Hate-CLIPper reports
85.8, so the qualitative effect reproduces and the absolute level does not.
Only within-pipeline comparisons are claimed.

## Not written

**Abstract, introduction, conclusion.** These are the sections that state the
contribution, and the contribution is not complete: the Persian evaluation set
is the data contribution, and it does not exist yet. Writing an abstract around
the HMC result alone would describe a different, smaller paper than the one
being built. They are quick to write once the Persian numbers exist.

**The Persian results section.** Waiting on collection and annotation, which
need people. See `docs/collection-protocol.md`.

## Status of the experiments

| cell | seeds | test AUROC |
| --- | --- | --- |
| CLIP · Hate-CLIPper · concat · n=1024 | 5 | 81.37 ± 0.28 |
| CLIP · Hate-CLIPper · align · n=1024 | 5 | 82.09 ± 0.39 |
| CLIP · MemeCLIP · align · n=1024 | 5 | 82.56 ± 0.59 |
| jina · Hate-CLIPper · concat · n=1024 | 5 | 79.46 ± 0.27 |
| jina · Hate-CLIPper · align · n=1024 | 5 | 79.20 ± 0.52 |
| jina · MemeCLIP · align · n=1024 | 5 | 77.88 ± 0.51 |
| jina · Hate-CLIPper · concat · n=256 | 5 | 79.40 ± 0.66 |
| jina · Hate-CLIPper · align · n=256 | 5 | 79.35 ± 0.40 |
| jina · Hate-CLIPper · cross · n=256 | 0 | **not run** — 67.6M trainable parameters, hours per seed on four CPU cores. Needs a GPU. Reported as absent in `results.md` §5, not left blank. |
| CLIP · Hate-CLIPper · cross | 0 | same, and at n=1024 it would not fit a free GPU at all |

Retrieval numbers in `runs/retrieval/` were measured on a 1,000-image subset of
Crossmodal-3600. Enough to have chosen an encoder on evidence; the published
table should be rerun over all 3,600 on a GPU.

## Before submission

**Five claims still need a primary source.** `docs/related-work.md` marks each
⚠️. They cannot be checked from this session: `arxiv.org` and `aclanthology.org`
are outside the environment's allowed network domains.

| claim | what is needed |
| --- | --- |
| ISSUES's reported AUROC | the number from the paper; a secondary source gives ~0.86 and marks it unverified |
| RGCL's publication venue | the search did not confirm it |
| MUTE / ExMute figures, shared-task macro F1 | from the papers, not from search results |
| ParsHate's exact scope and size | not independently verified |
| "no public Persian multimodal hate speech dataset" | **hedge to "to the best of our knowledge"** — absence is not something a literature search establishes |

**One wording inconsistency to settle:** `related-work.md` describes the
retrieval measurement as "1000 images" and `results.md` §5 calls it a subset of
Crossmodal-3600. Both are true; the paper should use one phrasing.

**Two licence conditions must appear in the paper**, not only in the repository:
jina-clip-v2 is CC BY-NC 4.0, research use and not commercial, and it requires
`trust_remote_code=True`, which executes code downloaded from its model
repository.
