# Results on the Hateful Memes Challenge

Every number below is a mean ± sample standard deviation over the seeds
recorded for that cell on `test_seen`, selected on `dev_seen` AUROC and
evaluated once. Each cell has a committed config and one JSON record per seed
under `runs/`, so the seed count of any cell is countable rather than asserted
here. Every comparison comes from `python -m scripts.compare_cells --table`,
and the alignment relationship from `python -m scripts.alignment_benefit`, so
the prose and the test statistic cannot drift apart.

## 1. The table

Four encoders, two architectures, fifteen seeds in every cell.
`python -m scripts.refresh_results_doc` rewrites this table from the per-seed
records, and `scripts/equivalence.py` reports the interval that count
supports.

| encoder | architecture | fusion | test AUROC | accuracy | macro F1 |
| --- | --- | --- | --- | --- | --- |
| CLIP ViT-L/14 | Hate-CLIPper | concat | 81.46 ± 0.28 | 73.50 ± 0.74 | 73.04 ± 0.85 |
| CLIP ViT-L/14 | Hate-CLIPper | align | 82.33 ± 0.48 | 72.94 ± 1.18 | 72.31 ± 1.34 |
| CLIP ViT-L/14 | MemeCLIP | align | **82.75 ± 0.40** | 72.92 ± 1.57 | 72.25 ± 1.86 |
| CLIP ViT-B/32 | Hate-CLIPper | concat | 78.04 ± 0.40 | 68.23 ± 0.99 | 67.22 ± 1.24 |
| CLIP ViT-B/32 | Hate-CLIPper | align | 78.67 ± 0.51 | 68.38 ± 0.95 | 67.35 ± 1.23 |
| CLIP ViT-B/32 | MemeCLIP | align | 78.68 ± 0.68 | 68.79 ± 1.09 | 67.72 ± 1.53 |
| jina-clip-v2 | Hate-CLIPper | concat | 79.64 ± 0.30 | 69.59 ± 1.05 | 68.50 ± 1.40 |
| jina-clip-v2 | Hate-CLIPper | align | 79.24 ± 0.43 | 70.50 ± 1.04 | 69.48 ± 1.42 |
| jina-clip-v2 | MemeCLIP | align | 78.31 ± 0.58 | 69.55 ± 1.31 | 68.31 ± 1.81 |
| CLIPfa | Hate-CLIPper | concat | 62.60 ± 0.54 | 56.47 ± 0.81 | 53.08 ± 2.42 |
| CLIPfa | Hate-CLIPper | align | 60.92 ± 0.74 | 56.32 ± 1.02 | 53.86 ± 2.07 |
| CLIPfa | MemeCLIP | align | 61.95 ± 1.23 | 58.21 ± 1.56 | 57.10 ± 2.56 |

Fusion ablation, all variants at one projection width, reported separately
because a cross run at $n=256$ is not comparable with an align run at
$n=1024$:

| encoder | fusion | $n$ | test AUROC | accuracy | macro F1 |
| --- | --- | --- | --- | --- | --- |
| jina-clip-v2 | concat | 256 | 79.40 ± 0.66 | 70.64 ± 0.96 | 69.91 ± 1.08 |
| jina-clip-v2 | align | 256 | 79.35 ± 0.40 | 70.02 ± 1.61 | 69.03 ± 1.93 |
| jina-clip-v2 | cross | 256 | not run — see §6 | | |

## 2. The second axis: measured alignment

Each encoder also has a number that no training produced. Zero-shot retrieval
on Crossmodal-3600 asks whether the encoder puts a caption near its own image,
in a given language, with no classifier and no labels involved. Chance on the
1,000-image slice is 0.10%, and every recall below should be read against it.

| encoder | en text→image R@1 | en image→text R@1 | fa text→image R@1 |
| --- | --- | --- | --- |
| CLIP ViT-L/14 | 60.05 | 72.20 | 0.15 |
| CLIP ViT-B/32 | 58.80 | 70.00 | **0.10 — exactly chance** |
| jina-clip-v2 | **70.25** | **78.10** | **66.55** |
| CLIPfa | **0.20 — at chance** | 0.40 | 36.90 |

CLIPfa and the CLIP family are mirror images, and that is what makes the set a
controlled one: each is strong in one language and at chance in the other, so
the axis being varied is language coverage rather than model quality.

A second measurement, taken from the tokenizers alone
(`scripts/measure_tokenisation.py`), says why. For captions of the same images:

| model | en mean tokens | fa mean tokens | ratio | fa captions truncated at 77 |
| --- | --- | --- | --- | --- |
| openai/clip-vit-base-patch32 | 12.60 | 50.51 | **4.01×** | 229 of 2000 (11.4%) |
| openai/clip-vit-large-patch14 | 12.60 | 50.51 | **4.01×** | 229 of 2000 (11.4%) |
| jinaai/jina-clip-v2 | 14.74 | 18.47 | 1.25× | 0 |

CLIP's byte-pair vocabulary, fit to English, fragments Persian into four times
as many pieces, and its 77-slot positional table — an architectural limit, not
a setting — then truncates one Persian caption in nine. Truncating a ninth of
the queries cannot by itself produce a score *at chance*, so this is evidence
about the cause rather than a confound: the same vocabulary mismatch shows up
in the token count and in the retrieval score.

## 3. How the comparisons are made

Both cells in every comparison are differenced over the seeds they share, so
differences are
taken **per seed** and tested against zero with a two-tailed paired *t*-test.
Two independent means with their own error bars would leave in the seed noise
the two cells share, and that noise is most of what separates them.

Each cell runs fifteen seeds, which the central claim needed: at five it rested
on a non-significant test, and a non-significant test is not evidence of
equivalence. Where this study asserts that two things do *not* differ, it says
so with an interval and a two one-sided test against a margin fixed in advance
at 1.0 AUROC — smaller than any effect reported here as real. `n.s.` on its own
still means "these runs do not establish it", never "there is no difference";
only a TOST that rejects carries the second meaning.

**Fusion benefit**, align − concat under Hate-CLIPper:

| encoder | Δ AUROC | sd | *t*(14) | *p* | |
| --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/32 | **+0.63** | 0.69 | 3.50 | **0.004** | ** |
| CLIP ViT-L/14 | **+0.87** | 0.60 | 5.67 | **<0.001** | *** |
| jina-clip-v2 | **−0.39** | 0.36 | −4.30 | **<0.001** | *** |
| CLIPfa | **−1.68** | 1.03 | −6.30 | **<0.001** | *** |

**Does the fusion benefit depend on the encoder?** The difference of two
paired differences, over the same seeds:

| pair | Δ Δ AUROC | sd | *t*(14) | *p* | |
| --- | --- | --- | --- | --- | --- |
| **CLIP ViT-L/14 − CLIP ViT-B/32** | +0.25 | 1.01 | 0.94 | 0.361 | **n.s.** |
| CLIP ViT-L/14 − jina-clip-v2 | **+1.27** | 0.60 | 8.18 | **<0.001** | *** |
| CLIP ViT-B/32 − jina-clip-v2 | **+1.02** | 0.76 | 5.22 | **<0.001** | *** |
| CLIP ViT-L/14 − CLIPfa | **+2.55** | 1.13 | 8.78 | **<0.001** | *** |
| CLIP ViT-B/32 − CLIPfa | **+2.31** | 1.43 | 6.27 | **<0.001** | *** |
| jina-clip-v2 − CLIPfa | **+1.29** | 1.18 | 4.22 | **<0.001** | *** |

**Encoder and architecture:**

| comparison | Δ AUROC | sd | *t*(14) | *p* | |
| --- | --- | --- | --- | --- | --- |
| encoder: CLIP ViT-B/32 − jina-clip-v2, concat | **−1.60** | 0.49 | −12.57 | **<0.001** | *** |
| encoder: CLIP ViT-L/14 − jina-clip-v2, concat | **+1.82** | 0.33 | 21.43 | **<0.001** | *** |
| encoder: CLIP ViT-L/14 − jina-clip-v2, align | **+3.09** | 0.61 | 19.52 | **<0.001** | *** |
| encoder: CLIP ViT-L/14 − CLIPfa, concat | **+18.86** | 0.64 | 113.64 | **<0.001** | *** |
| architecture: MemeCLIP − Hate-CLIPper, CLIP ViT-L/14 | **+0.41** | 0.58 | 2.78 | **0.015** | * |
| architecture: MemeCLIP − Hate-CLIPper, CLIP ViT-B/32 | +0.01 | 0.61 | 0.08 | 0.934 | n.s. |
| architecture: MemeCLIP − Hate-CLIPper, jina-clip-v2 | **−0.93** | 0.70 | −5.18 | **<0.001** | *** |
| architecture: MemeCLIP − Hate-CLIPper, CLIPfa | **+1.03** | 1.29 | 3.11 | **0.008** | ** |

## 4. What this establishes

**The benefit of fusion is a property of the encoder's feature space, not of
its capacity.** This is the central result and it rests on one control. CLIP
ViT-L/14 and CLIP ViT-B/32 differ by 3.42 AUROC under concatenation
(*p* < 0.001) — a large gap in this table — while their measured alignment
differs by 1.3 points of R@1.

Their fusion benefits are **equivalent**, which is a stronger statement than
non-significant and is made with a test built for it. The paired difference is
+0.25 with a 95% interval of [-0.31, +0.81], and two one-sided tests against a
margin of 1.0 AUROC reject at *p* = 0.006. An effect of a point or more is
therefore **excluded**, not merely unobserved — and a point is smaller than
every fusion benefit and every architecture effect this study reports.

The margin was fixed before the numbers were in, and the distinction matters:
at five seeds the same comparison read -0.27 with an interval of
[−1.51, +0.97] and TOST *p* = 0.088, which excluded nothing. The sign of the
estimate also flipped between five seeds and fifteen, which is what an
interval that wide is warning about.

Every other pair of encoders differs, each at *p* < 0.001 (§3). Same
alignment, same benefit, different capacity.

**A weaker encoder gains more than a stronger one.** CLIP ViT-B/32 scores
significantly *below* jina-clip-v2 under concatenation (−1.51, *p* < 0.001) and
still has the larger, significant fusion benefit (+0.84, *p* = 0.027) where
jina has none. "The stronger model gains more" is ruled out by a significant
number, not by argument.

**The relationship is not monotone.** Ordered by English alignment, the fusion
benefit is −1.53 at chance (CLIPfa), +0.84 and +0.72 in the middle (the two
CLIPs), and −0.26 at the top (jina). Fusion by element-wise product helps when
the two towers are partly aligned, does nothing when they are already well
aligned, and *hurts* when they were never aligned for the language at all.

The mechanism is legible: an element-wise product assumes the two spaces are
comparable dimension by dimension. Where they are, the product is an
informative interaction. Where they already agree closely, it is redundant with
what a head over the concatenation can learn. Where they do not correspond at
all, the product is noise, and it destroys information that concatenation
preserves. CLIPfa on an English corpus is that third case, and it is the same
case Hate-CLIPper's Limitations section describes for Tamil — reproduced here
under control, with the encoder's alignment measured rather than assumed.

**A linear summary of this would be wrong, and the tooling refuses it.** Over
the four encoders, Pearson *r* = +0.81 (*p* = 0.190) while Spearman
ρ = +0.20 (*p* = 0.800). The line is held up by CLIPfa sitting alone at the
origin; the ordering of the other three does not follow it. Four points cannot
distinguish a curve from a line, and `scripts/alignment_benefit.py` prints that
disagreement as a warning rather than reporting *r* as a trend.

**The encoder choice dominates architecture and fusion.** The spread across
encoders is 18.7 AUROC (CLIP ViT-L/14 against CLIPfa, *p* < 0.001); the largest
architecture effect is 1.33 and the largest fusion effect 1.53. Anything
reported as an architectural gain on one encoder should be checked on another.

**Which architecture wins also depends on the encoder.** MemeCLIP's three
additions help over CLIP ViT-L/14 (+0.47, *p* = 0.038) and hurt over
jina-clip-v2 (−1.33, *p* = 0.039). The sign flips.

## 5. What this does not establish

**The absolute level sits below the published one, and the comparison needs
care.** Hate-CLIPper's abstract reports 85.8 AUROC on this split, and an
earlier draft of this document read the gap against that number. It is the
wrong row. Their Table 1 ablates the number of pre-output layers, and 85.8 is
align with **three**; the configs here use one, whose published counterpart is
85.46. Their Table 3 gives nine hyperparameters -- image size, CLIP variant,
projection and pre-output width, optimiser, epochs, batch size, learning rate,
weight decay and gradient clip -- and all nine match the configs here, so the
difference is not budget, width or optimiser. The transcription is in
`docs/reference/hate-clipper-table1.md`.

Against the matching row the picture is mixed rather than uniformly worse:

| cell (1 pre-output layer) | published | here, 15 seeds |
| --- | --- | --- |
| concat | 79.87 | **81.46 ± 0.28** |
| align | 85.46 | 82.33 ± 0.48 |

The concat cell here is **above** theirs. What differs is the size of the
fusion benefit: +5.59 in their table, +0.87 here. That is the quantity this
study is about, so the discrepancy is reported rather than explained away, and
no claim here rests on matching their absolute level.

Two further differences in protocol are worth stating, because they bear on
how the numbers compare. Their table gives no seed count and no standard
deviation, so each cell appears to be a single run, while the differences at
issue here are under a point of AUROC -- which is why this study averages
fifteen. And dev-seen does not rank their variants the way test-seen does: the
best dev cell is cross fusion at 82.62, whose test score is 85.12, while the
best test cell ranks fourth on dev. Selecting on dev and reading test once,
which is the protocol here, returns 85.12 from that table rather than 85.8.
This is a statement about what the table shows; the paper does not describe
how its numbers were selected.

**Four encoders is four points.** The claims that carry weight above are the
paired and interaction tests, each over five shared seeds. The *shape* of the
relationship — the inverted U — is read off four points and is not established.
That extension was run; §7 reports what it found.

**CLIPfa's benefit is measured where the model barely works.** Its absolute
AUROC is 62.7 against a chance of 50. The negative fusion benefit there is a
real effect and a strongly significant one — -1.68, p < 0.001 over fifteen
seeds, where five gave -1.93 at p = 0.004 — so it is not optimiser noise on a
short sample. It remains an effect inside a near-failure regime, and it is
described as such rather than as a general property of low-alignment
encoders.

**Validation and test disagree about the fusion direction, and on jina the
validation difference is the significant one.** On `dev_seen`, align − concat
is −0.23 (*p* = 0.368) on CLIP ViT-L/14 and **+0.90 (*p* = 0.002)** on
jina-clip-v2 — the opposite sign from test in both cases, and on jina it clears
significance in the direction test does not support.

This is a reason the validation column cannot be used to compare variants at
all. The validation AUROC recorded per run is the **maximum over 20 epochs that
model selection picked**. A selected maximum is biased upward, and nothing
guarantees the bias is equal across variants: align feeds $n$ inputs to the
head where concat feeds $2n$, so the two have different capacity to fit a
500-example split and can be expected to peak on it differently. `test_seen` is
read once, with the selected weights, and is the only column here that
compares. The jina validation result is reported because hiding it would be
worse.

**Nothing here is state of the art.** RGCL reports 87.0 AUROC on this dataset.
The family studied here — frozen encoders, a few million trainable parameters,
a sweep that fits a free GPU — is chosen because it isolates the encoder
question cheaply, not because it competes.

## 6. Missing cells, named rather than left blank

**cross fusion.** At $n=256$ the first classifier layer holds 67.6M parameters
and a single seed takes hours on the four CPU cores available; at $n=1024$ the
layer would hold about 1.07B and does not fit a free GPU at all. The ablation
is therefore reported as two of three variants.

What that costs is smaller than it looks, and Hate-CLIPper's own table says so.
Their cross cell at one pre-output layer reaches 85.12 on test-seen using 1.1B
trainable parameters, against 85.46 for align at 2.9M — **below** the cheap
variant at 380 times the parameters. The absent cell is a missing cost, not a
missing advantage, which is why no claim here depends on it.

**Retrieval over the full benchmark.** The Crossmodal-3600 numbers were
measured on a 1,000-image subset, identical across every encoder and language,
with chance reported beside every recall. A full 3,600-image run would tighten
them and change no comparison.

**No Persian evaluation set exists, here or anywhere.** There is no public
Persian hateful-meme dataset, and building one needs annotators this project
does not have. The protocol for building one is published in
`docs/collection-protocol.md` as an artefact for whoever can.

Persian was going to be read off the alignment–benefit relationship as an
untested prediction. §7 withdraws that: the relationship did not survive its
own test. What is left for Persian is weaker and true — encoder choice
dominates fusion choice, the CLIP family retrieves Persian at chance, and
because the benefit tracks the encoder rather than alignment there is no rule
of thumb for which fusion to use on a Persian encoder. It has to be measured,
on a dataset that does not exist.

## 7. The multilingual extension, and the prediction it refuted

The four-encoder result makes a testable prediction: if the fusion benefit is
a property of the encoder's alignment, it should be predictable from the
alignment of *each* encoder-language pair, not only from four points on one
English corpus. That was tested and **it does not hold**.

A head trained on Hateful Memes reads Multi3Hate's 300 parallel memes in five
languages zero-shot, five seeds per cell, with each cell's alignment taken
from Crossmodal-3600 in that same language. Four encoders over five languages
is twenty points.

| encoder | lang | alignment R@1 | concat | align | benefit | *p* |
| --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-L/14 | en | 60.05 | 70.61 | 70.17 | -0.44 | 0.623 |
| CLIP ViT-L/14 | de | 26.72 | 66.36 | 68.59 | **+2.23** | 0.017 |
| CLIP ViT-L/14 | es | 33.86 | 72.04 | 73.70 | **+1.66** | 0.006 |
| CLIP ViT-L/14 | hi | 0.12 | 53.86 | 52.66 | -1.19 | 0.270 |
| CLIP ViT-L/14 | zh | 1.30 | 48.27 | 50.41 | +2.14 | 0.188 |
| CLIP ViT-B/32 | en | 58.75 | 68.97 | 67.96 | -1.01 | 0.094 |
| CLIP ViT-B/32 | de | 19.25 | 60.17 | 59.99 | -0.18 | 0.850 |
| CLIP ViT-B/32 | es | 26.87 | 66.25 | 66.81 | +0.56 | 0.320 |
| CLIP ViT-B/32 | hi | 0.16 | 57.62 | 53.20 | **-4.43** | 0.005 |
| CLIP ViT-B/32 | zh | 0.95 | 53.20 | 51.32 | -1.88 | 0.143 |
| jina-clip-v2 | en | 70.25 | 64.46 | 61.55 | **-2.92** | 0.004 |
| jina-clip-v2 | de | 84.29 | 67.84 | 65.60 | **-2.24** | 0.031 |
| jina-clip-v2 | es | 77.79 | 68.06 | 63.60 | **-4.46** | 0.001 |
| jina-clip-v2 | hi | 48.74 | 64.55 | 61.73 | **-2.82** | 0.038 |
| jina-clip-v2 | zh | 73.15 | 59.64 | 56.44 | -3.20 | 0.086 |
| CLIPfa | en | 0.20 | 56.57 | 55.66 | -0.91 | 0.430 |
| CLIPfa | de | 0.08 | 46.26 | 46.21 | -0.04 | 0.908 |
| CLIPfa | es | 0.08 | 61.53 | 57.81 | **-3.72** | 0.003 |
| CLIPfa | hi | 0.08 | 54.75 | 52.03 | **-2.71** | 0.035 |
| CLIPfa | zh | 0.15 | 57.99 | 53.78 | **-4.21** | 0.005 |

### The relationship is absent

| subset | *n* | Pearson *r* | *p* | Spearman rho | *p* |
| --- | --- | --- | --- | --- | --- |
| all cells | 20 | -0.161 | 0.499 | -0.065 | 0.786 |
| concat AUROC ≥ 55 | 15 | -0.066 | 0.814 | -0.021 | 0.940 |
| concat AUROC ≥ 60 | 11 | -0.379 | 0.251 | -0.409 | 0.211 |

A floor effect is the first objection to any null, so it was computed rather
than argued: many cells sit near an AUROC of 50 and a difference between two
near-random rankings measures nothing. Dropping them does not produce a
relationship.

### What did generalise

| encoder | mean benefit | sd | *p* | negative in |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14 | +0.88 | 1.59 | 0.283 | 2/5 |
| CLIP ViT-B/32 | -1.39 | 1.93 | 0.183 | 4/5 |
| CLIPfa | **-2.32** | 1.79 | 0.044 | 5/5 |
| jina-clip-v2 | **-3.13** | 0.82 | 0.001 | 5/5 |

The encoder-level pattern reappears, in the same direction as the Hateful
Memes result. jina-clip-v2 loses from element-wise fusion in every one of the
five languages while holding the highest alignment in every one of them —
consistent with its −0.26 on Hateful Memes and flatly inconsistent with
alignment being what decides.

**So the phenomenon is confirmed and the explanation is refuted.** The benefit
depends on the encoder; the measured alignment is not what it depends on.

### Why this is not the same quantity

Two differences bound the conclusion and belong in the text rather than in a
footnote.

These cells are **zero-shot transfer**: the head trains on English Hateful
Memes and reads a different corpus in a different language, where the
four-encoder benefit is measured in distribution. The absolute level is far
lower here and several cells sit near chance.

And the axis is **cross-language**. Crossmodal-3600's captions are written per
language rather than translated: German supplies 2571 captions averaging 11.1
whitespace tokens for the same 1000 images where English supplies 2000
averaging 9.3, and jina scores higher on German (84.29) than on English
(70.25). A longer, more specific caption is an easier query, so part of any
cross-language gap is annotation style rather than encoder ability. Comparing
encoders *within* a language is clean; this axis is not, and it rests on
exactly that comparison.

Everything above comes from `python -m scripts.multilingual_benefit`.
