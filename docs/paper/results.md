# Results on the Hateful Memes Challenge

Every number below is a mean ± sample standard deviation over five seeds (0–4)
on `test_seen`, selected on `dev_seen` AUROC and evaluated once. Each cell has
a committed config and five per-seed JSON records under `runs/`. Every
comparison comes from `python -m scripts.compare_cells --table`, and the
alignment relationship from `python -m scripts.alignment_benefit`, so the prose
and the test statistic cannot drift apart.

## 1. The table

Four encoders, two architectures, five seeds: sixty runs.

| encoder | architecture | fusion | test AUROC | accuracy | macro F1 |
| --- | --- | --- | --- | --- | --- |
| CLIP ViT-L/14 | Hate-CLIPper | concat | 81.37 ± 0.28 | 73.32 ± 0.97 | 72.90 ± 1.03 |
| CLIP ViT-L/14 | Hate-CLIPper | align | 82.09 ± 0.39 | 72.56 ± 0.99 | 71.86 ± 1.15 |
| CLIP ViT-L/14 | MemeCLIP | align | **82.56 ± 0.59** | 72.96 ± 0.63 | 72.40 ± 0.83 |
| CLIP ViT-B/32 | Hate-CLIPper | concat | 77.95 ± 0.29 | 68.26 ± 0.90 | 67.23 ± 1.16 |
| CLIP ViT-B/32 | Hate-CLIPper | align | 78.79 ± 0.51 | 69.10 ± 0.69 | 68.31 ± 0.85 |
| CLIP ViT-B/32 | MemeCLIP | align | 78.96 ± 0.35 | 68.96 ± 0.87 | 67.97 ± 1.19 |
| jina-clip-v2 | Hate-CLIPper | concat | 79.46 ± 0.27 | 69.52 ± 1.35 | 68.41 ± 1.79 |
| jina-clip-v2 | Hate-CLIPper | align | 79.20 ± 0.52 | 70.70 ± 1.01 | 69.74 ± 1.40 |
| jina-clip-v2 | MemeCLIP | align | 77.88 ± 0.51 | 69.78 ± 1.33 | 68.68 ± 1.80 |
| CLIPfa | Hate-CLIPper | concat | 62.64 ± 0.57 | 57.02 ± 1.11 | 54.38 ± 2.34 |
| CLIPfa | Hate-CLIPper | align | 61.10 ± 1.07 | 56.42 ± 0.98 | 54.13 ± 0.96 |
| CLIPfa | MemeCLIP | align | 61.89 ± 1.54 | 58.44 ± 1.87 | 57.72 ± 1.92 |

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

Both cells in every comparison ran the same five seeds, so differences are
taken **per seed** and tested against zero with a two-tailed paired *t*-test.
Two independent means with their own error bars would leave in the seed noise
the two cells share, and that noise is most of what separates them.

At five seeds the test has four degrees of freedom and very little power. It
can confirm a large effect and cannot rule out a small one, so **n.s. here
means "five runs do not establish this", never "there is no difference".**

**Fusion benefit**, align − concat under Hate-CLIPper:

| encoder | Δ AUROC | sd | *t*(4) | *p* | |
| --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/32 | **+0.84** | 0.55 | 3.43 | **0.027** | * |
| CLIP ViT-L/14 | +0.72 | 0.62 | 2.58 | 0.061 | n.s. |
| jina-clip-v2 | −0.26 | 0.38 | −1.54 | 0.198 | n.s. |
| CLIPfa | **−1.53** | 0.99 | −3.46 | **0.026** | * |

**Does the fusion benefit depend on the encoder?** The difference of two
paired differences, over the same seeds:

| pair | Δ Δ AUROC | sd | *t*(4) | *p* | |
| --- | --- | --- | --- | --- | --- |
| **CLIP ViT-L/14 − CLIP ViT-B/32** | **−0.12** | 0.90 | −0.29 | **0.785** | **n.s.** |
| CLIP ViT-L/14 − jina-clip-v2 | +0.98 | 0.41 | 5.30 | 0.006 | ** |
| CLIP ViT-B/32 − jina-clip-v2 | +1.10 | 0.85 | 2.89 | 0.045 | * |
| CLIP ViT-L/14 − CLIPfa | +2.25 | 1.56 | 3.22 | 0.032 | * |
| CLIP ViT-B/32 − CLIPfa | +2.37 | 1.06 | 5.00 | 0.008 | ** |
| jina-clip-v2 − CLIPfa | +1.27 | 1.26 | 2.26 | 0.087 | n.s. |

**Encoder and architecture:**

| comparison | Δ AUROC | sd | *t*(4) | *p* | |
| --- | --- | --- | --- | --- | --- |
| encoder: CLIP ViT-B/32 − jina-clip-v2, concat | **−1.51** | 0.12 | −28.23 | <0.001 | *** |
| encoder: CLIP ViT-L/14 − jina-clip-v2, concat | +1.91 | 0.28 | 15.06 | <0.001 | *** |
| encoder: CLIP ViT-L/14 − jina-clip-v2, align | +2.89 | 0.49 | 13.24 | <0.001 | *** |
| encoder: CLIP ViT-L/14 − CLIPfa, concat | +18.73 | 0.59 | 71.13 | <0.001 | *** |
| architecture: MemeCLIP − Hate-CLIPper, CLIP ViT-L/14 | +0.47 | 0.35 | 3.04 | 0.038 | * |
| architecture: MemeCLIP − Hate-CLIPper, CLIP ViT-B/32 | +0.17 | 0.23 | 1.58 | 0.189 | n.s. |
| architecture: MemeCLIP − Hate-CLIPper, jina-clip-v2 | −1.33 | 0.98 | −3.02 | 0.039 | * |
| architecture: MemeCLIP − Hate-CLIPper, CLIPfa | +0.78 | 1.75 | 1.00 | 0.373 | n.s. |

## 4. What this establishes

**The benefit of fusion is a property of the encoder's feature space, not of
its capacity.** This is the central result and it rests on one control. CLIP
ViT-L/14 and CLIP ViT-B/32 differ by 3.4 AUROC in absolute terms — a large gap
in this table — and their measured alignment differs by 1.3 points of R@1.
Their fusion benefits are statistically indistinguishable (−0.12, *p* = 0.785).
Both differ from jina-clip-v2 (*p* = 0.006, *p* = 0.045) and from CLIPfa
(*p* = 0.032, *p* = 0.008), whose alignments are 10 points higher and 59 points
lower respectively. Same alignment, same benefit, different capacity.

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

**The absolute level does not match the published one.** Hate-CLIPper reports
85.8 AUROC on this split; the align cell here reaches 82.09, **3.7 points
short**. The qualitative effect reproduces, but this pipeline is weaker in
absolute terms than the original. Every claim above is a comparison *within*
this pipeline, where everything but one variable is held fixed. **Equality with
the published table is not claimed**, and no number here should be read against
another paper's.

**Four encoders is four points.** The claims that carry weight above are the
paired and interaction tests, each over five shared seeds. The *shape* of the
relationship — the inverted U — is read off four points and is not established.
The multilingual extension exists to turn four points into twenty.

**CLIPfa's benefit is measured where the model barely works.** Its absolute
AUROC is 62.6 against a chance of 50. The negative fusion benefit there is a
real, significant effect, but it is an effect inside a near-failure regime, and
it should be described as such rather than as a general property of
low-alignment encoders.

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
and a single seed takes hours on the four CPU cores available, so five seeds
were not run. At $n=1024$ the layer would hold about 1.07B parameters and does
not fit a free GPU at all. The ablation is reported as two of three variants.

**Retrieval over the full benchmark.** The Crossmodal-3600 numbers were
measured on a 1,000-image subset, identical across every encoder and language,
with chance reported beside every recall. A full 3,600-image run would tighten
them and change no comparison.

**No Persian evaluation set exists, here or anywhere.** There is no public
Persian hateful-meme dataset, and building one needs annotators this project
does not have. The protocol for building one is published in
`docs/collection-protocol.md` as an artefact for whoever can.

This is why Persian appears as a *prediction* rather than a result. It has a
measured alignment on Crossmodal-3600 for every encoder in the table and no
labels anywhere. The multilingual extension fits the alignment–benefit
relationship over five languages that do have labels, and Persian is then read
off that relationship — stated as an untested prediction, and labelled as one
wherever it appears.
