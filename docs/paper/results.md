# Results on the Hateful Memes Challenge

Every number below is a mean ± sample standard deviation over five seeds (0–4)
on `test_seen`, selected on `dev_seen` AUROC and evaluated once. Each cell has
a committed config and five per-seed JSON records under `runs/`. Every
comparison comes from `python -m scripts.compare_cells --table`, so the prose
and the test statistic cannot drift apart.

## 1. The table

| encoder | architecture | fusion | test AUROC | accuracy | macro F1 |
| --- | --- | --- | --- | --- | --- |
| CLIP ViT-L/14 | Hate-CLIPper | concat | 81.37 ± 0.28 | 73.32 ± 0.97 | 72.90 ± 1.03 |
| CLIP ViT-L/14 | Hate-CLIPper | align | 82.09 ± 0.39 | 72.56 ± 0.99 | 71.86 ± 1.15 |
| CLIP ViT-L/14 | MemeCLIP | align | **82.56 ± 0.59** | 72.96 ± 0.63 | 72.40 ± 0.83 |
| jina-clip-v2 | Hate-CLIPper | concat | 79.46 ± 0.27 | 69.52 ± 1.35 | 68.41 ± 1.79 |
| jina-clip-v2 | Hate-CLIPper | align | 79.20 ± 0.52 | 70.70 ± 1.01 | 69.74 ± 1.40 |
| jina-clip-v2 | MemeCLIP | align | 77.88 ± 0.51 | 69.78 ± 1.33 | 68.68 ± 1.80 |

Fusion ablation, all three variants at one projection width, reported
separately because a cross run at $n=256$ is not comparable with an align run
at $n=1024$:

| encoder | fusion | $n$ | test AUROC | accuracy | macro F1 |
| --- | --- | --- | --- | --- | --- |
| jina-clip-v2 | concat | 256 | 79.40 ± 0.66 | 70.64 ± 0.96 | 69.91 ± 1.08 |
| jina-clip-v2 | align | 256 | 79.35 ± 0.40 | 70.02 ± 1.61 | 69.03 ± 1.93 |
| jina-clip-v2 | cross | 256 | not run — see §5 | | |

## 2. How the comparisons are made

Both variants in every comparison ran the same five seeds, so the differences
are taken **per seed** and tested against zero with a two-tailed paired
*t*-test. Two independent means with their own error bars would leave in the
seed noise the two variants share, and that noise is most of what separates
these cells.

At five seeds the test has four degrees of freedom and very little power. It
can confirm a large effect and cannot rule out a small one, so **n.s. here
means "five runs do not establish this", never "there is no difference".**

| comparison | Δ AUROC | sd | *t*(4) | *p* | |
| --- | --- | --- | --- | --- | --- |
| encoder: CLIP − jina, under align | **+2.89** | 0.49 | 13.24 | <0.001 | *** |
| encoder: CLIP − jina, under concat | **+1.91** | 0.28 | 15.06 | <0.001 | *** |
| **interaction: fusion benefit, CLIP − jina** | **+0.98** | 0.41 | 5.30 | **0.006** | ** |
| architecture: MemeCLIP − Hate-CLIPper, CLIP | +0.47 | 0.35 | 3.04 | 0.038 | * |
| architecture: MemeCLIP − Hate-CLIPper, jina | −1.33 | 0.98 | −3.02 | 0.039 | * |
| fusion: align − concat, CLIP | +0.72 | 0.62 | 2.58 | 0.061 | n.s. |
| fusion: align − concat, jina | −0.26 | 0.38 | −1.54 | 0.198 | n.s. |

## 3. What this establishes

**The encoder choice dominates everything else.** Swapping CLIP ViT-L/14 for
jina-clip-v2 costs 1.9 to 2.9 AUROC on English, on every one of the ten paired
runs, at *p* < 0.001 under both fusions. That is several times the size of any
architectural or fusion difference measured here.

**The benefit of align fusion depends on the encoder.** This is the central
result, and it is the *interaction* rather than either fusion comparison on its
own: align's advantage over concatenation is +0.98 AUROC larger on CLIP than on
jina-clip-v2 (*p* = 0.006). The two halves do not reach significance
separately — +0.72 (*p* = 0.061) on CLIP, −0.26 (*p* = 0.198) on jina — and the
double difference does, because it cancels the per-seed noise both halves
carry. Claiming "align beats concat on CLIP" from these five seeds would
overstate the evidence; claiming that *how much align helps depends on the
encoder* is what the data support.

Note the direction of the encoder gap: it is **larger under align (+2.89) than
under concat (+1.91)**. jina-clip-v2 loses most precisely where the fusion
mechanism is supposed to contribute. That is the mechanism Hate-CLIPper's
Limitations section proposes for Tamil — that for a language CLIP never saw,
the text and image spaces are not aligned, so there is no meaningful
interaction for the fusion to model — and here it is measured on *English*
data, with an encoder whose multilingual training is what differs.

**Which architecture wins also depends on the encoder.** MemeCLIP's three
additions — residual adapters, L2 normalisation before fusion, a cosine
classifier — help over CLIP (+0.47, *p* = 0.038) and hurt over jina-clip-v2
(−1.33, *p* = 0.039). The sign flips. An architecture's advantage reported on
CLIP features should not be assumed to carry to another encoder.

## 4. What this does not establish

**The absolute level does not match the published one.** Hate-CLIPper reports
85.8 AUROC on this split; the align cell here reaches 82.09, **3.7 points
short**. The qualitative effect reproduces — align ahead of concat on CLIP, in
the same direction and the same rough magnitude as published — but this
pipeline is weaker in absolute terms than the original. Every claim above is
therefore a comparison *within* this pipeline, where the encoder, the data, the
budget and the selection rule are held fixed and one variable moves. **Equality
with the published table is not claimed**, and no number here should be read
against another paper's.

**Validation and test disagree about the fusion direction, and on jina the
validation difference is the significant one.** On `dev_seen`, align − concat
is −0.23 (*p* = 0.368) on CLIP and **+0.90 (*p* = 0.002)** on jina-clip-v2 —
the opposite sign from test in both cases, and on jina it clears significance
in the direction test does not support.

This is not a borderline case to wave at; it is a reason the validation column
cannot be used to compare variants at all. The validation AUROC recorded per
run is the **maximum over 20 epochs that model selection picked**. A selected
maximum is biased upward, and nothing guarantees the bias is equal across
variants: align feeds $n$ inputs to the head where concat feeds $2n$, so the
two have different capacity to fit a 500-example split and can be expected to
peak on it differently. A difference in selected maxima is therefore a
difference in how reliably each variant touches a high point on `dev_seen`,
not a difference in what either generalises to.

`test_seen` is read once, with the selected weights, and is the only column
here that compares the variants. The jina validation result is reported
because hiding it would be worse, and because it is a concrete demonstration
of why selection numbers and comparison numbers must not be mixed — not
because it is evidence about fusion.

**Nothing here is state of the art.** RGCL reports 87.0 AUROC on this dataset.
The family studied here — frozen encoders, a few million trainable parameters,
a sweep that fits a free GPU — is chosen because it isolates the encoder
question cheaply, not because it competes.

## 5. Missing cells, named rather than left blank

**cross fusion.** At $n=256$ the first classifier layer holds 67.6M parameters,
and a single seed takes hours on the four CPU cores available, so the five
seeds were not run. At $n=1024$ the layer would hold about 1.07B parameters and
does not fit a free GPU at all. The variant is therefore absent from the
ablation, and the ablation is reported as two of three variants rather than as
a complete one.

**Retrieval numbers.** The Crossmodal-3600 numbers that decided the encoder
(`runs/retrieval/`) were measured on a 1,000-image subset. They were enough to
choose an encoder on evidence; the published table should be rerun over all
3,600 images on a GPU.

**The Persian evaluation set.** Collection is under way under the protocol in
`docs/collection-protocol.md`. No Persian number exists yet, and the
encoder-dependence result above is what motivates measuring one: it predicts
that on Persian the fusion machinery should contribute even less than it does
on English, and that a multilingual encoder's cost on English may be the price
of its being usable on Persian at all. That prediction is untested here.
