# Method and experimental setup

Draft sections. These describe what the code does, so every number and claim
here is checkable against the repository rather than against a memory of it,
and none of it depends on experiments still running. The results sections are
deliberately absent: see `docs/paper/README.md` for what is still needed and
why writing them now would be premature.

## 1. Task

A meme is an image with text rendered onto it. Given the image and its text,
predict whether the meme is hateful. The two modalities have to be read
together: the task is constructed so that neither alone is sufficient, which is
what the *benign confounders* in the Hateful Memes Challenge enforce — for a
hateful meme, the corpus also contains a minimally edited image or text that
flips the label (Kiela et al., 2020).

## 2. Architecture

### 2.1 Frozen encoders

A vision-language encoder maps an image and a text into a shared space, giving
$f_i, f_t \in \mathbb{R}^{d}$. Both towers stay frozen: no encoder parameter is
updated at any point. This is the setting Hate-CLIPper studies, and it is what
makes the comparison here affordable — features depend on the encoder alone, so
they are computed once and cached, and every later experiment reads tensors
from disk.

Two encoders are compared, chosen on measured evidence rather than on model
cards (§4.1):

| encoder | $d$ | licence |
| --- | --- | --- |
| OpenAI CLIP ViT-L/14 | 768 | MIT |
| jina-clip-v2 | 1024 | CC BY-NC 4.0 |

jina-clip-v2's licence permits research use and not commercial use, and it
requires `trust_remote_code=True`, which executes code downloaded from its
model repository. Both conditions are stated here because both constrain reuse
of this work.

### 2.2 Projection and fusion

Each modality is projected to a common width $n$ by its own linear map,
$p_i = W_i f_i$ and $p_t = W_t f_t$, and the two are combined by one of three
operations on the *feature interaction matrix* $p_i \otimes p_t$:

| fusion | definition | width |
| --- | --- | --- |
| concat | $[p_i ; p_t]$ | $2n$ |
| align | $\mathrm{diag}(p_i \otimes p_t) = p_i \odot p_t$ | $n$ |
| cross | $\mathrm{flatten}(p_i \otimes p_t)$ | $n^2$ |

**concat** is the control: no explicit cross-modal interaction, the head has to
discover any. **align** keeps the diagonal of the interaction matrix. Its
elements sum to $\langle p_i, p_t \rangle$, so the vector is a per-dimension
view of how far the two modalities agree, at $n$ features rather than $n^2$.
**cross** keeps the whole matrix.

The widths are the whole practical story of the comparison. At $n = 1024$,
cross fusion puts $n^2 \times$ `pre_output_dim` $\approx 1.07 \times 10^9$
parameters in the first layer of the head alone, which does not fit the free
GPUs this work is designed to run on; align puts $1024 \times 1024$ there.
Measured trainable parameter counts, over frozen encoders:

| encoder | architecture | fusion | $n$ | trainable |
| --- | --- | --- | --- | --- |
| CLIP ViT-L/14 | Hate-CLIPper | align | 1024 | 2,626,562 |
| CLIP ViT-L/14 | Hate-CLIPper | concat | 1024 | 3,675,138 |
| CLIP ViT-L/14 | MemeCLIP | align | 1024 | 3,675,136 |
| CLIP ViT-L/14 | Hate-CLIPper | cross | 256 | 67,505,666 |
| jina-clip-v2 | Hate-CLIPper | align | 1024 | 3,150,850 |
| jina-clip-v2 | Hate-CLIPper | concat | 1024 | 4,199,426 |
| jina-clip-v2 | MemeCLIP | align | 1024 | 4,199,424 |
| jina-clip-v2 | Hate-CLIPper | cross | 256 | 67,636,738 |

A cross-fusion run at $n = 256$ is not comparable with an align run at
$n = 1024$, so the fusion ablation (§4.3) holds $n$ fixed at 256 across all
three variants and is reported separately from the main table, never merged
into it.

### 2.3 Classification head

The fused vector goes through `num_pre_output_layers` blocks of
linear → ReLU → dropout, each of width `pre_output_dim`, and a final linear
layer to two logits. Trained with cross-entropy.

### 2.4 Second baseline: MemeCLIP

MemeCLIP (Shah et al., EMNLP 2024) is included as a second architecture over
the same cached features, so the comparison between the two changes the model
and nothing else. It adds three things to the same fusion:

1. **Residual adapters** on each projected feature,
   $p \leftarrow \alpha \cdot \mathrm{Adapter}(p) + (1 - \alpha) \cdot p$, with
   a bottleneck of width $n / r$.
2. **L2 normalisation** of each projection before fusion.
3. **A cosine classifier**: logits are scaled cosine similarities to class
   vectors rather than a plain linear layer, optionally initialised from the
   encoder's text embeddings of the class names.

One point is worth stating plainly, because the two papers present their
methods very differently. **MemeCLIP's fusion is the same operation as
Hate-CLIPper's align variant.** Its released implementation combines the two
modalities with `torch.mul`, an element-wise product, which is
$\mathrm{diag}(p_i \otimes p_t)$ written another way. The difference between
the two architectures here is therefore the adapters, the normalisation and the
cosine classifier — not the fusion.

## 3. Data

### 3.1 Hateful Memes Challenge

| split | memes | labelled | hateful |
| --- | --- | --- | --- |
| train | 8,500 | yes | 3,019 (35.5%) |
| dev_seen | 500 | yes | 247 (49.4%) |
| dev_unseen | 540 | yes | 200 (37.0%) |
| test_seen | 1,000 | yes | 490 (49.0%) |
| test_unseen | 2,000 | yes | 750 (37.5%) |

Training uses `train`, model selection uses `dev_seen`, and `test_seen` is read
once at the end. `test_seen` is the split Hate-CLIPper reports on, which is
what makes the numbers here comparable with its table.

The corpus is distributed by Meta under a click-through licence that forbids
redistribution, so it is neither committed to the repository nor downloaded
automatically. Obtaining it is left to the reader, and **a copy is verified
before it is used**: `scripts/check_hmc.py` checks every split's size against
the published counts, the labels, the referenced images, and train-to-evaluation
leakage.

That check is not a formality. The two most widely mirrored copies on the
Hugging Face Hub each ship 9,664 images while their own metadata references
12,140 ids, so 2,043 of the 10,000 images the three splits above need are
simply absent — and the download tool reports success. Training on such a copy
would have succeeded on 7,957 of 10,000 memes and produced a number comparable
with nothing. The experiments here use a copy completed from a third mirror,
after confirming on a sample of 40 overlapping images that the two mirrors
differ only in PNG encoding and decode to **identical pixels**. Those details
belong in a reproducibility statement rather than a footnote, because the
failure is silent.

### 3.2 Persian evaluation set

A small Persian meme set is being built as this work's data contribution,
annotated under a protocol following Multi3Hate (Bui et al., NAACL 2025): the
UN definition of hate speech, three labels with an explicit abstention,
per-annotator labels kept rather than only the majority, at least three
annotators per meme, and agreement reported as Krippendorff's $\alpha$.
`docs/annotation-guideline.md` is the full protocol and
`docs/collection-protocol.md` the scope decision and procedure. No meme image
will be released; labels, per-annotator labels, image hashes and source URLs
will be.

## 4. Experimental setup

### 4.1 Choosing the encoder on evidence

Encoder choice is decided by zero-shot retrieval on Crossmodal-3600 (Thapliyal
et al., EMNLP 2022), which provides human-authored captions in 36 languages
including Persian, rather than by what a model card claims about language
coverage. Recall at 1, 5 and 10 is reported in both directions, **each beside
its chance level**, because on a small image pool a recall figure is
meaningless without it.

### 4.2 Main comparison

Two encoders × {Hate-CLIPper align, Hate-CLIPper concat, MemeCLIP}, all over
the same cached features, at $n = 1024$.

### 4.3 Fusion ablation

{concat, align, cross} at $n = 256$, where all three fit, reported as its own
group.

### 4.4 Training

Hyperparameters follow Hate-CLIPper's Table 3 so that the numbers sit on the
same axis as its own:

| | |
| --- | --- |
| optimiser | AdamW |
| learning rate | $10^{-4}$ |
| weight decay | $10^{-4}$ |
| gradient clipping | 0.1 |
| batch size | 64 |
| epochs | 20 |
| dropout | 0.1 |
| pre-output layers | 1, width 1024 |

MemeCLIP is trained on the same budget rather than on its paper's 16 × 10,
because comparing two architectures under different training budgets would
measure the budgets. Its own additions keep the authors' released values:
adapter ratio 0.2, bottleneck reduction 4, cosine scale 30.

### 4.5 Model selection and reporting

The checkpoint with the best **validation** AUROC is kept, and the test split
is evaluated once, with those weights. AUROC is the headline metric, as on the
challenge; accuracy and macro F1 are reported alongside, since the Persian set
will be small and class-imbalanced, where AUROC alone hides a lot.

**Every number is a mean ± sample standard deviation over five seeds (0–4).**
Sample (`ddof=1`), because five seeds are a sample of the initialisations the
method could have been run under rather than the population of them. At this
$n$ the choice is not cosmetic: `ddof=0` reports a spread 11% narrower, in the
direction that makes a sub-point difference look resolved.
Published differences between fusion variants are under a point of AUROC, which
a single run cannot resolve. Where two variants are compared, the comparison is
also made **pairwise across seeds** — the per-seed difference, its mean and its
standard deviation — because two independent means with their own standard
deviations leave in seed noise that the variants share.

## 5. Reproducibility

Each number in the result tables has a committed config that produces it, and a
per-seed JSON record under `runs/` keyed by dataset, encoder, architecture and
fusion. Result paths name the encoder: without that, a second encoder's run
overwrites the first's seed files and the aggregate silently averages two
encoders. Feature cache files record the encoder, the split, the count, the
feature width, the write time and the library versions that produced them, and
training refuses a cache whose recorded encoder disagrees with the config.

Costs, measured rather than estimated, on the machine used here (4 CPU cores,
no GPU):

| step | cost |
| --- | --- |
| CLIP ViT-L/14 feature extraction, 10,000 memes | ~5 hours CPU; minutes on a free T4 |
| jina-clip-v2 feature extraction, 10,000 memes at 512px | ~2 hours on a free T4 |
| one training run over cached features | ~2 minutes CPU |
| one five-seed cell | ~10 minutes CPU |

Feature caching is what makes the sweeps cheap: the expensive step runs once
per encoder, and the reference MemeCLIP implementation's practice of
re-encoding every batch of every epoch is what this design avoids.

## References

- Kiela et al. The Hateful Memes Challenge. NeurIPS 2020.
- Kumar & Nandakumar. Hate-CLIPper. WOAH 2022.
- Shah et al. MemeCLIP. EMNLP 2024.
- Bui, von der Wense & Lauscher. Multi3Hate. NAACL 2025.
- Thapliyal et al. Crossmodal-3600. EMNLP 2022.
