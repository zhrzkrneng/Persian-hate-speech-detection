# What Hate-CLIPper's Table 1 actually reports

Transcribed from the paper (arXiv:2210.05916, Table 1 and Table 3) so that a
comparison against it can be checked rather than remembered. Every row uses
one projection layer and CLIP ViT-L/14 with frozen weights; the rows differ in
the fusion and in the number of pre-output layers.

| fusion | p.o. layers | dev seen | test seen | trainable |
| --- | --- | --- | --- | --- |
| Concat | 1 | 76.72 | 79.87 | 3.9M |
| Concat | 3 | 79.02 | 83.73 | 6M |
| Concat | 5 | 78.60 | 83.80 | 8.1M |
| Concat | 7 | 78.63 | 83.29 | 10.2M |
| CMAF | 1 | 77.36 | 80.15 | 4.5M |
| CMAF | 3 | 76.85 | 82.00 | 6.6M |
| CMAF | 5 | 79.51 | 83.35 | 8.7M |
| CMAF | 7 | 78.88 | 82.40 | 10.8M |
| Cross | 1 | **82.62** | 85.12 | 1.1B |
| Cross | 3 | 82.19 | 82.66 | 1.1B |
| Align | 1 | 81.18 | 85.46 | 2.9M |
| Align | 3 | 81.55 | **85.80** | 5M |
| Align | 5 | 80.88 | 85.46 | 7.1M |
| Align | 7 | 81.09 | 84.88 | 9.2M |

Their Table 3 gives the configuration: image size 224, ViT-Large-Patch14,
projection dimension 1024, pre-output dimension 1024, AdamW, 20 epochs, batch
size 64, learning rate 1e-4, weight decay 1e-4, gradient clip 0.1.

## Why this file exists

**Every one of those nine hyperparameters matches this project's configs.**
So the gap in absolute score is not a difference in optimiser, budget or
width, and saying "our pipeline is weaker" without saying in what respect
would have been a guess.

**The 85.8 in their abstract is Align with three pre-output layers.** This
project's configs use one, so the row to compare against is Align/1 at 85.46,
not 85.8. A comparison against 85.8 compares different architectures.

**Their Concat/1 is 79.87.** This project's concat cell reaches 81.4 over
fifteen seeds, so the pipeline here is not uniformly weaker than theirs -- it
is better on the row that shares its configuration. What differs is the size
of the fusion benefit: +5.59 for them at one pre-output layer, +0.87 here.

**Dev-seen does not rank these variants the way test-seen does.** The best dev
number in the table is Cross/1 at 82.62, whose test score is 85.12; the best
test number is Align/3 at 85.80, which ranks fourth on dev. Selecting on dev
and reading test once -- the protocol this project follows -- would have
returned 85.12 from this table, not 85.8. This is a statement about what the
table shows, not about how the numbers were obtained, which the paper does not
describe.

**No seed count or standard deviation is given** for any cell. Each number
appears to be a single run. The differences this project reports are under a
point of AUROC, which is why it averages fifteen.

**Cross fusion scores below Align while using 380 times the parameters**:
85.12 against 85.46, at 1.1B trainable against 2.9M. The cross cell missing
from this project's ablation is therefore a missing *cost*, not a missing
advantage, and the section that reports it as absent should say so.
