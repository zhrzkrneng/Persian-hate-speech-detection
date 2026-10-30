# Paper draft

What is written, what is not, and what each missing piece is waiting for.

Markdown rather than LaTeX for now: the content is what needs reviewing, and a
diff of it is readable in a pull request. Converting to a venue's template is
mechanical and happens once the content settles.

## Written

| section | file | depends on |
| --- | --- | --- |
| Method, experimental setup, reproducibility | `method.md` | the code only — every number in it is computed from the repository |

## Not written, and why

**Results, discussion, abstract, introduction, conclusion.** These wait on one
experiment, for a reason worth stating rather than working around.

On jina-clip-v2 features, align fusion gives no advantage over plain
concatenation. Paired across five seeds:

```
dim 1024   align − concat   mean −0.26   sd 0.38
dim  256   align − concat   mean −0.05   sd 0.89
```

And the best number so far, 79.46 ± 0.27 test AUROC, sits 6.3 points below
Hate-CLIPper's published 85.8 on the same split.

Both observations have two readings, and nothing collected so far separates
them:

1. **jina-clip-v2's feature space removes the fusion advantage.** This would be
   a finding, and a direct extension of Hate-CLIPper's own Limitations section,
   which reports exactly this on Tamil and attributes it to a feature space CLIP
   never aligned for a language it had not seen. Here it would be happening on
   *English* data, with an encoder that did see Persian — so the cost of
   multilinguality would be showing up on the language the encoder is good at.
2. **This pipeline does not reproduce the fusion effect at all.** That would be
   a defect in our work, and every number in it would be uninterpretable.

The experiment that separates them is the CLIP ViT-L/14 control: the same
pipeline, the same splits, the same budget, the other encoder. If CLIP-align
beats CLIP-concat and lands near 85.8, reading 1 holds. If CLIP behaves like
jina, reading 2 holds and the pipeline needs fixing before anything is written.

Writing a results section before that would mean writing it twice, and worse,
risks fixing a narrative in place before the evidence that decides it exists.

## Status of the experiments

| cell | seeds | status |
| --- | --- | --- |
| jina · Hate-CLIPper · align · n=1024 | 5 | done, 79.20 ± 0.52 |
| jina · Hate-CLIPper · concat · n=1024 | 5 | done, 79.46 ± 0.27 |
| jina · MemeCLIP · align · n=1024 | 5 | done, 77.88 ± 0.51 |
| jina · Hate-CLIPper · concat · n=256 | 5 | done, 79.40 ± 0.66 |
| jina · Hate-CLIPper · align · n=256 | 5 | done, 79.35 ± 0.40 |
| jina · Hate-CLIPper · cross · n=256 | 0 | deferred — ~67M trainable parameters, hours per seed on 4 CPU cores. Needs a GPU. If the final table ships without it, that has to be said, not left blank. |
| CLIP · all four cells | 0 | waiting on feature extraction (~5 h CPU, in progress) |

Retrieval numbers in `runs/retrieval/` were measured on a subset of
Crossmodal-3600. The final table should be rerun over all 3,600 images on a
GPU; the subset is enough to have chosen an encoder, not to be a published
number.

## Claims needing verification before submission

`docs/related-work.md` carries ⚠️ markers on every claim taken from a secondary
source. Those are to be checked against the primary source, not promoted by
being repeated. The "no public Persian multimodal hate speech dataset" claim in
particular must be hedged to "to the best of our knowledge" — absence is not
something a literature search can establish.
