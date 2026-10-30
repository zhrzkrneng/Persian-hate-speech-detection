# Related work — draft

A draft, not a submission. It is written so the argument is already in place
when the results arrive: each section ends at the gap this project fills.

**Before submitting, every claim marked ⚠️ has to be checked against the paper
itself.** They are the ones taken from search results or from a secondary
source rather than read directly. A wrong citation is the cheapest possible
way to lose a reviewer's trust.

---

## 1. Multimodal hateful meme detection

The Hateful Memes Challenge (HMC) [Kiela et al., NeurIPS 2020] defined the task
and the benchmark the field still reports on. Its design is the point: for
each hateful meme the authors built *benign confounders*, memes where changing
only the image or only the text flips the label. A model that reads either
modality alone therefore cannot succeed by construction. The gap the challenge
exposed was large — human annotators reached 82.65 AUROC, the best baseline
(VisualBERT) 75.44.

Fusion strategy is the axis most of the follow-up work moves along.
**Early fusion** combines raw inputs; **late fusion** combines the outputs of
two unimodal models. Hate-CLIPper [Kumar & Nandakumar, WOAH 2022] argues that
both are wrong for memes, because early fusion assumes the text describes the
image — true for captioning, false for memes, where the text and the image
often mean different things. Its answer is *intermediate* fusion over
multimodally pretrained encoders: freeze CLIP, add a trainable projection per
modality, and model their interaction explicitly as a feature interaction
matrix, the outer product of the two projections.

Hate-CLIPper reports 85.8 AUROC on the HMC test-seen split with its
align-fusion variant, above the 82.65 human figure. The variant matters for
cost: *cross-fusion* keeps the full n×n matrix and puts around 1.1 billion
parameters in the classifier at n=1024, while *align-fusion* keeps only the
diagonal — the element-wise product — and reaches the same or better accuracy
with 2.9–5 million.

MemeCLIP [Shah et al., EMNLP 2024] is usually described as a different
architecture. Reading its released implementation shows it is not, in the part
that matters: its fusion is `torch.mul(image_features, text_features)`, the
same element-wise product. What it adds is a residual adapter per modality, L2
normalisation before the product, and a cosine classifier whose weights can be
initialised from CLIP's text embeddings of the class names. It also introduces
PrideMM, 5,063 text-embedded images annotated for hate, target, stance and
humour.

Two stronger methods bound what the simple CLIP-plus-head family can claim.
ISSUES [Burbi, Baldrati, Agnolucci, Bertini & Del Bimbo, *Mapping Memes to
Words for Multimodal Hateful Meme Classification*, ICCV 2023 CLVL Workshop]
pairs CLIP with textual inversion, mapping the image into a pseudo-word token
so that the meme's combined meaning lives in the text space. Code at
`github.com/miccunifi/ISSUES`.

RGCL [Mei, Chen, Lin, Byrne & Tomalin, *Improving Hateful Meme Detection
through Retrieval-Guided Contrastive Learning*, arXiv:2311.08110, November
2023] retrieves similar examples during training and adds a contrastive
objective. Its abstract states, verbatim, "state-of-the-art performance on the
HatefulMemes dataset with an AUROC of 87.0, outperforming much larger
fine-tuned large multimodal models". Project page: `rgclmm.github.io`.

> ⚠️ Still unverified, and the three are not the same kind of gap:
>
> - **ISSUES's numeric AUROC.** Two aggregators give 0.855 and 0.86; both are
>   secondary and one marks itself unverified. The CVF proceedings page is the
>   primary source and is unreachable from this environment. Either read it or
>   cite ISSUES without a number.
> - **RGCL's venue.** The paper's own metadata gives arXiv and a project page,
>   no venue. Cite the arXiv entry unless a published version is confirmed.
> - **RGCL's 78.8 accuracy and the Flamingo-80B 86.6 comparison.** The abstract
>   confirms 87.0 AUROC and that it outperforms "much larger fine-tuned large
>   multimodal models", but neither of those two numbers. Drop them or read the
>   paper's table.

**Where this leaves us.** Hate-CLIPper and MemeCLIP are the right base and
control for a low-resource study: frozen encoders, a few million trainable
parameters, a full sweep that fits a free GPU. Neither is state of the art,
and this work does not claim to be.

## 2. Beyond English

The same architectures do not transfer. Hate-CLIPper's own evaluation on
TamilMemes found that its cross- and align-fusion variants gave **no advantage
over plain concatenation**, and that replacing the CLIP text encoder with
multilingual BERT did not help either. Its explanation is that CLIP's text and
image spaces are not aligned for a language it never saw in pretraining, so
there is no meaningful interaction left for the fusion to model. Its
Limitations section says as much, and asks for "more thorough ablation
studies" on whether learnable projection layers can overcome that
misalignment.

That paragraph is the opening this project works in, and we can now put a
number on it. Measured on Crossmodal-3600 [Thapliyal et al., EMNLP 2022] with
1000 images, the same CLIP ViT-L/14 reaches **60.05 R@1** on English
text-to-image retrieval and **0.15 R@1** on Persian, against a chance level of
0.10. The English control is what makes the Persian number mean something:
identical code, identical images, identical metric, only the caption language
changes.

⚠️ Work on other low-resource languages converges on the same recipe — a CLIP
vision tower with a language-specific or multilingual text encoder. Bengali
has MUTE and its extension ExMute; shared-task systems for Bengali, Hindi,
Gujarati, Bodo and Nepali report macro F1 around 0.6.

**Persian text-only hate speech is well served, and that is the contrast.**
ParsHate [Bokaei, Magdy & Webber, arXiv:2609.16393, September 2026] is a
manually annotated set of **10,000 Persian tweets spanning 2013–2022, 31%
hateful**, with multi-label target identification over **seven** target
categories, an explicit/implicit distinction, span-level rationales, and
score-stratified temporal sampling. State-of-the-art Persian models reach 79%
F1 on hate detection and only 25.5% macro F1 on target identification. Its
schema is close enough to this project's that the two are directly comparable:
seven targets and the explicit/implicit split are the same choices, made
independently. Other Persian text resources include PHate (multi-label, with
annotator rationales), PHOND (with a user following network) and PHICAD
(Instagram comments).

**To the best of our knowledge there is no Persian multimodal hateful meme
dataset.** A search of the Hugging Face Hub (datasets matching "persian meme",
"farsi meme", "persian multimodal"), Kaggle, and the web returned none. The
nearest multimodal meme hate dataset in a Perso-Arabic script is **Arabic**:
M3 [`mira-ai-lab/M3`] pairs 2,455 meme images with their posts across X, 4chan
and Weibo, with binary hate labels and eight fine-grained categories, and its
README states it "spans English, Chinese, and Arabic content" — Persian is not
among them. The closest Persian multimodal resource is MPerSocial
[`mebasiri/Multimodal-Persian-Sentiment-Analysis`], image–text pairs from
Instagram and Telegram annotated for **sentiment**, not hate.

> ⚠️ Still to verify: the MUTE/ExMute figures and the shared-task F1 scores,
> which come from search results rather than from the papers. The absence
> claim above is hedged on purpose: a literature search cannot establish that
> nothing exists, and it should be re-run against ACL Anthology and
> Persian-language venues before submission.

## 3. Whose labels?

Hate speech annotation is not a single ground truth. Multi3Hate [Bui, von der
Wense & Lauscher, NAACL 2025] makes this measurable: 300 parallel memes in
English, German, Spanish, Hindi and Mandarin, annotated by people from five
countries. Average pairwise agreement between countries is 74%, the lowest
being 67% between the USA and India, and the authors attribute the gap to
culture rather than noise. Krippendorff's α *within* each group runs from 0.39
to 0.47 — which they note is comparable to or higher than earlier hate speech
work, and which sets the expectation for any new annotation effort on this
task.

Their second finding concerns models. Five vision-language models evaluated
zero-shot align more closely with the US annotations than with any other
culture's, **even when the meme and the prompt are in the other culture's
language**. Translation does not carry the perspective with it.

Persian is not among their five languages, and no Iranian annotator group
appears in that comparison.

## 4. The gap

Three things hold together:

1. Intermediate fusion over frozen CLIP is cheap and strong in English, and
   its own authors report it failing on a low-resource language for a reason
   they could only hypothesise.
2. We can now measure that reason directly for Persian: CLIP's text space
   carries essentially no Persian signal, so any fusion built on it is
   effectively unimodal.
3. Cross-cultural disagreement is a measurable property of this task, and the
   Persian perspective is absent from the dataset that measures it.

This work addresses all three: a Persian meme evaluation set annotated under
the Multi3Hate protocol with per-annotator labels retained, an encoder
comparison that chooses the text encoder on evidence rather than on model-card
claims, and an evaluation of the Hate-CLIPper and MemeCLIP families on Persian
under that encoder.

---

## Citation checklist

| Claim | Status |
| --- | --- |
| HMC design, human 82.65, VisualBERT 75.44 | read in the Hate-CLIPper paper |
| Hate-CLIPper 85.8, parameter counts, TamilMemes result, Limitations | read in the paper |
| MemeCLIP fusion is an element-wise product | read in the released code |
| PrideMM, 5,063 images, four tasks | read in the repository README |
| Multi3Hate: 300 memes, 74% / 67%, α 0.39–0.47, VLM alignment | read in the paper |
| Crossmodal-3600: 3600 images, 36 languages, native-speaker captions | read in the paper and the data |
| Our own retrieval numbers | measured in this repository |
| ISSUES venue and authors | ✅ ICCV 2023 CLVL Workshop, from the CVF proceedings listing |
| ISSUES AUROC | ⚠️ aggregators give 0.855 and 0.86; CVF page unreachable from this environment |
| RGCL AUROC 87.0 | ✅ quoted verbatim from the paper's own abstract |
| RGCL venue | ⚠️ its metadata gives arXiv:2311.08110 and a project page, no venue |
| RGCL accuracy 78.8 and Flamingo-80B 86.6 | ⚠️ not in the abstract; drop or read the table |
| MUTE / ExMute / shared-task F1 | ⚠️ search results only |
| ParsHate scope and size | ✅ 10,000 tweets, 2013–2022, 31% hateful, 7 targets, from its abstract |
| "No Persian multimodal hateful meme dataset" | ✅ hedged, with the search recorded and M3's Arabic coverage named |
