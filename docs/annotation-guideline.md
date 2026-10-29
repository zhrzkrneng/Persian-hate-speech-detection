# Annotation guideline: Persian hateful memes

The protocol for building the Persian evaluation set. This is the document a
reviewer will ask for first, and the one that decides whether the labels mean
anything.

`annotation-guideline.fa.md` is the Persian version annotators actually read.
This one is the full protocol, including the research design they do not need.

## 1. What is being annotated

A **meme**: one image with text rendered onto it. Not an image with a separate
caption underneath — that is a different object, and the distinction matters,
because the whole difficulty of this task is that the image and the text mean
something together that neither means alone.

The unit of annotation is the meme as a whole. Annotators see the image with
its text in place and judge that, never the text alone.

## 2. Definition

We adopt the United Nations definition, as Multi3Hate does, so that labels are
comparable with that dataset rather than resting on a definition we invented:

> any kind of communication in speech, writing or behaviour, that attacks or
> uses pejorative or discriminatory language with reference to a person or a
> group on the basis of who they are, in other words, based on their religion,
> ethnicity, nationality, race, colour, descent, gender or other identity
> factor.

Source: <https://www.un.org/en/hate-speech/understanding-hate-speech/what-is-hate-speech>

Two consequences annotators get wrong most often:

- **Offensive is not the same as hateful.** A crude joke that targets nobody's
  identity is not hate speech, however distasteful.
- **The attack must be on identity.** Insulting a named individual for
  something they did is not hate speech under this definition; insulting them
  for what they are is.

## 3. Labels

| Label | Value |
| --- | --- |
| non-hate | 0 |
| hate | 1 |
| I don't know | abstain |

"I don't know" is a real option and must stay available. An annotator forced
to choose on a meme they do not understand adds noise that looks like signal.
Multi3Hate offers the same three options.

### Secondary labels, on memes labelled hate

**Target category**, following the identity factors in the definition:
gender, religion, ethnicity or nationality, political identity, disability,
sexual orientation, other.

**Explicitness**: *explicit* if the attack is stated in words or an unambiguous
image; *implicit* if it depends on irony, a stereotype, a visual pun, or
knowledge the reader has to supply. ParsHate makes the same distinction for
Persian text, so the two can be compared.

## 4. Procedure

1. Look at the whole meme before reading closely. First impression matters: a
   meme works on a reader in seconds.
2. Ask: does this attack or demean a person or group **on the basis of who
   they are**?
3. If yes, label hate, then choose the target category and the explicitness.
4. If the meaning depends on something you do not recognise — a reference, a
   person, a template — choose "I don't know" rather than guessing.

**Judge the meme, not the poster.** The question is what this meme
communicates, not whether whoever shared it meant harm.

**Judge it as it stands.** Do not imagine a context that would make it
innocent, and do not imagine one that would make it worse.

## 5. Cases that are not hate speech

Written out because these are where annotators disagree most:

- **Self-directed humour.** A group joking about itself, in its own voice.
- **Quoting or reporting hate in order to criticise it.** Mark it as
  non-hate, and flag it, since these are valuable and rare.
- **Criticism of an institution, a government, a policy or an organisation.**
  Institutions are not identities. Criticism of a religion's doctrine is not
  an attack on its believers; mockery of believers *as believers* is.
- **Offensive but untargeted vulgarity.**

## 6. How many annotators, and what agreement to expect

**At least three annotators per meme**, independently, with the final label by
majority. Multi3Hate collects a minimum of five and breaks ties by gathering
more; three is the floor below which majority voting stops meaning anything.

**One annotator is not an option.** A single-annotator dataset is the first
thing a reviewer rejects.

Report **Krippendorff's α** over all annotators. The number to expect:

| Study | α |
| --- | --- |
| Multi3Hate, US annotators | 0.47 |
| Multi3Hate, German | 0.45 |
| Multi3Hate, Chinese | 0.43 |
| Multi3Hate, Indian | 0.40 |
| Multi3Hate, Mexican | 0.39 |

Multi3Hate notes these are "higher than or comparable to" earlier hate speech
research. **An α around 0.4 on this task is a normal result, not a failed
one.** Hate speech annotation genuinely is subjective; a suspiciously high α
usually means the annotators were trained into agreement, or the sample was
too easy. Report the number honestly and discuss where the disagreement sits.

### The disagreement is a finding, not noise

Multi3Hate's central result is that annotators from different countries agreed
only 74% of the time on average, lowest between the USA and India at 67%, and
that vision-language models align more closely with US annotations than with
any other culture — even when the meme and the prompt are in the other
culture's language.

Persian is absent from that dataset. Adding it, with the same protocol, is
this project's data contribution. So **keep every individual annotator's label,
not only the majority**. The per-annotator labels are what make the
cross-cultural comparison possible, and they cost nothing to store.

## 7. Pilot before the real round

Annotate 50 memes with all annotators, compute α, and read the disagreements
together. Expect the guideline to need revision after this; that is what the
pilot is for. Only then annotate the rest, and do not change the guideline
again mid-round — if it has to change, the memes labelled under the old
version are re-annotated.

## 8. Collection

- **Sources**: public Persian-language meme accounts and channels. Record the
  platform and collection date for each item.
- **Size**: 300 to 500 memes. Multi3Hate works with 300. A small, carefully
  labelled set is worth more here than a large noisy one, and it is what three
  annotators can actually get through.
- **Balance**: aim for a mix of target categories, and do not let one topic
  dominate. A set that is 80% one category measures that category, not hate.
- **Deduplicate** by image hash and by text. Meme templates recur heavily.

### Anonymisation, at collection time

Remove or blur usernames, handles, profile pictures, phone numbers and any
other identifying detail of private individuals, before the meme reaches an
annotator. Multi3Hate releases its data with user IDs and personally
identifiable information removed, and the same applies here.

## 9. Ethics

**Annotator wellbeing.** This material is unpleasant to read in volume. Tell
annotators what the content is before they consent, let them stop at any time,
and keep sessions short. Multi3Hate obtained institutional ethical approval,
warned annotators explicitly, and took written consent; do the same, and if a
university is involved, get the approval before collection starts, not after.

**Compensation.** If annotators are paid, pay fairly and state the rate in the
paper. If they are volunteers — colleagues or fellow students — say that
instead. Either is acceptable; silence is not.

**Researcher safety.** Persian meme culture is heavily political, and the
political target category will be the most sensitive one in this dataset.
Decide deliberately, before collection, what is in scope and what is not, and
write that decision into the paper's Ethics Statement rather than leaving it
implicit. This is a judgement about your own circumstances that only you can
make.

**What gets released.** Labels, the per-annotator labels, the guideline, image
hashes and source URLs — never the meme images themselves. The images are
third-party content whose rights you do not hold, and the people in them did
not consent to appear in a hate speech dataset. State the terms of use and
prohibit malicious reuse, as Multi3Hate does.

## 10. Storage format

One JSONL line per meme. Keep the per-annotator labels:

```json
{
  "id": "fa_0001",
  "image_sha256": "…",
  "source": {"platform": "…", "url": "…", "collected": "2026-11-03"},
  "text": "متن روی تصویر",
  "annotations": [
    {"annotator": "A1", "label": 1, "target": "gender", "explicit": false},
    {"annotator": "A2", "label": 1, "target": "gender", "explicit": false},
    {"annotator": "A3", "label": 0, "target": null, "explicit": null}
  ],
  "label": 1,
  "agreement": 0.67
}
```

The `text` field is the meme's text, transcribed. Transcribe it by hand.
Persian OCR on meme typography is unreliable, and a wrong transcription is a
label error that no model can recover from.

## References

- UN, *What is hate speech* — the definition used here.
- Bui, von der Wense & Lauscher, [Multi3Hate](https://aclanthology.org/2025.naacl-long.490/), NAACL 2025 — protocol, label set, agreement expectations, ethics.
- Kiela et al., [The Hateful Memes Challenge](https://arxiv.org/abs/2005.04790), NeurIPS 2020 — the multimodal task definition.
- Bokaei et al., ParsHate — Persian text hate speech, explicit/implicit distinction and target categories.
