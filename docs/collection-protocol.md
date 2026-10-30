# Collection protocol: the Persian meme set

The annotation guideline (`annotation-guideline.md`) says *what* is labelled
and *how*. This says what is collected, what is deliberately in and out of
scope, and the exact commands a collecting session runs.

§9 of the guideline requires the scope decision to be made and written down
**before** collection, so that the paper's Ethics Statement states it rather
than implying it. This document is that record.

## 1. Scope, as decided

**Decided 2026-10-30 by the author.** All seven target categories are in
scope, political identity among them:

| Target category | In scope |
| --- | --- |
| gender | yes |
| religion | yes |
| ethnicity or nationality | yes |
| political identity | yes |
| disability | yes |
| sexual orientation | yes |
| other | yes |

Persian meme culture is heavily political, so excluding political identity
would have meant excluding much of what Persian memes actually do, and the
resulting set would have measured the categories that survived rather than
Persian hateful memes. The decision is to include it and to be explicit about
the boundary below.

### The boundary that makes the political category meaningful

The guideline's §5 rule is doing more work here than in any other category:

> **Criticism of an institution, a government, a policy or an organisation.**
> Institutions are not identities.

So, under the UN definition this project adopts:

- A meme attacking a government, a ministry, a policy, an office or an
  organisation is **not hate speech**, however harsh. It is political speech.
- A meme attacking *people* for a political identity they hold — supporters of
  a side, a diaspora, a generation, a movement, as the kind of people they are
  — **is** hate speech, with target `political`.

A predictable consequence, worth stating in the paper before a reviewer raises
it: a large share of political Persian memes will be labelled non-hate, and
the `political` target will be smaller than the volume of political material
collected. That is the definition working, not the sampling failing. Report
both numbers — how much political material was collected, how much of it was
labelled hate — because the gap is itself a finding about what the UN
definition does and does not capture in this language.

### Researcher and annotator safety

- **No annotator is identifiable from anything in this repository.** The
  schema stores pseudonyms (`A1`, `A2`, …). The mapping from pseudonym to
  person is never committed, never put in the annotation file, and does not
  need to exist in writing at all.
- **No meme image is ever released.** Labels, per-annotator labels, image
  hashes, source URLs and the guideline are released; the images are
  third-party content whose rights the project does not hold, and the people
  in them did not consent to appear in a hate speech dataset.
- Collection draws on **public** accounts and channels only. Nothing private,
  nothing requiring a follow request, no direct messages.

## 2. What to collect

From the guideline's §8, with the operational detail:

- **Size**: 300–500 memes. Multi3Hate works with 300.
- **Sources**: public Persian-language meme accounts and channels.
- **Balance**: no single target category should dominate. Aim for a spread,
  and keep collecting in the thin categories rather than trimming the fat one
  — trimming after labelling selects on the label.
- **Persian text on the image.** An image with a caption underneath is a
  different object and is out of scope.
- **Deduplicate**: handled by `prepare_memes.py`, by exact hash and by
  perceptual hash. Templates recur heavily.
- **Anonymise before an annotator sees it**: blur or crop usernames, handles,
  profile pictures and phone numbers belonging to private individuals. A
  channel's own watermark is a public account, not a private individual.

## 3. Folder layout

Everything under `data/` is gitignored and never committed.

```
data/persian/
  raw/                  the collected images, as saved
  sources.csv           provenance: file, platform, url, collected
  annotations.jsonl     the annotation file, built from the two above
```

## 4. A collecting session, in commands

**Step 1 — collect.** Save images into `data/persian/raw/`. Keep whatever
filenames they arrive with; nothing depends on them.

**Step 2 — record where they came from, the same day.**

```bash
python -m scripts.prepare_memes --images data/persian/raw --init-manifest
```

This adds a blank row to `data/persian/sources.csv` for each new file and
leaves every row already filled in alone. Open it in a spreadsheet and fill in
`platform`, `url` and `collected` (`YYYY-MM-DD`), then save it as CSV again.

Do this in the same session as the collecting. A folder of images a week later
does not say where anything came from, and three hundred rows cannot be
reconstructed from memory. `url` may be left empty — a forwarded meme has no
link — and the count of those is reported so the paper can state it.

**Step 3 — check before writing anything.**

```bash
python -m scripts.prepare_memes --images data/persian/raw --dry-run
```

Reports exact duplicates, near duplicates, filenames carrying identifying
detail, and any gap in the manifest. Act on these now: a duplicate annotated
twice is not an independent judgement, and it inflates agreement.

**Step 4 — build or extend the annotation file.**

```bash
python -m scripts.prepare_memes --images data/persian/raw \
    --out data/persian/annotations.jsonl
```

Safe to run after every session. Rows are matched by image hash: existing ids,
transcriptions and labels are kept exactly, new memes are appended with fresh
ids, and ids are never reassigned.

**Step 5 — transcribe each meme's text by hand**, into the `text` field, or in
the annotation page, which writes it for you. Not by OCR: Persian OCR on meme
typography is unreliable, and a wrong transcription is a label error no model
can recover from.

**Step 6 — annotate.** Open `tools/annotate.html` in a browser, load
`annotations.jsonl` and the image folder. It runs entirely in the browser,
loads nothing from the network and sends nothing anywhere. Each annotator uses
their own pseudonym and works from their own copy; the page merges rather than
overwrites, so copies can be combined by loading them in turn.

**Step 7 — check the round.**

```bash
python -m scripts.check_annotations --file data/persian/annotations.jsonl
```

Reports coverage, ties, duplicate images, missing provenance, the label and
target distributions, Krippendorff's α and pairwise agreement, with
Multi3Hate's α range printed beside it for context.

Run the **pilot** first: 50 memes, all annotators, then read the
disagreements together and expect the guideline to need revision. After the
main round starts, the guideline does not change; if it has to, the memes
labelled under the old version are re-annotated.

## 5. Cost

Nothing in this pipeline costs money. Collection is manual, the annotation
page is a single local HTML file, the checks run on a laptop CPU in seconds,
and no API is called at any point. The only cost is annotator time, which is
why the guideline caps the set at 300–500 memes and insists on a pilot: a
protocol error found at meme 50 is cheap, and at meme 400 it is a re-run.

## 6. Tools considered and not used

- **[Hengam](https://github.com/kargaranamir/Hengam)** (Mirzababaei et al.,
  [AACL 2022](https://aclanthology.org/2022.aacl-main.74), MIT) tags Persian
  temporal expressions inside a text — `B-TIM`, `B-DAT` spans. It is a good
  tool for a different problem: the `collected` date here is recorded by the
  collector at save time, not extracted from the meme's words, and a date
  mentioned *inside* a meme is content, not provenance. Worth revisiting only
  if the set is ever analysed for the events its memes refer to.
- **Persian OCR** for the `text` field: rejected by the guideline. Meme
  typography defeats it, and the error lands silently in the label.
