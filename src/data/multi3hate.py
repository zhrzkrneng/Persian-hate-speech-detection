"""Multi3Hate: parallel memes in five languages, labelled per country.

Bui, von der Wense & Lauscher, NAACL 2025. 300 memes built from 45
memegenerator.net templates, each rendered in English, German, Spanish, Hindi
and Mandarin, and labelled for hate speech by at least five native speakers in
each of five countries.

Why this dataset earns a place in a project about Persian. The memes are
**parallel**: the same image and the same intended meaning in all five
languages, with only the rendered text changing. So when a classifier's score
differs between two languages here, the language is the only thing that
differs -- not the topic, not the difficulty, not the label balance, not the
template. No amount of collection effort could buy that design; it exists
because the original authors built it for a cross-cultural comparison.

That makes it the right place to test this project's claim that the usefulness
of cross-modal fusion is a property of the encoder's feature space rather than
of the architecture. Crossmodal-3600 gives each encoder a measured retrieval
alignment per language, Persian included, so alignment can be regressed
against downstream score across languages -- and Persian, which has an
alignment score and no labelled memes anywhere, becomes a **prediction** from
that relationship rather than a claim about untested data.

Licence: CC BY-NC-ND 4.0, stated on the dataset's Hugging Face card. Research
use, no commercial use, and no distribution of adapted material -- so the
images are not redistributed here and neither is any feature cache built from
them. Reporting measurements is not distributing an adaptation; publishing the
encoded features would be.

Three hundred memes per language is an evaluation set, not a training set.
Models are trained on the Hateful Memes Challenge and read this zero-shot.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

LANGUAGES = ("en", "de", "es", "hi", "zh")

# Each language's label comes from the country whose native speakers annotated
# that language's memes. The mapping is the dataset's own design -- the
# countries were chosen as the largest population of native speakers for each
# target language -- and it is written out rather than inferred because the
# whole point of Multi3Hate is that the five countries disagree: taking the
# wrong column would measure a different culture's judgement of the same meme.
NATIVE_COUNTRY = {
    "en": "US",
    "de": "DE",
    "es": "MX",
    "hi": "IN",
    "zh": "CN",
}
COUNTRIES = ("US", "DE", "MX", "CN", "IN")

# The dataset separates a meme's upper and lower text with this marker. Joining
# with a newline matches how the text is laid out on the image, and how
# HatefulMemesSplit presents a meme's text as one string.
SEPARATOR = "<sep>"


class Multi3HateError(ValueError):
    """A local copy that does not match what this reader expects."""


@dataclass(frozen=True)
class Meme:
    uid: str
    language: str
    text: str
    image_path: Path
    label: int | None
    # Every country's label is kept, not only the native one. The cross-cultural
    # disagreement is this dataset's own finding, and it costs nothing to carry.
    labels_by_country: dict[str, int]
    template: str

    def majority(self) -> int | None:
        votes = Counter(self.labels_by_country.values())
        if not votes:
            return None
        ranked = votes.most_common()
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return None
        return ranked[0][0]


class Multi3HateSplit:
    """One language of Multi3Hate, read from the layout download_multi3hate writes.

        data/multi3hate/
          en.jsonl ... zh.jsonl
          img/en/<id>.jpg ...

    JSONL rather than the published parquet so that reading it needs nothing
    beyond the standard library: this module is imported by the feature
    extractor and by tests that run under CI's deliberately small dependency
    set, where neither `datasets` nor `pyarrow` is installed.
    """

    def __init__(
        self,
        root: str | Path,
        language: str,
        country: str | None = None,
    ) -> None:
        if language not in LANGUAGES:
            raise Multi3HateError(
                f"unknown language {language!r}, expected one of {list(LANGUAGES)}"
            )
        self.root = Path(root)
        self.language = language
        # None means the language's native country, which is the default a
        # monolingual evaluation wants. Naming another country is how the
        # cross-cultural comparison is run.
        self.country = country or NATIVE_COUNTRY[language]
        if self.country not in COUNTRIES:
            raise Multi3HateError(
                f"unknown country {self.country!r}, expected one of {list(COUNTRIES)}"
            )
        self.records = self._read()

    @property
    def path(self) -> Path:
        return self.root / f"{self.language}.jsonl"

    def _read(self) -> list[Meme]:
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found. Run scripts/download_multi3hate.py first."
            )

        memes: list[Meme] = []
        with self.path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                where = f"{self.path}:{number}"
                row = json.loads(line)
                for field in ("id", "img", "text", "labels"):
                    if field not in row:
                        raise Multi3HateError(f"{where}: missing field {field!r}")

                labels = {
                    country: int(value)
                    for country, value in row["labels"].items()
                    if value is not None
                }
                for country, value in labels.items():
                    if value not in (0, 1):
                        raise Multi3HateError(
                            f"{where}: {country} label {value!r} is not 0 or 1"
                        )

                memes.append(
                    Meme(
                        uid=str(row["id"]),
                        language=self.language,
                        text=row["text"],
                        image_path=self.root / row["img"],
                        label=labels.get(self.country),
                        labels_by_country=labels,
                        template=row.get("template", ""),
                    )
                )

        duplicates = [
            uid for uid, count in Counter(m.uid for m in memes).items() if count > 1
        ]
        if duplicates:
            raise Multi3HateError(f"{self.path}: duplicate id(s) {duplicates}")
        return memes

    def missing_images(self) -> list[str]:
        return [str(m.image_path) for m in self.records if not m.image_path.exists()]

    def label_counts(self) -> Counter:
        return Counter(m.label for m in self.records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Meme:
        return self.records[index]


def parallel_ids(root: str | Path) -> list[str]:
    """Ids present in every language, which is what makes a comparison parallel.

    The dataset should be complete, so this is a check rather than a filter: a
    score compared across languages over different sets of memes would not be
    measuring the language.
    """
    per_language = {}
    for language in LANGUAGES:
        split = Multi3HateSplit(root, language)
        per_language[language] = {m.uid for m in split.records}

    shared = set.intersection(*per_language.values())
    ragged = {
        language: sorted(ids - shared)[:5]
        for language, ids in per_language.items()
        if ids - shared
    }
    if ragged:
        raise Multi3HateError(
            f"languages do not cover the same memes; extras per language: {ragged}"
        )
    return sorted(shared, key=int)
