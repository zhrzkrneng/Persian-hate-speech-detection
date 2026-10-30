"""Multi3Hate's parallel structure is the whole reason it is here.

The memes are the same image and the same meaning in five languages, so a
score difference between languages is caused by the language. Two things can
quietly destroy that, and both are tested here: comparing languages over
different sets of memes, and reading the wrong country's label -- the dataset's
own finding is that the five countries disagree, so taking the US column for
Hindi memes would measure one culture's judgement of another's memes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.multi3hate import (  # noqa: E402
    COUNTRIES,
    LANGUAGES,
    NATIVE_COUNTRY,
    Multi3HateError,
    Multi3HateSplit,
    parallel_ids,
)

LABELS = {"US": 1, "DE": 0, "MX": 1, "CN": 0, "IN": 1}


def write(
    root: Path,
    language: str,
    ids=("0", "1"),
    labels: dict | None = None,
    make_images: bool = True,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "img" / language).mkdir(parents=True, exist_ok=True)
    rows = []
    for uid in ids:
        if make_images:
            (root / "img" / language / f"{uid}.jpg").write_bytes(b"not-a-real-jpeg")
        rows.append(
            {
                "id": uid,
                "img": f"img/{language}/{uid}.jpg",
                "text": f"{language} text {uid}",
                "template": "a template",
                "labels": dict(LABELS if labels is None else labels),
            }
        )
    path = root / f"{language}.jsonl"
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    return path


def test_every_language_is_known() -> None:
    assert set(LANGUAGES) == {"en", "de", "es", "hi", "zh"}
    assert set(NATIVE_COUNTRY) == set(LANGUAGES)
    assert set(NATIVE_COUNTRY.values()) <= set(COUNTRIES)


def test_each_language_defaults_to_its_native_country(tmp_path: Path) -> None:
    """The dataset's countries disagree, so the default must be deliberate."""
    for language in LANGUAGES:
        write(tmp_path, language)
        split = Multi3HateSplit(tmp_path, language)
        assert split.country == NATIVE_COUNTRY[language]
        assert split[0].label == LABELS[NATIVE_COUNTRY[language]]

    # Hindi memes labelled by India, not by the US: 1 and 1 happen to agree in
    # the fixture, so check the column actually selected rather than the value.
    assert Multi3HateSplit(tmp_path, "hi").country == "IN"
    assert Multi3HateSplit(tmp_path, "zh").country == "CN"


def test_another_country_can_be_asked_for(tmp_path: Path) -> None:
    """Cross-cultural comparison is reading a different column, nothing more."""
    write(tmp_path, "hi")
    assert Multi3HateSplit(tmp_path, "hi", country="IN")[0].label == 1
    assert Multi3HateSplit(tmp_path, "hi", country="DE")[0].label == 0


def test_every_country_label_is_kept(tmp_path: Path) -> None:
    write(tmp_path, "en")
    meme = Multi3HateSplit(tmp_path, "en")[0]
    assert meme.labels_by_country == LABELS
    assert meme.majority() == 1  # three of five


def test_a_tied_majority_is_unresolved(tmp_path: Path) -> None:
    write(tmp_path, "en", labels={"US": 1, "DE": 0, "MX": 1, "CN": 0})
    assert Multi3HateSplit(tmp_path, "en")[0].majority() is None


def test_an_unknown_language_is_refused(tmp_path: Path) -> None:
    with pytest.raises(Multi3HateError, match="unknown language"):
        Multi3HateSplit(tmp_path, "fa")


def test_an_unknown_country_is_refused(tmp_path: Path) -> None:
    write(tmp_path, "en")
    with pytest.raises(Multi3HateError, match="unknown country"):
        Multi3HateSplit(tmp_path, "en", country="IR")


def test_a_label_outside_zero_one_is_refused(tmp_path: Path) -> None:
    write(tmp_path, "en", labels={"US": 2, "DE": 0, "MX": 0, "CN": 0, "IN": 0})
    with pytest.raises(Multi3HateError, match="not 0 or 1"):
        Multi3HateSplit(tmp_path, "en")


def test_duplicate_ids_are_refused(tmp_path: Path) -> None:
    write(tmp_path, "en", ids=("7", "7"))
    with pytest.raises(Multi3HateError, match="duplicate id"):
        Multi3HateSplit(tmp_path, "en")


def test_a_missing_file_says_how_to_get_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download_multi3hate"):
        Multi3HateSplit(tmp_path, "en")


def test_missing_images_are_reported(tmp_path: Path) -> None:
    write(tmp_path, "en", make_images=False)
    assert len(Multi3HateSplit(tmp_path, "en").missing_images()) == 2


def test_parallel_ids_are_the_shared_ones(tmp_path: Path) -> None:
    for language in LANGUAGES:
        write(tmp_path, language, ids=("0", "1", "2"))
    assert parallel_ids(tmp_path) == ["0", "1", "2"]


def test_a_ragged_copy_is_refused(tmp_path: Path) -> None:
    """The failure this guards: comparing languages over different memes.

    A score gap between two languages measured on different subsets is not a
    language effect, and nothing downstream would reveal it.
    """
    for language in LANGUAGES:
        write(tmp_path, language, ids=("0", "1"))
    write(tmp_path, "zh", ids=("0", "1", "99"))

    with pytest.raises(Multi3HateError, match="same memes"):
        parallel_ids(tmp_path)


def test_the_separator_becomes_a_line_break(tmp_path: Path) -> None:
    """Upper and lower meme text, joined the way the image reads."""
    from scripts.download_multi3hate import read_captions

    source = tmp_path / "data" / "captions"
    source.mkdir(parents=True)
    (source / "en.csv").write_text(
        "Meme ID,Template Name,Original (English),Translation\n"
        '5,t,"UPPER <sep> LOWER","UPPER <sep> LOWER"\n',
        encoding="utf-8",
    )
    text, template = read_captions(tmp_path, "en")["5"]
    assert text == "UPPER\nLOWER"
    assert template == "t"
