"""Footnotes have to be referenced, not merely defined.

The journal asks for the Latin equivalent of a translated technical term in a
footnote at its first use, created as a real Word footnote and numbered afresh
on each page.

The first build satisfied none of that while looking like it did. `P()` wrapped
every run it was given in a `TextRun`, so the eight `FootnoteReferenceRun`
objects were destroyed on the way into their paragraphs: the document carried
eight footnote bodies and zero references to them. It opened, it passed schema
validation, and not one footnote was visible. These tests read the built
document and check both halves.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "paper" / "maghale.docx"


@pytest.fixture(scope="module")
def parts() -> dict[str, str]:
    if not DOCX.exists():
        pytest.skip("paper/maghale.docx not built")
    with zipfile.ZipFile(DOCX) as archive:
        names = set(archive.namelist())
        if "word/footnotes.xml" not in names:
            pytest.skip("document has no footnotes part")
        return {
            name: archive.read(name).decode("utf-8")
            for name in ("word/document.xml", "word/footnotes.xml", "word/settings.xml")
        }


def footnote_bodies(xml: str) -> dict[str, str]:
    """id -> text, for the real footnotes (Word reserves ids 0 and -1)."""
    out = {}
    for block in re.findall(
        r"<w:footnote [^>]*w:id=\"(-?\d+)\"(.*?)</w:footnote>", xml, re.DOTALL
    ):
        identifier, content = block
        if int(identifier) <= 0:
            continue
        text = "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", content))
        out[identifier] = text.strip()
    return out


def test_every_footnote_is_referenced(parts: dict[str, str]) -> None:
    bodies = footnote_bodies(parts["word/footnotes.xml"])
    referenced = set(
        re.findall(r"<w:footnoteReference w:id=\"(\d+)\"", parts["word/document.xml"])
    )
    assert bodies, "no footnotes in the document"
    orphans = sorted(set(bodies) - referenced, key=int)
    assert not orphans, "footnote bodies with no reference in the text: " + ", ".join(
        f"{i} ({bodies[i]!r})" for i in orphans
    )


def test_every_reference_has_a_body(parts: dict[str, str]) -> None:
    bodies = footnote_bodies(parts["word/footnotes.xml"])
    referenced = set(
        re.findall(r"<w:footnoteReference w:id=\"(\d+)\"", parts["word/document.xml"])
    )
    dangling = sorted(referenced - set(bodies), key=int)
    assert not dangling, f"references with no footnote body: {dangling}"


def test_footnote_text_is_latin(parts: dict[str, str]) -> None:
    """These footnotes exist to give the Latin term; a Persian one is a slip."""
    for identifier, text in footnote_bodies(parts["word/footnotes.xml"]).items():
        assert text, f"footnote {identifier} is empty"
        assert not re.search(r"[؀-ۿ]", text), (
            f"footnote {identifier} is in Persian: {text!r}"
        )


def test_numbering_restarts_on_each_page(parts: dict[str, str]) -> None:
    settings = parts["word/settings.xml"]
    assert "w:footnotePr" in settings, "no footnote properties in settings.xml"
    assert 'w:numRestart w:val="eachPage"' in settings, (
        "the journal requires footnote numbering to restart on each page"
    )


def test_no_term_is_footnoted_twice(parts: dict[str, str]) -> None:
    texts = list(footnote_bodies(parts["word/footnotes.xml"]).values())
    duplicates = {t for t in texts if texts.count(t) > 1}
    assert not duplicates, f"a footnote should mark the first use only: {duplicates}"
