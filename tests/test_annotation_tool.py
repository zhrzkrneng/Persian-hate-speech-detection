"""The annotation tool must write files the reader accepts.

A labelling interface that produces a file `src/data/persian.py` rejects is
worse than no interface: the error surfaces after the annotation round, when
the work is already done. So the page's vocabulary is checked against the
reader's, and its save logic is replicated here and fed through the reader.

The replication is the weak point — it is a second implementation of what the
page does, and the two can drift. It is kept deliberately short, and mirrors
the one block of JavaScript under `$("save")`.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.persian import TARGETS, AnnotationError, PersianMemeSet  # noqa: E402

PAGE = Path(__file__).resolve().parents[1] / "tools" / "annotate.html"
DIGEST = "a" * 64


def page_text() -> str:
    return PAGE.read_text(encoding="utf-8")


def test_page_exists() -> None:
    assert PAGE.is_file(), PAGE


def test_target_vocabulary_matches_the_reader() -> None:
    """Drift here writes files that fail validation after the work is done."""
    block = re.search(r"const TARGETS = \[(.*?)\];", page_text(), re.S).group(1)
    in_page = {t.strip().strip('"') for t in block.replace("\n", "").split(",")}
    in_page.discard("")
    assert in_page == TARGETS


def test_every_target_is_offered_in_the_dropdown() -> None:
    offered = set(re.findall(r'<option value="([a-z_]+)">', page_text()))
    assert TARGETS <= offered, TARGETS - offered


def test_page_does_not_phone_home() -> None:
    """Annotation content is sensitive; nothing should leave the browser."""
    html = page_text()
    for call in ("fetch(", "XMLHttpRequest", "navigator.sendBeacon", "WebSocket("):
        assert call not in html, call
    assert "<script src=" not in html, "no external script may be loaded"


# ── replication of the page's save step ──────────────────────────────────


def save(rows: list[dict], answers: list[dict], annotator: str) -> str:
    """Mirror of the JavaScript under $("save"), including its guard."""
    incomplete = [
        rows[i]["id"]
        for i, a in enumerate(answers)
        if a.get("label") == 1
        and (a.get("target") not in TARGETS or not isinstance(a.get("explicit"), bool))
    ]
    if incomplete:
        raise ValueError(f"incomplete hate labels: {incomplete}")

    out = []
    for row, a in zip(rows, answers, strict=True):
        copy = json.loads(json.dumps(row))
        copy["text"] = a.get("text", "")
        existing = copy.get("annotations") or []
        copy["annotations"] = [x for x in existing if x["annotator"] != annotator]
        if "label" in a:
            entry = {"annotator": annotator, "label": a["label"]}
            if a["label"] == 1:
                entry["target"] = a["target"]
                entry["explicit"] = a["explicit"]
            copy["annotations"].append(entry)
        out.append(json.dumps(copy, ensure_ascii=False))
    return "\n".join(out) + "\n"


def skeleton(n: int = 3) -> list[dict]:
    return [
        {
            "id": f"fa_{i:04d}",
            "image_sha256": DIGEST,
            "source": {"platform": "", "url": "", "collected": ""},
            "text": "",
            "annotations": [],
        }
        for i in range(n)
    ]


def test_output_is_accepted_by_the_reader(tmp_path: Path) -> None:
    rows = skeleton()
    answers = [
        {"label": 1, "target": "gender", "explicit": False, "text": "متن اول"},
        {"label": 0, "target": None, "explicit": None, "text": "متن دوم"},
        {"label": None, "target": None, "explicit": None, "text": "متن سوم"},
    ]
    path = tmp_path / "a.jsonl"
    path.write_text(save(rows, answers, "A1"), encoding="utf-8")

    data = PersianMemeSet(path)
    assert len(data) == 3
    assert data.annotators() == ["A1"]
    assert data[0].majority() == 1
    assert data[2].labels == (None,)  # "I don't know" survives as an abstention
    assert data[0].text == "متن اول"  # Persian text round-trips


def test_a_second_annotator_merges_rather_than_overwrites(tmp_path: Path) -> None:
    """Two annotators working from the same file must both end up in it."""
    rows = skeleton(1)
    first = json.loads(save(rows, [{"label": 0, "text": "x"}], "A1").strip())
    merged = save(
        [first], [{"label": 1, "target": "other", "explicit": True, "text": "x"}], "A2"
    )

    path = tmp_path / "a.jsonl"
    path.write_text(merged, encoding="utf-8")
    data = PersianMemeSet(path)
    assert data.annotators() == ["A1", "A2"]
    assert data[0].majority() is None  # one each: a tie, so unresolved


def test_rewriting_your_own_pass_does_not_duplicate_you(tmp_path: Path) -> None:
    rows = skeleton(1)
    once = json.loads(save(rows, [{"label": 0, "text": "x"}], "A1").strip())
    twice = save(
        [once], [{"label": 1, "target": "other", "explicit": True, "text": "x"}], "A1"
    )

    path = tmp_path / "a.jsonl"
    path.write_text(twice, encoding="utf-8")
    data = PersianMemeSet(path)
    assert len(data[0].annotations) == 1
    assert data[0].labels == (1,)


def test_incomplete_hate_label_is_refused_before_writing(tmp_path: Path) -> None:
    """Catching it in the page beats failing validation after the round."""
    with pytest.raises(ValueError, match="incomplete"):
        save(skeleton(1), [{"label": 1, "target": None, "explicit": None}], "A1")


def test_an_unfinished_file_is_still_readable(tmp_path: Path) -> None:
    """Saving halfway through must not produce something unreadable."""
    rows = skeleton(2)
    answers = [{"label": 0, "text": "done"}, {}]  # second not annotated yet
    path = tmp_path / "a.jsonl"
    path.write_text(save(rows, answers, "A1"), encoding="utf-8")

    data = PersianMemeSet(path)
    assert data[0].majority() == 0
    assert data[1].annotations == ()


def test_a_bad_target_would_be_caught_by_the_reader(tmp_path: Path) -> None:
    """Belt and braces: if the page ever offers a wrong value, this fires."""
    rows = skeleton(1)
    bad = json.loads(save(rows, [{"label": 0, "text": ""}], "A1").strip())
    bad["annotations"] = [
        {"annotator": "A1", "label": 1, "target": "football", "explicit": True}
    ]
    path = tmp_path / "a.jsonl"
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(AnnotationError, match="needs a target from"):
        PersianMemeSet(path)
