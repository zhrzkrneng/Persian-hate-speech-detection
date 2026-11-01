"""Every AUROC in docs/paper/results.md must still match runs/.

The manuscript is generated from the run records, so it cannot drift. This
document is written by hand, and it drifted: after the table was rebuilt on one
library version, nine of its twelve rows still carried the old numbers, and
nothing said so. It is the repository's own account of its results, so a reader
comparing it against the paper would have found two different tables.

This test recomputes each row rather than comparing against a stored copy, so
it fails the moment a cell is re-run and the document is not updated.
"""

from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "paper" / "results.md"
HMC = ROOT / "runs" / "hmc"

ENCODERS = {
    "CLIP ViT-L/14": "openai__clip-vit-large-patch14",
    "CLIP ViT-B/32": "openai__clip-vit-base-patch32",
    "jina-clip-v2": "jinaai__jina-clip-v2",
    "CLIPfa": "CLIPfa",
}
ARCHITECTURES = {"Hate-CLIPper": "hateclipper", "MemeCLIP": "memeclip"}

ROW = re.compile(
    r"^\| (CLIP ViT-[LB]/\d+|jina-clip-v2|CLIPfa) \| (Hate-CLIPper|MemeCLIP) \| "
    r"(concat|align) \| \*{0,2}([\d.]+) ± ([\d.]+)"
)


def rows() -> list[tuple[str, str, str, float, float]]:
    if not DOC.exists():
        return []
    found = []
    for line in DOC.read_text(encoding="utf-8").splitlines():
        match = ROW.match(line)
        if match:
            encoder, architecture, fusion, mean, sd = match.groups()
            found.append((encoder, architecture, fusion, float(mean), float(sd)))
    return found


def measured(
    encoder: str, architecture: str, fusion: str
) -> tuple[float, float] | None:
    directory = HMC / ENCODERS[encoder] / ARCHITECTURES[architecture] / fusion
    files = sorted(directory.glob("seed*.json"))
    if len(files) < 2:
        return None
    values = [
        json.loads(p.read_text(encoding="utf-8"))["test"]["auroc"] * 100 for p in files
    ]
    return round(st.mean(values), 2), round(st.stdev(values), 2)


def test_the_document_has_a_table_to_check() -> None:
    assert len(rows()) >= 4, "no result rows found; this test would pass vacuously"


@pytest.mark.parametrize(
    "row", rows(), ids=lambda r: f"{r[0]}-{r[1]}-{r[2]}".replace("/", "")
)
def test_each_row_matches_the_run_records(row) -> None:
    encoder, architecture, fusion, mean, sd = row
    actual = measured(encoder, architecture, fusion)
    if actual is None:
        pytest.skip(f"{encoder}/{architecture}/{fusion} has no complete cell here")
    assert (mean, sd) == actual, (
        f"{encoder} {architecture}/{fusion}: the document says "
        f"{mean:.2f} ± {sd:.2f}, the records say {actual[0]:.2f} ± {actual[1]:.2f}. "
        "Re-run the cell or update the document; do not leave them disagreeing."
    )
