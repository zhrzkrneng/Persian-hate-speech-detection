"""The manuscript carries no Arabic diacritics.

The author asked for them to go: the ezafe kasra and its relatives make the
text look formal and slow it down, and Persian readers do not need them. They
were removed from the source rather than from the output, so this guards the
source.

Five words became ambiguous without their mark and were rewritten instead of
re-marked -- نُه (nine) reads as نه (not), می‌بُرد (truncates) as می‌برد
(carries), بُعد (dimension) as بعد (after), مُد (modality) as مد (fashion) and
کَش (cache) as کش. Those rewrites are the reason this can be a blanket rule.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"

# fathatan, dammatan, kasratan, fatha, damma, kasra, shadda, sukun, maddah,
# hamza above, superscript alef
DIACRITICS = "ًٌٍَُِّْٰٓٔ"
PATTERN = re.compile(f"[{DIACRITICS}]")


def sources() -> list[Path]:
    return sorted(PAPER.glob("*.js"))


def test_the_sources_exist() -> None:
    assert sources(), "no manuscript sources found"


@pytest.mark.parametrize("path", sources(), ids=lambda p: p.name)
def test_no_arabic_diacritics(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    found = PATTERN.findall(text)
    if not found:
        return
    lines = [
        f"line {i + 1}: {line.strip()[:70]}"
        for i, line in enumerate(text.split("\n"))
        if PATTERN.search(line)
    ]
    raise AssertionError(
        f"{len(found)} Arabic diacritic(s) in {path.name}:\n" + "\n".join(lines[:5])
    )


def test_the_words_that_needed_rewriting_stayed_rewritten() -> None:
    """Dropping the mark on these made them mean something else."""
    joined = "\n".join(p.read_text(encoding="utf-8") for p in sources())
    assert "نه فراپارامتر" not in joined, "«نه فراپارامتر» reads as a negation"
    assert "نه زیربخش" not in joined, "«نه زیربخش» reads as a negation"
    assert "را می‌برد،" not in joined, "«می‌برد» should be «قطع می‌کند» here"
    assert "بعد‌به‌بعد" not in joined, "«بعد‌به‌بعد» should be «مؤلفه‌به‌مؤلفه»"
