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


def test_monotone_is_not_confused_with_uniform() -> None:
    """یکنوا is monotone; یکنواخت is uniform. They are different properties.

    A readability pass replaced یکنوا with یکنواخت in two places, which left
    the manuscript using two words for one concept, one of them meaning
    something else. The one legitimate یکنواخت is "uniformly more powerful",
    the standard phrase for Holm against Bonferroni.
    """
    allowed = "به‌طور یکنواخت از بونفرونی"
    for path in sources():
        text = path.read_text(encoding="utf-8").replace(allowed, "")
        assert "یکنواخت" not in text, (
            f"{path.name} uses یکنواخت (uniform) where یکنوا (monotone) is meant"
        )


def test_no_invented_compound() -> None:
    """A word coined by compressing a phrase is not a word.

    "داده‌ی موجودنبوده" came out of a pass meant to make the text easier to
    read, replacing "داده‌ای که وجود ندارد", which was already clear.
    """
    joined = "\n".join(p.read_text(encoding="utf-8") for p in sources())
    for coinage in ("موجودنبوده", "آزمودهنشده", "اندازهنگرفته"):
        assert coinage not in joined, f"invented compound: {coinage}"


def test_one_word_per_concept() -> None:
    """Two words for one idea means one of them arrived by accident.

    The readability pass introduced نوفه beside the نویز the manuscript
    already used, and یکنواخت beside یکنوا. Neither was a decision.
    """
    joined = "\n".join(p.read_text(encoding="utf-8") for p in sources())
    assert "نوفه" not in joined, "noise is نویز in this manuscript, not نوفه"


def test_no_zero_width_joiner_left_mid_word() -> None:
    """A blind replacement inside a word leaves the old ezafe behind.

    Replacing نوفه with نویز turned نوفه‌ی into نویز‌ی -- a ZWNJ sitting
    between a noun and a bare ی, which is not a form Persian has.
    """
    broken = re.compile(r"نویز\u200c")
    for path in sources():
        hit = broken.search(path.read_text(encoding="utf-8"))
        assert not hit, f"{path.name} has a ZWNJ stranded after نویز"


def test_additions_are_said_plainly_and_completely() -> None:
    """افزودن needs a «به», and in this manuscript it is اضافه کردن anyway.

    "سه چیز می‌افزاید" was missing its complement -- three things added *to
    what?* -- and the same sentence appeared correctly elsewhere, so the
    manuscript was both incomplete in one place and inconsistent across two.
    اضافه کردن is the plainer verb and carries its preposition visibly.
    """
    joined = "\n".join(p.read_text(encoding="utf-8") for p in sources())
    for form in ("می‌افزاید", "نمی‌افزاید", "می‌افزایند"):
        assert form not in joined, f"«{form}» hides its preposition; use اضافه می‌کند"
