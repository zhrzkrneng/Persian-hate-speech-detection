"""Structural checks on the English manuscript, standing in for a compiler.

There is no TeX toolchain in this container and installing one to typeset a
paper Overleaf will typeset for free is a poor trade, so the compiler cannot
be the thing that catches a broken file. These checks catch the faults that
actually break a LaTeX build -- unbalanced braces, unclosed environments, a
table row whose column count disagrees with its preamble, a stray placeholder
-- and they run in milliseconds.

They do not prove the document compiles. They prove it is not broken in the
ways a generated file gets broken.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TEMPLATE = ROOT / "english" / "main.tex.in"
RENDERED = ROOT / "english" / "main.tex"


@pytest.fixture(scope="module")
def tex() -> str:
    if not RENDERED.exists():
        pytest.skip("english/main.tex not generated")
    return RENDERED.read_text(encoding="utf-8")


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.split("\n"))


def test_no_placeholder_survives(tex: str) -> None:
    assert "<<" not in tex and ">>" not in tex, "an unrendered placeholder remains"


def test_no_number_is_typed_in_the_template() -> None:
    """A statistic in the template is a statistic that can go stale."""
    body = strip_comments(TEMPLATE.read_text(encoding="utf-8"))
    # Strip the preamble, where font sizes and package options are numbers.
    body = body.split(r"\begin{document}", 1)[-1]
    # A decimal with two or more places is a statistic; integers (section
    # numbers, the 77-token window, the 300 memes, the 50 chance level) are
    # prose and are allowed.
    suspicious = re.findall(r"(?<![\w.])\d+\.\d{2,}", body)
    assert not suspicious, f"typed statistics in the template: {suspicious}"


def test_braces_balance(tex: str) -> None:
    body = strip_comments(tex)
    depth = 0
    for i, char in enumerate(body):
        if char in "{}" and (i == 0 or body[i - 1] != "\\"):
            depth += 1 if char == "{" else -1
            assert depth >= 0, f"unmatched closing brace near offset {i}"
    assert depth == 0, f"{depth} unclosed brace(s)"


def test_math_mode_closes(tex: str) -> None:
    body = strip_comments(tex)
    dollars = [
        i for i, c in enumerate(body) if c == "$" and (i == 0 or body[i - 1] != "\\")
    ]
    assert len(dollars) % 2 == 0, f"{len(dollars)} unescaped $ -- math mode is open"


def test_environments_match(tex: str) -> None:
    body = strip_comments(tex)
    stack: list[str] = []
    for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", body):
        if kind == "begin":
            stack.append(name)
        else:
            assert stack, f"\\end{{{name}}} with nothing open"
            opened = stack.pop()
            assert opened == name, f"\\begin{{{opened}}} closed by \\end{{{name}}}"
    assert not stack, f"unclosed environment(s): {stack}"


def test_table_rows_match_their_preamble(tex: str) -> None:
    body = strip_comments(tex)
    for preamble, content in re.findall(
        r"\\begin\{tabular\}\{([^}]*)\}(.*?)\\end\{tabular\}", body, re.DOTALL
    ):
        columns = len(re.findall(r"[lcr]", preamble))
        for row in content.split(r"\\"):
            row = re.sub(r"\\(toprule|midrule|bottomrule|cmidrule)[^\n]*", "", row)
            if not row.strip():
                continue
            # & inside math or escaped does not separate a column
            cells = len(re.findall(r"(?<!\\)&", row)) + 1
            assert cells == columns, (
                f"row has {cells} cells for a {columns}-column preamble: "
                f"{row.strip()[:70]}"
            )


def test_every_citation_is_defined(tex: str) -> None:
    bib = ROOT / "english" / "references.bib"
    if not bib.exists():
        pytest.skip("references.bib not present")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib.read_text(encoding="utf-8")))
    cited = set()
    for group in re.findall(r"\\cite[a-z]*\{([^}]+)\}", strip_comments(tex)):
        cited.update(k.strip() for k in group.split(","))
    missing = cited - keys
    assert not missing, f"cited but not in references.bib: {sorted(missing)}"


def test_required_acl_sections_present(tex: str) -> None:
    """ACL requires a Limitations section; it is unnumbered and page-free."""
    assert r"\section*{Limitations}" in tex, "ACL requires a Limitations section"
    assert r"\begin{abstract}" in tex


def test_generator_is_idempotent() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.build_english_paper", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
