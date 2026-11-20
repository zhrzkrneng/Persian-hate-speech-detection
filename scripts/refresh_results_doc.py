"""Rewrite the numbers in docs/paper/results.md from the run records.

    python -m scripts.refresh_results_doc
    python -m scripts.refresh_results_doc --check

The manuscript is generated, so it cannot drift. This document was the last
artefact still maintained by hand, and it drifted twice: once after the table
was rebuilt on one library version, and again the moment a seed sweep started
landing new runs. tests/test_results_doc.py catches it, which is why it is
caught at all -- but catching is not fixing, and a guard that can only be
satisfied by retyping numbers invites someone to edit the test instead.

This rewrites every result row in place from the per-seed records, touching
nothing else in the file: the prose, the structure and the ordering are left
exactly as written. `--check` reports what would change without writing, which
is what to run before a commit.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DOC = Path("docs/paper/results.md")
HMC = Path("runs/hmc")

ENCODERS = {
    "CLIP ViT-L/14": "openai__clip-vit-large-patch14",
    "CLIP ViT-B/32": "openai__clip-vit-base-patch32",
    "jina-clip-v2": "jinaai__jina-clip-v2",
    "CLIPfa": "CLIPfa",
}
ARCHITECTURES = {"Hate-CLIPper": "hateclipper", "MemeCLIP": "memeclip"}

# | encoder | architecture | fusion | AUROC | accuracy | macro F1 |
ROW = re.compile(
    r"^\| (CLIP ViT-[LB]/\d+|jina-clip-v2|CLIPfa) \| (Hate-CLIPper|MemeCLIP) \| "
    r"(concat|align) \| (\*{0,2})([\d.]+ ± [\d.]+)\4 \| [\d.]+ ± [\d.]+ \| "
    r"[\d.]+ ± [\d.]+ \|$"
)


# The statistics tables in section 3. Each row names a comparison in its first
# cell and carries delta, sd, t, p and stars; the row label is the key, so the
# table's ordering and any emphasis around a label survive a refresh.
STAT_ROW = re.compile(
    r"^\| (\*{0,2})([^|*]+?)\1 \| \*{0,2}[+\u2212\-][\d.]+\*{0,2} \| [\d.]+ \| "
    r"[+\u2212\-]?[\d.]+ \| \*{0,2}[<\d.]+\*{0,2} \| \*{0,4}(?:n\.s\.)?\*{0,4} \|$"
)


def _signed(value: float) -> str:
    """Minus sign, not hyphen: the document uses U+2212 throughout."""
    return f"{'+' if value >= 0 else chr(0x2212)}{abs(value):.2f}"


def _stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def statistics() -> dict[str, dict]:
    """Every comparison the section 3 tables quote, keyed by its row label."""
    from scipy import stats

    def cell(encoder: str, architecture: str, fusion: str) -> dict[int, float]:
        directory = HMC / ENCODERS[encoder] / architecture / fusion
        return {
            json.loads(p.read_text(encoding="utf-8"))["seed"]: json.loads(
                p.read_text(encoding="utf-8")
            )["test"]["auroc"]
            * 100
            for p in sorted(directory.glob("seed*.json"))
        }

    def test(diffs: list[float]) -> dict:
        result = stats.ttest_1samp(diffs, 0.0)
        return {
            "mean": st.mean(diffs),
            "sd": st.stdev(diffs),
            "t": float(result.statistic),
            "p": float(result.pvalue),
            "df": len(diffs) - 1,
        }

    def paired(a: dict[int, float], b: dict[int, float]) -> list[float]:
        return [a[s] - b[s] for s in sorted(set(a) & set(b))]

    out: dict[str, dict] = {}
    benefit = {}
    for encoder in ENCODERS:
        diffs = paired(
            cell(encoder, "hateclipper", "align"),
            cell(encoder, "hateclipper", "concat"),
        )
        benefit[encoder] = diffs
        out[encoder] = test(diffs)
        out[f"architecture: MemeCLIP \u2212 Hate-CLIPper, {encoder}"] = test(
            paired(
                cell(encoder, "memeclip", "align"),
                cell(encoder, "hateclipper", "align"),
            )
        )
    for left, right in combinations(ENCODERS, 2):
        pairs = zip(benefit[left], benefit[right], strict=True)
        out[f"{left} \u2212 {right}"] = test([x - y for x, y in pairs])
    for fusion in ("concat", "align"):
        for left, right in combinations(ENCODERS, 2):
            out[f"encoder: {left} \u2212 {right}, {fusion}"] = test(
                paired(
                    cell(left, "hateclipper", fusion),
                    cell(right, "hateclipper", fusion),
                )
            )
    return out


def measured(encoder: str, architecture: str, fusion: str) -> dict[str, str] | None:
    directory = HMC / ENCODERS[encoder] / ARCHITECTURES[architecture] / fusion
    files = sorted(directory.glob("seed*.json"))
    if len(files) < 2:
        return None
    rows = [json.loads(p.read_text(encoding="utf-8"))["test"] for p in files]
    out = {}
    for metric in ("auroc", "accuracy", "macro_f1"):
        values = [r[metric] * 100 for r in rows]
        out[metric] = f"{st.mean(values):.2f} ± {st.stdev(values):.2f}"
    return out


DF_HEADER = re.compile(r"\*t\*\((\d+)\)")

# The multiplicity table: one row per corrected comparison, each ending in the
# procedure's verdict. Four columns and a holds/fails cell keep it apart from
# every other table in the document.
HOLM_ROW = re.compile(
    r"^\| (\*{0,2})([^|*]+?)\1 \| \*{0,2}[<\d.]+\*{0,2} \| [\d.]+ \| "
    r"\*{0,2}(?:holds|fails)\*{0,2} \|$"
)


HOLM_HEADER = "| comparison | *p* | Holm threshold | |"


def holm_line(label: str, row: dict) -> str:
    shown = "<0.001" if row["p"] < 0.001 else f"{row['p']:.3f}"
    mark = "**" if row["holds"] else ""
    verdict = "holds" if row["holds"] else "**fails**"
    return f"| {label} | {mark}{shown}{mark} | {row['threshold']:.4f} | {verdict} |"


def rewrite_holm_table(text: str) -> tuple[str, list[str]]:
    """Replace the whole multiplicity table, not its rows one by one.

    The family can gain or lose a test, and a generator that only rewrote the
    rows already present would keep reporting the old family size while every
    row in it looked current.
    """
    rows = text.split("\n")
    try:
        start = rows.index(HOLM_HEADER)
    except ValueError:
        return text, []
    end = start + 2  # past the header and the separator
    while end < len(rows) and HOLM_ROW.match(rows[end]):
        end += 1
    rebuilt = [holm_line(label, row) for label, row in multiplicity()]
    if rows[start + 2 : end] == rebuilt:
        return text, []
    return (
        "\n".join(rows[: start + 2] + rebuilt + rows[end:]),
        [f"multiplicity table: {len(rebuilt)} row(s) rewritten"],
    )


def multiplicity() -> list[tuple[str, dict]]:
    """The corrected family, in the order Holm examines it.

    Read from scripts/multiplicity.py rather than recomputed here: the family
    is a decision, and two places deciding it is how a correction quietly
    starts covering a different set of tests than the paper says it does.
    """
    from scripts.multiplicity import ALPHA, family, holm

    tests = family()
    corrected = holm({k: p for k, (_, p) in tests.items()}, ALPHA)
    return sorted(corrected.items(), key=lambda kv: kv[1]["rank"])


def rewrite(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    stats_by_label = statistics()
    text, holm_changes = rewrite_holm_table(text)
    changes.extend(holm_changes)
    # The column header names the degrees of freedom, which is data: it was
    # written as t(4) at five seeds and has to follow the records too.
    degrees = {v["df"] for v in stats_by_label.values()}
    header_df = str(max(degrees)) if len(degrees) == 1 else None
    lines = []
    for line in text.split("\n"):
        if header_df and DF_HEADER.search(line) and line.startswith("| "):
            fixed = DF_HEADER.sub(f"*t*({header_df})", line)
            if fixed != line:
                changes.append(f"degrees of freedom header -> t({header_df})")
            lines.append(fixed)
            continue

        stat = STAT_ROW.match(line)
        if stat and stat.group(2).strip() in stats_by_label:
            emphasis, label = stat.group(1), stat.group(2).strip()
            values = stats_by_label[label]
            star = _stars(values["p"])
            shown = "<0.001" if values["p"] < 0.001 else f"{values['p']:.3f}"
            mark = "**" if values["p"] < 0.05 else ""
            delta = f"{mark}{_signed(values['mean'])}{mark}"
            # U+2212 for a negative t too, to match the rest of the document.
            t_shown = (
                f"{chr(0x2212)}{abs(values['t']):.2f}"
                if values["t"] < 0
                else f"{values['t']:.2f}"
            )
            tail = "**" + star + "**" if emphasis else star
            rebuilt = (
                f"| {emphasis}{label}{emphasis} | {delta} | {values['sd']:.2f} "
                f"| {t_shown} | {mark}{shown}{mark} | {tail} |"
            )
            if rebuilt != line:
                changes.append(f"{label}: statistics refreshed")
            lines.append(rebuilt)
            continue

        match = ROW.match(line)
        if not match:
            lines.append(line)
            continue
        encoder, architecture, fusion, bold, old = match.groups()
        values = measured(encoder, architecture, fusion)
        if values is None:
            lines.append(line)
            continue
        if values["auroc"] != old:
            changes.append(
                f"{encoder} {architecture}/{fusion}: {old} -> {values['auroc']}"
            )
        lines.append(
            f"| {encoder} | {architecture} | {fusion} | "
            f"{bold}{values['auroc']}{bold} | {values['accuracy']} | "
            f"{values['macro_f1']} |"
        )
    return "\n".join(lines), changes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report drift without writing"
    )
    args = parser.parse_args()

    if not DOC.exists():
        raise SystemExit(f"{DOC} not found")
    text = DOC.read_text(encoding="utf-8")
    updated, changes = rewrite(text)

    if not changes:
        print(f"{DOC} matches the run records")
        return
    for change in changes:
        print(change)
    if args.check:
        raise SystemExit(f"\n{len(changes)} row(s) stale; run without --check to fix")
    DOC.write_text(updated, encoding="utf-8")
    print(f"\nrewrote {len(changes)} row(s) in {DOC}")


if __name__ == "__main__":
    main()
