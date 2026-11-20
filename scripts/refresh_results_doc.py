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


def rewrite(text: str) -> tuple[str, list[str]]:
    changes: list[str] = []
    lines = []
    for line in text.split("\n"):
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
