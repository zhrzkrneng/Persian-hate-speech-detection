"""Collect every number the paper quotes into one file, from runs/.

    python -m scripts.paper_numbers
    python -m scripts.paper_numbers --fallback-ref HEAD~4

The paper is generated from this file rather than typed, because a number
typed into prose is a copy, and a copy drifts. Everything the manuscript
states -- each cell's mean and spread, each paired test, each interaction,
each retrieval recall and its chance level, the tokenizer measurement -- is
computed here from the committed per-seed records and written once.

`--fallback-ref` fills a cell that is missing from the working tree from a git
revision, and records in the output which cells came from where. It exists for
one situation: a rebuild is in flight, a draft is wanted before it lands, and a
draft whose provenance is silent would be worse than no draft. Every fallback
is listed under `provisional` and the generator marks the document.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import subprocess
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HMC = Path("runs/hmc")
RETRIEVAL = Path("runs/retrieval")

ENCODERS = {
    "CLIP ViT-L/14": ("openai__clip-vit-large-patch14", "clip-l14"),
    "CLIP ViT-B/32": ("openai__clip-vit-base-patch32", "clip-b32"),
    "jina-clip-v2": ("jinaai__jina-clip-v2", "jina-v2"),
    "CLIPfa": ("CLIPfa", "clipfa"),
}
CELLS = (("hateclipper", "concat"), ("hateclipper", "align"), ("memeclip", "align"))


def from_git(path: Path, ref: str) -> dict | None:
    try:
        text = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return None
    return json.loads(text)


def read_cell(slug: str, arch: str, fusion: str, ref: str | None) -> tuple[dict, bool]:
    """{seed: metrics} and whether it came from the fallback revision."""
    directory = HMC / slug / arch / fusion
    files = sorted(directory.glob("seed*.json"))
    if files:
        return {
            json.loads(p.read_text())["seed"]: json.loads(p.read_text()) for p in files
        }, False
    if not ref:
        return {}, False

    found = {}
    for seed in range(5):
        payload = from_git(directory / f"seed{seed}.json", ref)
        if payload:
            found[payload["seed"]] = payload
    return found, bool(found)


def spread(values: list[float]) -> dict:
    return {
        "mean": round(st.mean(values), 2),
        "sd": round(st.stdev(values), 2) if len(values) > 1 else None,
        "n": len(values),
    }


def test(diffs: list[float]) -> dict:
    from scipy import stats

    result = stats.ttest_1samp(diffs, 0.0)
    return {
        "mean": round(st.mean(diffs), 2),
        "sd": round(st.stdev(diffs), 2),
        "t": round(float(result.statistic), 2),
        "p": round(float(result.pvalue), 3),
        "n": len(diffs),
        "per_seed": [round(d, 2) for d in diffs],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fallback-ref", help="git revision to fill missing cells from"
    )
    parser.add_argument("--out", default="runs/paper_numbers.json")
    args = parser.parse_args()

    cells: dict = {}
    provisional: list[str] = []
    for label, (slug, _) in ENCODERS.items():
        cells[label] = {}
        for arch, fusion in CELLS:
            rows, fell_back = read_cell(slug, arch, fusion, args.fallback_ref)
            if not rows:
                raise SystemExit(
                    f"no records for {label} {arch}/{fusion}; run it, or pass "
                    "--fallback-ref to fill it from a revision that has it"
                )
            if fell_back:
                provisional.append(f"{label} {arch}/{fusion}")
            cells[label][f"{arch}/{fusion}"] = {
                "seeds": sorted(rows),
                "test": {
                    metric: spread([r["test"][metric] * 100 for r in rows.values()])
                    for metric in ("auroc", "accuracy", "macro_f1")
                },
                "val_auroc": spread([r["val"]["auroc"] * 100 for r in rows.values()]),
                "auroc_by_seed": {
                    s: rows[s]["test"]["auroc"] * 100 for s in sorted(rows)
                },
            }

    # fusion benefit, align - concat, paired per seed
    benefit = {}
    for label in ENCODERS:
        a = cells[label]["hateclipper/align"]["auroc_by_seed"]
        c = cells[label]["hateclipper/concat"]["auroc_by_seed"]
        seeds = sorted(set(a) & set(c))
        benefit[label] = test([a[s] - c[s] for s in seeds])

    interaction = {}
    for left, right in combinations(ENCODERS, 2):
        interaction[f"{left} - {right}"] = test(
            [
                x - y
                for x, y in zip(
                    benefit[left]["per_seed"], benefit[right]["per_seed"], strict=True
                )
            ]
        )

    architecture = {}
    for label in ENCODERS:
        m = cells[label]["memeclip/align"]["auroc_by_seed"]
        h = cells[label]["hateclipper/align"]["auroc_by_seed"]
        seeds = sorted(set(m) & set(h))
        architecture[label] = test([m[s] - h[s] for s in seeds])

    encoder_gap = {}
    for fusion in ("align", "concat"):
        for left, right in combinations(ENCODERS, 2):
            a = cells[left][f"hateclipper/{fusion}"]["auroc_by_seed"]
            b = cells[right][f"hateclipper/{fusion}"]["auroc_by_seed"]
            seeds = sorted(set(a) & set(b))
            encoder_gap[f"{left} - {right} ({fusion})"] = test(
                [a[s] - b[s] for s in seeds]
            )

    retrieval = {}
    for label, (_, preset) in ENCODERS.items():
        retrieval[label] = {}
        for language in ("en", "fa", "de", "es", "hi", "zh"):
            path = RETRIEVAL / f"{preset}_{language}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            retrieval[label][language] = {
                "t2i_r1": payload["results"]["text_to_image"]["R@1"],
                "i2t_r1": payload["results"]["image_to_text"]["R@1"],
                "images": payload["images"],
                "chance": round(100 / payload["images"], 2),
                "versions": payload.get("versions", {}),
            }

    tokenisation = {}
    path = Path("runs/tokenisation.json")
    if path.exists():
        tokenisation = json.loads(path.read_text())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "cells": cells,
                "fusion_benefit": benefit,
                "interaction": interaction,
                "architecture": architecture,
                "encoder_gap": encoder_gap,
                "retrieval": retrieval,
                "tokenisation": tokenisation,
                "provisional": provisional,
                "fallback_ref": args.fallback_ref,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if provisional:
        print(f"PROVISIONAL: {len(provisional)} cell(s) taken from {args.fallback_ref}")
        for item in provisional:
            print(f"  {item}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
