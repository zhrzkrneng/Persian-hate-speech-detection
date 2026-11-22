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
    mean, sd = st.mean(diffs), st.stdev(diffs)
    sem = sd / len(diffs) ** 0.5
    half = stats.t.ppf(0.975, len(diffs) - 1) * sem
    return {
        "mean": round(mean, 2),
        "sd": round(sd, 2),
        "t": round(float(result.statistic), 2),
        "p": round(float(result.pvalue), 3),
        "n": len(diffs),
        # Cohen's d for a paired design, and the interval. A p-value says
        # whether an effect is distinguishable from zero, which enough seeds
        # always achieve; d says how large it is next to the noise, which they
        # do not change.
        "d": round(mean / sd, 2) if sd else None,
        "ci_low": round(mean - half, 2),
        "ci_high": round(mean + half, 2),
        # Unrounded, for the multiplicity correction: Holm compares p with a
        # threshold as small as alpha/m, which three decimals cannot resolve.
        # It is stripped before the file is written -- see strip_exact below.
        "p_exact": float(result.pvalue),
        "per_seed": [round(x, 2) for x in diffs],
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

    # fusion benefit, align - concat, paired per seed. The per-seed diffs are
    # kept unrounded and keyed by seed, because the interaction below is built
    # from them: it was built from the rounded `per_seed` list that test()
    # reports for display, and positionally rather than by seed, so the one
    # statistic behind the central equivalence claim disagreed with
    # scripts/equivalence.py in its last digit -- [-0.32, +0.81] against
    # [-0.31, +0.81] for the same fifteen runs.
    benefit = {}
    benefit_by_seed: dict[str, dict[int, float]] = {}
    for label in ENCODERS:
        a = cells[label]["hateclipper/align"]["auroc_by_seed"]
        c = cells[label]["hateclipper/concat"]["auroc_by_seed"]
        seeds = sorted(set(a) & set(c))
        benefit_by_seed[label] = {s: a[s] - c[s] for s in seeds}
        benefit[label] = test([benefit_by_seed[label][s] for s in seeds])

    interaction = {}
    for left, right in combinations(ENCODERS, 2):
        shared = sorted(set(benefit_by_seed[left]) & set(benefit_by_seed[right]))
        interaction[f"{left} - {right}"] = test(
            [benefit_by_seed[left][s] - benefit_by_seed[right][s] for s in shared]
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

    # The four-point alignment relationship, from scripts/alignment_benefit.py.
    # The limitations section says Pearson looks large while Spearman does not;
    # quoting the two coefficients is more use to a reader than saying so.
    alignment = {}
    path = Path("runs/alignment_benefit.json")
    if path.exists():
        alignment = json.loads(path.read_text())

    # The equivalence results, from the file scripts/equivalence.py writes.
    # The English manuscript quotes the TOST p for the capacity control, and
    # typing it there is how the English abstract came to say "five seeds".
    equivalence = {}
    path = Path("runs/equivalence.json")
    if path.exists():
        equivalence = json.loads(path.read_text())

    # The multilingual test of this study's own prediction. Collected from the
    # same file scripts/multilingual_benefit.py writes, so the manuscript and
    # that script cannot disagree about a result the manuscript reports as a
    # failure -- the one kind of number nobody re-checks.
    multilingual = {}
    path = Path("runs/multilingual_benefit.json")
    if path.exists():
        multilingual = json.loads(path.read_text())

    # Holm-Bonferroni over the fourteen tests the results section reports as
    # findings. The family is fixed in scripts/multiplicity.py, not here, so a
    # claim cannot be moved out of it to survive a threshold.
    from scripts.multiplicity import ALPHA, holm

    primary = {}
    for label, values in benefit.items():
        primary[f"fusion benefit: {label}"] = values["p_exact"]
    for label, values in architecture.items():
        primary[f"architecture: {label}"] = values["p_exact"]
    for key, values in interaction.items():
        left, right = key.split(" - ")
        primary[f"interaction: {left} \u2212 {right}"] = values["p_exact"]
    corrected = holm(primary, ALPHA)

    # An unrounded scipy p-value is not reproducible across environments: the
    # same fifteen runs gave 5.806825246529435e-05 here and ...443e-05 on the
    # 3.10 runner, one ULP apart, which turned the committed file stale and CI
    # red. It is needed to apply Holm and never afterwards, so it does not go
    # in the file. Everything written below is rounded.
    def strip_exact(section: dict) -> dict:
        return {
            k: {f: v for f, v in row.items() if f != "p_exact"}
            for k, row in section.items()
        }

    benefit = strip_exact(benefit)
    architecture = strip_exact(architecture)
    interaction = strip_exact(interaction)
    encoder_gap = strip_exact(encoder_gap)
    survivors = [row for row in corrected.values() if row["holds"]]
    tightest = min(survivors, key=lambda r: r["margin"]) if survivors else None
    multiplicity = {
        "alpha": ALPHA,
        "family_size": len(primary),
        "n_holds": len(survivors),
        "tests": {
            label: {
                "rank": row["rank"],
                "p": round(row["p"], 4),
                "threshold": round(row["threshold"], 4),
                "margin": round(row["margin"], 4),
                "holds": row["holds"],
            }
            for label, row in corrected.items()
        },
        "tightest": (
            next(k for k, v in corrected.items() if v is tightest) if tightest else None
        ),
        "tightest_margin": round(tightest["margin"], 4) if tightest else None,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "cells": cells,
                "fusion_benefit": benefit,
                "interaction": interaction,
                "architecture": architecture,
                "multiplicity": multiplicity,
                "encoder_gap": encoder_gap,
                "retrieval": retrieval,
                "tokenisation": tokenisation,
                "multilingual": multilingual,
                "equivalence": equivalence,
                "alignment_benefit": alignment,
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
