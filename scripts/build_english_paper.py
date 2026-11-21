"""Generate the English ACL-format manuscript from the run records.

    python -m scripts.build_english_paper
    python -m scripts.build_english_paper --check

The Persian manuscript is generated so that no number in it is typed, and the
one number that *was* typed -- "five seeds" in the English abstract -- is
exactly the one that went stale and survived two rounds of review. The English
paper is built the same way for the same reason.

Placeholders in `english/main.tex.in` are written `<<name>>`, not `$name` or
`{name}`, because LaTeX already uses `$` and `{}` and a template language that
collides with its host is a trap. Every placeholder must resolve; an unknown
one is an error rather than a silent empty string, and a leftover `<<` in the
output fails the build.

This writes `.tex` and does not compile: there is no TeX toolchain here, and
installing one to typeset a paper that Overleaf will typeset for free is not
a good trade. `english/README.md` says how to build it.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NUMBERS = Path("runs/paper_numbers.json")
TEMPLATE = Path("english/main.tex.in")
OUTPUT = Path("english/main.tex")

ENCODERS = ("CLIP ViT-L/14", "CLIP ViT-B/32", "jina-clip-v2", "CLIPfa")
SLUG = {
    "CLIP ViT-L/14": "l14",
    "CLIP ViT-B/32": "b32",
    "jina-clip-v2": "jina",
    "CLIPfa": "clipfa",
}
PLACEHOLDER = re.compile(r"<<([a-zA-Z0-9_]+)>>")


def fmt(value: float | None, digits: int = 2) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def signed(value: float, digits: int = 2) -> str:
    """LaTeX minus, not a hyphen: $-$ keeps the glyph and the spacing right."""
    return ("+" if value >= 0 else "$-$") + f"{abs(value):.{digits}f}"


def p_value(value: float) -> str:
    return "$<$0.001" if value < 0.001 else f"{value:.3f}"


def stars(value: float) -> str:
    if value < 0.001:
        return "***"
    if value < 0.01:
        return "**"
    if value < 0.05:
        return "*"
    return "n.s."


def substitutions(numbers: dict) -> dict[str, str]:
    """Every value the template may ask for, keyed by placeholder name."""
    out: dict[str, str] = {}
    cells = numbers["cells"]
    benefit = numbers["fusion_benefit"]
    architecture = numbers["architecture"]
    interaction = numbers["interaction"]
    retrieval = numbers["retrieval"]
    multilingual = numbers["multilingual"]
    multiplicity = numbers["multiplicity"]

    out["seeds"] = str(benefit["CLIP ViT-L/14"]["n"])

    for label in ENCODERS:
        slug = SLUG[label]
        for key, name in (
            ("hateclipper/concat", "concat"),
            ("hateclipper/align", "align"),
            ("memeclip/align", "memeclip"),
        ):
            cell = cells[label][key]["test"]
            for metric, short in (
                ("auroc", "auroc"),
                ("accuracy", "acc"),
                ("macro_f1", "f1"),
            ):
                out[f"{slug}_{name}_{short}"] = fmt(cell[metric]["mean"])
                out[f"{slug}_{name}_{short}_sd"] = fmt(cell[metric]["sd"])

        row = benefit[label]
        out[f"ben_{slug}"] = signed(row["mean"])
        out[f"ben_{slug}_sd"] = fmt(row["sd"])
        out[f"ben_{slug}_p"] = p_value(row["p"])
        out[f"ben_{slug}_t"] = fmt(row["t"])
        out[f"ben_{slug}_d"] = signed(row["d"])
        out[f"ben_{slug}_ci"] = f"[{signed(row['ci_low'])}, {signed(row['ci_high'])}]"
        out[f"ben_{slug}_stars"] = stars(row["p"])

        arch = architecture[label]
        out[f"arch_{slug}"] = signed(arch["mean"])
        out[f"arch_{slug}_p"] = p_value(arch["p"])
        out[f"arch_{slug}_d"] = signed(arch["d"])

        langs = retrieval.get(label, {})
        for language in ("en", "fa", "de", "es", "hi", "zh"):
            entry = langs.get(language)
            if entry:
                out[f"ret_{slug}_{language}"] = fmt(entry["t2i_r1"])
                out[f"ret_{slug}_{language}_i2t"] = fmt(entry["i2t_r1"])
        if langs.get("en"):
            out["chance"] = fmt(langs["en"]["chance"])

    key = "CLIP ViT-L/14 - CLIP ViT-B/32"
    control = interaction[key]
    out["control_mean"] = signed(control["mean"])
    out["control_p"] = p_value(control["p"])
    out["control_ci"] = f"[{signed(control['ci_low'])}, {signed(control['ci_high'])}]"
    out["control_gap"] = fmt(
        cells["CLIP ViT-L/14"]["hateclipper/concat"]["test"]["auroc"]["mean"]
        - cells["CLIP ViT-B/32"]["hateclipper/concat"]["test"]["auroc"]["mean"]
    )

    equivalence = numbers.get("equivalence", {})
    pairs = equivalence.get("pairs", {})
    if key in pairs:
        out["control_tost"] = fmt(pairs[key]["p_tost"], 3)
        out["control_verdict"] = pairs[key]["verdict"]
    if "margin" in equivalence:
        out["control_margin"] = fmt(equivalence["margin"], 1)

    overall = multilingual["overall"]
    out["ml_n"] = str(overall["n"])
    out["ml_r"] = fmt(overall["pearson_r"], 3)
    out["ml_p"] = fmt(overall["pearson_p"], 3)
    out["ml_rho"] = fmt(overall["spearman_rho"], 3)
    out["ml_rho_p"] = fmt(overall["spearman_p"], 3)
    out["ml_reliability"] = fmt(multilingual["reliability"]["reliability"], 2)
    out["ml_between"] = fmt(multilingual["reliability"]["between_sd"], 2)
    out["ml_within"] = fmt(multilingual["reliability"]["mean_sem"], 2)
    out["ml_bound"] = fmt(multilingual["attenuation_bound"], 3)
    within = multilingual["within_language"]
    slopes = sorted(v["pearson_r"] for v in within.values())
    out["ml_within_min"] = fmt(slopes[0], 2)
    out["ml_within_max"] = fmt(slopes[-1], 2)
    out["ml_within_n"] = str(len(within))
    out["ml_seeds"] = str(len(multilingual["cells"][0]["seeds"]))

    out["mult_family"] = str(multiplicity["family_size"])
    out["mult_holds"] = str(multiplicity["n_holds"])
    out["mult_fails"] = str(multiplicity["family_size"] - multiplicity["n_holds"])
    tight = multiplicity["tests"][multiplicity["tightest"]]
    out["mult_tight_p"] = fmt(tight["p"], 4)
    out["mult_tight_threshold"] = fmt(tight["threshold"], 4)
    out["mult_tight_margin"] = fmt(tight["margin"], 4)

    tok = numbers.get("tokenisation", {})
    for row in tok.get("rows", []):
        if row["model"] == "openai/clip-vit-base-patch32":
            out[f"tok_clip_{row['language']}"] = fmt(row["mean_tokens"])
            out[f"tok_clip_{row['language']}_cut"] = fmt(
                row["over_clip_context_pct"], 1
            )
        if row["model"] == "jinaai/jina-clip-v2":
            out[f"tok_jina_{row['language']}"] = fmt(row["mean_tokens"])
    return out


def render(template: str, values: dict[str, str]) -> tuple[str, list[str]]:
    missing: list[str] = []

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            missing.append(name)
            return match.group(0)
        return values[name]

    return PLACEHOLDER.sub(replace, template), missing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="report drift without writing"
    )
    args = parser.parse_args()

    if not TEMPLATE.exists():
        raise SystemExit(f"{TEMPLATE} not found")
    if not NUMBERS.exists():
        raise SystemExit(f"{NUMBERS} not found; run python -m scripts.paper_numbers")

    numbers = json.loads(NUMBERS.read_text(encoding="utf-8"))
    rendered, missing = render(
        TEMPLATE.read_text(encoding="utf-8"), substitutions(numbers)
    )
    if missing:
        unique = sorted(set(missing))
        raise SystemExit(
            f"{len(unique)} placeholder(s) have no value: {', '.join(unique)}"
        )
    if "<<" in rendered:
        raise SystemExit("a '<<' survived rendering; the template is malformed")

    if args.check:
        current = OUTPUT.read_text(encoding="utf-8") if OUTPUT.exists() else None
        if current == rendered:
            print(f"{OUTPUT} matches the run records")
            return
        raise SystemExit(f"{OUTPUT} is stale; run without --check to rewrite it")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT} ({len(rendered.splitlines())} lines)")


if __name__ == "__main__":
    main()
