"""How many tokens each encoder spends on a Persian caption, and what it loses.

    python -m scripts.measure_tokenisation --limit 1000

This exists because a measurement that looked like a technicality turned out to
be part of the result. The retrieval presets inherit `max_text_length = 77`,
OpenAI CLIP's hard context limit -- 77 is the length of its learned positional
embedding table, not a configurable budget -- while the classification configs
for jina-clip-v2 set 512. Before treating that difference as immaterial it was
measured, and it is immaterial for jina and not for CLIP:

    openai/clip-vit-base-patch32   en  max  33 tokens,   0 of 2000 over 77
    openai/clip-vit-base-patch32   fa  max 145 tokens, 229 of 2000 over 77
    openai/clip-vit-large-patch14  en  max  33 tokens,   0 of 2000 over 77
    openai/clip-vit-large-patch14  fa  max 145 tokens, 229 of 2000 over 77
    jinaai/jina-clip-v2            en  max  36 tokens,   0 of 2000 over 77
    jinaai/jina-clip-v2            fa  max  52 tokens,   0 of 2000 over 77

So jina's retrieval numbers are unaffected by the limit and need no re-run,
and CLIP truncates about one Persian caption in nine.

The means say the same thing without the limit involved at all. For captions
of the same images, OpenAI CLIP spends 50.51 tokens on a Persian one against
12.60 on an English one -- 4.0 times as many -- where jina spends 18.47 against
14.74, a factor of 1.25.

That truncation is not a confound to apologise for. It is forced by CLIP's own
architecture, and its cause is the same thing the retrieval score is measuring:
a byte-pair vocabulary fit to English fragments Persian into many short pieces,
so the same amount of description costs several times as many tokens. CLIP's
Persian retrieval sits at chance, which truncating a ninth of the queries
cannot explain -- but the token cost is independent evidence for why, and it is
measured here rather than asserted.

Crossmodal-3600's captions are written independently per language, not
translated, so the comparison is between captions of the same images, not
between renderings of the same sentence.

Writes runs/tokenisation.json.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.xm3600 import Crossmodal3600  # noqa: E402

# (model id, trust_remote_code). The tokenizer alone is downloaded; no weights.
TOKENIZERS = [
    ("openai/clip-vit-base-patch32", False),
    ("openai/clip-vit-large-patch14", False),
    ("jinaai/jina-clip-v2", True),
]
CLIP_CONTEXT = 77


def captions(root: Path, language: str, limit: int | None) -> list[str]:
    records = Crossmodal3600(root, language).with_images()
    if limit:
        records = records[:limit]
    return [caption for record in records for caption in record.captions]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/xm3600")
    parser.add_argument("--languages", nargs="+", default=["en", "fa"])
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--out", default="runs/tokenisation.json")
    args = parser.parse_args()

    # Imported here, not at module scope: tests/test_imports.py walks scripts/
    # under CI's small dependency set, and transformers is not installed there.
    from transformers import AutoTokenizer

    root = Path(args.root)
    texts = {
        language: captions(root, language, args.limit) for language in args.languages
    }
    for language, items in texts.items():
        print(f"{language}: {len(items)} captions")

    rows = []
    for model, remote in TOKENIZERS:
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=remote)
        for language, items in texts.items():
            lengths = [len(tokenizer(text)["input_ids"]) for text in items]
            over = sum(1 for n in lengths if n > CLIP_CONTEXT)
            rows.append(
                {
                    "model": model,
                    "language": language,
                    "captions": len(lengths),
                    "mean_tokens": round(statistics.mean(lengths), 2),
                    "median_tokens": statistics.median(lengths),
                    "max_tokens": max(lengths),
                    "over_clip_context": over,
                    "over_clip_context_pct": round(100 * over / len(lengths), 2),
                }
            )
            row = rows[-1]
            print(
                f"{model:34s} {language}  mean {row['mean_tokens']:6.2f}  "
                f"max {row['max_tokens']:4d}  over {CLIP_CONTEXT}: "
                f"{over:4d} ({row['over_clip_context_pct']:.1f}%)"
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"clip_context": CLIP_CONTEXT, "limit": args.limit, "rows": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
