"""Zero-shot retrieval on Crossmodal-3600, to choose an encoder on evidence.

    python -m scripts.eval_retrieval --encoder openai/clip-vit-large-patch14
    python -m scripts.eval_retrieval --encoder jinaai/jina-clip-v2 --trust-remote-code
    python -m scripts.eval_retrieval --encoder clipfa

The question this answers is narrow and the whole Persian plan rests on it:
does a given encoder put Persian text and the matching image near each other?
An encoder that cannot do that will not support a Persian meme classifier
either, whatever its model card claims about language coverage.

Protocol, the standard one for this benchmark:

- text -> image: every caption is a query; a hit is its own image in the top k.
- image -> text: every image is a query; a hit is any of its own captions in
  the top k.

English is worth running as a control. A multilingual encoder that scores well
on English and badly on Persian has a language problem; one that scores badly
on both has a different problem, and the two should not be confused.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.config import EncoderConfig  # noqa: E402
from src.data.xm3600 import Crossmodal3600  # noqa: E402
from src.encoders import build_encoder  # noqa: E402
from src.utils.device import resolve_device  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

# Named shorthands for the models under comparison, so a run is one flag.
PRESETS: dict[str, EncoderConfig] = {
    "clip-l14": EncoderConfig(name="openai/clip-vit-large-patch14"),
    "jina-v2": EncoderConfig(name="jinaai/jina-clip-v2", trust_remote_code=True),
    "clipfa": EncoderConfig(
        name="CLIPfa",
        kind="dual_tower",
        text_model="SajjadAyoubi/clip-fa-text",
        vision_model="SajjadAyoubi/clip-fa-vision",
        max_text_length=512,
    ),
}


def encode_images(encoder, paths: list[Path], batch_size: int) -> torch.Tensor:
    features = []
    for start in tqdm(range(0, len(paths), batch_size), desc="images", leave=False):
        batch = [
            Image.open(p).convert("RGB") for p in paths[start : start + batch_size]
        ]
        features.append(encoder.encode_images(batch))
        for image in batch:
            image.close()
    return torch.cat(features)


def encode_texts(encoder, texts: list[str], batch_size: int) -> torch.Tensor:
    features = []
    for start in tqdm(range(0, len(texts), batch_size), desc="texts", leave=False):
        features.append(encoder.encode_texts(texts[start : start + batch_size]))
    return torch.cat(features)


def recall_at_k(ranks: torch.Tensor, ks: tuple[int, ...]) -> dict[str, float]:
    """`ranks` holds the 0-based rank of the correct item for each query."""
    return {f"R@{k}": float((ranks < k).float().mean()) * 100 for k in ks}


def evaluate(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    text_to_image: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, dict[str, float]]:
    """`text_to_image[i]` is the image index caption i belongs to."""
    images = F.normalize(image_features, dim=-1)
    texts = F.normalize(text_features, dim=-1)
    similarity = texts @ images.T  # [n_captions, n_images]

    # text -> image: where does the caption's own image rank?
    gold = similarity[torch.arange(len(texts)), text_to_image].unsqueeze(1)
    t2i_ranks = (similarity > gold).sum(dim=1)

    # image -> text: where does the image's best own caption rank?
    per_image = similarity.T  # [n_images, n_captions]
    n_images = images.shape[0]
    i2t_ranks = torch.empty(n_images, dtype=torch.long)
    for index in range(n_images):
        row = per_image[index]
        own = row[text_to_image == index]
        i2t_ranks[index] = (row > own.max()).sum()

    return {
        "text_to_image": recall_at_k(t2i_ranks, ks),
        "image_to_text": recall_at_k(i2t_ranks, ks),
    }


def run(
    preset: str,
    config: EncoderConfig,
    device: torch.device,
    root: Path,
    language: str,
    limit: int | None,
    batch_size: int,
) -> dict:
    dataset = Crossmodal3600(root, language)
    records = dataset.with_images()
    missing = dataset.missing_images()
    if missing:
        print(f"warning: {len(missing)} image(s) not downloaded; evaluating the rest")
    if limit:
        records = records[:limit]
    if not records:
        raise RuntimeError("no usable records; run scripts/download_xm3600.py")

    texts: list[str] = []
    text_to_image: list[int] = []
    for index, record in enumerate(records):
        for caption in record.captions:
            texts.append(caption)
            text_to_image.append(index)

    print(
        f"{preset} / {language}: {len(records)} images, {len(texts)} captions, "
        f"device {device}"
    )
    encoder = build_encoder(config, device)

    image_features = encode_images(encoder, [r.image_path for r in records], batch_size)
    text_features = encode_texts(encoder, texts, batch_size)
    if image_features.shape[1] != text_features.shape[1]:
        raise RuntimeError(
            f"{preset}: image features are {image_features.shape[1]}-d but text "
            f"features are {text_features.shape[1]}-d; the two towers do not "
            "share a space, so retrieval between them is meaningless"
        )

    results = evaluate(image_features, text_features, torch.tensor(text_to_image))
    return {
        "encoder": preset,
        "model": config.name,
        "language": language,
        "images": len(records),
        "captions": len(texts),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoder", required=True, help=f"one of {sorted(PRESETS)}, or a model id"
    )
    parser.add_argument("--language", default="fa")
    parser.add_argument("--root", default="data/xm3600")
    parser.add_argument("--limit", type=int, help="use only the first N images")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default="runs/retrieval")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)

    if args.encoder in PRESETS:
        preset, config = args.encoder, PRESETS[args.encoder]
    else:
        preset = args.encoder.replace("/", "__")
        config = EncoderConfig(
            name=args.encoder, trust_remote_code=args.trust_remote_code
        )

    payload = run(
        preset,
        config,
        device,
        Path(args.root),
        args.language,
        args.limit,
        args.batch_size,
    )

    for direction, scores in payload["results"].items():
        line = "  ".join(f"{k} {v:5.2f}" for k, v in scores.items())
        print(f"  {direction:<14} {line}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{preset}_{args.language}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
