"""Encode Multi3Hate's five languages once per encoder, for zero-shot reading.

    python -m scripts.extract_multi3hate --config configs/hmc_align_clipb32.yaml

Multi3Hate is 300 memes per language, so this is minutes rather than the hours
the Hateful Memes cache costs. The point of it is not size: the memes are
parallel -- the same image and the same intended meaning in English, German,
Spanish, Hindi and Mandarin, with only the rendered text changing -- so a score
difference between two languages is caused by the language and by nothing else.
No collection effort could buy that design.

The config is a classification config, read only for its encoder. Nothing is
trained here and no architecture is involved, so one config per encoder is
enough and which fusion it names is irrelevant. Caches land under
`features/multi3hate/<encoder>/<language>.pt`, keyed by encoder exactly as the
Hateful Memes caches are, so that two encoders can never share one file.

**These caches must never leave the machine that computes them.** Multi3Hate is
published under CC BY-NC-ND 4.0: no distribution of adapted material, and an
encoded copy of the images is adapted material. The Hateful Memes caches were
backed up to a private Kaggle dataset, which its click-through licence allows
as storage; these may not be, not even privately. `features/` is gitignored, so
the risk is not a commit but a backup command written to sweep `features/` --
it would have taken both. Anything that copies feature files off this machine
must name the datasets it copies, never the directory.

Reporting a measurement made from these features is not distributing them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import Config, load_config  # noqa: E402
from src.data.features import FeatureCache  # noqa: E402
from src.data.multi3hate import (  # noqa: E402
    LANGUAGES,
    Multi3HateError,
    Multi3HateSplit,
    parallel_ids,
)
from src.encoders import Encoder, build_encoder, resolve_device  # noqa: E402

# Not config.feature_root / config.data.name: the dataset here is not the one
# the config names, and writing under the config's dataset would put Multi3Hate
# features where a Hateful Memes cache is looked for.
ROOT = Path("features/multi3hate")


def feature_dir(config: Config) -> Path:
    return ROOT / config.encoder_slug


def encode(
    encoder: Encoder, split: Multi3HateSplit, batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    missing = split.missing_images()
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} image(s) of {split.language} are missing, first: "
            f"{missing[0]}. Run scripts/download_multi3hate.py."
        )

    unlabelled = [m.uid for m in split.records if m.label is None]
    if unlabelled:
        raise Multi3HateError(
            f"{split.language}: {len(unlabelled)} meme(s) have no label from "
            f"{split.country}, first {unlabelled[0]}. A zero-shot score cannot "
            "be computed over them, and -1 would quietly enter the metric."
        )

    images: list[torch.Tensor] = []
    texts: list[torch.Tensor] = []
    for start in tqdm(
        range(0, len(split), batch_size), desc=split.language, leave=False
    ):
        records = split.records[start : start + batch_size]
        opened = [Image.open(m.image_path).convert("RGB") for m in records]
        images.append(encoder.encode_images(opened))
        texts.append(encoder.encode_texts([m.text for m in records]))
        for image in opened:
            image.close()

    return (
        torch.cat(images),
        torch.cat(texts),
        torch.tensor([int(m.label) for m in split.records], dtype=torch.long),
        [m.uid for m in split.records],
    )


def run(config: Config, root: Path, overwrite: bool, languages: list[str]) -> None:
    # Checked before any encoding: comparing languages over different sets of
    # memes would not be measuring the language, and it is cheap to find out
    # now rather than after an hour of encoding.
    shared = parallel_ids(root)
    print(f"{len(shared)} memes present in all {len(LANGUAGES)} languages")

    directory = feature_dir(config)
    device = resolve_device(config.train.device)
    print(f"loading {config.encoder.name} ({config.encoder.kind}) on {device}")
    encoder = None

    for language in languages:
        cache = FeatureCache(directory, language)
        if cache.exists() and not overwrite:
            print(f"{language}: cached at {cache.path}, skipping")
            continue
        # Loaded on first need, so a fully cached encoder costs nothing.
        if encoder is None:
            encoder = build_encoder(config.encoder, device)

        split = Multi3HateSplit(root, language)
        image_features, text_features, labels, uids = encode(
            encoder, split, config.encoder.batch_size
        )
        cache.save(image_features, text_features, labels, uids, config.encoder.name)
        hateful = int(labels.sum())
        print(
            f"{language}: {len(uids)} memes -> {cache.path}  "
            f"label {split.country}: {hateful} hateful "
            f"({100 * hateful / len(uids):.1f}%)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="read for its encoder only")
    parser.add_argument("--root", default="data/multi3hate")
    parser.add_argument("--languages", nargs="+", default=list(LANGUAGES))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    unknown = [language for language in args.languages if language not in LANGUAGES]
    if unknown:
        raise SystemExit(f"unknown language(s) {unknown}, expected {list(LANGUAGES)}")

    run(load_config(Path(args.config)), Path(args.root), args.overwrite, args.languages)


if __name__ == "__main__":
    main()
