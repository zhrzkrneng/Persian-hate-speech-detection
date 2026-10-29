"""Download Crossmodal-3600.

    python -m scripts.download_xm3600

Both parts come from the project's own download links: the captions as a zip
from the project page, the images as a single tarball. Captions are licensed
CC BY 4.0 by Google; the images carry their own attributions, which the script
fetches alongside them.

Produces:

    data/xm3600/
      captions.jsonl
      image_attributions.csv
      img/<image_key>.jpg

Resumable: anything already on disk is skipped.
"""

from __future__ import annotations

import argparse
import io
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tqdm import tqdm  # noqa: E402

CAPTIONS_URL = "https://google.github.io/crossmodal-3600/web-data/captions.zip"
ATTRIBUTIONS_URL = (
    "https://google.github.io/crossmodal-3600/web-data/image_attributions.csv"
)
IMAGES_URL = "https://open-images-dataset.s3.amazonaws.com/crossmodal-3600/images.tgz"
TIMEOUT = 60


def download(url: str, description: str) -> bytes:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
        total = int(response.headers.get("content-length", 0))
        buffer = io.BytesIO()
        with tqdm(
            total=total or None, unit="B", unit_scale=True, desc=description
        ) as bar:
            while chunk := response.read(1 << 20):
                buffer.write(chunk)
                bar.update(len(chunk))
    return buffer.getvalue()


def fetch_captions(root: Path) -> None:
    path = root / "captions.jsonl"
    if path.exists():
        print(f"captions: already at {path}")
        return
    payload = download(CAPTIONS_URL, "captions")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names = [n for n in archive.namelist() if n.endswith(".jsonl")]
        if len(names) != 1:
            raise RuntimeError(f"expected one .jsonl in the archive, found {names}")
        path.write_bytes(archive.read(names[0]))
    print(f"captions: wrote {path}")


def fetch_attributions(root: Path) -> None:
    path = root / "image_attributions.csv"
    if path.exists():
        return
    try:
        path.write_bytes(download(ATTRIBUTIONS_URL, "attributions"))
    except Exception as exc:  # not worth failing the run over
        print(f"attributions: skipped ({type(exc).__name__}: {exc})")


def fetch_images(root: Path, expected: int = 3600) -> None:
    directory = root / "img"
    existing = len(list(directory.glob("*.jpg"))) if directory.exists() else 0
    if existing >= expected:
        print(f"images: already have {existing} in {directory}")
        return

    payload = download(IMAGES_URL, "images")
    directory.mkdir(parents=True, exist_ok=True)

    written = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in tqdm(archive.getmembers(), desc="extracting"):
            if not member.isfile() or not member.name.lower().endswith(".jpg"):
                continue
            # Flatten, and take only the basename: never trust a path from an
            # archive to stay inside the directory we meant to write to.
            name = Path(member.name).name
            source = archive.extractfile(member)
            if source is None:
                continue
            (directory / name).write_bytes(source.read())
            written += 1

    print(f"images: wrote {written} files to {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/xm3600")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    fetch_captions(root)
    fetch_attributions(root)
    fetch_images(root)


if __name__ == "__main__":
    main()
