"""Preparing collected memes for annotation.

Between "I downloaded some memes" and "annotators can start" sit three jobs
the guideline requires, and all three are easier to do once, here, than to
undo later:

1. **Deduplicate.** Meme templates recur constantly. The same image annotated
   twice wastes annotator time and inflates agreement, because the second copy
   is not an independent judgement.
2. **Flag identifying detail.** The guideline requires anonymisation at
   collection time. Automatic detection cannot be trusted to catch everything,
   so this flags candidates for a human to confirm rather than deciding.
3. **Build the annotation skeleton**, so the file annotators work from is
   valid against the schema from the first line.

Nothing here is destructive. It reports and it writes a new file; it never
edits or deletes a collected image.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

# Digits as Persian text actually writes them: Western, Persian-Indic
# (U+06F0-U+06F9) and Arabic-Indic (U+0660-U+0669). A phone number in a
# Persian meme is usually not in ASCII, and a pattern that assumes it is
# misses the thing this scan exists to catch.
_D = r"0-9۰-۹٠-٩"
_NINE = r"[9۹٩]"

# Patterns worth a human look before a meme reaches an annotator. These are
# deliberately broad: a false positive costs one glance, a missed username
# costs someone's privacy.
IDENTIFIER_PATTERNS: dict[str, re.Pattern[str]] = {
    # The lookbehind excludes only another @, not any word character: a handle
    # embedded in a filename like "from_@channel.png" is exactly the case this
    # scan is for. An email's local part matches too, and that is fine — every
    # pattern reports independently, and two labels on one string tell a
    # reviewer more than one.
    "handle": re.compile(r"(?<!@)@[A-Za-z0-9_]{3,}"),
    # t.me and telegram.me with or without a scheme or path separator.
    "telegram_link": re.compile(r"(?:t|telegram)\.me[/_\s]\S*", re.I),
    "instagram_link": re.compile(r"instagram\.com\S*", re.I),
    "url": re.compile(r"https?://\S+", re.I),
    # The TLD must be letters, so "a_@b.3" and similar noise do not read as
    # an address.
    "email": re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}"),
    # Iranian mobile numbers: an optional +98 or a leading zero, then 9 and
    # nine more digits, in any of the three digit sets.
    "phone": re.compile(
        rf"(?:\+?[9۹٩][8۸٨]|[0۰٠])?"
        rf"{_NINE}[{_D}]{{9}}"
    ),
}


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def average_hash(path: str | Path, size: int = 8) -> int:
    """A small perceptual hash, for near-duplicates an exact hash misses.

    A meme re-saved at another quality, or cropped by a pixel, has a different
    sha256 but the same average hash. Reduce to size x size greyscale and set
    one bit per pixel above the mean.
    """
    with Image.open(path) as image:
        thumbnail = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
        # tobytes() rather than getdata(): one byte per pixel in "L" mode, and
        # it does not move under Pillow's deprecations.
        pixels = thumbnail.tobytes()
    mean = sum(pixels) / len(pixels)
    bits = 0
    for index, value in enumerate(pixels):
        if value > mean:
            bits |= 1 << index
    return bits


def hamming(first: int, second: int) -> int:
    return (first ^ second).bit_count()


@dataclass(frozen=True)
class CollectedMeme:
    path: Path
    sha256: str
    ahash: int
    text: str = ""


def scan(paths: Iterable[str | Path]) -> list[CollectedMeme]:
    """Hash every image. Unreadable files are reported, not silently skipped."""
    collected: list[CollectedMeme] = []
    broken: list[Path] = []
    for raw in paths:
        path = Path(raw)
        try:
            collected.append(
                CollectedMeme(
                    path=path, sha256=sha256_file(path), ahash=average_hash(path)
                )
            )
        except Exception:
            broken.append(path)
    if broken:
        raise ValueError(
            f"{len(broken)} file(s) could not be read as images, first: {broken[0]}"
        )
    return collected


def exact_duplicates(memes: Sequence[CollectedMeme]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for meme in memes:
        groups.setdefault(meme.sha256, []).append(meme.path)
    return {k: v for k, v in groups.items() if len(v) > 1}


def near_duplicates(
    memes: Sequence[CollectedMeme], threshold: int = 5
) -> list[tuple[Path, Path, int]]:
    """Pairs within `threshold` bits of each other.

    Quadratic, which is fine for the few hundred memes the guideline calls
    for and would not be for a hundred thousand.

    The threshold is a judgement, not a fact: lower it if it groups memes that
    share a template but say different things, raise it if re-encodings slip
    through. Every pair is reported for a human to decide, never dropped
    automatically — two memes on the same template with different text are
    different memes, and that is exactly what this task is about.
    """
    pairs: list[tuple[Path, Path, int]] = []
    for i in range(len(memes)):
        for j in range(i + 1, len(memes)):
            if memes[i].sha256 == memes[j].sha256:
                continue  # already an exact duplicate
            distance = hamming(memes[i].ahash, memes[j].ahash)
            if distance <= threshold:
                pairs.append((memes[i].path, memes[j].path, distance))
    return sorted(pairs, key=lambda p: p[2])


def find_identifiers(text: str) -> dict[str, list[str]]:
    """Possible identifying detail in a meme's transcribed text.

    A reporting aid, not a filter. The guideline puts a human in this loop
    because no pattern list catches a face, a logo, or a name written as
    ordinary words.
    """
    found: dict[str, list[str]] = {}
    for name, pattern in IDENTIFIER_PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            found[name] = matches
    return found
