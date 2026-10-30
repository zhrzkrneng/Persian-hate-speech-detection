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

import csv
import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
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


# ── provenance ───────────────────────────────────────────────────────────
#
# The guideline requires the platform and the collection date for every meme.
# Those are knowable only while collecting: a folder of images a week later
# does not say where anything came from, and a few hundred rows cannot be
# reconstructed from memory. So they are recorded in a sidecar CSV as the
# images are saved, and read back from it here.
#
# CSV because it opens in a spreadsheet. The person filling it in is doing so
# by hand, a hundred rows at a time, and a spreadsheet is the tool for that.

MANIFEST_FIELDS = ("file", "platform", "url", "collected")


class ManifestError(ValueError):
    """The provenance manifest is missing, malformed, or incomplete."""


@dataclass(frozen=True)
class Source:
    platform: str
    url: str
    collected: str

    def as_dict(self) -> dict[str, str]:
        return {
            "platform": self.platform,
            "url": self.url,
            "collected": self.collected,
        }


def _problems(source: Source) -> list[str]:
    """What is wrong with one manifest row, in words a filler-in can act on.

    `url` is asked for but not required: a meme that arrived as a forward has
    no link, and refusing it would push the collector into inventing one. The
    count of rows without a URL is reported instead, so the paper can state
    how much of the set is traceable rather than implying all of it is.
    """
    issues: list[str] = []
    if not source.platform:
        issues.append("platform is empty")
    if not source.collected:
        issues.append("collected is empty")
    else:
        try:
            date.fromisoformat(source.collected)
        except ValueError:
            issues.append(f"collected {source.collected!r} is not a YYYY-MM-DD date")
    return issues


def read_manifest(path: str | Path) -> dict[str, Source]:
    """Read `file,platform,url,collected` into {filename: Source}.

    Reads as utf-8-sig: a spreadsheet saving CSV writes a byte-order mark, and
    without this the first column name arrives as "\\ufefffile" and the header
    check fails on a file that looks correct in every editor.
    """
    path = Path(path)
    if not path.exists():
        raise ManifestError(
            f"{path} not found. Create it with --init-manifest, then fill in "
            "the platform, url and collection date of each meme."
        )

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = tuple(reader.fieldnames or ())
        if header[: len(MANIFEST_FIELDS)] != MANIFEST_FIELDS:
            raise ManifestError(
                f"{path}: header must start with {','.join(MANIFEST_FIELDS)}, "
                f"got {','.join(header) or '(empty file)'}"
            )

        sources: dict[str, Source] = {}
        for number, row in enumerate(reader, start=2):
            name = (row.get("file") or "").strip()
            if not name:
                continue  # a blank line a spreadsheet left behind
            if name in sources:
                raise ManifestError(f"{path}:{number}: {name} listed twice")
            sources[name] = Source(
                platform=(row.get("platform") or "").strip(),
                url=(row.get("url") or "").strip(),
                collected=(row.get("collected") or "").strip(),
            )
    return sources


def manifest_gaps(
    names: Sequence[str], sources: dict[str, Source]
) -> dict[str, list[str]]:
    """Per-image provenance problems: missing rows and incomplete ones."""
    gaps: dict[str, list[str]] = {}
    for name in names:
        if name not in sources:
            gaps[name] = ["not listed in the manifest"]
        elif issues := _problems(sources[name]):
            gaps[name] = issues
    return gaps


def extend_manifest(path: str | Path, names: Sequence[str]) -> list[str]:
    """Add a row for each image not already listed. Returns the names added.

    Rows already present are rewritten exactly as they were read, including
    any extra columns the collector added for their own use. This runs after
    every collecting session, so overwriting what was already filled in would
    cost the whole manifest.
    """
    path = Path(path)
    existing: list[dict[str, str]] = []
    extra: list[str] = []
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = tuple(reader.fieldnames or ())
            if fields and fields[: len(MANIFEST_FIELDS)] != MANIFEST_FIELDS:
                raise ManifestError(
                    f"{path}: header must start with {','.join(MANIFEST_FIELDS)}, "
                    f"got {','.join(fields)}"
                )
            extra = [f for f in fields[len(MANIFEST_FIELDS) :] if f]
            existing = [row for row in reader if (row.get("file") or "").strip()]

    listed = {(row.get("file") or "").strip() for row in existing}
    added = [name for name in names if name not in listed]

    columns = [*MANIFEST_FIELDS, *extra]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in existing:
            writer.writerow({key: row.get(key, "") or "" for key in columns})
        for name in added:
            writer.writerow({**{key: "" for key in columns}, "file": name})
    temporary.replace(path)
    return added
