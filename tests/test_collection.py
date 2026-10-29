"""Deduplication and identifier scanning.

Both are reporting aids a human acts on, so the thing to get right is that
they neither miss a real duplicate nor collapse two memes that only share a
template.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.collection import (  # noqa: E402
    average_hash,
    exact_duplicates,
    find_identifiers,
    hamming,
    near_duplicates,
    scan,
    sha256_file,
)


def make_image(path: Path, seed: int = 0, size: int = 64) -> Path:
    """A deterministic image with structure in both axes.

    Regular stripes are a trap here: a vertical pattern whose period matches
    the hash grid gives every seed the same average hash, which makes a
    near-duplicate test pass or fail for the wrong reason. Per-cell pseudo-
    random blocks keep distinct seeds genuinely distinct.
    """
    rng = random.Random(seed)
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    cell = size // 8
    for row in range(8):
        for column in range(8):
            shade = rng.randrange(256)
            draw.rectangle(
                [column * cell, row * cell, (column + 1) * cell, (row + 1) * cell],
                fill=(shade, (shade * 3) % 256, 255 - shade),
            )
    image.save(path)
    return path


def test_sha256_matches_for_identical_bytes(tmp_path: Path) -> None:
    first = make_image(tmp_path / "a.png", seed=1)
    second = tmp_path / "b.png"
    second.write_bytes(first.read_bytes())
    assert sha256_file(first) == sha256_file(second)


def test_exact_duplicates_are_grouped(tmp_path: Path) -> None:
    first = make_image(tmp_path / "a.png", seed=1)
    (tmp_path / "copy.png").write_bytes(first.read_bytes())
    make_image(tmp_path / "other.png", seed=2)

    groups = exact_duplicates(scan(tmp_path.glob("*.png")))
    assert len(groups) == 1
    assert {p.name for p in next(iter(groups.values()))} == {"a.png", "copy.png"}


def test_recompressed_copy_is_a_near_duplicate(tmp_path: Path) -> None:
    """A re-saved meme has a new sha256 but should still be caught."""
    original = make_image(tmp_path / "a.png", seed=3)
    with Image.open(original) as image:
        image.save(tmp_path / "a_resaved.jpg", quality=55)

    memes = scan([tmp_path / "a.png", tmp_path / "a_resaved.jpg"])
    assert exact_duplicates(memes) == {}  # different bytes
    assert near_duplicates(memes), "re-encoded copy was not flagged"


def test_different_images_are_not_near_duplicates(tmp_path: Path) -> None:
    memes = scan(
        [
            make_image(tmp_path / "a.png", seed=1),
            make_image(tmp_path / "b.png", seed=40),
        ]
    )
    assert near_duplicates(memes, threshold=2) == []


def test_near_duplicate_threshold_is_respected(tmp_path: Path) -> None:
    memes = scan(
        [
            make_image(tmp_path / "a.png", seed=1),
            make_image(tmp_path / "b.png", seed=40),
        ]
    )
    # A threshold of 64 bits accepts everything, by definition of the hash.
    assert len(near_duplicates(memes, threshold=64)) == 1


def test_hamming_counts_differing_bits() -> None:
    assert hamming(0b1010, 0b1010) == 0
    assert hamming(0b1010, 0b1011) == 1
    assert hamming(0, 0xFF) == 8


def test_average_hash_is_stable(tmp_path: Path) -> None:
    path = make_image(tmp_path / "a.png", seed=5)
    assert average_hash(path) == average_hash(path)


def test_unreadable_file_is_reported_not_skipped(tmp_path: Path) -> None:
    make_image(tmp_path / "a.png", seed=1)
    (tmp_path / "broken.png").write_text("not an image")
    with pytest.raises(ValueError, match="could not be read"):
        scan(sorted(tmp_path.glob("*.png")))


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("کانال ما @meme_channel", "handle"),
        ("t.me/somechannel", "telegram_link"),
        ("instagram.com/someone", "instagram_link"),
        ("https://example.com/x", "url"),
        ("someone@example.com", "email"),
        ("09121234567", "phone"),
    ],
)
def test_identifier_patterns(text: str, kind: str) -> None:
    assert kind in find_identifiers(text)


@pytest.mark.parametrize(
    "number",
    [
        "09121234567",  # Western digits
        "۰۹۱۲۱۲۳۴۵۶۷",  # Persian-Indic
        "٠٩١٢١٢٣٤٥٦٧",  # Arabic-Indic
        "9121234567",  # no leading zero
        "۹۱۲۱۲۳۴۵۶۷",
        "+989121234567",
        "۹۸۹۱۲۱۲۳۴۵۶۷",
        "تماس: ۰۹۱۲۱۲۳۴۵۶۷ بگیرید",  # embedded in Persian text
    ],
)
def test_phone_numbers_in_every_digit_set(number: str) -> None:
    """Persian memes rarely write numbers in ASCII, and a missed one is a
    privacy failure, not a cosmetic miss."""
    assert "phone" in find_identifiers(number), number


def test_short_digit_runs_are_not_phone_numbers() -> None:
    """Years and small numbers must not flood the report."""
    assert "phone" not in find_identifiers("سال ۱۴۰۳ بود و ۹۱۲ نفر آمدند")


def test_plain_persian_text_is_not_flagged() -> None:
    assert find_identifiers("این یک متن عادی فارسی است") == {}


def test_patterns_report_independently() -> None:
    """An address matches both email and handle, on purpose.

    The handle pattern is loose so it catches a handle embedded in a filename
    like "from_@channel.png", which a strict version misses. The cost is that
    an email's local part matches too. Since this is a human-review aid, two
    labels on one string cost a reviewer one glance and tell them more than
    one label would.
    """
    found = find_identifiers("someone@example.com")
    assert "email" in found
    assert "handle" in found


def test_handle_inside_a_filename_is_caught() -> None:
    """The case a strict lookbehind misses, and the reason for the loose one."""
    assert "handle" in find_identifiers("from_@persian_memes.png")


def test_telegram_link_in_a_filename_is_caught() -> None:
    found = find_identifiers("from_@channel_t.me_something.png")
    assert "telegram_link" in found


def test_noise_is_not_read_as_an_email() -> None:
    """The TLD must be letters, or version strings flood the report."""
    assert "email" not in find_identifiers("build_v2@host.3")
