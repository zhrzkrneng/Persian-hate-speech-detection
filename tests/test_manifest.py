"""The provenance manifest, and that preparing twice loses nothing.

Collection happens in sessions: save some memes, prepare, annotate, save some
more. The first version of `prepare_memes.py` rebuilt the annotation file from
scratch on every run, so the second session destroyed the first session's
labels and transcriptions, exited 0, and renumbered every id. The tests here
are the regression tests for that, which is why they go through the script's
real entry point rather than a helper: the destruction happened in the part
that writes the file.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.prepare_memes import main, next_index  # noqa: E402
from src.data.collection import (  # noqa: E402
    ManifestError,
    extend_manifest,
    manifest_gaps,
    read_manifest,
)
from src.data.persian import PersianMemeSet  # noqa: E402

HEADER = "file,platform,url,collected\n"


def make_image(path: Path, seed: int) -> Path:
    """Per-cell random blocks, so distinct seeds have distinct average hashes."""
    rng = random.Random(seed)
    image = Image.new("L", (64, 64))
    pixels = image.load()
    for block_y in range(8):
        for block_x in range(8):
            value = rng.randrange(256)
            for y in range(block_y * 8, block_y * 8 + 8):
                for x in range(block_x * 8, block_x * 8 + 8):
                    pixels[x, y] = value
    image.save(path)
    return path


def run(*args: str) -> None:
    sys.argv = ["prepare_memes", *args]
    main()


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ── the manifest ─────────────────────────────────────────────────────────


def test_read_manifest(tmp_path: Path) -> None:
    path = tmp_path / "sources.csv"
    path.write_text(
        HEADER + "a.png,telegram,https://t.me/c/1,2026-10-30\n", encoding="utf-8"
    )
    sources = read_manifest(path)
    assert sources["a.png"].platform == "telegram"
    assert sources["a.png"].collected == "2026-10-30"


def test_a_spreadsheet_byte_order_mark_does_not_break_the_header(
    tmp_path: Path,
) -> None:
    """Saving CSV from a spreadsheet writes a BOM; the file looks fine."""
    path = tmp_path / "sources.csv"
    path.write_bytes(
        b"\xef\xbb\xbf" + (HEADER + "a.png,telegram,,2026-10-30\n").encode()
    )
    assert read_manifest(path)["a.png"].platform == "telegram"


def test_a_wrong_header_names_what_it_got(tmp_path: Path) -> None:
    path = tmp_path / "sources.csv"
    path.write_text("filename,where,when\na.png,telegram,today\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="header must start with"):
        read_manifest(path)


def test_a_missing_manifest_says_how_to_make_one(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="--init-manifest"):
        read_manifest(tmp_path / "nothing.csv")


def test_a_file_listed_twice_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "sources.csv"
    path.write_text(
        HEADER + "a.png,telegram,,2026-10-30\na.png,x,,2026-10-30\n", encoding="utf-8"
    )
    with pytest.raises(ManifestError, match="listed twice"):
        read_manifest(path)


def test_gaps_name_the_problem(tmp_path: Path) -> None:
    path = tmp_path / "sources.csv"
    path.write_text(
        HEADER
        + "ok.png,telegram,https://t.me/c/1,2026-10-30\n"
        + "nodate.png,telegram,,\n"
        + "badrate.png,telegram,,30/10/2026\n"
        + "noplatform.png,,https://x.com/a/1,2026-10-30\n",
        encoding="utf-8",
    )
    gaps = manifest_gaps(
        ["ok.png", "nodate.png", "badrate.png", "noplatform.png", "absent.png"],
        read_manifest(path),
    )
    assert "ok.png" not in gaps
    assert gaps["nodate.png"] == ["collected is empty"]
    assert "not a YYYY-MM-DD date" in gaps["badrate.png"][0]
    assert gaps["noplatform.png"] == ["platform is empty"]
    assert gaps["absent.png"] == ["not listed in the manifest"]


def test_a_missing_url_is_not_a_gap(tmp_path: Path) -> None:
    """A forwarded meme has no link; refusing it invites an invented one."""
    path = tmp_path / "sources.csv"
    path.write_text(HEADER + "a.png,telegram,,2026-10-30\n", encoding="utf-8")
    assert manifest_gaps(["a.png"], read_manifest(path)) == {}


def test_extend_manifest_keeps_what_was_filled_in(tmp_path: Path) -> None:
    """It runs after every session, so clobbering costs the whole manifest."""
    path = tmp_path / "sources.csv"
    path.write_text(
        "file,platform,url,collected,note\n"
        "a.png,telegram,https://t.me/c/1,2026-10-30,checked\n",
        encoding="utf-8",
    )
    added = extend_manifest(path, ["a.png", "b.png"])

    assert added == ["b.png"]
    sources = read_manifest(path)
    assert sources["a.png"].url == "https://t.me/c/1"
    assert sources["b.png"].platform == ""
    assert "note" in path.read_text(encoding="utf-8").splitlines()[0]
    assert "checked" in path.read_text(encoding="utf-8")


def test_extend_manifest_creates_the_file(tmp_path: Path) -> None:
    path = tmp_path / "sources.csv"
    assert extend_manifest(path, ["a.png"]) == ["a.png"]
    assert read_manifest(path)["a.png"].platform == ""


def test_extend_manifest_twice_adds_nothing_the_second_time(tmp_path: Path) -> None:
    path = tmp_path / "sources.csv"
    extend_manifest(path, ["a.png"])
    assert extend_manifest(path, ["a.png"]) == []


# ── preparing twice ──────────────────────────────────────────────────────


def annotate(path: Path) -> None:
    """Stand in for a round of annotation: text, source and three labels."""
    rows = read(path)
    for index, row in enumerate(rows):
        row["text"] = f"متن {index}"
        row["source"] = {
            "platform": "telegram",
            "url": f"https://t.me/c/{index}",
            "collected": "2026-10-29",
        }
        row["annotations"] = [
            {"annotator": "A1", "label": 1, "target": "political", "explicit": False},
            {"annotator": "A2", "label": 0, "target": None, "explicit": None},
            {"annotator": "A3", "label": 1, "target": "political", "explicit": True},
        ]
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )


def test_a_second_session_keeps_the_first_session_annotations(tmp_path: Path) -> None:
    """The regression test. This cost nine labels and three transcriptions."""
    images = tmp_path / "raw"
    images.mkdir()
    out = tmp_path / "annotations.jsonl"
    for seed, name in enumerate(["b.png", "c.png", "d.png"]):
        make_image(images / name, seed)

    run("--images", str(images), "--out", str(out))
    annotate(out)
    before = {row["image_sha256"]: row for row in read(out)}

    # A meme collected later, whose filename sorts before every other.
    make_image(images / "a_later.png", 99)
    run("--images", str(images), "--out", str(out))
    after = read(out)

    assert len(after) == 4
    for row in after:
        kept = before.get(row["image_sha256"])
        if kept is None:
            assert row["annotations"] == []  # the new one
            continue
        # id, text and every label survive, unchanged.
        assert row["id"] == kept["id"]
        assert row["text"] == kept["text"]
        assert row["annotations"] == kept["annotations"]
        assert row["source"] == kept["source"]


def test_ids_are_never_reassigned(tmp_path: Path) -> None:
    """An id names a meme in notes and in disagreement lists; it cannot move."""
    images = tmp_path / "raw"
    images.mkdir()
    out = tmp_path / "annotations.jsonl"
    make_image(images / "b.png", 1)
    run("--images", str(images), "--out", str(out))
    first = read(out)[0]

    make_image(images / "a.png", 2)  # sorts first
    run("--images", str(images), "--out", str(out))
    rows = {row["image_sha256"]: row["id"] for row in read(out)}

    assert rows[first["image_sha256"]] == first["id"] == "fa_0000"
    assert sorted(rows.values()) == ["fa_0000", "fa_0001"]


def test_an_image_that_disappears_keeps_its_row(tmp_path: Path) -> None:
    """A renamed file must not take a round of annotation with it."""
    images = tmp_path / "raw"
    images.mkdir()
    out = tmp_path / "annotations.jsonl"
    make_image(images / "a.png", 1)
    make_image(images / "b.png", 2)
    run("--images", str(images), "--out", str(out))
    annotate(out)

    (images / "b.png").rename(images / "b_renamed.png")
    run("--images", str(images), "--out", str(out))

    rows = read(out)
    assert len(rows) == 2, "the renamed file is the same image, by hash"
    assert all(row["annotations"] for row in rows)


def test_ids_continue_past_a_gap(tmp_path: Path) -> None:
    """Deleting a row by hand must not make the next run reuse its id."""
    images = tmp_path / "raw"
    images.mkdir()
    out = tmp_path / "annotations.jsonl"
    for seed, name in enumerate(["a.png", "b.png"]):
        make_image(images / name, seed)
    run("--images", str(images), "--out", str(out))

    rows = read(out)
    out.write_text(json.dumps(rows[1]) + "\n", encoding="utf-8")  # keep fa_0001 only
    make_image(images / "c.png", 5)
    run("--images", str(images), "--out", str(out))

    ids = [row["id"] for row in read(out)]
    assert ids == ["fa_0001", "fa_0002", "fa_0003"], ids


def test_next_index_ignores_another_prefix() -> None:
    rows = [{"id": "fa_0007"}, {"id": "en_0100"}, {"id": "note"}]
    assert next_index(rows, "fa") == 8
    assert next_index([], "fa") == 0


def test_the_skeleton_is_valid_to_the_reader(tmp_path: Path) -> None:
    images = tmp_path / "raw"
    images.mkdir()
    make_image(images / "a.png", 1)
    out = tmp_path / "annotations.jsonl"
    run("--images", str(images), "--out", str(out))
    assert len(PersianMemeSet(out)) == 1


def test_source_is_filled_from_the_manifest(tmp_path: Path) -> None:
    images = tmp_path / "raw"
    images.mkdir()
    make_image(images / "a.png", 1)
    manifest = tmp_path / "sources.csv"
    manifest.write_text(
        HEADER + "a.png,telegram,https://t.me/c/1,2026-10-30\n", encoding="utf-8"
    )
    out = tmp_path / "annotations.jsonl"
    run(
        "--images",
        str(images),
        "--manifest",
        str(manifest),
        "--out",
        str(out),
    )
    assert read(out)[0]["source"] == {
        "platform": "telegram",
        "url": "https://t.me/c/1",
        "collected": "2026-10-30",
    }


def test_a_source_already_recorded_is_not_overwritten(tmp_path: Path) -> None:
    """Both files are edited by hand; the manifest does not get the last word."""
    images = tmp_path / "raw"
    images.mkdir()
    make_image(images / "a.png", 1)
    out = tmp_path / "annotations.jsonl"
    run("--images", str(images), "--out", str(out))
    annotate(out)

    manifest = tmp_path / "sources.csv"
    manifest.write_text(
        HEADER + "a.png,instagram,https://instagram.com/p/z,2026-01-01\n",
        encoding="utf-8",
    )
    run("--images", str(images), "--manifest", str(manifest), "--out", str(out))
    assert read(out)[0]["source"]["platform"] == "telegram"


def test_an_explicit_manifest_that_is_missing_is_an_error(tmp_path: Path) -> None:
    """A typo in --manifest must not read as "no provenance wanted"."""
    images = tmp_path / "raw"
    images.mkdir()
    make_image(images / "a.png", 1)
    with pytest.raises(SystemExit, match="not found"):
        run(
            "--images",
            str(images),
            "--manifest",
            str(tmp_path / "typo.csv"),
            "--dry-run",
        )


def test_init_manifest_lists_the_images(tmp_path: Path) -> None:
    images = tmp_path / "raw"
    images.mkdir()
    make_image(images / "a.png", 1)
    manifest = tmp_path / "sources.csv"
    run("--images", str(images), "--manifest", str(manifest), "--init-manifest")
    assert read_manifest(manifest).keys() == {"a.png"}


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    images = tmp_path / "raw"
    images.mkdir()
    make_image(images / "a.png", 1)
    out = tmp_path / "annotations.jsonl"
    run("--images", str(images), "--out", str(out), "--dry-run")
    assert not out.exists()


def test_a_corrupt_annotation_file_is_not_overwritten(tmp_path: Path) -> None:
    """Refusing is the point: the file holds work that cannot be regenerated."""
    images = tmp_path / "raw"
    images.mkdir()
    make_image(images / "a.png", 1)
    out = tmp_path / "annotations.jsonl"
    out.write_text('{"id": "fa_0000", "image_sha2\n', encoding="utf-8")
    with pytest.raises(SystemExit, match="invalid JSON"):
        run("--images", str(images), "--out", str(out))
    assert out.read_text(encoding="utf-8") == '{"id": "fa_0000", "image_sha2\n'
