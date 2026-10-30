#!/usr/bin/env bash
# Build the manuscript, then set the one thing the docx library cannot.
#
# A section's column order is decided by <w:bidi/> inside <w:sectPr>, not by
# the paragraphs' own direction. Without it the first column lands on the left
# and a Persian reader meets the sections out of order -- which is how the
# first draft read. docx-js exposes no section-level bidi, so it is inserted
# here. Schema order puts it after <w:cols> and before <w:docGrid>, and Word
# rejects the file if it sits anywhere else.
set -euo pipefail
cd "$(dirname "$0")"

OUT="${1:-maghale.docx}"
node make.js "../runs/paper_numbers.json" "$OUT"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
unzip -q "$OUT" -d "$WORK"
find "$WORK" -type l -delete
python3 - "$WORK/word/document.xml" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
xml = path.read_text(encoding="utf-8")
before = xml.count("<w:bidi/><w:docGrid")
xml = xml.replace("<w:docGrid", "<w:bidi/><w:docGrid")
# only inside sectPr: every <w:docGrid> in this document is a section's
added = xml.count("<w:bidi/><w:docGrid") - before
path.write_text(xml, encoding="utf-8")
print(f"section bidi set on {added} section(s)")
PY
(cd "$WORK" && rm -f "$OLDPWD/$OUT" && zip -qXr "$OLDPWD/$OUT" .)
echo "wrote $OUT"
