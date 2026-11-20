#!/usr/bin/env bash
# Build a document, then set the one thing the docx library cannot.
#
# A section's column order is decided by <w:bidi/> inside <w:sectPr>, not by
# the paragraphs' own direction. Without it the first column lands on the left
# and a Persian reader meets the sections out of order -- which is how the
# first draft read. docx-js exposes no section-level bidi, so it is inserted
# here. Schema order puts it after <w:cols> and before <w:docGrid>, and Word
# rejects the file if it sits anywhere else.
#
# Both documents go through this script, because the report did not and lost
# its section bidi the first time it was rebuilt by calling the generator
# directly: `node report_build.js` writes a file that opens, validates and
# reads in the wrong direction. A generator is not a build.
#
#   bash paper/build.sh                  # -> paper/maghale.docx
#   bash paper/build.sh gozaresh.docx    # -> paper/gozaresh.docx
#   bash paper/build.sh --all            # both
set -euo pipefail
cd "$(dirname "$0")"

set_section_bidi() {
  local out="$1"
  local work
  work="$(mktemp -d)"
  # shellcheck disable=SC2064
  trap "rm -rf '$work'" RETURN
  unzip -q "$out" -d "$work"
  find "$work" -type l -delete
  python3 - "$work/word/document.xml" <<'PY'
import sys
from pathlib import Path
path = Path(sys.argv[1])
xml = path.read_text(encoding="utf-8")
before = xml.count("<w:bidi/><w:docGrid")
xml = xml.replace("<w:docGrid", "<w:bidi/><w:docGrid")
# only inside sectPr: every <w:docGrid> in this document is a section's
added = xml.count("<w:bidi/><w:docGrid") - before
if added == 0:
    raise SystemExit("no <w:sectPr> took bidi: the generator's output changed")
path.write_text(xml, encoding="utf-8")
print(f"section bidi set on {added} section(s)")
PY
  # -D: no directory entries. A docx that carries them opens in Word but has
  # failed to open through cloud services, which is where a journal portal and
  # a co-author's browser both read it; every build before this one had five.
  (cd "$work" && rm -f "$OLDPWD/$out" && zip -qXrD "$OLDPWD/$out" .)
  echo "wrote $out"
}

build() {
  local out="$1"
  case "$out" in
    gozaresh.docx) node report_build.js ;;
    *) node make.js "../runs/paper_numbers.json" "$out" ;;
  esac
  set_section_bidi "$out"
}

if [ "${1:-}" = "--all" ]; then
  build maghale.docx
  build gozaresh.docx
else
  build "${1:-maghale.docx}"
fi
