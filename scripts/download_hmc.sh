#!/usr/bin/env bash
# The Hateful Memes Challenge data cannot be redistributed, so this script
# fetches nothing. It tells you where to get it and checks what you placed.
set -euo pipefail

ROOT="${1:-data/hmc}"

cat <<'EOF'
Hateful Memes Challenge (Kiela et al., NeurIPS 2020)

  https://hatefulmemeschallenge.com/

The data is released by Meta under a click-through licence agreement that you
must accept yourself. It does not permit redistribution, so it is neither
stored in this repository nor downloaded automatically. Third-party mirrors
carry inconsistent licence labels; use the official source.

After accepting the licence, unpack the archive so that it looks like this:

EOF

printf '  %s/train.jsonl\n' "$ROOT"
printf '  %s/dev_seen.jsonl\n' "$ROOT"
printf '  %s/dev_unseen.jsonl\n' "$ROOT"
printf '  %s/test_seen.jsonl\n' "$ROOT"
printf '  %s/img/<id>.png\n\n' "$ROOT"

if [ ! -d "$ROOT" ]; then
  echo "not found: $ROOT (nothing to check yet)"
  exit 0
fi

echo "checking $ROOT"
status=0
for split in train dev_seen dev_unseen test_seen; do
  file="$ROOT/$split.jsonl"
  if [ -f "$file" ]; then
    printf '  ok      %-16s %s lines\n' "$split.jsonl" "$(wc -l < "$file")"
  else
    printf '  MISSING %s\n' "$split.jsonl"
    status=1
  fi
done

if [ -d "$ROOT/img" ]; then
  printf '  ok      img/            %s files\n' "$(find "$ROOT/img" -type f | wc -l)"
else
  printf '  MISSING img/\n'
  status=1
fi

exit "$status"
