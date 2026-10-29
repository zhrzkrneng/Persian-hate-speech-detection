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
stored in this repository nor downloaded automatically.

Try the official site first. If it is unreachable -- the competition page has
stopped serving the data publicly and people now circulate mirrors -- then
whatever copy you obtain must be verified before you trust it. Mirrors
differ: some are subsets, some re-split the data, some drop a field. A copy
that does not match the published splits yields numbers comparable with no
paper, and training on it would simply succeed.

Cite Kiela et al. (2020) either way, and say in the paper where you got it.

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
echo

# Line counting is not enough: a mirror can have the right files and the wrong
# contents. Hand over to the verifier, which checks the split sizes against
# the published ones, the labels, the referenced images and train/eval
# leakage.
if command -v python >/dev/null 2>&1; then
  python -m scripts.check_hmc --root "$ROOT"
else
  python3 -m scripts.check_hmc --root "$ROOT"
fi
