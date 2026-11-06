#!/usr/bin/env bash
# Compare state_hash in a submission with the hash of replaying train labels.
# Usage: scripts/compare_hash.sh --raw
#        scripts/compare_hash.sh --gold
#        scripts/compare_hash.sh --gold --limit 100
#        scripts/compare_hash.sh --gold --begin 51 --end 100
#        scripts/compare_hash.sh
# --raw scores outputs/submission.jsonl. --gold scores outputs/gold/submission.jsonl.
# With neither flag, both files that exist are scored.
# --begin, --end, and --limit count rows in the train file, starting at 1.
# Omit them to score every row that is already in the submission file.
# A train row in the requested range with no submission line is missing.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
exec python3 "$ROOT/scripts/compare_hash.py" "$@"
