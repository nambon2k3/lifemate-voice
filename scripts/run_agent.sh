#!/usr/bin/env bash
# Run the agent on train samples from cached transcripts. Does not use gold text.
# A cache miss is sent to the ASR server, then corrected, then planned.
# Usage: scripts/run_agent.sh --begin 32 --end 50
#        scripts/run_agent.sh --limit 10
#        scripts/run_agent.sh 10
# Samples are counted from 1, the first row of the train file.
# Omit --begin to start at 1. Omit --end and --limit to run through the last row.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BEGIN=""
END=""
LIMIT=""

usage() {
  echo "Usage: scripts/run_agent.sh [--begin N] [--end N | --limit N]" >&2
  exit 2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --begin)
      [[ $# -ge 2 ]] || usage
      BEGIN="$2"
      shift 2
      ;;
    --begin=*)
      BEGIN="${1#*=}"
      shift
      ;;
    --end)
      [[ $# -ge 2 ]] || usage
      END="$2"
      shift 2
      ;;
    --end=*)
      END="${1#*=}"
      shift
      ;;
    --limit)
      [[ $# -ge 2 ]] || usage
      LIMIT="$2"
      shift 2
      ;;
    --limit=*)
      LIMIT="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      ;;
    *)
      if [[ -z "$LIMIT" && -z "$END" && "$1" =~ ^[0-9]+$ ]]; then
        LIMIT="$1"
        shift
      else
        usage
      fi
      ;;
  esac
done

if [[ -n "$END" && -n "$LIMIT" ]]; then
  echo "use either --end or --limit" >&2
  exit 2
fi
if [[ -z "$BEGIN" ]]; then
  BEGIN=1
fi
for value in "$BEGIN" ${END:+"$END"} ${LIMIT:+"$LIMIT"}; do
  if [[ ! "$value" =~ ^[0-9]+$ || "$value" -lt 1 ]]; then
    echo "begin, end, and limit must be integers starting at 1" >&2
    exit 2
  fi
done
if [[ -n "$END" && "$END" -lt "$BEGIN" ]]; then
  echo "--end must be at least --begin" >&2
  exit 2
fi

cd "$ROOT"
PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "missing $PYTHON" >&2
  echo "create it with: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
if [[ -n "$LIMIT" ]]; then
  END=$((BEGIN + LIMIT - 1))
elif [[ -z "$END" ]]; then
  END="$("$PYTHON" -c "from pathlib import Path; print(sum(1 for line in Path('train+public_test/public_test/public_test.jsonl').open() if line.strip()))")"
fi

MAIN="$ROOT/outputs/public_test/submission.jsonl"
ONE="$ROOT/outputs/.one_submission.jsonl"
mkdir -p "$ROOT/outputs/public_test"
if [[ "$BEGIN" -eq 1 ]]; then
  : > "$MAIN"
fi
skipped=0
for ((row=BEGIN; row<=END; row++)); do
    rm -f "$ONE"

    if "$PYTHON" src/run.py \
        --split public_test \
        --begin "$row" \
        --end "$row" \
        --output "$ONE"; then

        if [[ -s "$ONE" ]]; then
            cat "$ONE" >> "$MAIN"
        else
            echo "empty output at row $row" >&2
            skipped=$((skipped + 1))
        fi
    else
        echo "skip row $row" >&2
        skipped=$((skipped + 1))
    fi
done
rm -f "$ONE"
echo "skipped $skipped" >&2
