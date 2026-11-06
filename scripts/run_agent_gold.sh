#!/usr/bin/env bash
# Run the agent on train samples using gold audio_text as the latest request.
# Does not call the ASR server and does not read the transcript cache.
# Writes outputs/gold/submission.jsonl so the ASR submission is left in place.
# Usage: scripts/run_agent_gold.sh --begin 32 --end 50
#        scripts/run_agent_gold.sh --limit 10
#        scripts/run_agent_gold.sh 10
# Samples are counted from 1, the first row of the train file.
# Omit --begin to start at 1. Omit --end and --limit to run through the last row.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BEGIN=""
END=""
LIMIT=""
OUTPUT="outputs/gold/submission.jsonl"

usage() {
  echo "Usage: scripts/run_agent_gold.sh [--begin N] [--end N | --limit N]" >&2
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
ARGS=(src/run.py --split train --gold --output "$OUTPUT" --begin "$BEGIN")
if [[ -n "$END" ]]; then
  ARGS+=(--end "$END")
elif [[ -n "$LIMIT" ]]; then
  ARGS+=(--limit "$LIMIT")
fi
"$PYTHON" "${ARGS[@]}"
