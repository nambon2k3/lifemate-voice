#!/usr/bin/env bash
# Transcribe train wavs through the ASR server. Does not use gold text.
# Usage: scripts/run_asr.sh --begin 32 --end 50
#        scripts/run_asr.sh --limit 10
#        scripts/run_asr.sh 10
# Samples are counted from 1, the first row of the train file.
# Omit --begin to start at 1. Omit --end and --limit to transcribe through the last row.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JSONL="$ROOT/train+public_test/train/train.jsonl"
AUDIO_ROOT="$ROOT/train+public_test/train"
BEGIN=""
END=""
LIMIT=""

usage() {
  echo "Usage: scripts/run_asr.sh [--begin N] [--end N | --limit N]" >&2
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
if [[ -n "$LIMIT" ]]; then
  END=$((BEGIN + LIMIT - 1))
fi
if [[ -n "$END" && "$END" -lt "$BEGIN" ]]; then
  echo "--end must be at least --begin" >&2
  exit 2
fi

cd "$ROOT"
python3 - "$JSONL" "$AUDIO_ROOT" "$BEGIN" "$END" << 'PY'
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
from asr import transcribe
from env import apply_env, env_value

jsonl = Path(sys.argv[1])
audio_root = Path(sys.argv[2])
begin = int(sys.argv[3])
end = int(sys.argv[4]) if sys.argv[4] else None
if begin < 1:
    raise SystemExit("--begin must be at least 1")
if end is not None and end < begin:
    raise SystemExit("--end must be at least --begin")

settings = {
    "cache_dir": str(Path("cache/transcripts").resolve()),
    "kit_root": "train+public_test",
    "split": "train",
    "use_gold_transcript": False,
}
apply_env(settings)
settings["cache_dir"] = str(Path("cache/transcripts").resolve())
settings["kit_root"] = "train+public_test"
settings["split"] = "train"
settings["use_gold_transcript"] = False
settings["asr_served_model"] = env_value("ASR_SERVED_MODEL", "qwen3-asr")

rows = []
for line in jsonl.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    rows.append(json.loads(line))

rows = rows[begin - 1:end]
for index, row in enumerate(rows, start=begin):
    wav = audio_root / row["audio"]
    if not wav.is_file():
        raise SystemExit(f"missing wav for {row['id']}: {wav}")
    text = transcribe(row, row["audio"], settings)
    print(f"{index:02d} {row['id']}: {text}", flush=True)

print(f"{len(rows)} transcripts -> cache/transcripts", flush=True)
PY
