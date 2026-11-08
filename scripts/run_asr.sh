#!/usr/bin/env bash
# Transcribe train wavs through the ASR server. Does not use gold text.
# Usage: scripts/run_asr.sh --begin 32 --end 50
#        scripts/run_asr.sh --limit 10
#        scripts/run_asr.sh 10
# Samples are counted from 1, the first row of the train file.
# Omit --begin to start at 1. Omit --end and --limit to transcribe through the last row.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JSONL="$ROOT/train+public_test/public_test/public_test.jsonl"
AUDIO_ROOT="$ROOT/train+public_test/public_test"
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
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from asr import transcribe
from env import apply_env, env_value

jsonl = Path(sys.argv[1])
audio_root = Path(sys.argv[2])
begin = int(sys.argv[3])
end = int(sys.argv[4]) if sys.argv[4] else None

settings = {
    "cache_dir": str(Path("cache/gold/transcripts").resolve()),
    "kit_root": "train+public_test",
    "split": "public_test",
    "use_gold_transcript": False,
}
apply_env(settings)

settings["cache_dir"] = str(Path("cache/gold/transcripts").resolve())
settings["kit_root"] = "train+public_test"
settings["split"] = "public_test"
settings["use_gold_transcript"] = False
settings["asr_served_model"] = env_value(
    "ASR_SERVED_MODEL", "qwen3-asr"
)
settings["disable_cache"] = True
rows = [
    json.loads(line)
    for line in jsonl.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

rows = rows[begin - 1:end]

cache_dir = Path(settings["cache_dir"])
cache_dir.mkdir(parents=True, exist_ok=True)

for index, row in enumerate(rows, start=begin):
    base_wav = audio_root / row["audio"]

    if not base_wav.is_file():
        raise SystemExit(f"Missing wav: {base_wav}")

    pattern = re.compile(
        re.escape(base_wav.stem) + r"_u(\d+)$"
    )

    turn_files = []

    for candidate in base_wav.parent.glob(
        f"{base_wav.stem}_u*.wav"
    ):
        match = pattern.fullmatch(candidate.stem)
        if match:
            turn_files.append((int(match.group(1)), candidate))

    turn_files.sort(key=lambda item: item[0])

    audio_files = [
        path for _, path in turn_files
    ] + [base_wav]

    print(
        f"\n[{index}] {row['id']}: "
        f"{len(audio_files)} turns",
        flush=True,
    )

    conversation_transcripts = []

    for turn_index, wav in enumerate(audio_files):
        audio_rel = wav.relative_to(audio_root).as_posix()

        # Keep the original conversation ID.
        # Disable individual turn cache writes by collecting
        # results and saving the conversation ourselves.
        text = transcribe(
            row,
            audio_rel,
            settings,
        )

        conversation_transcripts.append({
            "turn": turn_index,
            "audio": audio_rel,
            "text": text,
        })

        print(
            f"  Turn {turn_index}: {text}",
            flush=True,
        )

    output_file = cache_dir / f"{row['id']}.json"

    output_file.write_text(
        json.dumps(
            {
                "id": row["id"],
                "turns": conversation_transcripts,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"  Saved: {output_file}", flush=True)

print(
    f"\nCompleted: {len(rows)} conversations",
    flush=True,
)
PY
