python3 <<'PY'
import json
from pathlib import Path

cache_dir = Path("cache/gold/transcripts")
count = 0

for path in cache_dir.glob("*.json"):
    data = json.loads(path.read_text(encoding="utf-8"))

    turns = data.get("turns", [])
    if len(turns) <= 1:
        continue

    # Move original turn (index 0) to the end
    turns = turns[1:] + turns[:1]

    # Re-index turns according to the new order
    for i, turn in enumerate(turns):
        turn["turn"] = i

    data["turns"] = turns

    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    count += 1

print(f"Reordered {count} JSON files")
PY