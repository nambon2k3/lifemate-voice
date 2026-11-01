"""Run one sample or a full split and write submission JSONL.

Each sample gets one ``ToolEnv``. ``tool_calls`` is ``env.get_tool_calls()``
and ``state_hash`` is ``env.get_state_hash()``, including any rejected calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from asr import transcribe
from context import index_catalog, render
from agent import run as run_agent

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    settings: dict = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        settings[key.strip()] = _coerce(value.strip())
    return settings


def _coerce(value: str):
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tool_env(kit_root: Path, split: str):
    split_dir = str(kit_root / split)
    # Append so this repo's src.validate is not shadowed by zaic_tools.validate.
    if split_dir not in sys.path:
        sys.path.append(split_dir)
    from zaic_tools import ToolEnv  # noqa: WPS433

    return ToolEnv


def submission_line(sample_id: str, env) -> dict:
    return {
        "id": sample_id,
        "tool_calls": env.get_tool_calls(),
        "state_hash": env.get_state_hash(),
    }


def run_split(settings: dict, sample_id: str | None, limit: int | None) -> int:
    kit_root = (REPO_ROOT / settings["kit_root"]).resolve()
    split = settings["split"]
    rows = load_jsonl(kit_root / split / f"{split}.jsonl")
    if sample_id is not None:
        rows = [row for row in rows if row["id"] == sample_id]
        if not rows:
            raise SystemExit(f"no sample {sample_id} in {split}")
    if limit is not None:
        rows = rows[:limit]

    catalog = json.loads(
        (kit_root / split / f"{split}_tools_catalog.json").read_text(encoding="utf-8")
    )
    tools_by_name = index_catalog(catalog)
    tool_env = load_tool_env(kit_root, split)
    output = REPO_ROOT / settings["output_path"]
    output.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for sample in rows:
            transcript = transcribe(sample, sample["audio"], settings)
            context_text = render(sample, tools_by_name, transcript)
            env = tool_env(split, sample["id"], root=kit_root)
            run_agent(sample, env, transcript, context_text, tools_by_name, settings)
            handle.write(
                json.dumps(submission_line(sample["id"], env), ensure_ascii=False) + "\n"
            )
            count += 1
    print(f"{count} lines -> {output}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a LifeMate submission.jsonl")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "agent.yaml")
    parser.add_argument("--split", help="Override configs/agent.yaml split")
    parser.add_argument("--id", help="Run a single sample id")
    parser.add_argument("--limit", type=int, help="Stop after this many samples")
    parser.add_argument(
        "--gold",
        action="store_true",
        help="Use train audio_text for the latest request instead of ASR",
    )
    args = parser.parse_args()
    settings = load_config(args.config)
    if args.split:
        settings["split"] = args.split
    if args.gold:
        settings["use_gold_transcript"] = True
    for key in ("cache_dir", "output_path", "kit_root"):
        settings[key] = str(settings[key])
    run_split(settings, args.id, args.limit)


if __name__ == "__main__":
    main()
