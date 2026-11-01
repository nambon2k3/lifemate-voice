"""Run one sample or a full split and write submission JSONL.

Each sample gets one ``ToolEnv``. ``tool_calls`` is ``env.get_tool_calls()``
and ``state_hash`` is ``env.get_state_hash()``, including any rejected calls.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import REPO_ROOT, load_config, run_split

def main() -> None:
    """Parse CLI flags, overlay them on agent.yaml, and run the split.

    Input:
        none. Reads ``sys.argv``: ``--config`` (Path), ``--split`` (str, optional),
        ``--id`` (str, optional), ``--limit`` (int, optional), ``--gold`` (flag).

    Output:
        None. Writes submission.jsonl through ``run_split`` and prints the line count.
    """
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
