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
        ``--id`` (str, optional), ``--begin`` (int, default 1), ``--end`` (int, optional),
        ``--limit`` (int, optional), ``--gold`` (flag), ``--output`` (Path, optional).

    Output:
        None. Writes submission.jsonl through ``run_split`` and prints the line count.
    """
    parser = argparse.ArgumentParser(description="Write a LifeMate submission.jsonl")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "agent.yaml")
    parser.add_argument("--split", help="Override configs/agent.yaml split")
    parser.add_argument("--id", help="Run a single sample id")
    parser.add_argument("--begin", type=int, default=1, help="First sample, counting from 1")
    parser.add_argument("--end", type=int, help="Last sample, counting from 1")
    parser.add_argument("--limit", type=int, help="How many samples to run, starting at --begin")
    parser.add_argument(
        "--gold",
        action="store_true",
        help="Use train audio_text for the latest request instead of ASR",
    )
    parser.add_argument("--output", type=Path, help="Override the submission JSONL path")
    args = parser.parse_args()
    settings = load_config(args.config)
    if args.split:
        settings["split"] = args.split
    if args.gold:
        settings["use_gold_transcript"] = True
    if args.output:
        settings["output_path"] = args.output
    for key in ("cache_dir", "output_path", "kit_root"):
        settings[key] = str(settings[key])
    if args.begin < 1:
        parser.error("--begin must be at least 1")
    if args.end is not None and args.limit is not None:
        parser.error("use either --end or --limit")
    end = args.end
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        end = args.begin + args.limit - 1
    if end is not None and end < args.begin:
        parser.error("--end must be at least --begin")
    run_split(settings, args.id, args.begin, end)


if __name__ == "__main__":
    main()
