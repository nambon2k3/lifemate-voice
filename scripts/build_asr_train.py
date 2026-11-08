"""Build a Qwen3-ASR fine-tune JSONL from the train split.

Reads ``train+public_test/train/train.jsonl``. Each user turn that has a wav
and ``audio_text`` becomes one line. Assistant turns have no audio and are
skipped. The current request and every user turn in ``history`` are both kept.

The output matches ``qwen3_asr_sft.py``: an ``audio`` path and a ``text``
field that starts with ``language Vietnamese<asr_text>``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "train+public_test" / "train" / "train.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "asr_train.jsonl"
TEXT_PREFIX = "language Vietnamese<asr_text>"


def user_turns(row: dict) -> list[tuple[str, str]]:
    """Collect the spoken user turns in one sample.

    Input:
        row (dict): one line of train.jsonl. ``history`` is oldest first.
        ``audio`` and ``audio_text`` are the latest request.

    Output:
        list[tuple[str, str]]: ``(wav path relative to the train dir, transcript)``
        in time order. Turns with no wav or no transcript are left out.
    """
    turns = []
    for item in row.get("history") or []:
        if item.get("role") not in (None, "user"):
            continue
        audio = item.get("audio")
        text = item.get("audio_text")
        if isinstance(audio, str) and isinstance(text, str) and audio.strip() and text.strip():
            turns.append((audio.strip(), " ".join(text.split())))
    audio = row.get("audio")
    text = row.get("audio_text")
    if isinstance(audio, str) and isinstance(text, str) and audio.strip() and text.strip():
        turns.append((audio.strip(), " ".join(text.split())))
    return turns


def audio_path(relative: str, prefix: str) -> str:
    """Join an optional directory in front of a split-relative wav path.

    Input:
        relative (str): path from train.jsonl, such as
        ``audio/train_0001/train_0001.wav``.
        prefix (str): directory that contains that relative path on the
        training machine. Empty leaves the path unchanged.

    Output:
        str: the path written into the fine-tune JSONL.
    """
    if not prefix:
        return relative
    return str(Path(prefix) / relative)


def build(input_path: Path, output_path: Path, prefix: str, train_root: Path | None) -> tuple[int, int]:
    """Write the fine-tune JSONL.

    Input:
        input_path (Path): the kit train.jsonl.
        output_path (Path): destination JSONL. Parent directories are created.
        prefix (str): prepended to each wav path. Empty keeps the kit-relative path.
        train_root (Path | None): directory that contains ``audio/``. When set,
        a line is skipped if that wav is not on disk.

    Output:
        tuple[int, int]: lines written, and lines skipped because the wav was missing.
    """
    written = 0
    missing = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as sink:
        for line in source:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for relative, text in user_turns(row):
                if train_root is not None and not (train_root / relative).is_file():
                    missing += 1
                    continue
                record = {"audio": audio_path(relative, prefix), "text": f"{TEXT_PREFIX}{text}"}
                sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
    return written, missing


def main() -> None:
    """Parse arguments and write the fine-tune JSONL."""
    parser = argparse.ArgumentParser(description="Build a Qwen3-ASR train JSONL from train.jsonl.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Kit train.jsonl.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="JSONL for qwen3_asr_sft.py.")
    parser.add_argument(
        "--audio-prefix",
        default="",
        help="Directory on the training machine that contains audio/. Example: /content/train.",
    )
    parser.add_argument(
        "--train-root",
        type=Path,
        default=DEFAULT_INPUT.parent,
        help="Local train directory used to check that each wav exists. Pass an empty value to skip the check.",
    )
    args = parser.parse_args()
    train_root = args.train_root if str(args.train_root) else None
    written, missing = build(args.input, args.output, args.audio_prefix, train_root)
    print(f"wrote {written} lines -> {args.output}")
    if missing:
        print(f"skipped {missing} turns whose wav was missing under {train_root}")


if __name__ == "__main__":
    main()
