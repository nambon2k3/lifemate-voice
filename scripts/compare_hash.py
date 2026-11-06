"""Compare submission state_hash values with the train ground truth.

The train file has no stored hash. Ground truth is the hash of the state after
replaying ``label.tool_calls`` on the sample's original ``state.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
KIT_ROOT = REPO_ROOT / "train+public_test"
RAW_PATH = REPO_ROOT / "outputs" / "submission.jsonl"
GOLD_PATH = REPO_ROOT / "outputs" / "gold" / "submission.jsonl"


def load_jsonl(path: Path, limit: int | None) -> list[dict]:
    """Load JSONL rows in file order.

    Input:
        path (Path): a submission or a split file such as train.jsonl.
        limit (int | None): keep only the first N rows. None keeps every row.

    Output:
        list[dict]: parsed rows. Blank lines are skipped.
    """
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def split_of(sample_id: str) -> str:
    """Return the kit split that owns a sample id.

    Input:
        sample_id (str): ``train_0001`` or ``public_0001``.

    Output:
        str: ``train`` or ``public_test``.
    """
    if sample_id.startswith("train_"):
        return "train"
    if sample_id.startswith("public_"):
        return "public_test"
    raise SystemExit(f"unknown split for {sample_id}")


def gold_hash(split: str, sample_id: str, calls: list[dict]) -> str:
    """Hash the state produced by the gold tool calls.

    Input:
        split (str): ``train`` or ``public_test``.
        sample_id (str): sample id, such as ``train_0001``.
        calls (list[dict]): ``label.tool_calls``. Each item has ``name`` and ``arguments``.

    Output:
        str: ``state_hash`` after replaying ``calls`` from the original state.
    """
    split_dir = str(KIT_ROOT / split)
    if split_dir not in sys.path:
        sys.path.append(split_dir)
    from zaic_tools import ToolEnv  # noqa: WPS433

    env = ToolEnv(split, sample_id, root=KIT_ROOT)
    for call in calls:
        env.call(call.get("name") or "", call.get("arguments") or {})
    return env.get_state_hash()


def miss_kind(gold_calls: list[dict], pred_calls: list[dict]) -> str:
    """Name why two call lists can still be told apart.

    Input:
        gold_calls (list[dict]): ``label.tool_calls``.
        pred_calls (list[dict]): ``tool_calls`` from the submission line.

    Output:
        str: ``tool`` when the tool names differ, ``argument`` when only
        arguments differ, ``hash_only`` when the call lists are equal.
    """
    gold_names = [call.get("name") for call in gold_calls]
    pred_names = [call.get("name") for call in pred_calls]
    if gold_names != pred_names:
        return "tool"
    if gold_calls != pred_calls:
        return "argument"
    return "hash_only"


def score_one(sample_id: str, gold: dict | None, pred: dict | None) -> tuple[str, dict | None]:
    """Score one sample against its gold tool calls.

    Input:
        sample_id (str): sample id, such as ``train_0001``.
        gold (dict | None): the split row, including ``label.tool_calls``.
        pred (dict | None): the submission row. None when that id was not written.

    Output:
        tuple[str, dict | None]: ``match`` when the hash agrees, ``missing_label``
        when the split row has no tool calls, or ``miss`` plus ``{id, kind}``.
        ``kind`` is ``missing`` when the submission has no row for this id.
    """
    label = (gold or {}).get("label") or {}
    calls = label.get("tool_calls")
    if gold is None or calls is None:
        return "missing_label", None
    if pred is None:
        return "miss", {"id": sample_id, "kind": "missing"}
    expected = gold_hash(split_of(sample_id), sample_id, calls)
    if pred.get("state_hash") == expected:
        return "match", None
    return "miss", {"id": sample_id, "kind": miss_kind(calls, pred.get("tool_calls") or [])}


def compare(
    submission: Path,
    labels: dict[str, dict],
    selected: list[dict] | None,
) -> dict:
    """Compare one submission file with ground-truth hashes.

    Input:
        submission (Path): JSONL with ``id`` and ``state_hash``.
        labels (dict[str, dict]): split rows keyed by id, including ``label.tool_calls``.
        selected (list[dict] | None): split rows to score, in file order.
        None scores every row already in the submission.

    Output:
        dict: ``path`` (str), ``compared`` (int), ``hash_match`` (int),
        ``hash_miss`` (int), ``missing_prediction`` (int), ``missing_label`` (int),
        and ``misses`` (list of ``{id, kind}``). ``kind`` ``missing`` is a split
        row in the requested range with no submission line.
    """
    preds = {row["id"]: row for row in load_jsonl(submission, None)}
    if selected is None:
        chosen = [labels[row["id"]] for row in load_jsonl(submission, None) if row["id"] in labels]
        missing_ids = [row["id"] for row in load_jsonl(submission, None) if row["id"] not in labels]
    else:
        chosen = selected
        missing_ids = []
    match = 0
    misses = []
    missing_label = len(missing_ids)
    missing_prediction = 0
    for gold in chosen:
        status, miss = score_one(gold["id"], gold, preds.get(gold["id"]))
        if status == "match":
            match += 1
        elif status == "missing_label":
            missing_label += 1
        elif miss and miss["kind"] == "missing":
            missing_prediction += 1
            misses.append(miss)
        elif miss:
            misses.append(miss)
    hash_misses = [miss for miss in misses if miss["kind"] != "missing"]
    return {
        "path": str(submission),
        "compared": match + len(hash_misses) + missing_prediction,
        "hash_match": match,
        "hash_miss": len(hash_misses),
        "missing_prediction": missing_prediction,
        "missing_label": missing_label,
        "misses": misses,
    }


def label_rows(split: str) -> list[dict]:
    """Load one split file in row order.

    Input:
        split (str): ``train`` or ``public_test``.

    Output:
        list[dict]: every row. Row 1 is the first line of the file.
    """
    path = KIT_ROOT / split / f"{split}.jsonl"
    if not path.is_file():
        raise SystemExit(f"no ground truth at {path}")
    return load_jsonl(path, None)


def main() -> None:
    """Compare one or both submission files and print the hash counts.

    Input:
        none. Reads ``sys.argv``: ``--raw``, ``--gold``, ``--begin``, ``--end``,
        and ``--limit``. ``--raw`` scores ``outputs/submission.jsonl``.
        ``--gold`` scores ``outputs/gold/submission.jsonl``. With neither flag,
        both files that exist are scored. ``--begin`` and ``--end`` are
        1-based row numbers in the split file. ``--limit`` counts rows from
        ``--begin``. Omit all three to score the rows the submission already has.

    Output:
        None. Prints one JSON object per file.
    """
    parser = argparse.ArgumentParser(description="Compare submission state_hash with train ground truth")
    parser.add_argument("--raw", action="store_true", help="Score outputs/submission.jsonl")
    parser.add_argument("--gold", action="store_true", help="Score outputs/gold/submission.jsonl")
    parser.add_argument("--begin", type=int, default=1, help="First split row, counting from 1")
    parser.add_argument("--end", type=int, help="Last split row, counting from 1")
    parser.add_argument("--limit", type=int, help="How many split rows to score, starting at --begin")
    parser.add_argument("count", nargs="?", type=int, help="Same as --limit")
    args = parser.parse_args()
    if args.count is not None and args.limit is not None:
        parser.error("use either --limit or a bare number")
    if args.count is not None:
        args.limit = args.count
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
    ranged = args.begin != 1 or end is not None
    if args.raw or args.gold:
        paths = []
        if args.raw:
            paths.append(RAW_PATH)
        if args.gold:
            paths.append(GOLD_PATH)
    else:
        paths = [path for path in (RAW_PATH, GOLD_PATH) if path.is_file()]
    if not paths:
        raise SystemExit("no submission file")
    cached: dict[str, dict[str, dict]] = {}
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"no submission at {path}")
        rows = load_jsonl(path, None)
        if not rows:
            raise SystemExit(f"empty submission at {path}")
        split = split_of(rows[0]["id"])
        if split not in cached:
            cached[split] = label_rows(split)
        by_id = {row["id"]: row for row in cached[split]}
        selected = cached[split][args.begin - 1:end] if ranged else None
        if ranged and not selected:
            raise SystemExit(f"no split rows in {args.begin}..{end}")
        report = compare(path, by_id, selected)
        if ranged:
            report["begin"] = args.begin
            report["end"] = args.begin + len(selected) - 1
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
