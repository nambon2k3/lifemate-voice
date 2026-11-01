"""Score a submission against train labels for the held-out ids.

Reads ``scripts/holdout_ids.txt`` (one sample id per line) and a submission
JSONL. Writes a short report under ``outputs/``. Compares each prediction's
``tool_calls`` with ``label.tool_calls`` and tags the first mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOLDOUT_PATH = Path(__file__).with_name("holdout_ids.txt")


def load_ids(path: Path) -> list[str]:
    """Read the holdout sample ids.

    Input:
        path (Path): text file such as scripts/holdout_ids.txt. Blank lines and
        '#' comments are ignored. A missing file is treated as empty.

    Output:
        list[str]: sample ids in file order, such as ``train_0001``.
    """
    ids = []
    if not path.is_file():
        return ids
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def load_jsonl(path: Path) -> dict[str, dict]:
    """Load a JSONL file and index its rows by sample id.

    Input:
        path (Path): train.jsonl or a submission.jsonl. Each line must contain ``id``.

    Output:
        dict[str, dict]: sample id to the parsed row. A later line with the same id replaces the earlier one.
    """
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows[row["id"]] = row
    return rows


def tag_miss(gold_calls: list, pred_calls: list) -> str:
    """Name the first way a prediction disagrees with the gold calls.

    Input:
        gold_calls (list[dict]): ``label.tool_calls``. Each item has ``name`` and ``arguments``.
        pred_calls (list[dict]): ``tool_calls`` from the submission line, same shape.

    Output:
        str: ``extra_call``, ``missing_call``, ``wrong_tool``, ``bad_argument``, or ``hash_mismatch``.
        ``hash_mismatch`` means the call lists match, so a future hash check is what failed.
    """
    if not gold_calls and pred_calls:
        return "extra_call"
    if gold_calls and not pred_calls:
        return "missing_call"
    gold_names = [call.get("name") for call in gold_calls]
    pred_names = [call.get("name") for call in pred_calls]
    if gold_names != pred_names:
        return "wrong_tool"
    if gold_calls != pred_calls:
        return "bad_argument"
    return "hash_mismatch"


def score(gold_rows: dict[str, dict], pred_rows: dict[str, dict], ids: list[str]) -> dict:
    """Compare predicted tool calls with train labels for the given ids.

    Input:
        gold_rows (dict[str, dict]): train rows keyed by id, including ``label.tool_calls``.
        pred_rows (dict[str, dict]): submission rows keyed by id, including ``tool_calls``.
        ids (list[str]): holdout ids to score. Ids missing from either map are counted, not scored.

    Output:
        dict: ``n`` (int, holdout size), ``scored`` (int), ``call_match`` (int, exact call-list matches),
        ``missing_prediction`` (int), and ``tags`` (dict[str, int] of miss reasons).
    """
    matched = 0
    tags: dict[str, int] = {}
    missing_pred = 0
    for sample_id in ids:
        gold = gold_rows.get(sample_id)
        pred = pred_rows.get(sample_id)
        if gold is None or pred is None:
            missing_pred += 1
            continue
        gold_calls = (gold.get("label") or {}).get("tool_calls") or []
        pred_calls = pred.get("tool_calls") or []
        if gold_calls == pred_calls:
            matched += 1
            continue
        label = tag_miss(gold_calls, pred_calls)
        tags[label] = tags.get(label, 0) + 1
    return {
        "n": len(ids),
        "scored": len(ids) - missing_pred,
        "call_match": matched,
        "missing_prediction": missing_pred,
        "tags": tags,
    }


def main() -> None:
    """Score a submission on the holdout ids and write the report.

    Input:
        none. Reads ``sys.argv``: ``--submission``, ``--holdout``, ``--train``, and ``--report`` (all Path).

    Output:
        None. Writes the score dict as JSON to ``--report`` and prints the same object.
    """
    parser = argparse.ArgumentParser(description="Score a train submission on the holdout ids")
    parser.add_argument("--submission", type=Path, default=REPO_ROOT / "outputs" / "submission.jsonl")
    parser.add_argument("--holdout", type=Path, default=HOLDOUT_PATH)
    parser.add_argument(
        "--train",
        type=Path,
        default=REPO_ROOT / "train+public_test" / "train" / "train.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=REPO_ROOT / "outputs" / "score_dev.json",
    )
    args = parser.parse_args()

    ids = load_ids(args.holdout)
    if not ids:
        raise SystemExit(f"add sample ids to {args.holdout}")
    if not args.submission.is_file():
        raise SystemExit(f"no submission at {args.submission}")

    report = score(load_jsonl(args.train), load_jsonl(args.submission), ids)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
