"""Shared loaders and the split runner used by the submission entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent import run as run_agent
from asr import transcribe
from context import index_catalog, render

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    """Read a flat YAML config into a settings dict.

    Input:
        path (Path): file such as configs/agent.yaml. '#' comments and blank lines are ignored.

    Output:
        dict[str, bool | int | float | str]: one entry per ``key: value`` line.
        Values are coerced by ``_coerce``.
    """
    settings: dict = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        settings[key.strip()] = _coerce(value.strip())
    return settings


def _coerce(value: str):
    """Turn one raw config value into a Python value.

    Input:
        value (str): text after the colon, already stripped.

    Output:
        bool: ``true`` or ``false``.
        int: a whole number such as ``8``.
        float: a decimal number.
        str: anything else, including model names and paths.
    """
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
    """Load a JSON Lines file in file order.

    Input:
        path (Path): a split file such as train.jsonl or public_test.jsonl.
        Blank lines are skipped.

    Output:
        list[dict]: one sample dict per line, with keys such as ``id``, ``audio``,
        ``context``, and ``history``. Train rows also have ``audio_text`` and ``label``.
    """
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_tool_env(kit_root: Path, split: str):
    """Import the official ``ToolEnv`` constructor for one split.

    Input:
        kit_root (Path): directory that contains the split folders, usually train+public_test.
        split (str): ``train`` or ``public_test``. Selects ``<kit_root>/<split>/zaic_tools``.

    Output:
        Callable: ``ToolEnv(split, question_id, root) -> Session``. The caller creates
        one session per sample. That session loads state.json and runs tool calls.
    """
    split_dir = str(kit_root / split)
    # Append so this repo's src.validate is not shadowed by zaic_tools.validate.
    if split_dir not in sys.path:
        sys.path.append(split_dir)
    from zaic_tools import ToolEnv  # noqa: WPS433

    return ToolEnv


def write_state(sample_id: str, env, output: Path) -> Path:
    """Write the session state that ``get_state_hash`` hashes.

    Input:
        sample_id (str): sample id, used as the file name.
        env (Session): the session after the agent has finished.
        output (Path): submission JSONL path. The state folder is created beside it.

    Output:
        Path: ``<output dir>/states/<sample_id>.json``, the app state from ``env.get_state()``.
    """
    folder = output.parent / "states"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{sample_id}.json"
    path.write_text(
        json.dumps(env.get_state(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def write_tool_results(sample_id: str, env, output: Path) -> Path:
    """Write the result of every tool call in this session.

    Input:
        sample_id (str): sample id, used as the file name.
        env (Session): the session after the agent has finished. ``env.log`` holds
        each call's name, arguments, and result, in the same order as ``get_tool_calls``.
        output (Path): submission JSONL path. The folder is created beside it.

    Output:
        Path: ``<output dir>/tool_results/<sample_id>.json``.
    """
    folder = output.parent / "tool_results"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{sample_id}.json"
    path.write_text(
        json.dumps(env.log, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def submission_line(sample_id: str, env) -> dict:
    """Build one submission record from the session the agent just used.

    Input:
        sample_id (str): sample id, such as ``train_0001``.
        env (Session): the single ToolEnv session for that sample. Its recorded calls
        may include calls the simulator rejected.

    Output:
        dict: ``id`` (str), ``tool_calls`` (list of ``{name, arguments}``), and
        ``state_hash`` (str, sha256 from the simulator).
    """
    return {
        "id": sample_id,
        "tool_calls": env.get_tool_calls(),
        "state_hash": env.get_state_hash(),
    }


def run_split(settings: dict, sample_id: str | None, limit: int | None) -> int:
    """Transcribe each selected sample, run the agent, and write submission.jsonl.

    Input:
        settings (dict): config from ``load_config``, including ``split``, ``kit_root``,
        ``cache_dir``, and ``output_path``.
        sample_id (str | None): run only this id, or every row in the split when None.
        limit (int | None): keep only the first N selected rows. None means no cap.

    Output:
        int: number of JSONL lines written to ``settings["output_path"]``.
        Each line is one ``submission_line``.
    """
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
            context_text = render(sample, transcript)
            env = tool_env(split, sample["id"], root=kit_root)
            run_agent(sample, env, context_text, tools_by_name, settings)
            handle.write(
                json.dumps(submission_line(sample["id"], env), ensure_ascii=False) + "\n"
            )
            write_state(sample["id"], env, output)
            write_tool_results(sample["id"], env, output)
            count += 1
    print(f"{count} lines -> {output}")
    return count
