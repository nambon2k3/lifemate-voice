"""Shared loaders and the split runner used by the submission entry point."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agent import run as run_agent
from asr import transcribe
from context import index_catalog, render
from correct import correct_slots
from env import ENV_KEYS, apply_env

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path) -> dict:
    """Read a flat YAML config into a settings dict.

    A matching variable in ``.env`` or the process environment replaces the
    yaml value. See ``env.ENV_KEYS``.

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
    apply_env(settings)
    for setting_key in ENV_KEYS.values():
        if isinstance(settings.get(setting_key), str):
            settings[setting_key] = _coerce(settings[setting_key])
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


def _fill_history_text(sample: dict, settings: dict) -> dict:
    """Transcribe history user turns that only have a wav path.

    Train rows already store ``audio_text``. Public rows store ``audio``.
    Each missing sentence goes through ``transcribe``, the same path as the
    latest request: the JSON transcript is the ASR ground truth while testing,
    and a cache miss calls the ASR server. Assistant tool calls stay as they
    are. The kit file is not written.

    Input:
        sample (dict): one JSONL row. ``history`` may mix user wavs and
        assistant tool results.
        settings (dict): ``cache_dir`` and the ASR settings ``transcribe`` reads.

    Output:
        dict: ``sample`` when every user turn already has text. A shallow copy
        with ``audio_text`` filled on the user turns that only had ``audio``.
    """
    history = sample.get("history") or []
    if not any(
        turn.get("role") == "user" and not turn.get("audio_text") and turn.get("audio")
        for turn in history
    ):
        return sample
    filled = []
    for turn in history:
        if turn.get("role") == "user" and not turn.get("audio_text") and turn.get("audio"):
            text = transcribe(sample, turn["audio"], settings)
            filled.append({**turn, "audio_text": text})
        else:
            filled.append(turn)
    return {**sample, "history": filled}


def _using_gold(settings: dict, sample: dict, transcript: str) -> bool:
    """Return whether this transcript is the train gold sentence.

    Input:
        settings (dict): ``use_gold_transcript`` selects the gold sentence for
        the latest request.
        sample (dict): one JSONL row. ``audio_text`` is that gold sentence.
        transcript (str): the text ``transcribe`` just returned.

    Output:
        bool: True when the planner was given the gold sentence, so slot
        correction must not edit it.
    """
    gold = sample.get("audio_text")
    return bool(settings.get("use_gold_transcript") and isinstance(gold, str) and transcript == gold)


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


def run_split(
    settings: dict,
    sample_id: str | None,
    begin: int = 1,
    end: int | None = None,
) -> int:
    """Transcribe each selected sample, run the agent, and write submission.jsonl.

    An ASR transcript is passed through ``correct_slots`` before ``render``.
    A gold sentence selected with ``use_gold_transcript`` is left unchanged.
    User turns in ``history`` that only have a wav are transcribed first.
    ``begin`` and ``end`` count rows in the split file, starting at 1.

    Input:
        settings (dict): config from ``load_config``, including ``split``, ``kit_root``,
        ``cache_dir``, and ``output_path``.
        sample_id (str | None): run only this id, or the selected rows when None.
        begin (int): first row to run. 1 is the first row in the file.
        end (int | None): last row to run, inclusive. None runs through the last row.

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
    else:
        if begin < 1:
            raise SystemExit("--begin must be at least 1")
        if end is not None and end < begin:
            raise SystemExit("--end must be at least --begin")
        rows = rows[begin - 1:end]

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
            sample = _fill_history_text(sample, settings)
            transcript = transcribe(sample, sample["audio"], settings)
            if not _using_gold(settings, sample, transcript):
                transcript = correct_slots(transcript, sample)
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
