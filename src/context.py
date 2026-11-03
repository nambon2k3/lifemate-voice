"""Render sample context and history as the planner text block.

Reads ``context`` and ``history``. Writes nothing. Tool schemas are bound
on the model separately, so this text does not list tools.
``current_time`` is parsed so today and tomorrow are explicit in the prompt.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta


def index_catalog(catalog: dict) -> dict[str, dict]:
    """Index every tool in a split catalog by its full name.

    Input:
        catalog (dict): parsed ``<split>_tools_catalog.json``, with an ``apps`` list.

    Output:
        dict[str, dict]: tool name such as ``alarm.list`` to that tool's schema
        (``kind``, ``description``, ``parameters``, and the rest).
    """
    tools: dict[str, dict] = {}
    for app in catalog.get("apps", []):
        for tool in app.get("tools", []):
            tools[tool["name"]] = tool
    return tools


def installed_tool_names(sample: dict) -> list[str]:
    """List the tools this sample is allowed to call.

    Input:
        sample (dict): one JSONL row. Reads ``context.installed_apps``.

    Output:
        list[str]: ``sys.ask_missing_info`` plus each installed app's tool names,
        in catalog order within each app, with duplicates removed.
    """
    names = ["sys.ask_missing_info"]
    apps = sample.get("context", {}).get("installed_apps", {})
    for info in apps.values():
        for name in info.get("tools", []):
            if name not in names:
                names.append(name)
    return names


def _parse_now(current_time: str) -> datetime | None:
    """Parse the sample clock into a timezone-aware datetime.

    Input:
        current_time (str): ISO time from ``context.current_time``, such as
        ``2026-11-28T07:10:00+07:00``. A trailing ``Z`` is accepted.

    Output:
        datetime: the parsed instant.
        None: when the string is not a valid ISO time.
    """
    text = current_time.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def render(sample: dict, tools_by_name: dict[str, dict], transcript: str) -> str:
    """Build the plain-text block the planner reads. Writes nothing.

    Tool names and parameter schemas are not included. ``bind_tools`` supplies them.

    Input:
        sample (dict): one JSONL row. Uses ``id``, ``context``, and ``history``.
        tools_by_name (dict[str, dict]): unused. Kept so existing callers stay valid.
        transcript (str): text of the latest user request.

    Output:
        str: clock, today, tomorrow, profile, history, and the request.
    """
    del tools_by_name
    context = sample.get("context", {})
    profile = context.get("user_profile") or {}
    location = context.get("location") or {}
    lines = [
        f"sample_id: {sample.get('id', '')}",
        f"current_time: {context.get('current_time', '')}",
        f"timezone: {context.get('timezone', '')}",
    ]
    now = _parse_now(str(context.get("current_time", "")))
    if now is not None:
        lines.append(f"today: {now.date().isoformat()}")
        lines.append(f"tomorrow: {(now + timedelta(days=1)).date().isoformat()}")
    lines.append(
        "location: "
        + json.dumps(location, ensure_ascii=False)
    )
    lines.append(f"name: {profile.get('name', '')}")
    lines.append(f"home_address: {profile.get('home_address', '')}")
    lines.append(f"work_address: {profile.get('work_address', '')}")
    lines.append(f"city: {profile.get('city', '')}")
    lines.append("history:")
    history = sample.get("history") or []
    if not history:
        lines.append("(none)")
    for turn in history:
        role = turn.get("role", "")
        if role == "user":
            text = turn.get("audio_text") or turn.get("audio") or ""
            lines.append(f"user: {text}")
        else:
            lines.append(f"assistant: {turn.get('text', '')}")
            if turn.get("tool_calls"):
                lines.append(
                    "tool_calls: "
                    + json.dumps(turn["tool_calls"], ensure_ascii=False)
                )
            if turn.get("tool_results"):
                lines.append(
                    "tool_results: "
                    + json.dumps(turn["tool_results"], ensure_ascii=False)
                )
    lines.append("latest_request:")
    lines.append(transcript)
    return "\n".join(lines) + "\n"
