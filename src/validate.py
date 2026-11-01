"""Reject a tool call before ``env.call``.

A call is refused when the tool is not installed on this sample, the arguments
miss the catalog schema, or a write argument marked ``x-ref`` was never seen in
context or in an earlier tool result. ``env.validate`` remains the organizer's
schema check and still runs inside ``env.call``.
"""

from __future__ import annotations


def collect_ids(payload) -> set[str]:
    """Gather id-like strings from a tool result or a context object."""
    found: set[str] = set()

    def walk(node, key: str | None = None) -> None:
        if isinstance(node, dict):
            for child_key, child in node.items():
                walk(child, child_key)
            return
        if isinstance(node, list):
            for child in node:
                walk(child, key)
            return
        if isinstance(node, str) and key and (key == "id" or key.endswith("_id")):
            found.add(node)

    walk(payload)
    return found


def reject_reason(
    name: str,
    arguments,
    installed: set[str],
    tools_by_name: dict[str, dict],
    seen_ids: set[str],
) -> str | None:
    """Return a short reason to skip ``env.call``, or ``None`` when the call may run."""
    if name not in installed:
        return f"tool not installed: {name}"
    tool = tools_by_name.get(name)
    if tool is None:
        return f"tool not in catalog: {name}"
    if not isinstance(arguments, dict):
        return "arguments must be an object"

    params = tool.get("parameters") or {}
    required = params.get("required") or []
    props = params.get("properties") or {}
    missing = [key for key in required if key not in arguments]
    if missing:
        return "missing required: " + ", ".join(missing)

    for key, value in arguments.items():
        spec = props.get(key)
        if spec is None:
            return f"unknown argument: {key}"
        enum = spec.get("enum")
        if enum is not None and value not in enum:
            return f"{key} not in enum"
        if spec.get("x-ref") and tool.get("kind") == "write" and str(value) not in seen_ids:
            return f"{key} was not returned by an earlier result"
    return None
