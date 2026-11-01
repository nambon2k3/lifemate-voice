"""Reject a tool call before ``env.call``.

A call is refused when the tool is not installed on this sample, the arguments
miss the catalog schema, or a write argument marked ``x-ref`` was never seen in
context or in an earlier tool result. ``env.validate`` remains the organizer's
schema check and still runs inside ``env.call``.
"""

from __future__ import annotations


def collect_ids(payload) -> set[str]:
    """Collect identifier strings from a tool result or a context object.

    Input:
        payload (dict | list | str | any): nested JSON. Only strings whose key is
        ``id`` or ends with ``_id`` are kept.

    Output:
        set[str]: those identifier values, for example ``evt_01`` or ``alm_02``.
    """
    found: set[str] = set()

    def walk(node, key: str | None = None) -> None:
        """Walk one JSON node and record id-like string leaves into ``found``.

        Input:
            node (any): current value.
            key (str | None): dict key that held ``node``, or None at the root.

        Output:
            None. Mutates ``found``.
        """
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
    """Decide whether a planned call may be sent to ``env.call``.

    Input:
        name (str): tool name, such as ``door_lock.set_lock``.
        arguments (dict | any): argument object the model produced.
        installed (set[str]): tool names this sample may call.
        tools_by_name (dict[str, dict]): catalog index from ``index_catalog``.
        seen_ids (set[str]): ids already present in context or earlier tool results.

    Output:
        str: a short refusal reason. The caller should not execute the call.
        None: the call may be executed. ``env.call`` still applies the organizer's schema check.
    """
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
