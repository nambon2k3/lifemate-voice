"""Turn the tools on one ToolEnv session into schemas for ``llm.bind_tools``."""

from __future__ import annotations


def tool_definitions(env) -> list[dict]:
    """Build OpenAI function schemas for the tools this session may call.

    Input:
        env (Session): one ToolEnv session. ``allowed`` is the installed tool
        names plus ``sys.ask_missing_info``. ``tools`` is the catalog spec keyed
        by those names.

    Output:
        list[dict]: one ``{"type": "function", "function": {...}}`` per allowed
        tool that exists in the catalog. Names are not hardcoded.
    """
    definitions = []
    for name in sorted(env.allowed):
        spec = env.tools.get(name)
        if not spec:
            continue
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": spec.get("name") or name,
                    "description": spec.get("description") or name,
                    "parameters": _compact_parameters(spec.get("parameters")),
                },
            }
        )
    return definitions


def _compact_parameters(parameters: dict | None) -> dict:
    """Keep the fields the model must fill, and drop catalog prose.

    Input:
        parameters (dict | None): JSON Schema from the tool catalog. None becomes
        an empty object schema.

    Output:
        dict: schema with ``type``, ``properties`` (``type`` and ``enum`` only),
        and ``required``. Descriptions and vendor keys are left out so the
        bound tools stay inside the planner context.
    """
    parameters = parameters or {}
    properties = {}
    for key, spec in (parameters.get("properties") or {}).items():
        if not isinstance(spec, dict):
            continue
        prop = {"type": spec.get("type", "string")}
        if spec.get("enum"):
            prop["enum"] = spec["enum"]
        properties[key] = prop
    compact = {"type": "object", "properties": properties}
    if parameters.get("required"):
        compact["required"] = parameters["required"]
    return compact
