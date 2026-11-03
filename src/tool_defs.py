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
        parameters = spec.get("parameters") or {"type": "object", "properties": {}}
        definitions.append(
            {
                "type": "function",
                "function": {
                    "name": spec.get("name") or name,
                    "description": spec.get("description") or name,
                    "parameters": parameters,
                },
            }
        )
    return definitions
