"""One tool call per step against a single ``ToolEnv`` session.

The planner calls the local model at temperature 0, ``validate.reject_reason``
runs before ``env.call``, and the tool result is appended before the next step.
Stop on a finish action, on ``sys.ask_missing_info``, or at ``max_steps``.

This module is the placeholder loop: it records no calls. The submission line
still comes from ``env.get_tool_calls()`` and ``env.get_state_hash()``.
"""

from __future__ import annotations


def run(
    sample: dict,
    env,
    transcript: str,
    context_text: str,
    tools_by_name: dict[str, dict],
    settings: dict,
) -> None:
    """Execute the step loop on the one session created for ``sample``.

    ``transcript`` and ``context_text`` are ready for the planner. ``settings``
    supplies ``planner_model``, ``temperature``, and ``max_steps``.
    """
    del sample, env, transcript, context_text, tools_by_name, settings
