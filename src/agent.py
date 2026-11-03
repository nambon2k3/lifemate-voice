"""One LangGraph loop per ToolEnv session.

The agent node calls the local model. Tool calls are executed with ``env.call``
and returned as ``ToolMessage`` values that keep the model's ``tool_call_id``.
A new graph is built for every sample, so sessions do not share messages or tools.
"""

from __future__ import annotations

import json
from typing import Annotated, TypedDict

from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from tool_defs import tool_definitions

_RULES = """
Bạn là LifeMate. Hãy thực hiện yêu cầu mới nhất bằng các tool đã được gắn.
Chỉ gọi tool đã gắn. Không đoán id, số tiền, người nhận hay thiết bị.
Nếu thiếu tham số bắt buộc và không suy ra được từ context, history hay kết quả tool trước, gọi sys.ask_missing_info với tool_name và danh sách params còn thiếu.
Nếu không app nào làm được việc, đừng gọi tool.
Gọi tool cho bước tiếp theo, đọc kết quả, rồi mới quyết định bước sau.
Khi xong việc, dừng và không gọi thêm tool.
"""

_CHAT = None
_CHAT_KEY: tuple | None = None


class AgentState(TypedDict):
    """Conversation and call count for a single sample."""

    messages: Annotated[list, add_messages]
    steps: int


def run(
    sample: dict,
    env,
    transcript: str,
    context_text: str,
    tools_by_name: dict[str, dict],
    settings: dict,
) -> None:
    """Run the tool loop on the one session created for this sample.

    Input:
        sample (dict): one JSONL row. Unused except to keep the runner contract.
        env (Session): the only ToolEnv session for this sample. Calls are recorded here.
        transcript (str): text of the latest request. Already included in ``context_text``.
        context_text (str): prompt block from ``context.render``.
        tools_by_name (dict[str, dict]): unused. Schemas come from ``env.tools``.
        settings (dict): ``planner_model``, ``temperature``, and ``max_steps``.

    Output:
        None. Submission fields stay on ``env`` via ``get_tool_calls`` and ``get_state_hash``.
    """
    del sample, transcript, tools_by_name
    definitions = tool_definitions(env)
    llm = _chat_model(settings).bind_tools(definitions)
    max_steps = int(settings["max_steps"])

    def agent_node(state: AgentState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def execute_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        messages = []
        for call in _tool_calls(last):
            if len(env.get_tool_calls()) >= max_steps:
                break
            name = call["name"]
            result = env.call(name, call["args"])
            messages.append(
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    tool_call_id=call["id"],
                    name=name,
                )
            )
            if name == "sys.ask_missing_info":
                break
        return {"messages": messages, "steps": len(env.get_tool_calls())}

    def after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        if not _tool_calls(last):
            return END
        if len(env.get_tool_calls()) >= max_steps:
            return END
        return "execute"

    def after_tools(state: AgentState) -> str:
        del state
        calls = env.get_tool_calls()
        if calls and calls[-1]["name"] == "sys.ask_missing_info":
            return END
        if len(calls) >= max_steps:
            return END
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("execute", execute_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_agent, {"execute": "execute", END: END})
    graph.add_conditional_edges("execute", after_tools, {"agent": "agent", END: END})
    app = graph.compile()
    app.invoke(
        {
            "messages": [SystemMessage(content=_RULES + "\n" + context_text)],
            "steps": 0,
        },
        config={"recursion_limit": max_steps * 4 + 4},
    )


def _tool_calls(message) -> list[dict]:
    """Normalize tool calls from an AIMessage.

    Input:
        message (AIMessage): the latest model reply.

    Output:
        list[dict]: each item has ``name`` (str), ``args`` (dict), and ``id`` (str).
    """
    direct = getattr(message, "tool_calls", None) or []
    parsed = []
    for call in direct:
        args = call.get("args") or {}
        if isinstance(args, str):
            args = _load_args(args)
        parsed.append(
            {
                "name": call.get("name") or "",
                "args": args if isinstance(args, dict) else {},
                "id": call.get("id") or "",
            }
        )
    if parsed:
        return parsed
    raw = (getattr(message, "additional_kwargs", None) or {}).get("tool_calls") or []
    for item in raw:
        function = item.get("function") or {}
        parsed.append(
            {
                "name": function.get("name") or item.get("name") or "",
                "args": _load_args(function.get("arguments")),
                "id": item.get("id") or "",
            }
        )
    return parsed


def _load_args(value) -> dict:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _chat_model(settings: dict):
    """Connect to the local vLLM server and reuse the client across samples.

    Input:
        settings (dict): ``planner_model`` (served name), ``planner_base_url``, and ``temperature``.

    Output:
        ChatOpenAI: unbound chat model. Callers bind tools per sample.
    """
    global _CHAT, _CHAT_KEY
    model_name = str(settings.get("planner_model") or "qwen-4b")
    base_url = str(settings.get("planner_base_url") or "http://127.0.0.1:8001/v1")
    temperature = float(settings.get("temperature") or 0)
    key = (model_name, base_url, temperature)
    if _CHAT is not None and _CHAT_KEY == key:
        return _CHAT
    _CHAT = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key="not-needed",
        temperature=temperature,
    )
    _CHAT_KEY = key
    return _CHAT
