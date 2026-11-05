"""One LangGraph loop per ToolEnv session.

The agent node calls the local model. Only the first tool call of a turn is
sent to ``env.call``, and only after ``reject_reason`` accepts it. Other calls
in that turn and refused calls come back as ``ToolMessage`` values with the
model's ``tool_call_id``, but they are not recorded on the session.
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
from validate import collect_ids, reject_reason

_RULES = """
Bạn là LifeMate. Hãy thực hiện yêu cầu mới nhất bằng các tool đã được gắn.
Mỗi lượt chỉ gọi một tool. Đọc result rồi mới quyết định bước sau.
Không bịa id, giờ, số tiền, người nhận hay thiết bị. Lấy các giá trị đó từ context, history, hoặc result. Muốn sửa một mục đã có thì gọi tool đọc của app đó trước, trừ khi id đã có trong history.
Chọn tool theo việc người dùng nói và app đang được cài. Gửi tin, gọi điện, chuyển tiền, đặt chỗ, bật thiết bị là tool ghi của đúng app đó. reminders.create chỉ khi người dùng muốn tạo nhắc việc.
Nếu câu có điều kiện, gọi tool đọc để kiểm tra trước. Điều kiện sai thì dừng, không gọi tool ghi.
Nếu thiếu tham số bắt buộc và không suy ra được từ context, history hay result, gọi sys.ask_missing_info với tool_name và danh sách params còn thiếu, rồi dừng.
Nếu không app nào làm được việc, đừng gọi tool.
Trước khi dừng, đối chiếu yêu cầu mới nhất với các call đã chạy. Chưa làm hết việc thì gọi tiếp.
"""

_NOT_RUN = "Chưa chạy. Hãy đọc kết quả tool vừa rồi, rồi gọi lại từng tool một."

_CHAT = None
_CHAT_KEY: tuple | None = None


class AgentState(TypedDict):
    """Conversation and agent turns for a single sample."""

    messages: Annotated[list, add_messages]
    turns: int


def run(
    sample: dict,
    env,
    context_text: str,
    tools_by_name: dict[str, dict],
    settings: dict,
) -> None:
    """Run the tool loop on the one session created for this sample.

    Input:
        sample (dict): one JSONL row. ``history`` supplies ids already returned
        by earlier turns. The latest request is already inside ``context_text``.
        env (Session): the only ToolEnv session for this sample. Accepted calls
        are recorded here. Refused calls are not.
        context_text (str): prompt block from ``context.render``, including the latest request.
        tools_by_name (dict[str, dict]): catalog index used by ``reject_reason``.
        settings (dict): ``planner_model``, ``temperature``, and ``max_steps``.

    Output:
        None. Submission fields stay on ``env`` via ``get_tool_calls`` and ``get_state_hash``.
    """
    definitions = tool_definitions(env)
    llm = _chat_model(settings).bind_tools(definitions)
    max_steps = int(settings["max_steps"])
    max_turns = max_steps + 4
    installed = set(env.allowed)
    seen_ids = _history_ids(sample)

    def agent_node(state: AgentState) -> dict:
        response = llm.invoke(state["messages"])
        return {"messages": [response], "turns": state["turns"] + 1}

    def execute_node(state: AgentState) -> dict:
        calls = _tool_calls(state["messages"][-1])
        messages = []
        if calls and len(env.get_tool_calls()) < max_steps:
            messages.append(_execute_one(calls[0]))
        for extra in calls[1:]:
            messages.append(
                ToolMessage(content=_NOT_RUN, tool_call_id=extra["id"], name=extra["name"])
            )
        return {"messages": messages}

    def _execute_one(call: dict) -> ToolMessage:
        name = call["name"]
        args = call["args"]
        reason = reject_reason(name, args, installed, tools_by_name, seen_ids)
        if reason:
            payload = {"ok": False, "error": reason}
        else:
            payload = env.call(name, args)
            seen_ids.update(collect_ids(payload))
        return ToolMessage(
            content=json.dumps(payload, ensure_ascii=False),
            tool_call_id=call["id"],
            name=name,
        )

    def after_agent(state: AgentState) -> str:
        if not _tool_calls(state["messages"][-1]):
            return END
        if len(env.get_tool_calls()) >= max_steps or state["turns"] >= max_turns:
            return END
        return "execute"

    def after_tools(state: AgentState) -> str:
        calls = env.get_tool_calls()
        if calls and calls[-1]["name"] == "sys.ask_missing_info":
            return END
        if len(calls) >= max_steps or state["turns"] >= max_turns:
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
            "turns": 0,
        },
        config={"recursion_limit": max_turns * 2 + 4},
    )


def _history_ids(sample: dict) -> set[str]:
    """Collect ids already returned in earlier turns of this sample.

    Input:
        sample (dict): one JSONL row. Reads ``history[].tool_results`` only.

    Output:
        set[str]: identifier strings from those results, such as ``c_10``.
        The current user text and ``state.json`` are not scanned.
    """
    found: set[str] = set()
    for turn in sample.get("history") or []:
        found |= collect_ids(turn.get("tool_results"))
    return found


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
