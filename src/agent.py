"""One LangGraph loop per ToolEnv session.

The plan node lists the objects in the request and the action on each one.
A later object can wait on a field from an earlier read. The agent node then
calls the local model. Only the first tool call of a turn is sent to
``env.call``, and only after ``reject_reason`` accepts it. Other calls in that
turn and refused calls come back as ``ToolMessage`` values with the model's
``tool_call_id``, but they are not recorded on the session.
A new graph is built for every sample, so sessions do not share messages or tools.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from env import env_value
from tool_defs import tool_definitions
from validate import collect_ids, reject_reason

_RULES = """
Bạn là LifeMate. Làm đúng từng đối tượng trong plan, bằng tool đã gắn.
Mỗi lượt một tool của đối tượng đang làm. Đối tượng có khi không khớp thì bỏ. Hết đối tượng thì dừng.
Không gọi lại tool với cùng tham số. Chỉ gọi lại khi là một đối tượng khác.
Không bịa id, giờ, số tiền, người nhận, thiết bị, địa chỉ. Lấy từ context, history hoặc result.
Giờ ghi dạng YYYY-MM-DDTHH:MM:SS với timezone của context. Ngày mai lấy từ tomorrow, hôm nay lấy từ today. Tối thì giờ cộng 12.
Dời một sự kiện thì cộng cùng một khoảng vào cả start và end của đúng sự kiện trong result.
Nội dung tin nhắn là việc vừa làm, không chép nguyên câu nghe.
Chỉ gửi tham số bắt buộc. weather.get_forecast chỉ có date. alarm.create chỉ có time. Đặt ngay thì không gửi schedule_time.
Đèn: brightness là một nửa brightness hiện tại của đúng phòng. Bốn chỗ hoặc bấm chỗ thì vehicle=car4.
Nạp điện thoại: amount là số ngàn trong câu nhân 1000, phone lấy từ contact vừa tìm hoặc số đọc trong câu.
Chuyển số tiền điện: amount là amount của hóa đơn điện, không lấy số tiền nạp.
Thiếu tham số bắt buộc và đối tượng không còn tool đọc thì gọi sys.ask_missing_info với tool_name và params còn thiếu, rồi dừng.
"""

_PLAN_HEAD = "Tool đang có:"

_PLAN_BODY = """
Lập plan trước khi gọi tool. Mỗi đối tượng đúng một dòng. Không viết dòng nào khác.

Đối tượng: <tên vật hoặc người> | việc: <hành động> | khi: luôn | tool: <tool đọc>, <tool ghi>

Cách tách câu:
- Mỗi việc là một đối tượng. Câu rút lại ("mà thôi", "à thôi", "chỉ cần", "giữ nguyên") thì không lập đối tượng cho ý đã bỏ.
- Việc cần đọc để biết điều kiện đứng trước, khi là luôn. Việc phía sau ghi khi bằng trường của kết quả đó. Nhánh "thôi" không thành đối tượng.
- Hai nhánh đều có việc thì hai đối tượng, hai khi ngược nhau. Chỉ đối tượng có khi khớp kết quả mới được gọi.
- Sửa hoặc hủy thứ đã có: tool đọc (list, search, get) rồi update, cancel hoặc delete. Không tạo mới.
- Dời, muộn, sớm một lịch đã có: calendar. Nhắc: reminders. Báo thức: alarm.
- Người nhận là tên người, kể cả khi một từ đứng giữa động từ và tên. contacts.search bằng tên đó, rồi tool gửi hoặc gọi.
- khi dùng tên trường của tool đọc: rain, temperature_c, buy, sell, price, balance, amount, paid, locked, acknowledged, available, won, status, battery. Ví dụ rain=true, balance>=amount, buy<123000000, temperature_c<20. 1 triệu = 1000000.
- Không có tool nào làm được việc thì một dòng: Đối tượng: ngoài phạm vi | việc: dừng | khi: luôn | tool:
"""

_NUDGE = (
    "Nếu không còn đối tượng nào cần gọi, không gọi tool. "
    "Nếu còn, gọi đúng một tool của đối tượng đang làm. Không gọi tool khác."
)

_RESULT_CHARS = 800
_SUMMARY_CHARS = 2000

_NOT_RUN = "Chưa chạy. Hãy đọc kết quả tool vừa rồi, rồi gọi lại từng tool một."

_CHAT = None
_CHAT_KEY: tuple | None = None


class AgentState(TypedDict):
    """Conversation and agent turns for a single sample."""

    messages: Annotated[list, add_messages]
    turns: int
    nudge: int


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
    chat = _chat_model(settings)
    llm = chat.bind_tools(definitions)
    max_steps = int(settings["max_steps"])
    max_turns = max_steps + 4
    installed = set(env.allowed)
    seen_ids = _history_ids(sample)
    done_calls: list[tuple[str, dict]] = []
    results: list = []
    request_text = _latest_request(context_text)
    facts: dict = {}
    guard_fails = 0

    def plan_node(state: AgentState) -> dict:
        request = _plan_request(_tool_menu(env, tools_by_name))
        response = chat.invoke(state["messages"] + [HumanMessage(content=request)])
        response.content = _repair_plan(
            _plan_lines(str(getattr(response, "content", "") or "")),
            request_text,
            installed,
            tools_by_name,
        )
        print(f"PLAN {sample.get('id', '')}:\n{response.content}\n", flush=True)
        return {"messages": [HumanMessage(content=request), response]}

    def agent_node(state: AgentState) -> dict:
        messages = _turn_messages(state)
        plan = _plan_text(state["messages"])
        hint = _guidance(plan, done_calls, facts, request_text)
        if hint:
            human = messages[-1]
            messages[-1] = HumanMessage(content=hint + "\n\n" + str(human.content))
        response = llm.invoke(messages)
        return {"messages": [response], "turns": state["turns"] + 1}

    def nudge_node(state: AgentState) -> dict:
        return {"messages": [HumanMessage(content=_NUDGE)], "nudge": 1}

    def execute_node(state: AgentState) -> dict:
        calls = _tool_calls(state["messages"][-1])
        messages = []
        if calls and len(env.get_tool_calls()) < max_steps:
            messages.append(_execute_one(calls[0], _plan_text(state["messages"])))
        for extra in calls[1:]:
            messages.append(
                ToolMessage(content=_NOT_RUN, tool_call_id=extra["id"], name=extra["name"])
            )
        return {"messages": messages, "nudge": 0}

    def _execute_one(call: dict, plan: str) -> ToolMessage:
        name = call["name"]
        args = _prepare_args(
            name,
            dict(call["args"]),
            request_text,
            context_text,
            results,
            facts,
            plan,
            done_calls,
        )
        nonlocal guard_fails
        clock = _time_block(args)
        schedule = _schedule_block(name, request_text)
        reuse = _reuse_reason(name, args, done_calls)
        guard = _plan_guard(plan, name, facts, done_calls, tools_by_name)
        if schedule:
            reason = schedule
            guard_fails += 1
        elif reuse:
            reason = reuse
            guard_fails += 1
        elif guard:
            reason = guard
            guard_fails += 1
        else:
            reason = clock or reject_reason(name, args, installed, tools_by_name, seen_ids)
            if not reason:
                guard_fails = 0
        if reason:
            payload = {"ok": False, "error": reason}
        else:
            payload = env.call(name, args)
            done_calls.append((name, args))
            results.append(payload)
            seen_ids.update(collect_ids(payload))
        _absorb_facts(payload, facts, _wanted_keys(plan), _bill_focus(request_text))
        return ToolMessage(
            content=_clip(json.dumps(payload, ensure_ascii=False), _RESULT_CHARS),
            tool_call_id=call["id"],
            name=name,
        )

    def after_agent(state: AgentState) -> str:
        if _tool_calls(state["messages"][-1]):
            if len(env.get_tool_calls()) >= max_steps or state["turns"] >= max_turns:
                return END
            return "execute"
        if state["nudge"] == 0:
            return "nudge"
        return END

    def after_tools(state: AgentState) -> str:
        calls = env.get_tool_calls()
        plan = _plan_text(state["messages"])
        if calls and calls[-1]["name"] == "sys.ask_missing_info":
            return END
        if guard_fails >= 2:
            return END
        if _plan_satisfied(plan, done_calls, facts):
            return END
        if len(calls) >= max_steps or state["turns"] >= max_turns:
            return END
        return "agent"

    graph = StateGraph(AgentState)
    graph.add_node("plan", plan_node)
    graph.add_node("agent", agent_node)
    graph.add_node("nudge", nudge_node)
    graph.add_node("execute", execute_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "agent")
    graph.add_edge("nudge", "agent")
    graph.add_conditional_edges(
        "agent", after_agent, {"execute": "execute", "nudge": "nudge", END: END}
    )
    graph.add_conditional_edges("execute", after_tools, {"agent": "agent", END: END})
    app = graph.compile()
    app.invoke(
        {
            "messages": [SystemMessage(content=_RULES + "\n" + context_text)],
            "turns": 0,
            "nudge": 0,
        },
        config={"recursion_limit": max_turns * 2 + 8},
    )


def _turn_messages(state: AgentState) -> list:
    """Build one planner request that stays small as the plan advances.

    Input:
        state (AgentState): full graph history. The system text, the plan, and
        tool results are read from it.

    Output:
        list: system message, then one human message with the plan and a short
        record of calls already made. Older tool transcripts are not resent.
    """
    system = state["messages"][0]
    plan = _clip(_plan_text(state["messages"]), _RESULT_CHARS)
    done = _clip(_done_text(state["messages"]), _SUMMARY_CHARS)
    extra = _NUDGE if state["nudge"] else (
        "Gọi đúng một tool của đối tượng đang làm. Không gọi tool của đối tượng khác. "
        "Đối tượng có khi không khớp thì bỏ. Hết đối tượng thì không gọi tool."
    )
    text = f"Plan:\n{plan}\n\nĐã làm:\n{done}\n\n{extra}"
    return [system, HumanMessage(content=text)]


def _plan_request(menu: str) -> str:
    """Build the plan prompt, including the tools this sample may use.

    Input:
        menu (str): one tool per line, ``name [kind] description``, from
        ``_tool_menu``.

    Output:
        str: the human message for the plan node. It starts with ``_PLAN_HEAD``
        so a later turn can find this message.
    """
    return f"{_PLAN_HEAD}\n{menu}\n{_PLAN_BODY}"


def _tool_menu(env, tools_by_name: dict[str, dict]) -> str:
    """List the tools installed on this sample, without parameter schemas.

    Input:
        env (Session): the sample session. ``allowed`` is the installed names.
        tools_by_name (dict[str, dict]): catalog index. Each spec may have
        ``kind`` and ``description``.

    Output:
        str: one line per allowed tool that exists in the catalog, sorted by name.
    """
    lines = []
    for name in sorted(env.allowed):
        spec = tools_by_name.get(name) or {}
        kind = spec.get("kind") or ""
        description = spec.get("description") or ""
        lines.append(f"{name} [{kind}] {description}")
    return "\n".join(lines)


def _plan_lines(text: str) -> str:
    """Keep the object lines and drop the reasoning around them.

    Input:
        text (str): the plan node's reply.

    Output:
        str: lines that start with ``Đối tượng:``, at most eight, each at most
        400 characters. The original text, clipped, when none of those lines appear.
    """
    kept = []
    for line in text.splitlines():
        line = line.strip().lstrip("-").strip()
        if len(line) > 400:
            line = line[:400]
        if line.startswith("Đối tượng:"):
            kept.append(line)
    if kept:
        return "\n".join(kept[:8])
    return text.strip()[:400]


_TOOL_NAME = re.compile(r"\b[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*\b")
_ATOM = re.compile(
    r"^(?P<left>[a-z_][a-z0-9_]*)\s*(?P<op><=|>=|!=|=|<|>)\s*(?P<right>.+)$",
    re.IGNORECASE,
)
_MONEY = re.compile(r"(\d+(?:[.,]\d+)?)\s*(tỷ|triệu)", re.IGNORECASE)
_STOP_ACTIONS = {"dừng", "dung", "thôi", "thoi"}

# Tools a gold train label calls twice. The second call always targets another
# item, except two loyalty.redeem rows. Every other tool is one call per sample.
_REPEATABLE = frozenset({
    "alarm.create",
    "appliances.start_cycle",
    "banking.transfer",
    "bike_rental.reserve",
    "bill_payment.pay",
    "calendar.create_event",
    "chat_ott.send_location",
    "chat_ott.send_message",
    "contacts.search",
    "door_lock.grant_guest_code",
    "door_lock.set_lock",
    "email.archive",
    "ewallet.transfer",
    "exchange_rate.convert",
    "fitness.get_activity",
    "flights.cancel",
    "food_delivery.cancel_order",
    "food_delivery.place_order",
    "home_automation.delete_rule",
    "home_security.acknowledge_alert",
    "household_chores.assign",
    "learning.log_progress",
    "learning.schedule_lesson",
    "library.reserve",
    "library.search",
    "lottery.check_ticket",
    "lottery.get_results",
    "loyalty.redeem",
    "maps.get_route",
    "maps.save_place",
    "maps.search_place",
    "medication.log_taken",
    "medication.remove",
    "music.add_to_playlist",
    "news.save",
    "news.search",
    "podcast_radio.queue",
    "reminders.complete",
    "reminders.create",
    "shopping_list.check",
    "smart_home.fan_set",
    "smart_home.light_set",
    "sms.send",
    "subscriptions.cancel",
    "tasks.add",
    "tasks.complete",
})


def _tool_names(text: str) -> list[str]:
    """Return tool names mentioned in one plan fragment, in order.

    Input:
        text (str): a plan line or branch, such as ``đúng thì calendar.cancel_event``.

    Output:
        list[str]: names matching ``app.tool``. Empty when the branch says to stop.
    """
    return _TOOL_NAME.findall(text or "")


def _parse_objects(plan: str) -> list[dict]:
    """Read one object from each ``Đối tượng:`` line.

    Input:
        plan (str): plan lines. Each object is
        ``Đối tượng: tên | việc: hành động | khi: điều kiện | tool: a, b``.

    Output:
        list[dict]: each item has ``name``, ``action``, ``when``, and ``tools``.
        ``tools`` is the tool names on that line, in order. Empty when the
        plan has no object line.
    """
    objects = []
    for raw in (plan or "").splitlines():
        line = raw.strip()
        if not line.startswith("Đối tượng:"):
            continue
        pieces = [part.strip() for part in line.split(":", 1)[1].split("|")]
        obj = {"name": pieces[0].strip() or "việc", "action": "", "when": "luôn", "tools": []}
        for piece in pieces[1:]:
            lower = piece.lower()
            if lower.startswith("việc:"):
                obj["action"] = piece.split(":", 1)[1].strip()
            elif lower.startswith("khi:"):
                obj["when"] = piece.split(":", 1)[1].strip() or "luôn"
            elif lower.startswith("tool:"):
                obj["tools"] = _tool_names(piece)
        objects.append(obj)
    return objects


def _obj(name: str, action: str, when: str, tools: list[str]) -> dict:
    """Build one plan object.

    Input:
        name (str): the thing the sentence talks about.
        action (str): what to do with it.
        when (str): ``luôn`` or a comparison such as ``rain=true``.
        tools (list[str]): tool names in call order.

    Output:
        dict: one object for ``_format_objects``.
    """
    return {"name": name, "action": action, "when": when, "tools": tools}


def _active_text(request: str) -> str:
    """Keep the clause that survives a retraction.

    Input:
        request (str): the latest user text.

    Output:
        str: the text after the last ``chỉ cần``, ``mà thôi``, or ``à thôi``.
        The whole request when none of those markers appear.
    """
    low = request.lower()
    cut_at = -1
    length = 0
    for marker in ("chỉ cần", "mà thôi", "à thôi"):
        index = low.rfind(marker)
        if index > cut_at:
            cut_at = index
            length = len(marker)
    if cut_at >= 0:
        return request[cut_at + length :]
    return request


def _check_only(text: str) -> bool:
    """Return whether the remaining clause only asks for a lookup.

    Input:
        text (str): the active clause from ``_active_text``.

    Output:
        bool: True when the clause asks about rain or a price and names no
        later write.
    """
    low = text.lower()
    writes = (
        "dời", "muộn", "sớm", "hủy", "huỷ", "nhắn", "gọi", "nạp", "chuyển",
        "đặt", "bật", "tắt", "nhắc", "báo thức", "trả", "giảm", "tăng",
    )
    if any(word in low for word in writes):
        return False
    return any(word in low for word in ("mưa", "thời tiết", "giá"))


_VN_DIGIT = {
    "không": 0, "một": 1, "mốt": 1, "hai": 2, "ba": 3, "bốn": 4, "tư": 4,
    "năm": 5, "lăm": 5, "sáu": 6, "bảy": 7, "bẩy": 7, "tám": 8, "chín": 9,
}


def _vn_under_1000(phrase: str) -> int:
    """Read a Vietnamese number below one thousand.

    Input:
        phrase (str): words such as ``một trăm hai mươi ba``.

    Output:
        int: that number. Zero when no digit word is present.
    """
    tokens = [token for token in phrase.lower().split() if token not in ("lẻ", "linh", "rưỡi")]
    number = 0
    if "trăm" in tokens:
        index = tokens.index("trăm")
        hundreds = _VN_DIGIT.get(tokens[index - 1], 1) if index else 1
        number += hundreds * 100
        tokens = tokens[index + 1 :]
    if "mươi" in tokens:
        index = tokens.index("mươi")
        tens = _VN_DIGIT.get(tokens[index - 1], 1) if index else 1
        number += tens * 10
        tokens = tokens[index + 1 :]
    elif tokens and tokens[0] == "mười":
        number += 10
        tokens = tokens[1:]
    if tokens and tokens[0] in _VN_DIGIT:
        number += _VN_DIGIT[tokens[0]]
    return number


def _trieu_amount(text: str) -> int | None:
    """Read the đồng amount written in front of ``triệu``.

    Input:
        text (str): a clause that may say ``123 triệu`` or ``một trăm hai mươi ba triệu``.

    Output:
        int: the amount in đồng.
        None: when ``triệu`` is not tied to a number.
    """
    low = text.lower()
    digits = re.search(r"(\d+(?:[.,]\d+)?)\s*triệu", low)
    if digits:
        return int(float(digits.group(1).replace(",", ".")) * 1_000_000)
    words = re.search(r"((?:[\w]+\s+){1,6})triệu", low)
    if not words:
        return None
    number = _vn_under_1000(words.group(1))
    if number <= 0:
        return None
    return number * 1_000_000


def _ngan_amount(text: str) -> int | None:
    """Read an amount written in thousands.

    Input:
        text (str): a clause that may say ``năm mươi ngàn`` or ``50 nghìn``.

    Output:
        int: the amount in đồng.
        None: when no thousand-amount is present.
    """
    low = text.lower()
    digits = re.search(r"(\d+)\s*(ngàn|nghìn)", low)
    if digits:
        return int(digits.group(1)) * 1000
    words = re.search(r"((?:[\w]+\s+){1,4})(ngàn|nghìn)", low)
    if not words:
        return None
    number = _vn_under_1000(words.group(1))
    if number <= 0:
        return None
    return number * 1000


def _normalize_when(when: str) -> str:
    """Turn a prose condition into a field comparison.

    Input:
        when (str): the ``khi`` text from the model, or ``luôn``.

    Output:
        str: ``luôn``, or a comparison such as ``rain=true`` or ``balance<amount``.
        A clock phrase with no condition stays ``luôn``.
    """
    text = (when or "").strip().lower()
    if text in ("", "luôn", "luon"):
        return "luôn"
    parsed = _money_to_int(text).replace(" ", "")
    if _ATOM.match(parsed):
        return parsed
    if "không mưa" in text or text in ("giữ nguyên", "thôi"):
        return "rain=false"
    if "mưa" in text:
        return "rain=true"
    if "không đủ" in text:
        return "balance<amount"
    if "đủ" in text:
        return "balance>=amount"
    amount = _trieu_amount(text)
    if amount and "dưới" in text:
        return f"buy<{amount}"
    if re.search(r"mưa|đủ|dưới|trên|>=|<=|=|>|<", text):
        return text
    return "luôn"


def _best_read(app: str, installed: set[str]) -> str | None:
    """Pick the list, search, or status tool for one app.

    Input:
        app (str): the app id, such as ``calendar`` or ``contacts``.
        installed (set[str]): tool names this sample may call.

    Output:
        str: the read tool name.
        None: when that app has no such read installed.
    """
    names = [name for name in installed if name.startswith(app + ".")]
    for head in ("list_", "search", "get_status"):
        for name in names:
            tail = name.split(".", 1)[1]
            if tail == head or tail.startswith(head):
                return name
    return None


def _reads_for(write: str, tools_by_name: dict[str, dict], installed: set[str]) -> list[str]:
    """Find the reads that produce ids for one write.

    Input:
        write (str): a tool name such as ``calendar.update_event``.
        tools_by_name (dict[str, dict]): catalog index. ``x-ref`` names the
        app that owns the id.
        installed (set[str]): tool names this sample may call.

    Output:
        list[str]: read tool names, without duplicates.
    """
    spec = tools_by_name.get(write) or {}
    props = (spec.get("parameters") or {}).get("properties") or {}
    found = []
    for prop in props.values():
        if not isinstance(prop, dict) or not isinstance(prop.get("x-ref"), str):
            continue
        read = _best_read(prop["x-ref"].split(".")[0], installed)
        if read and read != write and read not in found:
            found.append(read)
    return found


def _order_tools(tools: list[str], tools_by_name: dict[str, dict]) -> list[str]:
    """Put reads before writes and drop duplicates.

    Input:
        tools (list[str]): tool names on one object.
        tools_by_name (dict[str, dict]): catalog index, used for ``kind``.

    Output:
        list[str]: the same names, reads first, each name once.
    """
    seen = []
    for name in tools:
        if name not in seen:
            seen.append(name)
    reads = [name for name in seen if (tools_by_name.get(name) or {}).get("kind") != "write"]
    writes = [name for name in seen if name not in reads]
    return reads + writes


def _has_tool(objects: list[dict], name: str) -> bool:
    """Return whether any object already lists a tool.

    Input:
        objects (list[dict]): the plan objects so far.
        name (str): a tool name.

    Output:
        bool: True when ``name`` is on some object.
    """
    return any(name in obj["tools"] for obj in objects)


def _format_objects(objects: list[dict]) -> str:
    """Render objects as plan lines.

    Input:
        objects (list[dict]): objects with ``name``, ``action``, ``when``, and ``tools``.

    Output:
        str: one ``Đối tượng:`` line per object that still has a tool.
    """
    lines = []
    for obj in objects:
        if not obj["tools"]:
            continue
        tools = ", ".join(obj["tools"])
        lines.append(
            f"Đối tượng: {obj['name']} | việc: {obj['action']} | khi: {obj['when']} | tool: {tools}"
        )
    return "\n".join(lines)


def _person_after(text: str, verb: str) -> str | None:
    """Read the person named after a verb.

    Input:
        text (str): the active clause, original casing kept where the ASR has it.
        verb (str): ``nhắn`` or ``chuyển ví cho`` or ``điện thoại cho``.

    Output:
        str: a short query such as ``Dũng`` or ``Bác Ba``.
        None: when no person follows the verb.
    """
    match = re.search(verb + r"\s+(?:tiếp\s+|cho\s+)?([A-ZĐÀ-Ỹ][\w]*(?:\s+[A-ZĐÀ-Ỹ][\w]*)?)", text)
    if match:
        return match.group(1)
    match = re.search(verb + r"\s+(?:cho\s+)?((?:bác|cô|anh|chị|chú)\s+\w+)", text, re.IGNORECASE)
    if match:
        return " ".join(part.capitalize() for part in match.group(1).split())
    match = re.search(verb + r"\s+(?:cho\s+)?(bố|mẹ|ba|má|sếp)\b", text, re.IGNORECASE)
    if match:
        word = match.group(1)
        return word[:1].upper() + word[1:]
    match = re.search(verb + r"\s+con\s+(\w+)", text, re.IGNORECASE)
    if match:
        word = match.group(1)
        return word[:1].upper() + word[1:]
    return None


def _governing_when(text: str) -> str:
    """Choose the condition that guards the writes in this clause.

    Input:
        text (str): the active clause.

    Output:
        str: ``rain=true`` when rain decides the write, ``balance>=amount`` or
        ``balance<amount`` when the wallet does, otherwise ``luôn``.
    """
    low = text.lower()
    if "ví đủ" in low or "ví không đủ" in low:
        return "luôn"
    if "mưa" in low and any(word in low for word in ("không mưa", "thì thôi", "giữ nguyên")):
        return "rain=true"
    if "mưa" in low and any(word in low for word in ("muộn", "sớm", "dời", "hủy", "huỷ", "nhắn")):
        return "rain=true"
    return "luôn"


def _repair_plan(
    plan: str,
    request: str,
    installed: set[str],
    tools_by_name: dict[str, dict],
) -> str:
    """Rebuild the object plan from the sentence and the tools installed here.

    Input:
        plan (str): the lines the model wrote. Their tool names are kept when
        the sentence still asks for them.
        request (str): the latest user text. A retraction keeps only the tail.
        installed (set[str]): tool names this sample may call.
        tools_by_name (dict[str, dict]): catalog index, used to put a read
        before a write that needs an id.

    Output:
        str: one object line per remaining action. The model's text when no
        object can be built.
    """
    scope = _active_text(request)
    low = scope.lower()
    objects = []
    if not _check_only(scope):
        for obj in _parse_objects(plan):
            tools = []
            for name in obj["tools"]:
                if name == "sys.ask_missing_info" or name not in installed:
                    continue
                if name.startswith("alarm.") and "báo thức" not in low and "xoá" not in low and "xóa" not in low:
                    continue
                if name == "phone.call" and "gọi" not in low:
                    continue
                if name == "ewallet.transfer" and "chuyển" not in low:
                    continue
                tools.append(name)
            when = _normalize_when(obj["when"])
            if when == "rain=false" and any(word in low for word in ("thì thôi", "giữ nguyên", "không mưa")):
                continue
            if tools:
                objects.append(_obj(obj["name"], obj["action"], when, tools))

    def add(obj: dict) -> None:
        if obj["tools"] and all(name in installed for name in obj["tools"]):
            if len(obj["tools"]) == 1 and _has_tool(objects, obj["tools"][0]) and obj["when"] == "luôn":
                return
            objects.append(obj)

    if _check_only(scope):
        objects = []
        if "mưa" in low and "weather.get_forecast" in installed:
            add(_obj("thời tiết", "kiểm tra mưa", "luôn", ["weather.get_forecast"]))
        elif _trieu_amount(scope) and "gold_price.get_price" in installed:
            add(_obj("giá", "kiểm tra", "luôn", ["gold_price.get_price"]))
    elif re.search(r"v[aà]ng", low) and "triệu" in low and "gold_price.get_price" in installed:
        objects = [_obj("giá vàng", "kiểm tra giá mua, type=sjc nếu là SJC", "luôn", ["gold_price.get_price"])]
        amount = _trieu_amount(scope)
        if amount and "nhắc" in low and "reminders.create" in installed:
            add(_obj("nhắc mua vàng", "nhắc giờ trong câu", f"buy<{amount}", ["reminders.create"]))
    elif "ví đủ" in low and "bill_payment.pay" in installed:
        objects = []
        if "bill_payment.list_bills" in installed:
            add(_obj("hóa đơn", "xem hóa đơn đúng loại trong câu", "luôn", ["bill_payment.list_bills"]))
        if "ewallet.get_balance" in installed:
            add(_obj("ví", "xem số dư", "luôn", ["ewallet.get_balance"]))
        true_part, false_part = low, ""
        if "còn không" in low:
            true_part, false_part = low.split("còn không", 1)
        if "trả" in true_part:
            add(_obj("trả bằng ví", "source=ewallet", "balance>=amount", ["bill_payment.pay"]))
        person = _person_after(scope, "nhắn")
        if person and "nhắn" in true_part and "sms.send" in installed:
            add(_obj(person, f"nhắn, query {person}", "balance>=amount", ["contacts.search", "sms.send"]))
        if "ngân hàng" in false_part:
            add(_obj("trả bằng ngân hàng", "source=bank", "balance<amount", ["bill_payment.pay"]))
        if "nhắc" in false_part and "reminders.create" in installed:
            add(_obj("nhắc nạp ví", "nhắc giờ trong câu", "balance<amount", ["reminders.create"]))
    else:
        if "mưa" in low and "weather.get_forecast" in installed and not _has_tool(objects, "weather.get_forecast"):
            objects.insert(0, _obj("thời tiết", "kiểm tra mưa, chỉ gửi date", "luôn", ["weather.get_forecast"]))
        if any(word in low for word in ("trả", "hóa đơn", "tiền điện")) and "bill_payment.list_bills" in installed:
            if not _has_tool(objects, "bill_payment.list_bills"):
                objects.insert(0, _obj("hóa đơn", "xem hóa đơn đúng loại", "luôn", ["bill_payment.list_bills"]))
        when = _governing_when(scope)
        move = any(word in low for word in ("muộn", "sớm", "dời", "hủy", "huỷ")) and "báo thức" not in low and "nhắc" not in low
        if move and "calendar.list_events" in installed:
            write = "calendar.cancel_event" if any(word in low for word in ("hủy", "huỷ")) and "muộn" not in low and "dời" not in low and "sớm" not in low else "calendar.update_event"
            if write in installed and not _has_tool(objects, write):
                add(_obj("lịch", "dời hoặc hủy đúng việc trong câu", when, ["calendar.list_events", write]))
        if "nhắn" in low and "sms.send" in installed:
            person = _person_after(scope, "nhắn") or "người nhận"
            if not _has_tool(objects, "sms.send"):
                add(_obj(person, f"nhắn giờ mới, query {person}", when, ["contacts.search", "sms.send"]))
        topup = ("nạp" in low or "cho số" in low or ("điện thoại cho" in low and "gọi" not in low and _ngan_amount(scope)))
        transfer = "chuyển" in low and "ewallet.transfer" in installed
        if topup or transfer:
            money_tools = {"contacts.search", "contacts.get", "ewallet.transfer", "ewallet.topup_mobile", "phone.call"}
            kept = []
            for obj in objects:
                names = [name for name in obj["tools"] if name not in money_tools]
                if names:
                    kept.append(_obj(obj["name"], obj["action"], obj["when"], names))
            objects = kept
        if topup and "ewallet.topup_mobile" in installed:
            person = None if "cho số" in low else _person_after(scope, "điện thoại cho")
            tools = ["contacts.search", "ewallet.topup_mobile"] if person else ["ewallet.topup_mobile"]
            add(_obj(person or "số điện thoại", "nạp, amount là số ngàn trong câu", "luôn", tools))
        if transfer:
            person = _person_after(scope, "chuyển ví cho") or _person_after(scope, "chuyển cho") or "người nhận"
            add(_obj(person, f"chuyển ví, query {person}", "luôn", ["contacts.search", "ewallet.transfer"]))
        if "báo thức" in low and "alarm.create" in installed and not _has_tool(objects, "alarm.create"):
            add(_obj("báo thức", "chỉ gửi time", "luôn", ["alarm.create"]))
        if any(word in low for word in ("đèn", "độ sáng")) and "smart_home.light_set" in installed:
            if not _has_tool(objects, "smart_home.light_set"):
                add(_obj("đèn", "chỉnh đúng phòng", "luôn", ["smart_home.get_status", "smart_home.light_set"]))
        if "đặt" in low and "xe" in low and "ride_hailing.book" in installed:
            if not re.search(r"\d", scope) and "calendar.list_events" in installed and not _has_tool(objects, "calendar.list_events"):
                objects.insert(0, _obj("lịch", "lấy điểm đến của việc trong câu", "luôn", ["calendar.list_events"]))
            if not _has_tool(objects, "ride_hailing.book"):
                add(_obj("xe", "đặt ngay thì không gửi schedule_time", "luôn", ["ride_hailing.book"]))

    hoisted = []
    rest = []
    gate_reads = {"weather.get_forecast", "weather.get_current", "gold_price.get_price", "ewallet.get_balance", "bill_payment.list_bills"}
    for obj in objects:
        gates = [name for name in obj["tools"] if name in gate_reads]
        later = [name for name in obj["tools"] if name not in gates]
        if gates and later and obj["when"] != "luôn":
            hoisted.append(_obj(obj["name"], "kiểm tra", "luôn", gates))
            obj = _obj(obj["name"], obj["action"], obj["when"], later)
        if gates and not later and obj["when"] != "luôn":
            obj = _obj(obj["name"], obj["action"], "luôn", gates)
        obj["tools"] = _order_tools(obj["tools"], tools_by_name)
        rest.append(obj)
    objects = hoisted + rest
    seen_reads = set()
    for obj in objects:
        extra = []
        for name in list(obj["tools"]):
            for read in _reads_for(name, tools_by_name, installed):
                if read in obj["tools"] or read in extra:
                    continue
                if read != "contacts.search" and read in seen_reads:
                    continue
                extra.append(read)
        obj["tools"] = _order_tools(extra + obj["tools"], tools_by_name)
        for name in obj["tools"]:
            if (tools_by_name.get(name) or {}).get("kind") != "write":
                seen_reads.add(name)
    always = [obj for obj in objects if obj["when"] == "luôn"]
    later = [obj for obj in objects if obj["when"] != "luôn"]

    def later_key(obj: dict) -> int:
        names = " ".join(obj["tools"])
        if "calendar." in names:
            return 0
        if "bill_payment.pay" in names:
            return 1
        if "reminders." in names:
            return 3
        return 2

    later.sort(key=later_key)
    rendered = _format_objects(always + later)
    return rendered or plan


def _money_to_int(text: str) -> str:
    """Expand ``triệu`` and ``tỷ`` so a condition can be compared as a number.

    Input:
        text (str): one condition atom, such as ``buy<123 triệu``.

    Output:
        str: the same atom with those units replaced by đồng.
    """
    def repl(match: re.Match) -> str:
        number = float(match.group(1).replace(",", "."))
        scale = 1_000_000_000 if match.group(2).lower() == "tỷ" else 1_000_000
        return str(int(number * scale))

    return _MONEY.sub(repl, text)


def _literal(text: str):
    """Parse a condition's right-hand side when it is a bool or a number.

    Input:
        text (str): the text after the operator.

    Output:
        bool | int | float | None: the value. None when the text is a field
        name or some other word.
    """
    raw = text.strip().lower()
    if raw in ("true", "false"):
        return raw == "true"
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?\d+\.\d+", raw):
        return float(raw)
    return None


def _atom_status(part: str, facts: dict) -> str:
    """Decide one comparison against the facts collected so far.

    Input:
        part (str): one atom, such as ``rain=true`` or ``balance>=amount``.
        facts (dict): scalar fields copied from tool results.

    Output:
        str: ``run`` when the atom holds or cannot be parsed, ``skip`` when it
        is false, ``wait`` when a named field is not in ``facts`` yet.
    """
    match = _ATOM.match(_money_to_int(part.strip()))
    if not match:
        return "skip"
    left = match.group("left").lower()
    op = match.group("op")
    right_text = match.group("right").strip()
    if left not in facts:
        return "wait"
    left_value = facts[left]
    literal = _literal(right_text)
    if literal is None and re.fullmatch(r"[a-z_][a-z0-9_]*", right_text, re.IGNORECASE):
        key = right_text.lower()
        if key not in facts:
            return "wait"
        right_value = facts[key]
    elif literal is None:
        right_value = right_text.strip()
    else:
        right_value = literal
    if isinstance(left_value, bool) or isinstance(right_value, bool):
        if op not in ("=", "!="):
            return "run"
        same = bool(left_value) == bool(right_value)
        return "run" if (same if op == "=" else not same) else "skip"
    if isinstance(left_value, (int, float)) and isinstance(right_value, (int, float)):
        ok = {
            "=": left_value == right_value,
            "!=": left_value != right_value,
            "<": left_value < right_value,
            ">": left_value > right_value,
            "<=": left_value <= right_value,
            ">=": left_value >= right_value,
        }[op]
        return "run" if ok else "skip"
    same = str(left_value).lower() == str(right_value).lower()
    if op == "=":
        return "run" if same else "skip"
    if op == "!=":
        return "run" if not same else "skip"
    return "run"


def _when_status(expr: str, facts: dict) -> str:
    """Decide whether an object's condition allows it to run.

    Input:
        expr (str): the ``khi`` text. ``luôn`` always runs. Several atoms join
        with ``và``.
        facts (dict): scalar fields copied from tool results.

    Output:
        str: ``run``, ``skip``, or ``wait``. One false atom skips the object.
        A missing field waits, unless another atom is already false.
    """
    text = (expr or "").strip().lower()
    if text in ("", "luôn", "luon"):
        return "run"
    saw_wait = False
    for part in re.split(r"\s+và\s+", text):
        status = _atom_status(part, facts)
        if status == "skip":
            return "skip"
        if status == "wait":
            saw_wait = True
    return "wait" if saw_wait else "run"


def _is_stop(obj: dict) -> bool:
    """Return whether this object means the request is finished.

    Input:
        obj (dict): one parsed object. A stop object has no tools and an
        action such as ``dừng``.

    Output:
        bool: True when reaching this object should end the sample.
    """
    return not obj["tools"] and obj["action"].lower() in _STOP_ACTIONS


def _cursor(objects: list[dict], done_calls: list[tuple[str, dict]], facts: dict):
    """Find the next tool the plan still wants.

    Input:
        objects (list[dict]): objects in plan order.
        done_calls (list[tuple[str, dict]]): successful calls in this sample.
        facts (dict): fields used by ``khi``.

    Output:
        tuple[str, dict | None, str]: ``run`` plus the object and its next tool,
        ``wait`` when a later object needs a field that is not in yet,
        or ``done`` when every remaining object is finished or skipped.
    """
    index = 0
    for obj in objects:
        status = _when_status(obj["when"], facts)
        if _is_stop(obj):
            if status == "skip":
                continue
            if status == "wait":
                return "wait", obj, ""
            return "done", obj, ""
        if status == "skip":
            continue
        if status == "wait":
            return "wait", obj, ""
        for tool in obj["tools"]:
            if index < len(done_calls) and done_calls[index][0] == tool:
                index += 1
                continue
            return "run", obj, tool
    return "done", None, ""


def _wanted_keys(plan: str) -> set[str]:
    """Return the result fields named in the plan's conditions.

    Input:
        plan (str): the object plan.

    Output:
        set[str]: field names such as ``rain`` and ``amount``. Empty when every
        object says ``luôn``.
    """
    keys = set()
    for obj in _parse_objects(plan):
        text = (obj["when"] or "").lower()
        if text in ("", "luôn", "luon"):
            continue
        for part in re.split(r"\s+và\s+", text):
            match = _ATOM.match(_money_to_int(part.strip()))
            if not match:
                continue
            keys.add(match.group("left").lower())
            right = match.group("right").strip()
            if re.fullmatch(r"[a-z_][a-z0-9_]*", right, re.IGNORECASE) and right.lower() not in (
                "true",
                "false",
            ):
                keys.add(right.lower())
    return keys


def _bill_focus(request: str) -> str | None:
    """Return the bill type named in the request.

    Input:
        request (str): the latest user text.

    Output:
        str: a catalog bill type such as ``electricity``.
        None: when the sentence does not name a bill.
    """
    low = request.lower()
    if "tiền điện" in low or "trả điện" in low or re.search(r"điện(?!\s*thoại)", low):
        return "electricity"
    for word, code in (
        ("điện thoại", "phone"),
        ("truyền hình", "tv"),
        ("học phí", "tuition"),
        ("nước", "water"),
        ("mạng", "internet"),
    ):
        if word in low:
            return code
    return None


def _absorb_facts(payload, facts: dict, wanted: set[str] | None = None, focus: str | None = None) -> None:
    """Copy scalar result fields that a later condition can test.

    Input:
        payload (dict | any): one tool result. Nested lists are walked.
        facts (dict): sample-level facts. A repeated key keeps the latest value.
        wanted (set[str] | None): string fields named in the plan. Numbers and
        bools are kept either way.
        focus (str | None): bill type to keep, such as ``electricity``. Other
        bills do not overwrite ``amount`` or ``bill_id``.

    Output:
        None. Mutates ``facts``.
    """
    wanted = {key.lower() for key in (wanted or set())}

    def walk(node) -> None:
        if isinstance(node, dict):
            bill_type = node.get("type") if isinstance(node.get("amount"), (int, float)) else None
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    walk(value)
                    continue
                if key in ("ok", "error") or value is None:
                    continue
                if key == "amount" and focus and bill_type and bill_type != focus:
                    continue
                if key == "id" and isinstance(value, str) and bill_type and (not focus or bill_type == focus):
                    facts["bill_id"] = value
                    continue
                if key in ("phone", "phone_number") and isinstance(value, str):
                    facts["phone"] = value
                if isinstance(value, (bool, int, float)) or key.lower() in wanted:
                    facts[key] = value
            return
        if isinstance(node, list):
            for child in node:
                walk(child)

    walk(payload)


def _guidance(plan: str, done_calls: list[tuple[str, dict]], facts: dict, request: str = "") -> str:
    """Tell the model which object is next and which fact decided it.

    Input:
        plan (str): the object plan.
        done_calls (list[tuple[str, dict]]): successful calls so far.
        facts (dict): fields copied from those results.
        request (str): the latest user text, used to name the value to fill.

    Output:
        str: the current object and the one tool to call, or a stop sentence.
        Empty when the plan has no object line and no relevant fact.
    """
    lines = []
    wanted = _wanted_keys(plan)
    shown = [f"{key}={value}" for key, value in facts.items() if key.lower() in wanted or key in ("bill_id", "amount", "phone")]
    if shown:
        lines.append("Sự kiện: " + ", ".join(shown) + ".")
    objects = _parse_objects(plan)
    if not objects:
        return "\n".join(lines)
    kind, obj, tool = _cursor(objects, done_calls, facts)
    if kind == "run":
        lines.append(
            f"Đối tượng đang làm: {obj['name']}. Việc: {obj['action']}. "
            f"Gọi đúng tool {tool}. Không gọi tool khác."
        )
        lines.append(_fill_hint(tool, request, facts))
    else:
        lines.append("Không còn đối tượng nào cần gọi. Dừng, không gọi tool.")
    return "\n".join(line for line in lines if line)


def _fill_hint(tool: str, request: str, facts: dict) -> str:
    """Name the argument values the next tool still has to copy.

    Input:
        tool (str): the tool the cursor wants next.
        request (str): the latest user text.
        facts (dict): fields already copied from results.

    Output:
        str: one sentence. Empty when this tool has no extra fill rule.
    """
    low = request.lower()
    if tool == "weather.get_forecast":
        return "Chỉ gửi date. Mai thì dùng tomorrow. Không gửi location."
    if tool == "alarm.create":
        return "Chỉ gửi time, dạng YYYY-MM-DDTHH:MM:SS với timezone. Không gửi label hay repeat."
    if tool == "calendar.update_event":
        return "event_id đúng việc trong câu. Gửi cả start và end, cộng cùng khoảng vào cả hai."
    if tool == "smart_home.light_set" and ("một nửa" in low or "độ sáng" in low):
        return "Chọn đèn đúng phòng. brightness bằng một nửa brightness hiện tại. Chỉ gửi device_id và brightness."
    if tool == "ride_hailing.book":
        return "pickup là công ty nếu câu nói công ty. dropoff lấy location của sự kiện đúng buổi. Đặt ngay thì không gửi schedule_time. Bốn chỗ hoặc bấm chỗ thì vehicle=car4."
    if tool == "ewallet.topup_mobile":
        phone = facts.get("phone") or "số trong câu, viết liền"
        amount = _ngan_amount(request)
        amount_text = str(amount) if amount else "số ngàn nhân 1000"
        return f"phone={phone}. amount={amount_text}."
    if tool == "ewallet.transfer" and facts.get("amount"):
        return f"amount={facts['amount']} lấy từ hóa đơn, không lấy số tiền nạp. contact_id của người vừa tìm."
    if tool == "bill_payment.pay" and facts.get("bill_id"):
        return f"bill_id={facts['bill_id']}. source theo việc của đối tượng này."
    if tool == "sms.send":
        return "content là giờ mới hoặc việc vừa xong, không chép nguyên câu nghe. contact_id của người vừa tìm."
    if tool == "contacts.search":
        return "query là tên người trong việc, không gửi thêm trường khác."
    return ""


def _context_value(context: str, label: str) -> str:
    """Read one labeled line from the rendered context.

    Input:
        context (str): the prompt block from ``context.render``.
        label (str): the line prefix, such as ``tomorrow`` or ``work_address``.

    Output:
        str: the text after the colon. Empty when the line is missing.
    """
    match = re.search(rf"^{label}: (.*)$", context, re.MULTILINE)
    return match.group(1).strip() if match else ""


_HOUR_WORD = {
    "không": 0, "một": 1, "hai": 2, "ba": 3, "bốn": 4, "năm": 5,
    "sáu": 6, "bảy": 7, "tám": 8, "chín": 9, "mười": 10, "mười một": 11, "mười hai": 12,
}
_DIGIT_WORD = {
    "không": "0", "một": "1", "hai": "2", "ba": "3", "bốn": "4", "tư": "4",
    "năm": "5", "lăm": "5", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9", "mốt": "1",
}


def _last_spoken_hm(text: str) -> tuple[int, int] | None:
    """Read the last clock phrase in the sentence.

    Input:
        text (str): the latest user text. A later phrase replaces an earlier one,
        as in ``năm giờ, à không năm rưỡi``.

    Output:
        tuple[int, int]: hour and minute. ``rưỡi`` means minute 30.
        None: when the sentence has no ``giờ`` or ``rưỡi`` after a number.
    """
    low = text.lower()
    pattern = r"(mười một|mười hai|mười|không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|\d{1,2})\s*(giờ|rưỡi)"
    matches = list(re.finditer(pattern, low))
    if not matches:
        return None
    match = matches[-1]
    word = match.group(1)
    hour = int(word) if word.isdigit() else _HOUR_WORD[word]
    minute = 30 if match.group(2) == "rưỡi" or low[match.end():match.end() + 8].lstrip().startswith("rưỡi") else 0
    return hour, minute


def _spoken_time(request: str, context: str) -> str | None:
    """Build an ISO time from the clock phrase next to sáng, tối, nay, or mai.

    Input:
        request (str): the latest user text.
        context (str): the rendered context, for today, tomorrow, and the offset.

    Output:
        str: ``YYYY-MM-DDTHH:MM:SS`` with the context offset.
        None: when no clock phrase is present or the day line is missing.
    """
    found = _last_spoken_hm(request)
    if not found:
        return None
    hour, minute = found
    low = request.lower()
    match = list(re.finditer(r"(mười một|mười hai|mười|không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|\d{1,2})\s*(giờ|rưỡi)", low))[-1]
    window = low[max(0, match.start() - 16):match.end() + 16]
    if any(word in window for word in ("tối", "đêm")) and hour < 12:
        hour += 12
    if "sáng" in window and hour == 12:
        hour = 0
    if "mai" in window and "nay" not in window:
        day_label = "tomorrow"
    elif "nay" in window:
        day_label = "today"
    else:
        day_label = "tomorrow" if "mai" in low and "nay" not in low else "today"
    day = _context_value(context, day_label)
    offset = re.search(r"([+-]\d{2}:\d{2})", context)
    if not day:
        return None
    return f"{day}T{hour:02d}:{minute:02d}:00{offset.group(1) if offset else '+07:00'}"


def _spoken_phone(text: str) -> str | None:
    """Read the digit words that follow ``cho số``.

    Input:
        text (str): the latest user text.

    Output:
        str: the digits written together.
        None: when fewer than eight digit words follow ``cho số``.
    """
    low = text.lower()
    tail = low.split("cho số", 1)[1] if "cho số" in low else ""
    digits = []
    for token in re.findall(r"[0-9a-zà-ỹ]+", tail):
        if token.isdigit():
            digits.append(token)
        elif token in _DIGIT_WORD:
            digits.append(_DIGIT_WORD[token])
        elif digits:
            break
    phone = "".join(digits)
    if len(phone) < 8:
        return None
    return phone


def _matching_contact(results: list, query: str) -> dict | None:
    """Pick the contact whose name contains the planned query.

    Input:
        results (list): tool payloads. A contact has ``id`` starting with ``c_``.
        query (str): the person named on the current object.

    Output:
        dict: the contact with the longest matching name.
        None: when no returned contact contains that query.
    """
    needle = (query or "").strip().lower()
    if not needle or needle in ("số điện thoại", "người nhận", "liên hệ"):
        return None
    best = None
    best_len = 0
    for payload in results:
        for node in _walk_dicts(payload):
            name = node.get("name")
            contact_id = node.get("id")
            if not isinstance(name, str) or not isinstance(contact_id, str) or not str(contact_id).startswith("c_"):
                continue
            if needle in name.lower() and len(name) >= best_len:
                best = node
                best_len = len(name)
    return best


def _expand_clock(value, request: str, context: str):
    """Attach a date and timezone to a bare clock time.

    Input:
        value (any): an argument the model sent, such as ``05:30``.
        request (str): the latest user text. ``mai`` selects tomorrow.
        ``tối`` adds twelve hours when the hour is still in the morning.
        context (str): the rendered context, for today, tomorrow, and the offset.

    Output:
        str: a full ISO time when ``value`` is ``HH:MM``.
        The original value otherwise.
    """
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", str(value).strip())
    if not match:
        return value
    hour = int(match.group(1))
    minute = int(match.group(2))
    low = request.lower()
    if any(word in low for word in ("tối", "đêm")) and hour < 12:
        hour += 12
    if "sáng" in low and hour == 12:
        hour = 0
    day_label = "tomorrow" if "mai" in low and "nay" not in low else "today"
    day = _context_value(context, day_label)
    offset = re.search(r"([+-]\d{2}:\d{2})", context)
    if not day:
        return value
    return f"{day}T{hour:02d}:{minute:02d}:00{offset.group(1) if offset else '+07:00'}"


def _walk_dicts(node):
    """Yield every dict in a nested tool result.

    Input:
        node (any): a tool payload.

    Output:
        iterator: each dict, including nested ones.
    """
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk_dicts(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_dicts(item)


def _shift_minutes(request: str) -> int | None:
    """Read how many minutes a schedule move asks for.

    Input:
        request (str): the latest user text.

    Output:
        int: minutes to add. Negative when the sentence says sớm.
        None: when the sentence does not move a time by a number of hours.
    """
    low = request.lower()
    if not any(word in low for word in ("muộn", "sớm", "dời")):
        return None
    sign = -1 if "sớm" in low and "muộn" not in low else 1
    if "nửa tiếng" in low or "nửa giờ" in low:
        return sign * 30
    match = re.search(r"(một|hai|ba|\d+)\s*tiếng", low)
    if not match:
        return None
    word = match.group(1)
    number = {"một": 1, "hai": 2, "ba": 3}.get(word, int(word) if word.isdigit() else 0)
    if not number:
        return None
    return sign * number * 60


def _add_minutes(value: str, minutes: int) -> str:
    """Move an ISO timestamp by a number of minutes.

    Input:
        value (str): a timestamp such as ``2026-11-29T08:00:00+07:00``.
        minutes (int): the delta. Negative moves earlier.

    Output:
        str: the shifted timestamp with the same offset.
        The original string when it is not an ISO time.
    """
    try:
        stamp = datetime.fromisoformat(value)
    except ValueError:
        return value
    return (stamp + timedelta(minutes=minutes)).isoformat()


def _half_light(results: list, request: str) -> dict | None:
    """Find the lamp whose name is the longest phrase inside the request.

    Input:
        results (list): tool payloads. ``smart_home.get_status`` lists devices.
        request (str): the latest user text, including the room name.

    Output:
        dict: the device that has ``brightness`` and whose name sits inside the request.
        None: when no lamp name is mentioned.
    """
    low = request.lower()
    best = None
    best_len = 0
    for payload in results:
        for node in _walk_dicts(payload):
            name = node.get("name")
            if not isinstance(name, str) or name.lower() not in low:
                continue
            brightness = node.get("brightness")
            state = node.get("state")
            if isinstance(state, dict) and isinstance(state.get("brightness"), (int, float)):
                brightness = state["brightness"]
            if not isinstance(brightness, (int, float)):
                continue
            if len(name) > best_len:
                best = {"id": node.get("id"), "brightness": brightness}
                best_len = len(name)
    return best


def _event_dropoff(results: list, request: str) -> str | None:
    """Take the location of the one event in the named part of the day.

    Input:
        results (list): tool payloads already returned in this sample.
        request (str): the latest user text. Chiều, sáng, or tối picks the hour.

    Output:
        str: that event's location.
        None: when there is not exactly one matching event.
    """
    low = request.lower()
    if "chiều" in low:
        window = range(12, 18)
    elif "tối" in low or "đêm" in low:
        window = range(18, 24)
    elif "sáng" in low:
        window = range(5, 12)
    else:
        window = None
    found = []
    for payload in results:
        for node in _walk_dicts(payload):
            start = node.get("start")
            location = node.get("location")
            if not isinstance(start, str) or not isinstance(location, str) or "T" not in start:
                continue
            hour = int(start[11:13]) if len(start) >= 13 else -1
            if window is None or hour in window:
                found.append(location)
    if len(found) == 1:
        return found[0]
    return None


def _time_block(args: dict) -> str | None:
    """Refuse a clock field that is not a full ISO time.

    Input:
        args (dict): tool arguments after ``_prepare_args``.

    Output:
        str: a refusal telling the model the required shape.
        None: when every present clock field has a date and a timezone.
    """
    for key in ("time", "start", "end", "schedule_time"):
        if key not in args:
            continue
        if not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", str(args[key])):
            return f"{key} phải là YYYY-MM-DDTHH:MM:SS kèm timezone"
    return None


def _prepare_args(
    name: str,
    args: dict,
    request: str,
    context: str,
    results: list,
    facts: dict,
    plan: str,
    done_calls: list[tuple[str, dict]],
) -> dict:
    """Fill or drop arguments the sentence and earlier results already decide.

    Input:
        name (str): the tool about to run.
        args (dict): the model's arguments. This dict is copied by the caller.
        request (str): the latest user text.
        context (str): the rendered context, for addresses and the clock.
        results (list): payloads of successful calls in this sample.
        facts (dict): ``amount``, ``bill_id``, and ``phone`` when a result had them.
        plan (str): the object plan. The active object's việc names the pay source.
        done_calls (list[tuple[str, dict]]): successful calls, used to find that object.

    Output:
        dict: arguments to validate and send.
    """
    low = request.lower()
    for key in ("time", "start", "end", "schedule_time"):
        if key in args:
            args[key] = _expand_clock(args[key], request, context)
    if name in ("calendar.list_events", "smart_home.get_status"):
        args.clear()
    if name == "weather.get_forecast":
        location = str(args.get("location") or "")
        if location and location.lower() not in low:
            args.pop("location", None)
    if name in ("alarm.create", "reminders.create"):
        spoken = _spoken_time(request, context)
        if spoken:
            args["time"] = spoken
    if name == "alarm.create":
        args.pop("label", None)
        if "mỗi ngày" not in low and "hằng ngày" not in low:
            args.pop("repeat", None)
    if name == "contacts.search":
        kind, obj, tool = _cursor(_parse_objects(plan), done_calls, facts)
        if kind == "run" and tool == name and obj.get("name"):
            args.clear()
            args["query"] = obj["name"]
    if name == "ride_hailing.book":
        if any(word in low for word in ("ngay bây giờ", "đặt ngay", "ngay luôn")):
            args.pop("schedule_time", None)
        if "bảy chỗ" in low or "7 chỗ" in low:
            args["vehicle"] = "car7"
        elif "bốn chỗ" in low or "4 chỗ" in low or "bấm chỗ" in low:
            args["vehicle"] = "car4"
        elif "xe máy" in low:
            args["vehicle"] = "bike"
        if any(word in low for word in ("công ty", "chỗ làm")):
            work = _context_value(context, "work_address")
            if work:
                args["pickup"] = work
        dropoff = _event_dropoff(results, request)
        if dropoff and not re.search(r"\d", request):
            args["dropoff"] = dropoff
    if name == "smart_home.light_set" and ("một nửa" in low or "độ sáng" in low):
        device = _half_light(results, request)
        if device and "một nửa" in low:
            args["device_id"] = device.get("id", args.get("device_id"))
            brightness = device.get("brightness")
            if isinstance(brightness, (int, float)):
                args["brightness"] = int(brightness) // 2
        args.pop("color", None)
        args.pop("power", None)
    if name == "calendar.update_event":
        delta = _shift_minutes(request)
        event_id = args.get("event_id")
        if delta is not None and event_id:
            for payload in results:
                for node in _walk_dicts(payload):
                    if node.get("id") != event_id or not isinstance(node.get("start"), str):
                        continue
                    args["start"] = _add_minutes(node["start"], delta)
                    if isinstance(node.get("end"), str):
                        args["end"] = _add_minutes(node["end"], delta)
    spoken = _ngan_amount(request)
    if name == "ewallet.topup_mobile":
        if spoken:
            args["amount"] = spoken
        if "cho số" in low:
            phone = _spoken_phone(request)
            if phone:
                args["phone"] = phone
        else:
            kind, obj, _tool = _cursor(_parse_objects(plan), done_calls, facts)
            contact = _matching_contact(results, obj["name"]) if kind == "run" else None
            if contact and contact.get("phone"):
                args["phone"] = contact["phone"]
            else:
                current = re.sub(r"\D", "", str(args.get("phone") or ""))
                if facts.get("phone") and len(current) < 9:
                    args["phone"] = facts["phone"]
    if name == "ewallet.transfer":
        if "tiền điện" in low and facts.get("amount"):
            args["amount"] = facts["amount"]
        kind, obj, _tool = _cursor(_parse_objects(plan), done_calls, facts)
        contact = _matching_contact(results, obj["name"]) if kind == "run" else None
        if contact:
            args["contact_id"] = contact["id"]
    if name == "bill_payment.pay":
        if facts.get("bill_id"):
            args["bill_id"] = facts["bill_id"]
        kind, obj, tool = _cursor(_parse_objects(plan), done_calls, facts)
        if kind == "run" and tool == name:
            action = obj["action"].lower()
            if "bank" in action or "ngân hàng" in action:
                args["source"] = "bank"
            elif "ví" in action or "ewallet" in action:
                args["source"] = "ewallet"
    return args


def _plan_satisfied(plan: str, done_calls: list[tuple[str, dict]], facts: dict) -> bool:
    """Return whether the object plan has nothing left to call.

    Input:
        plan (str): the object plan.
        done_calls (list[tuple[str, dict]]): successful calls so far.
        facts (dict): fields used to skip objects whose ``khi`` is false.

    Output:
        bool: True when the next step is done or waiting on a missing field.
        False when an object still has a tool to call, or the plan names no object.
    """
    objects = _parse_objects(plan)
    if not objects:
        return False
    kind, _obj, _tool = _cursor(objects, done_calls, facts)
    return kind in ("done", "wait")


def _latest_request(context_text: str) -> str:
    """Return the latest user text from the rendered context block.

    Input:
        context_text (str): prompt from ``context.render``. The request follows
        the line ``latest_request:``.

    Output:
        str: that request, or the whole block when the marker is absent.
    """
    marker = "latest_request:"
    if marker in context_text:
        return context_text.split(marker, 1)[1]
    return context_text


_MOVE_WORDS = ("muộn", "sớm", "dời", "hủy", "huỷ")


def _schedule_block(name: str, request: str) -> str | None:
    """Keep schedule edits on calendar, reminders for nhắc, alarms for báo thức.

    Input:
        name (str): tool the model wants to call.
        request (str): the latest user text. In train, every
        ``reminders.create`` contains ``nhắc``, and an ``alarm.create`` that
        also says muộn, sớm, dời, or hủy does not occur.

    Output:
        str: a refusal when the tool is the wrong kind of time entry.
        None: the call matches that split.
    """
    text = request.lower()
    if name == "reminders.create" and "nhắc" not in text:
        return "reminders chỉ để nhắc nhở. Sửa lịch trình thì dùng calendar."
    moving = any(word in text for word in _MOVE_WORDS)
    if name == "alarm.create" and moving and "báo thức" not in text:
        return "alarm chỉ để báo thức. Dời giờ một việc đã có thì dùng calendar."
    return None


def _args_key(args: dict) -> str:
    """Canonical argument text used to detect the same call twice.

    Input:
        args (dict): tool arguments.

    Output:
        str: JSON with sorted keys.
    """
    return json.dumps(args, sort_keys=True, ensure_ascii=False)


def _reuse_reason(name: str, args: dict, done_calls: list[tuple[str, dict]]) -> str | None:
    """Refuse a second call the train labels do not make.

    Input:
        name (str): tool the model wants to call.
        args (dict): its arguments.
        done_calls (list[tuple[str, dict]]): successful calls in this sample.

    Output:
        str: a refusal reason when this tool already ran with the same
        arguments, or it is not in the gold repeat set.
        None: the call targets a new item of a tool gold does repeat.
    """
    prior = [old for old_name, old in done_calls if old_name == name]
    if not prior:
        return None
    key = _args_key(args)
    if any(_args_key(old) == key for old in prior):
        return "tool này đã chạy với cùng tham số, không gọi lại"
    if name not in _REPEATABLE:
        return "tool này chỉ gọi một lần, không gọi lại"
    return None


def _plan_guard(
    plan: str,
    name: str,
    facts: dict,
    done_calls: list[tuple[str, dict]],
    _tools_by_name: dict[str, dict],
) -> str | None:
    """Reject a call that is not the next tool of the active object.

    Input:
        plan (str): the object plan. No object lines means this guard stays quiet.
        name (str): tool the model wants to call.
        facts (dict): fields used by ``khi``.
        done_calls (list[tuple[str, dict]]): successful calls so far.
        _tools_by_name (dict[str, dict]): catalog index. The caller passes it
        with the schema check. Object order does not read it.

    Output:
        str: a refusal reason. The caller must not execute the call.
        None: this is the next tool, a missing-info question, or the plan
        has no objects. Schema checks still apply.
    """
    if name == "sys.ask_missing_info":
        return None
    objects = _parse_objects(plan)
    if not objects:
        return None
    kind, obj, tool = _cursor(objects, done_calls, facts)
    if kind == "run" and name == tool:
        return None
    if kind == "run":
        return f"đối tượng {obj['name']} cần {tool}"
    if kind == "wait":
        return "chưa có kết quả để chọn đối tượng này, dừng"
    return "các đối tượng đã xong hoặc điều kiện không chọn, dừng"


def _plan_text(messages: list) -> str:
    """Return the plan sentence the plan node wrote.

    Input:
        messages (list): graph messages. The plan is the assistant reply that
        follows the plan request.

    Output:
        str: plan text. Empty when that reply is missing.
    """
    for index, message in enumerate(messages):
        content = str(getattr(message, "content", "") or "")
        if isinstance(message, HumanMessage) and content.startswith(_PLAN_HEAD + "\n"):
            follow = messages[index + 1] if index + 1 < len(messages) else None
            return _plan_lines(str(getattr(follow, "content", "") or ""))
    return ""


def _done_text(messages: list) -> str:
    """Summarize tool calls already run in this sample.

    Input:
        messages (list): graph messages, including tool calls and tool results.

    Output:
        str: one line per call, ``name(args) -> result``. ``(none)`` when no
        tool has run. Each result is clipped.
    """
    lines = []
    pending = ""
    for message in messages:
        calls = _tool_calls(message)
        if calls:
            call = calls[0]
            args = json.dumps(call["args"], ensure_ascii=False)
            pending = f"{call['name']}({_clip(args, 120)})"
            continue
        if isinstance(message, ToolMessage) and pending:
            lines.append(f"{pending} -> {_clip(str(message.content), _RESULT_CHARS)}")
            pending = ""
    if pending:
        lines.append(pending)
    return "\n".join(lines) if lines else "(none)"


def _clip(text: str, limit: int) -> str:
    """Shorten a string so a later turn cannot grow without bound.

    Input:
        text (str): result, plan, or argument text.
        limit (int): maximum characters to keep.

    Output:
        str: ``text`` unchanged when it fits, otherwise the prefix plus ``...``.
    """
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


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
        settings (dict): ``planner_model`` (served name), ``planner_base_url``,
        ``planner_api_key``, and ``temperature``. Missing values fall back to
        ``PLANNER_MODEL``, ``PLANNER_BASE_URL``, ``PLANNER_API_KEY``, and ``TEMPERATURE``.

    Output:
        ChatOpenAI: unbound chat model. Callers bind tools per sample.
    """
    global _CHAT, _CHAT_KEY
    model_name = str(settings.get("planner_model") or env_value("PLANNER_MODEL", "qwen-4b"))
    base_url = str(settings.get("planner_base_url") or env_value("PLANNER_BASE_URL", "http://127.0.0.1:8001/v1"))
    api_key = str(settings.get("planner_api_key") or env_value("PLANNER_API_KEY", "not-needed"))
    if settings.get("temperature") is None:
        temperature = float(env_value("TEMPERATURE", "0"))
    else:
        temperature = float(settings["temperature"])
    key = (model_name, base_url, api_key, temperature)
    if _CHAT is not None and _CHAT_KEY == key:
        return _CHAT
    _CHAT = ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
    )
    _CHAT_KEY = key
    return _CHAT
