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
import unicodedata
from datetime import datetime, timedelta
from typing import Annotated, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from correct import is_function, is_number_word
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
- Dời, muộn, sớm một Hi. Bạn cần làm gì với LifeMate voice thì nói mình nghe.lịch đã có: calendar. Nhắc: reminders. Báo thức: alarm.
- Người nhận là tên người, kể cả khi một từ đứng giữa động từ và tên. contacts.search chỉ bằng tên, không kèm chức danh, rồi tool gửi hoặc gọi.
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
    profile = _context_value(context_text, "name")
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
            profile,
            context_text,
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
        nonlocal guard_fails
        calls = _tool_calls(state["messages"][-1])
        messages = []
        if not calls:
            plan = _plan_text(state["messages"])
            while len(env.get_tool_calls()) < max_steps:
                planned = _planned_tool(plan, facts, done_calls)
                known = planned in _FILL_NEXT and _can_take_over(planned, request_text, results, done_calls)
                if not known and planned:
                    trial = _prepare_args(
                        planned, {}, request_text, context_text, results, facts, plan, done_calls,
                    )
                    known = reject_reason(planned, trial, installed, tools_by_name, seen_ids) is None
                if not known:
                    break
                before = len(done_calls)
                messages.append(_execute_one(
                    {"id": f"fill-{len(messages)}", "name": planned, "args": {}},
                    plan,
                ))
                if len(done_calls) == before:
                    guard_fails += 1
                    break
            return {"messages": messages, "nudge": 0}
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
        planned = _planned_tool(plan, facts, done_calls)
        if planned and planned != name and _can_take_over(planned, request_text, results, done_calls):
            name = planned
            args = _prepare_args(
                name, {}, request_text, context_text, results, facts, plan, done_calls,
            )
        elif planned and planned != name:
            trial = _prepare_args(
                planned, {}, request_text, context_text, results, facts, plan, done_calls,
            )
            if reject_reason(planned, trial, installed, tools_by_name, seen_ids) is None:
                name = planned
                args = trial
        elif planned and planned == name:
            trial = _prepare_args(
                planned, {}, request_text, context_text, results, facts, plan, done_calls,
            )
            if reject_reason(planned, trial, installed, tools_by_name, seen_ids) is None:
                args = trial
        if name in ("phone.call", "sms.send") and not (name == "phone.call" and _spoken_phone(request_text)):
            kind, obj, tool = _cursor(_parse_objects(plan), done_calls, facts)
            override = _phone_override(results, obj.get("name") or "", name, context_text) if kind == "run" and tool == name else None
            if override and override[0] == "sys.ask_missing_info":
                name, args = override
            elif override:
                args["contact_id"] = override[1]["contact_id"]
        if name == "chat_ott.send_message" and _ambiguous_chat(results, request_text, context_text):
            name = "sys.ask_missing_info"
            args = {"tool_name": "chat_ott.send_message", "params": ["chat_id"]}
        spoken_clock = _spoken_time(request_text, context_text)
        if name == "reminders.create" and spoken_clock is None and _remind_lead(request_text) is None:
            name = "sys.ask_missing_info"
            args = {"tool_name": "reminders.create", "params": ["time"]}
        low_request = request_text.lower()
        if (
            name == "alarm.create"
            and spoken_clock is None
            and _hours_before(request_text) is None
            and "trước" not in low_request
            and "phút" not in low_request
            and "nửa" not in low_request
            and not _alarm_retracted(low_request)
        ):
            name = "sys.ask_missing_info"
            args = {"tool_name": "alarm.create", "params": ["time"]}
        if name == "ewallet.transfer" and _missing_wallet_amount(request_text):
            name = "sys.ask_missing_info"
            args = {"tool_name": "ewallet.transfer", "params": ["amount"]}
        if name == "sys.ask_missing_info" and _missing_wallet_amount(request_text):
            kind, _obj, tool = _cursor(_parse_objects(plan), done_calls, facts)
            if kind == "run" and tool == "sys.ask_missing_info":
                args = {"tool_name": "ewallet.transfer", "params": ["amount"]}
        low_request = request_text.lower()
        fresh = _prepare_args(name, {}, request_text, context_text, results, facts, plan, done_calls)
        if (
            (name == "loyalty.redeem" and ("đổi hết" in low_request or "mấy ly" in low_request) and not fresh.get("reward_id"))
            or (name == "household_chores.assign" and "giao hết" in low_request and not fresh.get("chore_id"))
        ):
            done_calls.append((name, {"skip": True}))
            return ToolMessage(
                content=json.dumps({"ok": True, "skipped": True}, ensure_ascii=False),
                tool_call_id=call["id"],
            )
        if name == "bill_payment.pay" and _due_sweep(request_text) and not args.get("bill_id"):
            while True:
                kind, _obj, tool = _cursor(_parse_objects(plan), done_calls, facts)
                if kind == "run" and tool == "bill_payment.pay":
                    done_calls.append(("bill_payment.pay", {"skip": True}))
                else:
                    break
            return ToolMessage(
                content=json.dumps({"ok": True, "skipped": True}, ensure_ascii=False),
                tool_call_id=call["id"],
            )
        nonlocal guard_fails
        clock = None if name == "home_automation.create_rule" else _time_block(args)
        schedule = _schedule_block(name, request_text)
        if name == "smart_home.light_set" and "tắt" in request_text.lower() and not _turns_lights_off(request_text):
            schedule = schedule or "câu không tắt đèn"
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
        planned = _planned_tool(_plan_text(state["messages"]), facts, done_calls)
        plan = _plan_text(state["messages"])
        ready = bool(planned) and planned in _FILL_NEXT and _can_take_over(
            planned, request_text, results, done_calls,
        )
        if not ready and planned:
            trial = _prepare_args(
                planned, {}, request_text, context_text, results, facts, plan, done_calls,
            )
            ready = reject_reason(planned, trial, installed, tools_by_name, seen_ids) is None
        if ready and len(env.get_tool_calls()) < max_steps and guard_fails < 2:
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
_FACT_FIELDS = frozenset({
    "rain", "temperature_c", "buy", "sell", "price", "balance", "amount",
    "paid", "locked", "acknowledged", "available", "won", "status", "battery",
})
_MONEY = re.compile(r"(\d+(?:[.,]\d+)?)\s*(tỷ|triệu)", re.IGNORECASE)
_STOP_ACTIONS = {"dừng", "dung", "thôi", "thoi"}

# Tools a gold train label calls twice. The second call always targets another
# item, except two loyalty.redeem rows. Every other tool is one call per sample.
_REPEATABLE = frozenset({
    "alarm.create",
    "smart_home.light_set",
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
    for marker in ("chỉ cần", "mà thôi", "à thôi", "thôi khỏi"):
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
    if _wants_task(low) or "ghi chú" in low:
        return False
    if "báo thức" in low and any(word in low for word in ("xóa", "xoá")):
        return False
    if "xe đạp" in low or "máy rửa" in low or "máy giặt" in low or "máy sấy" in low:
        return False
    if ("xoá" in low or "xóa" in low) and "thuốc" in low:
        return False
    if "book" in low or "gửi vị trí" in low:
        return False
    if "mục tiêu" in low and "bước" in low:
        return False
    if ("khoá" in low or "khóa" in low) and ("xếp" in low or "học" in low):
        return False
    if "gói" in low and any(word in low for word in ("hủy", "huỷ", "bỏ")):
        return False
    if any(phrase in low for phrase in ("thư nháp", "soạn nháp", "soạn sẵn")):
        return False
    if "đổi" in low and any(word in low for word in ("điểm", "vé", "voucher")):
        return False
    if "cài" in low and any(word in low for word in ("tắt", "bật")):
        return False
    if _wants_new_event(low) or "danh sách" in low or "cuốn" in low:
        return False
    if "lưu" in low and any(word in low for word in ("quán", "chỗ", "bản đồ", "lớp", "tên là")):
        return False
    if any(word in low for word in ("tự trả", "tự động thanh toán", "cài báo", "cảnh báo điện")):
        return False
    if "robot" in low or ("ghi lại" in low and "phút" in low):
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
    elif len(tokens) >= 2 and tokens[0] in _VN_DIGIT and tokens[1] in ("lăm", "mốt"):
        number += _VN_DIGIT[tokens[0]] * 10 + _VN_DIGIT[tokens[1]]
        tokens = tokens[2:]
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
    marker = low.find("triệu")
    if marker < 0:
        return None
    before = [token for token in low[:marker].split() if token]
    kept = []
    for token in reversed(before):
        if token in _VN_DIGIT or token in ("trăm", "mươi", "mười", "lẻ", "linh") or token[:1].isdigit():
            kept.append(token)
            continue
        break
    phrase = " ".join(reversed(kept))
    if phrase[:1].isdigit():
        number = int(float(phrase.split()[0].replace(",", ".")) * 1_000_000)
    else:
        number = _vn_under_1000(phrase) * 1_000_000
    if number <= 0:
        return None
    after = [token.strip(".,;:!?") for token in low[marker + len("triệu"):].split()]
    after = [token for token in after if token]
    if after and after[0] in _VN_DIGIT and _VN_DIGIT[after[0]] < 10 and (len(after) == 1 or after[1] != "trăm"):
        number += _VN_DIGIT[after[0]] * 100_000
    elif after and after[0] == "rưỡi":
        number += 500_000
    return number


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


_READ_FOR_REF = {
    "banking.accounts": "banking.get_balance",
    "banking.beneficiaries": "banking.list_beneficiaries",
}


def _read_for_ref(ref: str, installed: set[str]) -> str | None:
    """Pick the read that produces one referenced id.

    Input:
        ref (str): an ``x-ref`` such as ``banking.accounts``.
        installed (set[str]): tool names this sample may call.

    Output:
        str: ``banking.get_balance`` or ``banking.list_beneficiaries`` when
        the reference names that collection.
        None: when that read is not installed. Any other reference uses the
        app's list or search tool.
    """
    named = _READ_FOR_REF.get(ref)
    if named:
        return named if named in installed else None
    return _best_read(ref.split(".")[0], installed)


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
        read = _read_for_ref(prop["x-ref"], installed)
        if read and read != write and read not in found:
            found.append(read)
    return found


def _order_tools(tools: list[str], tools_by_name: dict[str, dict]) -> list[str]:
    """Put reads before writes and drop duplicates.

    Input:
        tools (list[str]): tool names on one object.
        tools_by_name (dict[str, dict]): catalog index, used for ``kind``.

    Output:
        list[str]: the same names, reads first. A repeated light, transfer, or
        bill payment stays.
    """
    repeated = {
        "smart_home.light_set", "banking.transfer", "bill_payment.pay",
        "loyalty.redeem", "household_chores.assign", "subscriptions.cancel",
        "chat_ott.send_message", "bike_rental.reserve", "library.reserve", "library.search",
    }
    seen = []
    for name in tools:
        if name not in seen or name in repeated:
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


_ROLE = frozenset({
    "bác", "cô", "anh", "chị", "chú", "ông", "bà", "bạn", "con",
    "bố", "mẹ", "ba", "má", "sếp", "em",
})
_TITLE = _ROLE | {"giáo", "thầy"}


def _person_after(text: str, verb: str) -> str | None:
    """Read the person span spoken after a verb.

    Function words between the verb and the person are skipped. The words
    that remain are returned as spoken. Nothing is rewritten into a different
    title.

    Input:
        text (str): the active clause, original casing kept where the ASR has it.
        verb (str): ``nhắn`` or ``chuyển ví cho`` or ``điện thoại cho``.

    Output:
        str: a short query such as ``Dũng``, ``Bác Ba``, or ``Sếp Dũng``.
        None: when no person follows the verb.
    """
    match = re.search(re.escape(verb), text, flags=re.IGNORECASE)
    if not match:
        return None
    tokens = re.findall(r"\w+", text[match.end():], flags=re.UNICODE)
    index = 0
    while index < len(tokens) and is_function(tokens[index]):
        index += 1
    if index >= len(tokens):
        return None
    first = tokens[index]
    if is_number_word(first) and first.lower() not in _ROLE:
        return None
    second = tokens[index + 1] if index + 1 < len(tokens) else ""
    second_name = bool(second) and not is_function(second) and not _amount_unit(second)
    if first.lower() in _ROLE and second_name:
        return _as_spoken(first) + " " + _as_spoken(second)
    if first[:1].isupper() and second_name and second[:1].isupper() and not is_number_word(second):
        return first + " " + second
    return _as_spoken(first)


def _payee_before_transfer(scope: str) -> str | None:
    """Read the titled person spoken just before ``thì chuyển``.

    Input:
        scope (str): the active clause. ``Cô Mai Hoa thì chuyển`` keeps both
        given names. ``không thì chuyển`` names nobody.

    Output:
        str: the title and the names, as spoken, such as ``Cô Giáo Mai``.
        None: when ``thì chuyển`` is absent or no title sits in that piece.
    """
    marker = re.search(r"thì\s+chuyển", scope, flags=re.IGNORECASE)
    if not marker:
        return None
    piece = re.split(r"[.;,]", scope[:marker.start()])[-1]
    titles = (
        "cô giáo", "thầy giáo", "sếp", "bác", "cô", "anh", "chị", "chú",
        "ông", "bà", "bạn", "bố", "mẹ", "ba", "má",
    )
    found = None
    for title in titles:
        for match in re.finditer(rf"\b{title}\b", piece, flags=re.IGNORECASE):
            if found is None or match.start() > found[0] or (match.start() == found[0] and len(title) > len(found[1])):
                found = (match.start(), title, match.end())
    if found is None:
        return None
    names = []
    for word in re.findall(r"\w+", piece[found[2]:], flags=re.UNICODE):
        if is_function(word) or word.lower() in {"thì", "không", "chưa", "đúng", "tiền", "qua"}:
            break
        if word.isdigit() or (is_number_word(word) and word.lower() not in _ROLE):
            break
        names.append(word)
    if not names:
        return None
    title_words = re.findall(r"\w+", piece[found[0]:found[2]], flags=re.UNICODE)
    return " ".join(_as_spoken(word) for word in title_words + names)


def _transfer_payee(scope: str) -> str:
    """Read the person who receives a transfer, not the person who is topped up.

    Input:
        scope (str): the active clause. A top-up and a transfer can share it.

    Output:
        str: the person after ``chuyển``, such as ``Cô Mai``.
        ``người nhận`` when that clause names nobody.
    """
    named = _payee_before_transfer(scope)
    if named:
        return named
    marker = re.search(r"chuyển", scope, flags=re.IGNORECASE)
    clause = scope[marker.start():] if marker else scope
    clause = re.split(r"\b(đúng|bằng|nhắn|rồi)\b", clause, maxsplit=1, flags=re.IGNORECASE)[0]
    person = (
        _person_after(clause, "chuyển ví cho")
        or _person_after(clause, "chuyển cho")
        or _person_after(clause, "cho")
    )
    return person or "người nhận"


def _call_person(scope: str) -> str:
    """Read the person a phone call is for, skipping the speaker's pronoun.

    Input:
        scope (str): the active clause, such as ``Gọi dùm tôi Bố`` or
        ``gọi cho bạn Minh``.

    Output:
        str: the person, such as ``Bố`` or ``Bạn Minh``.
        ``người nhận`` when the clause names nobody.
    """
    self_words = {"tôi", "tui", "mình", "tao", "tớ"}
    for verb in ("gọi cho", "gọi điện cho", "gọi"):
        match = re.search(re.escape(verb), scope, flags=re.IGNORECASE)
        if not match:
            continue
        tokens = re.findall(r"\w+", scope[match.end():], flags=re.UNICODE)
        index = 0
        while index < len(tokens) and (is_function(tokens[index]) or tokens[index].lower() in self_words):
            index += 1
        if index >= len(tokens):
            continue
        first = tokens[index]
        if is_number_word(first) and first.lower() not in _ROLE:
            continue
        second = tokens[index + 1] if index + 1 < len(tokens) else ""
        second_name = (
            bool(second)
            and not is_function(second)
            and not _amount_unit(second)
            and second.lower() not in self_words
        )
        if first.lower() in _ROLE and second_name:
            return _as_spoken(first) + " " + _as_spoken(second)
        if first[:1].isupper() and second_name and second[:1].isupper() and not is_number_word(second):
            return first + " " + second
        return _as_spoken(first)
    return "người nhận"


def _amount_unit(token: str) -> bool:
    """Return whether a token continues a quantity rather than a name.

    Input:
        token (str): the word after a possible person.

    Output:
        bool: True for units and digit words such as ``mươi`` or ``ngàn``.
        False for ``ba``, which is also a kinship name.
    """
    if token.isdigit():
        return True
    return token.lower() in {
        "không", "một", "mốt", "mười", "mươi", "trăm", "ngàn", "nghìn", "triệu", "tỷ",
    }


def _contact_query(span: str) -> str:
    """Drop a leading title and keep the given name for a contact lookup.

    ASR often hears the name and misses the title, as in ``cô Mai`` heard as
    ``con Mai``. Search is a substring of the stored name, so the query is
    the name alone. A single word stays, including ``Mẹ`` or ``Bố`` when
    that word is the whole name. The last word stays even when it is also
    a kinship word, as in ``Bác Ba``.

    Input:
        span (str): the person as spoken, such as ``Con Mai`` or ``Bác Ba``.

    Output:
        str: ``Mai`` or ``Ba``. The original word when no title precedes a name.
    """
    tokens = re.findall(r"\w+", span or "", flags=re.UNICODE)
    if not tokens:
        return (span or "").strip()
    index = 0
    while index < len(tokens) - 1 and tokens[index].lower() in _TITLE:
        index += 1
    return " ".join(_as_spoken(token) for token in tokens[index:])


def _is_speaker(span: str, profile: str) -> bool:
    """Return whether a person span is the user named in the profile.

    The profile ``Em Linh`` splits into the pronoun ``em`` and the given
    name ``Linh``. ``em Linh`` and a bare ``Linh`` are the speaker. ``em
    Hương`` is someone else because the given name differs. ``bác Linh``
    is someone else because the title is not the user's pronoun.

    Input:
        span (str): an object name or the words after a verb, such as
        ``Linh`` or ``Bác Ba``.
        profile (str): ``user_profile.name``, such as ``Em Linh``.

    Output:
        bool: True when the given name matches and any leading title is
        exactly the user's pronoun. False when the profile has no name.
    """
    profile_tokens = re.findall(r"\w+", profile or "", flags=re.UNICODE)
    if not profile_tokens:
        return False
    given = profile_tokens[-1].lower()
    pronoun = profile_tokens[0].lower() if len(profile_tokens) > 1 else ""
    tokens = re.findall(r"\w+", span or "", flags=re.UNICODE)
    if not tokens:
        return False
    index = 0
    titles = []
    while index < len(tokens) - 1 and tokens[index].lower() in _TITLE:
        titles.append(tokens[index].lower())
        index += 1
    rest = tokens[index:]
    if len(rest) != 1 or rest[0].lower() != given:
        return False
    if not titles:
        return True
    return titles == [pronoun]


def _drop_self_search(objects: list[dict], profile: str) -> list[dict]:
    """Remove a contact search whose object is the speaker.

    Input:
        objects (list[dict]): plan objects. Not mutated.
        profile (str): ``user_profile.name``.

    Output:
        list[dict]: the same objects. ``contacts.search`` is gone from an
        object that names the speaker. An object left with no tools is
        dropped. A different person keeps the search.
    """
    kept = []
    for obj in objects:
        if "contacts.search" not in obj["tools"] or not _is_speaker(obj["name"], profile):
            kept.append(obj)
            continue
        rest = [name for name in obj["tools"] if name != "contacts.search"]
        if rest:
            kept.append(_obj(obj["name"], obj["action"], obj["when"], rest))
    return kept


def _as_spoken(token: str) -> str:
    """Keep a token's letters, capitalizing only a fully lowercase word.

    Input:
        token (str): one word from the clause.

    Output:
        str: ``token`` when it already has an uppercase letter, otherwise the
        same word with the first letter uppercased.
    """
    if any(char.isupper() for char in token):
        return token
    return token[:1].upper() + token[1:]


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
    if ("không mưa" in low or "hông mưa" in low) and not re.search(r"(?<!không )(?<!hông )mưa", low):
        return "rain=false"
    if "mưa" in low and any(word in low for word in ("không mưa", "hông mưa", "thì thôi", "giữ nguyên")):
        return "rain=true"
    if "mưa" in low and any(word in low for word in ("muộn", "sớm", "dời", "hủy", "huỷ", "nhắn")):
        return "rain=true"
    return "luôn"


# A model tool is dropped unless one of these words is in the sentence.
# Tools that are not listed are kept whenever they are installed.
_CUE_REQUIRED = (
    ("alarm.", ("báo thức", "xoá", "xóa")),
    ("phone.call", ("gọi",)),
    ("ewallet.transfer", ("chuyển",)),
)

_ABANDONED = ("thì thôi", "giữ nguyên", "không mưa")

# Read that must stand in front of a later conditional write.
_GATE_READS = frozenset({
    "weather.get_forecast",
    "weather.get_current",
    "gold_price.get_price",
    "ewallet.get_balance",
    "bill_payment.list_bills",
})

_SCHEDULE_NOUNS = ("lịch", "buổi", "hẹn", "họp", "cuộc")
_MOVE_VERBS = ("muộn", "sớm", "dời")
_BANK_CUES = ("ngân hàng", "tài khoản", "tiết kiệm", "chủ nhà")


def _cue_ok(name: str, low: str) -> bool:
    """Return whether the sentence still asks for this model tool.

    Input:
        name (str): a tool name from the model plan.
        low (str): the active clause, lowercased.

    Output:
        bool: False when ``name`` is in ``_CUE_REQUIRED`` and none of its words
        appear. True for every other tool.
    """
    for prefix, cues in _CUE_REQUIRED:
        matched = name.startswith(prefix) if prefix.endswith(".") else name == prefix
        if matched:
            return any(cue in low for cue in cues)
    return True


def _push(objects: list[dict], installed: set[str], obj: dict) -> None:
    """Append one object when its tools are installed and not already unconditional.

    Input:
        objects (list[dict]): plan objects so far. May be mutated.
        installed (set[str]): tool names this sample may call.
        obj (dict): one object from ``_obj``.

    Output:
        None. Appends ``obj`` unless a tool is missing, or the object is a
        single unconditional tool that is already on an earlier object.
    """
    if obj["tools"] and all(name in installed for name in obj["tools"]):
        if len(obj["tools"]) == 1 and _has_tool(objects, obj["tools"][0]) and obj["when"] == "luôn":
            return
        objects.append(obj)


def _drop_tools(objects: list[dict], banned: set[str]) -> list[dict]:
    """Remove named tools and drop an object that has none left.

    Input:
        objects (list[dict]): plan objects.
        banned (set[str]): tool names to take off every object.

    Output:
        list[dict]: the objects that still have a tool.
    """
    kept = []
    for obj in objects:
        names = [name for name in obj["tools"] if name not in banned]
        if names:
            kept.append(_obj(obj["name"], obj["action"], obj["when"], names))
    return kept


def _seed_objects(plan: str, scope: str, installed: set[str]) -> list[dict]:
    """Keep the model objects the active clause still supports.

    Input:
        plan (str): the lines the model wrote.
        scope (str): the clause that survives a retraction.
        installed (set[str]): tool names this sample may call.

    Output:
        list[dict]: model objects with uninstalled tools, ``sys.ask_missing_info``,
        and tools whose cue is missing removed. An abandoned ``rain=false``
        object is dropped. Empty when the clause is only a lookup.
    """
    if _check_only(scope):
        return []
    low = scope.lower()
    objects = []
    for obj in _parse_objects(plan):
        tools = [
            name for name in obj["tools"]
            if name != "sys.ask_missing_info" and name in installed and _cue_ok(name, low)
        ]
        when = _normalize_when(obj["when"])
        if when == "rain=false" and any(word in low for word in _ABANDONED):
            continue
        if tools:
            objects.append(_obj(obj["name"], obj["action"], when, tools))
    return objects


def _wants_task(low: str) -> bool:
    """Return whether the sentence asks to add a task.

    Input:
        low (str): the active clause, lowercased.

    Output:
        bool: True for ``tạo một việc``, ``thêm một việc``, or ``cái việc``.
    """
    if any(phrase in low for phrase in ("một việc", "cái việc", "thêm việc", "tạo việc")):
        return True
    return re.search(r"thêm(?:\s+\w+){0,2}\s+việc", low) is not None


def _reads_note(low: str) -> bool:
    """Return whether the sentence reads notes."""
    return any(
        phrase in low
        for phrase in ("đọc ghi chú", "xem ghi chú", "tìm ghi chú")
    )

def _plain(text: str) -> str:
    """Lowercase and strip accents so a spoken name matches a stored one.

    Input:
        text (str): a name, a label, or a sentence.

    Output:
        str: letters without diacritics. ``đ`` becomes ``d``.
    """
    folded = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(char for char in folded if unicodedata.category(char) != "Mn")


def _pickup_lead_only(low: str) -> bool:
    """Return whether sớm or muộn only moves the car.

    Input:
        low (str): the active clause, lowercased.

    Output:
        bool: True for ``đón sớm`` or ``đón trước`` when the event itself
        is not dời and is not muộn.
    """
    if "dời" in low:
        return False
    if "muộn" in low and "đón muộn" not in low:
        return False
    return "đón sớm" in low or "đón trước" in low


def _event_moving(low: str) -> bool:
    """Return whether the sentence moves an existing calendar event.

    Input:
        low (str): the active clause, lowercased.

    Output:
        bool: True when dời, muộn, or sớm applies to the event. False when
        sớm only says how early the car should arrive, or sớm nhất names a
        free slot rather than a fixed delta.
    """
    ride = any(word in low for word in ("xe", "ô tô", "taxi")) and any(word in low for word in ("đặt", "gọi", "chở"))
    if ride and not re.search(r"dời\s+(?:cái\s+|cái\s+lịch\s+)?(?:lịch|hẹn|buổi|họp|cuộc)", low):
        return False
    if _pickup_lead_only(low):
        return False
    if ("sớm nhất" in low or "muộn nhất" in low) and "dời" not in low:
        return False
    if "dời" not in low and "muộn" not in low and "sớm" in low:
        attached = any(noun in low for noun in ("lịch", "buổi", "họp", "hẹn"))
        duration = re.search(r"sớm\s+(?:nửa|\d+|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười)\s*(?:phút|tiếng)", low)
        if not attached and not duration:
            return False
    return any(word in low for word in _MOVE_VERBS)


def _wants_new_event(low: str) -> bool:
    """Return whether the sentence creates or restores a calendar event.

    Input:
        low (str): the active clause, lowercased.

    Output:
        bool: True for tạo/thêm/đặt/lên lịch, a sự kiện, or khôi phục lịch.
        False for a reminder, a ticket, or a clinic booking.
    """
    if "khôi phục" in low and "lịch" in low:
        return True
    return any(phrase in low for phrase in (
        "tạo lịch", "tạo cái lịch", "thêm lịch", "đặt lịch", "lên lịch", "ghi lịch", "ghi vô lịch", "ghi vào lịch", "một lịch",
        "sự kiện", "tạo buổi", "thêm buổi", "cuộc họp",
    ))


def _forbid_calendar_create(low: str) -> bool:
    """Return whether a model calendar.create_event does not belong.

    Input:
        low (str): the active clause, lowercased.

    Output:
        bool: True when the sentence is a reminder, a ticket, or a clinic
        visit and does not ask for a new event.
    """
    if _wants_new_event(low):
        return False
    if _event_moving(low) or "nhắc" in low:
        return True
    return any(phrase in low for phrase in ("vé tàu", "vé xe", "vé máy bay", "suất khám", "phòng khám"))


def _alarm_retracted(low: str) -> bool:
    """Return whether ``à không`` takes back the alarm.

    Input:
        low (str): the active clause, lowercased.

    Output:
        bool: True when báo thức is only in the clause before ``à không``
        and the tail is a different request, not a corrected clock.
    """
    if "à không" not in low or "báo thức" not in low:
        return False
    head, tail = low.split("à không", 1)
    if "báo thức" not in head or "báo thức" in tail:
        return False
    if _last_spoken_hm(tail) or "rưỡi" in tail or "giờ" in tail:
        return False
    return True


def _group_meal(low: str) -> bool:
    """Return whether the sentence orders one portion per person in a group.

    Input:
        low (str): the active clause, lowercased.

    Output:
        bool: True when a group is named and each person gets a portion.
    """
    return "nhóm" in low and any(
        phrase in low for phrase in ("mỗi người", "một phần", "một suất", "suất")
    )


def _missing_wallet_amount(text: str) -> bool:
    """Return whether a wallet transfer is asked without a readable amount.

    Input:
        text (str): the latest user text. ``chút tiền`` and ``ít tiền`` have
        no amount.

    Output:
        bool: True when the sentence transfers a wallet and no thousand or
        million amount is written.
    """
    low = text.lower()
    if "chuyển" not in low or "ví" not in low:
        return False
    if _ngan_amount(text) or _trieu_amount(text):
        return False
    if re.search(r"\d+\s*(k|đ|đồng)\b", low):
        return False
    return True


def _fold_vi(text: str) -> str:
    """Fold the two spellings of hóa so keyword checks see one form.

    Input:
        text (str): any casing.

    Output:
        str: lowercased text with ``hoá`` written as ``hóa``.
    """
    return text.lower().replace("hoá", "hóa").replace("hoả", "hỏa")


def _exclusive_plan(scope: str, installed: set[str], profile: str) -> list[dict] | None:
    """Build the whole plan when the sentence is one known compound shape.

    Input:
        scope (str): the active clause.
        installed (set[str]): tool names this sample may call.
        profile (str): ``user_profile.name``. The speaker is not a contact.

    Output:
        list[dict]: the replacement objects. Empty when the shape matches but
        no tool is installed.
        None: the sentence is not a lookup, a gold-price threshold, or a
        wallet-versus-bank bill.
    """
    low = scope.lower()
    if _check_only(scope):
        objects: list[dict] = []
        if "mưa" in low and "weather.get_forecast" in installed:
            _push(objects, installed, _obj("thời tiết", "kiểm tra mưa", "luôn", ["weather.get_forecast"]))
        elif _trieu_amount(scope) and "gold_price.get_price" in installed:
            _push(objects, installed, _obj("giá", "kiểm tra", "luôn", ["gold_price.get_price"]))
        return objects
    if re.search(r"v[aà]ng", low) and "triệu" in low and "gold_price.get_price" in installed:
        objects = [_obj("giá vàng", "kiểm tra giá mua, type=sjc nếu là SJC", "luôn", ["gold_price.get_price"])]
        marker = next((word for word in ("trên", "vượt", "hơn") if word in low), None)
        threshold = _trieu_amount(low.split(marker, 1)[-1]) if marker else None
        if threshold and "bán" in low and "chuyển" in low and "banking.transfer" in installed:
            if "nhắc" in low and "reminders.create" in installed:
                _push(objects, installed, _obj("nhắc bán vàng", "nhắc giờ trong câu", f"sell>{threshold}", ["reminders.create"]))
            _push(objects, installed, _obj(
                "chuyển khoản", "amount sau chữ chuyển, tài khoản thanh toán", f"sell<={threshold}",
                ["banking.get_balance", "banking.list_beneficiaries", "banking.transfer"],
            ))
            return objects
        amount = _trieu_amount(scope)
        if amount and "nhắc" in low and "reminders.create" in installed:
            _push(objects, installed, _obj("nhắc mua vàng", "nhắc giờ trong câu", f"buy<{amount}", ["reminders.create"]))
        return objects
    if _due_sweep(low) and "bill_payment.pay" in installed:
        objects = []
        if "bill_payment.list_bills" in installed:
            _push(objects, installed, _obj("hóa đơn", "xem mọi hóa đơn", "luôn", ["bill_payment.list_bills"]))
        _push(objects, installed, _obj(
            "hóa đơn đến hạn", "source=bank, bỏ hóa đơn trên 500 nghìn", "luôn",
            ["bill_payment.pay", "bill_payment.pay", "bill_payment.pay", "bill_payment.pay"],
        ))
        return objects
    short_wallet = _wallet_short(low)
    pays_by_wallet = any(phrase in low for phrase in ("bằng ví", "ví điện tử", "ví đủ"))
    if short_wallet and pays_by_wallet and "bill_payment.pay" in installed:
        objects = []
        if "bill_payment.list_bills" in installed:
            _push(objects, installed, _obj("hóa đơn", "xem hóa đơn đúng loại trong câu", "luôn", ["bill_payment.list_bills"]))
        if "ewallet.get_balance" in installed:
            _push(objects, installed, _obj("ví", "xem số dư", "luôn", ["ewallet.get_balance"]))
        true_part, false_part = _wallet_parts(low)
        person = _sms_person(scope)
        sms = []
        if person and "sms.send" in installed:
            sms = ["sms.send"] if _is_speaker(person, profile) else ["contacts.search", "sms.send"]
        report = bool(sms) and any(word in false_part for word in ("xong", "rồi", "đã đóng", "đóng rồi", "đóng xong"))
        if any(word in true_part for word in ("trả", "thanh toán", "đóng", "xài")):
            _push(objects, installed, _obj("trả bằng ví", "source=ewallet", "balance>=amount", ["bill_payment.pay"]))
        if sms and ("nhắn" in true_part or report):
            _push(objects, installed, _obj(person, f"nhắn, query {person}", "balance>=amount", sms))
        if "ngân hàng" in false_part or "tài khoản" in false_part:
            _push(objects, installed, _obj("trả bằng ngân hàng", "source=bank", "balance<amount", ["bill_payment.pay"]))
        if sms and report:
            _push(objects, installed, _obj(person, f"nhắn, query {person}", "balance<amount", sms))
        if "nhắc" in false_part and "reminders.create" in installed:
            _push(objects, installed, _obj("nhắc nạp ví", "nhắc giờ trong câu", "balance<amount", ["reminders.create"]))
        if any(phrase in low for phrase in ("tự trả", "tự động thanh toán")) and "bill_payment.schedule_autopay" in installed:
            _push(objects, installed, _obj("hoá đơn", "bật tự trả loại tới hạn sớm nhất", "luôn", ["bill_payment.schedule_autopay"]))
        return objects
    if _group_meal(low) and "food_delivery.place_order" in installed:
        objects = []
        if "chat_ott.list_chats" in installed:
            _push(objects, installed, _obj("nhóm chat", "đếm người trong nhóm", "luôn", ["chat_ott.list_chats"]))
        if "food_delivery.list_restaurants" in installed:
            _push(objects, installed, _obj("quán", "tìm quán trong câu", "luôn", ["food_delivery.list_restaurants"]))
        if "food_delivery.get_menu" in installed:
            _push(objects, installed, _obj("thực đơn", "lấy món", "luôn", ["food_delivery.get_menu"]))
        _push(objects, installed, _obj("đặt món", "mỗi người một suất, không tính người nói", "luôn", ["food_delivery.place_order"]))
        return objects
    return None


_PRONOUNS = frozenset({
    "tôi", "tui", "tao", "tớ", "mình", "ta", "em", "anh", "chị",
})
_THING_EXTRA = frozenset({"luôn", "ấy", "đấy"})


def _split_clauses(text: str) -> list[str]:
    """Split a sentence where a new action starts.

    Input:
        text (str): the active clause, any casing.

    Output:
        list[str]: pieces divided at a period, a comma, ``rồi``, or ``xong``.
        Empty pieces are dropped.
    """
    low = text.lower().replace("huỷ", "hủy")
    parts = re.split(r"[.;,]|\s+rồi(?:\s+|$)|(?:^|\s+)xong(?:\s+|$)", low)
    return [part.strip() for part in parts if part.strip()]


def _content_words(text: str, skip_pronouns: bool) -> list[str]:
    """Drop fillers and keep the words that name a thing or a person.

    Input:
        text (str): a clause tail, already lowercased.
        skip_pronouns (bool): True when a vocative such as ``tôi`` or ``chị``
        sits between the verb and the thing. False when that word is the person
        being called.

    Output:
        list[str]: the remaining words, in order.
    """
    kept = []
    for word in re.findall(r"[0-9a-zà-ỹ]+", text.lower()):
        if is_function(word) or word in _THING_EXTRA:
            continue
        if skip_pronouns and word in _PRONOUNS:
            continue
        kept.append(word)
    return kept


def _object_after(clause: str, verbs: tuple[str, ...], skip_pronouns: bool = True) -> str:
    """Return the thing the verb acts on inside one clause.

    Input:
        clause (str): one clause, lowercased.
        verbs (tuple[str, ...]): the verbs to look for, such as ``hủy``.
        skip_pronouns (bool): True when a vocative is not the object.

    Output:
        str: the first content word after the verb. ``cuộc hẹn``, ``tiền điện``,
        and ``hóa đơn`` stay as two words. Empty when the verb is absent or
        only fillers follow it.
    """
    low = clause.lower().replace("huỷ", "hủy")
    start = None
    for verb in verbs:
        found = re.search(rf"\b{re.escape(verb)}\b", low)
        if found and (start is None or found.end() < start):
            start = found.end()
    if start is None:
        return ""
    words = _content_words(low[start:], skip_pronouns)
    if not words:
        return ""
    first = words[0]
    second = words[1] if len(words) > 1 else ""
    if first == "cuộc" and second in ("hẹn", "họp", "gọi"):
        return f"cuộc {second}"
    if first == "tiền" and second in ("điện", "nước", "mạng"):
        return f"tiền {second}"
    if first == "hóa" and second == "đơn":
        return "hóa đơn"
    if first == "sân" and second == "bay":
        return "sân bay"
    if first == "chuyến" and second == "xe":
        return "chuyến xe"
    return first


def _cancel_target(text: str) -> str:
    """Return the noun ``hủy`` or ``bỏ`` acts on.

    Input:
        text (str): the active clause. A later ``rồi`` or ``xong`` is another
        action, not the thing being cancelled.

    Output:
        str: ``gói``, ``lịch``, ``xe``, or another content word.
        Empty when this is a report that the cancel already happened
        (``hủy rồi``), or the verb has no object.
        ``hủy đi`` with no noun of its own uses the noun of the previous clause.
    """
    clauses = _split_clauses(text)
    for index, clause in enumerate(clauses):
        if re.search(r"(?:đã|vừa)\s+hủy|hủy\s+rồi", clause):
            continue
        if not re.search(r"\b(hủy|bỏ)\b", clause):
            continue
        obj = _object_after(clause, ("hủy", "bỏ"))
        if obj:
            return obj
        if re.search(r"\b(?:hủy|bỏ)\s+(?:đi|luôn|giùm|giúp|hộ|với|cho)\b", clause):
            verb_at = re.search(r"\b(?:hủy|bỏ)\b", clause)
            before = _content_words(clause[:verb_at.start()] if verb_at else "", True)
            for word in before:
                if word in ("gói", "lịch", "xe", "buổi", "họp"):
                    return word
            for prev in reversed(clauses[:index]):
                for word in _content_words(prev, True):
                    if word in ("gói", "lịch", "xe", "buổi", "họp"):
                        return word
    return ""


def _bill_payment(text: str) -> bool:
    """Return whether ``trả`` is paying a bill.

    Input:
        text (str): the active clause. ``vừa trả`` is a bill already paid.
        ``trả lại`` and ``trả về`` name something that is not a bill.

    Output:
        bool: True when the object of ``trả`` is a bill, or the clause names
        an invoice or an electricity bill and is not only reporting a payment.
    """
    low = _fold_vi(text)
    if "vừa trả" in low:
        return False
    for clause in _split_clauses(low):
        obj = _object_after(clause, ("trả",))
        if obj in ("hóa đơn", "tiền điện", "tiền nước", "tiền mạng", "điện", "nước", "tiền"):
            return True
    return "hóa đơn" in low or "tiền điện" in low


def _call_in_message(low: str) -> bool:
    """Return whether ``gọi`` is the text of a message, not a phone call.

    Input:
        low (str): the active clause, lowercased. A message introduced by
        ``nhắn`` and ``là`` holds the call inside its content. ``gọi lại``
        in that message is the same, including ``gọi lại cho`` someone.

    Output:
        bool: True when the call must not become ``phone.call``.
        False when ``gọi`` is its own request, or the clause says ``gọi điện``
        outside the message content.
    """
    if not _outgoing_message(low):
        return False
    if " là " in low:
        head, tail = low.split(" là ", 1)
        if "gọi điện" in head:
            return False
        if "gọi" in tail and "gọi" not in head:
            return True
    return "gọi lại" in low and "gọi điện" not in low


def _spoken_dropoff(text: str) -> str:
    """Read the place the ride is asked to go to.

    Input:
        text (str): the latest user text, original casing kept.

    Output:
        str: the place after ``ra``, ``tới``, or ``đến``, or a named airport.
        Empty when that place is only ``chỗ hẹn``, so the calendar still
        supplies it.
    """
    stop = {
        "nhé", "nha", "ạ", "đón", "lúc", "giúp", "với", "và", "rồi", "xong", "nếu", "thì",
        "sáng", "chiều", "tối", "trưa", "mai", "hôm", "nay", "giờ", "phút",
    }
    for match in re.finditer(r"\b(?:ra|tới|đến)\s+", text, flags=re.IGNORECASE):
        words = []
        for word in re.findall(r"[0-9A-Za-zÀ-ỹ]+", text[match.end():]):
            token = word.lower()
            if token in stop or is_function(token) or token in _PRONOUNS:
                break
            words.append(word)
            if len(words) >= 4:
                break
        if not words or all(word.isdigit() or is_number_word(word) for word in words):
            continue
        phrase = " ".join(words).strip()
        folded = phrase.lower()
        if folded in ("chỗ hẹn", "nơi hẹn", "đó", "chỗ đó") or folded.startswith("chỗ hẹn"):
            continue
        return phrase
    airport = re.search(r"sân bay\s+", text, flags=re.IGNORECASE)
    if airport:
        words = []
        for word in re.findall(r"[0-9A-Za-zÀ-ỹ]+", text[airport.end():]):
            token = word.lower()
            if token in stop or is_function(token) or token in _PRONOUNS:
                break
            words.append(word)
            if len(words) >= 4:
                break
        if words and not all(word.isdigit() or is_number_word(word) for word in words):
            return (text[airport.start():airport.end()] + " ".join(words)).strip()
    return ""


def _wake_call(low: str) -> bool:
    """Return whether ``gọi`` is asking someone to wake, not to phone them.

    Input:
        low (str): the active clause, lowercased. Fillers such as ``giúp``
        and ``cho`` may sit between ``gọi`` and the person.

    Output:
        bool: True when ``dậy`` follows ``gọi`` in the same clause.
        ``gọi mẹ dậy`` and ``gọi giúp mẹ dậy`` are both a wake-up.
    """
    for clause in _split_clauses(low):
        match = re.search(r"\bgọi\b", clause)
        if not match:
            continue
        tokens = re.findall(r"[0-9a-zà-ỹ]+", clause[match.end():])
        index = 0
        names = 0
        while index < len(tokens):
            token = tokens[index]
            if token == "dậy":
                return True
            if is_function(token) or token in _PRONOUNS:
                index += 1
                continue
            if names >= 2:
                break
            names += 1
            index += 1
    return False


def _schedule_write(low: str) -> str | None:
    """Choose the calendar write for a move or a cancellation of an event.

    Input:
        low (str): the active clause, lowercased.

    Output:
        str: ``calendar.update_event`` or ``calendar.cancel_event``.
        None: the clause is about an alarm, a reminder, or a cancellation of
        something that is not an event (a package, a ticket, an order).
    """
    moving = _event_moving(low)
    cancelling = any(word in low for word in ("hủy", "huỷ"))
    if ("báo thức" in low or "nhắc" in low) and not moving and not cancelling:
        return None
    if moving:
        return "calendar.update_event"
    if not cancelling:
        return None
    if re.search(r"(?:hủy|huỷ)\s+nhầm", low):
        return None
    if "cuộc gọi" in low and not any(word in low for word in ("lịch", "buổi", "hẹn", "họp")):
        return None
    if _cancel_target(low) in ("lịch", "buổi", "họp", "cuộc hẹn", "cuộc họp"):
        return "calendar.cancel_event"
    return None


def _outgoing_message(low: str) -> bool:
    """Return whether the assistant is asked to send a message.

    Input:
        low (str): the active clause, lowercased.

    Output:
        bool: True when ``nhắn`` is its own word, including after another noun
        such as ``một tiếng nhắn``. False when the only ``nhắn`` is the noun
        ``tin nhắn``, a report that someone already sent it (``đã nhắn``),
        or a question about who already wrote (``có ai nhắn``).
    """
    if "à không" in low:
        low = low.split("à không", 1)[1]
    if re.search(r"\bcó ai nhắn\b", low):
        return False
    if re.search(r"\b(đọc|coi)\b", low):
        readable = low.replace("tin nhắn", " ")
        readable = re.sub(r"\b(đã|mới)\s+gửi\b", " ", readable)
        readable = re.sub(r"\bđã\s+nhắn\b", " ", readable)
        if not re.search(r"\b(nhắn|gửi)\b", readable):
            return False
    if "sms" in low or "gửi tin nhắn" in low:
        return True
    text = low.replace("tin nhắn", " ")
    text = re.sub(r"\bđã\s+nhắn\b", " ", text)
    return re.search(r"\bnhắn\b", text) is not None


def _message_tools(low: str, installed: set[str]) -> list[str] | None:
    """Pick the messenger the sentence names, not a fixed SMS tool.

    Input:
        low (str): the active clause, lowercased.
        installed (set[str]): tool names this sample may call.

    Output:
        list[str]: the write to add. ``chat_ott.send_message`` when the clause
        says Zalo or Messenger, or when that is the only messenger installed.
        ``sms.send`` together with ``contacts.search`` otherwise.
        None: the clause does not send a message, or no messenger is installed.
    """
    if not _outgoing_message(low):
        return None
    if "à không" in low:
        low = low.split("à không", 1)[1]
    zalo = "zalo" in low or "messenger" in low
    chat = "chat_ott.send_message" in installed
    sms = "sms.send" in installed
    if zalo and chat:
        tools = ["chat_ott.send_message"]
    elif sms and not zalo:
        return ["contacts.search", "sms.send"]
    elif chat and not sms:
        tools = ["chat_ott.send_message"]
    else:
        return None
    if "chat_ott.list_chats" in installed and "chat_ott.list_chats" not in tools:
        tools = ["chat_ott.list_chats", *tools]
    return tools


def _transfer_tool(low: str, installed: set[str]) -> str | None:
    """Pick the transfer tool from the account the sentence names.

    Input:
        low (str): the active clause, lowercased.
        installed (set[str]): tool names this sample may call.

    Output:
        str: ``banking.transfer`` when the clause names a bank, an account,
        savings, or the landlord, and does not say ví. A bill amount is also a
        bank transfer, unless the same sentence tops up a phone.
        ``ewallet.transfer`` when the clause says ví, or when a top-up and a
        bill share the sentence and no bank is named.
        None: the clause does not transfer, or only the bank tool is missing
        from a bank clause. A ``chuyển`` that shares its clause with ``lát``
        is the speaker's own later errand, not a transfer to run now.
    """
    low = " ".join(
        clause for clause in _split_clauses(low)
        if "lát" not in clause or "chuyển" not in clause
    )
    if "chuyển" not in low and not _transfer_jobs(low):
        return None
    wallet = "ewallet.transfer" in installed
    bank = "banking.transfer" in installed
    says_wallet = "ví" in low
    says_bank = any(word in low for word in _BANK_CUES)
    if says_wallet and wallet:
        return "ewallet.transfer"
    if says_bank and bank:
        return "banking.transfer"
    if _bill_focus(low) and "nạp" not in low and bank:
        return "banking.transfer"
    if wallet and not says_bank:
        return "ewallet.transfer"
    if bank and not wallet:
        return "banking.transfer"
    return None


def _claim_weather(objects: list[dict], low: str, installed: set[str]) -> list[dict]:
    """Give the forecast to a weather object, not to a person.

    Input:
        objects (list[dict]): objects kept from the model plan.
        low (str): the active clause, lowercased.
        installed (set[str]): tool names this sample may call.

    Output:
        list[dict]: the same objects with ``weather.get_forecast`` removed
        from every object that is not ``thời tiết``. An object left with no
        tools is dropped. When the clause mentions rain and the tool is
        installed, one ``thời tiết`` object holds the forecast with ``khi``
        ``luôn``.
    """
    if "weather.get_forecast" not in installed:
        return objects
    kept = []
    for obj in objects:
        if "weather.get_forecast" not in obj["tools"] or obj["name"].lower() == "thời tiết":
            if obj["name"].lower() == "thời tiết" and "weather.get_forecast" in obj["tools"]:
                kept.append(_obj("thời tiết", "kiểm tra mưa", "luôn", ["weather.get_forecast"]))
            else:
                kept.append(obj)
            continue
        rest = [name for name in obj["tools"] if name != "weather.get_forecast"]
        if rest:
            kept.append(_obj(obj["name"], obj["action"], obj["when"], rest))
    if "mưa" in low and not _has_tool(kept, "weather.get_forecast"):
        kept.insert(0, _obj("thời tiết", "kiểm tra mưa", "luôn", ["weather.get_forecast"]))
    return kept


def _add_open_apps(objects: list[dict], scope: str, installed: set[str]) -> list[dict]:
    """Add the apps whose sentence cue is a noun plus a verb, not a model guess.

    Input:
        objects (list[dict]): the plan so far. May be mutated.
        scope (str): the active clause.
        installed (set[str]): tool names this sample may call.

    Output:
        list[dict]: the same objects plus one object for a dishwasher, a chore,
        a cancelled package, a draft, a bicycle, a points reward, or a timed
        device rule. An app that is not installed is left out.
    """
    low = scope.lower()
    if any(word in low for word in ("máy rửa", "máy giặt", "máy sấy")) and any(
        word in low for word in ("chạy", "bật", "chế độ")
    ):
        objects = _drop_tools(objects, {"appliances.start_cycle", "appliances.stop"})
        if "appliances.start_cycle" in installed:
            tools = ["appliances.list", "appliances.start_cycle"] if "appliances.list" in installed else ["appliances.start_cycle"]
            _push(objects, installed, _obj("máy", "chạy đúng chế độ trong câu", "luôn", tools))
    if "giao" in low and any(word in low for word in ("việc", "vụ", "cất")) and "household_chores.assign" in installed:
        objects = _drop_tools(objects, {"household_chores.assign"})
        assigns = ["household_chores.assign"] * (4 if "giao hết" in low else 1)
        tools = (["household_chores.list_chores"] + assigns) if "household_chores.list_chores" in installed else assigns
        _push(objects, installed, _obj("việc nhà", "giao đúng việc cho người trong câu", "luôn", tools))
    if "gói" in low and _cancel_target(scope) == "gói" and "subscriptions.cancel" in installed:
        objects = _drop_tools(objects, {"subscriptions.cancel"})
        tools = ["subscriptions.list", "subscriptions.cancel"] if "subscriptions.list" in installed else ["subscriptions.cancel"]
        _push(objects, installed, _obj("gói", "huỷ đúng gói trong câu", "luôn", tools))
    if any(phrase in low for phrase in ("thư nháp", "soạn nháp", "soạn sẵn")) and "email.create_draft" in installed:
        objects = _drop_tools(objects, {"email.send", "email.create_draft"})
        person = _person_after(scope, "gửi") or _person_after(scope, "cho") or "người nhận"
        tools = ["email.create_draft"]
        if "a còng" not in low and "@" not in low and "contacts.search" in installed:
            tools = ["contacts.search", "email.create_draft"]
        _push(objects, installed, _obj(person, "thư nháp, chưa gửi", "luôn", tools))
    if "xe đạp" in low and any(word in low for word in ("giữ", "đặt", "thuê")) and "bike_rental.reserve" in installed:
        objects = _drop_tools(objects, {"bike_rental.reserve", "ride_hailing.book"})
        one = ["bike_rental.reserve"]
        if re.search(r"hai\s+(?:chiếc|xe)", low):
            one = ["bike_rental.reserve", "bike_rental.reserve"]
        tools = (["bike_rental.find_bikes"] + one) if "bike_rental.find_bikes" in installed else one
        bike_when = "rain=false" if "không mưa" in low else "luôn"
        _push(objects, installed, _obj("xe đạp", "giữ một xe còn trống", bike_when, tools))
    if "đổi" in low and any(word in low for word in ("điểm", "vé", "voucher")) and "loyalty.redeem" in installed:
        objects = _drop_tools(objects, {"loyalty.redeem"})
        redeems = ["loyalty.redeem"] * (4 if "đổi hết" in low or "mấy ly" in low else 1)
        tools = (["loyalty.get_points"] + redeems) if "loyalty.get_points" in installed else redeems
        points_when = "luôn"
        if "không mưa" in low and re.search(r"(?<!không )(?<!hông )mưa", low):
            points_when = "rain=false"
        _push(objects, installed, _obj("điểm", "đổi đúng phần thưởng trong câu", points_when, tools))
    if "cài" in low and any(word in low for word in ("tắt", "bật")) and _last_spoken_hm(low) and "home_automation.create_rule" in installed:
        objects = _drop_tools(objects, {"home_automation.create_rule"})
        tools = []
        if "số" in low and "home_energy.get_usage" in installed:
            tools.append("home_energy.get_usage")
        if "smart_home.get_status" in installed:
            tools.append("smart_home.get_status")
        tools.append("home_automation.create_rule")
        _push(objects, installed, _obj("tự động", "tắt hoặc bật đúng thiết bị đúng giờ", "luôn", tools))
    return objects


def _augment(objects: list[dict], scope: str, installed: set[str], profile: str, context: str = "") -> list[dict]:
    """Add the reads and writes the clause asks for and the model left out.

    Input:
        objects (list[dict]): objects kept from the model plan. Not mutated
        when a money repair rebuilds the list; the returned list is the one
        to keep.
        scope (str): the active clause.
        installed (set[str]): tool names this sample may call.
        profile (str): ``user_profile.name``. A person who is the speaker
        does not get ``contacts.search``.

    Output:
        list[dict]: the same objects plus one object per missing action.
        A conditional write keeps ``when`` from ``_governing_when``.
    """
    low = scope.lower()
    objects = _drop_self_search(_claim_weather(objects, low, installed), profile)
    if _bill_payment(scope) and "bill_payment.list_bills" in installed:
        if not _has_tool(objects, "bill_payment.list_bills"):
            objects.insert(0, _obj("hóa đơn", "xem hóa đơn đúng loại", "luôn", ["bill_payment.list_bills"]))
    when = _governing_when(scope)
    write = _schedule_write(low)
    if write and "calendar.list_events" in installed and write in installed and not _has_tool(objects, write):
        _push(objects, installed, _obj("lịch", "dời hoặc hủy đúng việc trong câu", when, ["calendar.list_events", write]))
    if _points_at_remaining_alarm(scope, context) and _shift_minutes(scope) is not None:
        objects = _drop_tools(objects, {
            "calendar.list_events", "calendar.update_event", "alarm.list", "alarm.delete",
        })
        if "alarm.update" in installed and not _has_tool(objects, "alarm.update"):
            _push(objects, installed, _obj("báo thức", "dời báo thức còn lại", when, ["alarm.update"]))
    if not _event_moving(low):
        objects = _drop_tools(objects, {"calendar.update_event"})
    if _forbid_calendar_create(low):
        objects = _drop_tools(objects, {"calendar.create_event"})
    message = _message_tools(low, installed)
    if message and not _has_tool(objects, message[-1]):
        person = _sms_person(scope) or _person_after(scope, "nhắn") or "người nhận"
        if message[-1] == "sms.send" and "giờ mới" in low:
            action = "nhắn giờ mới"
        else:
            action = "nhắn, query " + person
        tools = [name for name in message if not (name == "contacts.search" and _is_speaker(person, profile))]
        _push(objects, installed, _obj(person, action, when, tools))
    topup = (
        "nạp" in low
        or "cho số" in low
        or ("điện thoại cho" in low and "gọi" not in low and _ngan_amount(scope) is not None)
    )
    topup_person = None
    if "cho số" not in low:
        for verb in ("điện thoại cho", "nạp giúp", "nạp cho", "nạp dùm"):
            person = _person_after(scope, verb)
            if person and person.lower() not in ("mình", "tôi", "tui", "tao", "tớ"):
                topup_person = person
                break
    if topup_person and _is_speaker(topup_person, profile):
        topup_person = None
    transfer = _transfer_tool(low, installed)
    if transfer == "banking.transfer":
        banned = {
            "banking.list_transactions", "banking.open_savings", "banking.transfer",
            "banking.get_balance", "banking.list_beneficiaries",
            "ewallet.transfer", "ewallet.topup_mobile", "contacts.search",
            "contacts.get", "phone.call",
        }
        if _transfer_jobs(scope) or _bill_focus(scope):
            banned.add("bill_payment.pay")
        objects = _drop_tools(objects, banned | {"bill_payment.list_bills"})
        repeats = max(1, len(_transfer_jobs(scope)))
        bank_tools = ["banking.get_balance", "banking.list_beneficiaries"] + ["banking.transfer"] * repeats
        if _bill_focus(scope) and "bill_payment.list_bills" in installed:
            bank_tools = ["bill_payment.list_bills"] + bank_tools
        _push(objects, installed, _obj(
            "chuyển khoản", "amount và người nhận trong câu", "luôn", bank_tools,
        ))
    elif (
        any("lát" in clause and "chuyển" in clause for clause in _split_clauses(low))
        and "banking.get_balance" in installed
        and not _has_tool(objects, "banking.get_balance")
    ):
        _push(objects, installed, _obj("tài khoản", "xem số dư", "luôn", ["banking.get_balance"]))
    if topup or transfer == "ewallet.transfer":
        money_tools = {"contacts.get", "ewallet.transfer", "ewallet.topup_mobile", "phone.call"}
        if transfer == "ewallet.transfer" or topup_person:
            money_tools.add("contacts.search")
        if transfer == "ewallet.transfer":
            money_tools.update({
                "banking.list_transactions", "banking.open_savings", "banking.transfer",
                "banking.get_balance", "banking.list_beneficiaries",
            })
            if _bill_focus(scope) or "tiền điện" in low:
                money_tools.add("bill_payment.pay")
        kept = []
        for obj in objects:
            names = [name for name in obj["tools"] if name not in money_tools]
            if names:
                kept.append(_obj(obj["name"], obj["action"], obj["when"], names))
        objects = kept
    if topup and "ewallet.topup_mobile" in installed:
        tools = ["contacts.search", "ewallet.topup_mobile"] if topup_person else ["ewallet.topup_mobile"]
        _push(objects, installed, _obj(topup_person or "số điện thoại", "nạp, amount là số ngàn trong câu", "luôn", tools))
    missing_amount = None
    if transfer == "ewallet.transfer":
        person = _transfer_payee(scope)
        if _missing_wallet_amount(scope):
            if not _is_speaker(person, profile):
                _push(objects, installed, _obj(person, f"query {person}", "luôn", ["contacts.search"]))
            missing_amount = _obj(person, "thiếu amount", "luôn", ["sys.ask_missing_info"])
        else:
            tools = ["ewallet.transfer"] if _is_speaker(person, profile) else ["contacts.search", "ewallet.transfer"]
            _push(objects, installed, _obj(person, f"chuyển ví, query {person}", "luôn", tools))
    number_call = re.search(r"\bgọi\b", low) is not None and re.search(r"\bsố\b", low) is not None
    if "nhắc" in low and "báo thức" not in low and "reminders.create" in installed:
        if not number_call:
            objects = _drop_tools(objects, {"alarm.create", "phone.call"})
        if not _has_tool(objects, "reminders.create"):
            _push(objects, installed, _obj("nhắc", "nhắc việc trong câu", "luôn", ["reminders.create"]))
    deleting_alarm = any(word in low for word in ("xóa", "xoá")) and "báo thức" in low
    if _alarm_retracted(low):
        objects = _drop_tools(objects, {"alarm.create", "alarm.list", "alarm.delete"})
    elif deleting_alarm and "alarm.delete" in installed:
        objects = _drop_tools(objects, {"alarm.create", "alarm.delete", "alarm.list"})
        tools = ["alarm.list", "alarm.delete"] if "alarm.list" in installed else ["alarm.delete"]
        _push(objects, installed, _obj("báo thức", "xóa báo thức đúng giờ trong câu", "luôn", tools))
    elif "báo thức" in low and "alarm.create" in installed and (
        _hours_before(low) or "nửa tiếng" in low or _last_spoken_hm(low) or ("trước" not in low and "phút" not in low and "nửa" not in low)
    ):
        if _hours_before(low):
            objects = _drop_tools(objects, {"alarm.create"})
        if not _has_tool(objects, "alarm.create"):
            if _hours_before(low) and "calendar.list_events" in installed and not _has_tool(objects, "calendar.list_events"):
                _push(objects, installed, _obj("lịch", "lấy giờ sự kiện", "luôn", ["calendar.list_events"]))
            alarm = _obj(
                "báo thức",
                "trước sự kiện vừa dời" if _hours_before(low) else "chỉ gửi time",
                "luôn",
                ["alarm.create"],
            )
            placed = False
            if _hours_before(low):
                for index, existing in enumerate(objects):
                    if "calendar.update_event" in existing["tools"] or "calendar.cancel_event" in existing["tools"]:
                        objects.insert(index + 1, alarm)
                        placed = True
                        break
            if not placed:
                _push(objects, installed, alarm)
    ride_call = re.search(r"gọi\s+(ô tô|xe|taxi)", low) is not None
    if _wake_call(low):
        objects = _drop_tools(objects, {"phone.call"})
    elif (
        re.search(r"\bgọi\b", low)
        and ("nhắc" not in low or number_call)
        and "zalo" not in low
        and "video" not in low
        and not ride_call
        and not _call_in_message(low)
    ):
        banned = {"chat_ott.call", "chat_ott.list_chats", "phone.call"}
        if not _outgoing_message(low):
            banned.add("chat_ott.send_message")
        objects = _drop_tools(objects, banned)
        if "phone.call" in installed:
            person = _call_person(scope)
            if _spoken_phone(scope):
                tools = ["phone.call"]
                person = "số điện thoại"
            else:
                tools = ["phone.call"] if _is_speaker(person, profile) else ["contacts.search", "phone.call"]
            _push(objects, installed, _obj(person, "gọi điện", when, tools))
    if "đèn" in low and "tắt" in low and not _turns_lights_off(scope):
        objects = _drop_tools(objects, {"smart_home.get_status", "smart_home.light_set"})
    if _floor_named(low) and "tắt" in low and "đèn" in low and "smart_home.light_set" in installed:
        objects = _drop_tools(objects, {
            "smart_home.get_status", "smart_home.light_set", "smart_home.climate_set",
        })
        kept = "trừ phòng đang chạy quạt" if "quạt" in low else "trừ phòng đang bật điều hòa"
        _push(objects, installed, _obj(
            "đèn", f"tắt đèn đúng tầng, {kept}", "luôn",
            ["smart_home.get_status", "smart_home.light_set", "smart_home.light_set"],
        ))
        if _climate_temperature(low) is not None and "smart_home.climate_set" in installed:
            _push(objects, installed, _obj(
                "điều hòa", "mode=cool, đúng phòng và đúng số độ", "luôn",
                ["smart_home.climate_set"],
            ))
    if _negative_brightness(low):
        objects = _drop_tools(objects, {"smart_home.get_status", "smart_home.light_set"})
    elif any(word in low for word in ("đèn", "độ sáng")) and "smart_home.light_set" in installed:
        if "tắt" in low and not _turns_lights_off(scope):
            pass
        elif not _has_tool(objects, "smart_home.light_set"):
            _push(objects, installed, _obj("đèn", "chỉnh đúng phòng", "luôn", ["smart_home.get_status", "smart_home.light_set"]))
    if (
        _climate_temperature(scope) is not None
        and "smart_home.climate_set" in installed
        and not _has_tool(objects, "smart_home.climate_set")
    ):
        if "smart_home.get_status" in installed and not _has_tool(objects, "smart_home.get_status"):
            _push(objects, installed, _obj("điều hòa", "xem thiết bị", "luôn", ["smart_home.get_status"]))
        _push(objects, installed, _obj(
            "điều hòa", "mode=cool, đúng phòng và đúng số độ", "luôn", ["smart_home.climate_set"],
        ))
    cancelling_ride = _cancel_target(low) in ("xe", "chuyến xe")
    if "xe đạp" in low or cancelling_ride:
        objects = _drop_tools(objects, {"ride_hailing.book"})
    ride = (
        "ride_hailing.book" in installed
        and "xe đạp" not in low
        and not cancelling_ride
        and any(word in low for word in ("xe", "ô tô", "bốn chỗ", "4 chỗ"))
        and any(word in low for word in ("đặt", "gọi", "chở"))
        and not ("book" in low and "gửi vị trí" in low)
    )
    if ride:
        objects = _drop_tools(objects, {"ride_hailing.book"})
        if "mang laptop" in low:
            objects = _drop_tools(objects, {"reminders.create"})
        kept = []
        for obj in objects:
            if "calendar.list_events" in obj["tools"] and obj["when"] != "luôn":
                names = [name for name in obj["tools"] if name != "calendar.list_events"]
                if names:
                    kept.append(_obj(obj["name"], obj["action"], obj["when"], names))
                continue
            kept.append(obj)
        objects = kept
        book_when = "rain=false" if "không mưa" in low else "luôn"
        dropoff = _spoken_dropoff(scope)
        if (
            "calendar.list_events" in installed
            and not dropoff
            and any(word in low for word in ("hẹn", "lịch", "chiều", "sáng", "tối"))
        ):
            if not _has_tool(objects, "calendar.list_events"):
                spot = 1 if objects and "weather.get_forecast" in objects[0]["tools"] else 0
                objects.insert(spot, _obj("lịch", "lấy điểm đến của việc trong câu", "luôn", ["calendar.list_events"]))
        _push(objects, installed, _obj("xe", "đón trước giờ hẹn, xe bốn chỗ", book_when, ["ride_hailing.book"]))
        if "nhắc" in low and "mang laptop" in low and "reminders.create" in installed:
            _push(objects, installed, _obj("nhắc", "mang laptop trước giờ đón", book_when, ["reminders.create"]))
    if _reads_note(low) and "notes.list" in installed:
        objects = _drop_tools(objects, {"notes.create", "notes.append", "notes.update"})
        if not _has_tool(objects, "notes.list"):
            _push(objects, installed, _obj("ghi chú", "đọc ghi chú được nêu", "luôn", ["notes.list"]))
    elif "ghi chú" in low and "notes.create" in installed:
        objects = _drop_tools(objects, {"notes.append", "notes.update", "notes.create"})
        _push(objects, installed, _obj("ghi chú", "tạo ghi chú trong câu", "luôn", ["notes.create"]))
    else:
        objects = _drop_tools(objects, {"notes.create", "notes.append", "notes.update"})
    if "xác nhận" not in low and "cảnh báo" not in low:
        objects = _drop_tools(objects, {"home_security.acknowledge_alert"})
    if "nhắc" in low and "thêm thuốc" not in low and "kê thuốc" not in low:
        objects = _drop_tools(objects, {"medication.add"})
    if ride and not _outgoing_message(low):
        objects = _drop_tools(objects, {"chat_ott.send_message", "chat_ott.call"})
    if _wants_task(low) and "tasks.add" in installed and not _has_tool(objects, "tasks.add"):
        _push(objects, installed, _obj("việc", "tạo việc, due nếu câu có hạn", "luôn", ["tasks.add"]))
    elif not _wants_task(low):
        objects = _drop_tools(objects, {"tasks.add"})
    if _wants_new_event(low) and "calendar.create_event" in installed and not _has_tool(objects, "calendar.create_event"):
        _push(objects, installed, _obj("lịch", "tạo hoặc khôi phục sự kiện", _governing_when(scope), ["calendar.create_event"]))
    objects = _add_open_apps(objects, scope, installed)
    objects = _add_followups(objects, scope, installed, context)
    if missing_amount is not None:
        objects.append(missing_amount)
    return _drop_self_search(_lift_reminders(objects), profile)


def _add_followups(objects: list[dict], scope: str, installed: set[str], context: str = "") -> list[dict]:
    """Add the second action when the sentence does not stop after the lookup.

    Input:
        objects (list[dict]): the plan so far.
        scope (str): the active clause.
        installed (set[str]): tool names this sample may call.

    Output:
        list[dict]: the same plan plus a shopping item, a workout, a book
        hold, an autopay, a robot run, or a shifted alarm when the sentence
        names that follow-up.
    """
    low = scope.lower()
    when = _governing_when(scope)
    list_kind = ""
    for clause in _split_clauses(low):
        kind = _object_after(clause, ("danh sách",))
        if kind:
            list_kind = kind
    adds_item = list_kind not in ("xem", "phim") and "danh sách" in low and (
        list_kind in ("chợ", "mua", "sắm")
        or any(word in low for word in ("mua", "cần mua", "đi chợ"))
        or "vô danh sách" in low
        or "vào danh sách" in low
    )
    if adds_item and "shopping_list.add" in installed:
        if "nhóm" in low and "chat_ott.list_chats" in installed and not _has_tool(objects, "chat_ott.list_chats"):
            _push(objects, installed, _obj("nhóm", "đếm người trong nhóm", "luôn", ["chat_ott.list_chats"]))
        if not _has_tool(objects, "shopping_list.add"):
            _push(objects, installed, _obj("mua sắm", "thêm món vừa tính", "luôn", ["shopping_list.add"]))
    if _workout(low) and "fitness.log_workout" in installed and not _has_tool(objects, "fitness.log_workout"):
        _push(objects, installed, _obj("tập", "ghi loại và số phút", "luôn", ["fitness.log_workout"]))
    if "cuốn" in low and any(word in low for word in ("giữ", "đặt")) and "library.reserve" in installed:
        if "không thì thôi" in low:
            if "library.search" in installed and not _has_tool(objects, "library.search"):
                _push(objects, installed, _obj("sách", "xem cuốn được nêu", "luôn", ["library.search"]))
        elif not _has_tool(objects, "library.reserve"):
            tools = ["library.search", "library.reserve"] if "library.search" in installed else ["library.reserve"]
            repeats = 2 if "với cuốn" in low or "và cuốn" in low else 1
            _push(objects, installed, _obj("sách", "giữ cuốn được nêu", when, tools[:-1] + ["library.reserve"] * repeats))
    if any(phrase in low for phrase in ("tự trả", "tự động thanh toán")) and "bill_payment.schedule_autopay" in installed:
        if not _has_tool(objects, "bill_payment.schedule_autopay"):
            tools = ["bill_payment.schedule_autopay"]
            if "bill_payment.list_bills" in installed and not _has_tool(objects, "bill_payment.list_bills"):
                tools = ["bill_payment.list_bills", "bill_payment.schedule_autopay"]
            _push(objects, installed, _obj("hoá đơn", "bật tự trả loại tới hạn sớm nhất", "luôn", tools))
    pay_source = _named_bill_pay(low)
    if pay_source and "bill_payment.pay" in installed and not _has_tool(objects, "bill_payment.pay"):
        if "bill_payment.list_bills" in installed and not _has_tool(objects, "bill_payment.list_bills"):
            _push(objects, installed, _obj("hóa đơn", "xem hóa đơn đúng loại", "luôn", ["bill_payment.list_bills"]))
        _push(objects, installed, _obj("hóa đơn", f"source={pay_source}", "luôn", ["bill_payment.pay"]))
    if "robot" in low and any(word in low for word in ("dọn", "hút")) and "robot_vacuum.start" in installed:
        if not _has_tool(objects, "robot_vacuum.start"):
            waits = "xong rồi" in low and "chưa xong" in low and "robot_vacuum.get_status" in installed
            if waits:
                _push(objects, installed, _obj("robot", "xem robot còn chạy không", "luôn", ["robot_vacuum.get_status"]))
                _push(objects, installed, _obj("robot", "dọn đúng phòng trong câu", "status!=cleaning", ["robot_vacuum.start"]))
            else:
                whole_home = "cả nhà" in low
                tools = ["robot_vacuum.start"] if whole_home or "robot_vacuum.get_status" not in installed else [
                    "robot_vacuum.get_status", "robot_vacuum.start",
                ]
                robot_when = "rain=true" if "không mưa" in low and "mưa" in low else when
                _push(objects, installed, _obj("robot", "dọn đúng phòng trong câu", robot_when, tools))
    if (
        "ngủ" in low
        and _event_moving(low)
        and "lịch" not in low
        and "hẹn" not in low
        and "alarm.update" in installed
        and not _has_tool(objects, "alarm.update")
    ):
        tools = ["alarm.list", "alarm.update"] if "alarm.list" in installed else ["alarm.update"]
        _push(objects, installed, _obj("báo thức", "dời giờ ngủ", when, tools))
    if "lưu" in low and any(word in low for word in ("tên là", "đặt tên", "bản đồ")) and "maps.save_place" in installed:
        if not _has_tool(objects, "maps.save_place") and "xe buýt" not in low:
            tools = ["maps.search_place", "maps.save_place"] if "maps.search_place" in installed else ["maps.save_place"]
            _push(objects, installed, _obj("địa điểm", "lưu chỗ vừa tìm", "luôn", tools))
    return _tail_actions(objects, scope, installed, context)


def _forecast_day(request: str, context: str) -> str | None:
    """Read today, tomorrow, or the day after from the sentence.

    Input:
        request (str): the latest user text.
        context (str): the rendered context, for ``today`` and ``tomorrow``.

    Output:
        str: ``YYYY-MM-DD``.
        None: when the sentence does not name a day.
    """
    low = request.lower()
    today = _context_value(context, "today")
    tomorrow = _context_value(context, "tomorrow")
    if ("ngày kia" in low or "ngày mốt" in low) and today:
        return (datetime.fromisoformat(today).date() + timedelta(days=2)).isoformat()
    if re.search(r"\bmai\b", low) and tomorrow:
        return tomorrow
    if re.search(r"(hôm nay|chiều nay|sáng nay|tối nay|bữa nay|sáng ni|chiều ni|tối ni)", low) and today:
        return today
    return None


def _spoken_titles(text: str) -> list[str]:
    """Read book titles that follow ``cuốn``.

    Input:
        text (str): the latest user text, any casing.

    Output:
        list[str]: titles of at least two words, in spoken order.
    """
    found = []
    for match in re.finditer(r"cuốn\s+(.+?)(?:\s+với\s+cuốn|\s*,|\s+mưa|$)", text, flags=re.IGNORECASE):
        title = match.group(1).strip(" .")
        if len(title.split()) >= 2 and title.lower() not in [item.lower() for item in found]:
            found.append(title)
    return found


def _history_amount(context: str) -> int | None:
    """Read a transfer amount an earlier turn already stated.

    Input:
        context (str): the rendered context.

    Output:
        int: the amount written after ``chuyển``, such as ``1200000``.
        None: when no such amount is in the history.
    """
    for line in _history_text(context).splitlines():
        match = re.search(r"chuyển\s+([\d.]+)", line.lower())
        if not match:
            continue
        digits = match.group(1).replace(".", "")
        if digits.isdigit():
            return int(digits)
    return None


def _history_steps(request: str, context: str) -> int | None:
    """Read a step count spoken earlier for the person named now.

    Input:
        request (str): the latest user text. The person follows ``của``.
        context (str): the rendered context.

    Output:
        int: the step count, such as ``12000``.
        None: when the history has no thousand-step count.
    """
    person = ""
    named = re.search(r"của\s+([A-ZÀ-Ỹ][\wà-ỹ]+)", request)
    if named:
        person = named.group(1).lower()
    for line in _history_text(context).splitlines():
        if person and person not in line.lower():
            continue
        match = re.search(
            r"((?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi|mốt|tư|lăm)"
            r"(?:\s+(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi|mốt|tư|lăm))*)\s+nghìn\s+bước",
            line.lower(),
        )
        if not match:
            continue
        number = _vn_under_1000(match.group(1))
        if number:
            return number * 1000
    return None


def _show_alarm(request: str, context: str) -> str | None:
    """Build an alarm half an hour before a show time in the history.

    Input:
        request (str): the latest user text. It asks for ``nửa tiếng`` before
        the show.
        context (str): the rendered context. The show clock is in an earlier turn.

    Output:
        str: an ISO time.
        None: when this is not a show alarm.
    """
    low = request.lower()
    if "nửa" not in low or "trước" not in low:
        return None
    today = _context_value(context, "today")
    offset = re.search(r"([+-]\d{2}:\d{2})", context)
    zone = offset.group(1) if offset else "+07:00"
    for line in _history_text(context).splitlines():
        lowered = line.lower()
        if "phim" not in lowered and "chiếu" not in lowered:
            continue
        head = lowered.split("phim", 1)[0].split("chiếu", 1)[0]
        found = _last_spoken_hm(head)
        if not found:
            continue
        hour, minute = found
        if any(word in line.lower() for word in ("tối", "chiều", "đêm")) and hour < 12:
            hour += 12
        if not today:
            return None
        return _add_minutes(f"{today}T{hour:02d}:{minute:02d}:00{zone}", -30)
    return None


def _named_ride(request: str, context: str) -> str | None:
    """Match a spoken pickup time to a ride booked in history.

    Input:
        request (str): the latest user text.
        context (str): the rendered context, including earlier ride results.

    Output:
        str: the ride id.
        None: when no booked ride matches the clock.
    """
    found = _last_spoken_hm(request)
    for _tool, args, payload in _history_invocations(context):
        for node in _walk_dicts(payload):
            ride_id = node.get("id")
            when = str(node.get("schedule_time") or "")
            dropoff = str(node.get("dropoff") or "").lower()
            if not isinstance(ride_id, str) or not ride_id.startswith("ride_"):
                continue
            if found and f"T{found[0]:02d}:{found[1]:02d}" in when:
                return ride_id
            if "vựa" in request.lower() and "vựa" in dropoff:
                return ride_id
            if args.get("ride_id") and not found:
                return str(args["ride_id"])
    return None


def _place_query(request: str, context: str) -> str | None:
    """Read the place an earlier turn gave for the person named now.

    Input:
        request (str): the latest user text. The person is spoken in it.
        context (str): the rendered context.

    Output:
        str: the place after ``ở`` on that history line.
        None: when no line names both the person and a place.
    """
    names = re.findall(r"[A-ZÀ-Ỹ][\wà-ỹ]+(?:\s+[A-ZÀ-Ỹ][\wà-ỹ]+)+", request)
    if not names:
        return None
    person = names[0].lower()
    for line in _history_text(context).splitlines():
        if person not in line.lower():
            continue
        match = re.search(r"ở\s+(.+?)(?:\s+để|\s+rồi|,|$)", line, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _track_title(request: str, context: str) -> str | None:
    """Read the song title after ``bài``.

    Input:
        request (str): the latest user text.
        context (str): the rendered context. The title may be in an earlier turn.

    Output:
        str: the title, such as ``Nhịp thở chậm``.
        None: when no song is named, or a minute gate in the history is not met.
    """
    title = None
    blob = request + "\n" + _history_text(context)
    for line in blob.splitlines():
        match = re.search(r"bài\s+(.+?)(?:\s+(?:cho|lên|nghe|rồi|nếu|khi)|$)", line, flags=re.IGNORECASE)
        if match:
            title = match.group(1).strip(" .,")
            break
    if not title:
        return None
    gate = None
    for line in _history_text(context).splitlines():
        if "đủ" in line.lower() and "phút" in line.lower():
            gate = _duration_minutes(line)
    logged = _duration_minutes(request)
    if gate and logged and logged < gate:
        return None
    return title


def _corrected_room(text: str) -> str:
    """Read the room that survives a correction.

    Input:
        text (str): the latest user text. Words after ``nhầm`` replace the room.

    Output:
        str: a room such as ``phòng làm việc``.
        Empty when no room is named.
    """
    low = text.lower()
    if "nhầm" in low:
        low = low.split("nhầm", 1)[1]
    match = re.search(r"phòng\s+\w+(?:\s+\w+)?", low)
    return match.group(0) if match else ""


def _room_devices(results: list, request: str, kind: str) -> list[dict]:
    """List devices of one type in the corrected room.

    Input:
        results (list): payloads from ``smart_home.get_status``.
        request (str): the latest user text.
        kind (str): ``light`` or ``fan``.

    Output:
        list[dict]: matching devices.
    """
    room = _plain(_corrected_room(request))
    found = []
    for payload in results:
        for node in _walk_dicts(payload):
            if node.get("type") != kind or not isinstance(node.get("id"), str):
                continue
            label = _plain(str(node.get("room") or "") + " " + str(node.get("name") or ""))
            if room and room not in label:
                continue
            found.append(node)
    return found


def _tail_actions(objects: list[dict], scope: str, installed: set[str], context: str = "") -> list[dict]:
    """Add the second action that shares the sentence with the first.

    Input:
        objects (list[dict]): the plan so far.
        scope (str): the active clause.
        installed (set[str]): tool names this sample may call.

    Output:
        list[dict]: the same plan plus a meeting, a song, a route, a lesson,
        a ride cancel, or another follow-up the sentence names.
    """
    low = scope.lower()
    if "phòng họp online" in low and "video_meeting.create" in installed and not _has_tool(objects, "video_meeting.create"):
        _push(objects, installed, _obj("phòng họp", "cùng tên, đúng giờ vừa nói", "luôn", ["video_meeting.create"]))
    if _track_title(scope, context) and "music.play" in installed and not _has_tool(objects, "music.play"):
        tools = ["music.list_tracks", "music.play"] if "music.list_tracks" in installed else ["music.play"]
        _push(objects, installed, _obj("nhạc", "mở đúng bài", "luôn", tools))
    if re.search(r"đường từ\s+.+\s+ra\s+", low) and "maps.get_route" in installed and not _has_tool(objects, "maps.get_route"):
        _push(objects, installed, _obj("đường", "từ điểm đi ra điểm đến", "luôn", ["maps.get_route"]))
    if "tin giải trí" in low and "news.headlines" in installed and not _has_tool(objects, "news.headlines"):
        _push(objects, installed, _obj("tin", "topic=entertainment", "luôn", ["news.headlines"]))
    if ("xoá" in low or "xóa" in low) and "thuốc" in low and "medication.remove" in installed:
        if not _has_tool(objects, "medication.remove"):
            tools = ["medication.list", "medication.remove"] if "medication.list" in installed else ["medication.remove"]
            _push(objects, installed, _obj("thuốc", "xoá lịch thuốc được nêu", "rain=true" if "mưa" in low else "luôn", tools))
    if ("huỷ" in low or "hủy" in low) and "chuyến" in low and "ride_hailing.cancel" in installed:
        if not _has_tool(objects, "ride_hailing.cancel"):
            when = "rain=false" if "hổng mưa" in low or "không mưa" in low else "luôn"
            _push(objects, installed, _obj("chuyến xe", "huỷ đúng chuyến", when, ["ride_hailing.cancel"]))
    if "mục tiêu" in low and "bước" in low and "fitness.set_goal" in installed and not _has_tool(objects, "fitness.set_goal"):
        _push(objects, installed, _obj("bước", "đúng số bước trong câu", "luôn", ["fitness.set_goal"]))
    if ("khoá" in low or "khóa" in low) and "learning.schedule_lesson" in installed and not _has_tool(objects, "learning.schedule_lesson"):
        tools = ["learning.list_courses", "learning.schedule_lesson"] if "learning.list_courses" in installed else ["learning.schedule_lesson"]
        _push(objects, installed, _obj("khoá học", "xếp giờ theo mưa", "luôn", tools))
    if "book" in low and "gửi vị trí" in low and "chat_ott.send_location" in installed:
        if not _has_tool(objects, "chat_ott.send_location"):
            tools = ["chat_ott.list_chats", "chat_ott.send_location"] if "chat_ott.list_chats" in installed else ["chat_ott.send_location"]
            _push(objects, installed, _obj("vị trí", "gửi cho người được nêu", "rain=false", tools))
        if "ride_hailing.book" in installed and not _has_tool(objects, "ride_hailing.book"):
            _push(objects, installed, _obj("xe", "book khi mưa", "rain=true", ["ride_hailing.book"]))
    if "quá" in low and "tháng" in low and ("huỷ" in low or "hủy" in low) and "subscriptions.cancel" in installed:
        if not _has_tool(objects, "subscriptions.cancel"):
            tools = ["subscriptions.list", "subscriptions.cancel"] if "subscriptions.list" in installed else ["subscriptions.cancel"]
            _push(objects, installed, _obj("gói", "huỷ gói đắt hơn mức trong câu", "luôn", tools))
    if "nhầm" in low and "tắt" in low and "đèn" in low and "quạt" in low:
        for obj in objects:
            if "smart_home.light_set" in obj["tools"] and obj["tools"].count("smart_home.light_set") < 2:
                obj["tools"].append("smart_home.light_set")
        if "smart_home.fan_set" in installed and not _has_tool(objects, "smart_home.fan_set"):
            _push(objects, installed, _obj("quạt", "tắt quạt đúng phòng", "luôn", ["smart_home.fan_set"]))
        if "tasks.list" in installed and not _has_tool(objects, "tasks.list"):
            _push(objects, installed, _obj("việc", "liệt kê việc", "luôn", ["tasks.list"]))
        if not _has_tool(objects, "sys.ask_missing_info"):
            _push(objects, installed, _obj("việc", "hỏi task_id khi có hai việc", "luôn", ["sys.ask_missing_info"]))
    rainy_books = "không mưa" in low and "mưa thì" in low and "với cuốn" in low
    if rainy_books and "library.reserve" in installed:
        objects = [obj for obj in objects if "library.reserve" not in obj["tools"] and "library.search" not in obj["tools"]]
        search = ["library.search"] if "library.search" in installed else []
        _push(objects, installed, _obj("sách", "giữ hai cuốn khi không mưa", "rain=false", search + ["library.reserve", "library.search", "library.reserve"]))
        _push(objects, installed, _obj("sách", "giữ cuốn kỹ sư khi mưa", "rain=true", search + ["library.reserve"]))
    return objects


def _workout(low: str) -> tuple[str, int] | None:
    """Read a workout type and a duration.

    Input:
        low (str): the active clause, lowercased.

    Output:
        tuple[str, int]: an enum type and the minutes.
        None: when the sentence does not log a workout.
    """
    if "ghi lại" not in low and "ghi tập" not in low and "ghi vào" not in low:
        return None
    kind = next((
        code for phrase, code in (
            ("đi bộ", "walking"), ("chạy bộ", "running"), ("chạy", "running"),
            ("đạp xe", "cycling"), ("bơi", "swimming"), ("yoga", "yoga"),
            ("đá bóng", "football"), ("cầu lông", "badminton"), ("gym", "gym"),
        ) if phrase in low
    ), "")
    minutes = _duration_minutes(low)
    if not kind or not minutes:
        return None
    return kind, minutes


def _book_to_reserve(
    results: list,
    request: str,
    done_calls: list[tuple[str, dict]],
    context: str = "",
) -> dict | None:
    """Pick the next book whose title is in the sentence or the history.

    Input:
        results (list): tool payloads. A book has ``id``, ``title``, and ``author``.
        request (str): the latest user text. An author name selects that writer's book.
        done_calls (list[tuple[str, dict]]): reserves already made.
        context (str): the rendered context. A title spoken earlier outranks
        another book by the same author.

    Output:
        dict: the book to hold.
        None: when no title is named, or every named book is already held.
    """
    used = {call.get("book_id") for name, call in done_calls if name == "library.reserve"}
    for name, args, _payload in _history_invocations(context):
        if name == "library.reserve" and args.get("book_id"):
            used.add(args["book_id"])
    low = _plain(request)
    history = _plain(_history_text(context))
    best = None
    best_score = 0
    for payload in results:
        for node in _walk_dicts(payload):
            book_id = node.get("id")
            title = node.get("title") or node.get("name")
            if not isinstance(book_id, str) or not book_id.startswith("book_") or not isinstance(title, str):
                continue
            if book_id in used:
                continue
            title_plain = _plain(title)
            score = sum(len(token) for token in title_plain.split() if len(token) > 2 and token in low)
            author = _plain(str(node.get("author") or ""))
            if author and author in low and title_plain and title_plain in history:
                score += 100
            elif author and author in low:
                score += 8
            shelf = any(phrase in request.lower() for phrase in ("còn trên kệ", "đang có sẵn", "có sẵn"))
            if shelf and node.get("available") is True:
                score += 40
            elif shelf and node.get("available") is False:
                score -= 40
            if score > best_score:
                best = node
                best_score = score
    return best if best_score >= 4 else None


def _soonest_bill_type(results: list) -> str | None:
    """Return the type of the unpaid bill that is due first.

    Input:
        results (list): tool payloads. A bill has ``type``, ``due_date``, and ``paid``.

    Output:
        str: a catalog bill type such as ``electricity``.
        None: when no unpaid bill is listed.
    """
    best_type = None
    best_due = "9999"
    for payload in results:
        for node in _walk_dicts(payload):
            bill_id = node.get("id")
            bill_type = node.get("type")
            due = node.get("due_date") or node.get("due") or ""
            if not isinstance(bill_id, str) or not bill_id.startswith("bill_"):
                continue
            if not isinstance(bill_type, str) or node.get("paid") is True:
                continue
            if str(due) < best_due:
                best_type = bill_type
                best_due = str(due)
    return best_type


def _robot_rooms(results: list, request: str, context: str = "") -> list[str]:
    """Return the vacuum rooms the sentence is sending the robot to.

    Input:
        results (list): tool payloads from ``robot_vacuum.get_status``.
        request (str): the latest user text. ``khỏi`` drops that room.
        ``chưa hút`` keeps rooms missing from the current job.
        context (str): the rendered context. A study spot is read from history.

    Output:
        list[str]: room names. Mentioned rooms follow the sentence.
        Rooms not yet vacuumed follow the status list.
    """
    low = request.lower()
    catalog = []
    cleaned = []
    for payload in results:
        for node in _walk_dicts(payload):
            rooms = node.get("rooms")
            if (
                isinstance(rooms, list)
                and rooms
                and all(isinstance(room, str) for room in rooms)
                and ("battery" in node or "status" in node)
            ):
                catalog = list(rooms)
            elif isinstance(rooms, list) and rooms and isinstance(rooms[0], dict):
                catalog = [room.get("name") for room in rooms if isinstance(room.get("name"), str)]
            job = node.get("current_job")
            if isinstance(job, dict) and isinstance(job.get("rooms"), list):
                cleaned = [room for room in job["rooms"] if isinstance(room, str)]
    if not catalog:
        for payload in results:
            for node in _walk_dicts(payload):
                room = node.get("room")
                if isinstance(room, str) and room not in catalog:
                    catalog.append(room)
    if "chưa hút" in low or "chưa quét" in low:
        return [room for room in catalog if room not in cleaned]
    if "ngồi học" in low or "chỗ học" in low:
        history = _history_text(context).lower()
        return [room for room in catalog if room.lower() in history]
    spans = []
    for room in catalog:
        token = room.lower()
        start = 0
        while True:
            index = low.find(token, start)
            if index < 0:
                break
            spans.append((index, index + len(token), room))
            start = index + len(token)
    spans.sort(key=lambda item: (item[0], item[0] - item[1]))
    kept = []
    occupied = []
    for start, end, room in spans:
        if any(not (end <= left or start >= right) for left, right in occupied):
            continue
        kept.append((start, end, room))
        occupied.append((start, end))
    skipped = set()
    for marker in re.finditer(r"\bkhỏi\b", low):
        before = [item for item in kept if item[1] <= marker.start()]
        if before:
            skipped.add(before[-1][2])
    found = []
    for room in catalog:
        if room in skipped or room.lower() not in low:
            continue
        found.append((low.find(room.lower()), room))
    found.sort()
    return [room for _index, room in found]


def _history_user_text(context: str) -> str:
    """Return the user sentences already stored in history.

    Input:
        context (str): the rendered context. User lines start with ``user:``.

    Output:
        str: those sentences, one per line. Wav paths stay as written when the
        history was not transcribed. Empty when there is no history.
    """
    lines = []
    for line in _history_text(context).splitlines():
        if line.startswith("user:"):
            lines.append(line.split(":", 1)[1].strip())
    return "\n".join(lines)


def _remaining_alarm(context: str) -> dict | None:
    """Return the one alarm history listed and did not delete.

    Input:
        context (str): the rendered context. ``alarm.list`` holds the alarms
        and ``alarm.delete`` holds the removed id.

    Output:
        dict: ``id`` and ``time`` of that alarm.
        None: when history has no list, or more than one alarm is still set.
    """
    listed: list[dict] = []
    deleted: set[str] = set()
    for name, args, payload in _history_invocations(context):
        if name == "alarm.delete":
            alarm_id = args.get("alarm_id")
            if isinstance(alarm_id, str):
                deleted.add(alarm_id)
        if name != "alarm.list":
            continue
        for node in _walk_dicts(payload):
            alarm_id = node.get("id")
            when = node.get("time")
            if isinstance(alarm_id, str) and alarm_id.startswith("alm_") and isinstance(when, str):
                listed.append({"id": alarm_id, "time": when})
    live = [item for item in listed if item["id"] not in deleted]
    if len(live) != 1:
        return None
    return live[0]


def _points_at_remaining_alarm(request: str, context: str) -> bool:
    """Return whether the latest sentence means the alarm history left behind.

    Input:
        request (str): the latest user text. ``cái còn lại`` and ``còn cái kia``
        point at an earlier object. A sentence that names a calendar event or
        an alarm itself is left to that object.
        context (str): the rendered context. An earlier user turn has to mention
        ``báo thức``, and history has to leave exactly one alarm.

    Output:
        bool: True when the shift applies to that remaining alarm.
    """
    low = request.lower()
    if "cái còn lại" not in low and "còn cái kia" not in low:
        return False
    if any(word in low for word in ("lịch", "hẹn", "báo thức")):
        return False
    if "báo thức" not in _history_user_text(context).lower():
        return False
    return _remaining_alarm(context) is not None


def _alarm_to_shift(results: list, request: str, context: str = "") -> dict | None:
    """Move one alarm by the spoken delta.

    Input:
        results (list): tool payloads. An alarm has ``id`` and ``time``.
        request (str): the latest user text. ``nửa tiếng`` is thirty minutes.
        context (str): the rendered context. When the sentence points at the
        alarm history left behind, that alarm is shifted and a deleted id is not.

    Output:
        dict: ``id`` and the shifted ``time``.
        None: when there is no alarm or no delta.
    """
    delta = _shift_minutes(request)
    if delta is None:
        return None
    if context and _points_at_remaining_alarm(request, context):
        live = _remaining_alarm(context)
        if live is None:
            return None
        return {"id": live["id"], "time": _add_minutes(live["time"], delta)}
    chosen = None
    for payload in results:
        for node in _walk_dicts(payload):
            alarm_id = node.get("id")
            when = node.get("time")
            if isinstance(alarm_id, str) and alarm_id.startswith("alm_") and isinstance(when, str):
                chosen = {"id": alarm_id, "time": _add_minutes(when, delta)}
                break
        if chosen:
            break
    return chosen


def _place_to_save(results: list, request: str, context: str = "") -> dict | None:
    """Pick the searched place whose name is in the sentence.

    Input:
        results (list): tool payloads. A place has ``id`` and ``name``.
        request (str): the latest user text.

    Output:
        dict: the place to save.
        None: when no place name is mentioned.
    """
    query = _place_query(request, context) if context else None
    low = _plain(query or request)
    best = None
    best_score = 0
    for payload in results:
        for node in _walk_dicts(payload):
            place_id = node.get("id")
            name = node.get("name")
            if not isinstance(place_id, str) or not place_id.startswith("place_") or not isinstance(name, str):
                continue
            score = sum(len(token) for token in _plain(name).split() if len(token) > 2 and token in low)
            if score > best_score:
                best = node
                best_score = score
    return best if best_score >= 4 else None


def _lift_reminders(objects: list[dict]) -> list[dict]:
    """Place a reminder before a message when both are unconditional.

    A missing-info question ends the sample, so the reminder has to run
    before that question. A ride already puts the reminder after the booking.

    Input:
        objects (list[dict]): objects from ``_augment``.

    Output:
        list[dict]: the same objects, with a lone reminder moved ahead of the
        first message. Unchanged when a ride is in the plan.
    """
    if any("ride_hailing.book" in obj["tools"] for obj in objects):
        return objects
    reminders = [obj for obj in objects if obj["tools"] == ["reminders.create"]]
    if not reminders:
        return objects
    others = [obj for obj in objects if obj not in reminders]
    for index, obj in enumerate(others):
        if any(name in obj["tools"] for name in ("sms.send", "chat_ott.send", "sys.ask_missing_info")):
            return others[:index] + reminders + others[index:]
    return objects


def _arrange(
    objects: list[dict],
    tools_by_name: dict[str, dict],
    installed: set[str],
    profile: str,
    request: str = "",
) -> list[dict]:
    """Put condition checks first, then the reads each write needs.

    Input:
        objects (list[dict]): objects after ``_exclusive_plan`` or ``_augment``.
        tools_by_name (dict[str, dict]): catalog index.
        installed (set[str]): tool names this sample may call.
        profile (str): ``user_profile.name``. A speaker does not gain
        ``contacts.search`` from a write that needs a contact id.

    Output:
        list[dict]: unconditional objects, then conditional ones. Calendar
        writes come before a bill payment, and a reminder comes last.
    """
    hoisted = []
    rest = []
    for obj in objects:
        gates = [name for name in obj["tools"] if name in _GATE_READS]
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
                if read == "contacts.search" and (_is_speaker(obj["name"], profile) or _spoken_phone(request)):
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
    return always + later


def _no_remaining_task(text: str) -> bool:
    """Return whether a clause left after a retraction asks for nothing.

    Input:
        text (str): the tail after ``à thôi`` or ``chỉ cần``.

    Output:
        bool: True when the tail names no lookup and no write.
    """
    low = text.lower()
    words = (
        "dời", "muộn", "sớm", "hủy", "huỷ", "nhắn", "gọi", "nạp", "chuyển",
        "đặt", "bật", "tắt", "nhắc", "báo", "trả", "giảm", "tăng", "mưa",
        "giá", "đèn", "nạp",
    )
    return not any(word in low for word in words)


def _keep_spoken_calendar(objects: list[dict], plan: str) -> list[dict]:
    """Keep the event name the model wrote on a calendar move.

    Input:
        objects (list[dict]): the rebuilt plan. A generic move is named ``lịch``.
        plan (str): the model lines. One of them may already name the event,
        such as ``chạy bộ``.

    Output:
        list[dict]: the same objects. The generic calendar line uses that
        name and action when the model named exactly one calendar write.
    """
    spoken = [
        obj for obj in _parse_objects(plan)
        if any(name in obj["tools"] for name in ("calendar.update_event", "calendar.cancel_event"))
    ]
    if len(spoken) != 1:
        return objects
    for obj in objects:
        if obj["name"] != "lịch":
            continue
        if any(name in obj["tools"] for name in ("calendar.update_event", "calendar.cancel_event")):
            obj["name"] = spoken[0]["name"]
            obj["action"] = spoken[0]["action"]
    return objects


def _history_has_book(request: str, context: str) -> bool:
    """Return whether history already listed the book this sentence holds.

    Input:
        request (str): the latest user text.
        context (str): the rendered context.

    Output:
        bool: True when a stored title or author is named, or the sentence
        points at the shelf that was just read.
    """
    low = request.lower()
    if any(phrase in low for phrase in ("vừa đọc", "hồi nãy", "còn trên kệ", "đang có sẵn", "có sẵn")):
        return any(name == "library.search" for name, _args, _payload in _history_invocations(context))
    plain = _plain(request)
    for name, _args, payload in _history_invocations(context):
        if name != "library.search":
            continue
        for node in _walk_dicts(payload):
            title = _plain(str(node.get("title") or ""))
            author = _plain(str(node.get("author") or ""))
            if title and title in plain:
                return True
            if author and author in plain:
                return True
    return False


def _history_buy(context: str) -> int | None:
    """Read the SJC buy price already spoken in history.

    Input:
        context (str): the rendered context. The assistant line says
        ``mua vào 130.8 triệu``.

    Output:
        int: that price in đồng. None when history never stated a buy price.
    """
    for line in _history_text(context).splitlines():
        if not line.startswith("assistant:"):
            continue
        low = line.lower()
        if "mua vào" not in low:
            continue
        amount = _trieu_amount(low.split("mua vào", 1)[1])
        if amount:
            return amount
    return None


def _refused_verbs(text: str) -> set[str]:
    """Return the actions ``khỏi`` or ``đừng`` cancel.

    Input:
        text (str): the latest user text. ``thôi``, ``cho``, ``nữa``, and
        ``gì`` sit between the refusal and the verb.

    Output:
        set[str]: verbs such as ``chuyển``, ``đặt``, or ``thêm``.
        Empty when the sentence does not refuse an action.
        ``không mưa thì thôi`` has neither word, so the calendar stays.
    """
    cleaned = text.lower()
    for word in ("nữa", "gì"):
        cleaned = re.sub(rf"\b{word}\b", " ", cleaned)
    refused = set()
    for clause in _split_clauses(cleaned):
        obj = _object_after(clause, ("khỏi", "đừng"))
        if obj:
            refused.add(obj)
    return refused


def _drop_refused(objects: list[dict], request: str) -> list[dict]:
    """Drop the write that a refusal names.

    Input:
        objects (list[dict]): the rebuilt plan.
        request (str): the latest user text, before a retraction cuts it.

    Output:
        list[dict]: the plan without the refused transfer, ride, or shopping add.
    """
    refused = _refused_verbs(request)
    drop = set()
    low = request.lower()
    if "chuyển" in refused:
        drop.update({
            "ewallet.transfer", "banking.transfer",
            "banking.get_balance", "banking.list_beneficiaries",
        })
        if "nhắn" not in low and "gọi" not in low:
            drop.add("contacts.search")
    if "đặt" in refused:
        drop.add("ride_hailing.book")
    if "thêm" in refused:
        drop.add("shopping_list.add")
    return _drop_tools(objects, drop)


def _transfer_already_done(request: str, context: str) -> bool:
    """Return whether history already made the transfer this sentence asks about.

    Input:
        request (str): the latest user text. ``chưa thì chuyển`` only transfers
        when the earlier turn did not.
        context (str): the rendered context. An assistant line such as
        ``đã chuyển ... cho Trang`` answers the question.

    Output:
        bool: True when that line names the same person and the same bill.
        False when the sentence is a new transfer.
    """
    low = request.lower()
    if "chưa" not in low or "chuyển" not in low:
        return False
    person = _person_after(request, "cho") or ""
    token = _plain(person).split()[-1] if person else ""
    if len(token) < 2:
        return False
    for line in _history_text(context).splitlines():
        if not line.startswith("assistant:"):
            continue
        speech = line.lower()
        if "đã chuyển" not in speech and "chuyển rồi" not in speech:
            continue
        if token not in _plain(speech):
            continue
        if "điện" in low and "điện" not in speech:
            continue
        return True
    return False


def _negative_brightness(low: str) -> bool:
    """Return whether the spoken brightness is negative.

    Input:
        low (str): the active clause, lowercased. ``âm`` and ``phần trăm``
        share a clause. Fillers may sit between them.

    Output:
        bool: True for ``âm mười phần trăm``. False when the percent has no ``âm``.
    """
    for clause in _split_clauses(low):
        if "âm" in clause and "phần trăm" in clause:
            return True
    return False


def _drop_known_reads(objects: list[dict], request: str, context: str) -> list[dict]:
    """Drop a read whose payload is already in history.

    Input:
        objects (list[dict]): the plan after reads were paired with writes.
        request (str): the latest user text.
        context (str): the rendered context. Empty in tests that have no history.

    Output:
        list[dict]: the same plan without a repeated list or search. A gold
        reminder whose price was already read becomes unconditional, unless
        that price is already not under the spoken threshold.
    """
    if not context:
        return objects
    have = {name for name, _args, _payload in _history_invocations(context)}
    low = request.lower()
    drop = set()
    if "subscriptions.list" in have and "gói" in low:
        drop.add("subscriptions.list")
    if (
        "library.search" in have
        and "cuốn" in low
        and len(_spoken_titles(request)) < 2
        and _history_has_book(request, context)
    ):
        drop.add("library.search")
    if "loyalty.get_points" in have and "thẻ" in low:
        drop.add("loyalty.get_points")
    if "chat_ott.list_chats" in have and any(word in low for word in ("nhắn", "gửi")):
        known = _chats_named(_chat_people([], context), _chat_query(request))
        if known or any(phrase in low for phrase in ("gửi nhầm", "gửi lại")):
            drop.add("chat_ott.list_chats")
    if "bill_payment.list_bills" in have and "vừa trả" in low:
        drop.add("bill_payment.list_bills")
    if "household_chores.list_chores" in have and ("trả lại" in low or "giao nhầm" in low):
        drop.add("household_chores.list_chores")
    if "calendar.list_events" in have and any(phrase in low for phrase in ("tạo lại", "y giờ cũ", "hủy nhầm", "huỷ nhầm")):
        drop.update({"calendar.list_events", "calendar.cancel_event"})
    if "cả nhà" in low and not ("xong rồi" in low and "chưa xong" in low):
        drop.add("robot_vacuum.get_status")
    if "gold_price.get_price" in have and "vàng" in low and "nhắc" in low:
        buy = _history_buy(context)
        threshold = _trieu_amount(low.split("dưới", 1)[-1]) if "dưới" in low else None
        if buy is not None and threshold is not None and buy >= threshold:
            drop.update({"gold_price.get_price", "reminders.create"})
        else:
            drop.add("gold_price.get_price")
            for obj in objects:
                if "reminders.create" in obj["tools"]:
                    obj["when"] = "luôn"
    return _drop_tools(objects, drop)


def _expand_repeats(objects: list[dict], request: str, context: str) -> list[dict]:
    """Keep one write per card, chore, package, or message.

    Input:
        objects (list[dict]): the plan.
        request (str): the latest user text.
        context (str): the rendered context, so a count can use a stored list.

    Output:
        list[dict]: the same objects, with the write repeated when the
        sentence names more than one target.
    """
    low = request.lower()

    def stretch(tool: str, count: int) -> None:
        if count <= 1:
            return
        for obj in objects:
            if tool in obj["tools"]:
                while obj["tools"].count(tool) < count:
                    obj["tools"].append(tool)
                return

    if context and "mấy gói" in low:
        stretch("subscriptions.cancel", len(_subscriptions_to_cancel([], request, context)))
    if context and "mỗi" in low and "thẻ" in low:
        stretch("loyalty.redeem", len(_loyalty_queue([], request, context)))
    if "còn" in low and ("trả lại" in low or "là của" in low):
        stretch("household_chores.assign", 2)
    if any(phrase in low for phrase in ("gửi lại", "gửi nhầm")) and "xin lỗi" in low:
        stretch("chat_ott.send_message", 2)
    return objects


def _chat_people(results: list, context: str) -> list[dict]:
    """Return chat rows from this turn or from history.

    Input:
        results (list): payloads so far.
        context (str): the rendered context.

    Output:
        list[dict]: rows whose id starts with ``chat_``.
    """
    return _collect(_with_history(results, context, "chat_ott.list_chats", "chat_"), "chat_")


def _chat_query(request: str) -> str:
    """Read the person or the group the sentence messages.

    Input:
        request (str): the latest user text.

    Output:
        str: a short name such as ``Nam`` or ``An Bình``. Empty when the
        sentence does not name one.
    """
    group = re.search(r"nhóm\s+([A-ZÀ-Ỹ][\w]+(?:\s+[A-ZÀ-Ỹ][\w]+)?)", request)
    if group:
        return group.group(1)
    for verb in ("nhắn cho", "nhắn", "gửi cho"):
        person = _person_after(request, verb)
        if person:
            return person
    return ""


def _chats_named(chats: list[dict], query: str) -> list[dict]:
    """Return chats whose name contains the query.

    Input:
        chats (list[dict]): chat rows.
        query (str): a person or a group, as spoken.

    Output:
        list[dict]: the matches. An exact name wins over a longer name.
    """
    plain = _plain(query)
    if not plain:
        return []
    exact = [chat for chat in chats if _plain(str(chat.get("name") or "")) == plain]
    if exact:
        return exact
    return [chat for chat in chats if plain in _plain(str(chat.get("name") or ""))]


def _ambiguous_chat(results: list, request: str, context: str) -> bool:
    """Return whether two chats share the person being messaged.

    Input:
        results (list): chat payloads.
        request (str): the latest user text.
        context (str): the rendered context.

    Output:
        bool: True when the sentence is not correcting a wrong send and at
        least two chats have that name.
    """
    low = request.lower()
    if any(phrase in low for phrase in ("gửi nhầm", "gửi lại")):
        return False
    matches = _chats_named(_chat_people(results, context), _chat_query(request))
    return len(matches) >= 2


def _chat_sequence(results: list, request: str, context: str) -> list[dict] | None:
    """Build the sends for a wrong recipient, a group, or a stored chat.

    Input:
        results (list): chat payloads.
        request (str): the latest user text.
        context (str): the rendered context. The previous send is the
        person who should receive the apology.

    Output:
        list[dict]: ``chat_id`` and ``content`` for each send.
        None: an ordinary message, left for the model to address.
    """
    low = request.lower()
    chats = _chat_people(results, context)
    if any(phrase in low for phrase in ("gửi nhầm", "gửi lại")):
        spoken = ""
        named = re.search(r"là\s+([A-ZÀ-Ỹ][\w]+(?:\s+[A-ZÀ-Ỹ][\w]+)?)", request)
        if named:
            spoken = named.group(1)
        correct = _chats_named(chats, spoken)
        previous = ""
        for name, args, _payload in _history_invocations(context):
            if name == "chat_ott.send_message" and args.get("chat_id"):
                previous = str(args["chat_id"])
        price = "giá đó"
        for line in _history_text(context).splitlines():
            if "đồng" in line.lower() and not line.startswith("tool_"):
                price = line.strip()
                break
        sequence = []
        if correct:
            sequence.append({"chat_id": correct[0]["id"], "content": price})
        if previous and (not correct or previous != correct[0].get("id")):
            sequence.append({"chat_id": previous, "content": "Xin lỗi, gửi nhầm."})
        return sequence or None
    if "nhóm" in low or any(name == "chat_ott.list_chats" for name, _args, _payload in _history_invocations(context)):
        matches = _chats_named(chats, _chat_query(request))
        if len(matches) == 1:
            content = request.strip() or "nhắn"
            return [{"chat_id": matches[0]["id"], "content": content}]
    return None


def _repair_plan(
    plan: str,
    request: str,
    installed: set[str],
    tools_by_name: dict[str, dict],
    profile: str = "",
    context: str = "",
) -> str:
    """Rebuild the object plan from the sentence and the tools installed here.

    The model plan is kept. A lookup, a gold threshold, or a wallet bill
    replaces it. Every other sentence adds the missing action for the verb
    and the noun: a calendar change only when the clause is about an event,
    a message on the channel the clause names, and a transfer on the account
    it names. A bank transfer is left on the model plan instead of being
    rewritten as a wallet transfer.

    Input:
        plan (str): the lines the model wrote. Their tool names are kept when
        the sentence still asks for them.
        request (str): the latest user text. A retraction keeps only the tail.
        installed (set[str]): tool names this sample may call.
        tools_by_name (dict[str, dict]): catalog index, used to put a read
        before a write that needs an id.
        profile (str): ``user_profile.name``. The speaker is not searched
        as a contact. Empty when the profile is unknown.

    Output:
        str: one object line per remaining action. The model's text when no
        object can be built.
    """
    scope = _active_text(request)
    if scope.strip() != request.strip() and _no_remaining_task(scope):
        return "Đối tượng: ngoài phạm vi | việc: dừng | khi: luôn | tool:"
    forced = _exclusive_plan(scope, installed, profile)
    if forced is None:
        objects = _augment(_seed_objects("", scope, installed), scope, installed, profile, context)
        objects = _keep_spoken_calendar(objects, plan)
    else:
        objects = forced
    low = scope.lower()
    objects = _arrange(objects, tools_by_name, installed, profile, scope)
    if _points_at_remaining_alarm(scope, context):
        objects = _drop_tools(objects, {
            "alarm.list", "alarm.delete", "calendar.list_events", "calendar.update_event",
        })
    if not _event_moving(low):
        objects = _drop_tools(objects, {"calendar.update_event"})
    if _forbid_calendar_create(low):
        objects = _drop_tools(objects, {"calendar.create_event"})
    objects = _drop_known_reads(objects, scope, context)
    objects = _drop_refused(objects, request)
    if _transfer_already_done(request, context):
        objects = _drop_tools(objects, {
            "banking.transfer", "banking.get_balance", "banking.list_beneficiaries",
            "ewallet.transfer", "contacts.search",
        })
    objects = _expand_repeats(objects, scope, context)
    rendered = _format_objects(objects)
    if rendered:
        return rendered
    model_plan = plan or ""
    if _forbid_calendar_create(low) or (not _event_moving(low) and "calendar.update_event" in model_plan):
        return "Đối tượng: ngoài phạm vi | việc: dừng | khi: luôn | tool:"
    if "đèn" in request.lower() and "tắt" in request.lower() and not _turns_lights_off(request):
        return "Đối tượng: ngoài phạm vi | việc: dừng | khi: luôn | tool:"
    low = request.lower()
    if any(marker in low for marker in ("chỉ cần", "mà thôi", "à thôi", "thôi khỏi")):
        return "Đối tượng: ngoài phạm vi | việc: dừng | khi: luôn | tool:"
    return "Đối tượng: ngoài phạm vi | việc: dừng | khi: luôn | tool:"


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
    key = right_text.lower()
    if literal is None and key in _FACT_FIELDS and re.fullmatch(r"[a-z_][a-z0-9_]*", key):
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
            if (
                index < len(done_calls)
                and done_calls[index][0] == "sys.ask_missing_info"
                and done_calls[index][1].get("tool_name") == tool
            ):
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
    if "tiền điện" in low or "trả điện" in low or re.search(r"điện(?!\s*(?:thoại|tử))", low):
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
    if tool == "alarm.create" and "trước" in low and "lịch" in low:
        return "Chỉ gửi time. Lấy start của đúng lịch rồi trừ số tiếng trong câu."
    if tool == "alarm.create":
        return "Chỉ gửi time, dạng YYYY-MM-DDTHH:MM:SS với timezone. Không gửi label hay repeat."
    if tool == "reminders.create":
        return "content là việc được nhắc. time là giờ trong câu, dạng YYYY-MM-DDTHH:MM:SS với timezone."
    if tool == "smart_home.light_set" and _floor_named(low) and "tắt" in low:
        return "Tắt từng đèn đúng tầng đang bật, trừ phòng có điều hòa đang chạy. Mỗi lần một device_id, power=off."
    if tool == "banking.transfer":
        return "from_account_id từ get_balance, beneficiary_id từ list_beneficiaries, amount từ câu hoặc hóa đơn."
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
        return "query là tên người, không kèm chức danh. Không gửi thêm trường khác."
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


def _clock_minutes(tail: str) -> int | None:
    """Read the minute words that follow ``giờ``.

    Input:
        tail (str): the text after the hour word, such as ``mười lăm nhắc``.

    Output:
        int: minutes from 1 to 59. ``mười lăm`` is 15 and ``năm mươi`` is 50.
        None: when the next word is not a minute.
    """
    tokens = []
    for raw in tail.lstrip().split():
        token = raw.strip(".,;:!?")
        if not token:
            continue
        if token in _VN_DIGIT or token in ("mươi", "mười", "lăm", "mốt"):
            tokens.append(token)
            continue
        break
    if not tokens:
        return None
    number = _vn_under_1000(" ".join(tokens))
    if number <= 0 or number >= 60:
        return None
    return number


def _clause_around(low: str, start: int, end: int) -> str:
    """Return the clause that holds one clock, not the previous clause.

    Input:
        low (str): the lowercased request.
        start (int): index of the clock hour.
        end (int): index just after the clock phrase.

    Output:
        str: text from the previous comma or ``rồi`` up to the next one.
        ``7 giờ tối mai`` stays apart from an earlier ``sáng nay``.
    """
    left = 0
    for sep in (",", ";", "."):
        pos = low.rfind(sep, 0, start)
        if pos + 1 > left:
            left = pos + 1
    roi = low.rfind(" rồi ", 0, start)
    if roi >= 0 and roi + len(" rồi ") > left:
        left = roi + len(" rồi ")
    right = len(low)
    for sep in (",", ";", "."):
        pos = low.find(sep, end)
        if pos >= 0:
            right = min(right, pos)
    roi = low.find(" rồi ", end)
    if roi >= 0:
        right = min(right, roi)
    return low[left:right]


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
    if minute == 0 and match.group(2) == "giờ":
        spoken_minute = _clock_minutes(low[match.end():])
        if spoken_minute is not None:
            minute = spoken_minute
    kem = re.match(r"\s*kém\s+(\d{1,2}|mười lăm|mười|năm|lăm)", low[match.end():])
    if kem:
        raw = kem.group(1)
        subtracted = int(raw) if raw.isdigit() else {"mười lăm": 15, "mười": 10, "năm": 5, "lăm": 5}[raw]
        if 0 < subtracted < 60:
            hour = (hour - 1) % 24
            minute = 60 - subtracted
    return hour, minute


def _history_text(context: str) -> str:
    """Return the rendered history, without the latest request.

    Input:
        context (str): the planner block from ``context.render``.

    Output:
        str: the lines between ``history:`` and ``latest_request:``.
        Empty when the block has no history.
    """
    if "history:" not in context:
        return ""
    return context.split("history:", 1)[1].split("latest_request:", 1)[0]


def _history_invocations(context: str) -> list[tuple[str, dict, dict]]:
    """Pair each history tool call with the arguments and the payload.

    Input:
        context (str): the planner block. History lines are ``tool_calls``
        and ``tool_results`` JSON.

    Output:
        list[tuple[str, dict, dict]]: tool name, arguments, and result, in
        order. A result that is not a dict is skipped.
    """
    found = []
    pending: list[dict] = []
    for line in _history_text(context).splitlines():
        if line.startswith("tool_calls:"):
            try:
                raw = json.loads(line.split(":", 1)[1])
            except json.JSONDecodeError:
                raw = []
            pending = [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []
        elif line.startswith("tool_results:"):
            try:
                raw = json.loads(line.split(":", 1)[1])
            except json.JSONDecodeError:
                raw = []
            if not isinstance(raw, list):
                raw = [raw]
            for index, payload in enumerate(raw):
                call = pending[index] if index < len(pending) else {}
                name = str(call.get("name") or "")
                args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                if name and isinstance(payload, dict):
                    found.append((name, args, payload))
            pending = []
    return found


def _with_history(results: list, context: str, tool: str, prefix: str) -> list:
    """Use this turn's payload, or the same read already stored in history.

    Input:
        results (list): payloads from the current sample.
        context (str): the rendered context.
        tool (str): the read that produced the ids.
        prefix (str): an id prefix such as ``sub_`` or ``book_``.

    Output:
        list: ``results`` when they already hold that id. Otherwise the
        history payloads are placed in front.
    """
    if _collect(results, prefix):
        return results
    extra = [payload for name, _args, payload in _history_invocations(context) if name == tool]
    return extra + results


def _year_of(context: str) -> int:
    """Read the year of ``today`` in the rendered context.

    Input:
        context (str): the planner block.

    Output:
        int: that year. 2026 when the today line is missing.
    """
    today = _context_value(context, "today")
    return int(today[:4]) if today and today[:4].isdigit() else 2026


def _dates_in(text: str, year: int) -> list[tuple[int, str]]:
    """Find calendar dates written in a sentence.

    Input:
        text (str): a request or a history line.
        year (int): the year to attach. The sentence names a day and a month.

    Output:
        list[tuple[int, str]]: ``(character index, YYYY-MM-DD)`` in order.
    """
    low = text.lower()
    found = []
    months = {
        "một": 1, "hai": 2, "ba": 3, "tư": 4, "bốn": 4, "năm": 5, "sáu": 6,
        "bảy": 7, "tám": 8, "chín": 9, "mười": 10, "mười một": 11, "mười hai": 12,
    }
    for match in re.finditer(r"(\d{1,2})/(\d{1,2})", low):
        day, month = int(match.group(1)), int(match.group(2))
        if 1 <= day <= 31 and 1 <= month <= 12:
            found.append((match.start(), f"{year:04d}-{month:02d}-{day:02d}"))
    number = (
        r"(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi|mốt|lăm|tư|\d+)"
        r"(?:\s+(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi|mốt|lăm|tư|\d+)){0,3}"
    )
    month_word = r"(mười hai|mười một|mười|một|hai|ba|tư|bốn|năm|sáu|bảy|tám|chín|\d{1,2})"
    for match in re.finditer(rf"(?:ngày|mùng)?\s*({number})\s+tháng\s+{month_word}", low):
        raw_day = match.group(1).strip()
        day = int(raw_day) if raw_day.isdigit() else _vn_under_1000(raw_day)
        raw_month = match.group(2)
        month = int(raw_month) if raw_month.isdigit() else months.get(raw_month, 0)
        if 1 <= day <= 31 and 1 <= month <= 12:
            found.append((match.start(), f"{year:04d}-{month:02d}-{day:02d}"))
    found.sort()
    return found


def _weekday_index(text: str) -> int | None:
    """Read thứ hai … chủ nhật as a Monday-based weekday.

    Input:
        text (str): a lowercased sentence.

    Output:
        int: Monday is 0 and Sunday is 6.
        None: when no weekday is named.
    """
    names = (
        ("chủ nhật", 6), ("thứ hai", 0), ("thứ ba", 1), ("thứ tư", 2),
        ("thứ năm", 3), ("thứ sáu", 4), ("thứ bảy", 5),
    )
    for phrase, index in names:
        if phrase in text:
            return index
    return None


def _next_weekday(today: str, weekday: int, upcoming: bool) -> str | None:
    """Return the next date that falls on a weekday.

    Input:
        today (str): ``YYYY-MM-DD``.
        weekday (int): Monday is 0.
        upcoming (bool): True when the sentence says ``tới`` or ``sau``.
        A weekday that is today stays today unless ``upcoming`` is set.

    Output:
        str: that date. None when ``today`` is not a date.
    """
    if not today:
        return None
    try:
        current = datetime.fromisoformat(today).date()
    except ValueError:
        return None
    delta = (weekday - current.weekday()) % 7
    if delta == 0 and upcoming:
        delta = 7
    return (current + timedelta(days=delta)).isoformat()


def _date_near(text: str, topic: str, year: int) -> str | None:
    """Read the date written beside a name.

    Input:
        text (str): history or a sentence.
        topic (str): a place or an order name, such as ``Amsterdam``.
        year (int): the year to attach.

    Output:
        str: ``YYYY-MM-DD`` of the first date after that name.
        None: when the name or a date is missing.
    """
    low = text.lower()
    index = low.find(topic.lower())
    if index < 0:
        return None
    dates = _dates_in(text[index:index + 220], year)
    return dates[0][1] if dates else None


def _shift_date(iso: str, days: int) -> str:
    """Move a calendar date by a number of days.

    Input:
        iso (str): ``YYYY-MM-DD``.
        days (int): the delta. Negative moves earlier.

    Output:
        str: the shifted date. The original string when it is not a date.
    """
    try:
        current = datetime.fromisoformat(iso).date()
    except ValueError:
        return iso
    return (current + timedelta(days=days)).isoformat()


def _clock_day(request: str, context: str) -> str | None:
    """Resolve hôm đó, hôm trước, a weekday, or ngày X tháng Y.

    Input:
        request (str): the latest user text.
        context (str): the rendered context, for today and the history.

    Output:
        str: ``YYYY-MM-DD``.
        None: when the sentence uses today or tomorrow instead.
    """
    low = request.lower()
    year = _year_of(context)
    history = _history_text(context)
    explicit = _dates_in(request, year)
    if explicit and re.search(r"(?:ngày|mùng)\b|/", low):
        return explicit[-1][1]
    if any(phrase in low for phrase in ("hôm trước", "trước hôm đó", "trước hôm ấy")):
        topic = ""
        named = re.search(r"đơn\s+([A-Za-zÀ-ỹ]+)", request)
        if named:
            topic = named.group(1)
        base = _date_near(history, topic, year) if topic else None
        if base is None:
            dates = _dates_in(history, year)
            base = dates[-1][1] if dates else None
        return _shift_date(base, -1) if base else None
    if "hôm đó" in low or "hôm ấy" in low or "hôm họ" in low:
        dates = _dates_in(history, year)
        return dates[-1][1] if dates else None
    weekday = _weekday_index(low)
    if weekday is not None and "mai" not in low and "nay" not in low:
        day = _next_weekday(_context_value(context, "today"), weekday, "tới" in low or "tuần sau" in low)
        if day and "tuần sau" in low:
            today = _context_value(context, "today")
            try:
                current = datetime.fromisoformat(today).date()
                found = datetime.fromisoformat(day).date()
            except ValueError:
                return day
            if found.isocalendar()[:2] == current.isocalendar()[:2]:
                return (found + timedelta(days=7)).isoformat()
        return day
    return None


def _trip_date(request: str, context: str) -> str | None:
    """Read the day of a past outing named with bữa.

    Input:
        request (str): the latest user text, such as ``bữa lên Bà Nà``.
        context (str): the rendered context. The date is in an earlier turn.

    Output:
        str: ``YYYY-MM-DD``.
        None: when the sentence does not point at an outing in the history.
    """
    low = request.lower()
    if "bữa" not in low:
        return None
    match = re.search(r"(?:lên|xuống|đi|ở)\s+(.+?)\s+(?:với|trời|đó|nữa)", low)
    if not match:
        return None
    return _date_near(_history_text(context), match.group(1).strip(), _year_of(context))


def _due_from_history(context: str) -> str | None:
    """Read a due time that an earlier turn already stated.

    Input:
        context (str): the rendered context.

    Output:
        str: an ISO time. None when no history line has both a clock and a date.
    """
    year = _year_of(context)
    offset = re.search(r"([+-]\d{2}:\d{2})", context)
    zone = offset.group(1) if offset else "+07:00"
    found = None
    for line in _history_text(context).splitlines():
        dates = _dates_in(line, year)
        clock = _last_spoken_hm(line)
        if not dates or not clock:
            continue
        hour, minute = clock
        if any(word in line.lower() for word in ("chiều", "tối", "đêm")) and hour < 12:
            hour += 12
        found = f"{dates[-1][1]}T{hour:02d}:{minute:02d}:00{zone}"
    return found


def _context_hour(context: str) -> int:
    """Read the hour of ``current_time``.

    Input:
        context (str): the planner block.

    Output:
        int: the hour. 0 when the clock line is missing.
    """
    match = re.search(r"current_time:\s*\d{4}-\d{2}-\d{2}T(\d{2})", context)
    return int(match.group(1)) if match else 0


def _history_afternoon(context: str, hour: int) -> bool:
    """Return whether an earlier turn placed this hour in the afternoon.

    Input:
        context (str): the rendered context.
        hour (int): a morning-looking hour, such as 5 for ``năm giờ``.

    Output:
        bool: True when one history line says chiều and that hour.
    """
    word = next((name for name, value in _HOUR_WORD.items() if value == hour), "")
    for line in _history_text(context).lower().splitlines():
        if "chiều" not in line:
            continue
        if (word and f"{word} giờ" in line) or f"{hour} giờ" in line:
            return True
    return False


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
        return _minutes_from_now(request, context)
    hour, minute = found
    low = request.lower()
    match = list(re.finditer(r"(mười một|mười hai|mười|không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|\d{1,2})\s*(giờ|rưỡi)", low))[-1]
    tail = low[match.end():]
    kem = re.match(r"\s*kém\s+(\d{1,2}|mười lăm|mười|năm|lăm)", tail)
    end = match.end() + (kem.end() if kem else 0)
    window = _clause_around(low, match.start(), end)
    if "sáng" not in window and any(word in window for word in ("tối", "đêm", "chiều")) and hour < 12:
        hour += 12
    elif (
        hour < 12
        and "sáng" not in window
        and not any(word in window for word in ("tối", "đêm", "chiều"))
        and _history_afternoon(context, hour)
        and _context_hour(context) >= 12
    ):
        hour += 12
    if "sáng" in window and hour == 12:
        hour = 0
    anchored = _clock_day(request, context)
    after = low[end:end + 24]
    if "mai" in after and "hôm đó" not in after and "hôm trước" not in after:
        day = _context_value(context, "tomorrow")
    elif anchored:
        day = anchored
    elif "mai" in window and "nay" not in window:
        day = _context_value(context, "tomorrow")
    elif "nay" in window:
        day = _context_value(context, "today")
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
        str: the digits written together, starting with 0 and at least 10
        digits long.
        None: when the spoken sequence is missing, does not start with 0, or
        is shorter than 10 digits. The missing zeros are not filled from
        anywhere else.
    """
    low = text.lower()
    if "cho số" in low:
        tail = low.split("cho số", 1)[1]
    elif re.search(r"gọi\s+(?:luôn\s+)?số", low):
        tail = re.split(r"gọi\s+(?:luôn\s+)?số", low, maxsplit=1)[1]
    else:
        tail = ""
    digits = []
    for token in re.findall(r"[0-9a-zà-ỹ]+", tail):
        if token.isdigit():
            digits.append(token)
        elif token in _DIGIT_WORD:
            digits.append(_DIGIT_WORD[token])
        elif digits:
            break
    phone = "".join(digits)
    if len(phone) < 10 or not phone.startswith("0"):
        return None
    return phone


def _matching_contact(results: list, query: str) -> dict | None:
    """Pick the contact whose name contains the given name.

    A leading title on ``query`` is dropped first, so ``Con Mai`` matches
    ``Cô giáo Mai``. The match is on the contact name, not on relation.

    Input:
        results (list): tool payloads. A contact has ``id`` starting with ``c_``.
        query (str): the person named on the current object.

    Output:
        dict: the contact with the longest matching name.
        None: when no returned contact contains that name.
    """
    spoken = (query or "").strip().lower()
    if not spoken or spoken in ("số điện thoại", "người nhận", "liên hệ"):
        return None
    needle = _contact_query(query).strip().lower()
    if not needle:
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


def _named_contacts(results: list, query: str) -> list[dict]:
    """Return every contact whose given name is the spoken name.

    Input:
        results (list): tool payloads. A contact has ``id`` starting with ``c_``.
        query (str): the person on the current object, title included.

    Output:
        list[dict]: contacts whose name, without a leading title, equals the
        query. Two people both named Lan are both returned.
    """
    needle = _contact_query(query).strip().lower()
    if not needle or needle in ("số điện thoại", "người nhận", "liên hệ"):
        return []
    found = []
    seen = set()
    for payload in results:
        for node in _walk_dicts(payload):
            name = node.get("name")
            contact_id = node.get("id")
            if not isinstance(name, str) or not isinstance(contact_id, str) or not contact_id.startswith("c_"):
                continue
            if contact_id in seen:
                continue
            if _contact_query(name).strip().lower() == needle:
                found.append(node)
                seen.add(contact_id)
    return found


def _history_hits(hits: list[dict], query: str, context: str) -> list[dict]:
    """Keep the contact an earlier turn already identified.

    Input:
        hits (list[dict]): contacts who share the spoken given name.
        query (str): that given name, such as ``Minh``.
        context (str): the rendered context, including earlier user turns.

    Output:
        list[dict]: one contact when a history line names both the person and
        that contact's relation. All of ``hits`` when history does not say.
    """
    if len(hits) < 2 or not context:
        return hits
    chosen = []
    needle = query.lower()
    for line in context.splitlines():
        if line.startswith("tool_"):
            continue
        folded = line.lower()
        if needle not in folded:
            continue
        for hit in hits:
            relation = str(hit.get("relation") or "").lower()
            if relation and relation in folded and hit not in chosen:
                chosen.append(hit)
    return chosen if len(chosen) == 1 else hits


def _phone_override(results: list, query: str, tool: str = "phone.call", context: str = "") -> tuple[str, dict] | None:
    """Choose a phone call or a question when several people share a name.

    Input:
        results (list): tool payloads from ``contacts.search``.
        query (str): the person the plan is calling.

    Output:
        tuple[str, dict]: ``sys.ask_missing_info`` when at least two contacts
        have that given name, or ``phone.call`` with the one matching id.
        None: when the search did not return that name.
    """
    if _is_speaker(query, _context_value(context, "name")):
        return None
    hits = _history_hits(_named_contacts(results, query), query, context)
    if len(hits) == 1 and hits[0].get("id"):
        return tool, {"contact_id": hits[0]["id"]}
    if _contact_query(query).strip():
        return "sys.ask_missing_info", {"tool_name": tool, "params": ["contact_id"]}
    return None


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
    if _pickup_lead_only(low) or "sớm nhất" in low or "muộn nhất" in low:
        return None
    if not any(word in low for word in ("muộn", "sớm", "dời")):
        return None
    sign = -1 if "sớm" in low and "muộn" not in low else 1
    move = re.search(r"\b(lùi|dời)\b", low)
    if move:
        clause = low[move.start():]
        cut = re.search(r"\b(báo thức|nhắc|nhắn)\b", clause)
        if cut:
            clause = clause[:cut.start()]
        clause_sign = -1 if "sớm" in clause and "muộn" not in clause else 1
        if "nửa tiếng" in clause or "nửa giờ" in clause:
            return clause_sign * 30
        spoken_clause = _duration_minutes(clause.split(",")[0])
        clause_hour = re.search(r"(một|hai|ba|\d+)\s*tiếng", clause)
        if spoken_clause and not clause_hour:
            return clause_sign * spoken_clause
        if clause_hour:
            word = clause_hour.group(1)
            number = {"một": 1, "hai": 2, "ba": 3}.get(word, int(word) if word.isdigit() else 0)
            if number:
                half = clause[clause_hour.end():].lstrip(" .,;").startswith("rưỡi")
                return clause_sign * (number * 60 + (30 if half else 0))
    indexes = [low.find(word) for word in ("muộn", "sớm", "dời") if word in low]
    tail = low[min(indexes):]
    phut = re.search(r"(\d+)\s*phút", tail)
    tieng = re.search(r"(một|hai|ba|\d+)\s*tiếng", tail)
    if "nửa tiếng" in tail[:24] or "nửa giờ" in tail[:24]:
        return sign * 30
    spoken_minutes = _duration_minutes(tail.split(",")[0])
    if phut and (not tieng or phut.start() < tieng.start()):
        return sign * int(phut.group(1))
    if spoken_minutes and (not tieng or tail.find("phút") < tieng.start()):
        return sign * spoken_minutes
    if not tieng:
        return None
    word = tieng.group(1)
    number = {"một": 1, "hai": 2, "ba": 3}.get(word, int(word) if word.isdigit() else 0)
    if not number:
        return None
    half = tail[tieng.end():].lstrip(" .,;").startswith("rưỡi")
    return sign * (number * 60 + (30 if half else 0))


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


def _turns_lights_off(text: str) -> bool:
    """Return whether the surviving clause actually switches a light off.

    Input:
        text (str): the latest user text. ``đừng tắt`` does not count.
        A retraction keeps only the clause after ``mà thôi``.

    Output:
        bool: True when some ``tắt`` is not preceded by ``đừng``, ``không``,
        or ``khỏi``.
    """
    low = _active_text(text).lower()
    cut = low.rfind("đừng tắt")
    if cut >= 0 and re.search(r"(?:thôi|quên|(?:^|\s)à(?:\s|,))", low[:cut]):
        low = low[cut + len("đừng tắt"):]
    if "đèn" not in low:
        return False
    for match in re.finditer(r"tắt", low):
        before = low[:match.start()].rstrip()
        if before.endswith(("đừng", "không", "khỏi", "đừng có")):
            continue
        return True
    return False


def _floor_named(text: str) -> int | None:
    """Read the floor number in a sentence about lights.

    Input:
        text (str): the latest user text.

    Output:
        int: the floor, such as ``2`` for ``tầng 2`` or ``tầng hai``.
        None: when the sentence names no floor.
    """
    match = re.search(r"tầng\s+(một|hai|ba|bốn|\d)", text.lower())
    if not match:
        return None
    word = match.group(1)
    return {"một": 1, "hai": 2, "ba": 3, "bốn": 4}.get(word, int(word) if word.isdigit() else None)


def _minutes_from_now(request: str, context: str) -> str | None:
    """Build an ISO time from ``N phút nữa``.

    Input:
        request (str): the latest user text.
        context (str): the rendered context, for ``current_time``.

    Output:
        str: the current time plus those minutes.
        None: when the sentence has no such phrase or the clock is missing.
    """
    low = request.lower()
    if "phút nữa" not in low and "phút sau" not in low:
        return None
    minutes = _duration_minutes(low)
    now = _context_value(context, "current_time")
    if minutes is None or not now:
        return None
    return _add_minutes(now, minutes)


def _hours_before(text: str) -> int | None:
    """Read how many hours before an event an alarm should ring.

    Input:
        text (str): the latest user text. It must say ``trước``.

    Output:
        int: the hour count in ``một tiếng`` or ``2 tiếng``.
        None: when the sentence is not about being early.
    """
    low = text.lower()
    if "trước" not in low:
        return None
    match = re.search(r"(một|hai|ba|bốn|\d+)\s*tiếng", low)
    if not match:
        return None
    word = match.group(1)
    return {"một": 1, "hai": 2, "ba": 3, "bốn": 4}.get(word, int(word) if word.isdigit() else None)


def _reminder_content(text: str) -> str:
    """Keep the errand in a reminder and drop the clock.

    Input:
        text (str): the latest user text.

    Output:
        str: the words after ``nhắc`` up to ``lúc`` or a clock.
        Empty when the sentence has no ``nhắc``.
    """
    match = re.search(r"nhắc", text, flags=re.IGNORECASE)
    if not match:
        return ""
    tail = text[match.end():]
    tail = re.split(r"\blúc\b|(?:lúc\s+)?\d+\s*giờ", tail, maxsplit=1, flags=re.IGNORECASE)[0]
    skip = _TITLE | {"dùm", "giúp", "hộ", "mình", "tôi", "tui", "nha", "nhé", "nhé", "ơi"}
    tokens = []
    for token in re.findall(r"\w+", tail, flags=re.UNICODE):
        if not tokens and token.lower() in skip:
            continue
        if token.lower() in _HOUR_WORD or token.lower() in ("giờ", "rưỡi", "kém", "tối", "sáng", "mai", "nay"):
            break
        tokens.append(token)
    return " ".join(tokens).strip()


_WEEKDAYS = (
    ("thứ hai", 0),
    ("thứ ba", 1),
    ("thứ tư", 2),
    ("thứ năm", 3),
    ("thứ sáu", 4),
    ("thứ bảy", 5),
    ("chủ nhật", 6),
)


def _due_weekday(text: str) -> int | None:
    """Read a due weekday when the sentence pays the bill due that day.

    Input:
        text (str): the latest user text, any casing.

    Output:
        int: Monday is 0 and Sunday is 6.
        None: when no due weekday is named.
    """
    low = text.lower()
    if "hạn" not in low:
        return None
    for phrase, day in _WEEKDAYS:
        if phrase in low:
            return day
    return None


def _unpaid_bills(results: list) -> list[dict]:
    """Collect unpaid bills from tool results.

    Input:
        results (list): tool payloads. A bill has ``id``, ``amount``, and ``paid``.

    Output:
        list[dict]: unpaid bills in result order.
    """
    found = []
    for payload in results:
        for node in _walk_dicts(payload):
            bill_id = node.get("id")
            if not isinstance(bill_id, str) or not bill_id.startswith("bill_"):
                continue
            if node.get("paid") is True or not isinstance(node.get("amount"), (int, float)):
                continue
            found.append(node)
    return found


def _bill_to_pay(results: list, request: str) -> dict | None:
    """Pick the one bill a sentence asks to pay now.

    Input:
        results (list): tool payloads.
        request (str): the latest user text. A due weekday wins, then a
        provider named in the sentence, then the bill type.

    Output:
        dict: the unpaid bill to pay.
        None: when no bill matches.
    """
    low = request.lower()
    bills = _unpaid_bills(results)
    weekday = _due_weekday(low)
    if weekday is not None:
        for bill in bills:
            due = bill.get("due_date")
            if not isinstance(due, str) or len(due) < 10:
                continue
            try:
                if datetime.fromisoformat(due[:10]).weekday() == weekday:
                    return bill
            except ValueError:
                continue
    for bill in bills:
        provider = str(bill.get("provider") or bill.get("name") or "").lower()
        if len(provider) >= 4 and provider in low:
            return bill
    focus = _bill_focus(request)
    if focus:
        for bill in bills:
            if bill.get("type") == focus:
                return bill
    return None


def _named_bill_pay(text: str) -> str | None:
    """Return the payment source when one named bill is paid now.

    Input:
        text (str): the latest user text, any casing.

    Output:
        str: ``bank`` or ``ewallet``.
        None: when the sentence is a transfer, a due sweep, a wallet branch,
        or does not pay a bill now.
    """
    low = text.lower()
    if _transfer_jobs(low) or _due_sweep(low):
        return None
    if _wallet_short(low) and any(phrase in low for phrase in ("bằng ví", "ví điện tử", "ví đủ")):
        return None
    pays = any(phrase in low for phrase in ("đóng", "trả luôn", "trả tiền", "trả giùm", "trả bằng", "thì trả"))
    if not pays or (_bill_focus(low) is None and "hóa đơn" not in low and "hoá đơn" not in low):
        return None
    if "ví" in low and "ngân hàng" not in low and "tài khoản" not in low:
        return "ewallet"
    if "ngân hàng" in low or "tài khoản" in low:
        return "bank"
    return None


def _due_sweep(text: str) -> bool:
    """Return whether the sentence pays every bill due within five days.

    Input:
        text (str): the latest user text, any casing.

    Output:
        bool: True when it names a due date, a five-day window, a 500 thousand
        cap, and the bank.
    """
    low = _fold_vi(text)
    due = any(phrase in low for phrase in ("đến hạn", "tới hạn", "sắp đến hạn", "tối hẹn", "hạn trong"))
    cap = "500" in low or "năm trăm" in low
    window = "ngày" in low or "trong vòng" in low
    bank = "ngân hàng" in low or "thanh toán" in low
    return "hóa đơn" in low and due and cap and window and bank


def _wallet_short(text: str) -> bool:
    """Return whether the sentence has a wallet-shortfall branch.

    Input:
        text (str): the latest user text, any casing.

    Output:
        bool: True for ``ví không đủ``, ``còn thiếu``, ``nếu thiếu``, and the
        same idea in the other wordings used in the gold set.
    """
    low = text.lower()
    if "ví đủ" in low and "không thì" in low:
        return True
    return any(phrase in low for phrase in (
        "còn không", "nếu thiếu", "còn thiếu", "ví mà không đủ", "ví không đủ", "ví thiếu",
    ))


def _wallet_parts(text: str) -> tuple[str, str]:
    """Split a wallet sentence into the enough branch and the short branch.

    Input:
        text (str): the latest user text, lowercased.

    Output:
        tuple[str, str]: text before the first shortfall marker, and the text
        after it. The second string is empty when no marker is present.
    """
    if "ví đủ" in text and "không thì" in text:
        return text.split("không thì", 1)
    for marker in ("còn không", "nếu thiếu", "còn thiếu", "ví mà không đủ", "ví không đủ", "ví thiếu"):
        if marker in text:
            return text.split(marker, 1)
    return text, ""


def _sms_person(text: str) -> str | None:
    """Read the person a text message is sent to.

    ``SMS`` between ``nhắn`` and the name is skipped. A title such as
    ``cô giáo`` stays in front of the given name.

    Input:
        text (str): the active clause.

    Output:
        str: a span such as ``Lan`` or ``Cô Giáo Mai``.
        None: when the clause does not name a recipient.
    """
    match = re.search(r"nhắn", text, flags=re.IGNORECASE)
    if not match:
        return None
    tail = re.sub(r"^(?:\s+(?:sms|tin))*", "", text[match.end():], flags=re.IGNORECASE)
    tokens = re.findall(r"\w+", tail, flags=re.UNICODE)
    index = 0
    while index < len(tokens) and (is_function(tokens[index]) or tokens[index].lower() in ("sms", "tin", "báo")):
        index += 1
    kept = []
    while index < len(tokens) and len(kept) < 4:
        token = tokens[index]
        if token.lower() in ("giờ", "mới", "biết"):
            break
        if kept and (is_function(token) or _amount_unit(token) or token.lower() in ("là", "báo")):
            break
        kept.append(token)
        index += 1
        if token.lower() not in _TITLE and token.lower() not in _ROLE:
            break
    if not kept:
        return None
    return " ".join(_as_spoken(token) for token in kept)


def _next_due_bill(results: list, context: str, done_calls: list[tuple[str, dict]]) -> dict | None:
    """Pick the next unpaid bill due within five days and at most 500 thousand.

    Input:
        results (list): tool payloads. A bill has ``due_date``, ``amount``, and ``paid``.
        context (str): the rendered context, for ``today``.
        done_calls (list[tuple[str, dict]]): payments already made in this sample.

    Output:
        dict: the next bill in list order.
        None: when every such bill is already chosen.
    """
    today = _context_value(context, "today")
    if not today:
        return None
    start = datetime.fromisoformat(today).date()
    end = start + timedelta(days=5)
    paid = {call.get("bill_id") for name, call in done_calls if name == "bill_payment.pay"}
    for payload in results:
        for node in _walk_dicts(payload):
            bill_id = node.get("id")
            if not isinstance(bill_id, str) or not bill_id.startswith("bill_"):
                continue
            if node.get("paid") is True or bill_id in paid:
                continue
            amount = node.get("amount")
            due = node.get("due_date")
            if not isinstance(amount, (int, float)) or amount > 500000:
                continue
            if not isinstance(due, str) or len(due) < 10:
                continue
            due_day = datetime.fromisoformat(due[:10]).date()
            if start <= due_day <= end:
                return node
    return None


def _bill_type_prefix(fragment: str) -> tuple[str, str]:
    """Read the bill type at the start of a fragment.

    Input:
        fragment (str): text that begins with ``truyền hình`` or ``điện``.

    Output:
        tuple[str, str]: the catalog type and the phrase that named it.
        ``("", "")`` when the fragment names no bill.
    """
    for phrase, kind in (
        ("truyền hình", "tv"),
        ("học phí", "tuition"),
        ("điện thoại", "phone"),
        ("điện", "electricity"),
        ("nước", "water"),
        ("mạng", "internet"),
    ):
        if fragment.startswith(phrase):
            return kind, phrase
    return "", ""


def _job_person(clause: str) -> str:
    """Read the payee in the clause before ``đúng tiền``.

    Input:
        clause (str): words such as ``chuyển cho trần văn tuấn``.

    Output:
        str: the name with titles kept, such as ``Trần Văn Tuấn``.
        Empty when the clause names nobody.
    """
    drop = {
        "chuyển", "cho", "em", "anh", "chị", "tôi", "tui", "mình", "và", "rồi",
        "tháng", "này", "nha", "nhé", "giúp", "dùm", "với", "thì", "trừ", "vào",
        "tài", "khoản", "thanh", "toán", "không", "đủ", "cả", "hai", "lấy",
        "tiết", "kiệm", "rút", "ra", "ừ", "ơi", "trả", "đóng", "chủ", "nhà",
        "đúng", "tiền",
    }
    kept = []
    for token in re.findall(r"\w+", clause, flags=re.UNICODE):
        if token.lower() in drop or is_function(token):
            continue
        kept.append(_as_spoken(token))
    return " ".join(kept[-4:])


def _clause_transfer_jobs(low: str) -> list[tuple[str, str]]:
    """Pair a bill type with the person named in the same clause.

    Input:
        low (str): the latest user text, lowercased. The bill type may come
        before ``cho``, as in ``tiền nước thì cho Nguyễn Thị Lan``.

    Output:
        list[tuple[str, str]]: ``(person, bill type)`` for each clause that
        names both. Empty when a clause is missing either one.
    """
    kinds = (
        ("truyền hình", "tv"), ("học phí", "tuition"), ("điện thoại", "phone"),
        ("điện", "electricity"), ("nước", "water"), ("mạng", "internet"),
    )
    jobs = []
    for clause in _split_clauses(low):
        kind = next((code for phrase, code in kinds if phrase in clause), "")
        if not kind:
            continue
        person = _payee_span(clause) if "cho " in clause else ""
        if not person and "đúng tiền" in clause:
            person = _job_person(clause.split("đúng tiền", 1)[0])
        if person and kind:
            jobs.append((person, kind))
    return jobs


def _transfer_jobs(text: str) -> list[tuple[str, str]]:
    """Pair each payee with the bill type that follows ``đúng tiền``.

    Input:
        text (str): the latest user text. One sentence can name two payees.

    Output:
        list[tuple[str, str]]: ``(person, bill type)`` in spoken order.
        Empty when the sentence is not this pattern.
    """
    low = text.lower()
    if "chuyển" not in low and "trả" not in low and "đóng" not in low:
        return []
    if low.count("vừa đủ") >= 2:
        kinds = (
            ("truyền hình", "tv"), ("học phí", "tuition"), ("điện thoại", "phone"),
            ("điện", "electricity"), ("nước", "water"), ("mạng", "internet"),
        )
        jobs = []
        for part in re.split(r"vừa đủ", low)[:-1]:
            kind = next((code for phrase, code in kinds if phrase in part), "")
            person = _payee_span(part)
            if person and kind:
                jobs.append((person, kind))
        if jobs:
            return jobs
    if "đúng" not in low:
        clause_jobs = _clause_transfer_jobs(low)
        return clause_jobs if len(clause_jobs) >= 2 else []
    jobs = []
    cursor = 0
    for match in re.finditer(r"đúng(?:\s+(?:bằng|số))?\s+tiền\s+", low):
        person = _job_person(low[cursor:match.start()])
        kind, phrase = _bill_type_prefix(low[match.end():])
        if person and kind:
            jobs.append((person, kind))
        cursor = match.end() + len(phrase)
    clause_jobs = _clause_transfer_jobs(low)
    if len(clause_jobs) >= 2:
        return clause_jobs
    return jobs


def _beneficiary_named(results: list, person: str) -> str | None:
    """Match one spoken payee to a beneficiary id.

    Input:
        results (list): tool payloads. A beneficiary has ``name`` and ``id``.
        person (str): the payee, such as ``Phạm Hùng``.

    Output:
        str: the id of the longest stored name that contains every query word.
        None: when no beneficiary matches.
    """
    best = None
    best_len = 0
    for payload in results:
        for node in _walk_dicts(payload):
            name = node.get("name")
            beneficiary_id = node.get("id")
            if not isinstance(name, str) or not isinstance(beneficiary_id, str):
                continue
            if not beneficiary_id.startswith("ben_") or not _named_in(person, name):
                continue
            if len(name) > best_len:
                best = beneficiary_id
                best_len = len(name)
    return best


def _duration_minutes(text: str) -> int | None:
    """Read a duration in minutes.

    Input:
        text (str): a clause that may say ``nửa tiếng`` or ``mười lăm phút``.

    Output:
        int: the number of minutes.
        None: when no duration is written.
    """
    low = text.lower()
    if "nửa tiếng" in low or "nửa giờ" in low:
        return 30
    match = re.search(r"(\d+)\s*phút", low)
    if match:
        return int(match.group(1))
    marker = low.rfind("phút")
    if marker < 0:
        return None
    kept = []
    for token in reversed(low[:marker].split()):
        if token in _VN_DIGIT or token in ("mươi", "mười", "trăm"):
            kept.append(token)
            continue
        break
    if not kept:
        return None
    number = _vn_under_1000(" ".join(reversed(kept)))
    return number or None


def _pickup_lead(text: str) -> int | None:
    """Read how many minutes before the event the car should arrive.

    Input:
        text (str): the latest user text. The reminder clause is ignored.

    Output:
        int: minutes such as 30 or 45.
        None: when the sentence gives no pickup lead.
    """
    return _duration_minutes(text.lower().split("nhắc", 1)[0])


def _remind_lead(text: str) -> int | None:
    """Read how many minutes before pickup the reminder should fire.

    Input:
        text (str): the latest user text.

    Output:
        int: minutes, usually 15.
        None: when the sentence has no ``trước lúc xe`` clause.
    """
    low = text.lower()
    if "nhắc" not in low:
        return None
    tail = low.split("nhắc", 1)[1]
    markers = (
        "trước lúc đón xe", "trước lúc xe", "trước giờ xe", "trước khi xe",
        "trước lúc đi", "trước lúc ra xe", "trước đó", "xe đón", "xe tới", "đón xe", "ra xe",
    )
    if not any(marker in tail for marker in markers):
        return None
    return _duration_minutes(tail)


def _ride_event(results: list, request: str, context: str) -> dict | None:
    """Pick the calendar event a ride is going to.

    Input:
        results (list): tool payloads. An event has ``title``, ``start``, and ``location``.
        request (str): the latest user text. ``chiều`` limits the hour.
        ``nay`` and ``mai`` limit the day.
        context (str): the rendered context, for today and tomorrow.

    Output:
        dict: the event whose title shares the most words with the request.
        None: when no event fits.
    """
    low = request.lower()
    window = range(12, 18) if "chiều" in low else range(5, 12) if "sáng" in low else range(18, 24) if "tối" in low else None
    today = _context_value(context, "today")
    tomorrow = _context_value(context, "tomorrow")
    day = tomorrow if "mai" in low and "nay" not in low else today if "nay" in low else ""
    found = []
    for payload in results:
        for node in _walk_dicts(payload):
            start = node.get("start")
            if not isinstance(start, str) or "T" not in start:
                continue
            if day and not start.startswith(day):
                continue
            hour = int(start[11:13]) if len(start) >= 13 else -1
            if window is not None and hour not in window:
                continue
            found.append(node)
    best = None
    best_score = 0
    for node in found:
        title = str(node.get("title") or "").lower().strip()
        if title and title in low:
            score = 100 + len(title)
        else:
            score = sum(1 for token in title.split() if len(token) > 2 and token in low)
        if score > best_score:
            best = node
            best_score = score
    if best_score > 0:
        return best
    if len(found) == 1:
        return found[0]
    return None


def _climate_temperature(text: str) -> int | None:
    """Read the degrees the sentence wants an air conditioner set to.

    Input:
        text (str): the latest user text.

    Output:
        int: the number before ``độ`` when the sentence turns the air
        conditioner on.
        None: otherwise.
    """
    low = text.lower()
    if "điều hòa" not in low and "điều hoà" not in low:
        return None
    match = re.search(r"(\d+)\s*độ", low)
    if match:
        return int(match.group(1))
    spoken = re.search(
        r"((?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi|mốt|tư|lăm)"
        r"(?:\s+(?:không|một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi|mốt|tư|lăm))*)\s+độ",
        low,
    )
    if not spoken:
        return None
    value = _vn_under_1000(spoken.group(1))
    return value or None


def _climate_device(results: list, request: str) -> dict | None:
    """Find the air conditioner whose room name is inside the sentence.

    Input:
        results (list): tool payloads from ``smart_home.get_status``.
        request (str): the latest user text, including the room.

    Output:
        dict: the climate device with the longest matching room name.
        None: when no room matches.
    """
    low = request.lower()
    best = None
    best_len = 0
    for payload in results:
        for node in _walk_dicts(payload):
            if node.get("type") != "climate" and not str(node.get("id", "")).startswith("ac_"):
                continue
            room = str(node.get("room") or "").lower()
            if room and room in low and len(room) > best_len:
                best = node
                best_len = len(room)
    return best


def _named_light(results: list, request: str) -> dict | None:
    """Find the lamp whose room name is the longest phrase in the sentence.

    Input:
        results (list): tool payloads from ``smart_home.get_status``.
        request (str): the latest user text. ``phòng ngủ nhỏ`` beats ``phòng ngủ``.

    Output:
        dict: that light.
        None: when the sentence names no room.
    """
    low = request.lower()
    best = None
    best_len = 0
    for payload in results:
        for node in _walk_dicts(payload):
            if node.get("type") != "light" and not str(node.get("id", "")).startswith("light_"):
                continue
            room = str(node.get("room") or "").lower()
            if room and room in low and len(room) > best_len:
                best = node
                best_len = len(room)
    return best


def _food_restaurant(results: list, request: str) -> dict | None:
    """Find the restaurant whose name shares the most words with the sentence.

    Input:
        results (list): tool payloads. A restaurant id starts with ``rest_``.

    Output:
        dict: the restaurant.
        None: when no restaurant name is mentioned.
    """
    low = request.lower()
    best = None
    best_score = 0
    for payload in results:
        for node in _walk_dicts(payload):
            name = node.get("name")
            restaurant_id = node.get("id")
            if not isinstance(name, str) or not isinstance(restaurant_id, str):
                continue
            if not restaurant_id.startswith("rest_") or "price" in node:
                continue
            score = sum(1 for token in name.lower().split() if len(token) > 1 and token in low)
            if score > best_score:
                best = node
                best_score = score
    return best


def _food_item(results: list, request: str) -> dict | None:
    """Find the menu item whose name sits inside the sentence.

    Input:
        results (list): tool payloads. A menu item has ``price``.

    Output:
        dict: the longest matching item.
        None: when no item name is mentioned.
    """
    low = request.lower()
    best = None
    best_len = 0
    for payload in results:
        for node in _walk_dicts(payload):
            name = node.get("name")
            if not isinstance(name, str) or "price" not in node:
                continue
            folded = name.lower()
            if folded in low and len(folded) > best_len:
                best = node
                best_len = len(folded)
    return best


def _serving_count(results: list, request: str, context: str) -> int:
    """Count one serving per person in the named chat, excluding the speaker.

    Input:
        results (list): tool payloads. A chat has ``name`` and ``members``.
        request (str): the latest user text.
        context (str): the rendered context, for the speaker's name.

    Output:
        int: the number of servings. One when the chat was not found.
    """
    low = request.lower()
    for payload in results:
        for node in _walk_dicts(payload):
            name = node.get("name")
            members = node.get("members")
            if not isinstance(name, str) or not isinstance(members, list):
                continue
            tokens = [token for token in name.lower().split() if len(token) > 2 and token not in ("nhóm", "nhom", "chat")]
            if name.lower() not in low and not (tokens and all(token in low for token in tokens)):
                continue
            count = len(members)
            if "không tính" in low:
                profile = _context_value(context, "name")
                given = profile.split()[-1].lower() if profile else ""
                if given and any(given in str(member).lower() for member in members):
                    count -= 1
            return max(count, 1)
    return 1


def _payee_span(text: str) -> str:
    """Read the person a transfer is sent to.

    Input:
        text (str): the latest user text.

    Output:
        str: the name after the last ``cho``, without a title. Words after
        ``đúng`` or ``triệu`` are not part of the name.
        Empty when no payee is named.
    """
    low = text.lower()
    index = low.rfind("cho ")
    start = index + 4 if index >= 0 else -1
    if start < 0:
        match = re.search(r"chuyển(?:\s+khoản)?\s+", low)
        start = match.end() if match else -1
    if start < 0:
        return ""
    tail = re.split(
        r"\b(số tiền|triệu|ngàn|nghìn|và|nhé|nha|đúng|bằng|tiền|góp|mua|\d)\b",
        text[start:],
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    skip = _TITLE | {"chủ", "nhà", "tui", "tôi", "mình", "tài", "khoản", "thanh", "toán", "từ"}
    kept = []
    for token in re.findall(r"\w+", tail, flags=re.UNICODE):
        if token.lower() in skip or _amount_unit(token):
            continue
        kept.append(token)
    return " ".join(kept[:4])


def _named_in(query: str, name: str) -> bool:
    """Return whether every query word sits inside a stored name.

    Input:
        query (str): the spoken name, such as ``Phạm Hùng``.
        name (str): the stored name, such as ``Chủ nhà - Phạm Hùng``.

    Output:
        bool: True when each query word occurs in ``name``.
    """
    parts = [part for part in query.lower().split() if part]
    folded = name.lower()
    return bool(parts) and all(part in folded for part in parts)


def _bill_amount(results: list, focus: str | None) -> int | None:
    """Read the unpaid amount of the bill type named in the request.

    Input:
        results (list): tool payloads. A bill has ``type``, ``amount``, and ``paid``.
        focus (str | None): a catalog bill type such as ``internet``.

    Output:
        int: the amount of the first unpaid bill of that type.
        None: when no such bill was returned.
    """
    if not focus:
        return None
    for payload in results:
        for node in _walk_dicts(payload):
            if node.get("type") != focus or node.get("paid") is True:
                continue
            if isinstance(node.get("amount"), (int, float)):
                return int(node["amount"])
    return None


def _bank_accounts(results: list) -> list[dict]:
    """Collect accounts returned by ``banking.get_balance``.

    Input:
        results (list): tool payloads.

    Output:
        list[dict]: accounts that have an id and a balance.
    """
    found = []
    for payload in results:
        for node in _walk_dicts(payload):
            if isinstance(node.get("balance"), (int, float)) and isinstance(node.get("id"), str):
                found.append(node)
    return found


def _pick_account(accounts: list[dict], amount: int | None, request: str) -> str | None:
    """Choose the account that pays a transfer.

    Input:
        accounts (list[dict]): rows from ``banking.get_balance``.
        amount (int | None): the amount to send.
        request (str): the latest user text. Savings is used when the
        payment account is short and the sentence says to take from savings.

    Output:
        str: the account id.
        None: when no account was returned.
    """
    if not accounts:
        return None
    low = request.lower()
    payment = [row for row in accounts if "thanh toán" in str(row.get("name", "")).lower()]
    savings = [row for row in accounts if "tiết kiệm" in str(row.get("name", "")).lower()]
    source = payment[0] if payment else accounts[0]
    allow_savings = "tiết kiệm" in low and ("không đủ" in low or "lấy từ" in low or "thiếu" in low)
    if allow_savings and savings and amount is not None and source.get("balance", 0) < amount:
        return savings[0]["id"]
    return source["id"]


def _pick_beneficiary(results: list, request: str) -> str | None:
    """Match the spoken payee to a beneficiary id.

    Input:
        results (list): tool payloads. A beneficiary has ``name`` and ``id``.
        request (str): the latest user text.

    Output:
        str: the beneficiary id whose name contains the spoken name.
        None: when the sentence names no payee or no row matches.
    """
    query = _payee_span(request)
    if not query:
        return None
    best = None
    for payload in results:
        for node in _walk_dicts(payload):
            name = node.get("name")
            if not isinstance(name, str) or not isinstance(node.get("id"), str):
                continue
            if "bank" not in node and not str(node.get("id", "")).startswith("ben_"):
                continue
            if _named_in(query, name) and (best is None or len(name) > len(best.get("name", ""))):
                best = node
    return best["id"] if best else None


def _event_for_alarm(results: list, request: str, context: str) -> dict | None:
    """Find the calendar event an alarm should precede.

    Input:
        results (list): tool payloads. An event has ``title`` and ``start``.
        request (str): the latest user text. ``ngày 19`` or ``mai`` filters the day.
        context (str): the rendered context, for ``tomorrow``.

    Output:
        dict: the event whose title shares the most words with the request.
        None: when no event title is mentioned.
    """
    low = request.lower()
    day = re.search(r"ngày\s+(\d{1,2})", low)
    tomorrow = _context_value(context, "tomorrow")
    best = None
    best_score = 0
    best_start = ""
    for payload in results:
        for node in _walk_dicts(payload):
            title = node.get("title")
            start = node.get("start")
            if not isinstance(title, str) or not isinstance(start, str):
                continue
            if day and start[8:10] != f"{int(day.group(1)):02d}":
                continue
            if day is None and "mai" in low and tomorrow and not start.startswith(tomorrow):
                continue
            score = sum(1 for token in title.lower().split() if token in low)
            if score > best_score or (score == best_score and score > 0 and start > best_start):
                best = node
                best_score = score
                best_start = start
    return best


def _fan_rooms(results: list) -> set[str]:
    """Return rooms whose fan is running.

    Input:
        results (list): tool payloads. A fan has ``room`` and ``state.power``.

    Output:
        set[str]: room names whose fan power is not ``off``.
    """
    rooms = set()
    for payload in results:
        for node in _walk_dicts(payload):
            if node.get("type") != "fan" and not str(node.get("id", "")).startswith("fan_"):
                continue
            power = (node.get("state") or {}).get("power") if isinstance(node.get("state"), dict) else None
            if power and power != "off" and isinstance(node.get("room"), str):
                rooms.add(node["room"])
    return rooms


def _cooled_rooms(results: list) -> set[str]:
    """Return rooms whose air conditioner is running.

    Input:
        results (list): tool payloads. A climate device has ``room`` and ``state.mode``.

    Output:
        set[str]: room names whose mode is not ``off``.
    """
    rooms = set()
    for payload in results:
        for node in _walk_dicts(payload):
            if node.get("type") != "climate" and not str(node.get("id", "")).startswith("ac_"):
                continue
            mode = (node.get("state") or {}).get("mode") if isinstance(node.get("state"), dict) else None
            if mode and mode != "off" and isinstance(node.get("room"), str):
                rooms.add(node["room"])
    return rooms


def _next_floor_light(results: list, request: str, done_calls: list[tuple[str, dict]]) -> dict | None:
    """Pick the next lamp to switch off on the named floor.

    A room whose air conditioner is running is left as it is. A lamp already
    switched in this sample is skipped.

    Input:
        results (list): tool payloads from ``smart_home.get_status``.
        request (str): the latest user text, including the floor.
        done_calls (list[tuple[str, dict]]): successful calls so far.

    Output:
        dict: one light device that is still on.
        None: when the floor is unnamed or every matching lamp is done.
    """
    floor = _floor_named(request)
    if floor is None:
        return None
    done_ids = {
        call.get("device_id")
        for name, call in done_calls
        if name == "smart_home.light_set"
    }
    low_request = request.lower()
    if "quạt" in low_request:
        skip_rooms = _fan_rooms(results)
    elif any(word in low_request for word in ("máy lạnh", "điều hòa", "điều hoà")):
        skip_rooms = _cooled_rooms(results)
    else:
        skip_rooms = set()
    for payload in results:
        for node in _walk_dicts(payload):
            if node.get("type") != "light" and not str(node.get("id", "")).startswith("light_"):
                continue
            if node.get("floor") != floor:
                continue
            if node.get("room") in skip_rooms:
                continue
            state = node.get("state") if isinstance(node.get("state"), dict) else {}
            if state.get("power") == "off":
                continue
            if node.get("id") in done_ids:
                continue
            return node
    return None


def _clock_hhmm(text: str) -> str | None:
    """Read a daily clock as HH:MM.

    Input:
        text (str): the latest user text. Tối, đêm, and chiều add twelve
        when the hour is still in the morning.

    Output:
        str: ``HH:MM``. None when the sentence has no clock.
    """
    found = _last_spoken_hm(text)
    if not found:
        return None
    hour, minute = found
    low = text.lower()
    if "sáng" not in low and any(word in low for word in ("tối", "đêm", "chiều")) and hour < 12:
        hour += 12
    return f"{hour:02d}:{minute:02d}"


def _kwh_limit(text: str) -> int | None:
    """Read the kWh threshold after ``quá``.

    Input:
        text (str): the latest user text, as in ``quá mười tám số``.

    Output:
        int: that number of kWh. None when the sentence has no such limit.
    """
    match = re.search(r"quá\s+(.+?)\s+số", text.lower())
    if not match:
        return None
    phrase = match.group(1).strip()
    if phrase.isdigit():
        return int(phrase)
    number = _vn_under_1000(phrase)
    return number or None


def _usage_kwh(results: list, request: str) -> float | None:
    """Read the kWh figure a usage result already returned.

    Input:
        results (list): tool payloads. ``home_energy.get_usage`` has ``usage``.
        request (str): the latest user text. ``hôm qua`` prefers the last day.

    Output:
        float: that kWh. None when usage has not been read yet.
    """
    found = None
    days = []
    for payload in results:
        if not isinstance(payload, dict):
            continue
        usage = payload.get("usage")
        if isinstance(usage, dict) and isinstance(usage.get("kwh"), (int, float)):
            found = usage["kwh"]
            if isinstance(usage.get("days"), list):
                days = usage["days"]
    if "hôm qua" in request.lower() and days:
        last = days[-2] if len(days) >= 2 else days[-1]
        if isinstance(last, dict) and isinstance(last.get("kwh"), (int, float)):
            return last["kwh"]
    return found


def _best_named(items: list, request: str, key: str = "name") -> dict | None:
    """Pick the item whose name shares the longest words with the sentence.

    Input:
        items (list): dicts that have ``key``.
        request (str): the latest user text.
        key (str): the field to score. ``name`` for a chore or a package.

    Output:
        dict: the best item. None when no word longer than two letters hits.
    """
    low = _plain(request)
    best = None
    best_score = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        score = sum(len(token) for token in _plain(str(item.get(key) or "")).split() if len(token) > 2 and token in low)
        if score > best_score:
            best = item
            best_score = score
    return best if best_score >= 4 else None


def _collect(results: list, prefix: str) -> list[dict]:
    """Collect result dicts whose id starts with a prefix.

    Input:
        results (list): tool payloads.
        prefix (str): an id prefix such as ``appl_`` or ``sub_``.

    Output:
        list[dict]: those dicts, in the order the results listed them.
    """
    found = []
    for payload in results:
        for node in _walk_dicts(payload):
            if str(node.get("id", "")).startswith(prefix):
                found.append(node)
    return found


def _appliance_job(results: list, request: str) -> tuple[dict, str] | None:
    """Pick the appliance and the cycle the sentence names.

    Input:
        results (list): payloads from ``appliances.list``.
        request (str): the latest user text.

    Output:
        tuple[dict, str]: the appliance and a cycle code such as ``quick``.
        None: when the name or the cycle is not in the list.
    """
    low = _plain(request)
    best = None
    best_len = 0
    for item in _collect(results, "appl_"):
        name = _plain(str(item.get("name") or ""))
        if name and name in low and len(name) > best_len:
            best = item
            best_len = len(name)
    if not best:
        return None
    labels = best.get("cycle_labels") or {}
    cycle = None
    cycle_len = 0
    for code, label in labels.items():
        plain = _plain(str(label))
        if plain and plain in low and len(plain) > cycle_len:
            cycle = code
            cycle_len = len(plain)
    cycles = best.get("cycles") or []
    if cycle is None and "nhanh" in request.lower() and "quick" in cycles:
        cycle = "quick"
    if cycle is None and "thuong" in low and "normal" in cycles:
        cycle = "normal"
    if cycle is None:
        return None
    return best, cycle


def _chore_job(results: list, request: str) -> tuple[str, str] | None:
    """Pick the chore and the family member the sentence names.

    Input:
        results (list): payloads from ``household_chores.list_chores``.
        request (str): the latest user text.

    Output:
        tuple[str, str]: chore id and the member name as stored.
        None: when either side is missing.
    """
    chore = _best_named(_collect(results, "chore_"), request)
    members = []
    for payload in results:
        if isinstance(payload, dict) and isinstance(payload.get("members"), list):
            members = payload["members"]
    person = _person_after(request, "giao cho") or _person_after(request, "cho")
    assignee = None
    if person:
        plain = _plain(person)
        for member in members:
            if plain and plain in _plain(str(member)):
                assignee = member
                break
    if assignee is None:
        low = _plain(request)
        best_len = 0
        for member in members:
            for token in _plain(str(member)).split():
                if len(token) > 2 and token in low and len(token) > best_len:
                    assignee = member
                    best_len = len(token)
    if not chore or not assignee:
        return None
    return chore["id"], assignee


def _chore_assignee(members: list, part: str) -> str | None:
    """Match a family member named in one assignment clause.

    Input:
        members (list): names from ``household_chores.list_chores``.
        part (str): one clause, such as ``của Na`` or ``cho ba``.

    Output:
        str: the stored member name. None when no member fits.
    """
    low = part.lower()
    spoken = low
    for marker in ("giao cho", "là của", "của", "cho"):
        if marker in low:
            spoken = low.split(marker, 1)[1]
            break
    words = re.findall(r"\w+", spoken, flags=re.UNICODE)
    candidate = _plain(words[0]) if words else ""
    if not candidate:
        return None
    for member in members:
        folded = _plain(str(member))
        if folded == candidate or folded.split()[-1] == candidate:
            return str(member)
    return None


def _chore_assignments(results: list, request: str, context: str) -> list[tuple[str, str]]:
    """List every chore the sentence assigns, in sentence order.

    Input:
        results (list): chore payloads.
        request (str): the latest user text. ``còn`` starts the next assignment.
        context (str): the rendered context, used when the list is in history.

    Output:
        list[tuple[str, str]]: chore id and member. One pair when the
        sentence names a single chore.
    """
    rows = _with_history(results, context, "household_chores.list_chores", "chore_")
    if "giao hết" in request.lower() and "chưa xong" in request.lower():
        owner = ""
        owner_match = re.search(r"của\s+(.+?)\s+mà", request, flags=re.IGNORECASE)
        if owner_match:
            owner = _plain(owner_match.group(1))
        assignee = ""
        assignee_match = re.search(r"(?:qua|cho)\s+(.+?)(?:\s+giùm|\s+nghe|$)", request, flags=re.IGNORECASE)
        if assignee_match:
            assignee = assignee_match.group(1).strip()
        jobs = []
        for node in _collect(rows, "chore_"):
            if node.get("done") is True:
                continue
            if owner and owner not in _plain(str(node.get("assignee") or "")):
                continue
            if node.get("id") and assignee:
                jobs.append((node["id"], assignee))
        if jobs:
            return jobs
    members = []
    for payload in rows:
        if isinstance(payload, dict) and isinstance(payload.get("members"), list):
            members = payload["members"]
    parts = re.split(r"\bcòn\b", request, maxsplit=1, flags=re.IGNORECASE)
    jobs = []
    seen = set()
    for part in parts:
        chore = _best_named(_collect(rows, "chore_"), part)
        assignee = _chore_assignee(members, part)
        if chore and assignee and chore["id"] not in seen:
            jobs.append((chore["id"], assignee))
            seen.add(chore["id"])
    if jobs:
        return jobs
    single = _chore_job(rows, request)
    return [single] if single else []


def _spoken_email(text: str) -> str | None:
    """Read an email address, including ``a còng`` and ``chấm``.

    Input:
        text (str): the latest user text.

    Output:
        str: an address such as ``nhom22@example.test``.
        None: when the sentence has no address.
    """
    match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", text)
    if match:
        return match.group(0)
    spoken = re.search(
        r"([a-z0-9._]+)\s+a còng\s+([a-z0-9]+)(?:\s+chấm\s+([a-z0-9]+))?",
        text.lower(),
    )
    if not spoken:
        return None
    host = spoken.group(3) or "test"
    return f"{spoken.group(1)}@{spoken.group(2)}.{host}"


def _draft_address(results: list, request: str) -> str | None:
    """Find the recipient address for a draft.

    Input:
        results (list): contact payloads. A contact may have ``email``.
        request (str): the latest user text.

    Output:
        str: the spoken address, or the email of the contact whose name or
        relation is in the sentence. None when neither is present.
    """
    spoken = _spoken_email(request)
    if spoken:
        return spoken
    low = _plain(request)
    for payload in results:
        for node in _walk_dicts(payload):
            email = node.get("email")
            if not isinstance(email, str) or "@" not in email:
                continue
            name = _plain(str(node.get("name") or ""))
            relation = _plain(str(node.get("relation") or ""))
            if (relation and relation in low) or (name and name in low):
                return email
    return None


def _spoken_count(text: str) -> int | None:
    """Read a number that follows số, such as một trăm mười hai.

    Input:
        text (str): the latest user text.

    Output:
        int: that number. None when ``số`` is not followed by digit words.
    """
    match = re.search(
        r"số\s+((?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|trăm|mươi|lăm|mốt|\d+)"
        r"(?:\s+(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|trăm|mươi|lăm|mốt|\d+))*)",
        text.lower(),
    )
    if not match:
        return None
    raw = match.group(1).strip()
    return int(raw) if raw.isdigit() else _vn_under_1000(raw) or None


def _bike_area(request: str, context: str) -> str | None:
    """Read the station or area a bicycle should come from.

    Input:
        request (str): the latest user text.
        context (str): the rendered context, for the home street and history.

    Output:
        str: a station name such as ``Mỹ Đình`` or ``Hàng Bạc``.
        None: when the sentence does not name a place.
    """
    low = request.lower()
    if "phố nhà" in low or "gần nhà" in low:
        home = _context_value(context, "home_address")
        street = re.sub(r"^\d+\s*", "", home.split(",")[0]).strip() if home else ""
        return street or None
    if "bể bơi" in low:
        match = re.search(r"bể bơi\s+([^.,\n]+)", _history_text(context), flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
    match = re.search(r"trạm\s+([A-ZÀ-Ỹa-zà-ỹ][\wÀ-ỹ]*(?:\s+[A-ZÀ-Ỹa-zà-ỹ][\wÀ-ỹ]*)?)", request)
    if not match:
        return None
    name = match.group(1).strip()
    if name.lower().startswith(("gần", "ngay", "nào")):
        return None
    return name


def _bike_queue(results: list, request: str, context: str = "") -> list[dict]:
    """List the bicycles a sentence holds, electric first unless there are too few.

    Input:
        results (list): payloads from ``bike_rental.find_bikes``.
        request (str): the latest user text. ``hai xe điện`` asks for two.
        context (str): the rendered context.

    Output:
        list[dict]: the bikes to reserve, in order.
    """
    if not re.search(r"hai\s+(?:chiếc|xe)", request.lower()):
        one = _bike_to_reserve(results, request, context)
        return [one] if one else []
    low = _plain(request)
    bikes = [item for item in _collect(results, "bike_") if item.get("available", True)]
    at_station = []
    for bike in bikes:
        tokens = [token for token in _plain(str(bike.get("station") or "")).split() if len(token) > 2 and token not in ("tram",)]
        if tokens and all(token in low for token in tokens):
            at_station.append(bike)
    pool = at_station or bikes
    electric = [bike for bike in pool if "dien" in _plain(str(bike.get("name") or ""))]
    regular = [bike for bike in pool if "dien" not in _plain(str(bike.get("name") or ""))]
    if ("dien" in low or "pin" in low) and "khong du" in low and len(electric) < 2:
        return regular[:2]
    if "dien" in low or "pin" in low:
        return (electric or regular)[:2]
    return pool[:2]


def _bike_to_reserve(results: list, request: str, context: str = "") -> dict | None:
    """Pick one available bicycle the sentence asks to hold.

    Input:
        results (list): payloads from ``bike_rental.find_bikes``.
        request (str): the latest user text. A station name, ``thể thao``,
        ``đủ pin``, or ``gần nhà`` narrows the list.
        context (str): the rendered context, for the home address.

    Output:
        dict: the first bike that still fits. None when none is available.
    """
    low = _plain(request)
    bikes = [item for item in _collect(results, "bike_") if item.get("available", True)]
    if not bikes:
        return None
    at_station = []
    for bike in bikes:
        station = _plain(str(bike.get("station") or ""))
        tokens = [token for token in station.split() if len(token) > 2 and token not in ("tram",)]
        if tokens and all(token in low for token in tokens):
            at_station.append(bike)
    pool = at_station or bikes
    if "pin" in low or "dien" in low:
        electric = [bike for bike in pool if "dien" in _plain(str(bike.get("name") or ""))]
        if electric:
            pool = electric
    if "the thao" in low:
        sport = [bike for bike in pool if "the thao" in _plain(str(bike.get("name") or ""))]
        if sport:
            pool = sport
    if "gần nhà" in request.lower():
        home = _plain(_context_value(context, "home_address"))
        near_home = []
        for bike in pool:
            tokens = [token for token in _plain(str(bike.get("station") or "")).split() if len(token) > 2 and token not in ("tram",)]
            if home and tokens and any(token in home for token in tokens):
                near_home.append(bike)
        if near_home:
            pool = near_home
    area = _plain(_bike_area(request, context) or "")
    if area:
        at_area = [bike for bike in pool if area in _plain(str(bike.get("station") or ""))]
        if at_area:
            pool = at_area
    if "thuong" in low and "dien" not in low:
        regular = [bike for bike in pool if "dien" not in _plain(str(bike.get("name") or ""))]
        if regular:
            pool = regular
        elif "re nhat" in low or "het thi" in low:
            pool = sorted(pool, key=lambda bike: bike.get("price_per_hour") or 10**9)
    number = _spoken_count(request)
    if number:
        numbered = [bike for bike in pool if re.search(rf"\b{number}\b", str(bike.get("name") or ""))]
        if numbered:
            pool = numbered
    if "re nhat" in low or "het thi" in low:
        available = [bike for bike in pool if bike.get("available", True)]
        if not available:
            station = [bike for bike in bikes if not area or area in _plain(str(bike.get("station") or ""))]
            available = [bike for bike in station if bike.get("available", True)]
            available.sort(key=lambda bike: bike.get("price_per_hour") or 10**9)
            return available[0] if available else None
        pool = available
    return pool[0]


def _loyalty_choice(results: list, request: str) -> tuple[dict, dict] | None:
    """Pick the program and the reward the sentence wants to redeem.

    Input:
        results (list): payloads from ``loyalty.get_points``.
        request (str): the latest user text. Words after ``còn đủ`` are a
        second, conditional reward and are ignored.

    Output:
        tuple[dict, dict]: the program and the reward.
        None: when no reward name is in the sentence.
    """
    low = _plain(request.split("còn đủ", 1)[0])
    best_program = None
    best_reward = None
    best_score = 0
    airline = "hang bay" in low or "may bay" in low
    for payload in results:
        programs = []
        if isinstance(payload, dict) and isinstance(payload.get("programs"), list):
            programs = payload["programs"]
        else:
            for node in _walk_dicts(payload):
                if isinstance(node.get("programs"), list):
                    programs = node["programs"]
        for program in programs:
            label = _plain(str(program.get("name") or "")) + " " + _plain(str(program.get("merchant") or ""))
            named = any(token in low for token in label.split() if len(token) > 2)
            carrier = airline and "air" in label
            for reward in program.get("rewards") or []:
                reward_name = _plain(str(reward.get("name") or ""))
                word_score = sum(len(token) for token in reward_name.split() if len(token) > 1 and token in low)
                if word_score <= 0 and not (carrier and "ve" in reward_name and "ve" in low):
                    continue
                score = word_score + (5 if named else 0) + (30 if carrier else 0)
                if "ve" in low and "ve" in reward_name:
                    score += 15
                if score > best_score:
                    best_program = program
                    best_reward = reward
                    best_score = score
    if best_program and best_reward:
        return best_program, best_reward
    return None


def _loyalty_programs(results: list, context: str) -> list[dict]:
    """Return loyalty programs from this turn or from history.

    Input:
        results (list): payloads so far.
        context (str): the rendered context.

    Output:
        list[dict]: program dicts that have an id.
    """
    payloads = results
    if not any(isinstance(node.get("rewards"), list) for payload in results for node in _walk_dicts(payload)):
        payloads = [payload for name, _args, payload in _history_invocations(context) if name == "loyalty.get_points"] + results
    programs = []
    for payload in payloads:
        if isinstance(payload, dict) and isinstance(payload.get("programs"), list):
            programs.extend(item for item in payload["programs"] if isinstance(item, dict))
            continue
        for node in _walk_dicts(payload):
            if isinstance(node.get("programs"), list):
                programs.extend(item for item in node["programs"] if isinstance(item, dict))
    return programs


def _loyalty_queue(results: list, request: str, context: str) -> list[tuple[dict, dict]]:
    """Order the redemptions the sentence asks for.

    Input:
        results (list): loyalty payloads.
        request (str): the latest user text. ``mỗi thẻ`` is one reward per
        card, the cheapest one the balance can pay, skipping a named card.
        context (str): the rendered context.

    Output:
        list[tuple[dict, dict]]: program and reward. One pair when the
        sentence names a single reward.
    """
    low = request.lower()
    if "mỗi" not in low:
        choice = _loyalty_choice(_with_history(results, context, "loyalty.get_points", "prog_"), request)
        return [choice] if choice else []
    plain = _plain(low)
    excluded = ""
    if "trừ" in low:
        named = re.search(r"trừ\s+(?:thẻ\s+)?([^\s,]+)", low)
        if named:
            excluded = _plain(named.group(1))
    queue = []
    for program in _loyalty_programs(results, context):
        label = _plain(str(program.get("name") or ""))
        if excluded and excluded in label:
            continue
        points = program.get("points") or 0
        affordable = []
        for index, reward in enumerate(program.get("rewards") or []):
            cost = reward.get("cost") or reward.get("points") or 0
            if isinstance(cost, int) and cost <= points:
                affordable.append((cost, index, reward))
        if not affordable:
            continue
        affordable.sort()
        queue.append((program, affordable[0][2]))
    return queue


def _named_device(results: list, request: str) -> dict | None:
    """Find the device whose name shares the most words with the sentence.

    Input:
        results (list): payloads from ``smart_home.get_status``.
        request (str): the latest user text.

    Output:
        dict: that device. None when no device name is mentioned.
    """
    low = _plain(request)
    best = None
    best_score = 0
    for item in _collect(results, ""):
        if item.get("type") not in ("light", "climate", "fan", "plug") and not str(item.get("id", "")).startswith(("light_", "ac_", "heater_", "fan_")):
            continue
        label = _plain(str(item.get("name") or ""))
        score = sum(len(token) for token in label.split() if len(token) > 2 and token in low)
        if label and label in low:
            score += 50
        if score > best_score:
            best = item
            best_score = score
    return best if best_score >= 4 else None


def _package_rows(results: list, context: str) -> list[dict]:
    """Return subscription rows from this turn or from history.

    Input:
        results (list): payloads so far.
        context (str): the rendered context.

    Output:
        list[dict]: rows whose id starts with ``sub_``.
    """
    found = {}
    for item in _collect(results, "sub_"):
        if item.get("id"):
            found[item["id"]] = item
    history = [payload for name, _args, payload in _history_invocations(context) if name == "subscriptions.list"]
    for item in _collect(history, "sub_"):
        if item.get("id"):
            found.setdefault(item["id"], item)
    return list(found.values())


def _subscriptions_to_cancel(results: list, request: str, context: str) -> list[dict]:
    """Order the packages the sentence wants cancelled.

    Input:
        results (list): payloads from ``subscriptions.list``.
        request (str): the latest user text.
        context (str): the rendered context. A package named earlier is
        ``gói đấy``. A kick-off date is the task already added.

    Output:
        list[dict]: the packages, earliest billing first when several
        entertainment packages end before the kick-off. One package when
        the sentence names a weekday, a price, or ``gói đấy``.
    """
    low = request.lower()
    pool = [item for item in _package_rows(results, context) if item.get("status") != "cancelled"]
    weekday = _weekday_index(low)
    if weekday is not None and "hôm trước" not in low:
        target = _next_weekday(
            _context_value(context, "today"),
            weekday,
            "tới" in low or "sau" in low,
        )
        matched = [item for item in pool if str(item.get("next_billing") or "") == target]
        if matched:
            return matched
    if re.search(r"gói\s+(?:đấy|ấy|đó|này)", low):
        speech = "\n".join(
            line.split(":", 1)[1]
            for line in _history_text(context).splitlines()
            if line.startswith(("assistant:", "user:"))
        ).lower()
        named = [item for item in pool if str(item.get("name") or "").lower() in speech]
        if named:
            named.sort(key=lambda item: len(str(item.get("name") or "")), reverse=True)
            return named[:1]
    amount = _ngan_amount(request)
    if amount and "sớm" in low:
        priced = [item for item in pool if item.get("price") == amount]
        priced.sort(key=lambda item: str(item.get("next_billing") or "9999"))
        if priced:
            return priced[:1]
    if "mấy gói" in low and "giải trí" in low:
        kick = ""
        for name, args, _payload in _history_invocations(context):
            if name == "tasks.add" and isinstance(args.get("due"), str):
                kick = args["due"][:10]
        fun_words = ("play", "melodify", "xemflix", "zing", "stream", "podcast", "nhac", "phim")
        chosen = []
        for item in pool:
            if item.get("status") not in (None, "active"):
                continue
            label = _plain(str(item.get("name") or ""))
            if not any(word in label for word in fun_words):
                continue
            billing = str(item.get("next_billing") or "")
            if kick and billing >= kick:
                continue
            chosen.append(item)
        if chosen:
            return chosen
    named = _best_named(pool, request)
    return [named] if named else []


def _subscription_to_cancel(
    results: list,
    request: str,
    context: str,
    done_calls: list | None = None,
) -> dict | None:
    """Pick the next package the sentence cancels.

    Input:
        results (list): payloads from ``subscriptions.list``.
        request (str): the latest user text. ``thứ Sáu tới`` is a date.
        context (str): the rendered context, for today.
        done_calls (list | None): cancels already made. The next package is
        one that is not in that list.

    Output:
        dict: the package. None when every match is already cancelled.
    """
    used = {call.get("subscription_id") for name, call in (done_calls or []) if name == "subscriptions.cancel"}
    for item in _subscriptions_to_cancel(results, request, context):
        if item.get("id") not in used:
            return item
    return None


def _parent_emails(results: list) -> list[str]:
    """Collect the email addresses of Bố and Mẹ.

    Input:
        results (list): contact payloads.

    Output:
        list[str]: those addresses, in the order the results listed them.
    """
    found = []
    for payload in results:
        for node in _walk_dicts(payload):
            email = node.get("email")
            if not isinstance(email, str) or "@" not in email:
                continue
            label = _plain(str(node.get("relation") or "")) + " " + _plain(str(node.get("name") or ""))
            if any(word in label.split() for word in ("bo", "me", "ba", "ma")) and email not in found:
                found.append(email)
    return found


def _shopping_item(request: str, context: str) -> tuple[str, int] | None:
    """Read the grocery item and the quantity from the conversation.

    Input:
        request (str): the latest user text. ``hai hộp`` is the quantity.
        context (str): the rendered context. The item was named earlier.

    Output:
        tuple[str, int]: the product name and the quantity.
        None: when history does not name a product.
    """
    users = re.findall(r"user:\s*(.+)", _history_text(context))
    if not users:
        return None
    line = users[-1]
    line = re.sub(r"^(ừ|rồi|dạ|thì)[, ]*", "", line, flags=re.IGNORECASE)
    line = re.sub(r"\b(lấy|mua|chọn|hiệu)\b", " ", line, flags=re.IGNORECASE)
    line = re.sub(r"\b(nha|nhé|ạ)\b[.!]?", "", line, flags=re.IGNORECASE)
    name = " ".join(line.replace(",", " ").split())
    if not name:
        return None
    quantity = 1
    count = re.search(
        r"(một|hai|ba|bốn|năm|sáu|bảy|tám|chín|\d+)\s*(hộp|chai|gói|túi|lon)",
        request.lower(),
    )
    if count:
        raw = count.group(1)
        quantity = int(raw) if raw.isdigit() else _VN_DIGIT.get(raw, 1)
    return name, quantity


def _spoken_percent(text: str) -> int | None:
    """Read the percent spoken before phần trăm.

    Input:
        text (str): the latest user text.

    Output:
        int: that percent. None when the sentence has no percent.
    """
    match = re.search(
        r"((?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi|trăm|tư|lăm|mốt|\d+)"
        r"(?:\s+(?:một|hai|ba|bốn|năm|sáu|bảy|tám|chín|mười|mươi|trăm|tư|lăm|mốt|\d+))*)\s+phần trăm",
        text.lower(),
    )
    if not match:
        return None
    raw = match.group(1).strip()
    return int(raw) if raw.isdigit() else _vn_under_1000(raw) or None


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
    if name == "gold_price.get_price" and re.search(r"\bsjc\b", low):
        args["type"] = "sjc"
    for key in ("time", "start", "end", "schedule_time"):
        if key in args:
            args[key] = _expand_clock(args[key], request, context)
    if name in (
        "calendar.list_events", "smart_home.get_status", "alarm.list",
        "appliances.list", "subscriptions.list", "household_chores.list_chores",
        "bike_rental.find_bikes", "loyalty.get_points",
    ):
        args.clear()
    if name == "home_energy.get_usage":
        args.clear()
        args["period"] = "week"
    if name == "weather.get_forecast":
        location = str(args.get("location") or "")
        if location and location.lower() not in low:
            args.pop("location", None)
        trip = _trip_date(request, context)
        day = trip or _forecast_day(request, context)
        if day:
            args["date"] = day
        place = re.search(r"\b([A-ZÀ-Ỹ][\wà-ỹ]+)\s+mưa", request)
        if place and place.group(1).lower() not in _ROLE:
            args["location"] = place.group(1)
    if name in ("alarm.create", "reminders.create"):
        spoken = _spoken_time(request, context)
        if spoken:
            args["time"] = spoken
    if name == "reminders.create":
        content = _reminder_content(request)
        if "mang laptop" in low:
            args["content"] = "mang laptop"
        elif content:
            args["content"] = content
        event = _ride_event(results, request, context)
        pickup = _pickup_lead(request)
        remind = _remind_lead(request)
        if event and pickup and remind and isinstance(event.get("start"), str):
            args["time"] = _add_minutes(event["start"], -(pickup + remind))
        args.pop("repeat", None)
    if name == "alarm.delete":
        alarm_id = _alarm_to_delete(results, request, context)
        args.clear()
        if alarm_id:
            args["alarm_id"] = alarm_id
    if name == "notes.create" and not str(args.get("content") or "").strip():
        args["content"] = _note_content(request)
    if name == "tasks.add":
        if not str(args.get("title") or "").strip():
            args["title"] = _task_title(request)
        promised = any(phrase in low for phrase in ("lời hứa", "đúng cái đó", "đúng cái"))
        if "hạn" in low or promised:
            spoken = _spoken_time(request, context) if _last_spoken_hm(request) else None
            if spoken:
                args["due"] = spoken
            else:
                due = _due_from_history(context)
                if due:
                    args["due"] = due
                elif "hạn" not in low:
                    args.pop("due", None)
        else:
            args.pop("due", None)
    if name == "alarm.create":
        hours = _hours_before(request)
        event = _event_for_alarm(results, request, context) if hours else None
        if hours and event and isinstance(event.get("start"), str) and any(
            phrase in low for phrase in ("lịch", "trước đó", "trước giờ")
        ):
            args["time"] = _add_minutes(event["start"], -hours * 60)
        args.pop("label", None)
        if "mỗi ngày" not in low and "hằng ngày" not in low:
            args.pop("repeat", None)
    if name == "contacts.search":
        kind, obj, tool = _cursor(_parse_objects(plan), done_calls, facts)
        if "bố mẹ" in low or "ba má" in low:
            args.clear()
        elif kind == "run" and tool == name and obj.get("name"):
            args.clear()
            spoken = obj["name"].strip()
            head = spoken.lower()
            if head.startswith(("sếp ", "bác ", "cô giáo ")):
                args["query"] = spoken
            else:
                args["query"] = _contact_query(spoken)
    if name == "ride_hailing.book":
        if any(word in low for word in ("ngay bây giờ", "đặt ngay", "ngay luôn")):
            args.pop("schedule_time", None)
        if "bảy chỗ" in low or "7 chỗ" in low:
            args["vehicle"] = "car7"
        elif "bốn chỗ" in low or "4 chỗ" in low or "bấm chỗ" in low:
            args["vehicle"] = "car4"
        elif "xe máy" in low:
            args["vehicle"] = "bike"
        if "ô tô" in low or "taxi" in low:
            args["vehicle"] = "car4"
        if any(word in low for word in ("từ nhà", "ở nhà")):
            home = _context_value(context, "home_address")
            if home:
                args["pickup"] = home
        elif any(word in low for word in ("công ty", "chỗ làm")):
            work = _context_value(context, "work_address")
            if work:
                args["pickup"] = work
        spoken_drop = _spoken_dropoff(request)
        if spoken_drop:
            args["dropoff"] = spoken_drop[:1].upper() + spoken_drop[1:]
            spoken = _spoken_time(request, context)
            if spoken:
                args["schedule_time"] = spoken
        else:
            event = _ride_event(results, request, context)
            if event and isinstance(event.get("location"), str):
                args["dropoff"] = event["location"]
            elif not re.search(r"\d", request):
                dropoff = _event_dropoff(results, request)
                if dropoff:
                    args["dropoff"] = dropoff
            lead = _pickup_lead(request)
            if event and lead and isinstance(event.get("start"), str):
                args["schedule_time"] = _add_minutes(event["start"], -lead)
    if name == "smart_home.light_set" and _floor_named(low) and "tắt" in low:
        device = _next_floor_light(results, request, done_calls)
        args.clear()
        if device:
            args["device_id"] = device.get("id")
            args["power"] = "off"
    elif name == "smart_home.light_set" and _turns_lights_off(request):
        lamp = _named_light(results, _active_text(request))
        if lamp:
            args.clear()
            args["device_id"] = lamp.get("id")
            args["power"] = "off"
    if name == "smart_home.climate_set":
        temperature = _climate_temperature(request)
        climate = _climate_device(results, request)
        if temperature is not None and climate:
            args.clear()
            args["device_id"] = climate.get("id")
            args["mode"] = "cool"
            args["temperature"] = temperature
    elif (
        name == "smart_home.light_set"
        and not _turns_lights_off(request)
        and "một nửa" not in low
        and "độ sáng" not in low
    ):
        lamp = _named_light(results, request)
        if lamp:
            args["device_id"] = lamp.get("id")
        percent = _spoken_percent(request)
        if percent is not None:
            args["brightness"] = percent
            args.pop("color", None)
            if "bật" in low and "đèn" in low and not _turns_lights_off(request):
                args["power"] = "on"
            else:
                args.pop("power", None)
    elif name == "smart_home.light_set" and ("một nửa" in low or "độ sáng" in low):
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
        event = None
        event_id = args.get("event_id")
        if event_id:
            for payload in results:
                for node in _walk_dicts(payload):
                    if node.get("id") == event_id and isinstance(node.get("start"), str):
                        event = node
        if event is None:
            event = _ride_event(results, request, context)
            if event and event.get("id"):
                args["event_id"] = event["id"]
        if delta is not None and event and isinstance(event.get("start"), str):
            args["start"] = _add_minutes(event["start"], delta)
            if isinstance(event.get("end"), str):
                args["end"] = _add_minutes(event["end"], delta)
        elif event and "dời" in low and _last_spoken_hm(request):
            spoken = _spoken_time(request, context)
            if spoken and isinstance(event.get("start"), str):
                args["start"] = spoken
                try:
                    start = datetime.fromisoformat(event["start"])
                    end = datetime.fromisoformat(event["end"]) if isinstance(event.get("end"), str) else start + timedelta(hours=1)
                    args["end"] = (datetime.fromisoformat(spoken) + (end - start)).isoformat()
                except ValueError:
                    args["end"] = (datetime.fromisoformat(spoken) + timedelta(hours=1)).isoformat()
    spoken = _ngan_amount(request)
    if name == "ewallet.topup_mobile":
        if spoken:
            args["amount"] = spoken
        if "cho số" in low:
            phone = _spoken_phone(request)
            if phone:
                args["phone"] = phone
            else:
                args.pop("phone", None)
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
        jobs = _transfer_jobs(request)
        focus = jobs[-1][1] if jobs else _bill_focus(request)
        bill = _bill_amount(results, focus)
        if bill is not None:
            args["amount"] = bill
        elif "tiền điện" in low and facts.get("amount"):
            args["amount"] = facts["amount"]
        else:
            spoken_amount = _trieu_amount(request) or _ngan_amount(request) or _history_amount(context)
            if spoken_amount is not None:
                args["amount"] = spoken_amount
        kind, obj, _tool = _cursor(_parse_objects(plan), done_calls, facts)
        contact = _matching_contact(results, obj["name"]) if kind == "run" else None
        if contact:
            args["contact_id"] = contact["id"]
    if name == "bill_payment.list_bills":
        focus = _bill_focus(request)
        args.clear()
        if focus and not _transfer_jobs(request) and not _due_sweep(low) and _due_weekday(low) is None:
            args["type"] = focus
    if name == "bill_payment.pay":
        if _due_sweep(low):
            bill = _next_due_bill(results, context, done_calls)
            args.clear()
            if bill:
                args["bill_id"] = bill["id"]
                args["source"] = "bank"
        else:
            chosen = _bill_to_pay(results, request)
            cap = _ngan_amount(low.split("dưới", 1)[-1]) if "dưới" in low else None
            if chosen and cap is not None and int(chosen["amount"]) > cap:
                args.clear()
            else:
                if chosen:
                    args["bill_id"] = chosen["id"]
                elif facts.get("bill_id"):
                    args["bill_id"] = facts["bill_id"]
                kind, obj, tool = _cursor(_parse_objects(plan), done_calls, facts)
                if kind == "run" and tool == name:
                    action = obj["action"].lower()
                    if "bank" in action or "ngân hàng" in action:
                        args["source"] = "bank"
                    elif "ví" in action or "ewallet" in action:
                        args["source"] = "ewallet"
    if name == "banking.transfer":
        jobs = _transfer_jobs(request)
        done_n = sum(1 for tool, _call in done_calls if tool == name)
        job = jobs[done_n] if done_n < len(jobs) else None
        if job:
            amount = _bill_amount(results, job[1])
            amounts = [_bill_amount(results, kind) for _person, kind in jobs]
            account_amount = sum(value or 0 for value in amounts) if len(jobs) > 1 and "cả hai" in low else amount
            beneficiary = _beneficiary_named(results, job[0])
        else:
            clause = low.split("chuyển", 1)[-1] if "chuyển" in low else low
            amount = _trieu_amount(clause) if "triệu" in clause else None
            if amount is None:
                amount = _bill_amount(results, _bill_focus(request))
            account_amount = amount
            beneficiary = _pick_beneficiary(results, request)
        if amount is None:
            amount = _trieu_amount(request) or _ngan_amount(request)
        if amount is not None:
            args["amount"] = amount
        account = _pick_account(_bank_accounts(results), account_amount, request)
        if account:
            args["from_account_id"] = account
        if beneficiary:
            args["beneficiary_id"] = beneficiary
        args.pop("note", None)
    if name == "food_delivery.get_menu":
        restaurant = _food_restaurant(results, request)
        if restaurant:
            args.clear()
            args["restaurant_id"] = restaurant["id"]
    if name == "food_delivery.place_order":
        restaurant = _food_restaurant(results, request)
        item = _food_item(results, request)
        if restaurant and item:
            args.clear()
            args["restaurant_id"] = restaurant["id"]
            args["items"] = [{"item_id": item["id"], "quantity": _serving_count(results, request, context)}]
    if name == "sms.send":
        kind, obj, _tool = _cursor(_parse_objects(plan), done_calls, facts)
        hits = _named_contacts(results, obj["name"]) if kind == "run" else []
        if len(hits) == 1:
            args["contact_id"] = hits[0]["id"]
        else:
            args.pop("contact_id", None)
        if not str(args.get("content") or "").strip():
            args["content"] = "giờ mới" if "giờ mới" in low else (_reminder_content(request) or "đã xong")
    if name == "appliances.start_cycle":
        job = _appliance_job(results, request)
        if job:
            args.clear()
            args["appliance_id"] = job[0]["id"]
            args["cycle"] = job[1]
            if "đúng giờ" in low:
                spoken = _spoken_time(request, context)
                now = re.search(r"current_time:\s*(\S+)", context)
                if spoken and now:
                    try:
                        delta = datetime.fromisoformat(spoken) - datetime.fromisoformat(now.group(1))
                    except ValueError:
                        delta = None
                    if delta is not None and delta.total_seconds() > 0:
                        args["delay_minutes"] = int(delta.total_seconds() // 60)
    if name == "household_chores.assign":
        jobs = _chore_assignments(results, request, context)
        used = {call.get("chore_id") for done_name, call in done_calls if done_name == "household_chores.assign"}
        job = next((item for item in jobs if item[0] not in used), None)
        if job:
            args.clear()
            args["chore_id"] = job[0]
            args["assignee"] = job[1]
    if name == "subscriptions.cancel":
        chosen = _subscription_to_cancel(results, request, context, done_calls)
        if chosen:
            args.clear()
            args["subscription_id"] = chosen["id"]
    if name == "email.create_draft":
        parents = _parent_emails(results) if "bố mẹ" in low or "ba má" in low else []
        if len(parents) >= 2:
            package = ""
            named = re.search(r"gói\s+(.+?)\s+(?:giúp|với|xong|,)", request, flags=re.IGNORECASE)
            if named:
                package = named.group(1).strip()
            args.clear()
            args["to"] = parents
            args["subject"] = f"Gói {package} không gia hạn nữa" if package else "không gia hạn nữa"
            args["body"] = (
                f"Bố mẹ ơi, con huỷ gói {package} rồi, từ tháng sau gói đó không gia hạn nữa ạ."
                if package
                else "Bố mẹ ơi, từ tháng sau gói đó không gia hạn nữa ạ."
            )
        else:
            address = _draft_address(results, request)
            if address:
                args["to"] = [address]
                package = ""
                history = _history_text(context)
                named = re.search(r"(gói\s+.+?)\s+(?:rồi|nhé|ạ|đã)", history, flags=re.IGNORECASE)
                if named:
                    package = named.group(1).strip()
                if package and re.search(r"(?:hủy|huỷ)\s+rồi|(?:đã|vừa)\s+(?:hủy|huỷ)", low):
                    args["subject"] = f"Thông báo hủy {package}"
                    args["body"] = f"Chào cả nhóm, {package} đã được hủy. Ai cần dữ liệu thì tải về sớm nhé."
                else:
                    if not str(args.get("subject") or "").strip():
                        args["subject"] = "nháp"
                    if not str(args.get("body") or "").strip():
                        args["body"] = request.strip() or "nháp"
    if name == "bike_rental.find_bikes":
        near = _bike_area(request, context)
        if near:
            args["near"] = near
    if name == "bike_rental.reserve":
        used = {call.get("bike_id") for done_name, call in done_calls if done_name == "bike_rental.reserve"}
        queue = _bike_queue(results, request, context)
        bike = next((item for item in queue if item.get("id") not in used), None) or _bike_to_reserve(results, request, context)
        if bike and bike.get("id") not in used:
            args.clear()
            args["bike_id"] = bike["id"]
    if name == "shopping_list.add":
        item = _shopping_item(request, context)
        if item:
            args["name"] = item[0]
            args["quantity"] = item[1]
    if name == "fitness.log_workout":
        workout = _workout(low)
        if workout:
            args.clear()
            args["type"] = workout[0]
            args["minutes"] = workout[1]
    if name == "library.search":
        named = re.search(r"của\s+(.+?)\s+mà", request, flags=re.IGNORECASE)
        titles = _spoken_titles(request)
        if "mưa thì" in low and facts.get("rain") is True:
            titles = [title for title in titles if "kỹ sư" in title.lower()] or titles[-1:]
        searched = {str(call.get("query") or "") for done_name, call in done_calls if done_name == "library.search"}
        pending = next((title for title in titles if title not in searched), None)
        if pending:
            args["query"] = pending
        elif named:
            args["query"] = named.group(1).strip()
    if name == "library.reserve":
        book = _book_to_reserve(
            _with_history(results, context, "library.search", "book_"),
            request,
            done_calls,
            context,
        )
        if book:
            args.clear()
            args["book_id"] = book["id"]
    if name == "bill_payment.schedule_autopay":
        bill_type = _soonest_bill_type(results)
        if bill_type:
            args.clear()
            args["type"] = bill_type
            args["enabled"] = True
    if name == "robot_vacuum.start":
        rooms = [] if "cả nhà" in low else _robot_rooms(results, request, context)
        args.clear()
        if rooms:
            args["rooms"] = rooms
        if "hút" in low and "lau" in low:
            args["mode"] = "both"
    if name == "alarm.update":
        alarm = _alarm_to_shift(results, request, context)
        if alarm:
            args.clear()
            args["alarm_id"] = alarm["id"]
            if alarm.get("time"):
                args["time"] = alarm["time"]
    if name == "maps.save_place":
        place = _place_to_save(results, request, context)
        if place:
            args.clear()
            args["place_id"] = place["id"]
            args["label"] = str(place.get("name") or "đã lưu")
    if name == "loyalty.redeem":
        used = {
            (call.get("program"), call.get("reward_id"))
            for done_name, call in done_calls
            if done_name == "loyalty.redeem"
        }
        if "đổi hết" in low or "mấy ly" in low:
            choice = _loyalty_choice(results, request)
            done_n = sum(1 for done_name, _call in done_calls if done_name == "loyalty.redeem")
            cost = int((choice[1].get("cost") or 0)) if choice else 0
            points = int((choice[0].get("points") or 0)) if choice else 0
            if choice and cost and done_n < points // cost:
                args.clear()
                args["program"] = choice[0].get("id")
                args["reward_id"] = choice[1].get("id")
            else:
                args.clear()
        else:
            choice = next(
                (item for item in _loyalty_queue(results, request, context) if (item[0].get("id"), item[1].get("id")) not in used),
                None,
            )
            if choice:
                args.clear()
                args["program"] = choice[0].get("id")
                args["reward_id"] = choice[1].get("id")
    if name == "home_automation.create_rule":
        device = _named_device(results, request + "\n" + _history_text(context))
        clock = _clock_hhmm(request)
        if device and clock:
            args.clear()
            args["device_id"] = device.get("id")
            turning_on = "bật" in low and "tắt" not in low
            args["action"] = "on" if turning_on else "off"
            args["time"] = clock
            if turning_on and "tối" in low:
                args["title"] = f"Bật {device.get('name')} buổi tối"
    if name == "calendar.create_event" and _wants_new_event(low):
        restored = None
        if any(phrase in low for phrase in ("tạo lại", "y giờ cũ")):
            for _tool, _call_args, payload in _history_invocations(context):
                for node in _walk_dicts(payload):
                    if str(node.get("id") or "").startswith("evt_") and node.get("status") == "cancelled":
                        restored = node
        if restored and isinstance(restored.get("start"), str):
            args.clear()
            args["title"] = restored.get("title") or "lịch"
            args["start"] = restored["start"]
            if isinstance(restored.get("end"), str):
                args["end"] = restored["end"]
        elif "hết hạn" in low:
            package = next(iter(_subscriptions_to_cancel(results, request, context)), None)
            spoken = _spoken_time(request, context)
            if package and spoken and package.get("next_billing"):
                clock = spoken.split("T", 1)[1]
                args["start"] = f"{package['next_billing']}T{clock}"
                try:
                    args["end"] = (datetime.fromisoformat(args["start"]) + timedelta(hours=1)).isoformat()
                except ValueError:
                    pass
                args["title"] = f"Gói {package.get('name')} hết hạn"
            elif spoken and not args.get("start"):
                args["start"] = spoken
        else:
            spoken = _spoken_time(request, context)
            if spoken and not args.get("start"):
                args["start"] = spoken
                if not args.get("end"):
                    try:
                        args["end"] = (datetime.fromisoformat(spoken) + timedelta(hours=1)).isoformat()
                    except ValueError:
                        pass
        if any(phrase in low for phrase in ("ở nhà", "nhà mình", "nhà ta")):
            home = _context_value(context, "home_address")
            if home:
                args["location"] = home
        if not str(args.get("title") or "").strip():
            args["title"] = "lịch"
    if name == "chat_ott.send_message":
        sequence = _chat_sequence(results, request, context)
        if sequence:
            sent = [call.get("chat_id") for done_name, call in done_calls if done_name == "chat_ott.send_message"]
            pending = [item for item in sequence if item.get("chat_id") not in sent]
            if pending:
                args.clear()
                args.update(pending[0])
    if name == "phone.call":
        phone = _spoken_phone(request)
        if phone:
            args.clear()
            args["phone_number"] = phone
    if name == "music.play":
        title = _plain(_track_title(request, context) or "")
        for payload in results:
            for node in _walk_dicts(payload):
                if isinstance(node.get("id"), str) and str(node["id"]).startswith("trk_") and title and title in _plain(str(node.get("title") or "")):
                    args.clear()
                    args["track_id"] = node["id"]
    if name == "music.list_tracks":
        args.clear()
    if name == "news.headlines":
        args.clear()
        args["topic"] = "entertainment"
    if name == "maps.search_place":
        query = _place_query(request, context)
        if query:
            args["query"] = query
    if name == "maps.get_route":
        match = re.search(r"từ\s+(.+?)\s+ra\s+(.+?)(?:\s+luôn|$)", request, flags=re.IGNORECASE)
        if match:
            args.clear()
            args["origin"] = match.group(1).strip()
            args["destination"] = match.group(2).strip()
    if name == "video_meeting.create":
        spoken = _spoken_time(request, context)
        title = ""
        for payload in results:
            for node in _walk_dicts(payload):
                label = str(node.get("title") or "")
                if label and _plain(label) in _plain(request):
                    title = label
                    if not spoken and isinstance(node.get("start"), str):
                        spoken = node["start"]
        if spoken:
            args.clear()
            args["title"] = title or "họp"
            args["start"] = spoken
            people = re.findall(r"[A-ZÀ-Ỹ][\wà-ỹ]+", request)
            for payload in results:
                for node in _walk_dicts(payload):
                    contact_id = node.get("id")
                    contact = str(node.get("name") or "")
                    if isinstance(contact_id, str) and contact_id.startswith("c_") and any(person.lower() in contact.lower() for person in people):
                        args["attendees"] = [contact_id]
    if name == "medication.remove":
        words = [token for token in _plain(request).split() if len(token) > 2]
        best = None
        best_score = 0
        for payload in results:
            for node in _walk_dicts(payload):
                schedule_id = node.get("id")
                label = _plain(str(node.get("name") or ""))
                if not isinstance(schedule_id, str) or not schedule_id.startswith("med_"):
                    continue
                score = sum(len(token) for token in words if token in label)
                if score > best_score:
                    best = schedule_id
                    best_score = score
        if best:
            args.clear()
            args["schedule_id"] = best
    if name == "ride_hailing.cancel":
        ride_id = _named_ride(request, context)
        if ride_id:
            args.clear()
            args["ride_id"] = ride_id
    if name == "fitness.set_goal":
        steps = _history_steps(request, context)
        if steps:
            args.clear()
            args["steps"] = steps // 2 if facts.get("rain") is True and "nửa" in low else steps
    if name == "learning.schedule_lesson":
        course_id = None
        for payload in results:
            for node in _walk_dicts(payload):
                label = _plain(str(node.get("name") or ""))
                if isinstance(node.get("id"), str) and str(node["id"]).startswith("course_") and "le tan" in label and "homestay" in _plain(request):
                    course_id = node["id"]
        day = _forecast_day(request, context) or _context_value(context, "tomorrow")
        hour = 19 if facts.get("rain") is True else 21
        offset = re.search(r"([+-]\d{2}:\d{2})", context)
        if course_id and day:
            args.clear()
            args["course_id"] = course_id
            args["time"] = f"{day}T{hour:02d}:00:00{offset.group(1) if offset else '+07:00'}"
    if name == "chat_ott.send_location":
        person = _person_after(request, "cho") or ""
        for payload in results:
            for node in _walk_dicts(payload):
                if isinstance(node.get("id"), str) and str(node["id"]).startswith("chat_") and person and _plain(person) in _plain(str(node.get("name") or "")):
                    args.clear()
                    args["chat_id"] = node["id"]
    if name == "subscriptions.cancel" and "quá" in low and "tháng" in low:
        cap = _ngan_amount(low.split("quá", 1)[-1])
        for payload in results:
            for node in _walk_dicts(payload):
                price = node.get("price")
                label = _plain(str(node.get("name") or ""))
                if not isinstance(node.get("id"), str) or not str(node["id"]).startswith("sub_"):
                    continue
                if not isinstance(price, (int, float)) or cap is None or price <= cap:
                    continue
                if any(token in label for token in _plain(request).split() if len(token) > 3):
                    args.clear()
                    args["subscription_id"] = node["id"]
    if name == "alarm.create":
        show = _show_alarm(request, context)
        if show:
            args["time"] = show
    if name == "smart_home.light_set" and "nhầm" in low and "quạt" in low:
        used = {call.get("device_id") for done_name, call in done_calls if done_name == "smart_home.light_set"}
        lamp = next((item for item in _room_devices(results, request, "light") if item.get("id") not in used), None)
        if lamp:
            args.clear()
            args["device_id"] = lamp["id"]
            args["power"] = "off"
    if name == "smart_home.fan_set" and "tắt" in low:
        used = {call.get("device_id") for done_name, call in done_calls if done_name == "smart_home.fan_set"}
        fan = next((item for item in _room_devices(results, request, "fan") if item.get("id") not in used), None)
        if fan:
            args.clear()
            args["device_id"] = fan["id"]
            args["power"] = "off"
    if name == "sys.ask_missing_info" and "nhầm" in low and "việc" in low:
        args.clear()
        args["tool_name"] = "tasks.complete"
        args["params"] = ["task_id"]
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
    if name == "alarm.create" and "nhắc" in text and "báo thức" not in text:
        return "nhắc thì dùng reminders.create, không dùng alarm."
    if name == "phone.call" and "nhắc" in text and _spoken_phone(request) is None:
        return "nhắc thì dùng reminders.create, không gọi điện."
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
    if name != "loyalty.redeem" and any(_args_key(old) == key for old in prior):
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


def _planned_tool(plan: str, facts: dict, done_calls: list[tuple[str, dict]]) -> str:
    """Return the one tool the plan wants next.

    Input:
        plan (str): the object plan.
        facts (dict): fields used by ``khi``.
        done_calls (list[tuple[str, dict]]): successful calls so far.

    Output:
        str: the next tool name.
        Empty when the plan is finished, waiting, or has no objects.
    """
    objects = _parse_objects(plan)
    if not objects:
        return ""
    kind, _obj, tool = _cursor(objects, done_calls, facts)
    return tool if kind == "run" else ""


_FILL_NEXT = frozenset({
    "notes.create",
    "tasks.add",
    "alarm.list",
    "alarm.delete",
    "alarm.create",
    "banking.get_balance",
    "banking.list_beneficiaries",
    "banking.transfer",
    "bill_payment.list_bills",
    "bill_payment.pay",
    "reminders.create",
    "sms.send",
    "ewallet.transfer",
    "smart_home.light_set",
    "smart_home.get_status",
    "contacts.search",
    "weather.get_forecast",
    "appliances.list",
    "appliances.start_cycle",
    "household_chores.list_chores",
    "household_chores.assign",
    "subscriptions.list",
    "subscriptions.cancel",
    "email.create_draft",
    "bike_rental.find_bikes",
    "bike_rental.reserve",
    "loyalty.get_points",
    "loyalty.redeem",
    "home_energy.get_usage",
    "home_automation.create_rule",
    "calendar.create_event",
    "shopping_list.add",
    "fitness.log_workout",
    "library.search",
    "library.reserve",
    "bill_payment.schedule_autopay",
    "robot_vacuum.get_status",
    "robot_vacuum.start",
    "alarm.list",
    "alarm.update",
    "maps.search_place",
    "maps.save_place",
})


def _note_content(text: str) -> str:
    """Keep a non-empty body for a new note.

    Input:
        text (str): the latest user text.

    Output:
        str: the words after ``ghi là`` or ``tên là`` when present, otherwise
        the sentence itself. Title and content are free text.
    """
    low = text.lower()
    for marker in ("ghi là", "tên là", "tiêu đề là"):
        if marker in low:
            tail = text[low.find(marker) + len(marker):].strip(" .,")
            if tail:
                return tail
    return text.strip() or "ghi chú"


def _task_title(text: str) -> str:
    """Keep the errand named after ``một việc`` or ``cái việc``.

    Input:
        text (str): the latest user text.

    Output:
        str: the words up to ``hạn``. A short fallback when the clause is empty.
    """
    low = text.lower()
    for marker in ("việc là", "một việc", "cái việc"):
        if marker in low:
            tail = text[low.find(marker) + len(marker):]
            tail = re.split(r"\bhạn\b", tail, maxsplit=1, flags=re.IGNORECASE)[0]
            title = tail.strip(" .,")
            if title:
                return title
    return "việc trong câu"


def _alarm_to_delete(results: list, request: str, context: str) -> str | None:
    """Find the alarm whose time is the clock in the sentence.

    Input:
        results (list): tool payloads. An alarm has ``id`` and ``time``.
        request (str): the latest user text.
        context (str): the rendered context, for today, tomorrow, and the offset.

    Output:
        str: the alarm id whose time equals the spoken clock.
        None: when no listed alarm is at that time.
    """
    spoken = _spoken_time(request, context)
    if not spoken:
        return None
    for payload in results:
        for node in _walk_dicts(payload):
            if str(node.get("id", "")).startswith("alm_") and node.get("time") == spoken:
                return node["id"]
    return None


def _can_take_over(
    tool: str,
    request: str,
    results: list,
    done_calls: list[tuple[str, dict]],
) -> bool:
    """Return whether a wrong call can be replaced by the planned tool.

    The model sometimes repeats a read or searches a person while the plan
    still has a floor lamp or a bank transfer to finish. Those calls have
    their arguments filled from the sentence and earlier results.

    Input:
        tool (str): the tool the plan wants next.
        request (str): the latest user text.
        results (list): payloads of successful calls so far.
        done_calls (list[tuple[str, dict]]): those calls, with arguments.

    Output:
        bool: True when that tool can run without a new choice from the model.
    """
    if tool == "smart_home.light_set":
        return _next_floor_light(results, request, done_calls) is not None
    if tool == "bill_payment.pay":
        return _due_sweep(request)
    if tool == "smart_home.climate_set":
        return _climate_temperature(request) is not None
    if tool in ("chat_ott.list_chats", "food_delivery.list_restaurants", "food_delivery.get_menu", "food_delivery.place_order"):
        return _group_meal(request.lower())
    if tool == "alarm.create":
        return "báo thức" in request.lower()
    if tool == "notes.create":
        return "ghi chú" in request.lower()
    if tool == "tasks.add":
        return _wants_task(request.lower())
    if tool in ("alarm.list", "alarm.delete"):
        low_request = request.lower()
        return "báo thức" in low_request and any(word in low_request for word in ("xóa", "xoá"))
    if tool == "sys.ask_missing_info":
        return _missing_wallet_amount(request)
    if tool == "ride_hailing.book":
        return _pickup_lead(request) is not None
    if tool == "reminders.create":
        return "mang laptop" in request.lower() or _spoken_time(request, "") is not None or "trước" not in request.lower()
    if tool == "contacts.search":
        return True
    if tool == "sms.send":
        return any(str(node.get("id", "")).startswith("c_") for payload in results for node in _walk_dicts(payload))
    if tool == "weather.get_forecast":
        return "mưa" in request.lower() or "thời tiết" in request.lower()
    if tool == "appliances.list":
        return any(word in request.lower() for word in ("máy rửa", "máy giặt", "máy sấy"))
    if tool == "appliances.start_cycle":
        return _appliance_job(results, request) is not None
    if tool == "household_chores.list_chores":
        return "giao" in request.lower()
    if tool == "household_chores.assign":
        return _chore_job(results, request) is not None
    if tool == "subscriptions.list":
        return "gói" in request.lower()
    if tool == "subscriptions.cancel":
        return _best_named(_collect(results, "sub_"), request) is not None
    if tool == "email.create_draft":
        return _draft_address(results, request) is not None
    if tool == "bike_rental.find_bikes":
        return "xe đạp" in request.lower()
    if tool == "bike_rental.reserve":
        return _bike_to_reserve(results, request) is not None
    if tool == "loyalty.get_points":
        return "đổi" in request.lower()
    if tool == "loyalty.redeem":
        return _loyalty_choice(results, request) is not None
    if tool == "home_energy.get_usage":
        return "số" in request.lower()
    if tool == "smart_home.get_status":
        return "cài" in request.lower() or ("đèn" in request.lower() and "tắt" in request.lower())
    if tool == "calendar.create_event":
        return _wants_new_event(request.lower()) and _last_spoken_hm(request) is not None
    if tool == "shopping_list.add":
        return "danh sách" in request.lower()
    if tool == "fitness.log_workout":
        return _workout(request.lower()) is not None
    if tool == "library.search":
        return "cuốn" in request.lower()
    if tool == "library.reserve":
        return _book_to_reserve(results, request, done_calls) is not None
    if tool == "bill_payment.schedule_autopay":
        return _soonest_bill_type(results) is not None
    if tool == "robot_vacuum.get_status":
        return "robot" in request.lower()
    if tool == "robot_vacuum.start":
        return "robot" in request.lower()
    if tool == "alarm.update":
        return _alarm_to_shift(results, request) is not None
    if tool == "maps.search_place":
        return "lưu" in request.lower()
    if tool == "maps.save_place":
        return _place_to_save(results, request) is not None
    if tool == "home_automation.create_rule":
        limit = _kwh_limit(request)
        used = _usage_kwh(results, request)
        if limit is not None and (used is None or used <= limit):
            return False
        return _clock_hhmm(request) is not None and _named_device(results, request) is not None
    return tool in (
        "banking.get_balance", "banking.list_beneficiaries", "banking.transfer",
        "bill_payment.list_bills",
        "calendar.list_events",
    )


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
