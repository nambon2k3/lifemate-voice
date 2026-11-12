"""Slot checks for the ASR prompt, the transcript edits, and the phone guard.

Run from the repo root: ``python scripts/test_slots.py``.
The fixtures are the cached ASR lines for train_0001, train_0003, train_0008,
and train_0009. Context comes from the train JSONL. State files are not read.
"""

from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _stub_langchain() -> None:
    """Register stand-in modules so agent.py can load without LangChain."""

    def add(name: str, **attrs: object) -> None:
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module

    class Marker:
        pass

    add("langchain_core")
    add("langchain_core.messages", HumanMessage=Marker, SystemMessage=Marker, ToolMessage=Marker)
    add("langchain_openai", ChatOpenAI=Marker)
    add("langgraph")
    add("langgraph.graph", END="END", START="START", StateGraph=Marker)
    add("langgraph.graph.message", add_messages=lambda left: left)


try:
    from agent import (  # noqa: E402
        _contact_query,
        _cursor,
        _is_speaker,
        _matching_contact,
        _ngan_amount,
        _parse_objects,
        _person_after,
        _phone_override,
        _prepare_args,
        _repair_plan,
        _spoken_phone,
        _spoken_time,
        _trieu_amount,
    )
except ModuleNotFoundError:
    _stub_langchain()
    from agent import (  # noqa: E402
        _contact_query,
        _cursor,
        _is_speaker,
        _matching_contact,
        _ngan_amount,
        _parse_objects,
        _person_after,
        _phone_override,
        _prepare_args,
        _repair_plan,
        _spoken_phone,
        _spoken_time,
        _trieu_amount,
    )
from asr import asr_instruction, recover_spoken_zeros, served_model  # noqa: E402
from correct import correct_slots, syllable_distance  # noqa: E402
from utils import _using_gold  # noqa: E402

TRAIN = ROOT / "train+public_test" / "train" / "train.jsonl"

ASR = {
    "train_0001": (
        "nếu mai mưa thì giờ chạy bộ muộn một tiếng nhắn tiếp Dũng giờ mới không mưa thì thôi."
    ),
    "train_0003": (
        "ơi, nặng kým mình năm mươi ngàn thì điện thoại cho bác ba với chuyển ví cho con mai "
        "số tiền điện hắn này nhé."
    ),
    "train_0008": "Cái còn lại nếu may mưa thì rời muộn nửa tiếng, không mưa giữ nguyên.",
    "train_0009": (
        "đặt dùm em liên năm mươi ngàn cho số chín năm năm ba sáu ba sáu bảy nha."
    ),
}
GOLD_PHONE = (
    "Nạp dùm em Linh 50 ngàn cho số không chín năm không năm ba sáu không ba sáu bảy nha."
)


def _sample(sample_id: str) -> dict:
    """Load one train row's context. Does not open state.json."""
    with TRAIN.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["id"] == sample_id:
                return {"id": row["id"], "context": row["context"], "audio_text": row.get("audio_text")}
    raise AssertionError(f"missing {sample_id}")


class SlotTests(unittest.TestCase):
    def test_asr_instruction_keeps_khong_and_names_no_contact(self) -> None:
        text = asr_instruction("Vietnamese")
        self.assertIn("không", text)
        self.assertIn("chữ số 0", text)
        self.assertNotIn("Sếp", text)
        self.assertNotIn("Mai", text)
        self.assertEqual(
            served_model({"asr_model": "vinai/PhoWhisper-large"}),
            "qwen3-asr",
        )

    def test_zero_retry_accepts_only_inserted_khong(self) -> None:
        base = "đặt dùm em liên năm mươi ngàn cho số chín năm năm ba sáu ba sáu bảy nha"
        good = (
            "đặt dùm em liên năm mươi ngàn cho số không chín năm không năm "
            "ba sáu không ba sáu bảy nha."
        )
        self.assertEqual(recover_spoken_zeros(base, good), good)
        moved = (
            "đặt cho mẹ em liên năm mươi ngàn cho số không chín năm không năm "
            "ba sáu không ba sáu bảy nha."
        )
        self.assertIsNone(recover_spoken_zeros(base, moved))
        changed_digit = (
            "đặt dùm em liên năm mươi ngàn cho số không chín năm không năm "
            "ba bảy không ba sáu bảy nha."
        )
        self.assertIsNone(recover_spoken_zeros(base, changed_digit))

    def test_spoken_phone_keeps_a_full_zero_sequence(self) -> None:
        self.assertEqual(_spoken_phone(GOLD_PHONE), "09505360367")

    def test_spoken_phone_rejects_a_sequence_that_dropped_khong(self) -> None:
        self.assertIsNone(_spoken_phone(ASR["train_0009"]))
        args = _prepare_args(
            "ewallet.topup_mobile",
            {"phone": "95536367", "amount": 1},
            ASR["train_0009"],
            "",
            [],
            {"phone": "09505360367"},
            "",
            [],
        )
        self.assertNotIn("phone", args)
        self.assertEqual(args["amount"], 50000)

    def test_spoken_phone_rejects_a_leading_zero_run_shorter_than_ten(self) -> None:
        short = "nạp cho số không chín một hai ba bốn năm sáu bảy nha"
        self.assertIsNone(_spoken_phone(short))

    def test_speaker_is_not_searched_and_a_different_name_is(self) -> None:
        self.assertTrue(_is_speaker("Linh", "Em Linh"))
        self.assertTrue(_is_speaker("Em Linh", "Em Linh"))
        self.assertFalse(_is_speaker("Em Hương", "Em Linh"))
        self.assertFalse(_is_speaker("Bác Ba", "Em Linh"))
        self.assertFalse(_is_speaker("Bác Linh", "Em Linh"))
        sample = _sample("train_0009")
        fixed = correct_slots(ASR["train_0009"], sample)
        repaired = _repair_plan(
            "Đối tượng: Linh | việc: tìm liên hệ | khi: luôn | tool: contacts.search",
            fixed,
            {"contacts.search", "ewallet.topup_mobile"},
            {},
            "Em Linh",
        )
        self.assertNotIn("contacts.search", repaired)
        self.assertIn("ewallet.topup_mobile", repaired)
        self.assertIn("số điện thoại", repaired)
        huong = _repair_plan(
            "",
            "nhắn em Hương giúp",
            {"contacts.search", "sms.send"},
            {},
            "Em Linh",
        )
        self.assertIn("contacts.search", huong)
        self.assertIn("Hương", huong)
        bac = _repair_plan(
            "",
            "nhắn bác Ba giúp",
            {"contacts.search", "sms.send"},
            {},
            "Em Linh",
        )
        self.assertIn("contacts.search", bac)
        self.assertIn("Bác Ba", bac)

    def test_train_0009_llm_slots_restore_nap_and_user_name(self) -> None:
        sample = _sample("train_0009")
        fixed = correct_slots(ASR["train_0009"], sample)
        self.assertIn("Linh", fixed)
        self.assertNotIn("liên", fixed)
        self.assertIn("đặt", fixed)
        self.assertEqual(_ngan_amount(fixed), 50000)
        self.assertIsNone(_spoken_phone(fixed))
        self.assertIn("chín năm năm ba sáu ba sáu bảy", fixed)

    def test_user_name_is_not_applied_to_lien_he(self) -> None:
        sample = _sample("train_0009")
        fixed = correct_slots("xem liên hệ của em", sample)
        self.assertIn("liên hệ", fixed)
        self.assertNotIn("Linh", fixed)

    def test_syllable_distance_is_one_step_not_a_word_pair(self) -> None:
        self.assertLessEqual(syllable_distance("liên", "Linh"), 1)
        self.assertLessEqual(syllable_distance("giờ", "dời"), 1)
        self.assertLessEqual(syllable_distance("rời", "dời"), 1)
        self.assertGreater(syllable_distance("đặt", "nạp"), 1)
        self.assertGreater(syllable_distance("tiếp", "sếp"), 1)
        self.assertGreater(syllable_distance("nặng", "nạp"), 1)

    def test_train_0001_restores_doi_and_does_not_invent_sep(self) -> None:
        sample = _sample("train_0001")
        fixed = correct_slots(ASR["train_0001"], sample)
        self.assertIn("dời chạy bộ muộn", fixed)
        self.assertIn("giờ mới", fixed)
        self.assertIn("tiếp", fixed)
        self.assertNotIn("Sếp", fixed)
        self.assertNotIn("sếp", fixed.lower())
        self.assertEqual(_person_after(fixed, "nhắn"), "Dũng")
        repaired = _repair_plan(
            "\n".join([
                "Đối tượng: Dũng | việc: kiểm tra thời tiết hôm sau | khi: luôn | tool: weather.get_forecast",
                "Đối tượng: chạy bộ | việc: dời sự kiện muộn một tiếng | khi: rain=true | tool: calendar.list_events, calendar.update_event",
            ]),
            fixed,
            {
                "weather.get_forecast",
                "calendar.list_events",
                "calendar.update_event",
                "contacts.search",
                "sms.send",
            },
            {},
        )
        self.assertEqual(
            repaired,
            "\n".join([
                "Đối tượng: thời tiết | việc: kiểm tra mưa | khi: luôn | tool: weather.get_forecast",
                "Đối tượng: chạy bộ | việc: dời sự kiện muộn một tiếng | khi: rain=true | tool: calendar.list_events, calendar.update_event",
                "Đối tượng: Dũng | việc: nhắn giờ mới | khi: rain=true | tool: contacts.search, sms.send",
            ]),
        )
        self.assertNotIn("Dũng | việc: kiểm tra", repaired)
        self.assertEqual(_person_after("nhắn Sếp Dũng giờ mới", "nhắn"), "Sếp Dũng")
        self.assertEqual(_person_after("nhắn giúp bác ba", "nhắn"), "Bác Ba")

    def test_train_0008_restores_doi_beside_muon(self) -> None:
        sample = _sample("train_0008")
        fixed = correct_slots(ASR["train_0008"], sample)
        self.assertIn("dời muộn", fixed)
        self.assertNotIn("rời", fixed)

    def test_move_verb_stays_when_calendar_is_not_installed(self) -> None:
        sample = {"context": {"user_profile": {"name": "Em Linh"}, "installed_apps": {}}}
        fixed = correct_slots(ASR["train_0001"], sample)
        self.assertIn("giờ chạy bộ", fixed)
        self.assertNotIn("dời", fixed)

    def test_dat_xe_is_not_rewritten_as_nap(self) -> None:
        sample = _sample("train_0009")
        text = "đặt giúp tôi xe bốn chỗ từ công ty"
        self.assertEqual(correct_slots(text, sample), text)

    def test_contact_query_drops_a_wrong_title(self) -> None:
        self.assertEqual(_contact_query("Con Mai"), "Mai")
        self.assertEqual(_contact_query("Bác Ba"), "Ba")
        self.assertEqual(_contact_query("Sếp Dũng"), "Dũng")
        self.assertEqual(_contact_query("Cô giáo Mai"), "Mai")
        self.assertEqual(_contact_query("Mẹ"), "Mẹ")
        self.assertEqual(_contact_query("Dũng"), "Dũng")
        plan = (
            "Đối tượng: Con Mai | việc: chuyển ví, query Con Mai | khi: luôn | "
            "tool: contacts.search, ewallet.transfer"
        )
        args = _prepare_args(
            "contacts.search",
            {"query": "Con Mai"},
            "chuyển ví cho con mai",
            "",
            [],
            {},
            plan,
            [],
        )
        self.assertEqual(args, {"query": "Mai"})
        found = _matching_contact(
            [{
                "contacts": [
                    {"id": "c_06", "name": "Cô giáo Mai", "phone": "09369191783"},
                    {"id": "c_07", "name": "Hương", "relation": "bạn"},
                ],
            }],
            "Con Mai",
        )
        self.assertEqual(found["id"], "c_06")
        bac = _matching_contact(
            [{
                "contacts": [
                    {"id": "c_07", "name": "Hương", "relation": "bạn"},
                    {"id": "c_02", "name": "Bác Ba"},
                ],
            }],
            "Bác Ba",
        )
        self.assertEqual(bac["id"], "c_02")

    def test_train_0003_keeps_bac_ba_and_does_not_invent_co_mai(self) -> None:
        sample = _sample("train_0003")
        fixed = correct_slots(ASR["train_0003"], sample)
        self.assertIn("bác ba", fixed)
        self.assertIn("con mai", fixed)
        self.assertNotIn("Cô giáo Mai", fixed)
        self.assertNotIn("nạp", fixed.lower())

    def test_gold_transcript_is_not_a_correction_target(self) -> None:
        sample = _sample("train_0009")
        self.assertTrue(_using_gold({"use_gold_transcript": True}, sample, sample["audio_text"]))
        self.assertFalse(_using_gold({"use_gold_transcript": False}, sample, ASR["train_0009"]))

    def test_retraction_does_not_keep_the_model_plan(self) -> None:
        repaired = _repair_plan(
            "Đối tượng: đèn phòng làm việc | việc: tắt | khi: luôn | tool: smart_home.get_status, smart_home.light_set",
            "Tắt đèn phòng làm việc đi à thôi khỏi, không cần.",
            {"smart_home.get_status", "smart_home.light_set"},
            {},
        )
        self.assertIn("dừng", repaired)
        self.assertNotIn("light_set", repaired)

    def test_nhac_is_a_reminder_and_kem_moves_the_clock_back(self) -> None:
        airport = "Nhắc dùm chị ra sân bay lúc 8 giờ kém 15 tối mai nha."
        repaired = _repair_plan(
            "Đối tượng: Chị | việc: nhắc | khi: luôn | tool: alarm.create",
            airport,
            {"alarm.create", "reminders.create", "phone.call"},
            {},
        )
        self.assertIn("reminders.create", repaired)
        self.assertNotIn("alarm.create", repaired)
        context = "today: 2026-11-17\ntomorrow: 2026-11-18\ncurrent_time: 2026-11-17T10:00:00+07:00\n"
        self.assertEqual(_spoken_time(airport, context), "2026-11-18T19:45:00+07:00")
        call_back = "Nhớ nhắc mình gọi lại cho khách lúc tám rưỡi tối mai nhé."
        call_plan = _repair_plan(
            "Đối tượng: khách | việc: gọi lại | khi: luôn | tool: phone.call",
            call_back,
            {"phone.call", "reminders.create", "contacts.search"},
            {},
        )
        self.assertIn("reminders.create", call_plan)
        self.assertNotIn("phone.call", call_plan)
        later = "today: 2026-11-05\ntomorrow: 2026-11-06\ncurrent_time: 2026-11-05T08:00:00+07:00\n"
        self.assertEqual(_spoken_time(call_back, later), "2026-11-06T20:30:00+07:00")

    def test_alarm_before_an_event_uses_the_event_start(self) -> None:
        one_hour = "Đặt báo thức cho chị một tiếng trước lịch họp team ngày 19 nhé."
        repaired = _repair_plan(
            "Đối tượng: Chị | việc: báo thức | khi: luôn | tool: alarm.create",
            one_hour,
            {"alarm.create", "calendar.list_events"},
            {"calendar.list_events": {"kind": "read"}, "alarm.create": {"kind": "write"}},
        )
        self.assertIn("calendar.list_events", repaired)
        self.assertIn("alarm.create", repaired)
        args = _prepare_args(
            "alarm.create",
            {"time": "2026-11-19T06:40:00+07:00"},
            one_hour,
            "today: 2026-11-18\ntomorrow: 2026-11-19\ncurrent_time: 2026-11-18T07:00:00+07:00\n",
            [{"events": [
                {"id": "evt_01", "title": "đi chơi công viên", "start": "2026-11-20T18:30:00+07:00"},
                {"id": "evt_02", "title": "họp team", "start": "2026-11-19T10:30:00+07:00"},
            ]}],
            {},
            repaired,
            [],
        )
        self.assertEqual(args["time"], "2026-11-19T09:30:00+07:00")
        two_hours = "Đặt giúp chị báo thức trước lịch họp team mai 2 tiếng nhé."
        later = _prepare_args(
            "alarm.create",
            {"time": "2026-11-08T09:00:00+07:00"},
            two_hours,
            "today: 2026-11-07\ntomorrow: 2026-11-08\ncurrent_time: 2026-11-07T09:00:00+07:00\n",
            [{"events": [
                {"id": "evt_02", "title": "đá bóng", "start": "2026-11-08T09:30:00+07:00"},
                {"id": "evt_01", "title": "họp team", "start": "2026-11-08T19:30:00+07:00"},
            ]}],
            {},
            "",
            [],
        )
        self.assertEqual(later["time"], "2026-11-08T17:30:00+07:00")

    def test_bank_transfer_reads_balance_and_the_payee(self) -> None:
        self.assertEqual(_trieu_amount("hai triệu tám"), 2800000)
        self.assertEqual(_trieu_amount("một triệu hai"), 1200000)
        installed = {
            "banking.get_balance", "banking.list_beneficiaries", "banking.list_transactions",
            "banking.transfer", "banking.open_savings", "bill_payment.list_bills",
            "contacts.search", "ewallet.topup_mobile", "ewallet.transfer",
        }
        both = _repair_plan(
            "Đối tượng: Phạm Hùng | việc: chuyển | khi: luôn | tool: banking.list_transactions, banking.transfer",
            "Bạn ơi, chuyển cho chủ nhà Phạm Hùng hai triệu tám và nạp giúp sếp Dũng hai trăm nghìn nhé.",
            installed,
            {},
        )
        self.assertIn("banking.get_balance", both)
        self.assertIn("banking.list_beneficiaries", both)
        self.assertIn("banking.transfer", both)
        self.assertIn("contacts.search", both)
        self.assertIn("ewallet.topup_mobile", both)
        self.assertNotIn("list_transactions", both)
        self.assertNotIn("ewallet.transfer", both)
        bill = _repair_plan(
            "Đối tượng: chú Phạm Hùng | việc: chuyển | khi: balance<amount | tool: banking.list_transactions, banking.transfer, banking.open_savings",
            "Chuyển tiền mạng tháng này cho chú Phạm Hùng nha, nếu tài khoản thanh toán không đủ thì lấy từ tiết kiệm dùm tôi hen.",
            installed,
            {},
        )
        self.assertIn("bill_payment.list_bills", bill)
        self.assertIn("khi: luôn", bill)
        self.assertNotIn("balance<amount", bill)
        self.assertNotIn("open_savings", bill)
        tv = _repair_plan(
            "Đối tượng: anh Trần Văn Tuấn | việc: chuyển | khi: luôn | tool: banking.list_transactions, banking.transfer\n"
            "Đối tượng: người nhận | việc: chuyển ví | khi: luôn | tool: contacts.search, ewallet.transfer",
            "Chuyển dùm tui cho anh Trần Văn Tuấn số tiền truyền hình tháng này nha.",
            installed,
            {},
        )
        self.assertIn("bill_payment.list_bills", tv)
        self.assertNotIn("ewallet.transfer", tv)
        args = _prepare_args(
            "banking.transfer",
            {"from_account_id": "acc_01", "amount": 1},
            "Chuyển tiền mạng tháng này cho chú Phạm Hùng nha, nếu tài khoản thanh toán không đủ thì lấy từ tiết kiệm dùm tôi hen.",
            "",
            [
                {"bills": [{"type": "internet", "amount": 218000, "paid": False}]},
                {"accounts": [
                    {"id": "acc_01", "name": "Tài khoản thanh toán", "balance": 112425},
                    {"id": "acc_02", "name": "Tài khoản tiết kiệm", "balance": 50000000},
                ]},
                {"beneficiaries": [
                    {"id": "ben_02", "name": "Chủ nhà - Phạm Hùng", "bank": "An Phát Bank"},
                ]},
            ],
            {},
            bill,
            [],
        )
        self.assertEqual(args["amount"], 218000)
        self.assertEqual(args["from_account_id"], "acc_02")
        self.assertEqual(args["beneficiary_id"], "ben_02")

    def test_floor_lights_skip_a_room_with_the_air_on(self) -> None:
        text = "Tắt hết đèn tầng 2 dùm chị, phòng nào đang bật máy lạnh thì để nguyên nha."
        repaired = _repair_plan(
            "Đối tượng: đèn tầng 2 | việc: tắt hết đèn | khi: luôn | tool: smart_home.get_status, smart_home.light_set",
            text,
            {"smart_home.get_status", "smart_home.light_set", "smart_home.climate_set"},
            {"smart_home.get_status": {"kind": "read"}, "smart_home.light_set": {"kind": "write"}},
        )
        self.assertEqual(repaired.count("smart_home.light_set"), 2)
        devices = {"devices": [
            {"id": "light_bedroom", "type": "light", "room": "phòng ngủ", "floor": 2, "state": {"power": "on"}},
            {"id": "ac_bedroom", "type": "climate", "room": "phòng ngủ", "floor": 2, "state": {"mode": "cool"}},
            {"id": "light_office", "type": "light", "room": "phòng làm việc", "floor": 2, "state": {"power": "on"}},
            {"id": "light_bedroom2", "type": "light", "room": "phòng ngủ nhỏ", "floor": 2, "state": {"power": "on"}},
            {"id": "light_living", "type": "light", "room": "phòng khách", "floor": 1, "state": {"power": "on"}},
        ]}
        first = _prepare_args("smart_home.light_set", {}, text, "", [devices], {}, repaired, [])
        self.assertEqual(first, {"device_id": "light_office", "power": "off"})
        second = _prepare_args(
            "smart_home.light_set", {}, text, "", [devices], {}, repaired,
            [("smart_home.light_set", first)],
        )
        self.assertEqual(second, {"device_id": "light_bedroom2", "power": "off"})

    def test_topup_for_the_named_person_not_minh(self) -> None:
        repaired = _repair_plan(
            "Đối tượng: mình | việc: nạp | khi: luôn | tool: ewallet.topup_mobile",
            "Ơi, nạp giúp mình 50 nghìn điện thoại cho bác Ba với chuyển ví cho cô Mai số tiền điện tháng này nhé.",
            {"bill_payment.list_bills", "contacts.search", "ewallet.topup_mobile", "ewallet.transfer"},
            {},
        )
        self.assertIn("Bác Ba", repaired)
        self.assertNotIn("Mình", repaired)
        args = _prepare_args(
            "bill_payment.list_bills",
            {"type": "phone"},
            "Chuyển tiền mạng tháng này cho chú Phạm Hùng nha.",
            "",
            [],
            {},
            "",
            [],
        )
        self.assertEqual(args, {"type": "internet"})

    def test_topup_and_bill_transfer_stay_on_the_wallet(self) -> None:
        installed = {
            "bill_payment.list_bills", "contacts.search", "ewallet.topup_mobile",
            "ewallet.transfer", "banking.get_balance", "banking.list_beneficiaries",
            "banking.transfer",
        }
        repaired = _repair_plan(
            "Đối tượng: cô Mai | việc: chuyển khoản | khi: luôn | tool: banking.transfer",
            "Nạp cho anh Tuấn năm mươi nghìn và chuyển tiền điện tháng này cho cô Mai nhé.",
            installed,
            {},
        )
        self.assertIn("ewallet.topup_mobile", repaired)
        self.assertIn("ewallet.transfer", repaired)
        self.assertIn("Tuấn", repaired)
        self.assertIn("Cô Mai", repaired)
        self.assertNotIn("banking.transfer", repaired)
        amount = _prepare_args(
            "ewallet.transfer",
            {"amount": 50000},
            "Nạp cho anh Tuấn năm mươi nghìn và chuyển tiền điện tháng này cho cô Mai nhé.",
            "",
            [{"bills": [{"type": "electricity", "amount": 845000, "paid": False}]}],
            {"amount": 50000},
            repaired,
            [],
        )
        self.assertEqual(amount["amount"], 845000)

    def test_a_plain_call_is_the_phone_and_two_names_ask(self) -> None:
        installed = {"alarm.create", "phone.call", "contacts.search", "chat_ott.list_chats", "chat_ott.call"}
        alarm = _repair_plan(
            "Đối tượng: Minh | việc: gọi | khi: luôn | tool: chat_ott.list_chats, chat_ott.call",
            "Đặt báo thức 6 rưỡi sáng mai với gọi cho Minh giúp tôi nhé.",
            installed,
            {},
        )
        self.assertIn("alarm.create", alarm)
        self.assertIn("phone.call", alarm)
        self.assertNotIn("chat_ott", alarm)
        call = _repair_plan(
            "Đối tượng: Lan | việc: gọi | khi: luôn | tool: phone.call",
            "Ờ à, gọi cho Lan nhé.",
            installed,
            {},
        )
        self.assertIn("contacts.search", call)
        self.assertIn("phone.call", call)
        asked = _phone_override(
            [{"contacts": [
                {"id": "c_05", "name": "Lan"},
                {"id": "c_10", "name": "Lan"},
            ]}],
            "Lan",
        )
        self.assertEqual(asked, ("sys.ask_missing_info", {"tool_name": "phone.call", "params": ["contact_id"]}))
        one = _phone_override([{"contacts": [{"id": "c_01", "name": "Cô giáo Mai"}]}], "Cô Mai")
        self.assertEqual(one, ("phone.call", {"contact_id": "c_01"}))
        father = _repair_plan(
            "Đối tượng: Tôi | việc: gọi | khi: luôn | tool: phone.call",
            "Gọi dùm tôi Bố nha.",
            {"contacts.search", "phone.call"},
            {},
        )
        self.assertIn("Đối tượng: Bố", father)
        self.assertNotIn("Tôi", father)
        rainy = _repair_plan(
            "Đối tượng: Minh | việc: gọi | khi: luôn | tool: phone.call",
            "Nếu mai mưa thì huỷ lịch đá bóng sáng mai với gọi cho bạn Minh nhé, còn không mưa thì thôi ạ.",
            {"weather.get_forecast", "calendar.list_events", "calendar.cancel_event", "contacts.search", "phone.call"},
            {},
        )
        self.assertLess(rainy.find("calendar.cancel_event"), rainy.find("phone.call"))
        self.assertIn("khi: rain=true | tool: contacts.search, phone.call", rainy)

    def test_a_short_wallet_pays_the_bill_from_the_bank(self) -> None:
        repaired = _repair_plan(
            "Đối tượng: tiền điện | việc: thanh toán | khi: luôn | tool: ewallet.pay_qr",
            "Trợ lý ơi, xem giúp tôi tiền điện tháng này bao nhiêu rồi thanh toán bằng ví, nếu thiếu thì trừ luôn qua ngân hàng nhé.",
            {"bill_payment.list_bills", "ewallet.get_balance", "ewallet.pay_qr", "bill_payment.pay", "banking.get_balance"},
            {},
        )
        self.assertIn("bill_payment.list_bills", repaired)
        self.assertIn("ewallet.get_balance", repaired)
        self.assertIn("balance>=amount", repaired)
        self.assertIn("balance<amount", repaired)
        self.assertIn("source=bank", repaired)
        self.assertNotIn("pay_qr", repaired)

    def test_wallet_shortfall_pays_by_wallet_or_by_bank(self) -> None:
        installed = {
            "bill_payment.list_bills", "ewallet.get_balance", "bill_payment.pay",
            "contacts.search", "sms.send", "reminders.create", "banking.transfer",
        }
        enough = _repair_plan(
            "Đối tượng: ví | việc: trả | khi: balance<amount | tool: banking.transfer",
            "Ơi, giúp tôi thanh toán tiền điện bằng ví nhé, à mà ví không đủ thì trừ qua ngân hàng giùm luôn nha.",
            installed,
            {},
        )
        self.assertIn("balance>=amount | tool: bill_payment.pay", enough)
        self.assertIn("source=bank", enough)
        self.assertIn("balance<amount | tool: bill_payment.pay", enough)
        self.assertNotIn("banking.transfer", enough)
        short = _repair_plan(
            "Đối tượng: ví | việc: trả | khi: luôn | tool: bill_payment.pay",
            "Nếu ví đủ thì trả tiền nước bằng ví và nhắn SMS cho cô giáo Mai là đóng rồi, còn thiếu thì trả bằng ngân hàng và 8 giờ tối nay nhắc tôi nạp ví.",
            installed,
            {},
        )
        self.assertIn("balance>=amount | tool: contacts.search, sms.send", short)
        self.assertIn("balance<amount | tool: bill_payment.pay", short)
        self.assertIn("balance<amount | tool: reminders.create", short)
        self.assertNotIn("balance<amount | tool: contacts.search", short)

    def test_due_bills_under_500_thousand_are_paid_from_the_bank(self) -> None:
        text = "Mấy hoá đơn đến hạn trong năm ngày tới thì trả bằng ngân hàng hết, mà cái nào trên 500 nghìn thì để lại đó chưa trả nhé."
        context = "today: 2026-11-21\n"
        bills = {"bills": [
            {"id": "bill_01", "type": "electricity", "amount": 920000, "due_date": "2026-11-23", "paid": False},
            {"id": "bill_02", "type": "water", "amount": 210000, "due_date": "2026-11-24", "paid": False},
            {"id": "bill_03", "type": "internet", "amount": 220000, "due_date": "2026-11-22", "paid": False},
            {"id": "bill_04", "type": "tv", "amount": 150000, "due_date": "2026-11-30", "paid": False},
        ]}
        plan = _repair_plan("Đối tượng: hóa đơn | việc: trả | khi: luôn | tool: bill_payment.pay", text, {"bill_payment.list_bills", "bill_payment.pay"}, {})
        self.assertEqual(plan.count("bill_payment.pay"), 4)
        first = _prepare_args("bill_payment.pay", {}, text, context, [bills], {}, plan, [])
        self.assertEqual(first, {"bill_id": "bill_02", "source": "bank"})
        second = _prepare_args(
            "bill_payment.pay", {}, text, context, [bills], {}, plan,
            [("bill_payment.pay", {"bill_id": "bill_02", "source": "bank"})],
        )
        self.assertEqual(second, {"bill_id": "bill_03", "source": "bank"})

    def test_two_transfers_share_savings_when_payment_cannot_cover_both(self) -> None:
        text = (
            "Chuyển cho chủ nhà Phạm Hùng đúng tiền truyền hình tháng này, "
            "cho Trần Văn Tuấn đúng tiền điện tháng này, trừ tài khoản thanh toán, "
            "mà không đủ cả hai thì lấy tiết kiệm."
        )
        results = [{
            "bills": [
                {"id": "bill_tv", "type": "tv", "amount": 150000, "paid": False},
                {"id": "bill_el", "type": "electricity", "amount": 847000, "paid": False},
            ],
            "accounts": [
                {"id": "acc_01", "name": "Thanh toán", "balance": 880418},
                {"id": "acc_02", "name": "Tiết kiệm", "balance": 50000000},
            ],
            "beneficiaries": [
                {"id": "ben_01", "name": "Chủ nhà - Phạm Hùng"},
                {"id": "ben_03", "name": "Trần Văn Tuấn"},
            ],
        }]
        plan = "Đối tượng: chuyển khoản | việc: chuyển | khi: luôn | tool: banking.transfer, banking.transfer"
        first = _prepare_args("banking.transfer", {}, text, "", results, {}, plan, [])
        second = _prepare_args(
            "banking.transfer", {}, text, "", results, {}, plan,
            [("banking.transfer", first)],
        )
        self.assertEqual(first["from_account_id"], "acc_02")
        self.assertEqual(first["beneficiary_id"], "ben_01")
        self.assertEqual(first["amount"], 150000)
        self.assertEqual(second["beneficiary_id"], "ben_03")
        self.assertEqual(second["amount"], 847000)
        self.assertEqual(second["from_account_id"], "acc_02")

    def test_a_running_fan_keeps_its_light_and_the_air_turns_on(self) -> None:
        text = "Tắt hết đèn tầng hai giúp tôi, trừ phòng nào đang chạy quạt thì cứ để đèn đó, rồi bật điều hoà phòng ngủ 25 độ."
        devices = {"devices": [
            {"id": "light_bedroom", "type": "light", "room": "phòng ngủ", "floor": 2, "state": {"power": "on"}},
            {"id": "ac_bedroom", "type": "climate", "room": "phòng ngủ", "floor": 2, "state": {"mode": "off"}},
            {"id": "light_office", "type": "light", "room": "phòng làm việc", "floor": 2, "state": {"power": "on"}},
            {"id": "fan_office", "type": "fan", "room": "phòng làm việc", "floor": 2, "state": {"power": "on"}},
            {"id": "light_bedroom2", "type": "light", "room": "phòng ngủ nhỏ", "floor": 2, "state": {"power": "on"}},
        ]}
        plan = _repair_plan(
            "Đối tượng: đèn | việc: tắt | khi: luôn | tool: smart_home.light_set",
            text,
            {"smart_home.get_status", "smart_home.light_set", "smart_home.climate_set"},
            {},
        )
        self.assertIn("smart_home.climate_set", plan)
        first = _prepare_args("smart_home.light_set", {}, text, "", [devices], {}, plan, [])
        second = _prepare_args(
            "smart_home.light_set", {}, text, "", [devices], {}, plan,
            [("smart_home.light_set", first)],
        )
        climate = _prepare_args("smart_home.climate_set", {}, text, "", [devices], {}, plan, [])
        self.assertEqual(first["device_id"], "light_bedroom")
        self.assertEqual(second["device_id"], "light_bedroom2")
        self.assertEqual(climate, {"device_id": "ac_bedroom", "mode": "cool", "temperature": 25})

    def test_a_ride_picks_up_before_the_event_and_reminds_before_that(self) -> None:
        text = (
            "Đặt xe bốn chỗ từ công ty tới chỗ hẹn cà phê với gia đình chiều nay, "
            "đón trước giờ hẹn nửa tiếng, nhắc tôi mang laptop trước lúc xe đón mười lăm phút."
        )
        context = "today: 2026-11-27\ntomorrow: 2026-11-28\nwork_address: Tòa nhà Kim Cương, Nam Từ Liêm, Hà Nội\n"
        events = {"events": [
            {"id": "evt_01", "title": "cà phê với gia đình", "start": "2026-11-27T15:30:00+07:00", "location": "Cà phê Giảng, 39 Nguyễn Hữu Huân, Hoàn Kiếm, Hà Nội"},
            {"id": "evt_02", "title": "họp phụ huynh", "start": "2026-11-28T08:00:00+07:00", "location": "Sân bóng Mini"},
        ]}
        plan = _repair_plan(
            "Đối tượng: xe | việc: đặt | khi: luôn | tool: ride_hailing.book",
            text,
            {"calendar.list_events", "ride_hailing.book", "reminders.create"},
            {},
        )
        self.assertIn("calendar.list_events", plan)
        self.assertIn("reminders.create", plan)
        book = _prepare_args("ride_hailing.book", {}, text, context, [events], {}, plan, [])
        reminder = _prepare_args("reminders.create", {}, text, context, [events], {}, plan, [])
        self.assertEqual(book["schedule_time"], "2026-11-27T15:00:00+07:00")
        self.assertEqual(book["dropoff"], "Cà phê Giảng, 39 Nguyễn Hữu Huân, Hoàn Kiếm, Hà Nội")
        self.assertEqual(book["pickup"], "Tòa nhà Kim Cương, Nam Từ Liêm, Hà Nội")
        self.assertEqual(book["vehicle"], "car4")
        self.assertEqual(reminder["time"], "2026-11-27T14:45:00+07:00")
        asked = _phone_override(
            [{"contacts": [{"id": "c_04", "name": "Hương"}, {"id": "c_08", "name": "Hương"}]}],
            "Hương",
            "sms.send",
        )
        self.assertEqual(asked, ("sys.ask_missing_info", {"tool_name": "sms.send", "params": ["contact_id"]}))


class ArgumentFixes(unittest.TestCase):
    """The 28 rows that called the right tool with the wrong arguments."""

    def test_wallet_transfer_uses_the_bill_and_the_payee(self) -> None:
        import agent

        text = "Lấy ví nạp 50 nghìn tiền điện thoại cho Sếp Dũng, còn Cô giáo Mai thì chuyển ví đúng bằng tiền mạng tháng này nha."
        self.assertEqual(agent._transfer_payee(text), "Cô Giáo Mai")
        bills = {"bills": [
            {"id": "bill_phone", "type": "phone", "amount": 50000, "paid": False},
            {"id": "bill_net", "type": "internet", "amount": 219000, "paid": False},
        ]}
        plan = "Đối tượng: Cô Giáo Mai | việc: chuyển ví | khi: luôn | tool: contacts.search, ewallet.transfer"
        amount = _prepare_args(
            "ewallet.transfer", {}, text, "", [bills, {"contacts": [{"id": "c_07", "name": "Cô giáo Mai"}]}],
            {}, plan, [("contacts.search", {"query": "Cô Giáo Mai"})],
        )
        self.assertEqual(amount["amount"], 219000)
        self.assertEqual(amount["contact_id"], "c_07")
        accounts = [
            {"id": "acc_01", "name": "Tài khoản thanh toán", "balance": 77780},
            {"id": "acc_02", "name": "Tài khoản tiết kiệm", "balance": 50000000},
        ]
        self.assertEqual(
            agent._pick_account(accounts, 215000, "thiếu thì dùng tiết kiệm"),
            "acc_02",
        )

    def test_relative_clocks_and_promised_due(self) -> None:
        import agent

        move = "Mai em bị muộn, cho lịch khám răng lùi lại nửa tiếng nhé, báo thức thì đặt trước giờ mới một tiếng."
        self.assertEqual(agent._shift_minutes(move), 30)
        context = (
            "today: 2026-10-01\ntomorrow: 2026-10-02\ncurrent_time: 2026-10-01T09:00:00+07:00\n"
            "history:\nuser: Hẹn rồi, chậm nhất năm giờ chiều thứ hai tuần sau, tức là mùng năm tháng mười.\n"
            "latest_request:\n"
        )
        self.assertEqual(
            _spoken_time("Dạ thế thêm cho em một việc, hạn năm giờ chiều ngày mười chín tháng mười.", context),
            "2026-10-19T17:00:00+07:00",
        )
        self.assertEqual(agent._due_from_history(context), "2026-10-05T17:00:00+07:00")
        afternoon = (
            "today: 2026-10-13\ncurrent_time: 2026-10-13T15:10:00+07:00\n"
            "history:\nassistant: Chiều nay có một nhắc lúc năm giờ: đón Minh ở bể bơi Láng Hạ.\n"
            "latest_request:\n"
        )
        self.assertEqual(_spoken_time("Năm giờ nhắc chị gọi Minh nhé.", afternoon), "2026-10-13T17:00:00+07:00")
        tuesday = "today: 2026-11-01\ntomorrow: 2026-11-02\ncurrent_time: 2026-11-01T14:30:00+07:00\n"
        self.assertEqual(
            _spoken_time("tám giờ tối thứ ba nhắc anh share thời tiết.", tuesday),
            "2026-11-03T20:00:00+07:00",
        )
        self.assertEqual(_prepare_args(
            "home_automation.create_rule",
            {"time": "2026-09-25T22:00:00+07:00"},
            "Cài tự động đi, cứ mười giờ tối là cho nó bật lên.",
            "history:\nuser: Tối nào tôi cũng phải bật cái quạt phòng ngủ.\nlatest_request:\n",
            [{"devices": [{"id": "fan_bedroom", "type": "fan", "name": "quạt phòng ngủ", "room": "phòng ngủ"}]}],
            {}, "", [],
        )["time"], "22:00")

    def test_ids_rooms_and_parents(self) -> None:
        import agent

        subs = {"subscriptions": [
            {"id": "sub_422", "name": "ViTube Premium", "next_billing": "2026-11-26"},
            {"id": "sub_3432", "name": "Gym Sunrise Fitness", "next_billing": "2026-11-27"},
        ]}
        self.assertEqual(
            agent._subscription_to_cancel([subs], "gói mô gia hạn vô thứ Sáu tới thì hủy", "today: 2026-11-22\n")["id"],
            "sub_3432",
        )
        bikes = {"bikes": [
            {"id": "bike_476", "name": "Xe đạp điện 112", "station": "Trạm Hàng Bạc", "available": True, "price_per_hour": 25000},
            {"id": "bike_7305", "name": "Xe đạp thường 112", "station": "Trạm Hàng Bạc", "available": True, "price_per_hour": 10000},
            {"id": "bike_9038", "name": "Xe đạp thường 121", "station": "Trạm Hàng Bạc", "available": True, "price_per_hour": 10000},
        ]}
        held = agent._bike_to_reserve(
            [bikes],
            "Giữ cho tôi cái xe đạp thường số một trăm mười hai ở trạm ngay phố nhà tôi nhé.",
            "home_address: 31 Hàng Bạc, Hoàn Kiếm, Hà Nội\n",
        )
        self.assertEqual(held["id"], "bike_7305")
        station = {"bikes": [
            {"id": "bike_3158", "name": "Xe đạp thường 315", "station": "Trạm Mỹ Đình", "available": False, "price_per_hour": 12000},
            {"id": "bike_3182", "name": "Xe đạp điện 318", "station": "Trạm Mỹ Đình", "available": True, "price_per_hour": 30000},
            {"id": "bike_3114", "name": "Xe đạp điện 311", "station": "Trạm Mỹ Đình", "available": True, "price_per_hour": 25000},
        ]}
        cheap = agent._bike_to_reserve(
            [station],
            "Giữ giùm em một chiếc xe đạp thường ở trạm Mỹ Đình, hết thì lấy chiếc rẻ nhất nha.",
            "",
        )
        self.assertEqual(cheap["id"], "bike_3114")
        rooms = agent._robot_rooms(
            [{"rooms": [
                {"name": "bếp"}, {"name": "phòng ngủ"}, {"name": "phòng khách"},
            ]}],
            "Cho robot vừa hút vừa lau phòng khách với bếp giùm anh, phòng ngủ khỏi nha.",
        )
        self.assertEqual(rooms, ["phòng khách", "bếp"])
        book = agent._book_to_reserve(
            [{"books": [
                {"id": "book_7730", "title": "Phối màu căn hộ nhỏ", "author": "Amber Interiors"},
                {"id": "book_2914", "title": "Nhà gỗ tối giản", "author": "Amber Interiors"},
            ]}],
            "giữ chỗ cuốn của Amber Interiors mà Trang nhờ mượn",
            [],
            "history:\nuser: Trang gửi tiền mượn hộ cuốn Nhà gỗ tối giản\nlatest_request:\n",
        )
        self.assertEqual(book["id"], "book_2914")
        item = agent._shopping_item(
            "Ừ, thêm đúng loại đó vào danh sách đi chợ cho em, hai hộp nha.",
            "history:\nuser: Rồi, lấy sữa yến mạch không đường hiệu Oat Farm nha.\nlatest_request:\n",
        )
        self.assertEqual(item, ("sữa yến mạch không đường Oat Farm", 2))
        draft = _prepare_args(
            "email.create_draft", {},
            "Huỷ cái gói Kho Ảnh Gia Đình giúp em với, xong soạn sẵn một thư nháp gửi bố mẹ em.",
            "",
            [{"contacts": [
                {"id": "c_03", "name": "Mẹ", "relation": "mẹ", "email": "mai.pham@example.test"},
                {"id": "c_04", "name": "Bố", "relation": "bố", "email": "hung.pham@example.test"},
            ]}],
            {}, "", [],
        )
        self.assertEqual(draft["to"], ["mai.pham@example.test", "hung.pham@example.test"])
        self.assertEqual(draft["subject"], "Gói Kho Ảnh Gia Đình không gia hạn nữa")


class ExtraReads(unittest.TestCase):
    """A sentence that points at history does not read that list again."""

    def test_known_package_and_false_calendar_writes(self) -> None:
        context = (
            "today: 2026-09-24\ntomorrow: 2026-09-25\n"
            "current_time: 2026-09-24T09:00:00+07:00\n"
            "history:\n"
            "assistant: Mùng 1 tháng 10 có gói Podcast Plus.\n"
            "tool_calls: "
            + json.dumps([{"name": "subscriptions.list", "arguments": {}}], ensure_ascii=False)
            + "\ntool_results: "
            + json.dumps([{
                "subscriptions": [
                    {"id": "sub_04", "name": "Xemflix gói Tiêu chuẩn", "price": 220000, "status": "paused"},
                    {"id": "sub_05", "name": "Podcast Plus", "price": 49000, "status": "active"},
                ],
            }], ensure_ascii=False)
            + "\nlatest_request:\n"
        )
        text = "Thế thì hủy luôn gói đấy cho anh, dạo này anh có nghe mấy đâu."
        plan = _repair_plan("", text, {"subscriptions.list", "subscriptions.cancel", "calendar.list_events", "calendar.cancel_event"}, {}, "", context)
        self.assertIn("subscriptions.cancel", plan)
        self.assertNotIn("subscriptions.list", plan)
        self.assertNotIn("calendar.cancel_event", plan)
        chosen = _prepare_args("subscriptions.cancel", {}, text, context, [], {}, "", [])
        self.assertEqual(chosen["subscription_id"], "sub_05")

    def test_early_without_a_duration_is_not_a_move(self) -> None:
        text = "Báo mọi người là gói lưu trữ hủy rồi, ai cần thì tải dữ liệu về sớm nhé."
        self.assertNotIn("calendar.update_event", _repair_plan(
            "", text,
            {"calendar.list_events", "calendar.update_event", "subscriptions.cancel", "email.create_draft"},
            {},
        ))

    def test_paid_bill_and_returned_chore_do_not_list_bills(self) -> None:
        paid = "Thêm cho anh một việc đối soát hóa đơn nước vừa trả, hạn năm giờ chiều ngày mười một tháng mười."
        self.assertNotIn("bill_payment.list_bills", _repair_plan(
            "", paid, {"bill_payment.list_bills", "tasks.add"}, {},
        ))
        chore = "Việc voucher đặt hoa mới là của Na, còn bánh kem thì trả lại cho ba."
        self.assertNotIn("bill_payment.list_bills", _repair_plan(
            "", chore, {"bill_payment.list_bills", "household_chores.assign", "household_chores.list_chores"}, {},
        ))

    def test_morning_tomorrow_ignores_an_earlier_today(self) -> None:
        context = "today: 2026-11-11\ntomorrow: 2026-11-12\ncurrent_time: 2026-11-11T10:15:00+07:00\n"
        text = "Nếu giá vàng SJC mua vào hôm nay dưới 119 triệu thì nhắc chị 9 giờ sáng mai đi mua vàng nhé."
        self.assertEqual(_spoken_time(text, context), "2026-11-12T09:00:00+07:00")

    def test_airport_ride_does_not_read_the_calendar(self) -> None:
        text = "Sáng hôm đó em gọi sẵn xe bốn chỗ từ nhà ra sân bay Nội Bài, đón lúc bốn giờ bốn lăm."
        plan = _repair_plan("", text, {"ride_hailing.book", "calendar.list_events"}, {}, "")
        self.assertIn("ride_hailing.book", plan)
        self.assertNotIn("calendar.list_events", plan)

    def test_two_chats_ask_instead_of_sending(self) -> None:
        import agent
        text = "Nhắn cho Nam giúp em là họp xong em gọi lại nhé."
        plan = _repair_plan("", text, {"chat_ott.list_chats", "chat_ott.send_message", "phone.call", "contacts.search"}, {}, "")
        self.assertIn("chat_ott.list_chats", plan)
        self.assertNotIn("phone.call", plan)
        self.assertNotIn("contacts.search", plan)
        self.assertTrue(agent._ambiguous_chat(
            [{"chats": [{"id": "chat_01", "name": "Nam"}, {"id": "chat_02", "name": "Nam"}]}],
            text,
            "",
        ))

    def test_each_card_except_one_redeems_the_cheapest(self) -> None:
        import agent
        payload = {"programs": [
            {"id": "prog_183", "name": "Cao Nguyên Rewards", "points": 650, "rewards": [
                {"id": "rw_02_1", "name": "Ly cà phê", "cost": 300},
                {"id": "rw_02_2", "name": "Voucher", "cost": 500},
            ]},
            {"id": "prog_4471", "name": "JetXanh Club", "points": 3100, "rewards": [
                {"id": "rw_05_1", "name": "Vé", "cost": 1500},
            ]},
            {"id": "prog_5520", "name": "CineViệt Membership", "points": 350, "rewards": [
                {"id": "rw_04_2", "name": "Combo", "cost": 400},
            ]},
        ]}
        text = "Thẻ nào đủ điểm thì đổi phần quà rẻ nhất, mỗi thẻ một phần, trừ thẻ JetXanh ra."
        queue = agent._loyalty_queue([payload], text, "")
        self.assertEqual([(item[0]["id"], item[1]["id"]) for item in queue], [("prog_183", "rw_02_1")])

    def test_the_object_of_the_verb_decides_the_tool(self) -> None:
        calendar = {"calendar.list_events", "calendar.cancel_event", "calendar.create_event", "calendar.update_event"}
        cancel = _repair_plan("", "Hủy giúp cho tôi cái lịch đi chơi nhé.", calendar, {})
        self.assertIn("calendar.cancel_event", cancel)
        later = _repair_plan("", "Hủy đi. Rồi ghi vào lịch ngày gói hết hạn.", calendar, {})
        self.assertNotIn("calendar.cancel_event", later)
        package = {"subscriptions.list", "subscriptions.cancel"}
        asked = _repair_plan("", "Hủy cho tôi cái gói xem phim này.", package, {})
        self.assertIn("subscriptions.cancel", asked)
        told = _repair_plan("", "Gói lưu trữ hủy rồi.", package, {})
        self.assertNotIn("subscriptions.cancel", told)
        moved = _repair_plan("", "Dời lịch sớm một chút giúp tôi.", calendar, {})
        self.assertIn("calendar.update_event", moved)
        early = _repair_plan("", "Ai cần thì tải về sớm nhé.", calendar, {})
        self.assertNotIn("calendar.update_event", early)
        bills = {"bill_payment.list_bills"}
        self.assertIn("bill_payment.list_bills", _repair_plan("", "Trả hộ tiền điện tháng này.", bills, {}))
        self.assertNotIn("bill_payment.list_bills", _repair_plan("", "Việc này trả lại cho anh.", bills, {}))
        self.assertNotIn("bill_payment.list_bills", _repair_plan("", "Trả về đúng chỗ cũ giúp tôi.", bills, {}))
        phone = {"phone.call", "contacts.search", "chat_ott.send_message", "chat_ott.list_chats"}
        note = _repair_plan("", "Nhắn cho Nam là chiều nay gọi lại cho Nam nhé.", phone, {})
        self.assertNotIn("phone.call", note)
        call = _repair_plan("", "Gọi cho Nam giúp tôi.", phone, {})
        self.assertIn("phone.call", call)
        ride = {"ride_hailing.book", "calendar.list_events", "subscriptions.cancel"}
        kept = _repair_plan("", "Hủy cho tôi cái gói, rồi gọi xe ra bệnh viện.", ride, {})
        self.assertIn("ride_hailing.book", kept)
        self.assertNotIn("calendar.list_events", kept)
        dropped = _repair_plan("", "Hủy xe hộ tôi, khỏi gọi nữa.", ride, {})
        self.assertNotIn("ride_hailing.book", dropped)
        airport = _repair_plan("", "Gọi xe ra sân bay Nội Bài sáng nay.", {"ride_hailing.book", "calendar.list_events"}, {})
        self.assertNotIn("calendar.list_events", airport)

    def test_a_name_runs_until_the_spoken_boundary(self) -> None:
        import agent

        self.assertEqual(agent._transfer_payee("Cô Mai Hoa thì chuyển ví giúp tôi."), "Cô Mai Hoa")
        self.assertEqual(_contact_query(agent._transfer_payee("Cô Mai Hoa thì chuyển ví giúp tôi.")), "Mai Hoa")
        self.assertEqual(
            agent._transfer_payee("không thì chuyển 2 triệu từ tài khoản thanh toán"),
            "người nhận",
        )
        self.assertEqual(
            agent._transfer_payee(
                "ví mà không đủ thì chuyển qua ngân hàng, rồi nhắn SMS cho sếp Dũng báo tôi đóng xong rồi."
            ),
            "người nhận",
        )
        woke = _repair_plan("", "Gọi giúp mẹ dậy lúc sáu giờ.", {"phone.call", "contacts.search", "alarm.create"}, {})
        self.assertNotIn("phone.call", woke)
        plain = _repair_plan("", "Gọi mẹ dậy giúp tôi.", {"phone.call", "contacts.search"}, {})
        self.assertNotIn("phone.call", plain)
        self.assertIn("phone.call", _repair_plan("", "Gọi cho Nam giúp tôi.", {"phone.call", "contacts.search"}, {}))
        self.assertEqual(agent._spoken_dropoff("Gọi xe sân bay Tân Sơn Nhất nhé."), "sân bay Tân Sơn Nhất")
        self.assertEqual(
            agent._spoken_dropoff("từ nhà ra sân bay Nội Bài, đón lúc bốn giờ."),
            "sân bay Nội Bài",
        )
        rooms = agent._robot_rooms(
            [{"rooms": [{"name": "bếp"}, {"name": "phòng ngủ"}, {"name": "phòng khách"}]}],
            "phòng khách phòng ngủ của bé thì khỏi",
        )
        self.assertEqual(rooms, ["phòng khách"])

    def test_an_empty_label_does_not_keep_a_write(self) -> None:
        gold = (
            "history:\nassistant: Vàng SJC hôm nay mua vào 130.8 triệu.\n"
            "tool_calls: " + json.dumps([{"name": "gold_price.get_price", "arguments": {"type": "sjc"}}]) + "\n"
            "tool_results: [{}]\nlatest_request:\n"
        )
        high = _repair_plan(
            "",
            "Nếu giá mua vào vàng SJC hôm nay dưới 129 triệu thì nhắc mình 9 giờ sáng mai đi mua vàng nhé.",
            {"gold_price.get_price", "reminders.create"},
            {},
            "",
            gold,
        )
        self.assertNotIn("reminders.create", high)
        low_price = gold.replace("130.8", "120")
        under = _repair_plan(
            "",
            "Nếu giá mua vào vàng SJC hôm nay dưới 129 triệu thì nhắc mình 9 giờ sáng mai đi mua vàng nhé.",
            {"gold_price.get_price", "reminders.create"},
            {},
            "",
            low_price,
        )
        self.assertIn("reminders.create", under)
        self.assertNotIn("ewallet.transfer", _repair_plan(
            "", "À khoan, thôi khỏi chuyển cho Minh nữa nhé.",
            {"contacts.search", "ewallet.transfer"}, {},
        ))
        self.assertNotIn("ride_hailing.book", _repair_plan(
            "", "Ơ thôi bạn ơi, khỏi đặt xe nữa nha.",
            {"ride_hailing.book"}, {},
        ))
        self.assertNotIn("shopping_list.add", _repair_plan(
            "", "Khoan khoan, đừng thêm gì vào danh sách đi chợ nữa nha.",
            {"shopping_list.add"}, {},
        ))
        self.assertIn("shopping_list.add", _repair_plan(
            "", "Thêm sữa vào danh sách đi chợ giúp em.",
            {"shopping_list.add"}, {},
        ))
        self.assertNotIn("shopping_list.add", _repair_plan(
            "", "Thêm tập hai bộ phim vào danh sách xem giúp chị.",
            {"shopping_list.add", "tv.guide"}, {},
        ))
        self.assertNotIn("light_set", _repair_plan(
            "", "Ơ, đặt độ sáng cái đèn bàn thành âm mười phần trăm giúp em với.",
            {"smart_home.get_status", "smart_home.light_set"}, {},
        ))
        self.assertIn("light_set", _repair_plan(
            "", "Đặt độ sáng cái đèn bàn thành mười phần trăm giúp em.",
            {"smart_home.get_status", "smart_home.light_set"}, {},
        ))
        self.assertNotIn("chat_ott.send_message", _repair_plan(
            "", "Ờ coi tin Zalo của Ba à không, tin nhắn SMS Ba mới gửi á, đọc em nghe coi.",
            {"chat_ott.list_chats", "chat_ott.send_message"}, {},
        ))
        done = (
            "history:\nassistant: Mình đã chuyển một trăm tám mươi nghìn cho Đỗ Thu Trang, nội dung tiền điện tháng này.\n"
            "latest_request:\n"
        )
        self.assertNotIn("banking.transfer", _repair_plan(
            "", "Ủa tiền điện cho Trang tớ chuyển chưa nhỉ? Chưa thì chuyển luôn đi nhé.",
            {"banking.get_balance", "banking.list_beneficiaries", "banking.transfer"},
            {}, "", done,
        ))
        rainy = _repair_plan(
            "", "Mai mà mưa thì dời lịch chạy bộ, còn không mưa thì thôi.",
            {"weather.get_forecast", "calendar.list_events", "calendar.update_event"}, {},
        )
        self.assertIn("calendar.update_event", rainy)

    def test_a_finished_gold_step_does_not_keep_a_later_write(self) -> None:
        high = (
            "history:\nassistant: Vàng SJC hôm nay mua vào 124.7 triệu.\n"
            "tool_calls: " + json.dumps([{"name": "gold_price.get_price", "arguments": {"type": "sjc"}}]) + "\n"
            "tool_results: [{}]\nlatest_request:\n"
        )
        sentence = "Nếu giá mua vào vàng SJC hôm nay dưới 123 triệu thì 9 giờ sáng mai nhắc tôi đi mua nhé."
        installed = {"gold_price.get_price", "reminders.create"}
        self.assertNotIn("reminders.create", _repair_plan("", sentence, installed, {}, "", high))
        under = _repair_plan("", sentence, installed, {}, "", high.replace("124.7", "118"))
        self.assertIn("reminders.create", under)
        args = _prepare_args("gold_price.get_price", {}, sentence, "", [], {}, "", [])
        self.assertEqual(args.get("type"), "sjc")
        robot = "Con robot còn đang chạy dở hả em? Nếu xong rồi thì cô nhờ cho nó hút cả nhà, chưa xong thì cứ kệ nó."
        plan = _repair_plan("", robot, {"robot_vacuum.get_status", "robot_vacuum.start"}, {})
        self.assertIn("status!=cleaning", plan)
        objects = _parse_objects(plan)
        kind, _obj, tool = _cursor(objects, [("robot_vacuum.get_status", {})], {"status": "cleaning"})
        self.assertEqual(kind, "done")
        self.assertNotEqual(tool, "robot_vacuum.start")
        _kind, _obj, tool = _cursor(objects, [("robot_vacuum.get_status", {})], {"status": "docked"})
        self.assertEqual(tool, "robot_vacuum.start")
        follow = _repair_plan(
            "", "Xong cho con robot dọn phòng ngủ luôn.",
            {"robot_vacuum.get_status", "robot_vacuum.start"}, {},
        )
        self.assertIn("robot_vacuum.start", follow)
        self.assertNotIn("status!=cleaning", follow)
        self.assertNotIn("chat_ott.send_message", _repair_plan(
            "", "Tối ni có ai nhắn Zalo cho anh không rứa?",
            {"chat_ott.list_chats", "chat_ott.send_message"}, {},
        ))
        self.assertNotIn("library.reserve", _repair_plan(
            "", "Thế còn cuốn tiếng Anh Tuấn dặn, xem còn thì giữ nốt cho tôi, không thì thôi.",
            {"library.search", "library.reserve"}, {},
        ))
        self.assertIn("library.reserve", _repair_plan(
            "", "Thằng Bi cần cuốn tám mươi ngày, coi thư viện còn thì giữ, bị mượn rồi thì giữ cuốn khác cùng tác giả.",
            {"library.search", "library.reserve"}, {},
        ))
        balance = _repair_plan(
            "", "Em coi nhanh tài khoản ngân hàng còn bao nhiêu giùm chị, lát chị chuyển cọc cho Laura Costa.",
            {"banking.get_balance", "banking.list_beneficiaries", "banking.transfer"}, {},
        )
        self.assertNotIn("banking.transfer", balance)
        self.assertIn("banking.get_balance", balance)

    def test_a_second_action_stays_on_the_plan(self) -> None:
        climate = _repair_plan(
            "", "Bật đèn phòng khách lên năm mươi phần trăm, điều hoà hai mươi sáu độ.",
            {"smart_home.get_status", "smart_home.light_set", "smart_home.climate_set"}, {},
        )
        self.assertIn("smart_home.climate_set", climate)
        transfers = _repair_plan(
            "", "Lấy tài khoản thanh toán trả Nguyễn Thị Lan đúng tiền nước, Phạm Hùng đúng tiền truyền hình.",
            {"bill_payment.list_bills", "banking.get_balance", "banking.list_beneficiaries", "banking.transfer"},
            {},
        )
        self.assertEqual(transfers.count("banking.transfer"), 2)
        phone = _repair_plan(
            "", "Nhắc tôi uống thuốc, xong gọi luôn số không chín không hai không không không không hai bảy không.",
            {"reminders.create", "phone.call", "contacts.search"}, {},
        )
        self.assertIn("reminders.create", phone)
        self.assertIn("phone.call", phone)
        workout = _repair_plan(
            "", "Hôm nay anh tập yoga được bốn lăm phút, ghi vào luôn hộ anh.",
            {"fitness.log_workout", "reminders.create"}, {},
        )
        self.assertIn("fitness.log_workout", workout)

    def test_a_remaining_alarm_uses_history_and_not_a_new_list(self) -> None:
        from agent import _history_invocations
        from context import render
        from utils import _fill_history_text

        context = "\n".join([
            "today: 2026-11-13",
            "tomorrow: 2026-11-14",
            "history:",
            "user: Mai tôi có những báo thức nào vậy?",
            "assistant: Mai có 2 báo thức: 06:15 và 07:00.",
            'tool_calls: [{"name": "alarm.list", "arguments": {}}]',
            'tool_results: [{"ok": true, "alarms": ['
            '{"id": "alm_01", "time": "2026-11-14T06:15:00+07:00"}, '
            '{"id": "alm_02", "time": "2026-11-14T07:00:00+07:00"}]}]',
            "user: Xóa cái sớm nhất giúp tôi nhé.",
            "assistant: Đã xoá báo thức 06:15.",
            'tool_calls: [{"name": "alarm.delete", "arguments": {"alarm_id": "alm_01"}}]',
            'tool_results: [{"ok": true}]',
            "latest_request:",
            "",
        ])
        request = "Cái còn lại nếu mai mưa thì dời muộn nửa tiếng, không mưa giữ nguyên."
        installed = {
            "weather.get_forecast", "alarm.list", "alarm.delete", "alarm.update",
            "calendar.list_events", "calendar.update_event",
        }
        catalog = {
            "alarm.update": {
                "kind": "write",
                "parameters": {"properties": {"alarm_id": {"type": "string", "x-ref": "alarm.alarms"}}},
            },
            "alarm.list": {"kind": "read", "parameters": {"properties": {}}},
            "calendar.update_event": {
                "kind": "write",
                "parameters": {"properties": {"event_id": {"type": "string", "x-ref": "calendar.events"}}},
            },
            "calendar.list_events": {"kind": "read", "parameters": {"properties": {}}},
        }
        plan = _repair_plan("", request, installed, catalog, "", context)
        self.assertIn("weather.get_forecast", plan)
        self.assertIn("alarm.update", plan)
        self.assertIn("rain=true", plan)
        self.assertNotIn("alarm.list", plan)
        self.assertNotIn("calendar.update_event", plan)
        forecast = _prepare_args("weather.get_forecast", {}, request, context, [], {}, plan, [])
        self.assertEqual(forecast.get("date"), "2026-11-14")
        shifted = _prepare_args("alarm.update", {}, request, context, [], {}, plan, [])
        self.assertEqual(shifted.get("alarm_id"), "alm_02")
        self.assertEqual(shifted.get("time"), "2026-11-14T07:30:00+07:00")

        public = ROOT / "train+public_test" / "public_test" / "public_test.jsonl"
        with public.open(encoding="utf-8") as handle:
            sample = next(json.loads(line) for line in handle if '"id": "public_0006"' in line)
        settings = {
            "cache_dir": str(ROOT / "cache" / "gold" / "transcripts"),
            "kit_root": str(ROOT / "train+public_test"),
            "split": "public_test",
        }
        filled = _fill_history_text(sample, settings)
        rendered = render(filled, "Cái còn lại nếu mai mưa thì dời muộn nửa tiếng, không mưa thì giữ nguyên.")
        self.assertIn("user: Mai mình có những báo thức nào thế?", rendered)
        self.assertIn("user: Xoá cái sớm nhắc giúp tôi nhé.", rendered)
        self.assertNotIn("public_0006_u1.wav", rendered)
        names = [name for name, _args, _payload in _history_invocations(rendered)]
        self.assertEqual(names, ["alarm.list", "alarm.delete"])
        public_plan = _repair_plan("", rendered.split("latest_request:", 1)[1], installed, catalog, "", rendered)
        public_shift = _prepare_args(
            "alarm.update", {}, rendered.split("latest_request:", 1)[1], rendered, [], {}, public_plan, [],
        )
        self.assertEqual(public_shift.get("alarm_id"), "alm_01")
        self.assertEqual(public_shift.get("time"), "2026-11-04T08:30:00+07:00")


if __name__ == "__main__":
    unittest.main()
