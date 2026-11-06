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
        _is_speaker,
        _matching_contact,
        _ngan_amount,
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
        _is_speaker,
        _matching_contact,
        _ngan_amount,
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


if __name__ == "__main__":
    unittest.main()
