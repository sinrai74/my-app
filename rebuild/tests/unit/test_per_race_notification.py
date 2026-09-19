"""per-race通知: formatter / message_key / Idempotency / skip のテスト。"""

from __future__ import annotations

import unittest

from models.evaluation import BuyDecision, Prediction, RaceEvaluation
from notification.body_formatter import format_race_body, format_race_subject
from notification.notifiers import NotificationRequest, NotificationResult
from notification.service import NotificationService, build_message_key
from output.renderers import RenderResult
from pipelines.notification_request_builder import build_per_race_mail_request


def _evaluation(**over):
    base = dict(
        eval_id="20260704_01_01", race_date="20260704", venue_num=1,
        venue_name="桐生", race_number=1, is_night=False, engine_name="ver4",
        engine_version="4.0", feature_schema_version=1, model_version="m",
        evaluated_at="t", danger_score=3.5, danger_breakdown={}, upset_score=9.5,
        upset_reasons=("超荒れ",), rank_index={}, featured_boats=None,
        win_probs=None, race_type="1残り荒れ型", match_index=50.0, features=None,
    )
    base.update(over)
    return RaceEvaluation(**base)


def _prediction(**over):
    base = dict(
        eval_id="20260704_01_01", pred_combo="6-1-3", pred_prob=0.04591,
        pred_ev=3.03, pred_odds=66.0, confidence=0.8007,
        why_bet="1号艇平均STやや遅め0.17 / 1号艇勝率低(3.7) / 超荒れ(9.5)",
        patterns=(),
    )
    base.update(over)
    return Prediction(**base)


def _decision(**over):
    base = dict(
        eval_id="20260704_01_01", purchased=True, buyscore=72.0,
        investment_type="通常", n_bets=1, cost=200, kelly_fraction=0.05,
        config_version="v", skip_reason=None, purchased_combos=("6-1-3",),
    )
    base.update(over)
    return BuyDecision(**base)


class TestBodyFormatter(unittest.TestCase):
    def test_normal_body_contains_key_values(self):
        body = format_race_body(_evaluation(), _prediction(), _decision())
        self.assertIn("【桐生 1R】1残り荒れ型", body)
        self.assertIn("波乱度: 9.5", body)          # 小数1桁
        self.assertIn("危険度: 3.5", body)
        self.assertIn("買い目: 6-1-3", body)
        self.assertIn("確率: 4.6%", body)            # %・小数1桁
        self.assertIn("EV: 3.03", body)              # 小数2桁
        self.assertIn("オッズ: 66.00", body)         # 小数2桁
        self.assertIn("信頼度: 0.80", body)          # 小数2桁
        self.assertIn("投資タイプ: 通常", body)
        self.assertIn("BuyScore: 72.00", body)       # 小数2桁
        self.assertIn("Kelly: 0.050", body)          # 小数3桁
        self.assertIn("点数: 1点 / 金額: 200円", body)

    def test_why_bet_used_not_upset_reasons(self):
        body = format_race_body(_evaluation(), _prediction(), _decision())
        self.assertIn("理由: 1号艇平均STやや遅め0.17", body)

    def test_optional_none_shows_missing(self):
        ev = _evaluation(danger_score=None)
        d = _decision(buyscore=None, kelly_fraction=None)
        body = format_race_body(ev, _prediction(), d)
        self.assertIn("危険度: 未記録", body)
        self.assertIn("BuyScore: 未記録", body)
        self.assertIn("Kelly: 未記録", body)

    def test_purchased_combos_all_shown(self):
        d = _decision(n_bets=3, purchased_combos=("6-1-3", "1-6-3", "6-3-1"))
        body = format_race_body(_evaluation(), _prediction(), d)
        self.assertIn("6-1-3", body)
        self.assertIn("1-6-3", body)
        self.assertIn("6-3-1", body)

    def test_subject_format(self):
        subject = format_race_subject(_evaluation())
        self.assertEqual(subject, "【競艇AI】桐生 1R 予想")


class TestMessageKey(unittest.TestCase):
    def _req(self, **over):
        base = dict(
            render_result=RenderResult(output_path="/tmp/x.html", summary={}),
            channel="mail", body="b", race_date="20260713",
            venue_num=12, race_number=5,
        )
        base.update(over)
        return NotificationRequest(**base)

    def test_message_key_format(self):
        key = build_message_key(self._req())
        self.assertEqual(key, "mail:arashi:20260713:12_05")

    def test_target_id_zero_padded(self):
        key = build_message_key(self._req(venue_num=1, race_number=1))
        self.assertEqual(key, "mail:arashi:20260713:01_01")

    def test_none_when_identifiers_missing(self):
        self.assertIsNone(build_message_key(self._req(race_date=None)))
        self.assertIsNone(build_message_key(self._req(venue_num=None)))


class _FakeIdemStore:
    def __init__(self, recorded=None):
        self._recorded = set(recorded or [])
        self.recorded_calls = []

    def is_recorded(self, channel, message_key):
        return (channel, message_key) in self._recorded

    def record(self, channel, message_key):
        self.recorded_calls.append((channel, message_key))
        self._recorded.add((channel, message_key))


class _FakeNotifier:
    channel = "mail"

    def __init__(self, sent=True):
        self._sent = sent
        self.calls = 0

    def notify(self, request):
        self.calls += 1
        return NotificationResult("mail", sent=self._sent)


def _req():
    return NotificationRequest(
        render_result=RenderResult(output_path="/tmp/x.html", summary={}),
        channel="mail", body="body-text", race_date="20260713",
        venue_num=12, race_number=5,
    )


class TestIdempotencyWiring(unittest.TestCase):
    def test_unrecorded_sends_then_records(self):
        store = _FakeIdemStore()
        notifier = _FakeNotifier(sent=True)
        svc = NotificationService({"mail": notifier}, idempotency_store=store)
        result = svc.send(_req())
        self.assertTrue(result.sent)
        self.assertEqual(notifier.calls, 1)
        self.assertEqual(store.recorded_calls, [("mail", "mail:arashi:20260713:12_05")])

    def test_recorded_skips_send(self):
        store = _FakeIdemStore(recorded=[("mail", "mail:arashi:20260713:12_05")])
        notifier = _FakeNotifier(sent=True)
        svc = NotificationService({"mail": notifier}, idempotency_store=store)
        result = svc.send(_req())
        self.assertFalse(result.sent)
        self.assertEqual(notifier.calls, 0)          # 送信されない
        self.assertEqual(store.recorded_calls, [])   # 記録もされない

    def test_send_failure_does_not_record(self):
        store = _FakeIdemStore()
        notifier = _FakeNotifier(sent=False)          # 送信失敗
        svc = NotificationService({"mail": notifier}, idempotency_store=store)
        result = svc.send(_req())
        self.assertFalse(result.sent)
        self.assertEqual(notifier.calls, 1)
        self.assertEqual(store.recorded_calls, [])    # 失敗時はrecordしない

    def test_no_store_backward_compat(self):
        notifier = _FakeNotifier(sent=True)
        svc = NotificationService({"mail": notifier})  # store未注入
        result = svc.send(_req())
        self.assertTrue(result.sent)
        self.assertEqual(notifier.calls, 1)


class TestPerRaceBuilder(unittest.TestCase):
    def test_builds_request_with_body_and_identifiers(self):
        rr = RenderResult(output_path="/tmp/x.html", summary={})
        req = build_per_race_mail_request(
            rr, subject="【競艇AI】桐生 1R 予想", body="本文",
            race_date="20260704", venue_num=1, race_number=1,
        )
        self.assertEqual(req.channel, "mail")
        self.assertEqual(req.title, "【競艇AI】桐生 1R 予想")
        self.assertEqual(req.body, "本文")
        self.assertEqual(req.race_date, "20260704")
        self.assertEqual(req.venue_num, 1)
        self.assertEqual(req.race_number, 1)
        self.assertIsNone(req.destination)


if __name__ == "__main__":
    unittest.main()
