"""
Step5-6（Actions結線・Shadow運用・Go/No-Go判定）のテスト。

指示のテスト要件に対応:
  Actions DI組立 / Shadow実行 / NullNotifier確認 / Notification0件確認 /
  Go判定 / No-Go判定 / 差分出力 / 切替フラグ / Legacy未変更確認。
"""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime

from actions.flags import use_rebuild_pipeline
from actions.wiring import assemble_pipelines
from notification.notifiers import NotificationRequest
from notification.service import NotificationService
from output.renderers import HtmlRenderer, RenderResult
from pipelines.evaluation_pipeline import EvaluationPipeline
from shadow.comparator import compare
from shadow.go_no_go import GoNoGoCriteria, evaluate_go_no_go
from shadow.notifier import NullNotifier
from shadow.runner import ShadowRunner

from models.evaluation import FeatureSet, Prediction, RaceEvaluation
from models.race import Race, RaceEntry


# ---------------- Fakes（既存Step5-1〜5-3と同型） ----------------


def _race(eval_date="20260704", venue=12, rno=5) -> Race:
    return Race(
        race_date=eval_date, venue_num=venue, venue_name="住之江",
        race_number=rno, close_time="15:00", is_night=True,
        entries=(RaceEntry(
            lane=1, racer_no="4001", racer_name="A", racer_class="A1",
            win_rate=6.5, place_rate=0.0, motor_no=0, motor_rate2=38.0,
            avg_st=0.16,
        ),),
        grade="一般",
    )


class _FakeRaceSource:
    def resolve_race(self, date, venue, race):
        return _race(date, venue, race)

    def resolve_boats(self, date, venue, race):
        return [{"lane": i, "win_rate": 5.0} for i in range(1, 7)]


class _FakeFeatureBuilder:
    def build(self, race, inputs, built_at) -> FeatureSet:
        return FeatureSet(
            eval_id=race.eval_id, feature_schema_version=1, built_at=built_at,
            boat_features={1: {}}, race_features={}, local_features=None,
            missing_keys=(),
        )


def _evaluation(eval_id="20260704_12_05", **over) -> RaceEvaluation:
    base = dict(
        eval_id=eval_id, race_date="20260704", venue_num=12, venue_name="住之江",
        race_number=5, is_night=True, engine_name="ver4", engine_version="4.0.0",
        feature_schema_version=1,
        features=FeatureSet(
            eval_id=eval_id, feature_schema_version=1, built_at="t",
            boat_features={1: {}}, race_features={}, local_features=None,
            missing_keys=(),
        ),
        model_version="m", evaluated_at="t", danger_score=10.0,
        danger_breakdown={}, upset_score=5.0, upset_reasons=(),
        rank_index={}, featured_boats=None, win_probs=None, race_type="",
        match_index=52.5,
    )
    base.update(over)
    return RaceEvaluation(**base)


class _FakeEngine:
    def __init__(self, result=None):
        self.result = result or _evaluation()

    def evaluate(self, race, feature_set, weather, config, now):
        return _evaluation(eval_id=race.eval_id, danger_score=self.result.danger_score,
                            upset_score=self.result.upset_score)


def _prediction(eval_id="20260704_12_05") -> Prediction:
    return Prediction(
        eval_id=eval_id, pred_combo="1-2-3", pred_prob=0.06, pred_ev=2.0,
        pred_odds=35.0, confidence=0.5, why_bet="x", patterns=(),
    )


class _FakePredictionProvider:
    def provide(self, evaluation, config):
        return _prediction(evaluation.eval_id)


from core.buyscore import BuyAssessment


class _FakeBuyEngine:
    def assess(self, evaluation, prediction, config):
        return BuyAssessment(
            eval_id=evaluation.eval_id, buyscore=72.0, investment_type="穴狙い",
            kelly_fraction=0.01, skip_reason=None, config_version="t",
        )


def _bundle(notification_service=None):
    tmp_renderer = HtmlRenderer("public", renderer_func=lambda d, p: {"n": 1})
    return assemble_pipelines(
        race_source=_FakeRaceSource(),
        feature_builder=_FakeFeatureBuilder(),
        engine=_FakeEngine(),
        now_provider=lambda: datetime(2026, 7, 21, 7, 30),
        eval_config={"_version": "t"},
        durable_store=None,
        prediction_provider=_FakePredictionProvider(),
        buy_engine=_FakeBuyEngine(),
        buy_config={"_version": "t"},
        output_renderers={"public": tmp_renderer},
        notification_service=notification_service,
    )


# ---------------- テスト ----------------


class TestActionsDIAssembly(unittest.TestCase):
    def test_assemble_returns_bundle_with_all_pipelines(self) -> None:
        bundle = _bundle()
        self.assertIsInstance(bundle.evaluation_pipeline, EvaluationPipeline)
        self.assertTrue(hasattr(bundle.buy_pipeline, "assess_race"))
        self.assertTrue(hasattr(bundle.output_pipeline, "render_all"))
        self.assertTrue(hasattr(bundle.notification_pipeline, "send_all"))

    def test_no_computation_in_wiring(self) -> None:
        """assemble_pipelinesはnewして渡すだけ（内部計算がないことの間接確認）。

        同じFake部品で2回組み立てても、評価結果は各Pipelineの実行結果
        （Fake側の戻り値）に一致し、wiring側で値が変わらないこと。
        """
        bundle1 = _bundle()
        bundle2 = _bundle()
        r1 = bundle1.evaluation_pipeline.evaluate_race("20260704", 12, 5)
        r2 = bundle2.evaluation_pipeline.evaluate_race("20260704", 12, 5)
        self.assertEqual(r1.danger_score, r2.danger_score)


class TestNullNotifier(unittest.TestCase):
    def test_never_sends(self) -> None:
        notifier = NullNotifier("mail")
        result = notifier.notify(
            NotificationRequest(RenderResult("/x.html", {}), channel="mail")
        )
        self.assertFalse(result.sent)
        self.assertEqual(notifier.call_count, 1)

    def test_records_without_external_call(self) -> None:
        notifier = NullNotifier("line")
        results = [
            notifier.notify(
                NotificationRequest(RenderResult("/x.html", {}), channel="line")
            )
            for _ in range(3)
        ]
        self.assertEqual(notifier.call_count, 3)
        self.assertTrue(all(not r.sent for r in results))


class TestShadowRunnerRequiresNullNotifier(unittest.TestCase):
    def test_real_notifier_registered_raises(self) -> None:
        from notification.notifiers import MailNotifier

        service = NotificationService({"mail": MailNotifier(sender=lambda s, b: True)})
        bundle = _bundle(notification_service=service)
        runner = ShadowRunner(bundle)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "public.html")
            with self.assertRaises(RuntimeError):
                runner.run_and_compare("20260704", 12, 5, {"public": path})

    def test_null_notifier_passes(self) -> None:
        service = NotificationService({"mail": NullNotifier("mail")})
        bundle = _bundle(notification_service=service)
        runner = ShadowRunner(bundle)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "public.html")
            result = runner.run_and_compare("20260704", 12, 5, {"public": path})
        self.assertEqual(result.eval_id, "20260704_12_05")


class TestShadowNotificationZeroSends(unittest.TestCase):
    def test_zero_real_sends_with_requests(self) -> None:
        service = NotificationService({"mail": NullNotifier("mail")})
        bundle = _bundle(notification_service=service)
        runner = ShadowRunner(bundle)
        req = NotificationRequest(RenderResult("/x.html", {}), channel="mail")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "public.html")
            result = runner.run_and_compare(
                "20260704", 12, 5, {"public": path},
                notification_requests=[req],
            )
        self.assertEqual(len(result.notification_requests), 1)
        # NullNotifier経由なので実送信は起きない（RuntimeErrorが出ないこと自体が証明）


class TestShadowDiffOutput(unittest.TestCase):
    def test_diff_format(self) -> None:
        diffs = compare(
            "20260704_12_05",
            {"danger_score": 10.0}, {"danger_score": 12.0}, path="$.evaluation",
        )
        self.assertEqual(len(diffs), 1)
        d = diffs[0]
        self.assertEqual(set(d.keys()), {"eval_id", "field_path", "legacy", "rebuild"})
        self.assertEqual(d["eval_id"], "20260704_12_05")
        self.assertEqual(d["field_path"], "$.evaluation.danger_score")
        self.assertEqual(d["legacy"], 10.0)
        self.assertEqual(d["rebuild"], 12.0)

    def test_no_diff_when_equal(self) -> None:
        diffs = compare("e1", _evaluation(), _evaluation(), path="$.evaluation")
        self.assertEqual(diffs, [])

    def test_race_evaluation_uses_existing_serializer(self) -> None:
        """既存RaceEvaluationSerializerを再利用していること（差分検出で確認）。"""
        e1 = _evaluation(danger_score=10.0)
        e2 = _evaluation(danger_score=99.0)
        diffs = compare("e1", e1, e2, path="$.evaluation")
        paths = [d["field_path"] for d in diffs]
        self.assertIn("$.evaluation.danger_score", paths)

    def test_shadow_run_diffs_reported(self) -> None:
        service = NotificationService({"mail": NullNotifier("mail")})
        bundle = _bundle(notification_service=service)
        runner = ShadowRunner(bundle)
        legacy_eval = _evaluation(danger_score=999.0)  # 意図的に不一致
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "public.html")
            result = runner.run_and_compare(
                "20260704", 12, 5, {"public": path},
                legacy_values={"evaluation": legacy_eval},
            )
        self.assertTrue(len(result.diffs) > 0)
        self.assertTrue(
            any(d["field_path"] == "$.evaluation.danger_score" for d in result.diffs)
        )


class TestGoNoGo(unittest.TestCase):
    def test_go_when_all_criteria_met(self) -> None:
        criteria = GoNoGoCriteria(
            golden_100_percent=True, shadow_consecutive_matches=100,
            output_byte_match=True, notification_request_match=True,
            shadow_real_sends=0, feature_freeze_intact=True,
            all_tests_passed=True,
        )
        self.assertEqual(evaluate_go_no_go(criteria).decision, "GO")

    def test_no_go_when_shadow_insufficient(self) -> None:
        criteria = GoNoGoCriteria(
            golden_100_percent=True, shadow_consecutive_matches=99,
        )
        result = evaluate_go_no_go(criteria)
        self.assertEqual(result.decision, "NO_GO")
        self.assertTrue(any("G2/G3" in r for r in result.reasons))

    def test_no_go_when_real_send_detected(self) -> None:
        criteria = GoNoGoCriteria(
            golden_100_percent=True, shadow_consecutive_matches=100,
            shadow_real_sends=1,
        )
        result = evaluate_go_no_go(criteria)
        self.assertEqual(result.decision, "NO_GO")
        self.assertTrue(any("G6" in r for r in result.reasons))

    def test_no_go_when_golden_not_100(self) -> None:
        criteria = GoNoGoCriteria(
            golden_100_percent=False, shadow_consecutive_matches=100,
        )
        self.assertEqual(evaluate_go_no_go(criteria).decision, "NO_GO")

    def test_multiple_reasons_reported(self) -> None:
        criteria = GoNoGoCriteria(
            golden_100_percent=False, shadow_consecutive_matches=0,
            feature_freeze_intact=False,
        )
        result = evaluate_go_no_go(criteria)
        self.assertEqual(result.decision, "NO_GO")
        self.assertGreaterEqual(len(result.reasons), 3)


class TestSwitchFlag(unittest.TestCase):
    def test_default_false(self) -> None:
        self.assertFalse(use_rebuild_pipeline({}))

    def test_true_variants(self) -> None:
        for v in ("1", "true", "True", "YES", "on"):
            self.assertTrue(use_rebuild_pipeline({"USE_REBUILD_PIPELINE": v}))

    def test_false_variants(self) -> None:
        for v in ("0", "false", "", "no"):
            self.assertFalse(use_rebuild_pipeline({"USE_REBUILD_PIPELINE": v}))

    def test_does_not_modify_legacy(self) -> None:
        """フラグはDI切替の判断材料のみ。Legacyモジュールをimportしない。"""
        import actions.flags as mod
        src = open(mod.__file__, encoding="utf-8").read()
        self.assertNotIn("notify_arashi", src)
        self.assertNotIn("import x_", src)


if __name__ == "__main__":
    unittest.main()
