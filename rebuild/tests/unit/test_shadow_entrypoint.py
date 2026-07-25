"""
Step6-2b（Shadow起動配線）のテスト。

検証対象:
  parse_target_race / build_bundle / run_shadow / main

方針: 実API・実Legacyへ接続せず、Fake/MockをDIして検証する。
起動配線が「呼ぶだけ」で計算・判定を持たないことを確認する。
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from actions.shadow_entrypoint import (
    main,
    parse_target_race,
    run_shadow,
)
from actions.wiring import PipelineBundle, assemble_pipelines
from notification.service import NotificationService
from shadow.notifier import NullNotifier


class TestParseTargetRace(unittest.TestCase):
    def test_valid_format(self) -> None:
        self.assertEqual(
            parse_target_race("20260704_12_5"), ("20260704", 12, 5)
        )

    def test_zero_padded_values(self) -> None:
        self.assertEqual(
            parse_target_race("20260704_01_01"), ("20260704", 1, 1)
        )

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            parse_target_race("")
        self.assertIn("TARGET_RACE is required", str(ctx.exception))

    def test_no_default_supplied(self) -> None:
        """未指定時に既定レースを補完しないこと。"""
        with self.assertRaises(ValueError) as ctx:
            parse_target_race("   ")
        self.assertIn("no default value is supplied", str(ctx.exception))

    def test_wrong_part_count_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_target_race("20260704_12")

    def test_invalid_date_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_target_race("2026_12_5")


# ---------------- build_bundle / run_shadow（Fake注入） ----------------


def _fake_bundle(prediction_raises: bool = True) -> PipelineBundle:
    """実API非依存のFake部品でPipelineBundleを組む。"""
    from core.buyscore import BuyAssessment
    from models.evaluation import FeatureSet, Prediction, RaceEvaluation
    from models.race import Race, RaceEntry

    def _race(date, venue, rno):
        return Race(
            race_date=date, venue_num=venue, venue_name="住之江",
            race_number=rno, close_time="15:00", is_night=True,
            entries=(RaceEntry(
                lane=1, racer_no="4001", racer_name="A", racer_class="A1",
                win_rate=6.5, place_rate=0.0, motor_no=0, motor_rate2=38.0,
                avg_st=0.16,
            ),),
            grade="一般",
        )

    class _Source:
        def resolve_race(self, d, v, r):
            return _race(d, v, r)

        def resolve_boats(self, d, v, r):
            return [{"lane": i} for i in range(1, 7)]

    class _Builder:
        def build(self, race, inputs, built_at):
            return FeatureSet(
                eval_id=race.eval_id, feature_schema_version=1,
                built_at=built_at, boat_features={1: {}}, race_features={},
                local_features=None, missing_keys=(),
            )

    class _Engine:
        def evaluate(self, race, feature_set, weather, config, now):
            return RaceEvaluation(
                eval_id=race.eval_id, race_date=race.race_date,
                venue_num=race.venue_num, venue_name=race.venue_name,
                race_number=race.race_number, is_night=race.is_night,
                engine_name="ver4", engine_version="4.0.0",
                feature_schema_version=1, features=feature_set,
                model_version="m", evaluated_at="t", danger_score=10.0,
                danger_breakdown={}, upset_score=5.0, upset_reasons=(),
                rank_index={}, featured_boats=None, win_probs=None,
                race_type="", match_index=52.5,
            )

    class _PredictionProvider:
        def provide(self, evaluation, config):
            if prediction_raises:
                raise NotImplementedError(
                    "PredictionContext resolver is not wired yet "
                    f"(reached eval_id={evaluation.eval_id})"
                )
            return Prediction(
                eval_id=evaluation.eval_id, pred_combo="1-2-3",
                pred_prob=0.06, pred_ev=2.0, pred_odds=35.0,
                confidence=0.5, why_bet="x", patterns=(),
            )

    class _BuyEngine:
        def assess(self, evaluation, prediction, config):
            return BuyAssessment(
                eval_id=evaluation.eval_id, buyscore=70.0,
                investment_type="堅実", kelly_fraction=0.0,
                skip_reason=None, config_version="t",
            )

    return assemble_pipelines(
        race_source=_Source(),
        feature_builder=_Builder(),
        engine=_Engine(),
        now_provider=lambda: datetime(2026, 7, 21, tzinfo=timezone.utc),
        eval_config={"_version": "t"},
        durable_store=None,
        prediction_provider=_PredictionProvider(),
        buy_engine=_BuyEngine(),
        buy_config={"_version": "t"},
        output_renderers={},
        notification_service=NotificationService({
            c: NullNotifier(c) for c in ("mail", "line", "discord", "x")
        }),
    )


class TestRunShadow(unittest.TestCase):
    def test_reaches_prediction_boundary(self) -> None:
        """Step6-2bの成功条件: PredictionContext以降まで到達すること。"""
        with self.assertRaises(NotImplementedError) as ctx:
            run_shadow("20260704", 12, 5, bundle=_fake_bundle())
        self.assertIn("PredictionContext", str(ctx.exception))
        self.assertIn("20260704_12_05", str(ctx.exception))

    def test_completes_when_prediction_wired(self) -> None:
        """PredictionProviderが結線されていれば完走すること。"""
        report = run_shadow(
            "20260704", 12, 5, bundle=_fake_bundle(prediction_raises=False)
        )
        self.assertEqual(report["eval_id"], "20260704_12_05")
        self.assertIn("diff_count", report)

    def test_no_real_send(self) -> None:
        """NullNotifier固定で実送信が発生しないこと。"""
        bundle = _fake_bundle(prediction_raises=False)
        run_shadow("20260704", 12, 5, bundle=bundle)
        notifiers = bundle.notification_pipeline._service._notifiers
        for notifier in notifiers.values():
            self.assertIsInstance(notifier, NullNotifier)
            self.assertEqual(notifier.call_count, 0)


class TestMain(unittest.TestCase):
    def test_missing_target_race_returns_1(self) -> None:
        import os

        saved = os.environ.pop("TARGET_RACE", None)
        try:
            self.assertEqual(main([]), 1)
        finally:
            if saved is not None:
                os.environ["TARGET_RACE"] = saved

    def test_invalid_target_race_returns_1(self) -> None:
        import os

        saved = os.environ.get("TARGET_RACE")
        os.environ["TARGET_RACE"] = "invalid"
        try:
            self.assertEqual(main([]), 1)
        finally:
            if saved is None:
                os.environ.pop("TARGET_RACE", None)
            else:
                os.environ["TARGET_RACE"] = saved


class TestBundleStructure(unittest.TestCase):
    def test_bundle_has_all_pipelines(self) -> None:
        bundle = _fake_bundle()
        self.assertTrue(hasattr(bundle.evaluation_pipeline, "evaluate_race"))
        self.assertTrue(hasattr(bundle.buy_pipeline, "assess_race"))
        self.assertTrue(hasattr(bundle.output_pipeline, "render_all"))
        self.assertTrue(hasattr(bundle.notification_pipeline, "send_all"))

    def test_only_null_notifiers_registered(self) -> None:
        bundle = _fake_bundle()
        notifiers = bundle.notification_pipeline._service._notifiers
        self.assertEqual(set(notifiers.keys()), {"mail", "line", "discord", "x"})
        for notifier in notifiers.values():
            self.assertIsInstance(notifier, NullNotifier)


if __name__ == "__main__":
    unittest.main()
