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


# ---------------- Step6-2c-1: boats配線のみ ----------------


class TestContextResolverBoatsOnly(unittest.TestCase):
    """Step6-2c-1: context_resolverがboatsのみ結線し、
    patterns/ml_probs/odds_map欠如で停止することを検証する。

    build_bundleは実Legacy(x_venue_stats等)をimportするため、ここでは
    context_resolverの挙動を単体で再現して検証する（実API非依存）。
    """

    def _resolver_boats_only(self, provider):
        """build_bundle内の_context_resolver_boats_onlyと同じ挙動を再現。"""
        from shadow.prediction_provider import PredictionContext

        def _resolver(evaluation):
            _p, boat_objs = provider._get_source(
                evaluation.race_date, evaluation.venue_num,
                evaluation.race_number,
            )
            return PredictionContext(boats=boat_objs)

        return _resolver

    def _evaluation(self):
        from models.evaluation import FeatureSet, RaceEvaluation

        eid = "20260704_12_05"
        return RaceEvaluation(
            eval_id=eid, race_date="20260704", venue_num=12,
            venue_name="住之江", race_number=5, is_night=True,
            engine_name="ver4", engine_version="4.0.0",
            feature_schema_version=1,
            features=FeatureSet(
                eval_id=eid, feature_schema_version=1, built_at="t",
                boat_features={1: {}}, race_features={}, local_features=None,
                missing_keys=(),
            ),
            model_version="m", evaluated_at="t", danger_score=10.0,
            danger_breakdown={}, upset_score=5.0, upset_reasons=(),
            rank_index={}, featured_boats=None, win_probs=None, race_type="",
            match_index=52.5,
        )

    def test_boats_are_wired_from_provider(self) -> None:
        """boatsがProviderのboat_objsから取得されること。"""
        from types import SimpleNamespace
        from adapters.providers import BoatsProvider

        marker_boats = [SimpleNamespace(lane=i) for i in range(1, 7)]
        program = {"race_stadium_number": 12, "race_number": 5,
                   "race_grade_number": 0, "race_closed_at": "15:00"}
        provider = BoatsProvider(
            programs_fetcher=lambda d: [program],
            boats_extractor=lambda p: marker_boats,
        )
        # PredictionContextの必須引数不足でTypeError（=期待到達点）だが、
        # その手前でboat_objsが正しく取得されることを、直接_get_sourceで確認
        _p, boat_objs = provider._get_source("20260704", 12, 5)
        self.assertIs(boat_objs, marker_boats)

    def test_stops_at_missing_required_args(self) -> None:
        """boats以外の必須引数(patterns/ml_probs/odds_map)欠如で停止すること。

        これはStep6-2c-1の期待到達点（boatsより先へ進んだ証拠）。
        """
        from types import SimpleNamespace
        from adapters.providers import BoatsProvider

        program = {"race_stadium_number": 12, "race_number": 5,
                   "race_grade_number": 0, "race_closed_at": "15:00"}
        provider = BoatsProvider(
            programs_fetcher=lambda d: [program],
            boats_extractor=lambda p: [SimpleNamespace(lane=i) for i in range(1, 7)],
        )
        resolver = self._resolver_boats_only(provider)
        with self.assertRaises(TypeError) as ctx:
            resolver(self._evaluation())
        msg = str(ctx.exception)
        # boatsではなくpatterns/ml_probs/odds_mapが原因であること
        self.assertIn("patterns", msg)
        self.assertIn("ml_probs", msg)
        self.assertIn("odds_map", msg)
        self.assertNotIn("boats", msg)


# ---------------- Step6-2c-2: PredictionContext必須3項目の結線 ----------------


class TestContextResolverRequiredFields(unittest.TestCase):
    """Step6-2c-2: patterns/ml_probs/odds_map（＋同一チェーンのupset_score/
    target_lanes）が結線され、PredictionContextが生成されることを検証する。

    Legツール関数はFakeをmonkeypatchで注入し、実API・実モデルへ接続しない。
    """

    def _evaluation(self):
        from models.evaluation import FeatureSet, RaceEvaluation

        eid = "20260704_12_05"
        return RaceEvaluation(
            eval_id=eid, race_date="20260704", venue_num=12,
            venue_name="住之江", race_number=5, is_night=True,
            engine_name="ver4", engine_version="4.0.0",
            feature_schema_version=1,
            features=FeatureSet(
                eval_id=eid, feature_schema_version=1, built_at="t",
                boat_features={1: {}}, race_features={}, local_features=None,
                missing_keys=(),
            ),
            model_version="m", evaluated_at="t", danger_score=10.0,
            danger_breakdown={}, upset_score=5.0, upset_reasons=(),
            rank_index={}, featured_boats=None, win_probs=None, race_type="",
            match_index=52.5,
        )

    def _build_resolver(self, captured):
        """build_bundle内の_context_resolver_requiredと同じ結線を、
        Legツール関数をFake差し替えで再現する。"""
        from types import SimpleNamespace
        from adapters.providers import BoatsProvider
        from shadow.prediction_provider import PredictionContext

        program = {"race_stadium_number": 12, "race_number": 5,
                   "race_grade_number": 3, "race_closed_at": "15:00"}
        boat_objs = [SimpleNamespace(lane=i) for i in range(1, 7)]
        provider = BoatsProvider(
            programs_fetcher=lambda d: [program],
            boats_extractor=lambda p: boat_objs,
        )

        def _fake_predict(boats):
            captured["ml_probs_input"] = boats
            return {1: 0.5, 2: 0.3}

        def _fake_upset(boats, race_grade, venue_num, is_night, config):
            captured["upset_args"] = dict(
                race_grade=race_grade, venue_num=venue_num, is_night=is_night,
            )
            return (7.5, {"d": 1}, [1, 2])

        def _fake_patterns(target_lanes, upset_score):
            captured["patterns_args"] = (target_lanes, upset_score)
            return {"honmei": [1, 2]}

        def _fake_odds(race_no, venue_code, race_date):
            captured["odds_args"] = (race_no, venue_code, race_date)
            return {"1-2-3": 30.0}

        def _resolver(evaluation):
            _p, bo = provider._get_source(
                evaluation.race_date, evaluation.venue_num,
                evaluation.race_number,
            )
            race_grade = int(_p.get("race_grade_number", 0) or 0)
            ml_probs = _fake_predict(bo)
            upset_score, _detail, target_lanes = _fake_upset(
                bo, race_grade, venue_num=evaluation.venue_num,
                is_night=evaluation.is_night, config={},
            )
            patterns = _fake_patterns(target_lanes, upset_score)
            venue_code = str(evaluation.venue_num).zfill(2)
            odds_map = _fake_odds(
                evaluation.race_number, venue_code, evaluation.race_date
            ) or {}
            return PredictionContext(
                patterns=patterns, ml_probs=ml_probs, odds_map=odds_map,
                boats=bo, upset_score=upset_score, target_lanes=target_lanes,
            )

        return _resolver

    def test_prediction_context_is_created(self) -> None:
        """必須3項目が揃いPredictionContextが生成されること。"""
        captured = {}
        resolver = self._build_resolver(captured)
        ctx = resolver(self._evaluation())
        # 必須3項目が入っている
        self.assertEqual(ctx.patterns, {"honmei": [1, 2]})
        self.assertEqual(ctx.ml_probs, {1: 0.5, 2: 0.3})
        self.assertEqual(ctx.odds_map, {"1-2-3": 30.0})
        # 同一チェーンの成果物も渡っている
        self.assertEqual(ctx.upset_score, 7.5)
        self.assertEqual(ctx.target_lanes, [1, 2])

    def test_race_grade_from_program(self) -> None:
        """race_gradeがprogramのrace_grade_numberから取られること。"""
        captured = {}
        resolver = self._build_resolver(captured)
        resolver(self._evaluation())
        self.assertEqual(captured["upset_args"]["race_grade"], 3)

    def test_patterns_uses_upset_chain(self) -> None:
        """patternsがupset_score/target_lanesの成果物から生成されること。"""
        captured = {}
        resolver = self._build_resolver(captured)
        resolver(self._evaluation())
        target_lanes, upset_score = captured["patterns_args"]
        self.assertEqual(target_lanes, [1, 2])
        self.assertEqual(upset_score, 7.5)

    def test_odds_uses_race_identifiers(self) -> None:
        """odds_mapがレース識別子（zfill venue_code含む）から取られること。"""
        captured = {}
        resolver = self._build_resolver(captured)
        resolver(self._evaluation())
        race_no, venue_code, race_date = captured["odds_args"]
        self.assertEqual(race_no, 5)
        self.assertEqual(venue_code, "12")  # zfill(2)
        self.assertEqual(race_date, "20260704")

    def test_ml_probs_from_boats(self) -> None:
        """ml_probsがboatsから取られること。"""
        captured = {}
        resolver = self._build_resolver(captured)
        resolver(self._evaluation())
        self.assertEqual(len(captured["ml_probs_input"]), 6)
