"""
Ver4Engine（core/engine.py）の単体テスト。

Step4-5テスト要件①②③に対応:
  ① EvaluationEngine単体（legacy dict構造・venue無し既定値・boat1欠損）
  ② Prediction単体（Strategy DI委譲・未結線時の明示失敗）
  ③ RaceEvaluation生成（全フィールド写像・正準形・predictor注入）
Golden回帰（④）は tests/regression/test_engine_golden.py。
"""

from __future__ import annotations

import unittest
from datetime import datetime

from core.engine import Ver4Engine, _grade_to_number
from models.evaluation import FeatureSet, Prediction
from models.race import Race
from tests.fakes import (
    FakePredictionStrategy,
    FakePredictor,
    FakeVenueStatsProvider,
)
from tests.unit.test_rank import _boat as _rank_boat
from tests.unit.test_rank import _config as _rank_config


def _boat(lane: int, **overrides) -> dict:
    """rank用_boatにcourse_rank（engineの_featuresが参照）を補完。"""
    merged = {"course_rank": [2.4, 2.7, 2.7, 3.2, 3.3, 4.0]}
    merged.update(overrides)
    return _rank_boat(lane, **merged)
from tests.unit.test_upset import _config as _upset_config


def _config() -> dict:
    """engine用合成config（upset系＋rank/featured系を統合）。"""
    cfg = _upset_config()  # model_version / boat_relative_score / upset_prob / danger最小
    rank_cfg = _rank_config()
    cfg["lane_rank_scores"] = rank_cfg["lane_rank_scores"]
    cfg["featured_racers"] = rank_cfg["featured_racers"]
    cfg["danger_score"] = rank_cfg["danger_score"]  # class_rank込み
    return cfg


def _six_boats() -> list[dict]:
    return [_boat(lane) for lane in range(1, 7)]


def _race(grade: str | None = None) -> Race:
    return Race(
        race_date="20260704",
        venue_num=12,
        venue_name="住之江",
        race_number=5,
        close_time="",
        is_night=False,
        entries=(),
        grade=grade,
    )


def _feature_set() -> FeatureSet:
    return FeatureSet(
        eval_id="20260704_12_05",
        feature_schema_version=1,
        built_at="t",
        boat_features={1: {}},
        race_features={},
        local_features=None,
        missing_keys=(),
    )


def _provider() -> FakeVenueStatsProvider:
    return FakeVenueStatsProvider(
        water_types={
            "住之江": {"type": "in_favorable", "label": "イン有利水面",
                       "course1_win_rate": 55.0, "source": "test", "samples": 100}
        },
        course_factors={"住之江": {"factor": 1.02, "samples": 120}},
    )


def _engine(boats=None, **kwargs) -> Ver4Engine:
    boats = boats if boats is not None else _six_boats()
    defaults = dict(
        boats_resolver=lambda race: boats,
        venue_stats=_provider(),
    )
    defaults.update(kwargs)
    return Ver4Engine(**defaults)


class TestLegacyDictStructure(unittest.TestCase):
    """① EvaluationEngine単体: v4 dictの構造と既定値。"""

    def test_top_level_keys_match_v4(self) -> None:
        legacy = _engine().build_legacy_evaluation(
            _six_boats(), _config(), venue_num=12
        )
        self.assertEqual(
            set(legacy.keys()),
            {
                "model_version", "venue", "venue_num", "boat1",
                "danger_score", "danger_breakdown", "water_type",
                "venue_factor_1c", "rank_index", "featured_boats",
                "upset_score", "upset_detail", "boat1_features",
            },
        )
        self.assertEqual(legacy["venue"], "住之江")
        self.assertIn("_features", legacy["upset_detail"])

    def test_features_dict_has_all_legacy_keys(self) -> None:
        legacy = _engine().build_legacy_evaluation(
            _six_boats(), _config(), venue_num=12
        )
        feats = legacy["upset_detail"]["_features"]
        expected_keys = {
            "win_rate", "motor", "avg_st", "racer_class", "course_st_1c",
            "course_rank_1c", "danger_breakdown", "danger_score_v3",
            "rank_index", "featured_boats", "model_version",
            "venue_water_type", "venue_factor", "ability_trend",
            "course_f_rate_1c", "course_l_rate_1c", "course_rentai2_1c",
            "course_sample_confidence",
        }
        self.assertEqual(set(feats.keys()), expected_keys)
        self.assertEqual(feats["venue_water_type"], "イン有利水面")
        self.assertEqual(feats["venue_factor"], 1.02)
        self.assertEqual(feats["course_sample_confidence"], 120)

    def test_no_venue_uses_transcribed_defaults(self) -> None:
        """venue未指定（venue_num=0）時のwater_type/venue_factor_1c既定dict。"""
        legacy = _engine().build_legacy_evaluation(
            _six_boats(), _config(), venue_num=0
        )
        self.assertEqual(
            legacy["water_type"],
            {"type": "unknown", "label": "不明", "course1_win_rate": 0.0,
             "source": "no_venue", "samples": 0},
        )
        self.assertEqual(
            legacy["venue_factor_1c"],
            {"factor": 1.0, "venue_win_rate": 0.0, "national_win_rate": 0.0,
             "samples": 0, "water_type": "unknown"},
        )
        feats = legacy["upset_detail"]["_features"]
        self.assertEqual(feats["venue_water_type"], "")
        self.assertEqual(feats["venue_factor"], "")

    def test_missing_boat1_yields_empty_strings(self) -> None:
        boats = [_boat(lane) for lane in range(2, 7)]
        legacy = _engine(boats=boats).build_legacy_evaluation(
            boats, _config(), venue_num=12
        )
        self.assertEqual(legacy["boat1"], {"lane": 1, "name": "", "racer_class": ""})
        feats = legacy["upset_detail"]["_features"]
        self.assertIsNone(feats["win_rate"])
        self.assertEqual(feats["ability_trend"], "")

    def test_ability_trend_empty_string_sentinel(self) -> None:
        """ability=0.0はfalsyで空文字センチネル（レガシー挙動保存）。"""
        boats = _six_boats()
        boats[0] = _boat(1, ability_curr=0.0)
        legacy = _engine(boats=boats).build_legacy_evaluation(
            boats, _config(), venue_num=12
        )
        self.assertEqual(legacy["upset_detail"]["_features"]["ability_trend"], "")

    def test_boat1_features_mirror_features(self) -> None:
        legacy = _engine().build_legacy_evaluation(
            _six_boats(), _config(), venue_num=12
        )
        feats = legacy["upset_detail"]["_features"]
        self.assertEqual(
            legacy["boat1_features"],
            {
                "ability_trend": feats["ability_trend"],
                "course_f_rate_1c": feats["course_f_rate_1c"],
                "course_l_rate_1c": feats["course_l_rate_1c"],
                "course_rentai2_1c": feats["course_rentai2_1c"],
                "course_sample_confidence": feats["course_sample_confidence"],
            },
        )


class TestRaceEvaluationConstruction(unittest.TestCase):
    """③ RaceEvaluation生成テスト。"""

    def test_all_fields_mapped(self) -> None:
        predictor = FakePredictor({"20260704_12_05": {1: 0.4, 2: 0.2}})
        engine = _engine(predictor=predictor)
        now = datetime(2026, 7, 18, 7, 30, 0)
        evaluation = engine.evaluate(_race(), _feature_set(), None, _config(), now)

        self.assertEqual(evaluation.eval_id, "20260704_12_05")
        self.assertEqual(evaluation.race_date, "20260704")
        self.assertEqual(evaluation.venue_num, 12)
        self.assertEqual(evaluation.venue_name, "住之江")
        self.assertEqual(evaluation.race_number, 5)
        self.assertFalse(evaluation.is_night)
        self.assertEqual(evaluation.engine_name, "ver4")
        self.assertEqual(evaluation.engine_version, "4.0.0")
        self.assertEqual(evaluation.feature_schema_version, 1)
        self.assertEqual(evaluation.evaluated_at, now.isoformat())
        self.assertEqual(evaluation.win_probs, {1: 0.4, 2: 0.2})
        self.assertEqual(evaluation.upset_reasons, ())
        self.assertEqual(evaluation.race_type, "")
        self.assertIsNotNone(evaluation.danger_score)
        self.assertAlmostEqual(
            evaluation.match_index,
            min(100, evaluation.upset_score * 10.5),
            delta=1e-9,
        )

    def test_rank_index_uses_str_lane_keys(self) -> None:
        evaluation = _engine().evaluate(
            _race(), _feature_set(), None, _config(), datetime(2026, 7, 18)
        )
        self.assertEqual(
            set(evaluation.rank_index.keys()), {"1", "2", "3", "4", "5", "6"}
        )

    def test_featured_boats_wrapped_in_canonical_dict(self) -> None:
        evaluation = _engine().evaluate(
            _race(), _feature_set(), None, _config(), datetime(2026, 7, 18)
        )
        self.assertEqual(set(evaluation.featured_boats.keys()), {"featured"})
        self.assertIsInstance(evaluation.featured_boats["featured"], list)

    def test_no_predictor_yields_none_win_probs(self) -> None:
        evaluation = _engine().evaluate(
            _race(), _feature_set(), None, _config(), datetime(2026, 7, 18)
        )
        self.assertIsNone(evaluation.win_probs)

    def test_grade_string_conversion(self) -> None:
        self.assertEqual(_grade_to_number("SG"), 4)
        self.assertEqual(_grade_to_number("G1"), 3)
        self.assertEqual(_grade_to_number("一般"), 0)
        self.assertEqual(_grade_to_number(None), 0)
        self.assertEqual(_grade_to_number("未知"), 0)

    def test_grade_string_roundtrips_unknown_numbers(self) -> None:
        """race_grade_numberはAPI生値で0-4に限らない。"gradeN"表現は
        Nへ復元できること（Golden回帰でrace_grade=5等が実在するため）。"""
        self.assertEqual(_grade_to_number("grade5"), 5)
        self.assertEqual(_grade_to_number("grade9"), 9)


class TestPrediction(unittest.TestCase):
    """② Prediction単体テスト。"""

    def test_strategy_delegation(self) -> None:
        strategy = FakePredictionStrategy()
        engine = _engine(prediction_strategy=strategy)
        evaluation = engine.evaluate(
            _race(), _feature_set(), None, _config(), datetime(2026, 7, 18)
        )
        prediction = engine.predict(evaluation, None, _config())
        self.assertIsInstance(prediction, Prediction)
        self.assertEqual(prediction.eval_id, evaluation.eval_id)
        self.assertEqual(len(strategy.calls), 1)
        called_eval, called_odds, called_cfg = strategy.calls[0]
        self.assertIs(called_eval, evaluation)
        self.assertIsNone(called_odds)

    def test_unwired_strategy_fails_explicitly(self) -> None:
        """戦略未注入は明示的に失敗する（サイレント失敗禁止⑫）。"""
        engine = _engine()
        evaluation = engine.evaluate(
            _race(), _feature_set(), None, _config(), datetime(2026, 7, 18)
        )
        with self.assertRaises(NotImplementedError):
            engine.predict(evaluation, None, _config())


if __name__ == "__main__":
    unittest.main()
