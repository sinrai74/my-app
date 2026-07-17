"""
実績・学習系データモデル（models/record.py）の単体テスト。

設計書 Phase0.5 v1.1 ③3.7〜3.11・3.14・3.15、⑤5.4（純粋性・再現性）に対応。
標準ライブラリ unittest のみを使用する。
"""

from __future__ import annotations

import unittest

from models.evaluation import BuyDecision, FeatureSet, Prediction, RaceEvaluation
from models.record import (
    HitRecord,
    LearningData,
    LocalCourseStats,
    MotorHistory,
    RaceResult,
    VenueStatistics,
    VerificationResult,
)


def _build_sample_feature_set() -> FeatureSet:
    return FeatureSet(
        eval_id="20260713_12_05",
        feature_schema_version=1,
        built_at="2026-07-13T06:00:00+09:00",
        boat_features={1: {"win_rate": 6.50}},
        race_features={"venue_factor": 1.02},
    )


def _build_sample_evaluation() -> RaceEvaluation:
    return RaceEvaluation(
        eval_id="20260713_12_05",
        race_date="20260713",
        venue_num=12,
        venue_name="住之江",
        race_number=5,
        is_night=False,
        engine_name="ver4",
        engine_version="4.2.0",
        feature_schema_version=1,
        model_version="v20260701",
        evaluated_at="2026-07-13T06:05:00+09:00",
        danger_score=78.5,
        danger_breakdown={"win_rate_low": 28},
        upset_score=42.0,
        upset_reasons=("1号艇平均ST遅め",),
        rank_index={"1": 0.80},
        featured_boats={"featured": [1, 4]},
        win_probs={1: 0.42, 2: 0.18, 3: 0.15, 4: 0.12, 5: 0.08, 6: 0.05},
        race_type="本命",
        match_index=44.1,
        features=_build_sample_feature_set(),
    )


def _build_sample_prediction() -> Prediction:
    return Prediction(
        eval_id="20260713_12_05",
        pred_combo="1-2-3",
        pred_prob=0.185,
        pred_ev=1.42,
        pred_odds=7.7,
        confidence=0.72,
        why_bet="1号艇信頼度高",
        patterns=({"combo": "1-2-3", "prob": 0.185},),
    )


def _build_sample_buy_decision(purchased: bool = True) -> BuyDecision:
    return BuyDecision(
        eval_id="20260713_12_05",
        purchased=purchased,
        buyscore=68.5,
        investment_type="通常" if purchased else "見送り",
        n_bets=3 if purchased else 0,
        cost=900 if purchased else 0,
        kelly_fraction=0.05 if purchased else 0.0,
        config_version="12",
        skip_reason=None if purchased else "danger_score閾値未達",
    )


def _build_sample_race_result() -> RaceResult:
    return RaceResult(
        eval_id="20260713_12_05",
        result_combo="1-3-2",
        payout=1970,
        hit=True,
        profit=1070,
    )


class TestRaceResult(unittest.TestCase):
    def test_construct(self) -> None:
        """必須項目でRaceResultが構築できること。"""
        result = _build_sample_race_result()
        self.assertTrue(result.hit)
        self.assertEqual(result.payout, 1970)

    def test_immutable(self) -> None:
        result = _build_sample_race_result()
        with self.assertRaises(Exception):
            result.hit = False  # type: ignore[misc]


class TestHitRecord(unittest.TestCase):
    def test_construct_without_result(self) -> None:
        """結果判明前はresultがNoneで構築できること
        （設計書⑥: BuyDecision確定時に追記、結果判明時に結果列のみ更新）。
        """
        record = HitRecord(
            eval_id="20260713_12_05",
            evaluation=_build_sample_evaluation(),
            prediction=_build_sample_prediction(),
            buy_decision=_build_sample_buy_decision(),
        )
        self.assertIsNone(record.result)
        self.assertEqual(record.evaluation.eval_id, "20260713_12_05")

    def test_construct_with_result(self) -> None:
        """結果判明後はresultを設定して構築できること。"""
        record = HitRecord(
            eval_id="20260713_12_05",
            evaluation=_build_sample_evaluation(),
            prediction=_build_sample_prediction(),
            buy_decision=_build_sample_buy_decision(),
            result=_build_sample_race_result(),
        )
        self.assertIsNotNone(record.result)
        self.assertTrue(record.result.hit)

    def test_aggregates_four_domain_models_by_type(self) -> None:
        """HitRecordがRaceEvaluation/Prediction/BuyDecision/RaceResultを
        型として保持する集約モデルであること（案A'の確認）。
        """
        record = HitRecord(
            eval_id="20260713_12_05",
            evaluation=_build_sample_evaluation(),
            prediction=_build_sample_prediction(),
            buy_decision=_build_sample_buy_decision(),
            result=_build_sample_race_result(),
        )
        self.assertIsInstance(record.evaluation, RaceEvaluation)
        self.assertIsInstance(record.prediction, Prediction)
        self.assertIsInstance(record.buy_decision, BuyDecision)
        self.assertIsInstance(record.result, RaceResult)

    def test_weather_is_optional_and_held_as_historical_record(self) -> None:
        """weatherが任意項目として保持できること（v1.1.3・案W-B）。

        Weatherは歴史的記録の再構築とレガシーCSV互換のためだけに保持され、
        評価の真実源ではない（RaceEvaluationはWeatherから独立を維持）。
        """
        from models.race import Weather

        record = HitRecord(
            eval_id="20260713_12_05",
            evaluation=_build_sample_evaluation(),
            prediction=_build_sample_prediction(),
            buy_decision=_build_sample_buy_decision(),
            weather=Weather(wind_speed_mps=3.0, wind_direction="横", wave_height_cm=3),
        )
        self.assertEqual(record.weather.wind_direction, "横")
        record_no_weather = HitRecord(
            eval_id="20260713_12_05",
            evaluation=_build_sample_evaluation(),
            prediction=_build_sample_prediction(),
            buy_decision=_build_sample_buy_decision(),
        )
        self.assertIsNone(record_no_weather.weather)

    def test_immutable(self) -> None:
        record = HitRecord(
            eval_id="20260713_12_05",
            evaluation=_build_sample_evaluation(),
            prediction=_build_sample_prediction(),
            buy_decision=_build_sample_buy_decision(),
        )
        with self.assertRaises(Exception):
            record.result = _build_sample_race_result()  # type: ignore[misc]


class TestMotorHistory(unittest.TestCase):
    def test_construct_matches_11_columns(self) -> None:
        """現行motor_history.csvの11列と1:1対応すること。"""
        history = MotorHistory(
            date="20260713",
            venue_num=12,
            venue="住之江",
            motor_no=12,
            racer_no="4999",
            racer_name="テスト選手",
            lane=1,
            place=1,
            ex_time=6.75,
            start_timing=0.16,
            race_number=5,
        )
        self.assertEqual(history.motor_no, 12)
        self.assertEqual(history.place, 1)

    def test_immutable(self) -> None:
        history = MotorHistory(
            date="20260713",
            venue_num=12,
            venue="住之江",
            motor_no=12,
            racer_no="4999",
            racer_name="テスト選手",
            lane=1,
            place=1,
            ex_time=6.75,
            start_timing=0.16,
            race_number=5,
        )
        with self.assertRaises(Exception):
            history.place = 2  # type: ignore[misc]


class TestVenueStatistics(unittest.TestCase):
    def test_construct(self) -> None:
        stats = VenueStatistics(
            venue_num=12,
            water_type="淡水",
            course_stats={"1": {"win_rate": 0.55}},
            venue_factor=1.02,
            updated_at="2026-07-13T06:00:00+09:00",
        )
        self.assertEqual(stats.water_type, "淡水")

    def test_immutable(self) -> None:
        stats = VenueStatistics(
            venue_num=12,
            water_type="淡水",
            course_stats={},
            venue_factor=1.02,
            updated_at="2026-07-13T06:00:00+09:00",
        )
        with self.assertRaises(Exception):
            stats.venue_factor = 1.0  # type: ignore[misc]


class TestLocalCourseStats(unittest.TestCase):
    def test_construct_matches_15_columns(self) -> None:
        """現行local_course_stats.csvの15列と1:1対応すること。"""
        stats = LocalCourseStats(
            racer_no="3415",
            venue_code=12,
            venue_name="住之江",
            course=1,
            starts=46,
            first=18,
            second=10,
            third=6,
            fourth=5,
            fifth=4,
            sixth=3,
            first_rate=39.1,
            top2_rate=60.9,
            top3_rate=73.9,
            last_updated="20260713",
        )
        self.assertEqual(stats.starts, 46)
        self.assertEqual(stats.first_rate, 39.1)

    def test_immutable(self) -> None:
        stats = LocalCourseStats(
            racer_no="3415",
            venue_code=12,
            venue_name="住之江",
            course=1,
            starts=46,
            first=18,
            second=10,
            third=6,
            fourth=5,
            fifth=4,
            sixth=3,
            first_rate=39.1,
            top2_rate=60.9,
            top3_rate=73.9,
            last_updated="20260713",
        )
        with self.assertRaises(Exception):
            stats.starts = 47  # type: ignore[misc]


class TestLearningData(unittest.TestCase):
    def test_construct_with_feature_set_type(self) -> None:
        """featuresがFeatureSet型そのものとして保持されること（案X採用の確認）。"""
        learning = LearningData(
            eval_id="20260713_12_05",
            features=_build_sample_feature_set(),
            race_result=_build_sample_race_result(),
            engine_version="4.2.0",
            feature_schema_version=1,
        )
        self.assertIsInstance(learning.features, FeatureSet)
        self.assertIsInstance(learning.race_result, RaceResult)

    def test_immutable(self) -> None:
        learning = LearningData(
            eval_id="20260713_12_05",
            features=_build_sample_feature_set(),
            race_result=_build_sample_race_result(),
            engine_version="4.2.0",
            feature_schema_version=1,
        )
        with self.assertRaises(Exception):
            learning.engine_version = "5.0.0"  # type: ignore[misc]


class TestVerificationResult(unittest.TestCase):
    def test_construct(self) -> None:
        result = VerificationResult(
            race_date="20260713",
            n_races=90,
            n_purchased=12,
            n_hit=4,
            total_cost=3600,
            total_payout=8200,
            roi=1.28,
            by_race_type={"本命": {"n": 60}},
            by_rank={"S": {"n": 10}},
            engine_version="4.2.0",
        )
        self.assertEqual(result.n_races, 90)
        self.assertAlmostEqual(result.roi, 1.28)

    def test_immutable(self) -> None:
        result = VerificationResult(
            race_date="20260713",
            n_races=90,
            n_purchased=12,
            n_hit=4,
            total_cost=3600,
            total_payout=8200,
            roi=1.28,
            by_race_type={},
            by_rank={},
            engine_version="4.2.0",
        )
        with self.assertRaises(Exception):
            result.roi = 0.0  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
