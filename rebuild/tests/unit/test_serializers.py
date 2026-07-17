"""
Serializer層（storage/serializers/）の単体テスト。

設計書 v1.1.6 ④、Step2実装計画書 §2 に対応。
標準ライブラリ unittest のみを使用する。
"""

from __future__ import annotations

import json
import unittest

from models.evaluation import BuyDecision, FeatureSet, Prediction, RaceEvaluation
from models.output import NewsArticle, RankingEntry, SystemMetrics
from models.record import RaceResult, VerificationResult
from storage.exceptions import ParseError
from storage.serializers import (
    SCHEMA_VERSION,
    BuyDecisionSerializer,
    NewsArticleSerializer,
    PredictionSerializer,
    RaceEvaluationSerializer,
    RaceResultSerializer,
    RankingEntrySerializer,
    SystemMetricsSerializer,
    VerificationResultSerializer,
)


def _feature_set() -> FeatureSet:
    return FeatureSet(
        eval_id="20260713_12_05",
        feature_schema_version=1,
        built_at="2026-07-13T06:00:00+09:00",
        boat_features={1: {"win_rate": 6.50, "motor_rate2": 38.5}},
        race_features={"venue_factor": 1.02},
        missing_keys=("ability_trend",),
    )


def _evaluation() -> RaceEvaluation:
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
        win_probs={1: 0.42, 2: 0.18},
        race_type="本命",
        match_index=44.1,
        features=_feature_set(),
    )


class TestRaceEvaluationSerializer(unittest.TestCase):
    def test_roundtrip(self) -> None:
        model = _evaluation()
        restored = RaceEvaluationSerializer.from_dict(
            RaceEvaluationSerializer.to_dict(model)
        )
        self.assertEqual(model, restored)

    def test_roundtrip_with_nulls(self) -> None:
        """Optional項目（v1.1.4/1.1.6のnull許容）がNoneのまま往復すること。"""
        base = _evaluation()
        model = RaceEvaluation(
            eval_id=base.eval_id,
            race_date=base.race_date,
            venue_num=base.venue_num,
            venue_name=base.venue_name,
            race_number=base.race_number,
            is_night=base.is_night,
            engine_name="legacy",
            engine_version="0.0.0",
            feature_schema_version=0,
            model_version=None,
            evaluated_at="",
            danger_score=None,
            danger_breakdown=None,
            upset_score=9.5,
            upset_reasons=(),
            rank_index=None,
            featured_boats=None,
            win_probs=None,
            race_type="1殴り荒れ型",
            match_index=None,
            features=base.features,
        )
        restored = RaceEvaluationSerializer.from_dict(
            RaceEvaluationSerializer.to_dict(model)
        )
        self.assertEqual(model, restored)
        self.assertIsNone(restored.win_probs)

    def test_schema_version_included(self) -> None:
        """④: 全出力JSONはトップレベルにschema_versionを持つこと。"""
        data = RaceEvaluationSerializer.to_dict(_evaluation())
        self.assertEqual(data["schema_version"], SCHEMA_VERSION)

    def test_output_is_json_compatible(self) -> None:
        """to_dict()の出力がそのままjson.dumps可能であること。"""
        data = RaceEvaluationSerializer.to_dict(_evaluation())
        text = json.dumps(data, ensure_ascii=False)
        self.assertEqual(json.loads(text)["eval_id"], "20260713_12_05")

    def test_win_probs_keys_str_in_json_int_in_model(self) -> None:
        data = RaceEvaluationSerializer.to_dict(_evaluation())
        self.assertIn("1", data["win_probs"])
        restored = RaceEvaluationSerializer.from_dict(data)
        self.assertIn(1, restored.win_probs)

    def test_unknown_keys_ignored(self) -> None:
        """④: 読み手は未知キーを無視する（前方互換）。"""
        data = RaceEvaluationSerializer.to_dict(_evaluation())
        data["future_field_added_in_v2"] = {"x": 1}
        restored = RaceEvaluationSerializer.from_dict(data)
        self.assertEqual(restored, _evaluation())

    def test_missing_required_key_raises(self) -> None:
        data = RaceEvaluationSerializer.to_dict(_evaluation())
        del data["upset_score"]
        with self.assertRaises(ParseError):
            RaceEvaluationSerializer.from_dict(data)


class TestPredictionAndBuyDecision(unittest.TestCase):
    def test_prediction_roundtrip_tuple_list(self) -> None:
        model = Prediction(
            eval_id="20260713_12_05",
            pred_combo="1-2-3",
            pred_prob=0.185,
            pred_ev=1.42,
            pred_odds=7.7,
            confidence=0.72,
            why_bet="1号艇信頼度高",
            patterns=({"combo": "1-2-3"},),
        )
        data = PredictionSerializer.to_dict(model)
        self.assertIsInstance(data["patterns"], list)
        restored = PredictionSerializer.from_dict(data)
        self.assertEqual(model, restored)
        self.assertIsInstance(restored.patterns, tuple)

    def test_buy_decision_roundtrip_with_none_kelly(self) -> None:
        """v1.1.6: kelly_fraction=None（未計算）が往復で保持されること。"""
        model = BuyDecision(
            eval_id="20260713_12_05",
            purchased=False,
            buyscore=None,
            investment_type="",
            n_bets=1,
            cost=0,
            kelly_fraction=None,
            config_version="",
            skip_reason="legacy",
        )
        restored = BuyDecisionSerializer.from_dict(BuyDecisionSerializer.to_dict(model))
        self.assertEqual(model, restored)
        self.assertIsNone(restored.kelly_fraction)

    def test_race_result_roundtrip(self) -> None:
        model = RaceResult(
            eval_id="20260713_12_05",
            result_combo="1-3-2",
            payout=1970,
            hit=True,
            profit=1070,
        )
        restored = RaceResultSerializer.from_dict(RaceResultSerializer.to_dict(model))
        self.assertEqual(model, restored)


class TestRankingEntrySerializer(unittest.TestCase):
    def test_roundtrip_keeps_source_eval_id_as_str(self) -> None:
        """案P維持: source_eval_idがstr参照IDのまま往復すること。"""
        model = RankingEntry(
            ranking_type="danger",
            race_date="20260713",
            venue_name="住之江",
            race_number=5,
            score=78.5,
            rank_label="S",
            subject={"racer_no": "4999"},
            reasons=("平均ST遅め",),
            ai_comment="1号艇に不安あり",
            source_eval_id="20260713_12_05",
        )
        data = RankingEntrySerializer.to_dict(model)
        self.assertIsInstance(data["reasons"], list)
        restored = RankingEntrySerializer.from_dict(data)
        self.assertEqual(model, restored)
        self.assertIsInstance(restored.source_eval_id, str)


class TestSystemMetricsSerializer(unittest.TestCase):
    def test_roundtrip_with_and_without_ai_summary(self) -> None:
        model = SystemMetrics(
            metrics_id="20260713_ranking_daily_run123",
            race_date="20260713",
            job_name="ranking_daily",
            run_id="run123",
            started_at="2026-07-13T07:30:00+09:00",
            finished_at="2026-07-13T07:35:00+09:00",
            duration_seconds=300.0,
            status="success",
            counters={"races_evaluated": 90},
            rates={"api_fetch_success_rate": 1.0},
            errors={},
            schema_version=1,
            ai_summary={"roi": 1.28},
        )
        restored = SystemMetricsSerializer.from_dict(
            SystemMetricsSerializer.to_dict(model)
        )
        self.assertEqual(model, restored)
        # ai_summaryなし（失敗ジョブ等）も往復すること
        model2 = SystemMetrics(
            metrics_id="x",
            race_date="20260713",
            job_name="arashi_watch",
            run_id="r",
            started_at="t1",
            finished_at="t2",
            duration_seconds=1.0,
            status="failed",
            counters={},
            rates={},
            errors={"DataFetchError": 1},
            schema_version=1,
        )
        restored2 = SystemMetricsSerializer.from_dict(
            SystemMetricsSerializer.to_dict(model2)
        )
        self.assertEqual(model2, restored2)


class TestVerificationAndNews(unittest.TestCase):
    def test_verification_roundtrip(self) -> None:
        model = VerificationResult(
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
        restored = VerificationResultSerializer.from_dict(
            VerificationResultSerializer.to_dict(model)
        )
        self.assertEqual(model, restored)

    def test_news_roundtrip(self) -> None:
        model = NewsArticle(
            race_date="20260713",
            sections={"featured_races": [{"eval_id": "20260713_12_05"}]},
            generated_at="2026-07-13T07:30:00+09:00",
            brand_version="1",
        )
        restored = NewsArticleSerializer.from_dict(NewsArticleSerializer.to_dict(model))
        self.assertEqual(model, restored)

    def test_missing_key_raises(self) -> None:
        data = NewsArticleSerializer.to_dict(
            NewsArticle(
                race_date="20260713",
                sections={},
                generated_at="t",
                brand_version="1",
            )
        )
        del data["sections"]
        with self.assertRaises(ParseError):
            NewsArticleSerializer.from_dict(data)


if __name__ == "__main__":
    unittest.main()
