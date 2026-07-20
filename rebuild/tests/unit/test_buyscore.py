"""
BuyEngine（core/buyscore.py）の単体テスト。

Step4-6指示のテスト要件に対応（Golden回帰と分離）:
  Kelly境界 / EV境界 / investment_type全分岐 / skip_reason全分岐 /
  buy_score境界 / None入力 / 0入力 / clamp / round / 閾値直前・直後。
"""

from __future__ import annotations

import unittest

from core.buyscore import (
    BuyAssessment,
    DefaultBuyEngine,
    DefaultKellyStrategy,
    _odds_band_bonus,
    calc_buyscore,
    check_passthrough,
    investment_type,
    kelly_fraction,
)
from core.exceptions import ValidationError
from models.evaluation import FeatureSet, Prediction, RaceEvaluation


def _config() -> dict:
    return {
        "_version": "test-1.0",
        "weights": {
            "ev": 0.35, "prob": 0.20, "match_index": 0.15, "market_gap": 0.10,
            "uncertainty": -0.10, "disagreement": -0.05, "calibration": 0.10,
            "race_type": 0.05,
        },
        "thresholds": {
            "buyscore_min": 60, "ev_min": 1.28, "match_index_min": 40,
            "uncertainty_max": 0.75, "disagreement_max": 0.80,
            "market_gap_max": 3.0,
        },
        "odds_band_bonus": {
            "1_5": -0.15, "6_12": 0.10, "13_25": 0.10,
            "26_40": 0.05, "41_80": -0.05, "80_over": -0.20,
        },
        "kelly": {"fraction": 0.25, "half_below": 60, "zero_below": 50},
        "star_thresholds": {"5star": 90, "4star": 80, "3star": 70, "2star": 60},
    }


class TestOddsBandBonus(unittest.TestCase):
    def test_band_boundaries(self) -> None:
        cfg = _config()["odds_band_bonus"]
        self.assertEqual(_odds_band_bonus(5.0, cfg), -0.15)   # <=5
        self.assertEqual(_odds_band_bonus(5.01, cfg), 0.10)   # ->6_12
        self.assertEqual(_odds_band_bonus(12.0, cfg), 0.10)
        self.assertEqual(_odds_band_bonus(25.0, cfg), 0.10)
        self.assertEqual(_odds_band_bonus(40.0, cfg), 0.05)
        self.assertEqual(_odds_band_bonus(80.0, cfg), -0.05)
        self.assertEqual(_odds_band_bonus(80.01, cfg), -0.20)


class TestCalcBuyscore(unittest.TestCase):
    def test_clamp_upper_100(self) -> None:
        cand = {"ev": 10.0, "prob": 1.0, "odds": 20.0, "composite": 1.0,
                "uncertainty": 0.0, "disagreement": 0.0}
        ctx = {"match_index": 100, "race_type": "本命戦", "market_gap": 3.0}
        self.assertEqual(calc_buyscore(cand, ctx, _config()), 100.0)

    def test_clamp_lower_0(self) -> None:
        cand = {"ev": 0.0, "prob": 0.0, "odds": 100.0, "composite": 0.0,
                "uncertainty": 1.0, "disagreement": 1.0}
        ctx = {"match_index": 0, "race_type": "混戦", "market_gap": 0.0,
               "has_exhibition": False}
        self.assertEqual(calc_buyscore(cand, ctx, _config()), 0.0)

    def test_round_to_1_decimal(self) -> None:
        cand = {"ev": 1.0, "prob": 0.04, "odds": 15.0, "composite": 0.3,
                "uncertainty": 0.5, "disagreement": 0.5}
        ctx = {"match_index": 50, "race_type": "混戦", "market_gap": 1.0}
        score = calc_buyscore(cand, ctx, _config())
        self.assertEqual(score, round(score, 1))

    def test_missing_keys_use_defaults(self) -> None:
        """空candidate/contextでも.getデフォルトで動く（0入力相当）。"""
        score = calc_buyscore({}, {}, _config())
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, 0.0)

    def test_exhibition_penalty(self) -> None:
        cand = {"ev": 1.5, "prob": 0.05, "odds": 15.0, "composite": 0.5,
                "uncertainty": 0.4, "disagreement": 0.3}
        ctx = {"match_index": 60, "race_type": "混戦", "market_gap": 1.0}
        with_ex = calc_buyscore(cand, {**ctx, "has_exhibition": True}, _config())
        without_ex = calc_buyscore(cand, {**ctx, "has_exhibition": False}, _config())
        self.assertAlmostEqual(with_ex - without_ex, 8.0, delta=0.11)  # -0.08*100

    def test_missing_config_section_raises(self) -> None:
        with self.assertRaises(ValidationError):
            calc_buyscore({}, {}, {"weights": {}})  # thresholds欠落


class TestInvestmentType(unittest.TestCase):
    def test_all_branches(self) -> None:
        # buyscore < 60 -> 見送り
        self.assertEqual(investment_type(59.9, 3.0, 50, "混戦"), "見送り")
        # 本命戦 -> 堅実
        self.assertEqual(investment_type(80, 3.0, 50, "本命戦"), "堅実")
        # ev>=2.0 and odds>=30 -> 穴狙い
        self.assertEqual(investment_type(80, 2.0, 30, "混戦"), "穴狙い")
        # ev>=1.5 -> 期待値重視
        self.assertEqual(investment_type(80, 1.5, 20, "混戦"), "期待値重視")
        # それ以外 -> 堅実
        self.assertEqual(investment_type(80, 1.4, 20, "混戦"), "堅実")

    def test_boundaries(self) -> None:
        self.assertEqual(investment_type(60, 1.9, 30, "混戦"), "期待値重視")  # ev<2.0
        self.assertEqual(investment_type(60, 2.0, 29, "混戦"), "期待値重視")  # odds<30
        self.assertEqual(investment_type(60, 2.0, 30, "混戦"), "穴狙い")
        self.assertEqual(investment_type(60, 1.49, 50, "混戦"), "堅実")


class TestKellyFraction(unittest.TestCase):
    def test_zero_below_threshold(self) -> None:
        self.assertEqual(kelly_fraction(0.1, 10, 49.9, _config()), 0.0)

    def test_negative_edge_returns_zero(self) -> None:
        # b*prob - q < 0
        self.assertEqual(kelly_fraction(0.01, 2.0, 80, _config()), 0.0)

    def test_b_zero_returns_zero(self) -> None:
        """odds=1.0 -> b=0 でゼロ除算せず0.0。"""
        self.assertEqual(kelly_fraction(0.5, 1.0, 80, _config()), 0.0)

    def test_half_below_halves(self) -> None:
        full = kelly_fraction(0.2, 10.0, 60, _config())
        half = kelly_fraction(0.2, 10.0, 59, _config())
        self.assertAlmostEqual(half, round(full * 0.5, 4), delta=1e-9)

    def test_round_to_4_decimals(self) -> None:
        k = kelly_fraction(0.15, 8.0, 70, _config())
        self.assertEqual(k, round(k, 4))

    def test_default_strategy_matches_function(self) -> None:
        strategy = DefaultKellyStrategy()
        self.assertEqual(
            strategy(0.2, 10.0, 70, _config()),
            kelly_fraction(0.2, 10.0, 70, _config()),
        )


class TestCheckPassthrough(unittest.TestCase):
    def _cand(self, **over) -> dict:
        base = {"buyscore": 70.0, "ev": 2.0, "uncertainty": 0.3,
                "disagreement": 0.3, "composite": 0.5}
        base.update(over)
        return base

    def test_empty_candidates(self) -> None:
        self.assertEqual(check_passthrough([], {}, _config()), "候補なし")

    def test_buyscore_shortfall(self) -> None:
        r = check_passthrough([self._cand(buyscore=57.0)], {"match_index": 90}, _config())
        self.assertEqual(r, "BuyScore不足(57)")

    def test_ev_shortfall(self) -> None:
        r = check_passthrough(
            [self._cand(ev=1.0)], {"match_index": 90}, _config()
        )
        self.assertEqual(r, "期待値不足(EV1.00)")

    def test_match_index_shortfall(self) -> None:
        r = check_passthrough([self._cand()], {"match_index": 30}, _config())
        self.assertEqual(r, "AI一致指数不足(30)")

    def test_match_index_approx_skips_that_check(self) -> None:
        r = check_passthrough(
            [self._cand()], {"match_index": 30, "match_index_approx": True}, _config()
        )
        self.assertIsNone(r)

    def test_uncertainty_high(self) -> None:
        r = check_passthrough(
            [self._cand(uncertainty=0.80)], {"match_index": 90}, _config()
        )
        self.assertEqual(r, "不確実性高(0.80)")

    def test_exhibition_and_low_composite(self) -> None:
        r = check_passthrough(
            [self._cand(composite=0.30)],
            {"match_index": 90, "has_exhibition": False}, _config(),
        )
        self.assertEqual(r, "展示なし+信頼度不足(0.30)")

    def test_disagreement_high(self) -> None:
        r = check_passthrough(
            [self._cand(disagreement=0.85)], {"match_index": 90}, _config()
        )
        self.assertEqual(r, "モデル不一致(0.85)")

    def test_pass_returns_none(self) -> None:
        r = check_passthrough([self._cand()], {"match_index": 90}, _config())
        self.assertIsNone(r)


class TestDefaultBuyEngine(unittest.TestCase):
    def _evaluation(self, **over) -> RaceEvaluation:
        base = dict(
            eval_id="20260704_12_05", race_date="20260704", venue_num=12,
            venue_name="住之江", race_number=5, is_night=False,
            engine_name="ver4", engine_version="4.0.0", feature_schema_version=1,
            model_version="m", evaluated_at="t",
            features=FeatureSet(
                eval_id="20260704_12_05", feature_schema_version=1, built_at="t",
                boat_features={1: {}}, race_features={}, local_features=None,
                missing_keys=(),
            ),
            danger_score=10.0,
            danger_breakdown={}, upset_score=5.0, upset_reasons=(),
            rank_index={}, featured_boats=None, win_probs=None, race_type="混戦",
            match_index=52.5,
        )
        base.update(over)
        return RaceEvaluation(**base)

    def _prediction(self, **over) -> Prediction:
        base = dict(
            eval_id="20260704_12_05", pred_combo="1-2-3", pred_prob=0.06,
            pred_ev=2.0, pred_odds=35.0, confidence=0.5, why_bet="x",
            patterns=(),
        )
        base.update(over)
        return Prediction(**base)

    def test_assess_returns_all_four_fields(self) -> None:
        engine = DefaultBuyEngine()
        result = engine.assess(self._evaluation(), self._prediction(), _config())
        self.assertIsInstance(result, BuyAssessment)
        self.assertEqual(result.eval_id, "20260704_12_05")
        self.assertIsInstance(result.buyscore, float)
        self.assertIn(
            result.investment_type,
            {"見送り", "堅実", "穴狙い", "期待値重視"},
        )
        self.assertIsInstance(result.kelly_fraction, float)
        self.assertEqual(result.config_version, "test-1.0")

    def test_eval_id_mismatch_raises(self) -> None:
        engine = DefaultBuyEngine()
        with self.assertRaises(ValidationError):
            engine.assess(
                self._evaluation(), self._prediction(eval_id="other"), _config()
            )

    def test_none_prediction_field_raises(self) -> None:
        engine = DefaultBuyEngine()
        with self.assertRaises(ValidationError):
            engine.assess(
                self._evaluation(), self._prediction(pred_prob=None), _config()
            )

    def test_custom_kelly_strategy_injected(self) -> None:
        class _FixedKelly:
            def __call__(self, prob, odds, buyscore, cfg):
                return 0.123

        engine = DefaultBuyEngine(kelly_strategy=_FixedKelly())
        result = engine.assess(self._evaluation(), self._prediction(), _config())
        self.assertEqual(result.kelly_fraction, 0.123)


if __name__ == "__main__":
    unittest.main()
