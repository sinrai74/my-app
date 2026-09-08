"""Stage 2: ShadowRunner が buy_decision（G3-B/G3-C）を比較することの統合テスト。

_evaluate_bets を1回しか呼ばないこと（K1維持）も検証する。
"""

from __future__ import annotations

import unittest
from datetime import datetime

from actions.wiring import assemble_pipelines
from core.buyscore import BuyAssessment
from models.evaluation import FeatureSet, LegacyPurchaseResult, Prediction, RaceEvaluation
from models.race import Race
from pipelines.buy_decision_builder import DefaultBuyDecisionBuilder
from shadow.notifier import NullNotifier
from shadow.runner import ShadowRunner


def _race():
    return Race(
        race_date="20260704", venue_num=12, venue_name="住之江",
        race_number=5, close_time="20:30", is_night=True, entries=(),
        grade=None, weather=None,
    )


def _evaluation(eval_id="20260704_12_05"):
    return RaceEvaluation(
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
        rank_index={}, featured_boats=None, win_probs=None, race_type="混戦",
        match_index=52.5,
    )


class _FakeRaceSource:
    def resolve_race(self, date, venue, race):
        return _race()

    def resolve_boats(self, date, venue, race):
        return []


class _FakeFeatureBuilder:
    def build(self, race, inputs, built_at):
        return FeatureSet(
            eval_id=race.eval_id, feature_schema_version=1, built_at=built_at,
            boat_features={1: {}}, race_features={}, local_features=None,
            missing_keys=(),
        )


class _FakeEngine:
    def evaluate(self, race, feature_set, weather, config, now):
        return _evaluation(race.eval_id)


class _CountingPredictionProvider:
    """_evaluate_bets呼び出し回数を数えるスタブ（capture代替）。"""

    def __init__(self, purchases):
        self.calls = 0
        self._purchases = purchases
        self._last_eval_id = None

    def provide(self, evaluation, config):
        self.calls += 1
        self._last_eval_id = evaluation.eval_id
        return Prediction(
            eval_id=evaluation.eval_id, pred_combo=self._purchases[0]["combo"],
            pred_prob=0.06, pred_ev=2.0, pred_odds=35.0, confidence=0.5,
            why_bet="x", patterns=(),
        )

    def last_purchase_result(self, eval_id):
        return LegacyPurchaseResult(eval_id=eval_id, purchases=tuple(self._purchases))


class _FakeBuyEngine:
    def assess(self, evaluation, prediction, config):
        return BuyAssessment(
            eval_id=evaluation.eval_id, buyscore=72.0, investment_type="穴狙い",
            kelly_fraction=0.01, skip_reason=None, config_version="t",
        )


def _bundle(provider):
    return assemble_pipelines(
        race_source=_FakeRaceSource(),
        feature_builder=_FakeFeatureBuilder(),
        engine=_FakeEngine(),
        now_provider=lambda: datetime(2026, 7, 21, 7, 30),
        eval_config={"_version": "t"},
        durable_store=None,
        prediction_provider=provider,
        buy_engine=_FakeBuyEngine(),
        buy_config={"_version": "t"},
        output_renderers={},
        notification_service=None,
        purchase_result_source=provider,
        decision_builder=DefaultBuyDecisionBuilder(),
    )


def _purchases():
    return [
        {"combo": "1-2-3", "purchased": True, "amount": 300},
        {"combo": "1-3-2", "purchased": True, "amount": 200},
    ]


class TestStage2RunnerBuyDecision(unittest.TestCase):
    def _run(self, legacy_values):
        provider = _CountingPredictionProvider(_purchases())
        runner = ShadowRunner(_bundle(provider))
        result = runner.run_and_compare(
            "20260704", 12, 5, output_paths={}, legacy_values=legacy_values,
        )
        return result, provider

    def test_evaluate_bets_called_once_even_with_buy_decision(self):
        legacy = {"buy_decision": {
            "purchased_combos": ["1-2-3", "1-3-2"],
            "purchased_amounts": [300, 200], "n_bets": 2, "cost": 500,
        }}
        result, provider = self._run(legacy)
        # provide() は1レースにつき1回のみ（assess_and_decideが単一化）
        self.assertEqual(provider.calls, 1)

    def test_buy_decision_match_no_diff(self):
        legacy = {"buy_decision": {
            "purchased_combos": ["1-2-3", "1-3-2"],
            "purchased_amounts": [300, 200], "n_bets": 2, "cost": 500,
        }}
        result, _ = self._run(legacy)
        bd_diffs = [d for d in result.diffs if "buy_decision" in d["field_path"]]
        self.assertEqual(bd_diffs, [])
        self.assertIsNotNone(result.buy_decision)

    def test_count_mismatch_detected(self):
        legacy = {"buy_decision": {
            "purchased_combos": ["1-2-3"], "purchased_amounts": [300],
            "n_bets": 1, "cost": 300,
        }}
        result, _ = self._run(legacy)
        bd_diffs = [d for d in result.diffs if "buy_decision" in d["field_path"]]
        self.assertTrue(bd_diffs)

    def test_amount_mismatch_detected(self):
        legacy = {"buy_decision": {
            "purchased_combos": ["1-2-3", "1-3-2"],
            "purchased_amounts": [999, 200], "n_bets": 2, "cost": 1199,
        }}
        result, _ = self._run(legacy)
        bd_diffs = [d for d in result.diffs if "buy_decision" in d["field_path"]]
        self.assertTrue(bd_diffs)

    def test_combo_order_mismatch_detected(self):
        # 順序違い（sortして一致させない）
        legacy = {"buy_decision": {
            "purchased_combos": ["1-3-2", "1-2-3"],
            "purchased_amounts": [200, 300], "n_bets": 2, "cost": 500,
        }}
        result, _ = self._run(legacy)
        bd_diffs = [d for d in result.diffs if "buy_decision" in d["field_path"]]
        self.assertTrue(bd_diffs)

    def test_no_buy_decision_in_legacy_values_no_comparison(self):
        legacy = {"race": {"venue_num": 12, "race_number": 5,
                           "is_night": True, "venue_name": "住之江"}}
        result, _ = self._run(legacy)
        bd_diffs = [d for d in result.diffs if "buy_decision" in d["field_path"]]
        self.assertEqual(bd_diffs, [])


if __name__ == "__main__":
    unittest.main()
