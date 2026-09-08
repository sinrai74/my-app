import unittest
from types import SimpleNamespace

from core.buyscore import BuyAssessment
from models.evaluation import LegacyPurchaseResult, Prediction, RaceEvaluation
from pipelines.buy_decision_builder import DefaultBuyDecisionBuilder
from pipelines.buy_pipeline import BuyPipeline


def _evaluation():
    return RaceEvaluation(
        eval_id="e1", race_date="20260101", venue_num=1, venue_name="桐生",
        race_number=1, is_night=False, engine_name="ver4", engine_version="4.0",
        feature_schema_version=1, model_version="m", evaluated_at="t",
        danger_score=1.0, danger_breakdown={}, upset_score=1.0, upset_reasons=(),
        rank_index={}, featured_boats=None, win_probs=None, race_type="混戦",
        match_index=50.0, features=None,
    )


def _prediction():
    return Prediction(
        eval_id="e1", pred_combo="1-2-3", pred_prob=0.1, pred_ev=1.8,
        pred_odds=18.0, confidence=0.6, why_bet="x", patterns=(),
    )


def _assessment():
    return BuyAssessment(
        eval_id="e1", buyscore=80.0, investment_type="堅実",
        kelly_fraction=0.08, skip_reason=None, config_version="t-1.0",
    )


def _purchase_result():
    return LegacyPurchaseResult(
        eval_id="e1",
        purchases=(
            {"combo": "1-2-3", "purchased": True, "amount": 500},
            {"combo": "1-3-2", "purchased": True, "amount": 300},
        ),
    )


class _FakePredictionProvider:
    def __init__(self, prediction=None):
        self.prediction = prediction or _prediction()
        self.called_with = None

    def provide(self, evaluation, config):
        self.called_with = SimpleNamespace(evaluation=evaluation, config=config)
        return self.prediction


class _FakeBuyEngine:
    def __init__(self, assessment=None):
        self.assessment = assessment or _assessment()
        self.called_with = None

    def assess(self, evaluation, prediction, config):
        self.called_with = SimpleNamespace(
            evaluation=evaluation, prediction=prediction, config=config)
        return self.assessment


class _FakePurchaseResultSource:
    """_EvaluateBetsCapture の代替フェイク。"""

    def __init__(self, result=None):
        self.result = result or _purchase_result()
        self.requested_eval_id = None

    def last_purchase_result(self, eval_id):
        self.requested_eval_id = eval_id
        if eval_id != self.result.eval_id:
            raise ValueError("eval_id mismatch in fake")
        return self.result


class TestBuyPipelineBackwardCompat(unittest.TestCase):
    def test_assess_race_unchanged_without_new_di(self):
        pipeline = BuyPipeline(
            prediction_provider=_FakePredictionProvider(),
            buy_engine=_FakeBuyEngine(),
            config={"_version": "t"},
        )
        result = pipeline.assess_race(_evaluation())
        self.assertIsInstance(result, BuyAssessment)

    def test_decide_race_without_di_raises(self):
        pipeline = BuyPipeline(
            prediction_provider=_FakePredictionProvider(),
            buy_engine=_FakeBuyEngine(),
            config={"_version": "t"},
        )
        with self.assertRaises(ValueError):
            pipeline.decide_race(_evaluation())

    def test_decide_race_missing_only_decision_builder_raises(self):
        pipeline = BuyPipeline(
            prediction_provider=_FakePredictionProvider(),
            buy_engine=_FakeBuyEngine(),
            config={"_version": "t"},
            purchase_result_source=_FakePurchaseResultSource(),
        )
        with self.assertRaises(ValueError):
            pipeline.decide_race(_evaluation())


class TestBuyPipelineDecideRace(unittest.TestCase):
    def setUp(self):
        self.provider = _FakePredictionProvider()
        self.engine = _FakeBuyEngine()
        self.source = _FakePurchaseResultSource()
        self.pipeline = BuyPipeline(
            prediction_provider=self.provider,
            buy_engine=self.engine,
            config={"_version": "t"},
            purchase_result_source=self.source,
            decision_builder=DefaultBuyDecisionBuilder(),
        )

    def test_decide_race_produces_buy_decision_from_purchase_result(self):
        decision = self.pipeline.decide_race(_evaluation())
        self.assertTrue(decision.purchased)
        self.assertEqual(decision.n_bets, 2)
        self.assertEqual(decision.cost, 800)
        self.assertEqual(decision.purchased_combos, ("1-2-3", "1-3-2"))
        self.assertEqual(decision.purchased_amounts, (500, 300))
        self.assertEqual(decision.buyscore, 80.0)

    def test_decide_race_requests_purchase_result_for_same_eval_id(self):
        ev = _evaluation()
        self.pipeline.decide_race(ev)
        self.assertEqual(self.source.requested_eval_id, ev.eval_id)
        self.assertIs(self.provider.called_with.evaluation, ev)
        self.assertIs(self.engine.called_with.evaluation, ev)

    def test_prediction_and_decision_are_independent_sources(self):
        # Predictionはbest1点（patterns=()の状態でも）、BuyDecisionは
        # 別経路(purchase_result_source)から複数点を得られることを確認。
        decision = self.pipeline.decide_race(_evaluation())
        self.assertEqual(self.provider.prediction.patterns, ())
        self.assertEqual(decision.n_bets, 2)


if __name__ == "__main__":
    unittest.main()
