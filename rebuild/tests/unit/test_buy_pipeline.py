"""
BuyPipeline（pipelines/buy_pipeline.py）とDI Adapterの単体テスト（Step5-3）。

指示のテスト要件に対応:
  BuyPipeline正常系 / Prediction Mock / BuyEngine Mock / BuyAssessment一致 /
  Shadow比較(Prediction) / Shadow比較(BuyAssessment) / DI Adapter単体 /
  DI組立 / EngineへCallable注入。

実部品へ接続せずMock/FakeをDIし、Pipelineが「結線のみ」で計算しないことを検証する。
BuyDecisionは生成しない（案A）ため、比較対象はBuyAssessmentの4項目。
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from pipelines.wiring import RaceArgBoatsResolver
from core.buyscore import BuyAssessment
from models.evaluation import FeatureSet, Prediction, RaceEvaluation
from models.race import Race
from pipelines.buy_pipeline import BuyPipeline


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


def _prediction(eval_id="20260704_12_05", **over) -> Prediction:
    base = dict(
        eval_id=eval_id, pred_combo="1-2-3", pred_prob=0.06, pred_ev=2.0,
        pred_odds=35.0, confidence=0.5, why_bet="x", patterns=(),
    )
    base.update(over)
    return Prediction(**base)


def _assessment(eval_id="20260704_12_05", **over) -> BuyAssessment:
    base = dict(
        eval_id=eval_id, buyscore=72.0, investment_type="穴狙い",
        kelly_fraction=0.0123, skip_reason=None, config_version="test-1.0",
    )
    base.update(over)
    return BuyAssessment(**base)


class _FakePredictionProvider:
    def __init__(self, prediction=None) -> None:
        self.prediction = prediction or _prediction()
        self.called_with = None

    def provide(self, evaluation, config) -> Prediction:
        self.called_with = SimpleNamespace(evaluation=evaluation, config=config)
        return self.prediction


class _FakeBuyEngine:
    def __init__(self, assessment=None) -> None:
        self.assessment = assessment or _assessment()
        self.called_with = None

    def assess(self, evaluation, prediction, config) -> BuyAssessment:
        self.called_with = SimpleNamespace(
            evaluation=evaluation, prediction=prediction, config=config
        )
        return self.assessment


def _pipeline(provider=None, engine=None) -> BuyPipeline:
    return BuyPipeline(
        prediction_provider=provider or _FakePredictionProvider(),
        buy_engine=engine or _FakeBuyEngine(),
        config={"_version": "test-1.0"},
    )


class TestWiring(unittest.TestCase):
    def test_returns_engine_assessment(self) -> None:
        engine = _FakeBuyEngine(_assessment(buyscore=88.0))
        result = _pipeline(engine=engine).assess_race(_evaluation())
        self.assertIs(result, engine.assessment)

    def test_prediction_flows_into_engine(self) -> None:
        provider = _FakePredictionProvider(_prediction(pred_combo="2-1-3"))
        engine = _FakeBuyEngine()
        _pipeline(provider=provider, engine=engine).assess_race(_evaluation())
        # Providerが返したPredictionがそのままEngineへ渡る（Pipelineは補正しない）
        self.assertEqual(engine.called_with.prediction.pred_combo, "2-1-3")

    def test_evaluation_passed_unchanged(self) -> None:
        provider = _FakePredictionProvider()
        engine = _FakeBuyEngine()
        ev = _evaluation(upset_score=9.9)
        _pipeline(provider=provider, engine=engine).assess_race(ev)
        self.assertIs(provider.called_with.evaluation, ev)
        self.assertIs(engine.called_with.evaluation, ev)


class TestPredictionMock(unittest.TestCase):
    def test_swap_prediction_provider(self) -> None:
        provider = _FakePredictionProvider(_prediction(pred_ev=3.3))
        engine = _FakeBuyEngine()
        _pipeline(provider=provider, engine=engine).assess_race(_evaluation())
        self.assertEqual(engine.called_with.prediction.pred_ev, 3.3)


class TestBuyEngineMock(unittest.TestCase):
    def test_swap_buy_engine(self) -> None:
        engine = _FakeBuyEngine(_assessment(investment_type="堅実"))
        result = _pipeline(engine=engine).assess_race(_evaluation())
        self.assertEqual(result.investment_type, "堅実")


class TestAssessmentConsistency(unittest.TestCase):
    def test_four_fields_match(self) -> None:
        expected = _assessment(
            buyscore=61.0, investment_type="期待値重視",
            kelly_fraction=0.05, skip_reason=None,
        )
        result = _pipeline(engine=_FakeBuyEngine(expected)).assess_race(_evaluation())
        self.assertEqual(result.buyscore, expected.buyscore)
        self.assertEqual(result.investment_type, expected.investment_type)
        self.assertEqual(result.kelly_fraction, expected.kelly_fraction)
        self.assertEqual(result.skip_reason, expected.skip_reason)


# ---------- Shadow比較（JSON正規化→再帰比較・差分 eval_id/path/legacy/rebuild） ----------


def _shadow_diff(eval_id, legacy_obj, rebuild_obj, path="$"):
    """Step5-0/5-2と同方式の再帰比較。差分を辞書リストで返す。"""
    diffs = []
    legacy = json.loads(json.dumps(legacy_obj, ensure_ascii=False, sort_keys=True))
    rebuild = json.loads(json.dumps(rebuild_obj, ensure_ascii=False, sort_keys=True))

    def rec(le, re, p):
        if isinstance(le, dict) and isinstance(re, dict):
            for k in sorted(set(le) | set(re)):
                rec(le.get(k), re.get(k), f"{p}.{k}")
        elif isinstance(le, list) and isinstance(re, list):
            if len(le) != len(re):
                diffs.append({"eval_id": eval_id, "field_path": f"{p}.__len__",
                              "legacy": len(le), "rebuild": len(re)})
            for i, (a, b) in enumerate(zip(le, re)):
                rec(a, b, f"{p}[{i}]")
        elif le != re:
            diffs.append({"eval_id": eval_id, "field_path": p,
                          "legacy": le, "rebuild": re})

    rec(legacy, rebuild, path)
    return diffs


def _prediction_to_dict(p: Prediction) -> dict:
    return {"eval_id": p.eval_id, "pred_combo": p.pred_combo,
            "pred_prob": p.pred_prob, "pred_ev": p.pred_ev, "pred_odds": p.pred_odds}


def _assessment_to_dict(a: BuyAssessment) -> dict:
    return {"eval_id": a.eval_id, "buyscore": a.buyscore,
            "investment_type": a.investment_type,
            "kelly_fraction": a.kelly_fraction, "skip_reason": a.skip_reason}


class TestShadowPrediction(unittest.TestCase):
    def test_prediction_equal(self) -> None:
        provider = _FakePredictionProvider(_prediction())
        rebuild_pred = provider.provide(_evaluation(), {})
        legacy_pred = _prediction()  # Shadowでのlegacy相当
        diffs = _shadow_diff(
            "20260704_12_05",
            _prediction_to_dict(legacy_pred), _prediction_to_dict(rebuild_pred),
        )
        self.assertEqual(diffs, [])

    def test_prediction_diff_detected(self) -> None:
        diffs = _shadow_diff(
            "20260704_12_05",
            _prediction_to_dict(_prediction(pred_combo="1-2-3")),
            _prediction_to_dict(_prediction(pred_combo="3-2-1")),
        )
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["field_path"], "$.pred_combo")
        self.assertEqual(diffs[0]["legacy"], "1-2-3")
        self.assertEqual(diffs[0]["rebuild"], "3-2-1")


class TestShadowAssessment(unittest.TestCase):
    def test_assessment_equal(self) -> None:
        engine = _FakeBuyEngine(_assessment())
        rebuild = _pipeline(engine=engine).assess_race(_evaluation())
        legacy = _assessment()
        diffs = _shadow_diff(
            "20260704_12_05",
            _assessment_to_dict(legacy), _assessment_to_dict(rebuild),
        )
        self.assertEqual(diffs, [])

    def test_assessment_diff_detected(self) -> None:
        diffs = _shadow_diff(
            "20260704_12_05",
            _assessment_to_dict(_assessment(buyscore=72.0)),
            _assessment_to_dict(_assessment(buyscore=60.0)),
        )
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["field_path"], "$.buyscore")


# ---------- DI Adapter ----------


class TestRaceArgBoatsResolver(unittest.TestCase):
    """DI Adapterは引数を詰め替えるだけ（判定・加工なし）。"""

    def _race(self) -> Race:
        return Race(
            race_date="20260704", venue_num=12, venue_name="住之江",
            race_number=5, close_time="", is_night=True, entries=(),
            grade="一般",
        )

    def test_adapts_race_to_positional_args(self) -> None:
        captured = {}

        class _Provider:
            def resolve_boats(self, date, venue, race):
                captured["args"] = (date, venue, race)
                return [{"lane": 1}]

        adapter = RaceArgBoatsResolver(_Provider())
        boats = adapter(self._race())
        self.assertEqual(captured["args"], ("20260704", 12, 5))
        self.assertEqual(boats, [{"lane": 1}])

    def test_returns_provider_result_unmodified(self) -> None:
        sentinel = [{"lane": i} for i in range(1, 7)]

        class _Provider:
            def resolve_boats(self, d, v, r):
                return sentinel

        adapter = RaceArgBoatsResolver(_Provider())
        self.assertIs(adapter(self._race()), sentinel)  # 加工しない

    def test_engine_callable_injection(self) -> None:
        """Ver4EngineのCallable[[Race],boats]としてそのまま注入できる形。"""
        class _Provider:
            def resolve_boats(self, d, v, r):
                return [{"lane": 1, "win_rate": 5.0}]

        adapter = RaceArgBoatsResolver(_Provider())
        # Callableとして呼べる（Ver4Engine.boats_resolver互換）
        self.assertTrue(callable(adapter))
        result = adapter(self._race())
        self.assertEqual(result[0]["lane"], 1)


class TestDIAssembly(unittest.TestCase):
    """DI組立: Fake部品でBuyPipelineを組み立てて1件通す。"""

    def test_assemble_and_run(self) -> None:
        pipeline = BuyPipeline(
            prediction_provider=_FakePredictionProvider(_prediction()),
            buy_engine=_FakeBuyEngine(_assessment(buyscore=75.0)),
            config={"_version": "test-1.0"},
        )
        result = pipeline.assess_race(_evaluation())
        self.assertIsInstance(result, BuyAssessment)
        self.assertEqual(result.buyscore, 75.0)


class TestCapitalManagerProtocol(unittest.TestCase):
    """CapitalManagerはProtocol定義のみ（具象実装しないことの確認）。"""

    def test_protocol_importable_and_no_concrete(self) -> None:
        from adapters import capital

        self.assertTrue(hasattr(capital, "CapitalManager"))
        # 具象クラス（Default*）が存在しないこと
        concretes = [n for n in dir(capital) if n.startswith("Default")]
        self.assertEqual(concretes, [])


if __name__ == "__main__":
    unittest.main()
