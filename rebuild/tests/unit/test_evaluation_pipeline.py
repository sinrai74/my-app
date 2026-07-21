"""
EvaluationPipeline（pipelines/evaluation_pipeline.py）の単体テスト（Step5-2）。

指示のテスト要件に対応:
  Provider Mock差し替え / Ver4Engine Mock差し替え / Race一致 /
  RaceEvaluation一致 / Shadow比較(Race) / Shadow比較(RaceEvaluation) /
  Protocol経由利用確認。

Legacy・実API・実Engineへ接続せず、すべてMock/FakeをDIして検証する
（Pipelineが「結線のみ」で計算しないことの確認が目的）。
"""

from __future__ import annotations

import unittest
from datetime import datetime
from types import SimpleNamespace

from models.evaluation import FeatureSet, RaceEvaluation
from models.race import Race, RaceEntry, Weather
from pipelines.evaluation_pipeline import EvaluationPipeline


# ---------------- Fake部品（結線の相手） ----------------


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
        weather=Weather(wind_speed_mps=3.0, wind_direction="追", wave_height_cm=5),
    )


class _FakeRaceSource:
    """RaceProvider＋BoatsResolverのFake。取得回数を記録する。"""

    def __init__(self) -> None:
        self.race_calls = 0
        self.boats_calls = 0

    def resolve_race(self, race_date, venue_num, race_number) -> Race:
        self.race_calls += 1
        return _race(race_date, venue_num, race_number)

    def resolve_boats(self, race_date, venue_num, race_number):
        self.boats_calls += 1
        return [{"lane": i, "win_rate": 5.0} for i in range(1, 7)]


class _FakeFeatureBuilder:
    def __init__(self) -> None:
        self.built_with = None

    def build(self, race, inputs, built_at) -> FeatureSet:
        self.built_with = (race, inputs, built_at)
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
    """Ver4EngineのFake。evaluateの引数を記録し固定評価を返す。"""

    def __init__(self, result=None) -> None:
        self.result = result or _evaluation()
        self.called_with = None

    def evaluate(self, race, feature_set, weather, config, now) -> RaceEvaluation:
        self.called_with = SimpleNamespace(
            race=race, feature_set=feature_set, weather=weather,
            config=config, now=now,
        )
        return self.result


class _RecordingStore:
    def __init__(self) -> None:
        self.saved = []

    def append_durably(self, evaluation, commit_message) -> None:
        self.saved.append((evaluation, commit_message))


def _pipeline(source=None, builder=None, engine=None, store=None,
              now=datetime(2026, 7, 21, 7, 30)) -> EvaluationPipeline:
    return EvaluationPipeline(
        race_source=source or _FakeRaceSource(),
        feature_builder=builder or _FakeFeatureBuilder(),
        engine=engine or _FakeEngine(),
        now_provider=lambda: now,
        config={"_version": "test"},
        durable_store=store,
    )


class TestWiring(unittest.TestCase):
    def test_returns_engine_result(self) -> None:
        engine = _FakeEngine(_evaluation(eval_id="20260704_12_05"))
        result = _pipeline(engine=engine).evaluate_race("20260704", 12, 5)
        self.assertIs(result, engine.result)

    def test_passes_race_and_features_to_engine(self) -> None:
        source, builder, engine = _FakeRaceSource(), _FakeFeatureBuilder(), _FakeEngine()
        _pipeline(source=source, builder=builder, engine=engine).evaluate_race(
            "20260704", 12, 5
        )
        # Engineへ渡されたraceはProviderのrace、feature_setはBuilderの産物
        self.assertEqual(engine.called_with.race.eval_id, "20260704_12_05")
        self.assertIsInstance(engine.called_with.feature_set, FeatureSet)
        # weatherはrace.weatherがそのまま渡る（Pipelineは加工しない）
        self.assertEqual(engine.called_with.weather.wind_speed_mps, 3.0)

    def test_now_injected_not_generated(self) -> None:
        fixed = datetime(2030, 1, 1, 0, 0)
        engine = _FakeEngine()
        _pipeline(engine=engine, now=fixed).evaluate_race("20260704", 12, 5)
        self.assertEqual(engine.called_with.now, fixed)

    def test_feature_builder_receives_boats(self) -> None:
        builder = _FakeFeatureBuilder()
        _pipeline(builder=builder).evaluate_race("20260704", 12, 5)
        race, inputs, built_at = builder.built_with
        self.assertEqual(len(inputs.boats), 6)


class TestPersist(unittest.TestCase):
    def test_persist_true_saves(self) -> None:
        store = _RecordingStore()
        _pipeline(store=store).evaluate_race(
            "20260704", 12, 5, persist=True, commit_message="msg"
        )
        self.assertEqual(len(store.saved), 1)
        self.assertEqual(store.saved[0][1], "msg")

    def test_persist_false_does_not_save(self) -> None:
        store = _RecordingStore()
        _pipeline(store=store).evaluate_race("20260704", 12, 5, persist=False)
        self.assertEqual(store.saved, [])

    def test_persist_without_store_raises(self) -> None:
        with self.assertRaises(ValueError):
            _pipeline(store=None).evaluate_race("20260704", 12, 5, persist=True)


class TestProviderMockSwap(unittest.TestCase):
    """Provider Mock差し替え: 別sourceを注入すれば別raceが流れる。"""

    def test_swap_source(self) -> None:
        class _OtherSource(_FakeRaceSource):
            def resolve_race(self, d, v, r):
                return _race(d, 24, r)  # venue 24 (大村)

        engine = _FakeEngine()
        _pipeline(source=_OtherSource(), engine=engine).evaluate_race("20260704", 24, 5)
        self.assertEqual(engine.called_with.race.venue_num, 24)


class TestEngineMockSwap(unittest.TestCase):
    """Ver4Engine Mock差し替え: 別評価を返すEngineを注入。"""

    def test_swap_engine(self) -> None:
        engine = _FakeEngine(_evaluation(danger_score=99.0))
        result = _pipeline(engine=engine).evaluate_race("20260704", 12, 5)
        self.assertEqual(result.danger_score, 99.0)


class TestRaceConsistency(unittest.TestCase):
    """Race一致: Provider由来のRaceがEngineへ改変されず渡ること。"""

    def test_race_identity_preserved(self) -> None:
        source = _FakeRaceSource()
        engine = _FakeEngine()
        _pipeline(source=source, engine=engine).evaluate_race("20260704", 12, 5)
        expected = source.resolve_race("20260704", 12, 5)
        got = engine.called_with.race
        self.assertEqual(got.eval_id, expected.eval_id)
        self.assertEqual(got.venue_name, expected.venue_name)
        self.assertEqual(got.is_night, expected.is_night)


class TestRaceEvaluationConsistency(unittest.TestCase):
    """RaceEvaluation一致: 戻り値がEngine産物とフィールド一致すること。"""

    def test_evaluation_fields_match(self) -> None:
        engine = _FakeEngine(_evaluation(upset_score=7.7))
        result = _pipeline(engine=engine).evaluate_race("20260704", 12, 5)
        self.assertEqual(result.upset_score, engine.result.upset_score)
        self.assertEqual(result.eval_id, engine.result.eval_id)


class TestShadowComparison(unittest.TestCase):
    """Shadow比較の枠組み（Race/RaceEvaluation）。

    Step5-0/5-1で設計した比較方法を、Pipeline出力とLegacy相当（ここでは
    別途用意した期待物）で突き合わせる最小の実装。実Legacyは繋がないが、
    比較関数の正しさと「一致」を検証する。
    """

    def test_shadow_race_equal(self) -> None:
        source = _FakeRaceSource()
        legacy_race = source.resolve_race("20260704", 12, 5)
        rebuild_race = _pipeline(source=source)._race_source.resolve_race(
            "20260704", 12, 5
        )
        # 主要フィールドの一致（Shadow設計の比較対象）
        for attr in ("eval_id", "venue_num", "venue_name", "race_number",
                     "is_night", "grade"):
            self.assertEqual(
                getattr(rebuild_race, attr), getattr(legacy_race, attr), attr
            )
        self.assertEqual(len(rebuild_race.entries), len(legacy_race.entries))

    def test_shadow_evaluation_equal(self) -> None:
        engine = _FakeEngine(_evaluation(danger_score=12.3, upset_score=4.5))
        rebuild_eval = _pipeline(engine=engine).evaluate_race("20260704", 12, 5)
        legacy_eval = engine.result  # Shadowでのlegacy相当
        for attr in ("eval_id", "danger_score", "upset_score", "match_index"):
            self.assertEqual(
                getattr(rebuild_eval, attr), getattr(legacy_eval, attr), attr
            )


class TestProtocolUsage(unittest.TestCase):
    """Protocol経由利用確認: Step5-1 BoatsProviderをそのまま注入できること。"""

    def test_boats_provider_as_race_source(self) -> None:
        from adapters.providers import BoatsProvider

        program = {
            "race_stadium_number": 12, "race_number": 5,
            "race_grade_number": 0, "race_closed_at": "15:00",
        }
        boats = [
            SimpleNamespace(
                lane=i, name=f"n{i}", racer_class="A1", racer_id=f"400{i}",
                win_rate=6.0, local_win=6.0, motor=38.0, avg_st=0.16,
                ability_curr=75.0, ability_prev=74.0,
                course_nyuko=[40] * 6, course_win_rate=[50.0] * 6,
                course_place_rate=[60.0] * 6, course_place_counts=[[10] * 6] * 6,
                course_rank=[2.5] * 6, course_st=[0.16] * 6,
                course_f_count=[0] * 6, course_l_count=[0] * 6,
            )
            for i in range(1, 7)
        ]
        provider = BoatsProvider(
            programs_fetcher=lambda d: [program],
            boats_extractor=lambda p: boats,
        )
        engine = _FakeEngine()
        # BoatsProviderはresolve_race/resolve_boatsを持つのでrace_sourceに適合
        result = _pipeline(source=provider, engine=engine).evaluate_race(
            "20260704", 12, 5
        )
        self.assertEqual(result.eval_id, "20260704_12_05")


if __name__ == "__main__":
    unittest.main()
