"""Production driver: 保存済み RaceEvaluation の再利用テスト（実送信・実pushなし）。

ケースA: 未評価 -> evaluate 1回・append_durably 1回
ケースB: 実行をまたいで評価済み -> evaluate 0回・保存済みを再利用
ケースC: 同一実行内の重複 -> evaluate 1回・2回目は再利用
ケースD: JSONL破損 -> 未評価扱いで再評価せずエラー
ケースE: eval_id重複レコード -> 選択・修復せず異常
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from actions.production_entrypoint import (
    build_evaluation_repository,
    build_local_evaluation_store,
    run_one_race,
    run_production_day,
)
from models.evaluation import BuyDecision, FeatureSet, Prediction, RaceEvaluation
from notification.notifiers import NotificationResult
from output.renderers import RenderResult
from storage.exceptions import ParseError, StorageError
from storage.serializers.evaluation_serializer import RaceEvaluationSerializer

DATE = "20260704"
PATHS = {"public": "/tmp/p.html"}


def _make_eval(venue_num=12, race_number=5):
    eval_id = f"{DATE}_{venue_num:02d}_{race_number:02d}"
    fs = FeatureSet(
        eval_id=eval_id, feature_schema_version=1, built_at="t",
        boat_features={1: {}}, race_features={}, local_features=None,
        missing_keys=(),
    )
    return RaceEvaluation(
        eval_id=eval_id, race_date=DATE, venue_num=venue_num,
        venue_name="桐生", race_number=race_number, is_night=False,
        engine_name="ver4", engine_version="4.0", feature_schema_version=1,
        model_version="m", evaluated_at="t", danger_score=1.0,
        danger_breakdown={}, upset_score=5.0, upset_reasons=(), rank_index={},
        featured_boats=None, win_probs=None, race_type="", match_index=50.0,
        features=fs,
    )


class _CountingDurableStore:
    """実 DurableEvaluationStore（no-op release/git）へ委譲し呼び出し回数を数える。"""

    def __init__(self, inner):
        self._inner = inner
        self.append_calls = 0
        self.appended = []

    def append_durably(self, evaluation, commit_message):
        self.append_calls += 1
        self.appended.append(evaluation)
        self._inner.append_durably(evaluation, commit_message)


class _CountingEvalPipeline:
    """evaluate_race の呼び出し回数を数え、persist 時は評価直後に保存する。"""

    def __init__(self, durable_store):
        self._store = durable_store
        self.calls = 0

    def evaluate_race(self, race_date, venue_num, race_number, *, persist=False):
        self.calls += 1
        ev = _make_eval(venue_num, race_number)
        if persist:
            self._store.append_durably(ev, f"eval {ev.eval_id}")
        return ev


class _RecordingBuyPipeline:
    def __init__(self):
        self.seen = []

    def assess_decide_predict(self, evaluation):
        self.seen.append(evaluation)
        decision = BuyDecision(
            eval_id=evaluation.eval_id, purchased=False, buyscore=0.0,
            investment_type="見送り", n_bets=0, cost=0, kelly_fraction=0.0,
            config_version="v", skip_reason="test", purchased_combos=(),
        )
        prediction = Prediction(
            eval_id=evaluation.eval_id, pred_combo="1-2-3", pred_prob=0.1,
            pred_ev=1.0, pred_odds=10.0, confidence=0.5, why_bet="",
            patterns=(),
        )
        return SimpleNamespace(kind="assessment"), decision, prediction


class _NullOutput:
    def render_all(self, date_str, output_paths):
        return {n: RenderResult(output_path=p, summary={}) for n, p in output_paths.items()}


class _NullNotif:
    def send_all(self, requests):
        return [NotificationResult(r.channel, sent=False) for r in requests]


class _Env:
    """一時JSONL・同一パスの durable_store と repository・Fake bundle を束ねる。"""

    def __init__(self):
        self.path = os.path.join(tempfile.mkdtemp(), "evaluations", "production.jsonl")
        self.store = _CountingDurableStore(build_local_evaluation_store(self.path))
        self.repository = build_evaluation_repository(self.path)
        self.eval_pipeline = _CountingEvalPipeline(self.store)
        self.buy = _RecordingBuyPipeline()
        self.bundle = SimpleNamespace(
            evaluation_pipeline=self.eval_pipeline,
            buy_pipeline=self.buy,
            output_pipeline=_NullOutput(),
            notification_pipeline=_NullNotif(),
        )

    def run(self):
        return run_one_race(
            self.bundle, DATE, 12, 5, PATHS,
            persist=True, evaluation_repository=self.repository,
        )

    def write_raw_lines(self, lines):
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            for line in lines:
                f.write(line + "\n")


class TestCaseA_Unevaluated(unittest.TestCase):
    def test_evaluate_once_and_append_once(self):
        env = _Env()
        result = env.run()
        self.assertEqual(env.eval_pipeline.calls, 1)
        self.assertEqual(env.store.append_calls, 1)
        self.assertFalse(result["evaluation_reused"])
        # 保存した同じ RaceEvaluation を後続（Buy）へ渡す
        self.assertIs(env.buy.seen[0], result["evaluation"])
        self.assertEqual(len(env.repository.load_all()), 1)


class TestCaseB_ReuseAcrossRuns(unittest.TestCase):
    def test_saved_evaluation_reused_without_evaluate(self):
        env = _Env()
        saved = _make_eval()
        env.repository.append(saved)  # 前回実行で保存済み（checkout済みJSONL相当）
        result = env.run()
        self.assertEqual(env.eval_pipeline.calls, 0)
        self.assertEqual(env.store.append_calls, 0)
        self.assertTrue(result["evaluation_reused"])
        self.assertEqual(result["evaluation"], saved)
        self.assertIs(env.buy.seen[0], result["evaluation"])
        self.assertEqual(len(env.repository.load_all()), 1)


class TestCaseC_DuplicateWithinRun(unittest.TestCase):
    def test_second_occurrence_reuses_first_saved(self):
        env = _Env()
        report = run_production_day(
            [(DATE, 12, 5), (DATE, 12, 5)],
            lambda d, v, r: PATHS,
            bundle=env.bundle, persist=True,
            evaluation_repository=env.repository,
        )
        self.assertEqual(report["status"], "success")
        self.assertEqual(env.eval_pipeline.calls, 1)
        self.assertEqual(env.store.append_calls, 1)
        first, second = report["results"]
        self.assertFalse(first["evaluation_reused"])
        self.assertTrue(second["evaluation_reused"])
        self.assertEqual(second["evaluation"], first["evaluation"])
        self.assertEqual(len(env.repository.load_all()), 1)


class TestCaseD_BrokenJsonl(unittest.TestCase):
    def test_parse_error_is_raised_not_reevaluated(self):
        env = _Env()
        env.write_raw_lines(["{broken json"])
        with self.assertRaises(ParseError):
            env.run()
        self.assertEqual(env.eval_pipeline.calls, 0)
        self.assertEqual(env.store.append_calls, 0)

    def test_day_records_error_without_reevaluation(self):
        env = _Env()
        env.write_raw_lines(["{broken json"])
        report = run_production_day(
            [(DATE, 12, 5)], lambda d, v, r: PATHS,
            bundle=env.bundle, persist=True,
            evaluation_repository=env.repository,
        )
        self.assertEqual(report["failure_count"], 1)
        self.assertIn("ParseError", report["errors"][0]["error"])
        self.assertEqual(env.eval_pipeline.calls, 0)


class TestCaseE_DuplicateRecords(unittest.TestCase):
    def test_duplicate_eval_id_records_are_abnormal(self):
        env = _Env()
        line = json.dumps(
            RaceEvaluationSerializer.to_dict(_make_eval()), ensure_ascii=False
        )
        env.write_raw_lines([line, line])
        before = open(env.path, encoding="utf-8").read()
        with self.assertRaises(StorageError):
            env.run()
        self.assertEqual(env.eval_pipeline.calls, 0)
        self.assertEqual(env.store.append_calls, 0)
        self.assertEqual(env.buy.seen, [])
        # 選択・修復・削除しない（ファイル無変更）
        self.assertEqual(open(env.path, encoding="utf-8").read(), before)


class TestReuseRequiresPersist(unittest.TestCase):
    def test_repository_without_persist_is_rejected(self):
        env = _Env()
        with self.assertRaises(ValueError):
            run_one_race(
                env.bundle, DATE, 12, 5, PATHS,
                persist=False, evaluation_repository=env.repository,
            )
        self.assertEqual(env.eval_pipeline.calls, 0)


# ---------- 実 EvaluationPipeline の persist 経路を通す監査テスト ----------

class _FakeRaceSource:
    """本番仕様どおり close_time（実データ形式）を持つ Race を返す。"""

    def resolve_race(self, race_date, venue_num, race_number):
        close_time = (
            f"{race_date[:4]}-{race_date[4:6]}-{race_date[6:]} 15:24:00"
        )
        return SimpleNamespace(weather=None, close_time=close_time)

    def resolve_boats(self, race_date, venue_num, race_number):
        return []


class _FakeFeatureBuilder:
    def build(self, race, inputs, built_at):
        return None


class _CountingEngine:
    def __init__(self):
        self.calls = 0

    def evaluate(self, race, feature_set, weather, config, now):
        self.calls += 1
        return _make_eval()


class _RealPipelineRun:
    """1回の実行分: 同一パスの実 DurableEvaluationStore・Repository・実 EvaluationPipeline。"""

    def __init__(self, path):
        from datetime import datetime

        from pipelines.evaluation_pipeline import EvaluationPipeline

        self.store = _CountingDurableStore(build_local_evaluation_store(path))
        self.repository = build_evaluation_repository(path)
        self.engine = _CountingEngine()
        self.buy = _RecordingBuyPipeline()
        self.bundle = SimpleNamespace(
            evaluation_pipeline=EvaluationPipeline(
                race_source=_FakeRaceSource(),
                feature_builder=_FakeFeatureBuilder(),
                engine=self.engine,
                now_provider=lambda: datetime(2026, 7, 4, 9, 0, 0),
                config={},
                durable_store=self.store,
            ),
            buy_pipeline=self.buy,
            output_pipeline=_NullOutput(),
            notification_pipeline=_NullNotif(),
        )


class TestRealPipelinePersistPath(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(
            tempfile.mkdtemp(), "evaluations", "production.jsonl"
        )

    def test_first_run_evaluates_saves_and_passes_same_instance(self):
        from actions.production_entrypoint import run_production_race

        run = _RealPipelineRun(self.path)
        result = run_production_race(
            DATE, 12, 5, PATHS, bundle=run.bundle, persist=True,
            evaluation_repository=run.repository,
        )
        self.assertEqual(run.engine.calls, 1)
        self.assertEqual(run.store.append_calls, 1)
        # 保存したインスタンス＝後続へ渡したインスタンス
        self.assertIs(run.store.appended[0], result["evaluation"])
        self.assertIs(run.buy.seen[0], result["evaluation"])
        self.assertFalse(result["evaluation_reused"])
        # 実際にJSONLへ追記されている（Repositoryと同一パス）
        self.assertEqual(run.repository.path, Path(self.path))
        self.assertEqual(len(run.repository.load_all()), 1)

    def test_next_run_reuses_without_evaluate_or_save(self):
        from actions.production_entrypoint import run_production_race

        first = _RealPipelineRun(self.path)
        r1 = run_production_race(
            DATE, 12, 5, PATHS, bundle=first.bundle, persist=True,
            evaluation_repository=first.repository,
        )
        second = _RealPipelineRun(self.path)  # 別実行（新しい部品・同一JSONL）
        r2 = run_production_race(
            DATE, 12, 5, PATHS, bundle=second.bundle, persist=True,
            evaluation_repository=second.repository,
        )
        self.assertEqual(second.engine.calls, 0)
        self.assertEqual(second.store.append_calls, 0)
        self.assertTrue(r2["evaluation_reused"])
        self.assertEqual(r2["evaluation"], r1["evaluation"])
        self.assertIs(second.buy.seen[0], r2["evaluation"])
        self.assertEqual(len(second.repository.load_all()), 1)

    def test_day_duplicate_uses_saved_content(self):
        run = _RealPipelineRun(self.path)
        report = run_production_day(
            [(DATE, 12, 5), (DATE, 12, 5)], lambda d, v, r: PATHS,
            bundle=run.bundle, persist=True,
            evaluation_repository=run.repository,
        )
        self.assertEqual(report["status"], "success")
        self.assertEqual(run.engine.calls, 1)
        self.assertEqual(run.store.append_calls, 1)
        first, second = report["results"]
        self.assertEqual(second["evaluation"], run.store.appended[0])
        self.assertEqual(second["evaluation"], first["evaluation"])
        self.assertEqual(
            [r["evaluation_reused"] for r in report["results"]], [False, True]
        )


class TestDefaultPathConsistency(unittest.TestCase):
    def test_repository_and_store_builders_share_default_path(self):
        import inspect

        from actions.production_entrypoint import build_github_evaluation_store

        def default(fn):
            return inspect.signature(fn).parameters["jsonl_path"].default

        self.assertEqual(
            default(build_evaluation_repository),
            default(build_local_evaluation_store),
        )
        self.assertEqual(
            default(build_evaluation_repository),
            default(build_github_evaluation_store),
        )


if __name__ == "__main__":
    unittest.main()
