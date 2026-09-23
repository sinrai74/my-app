"""pipelines/production_evaluation_entry のテスト（GitHub実書込・実送信なし）。"""

from __future__ import annotations

import functools
import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from actions.production_entrypoint import build_local_evaluation_store
from models.evaluation import BuyDecision, Prediction
from output.renderers import RenderResult
from pipelines.evaluation_pipeline import EvaluationPipeline
from pipelines.notification_request_builder import build_per_race_mail_request
from pipelines.production_evaluation_entry import (
    RELEASE_TAG,
    build_evaluation_only_bundle,
    evaluations_path_for,
    main,
    run_entry,
)
from tests.unit.test_production_evaluation_reuse import (
    DATE,
    _CountingDurableStore,
    _CountingEngine,
    _FakeFeatureBuilder,
    _FakeRaceSource,
)


class _PurchasingBuyPipeline:
    """purchased=True を返す（通知経路に入っても実送信されないことの確認用）。"""

    def __init__(self):
        self.seen = []

    def assess_decide_predict(self, evaluation):
        self.seen.append(evaluation)
        decision = BuyDecision(
            eval_id=evaluation.eval_id, purchased=True, buyscore=80.0,
            investment_type="本命", n_bets=1, cost=100, kelly_fraction=0.1,
            config_version="v", skip_reason=None, purchased_combos=("1-2-3",),
        )
        prediction = Prediction(
            eval_id=evaluation.eval_id, pred_combo="1-2-3", pred_prob=0.1,
            pred_ev=1.2, pred_odds=12.0, confidence=0.5, why_bet="",
            patterns=(),
        )
        return SimpleNamespace(kind="assessment"), decision, prediction


class _EntryRun:
    """1回の入口実行分の Fake 部品（Store は実ローカル DurableStore を包む）。"""

    def __init__(self):
        self.store_calls = []
        self.store = None
        self.engine = _CountingEngine()
        self.buy = _PurchasingBuyPipeline()

    def store_factory(self, jsonl_path, *, tag, env):
        self.store_calls.append({"path": jsonl_path, "tag": tag, "env": env})
        self.store = _CountingDurableStore(build_local_evaluation_store(jsonl_path))
        return self.store

    def production_bundle_factory(self, *, durable_store):
        return SimpleNamespace(
            evaluation_pipeline=EvaluationPipeline(
                race_source=_FakeRaceSource(),
                feature_builder=_FakeFeatureBuilder(),
                engine=self.engine,
                now_provider=lambda: datetime(2026, 7, 4, 9, 0, 0),
                config={},
                durable_store=durable_store,
            ),
            buy_pipeline=self.buy,
        )

    def run(self, races):
        return run_entry(
            races,
            env={},
            store_factory=self.store_factory,
            bundle_factory=functools.partial(
                build_evaluation_only_bundle,
                production_bundle_factory=self.production_bundle_factory,
            ),
        )


class _InTempDir(unittest.TestCase):
    def setUp(self):
        self._cwd = os.getcwd()
        self._tmp = tempfile.mkdtemp()
        os.chdir(self._tmp)

    def tearDown(self):
        os.chdir(self._cwd)

    def jsonl(self, race_date=DATE):
        return Path(self._tmp) / evaluations_path_for(race_date)


class TestEntryConstruction(_InTempDir):
    def test_store_and_repository_share_path_and_tag_is_v2(self):
        run = _EntryRun()
        run.run([(DATE, 12, 5)])
        self.assertEqual(run.store_calls[0]["path"], f"evaluations/{DATE}.jsonl")
        self.assertEqual(run.store_calls[0]["tag"], "data-store-v2")
        self.assertEqual(RELEASE_TAG, "data-store-v2")
        self.assertTrue(self.jsonl().exists())  # Repository と同一パスへ保存
        self.assertFalse((Path(self._tmp) / "evaluations/production.jsonl").exists())

    def test_date_maps_to_daily_jsonl(self):
        self.assertEqual(evaluations_path_for("20260922"), "evaluations/20260922.jsonl")

    def test_real_github_store_factory_uses_v2_tag_without_network(self):
        from actions.production_entrypoint import build_github_evaluation_store

        store = build_github_evaluation_store(
            evaluations_path_for("20260922"), tag=RELEASE_TAG,
            env={"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "o/r"},
        )
        self.assertEqual(store._release._tag, "data-store-v2")
        self.assertEqual(store._repository.path, Path("evaluations/20260922.jsonl"))

    def test_missing_credentials_is_configuration_error(self):
        with self.assertRaises(ValueError):
            run_entry([(DATE, 12, 5)], env={})


class TestSingleDatePerRun(_InTempDir):
    def test_mixed_dates_rejected_before_any_construction(self):
        run = _EntryRun()
        with self.assertRaises(ValueError):
            run.run([(DATE, 12, 5), ("20260705", 12, 5)])
        self.assertEqual(run.store_calls, [])
        self.assertEqual(run.engine.calls, 0)
        self.assertFalse((Path(self._tmp) / "evaluations").exists())

    def test_main_mixed_dates_returns_1(self):
        saved = os.environ.get("TARGET_RACES")
        os.environ["TARGET_RACES"] = f"{DATE}_12_5,20260705_12_5"
        try:
            self.assertEqual(main([]), 1)
        finally:
            if saved is None:
                os.environ.pop("TARGET_RACES", None)
            else:
                os.environ["TARGET_RACES"] = saved


class TestCaseA_FirstRun(_InTempDir):
    def test_evaluate_once_then_durable_save(self):
        run = _EntryRun()
        report = run.run([(DATE, 12, 5)])
        self.assertEqual(report["status"], "success")
        self.assertEqual(run.engine.calls, 1)
        self.assertEqual(run.store.append_calls, 1)
        result = report["results"][0]
        self.assertFalse(result["evaluation_reused"])
        self.assertIs(run.store.appended[0], result["evaluation"])
        self.assertIs(run.buy.seen[0], result["evaluation"])


class TestCaseB_NextRun(_InTempDir):
    def test_saved_jsonl_is_reused(self):
        first = _EntryRun()
        r1 = first.run([(DATE, 12, 5)])
        second = _EntryRun()
        r2 = second.run([(DATE, 12, 5)])
        self.assertEqual(second.engine.calls, 0)
        self.assertEqual(second.store.append_calls, 0)
        self.assertTrue(r2["results"][0]["evaluation_reused"])
        self.assertEqual(r2["results"][0]["evaluation"], r1["results"][0]["evaluation"])


class TestCaseC_DuplicateInRun(_InTempDir):
    def test_second_occurrence_reused(self):
        run = _EntryRun()
        report = run.run([(DATE, 12, 5), (DATE, 12, 5)])
        self.assertEqual(run.engine.calls, 1)
        self.assertEqual(run.store.append_calls, 1)
        self.assertEqual(
            [r["evaluation_reused"] for r in report["results"]], [False, True]
        )


class TestCaseD_NoRealSending(_InTempDir):
    def test_purchased_race_produces_no_request_and_no_output(self):
        run = _EntryRun()
        report = run.run([(DATE, 12, 5)])
        result = report["results"][0]
        self.assertTrue(result["buy_decision"].purchased)
        self.assertEqual(result["render_results"], {})
        self.assertEqual(result["requests"], [])
        self.assertEqual(result["notification_results"], [])

    def test_bundle_notification_is_null_only(self):
        run = _EntryRun()
        bundle = build_evaluation_only_bundle(
            None, production_bundle_factory=run.production_bundle_factory
        )
        request = build_per_race_mail_request(
            RenderResult(output_path="x", summary={}),
            subject="s", body="b", race_date=DATE, venue_num=12, race_number=5,
        )
        results = bundle.notification_pipeline.send_all([request])
        self.assertFalse(results[0].sent)
        self.assertEqual(results[0].detail, "shadow-noop")


class TestMain(_InTempDir):
    def test_help_exits_zero(self):
        with self.assertRaises(SystemExit) as cm:
            main(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_missing_target_races_returns_1(self):
        saved = os.environ.pop("TARGET_RACES", None)
        try:
            self.assertEqual(main([]), 1)
        finally:
            if saved is not None:
                os.environ["TARGET_RACES"] = saved


if __name__ == "__main__":
    unittest.main()
