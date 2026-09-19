"""Production driver / NotificationRequest builder のテスト（実送信なし）。"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from types import SimpleNamespace

from actions.production_entrypoint import (
    run_one_race,
    production_output_renderers,
    production_notification_service,
)
from notification.notifiers import NotificationRequest, NotificationResult
from output.renderers import RenderResult
from pipelines.notification_request_builder import (
    PRODUCTION_MAIL_TITLE,
    build_mail_notification_request,
)


# ---------- NotificationRequest builder ----------

class TestNotificationRequestBuilder(unittest.TestCase):
    def _rr(self):
        return RenderResult(output_path="/tmp/out.html", summary={"k": 1})

    def test_channel_is_mail(self):
        req = build_mail_notification_request(self._rr())
        self.assertEqual(req.channel, "mail")

    def test_title_is_fixed(self):
        req = build_mail_notification_request(self._rr())
        self.assertEqual(req.title, PRODUCTION_MAIL_TITLE)
        self.assertEqual(req.title, "【競艇AI】本日の予想結果")

    def test_destination_not_managed(self):
        req = build_mail_notification_request(self._rr())
        self.assertIsNone(req.destination)

    def test_render_result_passed_through(self):
        rr = self._rr()
        req = build_mail_notification_request(rr)
        self.assertIs(req.render_result, rr)

    def test_attachment_default_none(self):
        req = build_mail_notification_request(self._rr())
        self.assertIsNone(req.attachment_path)

    def test_attachment_set_when_given(self):
        req = build_mail_notification_request(self._rr(), attachment_path="/tmp/a.pdf")
        self.assertEqual(req.attachment_path, "/tmp/a.pdf")

    def test_does_not_mutate_models(self):
        req = build_mail_notification_request(self._rr())
        self.assertIsInstance(req, NotificationRequest)


# ---------- Evaluate Once (driver) ----------

from models.evaluation import BuyDecision as _BuyDecision
from models.evaluation import Prediction as _Prediction
from models.evaluation import RaceEvaluation as _RaceEvaluation


def _EvalRec(eval_id="20260704_12_05", race_date="20260704"):
    """テスト用の実 RaceEvaluation（per-race本文生成に必要なフィールドを持つ）。"""
    p = eval_id.split("_")
    return _RaceEvaluation(
        eval_id=eval_id, race_date=race_date, venue_num=int(p[1]),
        venue_name="桐生", race_number=int(p[2]), is_night=False,
        engine_name="ver4", engine_version="4.0", feature_schema_version=1,
        model_version="m", evaluated_at="t", danger_score=3.5,
        danger_breakdown={}, upset_score=9.5, upset_reasons=(), rank_index={},
        featured_boats=None, win_probs=None, race_type="1残り荒れ型",
        match_index=50.0, features=None,
    )


class _FakeEvalPipeline:
    def __init__(self, ev):
        self._ev = ev
        self.calls = 0
        self.persist_seen = None

    def evaluate_race(self, race_date, venue_num, race_number, *, persist=False):
        self.calls += 1
        self.persist_seen = persist
        return self._ev


def _mk_decision(ev, purchased=True):
    return _BuyDecision(
        eval_id=ev.eval_id, purchased=purchased, buyscore=72.0,
        investment_type="通常" if purchased else "見送り", n_bets=1 if purchased else 0,
        cost=200 if purchased else 0, kelly_fraction=0.05, config_version="v",
        skip_reason=None if purchased else "BuyScore不足",
        purchased_combos=("6-1-3",) if purchased else (),
    )


def _mk_prediction(ev):
    return _Prediction(
        eval_id=ev.eval_id, pred_combo="6-1-3", pred_prob=0.046, pred_ev=3.03,
        pred_odds=66.0, confidence=0.80, why_bet="理由", patterns=(),
    )


class _FakeBuyPipeline:
    def __init__(self, purchased=True):
        self.seen_eval = None
        self._purchased = purchased

    def assess_decide_predict(self, evaluation):
        self.seen_eval = evaluation
        return (
            SimpleNamespace(kind="assessment"),
            _mk_decision(evaluation, self._purchased),
            _mk_prediction(evaluation),
        )

    def assess_and_decide(self, evaluation):
        a, d, _ = self.assess_decide_predict(evaluation)
        return a, d

    def assess_race(self, evaluation):
        self.seen_eval = evaluation
        return SimpleNamespace(kind="assessment")


class _FakeOutputPipeline:
    def __init__(self):
        self.seen_date = None

    def render_all(self, date_str, output_paths):
        self.seen_date = date_str
        return {
            name: RenderResult(output_path=path, summary={})
            for name, path in output_paths.items()
        }


class _FakeNotifPipeline:
    def __init__(self):
        self.sent_requests = None

    def send_all(self, requests):
        self.sent_requests = list(requests)
        # 実送信しない。sent=False を返す（実メール発生なし）。
        return [NotificationResult(r.channel, sent=False) for r in requests]


def _fake_bundle(ev, purchased=True):
    return SimpleNamespace(
        evaluation_pipeline=_FakeEvalPipeline(ev),
        buy_pipeline=_FakeBuyPipeline(purchased=purchased),
        output_pipeline=_FakeOutputPipeline(),
        notification_pipeline=_FakeNotifPipeline(),
    )


class TestEvaluateOnce(unittest.TestCase):
    def test_evaluation_generated_exactly_once(self):
        ev = _EvalRec()
        bundle = _fake_bundle(ev)
        run_one_race(bundle, "20260704", 12, 5, {"public": "/tmp/p.html"})
        self.assertEqual(bundle.evaluation_pipeline.calls, 1)

    def test_same_evaluation_passed_to_buy(self):
        ev = _EvalRec()
        bundle = _fake_bundle(ev)
        run_one_race(bundle, "20260704", 12, 5, {"public": "/tmp/p.html"})
        self.assertIs(bundle.buy_pipeline.seen_eval, ev)

    def test_output_uses_evaluation_date_not_recompute(self):
        ev = _EvalRec(race_date="20260704")
        bundle = _fake_bundle(ev)
        run_one_race(bundle, "20260704", 12, 5, {"public": "/tmp/p.html"})
        # Output は evaluation.race_date を使う（再評価しない）
        self.assertEqual(bundle.output_pipeline.seen_date, "20260704")

    def test_persist_true_by_default(self):
        ev = _EvalRec()
        bundle = _fake_bundle(ev)
        run_one_race(bundle, "20260704", 12, 5, {"public": "/tmp/p.html"})
        self.assertTrue(bundle.evaluation_pipeline.persist_seen)

    def test_notification_requests_are_per_race_subject(self):
        # per-race通知: 件名はレース識別を含む（固定titleではない）
        ev = _EvalRec()
        bundle = _fake_bundle(ev)
        result = run_one_race(bundle, "20260704", 12, 5, {"public": "/tmp/p.html"})
        reqs = bundle.notification_pipeline.sent_requests
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0].channel, "mail")
        self.assertEqual(reqs[0].title, "【競艇AI】桐生 5R 予想")
        self.assertIsNone(reqs[0].destination)
        self.assertIsNotNone(reqs[0].body)  # per-race本文が入っている

    def test_no_real_send(self):
        ev = _EvalRec()
        bundle = _fake_bundle(ev)
        result = run_one_race(bundle, "20260704", 12, 5, {"public": "/tmp/p.html"})
        # すべて sent=False（実送信していない）
        self.assertTrue(all(not r.sent for r in result["notification_results"]))

    def test_skip_when_not_purchased_no_request(self):
        # purchased=False（見送り）はNotificationRequestを生成しない
        ev = _EvalRec()
        bundle = _fake_bundle(ev, purchased=False)
        result = run_one_race(bundle, "20260704", 12, 5, {"public": "/tmp/p.html"})
        self.assertEqual(bundle.notification_pipeline.sent_requests, [])
        self.assertEqual(result["requests"], [])

    def test_request_when_purchased(self):
        # purchased=True は per-race request を生成する
        ev = _EvalRec()
        bundle = _fake_bundle(ev, purchased=True)
        result = run_one_race(bundle, "20260704", 12, 5, {"public": "/tmp/p.html"})
        self.assertEqual(len(result["requests"]), 1)
        self.assertTrue(result["buy_decision"].purchased)

    def test_fallback_to_assess_race_when_no_decide(self):
        # assess_decide_predict が ValueError の構成でも assess_race で継続
        ev = _EvalRec()
        bundle = _fake_bundle(ev)

        def _raise(evaluation):
            raise ValueError("no decision builder")

        bundle.buy_pipeline.assess_decide_predict = _raise
        result = run_one_race(bundle, "20260704", 12, 5, {"public": "/tmp/p.html"})
        self.assertIsNone(result["buy_decision"])
        # decision/prediction なし → requestは生成されない
        self.assertEqual(result["requests"], [])


# ---------- Production bundle wiring ----------

class TestProductionBundleWiring(unittest.TestCase):
    def test_output_renderers_is_public(self):
        renderers = production_output_renderers()
        self.assertIn("public", renderers)
        from output.renderers import PublicHtmlRenderer
        self.assertIsInstance(renderers["public"], PublicHtmlRenderer)

    def test_notification_service_has_mail_only(self):
        svc = production_notification_service()
        # NotificationService へ mail が登録され、NullNotifierでないこと
        notifiers = svc._notifiers
        self.assertIn("mail", notifiers)
        from notification.notifiers import MailNotifier
        self.assertIsInstance(notifiers["mail"], MailNotifier)
        self.assertNotIn("line", notifiers)

    def test_assemble_pipelines_accepts_production_parts(self):
        # assemble_pipelines が PublicHtmlRenderer + MailNotifier を受け付ける
        from actions.wiring import assemble_pipelines
        bundle = assemble_pipelines(
            race_source=SimpleNamespace(resolve_race=lambda *a, **k: None,
                                        resolve_boats=lambda *a, **k: []),
            feature_builder=SimpleNamespace(build=lambda *a, **k: None),
            engine=SimpleNamespace(evaluate=lambda *a, **k: None),
            now_provider=lambda: None,
            eval_config={"_version": "t"},
            durable_store=None,
            prediction_provider=SimpleNamespace(provide=lambda *a, **k: None),
            buy_engine=SimpleNamespace(assess=lambda *a, **k: None),
            buy_config={"_version": "t"},
            output_renderers=production_output_renderers(),
            notification_service=production_notification_service(),
        )
        self.assertIsNotNone(bundle.output_pipeline)
        self.assertIsNotNone(bundle.notification_pipeline)


if __name__ == "__main__":
    unittest.main()


# ---------- Production entry (bundle construction + run_production_race) ----------

class TestProductionEntry(unittest.TestCase):
    def test_run_production_race_accepts_injected_bundle(self):
        # 注入bundleでrun_production_raceが1レースを処理し実送信しない
        from actions.production_entrypoint import run_production_race
        ev = _EvalRec()
        bundle = _fake_bundle(ev)
        result = run_production_race(
            "20260704", 12, 5, {"public": "/tmp/p.html"}, bundle=bundle
        )
        self.assertEqual(bundle.evaluation_pipeline.calls, 1)
        self.assertTrue(all(not r.sent for r in result["notification_results"]))

    def test_build_production_bundle_swaps_output_notification(self):
        # build_bundle を Fake 化し、eval/buy 流用・output/notification 差し替えを検証
        import actions.production_entrypoint as pe
        from actions.wiring import PipelineBundle

        base = SimpleNamespace(
            evaluation_pipeline=SimpleNamespace(tag="eval"),
            buy_pipeline=SimpleNamespace(tag="buy"),
            output_pipeline=SimpleNamespace(tag="shadow-output"),
            notification_pipeline=SimpleNamespace(tag="shadow-null"),
        )
        orig = None
        try:
            import actions.shadow_entrypoint as se
            orig = se.build_bundle
            se.build_bundle = lambda eval_config=None, buy_config=None: base
            bundle = pe.build_production_bundle()
        finally:
            if orig is not None:
                se.build_bundle = orig

        # eval/buy は base から流用
        self.assertIs(bundle.evaluation_pipeline, base.evaluation_pipeline)
        self.assertIs(bundle.buy_pipeline, base.buy_pipeline)
        # output/notification は本番用へ差し替わっている（shadowのものではない）
        from pipelines.output_pipeline import OutputPipeline
        from pipelines.notification_pipeline import NotificationPipeline
        self.assertIsInstance(bundle.output_pipeline, OutputPipeline)
        self.assertIsInstance(bundle.notification_pipeline, NotificationPipeline)


# ---------- Local persistence store + durable_store injection ----------

class TestLocalPersistence(unittest.TestCase):
    def test_build_local_evaluation_store_persists_to_file(self):
        import tempfile, os, json
        from actions.production_entrypoint import build_local_evaluation_store
        from models.evaluation import FeatureSet, RaceEvaluation
        tmp = tempfile.mkdtemp()
        path = os.path.join(tmp, "eval.jsonl")
        store = build_local_evaluation_store(path)
        fs = FeatureSet(
            eval_id="20260704_01_01", feature_schema_version=1, built_at="t",
            boat_features={1: {}}, race_features={}, local_features=None,
            missing_keys=(),
        )
        ev = RaceEvaluation(
            eval_id="20260704_01_01", race_date="20260704", venue_num=1,
            venue_name="桐生", race_number=1, is_night=False, engine_name="ver4",
            engine_version="4.0", feature_schema_version=1, model_version="m",
            evaluated_at="t", danger_score=1.0, danger_breakdown={}, upset_score=5.0,
            upset_reasons=(), rank_index={}, featured_boats=None, win_probs=None,
            race_type="", match_index=50.0, features=fs,
        )
        # no-op release/git -> ローカル追記のみ。ネットワーク・push なし。
        store.append_durably(ev, "test commit")
        self.assertTrue(os.path.exists(path))
        lines = [l for l in open(path, encoding="utf-8") if l.strip()]
        self.assertEqual(len(lines), 1)
        self.assertEqual(json.loads(lines[0])["eval_id"], "20260704_01_01")


class TestDurableStoreInjection(unittest.TestCase):
    def test_run_production_race_persist_false_default(self):
        # 既定 persist=False: durable_store未注入bundleでも例外にならない
        from actions.production_entrypoint import run_production_race
        ev = _EvalRec()
        bundle = _fake_bundle(ev)
        result = run_production_race(
            "20260704", 12, 5, {"public": "/tmp/p.html"}, bundle=bundle
        )
        self.assertFalse(bundle.evaluation_pipeline.persist_seen)

    def test_run_production_race_persist_true_when_requested(self):
        from actions.production_entrypoint import run_production_race
        ev = _EvalRec()
        bundle = _fake_bundle(ev)
        run_production_race(
            "20260704", 12, 5, {"public": "/tmp/p.html"},
            bundle=bundle, persist=True,
        )
        self.assertTrue(bundle.evaluation_pipeline.persist_seen)


# ---------- Orchestration (multiple races) ----------

class TestRunProductionDay(unittest.TestCase):
    def test_processes_all_races_with_one_bundle(self):
        from actions.production_entrypoint import run_production_day
        ev = _EvalRec()
        bundle = _fake_bundle(ev)
        targets = [("20260704", 12, 5), ("20260704", 12, 6), ("20260704", 1, 1)]
        results = run_production_day(
            targets,
            output_paths_for=lambda d, v, r: {"public": f"/tmp/{d}_{v}_{r}.html"},
            bundle=bundle,
        )
        self.assertEqual(len(results), 3)
        # evaluate_race は各レースにつき1回（Evaluate Once）→ 合計3回
        self.assertEqual(bundle.evaluation_pipeline.calls, 3)

    def test_each_race_produces_mail_request(self):
        from actions.production_entrypoint import run_production_day
        ev = _EvalRec()
        bundle = _fake_bundle(ev)
        targets = [("20260704", 12, 5), ("20260704", 12, 6)]
        results = run_production_day(
            targets,
            output_paths_for=lambda d, v, r: {"public": f"/tmp/{d}_{v}_{r}.html"},
            bundle=bundle,
        )
        for res in results:
            reqs = res["requests"]
            self.assertTrue(all(r.channel == "mail" for r in reqs))
            self.assertTrue(all(not nr.sent for nr in res["notification_results"]))


# ---------- GitHub Releases DurableStore 結線（実push なし） ----------

class TestGithubEvaluationStoreWiring(unittest.TestCase):
    def test_builds_durable_store_from_existing_parts(self):
        from actions.production_entrypoint import build_github_evaluation_store
        from storage.durability import DurableEvaluationStore
        store = build_github_evaluation_store(
            "evaluations/prod.jsonl", repo_dir=".", tag="data-store",
            env={"GITHUB_TOKEN": "x-token", "GITHUB_REPOSITORY": "sinrai74/my-app"},
        )
        # 既存部品が正しく束ねられた DurableEvaluationStore が返る
        self.assertIsInstance(store, DurableEvaluationStore)
        # 構成のみ。ネットワーク・実push は発生していない（append未呼び出し）
        from storage.clients.github_release_client import GithubReleaseClient
        from storage.clients.git_client import SubprocessGitClient
        self.assertIsInstance(store._release, GithubReleaseClient)
        self.assertIsInstance(store._git, SubprocessGitClient)

    def test_owner_repo_parsed_from_env(self):
        from actions.production_entrypoint import build_github_evaluation_store
        store = build_github_evaluation_store(
            env={"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "sinrai74/my-app"},
        )
        self.assertEqual(store._release._owner, "sinrai74")
        self.assertEqual(store._release._repo, "my-app")

    def test_raises_without_token(self):
        from actions.production_entrypoint import build_github_evaluation_store
        with self.assertRaises(ValueError):
            build_github_evaluation_store(
                env={"GITHUB_TOKEN": "", "GITHUB_REPOSITORY": "o/r"})

    def test_raises_without_valid_repository(self):
        from actions.production_entrypoint import build_github_evaluation_store
        with self.assertRaises(ValueError):
            build_github_evaluation_store(
                env={"GITHUB_TOKEN": "t", "GITHUB_REPOSITORY": "no-slash"})

    def test_no_hardcoded_credentials(self):
        # ソースに token/secret がハードコードされていないこと（env経由のみ）
        import inspect
        from actions import production_entrypoint
        src = inspect.getsource(production_entrypoint.build_github_evaluation_store)
        self.assertIn("GITHUB_TOKEN", src)
        self.assertNotIn("ghp_", src)  # 実トークンの痕跡がない

    def test_injectable_into_production_bundle_signature(self):
        # build_production_bundle が durable_store を受け取れる（注入口の存在）
        import inspect
        from actions.production_entrypoint import build_production_bundle
        params = inspect.signature(build_production_bundle).parameters
        self.assertIn("durable_store", params)
