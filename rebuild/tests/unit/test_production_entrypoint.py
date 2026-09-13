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

@dataclass
class _EvalRec:
    eval_id: str = "20260704_12_05"
    race_date: str = "20260704"


class _FakeEvalPipeline:
    def __init__(self, ev):
        self._ev = ev
        self.calls = 0
        self.persist_seen = None

    def evaluate_race(self, race_date, venue_num, race_number, *, persist=False):
        self.calls += 1
        self.persist_seen = persist
        return self._ev


class _FakeBuyPipeline:
    def __init__(self):
        self.seen_eval = None

    def assess_and_decide(self, evaluation):
        self.seen_eval = evaluation
        return SimpleNamespace(kind="assessment"), SimpleNamespace(kind="decision")

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


def _fake_bundle(ev):
    return SimpleNamespace(
        evaluation_pipeline=_FakeEvalPipeline(ev),
        buy_pipeline=_FakeBuyPipeline(),
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

    def test_notification_requests_are_mail_fixed_title(self):
        ev = _EvalRec()
        bundle = _fake_bundle(ev)
        result = run_one_race(bundle, "20260704", 12, 5, {"public": "/tmp/p.html"})
        reqs = bundle.notification_pipeline.sent_requests
        self.assertEqual(len(reqs), 1)
        self.assertEqual(reqs[0].channel, "mail")
        self.assertEqual(reqs[0].title, PRODUCTION_MAIL_TITLE)
        self.assertIsNone(reqs[0].destination)

    def test_no_real_send(self):
        ev = _EvalRec()
        bundle = _fake_bundle(ev)
        result = run_one_race(bundle, "20260704", 12, 5, {"public": "/tmp/p.html"})
        # すべて sent=False（実送信していない）
        self.assertTrue(all(not r.sent for r in result["notification_results"]))

    def test_fallback_to_assess_race_when_no_decide(self):
        # assess_and_decide が ValueError の構成でも assess_race で継続
        ev = _EvalRec()
        bundle = _fake_bundle(ev)

        def _raise(evaluation):
            raise ValueError("no decision builder")

        bundle.buy_pipeline.assess_and_decide = _raise
        result = run_one_race(bundle, "20260704", 12, 5, {"public": "/tmp/p.html"})
        self.assertIsNone(result["buy_decision"])


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
