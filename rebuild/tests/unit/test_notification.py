"""
Notification層のテスト（Step5-5）。

指示のテスト要件に対応:
  正常系 / MailNotifier Mock / LineNotifier Mock / DiscordNotifier Mock /
  XNotifier Mock / NotificationService Mock / Protocol利用確認 /
  RenderResult受け渡し確認 / render_all辞書API確認。

Shadow比較（指示）: 通知内容ではなくNotifierへの入力（NotificationRequest=
  RenderResult/通知先/添付/タイトル/種別）を比較する。送信API自体は比較しない。
実送信は行わず、senderをFakeで注入する。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from notification.notifiers import (
    DiscordNotifier,
    LineNotifier,
    MailNotifier,
    NotificationRequest,
    NotificationResult,
    Notifier,
    XNotifier,
)
from notification.service import NotificationService
from output.renderers import HtmlRenderer, RenderResult
from pipelines.notification_pipeline import NotificationPipeline
from pipelines.output_pipeline import OutputPipeline


def _render_result(path="/out/public.html", summary=None) -> RenderResult:
    return RenderResult(output_path=path, summary=summary or {"n": 3})


def _request(channel="mail", **over) -> NotificationRequest:
    base = dict(
        render_result=_render_result(), channel=channel,
        destination="dest@example.com", title="実績", attachment_path=None,
    )
    base.update(over)
    return NotificationRequest(**base)


class TestMailNotifier(unittest.TestCase):
    def test_calls_legacy_send_email(self) -> None:
        calls = []
        notifier = MailNotifier(sender=lambda subj, body: calls.append((subj, body)) or True)
        result = notifier.notify(_request(channel="mail", title="件名"))
        self.assertEqual(result.channel, "mail")
        self.assertTrue(result.sent)
        self.assertEqual(calls[0][0], "件名")  # subject=title
        self.assertEqual(calls[0][1], "/out/public.html")  # body=output_path

    def test_send_failure(self) -> None:
        notifier = MailNotifier(sender=lambda subj, body: False)
        self.assertFalse(notifier.notify(_request()).sent)


class TestLineNotifier(unittest.TestCase):
    def test_calls_legacy_send_line(self) -> None:
        calls = []
        notifier = LineNotifier(sender=lambda body: calls.append(body) or True)
        result = notifier.notify(_request(channel="line"))
        self.assertEqual(result.channel, "line")
        self.assertTrue(result.sent)
        self.assertEqual(calls[0], "/out/public.html")


class TestDiscordNotifier(unittest.TestCase):
    def test_with_injected_sender(self) -> None:
        calls = []
        notifier = DiscordNotifier(sender=lambda body: calls.append(body) or True)
        self.assertTrue(notifier.notify(_request(channel="discord")).sent)
        self.assertEqual(calls[0], "/out/public.html")

    def test_unwired_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            DiscordNotifier().notify(_request(channel="discord"))


class TestXNotifier(unittest.TestCase):
    def test_with_injected_sender(self) -> None:
        notifier = XNotifier(sender=lambda body: True)
        self.assertTrue(notifier.notify(_request(channel="x")).sent)

    def test_unwired_raises(self) -> None:
        with self.assertRaises(NotImplementedError):
            XNotifier().notify(_request(channel="x"))


class TestNotificationService(unittest.TestCase):
    def test_dispatches_by_channel(self) -> None:
        mail_calls, line_calls = [], []
        service = NotificationService({
            "mail": MailNotifier(sender=lambda s, b: mail_calls.append(b) or True),
            "line": LineNotifier(sender=lambda b: line_calls.append(b) or True),
        })
        service.send(_request(channel="mail"))
        service.send(_request(channel="line"))
        self.assertEqual(len(mail_calls), 1)
        self.assertEqual(len(line_calls), 1)

    def test_unknown_channel_raises(self) -> None:
        service = NotificationService({"mail": MailNotifier(sender=lambda s, b: True)})
        with self.assertRaises(ValueError):
            service.send(_request(channel="discord"))

    def test_service_with_mock_notifier(self) -> None:
        """NotificationService Mock: Notifier差し替えで結果を制御。"""
        class _MockNotifier:
            channel = "mail"

            def __init__(self):
                self.received = None

            def notify(self, request):
                self.received = request
                return NotificationResult("mail", True, "mock")

        mock = _MockNotifier()
        service = NotificationService({"mail": mock})
        result = service.send(_request(channel="mail"))
        self.assertEqual(result.detail, "mock")
        self.assertIsNotNone(mock.received)


class TestRenderResultPassing(unittest.TestCase):
    """RenderResult受け渡し確認: Notifierへ渡るのはRenderResultそのもの。"""

    def test_render_result_reaches_notifier_unmodified(self) -> None:
        received = {}

        class _Capture:
            channel = "mail"

            def notify(self, request):
                received["rr"] = request.render_result
                return NotificationResult("mail", True)

        rr = _render_result(path="/x/y.html", summary={"k": 1})
        service = NotificationService({"mail": _Capture()})
        service.send(NotificationRequest(render_result=rr, channel="mail"))
        # 同一オブジェクトが編集されず渡る
        self.assertIs(received["rr"], rr)
        self.assertEqual(received["rr"].output_path, "/x/y.html")
        self.assertEqual(received["rr"].summary, {"k": 1})

    def test_notification_does_not_reference_core_models(self) -> None:
        """入力はRenderResultのみ（Coreモデルを持たない）。"""
        req = _request()
        # NotificationRequestのフィールドにCoreモデル型が無いこと
        self.assertFalse(hasattr(req, "evaluation"))
        self.assertFalse(hasattr(req, "prediction"))
        self.assertFalse(hasattr(req, "assessment"))


class TestProtocolUsage(unittest.TestCase):
    def test_notifiers_satisfy_protocol(self) -> None:
        notifier: Notifier = MailNotifier(sender=lambda s, b: True)
        result = notifier.notify(_request())
        self.assertIsInstance(result, NotificationResult)

    def test_all_notifiers_have_channel(self) -> None:
        self.assertEqual(MailNotifier(sender=lambda s, b: True).channel, "mail")
        self.assertEqual(LineNotifier(sender=lambda b: True).channel, "line")
        self.assertEqual(DiscordNotifier(sender=lambda b: True).channel, "discord")
        self.assertEqual(XNotifier(sender=lambda b: True).channel, "x")


class TestNotificationPipeline(unittest.TestCase):
    def test_send_all(self) -> None:
        sent = []
        service = NotificationService({
            "mail": MailNotifier(sender=lambda s, b: sent.append("mail") or True),
            "line": LineNotifier(sender=lambda b: sent.append("line") or True),
        })
        pipeline = NotificationPipeline(service)
        results = pipeline.send_all([_request(channel="mail"), _request(channel="line")])
        self.assertEqual(len(results), 2)
        self.assertEqual(sent, ["mail", "line"])

    def test_normal_flow(self) -> None:
        """Notification正常系: RenderResult → Service → Notifier → 結果。"""
        service = NotificationService({"mail": MailNotifier(sender=lambda s, b: True)})
        result = NotificationPipeline(service).send_all([_request(channel="mail")])
        self.assertTrue(result[0].sent)


# ---------- Shadow比較（Notifierへの入力=NotificationRequestを比較） ----------


def _request_to_dict(req: NotificationRequest) -> dict:
    return {
        "channel": req.channel,
        "destination": req.destination,
        "title": req.title,
        "attachment_path": req.attachment_path,
        "output_path": req.render_result.output_path,
    }


def _shadow_diff(eval_id, legacy_obj, rebuild_obj, path="$"):
    """Step5-0/5-2/5-3と同方式の再帰比較。差分を返す。"""
    diffs = []
    legacy = json.loads(json.dumps(legacy_obj, ensure_ascii=False, sort_keys=True))
    rebuild = json.loads(json.dumps(rebuild_obj, ensure_ascii=False, sort_keys=True))

    def rec(le, re, p):
        if isinstance(le, dict) and isinstance(re, dict):
            for k in sorted(set(le) | set(re)):
                rec(le.get(k), re.get(k), f"{p}.{k}")
        elif le != re:
            diffs.append({"eval_id": eval_id, "field_path": p,
                          "legacy": le, "rebuild": re})

    rec(legacy, rebuild, path)
    return diffs


class TestShadowInputComparison(unittest.TestCase):
    def test_request_input_equal(self) -> None:
        legacy = _request(channel="mail", title="実績", destination="a@b.com")
        rebuild = _request(channel="mail", title="実績", destination="a@b.com")
        diffs = _shadow_diff(
            "20260704_12_05",
            _request_to_dict(legacy), _request_to_dict(rebuild),
        )
        self.assertEqual(diffs, [])

    def test_request_input_diff_detected(self) -> None:
        diffs = _shadow_diff(
            "20260704_12_05",
            _request_to_dict(_request(title="A")),
            _request_to_dict(_request(title="B")),
        )
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["field_path"], "$.title")


class TestRenderAllDictAPI(unittest.TestCase):
    """render_all が名前付きマッピング（dict）APIであることの確認。"""

    def test_render_all_takes_and_returns_dict(self) -> None:
        r = HtmlRenderer("pub", renderer_func=lambda d, p: {"n": 1})
        pipeline = OutputPipeline({"public": r})
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "public.html")
            results = pipeline.render_all("20260704", {"public": path})
        # 戻り値はdict（listではない）
        self.assertIsInstance(results, dict)
        self.assertIn("public", results)
        self.assertIsInstance(results["public"], RenderResult)

    def test_render_all_rejects_key_mismatch(self) -> None:
        r = HtmlRenderer("pub", renderer_func=lambda d, p: {})
        pipeline = OutputPipeline({"public": r})
        with self.assertRaises(ValueError):
            pipeline.render_all("20260704", {"results": "/x.html"})


if __name__ == "__main__":
    unittest.main()
