"""
Output層（output/renderers.py・pipelines/output_pipeline.py）の単体テスト（Step5-4）。

指示のテスト要件に対応:
  HTML一致 / CSV一致 / TXT一致 / Renderer Mock / Protocol利用確認 /
  Shadow比較 / Legacy一致。

Legacy Rendererへの実接続を避け、renderer_funcにFakeを注入して検証する
（Output層が「呼ぶだけ」で生成物に手を加えないことの確認が目的）。
Shadow比較は「既存Renderer出力」と「Wrapper経由出力」のbyte一致を確認する。
"""

from __future__ import annotations

import os
import tempfile
import unittest

from output.renderers import (
    DeveloperHtmlRenderer,
    HtmlRenderer,
    OutputRenderer,
    PublicHtmlRenderer,
    RenderResult,
    ResultsHtmlRenderer,
)
from pipelines.output_pipeline import OutputPipeline


class TestHtmlRendererWrapping(unittest.TestCase):
    """Wrapperは既存Rendererを呼ぶだけ・結果を加工しない。"""

    def test_calls_legacy_and_returns_result(self) -> None:
        calls = []

        def _fake_renderer(date_str, output_path):
            calls.append((date_str, output_path))
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("<html>legacy</html>")
            return {"count": 3}

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out_test.html")
            renderer = HtmlRenderer("test", renderer_func=_fake_renderer)
            result = renderer.render("20260704", path)
            self.assertEqual(calls, [("20260704", path)])
            self.assertEqual(result.output_path, path)
            self.assertEqual(result.summary, {"count": 3})

    def test_summary_passed_through_unmodified(self) -> None:
        summary = {"today": {"n": 5}, "d30": {"roi": 1.2}}
        renderer = HtmlRenderer(
            "test", renderer_func=lambda d, p: summary
        )
        result = renderer.render("20260704", os.path.join(tempfile.gettempdir(), "x.html"))
        self.assertIs(result.summary, summary)  # 加工せずそのまま

    def test_no_func_no_loader_raises(self) -> None:
        renderer = HtmlRenderer("broken", renderer_func=None, loader=None)
        with self.assertRaises(ValueError):
            renderer.render("20260704", os.path.join(tempfile.gettempdir(), "x.html"))


class TestByteIdentity(unittest.TestCase):
    """HTML/CSV/TXTのbyte一致: Wrapper経由の生成物が既存Renderer出力と同一。"""

    def _run_and_read(self, tmp_path, content):
        def _renderer(date_str, output_path):
            with open(output_path, "wb") as f:
                f.write(content)
            return {}
        HtmlRenderer("t", renderer_func=_renderer).render("20260704", tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()

    def test_html_byte_identity(self) -> None:
        html = b"<html><body>result</body></html>"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.html")
            # 既存Renderer直呼び
            def _legacy(date_str, output_path):
                with open(output_path, "wb") as f:
                    f.write(html)
                return {}
            _legacy("20260704", path)
            with open(path, "rb") as f:
                legacy_bytes = f.read()
            # Wrapper経由
            wrapper_bytes = self._run_and_read(path, html)
            self.assertEqual(legacy_bytes, wrapper_bytes)

    def test_csv_byte_identity(self) -> None:
        csv = b"date,venue,hit\n20260704,12,1\n"
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.csv")
            self.assertEqual(self._run_and_read(path, csv), csv)

    def test_txt_byte_identity(self) -> None:
        txt = "予測データ\n1-2-3\n".encode("utf-8")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r.txt")
            self.assertEqual(self._run_and_read(path, txt), txt)


class TestShadowComparison(unittest.TestCase):
    """Shadow比較: 既存Renderer出力とWrapper出力のbyte一致。"""

    def test_shadow_html_equal(self) -> None:

        html = b"<html>shadow</html>"

        def _legacy(date_str, output_path):
            with open(output_path, "wb") as f:
                f.write(html)
            return {"n": 1}

        with tempfile.TemporaryDirectory() as d:
            legacy_path = os.path.join(d, "legacy.html")
            rebuild_path = os.path.join(d, "rebuild.html")
            # legacy直呼び
            _legacy("20260704", legacy_path)
            # rebuild: 同じ関数をWrapper経由で
            HtmlRenderer("public", renderer_func=_legacy).render(
                "20260704", rebuild_path
            )
            with open(legacy_path, "rb") as f:
                legacy_bytes = f.read()
            with open(rebuild_path, "rb") as f:
                rebuild_bytes = f.read()
            # byte一致（差分なし）
            self.assertEqual(legacy_bytes, rebuild_bytes)


class TestConcreteRenderersUseLoaders(unittest.TestCase):
    """Public/Results/Developerが正しいLegacy loaderを指すこと（呼ばずに確認）。"""

    def test_loaders_are_set(self) -> None:
        for cls in (PublicHtmlRenderer, ResultsHtmlRenderer, DeveloperHtmlRenderer):
            renderer = cls()
            self.assertIsNotNone(renderer._loader)

    def test_injected_func_overrides_loader(self) -> None:
        called = []
        renderer = PublicHtmlRenderer(renderer_func=lambda d, p: called.append(1) or {})
        renderer.render("20260704", os.path.join(tempfile.gettempdir(), "x.html"))
        self.assertEqual(called, [1])  # loaderではなく注入funcが使われる


class TestProtocolUsage(unittest.TestCase):
    def test_renderers_satisfy_protocol(self) -> None:
        renderer: OutputRenderer = HtmlRenderer("t", renderer_func=lambda d, p: {})
        result = renderer.render("20260704", os.path.join(tempfile.gettempdir(), "x.html"))
        self.assertIsInstance(result, RenderResult)


class TestOutputPipeline(unittest.TestCase):
    def test_render_all_calls_each(self) -> None:
        calls = []
        r1 = HtmlRenderer("a", renderer_func=lambda d, p: calls.append(("a", p)) or {})
        r2 = HtmlRenderer("b", renderer_func=lambda d, p: calls.append(("b", p)) or {})
        pipeline = OutputPipeline({"public": r1, "results": r2})
        pa = os.path.join(tempfile.gettempdir(), "a.html")
        pb = os.path.join(tempfile.gettempdir(), "b.html")
        results = pipeline.render_all("20260704", {"public": pa, "results": pb})
        self.assertEqual(set(results.keys()), {"public", "results"})
        self.assertEqual(calls, [("a", pa), ("b", pb)])

    def test_key_mismatch_raises(self) -> None:
        pipeline = OutputPipeline({"public": HtmlRenderer("a", renderer_func=lambda d, p: {})})
        with self.assertRaises(ValueError):
            pipeline.render_all("20260704", {
                "public": os.path.join(tempfile.gettempdir(), "a.html"),
                "results": os.path.join(tempfile.gettempdir(), "b.html"),
            })

    def test_empty_renderers(self) -> None:
        pipeline = OutputPipeline({})
        self.assertEqual(pipeline.render_all("20260704", {}), {})


if __name__ == "__main__":
    unittest.main()
