"""
Output Renderer（Step5-4）: 既存Legacy Rendererを呼ぶだけのラッパー。

案C（レビュー確定）: 既存Rendererは date_str駆動（内部でCSV/Release取得→HTML生成）。
本層はそのシグネチャ (date_str, output_path) -> dict をそのまま呼ぶ薄い層。
Rendererの内部処理・レイアウト・集計・データ読込には一切触れない。

各ラッパーが行うのは:
  - Legacy Renderer関数を遅延importして呼ぶ（import利用のみ・コピー禁止）
  - 開始/終了ログ（本文は出力しない）
  - 生成物パスの返却
判定・計算・補正・変換は行わない。
"""

from __future__ import annotations

import logging
import time
from typing import Callable, Optional, Protocol

log = logging.getLogger(__name__)

# Legacy Renderer のシグネチャ: (date_str, output_path) -> summary dict
LegacyRenderer = Callable[[str, str], dict]


class OutputRenderer(Protocol):
    """Output層の抽象。date_str → 生成物パス。

    入力: date_str(YYYYMMDD), output_path
    出力: RenderResult（生成パス＋Legacyの返すsummary dict）
    責務: 既存Rendererを呼び生成物を得る
    禁止: 判定・計算・集計・データ読込・レイアウト生成・保存・通知
    """

    def render(self, date_str: str, output_path: str) -> "RenderResult": ...


class RenderResult:
    """生成結果（生成物パスとLegacyのsummary）。不変の単純データ保持。"""

    __slots__ = ("output_path", "summary")

    def __init__(self, output_path: str, summary: dict) -> None:
        self.output_path = output_path
        self.summary = summary

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, RenderResult)
            and self.output_path == other.output_path
            and self.summary == other.summary
        )

    def __repr__(self) -> str:
        return f"RenderResult(output_path={self.output_path!r})"


class HtmlRenderer:
    """Legacy HTML Renderer 汎用ラッパー。

    renderer_func を注入すればテストでFake差し替え可能。未注入時は
    loader() で遅延importする（Legacy本番関数）。処理は「呼ぶだけ」。
    """

    def __init__(
        self,
        name: str,
        renderer_func: Optional[LegacyRenderer] = None,
        loader: Optional[Callable[[], LegacyRenderer]] = None,
    ) -> None:
        self._name = name
        self._renderer_func = renderer_func
        self._loader = loader

    def render(self, date_str: str, output_path: str) -> RenderResult:
        start = time.monotonic()
        log.info("Output start renderer=%s date=%s", self._name, date_str)

        func = self._renderer_func
        if func is None:
            if self._loader is None:
                raise ValueError(f"no renderer func/loader for {self._name}")
            func = self._loader()

        summary = func(date_str, output_path)  # Legacyへ委譲（内部処理に触れない）

        log.info(
            "Output %s generated date=%s elapsed=%.3fs",
            self._name, date_str, time.monotonic() - start,
        )
        return RenderResult(output_path=output_path, summary=summary)


def _load_public() -> LegacyRenderer:
    from x_results_public import generate_public_html
    return generate_public_html


def _load_results() -> LegacyRenderer:
    from x_results_page import generate_results_html
    return generate_results_html


def _load_developer() -> LegacyRenderer:
    from x_results_developer import generate_developer_html
    return generate_developer_html


class PublicHtmlRenderer(HtmlRenderer):
    """x_results_public.generate_public_html のラッパー。"""

    def __init__(self, renderer_func: Optional[LegacyRenderer] = None) -> None:
        super().__init__("public_html", renderer_func, _load_public)


class ResultsHtmlRenderer(HtmlRenderer):
    """x_results_page.generate_results_html のラッパー。"""

    def __init__(self, renderer_func: Optional[LegacyRenderer] = None) -> None:
        super().__init__("results_html", renderer_func, _load_results)


class DeveloperHtmlRenderer(HtmlRenderer):
    """x_results_developer.generate_developer_html のラッパー。"""

    def __init__(self, renderer_func: Optional[LegacyRenderer] = None) -> None:
        super().__init__("developer_html", renderer_func, _load_developer)
