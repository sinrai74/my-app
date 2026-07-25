"""
OutputPipeline（Step5-4）: 複数のOutput Rendererを順に呼ぶ結線のみ。

責務: 注入されたRenderer群（Protocol）を date_str で順に呼び、生成結果を集める。
禁止: 判定・計算・集計・データ読込・レイアウト生成・保存・通知。
  Renderer選択のif分岐（if buyscore等）も持たない。呼ぶだけ。

依存: pipelines → output(Protocol)。具象Rendererは外から注入する。
"""

from __future__ import annotations

import logging
import time
from typing import Mapping

from output.renderers import OutputRenderer, RenderResult

log = logging.getLogger(__name__)


class OutputPipeline:
    """Renderer群の結線（計算しない）。名前付きマッピングで管理する。"""

    def __init__(self, renderers: Mapping[str, OutputRenderer]) -> None:
        self._renderers = dict(renderers)

    def render_all(
        self, date_str: str, output_paths: Mapping[str, str]
    ) -> dict[str, RenderResult]:
        """各Rendererを名前付きで対応するoutput_pathで呼ぶ（名前付きマッピングAPI）。

        renderersとoutput_pathsは同じキー集合で名前対応づける
        （順番依存のlistではなく辞書。Step5-5レビューでAPI明確化）。
        キーの対応づけの判定・並べ替えはしない（呼び出し側がキーで用意する）。
        戻り値も同じキーのRenderResult辞書。
        """
        if set(output_paths.keys()) != set(self._renderers.keys()):
            raise ValueError(
                f"renderers keys({sorted(self._renderers)}) and "
                f"output_paths keys({sorted(output_paths)}) mismatch"
            )
        start = time.monotonic()
        log.info("OutputPipeline start date=%s count=%d",
                 date_str, len(self._renderers))
        results = {
            name: self._renderers[name].render(date_str, output_paths[name])
            for name in self._renderers
        }
        log.info("OutputPipeline end date=%s elapsed=%.3fs",
                 date_str, time.monotonic() - start)
        return results
