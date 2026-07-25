"""
output層（Step5-4）: 既存Legacy Rendererのラッパー（date_str駆動）。

責務（Step5-4指示・案C確定）:
  date_str を受け取り、既存の Legacy Renderer（generate_public_html 等）を
  importして呼ぶだけ。HTMLの生成・レイアウト・集計・データ読込は
  すべて既存Renderer側の責務であり、本層は一切触れない。

禁止: 判定・計算・補正・データ読込・集計・レイアウト生成・モデル→HTML変換・
      Renderer再実装・保存(DurableStore)・通知。

依存: output → Legacy Renderer（import利用のみ）。Coreモデルには依存しない
  （date_str駆動のため）。
"""

from output.renderers import (
    HtmlRenderer,
    OutputRenderer,
    PublicHtmlRenderer,
    DeveloperHtmlRenderer,
    ResultsHtmlRenderer,
)

__all__ = [
    "DeveloperHtmlRenderer",
    "HtmlRenderer",
    "OutputRenderer",
    "PublicHtmlRenderer",
    "ResultsHtmlRenderer",
]
