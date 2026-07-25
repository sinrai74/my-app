"""
actions層（Step5-6）: GitHub ActionsからPipelineを「呼ぶだけ」の結線。

責務: Provider→EvaluationPipeline→BuyPipeline→OutputPipeline→
  NotificationPipeline をDIで組み立てて呼ぶ。
禁止: 判定・計算・補正・HTML生成・通知生成。すべて既存部品へ委譲する。

既存のGitHub Actions（.yml）は変更しない。本パッケージは.ymlから
呼び出される想定のPythonエントリポイントとして用意する。
"""

from actions.flags import use_rebuild_pipeline
from actions.wiring import PipelineBundle, assemble_pipelines

__all__ = ["PipelineBundle", "assemble_pipelines", "use_rebuild_pipeline"]
