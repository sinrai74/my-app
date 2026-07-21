"""
pipelines層（Step5-2〜）: 結線・オーケストレーション。

責務: Provider取得 → FeatureBuilder → Core Engine → (保存) を「呼ぶだけ」で束ねる。
禁止: 計算・判定・補正・分類変換・investment_type生成・race_type変換・
      評価ロジックの再実装。すべて既存部品（core/features/adapters）へ委譲する。

依存規則: pipelines → adapters(Protocol) / features / core / storage / models。
  逆方向依存は禁止。pipelinesは具象ではなくProtocolを参照する。
"""

from pipelines.evaluation_pipeline import EvaluationPipeline

__all__ = ["EvaluationPipeline"]
