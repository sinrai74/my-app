"""
Actions DI組立（Step5-6）: Provider→EvaluationPipeline→BuyPipeline→
OutputPipeline→NotificationPipeline を組み立てるだけ。

責務: 具象部品を受け取り、各Pipelineへ渡してPipelineBundleを返す。
禁止: 判定・計算・補正・HTML生成・通知生成。Pipeline内部へ処理を書かない
  （newした具象を渡すだけ）。

依存: actions → pipelines / notification / output（型参照のみ）。
  具象の生成は呼び出し側（実際のActionsエントリポイントや各テスト）が行う。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from notification.service import NotificationService
from pipelines.buy_pipeline import BuyPipeline, PredictionProvider, _BuyEngine
from pipelines.evaluation_pipeline import EvaluationPipeline, _EvaluationEngine, _RaceSource
from pipelines.notification_pipeline import NotificationPipeline
from pipelines.output_pipeline import OutputPipeline
from features.feature_builder import FeatureBuilder


@dataclass(frozen=True)
class PipelineBundle:
    """組み立て済みPipeline一式（DIの結果を束ねるだけの入れ物）。"""

    evaluation_pipeline: EvaluationPipeline
    buy_pipeline: BuyPipeline
    output_pipeline: OutputPipeline
    notification_pipeline: NotificationPipeline


def assemble_pipelines(
    *,
    race_source: _RaceSource,
    feature_builder: FeatureBuilder,
    engine: _EvaluationEngine,
    now_provider,
    eval_config: dict,
    durable_store=None,
    prediction_provider: PredictionProvider,
    buy_engine: _BuyEngine,
    buy_config: dict,
    output_renderers: dict,
    notification_service: Optional[NotificationService] = None,
) -> PipelineBundle:
    """各Pipelineをコンストラクタへ渡すだけの組立（計算・判定なし）。"""
    evaluation_pipeline = EvaluationPipeline(
        race_source=race_source,
        feature_builder=feature_builder,
        engine=engine,
        now_provider=now_provider,
        config=eval_config,
        durable_store=durable_store,
    )
    buy_pipeline = BuyPipeline(
        prediction_provider=prediction_provider,
        buy_engine=buy_engine,
        config=buy_config,
    )
    output_pipeline = OutputPipeline(output_renderers)
    notification_pipeline = NotificationPipeline(
        notification_service or NotificationService({})
    )
    return PipelineBundle(
        evaluation_pipeline=evaluation_pipeline,
        buy_pipeline=buy_pipeline,
        output_pipeline=output_pipeline,
        notification_pipeline=notification_pipeline,
    )
