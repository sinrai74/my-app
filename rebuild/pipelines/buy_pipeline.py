"""
BuyPipeline（Step5-3）: RaceEvaluation → Prediction → BuyAssessment の結線のみ。

役割: PredictionProvider（Prediction生成の実体をラップ）→ BuyEngine.assess を
      「順に呼ぶだけ」。

厳守（Step5-3指示・結線のみ・案A承認）:
  - BuyPipelineは計算・判定・補正・EV再計算・buyscore再計算・Kelly計算・
    investment_type生成・race_type変換をしない。
  - RaceEvaluation / Prediction を変更しない。
  - BuyDecisionは生成しない（purchased/n_bets/costは資金管理=CapitalManagerの
    責務で未実装。暫定値・既定値・仮値での穴埋めは禁止）。
  - 本Stepの終点は BuyAssessment（buyscore/investment_type/kelly_fraction/
    skip_reason/config_version）。

Prediction生成:
  ラップ方式（Step5-0確定）。既存 _evaluate_bets 等の実体は PredictionProvider
  としてDI注入し、Pipelineはその実装を知らない。Prediction生成ロジックの
  分解・再実装・コピーはしない。

依存（Protocolのみ参照・具象非依存）:
  - PredictionProvider: RaceEvaluation → Prediction（実体はDI）
  - _BuyEngine（core.buyscore.BuyEngine）: assess
  - config: freeze config（読むだけ）
  CapitalManagerはProtocol定義のみ（capital.py）。BuyPipelineは参照しない
  （BuyDecision昇格はStep5-4以降のため）。
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from core.buyscore import BuyAssessment
from models.evaluation import Prediction, RaceEvaluation

log = logging.getLogger(__name__)


class PredictionProvider(Protocol):
    """RaceEvaluation → Prediction（生成の実体はDI・Pipelineは実装を知らない）。

    入力: RaceEvaluation, config
    出力: Prediction
    責務: 既存のPrediction生成（_evaluate_bets等）をラップして呼ぶ
    禁止: EV/buyscore/Kelly/investment_typeの算出・補正（実体側の責務）
    """

    def provide(
        self, evaluation: RaceEvaluation, config: dict
    ) -> Prediction: ...


class _BuyEngine(Protocol):
    """core.buyscore.BuyEngine のうち本Pipelineが使う部分。"""

    def assess(
        self,
        evaluation: RaceEvaluation,
        prediction: Prediction,
        config: dict,
    ) -> BuyAssessment: ...


class BuyPipeline:
    """買い判定の結線パイプライン（計算しない・BuyAssessmentまで）。"""

    def __init__(
        self,
        prediction_provider: PredictionProvider,
        buy_engine: _BuyEngine,
        config: dict,
    ) -> None:
        self._prediction_provider = prediction_provider
        self._buy_engine = buy_engine
        self._config = config

    def assess_race(self, evaluation: RaceEvaluation) -> BuyAssessment:
        """1レースの評価から BuyAssessment を得る（結線のみ）。

        順序:
          1. prediction = prediction_provider.provide(evaluation, config)
          2. assessment = buy_engine.assess(evaluation, prediction, config)
        いずれも戻り値を次へ渡すだけ。加工・判定はしない。
        """
        start = time.monotonic()
        log.info("BuyPipeline start eval_id=%s", evaluation.eval_id)

        prediction = self._prediction_provider.provide(evaluation, self._config)
        log.info("BuyPipeline prediction ready eval_id=%s", evaluation.eval_id)

        log.info("BuyPipeline buy_engine start eval_id=%s", evaluation.eval_id)
        assessment = self._buy_engine.assess(evaluation, prediction, self._config)
        log.info("BuyPipeline assessment ready eval_id=%s", evaluation.eval_id)

        log.info(
            "BuyPipeline end eval_id=%s elapsed=%.3fs",
            evaluation.eval_id, time.monotonic() - start,
        )
        return assessment
