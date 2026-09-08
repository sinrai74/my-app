"""
BuyPipeline（Step5-3、W案で拡張）: RaceEvaluation → Prediction →
BuyAssessment / LegacyPurchaseResult → BuyDecision の結線。

役割: PredictionProvider（Prediction生成の実体をラップ）→ BuyEngine.assess →
      PurchaseResultSource（Legacy購入確定結果の取得）→
      BuyDecisionBuilder.build を「順に呼ぶだけ」。

厳守（Step5-3指示・結線のみ・案A承認、およびW案指示）:
  - BuyPipelineは計算・判定・補正・EV再計算・buyscore再計算・Kelly計算・
    investment_type生成・race_type変換をしない。
  - RaceEvaluation / Prediction / BuyAssessment / LegacyPurchaseResult を
    変更しない。
  - BuyDecisionの中身（combo/amount/n_bets/cost等）の算出はBuyDecisionBuilder
    の責務。BuyPipelineは戻り値をそのまま返すだけ。

【W案での設計】
  Predictionはbest1点モデルとして現状維持（K1、既存契約）。Legacyの実際の
  購入確定結果（最大4点）は、Prediction.patternsからではなく、
  PurchaseResultSource（実体は_EvaluateBetsCapture、shadow層でDI）から
  eval_id指定で取得する。これは「_evaluate_betsを1レースにつき1回だけ
  呼ぶ」という制約（資金管理ロジックの冪等性非保証）を満たすため、
  PredictionProvider.provide()の呼び出し直後・同一レース処理内で取得する
  必要がある。

  既存の assess_race() は変更しない（既存呼び出し元・既存テストへの後方互換
  を維持するため、シグネチャ・返り値ともに従来通りBuyAssessmentのみ返す）。
  decide_race() は、purchase_result_source が注入されている場合のみ利用可能
  （未注入の場合はBuyAssessmentのみの従来動作＝assess_raceと等価な範囲に
  留め、BuyDecisionは生成しない＝ValueError）。

Prediction生成:
  ラップ方式（Step5-0確定）。既存 _evaluate_bets 等の実体は PredictionProvider
  としてDI注入し、Pipelineはその実装を知らない。Prediction生成ロジックの
  分解・再実装・コピーはしない。

依存（Protocolのみ参照・具象非依存。pipelines層はshadow層を直接importしない）:
  - PredictionProvider: RaceEvaluation → Prediction（実体はDI）
  - _BuyEngine（core.buyscore.BuyEngine）: assess
  - PurchaseResultSource: eval_id → LegacyPurchaseResult（実体はDI・任意）
  - BuyDecisionBuilder: LegacyPurchaseResult + BuyAssessment → BuyDecision
    （実体はDI・任意）
  - config: freeze config（読むだけ）
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Protocol

from core.buyscore import BuyAssessment
from models.evaluation import (
    BuyDecision,
    LegacyPurchaseResult,
    Prediction,
    RaceEvaluation,
)

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


class PurchaseResultSource(Protocol):
    """Legacyの購入確定結果（最大4点）をeval_id指定で取得するProtocol。

    実体は shadow._EvaluateBetsCapture を想定するが、Pipelineは
    その実装を知らない。PredictionProviderと同一の_evaluate_bets呼び出し
    結果を共有するため、同一レース処理内でprovide()呼び出し直後に
    last_purchase_result(eval_id)を呼ぶことを前提とする（2回呼び出し禁止）。
    """

    def last_purchase_result(self, eval_id: str) -> LegacyPurchaseResult: ...


class _BuyDecisionBuilder(Protocol):
    """pipelines.buy_decision_builder.BuyDecisionBuilder のうち本Pipelineが使う部分。"""

    def build(
        self, purchase_result: LegacyPurchaseResult, assessment: BuyAssessment
    ) -> BuyDecision: ...


class BuyPipeline:
    """買い判定の結線パイプライン（計算しない）。"""

    def __init__(
        self,
        prediction_provider: PredictionProvider,
        buy_engine: _BuyEngine,
        config: dict,
        purchase_result_source: Optional[PurchaseResultSource] = None,
        decision_builder: Optional[_BuyDecisionBuilder] = None,
    ) -> None:
        self._prediction_provider = prediction_provider
        self._buy_engine = buy_engine
        self._config = config
        self._purchase_result_source = purchase_result_source
        self._decision_builder = decision_builder

    def assess_race(self, evaluation: RaceEvaluation) -> BuyAssessment:
        """1レースの評価から BuyAssessment を得る（結線のみ・従来と同一挙動）。

        順序:
          1. prediction = prediction_provider.provide(evaluation, config)
          2. assessment = buy_engine.assess(evaluation, prediction, config)
        いずれも戻り値を次へ渡すだけ。加工・判定はしない。
        """
        _, assessment = self._predict_and_assess(evaluation)
        return assessment

    def decide_race(self, evaluation: RaceEvaluation) -> BuyDecision:
        """1レースの評価から BuyDecision まで得る（結線のみ）。

        順序:
          1. prediction, assessment = 既存の predict→assess（assess_raceと同一処理）
             ※ この時点で purchase_result_source（_EvaluateBetsCapture）に
               同一レースの_evaluate_bets生結果が捕捉されている
          2. purchase_result = purchase_result_source.last_purchase_result(
                 evaluation.eval_id)
          3. decision = decision_builder.build(purchase_result, assessment)
        purchase_result_source / decision_builder が未注入の場合はValueError
        （暫定値での穴埋め禁止）。
        """
        if self._purchase_result_source is None or self._decision_builder is None:
            raise ValueError(
                "decide_race() requires both purchase_result_source and "
                "decision_builder to be configured"
            )
        prediction, assessment = self._predict_and_assess(evaluation)

        purchase_result = self._purchase_result_source.last_purchase_result(
            evaluation.eval_id
        )
        log.info("BuyPipeline decision_builder start eval_id=%s", evaluation.eval_id)
        decision = self._decision_builder.build(purchase_result, assessment)
        log.info("BuyPipeline decision ready eval_id=%s", evaluation.eval_id)
        return decision

    def _predict_and_assess(
        self, evaluation: RaceEvaluation
    ) -> tuple[Prediction, BuyAssessment]:
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
        return prediction, assessment
