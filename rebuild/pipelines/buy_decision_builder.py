"""
BuyDecisionBuilder（W案・最終）: LegacyPurchaseResult + BuyAssessment → BuyDecision
の組み立てのみ（計算・判定はしない）。

## 設計変更の経緯（Stage 1当初案からの訂正）

当初、本Builderは `Prediction.patterns` から購入結果を取り出す設計だった。
しかし、実リポジトリ検証の結果、`actions/shadow_entrypoint.py::build_bundle`
の既存配線（`_evaluate_bets_first`、Step6-2c-9・K1）が、Legacyの返り値list
のうち `result[0]` のみをPredictionへ渡し、残りを破棄することが判明した。
Predictionは「1レース1件のbest1点モデル」（Design Spec §3.5）として最初
から設計されており、これは正しい既存契約である。問題はPrediction側ではなく、
「Legacyの購入確定結果（最大4点）を運ぶ場所がシステムに存在しなかった」
ことであった。

この訂正を受け、本Builderの入力を `Prediction` から `LegacyPurchaseResult`
（Legacyの生返却値listをそのまま保持するimmutable Context、models.evaluation
で定義）へ変更した。`Prediction.patterns` を購入結果の入力元にすることは
今後も行わない。

## 責務

- LegacyPurchaseResult.purchases（Legacyが既に選定・金額確定した候補リスト）
  から、購入対象combo集合・各金額・購入点数・総投資額を「そのまま取り出す」。
  加工・再計算・フィルタリングは一切行わない。
- BuyAssessment（Rebuild独自評価: buyscore/investment_type/kelly_fraction/
  skip_reason）をそのまま転記する。
- 両者を単一のBuyDecisionへ束ねるだけ。

## 禁止事項

- assign_rank_labels相当の選定ロジックの再実装（不要と判明したため）
- get_bet_multiplier_extended相当の資金管理ロジックの再実装（同上）
- LegacyPurchaseResult.purchasesの中身の再計算・補正
- BuyAssessmentの値の上書き・補正
- Prediction.patternsを購入結果の入力元にすること
"""

from __future__ import annotations

import logging
from typing import Protocol

from core.buyscore import BuyAssessment
from models.evaluation import BuyDecision, LegacyPurchaseResult

log = logging.getLogger(__name__)


class BuyDecisionBuilder(Protocol):
    """LegacyPurchaseResult + BuyAssessment → BuyDecision（Protocol）。"""

    def build(
        self, purchase_result: LegacyPurchaseResult, assessment: BuyAssessment
    ) -> BuyDecision: ...


class DefaultBuyDecisionBuilder:
    """Legacyが既に確定した購入対象（LegacyPurchaseResult）をそのまま束ねる実装。"""

    def build(
        self, purchase_result: LegacyPurchaseResult, assessment: BuyAssessment
    ) -> BuyDecision:
        if purchase_result.eval_id != assessment.eval_id:
            raise ValueError(
                "eval_id mismatch: "
                f"purchase_result={purchase_result.eval_id!r} "
                f"assessment={assessment.eval_id!r}"
            )

        candidates = purchase_result.purchases

        if not candidates:
            return BuyDecision(
                eval_id=purchase_result.eval_id,
                purchased=False,
                buyscore=assessment.buyscore,
                investment_type=assessment.investment_type,
                n_bets=0,
                cost=0,
                kelly_fraction=assessment.kelly_fraction,
                config_version=assessment.config_version,
                skip_reason=assessment.skip_reason,
                purchased_combos=(),
                purchased_amounts=(),
            )

        is_purchased = bool(candidates[0].get("purchased", True))

        if not is_purchased:
            return BuyDecision(
                eval_id=purchase_result.eval_id,
                purchased=False,
                buyscore=assessment.buyscore,
                investment_type=assessment.investment_type,
                n_bets=0,
                cost=0,
                kelly_fraction=assessment.kelly_fraction,
                config_version=assessment.config_version,
                skip_reason=assessment.skip_reason,
                purchased_combos=(),
                purchased_amounts=(),
            )

        combos = tuple(c["combo"] for c in candidates)
        amounts = tuple(int(c.get("amount", 0)) for c in candidates)

        return BuyDecision(
            eval_id=purchase_result.eval_id,
            purchased=True,
            buyscore=assessment.buyscore,
            investment_type=assessment.investment_type,
            n_bets=len(candidates),
            cost=sum(amounts),
            kelly_fraction=assessment.kelly_fraction,
            config_version=assessment.config_version,
            skip_reason=None,
            purchased_combos=combos,
            purchased_amounts=amounts,
        )
