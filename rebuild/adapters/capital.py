"""
CapitalManager Protocol（Step5-3・定義のみ）。

資金管理（purchased/n_bets/cost確定）はStep5-4以降で実装する。
Step5-3ではProtocol境界のみ定義し、具象実装は行わない（指示準拠）。
BuyAssessment → BuyDecision への昇格に用いる予定。
"""

from __future__ import annotations

from typing import Protocol

from core.buyscore import BuyAssessment
from models.evaluation import BuyDecision


class CapitalManager(Protocol):
    """BuyAssessment＋資金状態 → BuyDecision（purchased/n_bets/cost確定）。

    入力: BuyAssessment, config（＋資金状態）
    出力: BuyDecision
    責務: 資金管理・点数制限を適用しBuyDecisionを確定する（Step5-4以降で実装）
    禁止: buyscore/investment_type/kelly/skip_reasonの再算出（BuyEngineの結果を用いる）

    注記: 本Protocolは境界定義のみ。Step5-3では具象実装しない。
    """

    def decide(
        self, assessment: BuyAssessment, config: dict
    ) -> BuyDecision: ...
