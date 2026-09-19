"""
per-race通知の本文formatter（整形のみ・計算しない）。

責務: RaceEvaluation / Prediction / BuyDecision の既存確定値を plain text へ
整形するだけ。評価・予測・Buy判定・スコア計算・ラベル生成は一切行わない
（Evaluate Once, Publish Everywhere: 値は上流で確定済み、本層は整形のみ）。

本文仕様（ユーザー確定・案C）:
  - 数値フォーマット: upset_score=小数1桁 / pred_prob=%・小数1桁 /
    pred_ev,pred_odds,confidence,buyscore=小数2桁 / kelly_fraction=小数第3位 /
    cost=整数+円 / n_bets=整数+点
  - Optional値がNone → "未記録"
  - 理由は Prediction.why_bet を使用（upset_reasonsから再構成しない）
  - investment_type はそのまま表示
  - purchased_combos は全件表示
"""

from __future__ import annotations

from typing import Optional

from core.buyscore import BuyAssessment  # noqa: F401 (型参照の明示用)
from models.evaluation import BuyDecision, Prediction, RaceEvaluation

_MISSING = "未記録"


def _fmt(value: Optional[float], digits: int, suffix: str = "") -> str:
    """Optional数値を指定桁で整形。Noneは"未記録"。計算はしない。"""
    if value is None:
        return _MISSING
    return f"{value:.{digits}f}{suffix}"


def _fmt_percent1(value: Optional[float]) -> str:
    """確率を%・小数1桁で表示（0.046 -> 4.6%）。Noneは"未記録"。"""
    if value is None:
        return _MISSING
    return f"{value * 100:.1f}%"


def format_race_body(
    evaluation: RaceEvaluation,
    prediction: Prediction,
    decision: BuyDecision,
) -> str:
    """per-race通知の plain text 本文を生成する（整形のみ）。

    purchased=False の扱い（通知するか否か）は呼び出し側の責務であり、
    本formatterは渡された値をそのまま整形する。
    """
    combos = "、".join(decision.purchased_combos) if decision.purchased_combos else _MISSING

    lines = [
        f"【{evaluation.venue_name} {evaluation.race_number}R】{evaluation.race_type}",
        f"波乱度: {_fmt(evaluation.upset_score, 1)} / "
        f"危険度: {_fmt(evaluation.danger_score, 1)}",
        "",
        "■ 予測",
        f"買い目: {prediction.pred_combo}",
        f"確率: {_fmt_percent1(prediction.pred_prob)} / "
        f"EV: {_fmt(prediction.pred_ev, 2)} / "
        f"オッズ: {_fmt(prediction.pred_odds, 2)} / "
        f"信頼度: {_fmt(prediction.confidence, 2)}",
        f"理由: {prediction.why_bet}",
        "",
        "■ 買い判断",
        f"購入: {'あり' if decision.purchased else 'なし'}",
        f"投資タイプ: {decision.investment_type}",
        f"BuyScore: {_fmt(decision.buyscore, 2)}",
        f"Kelly: {_fmt(decision.kelly_fraction, 3)}",
        f"点数: {decision.n_bets}点 / 金額: {decision.cost}円",
        f"購入買い目: {combos}",
    ]
    return "\n".join(lines)


def format_race_subject(evaluation: RaceEvaluation) -> str:
    """per-race通知の件名を生成する（整形のみ）。

    仕様（ユーザー確定）: 【競艇AI】{venue_name} {race_number}R 予想
    """
    return f"【競艇AI】{evaluation.venue_name} {evaluation.race_number}R 予想"
