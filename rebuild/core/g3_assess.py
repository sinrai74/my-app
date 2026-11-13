"""
正式G3（B案）：入力つきの評価。

既存の `core.buyscore.DefaultBuyEngine.assess`（294〜340行目）と同じ順序で、
core.buyscore の純関数を呼ぶ。計算の式は持たない。

- assess_equivalent: 入力の追加なし。既存の assess と同値であること（同値テストで検証）。
- assess_with_input: 入力の追加あり（BuyScoreInput と MlBonusResult）。

既存の DefaultBuyEngine.assess は変更しない。core/__init__.py からは import しない
（正式G2 の評価側 baseline の42ファイルを変更しないため）。
"""

from __future__ import annotations

from typing import Any, Optional

from core.buyscore import (
    BuyAssessment,
    DefaultKellyStrategy,
    KellyStrategy,
    calc_buyscore,
    check_passthrough,
    investment_type,
)
from core.exceptions import ValidationError
from core.rank import compute_match_index
from models.evaluation import Prediction, RaceEvaluation
from models.g3_input import BuyScoreInput, MlBonusResult


def _validate_common(evaluation: Any, prediction: Any) -> None:
    # 既存 assess の 300〜313行目と同じ順序・同じ条件
    if evaluation is None or prediction is None:
        raise ValidationError("evaluation and prediction are required")
    if evaluation.eval_id != prediction.eval_id:
        raise ValidationError(
            f"eval_id mismatch: evaluation={evaluation.eval_id!r} "
            f"prediction={prediction.eval_id!r}"
        )
    for field_name, value in (
        ("pred_prob", prediction.pred_prob),
        ("pred_ev", prediction.pred_ev),
        ("pred_odds", prediction.pred_odds),
    ):
        if value is None:
            raise ValidationError(f"prediction.{field_name} is required (None)")


def _finish(
    evaluation: RaceEvaluation,
    candidate: dict[str, Any],
    context: dict[str, Any],
    config: dict,
    kelly: KellyStrategy,
) -> BuyAssessment:
    # 既存 assess の 327〜340行目と同じ順序
    score = calc_buyscore(candidate, context, config)
    kelly_value = kelly(
        candidate.get("prob", 0), candidate.get("odds", 0), score, config
    )
    scored_candidate = dict(candidate)
    scored_candidate["buyscore"] = score
    skip = check_passthrough([scored_candidate], context, config)
    inv_type = investment_type(
        score, candidate.get("ev", 0.0), candidate.get("odds", 0.0),
        context.get("race_type", "混戦"),
    )
    return BuyAssessment(
        eval_id=evaluation.eval_id,
        buyscore=score,
        investment_type=inv_type,
        kelly_fraction=kelly_value,
        skip_reason=skip,
        config_version=str(config.get("_version", "")),
    )


def assess_equivalent(
    evaluation: RaceEvaluation,
    prediction: Prediction,
    config: dict,
    kelly_strategy: Optional[KellyStrategy] = None,
) -> BuyAssessment:
    kelly: KellyStrategy = kelly_strategy or DefaultKellyStrategy()
    _validate_common(evaluation, prediction)
    candidate: dict[str, Any] = {
        "combo": prediction.pred_combo,
        "prob": prediction.pred_prob,
        "ev": prediction.pred_ev,
        "odds": prediction.pred_odds,
    }
    context: dict[str, Any] = {
        "match_index": evaluation.match_index,
        "race_type": evaluation.race_type,
        "upset_score": evaluation.upset_score,
    }
    return _finish(evaluation, candidate, context, config, kelly)


def build_input_context(
    evaluation: RaceEvaluation, bonus: MlBonusResult, buy_input: BuyScoreInput
) -> dict[str, Any]:
    """入力ありの context（Legacy の buyscore_context と同じ7キーの形）。"""
    return {
        "match_index": compute_match_index(bonus.score_post),
        "match_index_approx": buy_input.match_index_approx,
        "race_type": evaluation.race_type,
        "has_exhibition": buy_input.has_exhibition,
        "market_gap": buy_input.market_gap,
        "upset_score": bonus.score_post,
        "ex_rank_1st": 0,
    }


def build_input_candidate(
    prediction: Prediction, buy_input: BuyScoreInput
) -> dict[str, Any]:
    return {
        "combo": prediction.pred_combo,
        "prob": prediction.pred_prob,
        "ev": prediction.pred_ev,
        "odds": prediction.pred_odds,
        "uncertainty": buy_input.uncertainty,
        "disagreement": buy_input.disagreement,
        "composite": buy_input.composite,
    }


def assess_with_input(
    evaluation: RaceEvaluation,
    prediction: Prediction,
    config: dict,
    bonus: MlBonusResult,
    buy_input: BuyScoreInput,
    kelly_strategy: Optional[KellyStrategy] = None,
) -> BuyAssessment:
    kelly: KellyStrategy = kelly_strategy or DefaultKellyStrategy()
    _validate_common(evaluation, prediction)
    if not isinstance(bonus, MlBonusResult):
        raise ValidationError("bonus must be MlBonusResult")
    if not isinstance(buy_input, BuyScoreInput):
        raise ValidationError("buy_input must be BuyScoreInput")
    if buy_input.eval_id != evaluation.eval_id:
        raise ValidationError(
            f"buy_input eval_id mismatch: {buy_input.eval_id!r} != {evaluation.eval_id!r}"
        )
    if buy_input.pred_combo != prediction.pred_combo:
        raise ValidationError(
            f"buy_input pred_combo mismatch: {buy_input.pred_combo!r} != {prediction.pred_combo!r}"
        )
    candidate = build_input_candidate(prediction, buy_input)
    context = build_input_context(evaluation, bonus, buy_input)
    return _finish(evaluation, candidate, context, config, kelly)
