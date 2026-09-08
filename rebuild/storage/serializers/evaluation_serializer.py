"""
Evaluation系Serializer: RaceEvaluation / Prediction / BuyDecision / RaceResult。
evaluations/{date}.jsonl の1行およびHitRecordのJSON表現に対応。

設計書 v1.1.6 ④（出力スキーマ固定）、Step2実装計画書 §2 に基づく。

共通規約:
- to_dict() の出力はトップレベルに schema_version（int）を必ず含める（④）
- from_dict() は未知キーを無視する（前方互換。④「読み手は未知キーを無視する」）
  ※ Mapperの未知列ParseErrorとは意図的に非対称: CSVは破損検知を、
    JSONは前方互換を目的とするため
- 必須キーの欠落は ParseError
- tuple⇔list の相互変換はSerializerが吸収する（モデル側tuple、JSON側list）
- json.dumps/loads の呼び出し・ファイルI/OはRepository（Step3以降）の責務
"""

from __future__ import annotations

from typing import Any

from storage.exceptions import ParseError
from storage.serializers.common import SCHEMA_VERSION, require_key as _req
from models.evaluation import BuyDecision, FeatureSet, Prediction, RaceEvaluation
from models.record import RaceResult


class RaceEvaluationSerializer:
    """RaceEvaluation ⇔ JSON互換dict。evaluations/{date}.jsonl の1行に対応。"""

    @staticmethod
    def to_dict(model: RaceEvaluation) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "eval_id": model.eval_id,
            "race_date": model.race_date,
            "venue_num": model.venue_num,
            "venue_name": model.venue_name,
            "race_number": model.race_number,
            "is_night": model.is_night,
            "engine_name": model.engine_name,
            "engine_version": model.engine_version,
            "feature_schema_version": model.feature_schema_version,
            "model_version": model.model_version,
            "evaluated_at": model.evaluated_at,
            "danger_score": model.danger_score,
            "danger_breakdown": model.danger_breakdown,
            "upset_score": model.upset_score,
            "upset_reasons": list(model.upset_reasons),
            "rank_index": model.rank_index,
            "featured_boats": model.featured_boats,
            "win_probs": (
                {str(k): v for k, v in model.win_probs.items()}
                if model.win_probs is not None
                else None
            ),
            "race_type": model.race_type,
            "match_index": model.match_index,
            "features": model.features.to_dict(),
            "hot_motor_score": model.hot_motor_score,
            "awakening_score": model.awakening_score,
            "local_advantage": model.local_advantage,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> RaceEvaluation:
        win_probs_raw = _req(data, "win_probs")
        try:
            features = FeatureSet.from_dict(_req(data, "features"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ParseError(f"broken 'features': {exc}") from exc
        return RaceEvaluation(
            eval_id=_req(data, "eval_id"),
            race_date=_req(data, "race_date"),
            venue_num=_req(data, "venue_num"),
            venue_name=_req(data, "venue_name"),
            race_number=_req(data, "race_number"),
            is_night=_req(data, "is_night"),
            engine_name=_req(data, "engine_name"),
            engine_version=_req(data, "engine_version"),
            feature_schema_version=_req(data, "feature_schema_version"),
            model_version=_req(data, "model_version"),
            evaluated_at=_req(data, "evaluated_at"),
            danger_score=_req(data, "danger_score"),
            danger_breakdown=_req(data, "danger_breakdown"),
            upset_score=_req(data, "upset_score"),
            upset_reasons=tuple(_req(data, "upset_reasons")),
            rank_index=_req(data, "rank_index"),
            featured_boats=_req(data, "featured_boats"),
            win_probs=(
                {int(k): float(v) for k, v in win_probs_raw.items()}
                if win_probs_raw is not None
                else None
            ),
            race_type=_req(data, "race_type"),
            match_index=_req(data, "match_index"),
            features=features,
            hot_motor_score=data.get("hot_motor_score"),
            awakening_score=data.get("awakening_score"),
            local_advantage=data.get("local_advantage"),
        )


class PredictionSerializer:
    @staticmethod
    def to_dict(model: Prediction) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "eval_id": model.eval_id,
            "pred_combo": model.pred_combo,
            "pred_prob": model.pred_prob,
            "pred_ev": model.pred_ev,
            "pred_odds": model.pred_odds,
            "confidence": model.confidence,
            "why_bet": model.why_bet,
            "patterns": list(model.patterns),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Prediction:
        return Prediction(
            eval_id=_req(data, "eval_id"),
            pred_combo=_req(data, "pred_combo"),
            pred_prob=_req(data, "pred_prob"),
            pred_ev=_req(data, "pred_ev"),
            pred_odds=_req(data, "pred_odds"),
            confidence=_req(data, "confidence"),
            why_bet=_req(data, "why_bet"),
            patterns=tuple(_req(data, "patterns")),
        )


class BuyDecisionSerializer:
    @staticmethod
    def to_dict(model: BuyDecision) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "eval_id": model.eval_id,
            "purchased": model.purchased,
            "buyscore": model.buyscore,
            "investment_type": model.investment_type,
            "n_bets": model.n_bets,
            "cost": model.cost,
            "kelly_fraction": model.kelly_fraction,
            "config_version": model.config_version,
            "skip_reason": model.skip_reason,
            "purchased_combos": list(model.purchased_combos),
            "purchased_amounts": list(model.purchased_amounts),
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> BuyDecision:
        return BuyDecision(
            eval_id=_req(data, "eval_id"),
            purchased=_req(data, "purchased"),
            buyscore=_req(data, "buyscore"),
            investment_type=_req(data, "investment_type"),
            n_bets=_req(data, "n_bets"),
            cost=_req(data, "cost"),
            kelly_fraction=_req(data, "kelly_fraction"),
            config_version=_req(data, "config_version"),
            skip_reason=data.get("skip_reason"),
            # 旧形式（本フィールド追加前に書かれたJSON）読み込み時は空タプル。
            purchased_combos=tuple(data.get("purchased_combos", ())),
            purchased_amounts=tuple(data.get("purchased_amounts", ())),
        )


class RaceResultSerializer:
    @staticmethod
    def to_dict(model: RaceResult) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "eval_id": model.eval_id,
            "result_combo": model.result_combo,
            "payout": model.payout,
            "hit": model.hit,
            "profit": model.profit,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> RaceResult:
        return RaceResult(
            eval_id=_req(data, "eval_id"),
            result_combo=_req(data, "result_combo"),
            payout=_req(data, "payout"),
            hit=_req(data, "hit"),
            profit=_req(data, "profit"),
        )
