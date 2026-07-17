"""
VerificationResultSerializer: verification_history に対応。

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

from storage.serializers.common import SCHEMA_VERSION, require_key as _req
from models.record import VerificationResult


class VerificationResultSerializer:
    @staticmethod
    def to_dict(model: VerificationResult) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "race_date": model.race_date,
            "n_races": model.n_races,
            "n_purchased": model.n_purchased,
            "n_hit": model.n_hit,
            "total_cost": model.total_cost,
            "total_payout": model.total_payout,
            "roi": model.roi,
            "by_race_type": model.by_race_type,
            "by_rank": model.by_rank,
            "engine_version": model.engine_version,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> VerificationResult:
        return VerificationResult(
            race_date=_req(data, "race_date"),
            n_races=_req(data, "n_races"),
            n_purchased=_req(data, "n_purchased"),
            n_hit=_req(data, "n_hit"),
            total_cost=_req(data, "total_cost"),
            total_payout=_req(data, "total_payout"),
            roi=_req(data, "roi"),
            by_race_type=_req(data, "by_race_type"),
            by_rank=_req(data, "by_rank"),
            engine_version=_req(data, "engine_version"),
        )
