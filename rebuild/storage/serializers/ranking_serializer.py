"""
RankingEntrySerializer: rankings/{date}.json のentries要素に対応。

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
from models.output import RankingEntry


class RankingEntrySerializer:
    """RankingEntry ⇔ JSON互換dict。rankings/{date}.json のentries要素に対応。

    案P維持: source_eval_idはstr参照IDのまま。RaceEvaluationの解決はRepositoryの責務。
    """

    @staticmethod
    def to_dict(model: RankingEntry) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "ranking_type": model.ranking_type,
            "race_date": model.race_date,
            "venue_name": model.venue_name,
            "race_number": model.race_number,
            "score": model.score,
            "rank_label": model.rank_label,
            "subject": model.subject,
            "reasons": list(model.reasons),
            "ai_comment": model.ai_comment,
            "source_eval_id": model.source_eval_id,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> RankingEntry:
        return RankingEntry(
            ranking_type=_req(data, "ranking_type"),
            race_date=_req(data, "race_date"),
            venue_name=_req(data, "venue_name"),
            race_number=_req(data, "race_number"),
            score=_req(data, "score"),
            rank_label=_req(data, "rank_label"),
            subject=_req(data, "subject"),
            reasons=tuple(_req(data, "reasons")),
            ai_comment=_req(data, "ai_comment"),
            source_eval_id=_req(data, "source_eval_id"),
        )
