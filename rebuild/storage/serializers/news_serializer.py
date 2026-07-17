"""
NewsArticleSerializer: AI新聞のsections JSONに対応。

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
from models.output import NewsArticle


class NewsArticleSerializer:
    @staticmethod
    def to_dict(model: NewsArticle) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "race_date": model.race_date,
            "sections": model.sections,
            "generated_at": model.generated_at,
            "brand_version": model.brand_version,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> NewsArticle:
        return NewsArticle(
            race_date=_req(data, "race_date"),
            sections=_req(data, "sections"),
            generated_at=_req(data, "generated_at"),
            brand_version=_req(data, "brand_version"),
        )
