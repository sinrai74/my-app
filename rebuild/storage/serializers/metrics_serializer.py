"""
SystemMetricsSerializer: system_metrics.json / metrics/{YYYYMM}.jsonl に対応。

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

from storage.serializers.common import require_key as _req
from models.output import SystemMetrics


class SystemMetricsSerializer:
    """SystemMetrics ⇔ JSON互換dict。system_metrics.json / metrics/{YYYYMM}.jsonl 対応。"""

    @staticmethod
    def to_dict(model: SystemMetrics) -> dict[str, Any]:
        return {
            "schema_version": model.schema_version,
            "metrics_id": model.metrics_id,
            "race_date": model.race_date,
            "job_name": model.job_name,
            "run_id": model.run_id,
            "started_at": model.started_at,
            "finished_at": model.finished_at,
            "duration_seconds": model.duration_seconds,
            "status": model.status,
            "counters": model.counters,
            "rates": model.rates,
            "errors": model.errors,
            "ai_summary": model.ai_summary,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> SystemMetrics:
        return SystemMetrics(
            metrics_id=_req(data, "metrics_id"),
            race_date=_req(data, "race_date"),
            job_name=_req(data, "job_name"),
            run_id=_req(data, "run_id"),
            started_at=_req(data, "started_at"),
            finished_at=_req(data, "finished_at"),
            duration_seconds=_req(data, "duration_seconds"),
            status=_req(data, "status"),
            counters=_req(data, "counters"),
            rates=_req(data, "rates"),
            errors=_req(data, "errors"),
            schema_version=_req(data, "schema_version"),
            ai_summary=data.get("ai_summary"),
        )
