"""
Mapper層: ドメインモデル ⇔ 行（dict[str, Any]）の純粋変換。

Step2実装計画書 §1に基づく。全Mapperは以下の不変条件を満たす:
    model --to_row()--> Row --from_row()--> model（完全往復、float許容1e-6）

Mapperはステートレスであり、ファイルI/O・パス解決・環境変数参照を行わない
（それらはRepository＝Step3以降の責務）。

Step2-1: row_types（Row型・正規化共通部品）
Step2-2: MotorHistoryCsvMapper, LocalCourseStatsCsvMapper（1:1 Mapper）
Step2-3: HitRecordCsvMapper（唯一の集約Mapper・44互換列＋拡張列）
"""

from storage.mappers.hit_record_mapper import HitRecordCsvMapper
from storage.mappers.local_course_stats_mapper import LocalCourseStatsCsvMapper
from storage.mappers.motor_history_mapper import MotorHistoryCsvMapper
from storage.mappers.row_types import (
    Row,
    format_bool01,
    format_optional,
    parse_bool01,
    parse_optional_float,
    parse_optional_int,
    parse_optional_str,
)

__all__ = [
    "HitRecordCsvMapper",
    "LocalCourseStatsCsvMapper",
    "MotorHistoryCsvMapper",
    "Row",
    "format_bool01",
    "format_optional",
    "parse_bool01",
    "parse_optional_float",
    "parse_optional_int",
    "parse_optional_str",
]
