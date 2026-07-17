"""
LocalCourseStatsCsvMapper: LocalCourseStats ⇔ CSV行 の変換。

Step2実装計画書 §1.3（1:1対応・低難易度）に基づく。
現行local_course_stats.csvの15列ヘッダーと完全一致することを実データで確認済み:
    racer_no,venue_code,venue_name,course,starts,first,second,third,fourth,
    fifth,sixth,first_rate,top2_rate,top3_rate,last_updated

Mapperはステートレス。ファイルI/O・パス解決を行わない（Repository＝Step3の責務）。
"""

from __future__ import annotations

from models.record import LocalCourseStats
from storage.mappers.row_types import Row

# 現行local_course_stats.csvの実ヘッダーからそのまま転記した列順（変更禁止）。
LEGACY_COLUMNS: tuple[str, ...] = (
    "racer_no",
    "venue_code",
    "venue_name",
    "course",
    "starts",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "first_rate",
    "top2_rate",
    "top3_rate",
    "last_updated",
)


class LocalCourseStatsCsvMapper:
    """LocalCourseStats ⇔ Row（dict[str, Any]）の純粋変換。

    不変条件: from_row(to_row(m)) == m （完全往復、float許容1e-6はテスト側で吸収）。
    全フィールドが必須（Optionalなし）のため、値なし（空欄）の扱いは考慮しない。
    """

    @staticmethod
    def to_row(model: LocalCourseStats) -> Row:
        """LocalCourseStatsをCSV行（dict）へ変換する。値の丸めは行わない。"""
        return {
            "racer_no": model.racer_no,
            "venue_code": model.venue_code,
            "venue_name": model.venue_name,
            "course": model.course,
            "starts": model.starts,
            "first": model.first,
            "second": model.second,
            "third": model.third,
            "fourth": model.fourth,
            "fifth": model.fifth,
            "sixth": model.sixth,
            "first_rate": model.first_rate,
            "top2_rate": model.top2_rate,
            "top3_rate": model.top3_rate,
            "last_updated": model.last_updated,
        }

    @staticmethod
    def from_row(row: Row) -> LocalCourseStats:
        """CSV行（dict）からLocalCourseStatsを復元する。

        全列が必須のため、型変換はint()/float()/str()を直接適用する
        （row_typesのOptional系パーサーは使わない。値なしを許容しない
        契約であるため、欠損はKeyError/ValueErrorとしてそのまま伝播させる）。
        """
        return LocalCourseStats(
            racer_no=str(row["racer_no"]),
            venue_code=int(row["venue_code"]),
            venue_name=str(row["venue_name"]),
            course=int(row["course"]),
            starts=int(row["starts"]),
            first=int(row["first"]),
            second=int(row["second"]),
            third=int(row["third"]),
            fourth=int(row["fourth"]),
            fifth=int(row["fifth"]),
            sixth=int(row["sixth"]),
            first_rate=float(row["first_rate"]),
            top2_rate=float(row["top2_rate"]),
            top3_rate=float(row["top3_rate"]),
            last_updated=str(row["last_updated"]),
        )
