"""
MotorHistoryCsvMapper: MotorHistory ⇔ CSV行 の変換。

Step2実装計画書 §1.3（1:1対応・低難易度）に基づく。
現行motor_history.csvの11列ヘッダーと完全一致することを実データで確認済み:
    date,venue_num,venue,motor_no,racer_no,racer_name,lane,place,
    ex_time,start_timing,race_number

Mapperはステートレス。ファイルI/O・パス解決を行わない（Repository＝Step3の責務）。
"""

from __future__ import annotations

from models.record import MotorHistory
from storage.mappers.row_types import Row

# 現行motor_history.csvの実ヘッダーからそのまま転記した列順（変更禁止）。
LEGACY_COLUMNS: tuple[str, ...] = (
    "date",
    "venue_num",
    "venue",
    "motor_no",
    "racer_no",
    "racer_name",
    "lane",
    "place",
    "ex_time",
    "start_timing",
    "race_number",
)


class MotorHistoryCsvMapper:
    """MotorHistory ⇔ Row（dict[str, Any]）の純粋変換。

    不変条件: from_row(to_row(m)) == m （完全往復、float許容1e-6はテスト側で吸収）。
    全フィールドが必須（Optionalなし）のため、値なし（空欄）の扱いは考慮しない。
    """

    @staticmethod
    def to_row(model: MotorHistory) -> Row:
        """MotorHistoryをCSV行（dict）へ変換する。値の丸めは行わない。"""
        return {
            "date": model.date,
            "venue_num": model.venue_num,
            "venue": model.venue,
            "motor_no": model.motor_no,
            "racer_no": model.racer_no,
            "racer_name": model.racer_name,
            "lane": model.lane,
            "place": model.place,
            "ex_time": model.ex_time,
            "start_timing": model.start_timing,
            "race_number": model.race_number,
        }

    @staticmethod
    def from_row(row: Row) -> MotorHistory:
        """CSV行（dict）からMotorHistoryを復元する。

        全列が必須のため、型変換はint()/float()/str()を直接適用する
        （row_typesのOptional系パーサーは使わない。値なしを許容しない
        契約であるため、欠損はKeyError/ValueErrorとしてそのまま伝播させる）。
        """
        return MotorHistory(
            date=str(row["date"]),
            venue_num=int(row["venue_num"]),
            venue=str(row["venue"]),
            motor_no=int(row["motor_no"]),
            racer_no=str(row["racer_no"]),
            racer_name=str(row["racer_name"]),
            lane=int(row["lane"]),
            place=int(row["place"]),
            ex_time=float(row["ex_time"]),
            start_timing=float(row["start_timing"]),
            race_number=int(row["race_number"]),
        )
