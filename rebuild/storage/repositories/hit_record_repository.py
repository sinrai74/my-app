"""
HitRecordCsvRepository: hit_record.csv の読み書き。

設計書 v1.1.6 ⑥、Step2-6承認スコープに基づく。

責務:
- csvモジュールによるファイルI/Oとヘッダーバリデーションをここに集約する
- 行⇔モデル変換はHitRecordCsvMapperへ完全委譲（変換ロジックの重複禁止）
- ParseError（Mapper由来）は握りつぶさず呼び出し元へ伝播させる
- Releases退避・git commit不可分化・migration起動はStep3以降（範囲外）

読み書き経路の定義:
- 読み込み: レガシー形式（44列ヘッダー）と新形式（44+拡張列）の両方を受理する。
  ヘッダーがどちらとも一致しない場合はStorageError（列並びの破損・タイポ検知）
- 書き込み: 常に新形式（ALL_COLUMNS）で出力する
- 追記: 新形式ヘッダーのファイルにのみ許可。レガシー形式への追記は
  StorageError（先にwrite_allで新形式へ変換することを促す）
"""

from __future__ import annotations

import csv
from pathlib import Path

from models.record import HitRecord
from storage.exceptions import StorageError
from storage.mappers.hit_record_mapper import (
    ALL_COLUMNS,
    LEGACY_COLUMNS,
    HitRecordCsvMapper,
)


class HitRecordCsvRepository:
    """1つのhit_record CSVファイルを担当するRepository（パスはDIで注入）。"""

    def __init__(self, csv_path: Path) -> None:
        self._path = Path(csv_path)

    @property
    def path(self) -> Path:
        """担当ファイルのパス（読み取り専用）。DurableStoreの退避・commit対象特定用（v1.1.7）。"""
        return self._path

    def read_all(self) -> list[HitRecord]:
        """全レコードを読み込む。ファイル無しは空リスト。

        ヘッダーがレガシー44列にも新形式にも一致しない場合はStorageError。
        行の解釈失敗（列欠落・破損JSON等）はMapperのParseErrorをそのまま伝播する。
        """
        if not self._path.exists():
            return []
        with open(self._path, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            header = tuple(reader.fieldnames or ())
            if header == ():
                return []
            if header not in (LEGACY_COLUMNS, ALL_COLUMNS):
                raise StorageError(
                    f"{self._path.name}: unexpected header (neither legacy 44 nor "
                    f"extended format). got {len(header)} columns"
                )
            return [HitRecordCsvMapper.from_row(row) for row in reader]

    def write_all(self, records: list[HitRecord]) -> None:
        """全レコードを新形式（ALL_COLUMNS）で書き出す（全置換）。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writeheader()
            for record in records:
                writer.writerow(HitRecordCsvMapper.to_row(record))

    def append(self, record: HitRecord) -> None:
        """レコードを1件追記する。

        追記先が存在しない場合は新形式ヘッダー付きで新規作成する。
        既存ファイルのヘッダーが新形式（ALL_COLUMNS）でない場合はStorageError
        （レガシー形式への列数不一致の追記はCSVを破壊するため。
        先にread_all -> write_allで新形式へ変換すること）。
        """
        if not self._path.exists():
            self.write_all([record])
            return
        with open(self._path, encoding="utf-8", newline="") as f:
            header = tuple(next(csv.reader(f), []))
        if header != ALL_COLUMNS:
            raise StorageError(
                f"{self._path.name}: append requires extended header "
                f"(convert legacy file via read_all/write_all first)"
            )
        with open(self._path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=ALL_COLUMNS)
            writer.writerow(HitRecordCsvMapper.to_row(record))
