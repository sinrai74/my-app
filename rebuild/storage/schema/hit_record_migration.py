"""
HitRecordMigration: レガシー44列 hit_record.csv を新形式（44+拡張列）へ変換する。

設計書 v1.1.8 ⑯（移行方法）、Step3計画書 §2（migration最小実装）に基づく。

スコープ（Step3-6指示・厳守）:
- 対象は「レガシー44列 -> 新形式」の1方向のみ（汎用移行フレームワークは作らない）
- 順序は backup作成 -> 変換 -> 検証 を厳守する
- rollbackは実装しない（Fail Fast。検証失敗時はStorageErrorで停止し、
  backupファイルは残るため運用者が手動復旧できる）
- 変換ロジックは持たず、HitRecordCsvRepository（Step2）へ完全委譲する
  （read_all=レガシー読込、write_all=新形式書込。Mapperの往復契約を利用）

依存: 標準ライブラリ ＋ storage内部（Repository/例外）のみ。
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from storage.exceptions import StorageError
from storage.mappers.hit_record_mapper import ALL_COLUMNS, LEGACY_COLUMNS
from storage.repositories.hit_record_repository import HitRecordCsvRepository


class HitRecordMigration:
    """レガシー hit_record.csv を新形式へ変換する（1回限りの移行操作）。"""

    def __init__(self, csv_path: Path) -> None:
        self._path = Path(csv_path)

    def migrate(self, backup_suffix: Optional[str] = None) -> Path:
        """backup作成 -> 変換 -> 検証 の順で移行する。

        戻り値: 作成したバックアップファイルのパス。
        失敗時はStorageError（rollbackしない。backupは残す）。

        - 対象ファイルが既に新形式（ALL_COLUMNS）の場合は何もせず、移行不要として
          StorageErrorを送出する（誤って新形式を再変換しないための安全弁）
        - レガシー44列でない・新形式でもない場合もStorageError（破損検知）
        """
        if not self._path.exists():
            raise StorageError(f"migration source not found: {self._path}")

        header = self._read_header()
        if header == ALL_COLUMNS:
            raise StorageError(
                f"{self._path.name}: already in extended format (no migration needed)"
            )
        if header != LEGACY_COLUMNS:
            raise StorageError(
                f"{self._path.name}: header is neither legacy 44 nor extended "
                f"(got {len(header)} columns); aborting migration"
            )

        # 1. backup作成（変換前に必ず退避）
        backup_path = self._make_backup(backup_suffix)

        # 2. 変換（レガシー読込 -> 新形式書込。変換はRepository/Mapperへ委譲）
        repo = HitRecordCsvRepository(self._path)
        records = repo.read_all()
        repo.write_all(records)  # write_allは常にALL_COLUMNSで出力する

        # 3. 検証（新形式ヘッダー・行数一致・再読込モデルの同値）
        self._verify(backup_path, records)
        return backup_path

    def _read_header(self) -> tuple[str, ...]:
        import csv

        with open(self._path, encoding="utf-8", newline="") as f:
            return tuple(next(csv.reader(f), []))

    def _make_backup(self, backup_suffix: Optional[str]) -> Path:
        if backup_suffix is None:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_suffix = f".backup_{stamp}"
        backup_path = self._path.with_name(self._path.name + backup_suffix)
        if backup_path.exists():
            raise StorageError(f"backup already exists: {backup_path}")
        shutil.copy2(self._path, backup_path)
        return backup_path

    def _verify(self, backup_path: Path, expected_records: list) -> None:
        """変換後の検証。失敗はStorageError（Fail Fast・rollbackなし）。"""
        header = self._read_header()
        if header != ALL_COLUMNS:
            raise StorageError(
                f"migration verify failed: output header is not extended format "
                f"(backup kept at {backup_path})"
            )
        reloaded = HitRecordCsvRepository(self._path).read_all()
        if len(reloaded) != len(expected_records):
            raise StorageError(
                f"migration verify failed: row count changed "
                f"{len(expected_records)} -> {len(reloaded)} (backup kept at {backup_path})"
            )
        if reloaded != expected_records:
            raise StorageError(
                f"migration verify failed: records differ after conversion "
                f"(backup kept at {backup_path})"
            )
