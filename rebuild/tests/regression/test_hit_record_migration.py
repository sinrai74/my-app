"""
HitRecordMigration の単体テスト＋Golden移行回帰。

設計書 v1.1.8 ⑯、Step3計画書 §6、Step3-6指示に対応。
tempfileによる実I/Oテスト。標準ライブラリ unittest のみを使用する。
"""

from __future__ import annotations

import csv
import shutil
import tempfile
import unittest
from pathlib import Path

from storage.exceptions import StorageError
from storage.mappers.hit_record_mapper import ALL_COLUMNS, LEGACY_COLUMNS
from storage.repositories.hit_record_repository import HitRecordCsvRepository
from storage.schema.hit_record_migration import HitRecordMigration

GOLDEN_CSV = Path("tests/regression/golden/hit_record_golden_100.csv")


class TestHitRecordMigration(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "hit_record.csv"
        # ゴールデン（レガシー44列）を作業ディレクトリへコピーして対象にする
        shutil.copy2(GOLDEN_CSV, self.path)

    def test_backup_then_convert_then_verify_success(self) -> None:
        """backup作成→変換→検証が成功し、backupパスが返ること。"""
        migration = HitRecordMigration(self.path)
        backup = migration.migrate(backup_suffix=".backup_test")
        # backupが作られ、中身はレガシー44列のまま
        self.assertTrue(backup.exists())
        with open(backup, encoding="utf-8", newline="") as f:
            self.assertEqual(tuple(next(csv.reader(f))), LEGACY_COLUMNS)
        # 本体は新形式になっている
        with open(self.path, encoding="utf-8", newline="") as f:
            self.assertEqual(tuple(next(csv.reader(f))), ALL_COLUMNS)

    def test_golden_migration_regression_preserves_models(self) -> None:
        """Golden回帰(Should-4): 移行前後でモデルが同値であること。"""
        before = HitRecordCsvRepository(self.path).read_all()
        self.assertEqual(len(before), 100)
        HitRecordMigration(self.path).migrate(backup_suffix=".bak")
        after = HitRecordCsvRepository(self.path).read_all()
        self.assertEqual(after, before)

    def test_backup_is_created_before_conversion(self) -> None:
        """backupの中身が『変換前（レガシー）』であること＝順序の保証。"""
        migration = HitRecordMigration(self.path)
        backup = migration.migrate(backup_suffix=".bak")
        # backupをレガシーとして読めば、元の100行が復元できる
        backup_records = HitRecordCsvRepository(backup).read_all()
        self.assertEqual(len(backup_records), 100)

    def test_already_extended_format_aborts(self) -> None:
        """既に新形式のファイルは再変換せずStorageError（安全弁）。"""
        # 一度移行して新形式にする
        HitRecordMigration(self.path).migrate(backup_suffix=".bak1")
        # もう一度migrateしようとするとStorageError
        with self.assertRaises(StorageError) as ctx:
            HitRecordMigration(self.path).migrate(backup_suffix=".bak2")
        self.assertIn("already in extended format", str(ctx.exception))

    def test_unknown_header_aborts(self) -> None:
        """44列でも新形式でもないヘッダーはStorageError（破損検知）。"""
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(["col_a", "col_b"])
        with self.assertRaises(StorageError):
            HitRecordMigration(self.path).migrate(backup_suffix=".bak")

    def test_missing_source_aborts(self) -> None:
        missing = Path(self._tmp.name) / "nope.csv"
        with self.assertRaises(StorageError):
            HitRecordMigration(missing).migrate(backup_suffix=".bak")

    def test_existing_backup_aborts(self) -> None:
        """backup先が既存ならStorageError（誤上書き防止）。"""
        backup = self.path.with_name(self.path.name + ".bak")
        backup.write_text("dummy", encoding="utf-8")
        with self.assertRaises(StorageError) as ctx:
            HitRecordMigration(self.path).migrate(backup_suffix=".bak")
        self.assertIn("backup already exists", str(ctx.exception))

    def test_backup_kept_on_no_rollback_policy(self) -> None:
        """Fail Fast(Should-3): 検証失敗時もbackupが残ること（rollbackしない）。

        write_all後に本体を破損させる検証失敗はモックしづらいため、
        ここでは『backupが変換後も物理的に残っている』ことを確認する
        （rollback=backup削除を行わない方針の担保）。
        """
        migration = HitRecordMigration(self.path)
        backup = migration.migrate(backup_suffix=".bak")
        self.assertTrue(backup.exists())


if __name__ == "__main__":
    unittest.main()
