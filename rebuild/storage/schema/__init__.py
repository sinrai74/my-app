"""
スキーマ管理層（設計書 v1.1.8 ⑩ storage/schema/）。

Step3-6: hit_record_migration.py（レガシー44列→新形式の移行、最小実装）
汎用migrationフレームワーク・schema_version検出の統合は必要になるStepで追加する。
"""

from storage.schema.hit_record_migration import HitRecordMigration

__all__ = ["HitRecordMigration"]
