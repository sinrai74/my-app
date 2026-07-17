"""
Repository層: ファイルI/O・JSON文字列化・永続化状態の管理。

設計書 v1.1.6 ⑥、Step2-6承認スコープに基づく。
Mapper/Serializer（純粋変換）を利用し、変換ロジックを持たない。
Releases退避・git commit不可分化・idempotency・migrationはStep3以降。
"""

from storage.repositories.evaluation_repository import EvaluationRepository
from storage.repositories.hit_record_repository import HitRecordCsvRepository

__all__ = [
    "EvaluationRepository",
    "HitRecordCsvRepository",
]
