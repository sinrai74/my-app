"""
DurableStore: 書込＝Releases退避＝git commit の不可分化。

設計書 v1.1.7 ⑥（hit_record行:「書込成功＝Releases退避＝git commitを
1トランザクション扱い、いずれか失敗で全体失敗としERROR通知」）、
Step3計画書 §4 に基づく。

hit_record欠落事故（2026-07-05〜07、docs/incidents参照）の構造的再発防止の本体。
「保存対象の書込」と「退避・commit」がコード上分離していたことが事故原因であり、
本モジュールは両者を1つの操作として提供する。

失敗時の方針（S5確定・Fail Fast）:
- ロールバックは行わない。いずれかの段階で失敗したらStorageErrorを送出して停止する
- Actionsコンテナはステートレスであり、失敗ジョブの局所状態はコンテナとともに
  消えるため、「静かに続行」より「大声で失敗」が正しい（設計書⑫サイレント失敗禁止）
- ERROR通知の発火はL5（Step4）。本モジュールは例外送出まで

依存はすべてコンストラクタ注入（Protocol）。ビジネスロジックは持たない。
"""

from __future__ import annotations

from models.evaluation import RaceEvaluation
from models.record import HitRecord
from storage.clients.protocols import GitClient, ReleaseClient
from storage.repositories.evaluation_repository import EvaluationRepository
from storage.repositories.hit_record_repository import HitRecordCsvRepository


class DurableHitRecordStore:
    """HitRecordの耐久書込。Repository（Step2）を包み、退避・commitまで保証する。"""

    def __init__(
        self,
        repository: HitRecordCsvRepository,
        release: ReleaseClient,
        git: GitClient,
    ) -> None:
        self._repository = repository
        self._release = release
        self._git = git

    def append_durably(self, record: HitRecord, commit_message: str) -> None:
        """1件追記し、Releases退避（最新版＋日次スナップショット）とcommitまで行う。

        実行順序（この順序は契約であり変更しない）:
        1. ローカル追記（失敗 -> 以降を実行しない）
        2. Releasesへ最新版 hit_record.csv をアップロード
        3. Releasesへ日次スナップショット hit_record_{race_date}.csv をアップロード
        4. git commit & push
        いずれかの失敗はStorageError等の例外として伝播する（ロールバックなし）。
        """
        self._repository.append(record)
        path = self._repository.path
        race_date = record.evaluation.race_date
        self._release.upload_asset(path, "hit_record.csv")
        self._release.upload_asset(path, f"hit_record_{race_date}.csv")
        self._git.commit_and_push([path], commit_message)


class DurableEvaluationStore:
    """RaceEvaluation（JSONL）の耐久書込。

    evaluations/{date}.jsonl はファイル自体が日付単位のため、
    アセット名 evaluations_{race_date}.jsonl（スナップショット=最新版）で退避する。
    """

    def __init__(
        self,
        repository: EvaluationRepository,
        release: ReleaseClient,
        git: GitClient,
    ) -> None:
        self._repository = repository
        self._release = release
        self._git = git

    def append_durably(self, evaluation: RaceEvaluation, commit_message: str) -> None:
        """1件追記し、Releases退避とcommitまで行う（順序契約はHitRecord側と同じ）。"""
        self._repository.append(evaluation)
        path = self._repository.path
        self._release.upload_asset(
            path, f"evaluations_{evaluation.race_date}.jsonl"
        )
        self._git.commit_and_push([path], commit_message)
