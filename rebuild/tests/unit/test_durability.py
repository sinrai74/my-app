"""
DurableStore（storage/durability.py）の単体テスト。

設計書 v1.1.7 ⑥（不可分化）、Step3計画書 §4・§6（失敗系網羅）に対応。
Fake注入（tests/fakes.py）による順序・失敗系の検証。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from models.evaluation import BuyDecision, FeatureSet, Prediction, RaceEvaluation
from models.record import HitRecord
from storage.durability import DurableEvaluationStore, DurableHitRecordStore
from storage.exceptions import StorageError
from storage.repositories.evaluation_repository import EvaluationRepository
from storage.repositories.hit_record_repository import HitRecordCsvRepository
from tests.fakes import FakeGitClient, FakeReleaseClient


def _evaluation(eval_id: str = "20260714_12_05") -> RaceEvaluation:
    date, venue, race = eval_id.split("_")
    return RaceEvaluation(
        eval_id=eval_id,
        race_date=date,
        venue_num=int(venue),
        venue_name="住之江",
        race_number=int(race),
        is_night=False,
        engine_name="ver4",
        engine_version="4.2.0",
        feature_schema_version=1,
        model_version="v20260701",
        evaluated_at="2026-07-14T09:00:00+09:00",
        danger_score=78.5,
        danger_breakdown={"win_rate_low": 28},
        upset_score=42.0,
        upset_reasons=(),
        rank_index={"1": 0.80},
        featured_boats={},
        win_probs={1: 0.42},
        race_type="本命",
        match_index=44.1,
        features=FeatureSet(
            eval_id=eval_id,
            feature_schema_version=1,
            built_at="2026-07-14T06:00:00+09:00",
            boat_features={1: {"win_rate": 6.50}},
            race_features={"venue_factor": 1.02},
        ),
    )


def _hit_record(eval_id: str = "20260714_12_05") -> HitRecord:
    return HitRecord(
        eval_id=eval_id,
        evaluation=_evaluation(eval_id),
        prediction=Prediction(
            eval_id=eval_id,
            pred_combo="1-2-3",
            pred_prob=0.185,
            pred_ev=1.42,
            pred_odds=7.7,
            confidence=0.72,
            why_bet="test",
            patterns=(),
        ),
        buy_decision=BuyDecision(
            eval_id=eval_id,
            purchased=True,
            buyscore=68.5,
            investment_type="通常",
            n_bets=3,
            cost=900,
            kelly_fraction=0.05,
            config_version="12",
        ),
    )


class TestDurableHitRecordStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "hit_record.csv"
        self.repo = HitRecordCsvRepository(self.path)

    def test_success_path_uploads_latest_and_snapshot_then_commits(self) -> None:
        """成功経路: ローカル追記→最新版→日次スナップショット→commitの順序契約。"""
        release = FakeReleaseClient()
        git = FakeGitClient()
        store = DurableHitRecordStore(self.repo, release, git)
        store.append_durably(_hit_record(), commit_message="feat: 実績追記")
        # ローカルに書けている
        self.assertEqual(len(self.repo.read_all()), 1)
        # 命名規約（S3確定）: 最新版 + 日次スナップショット
        self.assertEqual(
            release.calls,
            [("upload", "hit_record.csv"), ("upload", "hit_record_20260714.csv")],
        )
        self.assertIn("hit_record_20260714.csv", release.assets)
        # commitは最後に1回
        self.assertEqual(len(git.commits), 1)
        self.assertEqual(git.commits[0][1], "feat: 実績追記")

    def test_local_write_failure_prevents_upload_and_commit(self) -> None:
        """失敗系(a): ローカル書込失敗なら退避・commitは呼ばれない。"""
        # レガシー44列ヘッダーのファイルを用意し、appendがStorageErrorになる状況を作る
        import csv

        from storage.mappers.hit_record_mapper import LEGACY_COLUMNS

        with open(self.path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(LEGACY_COLUMNS)
        release = FakeReleaseClient()
        git = FakeGitClient()
        store = DurableHitRecordStore(self.repo, release, git)
        with self.assertRaises(StorageError):
            store.append_durably(_hit_record(), commit_message="x")
        self.assertEqual(release.calls, [])
        self.assertEqual(git.call_attempts, 0)  # Should-A: 未試行の明示検証
        self.assertEqual(git.commits, [])

    def test_upload_failure_prevents_commit(self) -> None:
        """失敗系(b): 退避失敗ならcommitは呼ばれずStorageError。ロールバックはしない。"""
        release = FakeReleaseClient(fail_on_upload=True)
        git = FakeGitClient()
        store = DurableHitRecordStore(self.repo, release, git)
        with self.assertRaises(StorageError):
            store.append_durably(_hit_record(), commit_message="x")
        # Should-A: commit/pushが「一切試行されていない」ことを明示的に検証
        self.assertEqual(git.call_attempts, 0)
        self.assertEqual(git.commits, [])
        # Fail Fast（S5）: ローカル追記はロールバックされず残る
        self.assertEqual(len(self.repo.read_all()), 1)

    def test_commit_failure_raises(self) -> None:
        """失敗系(c): commit失敗はStorageErrorとして伝播する。"""
        release = FakeReleaseClient()
        git = FakeGitClient(fail_on_commit=True)
        store = DurableHitRecordStore(self.repo, release, git)
        with self.assertRaises(StorageError):
            store.append_durably(_hit_record(), commit_message="x")
        # 退避までは完了している（後続の復旧はReleasesから可能）
        self.assertIn("hit_record.csv", release.assets)


class TestDurableEvaluationStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "evaluations" / "20260714.jsonl"
        self.repo = EvaluationRepository(self.path)

    def test_success_path(self) -> None:
        """成功経路: 追記→日付付きアセット退避→commit。"""
        release = FakeReleaseClient()
        git = FakeGitClient()
        store = DurableEvaluationStore(self.repo, release, git)
        store.append_durably(_evaluation(), commit_message="feat: 評価追記")
        self.assertEqual(len(self.repo.load_all()), 1)
        self.assertEqual(release.calls, [("upload", "evaluations_20260714.jsonl")])
        self.assertEqual(len(git.commits), 1)

    def test_duplicate_eval_id_prevents_upload(self) -> None:
        """一意性違反（Step2契約）でも退避・commitが走らないこと。"""
        release = FakeReleaseClient()
        git = FakeGitClient()
        store = DurableEvaluationStore(self.repo, release, git)
        store.append_durably(_evaluation(), commit_message="1回目")
        release.calls.clear()
        git.commits.clear()
        with self.assertRaises(StorageError):
            store.append_durably(_evaluation(), commit_message="2回目")
        self.assertEqual(release.calls, [])
        self.assertEqual(git.commits, [])


if __name__ == "__main__":
    unittest.main()
