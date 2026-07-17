"""
出力・KPI系データモデル（models/output.py）の単体テスト。

設計書 Phase0.5 v1.1 ③3.12〜3.13・3.16、⑦（表示層責務）・⑲（システムKPI）に対応。
標準ライブラリ unittest のみを使用する。
"""

from __future__ import annotations

import unittest

from models.output import NewsArticle, RankingEntry, SystemMetrics


class TestRankingEntry(unittest.TestCase):
    def test_construct(self) -> None:
        """必須項目でRankingEntryが構築できること。"""
        entry = RankingEntry(
            ranking_type="danger",
            race_date="20260713",
            venue_name="住之江",
            race_number=5,
            score=78.5,
            rank_label="S",
            subject={"racer_no": "4999", "racer_name": "テスト選手"},
            reasons=("平均ST遅め",),
            ai_comment="1号艇に不安あり",
            source_eval_id="20260713_12_05",
        )
        self.assertEqual(entry.ranking_type, "danger")
        self.assertEqual(entry.source_eval_id, "20260713_12_05")

    def test_source_eval_id_is_str_not_race_evaluation(self) -> None:
        """source_eval_idはstr型の参照IDのみであり、RaceEvaluation型の
        フィールドを持たないこと（案P採用の確認）。

        設計書③3.12: RankingEntryはRaceEvaluationの集約モデルではなく、
        参照関係のみを表現する軽量モデルである。
        """
        entry = RankingEntry(
            ranking_type="hot_motor",
            race_date="20260713",
            venue_name="住之江",
            race_number=5,
            score=90.0,
            rank_label="A",
            subject={"motor_no": 12},
            reasons=(),
            ai_comment="",
            source_eval_id="20260713_12_05",
        )
        self.assertIsInstance(entry.source_eval_id, str)
        field_names = {f for f in entry.__dataclass_fields__}
        self.assertNotIn("source_evaluation", field_names)
        self.assertNotIn("evaluation", field_names)

    def test_immutable(self) -> None:
        entry = RankingEntry(
            ranking_type="awakening",
            race_date="20260713",
            venue_name="住之江",
            race_number=5,
            score=60.0,
            rank_label="B",
            subject={},
            reasons=(),
            ai_comment="",
            source_eval_id="20260713_12_05",
        )
        with self.assertRaises(Exception):
            entry.score = 0.0  # type: ignore[misc]

    def test_reasons_as_tuple(self) -> None:
        """reasonsが複数件のタプルとして保持できること。"""
        entry = RankingEntry(
            ranking_type="manshuu",
            race_date="20260713",
            venue_name="住之江",
            race_number=5,
            score=8.2,
            rank_label="S",
            subject={},
            reasons=("荒れ要因1", "荒れ要因2"),
            ai_comment="",
            source_eval_id="20260713_12_05",
        )
        self.assertEqual(len(entry.reasons), 2)


class TestNewsArticle(unittest.TestCase):
    def test_construct(self) -> None:
        """必須項目でNewsArticleが構築できること。"""
        article = NewsArticle(
            race_date="20260713",
            sections={"featured_races": [], "featured_racers": []},
            generated_at="2026-07-13T07:30:00+09:00",
            brand_version="1",
        )
        self.assertEqual(article.race_date, "20260713")
        self.assertIn("featured_races", article.sections)

    def test_immutable(self) -> None:
        article = NewsArticle(
            race_date="20260713",
            sections={},
            generated_at="2026-07-13T07:30:00+09:00",
            brand_version="1",
        )
        with self.assertRaises(Exception):
            article.brand_version = "2"  # type: ignore[misc]


class TestSystemMetrics(unittest.TestCase):
    def test_construct_success_status(self) -> None:
        """ジョブ成功時のSystemMetricsが構築できること。ai_summaryは任意項目。"""
        metrics = SystemMetrics(
            metrics_id="20260713_ranking_daily_run123",
            race_date="20260713",
            job_name="ranking_daily",
            run_id="run123",
            started_at="2026-07-13T07:30:00+09:00",
            finished_at="2026-07-13T07:35:00+09:00",
            duration_seconds=300.0,
            status="success",
            counters={"races_evaluated": 90},
            rates={"api_fetch_success_rate": 1.0},
            errors={},
            schema_version=1,
        )
        self.assertEqual(metrics.status, "success")
        self.assertIsNone(metrics.ai_summary)

    def test_construct_failed_status_with_errors(self) -> None:
        """ジョブ失敗時もSystemMetricsが構築できること
        （設計書⑲19.3: 成功・失敗を問わず出力する契約）。
        """
        metrics = SystemMetrics(
            metrics_id="20260713_arashi_watch_run456",
            race_date="20260713",
            job_name="arashi_watch",
            run_id="run456",
            started_at="2026-07-13T09:00:00+09:00",
            finished_at="2026-07-13T09:00:15+09:00",
            duration_seconds=15.0,
            status="failed",
            counters={},
            rates={},
            errors={"DataFetchError": 1},
            schema_version=1,
        )
        self.assertEqual(metrics.status, "failed")
        self.assertEqual(metrics.errors["DataFetchError"], 1)

    def test_ai_summary_holds_verification_transcription(self) -> None:
        """ai_summaryはVerificationResultからの転記値を保持できること
        （設計書⑲: 再計算はしない、転記のみ）。
        """
        metrics = SystemMetrics(
            metrics_id="20260713_verify_results_run789",
            race_date="20260712",
            job_name="verify_results",
            run_id="run789",
            started_at="2026-07-13T22:00:00+09:00",
            finished_at="2026-07-13T22:02:00+09:00",
            duration_seconds=120.0,
            status="success",
            counters={"records_written": 90},
            rates={"storage_success_rate": 1.0},
            errors={},
            schema_version=1,
            ai_summary={"roi": 1.28, "n_hit": 4},
        )
        self.assertEqual(metrics.ai_summary["roi"], 1.28)

    def test_immutable(self) -> None:
        metrics = SystemMetrics(
            metrics_id="20260713_ranking_daily_run123",
            race_date="20260713",
            job_name="ranking_daily",
            run_id="run123",
            started_at="2026-07-13T07:30:00+09:00",
            finished_at="2026-07-13T07:35:00+09:00",
            duration_seconds=300.0,
            status="success",
            counters={},
            rates={},
            errors={},
            schema_version=1,
        )
        with self.assertRaises(Exception):
            metrics.status = "failed"  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
