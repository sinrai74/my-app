"""
MetricsStore（storage/metrics_store.py）の単体テスト。

設計書 v1.1.8 ⑥・⑲、Step3計画書 §2・§6、Step3-5指示（責務境界）に対応。
tempfileによる実I/Oテスト。標準ライブラリ unittest のみを使用する。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from models.output import SystemMetrics
from storage.exceptions import ParseError
from storage.metrics_store import MetricsStore


def _metrics(
    metrics_id: str = "20260714_ranking_daily_run1",
    race_date: str = "20260714",
    status: str = "success",
) -> SystemMetrics:
    return SystemMetrics(
        metrics_id=metrics_id,
        race_date=race_date,
        job_name="ranking_daily",
        run_id="run1",
        started_at="2026-07-14T07:30:00+09:00",
        finished_at="2026-07-14T07:35:00+09:00",
        duration_seconds=300.0,
        status=status,
        counters={"races_evaluated": 90},
        rates={"api_fetch_success_rate": 1.0},
        errors={},
        schema_version=1,
    )


class TestMetricsStoreSnapshot(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.snapshot = root / "system_metrics.json"
        self.monthly = root / "metrics"
        self.store = MetricsStore(self.snapshot, self.monthly)

    def test_missing_snapshot_reads_empty(self) -> None:
        self.assertEqual(self.store.read_snapshot(), [])

    def test_write_and_read_snapshot(self) -> None:
        items = [_metrics(), _metrics(metrics_id="x2", status="failed")]
        self.store.write_snapshot(items)
        self.assertEqual(self.store.read_snapshot(), items)

    def test_write_snapshot_is_full_replacement(self) -> None:
        """当日全置換: 2回目のwriteで1回目の内容が残らないこと。"""
        self.store.write_snapshot([_metrics(metrics_id="first")])
        self.store.write_snapshot([_metrics(metrics_id="second")])
        loaded = self.store.read_snapshot()
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].metrics_id, "second")

    def test_broken_snapshot_raises_parse_error(self) -> None:
        self.snapshot.write_text("{not array", encoding="utf-8")
        with self.assertRaises(ParseError):
            self.store.read_snapshot()

    def test_non_array_snapshot_raises(self) -> None:
        self.snapshot.write_text(json.dumps({"x": 1}), encoding="utf-8")
        with self.assertRaises(ParseError):
            self.store.read_snapshot()


class TestMetricsStoreMonthly(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.store = MetricsStore(root / "system_metrics.json", root / "metrics")

    def test_append_and_read_monthly(self) -> None:
        self.store.append_monthly(_metrics(metrics_id="a", race_date="20260714"))
        self.store.append_monthly(_metrics(metrics_id="b", race_date="20260720"))
        got = self.store.read_monthly("202607")
        self.assertEqual([m.metrics_id for m in got], ["a", "b"])

    def test_month_derived_from_race_date(self) -> None:
        """月は race_date の先頭6桁から決まること。"""
        self.store.append_monthly(_metrics(race_date="20260814"))
        self.assertEqual(len(self.store.read_monthly("202608")), 1)
        self.assertEqual(self.store.read_monthly("202607"), [])

    def test_missing_monthly_reads_empty(self) -> None:
        self.assertEqual(self.store.read_monthly("209901"), [])

    def test_invalid_race_date_raises(self) -> None:
        with self.assertRaises(ParseError):
            self.store.append_monthly(_metrics(race_date="2026"))

    def test_append_preserves_existing(self) -> None:
        """追記であり、既存行を消さないこと。"""
        self.store.append_monthly(_metrics(metrics_id="a", race_date="20260714"))
        self.store.append_monthly(_metrics(metrics_id="b", race_date="20260714"))
        self.assertEqual(len(self.store.read_monthly("202607")), 2)

    def test_failed_status_metrics_can_be_stored(self) -> None:
        """失敗ジョブのメトリクスも保存できること（⑲: 成否問わず記録）。"""
        self.store.append_monthly(
            _metrics(metrics_id="f", race_date="20260714", status="failed")
        )
        got = self.store.read_monthly("202607")
        self.assertEqual(got[0].status, "failed")


if __name__ == "__main__":
    unittest.main()
