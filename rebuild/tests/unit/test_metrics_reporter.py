"""S6 SystemMetrics のテスト（実Release・実git・実送信なし）。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from actions.metrics_reporter import (
    JOB_NAME,
    RELEASE_TAG,
    MetricsReporter,
    build_errors,
    metrics_id_for,
    monthly_asset_name,
)
from models.output import SystemMetrics
from storage.metrics_store import MetricsStore

JST = timezone(timedelta(hours=9))
DATE = "20260922"
RUN_ID = "123456"


class _FakeRelease:
    def __init__(self, assets=(), fail=False):
        self._assets = list(assets)
        self.uploaded = []
        self.fail = fail

    def list_assets(self):
        return sorted(self._assets)

    def upload_asset(self, file_path, asset_name):
        if self.fail:
            raise RuntimeError("upload failed")
        self.uploaded.append((asset_name, str(file_path)))
        self._assets.append(asset_name)


def _reporter(root: Path, release=None):
    store = MetricsStore(root / "system_metrics.json", root / "metrics")
    return MetricsReporter(
        store, monthly_dir=str(root / "metrics"), release=release
    )


def _metrics(reporter, *, race_date=DATE, run_id=RUN_ID, status="success",
             counters=None, errors=None):
    start = datetime(2026, 9, 22, 9, 0, 0, tzinfo=JST)
    end = start + timedelta(seconds=12.5)
    return reporter.build(
        race_date=race_date, run_id=run_id,
        started_at=start.isoformat(), finished_at=end.isoformat(),
        duration_seconds=(end - start).total_seconds(),
        status=status, counters=counters or {}, errors=errors or {},
    )


class TestMetricsIdAndJobName(unittest.TestCase):
    def test_metrics_id_format(self):
        self.assertEqual(
            metrics_id_for(DATE, RUN_ID), f"{DATE}_production_evaluation_{RUN_ID}"
        )

    def test_job_name_is_production_evaluation(self):
        root = Path(tempfile.mkdtemp())
        metrics = _metrics(_reporter(root))
        self.assertEqual(metrics.job_name, "production_evaluation")
        self.assertEqual(JOB_NAME, "production_evaluation")

    def test_race_date_is_used_not_execution_date(self):
        root = Path(tempfile.mkdtemp())
        metrics = _metrics(_reporter(root), race_date="20260701")
        self.assertTrue(metrics.metrics_id.startswith("20260701_"))
        self.assertEqual(metrics.race_date, "20260701")


class TestCounters(unittest.TestCase):
    def test_measured_counters_are_recorded(self):
        root = Path(tempfile.mkdtemp())
        counters = {
            "races_found": 3, "races_evaluated": 2,
            "races_skipped": 1, "records_written": 2,
        }
        metrics = _metrics(_reporter(root), counters=counters)
        self.assertEqual(metrics.counters, counters)

    def test_unmeasured_counters_are_absent_not_zero(self):
        root = Path(tempfile.mkdtemp())
        metrics = _metrics(_reporter(root), counters={"races_found": 3})
        for key in (
            "purchase_candidates", "purchases_decided", "rankings_generated",
            "deliveries_attempted", "deliveries_sent", "news_generated",
        ):
            self.assertNotIn(key, metrics.counters)

    def test_rates_are_empty_not_zero(self):
        root = Path(tempfile.mkdtemp())
        self.assertEqual(_metrics(_reporter(root)).rates, {})


class TestErrors(unittest.TestCase):
    def test_known_types_only(self):
        errors = build_errors([
            {"race": "a", "error": "StorageError: boom"},
            {"race": "b", "error": "ValueError: other"},
        ])
        self.assertEqual(errors["total"], 2)
        self.assertEqual(errors["StorageError"], 1)
        self.assertNotIn("ValueError", errors)
        self.assertNotIn("DataFetchError", errors)
        self.assertNotIn("DeliveryError", errors)
        self.assertNotIn("ModelError", errors)
        self.assertIn("other", errors["last_message"])

    def test_no_errors_is_empty_dict(self):
        self.assertEqual(build_errors([]), {})

    def test_exception_is_summarised(self):
        errors = build_errors(exception=ValueError("bad config"))
        self.assertEqual(errors["total"], 1)
        self.assertIn("bad config", errors["last_message"])


class TestSnapshotAndMonthly(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.reporter = _reporter(self.root)

    def _snapshot(self):
        return json.loads((self.root / "system_metrics.json").read_text("utf-8"))

    def _monthly_lines(self):
        path = self.root / "metrics" / "202609.jsonl"
        return [l for l in path.read_text("utf-8").splitlines() if l.strip()]

    def test_snapshot_written_and_monthly_appended(self):
        self.reporter.report(_metrics(self.reporter, counters={"races_found": 1}))
        self.assertEqual(len(self._snapshot()), 1)
        self.assertEqual(len(self._monthly_lines()), 1)

    def test_snapshot_replaces_other_days(self):
        self.reporter.report(_metrics(self.reporter, race_date="20260921"))
        self.reporter.report(_metrics(self.reporter, race_date=DATE))
        snapshot = self._snapshot()
        self.assertEqual([m["race_date"] for m in snapshot], [DATE])

    def test_same_day_other_run_is_kept(self):
        self.reporter.report(_metrics(self.reporter, run_id="1"))
        self.reporter.report(_metrics(self.reporter, run_id="2"))
        self.assertEqual(len(self._snapshot()), 2)

    def test_duplicate_metrics_id_is_not_appended(self):
        self.reporter.report(_metrics(self.reporter))
        with self.assertLogs("actions.metrics_reporter", level="WARNING") as cm:
            self.reporter.report(_metrics(self.reporter))
        self.assertEqual(len(self._monthly_lines()), 1)
        self.assertTrue(any("already exists" in l for l in cm.output))

    def test_duplicate_with_different_content_is_not_appended(self):
        self.reporter.report(_metrics(self.reporter, counters={"races_found": 1}))
        before = self._monthly_lines()
        self.reporter.report(
            _metrics(self.reporter, status="failed", counters={"races_found": 9})
        )
        self.assertEqual(self._monthly_lines(), before)  # 上書きもしない


class TestMonthlyRelease(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        (self.root / "metrics").mkdir(parents=True)

    def _write_month(self, yyyymm):
        (self.root / "metrics" / f"{yyyymm}.jsonl").write_text("{}\n", "utf-8")

    def test_current_month_is_excluded(self):
        self._write_month("202609")
        release = _FakeRelease()
        reporter = _reporter(self.root, release)
        self.assertEqual(reporter.unreleased_months(DATE), [])
        self.assertEqual(reporter.sync_monthly_releases(DATE), [])
        self.assertEqual(release.uploaded, [])

    def test_all_unreleased_months_oldest_first(self):
        for m in ("202606", "202607", "202608", "202609"):
            self._write_month(m)
        release = _FakeRelease(assets=[monthly_asset_name("202607")])
        reporter = _reporter(self.root, release)
        self.assertEqual(reporter.sync_monthly_releases(DATE), ["202606", "202608"])
        self.assertEqual(
            [name for name, _ in release.uploaded],
            ["metrics_202606.jsonl", "metrics_202608.jsonl"],
        )

    def test_existing_asset_is_skipped(self):
        self._write_month("202608")
        release = _FakeRelease(assets=[monthly_asset_name("202608")])
        reporter = _reporter(self.root, release)
        self.assertEqual(reporter.sync_monthly_releases(DATE), [])
        self.assertEqual(release.uploaded, [])

    def test_upload_failure_is_warning_and_keeps_local_file(self):
        self._write_month("202608")
        release = _FakeRelease(fail=True)
        reporter = _reporter(self.root, release)
        with self.assertLogs("actions.metrics_reporter", level="WARNING") as cm:
            uploaded = reporter.sync_monthly_releases(DATE)
        self.assertEqual(uploaded, [])
        self.assertTrue((self.root / "metrics" / "202608.jsonl").exists())
        self.assertTrue(any("upload failed" in l for l in cm.output))

    def test_retry_on_next_run(self):
        self._write_month("202608")
        failing = _FakeRelease(fail=True)
        _reporter(self.root, failing).sync_monthly_releases(DATE)
        ok = _FakeRelease()
        self.assertEqual(
            _reporter(self.root, ok).sync_monthly_releases(DATE), ["202608"]
        )

    def test_no_release_client_does_not_fail(self):
        self._write_month("202608")
        self.assertEqual(_reporter(self.root).sync_monthly_releases(DATE), [])

    def test_release_tag_is_data_store_v2(self):
        self.assertEqual(RELEASE_TAG, "data-store-v2")


class TestReportDoesNotRaise(unittest.TestCase):
    def test_store_failure_is_warning_only(self):
        root = Path(tempfile.mkdtemp())
        broken = SimpleNamespace(
            read_snapshot=lambda: (_ for _ in ()).throw(RuntimeError("x")),
            write_snapshot=lambda ms: (_ for _ in ()).throw(RuntimeError("y")),
            read_monthly=lambda m: (_ for _ in ()).throw(RuntimeError("z")),
            append_monthly=lambda m: None,
        )
        reporter = MetricsReporter(broken, monthly_dir=str(root / "metrics"))
        metrics = SystemMetrics(
            metrics_id=metrics_id_for(DATE, RUN_ID), race_date=DATE,
            job_name=JOB_NAME, run_id=RUN_ID, started_at="t", finished_at="t",
            duration_seconds=0.0, status="success", counters={}, rates={},
            errors={}, schema_version=1,
        )
        with self.assertLogs("actions.metrics_reporter", level="WARNING"):
            reporter.report(metrics)  # 例外を上げない


class TestEntryMetricsIntegration(unittest.TestCase):
    """入口の failure path / Actions外 / timings を確認する。"""

    def setUp(self):
        from pipelines import production_evaluation_entry as entry

        self.entry = entry
        self.calls = []
        self._saved = {
            k: os.environ.get(k) for k in ("TARGET_RACES", "GITHUB_RUN_ID")
        }
        os.environ["TARGET_RACES"] = f"{DATE}_1_1"
        os.environ["GITHUB_RUN_ID"] = RUN_ID
        self._orig_emit = entry.emit_metrics
        self._orig_run = entry.run_entry
        entry.emit_metrics = lambda **kw: self.calls.append(kw)

    def tearDown(self):
        self.entry.emit_metrics = self._orig_emit
        self.entry.run_entry = self._orig_run
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _report(self, status="success"):
        return {
            "status": status, "total": 1, "success_count": 1, "failure_count": 0,
            "incomparable_count": 0, "s4_excluded_count": 0,
            "races_evaluated": 1, "records_written": 1, "success_rate": 1.0,
            "results": [{"evaluation_reused": False}], "errors": [],
            "incomparable": [], "s4_excluded": [],
        }

    def test_success_path_emits_metrics(self):
        self.entry.run_entry = lambda races: self._report()
        self.assertEqual(self.entry.main([]), 0)
        kw = self.calls[0]
        self.assertEqual(kw["status"], "success")
        self.assertEqual(kw["race_date"], DATE)
        self.assertEqual(kw["run_id"], RUN_ID)
        self.assertEqual(
            kw["counters"],
            {"races_found": 1, "races_evaluated": 1,
             "races_skipped": 0, "records_written": 1},
        )
        self.assertGreaterEqual(kw["finished_at"], kw["started_at"])

    def test_failed_status_is_passed_through(self):
        self.entry.run_entry = lambda races: self._report("failed")
        self.entry.main([])
        self.assertEqual(self.calls[0]["status"], "failed")

    def test_partial_status_is_passed_through(self):
        self.entry.run_entry = lambda races: self._report("partial")
        self.entry.main([])
        self.assertEqual(self.calls[0]["status"], "partial")

    def test_config_error_still_emits_failed_metrics_and_exit_1(self):
        from actions.config_loader import ConfigError

        def boom(races):
            raise ConfigError("config missing")

        self.entry.run_entry = boom
        self.assertEqual(self.entry.main([]), 1)  # 既存exit code規約は維持
        kw = self.calls[0]
        self.assertEqual(kw["status"], "failed")
        self.assertEqual(kw["counters"], {"races_found": 1})  # 0で埋めない
        self.assertIn("config missing", kw["errors"]["last_message"])

    def test_unexpected_exception_emits_metrics_and_propagates(self):
        def boom(races):
            raise RuntimeError("bundle build failed")

        self.entry.run_entry = boom
        with self.assertRaises(RuntimeError):
            self.entry.main([])
        self.assertEqual(self.calls[0]["status"], "failed")
        self.assertNotIn("races_evaluated", self.calls[0]["counters"])

    def test_no_metrics_outside_actions(self):
        os.environ.pop("GITHUB_RUN_ID", None)
        self.entry.run_entry = lambda races: self._report()
        self.assertEqual(self.entry.main([]), 0)
        self.assertEqual(self.calls, [])

    def test_target_races_error_emits_no_metrics(self):
        os.environ.pop("TARGET_RACES", None)
        self.assertEqual(self.entry.main([]), 1)
        self.assertEqual(self.calls, [])


if __name__ == "__main__":
    unittest.main()
