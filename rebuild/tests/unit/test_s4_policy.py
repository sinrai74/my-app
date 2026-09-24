"""S4 本番運用ポリシー（締切・日次通知上限）と、その設定・カウンタのテスト。

確定仕様:
  - S4.1 締切前10分。now <= close_time - 10min が評価対象。評価前に適用。
    除外レースは評価せず s4_excluded に記録する。
  - S4.3 1日10件。生成された通知リクエスト1件を1件として数える。
    race_date のJST暦日単位。上限到達後も ジョブは継続する。
  - 有効対象 = total - s4_excluded_count - incomparable_count。
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

from actions.config_loader import (
    ConfigError,
    load_daily_notification_limit,
    load_deadline_minutes,
)
from actions.notification_counter import DailyNotificationCounter
from actions.production_entrypoint import (
    CLOSE_TIME_FORMAT,
    is_before_deadline,
    run_production_day,
)
from models.evaluation import BuyDecision, FeatureSet, Prediction, RaceEvaluation
from notification.notifiers import NotificationResult
from output.renderers import RenderResult

JST = timezone(timedelta(hours=9))
DATE = "20260922"
CLOSE = "2026-09-22 15:24:00"  # 実機確認した race_closed_at の実データ形式
PATHS = {"public": "/tmp/p.html"}


def _make_eval(venue_num=1, race_number=1):
    eval_id = f"{DATE}_{venue_num:02d}_{race_number:02d}"
    fs = FeatureSet(
        eval_id=eval_id, feature_schema_version=1, built_at="t",
        boat_features={1: {}}, race_features={}, local_features=None,
        missing_keys=(),
    )
    return RaceEvaluation(
        eval_id=eval_id, race_date=DATE, venue_num=venue_num, venue_name="桐生",
        race_number=race_number, is_night=False, engine_name="ver4",
        engine_version="4.0", feature_schema_version=1, model_version="m",
        evaluated_at="t", danger_score=1.0, danger_breakdown={}, upset_score=5.0,
        upset_reasons=(), rank_index={}, featured_boats=None, win_probs=None,
        race_type="", match_index=50.0, features=fs,
    )


class _RaceSource:
    def __init__(self, close_time=CLOSE):
        self.close_time = close_time
        self.calls = 0

    def resolve_race(self, race_date, venue_num, race_number):
        self.calls += 1
        return SimpleNamespace(close_time=self.close_time, weather=None)


class _EvalPipeline:
    def __init__(self):
        self.calls = 0

    def evaluate_race(self, race_date, venue_num, race_number, *, persist=False):
        self.calls += 1
        return _make_eval(venue_num, race_number)


class _BuyPipeline:
    def __init__(self, purchased=True):
        self.purchased = purchased

    def assess_decide_predict(self, evaluation):
        decision = BuyDecision(
            eval_id=evaluation.eval_id, purchased=self.purchased, buyscore=50.0,
            investment_type="本命", n_bets=1, cost=100, kelly_fraction=0.1,
            config_version="v", skip_reason=None, purchased_combos=("1-2-3",),
        )
        prediction = Prediction(
            eval_id=evaluation.eval_id, pred_combo="1-2-3", pred_prob=0.1,
            pred_ev=1.2, pred_odds=12.0, confidence=0.5, why_bet="", patterns=(),
        )
        return SimpleNamespace(kind="assessment"), decision, prediction


class _Output:
    def render_all(self, date_str, output_paths):
        return {n: RenderResult(output_path=p, summary={})
                for n, p in output_paths.items()}


class _Notif:
    def send_all(self, requests):
        return [NotificationResult(r.channel, sent=False) for r in requests]


def _bundle(race_source=None, purchased=True):
    return SimpleNamespace(
        race_source=race_source or _RaceSource(),
        evaluation_pipeline=_EvalPipeline(),
        buy_pipeline=_BuyPipeline(purchased),
        output_pipeline=_Output(),
        notification_pipeline=_Notif(),
    )


def _at(hour, minute):
    return lambda: datetime(2026, 9, 22, hour, minute, tzinfo=JST)


class TestDeadlineBoundary(unittest.TestCase):
    """S4.1 の境界（10分前ちょうどは対象）。"""

    def test_exactly_ten_minutes_before_is_included(self):
        self.assertTrue(is_before_deadline(CLOSE, _at(15, 14)(), 10))

    def test_one_minute_later_is_excluded(self):
        self.assertFalse(is_before_deadline(CLOSE, _at(15, 15)(), 10))

    def test_after_close_time_is_excluded(self):
        self.assertFalse(is_before_deadline(CLOSE, _at(15, 30)(), 10))

    def test_empty_close_time_is_error(self):
        with self.assertRaises(ValueError):
            is_before_deadline("", _at(12, 0)(), 10)

    def test_other_format_is_rejected(self):
        with self.assertRaises(ValueError):
            is_before_deadline("15:24", _at(12, 0)(), 10)

    def test_format_constant_matches_real_data(self):
        datetime.strptime(CLOSE, CLOSE_TIME_FORMAT)


class TestDeadlineInProductionDay(unittest.TestCase):
    def _run(self, now, races=None, bundle=None):
        bundle = bundle or _bundle()
        result = run_production_day(
            races or [(DATE, 1, 1)], lambda d, v, r: PATHS,
            bundle=bundle, deadline_minutes=10, now_provider=now,
        )
        return result, bundle

    def test_before_cutoff_is_evaluated(self):
        result, bundle = self._run(_at(12, 0))
        self.assertEqual(bundle.evaluation_pipeline.calls, 1)
        self.assertEqual(result["s4_excluded_count"], 0)
        self.assertEqual(result["success_count"], 1)
        self.assertEqual(result["status"], "success")

    def test_after_cutoff_is_excluded_without_evaluation(self):
        result, bundle = self._run(_at(15, 20))
        self.assertEqual(bundle.evaluation_pipeline.calls, 0)  # 評価も保存もしない
        self.assertEqual(result["s4_excluded_count"], 1)
        self.assertEqual(result["failure_count"], 0)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["s4_excluded"][0]["race"], f"{DATE}_1_1")
        self.assertIn("deadline", result["s4_excluded"][0]["reason"])

    def test_all_excluded_is_success_s5_2(self):
        result, _ = self._run(_at(15, 20), races=[(DATE, 1, 1), (DATE, 1, 2)])
        self.assertEqual(result["s4_excluded_count"], 2)
        self.assertEqual(result["status"], "success")  # S5.2 正常終了
        self.assertEqual(result["success_rate"], 0.0)

    def test_denominator_excludes_s4(self):
        # 2件中1件がS4除外・残り1件成功 → 分母1・rate=1.0
        bundle = _bundle()
        calls = {"n": 0}
        base = bundle.race_source.resolve_race

        def alternating(race_date, venue_num, race_number):
            calls["n"] += 1
            close = CLOSE if calls["n"] == 1 else "2026-09-22 12:00:00"
            return SimpleNamespace(close_time=close, weather=None)

        bundle.race_source.resolve_race = alternating
        self.assertIsNotNone(base)
        result = run_production_day(
            [(DATE, 1, 1), (DATE, 1, 2)], lambda d, v, r: PATHS,
            bundle=bundle, deadline_minutes=10, now_provider=_at(12, 0),
        )
        self.assertEqual(result["s4_excluded_count"], 1)
        self.assertEqual(result["success_count"], 1)
        self.assertAlmostEqual(result["success_rate"], 1.0)
        self.assertEqual(result["status"], "success")

    def test_no_deadline_minutes_keeps_previous_behaviour(self):
        bundle = _bundle()
        result = run_production_day(
            [(DATE, 1, 1)], lambda d, v, r: PATHS, bundle=bundle,
        )
        self.assertEqual(bundle.race_source.calls, 0)
        self.assertEqual(result["s4_excluded_count"], 0)


class TestS5_2AllExcluded(unittest.TestCase):
    """S5.2: 母集団あり・ポリシー適用後0件のINFOログ（正常終了）。"""

    LOGGER = "actions.production_entrypoint"
    S5_2 = "Production day S5.2"
    S5_1 = "no target races"

    def _run(self, races, **kwargs):
        return run_production_day(
            races, lambda d, v, r: PATHS, bundle=kwargs.pop("bundle", _bundle()),
            **kwargs,
        )

    def test_case1_all_s4_excluded_logs_s5_2(self):
        races = [(DATE, 1, 1), (DATE, 1, 2)]
        with self.assertLogs(self.LOGGER, level="INFO") as cm:
            result = self._run(
                races, deadline_minutes=10, now_provider=_at(15, 20),
            )
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["s4_excluded_count"], 2)
        self.assertEqual(result["incomparable_count"], 0)
        line = [l for l in cm.output if self.S5_2 in l]
        self.assertEqual(len(line), 1, cm.output)
        self.assertIn("total=2", line[0])
        self.assertIn("s4_excluded=2", line[0])

    def test_case2_zero_population_is_s5_1_not_s5_2(self):
        with self.assertLogs(self.LOGGER, level="INFO") as cm:
            result = self._run([], deadline_minutes=10, now_provider=_at(15, 20))
        self.assertEqual(result["status"], "success")
        self.assertFalse(any(self.S5_2 in l for l in cm.output), cm.output)
        self.assertTrue(any(self.S5_1 in l for l in cm.output), cm.output)

    def test_case3_all_incomparable_is_not_s5_2(self):
        bundle = _bundle()

        def _empty_list(race_date, venue_num, race_number, *, persist=False):
            raise ValueError(
                "_evaluate_bets returned an empty list; no bet candidate is "
                "available for this race"
            )

        bundle.evaluation_pipeline.evaluate_race = _empty_list
        with self.assertLogs(self.LOGGER, level="INFO") as cm:
            result = self._run([(DATE, 1, 1)], bundle=bundle)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["incomparable_count"], 1)
        self.assertFalse(any(self.S5_2 in l for l in cm.output), cm.output)

    def test_case4_mixed_s4_and_incomparable_is_not_s5_2(self):
        bundle = _bundle()
        closes = {(DATE, 1, 1): CLOSE, (DATE, 1, 2): "2026-09-22 12:00:00"}

        def resolve(race_date, venue_num, race_number):
            return SimpleNamespace(
                close_time=closes[(race_date, venue_num, race_number)],
                weather=None,
            )

        def _empty_list(race_date, venue_num, race_number, *, persist=False):
            raise ValueError(
                "_evaluate_bets returned an empty list; no bet candidate is "
                "available for this race"
            )

        bundle.race_source.resolve_race = resolve
        bundle.evaluation_pipeline.evaluate_race = _empty_list
        with self.assertLogs(self.LOGGER, level="INFO") as cm:
            result = self._run(
                [(DATE, 1, 1), (DATE, 1, 2)], bundle=bundle,
                deadline_minutes=10, now_provider=_at(12, 0),
            )
        # 1件はS4除外（close_time 12:00 は 12:00時点で締切10分前を過ぎている）、
        # 残り1件は incomparable → 有効対象0だが S5.2 ではない
        self.assertEqual(result["s4_excluded_count"], 1)
        self.assertEqual(result["incomparable_count"], 1)
        self.assertEqual(result["status"], "failed")
        self.assertFalse(any(self.S5_2 in l for l in cm.output), cm.output)


class TestDailyNotificationLimit(unittest.TestCase):
    def _counter(self):
        return DailyNotificationCounter(
            os.path.join(tempfile.mkdtemp(), "notification_counts")
        )

    def _run(self, counter, races, limit=2):
        bundle = _bundle()
        result = run_production_day(
            races, lambda d, v, r: PATHS, bundle=bundle,
            daily_notification_limit=limit, notification_counter=counter,
        )
        return result, bundle

    def test_requests_are_counted_per_race(self):
        counter = self._counter()
        result, _ = self._run(counter, [(DATE, 1, 1)])
        self.assertEqual(len(result["results"][0]["requests"]), 1)
        self.assertEqual(counter.count(DATE), 1)

    def test_limit_stops_new_requests_but_job_continues(self):
        counter = self._counter()
        races = [(DATE, 1, n) for n in (1, 2, 3)]
        result, bundle = self._run(counter, races, limit=2)
        self.assertEqual(counter.count(DATE), 2)
        made = [len(r["requests"]) for r in result["results"]]
        self.assertEqual(made, [1, 1, 0])
        self.assertEqual(result["success_count"], 3)  # ジョブは継続
        self.assertEqual(result["status"], "success")
        self.assertEqual(bundle.evaluation_pipeline.calls, 3)

    def test_count_persists_across_runs(self):
        counter = self._counter()
        self._run(counter, [(DATE, 1, 1)], limit=2)
        self._run(counter, [(DATE, 1, 2)], limit=2)
        self.assertEqual(counter.count(DATE), 2)
        result, _ = self._run(counter, [(DATE, 1, 3)], limit=2)
        self.assertEqual(len(result["results"][0]["requests"]), 0)
        self.assertEqual(counter.count(DATE), 2)

    def test_limit_requires_counter(self):
        with self.assertRaises(ValueError):
            run_production_day(
                [(DATE, 1, 1)], lambda d, v, r: PATHS, bundle=_bundle(),
                daily_notification_limit=10,
            )

    def test_counter_is_per_race_date(self):
        counter = self._counter()
        self._run(counter, [(DATE, 1, 1)], limit=2)
        self.assertEqual(counter.count("20260923"), 0)


class TestNotificationCounterStore(unittest.TestCase):
    def setUp(self):
        self.base = os.path.join(tempfile.mkdtemp(), "notification_counts")

    def test_starts_at_zero_and_accumulates(self):
        counter = DailyNotificationCounter(self.base)
        self.assertEqual(counter.count(DATE), 0)
        self.assertEqual(counter.add(DATE, 2), 2)
        self.assertEqual(counter.add(DATE, 1), 3)
        saved = json.loads((Path(self.base) / f"{DATE}.json").read_text("utf-8"))
        self.assertEqual(saved, {"race_date": DATE, "count": 3})

    def test_zero_delta_writes_nothing(self):
        counter = DailyNotificationCounter(self.base)
        counter.add(DATE, 0)
        self.assertFalse((Path(self.base) / f"{DATE}.json").exists())

    def test_uploads_and_commits_when_clients_given(self):
        uploads, commits = [], []
        counter = DailyNotificationCounter(
            self.base,
            release=SimpleNamespace(
                upload_asset=lambda tag, name, path: uploads.append((tag, name))
            ),
            git=SimpleNamespace(
                commit_and_push=lambda paths, message: commits.append(paths)
            ),
        )
        counter.add(DATE, 1)
        self.assertEqual(uploads, [("data-store-v2", f"notification_counts_{DATE}.json")])
        self.assertEqual(len(commits), 1)

    def test_broken_file_is_error_not_zero(self):
        Path(self.base).mkdir(parents=True, exist_ok=True)
        (Path(self.base) / f"{DATE}.json").write_text('{"count": -1}', "utf-8")
        with self.assertRaises(ValueError):
            DailyNotificationCounter(self.base).count(DATE)


class TestConfigLoader(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def _write(self, name, payload):
        Path(self.dir, name).write_text(
            json.dumps(payload, ensure_ascii=False), "utf-8"
        )

    def test_reads_repository_config_values(self):
        self.assertEqual(load_deadline_minutes("config"), 10)
        self.assertEqual(load_daily_notification_limit("config"), 10)

    def test_missing_file_is_config_error(self):
        with self.assertRaises(ConfigError):
            load_deadline_minutes(self.dir)

    def test_missing_version_is_config_error(self):
        self._write("pipeline.json", {"締切前分数": 10})
        with self.assertRaises(ConfigError):
            load_deadline_minutes(self.dir)

    def test_missing_key_is_config_error_not_default(self):
        self._write("pipeline.json", {"_version": 1})
        with self.assertRaises(ConfigError):
            load_deadline_minutes(self.dir)

    def test_wrong_type_is_config_error(self):
        self._write("delivery.json", {"_version": 1, "1日投稿数上限": "10"})
        with self.assertRaises(ConfigError):
            load_daily_notification_limit(self.dir)

    def test_broken_json_is_config_error(self):
        Path(self.dir, "pipeline.json").write_text("{", "utf-8")
        with self.assertRaises(ConfigError):
            load_deadline_minutes(self.dir)


if __name__ == "__main__":
    unittest.main()
