#!/usr/bin/env python3
"""
2026-10-09 の現行システム障害修正のテスト。
  python -m unittest test_b3_jst_date_20261009

- notify_arashi._run_main: 既定日付を JST で決める（B3）
  記録: docs/incidents/2026-10-09_run-main-date-utc.md
"""
from __future__ import annotations

import datetime as _datetime_module
import logging
import unittest
from unittest import mock

logging.disable(logging.CRITICAL)

_UTC = _datetime_module.timezone.utc


def _fixed_datetime(utc_now: _datetime_module.datetime) -> type:
    """now(tz) が固定時刻を返す datetime の代わり。"""

    class _FixedDateTime(_datetime_module.datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            if tz is None:
                return utc_now.replace(tzinfo=None)
            return utc_now.astimezone(tz)

    return _FixedDateTime


class _Stop(Exception):
    """最初のログ出力で _run_main を止めるための例外。"""


class TestRunMainDefaultDateJST(unittest.TestCase):
    def _race_date_at(self, utc_now: _datetime_module.datetime,
                      race_date: str | None = None) -> str:
        import notify_arashi as na

        captured: list[str] = []

        def _first_log(msg, *args, **kwargs):
            captured.append(args[0])
            raise _Stop

        with mock.patch.object(_datetime_module, "datetime", _fixed_datetime(utc_now)), \
             mock.patch.object(na.log, "info", side_effect=_first_log):
            with self.assertRaises(_Stop):
                na._run_main(race_date)
        return captured[0]

    def test_utc_2330_is_next_day_in_jst(self) -> None:
        # UTC 23:30 = JST 翌日 08:30
        now = _datetime_module.datetime(2026, 10, 9, 23, 30, tzinfo=_UTC)
        self.assertEqual(self._race_date_at(now), "20261010")

    def test_jst_2359_is_same_day(self) -> None:
        # UTC 14:59 = JST 23:59
        now = _datetime_module.datetime(2026, 10, 9, 14, 59, tzinfo=_UTC)
        self.assertEqual(self._race_date_at(now), "20261009")

    def test_jst_0000_is_next_day(self) -> None:
        # UTC 15:00 = JST 翌日 00:00
        now = _datetime_module.datetime(2026, 10, 9, 15, 0, tzinfo=_UTC)
        self.assertEqual(self._race_date_at(now), "20261010")

    def test_explicit_race_date_is_kept(self) -> None:
        now = _datetime_module.datetime(2026, 10, 9, 23, 30, tzinfo=_UTC)
        self.assertEqual(self._race_date_at(now, "20261001"), "20261001")


if __name__ == "__main__":
    unittest.main()
