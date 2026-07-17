"""
1:1 Mapper（MotorHistoryCsvMapper, LocalCourseStatsCsvMapper）の単体テスト。

Step2実装計画書 §1.3・§4（往復戦略）に対応。
標準ライブラリ unittest のみを使用する。
"""

from __future__ import annotations

import unittest

from models.record import LocalCourseStats, MotorHistory
from storage.mappers.local_course_stats_mapper import (
    LEGACY_COLUMNS as LOCAL_COURSE_STATS_COLUMNS,
)
from storage.mappers.local_course_stats_mapper import LocalCourseStatsCsvMapper
from storage.mappers.motor_history_mapper import LEGACY_COLUMNS as MOTOR_HISTORY_COLUMNS
from storage.mappers.motor_history_mapper import MotorHistoryCsvMapper
from tests.helpers import assert_row_equal


def _build_sample_motor_history() -> MotorHistory:
    return MotorHistory(
        date="20260704",
        venue_num=12,
        venue="住之江",
        motor_no=12,
        racer_no="4999",
        racer_name="テスト選手",
        lane=1,
        place=1,
        ex_time=6.75,
        start_timing=0.16,
        race_number=5,
    )


def _build_sample_local_course_stats() -> LocalCourseStats:
    return LocalCourseStats(
        racer_no="3415",
        venue_code=12,
        venue_name="住之江",
        course=1,
        starts=46,
        first=18,
        second=10,
        third=6,
        fourth=5,
        fifth=4,
        sixth=3,
        first_rate=39.1,
        top2_rate=60.9,
        top3_rate=73.9,
        last_updated="20260713",
    )


class TestMotorHistoryCsvMapper(unittest.TestCase):
    def test_to_row_contains_all_legacy_columns(self) -> None:
        """to_row()の出力キーが現行CSVの11列と完全一致すること。"""
        model = _build_sample_motor_history()
        row = MotorHistoryCsvMapper.to_row(model)
        self.assertEqual(set(row.keys()), set(MOTOR_HISTORY_COLUMNS))

    def test_roundtrip_model_to_row_to_model(self) -> None:
        """不変条件: from_row(to_row(m)) == m（完全往復）。"""
        model = _build_sample_motor_history()
        restored = MotorHistoryCsvMapper.from_row(MotorHistoryCsvMapper.to_row(model))
        self.assertEqual(model, restored)

    def test_roundtrip_row_to_model_to_row(self) -> None:
        """逆方向の往復: to_row(from_row(r)) == r。"""
        row = {
            "date": "20260704",
            "venue_num": 12,
            "venue": "住之江",
            "motor_no": 12,
            "racer_no": "4999",
            "racer_name": "テスト選手",
            "lane": 1,
            "place": 1,
            "ex_time": 6.75,
            "start_timing": 0.16,
            "race_number": 5,
        }
        restored_row = MotorHistoryCsvMapper.to_row(MotorHistoryCsvMapper.from_row(row))
        assert_row_equal(self, row, restored_row)

    def test_from_row_accepts_str_typed_csv_values(self) -> None:
        """csvモジュールが返す全て文字列の行からも正しく復元できること。"""
        row = {
            "date": "20260704",
            "venue_num": "12",
            "venue": "住之江",
            "motor_no": "12",
            "racer_no": "4999",
            "racer_name": "テスト選手",
            "lane": "1",
            "place": "1",
            "ex_time": "6.75",
            "start_timing": "0.16",
            "race_number": "5",
        }
        model = MotorHistoryCsvMapper.from_row(row)
        self.assertEqual(model.venue_num, 12)
        self.assertIsInstance(model.venue_num, int)
        self.assertAlmostEqual(model.ex_time, 6.75)

    def test_float_precision_within_tolerance_after_roundtrip(self) -> None:
        """ex_time/start_timingの往復でfloat精度が1e-6以内に収まること。"""
        model = _build_sample_motor_history()
        row = MotorHistoryCsvMapper.to_row(model)
        # str化を経由する現実のCSV書込・読込を模してから復元する
        row_as_str = {k: str(v) for k, v in row.items()}
        restored = MotorHistoryCsvMapper.from_row(row_as_str)
        self.assertAlmostEqual(model.ex_time, restored.ex_time, delta=1e-6)
        self.assertAlmostEqual(model.start_timing, restored.start_timing, delta=1e-6)


class TestLocalCourseStatsCsvMapper(unittest.TestCase):
    def test_to_row_contains_all_legacy_columns(self) -> None:
        """to_row()の出力キーが現行CSVの15列と完全一致すること。"""
        model = _build_sample_local_course_stats()
        row = LocalCourseStatsCsvMapper.to_row(model)
        self.assertEqual(set(row.keys()), set(LOCAL_COURSE_STATS_COLUMNS))

    def test_roundtrip_model_to_row_to_model(self) -> None:
        """不変条件: from_row(to_row(m)) == m（完全往復）。"""
        model = _build_sample_local_course_stats()
        restored = LocalCourseStatsCsvMapper.from_row(
            LocalCourseStatsCsvMapper.to_row(model)
        )
        self.assertEqual(model, restored)

    def test_roundtrip_row_to_model_to_row(self) -> None:
        """逆方向の往復: to_row(from_row(r)) == r。"""
        row = {
            "racer_no": "3415",
            "venue_code": 12,
            "venue_name": "住之江",
            "course": 1,
            "starts": 46,
            "first": 18,
            "second": 10,
            "third": 6,
            "fourth": 5,
            "fifth": 4,
            "sixth": 3,
            "first_rate": 39.1,
            "top2_rate": 60.9,
            "top3_rate": 73.9,
            "last_updated": "20260713",
        }
        restored_row = LocalCourseStatsCsvMapper.to_row(
            LocalCourseStatsCsvMapper.from_row(row)
        )
        assert_row_equal(self, row, restored_row)

    def test_from_row_accepts_str_typed_csv_values(self) -> None:
        """csvモジュールが返す全て文字列の行からも正しく復元できること。"""
        row = {
            "racer_no": "3415",
            "venue_code": "12",
            "venue_name": "住之江",
            "course": "1",
            "starts": "46",
            "first": "18",
            "second": "10",
            "third": "6",
            "fourth": "5",
            "fifth": "4",
            "sixth": "3",
            "first_rate": "39.1",
            "top2_rate": "60.9",
            "top3_rate": "73.9",
            "last_updated": "20260713",
        }
        model = LocalCourseStatsCsvMapper.from_row(row)
        self.assertEqual(model.venue_code, 12)
        self.assertIsInstance(model.venue_code, int)
        self.assertAlmostEqual(model.first_rate, 39.1)

    def test_float_precision_within_tolerance_after_roundtrip(self) -> None:
        """first_rate/top2_rate/top3_rateの往復でfloat精度が1e-6以内に収まること。"""
        model = _build_sample_local_course_stats()
        row = LocalCourseStatsCsvMapper.to_row(model)
        row_as_str = {k: str(v) for k, v in row.items()}
        restored = LocalCourseStatsCsvMapper.from_row(row_as_str)
        self.assertAlmostEqual(model.first_rate, restored.first_rate, delta=1e-6)
        self.assertAlmostEqual(model.top2_rate, restored.top2_rate, delta=1e-6)
        self.assertAlmostEqual(model.top3_rate, restored.top3_rate, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
