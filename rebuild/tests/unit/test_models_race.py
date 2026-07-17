"""
Race系データモデル（models/race.py）の単体テスト。

設計書 Phase0.5 v1.1 ③3.1・3.2、⑭（単体テスト: core全関数だが、
モデル自体の構築・不変性・eval_id算出ロジックもここで検証する）に対応。

標準ライブラリ unittest のみを使用する（外部テストランナーへの依存を追加しない）。
"""

from __future__ import annotations

import unittest

from models.race import CourseStats, OddsSnapshot, Race, RaceEntry, Weather


def _build_sample_entry(lane: int = 1) -> RaceEntry:
    """テスト用の最小構成 RaceEntry を組み立てるヘルパー。"""
    return RaceEntry(
        lane=lane,
        racer_no="4999",
        racer_name="テスト選手",
        racer_class="A1",
        win_rate=6.50,
        place_rate=45.0,
        motor_no=12,
        motor_rate2=38.5,
        avg_st=0.16,
    )


def _build_sample_race(entry_count: int = 6) -> Race:
    """テスト用の最小構成 Race を組み立てるヘルパー。"""
    entries = tuple(_build_sample_entry(lane=i + 1) for i in range(entry_count))
    return Race(
        race_date="20260713",
        venue_num=12,
        venue_name="住之江",
        race_number=5,
        close_time="15:40",
        is_night=False,
        entries=entries,
    )


class TestRaceEntry(unittest.TestCase):
    def test_construct_minimum_fields(self) -> None:
        """必須項目のみでRaceEntryが構築できること（任意項目は既定でNone）。"""
        entry = _build_sample_entry()
        self.assertEqual(entry.lane, 1)
        self.assertEqual(entry.racer_class, "A1")
        self.assertIsNone(entry.branch)
        self.assertIsNone(entry.course_stats)

    def test_immutable(self) -> None:
        """frozen dataclassのため属性再代入がFrozenInstanceErrorになること。

        設計書⑤5.4（純粋性の定義）を支える性質: データモデルが不変であることで、
        同一入力からの再現性・回帰テストでの比較が安全に行える。
        """
        entry = _build_sample_entry()
        with self.assertRaises(Exception):
            entry.lane = 2  # type: ignore[misc]

    def test_course_stats_optional_tuple(self) -> None:
        """course_statsに複数コース分のCourseStatsを保持できること。"""
        stats = (
            CourseStats(
                course=1,
                entry_count=46,
                place_rate=73.9,
                avg_start_timing=0.15,
                avg_start_rank=2.40,
            ),
            CourseStats(
                course=2,
                entry_count=28,
                place_rate=42.9,
                avg_start_timing=0.15,
                avg_start_rank=2.70,
            ),
        )
        entry = RaceEntry(
            lane=1,
            racer_no="4999",
            racer_name="テスト選手",
            racer_class="A1",
            win_rate=6.50,
            place_rate=45.0,
            motor_no=12,
            motor_rate2=38.5,
            avg_st=0.16,
            course_stats=stats,
        )
        self.assertEqual(len(entry.course_stats), 2)
        self.assertEqual(entry.course_stats[0].course, 1)


class TestRace(unittest.TestCase):
    def test_construct_with_six_entries(self) -> None:
        """6艇分のRaceEntryを保持できること。"""
        race = _build_sample_race(entry_count=6)
        self.assertEqual(len(race.entries), 6)
        self.assertEqual(race.entries[0].lane, 1)
        self.assertEqual(race.entries[5].lane, 6)

    def test_eval_id_format(self) -> None:
        """eval_idが設計書③3.4と同一の採番規則
        '{race_date}_{venue_num:02d}_{race_number:02d}' で生成されること。
        """
        race = _build_sample_race()
        self.assertEqual(race.eval_id, "20260713_12_05")

    def test_eval_id_zero_pads_single_digit_venue_and_race(self) -> None:
        """venue_num・race_numberが1桁でも2桁ゼロ埋めされること。"""
        race = Race(
            race_date="20260713",
            venue_num=1,
            venue_name="桐生",
            race_number=3,
            close_time="10:30",
            is_night=False,
            entries=(_build_sample_entry(),),
        )
        self.assertEqual(race.eval_id, "20260713_01_03")

    def test_grade_and_weather_default_to_none(self) -> None:
        """grade・weatherは任意項目であり、未指定時はNoneであること。"""
        race = _build_sample_race()
        self.assertIsNone(race.grade)
        self.assertIsNone(race.weather)

    def test_weather_is_optional_and_not_required_for_evaluation(self) -> None:
        """Weatherを保持できるが、必須ではないこと（朝刊ポリシー: 評価には不使用）。"""
        weather = Weather(
            wind_speed_mps=2.5,
            wind_direction="北",
            wave_height_cm=3,
        )
        race = Race(
            race_date="20260713",
            venue_num=12,
            venue_name="住之江",
            race_number=5,
            close_time="15:40",
            is_night=False,
            entries=(_build_sample_entry(),),
            weather=weather,
        )
        self.assertEqual(race.weather.wind_speed_mps, 2.5)


class TestOddsSnapshot(unittest.TestCase):
    def test_construct_with_trifecta_odds(self) -> None:
        """3連単オッズを組番文字列キーの辞書として保持できること。"""
        snapshot = OddsSnapshot(
            eval_id="20260713_12_05",
            fetched_at="2026-07-13T15:30:00+09:00",
            trifecta_odds={"1-2-3": 12.5, "1-3-2": 15.0},
        )
        self.assertEqual(snapshot.trifecta_odds["1-2-3"], 12.5)

    def test_trifecta_odds_defaults_to_empty_dict(self) -> None:
        """trifecta_oddsを省略した場合、空辞書が既定値になること
        （可変デフォルト引数の共有バグが起きないことをあわせて確認する）。
        """
        snap1 = OddsSnapshot(eval_id="a", fetched_at="t")
        snap2 = OddsSnapshot(eval_id="b", fetched_at="t")
        # dataclassのfield(default_factory=dict)により、各インスタンスが
        # 別々の辞書を持つことを確認する。
        snap1.trifecta_odds["1-2-3"] = 10.0
        self.assertEqual(snap2.trifecta_odds, {})


class TestCourseStats(unittest.TestCase):
    def test_construct(self) -> None:
        """FANファイル仕様書のコース別成績項目を保持できること。"""
        stats = CourseStats(
            course=1,
            entry_count=46,
            place_rate=73.9,
            avg_start_timing=0.15,
            avg_start_rank=2.40,
        )
        self.assertEqual(stats.course, 1)
        self.assertEqual(stats.place_rate, 73.9)


if __name__ == "__main__":
    unittest.main()
