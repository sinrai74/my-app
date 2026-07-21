"""
adapters/providers.py の単体テスト（Step5-1）。

指示のテスト最低要件に対応:
  正常取得 / 空データ / Fetch失敗 / Legacy取得件数一致。
Legacy本番モジュールへの実接続を避けるため、fetcher/extractorはすべて
DIしたFakeで差し替える（実API・実ファイルアクセスなし）。
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from adapters.exceptions import ProviderError
from adapters.providers import (
    BOAT_ATTRS,
    BoatsProvider,
    DefaultOddsProvider,
    DefaultRaceTypeProvider,
    DefaultVenueProvider,
    DefaultWeatherProvider,
    _boat_to_race_entry,
)
from models.race import OddsSnapshot, Race, Weather


def _boat_info(lane: int, **over) -> SimpleNamespace:
    base = dict(
        lane=lane, name=f"選手{lane}", racer_class="A1", racer_id=f"400{lane}",
        win_rate=6.5, local_win=6.0, motor=38.0, avg_st=0.16,
        ability_curr=75.0, ability_prev=74.0,
        course_nyuko=[40, 30, 20, 10, 5, 2],
        course_win_rate=[55.0, 40.0, 30.0, 20.0, 10.0, 5.0],
        course_place_rate=[70.0, 55.0, 45.0, 35.0, 20.0, 10.0],
        course_place_counts=[[20, 8, 4, 4, 2, 2]] * 6,
        course_rank=[2.4, 2.7, 2.7, 3.2, 3.3, 4.0],
        course_st=[0.15, 0.16, 0.16, 0.17, 0.18, 0.20],
        course_f_count=[0, 0, 0, 0, 0, 0],
        course_l_count=[0, 0, 0, 0, 0, 0],
    )
    base.update(over)
    return SimpleNamespace(**base)


def _program(venue_num: int = 12, race_number: int = 5, grade: int = 0) -> dict:
    return {
        "race_stadium_number": venue_num,
        "race_number": race_number,
        "race_grade_number": grade,
        "race_closed_at": "15:00",
    }


def _boats_provider(boats=None, program=None, **kwargs) -> BoatsProvider:
    boats = boats if boats is not None else [_boat_info(i) for i in range(1, 7)]
    program = program if program is not None else _program()
    return BoatsProvider(
        programs_fetcher=lambda date: [program],
        boats_extractor=lambda prog: boats,
        **kwargs,
    )


class TestVenueProvider(unittest.TestCase):
    def test_known_venue(self) -> None:
        self.assertEqual(DefaultVenueProvider().resolve_venue_name(12), "住之江")

    def test_unknown_venue_raises(self) -> None:
        with self.assertRaises(ProviderError):
            DefaultVenueProvider().resolve_venue_name(99)


class TestOddsProvider(unittest.TestCase):
    def test_normal_fetch(self) -> None:
        provider = DefaultOddsProvider(
            odds_fetcher=lambda rno, vc, date: {"1-2-3": 47.2, "1-3-2": 60.0}
        )
        snap = provider.resolve_odds("20260704", 12, 5)
        self.assertIsInstance(snap, OddsSnapshot)
        self.assertEqual(snap.eval_id, "20260704_12_05")
        self.assertEqual(snap.trifecta_odds["1-2-3"], 47.2)

    def test_empty_odds(self) -> None:
        provider = DefaultOddsProvider(odds_fetcher=lambda rno, vc, date: {})
        snap = provider.resolve_odds("20260704", 12, 5)
        self.assertEqual(snap.trifecta_odds, {})

    def test_fetch_failure_raises_provider_error(self) -> None:
        def _boom(rno, vc, date):
            raise RuntimeError("network")
        provider = DefaultOddsProvider(odds_fetcher=_boom)
        with self.assertRaises(ProviderError):
            provider.resolve_odds("20260704", 12, 5)


class TestBoatsProviderRace(unittest.TestCase):
    def test_resolve_race_builds_domain_model(self) -> None:
        race = _boats_provider().resolve_race("20260704", 12, 5)
        self.assertIsInstance(race, Race)
        self.assertEqual(race.eval_id, "20260704_12_05")
        self.assertEqual(race.venue_name, "住之江")
        self.assertEqual(len(race.entries), 6)
        self.assertTrue(race.is_night)  # 12はナイター場

    def test_race_entry_has_no_ver4_attrs(self) -> None:
        """RaceEntryにVer4専用属性が混入していないこと（Freeze厳守）。"""
        race = _boats_provider().resolve_race("20260704", 12, 5)
        entry = race.entries[0]
        for attr in ("local_win", "ability_curr", "course_win_rate"):
            self.assertFalse(hasattr(entry, attr), attr)

    def test_grade_number_conversion(self) -> None:
        p = _boats_provider(program=_program(grade=4))
        self.assertEqual(p.resolve_race("20260704", 12, 5).grade, "SG")
        p2 = _boats_provider(program=_program(grade=5))
        self.assertEqual(p2.resolve_race("20260704", 12, 5).grade, "grade5")


class TestBoatsProviderVer4Boats(unittest.TestCase):
    def test_resolve_boats_has_all_17_attrs(self) -> None:
        boats = _boats_provider().resolve_boats("20260704", 12, 5)
        self.assertEqual(len(boats), 6)
        for boat in boats:
            self.assertEqual(set(boat.keys()), set(BOAT_ATTRS))
        self.assertEqual(boats[0]["local_win"], 6.0)
        self.assertEqual(boats[0]["ability_curr"], 75.0)

    def test_single_fetch_for_both_conversions(self) -> None:
        """取得は1回のみ（resolve_race + resolve_boatsで二重取得しない）。"""
        calls = {"n": 0}

        def _counting_fetch(date):
            calls["n"] += 1
            return [_program()]

        provider = BoatsProvider(
            programs_fetcher=_counting_fetch,
            boats_extractor=lambda prog: [_boat_info(i) for i in range(1, 7)],
        )
        provider.resolve_race("20260704", 12, 5)
        provider.resolve_boats("20260704", 12, 5)
        self.assertEqual(calls["n"], 1)  # メモ化により1回

    def test_boats_count_matches_legacy(self) -> None:
        """Legacy取得件数（extractorが返す件数）と一致すること。"""
        legacy_boats = [_boat_info(i) for i in range(1, 7)]
        provider = _boats_provider(boats=legacy_boats)
        self.assertEqual(
            len(provider.resolve_boats("20260704", 12, 5)), len(legacy_boats)
        )
        self.assertEqual(
            len(provider.resolve_race("20260704", 12, 5).entries), len(legacy_boats)
        )


class TestBoatsProviderErrors(unittest.TestCase):
    def test_empty_programs_raises(self) -> None:
        provider = BoatsProvider(
            programs_fetcher=lambda date: [],
            boats_extractor=lambda prog: [],
        )
        with self.assertRaises(ProviderError):
            provider.resolve_race("20260704", 12, 5)

    def test_program_not_found_raises(self) -> None:
        provider = BoatsProvider(
            programs_fetcher=lambda date: [_program(venue_num=1, race_number=1)],
            boats_extractor=lambda prog: [_boat_info(1)],
        )
        with self.assertRaises(ProviderError):
            provider.resolve_boats("20260704", 12, 5)

    def test_empty_boats_raises(self) -> None:
        provider = BoatsProvider(
            programs_fetcher=lambda date: [_program()],
            boats_extractor=lambda prog: [],
        )
        with self.assertRaises(ProviderError):
            provider.resolve_boats("20260704", 12, 5)

    def test_fetch_failure_raises_provider_error(self) -> None:
        def _boom(date):
            raise RuntimeError("api down")
        provider = BoatsProvider(
            programs_fetcher=_boom, boats_extractor=lambda prog: []
        )
        with self.assertRaises(ProviderError):
            provider.resolve_race("20260704", 12, 5)


class TestWeatherProvider(unittest.TestCase):
    def test_normal_weather(self) -> None:
        provider = DefaultWeatherProvider(
            weather_fetcher=lambda d, v, r: {
                "wind_speed": 3.0, "wind_direction": "追",
                "wave_height": 5, "temperature": 20.0, "water_temperature": 18.0,
            }
        )
        w = provider.resolve_weather("20260704", 12, 5)
        self.assertIsInstance(w, Weather)
        self.assertEqual(w.wind_speed_mps, 3.0)
        self.assertEqual(w.wave_height_cm, 5)

    def test_empty_weather_returns_none(self) -> None:
        provider = DefaultWeatherProvider(weather_fetcher=lambda d, v, r: None)
        self.assertIsNone(provider.resolve_weather("20260704", 12, 5))

    def test_fetch_failure_raises(self) -> None:
        def _boom(d, v, r):
            raise RuntimeError("x")
        provider = DefaultWeatherProvider(weather_fetcher=_boom)
        with self.assertRaises(ProviderError):
            provider.resolve_weather("20260704", 12, 5)


class TestRaceTypeProvider(unittest.TestCase):
    def test_delegates_to_classifier(self) -> None:
        provider = DefaultRaceTypeProvider(classifier=lambda **kw: "イン逃げ型")
        self.assertEqual(
            provider.resolve_race_type("20260704", 12, 5), "イン逃げ型"
        )

    def test_classifier_failure_raises(self) -> None:
        def _boom(**kw):
            raise RuntimeError("x")
        provider = DefaultRaceTypeProvider(classifier=_boom)
        with self.assertRaises(ProviderError):
            provider.resolve_race_type("20260704", 12, 5)


class TestBoatToRaceEntry(unittest.TestCase):
    def test_field_mapping(self) -> None:
        entry = _boat_to_race_entry(_boat_info(1))
        self.assertEqual(entry.lane, 1)
        self.assertEqual(entry.racer_name, "選手1")
        self.assertEqual(entry.motor_rate2, 38.0)  # motor -> motor_rate2
        self.assertEqual(entry.racer_no, "4001")   # racer_id -> racer_no
        self.assertIsNotNone(entry.course_stats)
        self.assertEqual(entry.course_stats[0].course, 1)
        self.assertEqual(entry.course_stats[0].entry_count, 40)


if __name__ == "__main__":
    unittest.main()
