"""
calc_danger_score（core/danger.py）の単体テスト。

Step4計画書 C8（Golden回帰と分離した純関数検証）に対応。
移植元（x_asahi_scoring.py L210-L360）の相対評価・単体評価・ゲート条件・
上限クリップを小さな合成configと合成艇データで個別に検証する。
"""

from __future__ import annotations

import unittest

from core.danger import calc_danger_score
from tests.fakes import FakeVenueStatsProvider


def _config(**overrides) -> dict:
    base = {
        "danger_score": {
            "total_scale": 100,
            "relative_weights": {
                "_comment": "x",
                "win_rate": {"per_boat": 3.0, "max_weight": 12.0},
                "local_win": {"per_boat": 2.0, "max_weight": 8.0},
                "avg_st": {"per_boat": 2.0, "max_weight": 8.0},
                "motor": {"per_boat": 2.0, "max_weight": 8.0},
                "racer_class": {"per_boat": 2.0, "max_weight": 6.0},
                "ability_trend": {"per_boat": 1.0, "max_weight": 4.0},
                "course_rentai": {"per_boat": 2.0, "max_weight": 8.0},
            },
            "solo_weights": {
                "_comment": "x",
                "win_rate_low": {"weight": 2.0},
                "local_win_low": {"weight": 2.0},
                "motor_bad": {"weight": 2.0},
                "avg_st_slow": {"weight": 2.0},
                "course1_place_low": {"weight": 3.0},
                "f_risk": {"weight": 3.0},
                "venue_unfavorable": {"weight": 2.0},
            },
            "solo_thresholds": {
                "win_rate_low_abs": 5.0,
                "local_win_low_abs": 5.0,
                "motor_bad_abs": 35.0,
                "avg_st_slow_abs": 0.18,
                "course1_place_low_abs": 40.0,
                "f_risk_rate_abs": 2.0,
                "venue_unfavorable_factor_abs": 0.95,
            },
            "class_rank": {"A1": 4, "A2": 3, "B1": 2, "B2": 1, "": 0},
        }
    }
    base["danger_score"].update(overrides)
    return base


def _boat(lane: int, **overrides) -> dict:
    base = {
        "lane": lane,
        "racer_class": "A1",
        "win_rate": 6.0,
        "local_win": 6.0,
        "avg_st": 0.15,
        "motor": 40.0,
        "ability_curr": 75.0,
        "ability_prev": 74.0,
        "course_nyuko": [40, 30, 20, 10, 5, 2],
        "course_win_rate": [55.0, 40.0, 30.0, 20.0, 10.0, 5.0],
        "course_place_rate": [70.0, 55.0, 45.0, 35.0, 20.0, 10.0],
        "course_place_counts": [[20, 8, 4, 4, 2, 2]] * 6,
        "course_f_count": [0, 0, 0, 0, 0, 0],
    }
    base.update(overrides)
    return base


class TestGuards(unittest.TestCase):
    def test_no_boat1_returns_zero(self) -> None:
        score, breakdown = calc_danger_score(None, [_boat(1), _boat(2)], _config())
        self.assertEqual((score, breakdown), (0.0, {}))

    def test_less_than_two_boats_returns_zero(self) -> None:
        b1 = _boat(1)
        score, breakdown = calc_danger_score(b1, [b1], _config())
        self.assertEqual((score, breakdown), (0.0, {}))

    def test_no_other_boats_returns_zero(self) -> None:
        b1 = _boat(1)
        score, breakdown = calc_danger_score(b1, [b1, _boat(1)], _config())
        self.assertEqual((score, breakdown), (0.0, {}))


class TestRelativeItems(unittest.TestCase):
    def test_win_rate_counts_stronger_boats(self) -> None:
        b1 = _boat(1, win_rate=5.0)
        others = [_boat(2, win_rate=6.0), _boat(3, win_rate=7.0), _boat(4, win_rate=4.0)]
        _, breakdown = calc_danger_score(b1, [b1] + others, _config())
        item = breakdown["win_rate"]
        self.assertEqual(item["worse_count"], 2)  # 6.0と7.0が上
        self.assertEqual(item["worse_total"], 3)
        self.assertEqual(item["weighted"], 6.0)  # per_boat 3.0 × 2
        self.assertEqual(item["kind"], "relative")

    def test_max_weight_caps_relative(self) -> None:
        b1 = _boat(1, win_rate=1.0)
        others = [_boat(i, win_rate=7.0) for i in range(2, 7)]  # 5艇全部上
        _, breakdown = calc_danger_score(b1, [b1] + others, _config())
        self.assertEqual(breakdown["win_rate"]["weighted"], 12.0)  # 3.0×5=15 -> cap 12

    def test_avg_st_lower_is_better(self) -> None:
        b1 = _boat(1, avg_st=0.17)
        others = [_boat(2, avg_st=0.14), _boat(3, avg_st=0.19)]
        _, breakdown = calc_danger_score(b1, [b1] + others, _config())
        self.assertEqual(breakdown["avg_st"]["worse_count"], 1)  # 0.14のみ「小さい=上」

    def test_avg_st_missing_defaults_to_018(self) -> None:
        """欠損STは0.18で補完される（移植元と同一のor補完）。"""
        b1 = _boat(1, avg_st=None)
        others = [_boat(2, avg_st=0.17)]  # 0.17 < 0.18 -> 上が1艇
        _, breakdown = calc_danger_score(b1, [b1] + others, _config())
        self.assertEqual(breakdown["avg_st"]["worse_count"], 1)

    def test_racer_class_rank_comparison(self) -> None:
        b1 = _boat(1, racer_class="A2")
        others = [_boat(2, racer_class="A1"), _boat(3, racer_class="B1")]
        _, breakdown = calc_danger_score(b1, [b1] + others, _config())
        self.assertEqual(breakdown["racer_class"]["worse_count"], 1)  # A1のみ上

    def test_ability_trend_zero_quirk(self) -> None:
        """curr/prevのどちらかがfalsy（0.0含む）なら推移0.0（移植元の挙動保存）。"""
        b1 = _boat(1, ability_curr=0.0, ability_prev=74.0)  # -> 0.0
        others = [_boat(2, ability_curr=75.0, ability_prev=74.0)]  # -> +1.0
        _, breakdown = calc_danger_score(b1, [b1] + others, _config())
        self.assertEqual(breakdown["ability_trend"]["worse_count"], 1)

    def test_course_rentai_uses_place_rate_fallback(self) -> None:
        """着順回数が全ゼロなら複勝率で近似する（移植元と同一）。"""
        b1 = _boat(1, course_place_counts=[[0] * 6] * 6)  # -> place_rate[0]=70.0
        others = [_boat(2, course_place_counts=[[0] * 6] * 6)]  # lane2: place_rate[1]=55.0
        _, breakdown = calc_danger_score(b1, [b1] + others, _config())
        self.assertEqual(breakdown["course_rentai"]["worse_count"], 0)  # 70>55

    def test_course_rentai_venue_factor_applied(self) -> None:
        """venue指定時は各艇のレーンにfactorが乗る。"""
        provider = FakeVenueStatsProvider(
            course_factors={"住之江": {"factor": 0.5, "samples": 10}}
        )
        b1 = _boat(1)
        b2 = _boat(2)
        # factor一律0.5では相対関係は不変
        _, without = calc_danger_score(b1, [b1, b2], _config())
        _, with_venue = calc_danger_score(
            b1, [b1, b2], _config(), venue_name="住之江", venue_stats=provider
        )
        self.assertEqual(
            without["course_rentai"]["worse_count"],
            with_venue["course_rentai"]["worse_count"],
        )


class TestSoloItems(unittest.TestCase):
    def test_solo_thresholds_trigger(self) -> None:
        b1 = _boat(
            1, win_rate=4.0, local_win=4.0, motor=30.0, avg_st=0.19,
            course_win_rate=[35.0, 0, 0, 0, 0, 0],
            course_f_count=[2, 0, 0, 0, 0, 0],  # 2/40*100=5.0% >= 2.0%
        )
        others = [_boat(2)]
        _, breakdown = calc_danger_score(b1, [b1] + others, _config())
        for key in ("win_rate_low", "local_win_low", "motor_bad", "avg_st_slow",
                    "course1_place_low", "f_risk"):
            self.assertIn(key, breakdown, key)
            self.assertEqual(breakdown[key]["kind"], "solo")

    def test_course1_win_gate_requires_nyuko(self) -> None:
        """1コース進入0回ならcourse1_place_low判定自体を行わない。"""
        b1 = _boat(1, course_nyuko=[0, 30, 20, 10, 5, 2],
                   course_win_rate=[10.0, 0, 0, 0, 0, 0])
        _, breakdown = calc_danger_score(b1, [b1, _boat(2)], _config())
        self.assertNotIn("course1_place_low", breakdown)

    def test_venue_unfavorable_when_factor_low(self) -> None:
        provider = FakeVenueStatsProvider(
            course_factors={"戸田": {"factor": 0.90, "samples": 50}}
        )
        b1 = _boat(1)
        _, breakdown = calc_danger_score(
            b1, [b1, _boat(2)], _config(), venue_name="戸田", venue_stats=provider
        )
        self.assertIn("venue_unfavorable", breakdown)

    def test_no_venue_no_venue_items(self) -> None:
        b1 = _boat(1)
        _, breakdown = calc_danger_score(b1, [b1, _boat(2)], _config())
        self.assertNotIn("venue_unfavorable", breakdown)


class TestTotal(unittest.TestCase):
    def test_total_is_sum_of_weighted_rounded(self) -> None:
        b1 = _boat(1, win_rate=5.0)
        others = [_boat(2, win_rate=6.0)]
        score, breakdown = calc_danger_score(b1, [b1] + others, _config())
        self.assertAlmostEqual(
            score, round(sum(v["weighted"] for v in breakdown.values()), 2), delta=1e-9
        )

    def test_total_scale_caps_score(self) -> None:
        cfg = _config(total_scale=5)
        b1 = _boat(1, win_rate=1.0, local_win=1.0, motor=20.0, avg_st=0.25,
                   racer_class="B2")
        others = [_boat(i, win_rate=7.0) for i in range(2, 7)]
        score, _ = calc_danger_score(b1, [b1] + others, cfg)
        self.assertEqual(score, 5)


if __name__ == "__main__":
    unittest.main()
