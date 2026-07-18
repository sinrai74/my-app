"""
calc_lane_rank_scores / calc_rank_probabilities / calc_rank_index /
select_featured_boats / compute_match_index（core/rank.py）の単体テスト。

Step4計画書 C8（Golden回帰と分離した純関数検証）に対応。
移植元（x_asahi_scoring.py L613-L939・L536）の丸め・ゲート・フロア・
6号艇特別条件・featured閾値/マーク付与を合成入力で個別に検証する。
"""

from __future__ import annotations

import unittest

from core.rank import (
    calc_lane_rank_scores,
    calc_rank_index,
    calc_rank_probabilities,
    compute_match_index,
    select_featured_boats,
)
from tests.fakes import FakeVenueStatsProvider


def _config(**lrs_overrides) -> dict:
    def _weights(base=1.0, danger=0.0):
        return {
            "base_rate_weight": base,
            "win_rate_weight": 1.0,
            "local_win_weight": 0.5,
            "avg_st_weight": 1.0,
            "motor_weight": 0.2,
            "class_weight": 1.0,
            "course_perf_weight": 1.0,
            "ability_trend_weight": 0.5,
            "venue_factor_weight": 10.0,
            "f_risk_weight": -1.0,
            "course_st_weight": 0.5,
            "danger_penalty_weight": danger,
        }

    cfg = {
        "lane_rank_scores": {
            "lane_base_rate": {
                "1": {"first": 55, "second": 20, "third": 10},
                "2": {"first": 14, "second": 25, "third": 20},
                "3": {"first": 12, "second": 20, "third": 20},
                "4": {"first": 10, "second": 15, "third": 18},
                "5": {"first": 6, "second": 12, "third": 16},
                "6": {"first": 3, "second": 8, "third": 16},
            },
            "first_place": _weights(base=1.0, danger=1.0),
            "second_place": _weights(base=1.0, danger=0.5),
            "third_place": _weights(base=1.0, danger=0.2),
            "lane6_first_place_conditions": {
                "min_match_index": 85,
                "min_upset_score": 8.5,
                "normal_suppress_ratio": 0.35,
            },
        },
        "featured_racers": {
            "max_count": 3,
            "marks": ["◎", "○", "▲"],
            "exclude_boat1_threshold": 40,
            "min_composite_threshold": 0.0,
            "composite_weights": {
                "top1_weight": 0.5, "top2_weight": 0.3, "top3_weight": 0.2,
            },
        },
        "danger_score": {
            "total_scale": 100,
            "relative_weights": {},
            "solo_weights": {},
            "solo_thresholds": {
                "win_rate_low_abs": 0.0, "local_win_low_abs": 0.0,
                "motor_bad_abs": 0.0, "avg_st_slow_abs": 9.9,
                "course1_place_low_abs": 0.0, "f_risk_rate_abs": 999.0,
                "venue_unfavorable_factor_abs": 0.0,
            },
            "class_rank": {"A1": 4, "A2": 3, "B1": 2, "B2": 1, "": 0},
        },
    }
    cfg["lane_rank_scores"].update(lrs_overrides)
    return cfg


def _boat(lane: int, **overrides) -> dict:
    base = {
        "lane": lane,
        "name": f"選手{lane}",
        "racer_class": "B1",
        "win_rate": 5.5,
        "local_win": 5.5,
        "avg_st": 0.16,
        "motor": 36.0,
        "ability_curr": 60.0,
        "ability_prev": 60.0,
        "course_nyuko": [30, 30, 30, 30, 30, 30],
        "course_st": [0.16] * 6,
        "course_win_rate": [50.0, 15.0, 12.0, 10.0, 6.0, 3.0],
        "course_place_rate": [65.0, 45.0, 40.0, 35.0, 25.0, 15.0],
        "course_place_counts": [[10, 8, 5, 4, 2, 1]] * 6,
        "course_f_count": [0] * 6,
        "course_l_count": [0] * 6,
    }
    base.update(overrides)
    return base


def _six_boats() -> list[dict]:
    return [_boat(lane) for lane in range(1, 7)]


class TestComputeMatchIndex(unittest.TestCase):
    def test_formula(self) -> None:
        self.assertAlmostEqual(compute_match_index(4.0), 42.0, delta=1e-9)

    def test_capped_at_100(self) -> None:
        self.assertEqual(compute_match_index(10.0), 100)


class TestLaneRankScores(unittest.TestCase):
    def test_score_is_sum_of_rounded_contributions(self) -> None:
        boats = _six_boats()
        result = calc_lane_rank_scores(boats[1], boats, _config())
        contrib = result["contributions"]["first"]
        self.assertAlmostEqual(result["first"], sum(contrib.values()), delta=1e-9)

    def test_unknown_lane_uses_default_base_rates(self) -> None:
        boats = _six_boats()
        b7 = _boat(7)
        result = calc_lane_rank_scores(b7, boats + [b7], _config())
        # 既定 {"first":10,"second":15,"third":15} × base_rate_weight=1.0
        self.assertEqual(result["contributions"]["first"]["base_rate"], 10.0)
        self.assertEqual(result["contributions"]["second"]["base_rate"], 15.0)

    def test_avg_st_zero_treated_as_missing(self) -> None:
        boats = _six_boats()
        boats[2] = _boat(3, avg_st=0.0)
        result = calc_lane_rank_scores(boats[2], boats, _config())
        self.assertEqual(result["contributions"]["first"]["avg_st"], 0.0)

    def test_no_nyuko_gates_course_features(self) -> None:
        boats = _six_boats()
        boats[3] = _boat(4, course_nyuko=[30, 30, 30, 0, 30, 30])
        result = calc_lane_rank_scores(boats[3], boats, _config())
        c = result["contributions"]["first"]
        self.assertEqual(c["course_perf"], 0.0)
        self.assertEqual(c["other"], 0.0)
        self.assertEqual(c["f_risk"], 0.0)
        self.assertEqual(c["venue_factor"], 0.0)

    def test_venue_factor_contribution(self) -> None:
        provider = FakeVenueStatsProvider(
            course_factors={"住之江": {"factor": 1.10, "samples": 50}}
        )
        boats = _six_boats()
        with_v = calc_lane_rank_scores(
            boats[1], boats, _config(), venue_name="住之江", venue_stats=provider
        )
        without_v = calc_lane_rank_scores(boats[1], boats, _config())
        # factor 1.10 -> diff 0.10 × weight 10.0 = 1.0 (round3)
        self.assertAlmostEqual(
            with_v["contributions"]["first"]["venue_factor"], 1.0, delta=1e-9
        )
        self.assertEqual(without_v["contributions"]["first"]["venue_factor"], 0.0)

    def test_danger_penalty_only_lane1(self) -> None:
        cfg = _config()
        # dangerを確定値にする: soloのみ（win_rate_low常時成立 weight=30 -> danger=30）
        cfg["danger_score"]["solo_weights"] = {"win_rate_low": {"weight": 30.0}}
        cfg["danger_score"]["solo_thresholds"]["win_rate_low_abs"] = 99.0
        boats = _six_boats()
        b1 = calc_lane_rank_scores(boats[0], boats, cfg)
        b2 = calc_lane_rank_scores(boats[1], boats, cfg)
        # penalty_unit = 30/10 = 3.0、first weight=1.0 -> -3.0
        self.assertEqual(b1["contributions"]["first"]["danger_penalty"], -3.0)
        self.assertEqual(b1["contributions"]["second"]["danger_penalty"], -1.5)
        self.assertEqual(b1["contributions"]["third"]["danger_penalty"], -0.6)
        self.assertNotIn("danger_penalty", b2["contributions"]["first"])

    def test_floor_at_0_1(self) -> None:
        cfg = _config()
        cfg["danger_score"]["solo_weights"] = {"win_rate_low": {"weight": 100.0}}
        cfg["danger_score"]["solo_thresholds"]["win_rate_low_abs"] = 99.0
        # first weightを重くして大幅マイナスにする
        cfg["lane_rank_scores"]["first_place"]["danger_penalty_weight"] = 100.0
        boats = _six_boats()
        result = calc_lane_rank_scores(boats[0], boats, cfg)
        self.assertEqual(result["first"], 0.1)


class TestRankProbabilities(unittest.TestCase):
    def test_probs_sum_to_one(self) -> None:
        probs = calc_rank_probabilities(_six_boats(), None, _config())
        for key in ("first", "second", "third"):
            self.assertAlmostEqual(sum(probs[key].values()), 1.0, delta=1e-9)

    def test_lane6_suppressed_by_default(self) -> None:
        boats = _six_boats()
        cfg = _config()
        suppressed = calc_rank_probabilities(boats, None, cfg)
        allowed = calc_rank_probabilities(
            boats, {"match_index": 90, "upset_score": 0}, cfg
        )
        self.assertFalse(suppressed["lane6_first_allowed"])
        self.assertTrue(allowed["lane6_first_allowed"])
        self.assertLess(suppressed["first"][6], allowed["first"][6])

    def test_lane6_allowed_via_upset_score(self) -> None:
        probs = calc_rank_probabilities(
            _six_boats(), {"match_index": 0, "upset_score": 8.5}, _config()
        )
        self.assertTrue(probs["lane6_first_allowed"])


class TestRankIndex(unittest.TestCase):
    def test_index_is_rounded_cumulative_percent(self) -> None:
        boats = _six_boats()
        cfg = _config()
        probs = calc_rank_probabilities(boats, None, cfg)
        index = calc_rank_index(boats, None, cfg)
        lane = 2
        p1, p2, p3 = (probs[k][lane] for k in ("first", "second", "third"))
        self.assertEqual(index[lane]["top1"], round(p1 * 100, 1))
        self.assertEqual(index[lane]["top2"], round((p1 + p2) * 100, 1))
        self.assertEqual(index[lane]["top3"], round((p1 + p2 + p3) * 100, 1))

    def test_contributions_included_per_lane(self) -> None:
        index = calc_rank_index(_six_boats(), None, _config())
        self.assertIn("base_rate", index[1]["contributions"]["first"])
        self.assertIn("danger_penalty", index[1]["contributions"]["first"])
        self.assertNotIn("danger_penalty", index[2]["contributions"]["first"])


class TestSelectFeaturedBoats(unittest.TestCase):
    def _rank_index(self) -> dict[int, dict]:
        return {
            1: {"top1": 60.0, "top2": 75.0, "top3": 85.0},
            2: {"top1": 20.0, "top2": 45.0, "top3": 60.0},
            3: {"top1": 10.0, "top2": 30.0, "top3": 50.0},
            4: {"top1": 5.0, "top2": 20.0, "top3": 40.0},
            5: {"top1": 3.0, "top2": 12.0, "top3": 30.0},
            6: {"top1": 2.0, "top2": 8.0, "top3": 25.0},
        }

    def test_marks_assigned_in_composite_order(self) -> None:
        result = select_featured_boats(_six_boats(), self._rank_index(), 0.0, _config())
        self.assertEqual([x["lane"] for x in result], [1, 2, 3])
        self.assertEqual([x["mark"] for x in result], ["◎", "○", "▲"])
        # composite = 60*0.5 + 75*0.3 + 85*0.2 = 69.5
        self.assertEqual(result[0]["composite"], 69.5)

    def test_boat1_excluded_at_threshold(self) -> None:
        result = select_featured_boats(
            _six_boats(), self._rank_index(), 40.0, _config()
        )
        self.assertNotIn(1, [x["lane"] for x in result])
        self.assertEqual(result[0]["lane"], 2)

    def test_min_composite_filters_weak_boats(self) -> None:
        cfg = _config()
        cfg["featured_racers"]["min_composite_threshold"] = 30.0
        result = select_featured_boats(_six_boats(), self._rank_index(), 0.0, cfg)
        self.assertTrue(all(x["composite"] >= 30.0 for x in result))
        self.assertEqual([x["lane"] for x in result], [1, 2])

    def test_missing_lane_defaults_to_zero_index(self) -> None:
        index = self._rank_index()
        del index[6]
        result = select_featured_boats(_six_boats(), index, 0.0, _config())
        self.assertEqual(len(result), 3)  # 6号艇はcomposite 0で選外

    def test_max_count_limits(self) -> None:
        cfg = _config()
        cfg["featured_racers"]["max_count"] = 1
        result = select_featured_boats(_six_boats(), self._rank_index(), 0.0, cfg)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["mark"], "◎")

    def test_name_taken_from_boat(self) -> None:
        result = select_featured_boats(_six_boats(), self._rank_index(), 0.0, _config())
        self.assertEqual(result[0]["name"], "選手1")


if __name__ == "__main__":
    unittest.main()
