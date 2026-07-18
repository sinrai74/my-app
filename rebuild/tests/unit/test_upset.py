"""
calculate_upset_score / calc_boat_score（core/upset.py）の単体テスト。

Step4計画書 C8（Golden回帰と分離した純関数検証）に対応。
移植元（x_asahi_scoring.py L361-L536）の各効果・ゲート・クランプ・フィルタを
小さな合成configと合成艇データで個別に検証する。
"""

from __future__ import annotations

import unittest

from core.upset import (
    GRADE_EFFECTS,
    VENUE_NAMES,
    VENUE_UPSET_FACTOR,
    UpsetResult,
    _add_effect,
    _logit,
    _sigmoid,
    calc_boat_score,
    calculate_upset_score,
)


def _config(**up_overrides) -> dict:
    cfg = {
        "model_version": "test-model",
        "boat_relative_score": {
            "weights": {
                "avg_st_component": {"weight": 1.0},
                "course_st_component": {"weight": 1.0},
                "motor_component": {"weight": 1.0},
                "win_rate_component": {"weight": 1.0},
                "lane_weight": {"1": 1.5, "2": 0.5, "3": 0.2,
                                "4": 0.0, "5": -0.2, "6": -0.4},
            }
        },
        "upset_prob": {
            "night_factor": -0.08,
            "effects": {
                "st_risk_avg_st": {"value": 0.20},
                "st_risk_course_st": {"value": 0.15},
                "win_rate_risk_1": {"value": 0.30},
                "win_rate_risk_2": {"value": 0.15},
                "motor_risk": {"value": 0.15},
                "class_risk": {"value": 0.20},
                "boat2_strength_mult": {"value": 2.0},
                "class_gap_effect": {"value": 0.25},
            },
        },
        "danger_score": {
            "total_scale": 100,
            "relative_weights": {},
            "solo_weights": {},
            "solo_thresholds": {
                "win_rate_low_abs": 5.0, "local_win_low_abs": 5.0,
                "motor_bad_abs": 33.0, "avg_st_slow_abs": 0.17,
                "course1_place_low_abs": 45.0, "f_risk_rate_abs": 8.0,
                "venue_unfavorable_factor_abs": 0.85,
            },
        },
    }
    cfg["upset_prob"].update(up_overrides)
    return cfg


def _boat(lane: int, **overrides) -> dict:
    base = {
        "lane": lane,
        "racer_class": "B1",  # 既定はB1（A1/A2の等級フィルタで
        # スコアが0に張り付き比較テストが無効化されるのを防ぐ。フィルタ自体は
        # TestFiltersAndDetailで明示的に検証する）
        "win_rate": 6.0,
        "local_win": 6.0,
        "avg_st": 0.15,
        "motor": 40.0,
        "ability_curr": 75.0,
        "ability_prev": 74.0,
        "course_nyuko": [40, 30, 20, 10, 5, 2],
        "course_st": [0.15, 0.16, 0.16, 0.17, 0.18, 0.20],
        "course_rank": [2.4, 2.7, 2.7, 3.2, 3.3, 4.0],
        "course_win_rate": [55.0, 40.0, 30.0, 20.0, 10.0, 5.0],
        "course_place_rate": [70.0, 55.0, 45.0, 35.0, 20.0, 10.0],
        "course_place_counts": [[20, 8, 4, 4, 2, 2]] * 6,
        "course_f_count": [0] * 6,
        "course_l_count": [0] * 6,
    }
    base.update(overrides)
    return base


def _six_boats(**boat1_overrides) -> list[dict]:
    """6艇の合成レース。他艇はboat1に近い強さに設定する。

    boat1が強すぎるとboat1_prob>0.65の安心レースゼロ化が発動し、
    効果比較テストが0.0同士の比較になって無効化されるため
    （実測: 他艇win5.0/motor35/st0.17ではprob=0.871）、
    他艇をwin5.8/motor39/st0.16としてprob≒0.59に調整している。
    """
    boats = [_boat(1, **boat1_overrides)]
    for lane in range(2, 7):
        boats.append(
            _boat(lane, win_rate=5.8, motor=39.0, avg_st=0.16,
                  course_st=[0.16] * 6)
        )
    return boats


class TestMathHelpers(unittest.TestCase):
    def test_logit_clamps_to_0001_0999(self) -> None:
        self.assertEqual(_logit(0.0), _logit(0.001))
        self.assertEqual(_logit(1.0), _logit(0.999))

    def test_sigmoid_inverts_logit(self) -> None:
        self.assertAlmostEqual(_sigmoid(_logit(0.3)), 0.3, delta=1e-9)

    def test_add_effect_zero_is_identity_within_clamp(self) -> None:
        self.assertAlmostEqual(_add_effect(0.4, 0.0), 0.4, delta=1e-9)


class TestCalcBoatScore(unittest.TestCase):
    def test_lane_weight_applied(self) -> None:
        cfg = _config()
        b1 = _boat(1)
        b6 = _boat(6)
        boats = [b1, b6]
        s1 = calc_boat_score(b1, boats, cfg)
        s6 = calc_boat_score(b6, boats, cfg)
        # 同一能力なら枠の基礎優位差（1.5 - (-0.4)）とコースST差のみ
        self.assertGreater(s1, s6)

    def test_avg_st_zero_is_skipped(self) -> None:
        """avg_st=0はtruthyゲートでスキップされる（移植元の挙動保存）。"""
        cfg = _config()
        b = _boat(4, avg_st=0.0)
        b_slow = _boat(4, avg_st=0.30)  # 大幅減点されるはず
        score_zero = calc_boat_score(b, [b], cfg)
        score_slow = calc_boat_score(b_slow, [b_slow], cfg)
        self.assertGreater(score_zero, score_slow)

    def test_motor_relative_to_average(self) -> None:
        cfg = _config()
        strong = _boat(3, motor=50.0)
        weak = _boat(4, motor=30.0)
        boats = [strong, weak]
        self.assertGreater(
            calc_boat_score(strong, boats, cfg), calc_boat_score(weak, boats, cfg)
        )

    def test_softmax_probs_sum_to_one(self) -> None:
        result = calculate_upset_score(_six_boats(), _config())
        self.assertAlmostEqual(sum(result.lane_probs.values()), 1.0, delta=1e-9)


class TestRiskEffects(unittest.TestCase):
    def _score_with(self, **boat1_overrides) -> UpsetResult:
        return calculate_upset_score(_six_boats(**boat1_overrides), _config())

    def test_st_risk_increases_upset(self) -> None:
        slow = self._score_with(avg_st=0.18, course_st=[0.17] * 6)
        fast = self._score_with(avg_st=0.15, course_st=[0.15] * 6)
        self.assertGreater(slow.upset_score, fast.upset_score)

    def test_win_rate_risk_thresholds(self) -> None:
        risk1 = self._score_with(win_rate=4.9)
        risk2 = self._score_with(win_rate=5.4)
        no_risk = self._score_with(win_rate=6.0)
        self.assertGreater(risk1.upset_score, risk2.upset_score)
        self.assertGreater(risk2.upset_score, no_risk.upset_score)

    def test_class_gap_effect_requires_a1_rival(self) -> None:
        boats_with_a1 = _six_boats(racer_class="B1")
        boats_with_a1[3]["racer_class"] = "A1"
        boats_without = _six_boats(racer_class="B1")
        with_gap = calculate_upset_score(boats_with_a1, _config())
        without_gap = calculate_upset_score(boats_without, _config())
        self.assertGreater(with_gap.upset_score, without_gap.upset_score)

    def test_night_factor_reduces_upset(self) -> None:
        boats = _six_boats()
        day = calculate_upset_score(boats, _config(), venue_num=5)  # factor 0.0の場
        night = calculate_upset_score(boats, _config(), venue_num=5, is_night=True)
        self.assertLess(night.upset_score, day.upset_score)

    def test_venue_upset_factor_applied(self) -> None:
        boats = _six_boats()
        edogawa = calculate_upset_score(boats, _config(), venue_num=3)   # +0.15
        omura = calculate_upset_score(boats, _config(), venue_num=24)    # -0.15
        self.assertGreater(edogawa.upset_score, omura.upset_score)

    def test_grade_effects_reduce_upset(self) -> None:
        boats = _six_boats()
        general = calculate_upset_score(boats, _config(), race_grade=0)
        sg = calculate_upset_score(boats, _config(), race_grade=4)
        self.assertLess(sg.upset_score, general.upset_score)
        self.assertEqual(sg.detail["レース種別"], "SG")


class TestFiltersAndDetail(unittest.TestCase):
    def test_a1_filter_zeroes_score(self) -> None:
        result = calculate_upset_score(_six_boats(racer_class="A1"), _config())
        self.assertEqual(result.upset_score, 0.0)
        self.assertEqual(result.detail["等級フィルタ"], "1号艇A1 → スキップ")

    def test_a2_filter_subtracts_1_5_with_floor(self) -> None:
        result = calculate_upset_score(_six_boats(racer_class="A2"), _config())
        self.assertIn("等級フィルタ", result.detail)
        self.assertEqual(result.detail["等級フィルタ"], "1号艇A2 → -1.5")
        self.assertGreaterEqual(result.upset_score, 0.0)

    def test_b1_has_no_filter_note(self) -> None:
        result = calculate_upset_score(_six_boats(racer_class="B1"), _config())
        self.assertNotIn("等級フィルタ", result.detail)

    def test_safe_race_zeroed_when_boat1_dominant(self) -> None:
        """boat1_prob>0.65かつ最有力が下回るなら0.0（B級で等級フィルタ回避）。"""
        boats = _six_boats(racer_class="B1", win_rate=9.9, motor=70.0, avg_st=0.10)
        for b in boats[1:]:
            b.update(win_rate=2.0, motor=25.0, avg_st=0.25,
                     course_st=[0.25] * 6, racer_class="B2")
        result = calculate_upset_score(boats, _config())
        self.assertGreater(result.boat1_prob, 0.65)
        self.assertEqual(result.upset_score, 0.0)

    def test_detail_formats_are_exact(self) -> None:
        result = calculate_upset_score(_six_boats(), _config())
        self.assertRegex(result.detail["荒れ確率"], r"^\d+\.\d%$")
        self.assertRegex(result.detail["1号艇確率"], r"^\d+\.\d%\(rank\d\)$")
        self.assertRegex(result.detail["最有力"], r"^\d号艇\(\d+\.\d%\)$")
        self.assertEqual(result.detail["モデルVersion"], "test-model")

    def test_target_lanes_are_top2_others(self) -> None:
        result = calculate_upset_score(_six_boats(), _config())
        self.assertEqual(len(result.target_lanes), 2)
        self.assertNotIn(1, result.target_lanes)

    def test_constants_transcribed(self) -> None:
        """機械転記定数のスポット検証（件数と代表値）。"""
        self.assertEqual(len(VENUE_NAMES), 24)
        self.assertEqual(VENUE_NAMES[12], "住之江")
        self.assertEqual(VENUE_UPSET_FACTOR[3], 0.15)
        self.assertEqual(VENUE_UPSET_FACTOR[24], -0.15)
        self.assertEqual(GRADE_EFFECTS[4], -0.5)


class TestDangerBoost(unittest.TestCase):
    def test_danger_boost_via_config_weights(self) -> None:
        """danger>=60でupsetが1.30倍（9.5上限）へブーストされること。

        danger_scoreをconfig側の相対重みで制御して閾値をまたがせる
        （モック不使用・実経路での検証）。
        """
        cfg = _config()
        cfg["danger_score"]["relative_weights"] = {
            "win_rate": {"per_boat": 20.0, "max_weight": 100.0},
        }
        boats = _six_boats(racer_class="B1", win_rate=3.0)  # 全艇に劣後→danger=100
        boosted = calculate_upset_score(boats, cfg)
        self.assertGreaterEqual(boosted.danger_score, 60)

        cfg_low = _config()  # 重みなし→danger=0
        plain = calculate_upset_score(boats, cfg_low)
        self.assertEqual(plain.danger_score, 0.0)
        # ブースト有無で結果が異なる（かつ9.5上限内）
        self.assertLessEqual(boosted.upset_score, 9.5)
        self.assertGreater(boosted.upset_score, plain.upset_score - 1e-9)


if __name__ == "__main__":
    unittest.main()
