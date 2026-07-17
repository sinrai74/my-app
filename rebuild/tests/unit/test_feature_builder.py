"""
DefaultFeatureBuilder（features/feature_builder.py）の単体テスト。

Step4計画書 C8（純関数単位テストをGolden回帰と分離）に対応。
移植元（x_asahi_scoring.py L540-590）のゲート条件・丸め・欠損挙動を
小さな合成入力で個別に検証する。標準ライブラリ unittest のみを使用する。
"""

from __future__ import annotations

import unittest

from features.feature_builder import (
    BOAT1_FEATURE_KEYS,
    RACE_FEATURE_KEYS,
    WATER_TYPE_TYPE_TO_CODE,
    DefaultFeatureBuilder,
    FeatureInputs,
    create_feature_builder,
)
from models.race import Race
from tests.fakes import FakePredictor, FakeVenueStatsProvider


def _race(venue_name: str = "住之江") -> Race:
    return Race(
        race_date="20260704",
        venue_num=12,
        venue_name=venue_name,
        race_number=5,
        close_time="",
        is_night=False,
        entries=(),
    )


def _boat1(**overrides) -> dict:
    base = {
        "lane": 1,
        "name": "テスト太郎",
        "racer_class": "A1",
        "win_rate": 6.50,
        "avg_st": 0.16,
        "motor": 38.5,
        "ability_curr": 75.0,
        "ability_prev": 74.0,
        "course_nyuko": [46, 28, 24, 17, 9, 0],
        "course_st": [0.15, 0.15, 0.16, 0.15, 0.20, 0.0],
        "course_rank": [2.40, 2.70, 2.70, 3.20, 3.30, 0.0],
        "course_f_count": [1, 0, 0, 0, 0, 0],
        "course_l_count": [0, 0, 0, 0, 0, 0],
        "course_place_counts": [[20, 14, 5, 4, 2, 1]] * 6,
    }
    base.update(overrides)
    return base


def _builder(provider: FakeVenueStatsProvider | None = None) -> DefaultFeatureBuilder:
    provider = provider or FakeVenueStatsProvider(
        water_types={"住之江": {"type": "in_favorable", "label": "イン有利水面"}},
        course_factors={"住之江": {"factor": 1.02, "samples": 120}},
    )
    return DefaultFeatureBuilder(provider)


class TestDefaultFeatureBuilderValues(unittest.TestCase):
    def test_full_input_produces_all_features(self) -> None:
        fs = _builder().build(_race(), FeatureInputs(boats=(_boat1(),)), built_at="t")
        b1 = fs.boat_features[1]
        self.assertEqual(b1["win_rate"], 6.50)
        self.assertEqual(b1["motor_rate2"], 38.5)
        self.assertEqual(b1["avg_st"], 0.16)
        self.assertEqual(b1["course_st_1c"], 0.15)
        self.assertEqual(b1["course_rank_1c"], 2.40)
        self.assertEqual(b1["ability_trend"], 1.0)  # round(75.0-74.0, 2)
        # F率: 1/46*100 = 2.173... -> round(...,1) = 2.2
        self.assertEqual(b1["course_f_rate_1c"], 2.2)
        self.assertEqual(b1["course_l_rate_1c"], 0.0)
        # 2連対率: (20+14)/46*100 = 73.913 -> 73.9
        self.assertEqual(b1["course_rentai2_1c"], 73.9)
        self.assertEqual(b1["course_sample_confidence"], 120.0)
        self.assertEqual(fs.race_features["venue_factor"], 1.02)
        self.assertEqual(fs.race_features["venue_water_type_code"], 2.0)
        self.assertEqual(fs.eval_id, "20260704_12_05")

    def test_racer_class_score_is_always_missing(self) -> None:
        """級別は文字列であり数値化仕様がfreeze時点に無いため常に欠損（C2）。"""
        fs = _builder().build(_race(), FeatureInputs(boats=(_boat1(),)), built_at="t")
        self.assertNotIn("racer_class_score", fs.boat_features[1])
        self.assertIn("racer_class_score", fs.missing_keys)


class TestGatingQuirks(unittest.TestCase):
    """移植元のゲート条件（0値がfalsy扱いになる挙動を含む）の保存を検証する。"""

    def test_nyuko_zero_gates_course_features(self) -> None:
        boat = _boat1(course_nyuko=[0, 28, 24, 17, 9, 0])
        fs = _builder().build(_race(), FeatureInputs(boats=(boat,)), built_at="t")
        for key in ("course_st_1c", "course_rank_1c", "course_f_rate_1c",
                    "course_l_rate_1c", "course_rentai2_1c"):
            self.assertNotIn(key, fs.boat_features[1], key)
            self.assertIn(key, fs.missing_keys, key)

    def test_ability_zero_becomes_missing(self) -> None:
        """移植元は `if curr and prev` のため、0.0は欠損になる（挙動保存）。"""
        boat = _boat1(ability_curr=0.0, ability_prev=74.0)
        fs = _builder().build(_race(), FeatureInputs(boats=(boat,)), built_at="t")
        self.assertNotIn("ability_trend", fs.boat_features[1])
        self.assertIn("ability_trend", fs.missing_keys)

    def test_place_counts_all_zero_gates_rentai2(self) -> None:
        boat = _boat1(course_place_counts=[[0, 0, 0, 0, 0, 0]] * 6)
        fs = _builder().build(_race(), FeatureInputs(boats=(boat,)), built_at="t")
        self.assertNotIn("course_rentai2_1c", fs.boat_features[1])
        # F率/L率はcountsリスト自体がtruthyなら算出される（独立ゲート）
        self.assertIn("course_f_rate_1c", fs.boat_features[1])

    def test_missing_boat1_yields_missing_basics(self) -> None:
        boat2 = _boat1(lane=2)
        fs = _builder().build(_race(), FeatureInputs(boats=(boat2,)), built_at="t")
        for key in ("win_rate", "motor_rate2", "avg_st"):
            self.assertIn(key, fs.missing_keys)

    def test_empty_venue_name_gates_venue_features(self) -> None:
        fs = _builder().build(_race(venue_name=""), FeatureInputs(boats=(_boat1(),)), built_at="t")
        self.assertIn("venue_factor", fs.missing_keys)
        self.assertIn("venue_water_type_code", fs.missing_keys)
        self.assertIn("course_sample_confidence", fs.missing_keys)

    def test_unknown_water_type_key_becomes_missing(self) -> None:
        provider = FakeVenueStatsProvider(
            water_types={"住之江": {"type": "unknown", "label": "?"}},
            course_factors={"住之江": {"factor": 1.0, "samples": 0}},
        )
        fs = _builder(provider).build(_race(), FeatureInputs(boats=(_boat1(),)), built_at="t")
        self.assertIn("venue_water_type_code", fs.missing_keys)


class TestLayerConsistency(unittest.TestCase):
    def test_water_type_table_matches_mapper(self) -> None:
        """features層とstorage層の変換表が同一内容であること（機械同期検証）。"""
        from storage.mappers.hit_record_mapper import WATER_TYPE_LABEL_TO_CODE

        self.assertEqual(WATER_TYPE_TYPE_TO_CODE, WATER_TYPE_LABEL_TO_CODE)

    def test_feature_keys_match_mapper_bijection(self) -> None:
        """BOAT1_FEATURE_KEYSがMapperの全単射キー集合と一致すること。"""
        from storage.mappers.hit_record_mapper import FEAT_COLUMN_TO_FEATURE_KEY

        self.assertEqual(
            set(BOAT1_FEATURE_KEYS), set(FEAT_COLUMN_TO_FEATURE_KEY.values())
        )
        self.assertEqual(set(RACE_FEATURE_KEYS), {"venue_factor", "venue_water_type_code"})

    def test_builder_output_roundtrips_through_serializer(self) -> None:
        """生成したFeatureSetがto_dict/from_dictで完全往復すること。"""
        from models.evaluation import FeatureSet

        fs = _builder().build(_race(), FeatureInputs(boats=(_boat1(),)), built_at="t")
        self.assertEqual(FeatureSet.from_dict(fs.to_dict()), fs)


class TestFakePredictor(unittest.TestCase):
    def test_returns_fixed_prediction_and_records_calls(self) -> None:
        predictor = FakePredictor({"20260704_12_05": {1: 0.42, 2: 0.18}})
        self.assertEqual(predictor("20260704_12_05"), {1: 0.42, 2: 0.18})
        self.assertEqual(predictor.calls, ["20260704_12_05"])

    def test_unknown_eval_id_raises(self) -> None:
        predictor = FakePredictor({})
        with self.assertRaises(KeyError):
            predictor("20990101_01_01")


class TestDiWiring(unittest.TestCase):
    def test_factory_returns_default_builder(self) -> None:
        builder = create_feature_builder(FakeVenueStatsProvider())
        self.assertIsInstance(builder, DefaultFeatureBuilder)


if __name__ == "__main__":
    unittest.main()
