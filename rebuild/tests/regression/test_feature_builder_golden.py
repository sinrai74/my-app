"""
FeatureBuilder Golden回帰: Golden入力100件からのFeatureSet再現検証（Step4-1）。

Step4計画書 §5・C6/C7/C8 に対応。Golden一式（golden4/）が存在する環境
（運用者Windows等）で実行され、未生成環境ではskipする。

検証内容: 各Golden入力について DefaultFeatureBuilder.build() を実行し、
期待出力（build_race_evaluation_v4のupset_detail["_features"]）と比較する。
- 数値特徴: 1e-6許容で一致
- レガシーの空文字/None（未算出）: FeatureSetでは「キー欠損＋missing_keys記録」
  として表現されていること
- venue_water_type: 期待側のtypeキーとFeatureSetのコード値の対応一致

venue統計（水面タイプ・場補正）の扱い:
  これらはL1の外部入力（現行x_venue_statsの算出値）であり、FeatureBuilderの
  責務は「受け取った値の配置とコード変換」まで。よって本回帰ではProviderへ
  期待出力に記録された値（water_type / venue_factor_1c）を注入し、
  boats由来の計算特徴（ability_trend・course系レート等）の再現を主検証とする。
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from features.feature_builder import (
    WATER_TYPE_TYPE_TO_CODE,
    DefaultFeatureBuilder,
    FeatureInputs,
)
from models.race import Race
from tests.fakes import FakeVenueStatsProvider

GOLDEN4_DIR = Path("tests/regression/golden4")
MANIFEST = GOLDEN4_DIR / "manifest.json"

FLOAT_TOL = 1e-6

# 期待_featuresキー -> FeatureSet boat_features[1] キー
EXPECTED_TO_BOAT_KEY = {
    "win_rate": "win_rate",
    "motor": "motor_rate2",
    "avg_st": "avg_st",
    "course_st_1c": "course_st_1c",
    "course_rank_1c": "course_rank_1c",
    "ability_trend": "ability_trend",
    "course_f_rate_1c": "course_f_rate_1c",
    "course_l_rate_1c": "course_l_rate_1c",
    "course_rentai2_1c": "course_rentai2_1c",
    "course_sample_confidence": "course_sample_confidence",
}


def _is_legacy_missing(value) -> bool:
    return value is None or value == ""


@unittest.skipUnless(
    MANIFEST.exists(),
    "Ver4 Golden not generated yet (run tools/golden_wrapper.py first).",
)
class TestFeatureBuilderGolden(unittest.TestCase):
    """Golden 100件でのFeatureSet再現（100%一致が合否基準）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_all_100_inputs_reproduce_features(self) -> None:
        for entry in self.manifest["entries"]:
            eval_id = entry["eval_id"]
            with self.subTest(eval_id=eval_id):
                self._verify_one(eval_id)

    def _verify_one(self, eval_id: str) -> None:
        race_input = json.loads(
            (GOLDEN4_DIR / "inputs" / f"{eval_id}.json").read_text(encoding="utf-8")
        )
        expected = json.loads(
            (GOLDEN4_DIR / "expected" / f"{eval_id}.json").read_text(encoding="utf-8")
        )
        expected_features = expected["upset_detail"]["_features"]

        venue_name = race_input["venue"] or ""
        provider = FakeVenueStatsProvider(
            water_types={venue_name: expected["water_type"]},
            course_factors={venue_name: expected["venue_factor_1c"]},
        )
        race = Race(
            race_date=eval_id.split("_")[0],
            venue_num=race_input["venue_num"],
            venue_name=venue_name,
            race_number=int(eval_id.split("_")[2]),
            close_time="",
            is_night=race_input["is_night"],
            entries=(),
        )
        feature_set = DefaultFeatureBuilder(provider).build(
            race,
            FeatureInputs(boats=tuple(race_input["boats"])),
            built_at="golden",
        )

        boat1 = feature_set.boat_features[1]
        for exp_key, fs_key in EXPECTED_TO_BOAT_KEY.items():
            exp_value = expected_features.get(exp_key)
            if _is_legacy_missing(exp_value):
                self.assertNotIn(
                    fs_key, boat1, f"{eval_id}: {fs_key} should be missing"
                )
                self.assertIn(fs_key, feature_set.missing_keys, f"{eval_id}: {fs_key}")
            else:
                self.assertIn(
                    fs_key, boat1,
                    f"eval_id={eval_id} feature={fs_key} "
                    f"expected={exp_value!r} actual=<missing>",
                )
                self.assertAlmostEqual(
                    boat1[fs_key], float(exp_value), delta=FLOAT_TOL,
                    msg=(
                        f"eval_id={eval_id} feature={fs_key} "
                        f"expected={exp_value!r} actual={boat1[fs_key]!r}"
                    ),
                )

        # venue_factor（race_features側）
        exp_factor = expected_features.get("venue_factor")
        if _is_legacy_missing(exp_factor):
            self.assertIn("venue_factor", feature_set.missing_keys, eval_id)
        else:
            actual_factor = feature_set.race_features.get("venue_factor")
            self.assertAlmostEqual(
                actual_factor, float(exp_factor), delta=FLOAT_TOL,
                msg=(
                    f"eval_id={eval_id} feature=venue_factor "
                    f"expected={exp_factor!r} actual={actual_factor!r}"
                ),
            )

        # venue_water_type: 期待側のtypeキー -> コード対応
        water_type_key = str(expected["water_type"].get("type", ""))
        if water_type_key in WATER_TYPE_TYPE_TO_CODE:
            self.assertEqual(
                feature_set.race_features.get("venue_water_type_code"),
                WATER_TYPE_TYPE_TO_CODE[water_type_key],
                msg=f"{eval_id}: water type code",
            )
        else:
            self.assertIn("venue_water_type_code", feature_set.missing_keys, eval_id)

        # ラベル整合の追加確認: 期待_featuresのラベルとwater_typeのラベルが一致
        self.assertEqual(
            expected_features.get("venue_water_type", ""),
            expected["water_type"].get("label", ""),
            msg=f"{eval_id}: label consistency (golden self-check)",
        )


if __name__ == "__main__":
    unittest.main()
