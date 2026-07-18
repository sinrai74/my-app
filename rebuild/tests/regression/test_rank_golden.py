"""
Rank/Featured Golden回帰: Golden入力100件からのrank_index/featured_boats再現
（Step4-4）。

Step4計画書 §5・C2/C6/C7/C8 に対応。Golden一式（golden4/ v1.1.0以降）と
freeze configフィクスチャが存在する環境で実行され、未生成環境ではskipする。

検証内容（移植元v4の呼び出し連鎖の再現）:
  1. core.upset.calculate_upset_score で upset_score を算出（Golden検証済み）
  2. ctx = {"match_index": compute_match_index(upset), "upset_score": upset}
     （移植元L536のrank_index_ctxと同一）
  3. calc_rank_index(boats, ctx, config, venue) を期待出力の rank_index と比較
     （top1/top2/top3は1e-6、contributionsは全キー・全値1e-6）
  4. select_featured_boats(boats, rank_index, danger_score, config) を
     期待出力の featured_boats と比較（lane/name/mark完全一致、
     composite/top1/top2/top3は1e-6）

失敗時は eval_id / feature / expected / actual を表示する（Should2形式）。
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from core.rank import calc_rank_index, compute_match_index, select_featured_boats
from core.upset import calculate_upset_score

GOLDEN4_DIR = Path("tests/regression/golden4")
MANIFEST = GOLDEN4_DIR / "manifest.json"
CONFIG_FIXTURE = GOLDEN4_DIR / "asahi_config_freeze.json"
CONFIG_SHA256 = "6a7862b8dc36006854d557fa8e2bfd12823433a0c51c5f092bb7f357b57520b3"

FLOAT_TOL = 1e-6


class _RecordedFactorProvider:
    def __init__(self, course_rentai_factors: dict[str, float]) -> None:
        self._factors = course_rentai_factors

    def get_venue_course_factor(self, venue_name: str, course: int) -> dict:
        return {"factor": self._factors[str(course)]}


@unittest.skipUnless(
    MANIFEST.exists() and CONFIG_FIXTURE.exists(),
    "Ver4 Golden or freeze config fixture not present.",
)
class TestRankGolden(unittest.TestCase):
    """Golden 100件でのrank_index/featured再現（100%一致が合否基準）。"""

    @classmethod
    def setUpClass(cls) -> None:
        digest = hashlib.sha256(CONFIG_FIXTURE.read_bytes()).hexdigest()
        if digest != CONFIG_SHA256:
            raise AssertionError(f"asahi_config_freeze.json modified: {digest}")
        cls.config = json.loads(CONFIG_FIXTURE.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_all_100_inputs_reproduce_rank_and_featured(self) -> None:
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
        recorded = race_input.get("danger_venue_factors")
        if recorded is None:
            self.skipTest("golden inputs are v1.0.0; regenerate with wrapper v1.1.0")

        boats = race_input["boats"]
        provider = _RecordedFactorProvider(recorded["course_rentai"])
        venue_name = race_input["venue"] or None

        # 移植元v4の連鎖: upset -> rank_index_ctx -> rank_index -> featured
        upset = calculate_upset_score(
            boats=boats,
            config=self.config,
            race_grade=race_input["race_grade"],
            venue_num=race_input["venue_num"],
            is_night=race_input["is_night"],
            venue_stats=provider,
        )
        ctx = {
            "match_index": compute_match_index(upset.upset_score),
            "upset_score": upset.upset_score,
        }
        rank_index = calc_rank_index(
            boats, ctx, self.config, venue_name=venue_name, venue_stats=provider
        )
        featured = select_featured_boats(
            boats, rank_index, upset.danger_score, self.config
        )

        self._compare_rank_index(eval_id, expected["rank_index"], rank_index)
        self._compare_featured(eval_id, expected["featured_boats"], featured)

    def _compare_rank_index(
        self, eval_id: str, exp_index: dict, act_index: dict[int, dict]
    ) -> None:
        # JSON化で期待側のlaneキーは文字列になっている
        act_by_str = {str(lane): v for lane, v in act_index.items()}
        self.assertEqual(
            set(act_by_str.keys()), set(exp_index.keys()),
            msg=f"eval_id={eval_id} feature=rank_index.lanes "
                f"expected={sorted(exp_index)} actual={sorted(act_by_str)}",
        )
        for lane, exp_lane in exp_index.items():
            act_lane = act_by_str[lane]
            for key in ("top1", "top2", "top3"):
                self.assertAlmostEqual(
                    act_lane[key], float(exp_lane[key]), delta=FLOAT_TOL,
                    msg=f"eval_id={eval_id} feature=rank_index[{lane}].{key} "
                        f"expected={exp_lane[key]!r} actual={act_lane[key]!r}",
                )
            exp_contrib = exp_lane.get("contributions", {})
            act_contrib = act_lane.get("contributions", {})
            self.assertEqual(
                set(act_contrib.keys()), set(exp_contrib.keys()),
                msg=f"eval_id={eval_id} feature=rank_index[{lane}].contributions.keys "
                    f"expected={sorted(exp_contrib)} actual={sorted(act_contrib)}",
            )
            for rank_key, exp_items in exp_contrib.items():
                act_items = act_contrib[rank_key]
                self.assertEqual(
                    set(act_items.keys()), set(exp_items.keys()),
                    msg=f"eval_id={eval_id} "
                        f"feature=rank_index[{lane}].contributions.{rank_key}.keys "
                        f"expected={sorted(exp_items)} actual={sorted(act_items)}",
                )
                for feat, exp_value in exp_items.items():
                    act_value = act_items[feat]
                    self.assertAlmostEqual(
                        float(act_value), float(exp_value), delta=FLOAT_TOL,
                        msg=f"eval_id={eval_id} "
                            f"feature=rank_index[{lane}].contributions."
                            f"{rank_key}.{feat} "
                            f"expected={exp_value!r} actual={act_value!r}",
                    )

    def _compare_featured(
        self, eval_id: str, exp_featured: list, act_featured: list
    ) -> None:
        self.assertEqual(
            len(act_featured), len(exp_featured),
            msg=f"eval_id={eval_id} feature=featured_boats.length "
                f"expected={exp_featured!r} actual={act_featured!r}",
        )
        for i, (exp_item, act_item) in enumerate(zip(exp_featured, act_featured)):
            self.assertEqual(
                set(act_item.keys()), set(exp_item.keys()),
                msg=f"eval_id={eval_id} feature=featured_boats[{i}].keys "
                    f"expected={sorted(exp_item)} actual={sorted(act_item)}",
            )
            for field, exp_value in exp_item.items():
                act_value = act_item[field]
                if isinstance(exp_value, (int, float)) and not isinstance(
                    exp_value, bool
                ):
                    self.assertAlmostEqual(
                        float(act_value), float(exp_value), delta=FLOAT_TOL,
                        msg=f"eval_id={eval_id} feature=featured_boats[{i}].{field} "
                            f"expected={exp_value!r} actual={act_value!r}",
                    )
                else:
                    self.assertEqual(
                        act_value, exp_value,
                        msg=f"eval_id={eval_id} feature=featured_boats[{i}].{field} "
                            f"expected={exp_value!r} actual={act_value!r}",
                    )


if __name__ == "__main__":
    unittest.main()
