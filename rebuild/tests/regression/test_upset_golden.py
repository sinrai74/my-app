"""
UpsetScore Golden回帰: Golden入力100件からのupset_score/detail再現（Step4-3）。

Step4計画書 §5・C2/C6/C7/C8 に対応。Golden一式（golden4/ v1.1.0以降）と
freeze configフィクスチャが存在する環境で実行され、未生成環境ではskipする。

検証内容:
- upset_score: 期待出力のトップレベル値と1e-6一致
- upset_detail のコア部（荒れ確率/1号艇確率/1号艇危険度/最有力/レース種別/
  モデルVersion/等級フィルタ）: 文字列の完全一致（f-string書式まで再現）
- danger_score/danger_breakdown: 内部呼び出し結果が期待出力と一致
  （upset->danger内部連鎖の検証。単独のdanger Golden回帰と二重の担保）
※ upset_detail内の rank_index/featured_boats/_features 等はStep4-4/4-5の
  移植範囲であり、本回帰の比較対象外（各Stepで追加する）。
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from core.upset import calculate_upset_score

GOLDEN4_DIR = Path("tests/regression/golden4")
MANIFEST = GOLDEN4_DIR / "manifest.json"
CONFIG_FIXTURE = GOLDEN4_DIR / "asahi_config_freeze.json"
CONFIG_SHA256 = "6a7862b8dc36006854d557fa8e2bfd12823433a0c51c5f092bb7f357b57520b3"

FLOAT_TOL = 1e-6

# upset本体（Step4-3範囲）が生成するdetailキー
CORE_DETAIL_KEYS = (
    "荒れ確率",
    "1号艇確率",
    "1号艇危険度",
    "最有力",
    "レース種別",
    "モデルVersion",
    "等級フィルタ",  # 条件付きキー（不在同士も一致とみなす）
)


class _RecordedFactorProvider:
    def __init__(self, course_rentai_factors: dict[str, float]) -> None:
        self._factors = course_rentai_factors

    def get_venue_course_factor(self, venue_name: str, course: int) -> dict:
        return {"factor": self._factors[str(course)]}


@unittest.skipUnless(
    MANIFEST.exists() and CONFIG_FIXTURE.exists(),
    "Ver4 Golden or freeze config fixture not present.",
)
class TestUpsetGolden(unittest.TestCase):
    """Golden 100件でのupset再現（100%一致が合否基準）。"""

    @classmethod
    def setUpClass(cls) -> None:
        digest = hashlib.sha256(CONFIG_FIXTURE.read_bytes()).hexdigest()
        if digest != CONFIG_SHA256:
            raise AssertionError(f"asahi_config_freeze.json modified: {digest}")
        cls.config = json.loads(CONFIG_FIXTURE.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_all_100_inputs_reproduce_upset(self) -> None:
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

        result = calculate_upset_score(
            boats=race_input["boats"],
            config=self.config,
            race_grade=race_input["race_grade"],
            venue_num=race_input["venue_num"],
            is_night=race_input["is_night"],
            venue_stats=_RecordedFactorProvider(recorded["course_rentai"]),
        )

        exp_score = expected["upset_score"]
        self.assertAlmostEqual(
            result.upset_score, float(exp_score), delta=FLOAT_TOL,
            msg=f"eval_id={eval_id} feature=upset_score "
                f"expected={exp_score!r} actual={result.upset_score!r}",
        )

        exp_detail = expected["upset_detail"]
        exp_core = {k: exp_detail.get(k) for k in CORE_DETAIL_KEYS}
        act_core = {k: result.detail.get(k) for k in CORE_DETAIL_KEYS}
        # Should2: 失敗時はeval_idと期待/実際のdetail全体を表示する
        self.assertEqual(
            act_core, exp_core,
            msg=f"eval_id={eval_id}\n"
                f"expected detail={exp_core!r}\n"
                f"actual detail={act_core!r}",
        )

        # upset->danger 内部連鎖の一致（二重担保）
        self.assertAlmostEqual(
            result.danger_score, float(expected["danger_score"]), delta=FLOAT_TOL,
            msg=f"eval_id={eval_id} feature=danger_score(via upset) "
                f"expected={expected['danger_score']!r} actual={result.danger_score!r}",
        )
        self.assertEqual(
            set(result.danger_breakdown.keys()),
            set(expected["danger_breakdown"].keys()),
            msg=f"eval_id={eval_id} feature=danger_breakdown.keys(via upset)",
        )

        # venue_num->場名の導出が入力の場名と一致すること（定数転記の実データ検証）
        self.assertEqual(
            result.venue_name, race_input["venue"] or None,
            msg=f"eval_id={eval_id} feature=venue_name "
                f"expected={race_input['venue']!r} actual={result.venue_name!r}",
        )


if __name__ == "__main__":
    unittest.main()
