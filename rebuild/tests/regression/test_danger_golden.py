"""
DangerScore Golden回帰: Golden入力100件からのdanger_score/breakdown再現（Step4-2）。

Step4計画書 §5・C2/C6/C7/C8 に対応。Golden一式（golden4/）とfreeze時点configの
フィクスチャが存在する環境で実行され、未生成環境ではskipする。

config: asahi_config_freeze.json（freeze基準SHA-256=6a7862b8...で固定。
テスト実行時にハッシュを検証し、改変を検出する）。

venue補正の注入について（v1.1.0で確定）:
  get_venue_course_factorは場×レーン依存（motor_history.csvの実測1着率÷全国平均）。
  Golden入力v1.1.0以降は、生成時に実際に使われたレーン別factorが
  danger_venue_factors として各入力JSONへ記録されており、本回帰はそれを
  そのままProviderへ注入する（生成条件の完全再現）。
  旧形式（v1.0.0）の入力にはこのフィールドが無いため、その場合は
  再生成を促すメッセージ付きでskipする。
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from core.danger import calc_danger_score

GOLDEN4_DIR = Path("tests/regression/golden4")
MANIFEST = GOLDEN4_DIR / "manifest.json"
CONFIG_FIXTURE = GOLDEN4_DIR / "asahi_config_freeze.json"
CONFIG_SHA256 = "6a7862b8dc36006854d557fa8e2bfd12823433a0c51c5f092bb7f357b57520b3"

FLOAT_TOL = 1e-6


class _RecordedFactorProvider:
    """Golden入力に記録された生成時レーン別factorをそのまま返すProvider。"""

    def __init__(self, course_rentai_factors: dict[str, float]) -> None:
        self._factors = course_rentai_factors

    def get_venue_course_factor(self, venue_name: str, course: int) -> dict:
        return {"factor": self._factors[str(course)]}


@unittest.skipUnless(
    MANIFEST.exists() and CONFIG_FIXTURE.exists(),
    "Ver4 Golden or freeze config fixture not present.",
)
class TestDangerGolden(unittest.TestCase):
    """Golden 100件でのdanger_score/breakdown再現（100%一致が合否基準）。"""

    @classmethod
    def setUpClass(cls) -> None:
        digest = hashlib.sha256(CONFIG_FIXTURE.read_bytes()).hexdigest()
        if digest != CONFIG_SHA256:
            raise AssertionError(
                f"asahi_config_freeze.json modified: sha256={digest} "
                f"(expected {CONFIG_SHA256})"
            )
        cls.config = json.loads(CONFIG_FIXTURE.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_all_100_inputs_reproduce_danger(self) -> None:
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
        boats = race_input["boats"]
        boat1 = next((b for b in boats if b.get("lane") == 1), None)
        venue_name = race_input["venue"] or None
        recorded = race_input.get("danger_venue_factors")
        if recorded is None:
            self.skipTest(
                "golden inputs are v1.0.0 (no danger_venue_factors). "
                "Regenerate with tools/golden_wrapper.py v1.1.0."
            )
        provider = _RecordedFactorProvider(recorded["course_rentai"])
        # ポート設計の前提検証: venue_unfavorable用のlane1 factor（cfgなし呼び出し）が
        # course_rentai用lane1（cfg付き）と一致すること。不一致なら本ポートの
        # 単一Provider設計では表現できないため、明示的に失敗させて再検討する。
        self.assertAlmostEqual(
            float(recorded["venue_unfavorable"]),
            float(recorded["course_rentai"]["1"]),
            delta=FLOAT_TOL,
            msg=f"eval_id={eval_id} feature=venue_factor_lane1_variants "
                f"expected(cfg付き)={recorded['course_rentai']['1']!r} "
                f"actual(cfgなし)={recorded['venue_unfavorable']!r} "
                "-> 単一Provider設計の見直しが必要",
        )

        score, breakdown = calc_danger_score(
            boat1, boats, self.config, venue_name=venue_name, venue_stats=provider
        )

        exp_score = expected["danger_score"]
        self.assertAlmostEqual(
            score, float(exp_score), delta=FLOAT_TOL,
            msg=f"eval_id={eval_id} feature=danger_score "
                f"expected={exp_score!r} actual={score!r}",
        )

        exp_breakdown = expected["danger_breakdown"]
        self.assertEqual(
            set(breakdown.keys()), set(exp_breakdown.keys()),
            msg=f"eval_id={eval_id} feature=danger_breakdown.keys "
                f"expected={sorted(exp_breakdown)} actual={sorted(breakdown)}",
        )
        for key, exp_item in exp_breakdown.items():
            act_item = breakdown[key]
            for field, exp_value in exp_item.items():
                act_value = act_item.get(field)
                if isinstance(exp_value, (int, float)) and not isinstance(exp_value, bool):
                    self.assertAlmostEqual(
                        float(act_value), float(exp_value), delta=FLOAT_TOL,
                        msg=f"eval_id={eval_id} feature=danger_breakdown.{key}.{field} "
                            f"expected={exp_value!r} actual={act_value!r}",
                    )
                else:
                    self.assertEqual(
                        act_value, exp_value,
                        msg=f"eval_id={eval_id} feature=danger_breakdown.{key}.{field} "
                            f"expected={exp_value!r} actual={act_value!r}",
                    )


if __name__ == "__main__":
    unittest.main()
