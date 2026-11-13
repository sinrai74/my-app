"""正式G3（B案）：実際の Legacy で「購入」になる合成入力の、候補の生成から BuyAssessment までの値の流れ。

試験ケース（設計）
  P1 実際の `_evaluate_bets`・実際の apply_buyscore（写し取りの既定の対象）・実際の `_generate_patterns` を、
     evaluate_race_g3 を通して1回呼ぶ。期待：
       - outcome = compared、同一性の条件がすべて ok
       - Legacy は購入（passthrough=False）、Prediction は buyscore 最高の候補（ずれなし）
       - 写し取った候補（呼び出し時）に Legacy の buyscore・kelly が無い
       - Rebuild の4項目（buyscore・kelly・skip_reason・investment_type）が Legacy の値と完全一致
       - Rebuild は、Legacy が apply_buyscore の中で読み込んだ設定（cfg）で評価している
  （未確認として残すもの：ML 加点が起きる購入のレース。Legacy の boats の実物の組み立てが要るため、本テストでは扱わない）

- ネットワークを使わない。GITHUB_TOKEN が設定されていれば実行しない（Release に触れないため）。
- 一時フォルダを作業ディレクトリにして実行する（Legacy が設定・ログ・成績をローカルに読み書きするため）。
"""

from __future__ import annotations

import itertools
import os
import tempfile
import unittest

from actions import g3_entrypoint as G
from core.upset import UpsetResult
from tests.unit import g3_support as S

_RAW_ML = {1: 0.349, 2: 0.0354, 3: 0.013, 4: 0.0332, 5: 0.2639, 6: 0.3056}
_SUM = sum(_RAW_ML.values())
ML_PROBS = {k: v / _SUM for k, v in _RAW_ML.items()}
UPSET = 5.766
ODDS = {f"{a}-{b}-{c}": round(8.639 + (i % 37) * 1.125, 2)
        for i, (a, b, c) in enumerate(itertools.permutations(range(1, 7), 3))}


def _upset():
    return UpsetResult(upset_score=UPSET, detail={}, target_lanes=(2, 3), lane_probs={},
                       boat1_prob=0.5, boat1_rank=1, best_other_lane=6, best_other_prob=0.3,
                       danger_score=10.0, danger_breakdown={}, venue_name=None)


@unittest.skipIf(os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"),
                 "token present: skip to avoid touching Release")
class RealPurchaseFlowTests(unittest.TestCase):
    def test_p1_purchase_flow_real_legacy(self):
        cwd = os.getcwd()
        tmp = tempfile.mkdtemp(prefix="g3_buy_")
        os.chdir(tmp)
        try:
            import notify_arashi as N  # Legacy（読み込むだけ）
            ev = S.evaluation(upset_score=UPSET, win_probs=ML_PROBS)
            rec = G.evaluate_race_g3(
                evaluation=ev, upset=_upset(), legacy_boats=[], odds_map=ODDS,
                weather=N.WeatherInfo(), buy_config={"_version": "unused"},
                evaluate_bets=N._evaluate_bets, generate_patterns=N._generate_patterns,
            )
        finally:
            os.chdir(cwd)

        self.assertEqual(rec["outcome"], G.OUTCOME_COMPARED, rec)
        self.assertTrue(all(v["ok"] for v in rec["conditions"].values()), rec["conditions"])
        self.assertFalse(rec["ml_bonus"]["applied"])  # この入力では 1号艇の確率が最大
        # Legacy は購入、Prediction は buyscore 最高の候補
        self.assertIs(rec["legacy"]["passthrough"], False)
        self.assertIsNone(rec["legacy"]["passthrough_reason"])
        self.assertIsNotNone(rec["legacy"]["buyscore_top_combo"])
        # 写し取り：呼び出し時の context は Legacy の7キー、展示なし、近似の印
        ctx = rec["capture"]["call_context"]
        self.assertIs(ctx["has_exhibition"], False)
        self.assertIs(ctx["match_index_approx"], True)
        self.assertEqual(ctx["upset_score"], UPSET)
        self.assertIsNotNone(rec["capture"]["cfg_sha256"])
        # 4項目の完全一致（Legacy の値は Legacy の計算結果、investment_type は再計算値）
        self.assertTrue(rec["match"]["all"], rec["match"])
        self.assertIsNone(rec["rebuild"]["skip_reason"])
        self.assertGreaterEqual(rec["rebuild"]["buyscore"], 60)
        self.assertEqual(rec["rebuild"]["buyscore"], rec["legacy"]["pred_combo_candidate"]["buyscore"])
        self.assertEqual(rec["rebuild"]["kelly_fraction"], rec["legacy"]["pred_combo_candidate"]["kelly"])
        self.assertEqual(rec["rebuild"]["investment_type"], rec["legacy"]["investment_type_recomputed"])
        # Rebuild は Legacy が読み込んだ設定で評価している（buy_config の "_version" は使われない）
        self.assertNotEqual(rec["rebuild"]["config_version"], "unused")


if __name__ == "__main__":
    unittest.main()
