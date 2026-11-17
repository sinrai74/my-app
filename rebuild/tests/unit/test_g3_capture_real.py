"""正式G3（B案）：実際の Legacy `_evaluate_bets` を固定の合成入力で呼んだときの写し取り（04 T12・T13、08 §4 R4）。

- N-01：偽のトークンを環境に置いたうえで、shadow.g3_isolation の境界の中で実行する。
  Release の関数は差し替えられ、ネットワーク・外部コマンド・外への書き込みは止まる（止めた記録が0件であることも確かめる）。
- 作業ディレクトリは境界が作る一時フォルダ（freeze 版の設定だけを置く）。
"""

from __future__ import annotations

import itertools
import unittest

from shadow.g3_capture import STATUS_CAPTURED, ApplyBuyscoreCapture
from shadow.g3_isolation import G3Isolation
from tests.unit import g3_support as S


class RealEvaluateBetsCaptureTests(unittest.TestCase):
    def test_t12_t13_real_evaluate_bets(self):
        with S.fake_token_env(), G3Isolation({"buyscore_config.json": S.CONFIG_FIXTURE}) as iso:
            self.assertTrue(iso.established, iso.errors)
            import notify_arashi as N  # Legacy（読み込むだけ）
            patterns = N._generate_patterns([2, 3], 5.0)
            ml_probs = {1: 0.45, 2: 0.25, 3: 0.12, 4: 0.08, 5: 0.06, 6: 0.04}
            odds_map = {}
            for i, (a, b, c) in enumerate(itertools.permutations(range(1, 7), 3)):
                odds_map[f"{a}-{b}-{c}"] = 20.0 + (i % 37) * 10.0
            cap = ApplyBuyscoreCapture()
            with cap:
                result = N._evaluate_bets(
                    patterns, ml_probs, odds_map, target_lanes=[2, 3], has_exhibition=False,
                    boats=None, upset_score=5.0, odds_dropped=[], weather=N.WeatherInfo(),
                    venue_num=12, race_number=5, race_date="20261110", venue_name="住之江",
                )
            without = N._evaluate_bets(
                patterns, ml_probs, odds_map, target_lanes=[2, 3], has_exhibition=False,
                boats=None, upset_score=5.0, odds_dropped=[], weather=N.WeatherInfo(),
                venue_num=12, race_number=5, race_date="20261110", venue_name="住之江",
            )
        summary = iso.summary()
        self.assertTrue(summary["verified"], summary)
        self.assertEqual(summary["blocked_event_count"], 0, summary["blocked_events"])
        # 実際の Legacy が Release に触れようとした経路（2回の _evaluate_bets 分）は、すべて差し替えで止まった
        self.assertGreaterEqual(len(iso.release_calls_of("download_file", "buyscore_config.json")), 2)
        self.assertGreaterEqual(len(iso.release_calls_of("download_file", "hit_record.csv")), 2)
        self.assertGreaterEqual(len(iso.release_calls_of("upload_file", "buyscore_log.jsonl")), 1)
        self.assertTrue(all(c["blocked"] for c in summary["release_calls"]))
        self.assertTrue(result, "no candidates were produced by the synthetic input")
        self.assertEqual(cap.status, STATUS_CAPTURED, cap.errors)
        self.assertEqual(sorted(cap.call_context), sorted(
            ["match_index", "match_index_approx", "race_type", "has_exhibition", "market_gap",
             "upset_score", "ex_rank_1st"]))
        self.assertIs(cap.call_context["match_index_approx"], True)
        self.assertIs(cap.call_context["has_exhibition"], False)
        self.assertTrue(all("buyscore" not in c for c in cap.call_candidates))
        self.assertIsNotNone(cap.return_cfg)
        # T13：写し取りの値と戻り値の値の一致
        by_combo = {c["combo"]: c for c in cap.call_candidates}
        for r in result:
            src = by_combo.get(r["combo"])
            self.assertIsNotNone(src)
            for k in ("uncertainty", "disagreement", "composite"):
                self.assertEqual(r.get(k), src.get(k), (r["combo"], k))
        # T4：写し取りの有無で戻り値が同じ（金額など外部状態に依存しない項目で比べる）
        keys = ("combo", "prob", "ev", "odds", "buyscore", "kelly", "skip_reason")
        self.assertEqual([{k: r.get(k) for k in keys} for r in result],
                         [{k: r.get(k) for k in keys} for r in without])


if __name__ == "__main__":
    unittest.main()
