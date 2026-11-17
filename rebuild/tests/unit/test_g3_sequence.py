"""段階4-2：g3_sequence（連続カウント集約）の回帰テスト。"""
import unittest

from core.g3_compare import INCOMPARABLE, MATCH, MISMATCH
from shadow.g3_sequence import count_consecutive


def _r(judge=None, outcome="compared", mode="formal"):
    d = {"outcome": outcome, "mode": mode}
    if outcome == "compared":
        d["race_judge"] = judge
    return d


class CountConsecutiveTests(unittest.TestCase):
    def test_match_increments(self):
        res = count_consecutive([_r(MATCH), _r(MATCH), _r(MATCH)])
        self.assertEqual(res["streak_current"], 3)
        self.assertEqual(res["best_streak"], 3)
        self.assertEqual(res["compared_count"], 3)
        self.assertFalse(res["achieved"])

    def test_mismatch_resets_but_counts(self):
        res = count_consecutive([_r(MATCH), _r(MISMATCH), _r(MATCH)])
        self.assertEqual(res["streak_current"], 1)
        self.assertEqual(res["best_streak"], 1)
        self.assertEqual(res["compared_count"], 3)   # mismatch も比較成立なので件数に入る

    def test_incomparable_resets_and_not_counted(self):
        res = count_consecutive([_r(MATCH), _r(INCOMPARABLE), _r(MATCH)])
        self.assertEqual(res["streak_current"], 1)
        self.assertEqual(res["compared_count"], 2)   # incomparable は件数に入らない

    def test_non_compared_resets_and_not_counted(self):
        res = count_consecutive([_r(MATCH), _r(outcome="no_prediction"), _r(outcome="no_odds"), _r(MATCH)])
        self.assertEqual(res["streak_current"], 1)
        self.assertEqual(res["compared_count"], 2)

    def test_investigation_excluded(self):
        # investigation は連続に影響しない（間に挟んでも継続）
        res = count_consecutive([_r(MATCH), _r(MATCH, mode="investigation"), _r(MATCH)])
        self.assertEqual(res["streak_current"], 2)
        self.assertEqual(res["compared_count"], 2)

    def test_achieved_at_target_and_no_extra(self):
        recs = [_r(MATCH) for _ in range(102)]
        res = count_consecutive(recs, target=100)
        self.assertTrue(res["achieved"])
        self.assertEqual(res["achieved_at_index"], 99)
        self.assertEqual(res["best_streak"], 100)     # 到達後の追加一致は加えない
        self.assertEqual(res["streak_current"], 100)

    def test_mixed_days_accumulate(self):
        # 複数日通算：formal の連結で連続が積み上がる
        day1 = [_r(MATCH) for _ in range(50)]
        day2 = [_r(MATCH) for _ in range(50)]
        res = count_consecutive(day1 + day2, target=100)
        self.assertTrue(res["achieved"])
        self.assertEqual(res["best_streak"], 100)


if __name__ == "__main__":
    unittest.main()
