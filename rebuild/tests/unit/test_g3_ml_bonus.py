"""正式G3（B案）：ML 加点の純関数（07 §7）。

Legacy の加点は関数ではなく run() のループ内のコード（notify_arashi.py 3463〜3476行目）なので、
テストの中で同じ式を逐語的に再現し、固定・ランダムの入力で比べる。Legacy は変更しない。
"""

from __future__ import annotations

import random
import unittest
from types import SimpleNamespace

from core.ml_bonus import apply_ml_bonus


def _legacy_loop(score, target, ml_probs, boats):
    """notify_arashi.py 3462〜3476行目の逐語的な再現（ログ・detail を除く）。"""
    if ml_probs:
        prob_1 = ml_probs.get(1, 0.0)
        other_probs = {l: p for l, p in ml_probs.items() if l != 1}
        best_other_lane = max(other_probs, key=other_probs.get) if other_probs else None
        best_other_prob = other_probs.get(best_other_lane, 0.0)
        if best_other_prob > prob_1:
            ml_score = min(best_other_prob * 8.0, 4.0)
            score += ml_score
            sorted_lanes = sorted(
                [b.lane for b in boats if b.lane != 1],
                key=lambda l: ml_probs.get(l, 0), reverse=True,
            )
            target = sorted_lanes[:2]
    return score, list(target)


def _boats(order):
    return [SimpleNamespace(lane=l) for l in order]


class MlBonusTests(unittest.TestCase):
    def test_applied(self):
        r = apply_ml_bonus(3.0, (2, 3), {1: 0.2, 2: 0.1, 3: 0.3, 4: 0.25}, [1, 2, 3, 4, 5, 6])
        self.assertTrue(r.applied)
        self.assertEqual(r.bonus, min(0.3 * 8.0, 4.0))
        self.assertEqual(r.score_post, 3.0 + 2.4)
        self.assertEqual(r.target_post, (3, 4))

    def test_not_applied_when_equal(self):
        r = apply_ml_bonus(3.0, (2, 3), {1: 0.3, 2: 0.3}, [1, 2, 3, 4, 5, 6])
        self.assertFalse(r.applied)
        self.assertEqual((r.score_post, r.target_post, r.bonus), (3.0, (2, 3), 0.0))

    def test_cap_and_zero_pre(self):
        r = apply_ml_bonus(0.0, (2, 3), {1: 0.01, 5: 0.9}, [1, 2, 3, 4, 5, 6])
        self.assertEqual(r.bonus, 4.0)
        self.assertEqual(r.score_post, 4.0)
        r2 = apply_ml_bonus(9.5, (2, 3), {1: 0.01, 5: 0.9}, [1, 2, 3, 4, 5, 6])
        self.assertEqual(r2.score_post, 13.5)

    def test_ties_follow_lane_order_and_missing_lane_is_zero(self):
        r = apply_ml_bonus(2.0, (2, 3), {1: 0.1, 4: 0.3, 6: 0.3}, [1, 6, 5, 4, 3, 2])
        self.assertEqual(r.target_post, (6, 4))

    def test_empty(self):
        r = apply_ml_bonus(2.0, (2, 3), {}, [1, 2, 3])
        self.assertFalse(r.applied)
        self.assertEqual(r.target_post, (2, 3))

    def test_matches_legacy_loop_random(self):
        rng = random.Random(20261114)
        for _ in range(5000):
            order = [1, 2, 3, 4, 5, 6]
            rng.shuffle(order)
            lanes = rng.sample(range(1, 7), rng.randint(1, 6))
            probs = {l: round(rng.random(), rng.choice((1, 2, 3))) for l in lanes}
            score = rng.uniform(0.0, 9.5)
            target = tuple(sorted(rng.sample(range(2, 7), 2)))
            want = _legacy_loop(score, list(target), probs, _boats(order))
            got = apply_ml_bonus(score, target, probs, order)
            self.assertEqual((got.score_post, list(got.target_post)), want, (score, target, probs, order))


if __name__ == "__main__":
    unittest.main()
