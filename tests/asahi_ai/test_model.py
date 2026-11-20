"""モデル（Project 11 §4）の試験：確率の整合と、学習が強さを取り戻せること。"""
from __future__ import annotations

import unittest

import importlib.util

import numpy as np
import pandas as pd

from asahi_ai import model

HAS_LGB = importlib.util.find_spec("lightgbm") is not None and importlib.util.find_spec("scipy") is not None


def synth_races(n_races=3000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_races):
        x = rng.normal(size=6)
        lane_eff = np.array([1.0, 0.3, 0.1, 0.0, -0.2, -0.4])
        s = 1.2 * x + lane_eff
        # Plackett–Luce（λ=1）で着順を作る
        order, left = [], list(range(6))
        for _ in range(6):
            w = np.exp(s[left])
            k = rng.choice(len(left), p=w / w.sum())
            order.append(left.pop(k))
        fin = np.empty(6, int)
        fin[order] = np.arange(1, 7)
        for i in range(6):
            rows.append({"race_id": f"R{r:05d}", "lane": i + 1, "x": x[i], "lane_f": i + 1,
                         "finish_position": fin[i], "won": int(fin[i] == 1), "result_class": "finished"})
    return pd.DataFrame(rows)


class TestTrifecta(unittest.TestCase):
    def test_sums_to_one_and_harville(self):
        s = np.log(np.array([[6, 5, 4, 3, 2, 1.0]]))
        p = model._trifecta_matrix(s, (1.0, 1.0))
        self.assertAlmostEqual(p.sum(), 1.0, places=10)
        w = np.array([6, 5, 4, 3, 2, 1.0])
        exp = w[0] / 21 * w[1] / 15 * w[2] / 10
        self.assertAlmostEqual(p[0, model.trifecta_index((1, 2, 3))], exp, places=12)

    def test_missing_boat(self):
        s = np.array([[0.5, 0.1, -np.inf, 0.0, 0.3, -0.2]])
        for lam in [(1.0, 1.0), (0.7, 0.4)]:
            p = model._trifecta_matrix(s, lam)
            self.assertAlmostEqual(p.sum(), 1.0, places=10)
            bad = [i for i, t in enumerate(model.PERMS) if 2 in t]
            self.assertEqual(p[0, bad].sum(), 0.0)

    def test_actual_trifecta(self):
        d = pd.DataFrame({"race_id": ["A"] * 6, "lane": range(1, 7), "finish_position": [3, 1, 2, 4, 5, 6]})
        self.assertEqual(model.actual_trifecta(d)["A"], model.trifecta_index((2, 3, 1)))
        d2 = d.assign(finish_position=[1, 1, 3, 4, 5, 6])  # 同着は除く
        self.assertNotIn("A", model.actual_trifecta(d2).index)


@unittest.skipUnless(HAS_LGB, "lightgbm・scipy が無い")
class TestStrengthModel(unittest.TestCase):
    def test_learns_and_probabilities_consistent(self):
        df = synth_races(4000)
        tr = df[df["race_id"] < "R03000"]
        va = df[(df["race_id"] >= "R03000") & (df["race_id"] < "R03500")]
        te = df[df["race_id"] >= "R03500"]
        m = model.StrengthModel({"min_data_in_leaf": 50, "learning_rate": 0.1}, num_rounds=300)
        m.fit(tr, va, ["x", "lane_f"])
        pr = m.predict(te)
        self.assertTrue(np.allclose(pr["win_sum"], 1.0))
        self.assertTrue(np.allclose(pr["tri_sum"], 1.0))
        # 真の強さ（PL, λ=1）での1着の対数損失に近いこと、一様（log 6）より十分良いこと
        te = te.sort_values(["race_id", "lane"])
        win_lane = te[te["won"] == 1].set_index("race_id")["lane"]
        p = np.array([pr.set_index("race_id").loc[r, f"p_win_{l}"] for r, l in win_lane.items()])
        ll = -np.log(p).mean()
        self.assertLess(ll, np.log(6) - 0.25)
        self.assertTrue(0.6 < m.lam[0] < 1.4 and 0.5 < m.lam[1] < 1.5, m.lam)  # 作り方は λ=1


if __name__ == "__main__":
    unittest.main()
