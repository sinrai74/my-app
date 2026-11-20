"""特徴量の時点の規則（Project 05 §7・11 §2）の試験。未来の結果が混ざらないこと。"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from asahi_ai import features


def hist_rows(rows):
    """rows: (date, racer_no, venue, lane, finish, motor_no)"""
    recs = []
    for i, (d, racer, venue, lane, fin, motor) in enumerate(rows):
        recs.append({"race_id": f"R{i}", "date": pd.Timestamp(d), "racer_no": racer, "venue": venue,
                     "lane": lane, "finish_position": fin, "result_class": "finished",
                     "st_value": 0.15, "st_flag": "", "exhibition_time": 6.8, "course": lane,
                     "motor_no": motor})
    h = pd.DataFrame(recs)
    data = {"races": pd.DataFrame({"race_id": h["race_id"], "date": h["date"], "venue": h["venue"],
                                   "race_status": "completed", "quarantined": False}),
            "results": h.drop(columns=["date", "venue"]),
            "days": pd.DataFrame({"date": h["date"].dt.strftime("%Y-%m-%d").unique(), "day_verdict": "ok"})}
    return features.history(data)


def target(date="2024-03-10", racer="4001", venue="01", lane=1, motor=10):
    return pd.DataFrame([{"race_id": "T", "date": pd.Timestamp(date), "venue": venue, "lane": lane,
                          "racer_no": racer, "motor_no": float(motor)}])


class TestAsOf(unittest.TestCase):
    def test_boundaries_lag1(self):
        # 予測日 3/10、lag=1 → 3/9 までを使う。r180 の窓は (3/9 − 180日, 3/9]
        h = hist_rows([
            ("2024-03-09", "4001", "01", 1, 1, 10),   # 入る（前日）
            ("2024-03-10", "4001", "01", 1, 1, 10),   # 入らない（当日）
            ("2024-03-11", "4001", "01", 1, 1, 10),   # 入らない（未来）
            ("2023-09-11", "4001", "01", 1, 2, 10),   # 3/9 − 180日 = 2023-09-11 → 入らない（窓の外）
            ("2023-09-12", "4001", "01", 1, 3, 10),   # 入る
        ])
        w = features.window_features(target(), h, lag=1)
        self.assertEqual(w.loc[0, "r180_n"], 2)
        self.assertAlmostEqual(w.loc[0, "r180_win"], 0.5)
        self.assertAlmostEqual(w.loc[0, "r180_fin"], 2.0)

    def test_lag2_excludes_previous_day(self):
        h = hist_rows([("2024-03-09", "4001", "01", 1, 1, 10), ("2024-03-08", "4001", "01", 1, 4, 10)])
        self.assertEqual(features.window_features(target(), h, lag=1).loc[0, "r180_n"], 2)
        self.assertEqual(features.window_features(target(), h, lag=2).loc[0, "r180_n"], 1)

    def test_future_changes_do_not_change_features(self):
        base = [("2024-01-0%d" % d, "4001", "01", 1 + d % 6, 1 + d % 6, 10) for d in range(1, 10)]
        future = [("2024-03-%02d" % d, "4001", "01", 1, 1, 10) for d in range(10, 28)]
        w1 = features.window_features(target(), hist_rows(base + future), lag=1)
        future2 = [(d, r, v, l, 6, m) for d, r, v, l, _, m in future] + [("2024-03-10", "4001", "01", 1, 6, 10)]
        w2 = features.window_features(target(), hist_rows(base + future2), lag=1)
        pd.testing.assert_frame_equal(w1, w2)

    def test_keys_are_separate(self):
        h = hist_rows([("2024-03-01", "4001", "02", 3, 1, 10), ("2024-03-02", "4002", "01", 1, 1, 10),
                       ("2024-03-03", "4001", "01", 1, 2, 11)])
        w = features.window_features(target(), h, lag=1)
        self.assertEqual(w.loc[0, "r180_n"], 2)      # 選手 4001 の2走
        self.assertEqual(w.loc[0, "rv730_n"], 1)     # 当地（01）は1走
        self.assertEqual(w.loc[0, "rl365_n"], 1)     # 1号艇は1走
        self.assertEqual(w.loc[0, "m120_n"], 1)      # 場01・モーター10 は 4002 の1走
        self.assertEqual(w.loc[0, "vl365_n"], 2)     # 場01・1号艇は2走
        self.assertEqual(w.loc[0, "rv730_win"], 0.0)  # 当地の1走は2着

    def test_absent_rows_excluded_from_history(self):
        h = hist_rows([("2024-03-01", "4001", "01", 1, None, 10)])
        self.assertEqual(len(h), 1)
        data = {"races": pd.DataFrame({"race_id": ["X"], "date": [pd.Timestamp("2024-03-01")], "venue": ["01"],
                                       "race_status": ["completed"], "quarantined": [False]}),
                "results": pd.DataFrame([{"race_id": "X", "lane": 1, "racer_no": "4001", "finish_position": None,
                                          "result_class": "absent", "st_value": None, "st_flag": "K",
                                          "exhibition_time": None, "course": None, "motor_no": 10}]),
                "days": pd.DataFrame({"date": ["2024-03-01"], "day_verdict": ["ok"]})}
        self.assertEqual(len(features.history(data)), 0)


if __name__ == "__main__":
    unittest.main()
