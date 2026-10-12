"""
再学習の結果付与（parsers.attach_k_results）で、日付をまたいだラベル混在が起きないことのテスト。

docs/incidents/2026-10-12_retrain-label-cross-date-mix.md の修正の確認用。
合成データだけを使う（公式の B/K ファイルは使わない）。

実行: python -m unittest test_retrain_label_date_key_20261012 -v
"""
from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import pandas as pd

from parsers import attach_k_results
from retrain import check_label_integrity

# (日付, 場コード, レースNo) → 着順（1着・2着・3着の艇番）
RESULTS = {
    ("2026-10-01", "01", 1): (1, 2, 3),
    ("2026-10-02", "01", 1): (4, 5, 6),
    ("2026-10-01", "01", 2): (6, 1, 2),
    ("2026-10-02", "02", 1): (2, 3, 4),
}


def _bangumi() -> pd.DataFrame:
    rows = []
    for (d, vc, rno) in RESULTS:
        for lane in range(1, 7):
            rows.append({"場コード": vc, "レースNo": rno, "艇番": lane,
                         "開催日": d, "_file_date": d})
    return pd.DataFrame(rows)


def _k() -> pd.DataFrame:
    rows = []
    for (d, vc, rno), (a, b, c) in RESULTS.items():
        rows.append({"場コード": vc, "レースNo": rno, "組番": f"{a}-{b}-{c}",
                     "1着": a, "2着": b, "3着": c, "払戻": 1000 + a * 100 + b * 10 + c,
                     "返還フラグ": 0, "_file_date": d})
    return pd.DataFrame(rows)


def _attach(b, k, **kw) -> pd.DataFrame:
    with redirect_stdout(io.StringIO()):
        return attach_k_results(b, k, **kw)


class TestAttachWithDateKey(unittest.TestCase):

    def test_old_behaviour_mixes_dates(self):
        # 日付を指定しない（従来の呼び方）と、同じ場・同じレース番号の別の日の結果が混ざる
        m = _attach(_bangumi(), _k())
        day1 = m[(m["_file_date_x"] if "_file_date_x" in m else m["_file_date"]) == "2026-10-01"]
        race = day1[(day1["場コード"] == "01") & (day1["レースNo"] == 1)]
        self.assertEqual(int((race["着順"] == 1).sum()), 2)

    def test_labels_match_each_day(self):
        b = _bangumi()
        m = _attach(b, _k(), date_col="_file_date")
        self.assertEqual(len(m), len(b))
        for (d, vc, rno), order in RESULTS.items():
            race = m[(m["_file_date"] == d) & (m["場コード"] == vc) & (m["レースNo"] == rno)]
            got = {int(r["艇番"]): int(r["着順"]) for _, r in race.iterrows()}
            for rank, lane in enumerate(order, start=1):
                self.assertEqual(got[lane], rank, (d, vc, rno, lane))
            self.assertEqual(int((race["着順"] == 1).sum()), 1)
            self.assertEqual(sorted(got.values()), [1, 2, 3, 4, 5, 6])
            pay = race["払戻"].unique()
            self.assertEqual(len(pay), 1)

    def test_race_without_result_has_no_winner(self):
        k = _k()
        k = k[~((k["_file_date"] == "2026-10-02") & (k["場コード"] == "02"))]
        m = _attach(_bangumi(), k, date_col="_file_date")
        race = m[(m["_file_date"] == "2026-10-02") & (m["場コード"] == "02")]
        self.assertEqual(int((race["着順"] == 1).sum()), 0)
        check_label_integrity(m, len(_bangumi()))

    def test_missing_date_column_raises(self):
        with self.assertRaises(ValueError):
            _attach(_bangumi(), _k().drop(columns=["_file_date"]), date_col="_file_date")

    def test_empty_date_raises(self):
        b = _bangumi()
        b.loc[0, "_file_date"] = ""
        with self.assertRaises(ValueError):
            _attach(b, _k(), date_col="_file_date")

    def test_duplicated_result_raises(self):
        k = pd.concat([_k(), _k().iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            _attach(_bangumi(), k, date_col="_file_date")


class TestCheckLabelIntegrity(unittest.TestCase):

    def setUp(self):
        self.b = _bangumi()
        self.m = _attach(self.b, _k(), date_col="_file_date")

    def test_passes_on_clean_data(self):
        with self.assertLogs("retrain", level="INFO") as cm:
            check_label_integrity(self.m, len(self.b))
        self.assertTrue(any("ラベル検証" in s for s in cm.output))

    def test_row_count_change_raises(self):
        with self.assertRaises(ValueError):
            check_label_integrity(self.m, len(self.b) + 1)

    def test_duplicate_rows_raise(self):
        dup = pd.concat([self.m, self.m.iloc[[0]]], ignore_index=True)
        with self.assertRaises(ValueError):
            check_label_integrity(dup, len(dup))

    def test_two_winners_raise(self):
        bad = self.m.copy()
        idx = bad[(bad["_file_date"] == "2026-10-01") & (bad["場コード"] == "01")
                  & (bad["レースNo"] == 1) & (bad["艇番"] == 6)].index
        bad.loc[idx, "着順"] = 1
        with self.assertRaises(ValueError):
            check_label_integrity(bad, len(bad))


if __name__ == "__main__":
    unittest.main()
