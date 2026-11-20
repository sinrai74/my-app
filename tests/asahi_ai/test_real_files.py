"""実ファイルでの試験（任意）。ASAHI_TEST_RAW_DIR に b230101/k230101・b240615/k240615・b261116/k261116 の
LZH を置いたときだけ動く。期待値は Project 03・04（本人が添付した3日分の手作業の照合）。"""
from __future__ import annotations

import datetime as dt
import os
import unittest
from pathlib import Path

from asahi_ai import cli

RAW = os.environ.get("ASAHI_TEST_RAW_DIR")
EXPECT = {  # 日付: (場, レース, B の行, K の行, 欠場)
    "230101": (15, 180, 1080, 1080, 6),
    "240615": (15, 180, 1080, 1080, 2),   # 旧パーサは福岡（22）を落として 1,008 行
    "261116": (10, 120, 720, 720, 2),
}


@unittest.skipUnless(RAW, "ASAHI_TEST_RAW_DIR が無い")
class TestRealFiles(unittest.TestCase):
    def test_three_days(self):
        for ds, (v, r, nb, nk, absent) in EXPECT.items():
            d = dt.datetime.strptime(ds, "%y%m%d").date()
            b = (Path(RAW) / f"b{ds}.lzh").read_bytes()
            k = (Path(RAW) / f"k{ds}.lzh").read_bytes()
            t = cli.process(d, b, k)
            c = t.counts
            self.assertEqual(t.day_verdict, "ok", ds)
            self.assertEqual((c["b_venues"], c["k_venues"], c["b_races"], c["k_races"]), (v, v, r, r), ds)
            self.assertEqual((c["b_entries"], c["k_results"], c["absent"]), (nb, nk, absent), ds)
            self.assertEqual(c["status_completed"], r, ds)
            self.assertNotIn("stop", {x["level"] for x in t.checks}, ds)


if __name__ == "__main__":
    unittest.main()
