"""独立レビュー（11/21）の指摘への対応の試験。"""
from __future__ import annotations

import datetime as dt
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from asahi_ai import archive, compare_legacy, dataset, evaluate, fetch, model


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def clock(self):
        return self.t

    def sleep(self, s):
        self.t += s


class TestFetchFixes(unittest.TestCase):
    def _fetcher(self, tmp, script):
        def opener(url, timeout):
            r = script.pop(0)
            if isinstance(r, Exception):
                raise r
            return r
        fc = FakeClock()
        return fetch.Fetcher(Path(tmp), opener=opener, interval=0, clock=fc.clock, sleep=fc.sleep)

    def test_bad_archive_not_saved_as_original(self):
        d = dt.date(2023, 1, 1)
        with tempfile.TemporaryDirectory() as tmp:
            f = self._fetcher(tmp, [(200, b"broken", "Sun, 01 Jan 2023 12:00:00 GMT"), (200, b"good", None)])
            with mock.patch.object(archive, "extract_single", side_effect=archive.ArchiveError("CRC")):
                rec, data = f.get("B", d)
            self.assertEqual((rec["capture_status"], data), ("bad_archive", None))
            self.assertFalse(f.path_for("B", d).exists())                 # 原本の場所には置かない
            self.assertEqual(len(list((Path(tmp) / "_bad").rglob("*.lzh"))), 1)
            with mock.patch.object(archive, "extract_single",
                                   return_value=archive.Member("B230101.TXT", "-lh5-", b"x", 0, 0)):
                rec2, _ = f.get("B", d)                                     # 次は取り直せる（衝突しない）
            self.assertEqual(rec2["capture_status"], "saved")
            self.assertEqual(rec["last_modified"], "Sun, 01 Jan 2023 12:00:00 GMT")

    def test_5xx_retried_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            e = urllib.error.HTTPError("u", 503, "Service Unavailable", {}, None)
            f = self._fetcher(tmp, [e, e])
            rec, _ = f.get("K", dt.date(2023, 1, 1))
            self.assertEqual((rec["capture_status"], rec["http_status"]), ("fetch_failed", 503))
            self.assertIn("503", rec["error"])
            f2 = self._fetcher(tmp, [e, urllib.error.HTTPError("u", 404, "nf", {}, None)])
            self.assertEqual(f2.get("K", dt.date(2023, 1, 1))[0]["capture_status"], "not_found")


class TestCacheKey(unittest.TestCase):
    def test_fingerprint_changes_with_code(self):
        a = dataset.code_fingerprint()
        real = Path.read_bytes

        def fake(self):
            b = real(self)
            return b + b"#changed" if self.name == "parse_k.py" else b
        with mock.patch.object(Path, "read_bytes", fake):
            self.assertNotEqual(dataset.code_fingerprint(), a)


def _test_frame(rows):
    """rows: (race_id, lane, finish_position, result_class)"""
    df = pd.DataFrame(rows, columns=["race_id", "lane", "finish_position", "result_class"])
    df["won"] = (df["finish_position"] == 1).astype(int)
    return df


def _pred_with_pick(race_id, pick):
    tri = np.zeros(120)
    tri[model.trifecta_index(pick)] = 1.0
    row = {"race_id": race_id, "tri": tri}
    for i in range(1, 7):
        row[f"p_win_{i}"] = 1.0 if i == pick[0] else 0.0
    return row


class TestEvaluateFixes(unittest.TestCase):
    def test_roi_refund_special_and_hit(self):
        fin = [1, 2, 3, 4, 5, 6]
        rows = []
        for r in ("A", "B", "C", "D"):
            for lane, f in zip(range(1, 7), fin):
                rows.append((r, lane, f, "finished"))
        rows = [(r, l, (None if (r == "C" and l == 6) else f), ("absent" if (r == "C" and l == 6) else c))
                for r, l, f, c in rows]
        t = _test_frame(rows)
        pay = pd.DataFrame([
            {"race_id": "A", "bet_type": "trifecta", "payout_kind": "hit", "combination": "1-2-3", "payout_yen": 1500},
            {"race_id": "B", "bet_type": "trifecta", "payout_kind": "不成立", "combination": "", "payout_yen": None},
            {"race_id": "C", "bet_type": "trifecta", "payout_kind": "hit", "combination": "1-2-3", "payout_yen": 900},
            {"race_id": "D", "bet_type": "trifecta", "payout_kind": "特払い", "combination": "", "payout_yen": 70},
        ])
        pred = pd.DataFrame([_pred_with_pick("A", (1, 2, 3)), _pred_with_pick("B", (4, 5, 6)),
                             _pred_with_pick("C", (6, 1, 2)), _pred_with_pick("D", (2, 3, 4))])
        m = evaluate._metrics(pred, t, pay)
        # A 的中 1500、B 不成立で返還 100、C 欠場の艇を含むので返還 100、D 特払い 70
        self.assertAlmostEqual(m["roi_tri_top_pick"], (1500 + 100 + 100 + 70) / 400)

    def test_lane_baseline_uses_training_labels_only(self):
        te = _test_frame([("T", l, (1 if l == 6 else l + 1), "finished") for l in range(1, 7)])
        te["vl365_win"] = np.nan
        te["vl365_n"] = 0
        tr = _test_frame([("X", l, l, "finished") for l in range(1, 7)])  # 学習期間は1号艇が勝つ
        p = evaluate._baseline_lane(te, tr).set_index("race_id")
        self.assertGreater(p.at["T", "p_win_1"], p.at["T", "p_win_6"])  # テストの正解（6号艇）に引きずられない


class TestCompareLegacy(unittest.TestCase):
    def test_excludes_signals_committed_after_deadline(self):
        races = pd.DataFrame({"race_id": [f"R{i}" for i in range(12)], "date": ["2026-07-01"] * 12,
                              "venue": ["01"] * 12, "race_no": list(range(1, 13)),
                              "deadline_time": ["10:00"] * 6 + ["20:00"] * 6})
        results = pd.DataFrame([{"race_id": f"R{i}", "lane": l, "finish_position": (1 if l == 2 else None)}
                                for i in range(12) for l in range(1, 7)])
        pay = pd.DataFrame([{"race_id": f"R{i}", "bet_type": "trifecta", "payout_kind": "hit",
                             "combination": "2-1-3", "payout_yen": 20000} for i in range(12)])
        sig = pd.DataFrame([{"date": "20260701", "kind": "all_danger", "venue": "01", "race_no": i + 1,
                             "score": 50, "commit": "c", "committed_at": "2026-07-01T03:00:00+00:00"}
                            for i in range(12)])  # 12:00 JST の commit
        pred = pd.DataFrame([{"race_id": f"R{i}", **{f"p_win_{l}": 1 / 6 for l in range(1, 7)},
                              "tri": np.full(120, 1 / 120)} for i in range(12)])
        out = compare_legacy.compare(sig, pred, races, results, pay, top_n=3)
        self.assertEqual(out["excluded_after_deadline"], 6)        # 締切 10:00 の6レースは除く
        self.assertEqual(out["boat1_lost"]["picked"], 3)


if __name__ == "__main__":
    unittest.main()
