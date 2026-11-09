"""race_closed_at の検証（07f §6.4・§6.5 の 8。GPT判断 11/04。段階3、GPT 指示 2026-11-09）。

集約側で、固定した対象一覧の順について次を確かめる。合わない・確かめられない場合は、実行全体を外す。
- 日時として読めること（実データの形 "YYYY-MM-DD HH:MM:SS" だけ）
- 日時の昇順
- 同じ時刻なら、会場番号、R 番号の順
条件5 (iii)（文字列として並べ直した一覧の照合）とは別の確認。
"""
import os
import tempfile
import unittest
from unittest import mock

import test_aggregate as ta
from formal_g2.aggregate import main as agg
from test_aggregate import BASELINE, BASELINE_SHA, FORMAL_OK, PERMS, _race, _write_run


def T(eid, ca, vn=1, rn=1):
    return {"eval_id": eid, "venue_num": vn, "race_number": rn, "race_closed_at": ca}


class UnitTests(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(agg.race_closed_at_check([T("a", "2026-11-01 10:45:00", 1, 1),
                                                   T("b", "2026-11-01 11:12:00", 1, 2)]), [])

    def test_same_time_ordered_by_venue_then_race(self):
        ok = [T("a", "2026-11-01 15:00:00", 2, 5), T("b", "2026-11-01 15:00:00", 10, 1),
              T("c", "2026-11-01 15:00:00", 10, 3)]
        self.assertEqual(agg.race_closed_at_check(ok), [])
        # 会場番号は数として比べる（"10" < "2" の文字列の順ではない）
        self.assertTrue(agg.race_closed_at_check([ok[1], ok[0]]))
        self.assertTrue(agg.race_closed_at_check([ok[0], ok[2], ok[1]]))  # 同じ会場で R の逆順

    def test_descending_and_duplicate(self):
        self.assertTrue(agg.race_closed_at_check([T("a", "2026-11-01 11:00:00", 1, 1),
                                                  T("b", "2026-11-01 10:59:59", 1, 2)]))
        self.assertTrue(agg.race_closed_at_check([T("a", "2026-11-01 11:00:00", 1, 1),
                                                  T("b", "2026-11-01 11:00:00", 1, 1)]))

    def test_unreadable_values(self):
        for bad in (None, "", "15:00", "2026-11-01T10:45:00", "2026-11-01 10:45:00+09:00", "2026-11-01 10:45",
                    "2026-11-01 10:45:00 ", "2026-02-30 10:00:00", "2026-11-01 24:00:00", "２０２６-11-01 10:45:00",
                    1730425500, ["2026-11-01 10:45:00"]):
            with self.subTest(bad=bad):
                probs = agg.race_closed_at_check([T("a", "2026-11-01 10:00:00"), T("b", bad, 1, 2)])
                self.assertTrue(probs)
                self.assertIn("not readable", probs[0])

    def test_bad_numbers_and_shapes(self):
        for vn, rn in (("1", 1), (1, "2"), (True, 1), (1, None), (1.0, 2)):
            with self.subTest(vn=vn, rn=rn):
                self.assertTrue(agg.race_closed_at_check([T("a", "2026-11-01 10:00:00", vn, rn)]))
        for targets in (None, [], "x", [1], {"a": 1}):
            with self.subTest(targets=targets):
                self.assertTrue(agg.race_closed_at_check(targets))

    def test_across_midnight_is_compared_as_datetime(self):
        self.assertEqual(agg.race_closed_at_check([T("a", "2026-11-01 23:59:59"), T("b", "2026-11-02 00:00:00", 1, 2)]),
                         [])

    def test_real_data_shape(self):
        """実データ（2026-11-01、168件）と同じ形：同じ時刻のレースが3組ある日も通る。"""
        ts = [T(f"e{i}", f"2026-11-01 {10 + i // 4:02d}:{(i % 4) * 15:02d}:00", 1 + i, 1) for i in range(8)]
        ts.insert(2, T("tie", ts[1]["race_closed_at"], ts[1]["venue_num"] + 0, 2))
        self.assertEqual(agg.race_closed_at_check(ts), [])


def run_with(closed_at, labels, condition5_ok=False):
    old = ta.CLOSED_AT_OVERRIDE
    ta.CLOSED_AT_OVERRIDE = closed_at
    try:
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race(x) for x in labels], labels)
            v = [agg.verify_run(os.path.join(base, x)) for x in sorted(os.listdir(base))]
    finally:
        ta.CLOSED_AT_OVERRIDE = old
    if condition5_ok:
        for x in v:
            x["condition5"] = {"status": "ok", "problems": []}
    return agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)


class AggregateTests(unittest.TestCase):
    def test_valid_run_counts_and_records_ok(self):
        res = run_with(None, ["a", "b", "c"])
        self.assertEqual(res["days"][0]["race_closed_at_check"], "ok")
        self.assertEqual([x.get("state") for x in res["sequence"]], ["added"] * 3)

    def test_unreadable_value_excludes_whole_run(self):
        """条件5 は文字列として並べるので通る。race_closed_at の検証で実行全体を外す。"""
        res = run_with(["2099-01-01 10:00:00", "2099-01-01 10:01", "2099-01-01 10:02:00"], ["a", "b", "c"])
        self.assertEqual(res["days"][0]["condition5_check"], "ok")
        self.assertEqual(res["days"][0]["excluded"], "race_closed_at")
        self.assertEqual([x for x in res["sequence"] if "eval_id" in x], [])
        self.assertEqual(res["sequence"], [{"break": "race_closed_at", "race_date": "20990101"}])
        self.assertEqual(res["best_segment"], 0)

    def test_out_of_order_targets_excluded_even_if_condition5_were_ok(self):
        res = run_with(["2099-01-01 10:02:00", "2099-01-01 10:01:00"], ["a", "b"])
        self.assertNotEqual(res["days"][0]["condition5_check"], "ok")  # 条件5 でも外れる
        res = run_with(["2099-01-01 10:02:00", "2099-01-01 10:01:00"], ["a", "b"], condition5_ok=True)
        self.assertEqual(res["days"][0]["excluded"], "race_closed_at")

    def test_invalid_run_does_not_affect_100th_identification(self):
        """不正な値の実行は加算されないので、連続件数・100件目の特定に入らない（区切る）。"""
        from test_streak_reach import _inst
        good = [f"g{i:03d}" for i in range(1, 101)]
        with tempfile.TemporaryDirectory() as base:
            ta.CLOSED_AT_OVERRIDE = ["bad"] * 3
            try:
                _write_run(base, 1, 1, "20990101", [_race(x) for x in "xyz"], list("xyz"))
            finally:
                ta.CLOSED_AT_OVERRIDE = None
            _write_run(base, 2, 1, "20990102", [_race(x) for x in good], good,
                       run_overrides={"instrument": _inst(100)}, started="2099-01-03T00:00:00Z")
            v = [agg.verify_run(os.path.join(base, x)) for x in sorted(os.listdir(base))]
        perms = PERMS + [dict(PERMS[0], race_date="20990102", permission_id=PERMS[0]["permission_id"] + "-2",
                              window_end="2099-01-04T00:00:00Z")]
        for x in v:
            if x["meta"]["run_id"] in ("2", 2):
                x["manifest"]["run"]["permission_id"] = perms[-1]["permission_id"]
        res = agg.aggregate(v, perms, "20990101", "20990102", BASELINE, BASELINE_SHA, FORMAL_OK)
        self.assertEqual(res["days"][0]["excluded"], "race_closed_at")
        self.assertEqual(res["reach"]["first_race_key"], "20990102_01_01")
        self.assertEqual(res["reach"]["hundredth_race_key"], "20990102_01_100")
        self.assertEqual(res["reach"]["hundredth_race_closed_at"], "2099-01-02 11:39:00")

    def test_v4_recorded_separately_from_baseline_failures(self):
        """V4：baseline_failures に入れず、input_order_failures に原因とともに記録する。"""
        res = run_with(["2099-01-01 10:00:00", "2099-01-01 10:01"], ["a", "b"])
        self.assertEqual(res["baseline_failures"], [])
        self.assertEqual([f["check"] for f in res["input_order_failures"]], ["race_closed_at"])
        self.assertIn("possible API format change", res["input_order_failures"][0]["problems"][0])

    def test_v4_exit_code_4_and_3_takes_precedence(self):
        import json
        from contextlib import redirect_stdout
        from io import StringIO
        from test_aggregate import BASELINE_RAW
        self.assertEqual(agg.EXIT_INPUT_ORDER_FAILED, 4)
        self.assertNotEqual(agg.EXIT_INPUT_ORDER_FAILED, agg.EXIT_VERIFICATION_FAILED)
        for with_verification_failure, want in ((False, 4), (True, 3)):
            with self.subTest(with_verification_failure=with_verification_failure), \
                    tempfile.TemporaryDirectory() as tmp:
                base = os.path.join(tmp, "a")
                os.makedirs(base)
                ta.CLOSED_AT_OVERRIDE = ["2099-01-01 10:00:00", "bad"]
                try:
                    _write_run(base, 1, 1, "20990101", [_race("a"), _race("b")], ["a", "b"])
                finally:
                    ta.CLOSED_AT_OVERRIDE = None
                if with_verification_failure:  # 別の実行の artifact の digest が合わない（照合の失敗＝3）
                    _write_run(base, 2, 1, "20990101", [_race("a")], ["a"], digest_ok=False,
                               started="2099-01-02T05:00:00Z")
                bp = os.path.join(tmp, "b.json")
                with open(bp, "wb") as f:
                    f.write(BASELINE_RAW)
                with ta.permitted_by(), redirect_stdout(StringIO()):
                    rc = agg.main(["--artifacts-dir", base, "--first-date", "20990101", "--last-date", "20990101",
                                   "--out", os.path.join(tmp, "o"), "--baseline", bp])
                with open(os.path.join(tmp, "o", "aggregate_result.json"), encoding="utf-8") as f:
                    res = json.load(f)
                self.assertEqual(rc, want)
                self.assertEqual(len(res["input_order_failures"]), 1)
                self.assertEqual(res["baseline_failures"], [])

    def test_takes_priority_over_config_detected(self):
        old = ta.CLOSED_AT_OVERRIDE
        ta.CLOSED_AT_OVERRIDE = ["2099-01-01 10:00:00", "2099-01-01 10:01"]
        try:
            with tempfile.TemporaryDirectory() as base:
                _write_run(base, 1, 1, "20990101", [_race("a"), _race("b")], ["a", "b"], config_detected=True)
                v = [agg.verify_run(os.path.join(base, x)) for x in sorted(os.listdir(base))]
        finally:
            ta.CLOSED_AT_OVERRIDE = old
        res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)
        self.assertEqual(res["days"][0]["excluded"], "race_closed_at")

    def test_stopped_run_not_applicable(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [], ["a"], run_overrides={"status": "stopped"})
            v = [agg.verify_run(os.path.join(base, x)) for x in sorted(os.listdir(base))]
        res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)
        self.assertEqual(res["days"][0].get("race_closed_at_check"), "not_applicable_stopped_run")


if __name__ == "__main__":
    unittest.main()
