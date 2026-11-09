"""段階5：P4 10・14・15・15-2（07f §6.4・§6.5 の 7。GPT判断：確定 11/04）。GPT 指示 2026-11-09。

不変条件：過去の正式実行記録・100件到達記録・過去の集約結果を上書き・削除しない。H1 と H2 を同一視しない。
② を正本に繰り上げない。① の記録を自動で 0 にしない。比較不能・順序不明を正常扱いにしない。
H1 の保留は到達区間の中だけ（W3）。H1・H2 は自動では解かない。
"""
import copy
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import test_aggregate as ta
from formal_g2.aggregate import main as agg
from test_aggregate import BASELINE_RAW, FORMAL_OK, _race, _write_run

H = "c" * 64


def R(label, cat="diff_zero", psha=H, b_L=H, chash="a" * 64):
    r = _race(label, cat)
    r["program_sha256"] = psha
    for pt in ("before", "after_rebuild", "after_v4"):
        r["hashes"][pt].update({"b_L": b_L, "d_cfg": H, "e_venue_stats": H, "e_venue_config": H, "a_boatinfo": H,
                                  "c_legacy_boats": H})
    r["comparable_hash"] = {"legacy": chash, "rebuild": chash}
    return r


PERMS2 = [dict(ta.PERMS[0], permission_id="P-A"), dict(ta.PERMS[0], permission_id="P-B")]


def run_main(tmp, runs, perms=PERMS2, extra=(), last="20990101"):
    """runs: (rid, att, races, targets, started, permission, mode, overrides)"""
    base = os.path.join(tmp, "a")
    os.makedirs(base, exist_ok=True)
    for rid, att, races, targets, started, pid, mode, over in runs:
        kw = {"permission": pid, "started": started}
        if mode:
            kw["mode"] = mode
        if over:
            kw["run_overrides"] = over
        _write_run(base, rid, att, "20990101", races, targets, **kw)
    bp = os.path.join(tmp, "b.json")
    with open(bp, "wb") as f:
        f.write(BASELINE_RAW)
    out = os.path.join(tmp, "out%d" % len(os.listdir(tmp)))
    real = agg.aggregate

    def ok_ctx(v, ps, f, l, b, bs, ctx, **kw):
        return real(v, ps, f, l, b, bs, FORMAL_OK, **kw)
    with ta.permitted_by(perms), mock.patch.object(agg, "aggregate", side_effect=ok_ctx), \
            redirect_stdout(StringIO()):
        rc = agg.main(["--artifacts-dir", base, "--first-date", "20990101", "--last-date", last, "--out", out,
                       "--baseline", bp, *extra])
    path = os.path.join(out, "aggregate_result.json")
    with open(path, encoding="utf-8") as f:
        return rc, json.load(f), path


T0, T1, T2 = "2099-01-02T00:00:00Z", "2099-01-02T03:00:00Z", "2099-01-02T05:00:00Z"
L3 = ["a", "b", "c"]


def first(races=None):
    return (1, 1, races or [R(x) for x in L3], L3, T0, "P-A", None, None)


def second(races, started=T1, rid=2, att=1, pid="P-A", mode=None, over=None):
    return (rid, att, races, L3, started, pid, mode, over)


def eid(n):
    return f"20990101_01_{n:02d}"


def states(res):
    return [x.get("state") for x in res["sequence"] if "eval_id" in x]


class ClassificationTests(unittest.TestCase):
    def test_second_formal_match_new_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [first(), second([R(x) for x in L3])])
        c = res["stage5"]["run_classification"][0]
        self.assertEqual(c["first_formal"], "1-1")
        self.assertEqual(c["second_formal"], [{"run": "2-1", "kind": "new_dispatch"}])
        ev = [e["event"] for e in res["stage5"]["events"]]
        self.assertIn("formal_processing_overlap", ev)
        self.assertEqual(res["doubts"], [])
        self.assertEqual(states(res), ["added"] * 3)  # ① の加算は変わらない（② は加えない）
        # Y1：同じ許可の2回目は手続き違反として記録し、それだけで判定を止めない
        v = res["permission_violations"][0]
        self.assertEqual((v["run"], v["status"], v["classification"]), ("2-1", "permission_already_used",
                                                                          "procedure_violation"))
        self.assertNotIn("achievement_halted", res)
        self.assertNotEqual(res["g2_status"], "halted_permission_violation")

    def test_y1_second_use_with_unknown_order_is_held_not_halted(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(agg, "_order_vs_reach", side_effect=lambda m, r: "unknown"
                                   if (m or {}).get("run_id") == "2" and (r or {}).get("run_id") == "1"
                                   else agg_order(m, r)):
                rc, res, _ = run_main(tmp, [first(), second([R(x) for x in L3])])
        self.assertNotIn("achievement_halted", res)
        self.assertEqual(res["g2_status"], "held_p4")
        self.assertIn("second use of a permission whose order is unknown (Y1)",
                      [h["reason"] for h in res["achievement_holds"]])

    def test_y3_two_permissions_same_date_is_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [first(), second([R(x) for x in L3], pid="P-B")])
        self.assertEqual(res["days"][0]["pending_reason"], "multiple_permissions_for_race_date")
        self.assertEqual(res["days"][0]["permissions"], ["P-A", "P-B"])
        self.assertEqual(states(res), [])
        self.assertIn("multiple_permissions_for_race_date", [e["event"] for e in res["stage5"]["events"]])

    def test_technical_retry_is_second_formal_not_investigation(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [first(), second([R(x) for x in L3], rid=1, att=2, pid="P-A")])
        c = res["stage5"]["run_classification"][0]
        self.assertEqual(c["second_formal"], [{"run": "1-2", "kind": "technical_retry"}])
        self.assertEqual(c["investigation"], [])

    def test_undetermined_formal_run_before_first(self):
        """Y4：未判定の実行がある日は ① を数えず保留する。"""
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [first(), second([R(x) for x in L3], started="2099-01-01T23:00:00Z",
                                                        pid="P-X")])
        c = res["stage5"]["run_classification"][0]
        self.assertEqual(c["first_formal"], "1-1")
        self.assertEqual([u["run"] for u in c["undetermined"]], ["2-1"])
        self.assertIn("run_class_undetermined", [e["event"] for e in res["stage5"]["events"]])
        self.assertEqual(res["days"][0]["pending_reason"], "undetermined_run_on_day")
        self.assertEqual(states(res), [])

    def test_y4_other_mode_is_undetermined(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [first(), second([R(x) for x in L3], mode="shadow", pid=None)])
        self.assertEqual(res["days"][0]["pending_reason"], "undetermined_run_on_day")
        self.assertEqual(states(res), [])

    def test_order_unknown_day_is_not_normal(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [first(), second([R(x) for x in L3], started=T0, pid="P-B")])
        self.assertEqual(res["days"][0]["pending_reason"], "formal_order_unknown")
        self.assertEqual(states(res), [])

    def test_attempt_2_can_be_first_formal(self):
        """① は run_attempt=1 であることだけでは決めない。"""
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [(5, 2, [R(x) for x in L3], L3, T0, "P-A", None, None)], perms=PERMS2[:1])
        self.assertEqual(res["stage5"]["run_classification"][0]["first_formal"], "5-2")
        self.assertEqual(states(res), ["added"] * 3)


class Comparison15Tests(unittest.TestCase):
    def _doubts(self, races2, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [first(), second(races2, **kw)])
        return res

    def test_15a_unfit_second_is_excluded_no_holds(self):
        res = self._doubts([R(x) for x in L3], over={"config_detection": {"detected": True}})
        cmp = res["stage5"]["comparisons"][0]
        self.assertEqual(cmp["run_unfit"], "config_detected")
        self.assertEqual(res["doubts"], [])
        self.assertEqual(res["p4_holds"]["H1"], [])
        self.assertEqual(res["p4_holds"]["H2"], [])
        self.assertIn("compared_run_unfit", [e["event"] for e in res["stage5"]["events"]])

    def test_15b_not_comparable_race_no_holds(self):
        res = self._doubts([R("a"), R("b", "not_established"), R("c")])
        races = res["stage5"]["comparisons"][0]["races"]
        self.assertEqual([(r["eval_id"], r["case"]) for r in races], [(eid(2), "15b_not_comparable")])
        self.assertEqual(res["p4_holds"]["H1"], [])

    def test_15c_p4a_h1_only(self):
        res = self._doubts([R("a"), R("b", "mismatch"), R("c")])
        self.assertEqual([(d["eval_id"], d["case"], d["H1"], d["H2"]) for d in res["doubts"]],
                         [(eid(2), "P4-A", True, False)])
        self.assertEqual(res["p4_holds"]["H1"], [eid(2)])
        self.assertEqual(res["p4_holds"]["H2"], [])
        self.assertEqual(states(res), ["added"] * 3)  # ① を自動で 0 にしない

    def test_15c_hash_difference_counts_as_mismatch(self):
        """race_state が同じでも comparable_hash が違えば一致しない（§6.4・§6.5 の 3）。"""
        res = self._doubts([R("a"), R("b", chash="d" * 64), R("c")])
        self.assertEqual([d["eval_id"] for d in res["doubts"]], [eid(2)])

    def test_15c_p4b_h2_only_if_second_mismatch(self):
        res = self._doubts([R("a"), R("b", b_L="e" * 64, chash="d" * 64), R("c")])
        self.assertEqual([(d["case"], d["H2"]) for d in res["doubts"]], [("P4-B", False)])
        res = self._doubts([R("a"), R("b", "mismatch", b_L="e" * 64), R("c")])
        self.assertEqual([(d["case"], d["H2"]) for d in res["doubts"]], [("P4-B", True)])
        self.assertEqual(res["g2_status"], "held_p4")
        self.assertIn("H2", res["achievement_holds"][0]["reason"])

    def test_15c_p4c_h1_and_h2(self):
        res = self._doubts([R("a"), R("b", "mismatch", b_L=None), R("c")])
        self.assertEqual([(d["case"], d["H1"], d["H2"]) for d in res["doubts"]], [("P4-C", True, True)])
        res = self._doubts([R("a"), R("b", "mismatch", b_L="UNSERIALIZABLE:x"), R("c")])
        self.assertEqual([d["case"] for d in res["doubts"]], ["P4-C"])

    def test_first_zero_race_difference_is_event_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [first([R("a"), R("b", "mismatch"), R("c")]),
                                        second([R("a"), R("b"), R("c")])])
        self.assertEqual(res["doubts"], [])
        races = res["stage5"]["comparisons"][0]["races"]
        self.assertEqual([(r["eval_id"], r["case"]) for r in races], [(eid(2), "event_only_first_not_added")])

    def test_second_is_not_promoted_when_first_unfit(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [(1, 1, [R(x) for x in L3], L3, T0, "P-A", None,
                                         {"config_detection": {"detected": True}}),
                                        second([R(x) for x in L3])])
        self.assertEqual(res["days"][0]["excluded"], "config_detected")
        self.assertEqual(states(res), [])
        self.assertEqual(res["best_segment"], 0)


class AuditFixTests(unittest.TestCase):
    """独立監査（2026-11-09、段階5）の指摘の修正。"""

    def test_stopped_second_is_15a(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [first(), second([R("a"), R("b", "mismatch")],
                                                        over={"status": "stopped"})])
        self.assertTrue(res["stage5"]["comparisons"][0]["run_unfit"].startswith("stopped"))
        self.assertEqual(res["doubts"], [])

    def test_second_failing_run_level_checks_is_15a(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [first(), second([R("a"), R("b", "mismatch"), R("c")],
                                                        over={"preload": {"ok": False}})])
        self.assertIsNotNone(res["stage5"]["comparisons"][0]["run_unfit"])
        self.assertEqual(res["doubts"], [])

    def test_malformed_compared_run_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(agg, "_states_of", side_effect=KeyError("eval_id")):
                rc, res, _ = run_main(tmp, [first(), second([R(x) for x in L3])])
        self.assertTrue(res["stage5"]["comparisons"][0]["run_unfit"].startswith("check_error"))
        self.assertEqual(states(res), ["added"] * 3)

    def test_investigation_off_main_still_compared(self):
        """③ は正式の関門に関わらず比べる（関門は記録だけ）。"""
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [first(), second([R("a"), R("b", "mismatch"), R("c")], mode="investigation",
                                                        pid=None, over={"preload": {"ok": False}})])
        cmp = res["stage5"]["comparisons"][0]
        self.assertIsNone(cmp["run_unfit"])
        self.assertIs(cmp["formal_gates_observed"]["run_level_ok"], False)
        self.assertEqual([d["eval_id"] for d in res["doubts"]], [eid(2)])

    def test_unavailable_second_after_first_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            d = _write_run(base, 9, 1, "20990101", [R("a")], ["a"], started=T2)
            m = json.load(open(os.path.join(d, "meta.json")))
            m.update(artifact_status="expired", artifact_name="g2-20990101-9-1")
            json.dump(m, open(os.path.join(d, "meta.json"), "w"))
            os.remove(os.path.join(d, "artifact.zip"))
            rc, res, _ = run_main(tmp, [first()])
        ev = [e for e in res["stage5"]["events"] if e["event"] == "compared_run_unfit"]
        self.assertEqual([(e["run"], e["reason"]) for e in ev], [("9-1", "unavailable (retention_expired)")])
        self.assertEqual(states(res), ["added"] * 3)

agg_order = agg._order_vs_reach


class Y5Tests(unittest.TestCase):
    def _res(self, races2):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [first(), second(races2, mode="investigation", pid=None)])
        return res

    def test_race_level_not_comparable_is_event_only(self):
        res = self._res([R("a"), R("b", "not_established"), R("c")])
        self.assertEqual(res["doubts"], [])
        self.assertEqual([r["case"] for r in res["stage5"]["comparisons"][0]["races"]],
                         ["investigation_race_not_comparable"])

    def test_input_undetermined_is_p4c(self):
        res = self._res([R("a"), R("b", "not_established", b_L=None), R("c")])
        self.assertEqual([(d["case"], d["H1"], d["H2"]) for d in res["doubts"]], [("P4-C", True, True)])

    def test_record_missing_is_held_as_insufficient(self):
        res = self._res([R("a"), R("c")])
        self.assertEqual(res["doubts"], [])
        self.assertEqual(res["stage5"]["evidence_insufficient"][0]["eval_id"], eid(2))
        self.assertEqual(res["g2_status"], "held_p4")


class InvestigationTests(unittest.TestCase):
    def _res(self, races2):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(tmp, [first(), second(races2, mode="investigation", pid=None)])
        return res

    def test_p4a_p4b_p4c(self):
        self.assertEqual([(d["source"], d["case"], d["H2"]) for d in self._res(
            [R("a"), R("b", "mismatch"), R("c")])["doubts"]], [("P4-8", "P4-A", False)])
        self.assertEqual([(d["case"], d["H2"]) for d in self._res(
            [R("a"), R("b", b_L="e" * 64, chash="d" * 64), R("c")])["doubts"]], [("P4-B", True)])
        self.assertEqual([(d["case"], d["H2"]) for d in self._res(
            [R("a"), R("b", "mismatch", b_L=None), R("c")])["doubts"]], [("P4-C", True)])

    def test_no_difference_no_doubt_and_no_count_change(self):
        res = self._res([R(x) for x in L3])
        self.assertEqual(res["doubts"], [])
        self.assertEqual(res["stage5"]["run_classification"][0]["investigation"][0]["run"], "2-1")
        self.assertEqual(states(res), ["added"] * 3)


class ReachAndHoldTests(unittest.TestCase):
    def _reach(self, tmp, diff_n, extra=()):
        from test_streak_reach import _inst
        labels = [f"r{i:03d}" for i in range(1, 106)]
        inst = {"instrument": _inst(105)}
        r1 = [R(x) for x in labels]
        r2 = [R(x, "mismatch") if i == diff_n else R(x) for i, x in enumerate(labels, 1)]
        return run_main(tmp, [(1, 1, r1, labels, T0, "P-A", None, inst),
                              (2, 1, r2, labels, T1, "P-A", None, inst)], extra=extra)

    def test_h1_inside_reached_segment_holds_reach_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = self._reach(tmp, 50)
        self.assertEqual(res["reach"]["hundredth_race_key"], eid(100))
        self.assertEqual(res["reach"]["status"], "reached_not_yet_formally_achieved")
        self.assertEqual(res["best_segment"], 100)
        self.assertEqual(res["g2_status"], "held_p4")
        self.assertEqual(res["achievement_holds"][0]["eval_ids"], [eid(50)])

    def test_h1_outside_reached_segment_is_record_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = self._reach(tmp, 103)
        self.assertEqual(res["p4_holds"]["H1"], ["20990101_01_103"])
        self.assertEqual(res["achievement_holds"], [])
        self.assertEqual(res["g2_status"], "reached_100_unconfirmed")

    def test_holds_are_carried_and_not_auto_released(self):
        """前回の正本の H1・H2 は、今回の入力で疑義が出なくても引き継ぐ（承認された解除だけが解く）。"""
        from test_streak_reach import _inst
        labels = [f"r{i:03d}" for i in range(1, 106)]
        with tempfile.TemporaryDirectory() as tmp:
            inst = {"instrument": _inst(105)}
            r2 = [R(x, "mismatch", b_L=None) if i == 50 else R(x) for i, x in enumerate(labels, 1)]
            rc, res1, p1 = run_main(tmp, [(1, 1, [R(x) for x in labels], labels, T0, "P-A", None, inst),
                                          (2, 1, r2, labels, T1, "P-A", None, inst)])
            self.assertEqual([(d["case"], d["H1"], d["H2"]) for d in res1["doubts"]], [("P4-C", True, True)])
            repo = os.path.join(tmp, "repo")
            os.makedirs(repo)
            rel, sha = ta.canonical_previous(os.path.dirname(p1), repo)
            ev_before = open(os.path.join(repo, "g2_evidence/_aggregate/events/h1_events.jsonl"), "rb").read()
            base = os.path.join(tmp, "only1")
            os.makedirs(base)
            _write_run(base, 1, 1, "20990101", [R(x) for x in labels], labels, permission="P-A", started=T0,
                       run_overrides=inst)
            res2 = self._main_with_state(tmp, base, repo, rel, sha)
            ev_after = open(os.path.join(repo, "g2_evidence/_aggregate/events/h1_events.jsonl"), "rb").read()
        self.assertEqual(ev_before, ev_after)  # 過去のイベントを変えない
        self.assertEqual(res2["doubts"], [])
        self.assertEqual(res2["p4_holds"]["H1"], [eid(50)])
        self.assertEqual(res2["p4_holds"]["carried_from_previous"]["H1"], res1["p4_holds"]["H1_event_ids"])
        self.assertEqual([h["eval_id"] for h in res2["p4_holds"]["H2"]], [eid(50)])
        self.assertEqual(res2["g2_status"], "held_p4")
        self.assertEqual(res2["reaggregation"]["previous_reach"], res1["reach"])  # 到達の記録は不変
        self.assertEqual(res2["reaggregation"]["previous_result_source"]["path"], rel)
        reasons = [h["reason"] for h in res2["achievement_holds"]]
        self.assertEqual(len(reasons), len(set(reasons)))  # 同じ事実を二重に数えない
        self.assertIn("H2 (P4 11, 15-2): whole G2 achievement judgment held", reasons)
        self.assertEqual(res2["h1_events_new"], [])  # 同じ疑義を2回は加えない

    def _main_with_state(self, tmp, base, repo, rel, sha):
        bp = os.path.join(tmp, "b.json")
        real = agg.aggregate

        def ok_ctx(v, ps, f, l, b, bs, ctx, **kw):
            return real(v, ps, f, l, b, bs, FORMAL_OK, **kw)
        out = os.path.join(tmp, "o2")
        with ta.permitted_by(PERMS2, genesis=False), mock.patch.object(agg, "aggregate", side_effect=ok_ctx), \
                mock.patch.object(agg, "_repo_root", return_value=repo), \
                mock.patch.dict(os.environ, {"GITHUB_RUN_ID": "2"}), redirect_stdout(StringIO()):
            agg.main(["--artifacts-dir", base, "--first-date", "20990101", "--last-date", "20990101",
                      "--out", out, "--baseline", bp, "--previous-state", rel, "--previous-state-sha256", sha])
        with open(os.path.join(out, "aggregate_result.json"), encoding="utf-8") as f:
            return json.load(f)


class ExitStatusTests(unittest.TestCase):
    def test_exit_status_records_both_causes(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            ta.CLOSED_AT_OVERRIDE = ["2099-01-01 10:00:00", "bad"]
            try:
                _write_run(base, 1, 1, "20990101", [R("a"), R("b")], ["a", "b"])
            finally:
                ta.CLOSED_AT_OVERRIDE = None
            _write_run(base, 2, 1, "20990101", [R("a")], ["a"], digest_ok=False, started="2099-01-02T05:00:00Z")
            bp = os.path.join(tmp, "b.json")
            with open(bp, "wb") as f:
                f.write(BASELINE_RAW)
            with ta.permitted_by(), redirect_stdout(StringIO()):
                rc = agg.main(["--artifacts-dir", base, "--first-date", "20990101", "--last-date", "20990101",
                               "--out", os.path.join(tmp, "o"), "--baseline", bp])
            with open(os.path.join(tmp, "o", "aggregate_result.json"), encoding="utf-8") as f:
                res = json.load(f)
        self.assertEqual(rc, 3)
        es = res["exit_status"]
        self.assertEqual(es["code"], 3)
        self.assertTrue(es["verification_failed"] and es["input_order_failed"])
        self.assertEqual(es["input_order_causes"][0]["check"], "race_closed_at")
        self.assertEqual(es["input_order_causes"][0]["run"], "1")
        self.assertTrue(es["input_order_causes"][0]["problems"])


class InputCaseTests(unittest.TestCase):
    def test_id_keys_are_not_compared(self):
        r1, r2 = R("a"), R("a")
        r1["hashes"]["before"]["e_venue_stats_id"] = 1
        self.assertEqual(agg.input_case(R("a"), R("a"))[0], "A")
        r2["hashes"]["before"]["e_venue_stats_id"] = 2
        self.assertEqual(agg.input_case(r1, r2), ("A", []))

    def test_cases(self):
        self.assertEqual(agg.input_case(R("a"), R("a", b_L="e" * 64)), ("B", ["b_L"]))
        self.assertEqual(agg.input_case(R("a", psha=None), R("a"))[0], "C")
        r0 = R("a")
        r0["hashes"]["before"]["c_legacy_boats"] = "e" * 64
        self.assertEqual(agg.input_case(R("a"), r0), ("B", ["c_legacy_boats"]))
        r = R("a")
        r.pop("hashes")
        self.assertEqual(agg.input_case(R("a"), r)[0], "C")


if __name__ == "__main__":
    unittest.main()
