"""P4 4・5（07f §6.4・§6.5 の 6。GPT判断 11/04・11/05。段階4、GPT 指示 2026-11-09）。

4：artifact の保存の失敗、正本の保存の失敗、証拠の欠落、保持期限切れを、別々の事実として記録する。
5：同じ run_id・同じ成果物の SHA-256 での集約のやり直しは、評価のやり直しではない。結果が違えば、
   先の結果を上書きせずに別記録として残し、P4-D の疑義として H1 を付ける（前回 +1 のレースに限る）。
   H2 は自動で付けない。正式実行記録を自動で 0 にしない。100件目の到達の記録は変えない。
"""
import hashlib
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


def _meta(d):
    with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
        return json.load(f)


def _set_meta(d, **kw):
    m = _meta(d)
    m.update(kw)
    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(m, f)


class RecordFactTests(unittest.TestCase):
    def test_artifact_status_facts(self):
        for status, fact in (("missing", "artifact_save_failure"), ("expired", "retention_expired"),
                             ("download_failed", "artifact_download_failed")):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as base:
                d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
                _set_meta(d, artifact_status=status, artifact_expires_at="2099-04-01T00:00:00Z")
                os.remove(os.path.join(d, "artifact.zip"))
                v = agg.verify_run(d)
                self.assertFalse(v["verified"])
                self.assertEqual(v["record_fact"], fact)
                self.assertEqual(agg.record_facts([v])[0]["fact"], fact)

    def test_unknown_status_with_zip_present_is_not_verified(self):
        with tempfile.TemporaryDirectory() as base:
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            _set_meta(d, artifact_status="weird")
            v = agg.verify_run(d)
        self.assertFalse(v["verified"])
        self.assertEqual(v["record_fact"], "evidence_missing")

    def test_marker_dir_without_attempt(self):
        with tempfile.TemporaryDirectory() as base:
            d = os.path.join(base, "77-none")
            os.makedirs(d)
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump({"run_id": "77", "run_attempt": None, "latest_attempt": "1", "artifact_status": "missing"}, f)
            v = agg.verify_run(d)
        self.assertEqual(agg.record_facts([v]), [{"run": "77-none", "run_id": "77", "run_attempt": None,
                                                 "fact": "artifact_save_failure", "artifact_expires_at": None}])

    def test_evidence_missing_kinds(self):
        with tempfile.TemporaryDirectory() as base:
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], complete=False)
            self.assertEqual(agg.verify_run(d)["record_fact"], "evidence_missing")
        with tempfile.TemporaryDirectory() as base:
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            os.remove(os.path.join(d, "artifact.zip"))
            self.assertEqual(agg.verify_run(d)["record_fact"], "evidence_missing")
        with tempfile.TemporaryDirectory() as base:
            d = os.path.join(base, "1-1")
            os.makedirs(d)
            with open(os.path.join(d, "meta.json"), "w") as f:
                f.write("{broken")
            self.assertEqual(agg.verify_run(d)["record_fact"], "evidence_missing")

    def test_digest_mismatch_is_not_one_of_the_four_facts(self):
        """digest の不一致は改ざん・取り違えの照合の失敗で、4つの事実のどれでもない（照合の失敗として 3）。"""
        with tempfile.TemporaryDirectory() as base:
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], digest_ok=False)
            v = agg.verify_run(d)
        self.assertFalse(v["verified"])
        self.assertNotIn("record_fact", v)

    def test_old_meta_without_status_is_downloaded(self):
        with tempfile.TemporaryDirectory() as base:
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            m = _meta(d)
            m.pop("artifact_status", None)
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump(m, f)
            self.assertTrue(agg.verify_run(d)["verified"])


def run_main(base, tmp, extra=(), first="20990101", last="20990101", repo=None):
    bp = os.path.join(tmp, "b.json")
    with open(bp, "wb") as f:
        f.write(BASELINE_RAW)
    out = os.path.join(tmp, "out-" + hashlib.sha1(repr(extra).encode()).hexdigest()[:6] + first + last)
    real = agg.aggregate

    def ok_ctx(v, perms, f, l, b, bs, ctx, **kw):
        return real(v, perms, f, l, b, bs, FORMAL_OK, **kw)
    import contextlib
    with ta.permitted_by(genesis=repo is None), mock.patch.object(agg, "aggregate", side_effect=ok_ctx), \
            (mock.patch.object(agg, "_repo_root", return_value=repo) if repo else contextlib.nullcontext()), \
            redirect_stdout(StringIO()):
        rc = agg.main(["--artifacts-dir", base, "--first-date", first, "--last-date", last, "--out", out,
                       "--baseline", bp, *extra])
    path = os.path.join(out, "aggregate_result.json")
    with open(path, encoding="utf-8") as f:
        return rc, json.load(f), path


def _streak_run(base, n=100, rid=1, att=1, cats=None, started="2099-01-02T00:00:00Z"):
    from test_streak_reach import _inst
    labels = [f"r{i:03d}" for i in range(1, n + 1)]
    races = [_race(x, (cats or {}).get(i, "diff_zero")) for i, x in enumerate(labels, 1)]
    return _write_run(base, rid, att, "20990101", races, labels, run_overrides={"instrument": _inst(n)},
                      started=started)


class MainRecordTests(unittest.TestCase):
    def test_missing_artifact_run_is_recorded_and_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            d = os.path.join(base, "2-none")
            os.makedirs(d)
            with open(os.path.join(d, "meta.json"), "w") as f:
                json.dump({"run_id": "2", "run_attempt": None, "artifact_status": "missing"}, f)
            rc, res, _ = run_main(base, tmp)
        self.assertEqual(rc, 3)
        self.assertEqual([x["fact"] for x in res["record_facts"]], ["artifact_save_failure"])
        self.assertEqual(res["canonical_save"]["reached_segment_runs"], [])
        self.assertIn("2-none", res["verification_failed_runs"])

    def test_input_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            _write_run(base, 2, 1, "20990101", [_race("a")], ["a"])
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            rc, res, _ = run_main(base, tmp)
        ii = res["input_identity"]
        self.assertEqual((ii["first_date"], ii["last_date"]), ("20990101", "20990101"))
        self.assertEqual([r["run_id"] for r in ii["runs"]], ["1", "2"])
        self.assertTrue(all(len(r["artifact_sha256"]) == 64 for r in ii["runs"]))


class ReaggregationTests(unittest.TestCase):
    def _first_and_second(self, change=None, period=None, previous_mutator=None):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            repo = os.path.join(tmp, "repo")
            os.makedirs(repo)
            _streak_run(base, 100)
            rc1, first, path1 = run_main(base, tmp)
            out1 = os.path.dirname(path1)
            rel, sha = ta.canonical_previous(out1, repo, mutate_result=previous_mutator)
            if previous_mutator:
                previous_mutator(first)
            with open(os.path.join(repo, *rel.split("/")), "rb") as f:
                before = f.read()
            extra = ("--previous-state", rel, "--previous-state-sha256", sha)
            with mock.patch.dict(os.environ, {"GITHUB_RUN_ID": "2"}):
                if change:
                    with mock.patch.object(agg, "race_state", side_effect=change):
                        rc2, second, _ = run_main(base, tmp, extra, *(period or ()), repo=repo)
                else:
                    rc2, second, _ = run_main(base, tmp, extra, *(period or ()), repo=repo)
            with open(os.path.join(repo, *rel.split("/")), "rb") as f:
                after = f.read()
        self.assertEqual(before, after, "前回の正本を変えない")
        self.assertEqual(second["previous_state"]["status"], "ok", second["previous_state"])
        return first, second

    def test_same_input_same_result(self):
        first, second = self._first_and_second()
        r = second["reaggregation"]
        self.assertEqual(r["comparison"], "same_input")
        self.assertEqual(r["doubts"], [])
        self.assertTrue(r["reach_same"])
        self.assertEqual(second["p4_holds"]["H1"], [])
        self.assertEqual(r["previous_reach"], first["reach"])

    def test_same_input_different_result_gives_p4d_h1_and_keeps_reach(self):
        real = agg.race_state

        def change(rec, ok):
            st = real(rec, ok)
            return "mismatch" if rec.get("eval_id") == "20990101_01_50" else st
        first, second = self._first_and_second(change=change)
        r = second["reaggregation"]
        self.assertEqual(r["comparison"], "same_input")
        self.assertEqual([(d["eval_id"], d["case"], d["hold"]) for d in r["doubts"]],
                         [("20990101_01_50", "P4-D", "H1")])
        self.assertEqual(second["p4_holds"]["H1"], ["20990101_01_50"])
        self.assertEqual(second["p4_holds"]["H2"], [])  # H2 は自動で付けない
        # 100件目の到達の記録（前回）は不変で、そのまま写す。今回の結果で置き換えない
        self.assertEqual(first["reach"]["hundredth_race_key"], "20990101_01_100")
        self.assertEqual(r["previous_reach"], first["reach"])
        self.assertFalse(r["reach_same"])
        self.assertIsNone(second["reach"])  # 今回の計算では到達しない（別記録として残る）
        self.assertEqual(second["g2_status"], "held_p4")
        self.assertEqual([h["reason"].split(" (")[0] for h in second["achievement_holds"]],
                         ["H1 doubt inside the reached segment", "reach differs from the previous aggregate result"])

    def test_previous_zero_race_change_is_event_only(self):
        def prev_mut(first):
            for x in first["sequence"]:
                if x.get("eval_id") == "20990101_01_07":
                    x["state"] = "mismatch"
        first, second = self._first_and_second(previous_mutator=prev_mut)
        r = second["reaggregation"]
        self.assertEqual(r["doubts"], [])
        self.assertEqual(r["events_only"], ["20990101_01_07"])

    def test_period_change_is_recorded_not_p4d(self):
        first, second = self._first_and_second(period=("20990101", "20990102"))
        r = second["reaggregation"]
        self.assertEqual(r["comparison"], "period_changed")
        self.assertEqual(r["doubts"], [])
        self.assertTrue(r["reach_same"])
        self.assertEqual(second["achievement_holds"], [])

    def test_doubt_is_per_run_even_if_period_changed(self):
        """同じ run・同じ成果物のレースの結果が違えば、期間が変わっていても P4-D・H1（監査の指摘3）。"""
        real = agg.race_state

        def change(rec, ok):
            st = real(rec, ok)
            return "mismatch" if rec.get("eval_id") == "20990101_01_10" else st
        first, second = self._first_and_second(change=change, period=("20990101", "20990102"))
        r = second["reaggregation"]
        self.assertEqual(r["comparison"], "period_changed")
        self.assertEqual([d["eval_id"] for d in r["doubts"]], ["20990101_01_10"])
        self.assertEqual(second["g2_status"], "held_p4")

    def test_different_input(self):
        def prev_mut(first):
            first["input_identity"]["runs"][0]["artifact_sha256"] = "0" * 64
        first, second = self._first_and_second(previous_mutator=prev_mut)
        self.assertEqual(second["reaggregation"]["comparison"], "different_input")
        self.assertEqual(second["reaggregation"]["doubts"], [])
        self.assertTrue(second["reaggregation"]["reach_same"])

    def test_extra_run_does_not_hide_doubt(self):
        """ほかの実行が増えても（different_input）、変わらない実行のレースの違いは P4-D・H1。"""
        real = agg.race_state

        def change(rec, ok):
            st = real(rec, ok)
            return "mismatch" if rec.get("eval_id") == "20990101_01_20" else st

        def prev_mut(first):
            first["input_identity"]["runs"].append({"run_id": "9", "run_attempt": "1", "artifact_sha256": "e" * 64})
        first, second = self._first_and_second(change=change, previous_mutator=prev_mut)
        r = second["reaggregation"]
        self.assertEqual(r["comparison"], "different_input")
        self.assertEqual([d["eval_id"] for d in r["doubts"]], ["20990101_01_20"])
        self.assertEqual(r["previous_reach"], first["reach"])  # 100件目の到達の記録は不変
        self.assertEqual(second["g2_status"], "held_p4")

    def test_changed_artifact_is_event_only_but_reach_change_holds(self):
        real = agg.race_state

        def change(rec, ok):
            st = real(rec, ok)
            return "mismatch" if rec.get("eval_id") == "20990101_01_30" else st

        def prev_mut(first):
            first["input_identity"]["runs"][0]["artifact_sha256"] = "0" * 64
        first, second = self._first_and_second(change=change, previous_mutator=prev_mut)
        r = second["reaggregation"]
        self.assertEqual(r["doubts"], [])
        self.assertIn("20990101_01_30", r["events_only"])
        self.assertEqual([h["reason"].split(" (")[0] for h in second["achievement_holds"]],
                         ["reach differs from the previous aggregate result"])

    def test_aggregator_version_difference_recorded(self):
        def prev_mut(first):
            first["aggregator_version"] = {"rebuild/formal_g2/aggregate/main.py": "0" * 64}
        first, second = self._first_and_second(previous_mutator=prev_mut)
        self.assertFalse(second["reaggregation"]["aggregator_version"]["same"])

    def test_previous_unusable_or_unreadable(self):
        self.assertEqual(agg.compare_with_previous({}, {"input_identity": {}})["comparison"], "previous_unusable")
        self.assertEqual(agg.compare_with_previous([], {"input_identity": {}})["comparison"], "previous_unusable")
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            for extra in ((), ("--previous-state", "g2_evidence/_aggregate/aggregations/x/manifest.json",
                               "--previous-state-sha256", "0" * 64)):
                rc, res, _ = run_main(base, tmp, extra, repo=tmp)
                self.assertNotEqual(res["previous_state"]["status"], "ok")
                self.assertEqual(res["g2_status"], "held_p4")
            with self.assertRaises(SystemExit), redirect_stdout(StringIO()), mock.patch("sys.stderr", new=StringIO()):
                run_main(base, tmp, ("--previous-result", "x.json"), repo=tmp)


class UnavailableRunDateTests(unittest.TestCase):
    """P4 4：artifact を確かめられない実行の対象レースは 0。あとから別の実行で評価し直して連続数に入れない。"""

    def _base(self, tmp, status="expired", name="g2-20990101-1-1", started="2099-01-02T00:00:00Z"):
        base = os.path.join(tmp, "a")
        os.makedirs(base)
        d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], started=started)
        _set_meta(d, artifact_status=status, artifact_name=name)
        os.remove(os.path.join(d, "artifact.zip"))
        _write_run(base, 2, 1, "20990101", [_race("a")], ["a"], started="2099-01-02T03:00:00Z")
        return base

    def test_earlier_unavailable_run_blocks_the_day(self):
        for status in ("expired", "download_failed"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                rc, res, _ = run_main(self._base(tmp, status), tmp)
                self.assertEqual(res["days"][0]["excluded"], "artifact_unavailable")
                self.assertEqual([x for x in res["sequence"] if "eval_id" in x], [])
                self.assertEqual(rc, 3)

    def test_later_unavailable_run_does_not_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(self._base(tmp, started="2099-01-02T05:00:00Z"), tmp)
        self.assertEqual([x.get("state") for x in res["sequence"] if "eval_id" in x], ["added"])

    def test_same_start_is_unknown_and_blocks(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = run_main(self._base(tmp, started="2099-01-02T03:00:00Z"), tmp)
        self.assertEqual(res["days"][0]["excluded"], "artifact_unavailable")

    def test_unattributable_run_holds_judgment(self):
        for name in (None, "g2-x", "g2-20990101-9-1", "g2-20990231-1-1"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                rc, res, _ = run_main(self._base(tmp, "missing", name=name), tmp)
                self.assertEqual(res["unattributed_unavailable_runs"], ["1-1"])
                self.assertEqual(res["g2_status"], "held_p4")
                self.assertIn("P4 4", res["achievement_holds"][0]["reason"])

    def test_unavailable_run_date_parsing(self):
        f = agg.unavailable_run_date
        self.assertEqual(f({"meta": {"artifact_name": "g2-20990101-5-2", "run_id": "5", "run_attempt": "2"}}),
                         "20990101")
        self.assertIsNone(f({"meta": {"artifact_name": "g2-20990101-5-2", "run_id": "5", "run_attempt": "1"}}))
        self.assertIsNone(f({"manifest": {}, "meta": {"artifact_name": "g2-20990101-5-2", "run_id": "5",
                                                     "run_attempt": "2"}}))


class SubsequentRunTests(unittest.TestCase):
    def test_difference_with_later_investigation_run_is_recorded_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            _write_run(base, 1, 1, "20990101", [_race("a"), _race("b")], ["a", "b"])
            _write_run(base, 2, 1, "20990101", [_race("a"), _race("b", "mismatch")], ["a", "b"],
                       mode="investigation", started="2099-01-02T05:00:00Z")
            rc, res, _ = run_main(base, tmp)
        self.assertEqual([x.get("state") for x in res["sequence"] if "eval_id" in x], ["added", "added"])
        self.assertEqual(len(res["subsequent_run_differences"]), 1)
        d = res["subsequent_run_differences"][0]
        self.assertEqual((d["eval_id"], d["other_run"], d["other_mode"], d["order_vs_selected"],
                          d["selected_category"], d["other_category"]),
                         ("20990101_01_02", "2-1", "investigation", "after", "diff_zero", "mismatch"))
        # 段階5（P4 8・10）：① で +1 のレースに調査用の実行との差分があるので疑義（H1）。① の記録は変えない
        self.assertEqual(res["p4_holds"]["H1"], ["20990101_01_02"])


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        with open(".github/workflows/formal_g2_aggregate.yml", encoding="utf-8") as f:
            self.text = f.read()

    def test_artifact_status_and_previous_result(self):
        t = self.text
        self.assertIn('artifact_status:"missing"', t)
        self.assertIn("artifact_status=expired", t)
        self.assertIn("artifact_status=download_failed", t)
        self.assertIn('elif ! gh api "repos/${REPO}/actions/artifacts/${aid}/zip" > "$d/artifact.zip"; then', t)
        self.assertIn('done <<<"$arts"', t)
        self.assertIn("IFS=$'\\x1f' read -r aid aname adigest aexpired aexpires", t)
        self.assertNotIn("@tsv", t)
        self.assertIn('for a in $(seq 1 "$attempt"); do', t)
        self.assertNotIn("previous_result_path", t)
        self.assertIn('args+=(--previous-state "$PREVIOUS_STATE_PATH" --previous-state-sha256 "$PREVIOUS_STATE_SHA256")', t)
        i = t.index("- name: Aggregate")
        self.assertNotIn("GH_TOKEN", t[i:])


if __name__ == "__main__":
    unittest.main()
