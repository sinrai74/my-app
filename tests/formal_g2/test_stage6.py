"""段階6（d5）：正本の束・保存・検証、前回の正本の必須の読み込み（Y6）、H1・H2 の解除イベント（GPT 指示 2026-11-09）。

テストは一時ディレクトリをリポジトリに見立てる。実際の g2_evidence/・g2_permissions/ には触れない。
実際の解除イベントは作らない（承認の確認の方法は未決定なので、既定ではどの解除も有効にならない）。
"""
import glob
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import test_aggregate as ta
from formal_g2.aggregate import canonical as cn
from formal_g2.aggregate import main as agg
from test_aggregate import BASELINE_RAW, FORMAL_OK, _write_run
from test_stage5 import PERMS2, R, T0, T1

LABELS = [f"r{i:03d}" for i in range(1, 106)]


def eid(n):
    return f"20990101_01_{n:02d}"


def write_genesis(repo):
    g = {"schema_version": 1, "kind": "g2_state_genesis", "state_version_id": "genesis", "h1_active": [],
         "h2_active": [], "reach": None}
    p = os.path.join(repo, *cn.GENESIS.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    data = cn._dumps(g)
    open(p, "wb").write(data)
    return cn.GENESIS, cn.sha256(data)


def make_runs(base, diff=None):
    from test_streak_reach import _inst
    inst = {"instrument": _inst(len(LABELS))}
    _write_run(base, 1, 1, "20990101", [R(x) for x in LABELS], LABELS, permission="P-A", started=T0,
               run_overrides=inst)
    if diff:
        n, kw = diff
        r2 = [R(x, **kw) if i == n else R(x) for i, x in enumerate(LABELS, 1)]
        _write_run(base, 2, 1, "20990101", r2, LABELS, permission="P-A", started=T1, run_overrides=inst)


def run(tmp, base, repo, prev, run_id, verifier=None, git_ok=True):
    bp = os.path.join(tmp, "b.json")
    open(bp, "wb").write(BASELINE_RAW)
    out = os.path.join(tmp, f"out{run_id}")
    real = agg.aggregate

    def ok_ctx(v, ps, f, l, b, bs, ctx, **kw):
        return real(v, ps, f, l, b, bs, FORMAL_OK, **kw)
    args = ["--artifacts-dir", base, "--first-date", "20990101", "--last-date", "20990101", "--out", out,
            "--baseline", bp]
    if prev is not None:
        args += ["--previous-state", prev[0], "--previous-state-sha256", prev[1]]
    patches = [ta.permitted_by(PERMS2, genesis=False, git_ok=git_ok), mock.patch.object(agg, "aggregate", side_effect=ok_ctx),
               mock.patch.object(agg, "_repo_root", return_value=repo),
               # 登録（案A）は別のテストで確かめる。ここでは正本の保存の確認だけを見る
               mock.patch.object(agg, "registration_check", return_value={"status": "ok", "problems": [],
                                                                          "entries": {}, "registry": "x",
                                                                          "registry_sha256": None}),
               mock.patch.dict(os.environ, {"GITHUB_RUN_ID": str(run_id), "GITHUB_RUN_ATTEMPT": "1"})]
    if verifier is not None:
        patches.append(mock.patch.object(agg, "_approval_verifier", side_effect=verifier))
    import contextlib
    with contextlib.ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        with redirect_stdout(StringIO()):
            rc = agg.main(args)
    res = json.load(open(os.path.join(out, "aggregate_result.json"), encoding="utf-8"))
    return rc, res, out


def setup_repo(tmp):
    repo = os.path.join(tmp, "repo")
    os.makedirs(repo)
    return repo, write_genesis(repo)


class CanonicalSaveTests(unittest.TestCase):
    def _first(self, tmp, diff=None):
        repo, gen = setup_repo(tmp)
        base = os.path.join(tmp, "a")
        os.makedirs(base)
        make_runs(base, diff)
        rc, res, out = run(tmp, base, repo, gen, 1)
        return repo, base, res, out

    def test_normal_save_readback_and_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res, out = self._first(tmp)
            st = cn.stage(os.path.join(out, "canonical_bundle"), repo)
            self.assertEqual(st["status"], "ok", st)
            self.assertTrue(st["written"])
            self.assertTrue(all(m["status"] == "ok" for m in st["manifests"]))
            # 評価の実行の正本：artifact の元のバイト列と同じ SHA-256
            man = json.load(open(os.path.join(repo, "g2_evidence/P-A/1-1/manifest.json")))
            art = [e for e in man["files"] if e["path"].endswith("artifact.zip")][0]
            raw = open(os.path.join(base, "1-1", "artifact.zip"), "rb").read()
            self.assertEqual(art["sha256"], cn.sha256(raw))
            ev = json.load(open(os.path.join(repo, "g2_evidence/P-A/1-1/evaluation_evidence.json")))
            for k in ("race_date", "permission_id", "evaluation_run_id", "run_attempt", "run_started_at",
                      "workflow", "ref", "head_sha", "artifact", "provenance"):
                self.assertIn(k, ev)
            # 集約ごとに変わりうる値（実行区分・照合の結果・基準・取得の時刻）は集約の正本に記録する（監査 H-2）
            inp = json.load(open(glob.glob(os.path.join(repo, "g2_evidence/_aggregate/aggregations/*/input_manifest.json"))[0]))
            r11 = [r for r in inp["evaluation_runs"] if r["run"] == "1-1"][0]
            self.assertEqual(r11["classification"], "first_formal")
            for k in ("verified", "problems", "baseline", "fetch", "artifact_sha256"):
                self.assertIn(k, r11)
            am = json.load(open(os.path.join(repo, "g2_evidence/P-A/1-1/artifact_manifest.json")))
            self.assertTrue(am["readable"] and am["files"])
            amg = json.load(open(glob.glob(os.path.join(repo, "g2_evidence/_aggregate/aggregations/*/manifest.json"))[0]))
            for k in ("aggregate_run", "commit_sha", "aggregate_result", "snapshot", "inputs", "result_sha256",
                      "reach", "best_segment", "h1_active", "h2_active", "files", "append_only_prefixes"):
                self.assertIn(k, amg)

    def test_identical_resave_and_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res, out = self._first(tmp)
            bundle = os.path.join(out, "canonical_bundle")
            self.assertEqual(cn.stage(bundle, repo)["status"], "ok")
            again = cn.stage(bundle, repo)
            self.assertEqual(again["status"], "ok")
            self.assertEqual(again["written"], [])
            self.assertTrue(again["identical"])
            # 同じ識別子で内容が違う：衝突。何も書かない
            p = os.path.join(bundle, "g2_evidence/P-A/1-1/evaluation_evidence.json")
            open(p, "ab").write(b" ")
            extra = os.path.join(bundle, "g2_evidence/P-A/1-1/new_file.txt")
            open(extra, "w").write("x")
            st = cn.stage(bundle, repo)
            self.assertEqual(st["status"], "conflict")
            self.assertFalse(os.path.exists(os.path.join(repo, "g2_evidence/P-A/1-1/new_file.txt")))

    def test_missing_corrupt_and_tampered_files_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res, out = self._first(tmp)
            cn.stage(os.path.join(out, "canonical_bundle"), repo)
            rel = "g2_evidence/P-A/1-1/manifest.json"
            self.assertEqual(cn.verify_manifest(repo, rel)["status"], "ok")
            f = os.path.join(repo, "g2_evidence/P-A/1-1/source_meta.json")
            orig = open(f, "rb").read()
            open(f, "wb").write(orig[:-1] + b"X")  # 同じサイズで中身を変える（保存後の改変）
            v = cn.verify_manifest(repo, rel)
            self.assertEqual(v["status"], "mismatch")
            self.assertIn("sha256 mismatch", v["problems"][0])
            open(f, "wb").write(orig + b"\n")
            self.assertIn("size mismatch", cn.verify_manifest(repo, rel)["problems"][0])
            os.remove(f)
            self.assertIn("missing", cn.verify_manifest(repo, rel)["problems"][0])
            open(os.path.join(repo, rel), "wb").write(b"{broken")
            self.assertEqual(cn.verify_manifest(repo, rel)["status"], "manifest_unavailable")

    def test_manifest_and_files_disagree(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res, out = self._first(tmp)
            bundle = os.path.join(out, "canonical_bundle")
            mp = os.path.join(bundle, "g2_evidence/P-A/1-1/manifest.json")
            man = json.load(open(mp))
            man["files"][0]["sha256"] = "0" * 64
            open(mp, "wb").write(cn._dumps(man))
            st = cn.stage(bundle, repo)
            self.assertEqual(st["status"], "verify_failed")  # 保存しても成功にしない

    def test_partial_write_failure_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res, out = self._first(tmp)
            real = os.replace
            calls = {"n": 0}

            def flaky(a, b):
                calls["n"] += 1
                if calls["n"] == 3:
                    raise OSError("disk full")
                return real(a, b)
            with mock.patch.object(cn.os, "replace", side_effect=flaky):
                st = cn.stage(os.path.join(out, "canonical_bundle"), repo)
            self.assertEqual(st["status"], "write_failed")

    def test_append_only_events_cannot_be_rewritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res, out = self._first(tmp, diff=(50, {"cat": "mismatch"}))
            bundle = os.path.join(out, "canonical_bundle")
            self.assertEqual(cn.stage(bundle, repo)["status"], "ok")
            ev = os.path.join(repo, *cn.H1_EVENTS.split("/"))
            self.assertEqual(len(open(ev, "rb").read().splitlines()), 1)
            # 束の側で過去の行を変えると衝突（追記だけを許す）
            bp = os.path.join(bundle, *cn.H1_EVENTS.split("/"))
            open(bp, "wb").write(open(bp, "rb").read().replace(b"P4-A", b"P4-B"))
            self.assertEqual(cn.stage(bundle, repo)["status"], "conflict")

    def test_reserved_and_unsafe_names(self):
        self.assertTrue(cn.run_dir_for({"permission_id": "_aggregate", "race_date": "20990101"},
                                       {"run_id": 1, "run_attempt": 1}).startswith(cn.AGG + "/unattached/"))
        self.assertTrue(cn.run_dir_for({"permission_id": "../x"}, {"run_id": 1, "run_attempt": 1})
                        .startswith(cn.AGG + "/unattached/unknown_date/"))
        with tempfile.TemporaryDirectory() as tmp:
            b = os.path.join(tmp, "b", "g2_evidence", "_other")
            os.makedirs(b)
            open(os.path.join(b, "x.json"), "w").write("{}")
            st = cn.stage(os.path.join(tmp, "b"), os.path.join(tmp, "r"))
            self.assertEqual(st["status"], "conflict")

    def test_achievement_ready_only_after_canonical_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res1, out1 = self._first(tmp)
            self.assertIsNotNone(res1["reach"])
            self.assertFalse(res1["achievement_readiness"]["ready_for_human_record"])
            self.assertIn("canonical save not verified for the reached segment",
                          res1["achievement_readiness"]["blocking"])
            self.assertEqual(res1["canonical_save"]["reached_segment_runs"][0]["status"], "not_saved")
            rel, sha = ta.canonical_previous(out1, repo)
            rc, res2, out2 = run(tmp, base, repo, (rel, sha), 2)
            self.assertEqual(res2["previous_state"]["status"], "ok")
            self.assertEqual(res2["canonical_save"]["reached_segment_runs"][0]["status"], "saved_verified")
            self.assertTrue(res2["achievement_readiness"]["ready_for_human_record"],
                            res2["achievement_readiness"])
            self.assertNotIn("achieved", res2["g2_status"])  # 達成は人が記録する（集約は書かない）
            # 正本を改変すると、また記録できない状態になる
            f = os.path.join(repo, "g2_evidence/P-A/1-1/artifact.zip")
            open(f, "ab").write(b"x")
            rc, res3, _ = run(tmp, base, repo, (rel, sha), 3)
            self.assertEqual(res3["canonical_save"]["reached_segment_runs"][0]["status"], "mismatch")
            self.assertFalse(res3["achievement_readiness"]["ready_for_human_record"])


class CanonicalSaveExtraTests(unittest.TestCase):
    def test_readback_difference_is_not_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res, out = CanonicalSaveTests()._first(tmp)
            with mock.patch.object(cn, "read_repo_file", return_value=b"corrupt"):
                st = cn.stage(os.path.join(out, "canonical_bundle"), repo)
            self.assertEqual(st["status"], "verify_failed")
            self.assertTrue(any(p.startswith("read-back differs") for p in st["problems"]))

    def test_consistently_replaced_artifact_is_not_saved_verified(self):
        """manifest と中身をそろえて差し替えても、今回の入力の artifact の SHA-256 と違えば正本と認めない。"""
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res1, out1 = CanonicalSaveTests()._first(tmp)
            rel, sha = ta.canonical_previous(out1, repo)
            d = os.path.join(repo, "g2_evidence/P-A/1-1")
            open(os.path.join(d, "artifact.zip"), "wb").write(b"other")
            man = json.load(open(os.path.join(d, "manifest.json")))
            for e in man["files"]:
                if e["path"].endswith("artifact.zip"):
                    e.update(size=5, sha256=cn.sha256(b"other"))
            open(os.path.join(d, "manifest.json"), "wb").write(cn._dumps(man))
            self.assertEqual(cn.verify_manifest(repo, "g2_evidence/P-A/1-1/manifest.json")["status"], "ok")
            rc, res2, _ = run(tmp, base, repo, (rel, sha), 2)
            self.assertEqual(res2["canonical_save"]["reached_segment_runs"][0]["status"], "mismatch")
            self.assertFalse(res2["achievement_readiness"]["ready_for_human_record"])


class PreviousStateTests(unittest.TestCase):
    def _prev(self, tmp, diff=None):
        repo, gen = setup_repo(tmp)
        base = os.path.join(tmp, "a")
        os.makedirs(base)
        make_runs(base, diff)
        rc, res1, out1 = run(tmp, base, repo, gen, 1)
        rel, sha = ta.canonical_previous(out1, repo)
        return repo, base, res1, rel, sha

    def test_normal_and_genesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen = setup_repo(tmp)
            st = cn.load_previous_state(repo, *gen)
            self.assertEqual((st["status"], st["kind"]), ("ok", "genesis"))
            repo, base, res1, rel, sha = self._prev(tmp + "/x" if False else tempfile.mkdtemp(dir=tmp))
            st = cn.load_previous_state(repo, rel, sha)
            self.assertEqual((st["status"], st["kind"]), ("ok", "aggregation"))

    def test_failures_hold_and_do_not_drop_holds(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res1, rel, sha = self._prev(tmp, diff=(50, {"cat": "mismatch"}))
            self.assertEqual(res1["p4_holds"]["H1"], [eid(50)])
            cases = {
                "not_given": None,
                "missing": ("g2_evidence/_aggregate/aggregations/9-1/manifest.json", sha),
                "sha_mismatch": (rel, "0" * 64),
                "bad_sha_form": (rel, "XYZ"),
                "not_manifest_path": ("g2_evidence/P-A/1-1/manifest.json", sha),
            }
            for name, prev in cases.items():
                with self.subTest(name=name):
                    rc, res, _ = run(tmp, base, repo, prev, 50 + len(name))
                    self.assertNotEqual(res["previous_state"]["status"], "ok")
                    self.assertEqual(res["g2_status"], "held_p4")
                    self.assertIn("previous_state_unavailable (Y6)", [h["reason"] for h in res["achievement_holds"]])
                    # 今回の ② の疑義は記録する（前回が不明でも H1 なしとして扱わない）
                    self.assertEqual(res["p4_holds"]["H1"], [eid(50)])
                    self.assertFalse(res["achievement_readiness"]["ready_for_human_record"])

    def test_broken_json_and_provenance_and_unrestorable(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res1, rel, sha = self._prev(tmp, diff=(50, {"cat": "mismatch"}))
            mp = os.path.join(repo, *rel.split("/"))
            orig = open(mp, "rb").read()
            man = json.loads(orig)
            # 由来：manifest の中の集約の識別が場所と合わない
            man2 = dict(man, aggregate_run="other-1")
            data = cn._dumps(man2)
            open(mp, "wb").write(data)
            self.assertEqual(cn.load_previous_state(repo, rel, cn.sha256(data))["status"],
                             "previous_state_provenance_unknown")
            open(mp, "wb").write(b"{broken")
            self.assertEqual(cn.load_previous_state(repo, rel, cn.sha256(b"{broken"))["status"],
                             "previous_state_unavailable")
            open(mp, "wb").write(orig)
            # H1 の記録が消えている：復元できない
            ev = os.path.join(repo, *cn.H1_EVENTS.split("/"))
            saved = open(ev, "rb").read()
            open(ev, "wb").write(b"")
            st = cn.load_previous_state(repo, rel, sha)
            self.assertIn(st["status"], ("previous_state_unrestorable", "previous_state_invalid"))
            open(ev, "wb").write(saved + b"{bad\n")
            self.assertEqual(cn.load_previous_state(repo, rel, sha)["status"], "previous_state_invalid")
            open(ev, "wb").write(saved)
            self.assertEqual(cn.load_previous_state(repo, rel, sha)["status"], "ok")

    def test_snapshot_active_hold_without_event_is_unrestorable(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen = setup_repo(tmp)
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            make_runs(base)
            rc, res1, out1 = run(tmp, base, repo, gen, 1)
            rel, sha = ta.canonical_previous(out1, repo,
                                             mutate_snapshot=lambda d: d["h2_active"].append("h2-unknown"))
            self.assertEqual(cn.load_previous_state(repo, rel, sha)["status"], "previous_state_unrestorable")
            rc, res2, _ = run(tmp, base, repo, (rel, sha), 2)
            self.assertEqual(res2["g2_status"], "held_p4")

    def test_carried_holds_and_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res1, rel, sha = self._prev(tmp, diff=(50, {"cat": "mismatch"}))
            base2 = os.path.join(tmp, "b")
            os.makedirs(base2)
            make_runs(base2)  # 今回は ② を入れない
            rc, res2, out2 = run(tmp, base2, repo, (rel, sha), 2)
            self.assertEqual(res2["doubts"], [])
            self.assertEqual(res2["p4_holds"]["H1"], [eid(50)])
            self.assertEqual(res2["p4_holds"]["carried_from_previous"]["H1"], res1["p4_holds"]["H1_event_ids"])
            self.assertEqual(res2["g2_status"], "held_p4")
            snap = json.load(open(glob.glob(os.path.join(out2, "canonical_bundle", "g2_evidence/_aggregate/state/snapshots/*.json"))[0]))
            self.assertEqual(snap["h1_active"], res1["p4_holds"]["H1_event_ids"])

    def test_old_option_rejected(self):
        with self.assertRaises(SystemExit), redirect_stdout(StringIO()), mock.patch("sys.stderr", new=StringIO()):
            agg.main(["--artifacts-dir", "x", "--first-date", "20990101", "--last-date", "20990101",
                      "--previous-result", "p.json"])


def ok_verifier(repo_root, res):
    return True, "test verifier"


class ResolutionTests(unittest.TestCase):
    """解除イベントの形式と検証（実際の解除は作らない。承認の確認はテストでだけ差し替える）。"""

    def _state(self, tmp):
        repo, gen = setup_repo(tmp)
        base = os.path.join(tmp, "a")
        os.makedirs(base)
        make_runs(base, diff=(50, {"cat": "mismatch", "b_L": None}))  # P4-C 相当：H1 と H2
        rc, res1, out1 = run(tmp, base, repo, gen, 1)
        self.assertEqual([(d["case"], d["H1"], d["H2"]) for d in res1["doubts"]], [("P4-C", True, True)])
        rel, sha = ta.canonical_previous(out1, repo)
        ev = os.path.join(repo, "g2_evidence/_aggregate/evidence_doc.txt")
        open(ev, "w").write("investigation notes")
        return repo, base, res1, rel, sha

    def _res(self, kind, res1, **over):
        common = {"schema_version": 1, "resolution_id": f"res-{kind}", "reason": "investigated",
                  "evidence": [{"path": "g2_evidence/_aggregate/evidence_doc.txt",
                                "sha256": cn.sha256(b"investigation notes")}],
                  "approval_reference": "g2_evidence/_aggregate/approvals/x.json", "approved_by": "sinrai74",
                  "approved_at": "2099-01-03T00:00:00Z", "created_at": "2099-01-03T00:00:00Z"}
        if kind == "h1":
            r = dict(common, event_type="h1_resolution", target_h1_event_id=res1["p4_holds"]["H1_event_ids"][0],
                     target_race_key=eid(50), decision="zero")
        else:
            r = dict(common, event_type="h2_resolution", target_h2_event_id=res1["p4_holds"]["H2"][0]["event_id"],
                     decision="closed")
        r.update(over)
        return r

    def _write_res(self, repo, rows):
        p = os.path.join(repo, *cn.RESOLUTION_EVENTS.split("/"))
        open(p, "wb").write(b"".join(cn._line(r) for r in rows))

    def _run(self, tmp, repo, base, rel, sha, n, verifier=ok_verifier):
        return run(tmp, base, repo, (rel, sha), n, verifier=verifier)[1]

    def test_h1_only_resolution_keeps_h2(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res1, rel, sha = self._state(tmp)
            self._write_res(repo, [self._res("h1", res1)])
            h1_before = open(os.path.join(repo, *cn.H1_EVENTS.split("/")), "rb").read()
            res = self._run(tmp, repo, base, rel, sha, 2)
            self.assertEqual(res["p4_holds"]["H1"], [])
            self.assertEqual([h["eval_id"] for h in res["p4_holds"]["H2"]], [eid(50)])  # H2 は残る
            self.assertEqual(res["corrections_applied"], {eid(50): "zero"})
            # P4 13：訂正（0）を当てて計算し直す。元の記録は消さない（original_state に残る）
            x = [s for s in res["sequence"] if s.get("eval_id") == eid(50)][0]
            self.assertEqual((x["state"], x["original_state"]), ("corrected_zero", "added"))
            self.assertEqual(res["g2_status"], "held_p4")
            self.assertEqual(open(os.path.join(repo, *cn.H1_EVENTS.split("/")), "rb").read(), h1_before)

    def test_h2_only_resolution_keeps_h1(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res1, rel, sha = self._state(tmp)
            self._write_res(repo, [self._res("h2", res1)])
            res = self._run(tmp, repo, base, rel, sha, 2)
            self.assertEqual(res["p4_holds"]["H2"], [])
            self.assertEqual(res["p4_holds"]["H1"], [eid(50)])
            self.assertEqual(res["g2_status"], "held_p4")

    def test_default_has_no_approval_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res1, rel, sha = self._state(tmp)
            self._write_res(repo, [self._res("h1", res1), self._res("h2", res1)])
            res = run(tmp, base, repo, (rel, sha), 2)[1]  # 承認の確認は「未決定」のまま
            self.assertTrue(all(not r["valid"] for r in res["resolutions"]))
            self.assertIn("approval verification method not decided", res["resolutions"][0]["problems"][0])
            self.assertEqual(res["p4_holds"]["H1"], [eid(50)])
            self.assertEqual(len(res["p4_holds"]["H2"]), 1)

    def test_invalid_resolutions(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res1, rel, sha = self._state(tmp)
            r = self._res("h1", res1)
            cases = {
                "no_approval": dict(approval_reference=""),
                "no_approver": dict(approved_by=None),
                "target_missing": dict(target_h1_event_id="h1-nope"),
                "race_mismatch": dict(target_race_key=eid(51)),
                "evidence_missing": dict(evidence=[]),
                "evidence_file_missing": dict(evidence=[{"path": "g2_evidence/none.txt", "sha256": "0" * 64}]),
                "evidence_sha": dict(evidence=[{"path": "g2_evidence/_aggregate/evidence_doc.txt",
                                                "sha256": "1" * 64}]),
                "bad_decision": dict(decision="corrected_zero_or_maintained"),
                "maintain_p4c": dict(decision="maintain"),
                "schema": dict(schema_version=2),
                "extra_key": dict(note="x"),
            }
            for name, over in cases.items():
                with self.subTest(name=name):
                    out = cn.validate_resolutions(repo, [dict(r, **over)], *self._events(repo), verifier=ok_verifier)
                    self.assertEqual(out["valid"], [], name)
            missing = dict(r)
            missing.pop("reason")
            self.assertEqual(cn.validate_resolutions(repo, [missing], *self._events(repo), verifier=ok_verifier)["valid"], [])
            # 重複・矛盾
            dup = cn.validate_resolutions(repo, [r, r], *self._events(repo), verifier=ok_verifier)
            self.assertEqual(dup["valid"], [])
            contra = cn.validate_resolutions(repo, [r, dict(r, resolution_id="res-2", decision="maintain")],
                                             *self._events(repo), verifier=ok_verifier)
            self.assertEqual(contra["valid"], [])
            ok = cn.validate_resolutions(repo, [r], *self._events(repo), verifier=ok_verifier)
            self.assertEqual(len(ok["valid"]), 1)

    def _events(self, repo):
        h1 = cn.parse_jsonl(open(os.path.join(repo, *cn.H1_EVENTS.split("/")), "rb").read())[0]
        h2 = cn.parse_jsonl(open(os.path.join(repo, *cn.H2_EVENTS.split("/")), "rb").read())[0]
        return h1, h2

    def test_history_is_traceable_and_past_events_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, base, res1, rel, sha = self._state(tmp)
            self._write_res(repo, [self._res("h1", res1)])
            res2, = [self._run(tmp, repo, base, rel, sha, 2)]
            out2 = os.path.join(tmp, "out2")
            snap = json.load(open(glob.glob(os.path.join(out2, "canonical_bundle",
                                                       "g2_evidence/_aggregate/state/snapshots/2-1.json"))[0]))
            self.assertEqual(snap["h1_active"], [])
            self.assertEqual(snap["previous_state"]["path"], rel)
            self.assertEqual(snap["previous_state"]["sha256"], sha)
            # 過去の到達の記録は写すだけで変えない
            self.assertEqual(res2["reaggregation"]["previous_reach"], res1["reach"])
            st = cn.stage(os.path.join(out2, "canonical_bundle"), repo)
            self.assertEqual(st["status"], "ok", st)
            # 前の集約の正本も、そのまま検証できる（追記だけのファイルは先頭が同じ）
            self.assertEqual(cn.verify_manifest(repo, rel, sha)["status"], "ok")


if __name__ == "__main__":
    unittest.main()


class AuditStage6FixTests(unittest.TestCase):
    """独立監査（2026-11-09、段階6）の指摘の修正。"""

    def _chain(self, tmp, diff=(50, {"cat": "mismatch"})):
        repo, gen = setup_repo(tmp)
        base = os.path.join(tmp, "a")
        os.makedirs(base)
        make_runs(base, diff)
        rc, res1, out1 = run(tmp, base, repo, gen, 1)
        rel1, sha1 = ta.canonical_previous(out1, repo)
        return repo, gen, base, res1, rel1, sha1

    def test_h1_genesis_replay_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp)
            st = cn.load_previous_state(repo, *gen)
            self.assertEqual(st["status"], "previous_state_not_latest")
            rc, res, _ = run(tmp, base, repo, gen, 2)
            self.assertEqual(res["g2_status"], "held_p4")
            self.assertEqual(res["p4_holds"]["H1"], [eid(50)])  # 今回の疑義も、ログの保留も消えない
            self.assertFalse(res["achievement_readiness"]["ready_for_human_record"])

    def test_genesis_rejected_when_log_exists_without_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp, diff=None)
            self.assertEqual(cn.load_previous_state(repo, *gen)["status"], "previous_state_not_latest")

    def test_genesis_rejected_when_events_exist_without_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp)
            os.remove(os.path.join(repo, *cn.AGGREGATION_LOG.split("/")))
            self.assertEqual(cn.load_previous_state(repo, *gen)["status"], "previous_state_not_latest")

    def test_events_appended_outside_the_chain_are_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp)
            ev = os.path.join(repo, *cn.H2_EVENTS.split("/"))
            open(ev, "ab").write(cn._line({"schema_version": 1, "event_type": "h2", "event_id": "h2-x",
                                           "eval_id": eid(9)}))
            self.assertEqual(cn.load_previous_state(repo, rel1, sha1)["status"], "previous_state_not_latest")

    def test_h1_older_manifest_replay_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp)
            base2 = os.path.join(tmp, "b")
            os.makedirs(base2)
            make_runs(base2)
            rc, res2, out2 = run(tmp, base2, repo, (rel1, sha1), 2)
            self.assertEqual(cn.stage(os.path.join(out2, "canonical_bundle"), repo)["status"], "ok")
            self.assertEqual(cn.load_previous_state(repo, rel1, sha1)["status"], "previous_state_not_latest")
            rc, res3, _ = run(tmp, base2, repo, (rel1, sha1), 3)
            self.assertEqual(res3["g2_status"], "held_p4")
            self.assertFalse(res3["achievement_readiness"]["ready_for_human_record"])

    def test_holds_come_from_event_log_not_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp)
            base2 = os.path.join(tmp, "b")
            os.makedirs(base2)
            make_runs(base2)
            # スナップショットの有効な一覧を空にしても、ログの H1 は残る
            out1 = os.path.join(tmp, "out1")
            repo2 = os.path.join(tmp, "repo2")
            os.makedirs(repo2)
            write_genesis(repo2)
            rel, sha = ta.canonical_previous(out1, repo2, mutate_snapshot=lambda d: d.update(h1_active=[]))
            rc, res2, _ = run(tmp, base2, repo2, (rel, sha), 2)
            self.assertEqual(res2["p4_holds"]["H1"], [eid(50)])

    def test_h2_restage_after_refetch_is_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp, diff=None)
            for d in glob.glob(os.path.join(base, "*", "meta.json")):
                m = json.load(open(d))
                m.update(log_fetched_at="2099-02-02T00:00:00Z", latest_attempt="3",
                         artifact_expires_at="2099-05-01T00:00:00Z")
                json.dump(m, open(d, "w"))
            rc, res2, out2 = run(tmp, base, repo, (rel1, sha1), 2)
            st = cn.stage(os.path.join(out2, "canonical_bundle"), repo)
            self.assertEqual(st["status"], "ok", st["conflicts"])
            # 既にある実行の正本は束に入れない（書き直さない）。集約の正本に「already_saved」と記録する
            inp = json.load(open(glob.glob(os.path.join(out2, "canonical_bundle", "g2_evidence/_aggregate/aggregations/*/input_manifest.json"))[0]))
            self.assertEqual({r["canonical_in_bundle"] for r in inp["evaluation_runs"]}, {"already_saved"})
            # 実行の正本の中身は、取得の時刻などに左右されない
            vr = agg.verify_run(os.path.join(base, "1-1"))
            a1 = cn.build_run_files(vr, "g2_evidence/P-A/1-1", os.path.join(base, "1-1"))
            vr["meta"]["log_fetched_at"] = "2100-01-01T00:00:00Z"
            self.assertEqual(a1, cn.build_run_files(vr, "g2_evidence/P-A/1-1", os.path.join(base, "1-1")))

    def test_later_fetch_failure_does_not_block_staging(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp, diff=None)
            for d in glob.glob(os.path.join(base, "*", "job.log")):
                os.remove(d)
            for d in glob.glob(os.path.join(base, "*", "meta.json")):
                m = json.load(open(d))
                m.update(log_fetch_ok=False, log_file=None, log_sha256=None, log_bytes=None)
                json.dump(m, open(d, "w"))
            rc, res2, out2 = run(tmp, base, repo, (rel1, sha1), 2)
            self.assertEqual(cn.stage(os.path.join(out2, "canonical_bundle"), repo)["status"], "ok")

    def test_unverified_new_run_is_not_put_in_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen = setup_repo(tmp)
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            _write_run(base, 1, 1, "20990101", [R("a")], ["a"], digest_ok=False, permission="P-A")
            rc, res, out = run(tmp, base, repo, gen, 1)
            self.assertFalse(os.path.exists(os.path.join(out, "canonical_bundle", "g2_evidence", "P-A")))
            inp = json.load(open(glob.glob(os.path.join(out, "canonical_bundle", "g2_evidence/_aggregate/aggregations/*/input_manifest.json"))[0]))
            self.assertEqual(inp["evaluation_runs"][0]["canonical_in_bundle"], "not_included_unverified")

    def test_genesis_refused_when_aggregations_exist_even_if_logs_deleted(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp)
            for rel in (cn.AGGREGATION_LOG, cn.H1_EVENTS, cn.H2_EVENTS):
                p_ = os.path.join(repo, *rel.split("/"))
                if os.path.exists(p_):
                    os.remove(p_)
            self.assertEqual(cn.load_previous_state(repo, *gen)["status"], "previous_state_not_latest")

    def test_aggregation_log_chain_is_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp, diff=None)
            log_p = os.path.join(repo, *cn.AGGREGATION_LOG.split("/"))
            orig = open(log_p, "rb").read()
            row = json.loads(orig.splitlines()[0])
            # 古い manifest を指す行を足しても、鎖がつながらない
            open(log_p, "ab").write(cn._line(dict(row, previous={"path": cn.GENESIS})))
            self.assertEqual(cn.load_previous_state(repo, rel1, sha1)["status"], "previous_state_invalid")
            open(log_p, "wb").write(orig)
            self.assertEqual(cn.load_previous_state(repo, rel1, sha1)["status"], "ok")

    def test_event_without_eval_id_is_invalid_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp)
            ev = os.path.join(repo, *cn.H1_EVENTS.split("/"))
            rows = [json.loads(x) for x in open(ev, "rb").read().splitlines()]
            rows[0].pop("eval_id")
            open(ev, "wb").write(b"".join(cn._line(r) for r in rows))
            self.assertIn(cn.load_previous_state(repo, rel1, sha1)["status"],
                          ("previous_state_invalid", "previous_state_sha_mismatch"))
            rc, res, _ = run(tmp, base, repo, (rel1, sha1), 2)
            self.assertEqual(res["g2_status"], "held_p4")

    def test_no_bundle_when_previous_state_not_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen = setup_repo(tmp)
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            make_runs(base)
            rc, res, out = run(tmp, base, repo, None, 1)
            self.assertFalse(os.path.exists(os.path.join(out, "canonical_bundle")))
            self.assertIn("previous state not verified", res["canonical_bundle_error"])

    def test_m1_exit_3_blocks_readiness_but_procedure_violation_does_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp, diff=None)
            rc, res2, out2 = run(tmp, base, repo, (rel1, sha1), 2)
            self.assertTrue(res2["achievement_readiness"]["ready_for_human_record"])
            # 到達した区間の外の、照合に失敗した実行（終了コード 3）
            _write_run(base, 7, 1, "20990101", [R("zz")], ["zz"], digest_ok=False, started="2099-01-02T09:00:00Z",
                       permission="P-A")
            rc, res3, _ = run(tmp, base, repo, (rel1, sha1), 3)
            self.assertEqual(rc, 3)
            self.assertFalse(res3["achievement_readiness"]["ready_for_human_record"])
            self.assertIn("aggregate exit code 3", res3["achievement_readiness"]["blocking"])

    def test_m2_verify_append_only_prefix_and_newline(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp)
            ev = os.path.join(repo, *cn.H1_EVENTS.split("/"))
            orig = open(ev, "rb").read()
            open(ev, "wb").write(orig.replace(b"P4-A", b"P4-B"))
            self.assertEqual(cn.verify_manifest(repo, rel1, sha1)["status"], "mismatch")
            open(ev, "wb").write(orig)
            # 改行で終わらない既存のファイルへの追記は衝突
            b = os.path.join(tmp, "bundle", *cn.H1_EVENTS.split("/"))
            os.makedirs(os.path.dirname(b))
            open(ev, "wb").write(orig.rstrip(b"\n"))
            open(b, "wb").write(orig.rstrip(b"\n") + b'{"x":1}\n')
            self.assertEqual(cn.stage(os.path.join(tmp, "bundle"), repo)["status"], "conflict")

    def test_m2_provenance_and_listing_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp, diff=None)
            mp = os.path.join(repo, *rel1.split("/"))
            orig = open(mp, "rb").read()
            man = json.loads(orig)
            log_p = os.path.join(repo, *cn.AGGREGATION_LOG.split("/"))
            log_orig = open(log_p, "rb").read()

            def with_manifest(m):
                data = cn._dumps(m)
                open(mp, "wb").write(data)
                last = json.loads(log_orig.splitlines()[-1])
                last["manifest_sha256"] = cn.sha256(data)
                open(log_p, "wb").write(b"".join(log_orig.splitlines(keepends=True)[:-1]) + cn._line(last))
                return cn.load_previous_state(repo, rel1, cn.sha256(data))["status"]
            self.assertEqual(with_manifest(dict(man, commit_sha="local")), "previous_state_provenance_unknown")
            self.assertEqual(with_manifest(dict(man, files=[e for e in man["files"]
                                                             if e["path"] != man["snapshot"]])),
                             "previous_state_invalid")
            open(mp, "wb").write(orig)
            open(log_p, "wb").write(log_orig)
            self.assertEqual(cn.load_previous_state(repo, rel1, sha1)["status"], "ok")

    def test_m2_readiness_needs_segment_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp, diff=None)
            with mock.patch.object(agg, "_segment_ids", return_value=[]):
                rc, res2, _ = run(tmp, base, repo, (rel1, sha1), 2)
            self.assertFalse(res2["achievement_readiness"]["ready_for_human_record"])

    def test_m2_evidence_entry_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen = setup_repo(tmp)
            out = cn.validate_resolutions(repo, [{"event_type": "h1_resolution", "evidence": [
                {"path": "g2_evidence/x", "sha256": "0" * 64, "extra": 1}]}], [], [], verifier=ok_verifier)
            self.assertTrue(any("bad evidence entry" in p for p in out["results"][0]["problems"]))

    def test_low_previous_without_verified_previous_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp, diff=None)
            out1 = os.path.join(tmp, "out1")
            repo2 = os.path.join(tmp, "repo2")
            os.makedirs(repo2)
            rel, sha = ta.canonical_previous(out1, repo2,
                                             mutate_snapshot=lambda d: d["previous_state"].update(status="x"))
            self.assertEqual(cn.load_previous_state(repo2, rel, sha)["status"], "previous_state_invalid")

    def test_low_symlinked_parent_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = os.path.join(tmp, "repo")
            outside = os.path.join(tmp, "outside")
            os.makedirs(os.path.join(repo, "g2_evidence"))
            os.makedirs(outside)
            os.symlink(outside, os.path.join(repo, "g2_evidence", "P-A"))
            b = os.path.join(tmp, "b", "g2_evidence", "P-A", "1-1")
            os.makedirs(b)
            open(os.path.join(b, "x.json"), "w").write("{}")
            st = cn.stage(os.path.join(tmp, "b"), repo)
            self.assertEqual(st["status"], "conflict")
            self.assertEqual(os.listdir(outside), [])

    def test_low_p4_5_event_id_is_stable(self):
        a = cn.event_id("h1", {"eval_id": "e", "source": "P4-5", "case": "P4-D", "previous_result_sha256": "1"})
        b = cn.event_id("h1", {"eval_id": "e", "source": "P4-5", "case": "P4-D", "previous_result_sha256": "2"})
        self.assertEqual(a, b)

    def test_low_bundle_failure_blocks_readiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen, base, res1, rel1, sha1 = self._chain(tmp, diff=None)
            with mock.patch.object(agg, "write_canonical_bundle", side_effect=OSError("disk")):
                rc, res2, out2 = run(tmp, base, repo, (rel1, sha1), 2)
            self.assertFalse(res2["achievement_readiness"]["ready_for_human_record"])
            self.assertIn("canonical bundle could not be built", res2["achievement_readiness"]["blocking"])
            self.assertFalse(os.path.exists(os.path.join(out2, "canonical_bundle")))


class GitHistoryTests(unittest.TestCase):
    """Z3（GPT 決定 2026-11-09）：g2_evidence/ の commit の履歴を、手元の git で確かめる。"""

    def _repo(self, tmp):
        import subprocess
        repo = os.path.join(tmp, "g")
        os.makedirs(repo)
        env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x", GIT_COMMITTER_NAME="t",
                   GIT_COMMITTER_EMAIL="t@x")

        def git(*a):
            subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True, env=env)
        git("init", "-q")
        return repo, git

    def _put(self, repo, rel, data):
        p = os.path.join(repo, *rel.split("/"))
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(data)

    def test_ok_chain_of_appends_and_adds(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, git = self._repo(tmp)
            self.assertEqual(cn.git_history_check(repo)["status"], "ok")  # まだ g2_evidence が無い
            self._put(repo, cn.H1_EVENTS, b'{"a":1}\n')
            self._put(repo, "g2_evidence/P-A/1-1/manifest.json", b"{}")
            git("add", "-A")
            git("commit", "-qm", "1")
            self._put(repo, cn.H1_EVENTS, b'{"a":1}\n{"b":2}\n')
            self._put(repo, "g2_evidence/P-B/2-1/manifest.json", b"{}")
            git("add", "-A")
            git("commit", "-qm", "2")
            r = cn.git_history_check(repo)
            self.assertEqual(r["status"], "ok", r)
            self.assertEqual(r["commits"], 2)

    def test_rewritten_append_only_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, git = self._repo(tmp)
            self._put(repo, cn.H1_EVENTS, b'{"a":1}\n')
            git("add", "-A")
            git("commit", "-qm", "1")
            self._put(repo, cn.H1_EVENTS, b'{"z":9}\n')  # 過去の行を書き換えた
            git("add", "-A")
            git("commit", "-qm", "2")
            r = cn.git_history_check(repo)
            self.assertEqual(r["status"], "history_mismatch")
            self.assertIn("rewritten", r["problems"][0])

    def test_modified_or_deleted_canonical_file(self):
        for action in ("modify", "delete"):
            with self.subTest(action=action), tempfile.TemporaryDirectory() as tmp:
                repo, git = self._repo(tmp)
                rel = "g2_evidence/P-A/1-1/artifact.zip"
                self._put(repo, rel, b"zip")
                git("add", "-A")
                git("commit", "-qm", "1")
                if action == "modify":
                    self._put(repo, rel, b"other")
                else:
                    os.remove(os.path.join(repo, *rel.split("/")))
                git("add", "-A")
                git("commit", "-qm", "2")
                r = cn.git_history_check(repo)
                self.assertEqual(r["status"], "history_mismatch")

    def test_deleted_log_and_uncommitted_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, git = self._repo(tmp)
            self._put(repo, cn.AGGREGATION_LOG, b'{"a":1}\n')
            git("add", "-A")
            git("commit", "-qm", "1")
            self._put(repo, cn.AGGREGATION_LOG, b'{"a":1}\n{"b":2}\n')  # commit していない
            self.assertIn("working tree differs", cn.git_history_check(repo)["problems"][0])
            os.remove(os.path.join(repo, *cn.AGGREGATION_LOG.split("/")))
            git("add", "-A")
            git("commit", "-qm", "2")
            self.assertIn("deleted", cn.git_history_check(repo)["problems"][0])

    def test_evil_merge_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, git = self._repo(tmp)
            rel = "g2_evidence/P-A/1-1/manifest.json"
            self._put(repo, rel, b"{}")
            git("add", "-A")
            git("commit", "-qm", "1")
            git("checkout", "-qb", "side")
            self._put(repo, "other.txt", b"x")
            git("add", "-A")
            git("commit", "-qm", "side")
            git("checkout", "-q", "-")
            git("merge", "--no-ff", "--no-commit", "-q", "side")
            self._put(repo, rel, b'{"changed": 1}')  # merge の解決で正本を変える
            git("add", "-A")
            git("commit", "-qm", "merge")
            self.assertEqual(cn.git_history_check(repo)["status"], "history_mismatch")

    def test_uncommitted_or_untracked_canonical_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, git = self._repo(tmp)
            rel = "g2_evidence/P-A/1-1/manifest.json"
            self._put(repo, rel, b"{}")
            git("add", "-A")
            git("commit", "-qm", "1")
            self._put(repo, "g2_evidence/P-A/1-1/extra.json", b"{}")
            self.assertIn("uncommitted or untracked", cn.git_history_check(repo)["problems"][0])

    def test_not_a_repository_or_shallow(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(cn.git_history_check(tmp)["status"], "history_unavailable")

    def test_main_holds_when_history_check_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen = setup_repo(tmp)
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            make_runs(base)
            with mock.patch.object(cn, "git_history_check",
                                   return_value={"status": "history_mismatch", "problems": ["x"]}):
                rc, res, _ = run(tmp, base, repo, gen, 1, git_ok=False)
            self.assertEqual(res["g2_status"], "held_p4")
            self.assertIn("git history check of g2_evidence failed (Z3)",
                          [h["reason"] for h in res["achievement_holds"]])

    def test_workflow_fetches_full_history(self):
        t = open(".github/workflows/formal_g2_aggregate.yml", encoding="utf-8").read()
        self.assertIn("fetch-depth: 0", t)
        self.assertIn("persist-credentials: false", t)


class EndToEndWithGitTests(unittest.TestCase):
    """Z4（GPT 決定 2026-11-09）の順：束を commit → その commit で再集約 → 記録してよい状態を確かめる。
    実際の git のリポジトリで、Z3 の履歴の確かめも本物を使う。"""

    def test_bundle_commit_then_reaggregate_is_ready(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            repo, gen = setup_repo(tmp)
            env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@x", GIT_COMMITTER_NAME="t",
                       GIT_COMMITTER_EMAIL="t@x")

            def git(*a):
                subprocess.run(["git", "-C", repo, *a], check=True, capture_output=True, env=env)
            git("init", "-q")
            git("add", "-A")
            git("commit", "-qm", "genesis")
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            make_runs(base)
            rc, res1, out1 = run(tmp, base, repo, gen, 1, git_ok=False)
            self.assertEqual(res1["git_history_check"]["status"], "ok")
            self.assertFalse(res1["achievement_readiness"]["ready_for_human_record"])  # 正本がまだ無い
            self.assertEqual(cn.stage(os.path.join(out1, "canonical_bundle"), repo)["status"], "ok")
            git("add", "-A")
            git("commit", "-qm", "bundle 1")
            last = json.loads(open(os.path.join(repo, *cn.AGGREGATION_LOG.split("/")), "rb").read().splitlines()[-1])
            rc, res2, out2 = run(tmp, base, repo, (last["manifest"], last["manifest_sha256"]), 2, git_ok=False)
            self.assertEqual(res2["git_history_check"]["status"], "ok", res2["git_history_check"])
            self.assertEqual(res2["previous_state"]["status"], "ok")
            self.assertTrue(res2["achievement_readiness"]["ready_for_human_record"], res2["achievement_readiness"])
            self.assertEqual(rc, 0)
            # 正本を改変して commit すると、履歴の確かめで保留になる
            f = os.path.join(repo, "g2_evidence/P-A/1-1/artifact_manifest.json")
            open(f, "ab").write(b" ")
            git("add", "-A")
            git("commit", "-qm", "tamper")
            rc, res3, _ = run(tmp, base, repo, (last["manifest"], last["manifest_sha256"]), 3, git_ok=False)
            self.assertEqual(res3["git_history_check"]["status"], "history_mismatch")
            self.assertFalse(res3["achievement_readiness"]["ready_for_human_record"])
