"""段階4-2：formal_g3.permission（G3 permission 形式・窓・日付・mode）の回帰テスト。"""
import unittest
from datetime import datetime, timezone

from formal_g3 import permission as P


def _doc(**kw):
    base = {
        "schema_version": 1,
        "permission_id": "G3-20261116-01",
        "race_date": "20261116",
        "window_start": "2026-11-16T00:00:00Z",
        "window_end": "2026-11-17T15:00:00Z",
        "workflow": ".github/workflows/formal_g3_evaluate.yml",
        "ref": "refs/heads/main",
        "mode": "formal",
        "approved_on": "2026-11-15",
        "approved_by": "sinrai74",
        "approval_method": "github_web_ui",
    }
    base.update(kw)
    return base


class PermissionFileProblemsTests(unittest.TestCase):
    def test_valid_formal_default_workflow_confirmed(self):
        # GPT確定で G3_PERMISSION_WORKFLOW がセット済み。既定で形式 OK
        self.assertEqual(P.G3_PERMISSION_WORKFLOW, ".github/workflows/formal_g3_evaluate.yml")
        probs = P.permission_file_problems(_doc(), "G3-20261116-01")
        self.assertEqual(probs, [])

    def test_workflow_undetermined_flagged_when_none(self):
        # expected_workflow=None を明示すると未確定が problems に出る
        probs = P.permission_file_problems(_doc(), "G3-20261116-01", expected_workflow=None)
        self.assertTrue(any("undetermined" in p for p in probs))

    def test_workflow_mismatch(self):
        probs = P.permission_file_problems(_doc(workflow=".github/workflows/other.yml"), "G3-20261116-01")
        self.assertTrue(any("workflow must be" in p for p in probs))

    def test_investigation_mode_rejected_for_formal(self):
        probs = P.permission_file_problems(_doc(mode="investigation"), "G3-20261116-01",
                                           expected_workflow=".github/workflows/formal_g3_evaluate.yml")
        self.assertTrue(any("mode must be formal" in p for p in probs))

    def test_investigation_ok_when_not_require_formal(self):
        probs = P.permission_file_problems(_doc(mode="investigation"), "G3-20261116-01",
                                           expected_workflow=".github/workflows/formal_g3_evaluate.yml",
                                           require_formal=False)
        self.assertEqual(probs, [])

    def test_bad_fields(self):
        self.assertTrue(P.permission_file_problems(_doc(approved_by="someone"), "G3-20261116-01",
                        expected_workflow="w"))
        self.assertTrue(P.permission_file_problems(_doc(window_start="2026-11-16 00:00"), "G3-20261116-01",
                        expected_workflow="w"))
        self.assertTrue(P.permission_file_problems(_doc(permission_id="X"), "G3-20261116-01",
                        expected_workflow="w"))
        self.assertTrue(P.permission_file_problems(_doc(ref="refs/heads/dev"), "G3-20261116-01",
                        expected_workflow="w"))

    def test_window_start_after_end(self):
        probs = P.permission_file_problems(_doc(window_start="2026-11-18T00:00:00Z"), "G3-20261116-01",
                                           expected_workflow="w")
        self.assertTrue(any("window_start must be before" in p for p in probs))


class GuardTests(unittest.TestCase):
    def test_race_date_formal(self):
        now = datetime(2026, 11, 17, 3, 0, tzinfo=timezone.utc)  # JST 12:00 2026-11-17
        self.assertTrue(P.check_race_date_formal("20261116", now)["ok"])
        self.assertFalse(P.check_race_date_formal("20261117", now)["ok"])  # 当日は不可

    def test_within_window(self):
        d = _doc()
        self.assertTrue(P.within_window(d, "2026-11-16T09:00:00Z")["ok"])
        self.assertFalse(P.within_window(d, "2026-11-18T00:00:00Z")["ok"])
        self.assertFalse(P.within_window(d, "bad")["ok"])

    def test_investigation_needs_no_permission(self):
        self.assertFalse(P.investigation_needs_permission())

    def test_sha_helpers(self):
        self.assertTrue(P.is_sha256("a" * 64))
        self.assertFalse(P.is_sha256("xyz"))
        self.assertEqual(len(P.file_sha256(b"abc")), 64)


class PermissionCheck12Tests(unittest.TestCase):
    """git 履歴12条件（evidence 入力の純粋判定）。actions 層が取得した evidence を模す。"""

    PID = "G3-20261116-01"

    def _setup_repo(self, tmp):
        import json as _j
        import os
        os.makedirs(os.path.join(tmp, "g3_permissions"))
        os.makedirs(os.path.join(tmp, ".github", "workflows"))
        with open(os.path.join(tmp, ".github", "workflows", "formal_g3_evaluate.yml"), "w") as f:
            f.write("name: g3\n")
        doc = _doc(permission_id=self.PID)
        data = _j.dumps(doc).encode("utf-8")
        with open(os.path.join(tmp, "g3_permissions", f"{self.PID}.json"), "wb") as f:
            f.write(data)
        return doc, data

    def _vr(self, data, tmp, **overrides):
        blob = P._git_blob_sha(data)
        commit = {
            "sha": "c0ffee", "files_count": 1,
            "files": [{"filename": f"g3_permissions/{self.PID}.json", "status": "added", "sha": blob}],
            "author_login": "sinrai74", "committer_login": "web-flow", "verified": True,
            "committer_date": "2026-11-15T10:00:00Z",
        }
        ev = {"fetch_ok": True, "permission_id": self.PID, "path": f"g3_permissions/{self.PID}.json",
              "history_run_head": ["c0ffee"], "history_aggregate_head": ["c0ffee"], "commits": [commit],
              "aggregate_ref": "refs/heads/main",
              "compare": {"base": "c0ffee", "head": "HEADSHA", "status": "ahead"}}
        vr = {
            "manifest": {"run": {"permission_id": self.PID, "race_date": "20261116", "mode": "formal",
                                 "github_ref": "refs/heads/main"}},
            "meta": {"run_started_at": "2026-11-16T09:00:00Z", "api_head_sha": "HEADSHA",
                     "api_workflow_path": ".github/workflows/formal_g3_evaluate.yml"},
            "permission_evidence": ev,
        }
        for k, v in overrides.items():
            vr[k] = v
        return vr, ev, commit

    def test_all_12_ok(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _, data = self._setup_repo(tmp)
            vr, _ev, _c = self._vr(data, tmp)
            res = P.permission_check(tmp, vr)
            self.assertEqual(res["status"], "ok", res)
            self.assertEqual(len(res["conditions"]), 12)
            self.assertTrue(all(res["conditions"].values()))

    def test_wrong_author_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _, data = self._setup_repo(tmp)
            vr, ev, _c = self._vr(data, tmp)
            ev["commits"][0]["author_login"] = "someone"
            res = P.permission_check(tmp, vr)
            self.assertEqual(res["status"], "not_permitted")
            self.assertFalse(res["conditions"]["6_author"])

    def test_blob_mismatch_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _, data = self._setup_repo(tmp)
            vr, ev, _c = self._vr(data, tmp)
            ev["commits"][0]["files"][0]["sha"] = "deadbeef"
            res = P.permission_check(tmp, vr)
            self.assertEqual(res["status"], "not_permitted")
            self.assertFalse(res["conditions"]["11_blob_matches"])

    def test_missing_evidence_fails_closed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _, data = self._setup_repo(tmp)
            vr, _ev, _c = self._vr(data, tmp)
            del vr["permission_evidence"]
            res = P.permission_check(tmp, vr)
            self.assertEqual(res["status"], "not_permitted")

    def test_signature_not_verified_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _, data = self._setup_repo(tmp)
            vr, ev, _c = self._vr(data, tmp)
            ev["commits"][0]["verified"] = False
            res = P.permission_check(tmp, vr)
            self.assertFalse(res["conditions"]["8_signature_verified"])
            self.assertEqual(res["status"], "not_permitted")

    def test_commit_after_run_fails(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _, data = self._setup_repo(tmp)
            vr, ev, _c = self._vr(data, tmp)
            ev["commits"][0]["committer_date"] = "2026-11-16T12:00:00Z"  # run開始(09:00)より後
            res = P.permission_check(tmp, vr)
            self.assertFalse(res["conditions"]["9_committed_before_run"])


class WorkflowConsistencyTests(unittest.TestCase):
    def test_consistency_ok_and_mismatch(self):
        import os
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, ".github", "workflows"))
            with open(os.path.join(tmp, ".github", "workflows", "formal_g3_evaluate.yml"), "w") as f:
                f.write("name: g3\n")
            doc = _doc()
            meta = {"api_workflow_path": ".github/workflows/formal_g3_evaluate.yml"}
            info = {"github_ref": "refs/heads/main"}
            self.assertTrue(P.workflow_consistency(tmp, doc, meta, info)["ok"])
            # ref 不一致
            self.assertFalse(P.workflow_consistency(tmp, doc, meta, {"github_ref": "refs/heads/dev"})["ok"])
            # workflow ファイル不在
            with tempfile.TemporaryDirectory() as empty:
                self.assertFalse(P.workflow_consistency(empty, doc, meta, info)["ok"])


if __name__ == "__main__":
    unittest.main()
