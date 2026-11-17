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
    def test_valid_formal_with_explicit_workflow(self):
        probs = P.permission_file_problems(_doc(), "G3-20261116-01",
                                           expected_workflow=".github/workflows/formal_g3_evaluate.yml")
        self.assertEqual(probs, [])

    def test_workflow_undetermined_is_flagged(self):
        # 既定（G3_PERMISSION_WORKFLOW=None）では workflow 未確定が problems に出る
        probs = P.permission_file_problems(_doc(), "G3-20261116-01")
        self.assertTrue(any("workflow is undetermined" in p for p in probs))

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


if __name__ == "__main__":
    unittest.main()
