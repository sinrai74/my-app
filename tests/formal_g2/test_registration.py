"""baseline の登録の確認（案A。07h §2.8、2026-11-08 に本人が採用）のテスト。

承認記録 g2_baseline/registered/approvals.json と、--baseline・--aggregate-baseline に渡した基準の
パス・内容の SHA-256 を照合する。どれか1つでも満たさなければ、登録を確認できない（正式な合格を出さない）。
"""
import hashlib
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

from formal_g2.aggregate import main as agg
from test_aggregate import BASELINE_RAW, BASELINE_SHA, PERMS, _race, _write_run

EVAL_REL = "g2_baseline/registered/evaluate/baseline_evaluate.json"
AGG_REL = "g2_baseline/registered/aggregate/baseline_aggregate.json"


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _agg_baseline_raw() -> bytes:
    base = {"kind": "aggregate", "files": agg.self_version(), "python_version": list(sys.version_info[:3])}
    return json.dumps(base, sort_keys=True).encode("utf-8")


def _write(root: str, rel: str, data: bytes) -> str:
    p = os.path.join(root, *rel.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "wb") as f:
        f.write(data)
    return p


def _entry(t, rel, sha, date="2026-11-08", **extra):
    return dict({"type": t, "path": rel, "sha256": sha, "approved_on": date}, **extra)


def _registry(entries, **override):
    reg = {"kind": agg.REGISTRY_KIND, "format_version": agg.REGISTRY_FORMAT_VERSION, "entries": entries}
    reg.update(override)
    return json.dumps(reg, ensure_ascii=False).encode("utf-8")


class RegistrationCheckTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.agg_raw = _agg_baseline_raw()
        self.eval_path = _write(self.root, EVAL_REL, BASELINE_RAW)
        self.agg_path = _write(self.root, AGG_REL, self.agg_raw)
        self.entries = [_entry("evaluate", EVAL_REL, BASELINE_SHA), _entry("aggregate", AGG_REL, _sha(self.agg_raw))]

    def tearDown(self):
        self._tmp.cleanup()

    def _check(self, registry=None, eval_path=None, eval_sha=BASELINE_SHA, agg_path=None, agg_sha=None):
        if registry is not None:
            _write(self.root, agg.REGISTRY_REL, registry)
        return agg.registration_check(self.root,
                                      self.eval_path if eval_path is None else eval_path, eval_sha,
                                      self.agg_path if agg_path is None else agg_path,
                                      _sha(self.agg_raw) if agg_sha is None else agg_sha)

    def test_valid_registration_is_ok(self):
        res = self._check(_registry(self.entries))
        self.assertEqual(res["status"], "ok", res["problems"])
        self.assertEqual(res["entries"]["evaluate"]["sha256"], BASELINE_SHA)
        self.assertEqual(res["problems"], [])
        # note は任意の項目として許す
        notes = [dict(e, note="本人承認 2026-11-08") for e in self.entries]
        self.assertEqual(self._check(_registry(notes))["status"], "ok")

    def test_missing_registry(self):
        self.assertEqual(self._check()["status"], "registry_missing")

    def test_candidate_files_alone_are_not_registration(self):
        # 基準のファイルが registered に置かれていても、承認記録が無ければ登録とみなさない
        self.assertTrue(os.path.isfile(self.eval_path))
        self.assertNotEqual(self._check()["status"], "ok")

    def test_invalid_registry_structure(self):
        cases = {
            "not_json": b"{not json",
            "array": b"[]",
            "wrong_kind": _registry(self.entries, kind="other"),
            "wrong_version": _registry(self.entries, format_version=2),
            "bool_version": _registry(self.entries, format_version=True),
            "extra_top_key": _registry(self.entries, extra=1),
            "entries_not_list": _registry({"a": 1}),
            "empty": _registry([]),
            "missing_aggregate": _registry(self.entries[:1]),
            "duplicate_evaluate": _registry(self.entries + [self.entries[0]]),
        }
        for name, raw in cases.items():
            with self.subTest(case=name):
                self.assertEqual(self._check(raw)["status"], "registry_invalid")

    def test_invalid_entries(self):
        ev = self.entries[0]
        bad = {
            "missing_date": {k: v for k, v in ev.items() if k != "approved_on"},
            "missing_sha": {k: v for k, v in ev.items() if k != "sha256"},
            "extra_key": dict(ev, approved_by="x"),
            "bad_type": dict(ev, type="eval"),
            "path_other_dir": dict(ev, path="g2_baseline/candidates/baseline_evaluate.json"),
            "path_wrong_type_dir": dict(ev, path="g2_baseline/registered/aggregate/baseline_evaluate.json"),
            "path_dotdot": dict(ev, path="g2_baseline/registered/evaluate/../evaluate/baseline_evaluate.json"),
            "path_subdir": dict(ev, path="g2_baseline/registered/evaluate/x/baseline_evaluate.json"),
            "path_absolute": dict(ev, path="/g2_baseline/registered/evaluate/baseline_evaluate.json"),
            "path_not_json": dict(ev, path="g2_baseline/registered/evaluate/baseline_evaluate.txt"),
            "sha_upper": dict(ev, sha256=BASELINE_SHA.upper()),
            "sha_short": dict(ev, sha256=BASELINE_SHA[:63]),
            "date_format": dict(ev, approved_on="2026/11/08"),
            "date_invalid": dict(ev, approved_on="2026-02-30"),
            "note_not_str": dict(ev, note=1),
            "not_object": "evaluate",
        }
        for name, e in bad.items():
            with self.subTest(case=name):
                res = self._check(_registry([e, self.entries[1]]))
                self.assertEqual(res["status"], "registry_invalid")

    def test_duplicate_json_keys_are_rejected(self):
        ev, ag = self.entries
        top = ('{"kind": "%s", "kind": "%s", "format_version": 1, "entries": %s}'
               % (agg.REGISTRY_KIND, agg.REGISTRY_KIND, json.dumps(self.entries))).encode("utf-8")
        entry = json.dumps(ev)[:-1] + ', "sha256": "%s"}' % ("0" * 64)  # 後の値で上書きされる形
        inner = ('{"kind": "%s", "format_version": 1, "entries": [%s, %s]}'
                 % (agg.REGISTRY_KIND, entry, json.dumps(ag))).encode("utf-8")
        for name, raw in (("top_level", top), ("entry", inner)):
            with self.subTest(case=name):
                res = self._check(raw)
                self.assertEqual(res["status"], "registry_invalid")
                self.assertIn("duplicate key", res["problems"][0])

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink not available")
    def test_registry_through_symlinked_directory_is_rejected(self):
        real = os.path.join(self.root, "elsewhere")
        os.makedirs(real)
        with open(os.path.join(real, "approvals.json"), "wb") as f:
            f.write(_registry(self.entries))
        reg_dir = os.path.join(self.root, "g2_baseline", "registered")
        os.replace(reg_dir, os.path.join(self.root, "moved"))
        os.makedirs(os.path.join(self.root, "g2_baseline"), exist_ok=True)
        os.symlink(real, reg_dir)
        self.assertNotEqual(agg.registration_check(self.root, self.eval_path, BASELINE_SHA,
                                                   self.agg_path, _sha(self.agg_raw))["status"], "ok")
        self.assertEqual(agg.registration_check(self.root, None, None, None, None)["status"], "registry_invalid")

    def test_unreadable_registered_file_fails_closed(self):
        reg = _registry(self.entries)
        _write(self.root, agg.REGISTRY_REL, reg)
        real_open = open
        target = os.path.join(self.root, *EVAL_REL.split("/"))

        def fake_open(path, *a, **kw):
            if os.path.abspath(path) == os.path.abspath(target):
                raise PermissionError("denied")
            return real_open(path, *a, **kw)

        with mock.patch("builtins.open", side_effect=fake_open):
            res = agg.registration_check(self.root, self.eval_path, BASELINE_SHA, self.agg_path, _sha(self.agg_raw))
        self.assertEqual(res["status"], "registered_file_invalid")

    def test_registered_file_missing_or_changed(self):
        os.remove(self.agg_path)
        self.assertEqual(self._check(_registry(self.entries))["status"], "registered_file_invalid")
        _write(self.root, AGG_REL, self.agg_raw + b" ")
        self.assertEqual(self._check(_registry(self.entries))["status"], "registered_file_invalid")

    @unittest.skipUnless(hasattr(os, "symlink"), "symlink not available")
    def test_registered_file_symlink_is_rejected(self):
        outside = os.path.join(self.root, "outside.json")
        with open(outside, "wb") as f:
            f.write(BASELINE_RAW)
        os.remove(self.eval_path)
        os.symlink(outside, self.eval_path)
        self.assertEqual(self._check(_registry(self.entries))["status"], "registered_file_invalid")

    def test_used_baseline_must_match_registration(self):
        reg = _registry(self.entries)
        copy_elsewhere = _write(self.root, "g2_baseline/candidates/baseline_evaluate.json", BASELINE_RAW)
        cases = {
            "evaluate_not_given": dict(eval_path="", eval_sha=None),
            "evaluate_unreadable": dict(eval_sha=None),
            "evaluate_same_content_other_path": dict(eval_path=copy_elsewhere),
            "evaluate_sha_differs": dict(eval_sha="0" * 64),
        }
        for name, kw in cases.items():
            with self.subTest(case=name):
                self.assertEqual(self._check(reg, **kw)["status"], "evaluate_baseline_not_registered")
        cases = {
            "aggregate_not_given": dict(agg_path="", agg_sha=""),
            "aggregate_sha_differs": dict(agg_sha="0" * 64),
            "aggregate_path_differs": dict(agg_path=_write(self.root, "elsewhere/a.json", self.agg_raw)),
        }
        for name, kw in cases.items():
            with self.subTest(case=name):
                self.assertEqual(self._check(reg, **kw)["status"], "aggregate_baseline_not_registered")

    def test_relative_paths_are_resolved_from_cwd(self):
        reg = _registry(self.entries)
        cwd = os.getcwd()
        try:
            os.chdir(self.root)
            res = self._check(reg, eval_path=EVAL_REL, agg_path=AGG_REL)
        finally:
            os.chdir(cwd)
        self.assertEqual(res["status"], "ok", res["problems"])


class MainRegistrationTests(unittest.TestCase):
    """main()：登録を確認できたときだけ正式な合格を出す（ほかの条件はすべて満たす実行で確かめる）。"""

    def _main(self, tmp, registry_entries="valid", eval_rel=EVAL_REL):
        root = os.path.join(tmp, "repo")
        agg_raw = _agg_baseline_raw()
        eval_path = _write(root, eval_rel, BASELINE_RAW)
        agg_path = _write(root, AGG_REL, agg_raw)
        if registry_entries == "valid":
            registry_entries = [_entry("evaluate", EVAL_REL, BASELINE_SHA), _entry("aggregate", AGG_REL, _sha(agg_raw))]
        if registry_entries is not None:
            _write(root, agg.REGISTRY_REL, _registry(registry_entries))
        base = os.path.join(tmp, "a")
        os.makedirs(base)
        _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
        perms = os.path.join(tmp, "permissions.json")
        with open(perms, "w", encoding="utf-8") as f:
            json.dump(PERMS, f)
        out = os.path.join(tmp, "out")
        buf = StringIO()
        with mock.patch.object(agg, "_repo_root", return_value=root), redirect_stdout(buf):
            rc = agg.main(["--artifacts-dir", base, "--first-date", "20990101", "--last-date", "20990101",
                           "--out", out, "--baseline", eval_path, "--aggregate-baseline", agg_path,
                           "--permissions", perms])
        with open(os.path.join(out, "aggregate_result.json"), encoding="utf-8") as f:
            return rc, json.load(f), buf.getvalue()

    def test_registered_baselines_allow_formal_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, log = self._main(tmp)
        self.assertEqual(res["registration"]["status"], "ok", res["registration"]["problems"])
        self.assertIs(res["formal_judgment"]["registration_verified"], True)
        self.assertEqual(res["aggregate_baseline"]["status"], "ok")
        self.assertEqual(res["days"][0]["registration_check"], "ok")
        self.assertNotIn("excluded", res["days"][0])
        self.assertEqual(res["best_segment"], 1)
        self.assertEqual(rc, 0)
        self.assertIn("registration_verified=True registration=ok", log)

    def test_unregistered_cases_never_pass(self):
        cases = {
            "no_registry": dict(registry_entries=None),
            "evaluate_candidate_not_in_registered_dir": dict(
                eval_rel="g2_baseline/candidates/baseline_evaluate.json"),
            "registry_sha_mismatch": dict(registry_entries=[
                _entry("evaluate", EVAL_REL, "0" * 64),
                _entry("aggregate", AGG_REL, _sha(_agg_baseline_raw()))]),
        }
        for name, kw in cases.items():
            with self.subTest(case=name), tempfile.TemporaryDirectory() as tmp:
                rc, res, log = self._main(tmp, **kw)
                self.assertNotEqual(res["registration"]["status"], "ok")
                self.assertIs(res["formal_judgment"]["registration_verified"], False)
                self.assertEqual(res["days"][0]["excluded"], "baseline_registration_unverified")
                self.assertEqual(res["best_segment"], 0)
                self.assertEqual(rc, agg.EXIT_VERIFICATION_FAILED)
                self.assertIn("registration_verified=False", log)

    def test_unexpected_exception_in_check_is_not_registration(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(agg, "registration_check", side_effect=RuntimeError("boom")):
            rc, res, log = self._main(tmp)
        self.assertEqual(res["registration"]["status"], "check_error")
        self.assertIs(res["formal_judgment"]["registration_verified"], False)
        self.assertEqual(res["days"][0]["excluded"], "baseline_registration_unverified")
        self.assertEqual(res["best_segment"], 0)
        self.assertEqual(rc, agg.EXIT_VERIFICATION_FAILED)

    def test_repository_has_no_registry_yet(self):
        # 今のリポジトリには承認記録が無い（登録は本人の許可を得て別に行う）
        self.assertFalse(os.path.exists(os.path.join(agg._repo_root(), agg.REGISTRY_REL)))


if __name__ == "__main__":
    unittest.main()
