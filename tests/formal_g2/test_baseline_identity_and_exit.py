"""正式G2：基準の同一性（評価側と集約側の SHA-256）、環境変数の扱い、集約の終了コード。

実行（リポジトリ直下で）：
  PYTHONPATH="rebuild:." python -P -B -m unittest discover -s tests/formal_g2 -t tests/formal_g2
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from formal_g2 import env_baseline as eb
from formal_g2.aggregate import main as agg
from test_aggregate import BASELINE, BASELINE_RAW, BASELINE_SHA, FORMAL_OK, PERMS, _race, _write_run, permitted_by

ROOT = os.path.realpath(os.getcwd())


def _verify_all(base):
    return [agg.verify_run(os.path.join(base, x)) for x in sorted(os.listdir(base))]


def _ident(sha=BASELINE_SHA, end=None, unchanged=True):
    return {"path": "baseline.json", "sha256": sha, "sha256_at_end": sha if end is None else end,
            "unchanged_during_run": unchanged}


class AggregateIdentityTests(unittest.TestCase):
    def _day(self, check_overrides=None, baseline=BASELINE, sha=BASELINE_SHA, run_overrides=None):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], check_overrides=check_overrides,
                       run_overrides=run_overrides)
            return agg.aggregate(_verify_all(base), PERMS, "20990101", "20990101", baseline, sha, FORMAL_OK)

    def test_match(self):
        res = self._day()
        self.assertEqual(res["days"][0]["baseline_identity_check"], "ok")
        self.assertEqual(res["best_segment"], 1)
        self.assertEqual(res["baseline_failures"], [])

    def test_mismatch_unrecorded_changed(self):
        cases = (
            ({"baseline_identity": _ident(sha="0" * 64)}, "baseline_identity_mismatch"),
            ({"baseline_identity": None}, "baseline_identity_unrecorded"),
            ({"baseline_identity": _ident(sha=None)}, "baseline_identity_unrecorded"),
            ({"baseline_identity": _ident(end="1" * 64, unchanged=False)}, "baseline_identity_changed_during_run"),
            ({"baseline_identity": _ident(unchanged=None)}, "baseline_identity_changed_during_run"),
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected, overrides=overrides):
                res = self._day(check_overrides=overrides)
                self.assertEqual(res["days"][0]["excluded"], expected)
                self.assertEqual(res["best_segment"], 0)
                self.assertEqual(res["baseline_failures"][0]["check"], expected)

    def test_no_baseline_is_not_added(self):
        res = self._day(baseline=None, sha=None)
        self.assertEqual(res["days"][0]["excluded"], "baseline_no_baseline")
        self.assertEqual(res["best_segment"], 0)

    def test_stopped_run_is_not_a_baseline_failure(self):
        # 設計どおりの安全停止（例：CONFIG）。基準を記録する前に止まった場合も含む
        res = self._day(run_overrides={"status": "stopped", "config_detection": {"detected": True}},
                        check_overrides={"baseline_identity": None})
        self.assertEqual(res["days"][0]["excluded"], "config_detected")
        self.assertEqual(res["days"][0]["baseline_identity_check"], "not_applicable_stopped_run")
        self.assertEqual(res["baseline_failures"], [])
        res = self._day(run_overrides={"status": "stopped", "preload": {"ok": False}},
                        check_overrides={"baseline_identity": None})
        self.assertEqual(res["baseline_failures"], [])
        self.assertEqual(res["best_segment"], 0)


class ExitCodeTests(unittest.TestCase):
    def _main(self, tmp, baseline_bytes=BASELINE_RAW, baseline_name="baseline.json", **run_kw):
        base = os.path.join(tmp, "a")
        os.makedirs(base)
        _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], **run_kw)
        bpath = os.path.join(tmp, baseline_name)
        if baseline_bytes is not None:
            with open(bpath, "wb") as f:
                f.write(baseline_bytes)
        perms = os.path.join(tmp, "permissions.json")
        with open(perms, "w", encoding="utf-8") as f:
            json.dump(PERMS, f)
        out = os.path.join(tmp, "out")
        buf = StringIO()
        with permitted_by(), redirect_stdout(buf):
            rc = agg.main(["--artifacts-dir", base, "--first-date", "20990101", "--last-date", "20990101",
                           "--out", out, "--baseline", bpath])
        with open(os.path.join(out, "aggregate_result.json"), encoding="utf-8") as f:
            return rc, json.load(f), buf.getvalue()

    def test_identity_ok_but_formal_pass_is_never_given_by_main(self):
        # 2026-11-06 の GPT 判断：登録の確認ができなければ正式な合格を出さない（候補の基準と一致しても）。
        # 旧テスト test_ok_is_0_and_path_does_not_matter（終了コード 0・1件加算）を置き換えた。
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = self._main(tmp, baseline_name="registered_copy_elsewhere.json")
            self.assertEqual(res["days"][0]["baseline_identity_check"], "ok")  # パスは問わず内容で照合（従来どおり）
            self.assertEqual(res["baseline"]["sha256"], BASELINE_SHA)
            self.assertEqual(rc, agg.EXIT_VERIFICATION_FAILED)
            self.assertEqual(res["best_segment"], 0)
            self.assertIs(res["formal_judgment"]["registration_verified"], False)

    def test_identity_mismatch_is_3(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, log = self._main(tmp, check_overrides={"baseline_identity": _ident(sha="0" * 64)})
            self.assertEqual(rc, agg.EXIT_VERIFICATION_FAILED)
            self.assertIn("check=baseline_identity_mismatch", log)

    def test_runner_image_mismatch_is_3(self):
        # 2026-11-06 の GPT 判断：基準の固定条件は Image だけ（Version は照合しない）。Image が違う場合に除外する
        other = dict(BASELINE, runner_image={"image": "ubuntu-22.04"})
        raw = json.dumps(other, sort_keys=True).encode("utf-8")
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = self._main(tmp, baseline_bytes=raw,
                                    check_overrides={"baseline_identity": _ident(sha=hashlib.sha256(raw).hexdigest())})
            self.assertEqual(rc, agg.EXIT_VERIFICATION_FAILED)
            self.assertEqual(res["days"][0]["excluded"], "runner_image_mismatch")

    def test_unreadable_baseline_is_3(self):
        for raw in (None, b"{not json", b"[1, 2]"):
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as tmp:
                rc, res, _ = self._main(tmp, baseline_bytes=raw)
                self.assertEqual(rc, agg.EXIT_VERIFICATION_FAILED)
                self.assertIsNotNone(res["baseline"]["read_error"])
                self.assertEqual(res["best_segment"], 0)

    def test_safe_stop_is_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = self._main(tmp, run_overrides={"status": "stopped", "config_detection": {"detected": True}},
                                    check_overrides={"baseline_identity": None})
            self.assertEqual(rc, 0)
            self.assertEqual(res["days"][0]["excluded"], "config_detected")
            self.assertEqual(res["best_segment"], 0)


class EnvObservationTests(unittest.TestCase):
    def _snap(self, image_os, image_version):
        os_ok = {"available": True}
        snap = {"lock": {"sha256": "h", "check": {"ok": True}}, "settings": {"cfg_has_venue_cfg_key": False},
                "code": {"a.py": "x"}, "motor_history_header_sha256": "h", "os": os_ok}
        snap["image"] = {"ImageOS": image_os, "ImageVersion": image_version,
                         "identified": bool(image_os) and bool(image_version)}
        return snap

    def test_missing_env_alone_is_not_ineligible(self):
        for io_, iv in (("ubuntu24", None), (None, "20261026.354.1"), (None, None), ("ubuntu24", "20261026.354.1")):
            with self.subTest(ImageOS=io_, ImageVersion=iv):
                self.assertEqual(eb.self_check(self._snap(io_, iv)), [])

    def test_other_m7_self_checks_remain(self):
        snap = self._snap(None, None)
        snap["lock"]["check"]["ok"] = False
        snap["os"]["available"] = False
        problems = eb.self_check(snap)
        self.assertIn("installed packages do not match requirements-g2.lock", problems)
        self.assertIn("OS information unavailable (os.uname)", problems)

    def test_collect_snapshot_records_unavailable_env(self):
        from unittest import mock
        env = {k: v for k, v in os.environ.items() if k not in ("ImageOS", "ImageVersion")}
        with mock.patch.dict(os.environ, env, clear=True):
            snap = eb.collect_snapshot(ROOT, {})
        self.assertEqual(snap["image"]["unavailable"], ["ImageOS", "ImageVersion"])
        self.assertEqual(snap["image"]["source"], "process environment (observation only)")
        with mock.patch.dict(os.environ, dict(env, ImageOS="ubuntu24", ImageVersion="20261026.354.1"), clear=True):
            snap = eb.collect_snapshot(ROOT, {})
        self.assertEqual(snap["image"]["unavailable"], [])
        self.assertEqual(snap["image"]["ImageVersion"], "20261026.354.1")


# 評価側：run.json に基準の内容の SHA-256 を記録する（合成した出走表。ネットワークは使わない）
_EVAL_CODE = r"""
import json, os, sys
import requests
sys.path.append(os.path.join(os.getcwd(), "tests", "formal_g2"))  # 探索順の先頭は変えない
from test_programs_synthetic import RAW
import formal_g2.programs as P
mode, out, baseline = sys.argv[1], sys.argv[2], sys.argv[3]
def fake(url):
    if mode == "change":
        with open(baseline, "ab") as f:
            f.write(b" ")
    r = requests.models.Response(); r.status_code = 200; r._content = RAW; r.url = url
    return r
P.default_http_get = fake
from formal_g2 import evaluate_main
rc = evaluate_main.main(["--race-date", "20990101", "--mode", "investigation", "--out", out,
                         "--baseline", baseline])
print("RC", rc)
"""


class EvaluateSideIdentityTests(unittest.TestCase):
    def _run(self, mode, baseline_bytes):
        with tempfile.TemporaryDirectory() as tmp:
            bpath = os.path.join(tmp, "baseline.json")
            if baseline_bytes is not None:
                with open(bpath, "wb") as f:
                    f.write(baseline_bytes)
            out = os.path.join(tmp, "out")
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "rebuild"), ROOT]))
            env.pop("GITHUB_TOKEN", None)
            p = subprocess.run([sys.executable, "-B", "-P", "-c", _EVAL_CODE, mode, out, bpath], cwd=ROOT,
                               env=env, capture_output=True, text=True, timeout=600)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            rc = int([ln for ln in p.stdout.splitlines() if ln.startswith("RC ")][-1].split()[1])
            with open(os.path.join(out, "run.json"), encoding="utf-8") as f:
                run = json.load(f)
            stop = None
            if os.path.exists(os.path.join(out, "stop.json")):
                with open(os.path.join(out, "stop.json"), encoding="utf-8") as f:
                    stop = json.load(f)
            return rc, run, stop

    def test_sha_recorded_and_unchanged(self):
        rc, run, stop = self._run("same", BASELINE_RAW)
        ident = run["checks"]["baseline_identity"]
        self.assertEqual(rc, 0, stop)
        self.assertEqual(ident["sha256"], BASELINE_SHA)
        self.assertEqual(ident["sha256_at_end"], BASELINE_SHA)
        self.assertTrue(ident["unchanged_during_run"])
        self.assertIsNone(ident["read_error"])
        self.assertNotIn("_raw", ident)

    def test_change_during_run_stops_A1(self):
        rc, run, stop = self._run("change", BASELINE_RAW)
        self.assertEqual(rc, 2)
        self.assertEqual(stop["code"], "A-1")
        self.assertEqual(stop["reason"], "baseline changed during run")
        self.assertFalse(run["checks"]["baseline_identity"]["unchanged_during_run"])

    def test_unreadable_recorded(self):
        for raw in (None, b"{not json"):
            with self.subTest(raw=raw):
                rc, run, stop = self._run("same", raw)
                self.assertIsNotNone(run["checks"]["baseline_identity"]["read_error"])
                # 調査用の実行は止めない（正式な実行では A-1 で止まる）
                self.assertEqual(rc, 0, stop)


if __name__ == "__main__":
    unittest.main()
