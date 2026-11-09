"""正式G2：M7-3 Runner Image（ジョブログ由来）の照合と、環境変数の観測値の分離。

Q8（run 41401359888）の値を使う：ジョブログの Runner Image は 20261025.353.1、
artifact の環境変数 ImageVersion は 20261026.354.1。

実行（リポジトリ直下で）：
  PYTHONPATH="rebuild:." python -P -B -m unittest discover -s tests/formal_g2 -t tests/formal_g2
"""

import hashlib
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from formal_g2 import env_baseline as eb
from formal_g2.aggregate import main as agg
from test_aggregate import BASELINE, BASELINE_RAW, BASELINE_SHA, FORMAL_OK, PERMS, Q8_ENV_IMAGE, Q8_LOG_HEAD, _race, _write_run, permitted_by

ROOT = os.path.realpath(os.getcwd())


def _verify_all(base):
    return [agg.verify_run(os.path.join(base, x)) for x in sorted(os.listdir(base))]


class ParseTests(unittest.TestCase):
    """1・3：抽出と、グループ欠落・解析失敗。"""

    def test_extracts_q8_values(self):
        r = agg.parse_runner_image(Q8_LOG_HEAD)
        self.assertTrue(r["ok"], r["problems"])
        self.assertEqual((r["image"], r["version"]), ("ubuntu-24.04", "20261025.353.1"))
        self.assertEqual(r["lines"], {"Image": 4, "Version": 5})

    def test_group_missing(self):
        text = Q8_LOG_HEAD.replace("##[group]Runner Image", "##[group]Runner Imagex")
        self.assertFalse(agg.parse_runner_image(text)["ok"])

    def test_group_duplicated(self):
        self.assertFalse(agg.parse_runner_image(Q8_LOG_HEAD + Q8_LOG_HEAD)["ok"])

    def test_group_not_closed(self):
        text = Q8_LOG_HEAD.rsplit("2026-11-02T06:40:09.9053848Z ##[endgroup]", 1)[0]
        self.assertFalse(agg.parse_runner_image(text)["ok"])

    def test_version_line_missing(self):
        text = Q8_LOG_HEAD.replace("Version: 20261025.353.1", "Ver: 20261025.353.1")
        r = agg.parse_runner_image(text)
        self.assertFalse(r["ok"])
        self.assertIsNone(r["version"])

    def test_version_unparseable(self):
        for bad in ("Version: ", "Version: 2026-10-25", "Version: latest"):
            with self.subTest(bad=bad):
                text = Q8_LOG_HEAD.replace("Version: 20261025.353.1", bad)
                self.assertFalse(agg.parse_runner_image(text)["ok"])

    def test_values_outside_group_are_ignored(self):
        text = ("2026-11-02T06:40:09.9Z Version: 20990101.1.1\n"
                + Q8_LOG_HEAD.replace("Version: 20261025.353.1", "Note: none"))
        self.assertFalse(agg.parse_runner_image(text)["ok"])


class LogVerificationTests(unittest.TestCase):
    """2・3・7：取得失敗・欠落・改変は不合格。環境変数で代わりにしない。"""

    def test_fetch_failure_is_not_verified(self):
        with tempfile.TemporaryDirectory() as base:
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], log_fetch_ok=False)
            v = agg.verify_run(d)
            self.assertFalse(v["verified"])
            self.assertTrue(any("log fetch failed" in p for p in v["problems"]))
            self.assertIsNone(v["runner_image"]["log"])
            # 環境変数の値はあるが、代わりには使わない
            self.assertEqual(v["runner_image"]["env_observed"]["ImageVersion"], "20261026.354.1")

    def test_missing_log_file_is_not_verified(self):
        with tempfile.TemporaryDirectory() as base:
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            os.remove(os.path.join(d, "job.log"))
            self.assertFalse(agg.verify_run(d)["verified"])

    def test_tampered_log_is_not_verified(self):
        with tempfile.TemporaryDirectory() as base:
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            with open(os.path.join(d, "job.log"), "a", encoding="utf-8") as f:
                f.write("x\n")
            v = agg.verify_run(d)
            self.assertFalse(v["verified"])
            self.assertIn("job log: log sha256 mismatch", v["problems"])

    def test_group_missing_is_not_verified(self):
        with tempfile.TemporaryDirectory() as base:
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"],
                           log_text="2026-11-02T06:40:09.9Z Current runner version: '2.338.2'\n")
            self.assertFalse(agg.verify_run(d)["verified"])


class BaselineTests(unittest.TestCase):
    """4・5・6：ログ値で照合し、環境変数の値は観測値として別に保持する。"""

    def test_log_value_is_used_even_when_env_differs(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            v = _verify_all(base)
            self.assertTrue(v[0]["verified"], v[0]["problems"])
            ri = v[0]["runner_image"]
            self.assertEqual(ri["log"], {"image": "ubuntu-24.04", "version": "20261025.353.1"})
            self.assertEqual(ri["env_observed"], Q8_ENV_IMAGE)
            self.assertTrue(ri["env_version_differs_from_log"])
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)
            self.assertEqual(res["days"][0]["runner_image_check"], "ok")
            self.assertEqual(res["best_segment"], 1)

    def test_log_value_differs_from_baseline_is_A1(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            # 2026-11-06：Image が違えば除外する（Version は照合しない）
            other = {"runner_image": {"image": "ubuntu-22.04"}}
            res = agg.aggregate(_verify_all(base), PERMS, "20990101", "20990101", other, BASELINE_SHA, FORMAL_OK)
            self.assertEqual(res["days"][0]["excluded"], "runner_image_mismatch")
            self.assertEqual(res["best_segment"], 0)

    def test_version_differs_is_recorded_but_not_excluded(self):
        # 2026-11-06：ジョブログの Version（20261025.353.1）は観測値。基準は Image だけなので除外しない
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            v = _verify_all(base)
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)
            self.assertEqual(v[0]["runner_image"]["log"]["version"], "20261025.353.1")
            self.assertEqual(res["days"][0]["runner_image_check"], "ok")
            self.assertEqual(res["best_segment"], 1)

    def test_no_or_invalid_baseline_is_not_added(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            v = _verify_all(base)
            for b, expected in ((None, "baseline_no_baseline"),
                                ({"runner_image": {"version": "20261025.353.1"}}, "runner_image_baseline_invalid"),
                                # Version を持つ旧形式の基準は形が違うとする
                                ({"runner_image": {"image": "ubuntu-24.04", "version": "20261025.353.1"}},
                                 "runner_image_baseline_invalid"),
                                ({"runner_image": {"image": ""}}, "runner_image_baseline_invalid")):
                with self.subTest(baseline=b):
                    res = agg.aggregate(v, PERMS, "20990101", "20990101", b, BASELINE_SHA if b else None)
                    self.assertEqual(res["days"][0]["excluded"], expected)
                    self.assertEqual(res["best_segment"], 0)

    def _snap(self):
        return {"image": dict(Q8_ENV_IMAGE), "lock": {"sha256": "h", "check": {"ok": True}},
                "settings": {"cfg_has_venue_cfg_key": False}, "code": {"a.py": "x"},
                "motor_history_header_sha256": "h", "python": {"version_info": [3, 11, 9, "final", 0]},
                "installed": {"pip": "24.2"}}

    def test_evaluate_side_ignores_env_image_and_requires_runner_image(self):
        snap = self._snap()
        view = eb.baseline_view(snap)
        self.assertNotIn("image", view)
        self.assertEqual(view["image_env_observed"], {"ImageOS": "ubuntu24", "ImageVersion": "20261026.354.1"})
        registered = dict(view, runner_image=dict(BASELINE["runner_image"]))
        registered["image_env_observed"] = {"ImageOS": "ubuntu24", "ImageVersion": "20261025.353.1"}
        # 環境変数の値の違いだけでは差にしない
        self.assertEqual(eb.compare_to_baseline(snap, registered), [])
        # runner_image が無い・形が違う基準は差にする（＝正式な実行では A-1）
        # 2026-11-06：基準の形は {"image": str} だけ。Version を持つ旧形式・Image が空・Image が無いものは差にする
        for bad in (None, {"image": "ubuntu-24.04", "version": "20261025.353.1"}, {"image": ""},
                    {"version": "20261025.353.1"}, {"image": 1}):
            with self.subTest(bad=bad):
                b = dict(registered)
                if bad is None:
                    b.pop("runner_image")
                else:
                    b["runner_image"] = bad
                self.assertEqual(eb.compare_to_baseline(snap, b), ["runner_image"])

    def test_other_m7_items_are_still_compared(self):
        snap = self._snap()
        registered = dict(eb.baseline_view(snap), runner_image=dict(BASELINE["runner_image"]))
        for key, value in (("python_version_info", [3, 11, 10, "final", 0]), ("lock_sha256", "other"),
                           ("installed", {"pip": "24.3"}), ("settings", {}),
                           ("motor_history_header_sha256", "other")):
            with self.subTest(key=key):
                self.assertEqual(eb.compare_to_baseline(snap, dict(registered, **{key: value})), [key])


class OutputAndExclusionTests(unittest.TestCase):
    """7・8：出力の目録とログ原本の写し、ログ検証失敗の除外、終了コード。"""

    def _main(self, base, out, baseline_path):
        perms = os.path.join(os.path.dirname(baseline_path), "permissions.json")
        with open(perms, "w", encoding="utf-8") as f:
            json.dump(PERMS, f)
        buf = StringIO()
        with permitted_by(), redirect_stdout(buf):
            rc = agg.main(["--artifacts-dir", base, "--first-date", "20990101", "--last-date", "20990101",
                           "--out", out, "--baseline", baseline_path])
        return rc, buf.getvalue()

    def _baseline_file(self, tmp):
        p = os.path.join(tmp, "baseline.json")
        with open(p, "wb") as f:
            f.write(BASELINE_RAW)
        return p

    def test_manifest_lists_log_copy_and_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            out = os.path.join(tmp, "out")
            rc, log = self._main(base, out, self._baseline_file(tmp))
            # 2026-11-06：main() は正式な合格を出さない（集約側の基準が無く、登録も確認できない）ので終了コード 3
            self.assertEqual(rc, agg.EXIT_VERIFICATION_FAILED)
            with open(os.path.join(out, "manifest.json"), encoding="utf-8") as f:
                man = json.load(f)
            self.assertTrue(man["complete"])
            self.assertEqual(sorted(k for k in man["files"] if not k.startswith("canonical_bundle/")),
                             ["aggregate_result.json", "logs/1-1.log"])
            # 段階6（d5）：正本の束も出力の manifest に載る
            self.assertIn("canonical_bundle/g2_evidence/P1/1-1/artifact.zip", man["files"])
            for rel, info in man["files"].items():
                with open(os.path.join(out, rel), "rb") as f:
                    blob = f.read()
                self.assertEqual(info, {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob)})
            with open(os.path.join(d, "meta.json"), encoding="utf-8") as f:
                meta = json.load(f)
            self.assertEqual(man["files"]["logs/1-1.log"]["sha256"], meta["log_sha256"])
            with open(os.path.join(out, "aggregate_result.json"), encoding="utf-8") as f:
                result = json.load(f)
            run = result["runs"][0]
            self.assertEqual(run["runner_image"]["log"]["version"], "20261025.353.1")
            self.assertEqual(run["runner_image"]["env_observed"]["ImageVersion"], "20261026.354.1")
            self.assertEqual(run["job_log"]["sha256"], meta["log_sha256"])
            self.assertEqual(result["days"][0]["runner_image_check"], "ok")
            self.assertEqual(result["days"][0]["excluded"], "aggregate_baseline_no_baseline")
            self.assertEqual(result["best_segment"], 0)
            self.assertIn("runner_image_log={'image': 'ubuntu-24.04', 'version': '20261025.353.1'}", log)

    def test_log_failure_run_is_excluded_and_exit_code_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], log_fetch_ok=False)
            out = os.path.join(tmp, "out")
            rc, log = self._main(base, out, self._baseline_file(tmp))
            self.assertEqual(rc, agg.EXIT_VERIFICATION_FAILED)
            self.assertIn("failed=1", log)
            with open(os.path.join(out, "aggregate_result.json"), encoding="utf-8") as f:
                result = json.load(f)
            self.assertEqual(result["best_segment"], 0)
            self.assertEqual(result["sequence"][0]["break"], "verification_failed")
            self.assertEqual(result["verification_failed_runs"], ["1-1"])
            self.assertTrue(os.path.isfile(os.path.join(out, "manifest.json")))


class PermissionSeparationTests(unittest.TestCase):
    """9：C-4 の権限分離。評価の workflow はトークンを持たず、変更もしていない。"""

    def _read(self, rel):
        with open(os.path.join(ROOT, rel), "rb") as f:
            return f.read()

    # 2026-11-06：評価 workflow を変えた（匿名の取得・permissions: {}・SHA の固定）。旧 Q8 の SHA
    # （8dcd5f8b…）を固定していたテストを、変更後の内容の SHA に置き換えた。この値は評価側の新しい基準の
    # 候補（code）にも入る。workflow を変えるたびに、ここと基準の候補を作り直す。
    EVALUATE_WORKFLOW_SHA256 = "7f73606d7d20c21dc8909410a33fab8cd7e59ccf51d59b19de3da253b22a1129"

    def test_evaluate_workflow_pinned_content(self):
        self.assertEqual(hashlib.sha256(self._read(".github/workflows/formal_g2_evaluate.yml")).hexdigest(),
                         self.EVALUATE_WORKFLOW_SHA256)

    def test_evaluate_workflow_has_no_token(self):
        text = self._read(".github/workflows/formal_g2_evaluate.yml").decode("utf-8")
        code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
        for word in ("GITHUB_TOKEN", "GH_TOKEN", "github.token", "secrets.", "actions:", "actions/checkout"):
            self.assertNotIn(word, code)
        self.assertRegex(code, r"\npermissions: \{\}\n")

    def test_aggregate_token_only_in_download_step(self):
        text = self._read(".github/workflows/formal_g2_aggregate.yml").decode("utf-8")
        code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
        self.assertRegex(code, r"permissions:\n  contents: read\n  actions: read\n\n")
        self.assertNotIn("write", re.search(r"permissions:\n((?:  .*\n)+)", code).group(1))
        steps = re.split(r"\n      - name: ", code)
        with_token = [s.splitlines()[0] for s in steps if "github.token" in s or "GH_TOKEN" in s]
        self.assertEqual(with_token, ["Download artifacts (zip) and digests"])


if __name__ == "__main__":
    unittest.main()
