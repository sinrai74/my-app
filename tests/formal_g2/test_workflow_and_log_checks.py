"""正式G2：評価・集約 workflow と、評価ジョブのログ・環境の検査（2026-11-06、§6.8 B の条件1・3・5・6）。

検査の層（「ログに *** が無い」だけを合格の根拠にしない）：
- E1 workflow の静的な検査（このファイル）
- E2 アクションの入力（ジョブログ）
- E3 ジョブの GITHUB_TOKEN Permissions の表示（ジョブログ。想定の表示は Actions で確かめるまで未決定）
- E4 評価プロセスの環境（run.json の checks.token）
- E5 取得のステップの証跡（ジョブログの g2_fetch_head・g2_credentials_check と API の head_sha）
あわせて、集約側の基準の照合と、登録の確認ができない場合に正式な合格を出さないことを確かめる。

ここで確かめられないこと（Actions でしか確かめられない）：permissions: {} での artifact の保存、
匿名での取得、E3 の実際の表示、setup-python の token: "" での取得、ACTIONS_RUNTIME_TOKEN の実際の扱い。

実行（リポジトリ直下で）：
  PYTHONPATH="rebuild:." python -P -B -m unittest discover -s tests/formal_g2 -t tests/formal_g2
"""

import hashlib
import json
import os
import re
import sys
import tempfile
import unittest

from formal_g2.aggregate import main as agg
from test_aggregate import (BASELINE, BASELINE_RAW, BASELINE_SHA, EVAL_LOG_TAIL, EXPECTED_TOKEN_PERMISSIONS,
                            FORMAL_OK, PERMS, Q8_LOG_HEAD, _race, _write_run)

ROOT = os.path.realpath(os.getcwd())
EVAL_WF = ".github/workflows/formal_g2_evaluate.yml"
AGG_WF = ".github/workflows/formal_g2_aggregate.yml"
PINNED = {
    "actions/checkout": "11d5960a326750d5838078e36cf38b85af677262",  # v4.4.0
    "actions/setup-python": "a26af69be951a213d495a4c3e4e4022e16d87065",  # v5.6.0
    "actions/upload-artifact": "ea165f8d65b6e75b540449e92b4886f43607fa02",  # v4.6.2
}


def _read(rel):
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def _code(rel):
    return "\n".join(ln for ln in _read(rel).splitlines() if not ln.lstrip().startswith("#"))


def _steps(code):
    """`      - name: ` で区切ったステップの本文（名前 -> 本文）。"""
    parts = re.split(r"\n      - name: ", code)
    return {p.splitlines()[0].strip(): p for p in parts[1:]}


def _uses(code):
    return re.findall(r"uses:\s*([^\s#]+)", code)


class E1EvaluateWorkflowStaticTests(unittest.TestCase):
    def test_no_permissions_and_no_token_reference(self):
        code = _code(EVAL_WF)
        self.assertRegex(code, r"\npermissions: \{\}\n")
        for word in ("GITHUB_TOKEN", "GH_TOKEN", "github.token", "secrets.", "actions/checkout",
                     "persist-credentials", "extraheader =", "credential.helper=store"):
            self.assertNotIn(word, code)

    def test_uses_only_pinned_full_sha(self):
        for ref in _uses(_code(EVAL_WF)):
            name, _, sha = ref.partition("@")
            with self.subTest(uses=ref):
                self.assertIn(name, ("actions/setup-python", "actions/upload-artifact"))
                self.assertRegex(sha, r"^[0-9a-f]{40}$")
                self.assertEqual(sha, PINNED[name])

    def test_setup_python_token_is_explicitly_empty(self):
        step = _steps(_code(EVAL_WF))["Set up Python"]
        self.assertRegex(step, r'\n\s+token: ""\n')
        self.assertIn('python-version: "3.11.9"', step)

    def test_fetch_step_checks_head_tree_and_credentials(self):
        step = _steps(_code(EVAL_WF))["Fetch source anonymously"]
        self.assertIn('fetch --no-tags --depth=1 "https://github.com/${REPO}.git" "${SHA}"', step)
        self.assertIn("-c credential.helper=", step)
        self.assertIn('if [ "${head}" != "${SHA}" ]', step)  # HEAD の検証は取得のステップの中
        self.assertIn("git status --porcelain", step)
        self.assertIn("extraheader", step)
        self.assertIn(".git-credentials", step)
        self.assertIn("GIT_ASKPASS", step)
        self.assertIn("token[^=]*=", step)  # TOKEN を含む環境変数が無いこと
        self.assertIn('echo "g2_credentials_check=none"', step)
        self.assertIn('echo "g2_fetch_head=${head}"', step)
        # 取得の失敗・不一致では止まる（後のステップへ進まない）
        self.assertIn("set -euo pipefail", step)
        self.assertGreaterEqual(step.count("exit 1"), 4)

    def test_fetch_is_first_step_and_no_env_passes_token(self):
        names = list(_steps(_code(EVAL_WF)))
        self.assertEqual(names[0], "Fetch source anonymously")
        self.assertEqual(_steps(_code(EVAL_WF))["Evaluate"].count("${{ secrets"), 0)

    def test_runs_on_is_fixed(self):
        self.assertIn("runs-on: ubuntu-24.04", _code(EVAL_WF))


class E1AggregateWorkflowStaticTests(unittest.TestCase):
    def test_permissions_and_pinned_actions(self):
        code = _code(AGG_WF)
        self.assertRegex(code, r"permissions:\n  contents: read\n  actions: read\n\n")
        for ref in _uses(code):
            name, _, sha = ref.partition("@")
            with self.subTest(uses=ref):
                self.assertEqual(sha, PINNED[name])

    def test_api_fields_come_from_attempts_response(self):
        step = _steps(_code(AGG_WF))["Download artifacts (zip) and digests"]
        self.assertIn('att_json=$(gh api "repos/${REPO}/actions/runs/${rid}/attempts/${att}")', step)
        for field, jq in (("api_run_id", ".id"), ("api_run_attempt", ".run_attempt"), ("api_head_sha", ".head_sha"),
                          ("api_head_branch", ".head_branch"), ("api_workflow_path", ".path"),
                          ("api_repository", ".repository.full_name"), ("api_event", ".event")):
            with self.subTest(field=field):
                self.assertIn(f"{field}=$(jq -r '{jq}", step)
                self.assertIn(f"{field}:${field}", step)

    def test_aggregate_step_passes_aggregate_baseline_and_tools(self):
        step = _steps(_code(AGG_WF))["Aggregate"]
        self.assertIn("--aggregate-baseline", step)
        self.assertIn("--tools-json", step)
        self.assertNotIn("GH_TOKEN", step)


class LogTextTests(unittest.TestCase):
    """E2・E3・E5 の取り出し（合成ログ）。"""

    def test_clean_log(self):
        r = agg.eval_log_check(Q8_LOG_HEAD + EVAL_LOG_TAIL)
        self.assertTrue(r["token_inputs"]["ok"])
        self.assertEqual(r["token_permissions"]["display"], EXPECTED_TOKEN_PERMISSIONS)
        self.assertEqual(r["fetch"]["head_count"], 1)
        self.assertEqual(r["fetch"]["heads"], ["0" * 40])
        self.assertEqual(r["fetch"]["credentials_check_count"], 1)

    def test_echo_line_in_script_header_is_not_counted(self):
        # スクリプトの表示（echo "g2_fetch_head=${head}"）は数えない
        r = agg.eval_log_check(EVAL_LOG_TAIL)
        self.assertEqual(r["fetch"]["head_count"], 1)

    def test_token_input_is_detected(self):
        for line in ("  token: ***", "  github-token: ***", "  TOKEN: ***"):
            with self.subTest(line=line):
                text = Q8_LOG_HEAD + EVAL_LOG_TAIL + f"2026-11-02T06:40:12.0Z {line}\n"
                self.assertFalse(agg.eval_log_check(text)["token_inputs"]["ok"])
        # 空の token は合格（setup-python の token: ""）
        text = Q8_LOG_HEAD + EVAL_LOG_TAIL + "2026-11-02T06:40:12.0Z   token: \n"
        self.assertTrue(agg.eval_log_check(text)["token_inputs"]["ok"])

    def test_permissions_group_missing_or_duplicated(self):
        self.assertIsNotNone(agg.eval_log_check(Q8_LOG_HEAD)["token_permissions"]["problem"])
        self.assertIsNotNone(agg.eval_log_check(EVAL_LOG_TAIL + EVAL_LOG_TAIL)["token_permissions"]["problem"])


class CheckFunctionTests(unittest.TestCase):
    def _log(self, text):
        return agg.eval_log_check(text)

    def test_fetch_check(self):
        meta = {"api_head_sha": "0" * 40}
        self.assertEqual(agg.fetch_check(self._log(EVAL_LOG_TAIL), meta), [])
        self.assertTrue(agg.fetch_check(self._log(EVAL_LOG_TAIL), {"api_head_sha": "1" * 40}))
        self.assertTrue(agg.fetch_check(self._log(EVAL_LOG_TAIL), {}))  # API の値が無い
        self.assertTrue(agg.fetch_check(self._log(Q8_LOG_HEAD), meta))  # 証跡が無い
        self.assertTrue(agg.fetch_check(self._log(EVAL_LOG_TAIL + EVAL_LOG_TAIL), meta))  # 2行ある
        self.assertTrue(agg.fetch_check(None, meta))

    def test_token_permissions_check(self):
        log = self._log(EVAL_LOG_TAIL)
        self.assertEqual(agg.token_permissions_check(log, EXPECTED_TOKEN_PERMISSIONS), [])
        self.assertTrue(agg.token_permissions_check(log, None))  # 想定の表示が未決定 -> 不合格
        self.assertTrue(agg.token_permissions_check(log, ["Contents: read", "Metadata: read"]))
        self.assertTrue(agg.token_permissions_check(self._log(Q8_LOG_HEAD), EXPECTED_TOKEN_PERMISSIONS))

    def test_process_env_check(self):
        ok = {"checks": {"token": {"github_token_present": False, "token_like_env_names": []}}}
        self.assertEqual(agg.process_env_check(ok), [])
        for bad in ({"checks": {}}, {"checks": {"token": {"github_token_present": True, "token_like_env_names": []}}},
                    {"checks": {"token": {"github_token_present": False,
                                          "token_like_env_names": ["ACTIONS_RUNTIME_TOKEN"]}}},
                    {"checks": {"token": {"token_like_env_names": []}}}):
            with self.subTest(bad=bad):
                self.assertTrue(agg.process_env_check(bad))


class AggregateBaselineTests(unittest.TestCase):
    def _write(self, tmp, obj):
        p = os.path.join(tmp, "agg_base.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        return p

    def _good(self):
        return {"kind": "aggregate", "files": agg.self_version(), "python_version": list(sys.version_info[:3])}

    def test_ok_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(agg.aggregate_baseline_check(self._write(tmp, self._good()))["status"], "ok")
            bad = self._good()
            first = sorted(bad["files"])[0]
            bad["files"][first] = "0" * 64
            r = agg.aggregate_baseline_check(self._write(tmp, bad))
            self.assertEqual(r["status"], "mismatch")
            self.assertIn(f"file:{first}", r["diffs"])
            bad = self._good()
            bad["python_version"] = [3, 11, 10]
            self.assertEqual(agg.aggregate_baseline_check(self._write(tmp, bad))["diffs"], ["python_version"])
            bad = self._good()
            bad["files"].pop(first)
            self.assertEqual(agg.aggregate_baseline_check(self._write(tmp, bad))["status"], "mismatch")

    def test_missing_unreadable_invalid(self):
        self.assertEqual(agg.aggregate_baseline_check(None)["status"], "no_baseline")
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "x.json")
            with open(p, "w") as f:
                f.write("{not json")
            self.assertEqual(agg.aggregate_baseline_check(p)["status"], "read_error")
            self.assertEqual(agg.aggregate_baseline_check(os.path.join(tmp, "none.json"))["status"], "read_error")
            self.assertEqual(agg.aggregate_baseline_check(self._write(tmp, {"kind": "evaluate", "files": {}}))["status"],
                             "invalid")

    def test_self_version_lists_four_files(self):
        self.assertEqual(sorted(agg.self_version()), sorted([
            "rebuild/formal_g2/aggregate/main.py", "rebuild/formal_g2/aggregate/__init__.py",
            "rebuild/formal_g2/aggregate/fetch_log.py", AGG_WF]))
        self.assertTrue(all(v for v in agg.self_version().values()))


class FormalGateTests(unittest.TestCase):
    """加算されうる実行に当てはめる条件と、止まった実行にも当てはめる条件。"""

    def _agg(self, ctx=None, log_text=None, run_overrides=None, check_overrides=None, meta_edit=None):
        with tempfile.TemporaryDirectory() as base:
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], log_text=log_text,
                           run_overrides=run_overrides, check_overrides=check_overrides)
            if meta_edit:
                p = os.path.join(d, "meta.json")
                with open(p, encoding="utf-8") as f:
                    m = json.load(f)
                m.update(meta_edit)
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(m, f)
            v = [agg.verify_run(os.path.join(base, x)) for x in os.listdir(base)]
            return agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, ctx)

    def test_all_conditions_met_counts(self):
        res = self._agg(FORMAL_OK)
        self.assertEqual(res["best_segment"], 1)
        self.assertEqual(res["days"][0]["registration_check"], "ok")

    def test_default_context_never_passes(self):
        res = self._agg(None)  # 既定：集約側の基準・E3 の想定・登録の確認、どれも無い
        self.assertEqual(res["best_segment"], 0)
        self.assertEqual(res["days"][0]["excluded"], "aggregate_baseline_not_checked")
        self.assertTrue(res["baseline_failures"])

    def test_each_counting_condition(self):
        cases = (
            (dict(FORMAL_OK, aggregate_baseline_check="mismatch"), None, "aggregate_baseline_mismatch"),
            (FORMAL_OK, Q8_LOG_HEAD, None),  # 取得の証跡・権限の表示が無いログ（下で理由を確かめる）
            (dict(FORMAL_OK, expected_eval_token_permissions=None), None, "eval_token_permissions"),
            (dict(FORMAL_OK, registration_verified=False), None, "baseline_registration_unverified"),
        )
        for ctx, log_text, expected in cases:
            with self.subTest(expected=expected, log=bool(log_text)):
                res = self._agg(ctx, log_text=log_text)
                self.assertEqual(res["best_segment"], 0)
                self.assertEqual(res["days"][0]["excluded"], expected or "eval_fetch")
                self.assertEqual(res["baseline_failures"][0]["check"], expected or "eval_fetch")

    def test_fetch_head_must_match_api(self):
        res = self._agg(FORMAL_OK, meta_edit={"api_head_sha": "1" * 40},
                        run_overrides={"info": {"race_date": "20990101", "github_ref": "refs/heads/main",
                                                "github_sha": "1" * 40,
                                                "github_workflow_ref": "o/r/.github/workflows/formal_g2_evaluate.yml@refs/heads/main"}})
        self.assertEqual(res["days"][0]["excluded"], "eval_fetch")

    def test_candidate_baseline_alone_is_not_a_formal_pass(self):
        # 候補の基準と一致し、ほかの条件がそろっていても、登録を確認できなければ合格にしない
        res = self._agg(dict(FORMAL_OK, registration_verified=False))
        self.assertEqual(res["days"][0]["baseline_identity_check"], "ok")
        self.assertEqual(res["days"][0]["excluded"], "baseline_registration_unverified")
        self.assertEqual(res["g2_status"], "undetermined")

    def test_token_input_and_process_env_apply_to_stopped_runs(self):
        stopped = {"status": "stopped", "config_detection": {"detected": True}}
        res = self._agg(FORMAL_OK, run_overrides=stopped, check_overrides={"baseline_identity": None},
                        log_text=Q8_LOG_HEAD + EVAL_LOG_TAIL + "2026-11-02T06:40:12.0Z   token: ***\n")
        self.assertEqual(res["days"][0]["excluded"], "eval_token_input")
        res = self._agg(FORMAL_OK, run_overrides=stopped,
                        check_overrides={"baseline_identity": None,
                                         "token": {"github_token_present": False,
                                                   "token_like_env_names": ["ACTIONS_RUNTIME_TOKEN"]}})
        self.assertEqual(res["days"][0]["excluded"], "eval_process_token_env")
        # 安全停止で、トークンの条件を満たしていれば従来どおり config_detected（加算の条件は当てはめない）
        res = self._agg(None, run_overrides=stopped, check_overrides={"baseline_identity": None})
        self.assertEqual(res["days"][0]["excluded"], "config_detected")
        self.assertEqual(res["baseline_failures"], [])

    def test_investigation_runs_are_not_judged(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], mode="investigation", log_text=Q8_LOG_HEAD)
            v = [agg.verify_run(os.path.join(base, x)) for x in os.listdir(base)]
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA)
            self.assertEqual(res["baseline_failures"], [])
            self.assertEqual(res["best_segment"], 0)


class MainFormalJudgmentTests(unittest.TestCase):
    def test_main_with_valid_aggregate_baseline_still_has_no_formal_pass(self):
        from contextlib import redirect_stdout
        from io import StringIO
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            paths = {}
            for name, data in (("b.json", BASELINE_RAW), ("p.json", json.dumps(PERMS).encode()),
                               ("ab.json", json.dumps({"kind": "aggregate", "files": agg.self_version(),
                                                       "python_version": list(sys.version_info[:3])}).encode())):
                paths[name] = os.path.join(tmp, name)
                with open(paths[name], "wb") as f:
                    f.write(data)
            out = os.path.join(tmp, "out")
            with redirect_stdout(StringIO()):
                rc = agg.main(["--artifacts-dir", base, "--first-date", "20990101", "--last-date", "20990101",
                               "--out", out, "--baseline", paths["b.json"], "--permissions", paths["p.json"],
                               "--aggregate-baseline", paths["ab.json"],
                               "--tools-json", json.dumps({"gh": "gh version x", "jq": "jq-1.7"})])
            with open(os.path.join(out, "aggregate_result.json"), encoding="utf-8") as f:
                res = json.load(f)
        self.assertEqual(res["aggregate_baseline"]["status"], "ok")
        self.assertEqual(rc, agg.EXIT_VERIFICATION_FAILED)
        # E3 の想定の表示が未決定なので、登録の確認より先にそこで止まる
        self.assertEqual(res["days"][0]["excluded"], "eval_token_permissions")
        self.assertEqual(res["best_segment"], 0)
        self.assertEqual(res["aggregate_environment"]["tools"], {"gh": "gh version x", "jq": "jq-1.7"})
        self.assertIs(res["formal_judgment"]["registration_verified"], False)
        self.assertEqual(res["runs"][0]["eval_fetch"], [])
        self.assertEqual(res["runs"][0]["eval_process_env"], [])


if __name__ == "__main__":
    unittest.main()
