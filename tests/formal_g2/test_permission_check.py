"""項目4（2026-11-08 本人決定、GPT の照合条件 1〜12）：正式な実行の許可ファイル g2_permissions/<permission_id>.json と、
集約 workflow が API で取ったその履歴（permission_evidence.json）の照合。

- 集約のプログラムは通信しない。渡された履歴とチェックアウトしたファイルだけで照合する。
- どの失敗も「許可されていない」とする（fail-closed）。許可の無い formal の実行は黙って外さず、違反として記録し、
  達成の判定を止める（終了コード 3）。
- 許可ファイルは本人が GitHub のブラウザ画面で作る。テストは一時ディレクトリの中に作るだけで、リポジトリの
  g2_permissions/ には触れない。
"""
import copy
import json
import os
import re
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

from formal_g2.aggregate import main as agg
from test_aggregate import BASELINE_RAW, FORMAL_OK, _race, _write_run
from test_aggregate import genesis_ok as ta_genesis_ok
from test_aggregate import git_history_ok as ta_git_ok

PID = "G2-20990101-01"
REL = f"g2_permissions/{PID}.json"
RUN_HEAD = "a" * 40
AGG_HEAD = "b" * 40
PSHA = "c" * 40


def doc(**over):
    d = {"schema_version": 1, "permission_id": PID, "race_date": "20990101",
         "window_start": "2099-01-01T00:00:00Z", "window_end": "2099-01-03T00:00:00Z",
         "workflow": ".github/workflows/formal_g2_evaluate.yml", "ref": "refs/heads/main", "mode": "formal",
         "approved_on": "2098-12-31", "approved_by": "sinrai74", "approval_method": "github_web_ui"}
    d.update(over)
    return d


def raw(d):
    return (json.dumps(d, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def evidence(data, **over):
    ev = {"fetch_ok": True, "permission_id": PID, "path": REL, "run_head": RUN_HEAD, "aggregate_head": AGG_HEAD,
          "aggregate_ref": "refs/heads/main",
          "history_run_head": [PSHA], "history_aggregate_head": [PSHA],
          "commits": [{"sha": PSHA, "author_login": "sinrai74", "committer_login": "web-flow", "verified": True,
                       "verification_reason": "valid", "committer_date": "2099-01-01T12:00:00Z",
                       "files_count": 1, "files": [{"filename": REL, "status": "added", "sha": agg._git_blob_sha(data)}]}],
          "compare": {"base": PSHA, "head": RUN_HEAD, "status": "ahead"}}
    ev.update(over)
    return ev


def vr(ev, **info_over):
    info = {"mode": "formal", "permission_id": PID, "race_date": "20990101", "github_ref": "refs/heads/main"}
    info.update(info_over)
    return {"dir": "1-1", "manifest": {"run": info},
            "meta": {"run_started_at": "2099-01-02T00:00:00Z", "api_head_sha": RUN_HEAD,
                     "api_workflow_path": ".github/workflows/formal_g2_evaluate.yml"},
            "permission_evidence": ev}


class _Repo:
    """一時ディレクトリをリポジトリの直下に見立てて、許可ファイルを置く。"""

    def __init__(self, data=None, name=PID):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self.tmp.name)
        if data is not None:
            os.makedirs(os.path.join(self.root, "g2_permissions"))
            with open(os.path.join(self.root, "g2_permissions", f"{name}.json"), "wb") as f:
                f.write(data)

    def __enter__(self):
        return self.root

    def __exit__(self, *a):
        self.tmp.cleanup()


def check(data=None, ev=None, v=None, name=PID):
    data = raw(doc()) if data is None else data
    with _Repo(data, name) as root:
        return agg.permission_check(root, v if v is not None else vr(ev if ev is not None else evidence(data)))


def failed(res):
    return sorted(k for k, x in res["conditions"].items() if x is not True)


class PermissionCheckTests(unittest.TestCase):
    def test_all_12_conditions_ok(self):
        res = check()
        self.assertEqual(res["status"], "ok", res["problems"])
        self.assertEqual(len(res["conditions"]), 12)
        self.assertEqual(failed(res), [])
        self.assertEqual(res["permission"], {"permission_id": PID, "race_date": "20990101",
                                             "window_start": "2099-01-01T00:00:00Z",
                                             "window_end": "2099-01-03T00:00:00Z"})
        self.assertEqual(res["author"], "sinrai74")
        self.assertEqual(res["committer"], "web-flow")
        self.assertIs(res["verified"], True)

    def test_identical_compare_is_ok(self):
        data = raw(doc())
        res = check(data, evidence(data, compare={"base": PSHA, "head": RUN_HEAD, "status": "identical"}))
        self.assertEqual(res["status"], "ok")

    # 1・2
    def test_1_file_missing(self):
        with _Repo(None) as root:
            res = agg.permission_check(root, vr(evidence(raw(doc()))))
        self.assertEqual(res["status"], "not_permitted")
        self.assertIs(res["conditions"]["1_file_exists"], False)

    def test_1_symlink_rejected(self):
        data = raw(doc())
        with _Repo(None) as root:
            os.makedirs(os.path.join(root, "g2_permissions"))
            real = os.path.join(root, "real.json")
            with open(real, "wb") as f:
                f.write(data)
            os.symlink(real, os.path.join(root, "g2_permissions", f"{PID}.json"))
            res = agg.permission_check(root, vr(evidence(data)))
        self.assertIs(res["conditions"]["1_file_exists"], False)
        self.assertEqual(res["status"], "not_permitted")

    def test_1_directory_symlink_rejected(self):
        data = raw(doc())
        with _Repo(None) as root:
            other = os.path.join(root, "elsewhere")
            os.makedirs(other)
            with open(os.path.join(other, f"{PID}.json"), "wb") as f:
                f.write(data)
            os.symlink(other, os.path.join(root, "g2_permissions"))
            res = agg.permission_check(root, vr(evidence(data)))
        self.assertIs(res["conditions"]["1_file_exists"], False)

    def test_2_id_differs_from_file_name(self):
        data = raw(doc(permission_id="OTHER"))
        res = check(data)
        self.assertIs(res["conditions"]["2_name_matches_id"], False)
        self.assertEqual(res["status"], "not_permitted")

    def test_invalid_permission_id_in_run(self):
        for bad in (None, "", "../x", "a/b", ".", "..", "x" * 101, "日本"):
            with self.subTest(bad=bad):
                res = check(v=vr(evidence(raw(doc())), permission_id=bad))
                self.assertEqual(res["status"], "not_permitted")
                self.assertIsNone(res["permission"])

    # 3
    def test_3_format_problems(self):
        cases = {
            "schema": doc(schema_version=2), "schema_bool": doc(schema_version=True),
            "race_date": doc(race_date="2099-01-01"), "race_date_invalid": doc(race_date="20990231"),
            "window_no_z": doc(window_start="2099-01-01T00:00:00+00:00"),
            "window_jst": doc(window_start="2099-01-01T09:00:00+09:00"),
            "window_order": doc(window_start="2099-01-03T00:00:00Z", window_end="2099-01-01T00:00:00Z"),
            "workflow": doc(workflow=".github/workflows/other.yml"), "ref": doc(ref="refs/heads/dev"),
            "mode": doc(mode="investigation"), "approved_on": doc(approved_on="2098/12/31"),
            "approved_by": doc(approved_by="someone"), "method": doc(approval_method="api"),
            "extra": dict(doc(), note="x"),
        }
        for name, d in cases.items():
            with self.subTest(name=name):
                data = raw(d)
                res = check(data, evidence(data))
                self.assertIs(res["conditions"]["3_format_ok"], False, name)
                self.assertEqual(res["status"], "not_permitted")
        d = doc()
        d.pop("approved_by")
        data = raw(d)
        self.assertIs(check(data, evidence(data))["conditions"]["3_format_ok"], False)

    def test_3_duplicate_keys_and_broken_json(self):
        dup = b'{"schema_version": 1, "permission_id": "' + PID.encode() + b'", "mode": "formal", "mode": "formal"}'
        for data in (dup, b"{not json", b"[]", b"\xff\xfe"):
            with self.subTest(data=data[:20]):
                res = check(data, evidence(data))
                self.assertEqual(res["status"], "not_permitted")
                self.assertIs(res["conditions"]["3_format_ok"], False)

    # 4・5
    def test_4_5_history(self):
        data = raw(doc())
        base = evidence(data)
        modified = copy.deepcopy(base)
        modified["commits"][0]["files"][0]["status"] = "modified"
        cases = {
            "no_history": dict(history_run_head=[], history_aggregate_head=[], commits=[]),
            "two_commits_at_run": dict(history_run_head=[PSHA, "d" * 40]),
            "changed_after_at_agg": dict(history_aggregate_head=["d" * 40, PSHA]),
            "deleted_at_agg": dict(history_aggregate_head=[]),
            "not_added": dict(commits=modified["commits"]),
            "commit_details_missing": dict(commits=[]),
            "file_not_in_commit": dict(commits=[dict(base["commits"][0], files=[])]),
            "truncated": dict(history_run_head=[PSHA] * 100, history_aggregate_head=[PSHA] * 100),
            "not_lists": dict(history_run_head=None),
        }
        for name, over in cases.items():
            with self.subTest(name=name):
                res = check(data, evidence(data, **over))
                self.assertEqual(res["status"], "not_permitted", name)
                self.assertTrue({"4_added_once", "5_not_changed_after"} & set(failed(res)), name)

    def test_5_aggregate_ref_must_be_main(self):
        data = raw(doc())
        for ref in ("refs/heads/dev", "", None):
            with self.subTest(ref=ref):
                res = check(data, evidence(data, aggregate_ref=ref))
                self.assertIn("5_not_changed_after", failed(res))
        ev = evidence(data)
        ev.pop("aggregate_ref")
        self.assertIn("5_not_changed_after", failed(check(data, ev)))

    def test_truncation_guard(self):
        data = raw(doc())
        # hm だけが上限の件数（hr は正しい1件）：履歴が切れている可能性があるので不合格
        res = check(data, evidence(data, history_aggregate_head=[PSHA] + ["d" * 40] * 99))
        self.assertEqual(res["status"], "not_permitted")
        self.assertIn("permission history may be truncated", res["problems"])
        res = check(data, evidence(data, history_run_head=[PSHA] * 100))
        self.assertIn("permission history may be truncated", res["problems"])

    # 6〜8
    def test_6_7_8_identity_and_signature(self):
        data = raw(doc())
        for name, over, key in (
                ("author", {"author_login": "claude"}, "6_author"),
                ("author_none", {"author_login": None}, "6_author"),
                ("committer", {"committer_login": "claude"}, "7_committer"),
                ("committer_user", {"committer_login": "sinrai74"}, "7_committer"),
                ("unverified", {"verified": False, "verification_reason": "unsigned"}, "8_signature_verified"),
                ("verified_str", {"verified": "true"}, "8_signature_verified"),
                ("verified_none", {"verified": None}, "8_signature_verified")):
            with self.subTest(name=name):
                ev = evidence(data)
                ev["commits"][0].update(over)
                res = check(data, ev)
                self.assertEqual(res["status"], "not_permitted")
                self.assertIn(key, failed(res))

    def test_claude_push_shape_is_rejected(self):
        """FACT（2026-11-09、API で確認）：Claude の push の commit は author・committer とも claude、署名は valid。
        署名が有効でも、author・committer で許可にならない。"""
        data = raw(doc())
        ev = evidence(data)
        ev["commits"][0].update(author_login="claude", committer_login="claude", verified=True,
                                verification_reason="valid")
        res = check(data, ev)
        self.assertEqual(res["status"], "not_permitted")
        self.assertEqual(failed(res), ["6_author", "7_committer"])

    # 9
    def test_9_commit_time(self):
        data = raw(doc())
        for name, t in (("after", "2099-01-02T00:00:01Z"), ("equal", "2099-01-02T00:00:00Z"),
                        ("after_by_tz", "2099-01-02T08:59:59+09:00x"), ("no_tz", "2099-01-01T12:00:00"),
                        ("none", None), ("garbage", "yesterday")):
            with self.subTest(name=name):
                ev = evidence(data)
                ev["commits"][0]["committer_date"] = t
                self.assertIn("9_committed_before_run", failed(check(data, ev)))

    def test_9_timezone_is_interpreted(self):
        data = raw(doc())
        ev = evidence(data)
        # 2099-01-02T08:59:59+09:00 は 2099-01-01T23:59:59Z（実行の開始より前）。文字列で比べると後になる。
        ev["commits"][0]["committer_date"] = "2099-01-02T08:59:59+09:00"
        self.assertEqual(check(data, ev)["status"], "ok")
        ev["commits"][0]["committer_date"] = "2099-01-02T09:00:01+09:00"  # 開始の1秒後
        self.assertIn("9_committed_before_run", failed(check(data, ev)))

    # 10
    def test_10_in_run_head(self):
        data = raw(doc())
        for name, cmp in (("behind", {"base": PSHA, "head": RUN_HEAD, "status": "behind"}),
                          ("diverged", {"base": PSHA, "head": RUN_HEAD, "status": "diverged"}),
                          ("other_head", {"base": PSHA, "head": "e" * 40, "status": "ahead"}),
                          ("other_base", {"base": "e" * 40, "head": RUN_HEAD, "status": "ahead"}),
                          ("missing", {}), ("none", None)):
            with self.subTest(name=name):
                res = check(data, evidence(data, compare=cmp))
                self.assertIn("10_in_run_head", failed(res))

    # 11
    def test_11_blob_mismatch(self):
        data = raw(doc())
        changed = raw(doc(window_end="2099-01-04T00:00:00Z"))  # 今のファイルが、許可の commit のときと違う
        res = check(changed, evidence(data))
        self.assertIn("11_blob_matches", failed(res))
        self.assertEqual(res["status"], "not_permitted")

    # 12
    def test_12_matches_run(self):
        data = raw(doc())
        for name, kw in (("race_date", {"race_date": "20990102"}), ("ref", {"github_ref": "refs/heads/dev"})):
            with self.subTest(name=name):
                self.assertIn("12_matches_run", failed(check(data, v=vr(evidence(data), **kw))))
        v = vr(evidence(data))
        v["meta"]["api_workflow_path"] = ".github/workflows/other.yml"
        self.assertIn("12_matches_run", failed(check(data, v=v)))
        for started in ("2098-12-31T23:59:59Z", "2099-01-03T00:00:01Z", None, "2099-01-02T00:00:00"):
            with self.subTest(started=started):
                v = vr(evidence(data))
                v["meta"]["run_started_at"] = started
                self.assertIn("12_matches_run", failed(check(data, v=v)))

    # 取得の失敗
    def test_fetch_failures_are_not_permitted(self):
        data = raw(doc())
        for name, ev in (("missing", None), ("not_dict", []),
                         ("fetch_failed", {"fetch_ok": False, "permission_id": PID}),
                         ("fetch_ok_str", evidence(data, fetch_ok="true")),
                         ("other_id", evidence(data, permission_id="OTHER")),
                         ("other_path", evidence(data, path="g2_permissions/OTHER.json"))):
            with self.subTest(name=name):
                v = vr(ev)
                if ev is None:
                    v.pop("permission_evidence")
                res = check(data, v=v)
                self.assertEqual(res["status"], "not_permitted")
                self.assertEqual(len(res["conditions"]), 12)
                self.assertIsNone(res["permission"])

    def test_git_blob_sha(self):
        self.assertEqual(agg._git_blob_sha(b""), "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391")
        self.assertEqual(agg._git_blob_sha(b"hello\n"), "ce013625030ba8dba906f756967f9e9ca394464a")

    def test_parse_api_time(self):
        self.assertIsNone(agg._parse_api_time("2099-01-01T00:00:00"))
        self.assertIsNone(agg._parse_api_time(""))
        self.assertIsNone(agg._parse_api_time(None))
        self.assertEqual(agg._parse_api_time("2099-01-01T09:00:00+09:00"), agg._parse_api_time("2099-01-01T00:00:00Z"))


class IsFormalTimeTests(unittest.TestCase):
    def test_window_compared_as_time_not_string(self):
        v = {"manifest": {"run": {"mode": "formal", "permission_id": "P1", "race_date": "20990101"}},
             "meta": {"run_started_at": "2099-01-01T08:00:00+09:00"}}  # = 2098-12-31T23:00:00Z（窓の前）
        perms = [{"permission_id": "P1", "race_date": "20990101",
                  "window_start": "2099-01-01T00:00:00Z", "window_end": "2099-01-03T00:00:00Z"}]
        self.assertFalse(agg.is_formal(v, perms))
        v["meta"]["run_started_at"] = "2099-01-01T10:00:00+09:00"  # = 01:00Z（窓の中）
        self.assertTrue(agg.is_formal(v, perms))
        v["meta"]["run_started_at"] = "2099-01-01T10:00:00"  # タイムゾーンなし
        self.assertFalse(agg.is_formal(v, perms))


class CommitFilesLimitAndRenameTests(unittest.TestCase):
    """U4（GPT 決定 2026-11-09）：変更ファイルが 300 を超えうる commit、改名・削除・再追加の疑いは不合格。"""

    def test_files_count(self):
        data = raw(doc())
        for fc, ok in ((1, True), (299, True), (300, False), (301, False), (0, False), (None, False),
                       ("1", False), (True, False)):
            with self.subTest(fc=fc):
                ev = evidence(data)
                ev["commits"][0]["files_count"] = fc
                if fc is None:
                    ev["commits"][0].pop("files_count")
                res = check(data, ev)
                self.assertEqual(res["status"] == "ok", ok, res["problems"])
                if not ok:
                    self.assertIn("4_added_once", failed(res))

    def test_rename_and_previous_filename(self):
        data = raw(doc())
        for name, f in (("renamed_in", {"filename": REL, "status": "renamed", "previous_filename": "x.json"}),
                        ("added_with_prev", {"filename": REL, "status": "added", "previous_filename": "x.json"}),
                        ("renamed_out", {"filename": "g2_permissions/other.json", "status": "renamed",
                                         "previous_filename": REL}),
                        ("removed", {"filename": REL, "status": "removed"}),
                        ("modified", {"filename": REL, "status": "modified"})):
            with self.subTest(name=name):
                ev = evidence(data)
                ev["commits"][0]["files"] = [dict(f, sha=agg._git_blob_sha(data))]
                res = check(data, ev)
                self.assertEqual(res["status"], "not_permitted")
                self.assertIn("4_added_once", failed(res))

    def test_history_limits_recorded(self):
        data = raw(doc())
        lim = {"follows_renames": False, "per_page": 100, "commit_files_max": 300}
        self.assertEqual(check(data, evidence(data, history_limits=lim))["history_limits"], lim)


def _main_runs(tmp, runs, first="20990101", last="20990101", check_fn=None):
    """runs: (rid, att, date, races, targets, started, overrides) の並び。permission_check は check_fn に置き換える。"""
    base = os.path.join(tmp, "a")
    os.makedirs(base)
    for rid, att, date, races, targets, started, over in runs:
        _write_run(base, rid, att, date, races, targets, permission=PID, started=started, run_overrides=over)
    bpath = os.path.join(tmp, "b.json")
    with open(bpath, "wb") as f:
        f.write(BASELINE_RAW)
    out = os.path.join(tmp, "out")
    real = agg.aggregate

    def formal_ok_aggregate(v, perms, f, l, b, bs, ctx, **kw):
        # 登録・E3・集約側の基準は別のテストで確かめる。ここでは許可の扱いだけを見るため、それらは合格とする
        return real(v, perms, f, l, b, bs, FORMAL_OK, **kw)
    with mock.patch.object(agg, "permission_check", side_effect=check_fn), ta_genesis_ok(), ta_git_ok(), \
            mock.patch.object(agg, "aggregate", side_effect=formal_ok_aggregate), redirect_stdout(StringIO()):
        rc = agg.main(["--artifacts-dir", base, "--first-date", first, "--last-date", last,
                       "--out", out, "--baseline", bpath])
    with open(os.path.join(out, "aggregate_result.json"), encoding="utf-8") as f:
        return rc, json.load(f)


def _ok_for(pid=PID, race_date="20990101", only=None):
    def fn(root, v):
        if only is not None and v["dir"] not in only:
            return {"status": "not_permitted", "conditions": {"9_committed_before_run": False},
                    "problems": ["x"], "permission": None}
        info = v["manifest"]["run"]
        return {"status": "ok", "conditions": {}, "problems": [],
                "permission": {"permission_id": info["permission_id"], "race_date": info["race_date"],
                               "window_start": "2099-01-01T00:00:00Z", "window_end": "2099-01-05T00:00:00Z"}}
    return fn


class SingleUseTests(unittest.TestCase):
    """U1（GPT 決定 2026-11-09）：許可1件で formal の実行は1回だけ。最初に条件を満たした実行だけを許可する。"""

    def test_second_run_with_same_id_is_violation_and_does_not_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res = _main_runs(tmp, [
                (1, 1, "20990101", [_race("a")], ["a"], "2099-01-02T00:00:00Z", None),
                (2, 1, "20990101", [_race("a", "mismatch")], ["a"], "2099-01-02T01:00:00Z", None)],
                check_fn=_ok_for())
        self.assertEqual(rc, 0)  # Y1：手続き違反は記録するが、照合の失敗（3）にはしない
        self.assertEqual(res["exit_status"]["verification_causes"]["procedure_violations_recorded"], 1)
        self.assertEqual([(v["run"], v["status"]) for v in res["permission_violations"]],
                         [("2-1", "permission_already_used")])
        self.assertEqual(res["permission_violations"][0]["event_type"], "g2_permission_violation")
        self.assertEqual([x.get("state") for x in res["sequence"]], ["added"])  # 1回目の結果のまま

    def test_order_is_by_start_time_not_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            # run 2 の方が早く始まった：run 2 が許可され、run 1 が違反になる
            rc, res = _main_runs(tmp, [
                (1, 1, "20990101", [_race("a")], ["a"], "2099-01-02T05:00:00Z", None),
                (2, 1, "20990101", [_race("a")], ["a"], "2099-01-02T01:00:00Z", None)],
                check_fn=_ok_for())
        self.assertEqual([v["run"] for v in res["permission_violations"]], ["1-1"])

    def test_rerun_attempt_is_second_use(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res = _main_runs(tmp, [
                (1, 1, "20990101", [_race("a")], ["a"], "2099-01-02T00:00:00Z", None),
                (1, 2, "20990101", [_race("a")], ["a"], "2099-01-02T03:00:00Z", None)],
                check_fn=_ok_for())
        self.assertEqual([(v["run"], v["status"]) for v in res["permission_violations"]],
                         [("1-2", "permission_already_used")])

    def test_same_start_and_run_id_ordered_by_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res = _main_runs(tmp, [
                (1, 2, "20990101", [_race("a")], ["a"], "2099-01-02T00:00:00Z", None),
                (1, 1, "20990101", [_race("a")], ["a"], "2099-01-02T00:00:00Z", None)],
                check_fn=_ok_for())
        self.assertEqual([v["run"] for v in res["permission_violations"]], ["1-2"])

    def test_first_that_satisfies_conditions_is_used(self):
        """1回目が許可の条件を満たさず、2回目が満たした：2回目を許可し、1回目は違反（条件の不成立）。"""
        with tempfile.TemporaryDirectory() as tmp:
            rc, res = _main_runs(tmp, [
                (1, 1, "20990101", [_race("a")], ["a"], "2099-01-02T00:00:00Z", None),
                (2, 1, "20990101", [_race("a")], ["a"], "2099-01-02T01:00:00Z", None)],
                check_fn=_ok_for(only={"2-1"}))
        self.assertEqual([(v["run"], v["status"]) for v in res["permission_violations"]],
                         [("1-1", "not_permitted")])
        # Y4（2026-11-09）：① より前の正式の表示の実行（未判定）がある日は、① を数えず保留する
        self.assertEqual(res["days"][0]["pending_reason"], "undetermined_run_on_day")


class ViolationAfterReachTests(unittest.TestCase):
    """U2・U3（GPT 決定 2026-11-09）：100件目より後の違反だけでは、到達・連続件数を止めない・取り消さない。"""

    def _runs(self, second_date, second_started):
        labels = [f"r{i:03d}" for i in range(1, 101)]
        inst = __import__("test_streak_reach")._inst(100)
        return [(1, 1, "20990101", [_race(x) for x in labels], labels, "2099-01-02T00:00:00Z",
                 {"instrument": inst}),
                (2, 1, second_date, [_race("z")], ["z"], second_started, None)]

    def test_violation_after_reach_day_does_not_halt(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res = _main_runs(tmp, self._runs("20990102", "2099-01-03T00:00:00Z"), last="20990102",
                                 check_fn=_ok_for(only={"1-1"}))
        self.assertEqual(res["reach"]["status"], "reached_not_yet_formally_achieved")  # U3：reach は変えない
        self.assertEqual(res["best_segment"], 100)
        self.assertEqual(res["g2_status"], "reached_100_unconfirmed")
        self.assertNotIn("achievement_halted", res)
        self.assertEqual([(v["run"], v["violation_order_vs_reach"]) for v in res["permission_violations"]],
                         [("2-1", "after")])
        self.assertEqual(rc, 3)  # 違反のイベントがあるので、集約のステップは失敗として知らせる

    def _one(self, date, started, last="20990101", over=None):
        runs = self._runs(date, started)
        if over:
            runs[1] = over
        with tempfile.TemporaryDirectory() as tmp:
            return _main_runs(tmp, runs, last=last, check_fn=_ok_for(only={"1-1"}))

    def test_u6_same_day_violation_started_after_reach_run_does_not_halt(self):
        rc, res = self._one("20990101", "2099-01-02T03:00:00Z")
        self.assertEqual(res["permission_violations"][0]["violation_order_vs_reach"], "after")
        self.assertEqual(res["g2_status"], "reached_100_unconfirmed")
        self.assertNotIn("achievement_halted", res)
        self.assertEqual(res["reach"]["status"], "reached_not_yet_formally_achieved")

    def test_u6_started_before_reach_run_halts_but_keeps_reach(self):
        # 日付は後でも、開始が到達した実行より前なら「到達前」
        rc, res = self._one("20990102", "2099-01-01T23:00:00Z", last="20990102")
        self.assertEqual(res["permission_violations"][0]["violation_order_vs_reach"], "before")
        self.assertEqual(res["reach"]["status"], "reached_not_yet_formally_achieved")
        self.assertEqual(res["best_segment"], 100)
        self.assertEqual(res["g2_status"], "halted_permission_violation")
        self.assertEqual(res["achievement_halted"]["runs"], ["2-1"])

    def test_u6_same_time_is_unknown_and_halts(self):
        rc, res = self._one("20990102", "2099-01-02T00:00:00Z", last="20990102")
        self.assertEqual(res["permission_violations"][0]["violation_order_vs_reach"], "unknown")
        self.assertEqual(res["g2_status"], "halted_permission_violation")

    def test_u6_rerun_attempt_of_reach_run_uses_attempt(self):
        labels = [f"r{i:03d}" for i in range(1, 101)]
        inst = __import__("test_streak_reach")._inst(100)
        over = (1, 2, "20990101", [_race(x) for x in labels], labels, "2099-01-02T00:00:00Z", {"instrument": inst})
        rc, res = self._one("20990101", None, over=over)
        self.assertEqual([(v["run"], v["status"], v["violation_order_vs_reach"])
                          for v in res["permission_violations"]], [("1-2", "not_permitted", "after")])
        self.assertEqual(res["g2_status"], "reached_100_unconfirmed")

    def test_u6_ids_alone_do_not_order_different_runs(self):
        self.assertEqual(agg._order_vs_reach({"run_id": "9", "run_attempt": "1", "run_started_at": None},
                                             {"run_id": "1", "run_attempt": "1",
                                              "run_started_at": "2099-01-02T00:00:00Z"}), "unknown")
        self.assertEqual(agg._order_vs_reach({"run_id": "1", "run_attempt": "2",
                                              "run_started_at": "2099-01-02T00:00:00Z"},
                                             {"run_id": "1", "run_attempt": "1",
                                              "run_started_at": "2099-01-02T00:00:00Z"}), "after")
        self.assertEqual(agg._order_vs_reach({"run_started_at": "x"}, None), "unknown")
        # 別の run は attempt の大小でも順序を決めない
        self.assertEqual(agg._order_vs_reach({"run_id": "2", "run_attempt": "2",
                                              "run_started_at": "2099-01-02T00:00:00Z"},
                                             {"run_id": "1", "run_attempt": "1",
                                              "run_started_at": "2099-01-02T00:00:00Z"}), "unknown")

    def test_u7_scope_is_recorded(self):
        rc, res = self._one("20990102", "2099-01-03T00:00:00Z", last="20990102")
        self.assertEqual(res["permission_single_use_scope"]["checked_within"], "input run_ids only")
        self.assertEqual(res["permission_single_use_scope"]["run_ids_checked"], ["1", "2"])

    def test_no_reach_violation_halts(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res = _main_runs(tmp, [(1, 1, "20990101", [_race("a")], ["a"], "2099-01-02T00:00:00Z", None)],
                                 check_fn=_ok_for(only=set()))
        self.assertIsNone(res["reach"])
        self.assertEqual(res["g2_status"], "halted_permission_violation")
        self.assertEqual(res["permission_violations"][0]["violation_order_vs_reach"], "no_reach")


class PermissionBoundToRunTests(unittest.TestCase):
    """監査（2026-11-09）の指摘1：合格した許可は、その実行にだけ使う。同じ permission_id の別の実行は加算しない。"""

    def test_is_formal_requires_same_run(self):
        v = {"dir": "2-1", "manifest": {"run": {"mode": "formal", "permission_id": "P1", "race_date": "20990101"}},
             "meta": {"run_started_at": "2099-01-02T00:00:00Z"}}
        p = {"permission_id": "P1", "race_date": "20990101", "window_start": "2099-01-01T00:00:00Z",
             "window_end": "2099-01-03T00:00:00Z", "run": "1-1"}
        self.assertFalse(agg.is_formal(v, [p]))
        self.assertTrue(agg.is_formal(v, [dict(p, run="2-1")]))

    def test_unpermitted_twin_run_is_not_selected(self):
        """1-1 は許可に合格、2-1（同じ permission_id、より早い開始）は不合格：2-1 は加算されず、違反になる。"""
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], permission=PID,
                       started="2099-01-02T00:00:00Z")
            _write_run(base, 2, 1, "20990101", [_race("a", "mismatch")], ["a"], permission=PID,
                       started="2099-01-01T12:00:00Z")
            bpath = os.path.join(tmp, "b.json")
            with open(bpath, "wb") as f:
                f.write(BASELINE_RAW)

            def fake(root, v):
                if v["dir"] == "1-1":
                    return {"status": "ok", "conditions": {}, "problems": [],
                            "permission": {"permission_id": PID, "race_date": "20990101",
                                           "window_start": "2099-01-01T00:00:00Z",
                                           "window_end": "2099-01-03T00:00:00Z"}}
                return {"status": "not_permitted", "conditions": {"9_committed_before_run": False},
                        "problems": ["x"], "permission": None}
            out = os.path.join(tmp, "out")
            with mock.patch.object(agg, "permission_check", side_effect=fake), redirect_stdout(StringIO()):
                rc = agg.main(["--artifacts-dir", base, "--first-date", "20990101", "--last-date", "20990101",
                               "--out", out, "--baseline", bpath])
            with open(os.path.join(out, "aggregate_result.json"), encoding="utf-8") as f:
                res = json.load(f)
        self.assertEqual(rc, 3)
        self.assertEqual([v["run"] for v in res["permission_violations"]], ["2-1"])
        self.assertNotIn("2-1", json.dumps(res.get("sequence")))  # 2-1 の mismatch は列に入らない
        self.assertNotIn("mismatch", json.dumps([x.get("state") for x in res.get("sequence") or []]))


class MainPermissionTests(unittest.TestCase):
    def _main(self, tmp, mode="formal", evidence_obj=None, check_result=None, extra_args=()):
        base = os.path.join(tmp, "a")
        os.makedirs(base)
        d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], mode=mode, permission=PID)
        if evidence_obj is not None:
            with open(os.path.join(d, "permission_evidence.json"), "w", encoding="utf-8") as f:
                json.dump(evidence_obj, f)
        bpath = os.path.join(tmp, "b.json")
        with open(bpath, "wb") as f:
            f.write(BASELINE_RAW)
        out = os.path.join(tmp, "out")
        buf = StringIO()
        patches = [mock.patch.object(agg, "_repo_root", return_value=tmp)]
        if check_result is not None:
            patches.append(mock.patch.object(agg, "permission_check", side_effect=check_result))
        with redirect_stdout(buf):
            for p in patches:
                p.start()
            try:
                rc = agg.main(["--artifacts-dir", base, "--first-date", "20990101", "--last-date", "20990101",
                               "--out", out, "--baseline", bpath, *extra_args])
            finally:
                for p in patches:
                    p.stop()
        with open(os.path.join(out, "aggregate_result.json"), encoding="utf-8") as f:
            return rc, json.load(f), buf.getvalue()

    def test_formal_run_without_permission_halts(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, out = self._main(tmp)  # 許可ファイルも履歴も無い
        self.assertEqual(rc, 3)
        self.assertEqual(res["g2_status"], "halted_permission_violation")
        self.assertEqual(len(res["permission_violations"]), 1)
        self.assertEqual(res["permission_violations"][0]["permission_id"], PID)
        self.assertIn("1_file_exists", res["permission_violations"][0]["failed_conditions"])
        self.assertIn("achievement_halted", res)
        self.assertEqual(res["best_segment"], 0)
        self.assertIn("permission_violations=1", out)

    def test_fetch_failed_evidence_halts(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = self._main(tmp, evidence_obj={"fetch_ok": False, "permission_id": PID,
                                                       "fetch_errors": ["permission history fetch failed"]})
        self.assertEqual(rc, 3)
        self.assertEqual(res["g2_status"], "halted_permission_violation")

    def test_duplicate_key_evidence_is_fetch_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], permission=PID)
            with open(os.path.join(d, "permission_evidence.json"), "w") as f:
                f.write('{"fetch_ok": true, "fetch_ok": true}')
            v = agg.verify_run(d)
        self.assertIs(v["permission_evidence"]["fetch_ok"], False)

    def test_check_exception_is_violation(self):
        def boom(root, v):
            raise RuntimeError("boom")
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = self._main(tmp, check_result=boom)
        self.assertEqual(rc, 3)
        self.assertEqual(res["permission_violations"][0]["status"], "check_error")

    def test_investigation_run_is_not_checked(self):
        calls = []

        def spy(root, v):
            calls.append(v)
            return {"status": "ok", "permission": None, "conditions": {}, "problems": []}
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = self._main(tmp, mode="investigation", check_result=spy)
        self.assertEqual(calls, [])
        self.assertEqual(res["permission_violations"], [])
        self.assertNotEqual(res["g2_status"], "halted_permission_violation")

    def test_ok_without_permission_record_is_violation(self):
        """status ok でも permission の記録が無ければ、許可として使わない（fail-closed）。"""
        with tempfile.TemporaryDirectory() as tmp:
            rc, res, _ = self._main(tmp, check_result=lambda r, v: {"status": "ok", "conditions": {},
                                                                    "problems": []})
        self.assertEqual(rc, 3)
        self.assertEqual(len(res["permission_violations"]), 1)

    def test_old_permissions_option_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(StringIO()), \
                mock.patch("sys.stderr", new=StringIO()):
            with self.assertRaises(SystemExit) as cm:
                agg.main(["--artifacts-dir", tmp, "--first-date", "20990101", "--last-date", "20990101",
                          "--out", os.path.join(tmp, "o"), "--permissions", os.path.join(tmp, "p.json")])
        self.assertEqual(cm.exception.code, 2)

    def test_real_permission_file_in_main(self):
        """許可ファイルと履歴がそろえば、main() で違反にならない（登録が無いので正式な合格は出ない）。"""
        data = raw(doc())
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "g2_permissions"))
            with open(os.path.join(tmp, "g2_permissions", f"{PID}.json"), "wb") as f:
                f.write(data)
            ev = evidence(data, run_head="0" * 40, compare={"base": PSHA, "head": "0" * 40, "status": "ahead"})
            rc, res, _ = self._main(os.path.realpath(tmp), evidence_obj=ev)
        self.assertEqual(res["permission_violations"], [])
        self.assertNotEqual(res["g2_status"], "halted_permission_violation")
        self.assertEqual(res["runs"][0]["dir"], "1-1")


class WorkflowPermissionFetchTests(unittest.TestCase):
    WF = ".github/workflows/formal_g2_aggregate.yml"

    def setUp(self):
        with open(self.WF, encoding="utf-8") as f:
            self.text = f.read()

    def test_permissions_path_input_removed(self):
        self.assertNotIn("permissions_path", self.text)
        self.assertNotIn("--permissions", self.text)
        self.assertNotIn("PERMISSIONS_PATH", self.text)

    def test_fetch_is_in_download_step_and_read_only(self):
        i = self.text.index("- name: Download artifacts (zip) and digests")
        j = self.text.index("- name: Aggregate")
        step = self.text[i:j]
        self.assertIn("fetch_permission_evidence()", step)
        self.assertIn("commits?path=${path}&sha=${run_head}&per_page=100", step)
        self.assertIn("commits?path=${path}&sha=${GITHUB_SHA}&per_page=100", step)
        self.assertIn("compare/${sha}...${run_head}", step)
        self.assertIn("permission_evidence.json", step)
        self.assertIn("fetch_ok:false", step)
        self.assertNotIn("fetch_permission_evidence", self.text[j:])
        # 書き込みの API を使わない（gh api の既定は GET。-X・--method・-f・-F・--input を使わない）
        for line in step.splitlines():
            if "gh api" in line:
                self.assertIsNone(re.search(r"\s(-X|--method|-f|-F|--field|--raw-field|--input)\s", line), line)
        self.assertNotIn("contents/g2_permissions", self.text)

    def test_per_page_matches_truncation_limit(self):
        self.assertEqual(agg.PERMISSION_HISTORY_LIMIT, 100)
        self.assertEqual(self.text.count("&per_page=100\" --jq"), 2)


if __name__ == "__main__":
    unittest.main()
