"""正式G3（B案）段階3：診断の入口・取得・strace の検査・workflow の試験。

- 実際の出走表1レース（2026-11-07 桐生 1R、公開データ）を fixtures に置き、子のプロセスの手順を実際の
  Legacy・ML で通す（scikit-learn 1.5.2 が無ければ skip。理由を表示）。オッズは合成（実データが無い）。
- 親のプロセスの扱い（成功・SystemExit・時間切れ・出力なし・取り違え）は、小さな偽の子のプロセスで確かめる。
- C-04：この試験は外へ通信しない。取得の関数・requests・G2 の取得は、すべて差し替えるか、呼ばれたら失敗にする。
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import random
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from actions import g3_diagnostic_collect as C
from actions import g3_diagnostic_entry as D
from actions import g3_diagnostic_fetch as FT
from actions import g3_diagnostic_verify as V
from tests.unit import g3_support as S

FIXTURE = S.REBUILD_DIR / "tests/unit/fixtures/g3_diag_programs_20261107_01_01.json"
WORKFLOW = S.REPO_DIR / ".github/workflows/g3_diagnostic.yml"
LOCK = S.REPO_DIR / "requirements-g3-diagnostic.lock"
EID = "20261107_01_01"
PY = "/opt/py/bin/python"
LSCPU = "/usr/bin/lscpu"


def _tmp(test):
    d = Path(tempfile.mkdtemp(prefix="g3_diag_test_"))
    test.addCleanup(shutil.rmtree, d, True)
    return d


def _synth_odds(eids, seed="x"):
    races = {}
    for e in eids:
        rng = random.Random(f"{seed}:{e}")
        races[e] = {"fetched_at": "synthetic", "kind": "synthetic_for_test", "count": 120, "error": None,
                    "odds": {f"{a}-{b}-{c}": round(rng.uniform(3, 300), 1)
                             for a, b, c in itertools.permutations(range(1, 7), 3)}}
    return {"race_date": "20261107", "races": races}


def _no_network():
    """C-04：外への取得が呼ばれたら失敗にする（試験が外へ出ないための防御）。"""
    import requests

    def boom(*a, **k):
        raise AssertionError("network access attempted in test")
    return mock.patch.multiple(requests, get=boom, post=boom, request=boom)


CHECK_KEYS = {"outcome_compared", "comparison_4_items_equal", "config_source_is_workdir_file",
              "matches_freeze_is_true", "seeded_config_sha256_is_freeze", "eval_isolation_verified",
              "prep_isolation_verified", "eval_id_consistent", "evaluate_bets_call_count_is_1",
              "odds_error", "odds_kind", "target_status", "child_returncode_zero", "strace_evidence", "note"}


class ListAndChecksTests(unittest.TestCase):
    def test_list_races(self):
        progs = [{"race_stadium_number": 2, "race_number": 1, "race_closed_at": "2026-11-07 11:00:00"},
                 {"race_stadium_number": 1, "race_number": 2, "race_closed_at": "2026-11-07 10:30:00"}]
        self.assertEqual([r["eval_id"] for r in D.list_races("20261107", progs)],
                         ["20261107_01_02", "20261107_02_01"])

    def _ok_rec(self):
        return {"outcome": "compared", "match": {"all": True}, "evaluate_bets_call_count": 1,
                "target": {"status": "legacy_target_not_provided"},
                "config": {"source": "workdir_file", "matches_freeze": True},
                "isolation": {"verified": True, "seeded": {"buyscore_config.json": D.FREEZE_CONFIG_SHA256}}}

    def _diag(self):
        return {"eval_id_consistency": {"evaluation_eval_id_matches": True},
                "odds_meta": {"error": None, "kind": "k"}}

    def test_keys_exact_no_aggregate(self):                         # E21・C-10
        c = D.count_candidate_checks(self._ok_rec(), {"verified": True}, self._diag())
        self.assertEqual(set(c), CHECK_KEYS)  # 合計の判定（all_ok・countable など）を持たない

    def test_each_condition(self):
        prep, diag = {"verified": True}, self._diag()
        ok = D.count_candidate_checks(self._ok_rec(), prep, diag)
        for k in ("outcome_compared", "comparison_4_items_equal", "config_source_is_workdir_file",
                  "matches_freeze_is_true", "seeded_config_sha256_is_freeze", "eval_isolation_verified",
                  "prep_isolation_verified", "eval_id_consistent", "evaluate_bets_call_count_is_1"):
            self.assertIs(ok[k], True, k)
        self.assertEqual((ok["strace_evidence"], ok["child_returncode_zero"]),
                         ("pending_external", "pending_parent"))
        self.assertEqual((ok["odds_kind"], ok["target_status"]), ("k", "legacy_target_not_provided"))
        bad = [
            ("outcome_compared", lambda r: r.update(outcome="no_candidate")),
            ("comparison_4_items_equal", lambda r: r["match"].update(all=False)),
            ("config_source_is_workdir_file", lambda r: r["config"].update(source="builtin_defaults")),
            ("matches_freeze_is_true", lambda r: r["config"].update(matches_freeze=False)),
            ("seeded_config_sha256_is_freeze", lambda r: r["isolation"]["seeded"].update(
                {"buyscore_config.json": "0" * 64})),
            ("eval_isolation_verified", lambda r: r["isolation"].update(verified=False)),
            ("evaluate_bets_call_count_is_1", lambda r: r.update(evaluate_bets_call_count=2)),
        ]
        for key, mutate in bad:
            r = self._ok_rec()
            mutate(r)
            self.assertIs(D.count_candidate_checks(r, prep, diag)[key], False, key)
        self.assertIs(D.count_candidate_checks(self._ok_rec(), {"verified": False}, diag)["prep_isolation_verified"],
                      False)
        d2 = self._diag()
        d2["eval_id_consistency"]["evaluation_eval_id_matches"] = False
        self.assertIs(D.count_candidate_checks(self._ok_rec(), prep, d2)["eval_id_consistent"], False)

    def test_not_evaluated_is_none(self):                           # C-10
        """no_candidate など、評価に至らない項目は False ではなく None。"""
        r = {"outcome": "no_candidate", "match": None, "config": None, "isolation": {"verified": True,
             "seeded": {"buyscore_config.json": D.FREEZE_CONFIG_SHA256}}, "evaluate_bets_call_count": 1}
        c = D.count_candidate_checks(r, {"verified": True}, self._diag())
        self.assertIs(c["outcome_compared"], False)
        for k in ("comparison_4_items_equal", "config_source_is_workdir_file", "matches_freeze_is_true"):
            self.assertIsNone(c[k], k)
        empty = D.count_candidate_checks({}, None)
        self.assertFalse(any(v is True for v in empty.values()))
        self.assertIsNone(empty["outcome_compared"])


class ParentProcessTests(unittest.TestCase):
    """B-06・C-13：子のプロセスの成功・SystemExit・時間切れ・出力なし・取り違えを、親が記録する。"""

    def _progs(self, d, n=6):
        p = d / "programs.json"
        p.write_text(json.dumps({"programs": [
            {"race_stadium_number": 1, "race_number": k, "race_closed_at": f"2026-11-07 1{k}:00:00"}
            for k in range(1, n + 1)]}), encoding="utf-8")
        o = d / "odds.json"
        o.write_text("{}", encoding="utf-8")
        return p, o

    def _cmd(self, behaviours):
        def child_cmd(race_date, eval_id, programs, odds_p, out, nonce=""):
            b = behaviours[eval_id]
            good = f"{{'eval_id':{eval_id!r},'nonce':{nonce!r},'stage':'done','error':None,'g3':{{'outcome':'compared'}}}}"
            code = {
                "ok": f"import json;open({str(out)!r},'w').write(json.dumps({good}))",
                "exit3_with_output": f"import json,sys;open({str(out)!r},'w').write(json.dumps({good}));sys.exit(3)",
                "exit3": "import sys;sys.stderr.write('Traceback\\n' + 'x'*3000 + "
                         "'\\nhttps://x:SECRETPW@h.invalid/p?token=SECRETQ\\nFINAL-LINE-MARKER');sys.exit(3)",
                "sleep": "import time;time.sleep(30)",
                "noout": "pass",
                "wrong_eid": f"import json;open({str(out)!r},'w').write(json.dumps({{'eval_id':'other','nonce':{nonce!r}}}))",
                "wrong_nonce": f"import json;open({str(out)!r},'w').write(json.dumps({{'eval_id':{eval_id!r},'nonce':'stale'}}))",
                "invalid": f"open({str(out)!r},'w').write('not json')",
                "pythonpath": "import os,sys;sys.exit(0 if 'rebuild' in os.environ.get('PYTHONPATH','') else 9)",
                "no_odds": f"import json;open({str(out)!r},'w').write(json.dumps({{'eval_id':{eval_id!r},"
                           f"'nonce':{nonce!r},'stage':'done','error':None,'g3':{{'outcome':'no_odds'}}}}))",
            }[b]
            return [sys.executable, "-c", code]
        return child_cmd

    def test_child_outcomes_recorded(self):
        d = _tmp(self)
        progs, odds = self._progs(d, 8)
        eids = [f"20261107_01_0{k}" for k in range(1, 9)]
        beh = dict(zip(eids, ["ok", "exit3_with_output", "exit3", "sleep", "noout", "wrong_eid",
                              "wrong_nonce", "invalid"]))
        s = D.run_parent("20261107", progs, odds, d / "out", child_cmd=self._cmd(beh), timeout=3)
        recs = {json.loads(l)["eval_id"]: json.loads(l)
                for l in (d / "out" / "records.jsonl").read_text(encoding="utf-8").splitlines()}
        self.assertEqual(recs[eids[0]]["g3"]["outcome"], "compared")
        self.assertEqual(recs[eids[0]]["child_returncode"], 0)        # E26
        self.assertIs(recs[eids[0]]["count_candidate_checks"]["child_returncode_zero"], True)
        self.assertEqual(recs[eids[1]]["error"], "child_failed")       # E08：rc≠0 なら出力があっても採らない
        self.assertEqual(recs[eids[1]]["child_returncode"], 3)
        tail = recs[eids[2]]["child_stderr_tail"]                     # G-17：本文でなく digest（C-06 廃止：末尾も残さない）
        self.assertEqual(tail["redacted"], True)
        self.assertIsInstance(tail["len"], int)
        self.assertGreater(tail["len"], 0)                            # 非空の stderr を digest 化したこと
        self.assertIsInstance(tail["lines"], int)
        self.assertNotIn("SECRET", json.dumps(tail, ensure_ascii=False))  # 本文・末尾を一切残さない
        self.assertNotIn("FINAL-LINE-MARKER", json.dumps(tail, ensure_ascii=False))
        self.assertEqual(recs[eids[3]]["child_returncode"], "timeout")
        self.assertEqual(recs[eids[4]]["error"], "child_output_missing")
        self.assertEqual(recs[eids[5]]["error"], "child_output_mismatch")
        self.assertEqual(recs[eids[6]]["error"], "child_output_mismatch")
        self.assertEqual(recs[eids[7]]["error"], "child_output_invalid")
        for e in eids[1:]:
            self.assertIs(recs[e]["count_candidate_checks"]["child_returncode_zero"], recs[e]["child_returncode"] == 0)
            self.assertIn("child_returncode", recs[e])                # E26
        self.assertEqual(s["child_failures"], 7)
        self.assertEqual(s["records_count"], 8)
        self.assertFalse(s["all_children_failed"])
        summ = json.loads((d / "out" / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summ["records_sha256"], D._sha256_file(d / "out" / "records.jsonl"))
        self.assertIn("code_sha256", summ["environment"])
        names = {p.name for p in (d / "out" / "races").iterdir()}
        self.assertEqual(len(names), 5)  # 出力を書いた5件。名前に nonce が入り、以前の実行と取り違えない
        self.assertTrue(all(re.match(r"^20261107_01_0\d\.[0-9a-f]{32}\.json$", n) for n in names), names)

    def test_only_filter_and_all_failed_and_no_targets(self):       # E18・C-05
        d = _tmp(self)
        progs, odds = self._progs(d, 3)
        beh = {f"20261107_01_0{k}": "noout" for k in range(1, 4)}
        s = D.run_parent("20261107", progs, odds, d / "o1", only=["20261107_01_02"], child_cmd=self._cmd(beh))
        self.assertEqual((s["races"], s["records_count"]), (1, 1))
        self.assertTrue(s["all_children_failed"])
        s = D.run_parent("20261107", progs, odds, d / "o2", only=["nothing"], child_cmd=self._cmd(beh))
        self.assertEqual((s["races"], s["records_count"], s["no_target_races"]), (0, 0, True))

    def test_no_odds_count(self):                                    # N19
        d = _tmp(self)
        progs, odds = self._progs(d, 3)
        beh = {"20261107_01_01": "no_odds", "20261107_01_02": "no_odds", "20261107_01_03": "ok"}
        s = D.run_parent("20261107", progs, odds, d / "o", child_cmd=self._cmd(beh))
        self.assertEqual(s["no_odds_count"], 2)
        self.assertEqual(s["eval_ids"], ["20261107_01_01", "20261107_01_02", "20261107_01_03"])

    def test_child_env(self):                                        # E25
        d = _tmp(self)
        progs, odds = self._progs(d, 1)
        s = D.run_parent("20261107", progs, odds, d / "o", child_cmd=self._cmd({EID: "pythonpath"}))
        rec = json.loads((d / "o" / "records.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(rec["child_returncode"], 0)
        with S.fake_token_env():
            env = D._child_env()
        for k in ("GITHUB_TOKEN", "GH_TOKEN", "GITHUB_REPOSITORY"):
            self.assertNotIn(k, env)

    def test_run_parent_refuses_nonempty_out_dir(self):              # 本人判断2
        d = _tmp(self)
        progs, odds = self._progs(d, 1)
        out = d / "o"
        out.mkdir()
        (out / "stale.txt").write_text("x", encoding="utf-8")             # 非空 out_dir
        with self.assertRaises(SystemExit):
            D.run_parent("20261107", progs, odds, out, child_cmd=self._cmd({EID: "ok"}))
        s = D.run_parent("20261107", progs, odds, d / "o2", child_cmd=self._cmd({EID: "ok"}))  # 空は許可
        self.assertEqual(s["races"], 1)


def _line(pid, text):
    return f"{pid} {text}"


class VerifyStraceTests(unittest.TestCase):
    BASE = [
        _line(100, f'execve("{PY}", ["python"], 0x0 /* 3 vars */) = 0'),
        _line(100, 'connect(3<UNIX:[111]>, {sa_family=AF_UNIX, sun_path="/var/run/nscd/socket"}, 110) = -1 ENOENT'),
        _line(101, 'execve("/usr/local/bin/lscpu", ["lscpu"], 0x0) = -1 ENOENT (No such file)'),
        _line(101, f'execve("{LSCPU}", ["lscpu", "--parse=core"], 0x0) = 0'),
        _line(102, 'sendmsg(5<NETLINK:[222]>, {msg_name={sa_family=AF_NETLINK, nl_pid=0}, msg_iov=[]}, 0) = 20'),
        _line(103, 'socket(AF_INET, SOCK_STREAM|SOCK_CLOEXEC, IPPROTO_TCP) = 3<TCP:[333]>'),
    ]

    def _a(self, extra=(), allow=(PY, LSCPU)):
        return V.analyze(list(self.BASE) + list(extra), list(allow))

    def test_base_ok(self):
        r = self._a()
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["exec_allowed"], sorted([LSCPU, PY]))
        self.assertEqual(r["unix_connect_failed"], ["/var/run/nscd/socket"])

    def test_failures(self):
        cases = {
            "connect_inet": 'connect(3<TCP:[1]>, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("1.1.1.1")}, 16) = -1 ENETUNREACH',
            "connect_inet6": 'connect(3<TCPv6:[1]>, {sa_family=AF_INET6, sin6_port=htons(443)}, 28) = 0',
            "bind": 'bind(3<UDP:[1]>, {sa_family=AF_INET, sin_port=htons(0), sin_addr=inet_addr("0.0.0.0")}, 16) = 0',  # V11
            "sendmsg": 'sendmsg(3<UDP:[1]>, {msg_name={sa_family=AF_INET, sin_port=htons(53)}}, 0) = 1',            # V12
            "sendmmsg": 'sendmmsg(3<UDP:[1]>, [{msg_hdr={msg_name={sa_family=AF_INET, sin_port=htons(53)}}}], 1, 0) = 1',
            "listen": 'listen(3<TCP:[1]>, 128) = 0',
            "sendto_noaddr_tcp": 'sendto(3<TCP:[1->2]>, "x", 1, 0, NULL, 0) = 1',
            "sendto_unknown_fd": 'sendto(3, "x", 1, 0, NULL, 0) = 1',
            "data_trick": 'sendto(3<UDP:[1]>, "sa_family=AF_UNIX", 17, 0, {sa_family=AF_INET, sin_port=htons(9)}, 16) = 17',
            "unix_success": 'connect(4<UNIX:[9]>, {sa_family=AF_UNIX, sun_path="/var/run/docker.sock"}, 110) = 0',
            "exec_other": 'execve("/usr/bin/curl", ["curl"], 0x0) = 0',
            "exec_basename_trick": 'execve("/tmp/x/python", ["python"], 0x0) = 0',               # V16
            "execveat": 'execveat(3, "/bin/true", ["true"], 0x0, 0) = 0',
            "exec_escaped_quote": r'execve("/tmp/a\"b", ["x"], 0x0) = 0',
        }
        for name, text in cases.items():
            r = self._a([_line(200, text)])
            self.assertFalse(r["ok"], name)

    def test_split_execve_joined(self):                             # C-01
        split = [_line(300, 'execve("/bin/true", ["true"], 0x0 /* 3 vars */ <unfinished ...>'),
                 _line(301, 'connect(3<UNIX:[1]>, {sa_family=AF_UNIX, sun_path="/x"}, 110) = -1 ENOENT'),
                 _line(300, '<... execve resumed>) = 0')]
        r = self._a(split)
        self.assertFalse(r["ok"])
        self.assertEqual(r["exec_not_allowed"], ["/bin/true"])
        r2 = self._a([split[0]])  # 結果が来ない unfinished は判定できないので不合格
        self.assertFalse(r2["ok"])
        self.assertEqual(len(r2["exec_unresolved"]), 1)
        ok_split = [_line(300, f'execve("{LSCPU}", ["lscpu"], 0x0 <unfinished ...>'),
                    _line(300, '<... execve resumed>) = 0')]
        self.assertTrue(self._a(ok_split)["ok"])
        pid_form = [f'[pid   400] execve("/bin/sh", ["sh"], 0x0) = 0']
        self.assertFalse(self._a(pid_form)["ok"])

    def test_d01_quoted_return_value_not_trusted(self):              # D-01
        tricks = [
            'execve("/bin/echo", ["echo", ") = -1 ENOENT (x)"], 0x0 /* 3 vars */) = 0',
            'connect(4<UNIX-STREAM:[9]>, {sa_family=AF_UNIX, sun_path="e) = -1 ENOENT.sock"}, 110) = 0',
        ]
        for t in tricks:
            self.assertFalse(self._a([_line(500, t)])["ok"], t)
        r = self._a([_line(501, f'execve("{PY}", ["python", ") = -1 ENOENT"], 0x0) = 0')])
        self.assertTrue(r["ok"] and r["python_seen"], r)            # 逆向きの誤りも無い

    def test_more_net_and_exec_forms(self):                         # N03・N08・N09・N15・N49
        cases = {
            "accept4_inet": 'accept4(3<TCP:[1]>, {sa_family=AF_INET, sin_port=htons(5)}, [16], 0) = 4',
            "data_trick_no_fd": 'sendto(3, "sa_family=AF_UNIX", 17, 0, {sa_family=AF_INET, sin_port=htons(9)}, 16) = 17',
            "tcp_fd_unix_family": 'connect(3<TCP:[1]>, {sa_family=AF_UNIX, sun_path="/x"}, 110) = -1 ENOENT',
            "exec_ret_question": f'execve("{LSCPU}", ["lscpu"], 0x0) = ?',
            "resumed_without_start": '<... execve resumed>) = 0',
        }
        for name, text in cases.items():
            self.assertFalse(self._a([_line(600, text)])["ok"], name)

    def test_real_strace68_output(self):                            # D-02・C-01（strace 6.8 の実際の出力）
        fx = S.REBUILD_DIR / "tests/unit/fixtures/g3_strace68"
        py = "/opt/g3py/bin/python"

        def run(name, allow):
            return V.analyze((fx / f"{name}.txt").read_text(encoding="utf-8").splitlines(), allow)
        local = run("local", [py])
        self.assertEqual(local["network_non_local"], [])
        self.assertEqual(local["network_unknown_family"], [])         # UNIX-STREAM は局所
        self.assertGreaterEqual(local["network_local_count"], 4)
        self.assertEqual(len(local["unix_connect_succeeded"]), 1)      # 成功した AF_UNIX の接続は不合格
        # missing.sock への接続は、パスが AF_UNIX の上限（108 バイト）を超え、Python がシステムコールの前に止めた
        self.assertEqual(local["unix_connect_failed"], [])
        self.assertFalse(local["ok"])
        inet = run("inet", [py])
        self.assertEqual(len(inet["network_non_local"]), 2)            # connect と listen
        self.assertFalse(inet["ok"])
        conc = run("conc", [py])
        self.assertEqual(len(conc["exec_not_allowed"]), 20)            # 分かれた行も含め20件すべて
        self.assertFalse(conc["ok"])
        self.assertTrue(run("conc", [py, "/bin/true"])["ok"])
        th = run("thread_exec", [py])                                  # "pid changed to" の形
        self.assertEqual(th["exec_not_allowed"], ["/bin/true"])
        self.assertTrue(run("thread_exec", [py, "/bin/true"])["ok"])
        self.assertFalse(run("execveat", [py])["ok"])

    def test_requires_python_and_nonempty(self):
        self.assertFalse(V.analyze([], [PY])["ok"])
        self.assertFalse(V.analyze([_line(1, f'execve("{LSCPU}", ["lscpu"], 0x0) = 0')], [PY, LSCPU])["ok"])

    def test_e07_parser_cases(self):                                # E-07（V-n01・V-n06・V-n07・V-n19）
        # V-n01：戻り値は最後の ") = N"。引用符の外の注記にある ") = -1" にだまされない
        r = self._a([_line(700, 'execve("/bin/sh", ["sh"], 0x0 /* x) = -1 ENOENT */) = 0')])
        self.assertFalse(r["ok"])
        self.assertEqual(r["exec_not_allowed"], ["/bin/sh"])
        # V-n06：別の pid の resumed とつなぐのは "pid changed to" があるときだけ
        cross = [_line(701, 'execve("/bin/true", ["true"], 0x0 <unfinished ...>'),
                 _line(702, '<... execve resumed>) = 0')]
        r = self._a(cross, allow=(PY, LSCPU, "/bin/true"))
        self.assertFalse(r["ok"])
        self.assertNotIn("/bin/true", r["exec_allowed"])
        self.assertEqual(len(r["exec_unresolved"]), 2)                # 開始の無い resumed と、結果の無い unfinished
        cross_bad = [_line(701, 'execve("/bin/true", ["true"], 0x0 <unfinished ...>'),
                     _line(702, '<... execve resumed>) = 0')]
        self.assertEqual(self._a(cross_bad)["exec_not_allowed"], [])
        # V-n07：結果の来ない通信の呼び出しは不明として不合格
        r = self._a([_line(703, 'connect(3<UNIX:[1]>, {sa_family=AF_UNIX, sun_path="/x"}, 110 <unfinished ...>')])
        self.assertFalse(r["ok"])
        self.assertEqual(len(r["network_unknown_family"]), 1)
        # V-n19：文字列の中の \" で引用符の範囲がずれない
        r = self._a([_line(704, r'sendto(3, "x\" sa_family=AF_UNIX y", 17, 0, {sa_family=AF_INET, sin_port=htons(9)}, 16) = 17')])
        self.assertFalse(r["ok"])
        self.assertEqual(len(r["network_non_local"]), 1)

    def test_e05_unix_send_to_path(self):                           # E-05
        for t in ('sendto(4<UNIX-DGRAM:[9]>, "x", 1, 0, {sa_family=AF_UNIX, sun_path="/dev/log"}, 110) = 1',
                  'sendmsg(4<UNIX-DGRAM:[9]>, {msg_name={sa_family=AF_UNIX, sun_path="/run/x.sock"}, msg_iov=[]}, 0) = 1',
                  'sendmmsg(4<UNIX-DGRAM:[9]>, [{msg_hdr={msg_name={sa_family=AF_UNIX, sun_path="/run/y"}}}], 1, 0) = 1'):
            r = self._a([_line(710, t)])
            self.assertFalse(r["ok"], t)
            self.assertEqual(len(r["unix_send_to_path"]), 1, t)
        self.assertEqual(self._a([_line(710, 'sendto(4<UNIX-DGRAM:[9]>, "x", 1, 0, {sa_family=AF_UNIX, '
                                             'sun_path="/dev/log"}, 110) = 1')])["unix_send_to_path"], ["/dev/log"])
        # 宛先の無い送信（つながったソケット・NETLINK）は対象にしない
        r = self._a([_line(711, 'sendto(4<UNIX-STREAM:[9->10]>, "x", 1, 0, NULL, 0) = 1')])
        self.assertTrue(r["ok"], r)
        self.assertEqual(r["unix_send_to_path"], [])
        self.assertIn("limits", r)                                        # E-03・F-09：限界を記録に書く
        self.assertTrue(any("unshare --net" in x for x in r["limits"]))
        self.assertTrue(any("sha256 shows only" in x for x in r["limits"]))

    def test_namespace_and_server_calls(self):                      # F-03・F-06（名前空間・AF_UNIX の server）
        import hashlib
        # F-03：unshare・setns・mount 系・ptrace・bpf の成功は不合格
        for t in ('unshare(CLONE_NEWUSER) = 0', 'setns(3, CLONE_NEWNET) = 0',
                  'mount("none", "/x", "tmpfs", 0, NULL) = 0', 'ptrace(PTRACE_TRACEME) = 0',
                  'bpf(BPF_PROG_LOAD, {...}, 120) = 4', 'pivot_root("/a", "/b") = 0'):
            r = self._a([_line(720, t)])
            self.assertFalse(r["ok"], t)
            self.assertEqual(len(r["namespace_calls"]), 1, t)
        # 失敗した unshare は合格の側（試みても作れなかった）
        self.assertTrue(self._a([_line(721, 'unshare(CLONE_NEWUSER) = -1 EPERM (Operation not permitted)')])["ok"])
        # F-03：CLONE_NEW* か CLONE_UNTRACED の旗の clone・clone3 は、試しただけで不合格
        for t in ('clone(child_stack=NULL, flags=CLONE_NEWUSER|SIGCHLD) = -1 EPERM',
                  'clone3({flags=CLONE_NEWNET, ...}, 88) = -1 EPERM',
                  'clone(child_stack=0x1, flags=CLONE_UNTRACED|SIGCHLD) = 1234'):
            r = self._a([_line(722, t)])
            self.assertFalse(r["ok"], t)
            self.assertEqual(len(r["clone_flags_rejected"]), 1, t)
        self.assertTrue(self._a([_line(723, 'clone(child_stack=NULL, flags=SIGCHLD) = 999')])["ok"])  # 普通の fork
        # F-03：io_uring_setup は、成否を問わず不合格
        self.assertFalse(self._a([_line(724, 'io_uring_setup(8, {flags=0}) = 5')])["ok"])
        self.assertFalse(self._a([_line(725, 'io_uring_setup(8, {flags=0}) = -1 ENOSYS')])["ok"])
        # F-06：AF_UNIX の bind・listen・accept の成功は不合格（つなぎ役になれる）。NETLINK の bind は合格
        for t in ('bind(3<UNIX:[9]>, {sa_family=AF_UNIX, sun_path="/tmp/x"}, 110) = 0',
                  'listen(3<UNIX:[9]>, 5) = 0', 'accept(3<UNIX:[9]>, NULL, NULL) = 4',
                  'bind(3<UNIX:[9]>, {sa_family=AF_UNIX}, 110) = 0'):  # 抽象のソケット（sun_path なし）
            r = self._a([_line(726, t)])
            self.assertFalse(r["ok"], t)
            self.assertEqual(len(r["unix_server_succeeded"]), 1, t)
        self.assertTrue(self._a([_line(727, 'bind(3<NETLINK:[9]>, {sa_family=AF_NETLINK}, 12) = 0')])["ok"])
        self.assertTrue(self._a([_line(728, 'listen(3<UNIX:[9]>, 5) = -1 EADDRINUSE')])["ok"])   # 失敗は合格の側

    def test_redaction_preserves_judgement(self):                   # F-10
        lines = list(self.BASE) + [
            _line(730, 'sendto(3<UDP:[1]>, "secret_token=abcd1234 body", 26, 0, '
                       '{sa_family=AF_INET, sin_port=htons(53)}, 16) = 26'),
            _line(731, f'execve("{PY}", ["python", "password=s3cr3t x"], 0x0) = 0')]
        raw = V.analyze(lines, [PY, LSCPU])
        red = V.analyze([V.redact_strace_line(l) for l in lines], [PY, LSCPU])
        self.assertEqual(V.judgement(raw), V.judgement(red))              # 判定は変わらない（核心）
        # 伏せ字は「認識できる秘密の型」だけを対象にする：正規の実行パスに「token」が含まれるだけでは壊さない
        # （実行判定を保つための望ましい挙動）。本当に秘密の型（argv の password=…）は伏せ字になる。
        keep_path = _line(732, f'execve("{PY}token", ["x"], 0x0) = 0')     # パスに文字列 token を含むだけ
        self.assertEqual(V.redact_strace_line(keep_path), keep_path)       # 非改変（パスを壊さない）
        secret_arg = _line(733, f'execve("{PY}", ["python", "password=s3cr3t"], 0x0) = 0')
        self.assertNotIn("s3cr3t", V.redact_strace_line(secret_arg))       # argv の秘密は伏せ字
        # 送ったデータは、長さと SHA-256 の先頭だけになり、元の文字は残らない
        red_line = V.redact_strace_line(lines[6])
        self.assertNotIn("abcd1234", red_line)
        self.assertIn("redacted len=", red_line)

    # ---- main()：証拠は実行者（root）の持ち物として扱う。テストでは自分の uid を渡す
    def _write(self, d, strace_lines, records, races, ids=None, sha=None, summary_ids=None,
               drop_ids=False, state="ok", sysctl="restored", apparmor="present", writable_hits=0,
               writable_hard=0, apparmor_file=True, writable_file=True):
        import hashlib

        def put(name, text):
            p = d / name
            p.write_text(text, encoding="utf-8")
            os.chmod(p, 0o644)
            return p
        put("strace.txt", "\n".join(strace_lines) + "\n")
        ids_rec = ids if ids is not None else [f"e{i}" for i in range(records)]
        rec_text = "".join(json.dumps({"eval_id": e, "count_candidate_checks": {
            "strace_evidence": "pending_external"}}) + "\n" for e in ids_rec)
        put("records.jsonl", rec_text)
        real_sha = D._sha256_file(d / "records.jsonl")
        summ = {"races": races, "records_sha256": sha or real_sha,
                "eval_ids": summary_ids if summary_ids is not None else [f"e{i}" for i in range(races)]}
        if drop_ids:
            del summ["eval_ids"]
        put("summary.json", json.dumps(summ))
        # root が書くもの：状態・sysctl・strace の SHA-256・出力の一覧（F-01・F-08）

        def sha(p):
            return hashlib.sha256((d / p).read_bytes()).hexdigest()
        put("status.txt", f"state={state}\nkill_ok=1\nwritable_after_ok=1\n")
        put("sysctl.txt", f"sysctl_state={sysctl}\nmax_user_namespaces_prev=0\n")
        # R-1：AppArmor の記録（present/absent/read_failed を区別）。R-2：診断後の走査の結果。
        # 同じ一時フォルダを小ケースで使い回すため、「ファイル無し」を試すときは前のケースの残りを必ず消す
        # （本番では BASE は毎回新規なので残留は起きない。これはテスト専用の後始末）。
        if apparmor_file:
            put("apparmor.txt", f"apparmor_restrict_unprivileged_userns_state={apparmor}\n"
                                f"apparmor_restrict_unprivileged_userns_value=1\n")
        else:
            (d / "apparmor.txt").unlink(missing_ok=True)
        if writable_file:
            put("writable_after.meta", f"# R-2 writable scan (after)\nwritable_hits={writable_hits}\n"
                                       f"unexplored_permission_denied=3\nhard_errors={writable_hard}\n")
        else:
            (d / "writable_after.meta").unlink(missing_ok=True)
        put("strace.sha256", f"{sha('strace.txt')}  {d / 'strace.txt'}\n")
        out_list = [{"path": "./records.jsonl", "type": "f", "size": (d / "records.jsonl").stat().st_size,
                     "sha256": sha("records.jsonl")},
                    {"path": "./summary.json", "type": "f", "size": (d / "summary.json").stat().st_size,
                     "sha256": sha("summary.json")}]
        put("out.list", json.dumps(out_list))

    def _main(self, d, extra=(), gate=True):
        args = ["--strace", str(d / "strace.txt"), "--records", str(d / "records.jsonl"),
                "--summary", str(d / "summary.json"), "--allow-exec", PY, "--allow-exec", LSCPU,
                "--out-dir", str(d / "verifyout"), "--evidence-owner-uid", str(os.getuid())]
        if gate:
            args += ["--status-file", str(d / "status.txt"), "--sysctl-file", str(d / "sysctl.txt"),
                     "--expect-strace-sha256-file", str(d / "strace.sha256"), "--expect-out-list", str(d / "out.list")]
        return V.main(args + list(extra))

    def _summary(self, d):
        return json.loads((d / "verifyout" / "strace_summary.json").read_text(encoding="utf-8"))

    def test_main_and_integrity(self):                              # C-05・E-02・F-08
        d = _tmp(self)
        # 伏せ字で必ずバイトが変わるよう、許可された実行の argv に秘密を含む1行を足す（通信は起こさない）
        secret_line = _line(40, f'execve("{PY}", ["python", "token=LEAK12345"], 0x0) = 0')
        self._write(d, self.BASE + [secret_line], 2, 2)
        self.assertEqual(self._main(d), 0)
        r = json.loads((d / "verifyout" / "records_final.jsonl").read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(r["count_candidate_checks"]["strace_evidence"], "ok")
        # 生の strace は外に出さず、伏せ字の写しと2つの SHA-256 を書く（F-10）
        self.assertTrue((d / "verifyout" / "strace_redacted.txt").is_file())
        s = self._summary(d)
        self.assertNotEqual(s["strace_sha256_raw"], s["strace_sha256_redacted"])    # 伏せ字で中身が変わる
        self.assertTrue(s["redaction_preserves_judgement"])                          # 判定は不変（核心）
        # 外に出る伏せ字済みの写しに、秘密の送信データが残らない（F-10）
        self.assertNotIn("LEAK12345", (d / "verifyout" / "strace_redacted.txt").read_text(encoding="utf-8"))
        # 通信が外向きなら rc 2
        self._write(d, self.BASE + [_line(9, 'connect(3<TCP:[1]>, {sa_family=AF_INET, sin_port=htons(443)}, 16) = 0')], 2, 2)
        self.assertEqual(self._main(d), 2)
        # 行数と races が違う → rc 3
        self._write(d, self.BASE, 1, 2)
        self.assertEqual(self._main(d), 3)

    def test_gate_status_sysctl(self):                              # F-02・F-03（状態の門）
        d = _tmp(self)
        self._write(d, self.BASE, 2, 2, state="kill_failed")
        self.assertEqual(self._main(d), 2)                               # out を開かない
        s = self._summary(d)
        self.assertIs(s["opened_outputs"], False)
        self.assertFalse((d / "verifyout" / "records_final.jsonl").exists())
        self._write(d, self.BASE, 2, 2, sysctl="restore_failed")
        self.assertEqual(self._main(d), 2)
        self._write(d, self.BASE, 2, 2, sysctl="restore_unconfirmed")
        self.assertEqual(self._main(d), 2)
        self._write(d, self.BASE, 2, 2)                                   # ok に戻せば合格
        self.assertEqual(self._main(d), 0)

    def test_r2_writable_after_gate(self):                          # R-2：診断後の走査が門になる
        d = _tmp(self)
        self._write(d, self.BASE, 2, 2, writable_hits=1)                  # 書ける場所が1件 → 不合格
        self.assertEqual(self._main(d), 2)
        self.assertFalse((d / "verifyout" / "records_final.jsonl").exists())
        self._write(d, self.BASE, 2, 2, writable_hard=1)                  # 硬い誤りがある → 不合格
        self.assertEqual(self._main(d), 2)
        self._write(d, self.BASE, 2, 2, writable_file=False)             # 走査の記録が無い → 不合格
        self.assertEqual(self._main(d), 2)
        self.assertIn("writable_after:file_absent", self._summary(d)["gate"]["problems"])
        self._write(d, self.BASE, 2, 2, writable_hits=0, writable_hard=0)  # 0件・誤り無し → 合格
        self.assertEqual(self._main(d), 0)

    def test_r1_apparmor_recorded_not_gated(self):                  # R-1：AppArmor は記録のみ（門にしない）
        d = _tmp(self)
        for st in ("present", "absent", "read_failed"):
            self._write(d, self.BASE, 2, 2, apparmor=st)
            self.assertEqual(self._main(d), 0, st)                        # 値が何であれ合格を妨げない
            self.assertEqual(self._summary(d)["gate"]["apparmor"]
                             ["apparmor_restrict_unprivileged_userns_state"], st)
        self._write(d, self.BASE, 2, 2, apparmor_file=False)             # ファイルが無くても門にはしない
        self.assertEqual(self._main(d), 0)
        self.assertEqual(self._summary(d)["gate"]["apparmor"], "file_absent")

    def test_e02_expect_sha_and_list(self):                         # E-02・F-08
        d = _tmp(self)
        self._write(d, self.BASE, 2, 2)
        self.assertEqual(self._main(d), 0)
        self.assertEqual(set(self._summary(d)["evidence_integrity"][k] for k in
                             ("strace_sha256_matches_root", "records_matches_out_list", "summary_matches_out_list")),
                         {True})
        for target in ("strace.txt", "records.jsonl", "summary.json"):
            self._write(d, self.BASE, 2, 2)
            (d / target).write_bytes((d / target).read_bytes() + b"\n")   # root の記録の後に書き換えられた
            self.assertEqual(self._main(d), 2, target)
        # F-08：別の場所に同じ名前のファイルがあっても、./records.jsonl の照合は本物で True（偽の合格にならない）
        self._write(d, self.BASE, 2, 2)
        lst = json.loads((d / "out.list").read_text(encoding="utf-8"))
        lst.append({"path": "a/records.jsonl", "type": "f", "size": 1, "sha256": "0" * 64})
        (d / "out.list").write_text(json.dumps(lst), encoding="utf-8")
        self.assertEqual(self._main(d), 0)
        # 同じパスが2行あれば不一致（F-08）
        self._write(d, self.BASE, 2, 2)
        lst = json.loads((d / "out.list").read_text(encoding="utf-8"))
        lst.append(dict(lst[0]))
        (d / "out.list").write_text(json.dumps(lst), encoding="utf-8")
        self.assertEqual(self._main(d), 2)

    def test_e07_eval_ids(self):                                     # E-07（V-n15・V-n16）
        d = _tmp(self)
        self._write(d, self.BASE, 2, 2, summary_ids=["e0", "e1", "e1"])   # 集合は同じでも、重複があれば違う
        self.assertEqual(self._main(d), 3)
        self._write(d, self.BASE, 2, 2, summary_ids=["e1", "e0"])         # 順番は問わない
        self.assertEqual(self._main(d), 0)
        self._write(d, self.BASE, 2, 2, drop_ids=True)                    # summary に eval_ids が無い
        self.assertEqual(self._main(d), 3)
        self.assertIs(self._summary(d)["integrity"]["eval_ids_match"], False)

    def test_e06_corrupt_lines(self):                                # E-06（V-n17：--summary が無くても壊れた行で rc 3）
        d = _tmp(self)
        self._write(d, self.BASE, 2, 2)
        with open(d / "records.jsonl", "a", encoding="utf-8") as f:
            f.write("{not json\n[1, 2]\n")
        os.chmod(d / "records.jsonl", 0o644)
        # records の中身が変わったので、照合（out.list）を外して壊れた行だけを見る
        self.assertEqual(self._main(d, gate=False), 3)
        s = self._summary(d)
        self.assertEqual(s["integrity"]["records_corrupt_lines"], 2)
        self.assertEqual(s["integrity"]["records"], 2)
        self.assertEqual(len((d / "verifyout" / "records_final.jsonl").read_text(
            encoding="utf-8").splitlines()), 2)
        # --summary が壊れていても rc 3
        self._write(d, self.BASE, 2, 2)
        (d / "summary.json").write_text("{broken", encoding="utf-8")
        os.chmod(d / "summary.json", 0o644)
        self.assertEqual(self._main(d, gate=False), 3)
        self.assertTrue(self._summary(d)["integrity"]["summary_corrupt"])

    def test_records_sha_and_duplicate_ids(self):                   # D-07
        d = _tmp(self)
        self._write(d, self.BASE, 2, 2, sha="0" * 64)                    # records の SHA-256 が違う
        self.assertEqual(self._main(d, gate=False), 3)
        self._write(d, self.BASE, 2, 2, ids=["e0", "e0"])               # eval_id の重複
        self.assertEqual(self._main(d, gate=False), 3)
        self._write(d, self.BASE, 2, 2, ids=["e0", "zz"])               # eval_id の集合が違う
        self.assertEqual(self._main(d, gate=False), 3)

    def test_missing_outputs(self):
        d = _tmp(self)
        self._write(d, self.BASE, 2, 2)
        os.remove(d / "records.jsonl")                                   # records が無い：落ちずに rc 3
        self.assertEqual(self._main(d, gate=False), 3)
        self.assertFalse(self._summary(d)["integrity"]["records_present"])
        self._write(d, self.BASE, 2, 2)
        os.remove(d / "strace.txt")                                      # strace が無い：rc 2
        self.assertEqual(self._main(d, gate=False), 2)


class VerifyUnclassifiedTests(VerifyStraceTests):
    """G-31：apply_to_records の fail-closed（F-10）。未分類キーがあれば records_final.jsonl を
    新規作成も上書きもせず、integrity と rc がそれに整合する。"""

    def _restamp_records(self, d, rec_text):
        """records.jsonl を差し替え、summary.records_sha256 と out.list を合わせて、
        未分類キーだけが唯一の失敗要因になるようにする（SHA・一覧の不一致で濁らせない）。"""
        import hashlib
        p = d / "records.jsonl"
        p.write_text(rec_text, encoding="utf-8")
        os.chmod(p, 0o644)
        summ = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        summ["records_sha256"] = D._sha256_file(p)
        (d / "summary.json").write_text(json.dumps(summ), encoding="utf-8")
        os.chmod(d / "summary.json", 0o644)
        lst = json.loads((d / "out.list").read_text(encoding="utf-8"))
        for e in lst:
            if e.get("path") in ("./records.jsonl", "./summary.json"):
                f = d / e["path"][2:]
                e["size"] = f.stat().st_size
                e["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        (d / "out.list").write_text(json.dumps(lst), encoding="utf-8")
        os.chmod(d / "out.list", 0o644)

    def test_unclassified_skips_records_final(self):                # 生本文のキー → 成果物を作らない
        d = _tmp(self)
        self._write(d, self.BASE, 1, 1)
        self._restamp_records(d, json.dumps({"eval_id": "e0", "child_stderr_tail": "boom SECRET"}) + "\n")
        self.assertEqual(self._main(d), 2)
        integ = self._summary(d)["integrity"]
        self.assertIn("child_stderr_tail", integ["unclassified_output_keys"])
        self.assertIs(integ["records_final_written"], False)
        self.assertIs(integ["ok"], False)
        self.assertFalse((d / "verifyout" / "records_final.jsonl").exists())

    def test_unclassified_preserves_existing_records_final(self):   # 既存ファイルを truncate も上書きもしない
        d = _tmp(self)
        self._write(d, self.BASE, 1, 1)
        self._restamp_records(d, json.dumps({"eval_id": "e0", "prep_trace_tail": "x"}) + "\n")
        (d / "verifyout").mkdir(parents=True, exist_ok=True)
        sentinel = d / "verifyout" / "records_final.jsonl"
        sentinel.write_text("PRIOR-CONTENT\n", encoding="utf-8")
        self.assertEqual(self._main(d), 2)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "PRIOR-CONTENT\n")
        self.assertIn("prep_trace_tail", self._summary(d)["integrity"]["unclassified_output_keys"])

    def test_classified_writes_records_final(self):                 # 対照：分類済みなら成果物を書き、integrity 一致
        d = _tmp(self)
        self._write(d, self.BASE, 1, 1)                                  # 既定レコードは全キー分類済み
        self.assertEqual(self._main(d), 0)
        integ = self._summary(d)["integrity"]
        self.assertEqual(integ["unclassified_output_keys"], [])
        self.assertIs(integ["records_final_written"], True)
        self.assertIs(integ["ok"], True)
        self.assertTrue((d / "verifyout" / "records_final.jsonl").is_file())


class FetchTests(unittest.TestCase):
    def setUp(self):
        p = _no_network()
        p.start()
        self.addCleanup(p.stop)

    def test_fetch_with_fakes(self):
        d = _tmp(self)
        raw = json.dumps({"programs": [
            {"race_stadium_number": 1, "race_number": n, "race_closed_at": f"2026-11-07 1{n}:00:00"}
            for n in (1, 2, 3)]}).encode()
        calls = []

        def fake_odds(race_no, venue_code, race_date):
            calls.append((race_no, venue_code, race_date))
            if race_no == 2:
                raise RuntimeError("down")
            return {} if race_no == 3 else {"1-2-3": 10.0}
        fp = lambda rd: {"raw_bytes": raw, "url": "u", "status_code": 200, "targets_sha256": "t"}
        m = FT.fetch("20261107", d, fix_programs=fp, fetch_odds=fake_odds, sleep_sec=0)
        self.assertEqual(calls, [(1, "01", "20261107"), (2, "01", "20261107"), (3, "01", "20261107")])
        self.assertEqual((d / "programs_raw.json").read_bytes(), raw)
        odds = json.loads((d / "odds.json").read_text(encoding="utf-8"))["races"]
        self.assertEqual(odds["20261107_01_01"]["odds"], {"1-2-3": 10.0})
        self.assertIsNone(odds["20261107_01_01"]["error"])
        self.assertEqual(odds["20261107_01_02"]["error"], "unexpected_error")  # G-21：クラス名を残さない
        self.assertEqual(odds["20261107_01_03"]["error"], "empty")          # F03
        self.assertEqual(odds["20261107_01_01"]["kind"], FT.ODDS_KIND)
        self.assertEqual(m["odds_errors"], 2)
        d2 = _tmp(self)
        calls.clear()
        m = FT.fetch("20261107", d2, fix_programs=fp, fetch_odds=fake_odds, sleep_sec=0,
                     only=["20261107_01_03"])                               # F04
        self.assertEqual((calls, m["races"]), ([(3, "01", "20261107")], 1))

    def test_main_refuses_without_fetching(self):                   # C-04・F01・F08
        def must_not_fetch(*a, **k):
            raise AssertionError("fetch must not be called")
        for name in ("GITHUB_TOKEN", "MY_SECRET", "GMAIL_APP_PASS", "DB_PASSWORD", "GH_PAT", "SLACK_WEBHOOK_URL",
                     "AWS_ACCESS_KEY_ID", "NETRC", "SSH_AUTH_SOCK"):
            with mock.patch.dict(os.environ, {"PATH": os.environ.get("PATH", ""), name: "v"}, clear=True), \
                    mock.patch.object(FT, "fetch", must_not_fetch):
                self.assertEqual(FT.main(["--race-date", "20261107", "--out", str(_tmp(self))]), 1, name)

    def test_github_default_names_do_not_refuse(self):
        defaults = {k: "x" for k in ("GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_WORKSPACE", "GITHUB_REPOSITORY",
                                     "GITHUB_ACTIONS", "GITHUB_ENV", "GITHUB_OUTPUT", "GITHUB_EVENT_PATH",
                                     "RUNNER_OS", "RUNNER_TEMP", "RUNNER_TOOL_CACHE", "ImageOS", "ImageVersion")}
        with mock.patch.dict(os.environ, {"PATH": os.environ.get("PATH", ""), **defaults}, clear=True):
            self.assertEqual(FT.sensitive_env_names(), [])


class WorkflowStaticTests(unittest.TestCase):
    def setUp(self):
        self.text = WORKFLOW.read_text(encoding="utf-8")
        self.code = "\n".join(l for l in self.text.splitlines() if not l.lstrip().startswith("#"))

    def _step(self, name):
        m = re.search(r"- name: " + re.escape(name) + r"\n(.*?)(?=\n      - name: |\Z)", self.code, re.S)
        self.assertIsNotNone(m, name)
        return m.group(1)

    def _run_block(self, name):
        m = re.search(r"run: \|\n(.*)", self._step(name), re.S)
        self.assertIsNotNone(m, name)
        return m.group(1)

    def _logical_commands(self, block):
        return [re.sub(r"\s+", " ", c) for c in re.sub(r"\\\n\s*", " ", block).splitlines()]

    def test_trigger_permissions_runner(self):                      # W07
        self.assertRegex(self.code, r"(?m)^on:\n  workflow_dispatch:")
        for t in ("schedule:", "push:", "pull_request", "workflow_run", "repository_dispatch"):
            self.assertNotIn(t, self.code)
        self.assertRegex(self.code, r"(?m)^permissions: \{\}$")
        self.assertEqual(self.code.count("permissions:"), 1)
        self.assertEqual(re.findall(r"runs-on: (.+)", self.code), ["ubuntu-24.04"])

    def test_expressions_allowlisted(self):                         # W05・W06
        allowed = {"github.repository", "github.sha", "inputs.race_date", "inputs.only", "github.run_id",
                   "github.run_attempt", "steps.upload.outputs.artifact-digest", "steps.upload.outputs.artifact-id",
                   "runner.environment", "always()"}
        exprs = set(re.findall(r"\$\{\{\s*(.*?)\s*\}\}", self.code))
        self.assertTrue(exprs <= allowed, exprs - allowed)
        for t in ("secrets", "GITHUB_TOKEN", "github.token", "toJSON", "github[", "actions/checkout",
                  "contents: write", "gh release", "releases", "continue-on-error", "hit_record"):   # W17・W18
            self.assertNotIn(t, self.code)
        self.assertIn('token: ""', self.code)

    def test_actions_pinned(self):
        uses = re.findall(r"uses:\s*([^\s#]+)", self.code)
        self.assertEqual(len(uses), 2)
        for u in uses:
            self.assertRegex(u, r"@[0-9a-f]{40}$")

    def test_set_e_and_token_check(self):                           # W12・W16
        for name in ("Fetch source anonymously", "Install dependencies and strace", "Fetch inputs (network)",
                     "Diagnose (no network)", "Verify strace evidence"):
            self.assertEqual(self._run_block(name).strip().splitlines()[0], "set -euo pipefail", name)
        self.assertIn("if env | grep -qiE '^[^=]*token[^=]*='; then echo \"token-like environment variable is present\"; exit 1; fi",
                      self._step("Fetch source anonymously"))

    def test_diagnose_calls_root_script(self):                      # W08・W09・W13・F-03
        names = re.findall(r"- name: (.+)", self.code)
        self.assertLess(names.index("Fetch inputs (network)"), names.index("Diagnose (no network)"))
        self.assertLess(names.index("Diagnose (no network)"), names.index("Verify strace evidence"))
        block = self._run_block("Diagnose (no network)")
        cmds = [c.strip() for c in self._logical_commands(block) if c.strip()]
        # 診断の段は root の台本だけを呼ぶ。cgroup・専用の利用者・写し・証拠・unshare(net,pid,mount)・strace は台本の中
        self.assertIn('sudo bash "${GITHUB_WORKSPACE}/rebuild/actions/g3_diagnostic_run.sh" '
                      '"${GITHUB_WORKSPACE}" /opt/g3diag "$PY" "$RACE_DATE" "$ONLY" "$G3_RUN_ID" '
                      '"github-hosted" "${LD_LIBRARY_PATH:-}"', cmds)
        # F-03：runner が GitHub の使い捨ての VM であることを確かめてから動かす。
        # env: の割り当ては run: 本文ではなくステップ全体に現れるので、ステップ全文で照合する
        step = self._step("Diagnose (no network)")
        self.assertIn("RUNNER_ENVIRONMENT: ${{ runner.environment }}", step)
        self.assertIn('[ "${RUNNER_ENVIRONMENT}" = "github-hosted" ]', block)
        self.assertEqual(self.code.count("g3_diagnostic_run.sh"), 1)
        self.assertNotIn("unshare", self._step("Fetch inputs (network)"))
        self.assertIn("actions.g3_diagnostic_fetch", self._step("Fetch inputs (network)"))

    def test_verify_and_collect(self):                              # W15・E-02・F-05・F-08・F-10
        names = re.findall(r"- name: (.+)", self.code)
        self.assertEqual(names[names.index("Diagnose (no network)"):],
                         ["Diagnose (no network)", "Verify strace evidence", "Collect evidence", "Upload evidence",
                          "Record artifact digest"])
        v = self._step("Verify strace evidence")
        self.assertIn("if: always()", v)
        cmd = [c for c in self._logical_commands(self._run_block("Verify strace evidence"))
               if "g3_diagnostic_verify.py" in c]
        self.assertEqual(len(cmd), 1)
        # F-05：検査は root の持つ写し（BASE/src）のコードを -I で動かす
        self.assertIn('sudo "$PY" -I -P -B "$B/src/rebuild/actions/g3_diagnostic_verify.py"', cmd[0])
        # 状態・sysctl・SHA-256・一覧を読み、生の strace（root 専用）を照合する
        for part in ('--strace "$B/evidence/strace.txt"', '--records "$B/out/records.jsonl"',
                     '--summary "$B/out/summary.json"', '--status-file "$B/evidence/public/status.txt"',
                     '--sysctl-file "$B/evidence/public/sysctl.txt"',
                     '--expect-strace-sha256-file "$B/evidence/public/strace.sha256"',
                     '--expect-out-list "$B/evidence/public/out.list"', "--out-dir g3_out"):
            self.assertIn(part, cmd[0])
        self.assertIn("B=/opt/g3diag", v)
        # 起動を許可するのは、setpriv・env・python・lscpu の4つだけ
        self.assertEqual(re.findall(r"--allow-exec (\S+)", v),
                         ['"$SETPRIV"', '"$ENVBIN"', '"$PY"', '"$LSCPU"'])
        for var, cmd_name in (("PY", "python"), ("LSCPU", "lscpu"), ("SETPRIV", "setpriv"), ("ENVBIN", "env")):
            self.assertIn(f'{var}="$(command -v {cmd_name})"', v)
        # F-10・F-01：集めるのは BASE/collect と外に出してよい証拠（public）だけ。生の strace は含めない。リンクはたどらない
        c = self._step("Collect evidence")
        self.assertIn("if: always()", c)
        self.assertIn("/opt/g3diag/collect/.", c)
        self.assertIn("/opt/g3diag/evidence/public/.", c)
        self.assertNotIn("/opt/g3diag/evidence/strace.txt", c)
        self.assertNotIn("/opt/g3diag/evidence/.", c)             # evidence 全体は写さない（生の strace を外に出さない）
        self.assertIn("cp -r -P", c)                              # リンクをたどらない
        self.assertIn("g3_diagnostic_collect.py check", c)        # 集めた run を一覧と照合する
        self.assertIn("if: always()", self._step("Upload evidence"))

    def test_install_locked(self):                                  # W19
        inst = self._run_block("Install dependencies and strace")
        self.assertIn("python -m pip install --no-deps -r requirements-g3-diagnostic.lock", inst)
        self.assertIn("python -m pip check", inst)
        lines = [l for l in LOCK.read_text(encoding="utf-8").splitlines() if l and not l.startswith("#")]
        for l in lines:
            self.assertRegex(l, r"^[A-Za-z0-9_.\-]+==[A-Za-z0-9_.+\-]+$")
        self.assertIn("scikit-learn==1.5.2", lines)
        self.assertIn("lightgbm==4.5.0", lines)

class RunScriptStaticTests(unittest.TestCase):
    """第3版（F-01〜F-10）：root の台本 g3_diagnostic_run.sh と、名前空間の中の inner・ソケット探索の台本。
    実際の攻撃に対する耐性は、ここでは確かめない（静的な組み立ての確認のみ）。"""

    SCRIPT = S.REBUILD_DIR / "actions/g3_diagnostic_run.sh"
    INNER = S.REBUILD_DIR / "actions/g3_diagnostic_inner.sh"
    SOCK = S.REBUILD_DIR / "actions/g3_find_writable_sockets.sh"

    def setUp(self):
        self.raw = self.SCRIPT.read_text(encoding="utf-8")
        self.code = "\n".join(l for l in self.raw.splitlines() if not l.lstrip().startswith("#"))
        self.inner = "\n".join(l for l in self.INNER.read_text(encoding="utf-8").splitlines()
                               if not l.lstrip().startswith("#"))

    def _pos(self, s, code=None):
        i = (code or self.code).find(s)
        self.assertGreaterEqual(i, 0, s)
        return i

    def test_header_and_guards(self):
        for p in (self.SCRIPT, self.INNER, self.SOCK):
            self.assertTrue(p.read_text(encoding="utf-8").startswith("#!/usr/bin/env bash\n"), p)
        self.assertIn("set -euo pipefail", self.code)
        self.assertIn('[ "$(id -u)" = "0" ] || die "must run as root"', self.code)
        self.assertIn('[ -e "$BASE" ] && die "base already exists: $BASE"', self.code)
        # 削除・ネットワーク取得・特権付与はしない（禁止事項）
        # 破壊的な全消し・ネットワーク取得・特権付与はしない。rmdir は cgroup（空ディレクトリ）の後始末に必要なので除く
        for t in ("rm -rf", "curl", "wget", "GITHUB_TOKEN", "secrets", "chmod 777", "--privileged", "usermod"):
            self.assertNotIn(t, self.code)
        # rmdir を使うのは、作った cgroup の後始末だけ（利用者のファイル・出力は消さない）
        self.assertTrue(all("cgroup" in l.lower() or "CG" in l for l in self.code.splitlines() if "rmdir" in l))
        # runner の種類を確かめる（F-03）
        self.assertIn('local-test) [ -f /opt/g3test/LOCAL_TEST_HOST ]', self.code)
        self.assertIn('*) die "runner_kind must be github-hosted or local-test"', self.code)

    def test_cgroup_precheck(self):                                   # F-02（C1〜C6）
        self.assertIn('"$FLOCK" -n "$LOCKFD" || die', self.code)       # C6：flock を最初から持つ
        self.assertIn('exec {LOCKFD}>/run/g3diag.lock', self.code)
        self.assertIn('awk \'$3=="cgroup2"{print $2; exit}\' /proc/self/mountinfo', self.code)  # C1
        self.assertIn('mkdir "$CG" 2>/dev/null || die "cannot create run cgroup (C2)"', self.code)  # C2
        self.assertIn('[ -e "$CG/cgroup.kill" ] && [ -e "$CG/cgroup.events" ]', self.code)      # C3
        self.assertIn('echo 1 > "$CGTEST/cgroup.kill"', self.code)                              # C4
        self.assertIn('populated 0', self.code)
        # C5：追跡される側が cgroup.procs に書けないこと。探索の誤り・未探索があれば合格にしない
        self.assertIn('find "$CG2ROOT" -name cgroup.procs -writable', self.code)
        self.assertIn('die "dedicated user can write cgroup.procs (C5)"', self.code)
        self.assertIn('die "C5 cgroup scan had errors', self.code)
        # cgroup が使えなければ診断を始めない（uid へ切り替えない）
        self.assertNotRegex(self.code, r"cgroup.*(fall ?back|切り替え).*uid")
        # uid で止めるのは「予備」で、cgroup.kill の後だけ
        self.assertLess(self._pos('echo 1 > "$CG/cgroup.kill" 2>/dev/null || true'),
                        self._pos('pkill -KILL -u "$DUID"'))

    def test_dedicated_user_and_preexisting(self):                    # E-01・F-06
        self.assertIn('useradd --system --no-create-home --home-dir "$BASE/home" --shell /usr/sbin/nologin '
                      '--user-group "$USER_NAME"', self.code)
        self.assertIn('{ [ "$DUID" != "0" ] && [ "$DGID" != "0" ]; } || die "dedicated user must not be root"',
                      self.code)
        self.assertIn('[ "$(id -G "$USER_NAME")" = "$DGID" ] || die "dedicated user has extra groups"', self.code)
        self.assertIn('getent passwd | awk -F: -v u="$DUID" \'$3==u\'', self.code)   # uid が専用（予備の前提）
        self.assertIn('if pgrep -u "$DUID" >/dev/null 2>&1; then die "processes of the dedicated user already exist"',
                      self.code)
        for s in ("/var/run/docker.sock", "/run/docker.sock", "/run/containerd/containerd.sock"):
            self.assertIn(s, self.code)
        self.assertIn('if [ "$sg" = "$DGID" ] || [ "$so" = "$DUID" ]; then die "dedicated user can reach $s"',
                      self.code)

    def test_read_only_source_and_root_evidence(self):               # E-02・F-05・F-10
        self.assertIn('tar -C "$WS" --exclude=./.git -cf - . | tar -C "$BASE/src" -xf -', self.code)
        self.assertIn('chown -R root:root "$BASE/src" "$BASE/evidence"', self.code)
        self.assertIn('chmod -R u+rwX,go+rX,go-w "$BASE/src"', self.code)
        self.assertIn('mkdir -m 0700 "$BASE/evidence"', self.code)       # F-10：生の strace は root 専用
        self.assertIn('install -d -m 0755 "$BASE/evidence/public"', self.code)
        # 出力のフォルダ（out）だけを専用の利用者に渡す。src・evidence は root のまま。home は名前空間の中の tmpfs
        self.assertIn('chown "$DUID:$DGID" "$BASE/out"', self.code)
        self.assertEqual(len(re.findall(r'chown "\$DUID:\$DGID" "\$BASE/out"', self.code)), 1)
        self.assertNotIn('chown "$DUID:$DGID" "$BASE/src"', self.code)
        self.assertNotIn('chown "$DUID:$DGID" "$BASE/evidence"', self.code)
        self.assertIn('install -d -m 0700 -o "$DUID" -g "$DGID" "$RUNHOME"', self.inner)
        # F-05：書けないことの確認は、workspace・写し・collect の3ファイル＋ src 全体の SHA-256
        for t in ('"$WS/rebuild/actions/g3_diagnostic_verify.py"',
                  '"$BASE/src/rebuild/actions/g3_diagnostic_verify.py"',
                  '"$BASE/src/rebuild/actions/g3_diagnostic_collect.py"'):
            self.assertIn(t, self.code)
        self.assertIn('SRC_SHA_BEFORE=', self.code)
        self.assertIn('SRC_SHA_AFTER=', self.code)
        self.assertIn('[ "$SRC_SHA_AFTER" = "$SRC_SHA_BEFORE" ] && SRC_UNCHANGED=1', self.code)

    def test_sysctl(self):                                            # F-03
        self.assertIn('trap cleanup EXIT', self.code)
        self.assertIn("trap 'exit 130' INT", self.code)
        self.assertIn("trap 'exit 143' TERM HUP", self.code)
        # trap は sysctl を変えるより前
        self.assertLess(self._pos('trap cleanup EXIT'), self._pos('echo 0 > /proc/sys/user/max_user_namespaces'))
        self.assertIn('SYSCTL_PREV="$(cat /proc/sys/user/max_user_namespaces', self.code)
        self.assertIn('die "cannot set user.max_user_namespaces=0" 95', self.code)
        for st in ("restored", "restore_failed", "restore_unconfirmed"):
            self.assertIn(st, self.code)
        # 事前の確認：追跡される側が user・mount の名前空間を作れないこと
        self.assertIn('"$UNSHARE" -U -r true', self.code)
        self.assertIn('"$UNSHARE" -m true', self.code)

    def test_namespace_launch(self):                                  # F-02・F-03・F-04・F-07
        # 包みの台本が自分を cgroup に入れてから unshare(net,pid,mount) を exec する
        self.assertIn('echo \\$\\$ > "$CG/cgroup.procs"', self.code)
        self.assertIn('exec "$UNSHARE" --net --pid --fork --kill-child --mount --propagation private --mount-proc --',
                      self.code)
        # inner：マウントを private にして tmpfs をかぶせ、ソケットを調べ、strace を exec する
        self.assertIn('mount --make-rprivate /', self.inner)
        for d in ("/run", "/tmp", "/var/tmp", "/dev/shm"):
            self.assertIn(d, self.inner)
        self.assertIn('readlink -f /var/run', self.inner)
        self.assertIn('bash "$SOCK_FINDER"', self.inner)
        self.assertIn('log "writable sockets are reachable by the dedicated user"', self.inner)
        self.assertIn('exec "$STRACE" --kill-on-exit -f -qq -yy -s 256 -e "trace=${G3_STRACE_TRACE}"', self.inner)
        self.assertIn("--clear-groups --no-new-privs", self.inner)
        self.assertIn("actions.g3_diagnostic_entry parent", self.inner)
        # ソケット探索は proc・sys・cgroup を除く
        sock = self.SOCK.read_text(encoding="utf-8")
        self.assertIn("-type s -writable", sock)
        for fs in ("proc", "sysfs", "cgroup", "cgroup2"):
            self.assertIn(fs, sock)

    def test_strace_trace_matches_verifier(self):                     # F-03：記録する呼び出しは検査と同じ
        self.assertIn("from actions.g3_diagnostic_verify import STRACE_TRACE", self.code)
        self.assertIn('trace=${G3_STRACE_TRACE}', self.inner)
        for call in ("%network", "execve", "unshare", "setns", "mount", "clone", "io_uring_setup"):
            self.assertIn(call, V.STRACE_TRACE)

    def test_kill_and_freeze_order(self):                             # F-02・F-01
        # section 6 以降だけを見る（cleanup()・C4 プリチェックにも現れる目印を拾わないよう、起点を固定）
        start = self._pos('bash "$WRAP" || rc=$?')
        body = self.code[start:]

        def rel(s):
            i = body.find(s)
            self.assertGreaterEqual(i, 0, s)
            return i
        order = [rel('bash "$WRAP" || rc=$?'),
                 rel('echo 1 > "$CG/cgroup.kill" 2>/dev/null || true'),
                 rel("grep -q '^populated 0$' \"$CG/cgroup.events\""),
                 rel('pkill -KILL -u "$DUID"'),
                 rel('readlink "$l"'),                         # 開いた FD の確認
                 rel('chown -R root:root "$BASE/out"'),        # 凍結
                 rel('chmod -R a-w,a+rX "$BASE/out"'),
                 rel('"$COLLECT" list "$BASE/out" "$PUB/out.list"'),   # 一覧
                 rel('"$COLLECT" copy "$BASE/out" "$BASE/collect"'),   # 写し
                 rel('echo "state=$STATE"')]
        self.assertEqual(order, sorted(order))
        # 止まったことを確かめる4つの条件
        self.assertIn('if [ "$CG_EMPTY" = 1 ] && [ "$UID_AFTER" = 0 ] && [ "$STRAY" = 0 ] && [ "$OPEN_FDS" = 0 ]',
                      self.code)
        # kill_failed・src_changed・writable_dirty・collect_invalid は、それぞれ別の終了コード（検査が必ず不合格にできる）
        self.assertIn('case "$STATE" in kill_failed) exit 93;; src_changed) exit 92;; '
                      'writable_dirty) exit 98;; collect_invalid) exit 94;;',
                      self.code)
        # 生の strace の SHA-256 は、止まったかどうかに関わらず書く
        self.assertIn('sha256sum "$BASE/evidence/strace.txt" > "$PUB/strace.sha256"', self.code)

    def test_state_file(self):                                        # F-02・F-03（検査の門）
        self.assertIn('echo "state=$STATE"', self.code)
        self.assertIn('> "$PUB/status.txt"', self.code)
        self.assertIn('chmod 0444 "$PUB/status.txt"', self.code)

    def test_r1_apparmor_record(self):                                # R-1
        self.assertIn('APPARMOR_SYSCTL="/proc/sys/kernel/apparmor_restrict_unprivileged_userns"', self.code)
        self.assertIn("_write_apparmor()", self.code)
        # 不存在・取得失敗・値 を区別する（同じ扱いにしない）。変更はしない（echo で書き込まない）
        self.assertIn('state="absent"', self.code)
        self.assertIn('state="read_failed"', self.code)
        self.assertIn('state="present"', self.code)
        self.assertNotRegex(self.code, r">\s*\"?\$APPARMOR_SYSCTL")          # 値を書き換えない
        self.assertNotRegex(self.code, r"sysctl\s+-w\s+kernel\.apparmor")
        # sysctl の「読み取り」より前に記録すること。区切りは初期化行の SYSCTL_PREV="" ではなく読み取り行を使う
        self.assertIn('_write_apparmor', self.code.split('SYSCTL_PREV="$(cat')[0])
        # 証拠は public に残す（verify が読む）
        self.assertIn('"$PUB/apparmor.txt"', self.code)

    def test_r2_writable_scan(self):                                  # R-2
        self.assertIn('WRITABLE_FINDER="$SELF_DIR/g3_find_writable_paths.sh"', self.code)
        self.assertIn("_writable_scan()", self.code)
        # 対象：workspace・BASE/src・Python の検索パス（sys.path）
        self.assertIn('local -a roots=("$WS" "$BASE/src")', self.code)
        self.assertIn("import sys", self.code)
        self.assertIn("sys.path", self.code)
        # 検査の主体は本番と同じ RUN_AS（g3diag・補助グループなし）
        self.assertIn('"${RUN_AS[@]}" "${ENVS_CHK[@]}" bash "$WRITABLE_FINDER"', self.code)
        # 書ける場所>0 か 硬い誤り>0 なら合格にしない。権限拒否は「未探索」として数える
        self.assertIn('[ "$hits" = 0 ] && [ "$hard" = 0 ]', self.code)
        self.assertIn("unexplored_permission_denied", self.code)
        # 前後の両方で検査する
        self.assertIn("_writable_scan before", self.code)
        self.assertIn("_writable_scan after", self.code)
        self.assertIn('die "dedicated user can write inside the scanned trees (before)"', self.code)
        self.assertIn('WRITABLE_AFTER_OK=1', self.code)
        # 後の走査を凍結の門に入れる・新しい状態 writable_dirty
        self.assertIn('[ "$WRITABLE_AFTER_OK" = 1 ]', self.code)
        self.assertIn('STATE="writable_dirty"', self.code)
        # 既存の BASE/src の SHA-256 前後照合は維持
        self.assertIn('SRC_SHA_BEFORE=', self.code)
        self.assertIn('SRC_SHA_AFTER=', self.code)
        # NUL 区切りで探し、改行入りのファイル名でも壊れない
        finder_raw = self.SOCK.parent.joinpath("g3_find_writable_paths.sh").read_text(encoding="utf-8")
        finder = "\n".join(l for l in finder_raw.splitlines() if not l.lstrip().startswith("#"))  # コメント除外
        self.assertIn("-writable -print0", finder)
        for fs in ("proc", "sysfs", "cgroup", "cgroup2"):
            self.assertIn(fs, finder)
        self.assertNotIn("-xdev", finder)                                  # 木の下の別マウントも見る（コメント除く本文）


class CollectTests(unittest.TestCase):
    """F-01：出力の一覧・写し・照合（g3_diagnostic_collect）。通常のファイルの木と、
    通常のファイル以外（リンク・hardlink・FIFO）を使った検査。実際の攻撃の実行ではなく、収集の正しさの確認。"""

    def _tree(self):
        d = _tmp(self)
        src = d / "out"
        (src / "races").mkdir(parents=True)
        (src / "records.jsonl").write_text('{"eval_id": "e0"}\n', encoding="utf-8")
        (src / "summary.json").write_text('{"races": 1}', encoding="utf-8")
        (src / "races" / "e0.json").write_text("{}", encoding="utf-8")
        return d, src

    def test_list_and_copy_and_check(self):
        d, src = self._tree()
        self.assertEqual(C.main(["list", str(src), str(d / "out.list")]), 0)
        listing = json.loads((d / "out.list").read_text(encoding="utf-8"))
        paths = {e["path"] for e in listing}
        # walk は配下の要素を列挙する（根ノード "." は出さない。copy・check は "." を必要としない）
        self.assertEqual(paths, {"./races", "./records.jsonl", "./summary.json", "./races/e0.json"})
        (d / "collect").mkdir()   # 写し先は空の既存フォルダ（本番の run.sh も事前に mkdir する）
        self.assertEqual(C.main(["copy", str(src), str(d / "collect"), str(d / "out.list"),
                                 str(d / "collect.list")]), 0)
        self.assertEqual((d / "collect" / "records.jsonl").read_text(encoding="utf-8"), '{"eval_id": "e0"}\n')
        self.assertEqual(C.main(["check", str(d / "collect"), str(d / "collect.list")]), 0)
        # 中身が変わったら check は不合格
        (d / "collect" / "records.jsonl").chmod(0o644)
        (d / "collect" / "records.jsonl").write_text("changed", encoding="utf-8")
        self.assertEqual(C.main(["check", str(d / "collect"), str(d / "collect.list")]), 94)

    def test_rejects_symlink(self):                                  # F-01：シンボリックリンク
        d, src = self._tree()
        secret = d / "secret.txt"
        secret.write_text("SECRETDATA", encoding="utf-8")
        os.symlink(str(secret), str(src / "link"))
        self.assertEqual(C.main(["list", str(src), str(d / "out.list")]), 94)   # 一覧の時点で拒む
        # フォルダへのリンクも拒む
        d2, src2 = self._tree()
        os.symlink(str(d2), str(src2 / "dirlink"))
        self.assertEqual(C.main(["list", str(src2), str(d2 / "l.list")]), 94)

    def test_rejects_hardlink_and_fifo(self):                        # F-01：外への hardlink・FIFO
        d, src = self._tree()
        os.link(str(src / "records.jsonl"), str(src / "hard"))       # 通常のファイルのリンク数が2
        self.assertEqual(C.main(["list", str(src), str(d / "out.list")]), 94)
        d2, src2 = self._tree()
        os.mkfifo(str(src2 / "pipe"))
        self.assertEqual(C.main(["list", str(src2), str(d2 / "l.list")]), 94)

    def test_copy_does_not_follow_links(self):                      # F-01：写しでリンク先を取り込まない
        d, src = self._tree()
        self.assertEqual(C.main(["list", str(src), str(d / "out.list")]), 0)
        # 一覧の後に写し元へリンクを足すのは「一覧後の構造変化」。写しは事後再走査で検知して fail closed（94）になる
        # （リンクをたどって中身を取り込むことは無い）。本体のより安全な挙動を明文化する。
        secret = d / "secret.txt"
        secret.write_text("SECRETDATA", encoding="utf-8")
        os.symlink(str(secret), str(src / "records.jsonl.lnk"))
        (d / "collect").mkdir()   # 写し先は空の既存フォルダ（本番と同じ）。94 はリンク検知が理由であって不在ではない
        rc = C.main(["copy", str(src), str(d / "collect"), str(d / "out.list"), str(d / "collect.list")])
        self.assertEqual(rc, 94)   # 一覧に無いリンクの出現を検知して不合格
        self.assertFalse((d / "collect" / "records.jsonl.lnk").exists())
        # リンク先の秘密が collect のどのファイルにも漏れていない（artifact 側へ出ない）
        grep = "".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in (d / "collect").rglob("*") if p.is_file())
        self.assertNotIn("SECRETDATA", grep)

    def test_detects_change_between_list_and_copy(self):            # F-01：一覧の後に中身が変わると不合格
        d, src = self._tree()
        self.assertEqual(C.main(["list", str(src), str(d / "out.list")]), 0)
        (src / "records.jsonl").write_text("tampered after listing", encoding="utf-8")
        self.assertEqual(C.main(["copy", str(src), str(d / "collect"), str(d / "out.list"),
                                 str(d / "collect.list")]), 94)


class ChildGuardTests(unittest.TestCase):
    """E12・E15・E16：子の手順の検査（ML のモデルは差し替える。準備の境界の中で動く）。"""

    def setUp(self):
        self.d = _tmp(self)
        self.odds = self.d / "odds.json"
        self.odds.write_text(json.dumps(_synth_odds([EID])), encoding="utf-8")
        # 子の手順は境界の中で動くので、外への通信は境界（自己試験つき）が止める。requests は差し替えない

    def _probs(self, boats):
        return {1: 0.5, 2: 0.2, 3: 0.1, 4: 0.1, 5: 0.05, 6: 0.05}

    def _run(self):
        import notify_arashi as N
        with S.fake_token_env(), mock.patch.object(N, "_predict_win_prob", self._probs):
            return D.run_child("20261107", EID, FIXTURE, self.odds)

    def test_prep_violation_stops(self):                            # E16
        def bad(boats):
            try:
                os.system("true")
            except OSError:
                pass
            return self._probs(boats)
        import notify_arashi as N
        with S.fake_token_env(), mock.patch.object(N, "_predict_win_prob", bad):
            rec = D.run_child("20261107", EID, FIXTURE, self.odds)
        self.assertEqual(rec["error"], "prep_isolation_not_verified")
        self.assertNotIn("g3", rec)
        self.assertIs(rec["count_candidate_checks"]["prep_isolation_verified"], False)

    def test_prep_trace_redacted(self):                             # N24
        def boom(boats):
            raise RuntimeError("failed token=SECRETX at https://u:p@h.invalid/x?k=SECRETY")
        import notify_arashi as N
        with S.fake_token_env(), mock.patch.object(N, "_predict_win_prob", boom):
            rec = D.run_child("20261107", EID, FIXTURE, self.odds)
        self.assertEqual(rec["error"], "unexpected_error")            # G-19：例外クラス名を固定ラベルに
        self.assertEqual(rec["prep_trace_tail"]["redacted"], True)    # G-18：本文でなく digest
        self.assertGreater(rec["prep_trace_tail"]["len"], 0)
        text = json.dumps(rec, ensure_ascii=False, default=str)
        self.assertNotIn("RuntimeError", text)                        # クラス名も残らない
        self.assertNotIn("SECRETX", text)
        self.assertNotIn("SECRETY", text)

    def test_asahi_flag_reflects_workdir(self):                     # N25
        with mock.patch.object(D, "PREP_SEED_NAMES", D.PREP_SEED_NAMES + ("asahi_config.json",)):
            rec = self._run()
        self.assertIs(rec["prep_inputs"]["asahi_config_json_in_workdir"], True)

    def test_freeze_sha_checked(self):                              # E12
        with mock.patch.object(D, "FREEZE_CONFIG_SHA256", "0" * 64):
            rec = self._run()
        self.assertEqual(rec["error"], "freeze_config_sha_mismatch")
        self.assertNotIn("g3", rec)

    def test_eval_id_mismatch_stops(self):                          # E15
        from actions import g3_entrypoint as G
        real = G.build_rebuild_evaluation

        def wrong(**kw):
            ev, up = real(**kw)
            return dataclasses.replace(ev, eval_id="20261107_99_99"), up
        with mock.patch.object(G, "build_rebuild_evaluation", wrong):
            rec = self._run()
        self.assertEqual(rec["error"], "eval_id_mismatch")
        self.assertIs(rec["count_candidate_checks"]["eval_id_consistent"], False)

    def test_happy_path_records(self):                              # C-11・C-12・C-13・E24
        rec = self._run()
        self.assertEqual(rec["stage"], "done", rec.get("error"))
        self.assertEqual(rec["odds_meta"], {"fetched_at": "synthetic", "kind": "synthetic_for_test",
                                            "count": 120, "error": None})
        self.assertEqual(rec["prep_inputs"]["asahi_config_json_in_workdir"], False)
        self.assertEqual(len(rec["prep_inputs"]["asahi_config_sha256"]), 64)
        self.assertTrue(rec["prep_inputs"]["fan_file_used"].startswith("fan"))
        g3 = rec["g3"]
        if g3["evaluate_bets_args"] is not None:
            self.assertEqual(g3["evaluate_bets_args"]["matches_record"],
                             {"patterns": True, "ml_probs": True, "odds_map": True, "boats": True})
        rec2 = self._run()
        self.assertEqual(rec["evaluation"]["sha256"], rec2["evaluation"]["sha256"])  # 時刻を除いた SHA-256


def _ml_skip() -> str:
    try:
        import sklearn
    except ImportError:
        return "scikit-learn missing"
    return "" if sklearn.__version__ == "1.5.2" else f"scikit-learn {sklearn.__version__} != 1.5.2"


@unittest.skipIf(_ml_skip(), _ml_skip())
class RealChildTests(unittest.TestCase):
    """実際の出走表1レース・実際の Legacy・ML で、子のプロセスの手順を通す（オッズは合成）。"""

    def test_run_child_real_race(self):
        d = _tmp(self)
        odds = d / "odds.json"
        odds.write_text(json.dumps(_synth_odds([EID])), encoding="utf-8")
        with S.fake_token_env():
            rec = D.run_child("20261107", EID, FIXTURE, odds)
        self.assertEqual(rec["stage"], "done", rec.get("error"))
        self.assertTrue(rec["prep_isolation"]["verified"], rec["prep_isolation"])
        self.assertEqual(rec["prep_isolation"]["network_attempts_observed"], 0)
        self.assertEqual(sorted(rec["prep_isolation"]["seeded"]),
                         sorted(["model_all.pkl", "motor_history.csv"] +
                                [p.name for p in S.REPO_DIR.glob("fan*.txt")]))
        self.assertTrue(rec["eval_id_consistency"]["evaluation_eval_id_matches"])
        self.assertEqual(rec["eval_id_consistency"]["boats_lanes"], [1, 2, 3, 4, 5, 6])
        g3 = rec["g3"]
        self.assertEqual(sorted(g3["isolation"]["seeded"]), ["buyscore_config.json"])
        self.assertEqual(g3["isolation"]["seeded"]["buyscore_config.json"], D.FREEZE_CONFIG_SHA256)
        self.assertTrue(g3["isolation"]["verified"])
        self.assertNotIn("hit_record.csv", g3["isolation"]["workdir_files_start"])
        self.assertEqual(rec["evaluation"]["eval_id"], EID)
        self.assertEqual(g3["win_probs"], rec["evaluation"]["win_probs"])
        self.assertEqual(rec["weather_source"], D.WEATHER_SOURCE)
        # E27：記録の win_probs は、実際の Legacy の predictor の値（境界の中で同じ boats に対して呼び直して照合）
        import notify_arashi as N
        from shadow.g3_isolation import G3Isolation
        boats = [N.BoatInfo(**b) for b in g3["inputs"]["legacy_boats"]]
        with G3Isolation({"model_all.pkl": S.REPO_DIR / "model_all.pkl"}) as iso:
            again = N._predict_win_prob(boats)
        self.assertTrue(iso.verified)
        self.assertEqual({str(k): v for k, v in again.items()}, rec["evaluation"]["win_probs"])
        self.assertNotEqual(len(set(again.values())), 1)
        c = rec["count_candidate_checks"]
        self.assertEqual(c["strace_evidence"], "pending_external")
        if g3["outcome"] == "compared":
            self.assertTrue(c["outcome_compared"] and c["config_source_is_workdir_file"] and
                            c["matches_freeze_is_true"] and c["seeded_config_sha256_is_freeze"])
        self.assertNotIn(S.FAKE_TOKEN_ENV["GITHUB_TOKEN"], json.dumps(rec, ensure_ascii=False, default=str))
        # F-10 end-to-end：実レコード（記録される redact 後の形）が検出器を通ること（B のスコープ確証）
        from shadow.g3_isolation import unclassified_output_keys as _U, redact_free_text as _R
        self.assertEqual(_U(_R(rec)), [], "real record must be detector-clean (F-10)")


class RunShCgroupDetectTests(unittest.TestCase):
    """run.sh の C1 cgroup2 検出（CG2ROOT）が /proc/self/mountinfo を正しく解析することを保証（C1 回帰防止）。
    mountinfo のフィールド3は maj:min で、fstype は '-' 区切りの後ろ。run.sh 内の実 awk 式を抽出して検証する。"""

    def _cg2_awk(self):
        import re
        src = (S.REBUILD_DIR / "actions/g3_diagnostic_run.sh").read_text(encoding="utf-8")
        m = re.search(r"CG2ROOT=\"\$\(awk '([^']*)' /proc/self/mountinfo", src)
        self.assertIsNotNone(m, "run.sh の CG2ROOT awk 行が見つからない")
        return m.group(1)

    def _run_awk(self, prog, mountinfo):
        import subprocess
        return subprocess.run(["awk", prog], input=mountinfo, capture_output=True, text=True).stdout.strip()

    def test_detects_cgroup2_mount(self):
        prog = self._cg2_awk()
        mi = ("38 28 0:32 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:10 "
              "- cgroup2 cgroup2 rw,nsdelegate,memory_recursiveprot\n")
        self.assertEqual(self._run_awk(prog, mi), "/sys/fs/cgroup")

    def test_empty_when_no_cgroup2(self):
        prog = self._cg2_awk()
        mi = ("23 28 0:21 / /sys/fs/cgroup/devices rw,nosuid shared:9 - cgroup cgroup rw,devices\n"
              "24 28 0:22 / /proc rw,nosuid - proc proc rw\n")
        self.assertEqual(self._run_awk(prog, mi), "")


class RunShFinderPathTests(unittest.TestCase):
    """run.sh の finder は g3diag（補助グループなし）が実行するため、ワークスペース(SELF_DIR)ではなく
    root 所有・go+rX の BASE/src の写しから実行しなければならない（rc=126 Permission denied の回帰防止）。"""

    def _src(self):
        return (S.REBUILD_DIR / "actions/g3_diagnostic_run.sh").read_text(encoding="utf-8")

    def test_finders_run_from_base_src(self):
        src = self._src()
        self.assertIn('SOCK_FINDER="$BASE/src/rebuild/actions/g3_find_writable_sockets.sh"', src)
        self.assertIn('WRITABLE_FINDER="$BASE/src/rebuild/actions/g3_find_writable_paths.sh"', src)

    def test_base_src_is_group_other_readable(self):
        # BASE/src を go+rX にしている（finder を g3diag が読めるための前提）
        self.assertIn("chmod -R u+rwX,go+rX,go-w \"$BASE/src\"", self._src())

    def test_python_prefix_stripped_of_group_other_write(self):
        # world-writable な toolcache の Python 実行環境から go-w を剥がす（g3diag が lib に書けないように）
        src = self._src()
        self.assertIn('PYPREFIX="$(cd "$(dirname "$PY")/.." 2>/dev/null && pwd || true)"', src)
        self.assertIn('chmod -R go-w "$PYPREFIX"', src)


if __name__ == "__main__":
    unittest.main()
