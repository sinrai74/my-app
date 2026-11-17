"""正式G3（B案）再監査 B-01：Fable の独自の変異で生き残ったものに対応する試験。

境界（shadow.g3_isolation）と入口（actions.g3_entrypoint）のコードは変えず、試験だけを足す。
exec・fork・spawn は、境界が働かないとこの試験のプロセス自体が置き換わる・増えるので、
別の子のプロセスで試す（子は終わるとすぐ消える）。
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import smtplib
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

from actions import g3_entrypoint as G
from shadow import g3_isolation as GI
from shadow.g3_isolation import G3Isolation, G3IsolationBlocked
from tests.unit import g3_support as S


def _f(sha, parsed=None, err=None):
    return {"sha256": sha, "parsed": parsed, "parse_error": err}


CFG = {"_version": "x", "a": 1}
OTHER = {"_version": "y", "a": 2}


class ConfigSourceBranchTests(unittest.TestCase):
    """determine_config_source の各分岐（F28b・F28c・X10・X11）。"""

    def _src(self, start, end, captured=CFG, reload=CFG):
        return G.determine_config_source(start, end, captured, reload, [])

    def test_workdir_file_exact(self):
        r = self._src({"buyscore_config.json": _f("a", CFG)}, {"buyscore_config.json": _f("a", CFG)})
        self.assertEqual(r["source"], "workdir_file")

    def test_workdir_file_missing_at_end(self):                      # F28b
        r = self._src({"buyscore_config.json": _f("a", CFG)}, {"buyscore_config.json": None})
        self.assertEqual((r["source"], r["source_detail"]["reason"]), ("unknown", "config_file_changed"))

    def test_default_copy_exact(self):
        r = self._src({"buyscore_config.json": None, "buyscore_config.default.json": _f("d", CFG)},
                      {"buyscore_config.json": _f("d", CFG), "buyscore_config.default.json": _f("d", CFG)})
        self.assertEqual(r["source"], "default_template_copy")
        self.assertEqual(r["source_detail"]["source_file_sha256"], "d")

    def test_default_copy_not_observed(self):                        # F28c
        for end_cfg in (None, _f("other", CFG)):
            r = self._src({"buyscore_config.json": None, "buyscore_config.default.json": _f("d", CFG)},
                          {"buyscore_config.json": end_cfg, "buyscore_config.default.json": _f("d", CFG)})
            self.assertEqual((r["source"], r["source_detail"]["reason"]),
                             ("unknown", "default_copy_not_observed"))

    def test_default_not_equal_captured(self):                       # X10
        r = self._src({"buyscore_config.json": None, "buyscore_config.default.json": _f("d", OTHER)},
                      {"buyscore_config.json": _f("d", OTHER), "buyscore_config.default.json": _f("d", OTHER)})
        self.assertEqual((r["source"], r["source_detail"]["reason"]),
                         ("unknown", "default_file_not_equal_captured"))

    def test_default_invalid(self):
        r = self._src({"buyscore_config.json": None, "buyscore_config.default.json": _f("d", None, "ValueError")},
                      {"buyscore_config.json": None, "buyscore_config.default.json": _f("d", None, "ValueError")})
        self.assertEqual(r["source_detail"]["reason"], "default_file_invalid")

    def test_builtin_defaults_and_file_appeared(self):               # X11
        none = {"buyscore_config.json": None, "buyscore_config.default.json": None}
        self.assertEqual(self._src(none, none)["source"], "builtin_defaults")
        r = self._src(none, {"buyscore_config.json": _f("n", CFG), "buyscore_config.default.json": None})
        self.assertEqual((r["source"], r["source_detail"]["reason"]), ("unknown", "config_file_appeared"))
        r = self._src(none, {"buyscore_config.json": None, "buyscore_config.default.json": _f("n", CFG)})
        self.assertEqual((r["source"], r["source_detail"]["reason"]), ("unknown", "config_file_appeared"))

    def test_no_captured_cfg(self):
        none = {"buyscore_config.json": None, "buyscore_config.default.json": None}
        r = G.determine_config_source(none, none, None, None, [])
        self.assertEqual(r["source_detail"]["reason"], "no_captured_cfg")


class EntrypointRecordTests(unittest.TestCase):
    """X32・X44・X48・X51。"""

    def test_freeze_sha_is_verified(self):                           # X32
        self.assertIsNotNone(G._freeze_config())
        with mock.patch.object(G, "FREEZE_CONFIG_SHA256", "0" * 64):
            self.assertIsNone(G._freeze_config())

    def _run(self, **kw):
        from tests.unit import g3_fake_legacy as F
        from tests.unit.test_g3_entrypoint import ODDS, _boats, _gen_patterns, _upset
        from shadow.g3_capture import ApplyBuyscoreCapture
        F.FAKE_CFG.clear()
        F.FAKE_CFG.update(S.load_freeze_config())
        for k in F.MODE:
            F.MODE[k] = False
        ev = S.evaluation(upset_score=5.0, win_probs={1: 0.2, 2: 0.1, 3: 0.3, 4: 0.25})
        return G.evaluate_race_g3(
            evaluation=ev, upset=_upset(), legacy_boats=_boats(), odds_map=ODDS, weather=object(),
            buy_config=F.FAKE_CFG, evaluate_bets=F.evaluate_bets, generate_patterns=_gen_patterns,
            capture_factory=lambda: ApplyBuyscoreCapture(target=F.apply_buyscore),
            isolation_factory=S.iso_factory(), **kw), _boats()

    def test_boats_sha_and_target_status_and_reload_phase(self):     # X44・X48・X51
        rec, boats = self._run(legacy_target=[2, 3])
        self.assertEqual(rec["outcome"], G.OUTCOME_COMPARED, rec["conditions"])
        self.assertEqual(rec["inputs"]["legacy_boats_sha256"],
                         G._sha256_json([G._boat_dict(b) for b in boats]))
        self.assertEqual(rec["target"]["status"], "compared")
        reload_calls = [c for c in rec["isolation"]["release_calls"] if c["phase"] == "config_reload"]
        self.assertEqual([(c["function"], c["args"][0]) for c in reload_calls],
                         [("download_file", "buyscore_config.json")])


class VerifiedConditionTests(unittest.TestCase):
    """X05・X06・X25〜X29：verified は post_check の各項目と後片付けの失敗を反映する。"""

    def test_each_post_check_item_lowers_verified(self):
        keys = ("release_stubs_intact", "release_module_identity", "legacy_uses_patched_module",
                "hook_active", "env_still_scrubbed", "cwd_is_workdir")
        original = G3Isolation._post_checks
        for key in keys:
            def patched(self, _key=key):
                original(self)
                self.post_check[_key] = False
            with mock.patch.object(G3Isolation, "_post_checks", patched), G3Isolation({}) as iso:
                self.assertTrue(iso.established)
            self.assertFalse(iso.verified, key)
        with G3Isolation({}) as ok:
            pass
        self.assertEqual(set(ok.post_check), set(keys))
        self.assertTrue(ok.verified)

    def test_teardown_error_after_post_check_lowers_verified(self):
        with mock.patch.object(G3Isolation, "_snapshot", side_effect=[{}, RuntimeError("snap")]):
            with G3Isolation({}) as iso:
                pass
        self.assertTrue(all(iso.post_check.values()))
        self.assertIn("teardown_error", iso.errors)                  # G-10：固定ラベル（teardown は記録して verified を下げる）
        self.assertNotIn("teardown_error:RuntimeError", iso.errors)   # クラス名は公開しない（意味を維持）
        self.assertFalse(any("RuntimeError" in e for e in iso.errors))
        self.assertFalse(iso.verified)


class WriteModeTests(unittest.TestCase):
    """X02・X03・X04・X16〜X19・F14b：外への書き込みの各形。"""

    def setUp(self):
        self.out = Path(tempfile.mkdtemp(prefix="g3_b01_outside_"))
        self.addCleanup(shutil.rmtree, self.out, True)
        (self.out / "existing.txt").write_text("keep")
        (self.out / "sub").mkdir()

    def _blocked(self, fn):
        with self.assertRaises(G3IsolationBlocked):
            fn()

    def test_write_modes_and_path_events(self):
        o = self.out
        with G3Isolation({}) as iso:
            Path("inside.txt").write_text("i")
            self._blocked(lambda: open(o / "existing.txt", "a").close())           # X02
            self._blocked(lambda: open(o / "existing.txt", "r+").close())          # X03
            self._blocked(lambda: os.open(str(o / "new.txt"), os.O_WRONLY | os.O_CREAT))  # X04
            self._blocked(lambda: os.mkdir(str(o / "newdir")))                     # X16
            self._blocked(lambda: os.symlink("inside.txt", str(o / "ln")))          # X17
            self._blocked(lambda: os.link("inside.txt", str(o / "hl")))             # X17
            self._blocked(lambda: os.truncate(str(o / "existing.txt"), 0))          # X18
            self._blocked(lambda: os.chmod(str(o / "existing.txt"), 0o600))         # X18
            self._blocked(lambda: shutil.rmtree(str(o / "sub")))                    # X19
            self._blocked(lambda: shutil.move("inside.txt", str(o / "moved.txt")))  # X19
            sibling = f"{iso.workdir}_sibling.txt"                                  # F14b（前方一致）
            self._blocked(lambda: open(sibling, "w").close())
        self.assertEqual((o / "existing.txt").read_text(), "keep")
        self.assertTrue((o / "sub").is_dir())
        for name in ("new.txt", "newdir", "ln", "hl", "moved.txt"):
            self.assertFalse((o / name).exists(), name)
        self.assertFalse(Path(sibling).exists())
        events = [e["event"] for e in iso.summary()["blocked_events"]]
        for ev in ("open", "os.mkdir", "os.symlink", "os.link", "os.truncate", "os.chmod",
                   "shutil.rmtree", "shutil.move"):
            self.assertIn(ev, events)


class ProcessAndNetworkTests(unittest.TestCase):
    """X21・X23・X24。exec・fork・spawn は子のプロセスで試す。"""

    def test_smtplib_connect_recorded(self):                         # X21
        with G3Isolation({}) as iso:
            with self.assertRaises(G3IsolationBlocked):
                smtplib.SMTP("127.0.0.1", 9, timeout=0.5)
        self.assertEqual(iso.summary()["blocked_events"][0]["event"], "smtplib.connect")

    def test_exec_fork_spawn_blocked_in_child(self):                 # X23・X24
        code = textwrap.dedent("""
            import os, sys
            from shadow.g3_isolation import G3Isolation, G3IsolationBlocked
            res = {}
            with G3Isolation({}) as iso:
                for name, fn in (
                    ("posix_spawn", lambda: os.posix_spawn("/bin/true", ["true"], dict(os.environ))),
                    ("spawnv", lambda: os.spawnv(os.P_WAIT, "/bin/true", ["true"])),
                    ("fork", lambda: os.fork()),
                    ("execv", lambda: os.execv("/bin/true", ["true"])),
                ):
                    try:
                        r = fn()
                        if name == "fork" and r == 0:
                            os._exit(0)
                        res[name] = "NOT_BLOCKED"
                    except G3IsolationBlocked:
                        res[name] = "BLOCKED"
            print("RESULT", sorted(res.items()), sorted({e["event"] for e in iso.summary()["blocked_events"]}))
        """)
        env = {k: v for k, v in os.environ.items() if not GI.env_name_is_sensitive(k)}
        env["PYTHONPATH"] = os.pathsep.join([str(S.REBUILD_DIR), str(S.REPO_DIR)])
        p = subprocess.run([sys.executable, "-B", "-c", code], cwd=str(S.REBUILD_DIR), env=env,
                           capture_output=True, text=True, timeout=60)
        line = [x for x in p.stdout.splitlines() if x.startswith("RESULT")]
        self.assertEqual(len(line), 1, (p.stdout, p.stderr[-2000:]))
        self.assertIn("('execv', 'BLOCKED')", line[0])
        self.assertIn("('fork', 'BLOCKED')", line[0])
        self.assertIn("('posix_spawn', 'BLOCKED')", line[0])
        self.assertIn("('spawnv', 'BLOCKED')", line[0])
        for ev in ("os.exec", "os.fork", "os.posix_spawn"):
            self.assertIn(ev, line[0])


class SelfTestItemTests(unittest.TestCase):
    """F21d〜f・X40：自己試験の requests・release_stub・env_scrubbed の各項目。"""

    def test_requests_item_fails_without_network_layer(self):        # F21d
        with mock.patch.object(GI, "_NETWORK_EVENTS", frozenset()), G3Isolation({}) as iso:
            pass
        self.assertFalse(iso.self_test["requests"])

    def test_requests_missing_fails(self):                           # X40
        with mock.patch.dict(sys.modules, {"requests": None}), G3Isolation({}) as iso:
            self.assertFalse(iso.established)
        self.assertFalse(iso.self_test["requests"])

    def test_release_stub_item(self):                                # F21e
        with mock.patch.object(GI, "_STUB_RETURNS", {}), G3Isolation({}) as iso:
            self.assertFalse(iso.established)
        self.assertFalse(iso.self_test["release_stub"])

    def test_env_scrubbed_item(self):                                # F21f
        original = G3Isolation._scrub_env

        def leaky(self):
            original(self)
            os.environ["GITHUB_TOKEN"] = S.FAKE_TOKEN_ENV["GITHUB_TOKEN"]
        with S.fake_token_env():
            with mock.patch.object(G3Isolation, "_scrub_env", leaky), G3Isolation({}) as iso:
                self.assertFalse(iso.established)
        self.assertFalse(iso.self_test["env_scrubbed"])


class EnvAndLifecycleTests(unittest.TestCase):
    """X37・X38・X42・X43・X53・X56。"""

    def test_home_and_tempdir_inside_workdir(self):                  # X37・X38
        with G3Isolation({}) as iso:
            root = os.path.realpath(str(iso.workdir))
            self.assertTrue(os.path.realpath(os.environ["HOME"]).startswith(root + os.sep))
            self.assertTrue(os.path.realpath(tempfile.gettempdir()).startswith(root + os.sep))
            with tempfile.NamedTemporaryFile("w") as fh:          # 中に作られる（止まらない）
                fh.write("x")
        self.assertTrue(iso.verified, iso.summary()["blocked_events"])

    def test_sensitive_name_patterns(self):                          # X42・X43
        names = {"MY_SERVICE_TOKEN": "v", "SMTPHOST": "v", "MAIL_FROM": "v", "GMAIL_USER": "v",
                 "LINE_CHANNEL": "v", "SOME_API_KEY": "v", "DB_PASSWORD": "v"}
        with mock.patch.dict(os.environ, names), G3Isolation({}) as iso:
            for n in names:
                self.assertNotIn(n, os.environ)
        for n in names:
            self.assertIn(n, iso.env_removed_names)

    def test_single_use(self):                                       # X53
        iso = G3Isolation({})
        with iso:
            pass
        with self.assertRaises(RuntimeError):
            with iso:
                pass

    def test_outside_probe_dir_removed(self):                        # X56
        with G3Isolation({}) as iso:
            probe = iso._outside_probe_dir
            self.assertTrue(probe.exists())
        self.assertFalse(probe.exists())


class PostCheckRealConditionTests(unittest.TestCase):
    """X26〜X29：終了時の確認の各項目が、実際の条件の崩れを検出する。"""

    def test_env_reappears(self):                                    # X26
        with mock.patch.dict(os.environ):  # 試験のプロセスの環境変数を元に戻す
            with G3Isolation({}) as iso:
                os.environ["GITHUB_TOKEN"] = S.FAKE_TOKEN_ENV["GITHUB_TOKEN"]
        self.assertFalse(iso.post_check["env_still_scrubbed"])
        self.assertFalse(iso.verified)

    def test_cwd_moved(self):                                        # X27
        other = Path(tempfile.mkdtemp(prefix="g3_b01_cwd_"))
        self.addCleanup(shutil.rmtree, other, True)
        cwd = os.getcwd()
        with G3Isolation({}) as iso:
            os.chdir(other)
        self.assertEqual(os.getcwd(), cwd)
        self.assertFalse(iso.post_check["cwd_is_workdir"])
        self.assertFalse(iso.verified)

    def test_release_module_swapped(self):                           # X28
        import types
        real = sys.modules["x_release_storage"]
        try:
            with G3Isolation({}) as iso:
                sys.modules["x_release_storage"] = types.ModuleType("x_release_storage")
        finally:
            sys.modules["x_release_storage"] = real
        self.assertFalse(iso.post_check["release_module_identity"])
        self.assertFalse(iso.verified)

    def test_legacy_reference_swapped(self):                         # X29
        import types
        import x_buyscore
        real = x_buyscore.x_release_storage
        try:
            with G3Isolation({}) as iso:
                x_buyscore.x_release_storage = types.ModuleType("x_release_storage_other")
        finally:
            x_buyscore.x_release_storage = real
        self.assertFalse(iso.post_check["legacy_uses_patched_module"])
        self.assertFalse(iso.verified)


class RedactionTests(unittest.TestCase):
    """B-04：URL・認証情報の伏せ字。"""

    def test_redact_forms(self):
        cases = [
            "https://hooks.example.invalid/services/T000/B000/SECRETPATH1",
            "https://user:p@ss/wo@rd@example.invalid/x",
            "https://example.invalid/a;key=SECRETPARAM",
            "https://example.invalid/x?token=SECRETQ#SECRETFRAG",
            "git push https://SECRETTOK@github.com/o/r.git main",
            "Authorization: Bearer SECRETBEARER",
            "token=SECRETKV password:SECRETPW",
            "ghp_SECRETGHP0123456789 and github_pat_SECRETPAT_0123456789",
        ]
        for c in cases:
            out = GI._redact(c)
            self.assertNotIn("SECRET", out, (c, out))
        self.assertEqual(GI._redact("https://api.github.com:443/repos/x?y=1"), "https://api.github.com:443/***")
        self.assertEqual(GI._redact("buyscore_config.json"), "buyscore_config.json")  # ファイル名は残す

    def test_release_stub_args_redacted(self):
        import x_release_storage as R
        with G3Isolation({}) as iso:
            R.download_file("https://example.invalid/x?token=SECRETARG", "ghp_SECRETGHP0123456789")
        self.assertNotIn("SECRET", repr(iso.summary()["release_calls"]))


class RedactionC07Tests(unittest.TestCase):
    """再監査 C-06・C-07・I04：伏せ字の形の追加、末尾を残す切り方、差し替えた関数のキーワード引数。"""

    def test_more_forms(self):
        cases = ["token = SECRET1", "token SECRET2", "Authorization: token SECRET3", '{"token": "SECRET4"}',
                 "password SECRET5", "x-api-key: SECRET6", "api-key=SECRET7", "access_key=SECRET8",
                 "pwd=SECRET9", "secret_key: SECRETA", "cookie: session=SECRETB", "user:SECRETC@host.example",
                 "glpat-SECRETD0123456789", "sk-SECRETE0123456789abcdef", "AKIASECRETFGHIJKLMNO"]
        for c in cases:
            self.assertNotIn("SECRET", GI.redact_text(c), c)
        self.assertEqual(GI.redact_text("model_all.pkl"), "model_all.pkl")

    def test_tail_kept(self):
        # C-06 廃止：keep="tail" の「末尾保持で例外の最終行を残す」保証（旧 out.endswith("FINAL")）は廃止。
        # 末尾側の秘密はマスクされ得るため最終行の生残は保証しない。以下はマスキングと切り出しを分けて検証する。
        # (1) マスキングは弱めない：末尾側に秘密があっても keep="tail" の出力に生の秘密は残らない
        text = "head " + "x" * 500 + " token=SECRETZ FINAL"
        out_tail = GI.redact_text(text, 50, "tail")
        self.assertNotIn("SECRET", out_tail)
        self.assertNotIn("SECRETZ", out_tail)
        # (2) 切り出し動作（マスキングと分離）：秘密を含まない入力で keep="tail" は末尾 limit 文字を返す（既存仕様 text[-limit:]）
        plain = "A" * 10 + "B" * 50
        self.assertEqual(GI.redact_text(plain, 50, "tail"), plain[-50:])
        self.assertTrue(GI.redact_text(plain, 50, "tail").endswith("B" * 50))
        # (3) 既定(head)は先頭 limit 文字（既存仕様 text[:limit]）
        self.assertEqual(GI.redact_text(plain, 50), plain[:50])
        self.assertTrue(GI.redact_text(text, 50).startswith("head"))

    def test_stub_kwargs_redacted(self):                             # I04
        import x_release_storage as R
        with G3Isolation({}) as iso:
            R.download_file("a.json", "a.json", url="https://example.invalid/x?token=SECRETKW",
                            header="Bearer SECRETBEARER")
        call = iso.summary()["release_calls"][0]
        self.assertNotIn("SECRET", repr(call))
        self.assertEqual(set(call["kwargs"]), {"url", "header"})

    def test_new_sensitive_names_removed(self):
        names = {"GMAIL_APP_PASS": "v", "MY_PAT": "v", "SLACK_WEBHOOK_URL": "v", "SOME_ACCESS_KEY_ID": "v"}
        with mock.patch.dict(os.environ, names), G3Isolation({}) as iso:
            for n in names:
                self.assertNotIn(n, os.environ)
        for n in names:
            self.assertIn(n, iso.env_removed_names)
            self.assertTrue(GI.env_name_must_refuse(n), n)
        for n in ("GITHUB_SHA", "RUNNER_OS", "GITHUB_WORKSPACE", "ImageOS", "XDG_SESSION_ID", "XDG_SESSION_TYPE",
                  "DBUS_SESSION_BUS_ADDRESS"):
            self.assertFalse(GI.env_name_must_refuse(n), n)
        for n in ("MY_SESSION_TOKEN", "SESSION_SECRET", "GITHUB_TOKEN"):
            self.assertTrue(GI.env_name_must_refuse(n), n)

    def test_d05_forms(self):                                        # D-05
        for c in ("Authorization: Basic U0VDUkVUX0I2NA==", "Authorization: Bearer SECRETB", "X_TOKEN=SECRETS",
                  "aws_secret_access_key = SECRETA", "GH_PAT=SECRETP", "MY_PAT: SECRETQ"):
            out = GI.redact_text(c)
            self.assertNotIn("SECRET", out, (c, out))
            self.assertNotIn("U0VDUkVU", out, (c, out))
        for keep in ("buyscore_config.json", "model_all.pkl", 'File "/x/y.py", line 3', "passthrough_reason"):
            self.assertEqual(GI.redact_text(keep), keep)

    def test_e04_forms(self):                                        # E-04
        cases = ['password="abc SECRETW1 ghi"', "api_key='SECRETW2 x'", '{"access_token": "SECRETW3"}',
                 "{'client_secret': b'SECRETW4'}", 'auth = "SECRETW5 two"', r'password="a\"SECRETW6"',
                 "-----BEGIN RSA PRIVATE KEY-----\nMIISECRETW7\nmore", "-----BEGIN PRIVATE KEY-----\nSECRETW8",
                 "eyJhbGciOiJ.eyJzdWIiOiJ.SECRETW9sig", "GMAIL_APP_PASS=abcd SECRETWA ijkl",
                 "db_password: one two SECRETWB"]
        for c in cases:
            out = GI.redact_text(c)
            self.assertNotIn("SECRET", out, (c, out))
        self.assertEqual(GI.redact_text('auth = "x y" tail'), "*** tail")   # 引用符の後ろは残す
        for keep in ('File "/x/y.py", line 3', "passthrough_reason", "race_number=3"):
            self.assertEqual(GI.redact_text(keep), keep)

    def test_rn05_exemption_not_widened(self):                       # E-07（R-n05）
        for n in ("XDG_SESSION_ID", "XDG_SESSION_TYPE", "XDG_SEAT", "XDG_VTNR", "XDG_SESSION_CLASS",
                  "DBUS_SESSION_BUS_ADDRESS"):
            self.assertFalse(GI.env_name_must_refuse(n), n)
        for n in ("XDG_SESSION_TOKEN", "XDG_SESSION_KEY", "XDG_SEAT_SECRET", "XDG_CONFIG_SECRET",
                  "XDG_RUNTIME_TOKEN", "DBUS_SESSION_TOKEN", "XDG_SESSIONTOKEN"):
            self.assertTrue(GI.env_name_must_refuse(n), n)


class RedactionF10Tests(unittest.TestCase):
    """F-10：伏せ字の漏れの修正（第3版）。実際の攻撃ではなく、合成した文字列での確認。"""

    def test_more_forms(self):
        cases = {
            "pgp_block": "-----BEGIN PGP PRIVATE KEY BLOCK-----\nlQOSECRETKEYMATERIAL\n-----END PGP PRIVATE KEY BLOCK-----",
            "token_space_value": "token = abcd efgh SECRETX",
            "channel_token": "line_channel_access_token SECRETY1234567890",
            "credential_eq": 'credential="a b SECRETZ"',
            "client_secret": "client_secret=abc def SECRETW",
            "creds": "aws_creds: AKIA SECRETA trailing",
        }
        for name, c in cases.items():
            out = GI.redact_text(c)
            self.assertNotIn("SECRET", out, (name, out))
        # 残すべきもの（値の欄・普通の記録）は変えない
        for keep in ("race_number=3", "passthrough_reason", 'File "/x/y.py", line 3', "eval_id=20261107_01_01"):
            self.assertEqual(GI.redact_text(keep), keep)

    def test_pgp_block_across_lines(self):                           # 複数行の秘密鍵
        text = "before\n-----BEGIN PGP PRIVATE KEY BLOCK-----\nLINE1SECRET\nLINE2SECRET\n-----END PGP PRIVATE KEY BLOCK-----\nafter"
        out = "\n".join(GI.redact_lines(text.split("\n")))
        self.assertNotIn("SECRET", out)
        self.assertIn("before", out)
        self.assertIn("after", out)

    def test_env_name_pwd_and_creds(self):                          # 名前の型：_PWD・CREDS（PWD・OLDPWD は除く）
        for n in ("DB_PWD", "MYSQL_PWD", "CLIENT_CREDS", "AWS_CREDS"):
            self.assertTrue(GI.env_name_must_refuse(n), n)
            self.assertTrue(GI.env_name_is_sensitive(n), n)
        for n in ("PWD", "OLDPWD", "RUNNER_OS", "GITHUB_SHA"):
            self.assertFalse(GI.env_name_must_refuse(n), n)
        # 止める名前は、必ず外す側にも入る（食い違いをなくす）
        for n in ("XDG_SESSION_TOKEN", "GH_PAT", "DB_PWD"):
            self.assertTrue(GI.env_name_is_sensitive(n), n)

    def test_redact_free_text(self):                                # 記録の自由な文だけに伏せ字をかける
        rec = {"eval_id": "e0", "sha256": "a" * 64, "error": "RuntimeError: token=SECRET1",
               "prep_trace_tail": "line\ntoken=SECRET2\n", "count_candidate_checks": {"strace_evidence": "ok"},
               "note": "individual_results_only", "child_stderr_tail": "boom SECRET3 secret=SECRET4"}
        out = GI.redact_free_text(rec)
        self.assertEqual(out["eval_id"], "e0")                       # 値の欄は変えない
        self.assertEqual(out["sha256"], "a" * 64)
        self.assertEqual(out["count_candidate_checks"]["strace_evidence"], "ok")
        self.assertNotIn("SECRET", json.dumps(out["error"]))
        self.assertNotIn("SECRET", json.dumps(out["prep_trace_tail"]))
        self.assertNotIn("SECRET", json.dumps(out["child_stderr_tail"]))

    def test_redacting_stream(self):                                # 標準出力・標準エラーの伏せ字
        import io

        buf = io.StringIO()
        st = GI.RedactingStream(buf)
        st.write("normal line\n")
        st.write("token=SECRET1 more\n")
        st.write("-----BEGIN RSA PRIVATE KEY-----\n")
        st.write("SECRETKEYBODY\n")
        st.write("-----END RSA PRIVATE KEY-----\n")
        st.write("tail secret=SECRET2\n")
        st.flush()
        out = buf.getvalue()
        self.assertIn("normal line", out)
        self.assertNotIn("SECRET", out)


class EvaluateBetsArgsTests(unittest.TestCase):
    """B-05：Legacy に渡した weather・venue_num・race_number・race_date などの記録。"""

    _run = EntrypointRecordTests._run

    def test_evaluate_bets_args_recorded(self):
        rec, _boats = self._run()
        a = rec["evaluate_bets_args"]
        ev = S.evaluation(upset_score=5.0, win_probs={1: 0.2, 2: 0.1, 3: 0.3, 4: 0.25})
        self.assertEqual((a["venue_num"], a["race_number"], a["race_date"]),
                         (ev.venue_num, ev.race_number, ev.race_date))
        self.assertIs(a["has_exhibition"], False)
        self.assertEqual(a["weather_type"], "object")
        self.assertEqual(a["target_lanes"], list(rec["ml_bonus"]["target_post"]))
        self.assertEqual(a["upset_score"], rec["ml_bonus"]["score_post"])
        for k in ("bankroll", "venue_name"):
            self.assertIn(k, a)
        self.assertEqual(a["matches_record"], {"patterns": True, "ml_probs": True, "odds_map": True, "boats": True})

    def test_matches_record_detects_difference(self):                # N29・N30
        """記録した値と、実際に Legacy に渡した値の SHA-256 が違えば matches_record が False になる。"""
        from tests.unit import g3_fake_legacy as F
        from tests.unit.test_g3_entrypoint import ODDS, _boats, _gen_patterns, _upset
        from shadow.g3_capture import ApplyBuyscoreCapture

        class Unstable:
            n = 0

            def __str__(self):
                Unstable.n += 1
                return f"u{Unstable.n}"

            def __deepcopy__(self, memo):
                return self

        @dataclasses.dataclass
        class Boat:
            lane: int
            tag: object

        def gen(target, score):
            p = _gen_patterns(target, score)
            p["unstable"] = Unstable()
            return p
        F.FAKE_CFG.clear()
        F.FAKE_CFG.update(S.load_freeze_config())
        for k in F.MODE:
            F.MODE[k] = False
        ev = S.evaluation(upset_score=5.0, win_probs={1: 0.2, 2: 0.1, 3: 0.3, 4: 0.25})
        boats = [Boat(lane=b.lane, tag=Unstable()) for b in _boats()]
        rec = G.evaluate_race_g3(
            evaluation=ev, upset=_upset(), legacy_boats=boats, odds_map=ODDS, weather=None,
            buy_config=F.FAKE_CFG, evaluate_bets=F.evaluate_bets, generate_patterns=gen,
            capture_factory=lambda: ApplyBuyscoreCapture(target=F.apply_buyscore),
            isolation_factory=S.iso_factory())
        m = rec["evaluate_bets_args"]["matches_record"]
        self.assertIs(m["patterns"], False)
        self.assertIs(m["boats"], False)
        self.assertIs(m["odds_map"], True)
        self.assertIs(m["ml_probs"], True)

    def test_weather_dataclass_recorded(self):                       # G01
        from tests.unit import g3_fake_legacy as F
        from tests.unit.test_g3_entrypoint import ODDS, _boats, _gen_patterns, _upset
        from shadow.g3_capture import ApplyBuyscoreCapture

        @dataclasses.dataclass
        class W:
            wind: float = 3.5
            wave: int = 2
        F.FAKE_CFG.clear()
        F.FAKE_CFG.update(S.load_freeze_config())
        for k in F.MODE:
            F.MODE[k] = False
        ev = S.evaluation(upset_score=5.0, win_probs={1: 0.2, 2: 0.1, 3: 0.3, 4: 0.25})
        rec = G.evaluate_race_g3(
            evaluation=ev, upset=_upset(), legacy_boats=_boats(), odds_map=ODDS, weather=W(),
            buy_config=F.FAKE_CFG, evaluate_bets=F.evaluate_bets, generate_patterns=_gen_patterns,
            capture_factory=lambda: ApplyBuyscoreCapture(target=F.apply_buyscore),
            isolation_factory=S.iso_factory())
        self.assertEqual(rec["evaluate_bets_args"]["weather"], {"wind": 3.5, "wave": 2})
        self.assertEqual(rec["evaluate_bets_args"]["weather_type"], "W")


if __name__ == "__main__":
    unittest.main()
