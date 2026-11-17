"""正式G3（B案）N-01：G3 の診断の経路で Release・ネットワークに触れない境界（shadow.g3_isolation）。

試験の方針
- トークンがある環境：偽のトークン（g3_support.FAKE_TOKEN_ENV。実在しないリポジトリ名）を環境に置いて実行する。
- 実際の Legacy の3経路（設定の取得、学習用ログの取得・アップロード、成績の取得）を呼び、
  層1（Release の関数の差し替え）で止まり、層2 が止めた件数が0件であることを確かめる。
- 層2 だけで止まることの確認（否定の対照）：境界の中で、差し替える前の本物の x_release_storage の関数を
  偽のトークンつきで呼び、ソケットの段階で止まって記録されることを確かめる。
  宛先は層3 で接続できない代理（127.0.0.1:9）になっているので、層2 が働かなくても外へは出ない。
- 自己試験の失敗・入れ子・後片付け（環境変数・作業ディレクトリ・差し替えの復元）。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import unittest
from pathlib import Path

from shadow import g3_isolation as GI
from shadow.g3_isolation import G3Isolation, G3IsolationBlocked
from tests.unit import g3_support as S


class SelfTestAndRestoreTests(unittest.TestCase):
    def test_self_test_all_layers_and_restore(self):
        import x_release_storage as R

        original_download = R.download_file
        cwd = os.getcwd()
        with S.fake_token_env():
            env_before = dict(os.environ)
            with G3Isolation({"buyscore_config.json": S.CONFIG_FIXTURE}) as iso:
                self.assertTrue(iso.established, iso.errors)
                self.assertEqual(set(iso.self_test), {
                    "socket_connect", "getaddrinfo", "subprocess", "os_system", "write_outside",
                    "remove_outside", "requests", "write_inside", "release_stub", "env_scrubbed"})
                self.assertTrue(all(iso.self_test.values()), iso.self_test)
                self.assertNotIn("GITHUB_TOKEN", os.environ)
                self.assertNotIn("GH_TOKEN", os.environ)
                self.assertNotIn("GITHUB_REPOSITORY", os.environ)
                self.assertNotEqual(os.getcwd(), cwd)
                self.assertTrue(Path("buyscore_config.json").exists())
                self.assertIsNot(R.download_file, original_download)
                workdir = iso.workdir
            self.assertEqual(dict(os.environ), env_before)  # 環境変数は元どおり
        self.assertEqual(os.getcwd(), cwd)
        self.assertIs(R.download_file, original_download)
        self.assertFalse(workdir.exists())  # N-08：作業フォルダを片付ける
        s = iso.summary()
        self.assertTrue(s["verified"], s)
        self.assertEqual(s["release_calls"], [])  # 自己試験の呼び出しは実行の記録に入れない
        self.assertEqual(s["seeded"]["buyscore_config.json"], S.CONFIG_SHA256)
        self.assertIn("GITHUB_TOKEN", s["env_removed_names"])

    def test_nested_isolation_is_not_established(self):
        with G3Isolation({}) as outer:
            self.assertTrue(outer.established)
            with G3Isolation({}) as inner:
                self.assertFalse(inner.established)
                self.assertIn("another_isolation_active", inner.errors)
        self.assertTrue(outer.summary()["verified"])
        self.assertFalse(inner.summary()["verified"])

    def test_bad_seed_name_is_not_established(self):
        with G3Isolation({"../escape.json": b"{}"}) as iso:
            self.assertFalse(iso.established)
        self.assertFalse(iso.verified)
        self.assertIn("setup_error", iso.errors)                     # G-10：クラス名を付けず固定ラベル

    def test_missing_seed_file_is_not_established(self):
        with G3Isolation({"model_all.pkl": S.REPO_DIR / "does_not_exist.pkl"}) as iso:
            self.assertFalse(iso.established)

    def test_hook_inactive_outside(self):
        with G3Isolation({}):
            pass
        # 境界の外では何も止めない（ローカルの接続を試み、境界の例外にならないこと）
        s = socket.socket()
        s.settimeout(0.2)
        try:
            s.connect(("127.0.0.1", 9))
        except G3IsolationBlocked:
            self.fail("hook is active outside the isolation")
        except OSError:
            pass
        finally:
            s.close()


class RealLegacyReleasePathTests(unittest.TestCase):
    """実際の Legacy の Release の3経路（x_buyscore 47・632〜635、notify_arashi 2765）。"""

    def test_three_paths_blocked_with_token(self):
        import notify_arashi as N
        import x_buyscore as X

        with S.fake_token_env(), G3Isolation({"buyscore_config.json": S.CONFIG_FIXTURE}) as iso:
            self.assertTrue(iso.established, iso.errors)
            cfg = X.load_config()                       # 設定の取得（_ensure_config_file）
            X.save_buyscore_log("v", "1", "20261113", [{"combo": "1-2-3"}], None, {}, "通常")  # ログ
            mult = N._get_bet_multiplier()              # 成績の取得
            log_written = Path("buyscore_log.jsonl").read_text(encoding="utf-8")
        s = iso.summary()
        self.assertTrue(s["verified"], s)
        self.assertEqual(s["blocked_event_count"], 0, s["blocked_events"])
        self.assertEqual(s["network_attempts_observed"], 0)
        self.assertEqual(cfg, S.load_freeze_config())   # Release が使えないので、置いた設定を使った
        self.assertEqual(mult, 1.0)                      # 成績のファイルが無いときの既存の分岐
        self.assertIn('"combo": "1-2-3"', log_written)   # ログは作業フォルダにだけ書かれた
        got = [(c["function"], c["args"][0]) for c in s["release_calls"]]
        self.assertEqual(got, [("download_file", "buyscore_config.json"),
                               ("download_file", "buyscore_log.jsonl"),
                               ("upload_file", "buyscore_log.jsonl"),
                               ("download_file", "hit_record.csv")])
        self.assertTrue(all(c["blocked"] for c in s["release_calls"]))
        self.assertIn("buyscore_log.jsonl", s["workdir_files_end"])

    def test_release_create_path_is_also_stubbed(self):
        """_get_or_create_release（404 なら Release を作る POST）も差し替えの対象。"""
        import x_release_storage as R

        with S.fake_token_env(), G3Isolation({}) as iso:
            self.assertIsNone(R._get_or_create_release())
            self.assertIsNone(R._upload_asset({}, "x", "x"))
            self.assertFalse(R._delete_asset(1))
            self.assertFalse(R.is_available())
        s = iso.summary()
        self.assertTrue(s["verified"], s)
        self.assertEqual([c["function"] for c in s["release_calls"]],
                         ["_get_or_create_release", "_upload_asset", "_delete_asset", "is_available"])
        for name in ("download_file", "upload_file", "restore_from_backup", "_get_or_create_release",
                     "_download_asset", "_upload_asset", "_delete_asset", "_headers", "_get_token"):
            self.assertIn(f"x_release_storage.{name}", s["patched"])

    def test_network_layer_alone_stops_original_release_code(self):
        """否定の対照：層1 を通らず、本物の download_file を偽のトークンつきで呼ぶと、層2 が止める。"""
        import x_release_storage as R

        with S.fake_token_env(), G3Isolation({}) as iso:
            self.assertTrue(iso.established)
            os.environ["GITHUB_TOKEN"] = S.FAKE_TOKEN_ENV["GITHUB_TOKEN"]      # 層3 をわざと外す
            os.environ["GITHUB_REPOSITORY"] = S.FAKE_TOKEN_ENV["GITHUB_REPOSITORY"]
            for (o, n, v) in list(iso._saved_attrs):                              # 層1 をわざと外す
                if o is R:
                    setattr(o, n, v)   # 本物の関数と本物の URL（宛先は層3 の接続できない代理）
            self.assertEqual(R.API_BASE, "https://api.github.com")
            original = R.download_file
            ok = original("buyscore_config.json", "buyscore_config.json")
            del os.environ["GITHUB_TOKEN"]
            del os.environ["GITHUB_REPOSITORY"]
        s = iso.summary()
        self.assertFalse(ok)
        self.assertGreaterEqual(s["network_attempts_observed"], 1, s["blocked_events"])
        self.assertFalse(s["verified"])                          # 想定外の経路として違反になる
        self.assertFalse(s["post_check"]["release_stubs_intact"])  # 差し替えを外したことも検出する


class ProcessAndWriteTests(unittest.TestCase):
    def test_os_system_git_push_is_blocked(self):
        """notify_arashi.update_fan_files の os.system("git push ...") の形。"""
        with S.fake_token_env(), G3Isolation({}) as iso:
            with self.assertRaises(G3IsolationBlocked):
                os.system("git push https://example.invalid/none.git main")
        s = iso.summary()
        self.assertFalse(s["verified"])
        self.assertEqual([e["kind"] for e in s["blocked_events"]], ["process"])

    def test_write_outside_blocked_inside_allowed(self):
        outside_dir = Path(tempfile.mkdtemp(prefix="g3_outside_test_"))  # 境界の外
        self.addCleanup(shutil.rmtree, outside_dir, True)
        with G3Isolation({}) as iso:
            Path("inside.txt").write_text("ok")
            with self.assertRaises(G3IsolationBlocked):
                (outside_dir / "g3_should_not_exist.txt").write_text("x")
            with self.assertRaises(G3IsolationBlocked):
                os.rename("inside.txt", str(outside_dir / "g3_should_not_exist2.txt"))
        self.assertFalse((outside_dir / "g3_should_not_exist.txt").exists())
        self.assertFalse((outside_dir / "g3_should_not_exist2.txt").exists())
        s = iso.summary()
        self.assertEqual([e["event"] for e in s["blocked_events"]], ["open", "os.rename"])
        self.assertIn("inside.txt", s["workdir_files_end"])

    def test_bytecode_not_written(self):
        with G3Isolation({}) as iso:
            self.assertTrue(sys.dont_write_bytecode)
        self.assertTrue(iso.verified)


class HardeningTests(unittest.TestCase):
    """監査 A-03（生き残った変異）・A-06（伏せ字）・A-08（後片付け）。"""

    def _outside(self):
        d = Path(tempfile.mkdtemp(prefix="g3_outside_test_"))
        self.addCleanup(shutil.rmtree, d, True)
        return d

    def test_self_test_detects_ineffective_layer(self):          # M06・M21
        from unittest import mock
        with mock.patch.object(GI, "_NETWORK_EVENTS", frozenset()), G3Isolation({}) as iso:
            self.assertFalse(iso.established)
        self.assertFalse(iso.self_test["socket_connect"])
        self.assertFalse(iso.self_test["getaddrinfo"])
        self.assertIn("self_test_failed", iso.errors)               # G-10：クラス名を付けず固定ラベル
        self.assertFalse(iso.verified)
        with mock.patch.object(GI, "_PROCESS_EVENTS", frozenset()), G3Isolation({}) as iso2:
            self.assertFalse(iso2.established)
        self.assertFalse(iso2.self_test["os_system"])

    def test_self_test_requires_block_record(self):               # M21
        """止めても記録が残らない層は、自己試験で不合格にする。"""
        from unittest import mock
        with mock.patch.object(G3Isolation, "_record_block", lambda self, *a, **k: None), \
                G3Isolation({}) as iso:
            self.assertFalse(iso.established)
        self.assertFalse(iso.self_test["socket_connect"])
        self.assertFalse(iso.self_test["write_outside"])

    def test_symlink_to_outside_is_blocked(self):                 # M14
        outside = self._outside()
        with G3Isolation({}) as iso:
            os.symlink(str(outside), "link")      # 作成は中なので許す
            with self.assertRaises(G3IsolationBlocked):
                open("link/escape.txt", "w").close()
        self.assertFalse((outside / "escape.txt").exists())
        self.assertEqual([e["kind"] for e in iso.summary()["blocked_events"]], ["write_outside"])

    def test_sendto_sendmsg_bind_blocked(self):                   # M15
        with G3Isolation({}) as iso:
            for fn in (lambda s: s.sendto(b"x", ("127.0.0.1", 9)),
                       lambda s: s.sendmsg([b"x"], [], 0, ("127.0.0.1", 9)),
                       lambda s: s.bind(("127.0.0.1", 0))):
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                try:
                    with self.assertRaises(G3IsolationBlocked):
                        fn(s)
                finally:
                    s.close()
        self.assertEqual([e["event"] for e in iso.summary()["blocked_events"]],
                         ["socket.sendto", "socket.sendmsg", "socket.bind"])

    def test_from_import_copies_are_stubbed(self):                # M17
        import types
        import x_release_storage as R
        m = types.ModuleType("g3_fake_importer")
        m.download_file = R.download_file
        sys.modules["g3_fake_importer"] = m
        self.addCleanup(sys.modules.pop, "g3_fake_importer", None)
        original = R.download_file
        with G3Isolation({}) as iso:
            self.assertFalse(m.download_file("x.json", "x.json"))
        self.assertIs(m.download_file, original)
        self.assertIn("g3_fake_importer.download_file", iso.summary()["patched"])
        self.assertEqual([c["function"] for c in iso.summary()["release_calls"]], ["download_file"])

    def test_bytecode_flag_set_and_restored(self):                # M18
        saved = sys.dont_write_bytecode
        sys.dont_write_bytecode = False
        try:
            with G3Isolation({}):
                self.assertTrue(sys.dont_write_bytecode)
            self.assertFalse(sys.dont_write_bytecode)
        finally:
            sys.dont_write_bytecode = saved

    def test_proxy_and_release_urls_unreachable(self):            # M24・M27
        import x_release_storage as R
        api, up = R.API_BASE, R.UPLOAD_BASE
        with G3Isolation({}):
            for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy"):
                self.assertEqual(os.environ[k], "http://127.0.0.1:9")
            self.assertEqual(os.environ["NO_PROXY"], "")
            self.assertEqual((R.API_BASE, R.UPLOAD_BASE), ("http://127.0.0.1:9", "http://127.0.0.1:9"))
        self.assertEqual((R.API_BASE, R.UPLOAD_BASE), (api, up))

    def test_blocked_url_is_redacted(self):                       # A-06
        import urllib.request
        with G3Isolation({}) as iso:
            with self.assertRaises(G3IsolationBlocked):
                urllib.request.urlopen("https://user:pw-SECRET@example.invalid/p?token=SECRET123#frag")
            s = socket.socket()
            try:
                with self.assertRaises(G3IsolationBlocked):
                    s.connect(("127.0.0.1", 9))
            finally:
                s.close()
        text = repr(iso.summary())
        self.assertNotIn("SECRET123", text)
        self.assertNotIn("pw-SECRET", text)
        self.assertIn("https://example.invalid/***", text)
        connect = [e for e in iso.summary()["blocked_events"] if e["event"] == "socket.connect"]
        self.assertEqual(connect[0]["target"][0], ["127.0.0.1", 9])  # G-09 追従（detail→target）

    def test_teardown_restores_when_post_check_raises(self):      # A-08
        from unittest import mock
        import x_release_storage as R
        cwd, env, original = os.getcwd(), dict(os.environ), R.download_file
        with mock.patch.object(G3Isolation, "_post_checks", side_effect=RuntimeError("boom")):
            with S.fake_token_env():
                env = dict(os.environ)
                with G3Isolation({}) as iso:
                    workdir = iso.workdir
                self.assertEqual(dict(os.environ), env)
        self.assertEqual(os.getcwd(), cwd)
        self.assertIs(R.download_file, original)
        self.assertFalse(workdir.exists())
        self.assertFalse(iso.verified)
        self.assertIn("teardown_error", iso.errors)                  # G-10：クラス名を付けず固定ラベル

    def test_keyboard_interrupt_during_setup_restores(self):      # A-08
        from unittest import mock
        import x_release_storage as R
        cwd, original, saved_tmp = os.getcwd(), R.download_file, tempfile.tempdir
        iso = G3Isolation({})
        with S.fake_token_env():
            env = dict(os.environ)
            with mock.patch.object(G3Isolation, "_run_self_test", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    with iso:
                        self.fail("body must not run")
            self.assertEqual(dict(os.environ), env)
        self.assertEqual(os.getcwd(), cwd)
        self.assertIs(R.download_file, original)
        self.assertIs(tempfile.tempdir, saved_tmp)
        self.assertFalse(iso.workdir.exists())
        self.assertIs(GI._ACTIVE, None)

    def test_nested_failure_keeps_outer_tempdir(self):            # A-08
        saved_tmp = tempfile.tempdir
        with G3Isolation({}) as outer:
            inner_tmp = tempfile.tempdir
            with G3Isolation({}) as inner:
                self.assertFalse(inner.established)
            self.assertEqual(tempfile.tempdir, inner_tmp)
        self.assertIs(tempfile.tempdir, saved_tmp)
        self.assertTrue(outer.verified)


class UnclassifiedOutputKeysTests(unittest.TestCase):
    """G-30：公開レコードの fail-closed 検出器（F-10）。許可された形だけ受理し、生本文・クラス名・不正値を検出する。"""

    def test_allowed_shapes_return_empty(self):
        ok_cases = [
            {"child_stderr_tail": {"redacted": True, "len": 5, "lines": 1}},   # 正常 digest
            {"prep_trace_tail": None},                                          # None も可
            {"error": "unexpected_error"},                                      # 固定ラベル
            {"odds_meta": {"error": "empty"}},
            {"g3": {"config": {"source_detail": {"reason": "no_captured_cfg"}}}},
            {"g3": {"config": {"source_detail": {"workdir_config_parse_error_start": "not_a_dict"}}}},
            {"g3": {"config": {"source_detail": {"workdir_default_parse_error_start": "value_error"}}}},
            {"g3": {"capture": {"errors": ["unexpected_error", "cfg_not_captured"]}}},
            {"prep_isolation": {"errors": ["teardown_error", "self_test_failed"]}},
            {"c1_upset_pre": {"check_detail": {"upset": 1.0, "evaluation": 2.0}}},  # 構造化
            {"c3_post_values": {"check_detail": "assembly_check_only"}},
            {"c6_context": {"check_detail": {"diff_keys": ["upset_score", "match_index"]}}},
            {"context_keys": {"check_detail": {"keys": ["match_index", "upset_score"]}}},
            {"t13_capture_vs_return": {"check_detail": None}},
            {"g3": {"inputs": {"legacy_boats": [{"repr_type": "dict"}]}}},       # 許可された型名の位置
            {"g3": {"evaluate_bets_args": {"weather_type": "W"}}},
            {"message": None, "detail": None},
            {"stderr_count": 0, "stdout_bytes": 10, "trace_count": 2, "trace_id": "abcd1234"},
        ]
        for c in ok_cases:
            self.assertEqual(GI.unclassified_output_keys(c), [], c)

    def test_disallowed_shapes_detected(self):
        bad_cases = [
            ({"child_stderr_tail": "boom SECRET"}, "child_stderr_tail"),                 # 生本文
            ({"child_stderr_tail": {"redacted": True, "len": 5}}, "child_stderr_tail"),  # 不正 digest（キー欠落）
            ({"prep_trace_tail": {"redacted": False, "len": 1, "lines": 1}}, "prep_trace_tail"),  # redacted≠True
            ({"trace_tail": None}, "trace_tail"),                                        # 許可パス外の生出力キー
            ({"error": "RuntimeError"}, "error"),                                        # クラス名
            ({"joblib_prewarm_error": "KeyError"}, "joblib_prewarm_error"),              # ラベル規則違反
            ({"some_error": "x"}, "some_error"),                                         # 未知の _error 接尾
            ({"traceback_blob": "x"}, "traceback_blob"),                                 # 未知の stream 語キー
            ({"c7_pred_combo_once": {"check_detail": {"count": -1}}}, "c7_pred_combo_once.check_detail"),
            ({"context_keys": {"check_detail": {"keys": ["unknown_key"]}}}, "context_keys.check_detail"),
            ({"repr_type": "dict"}, "repr_type"),                                        # 許可パス外の型名キー
            ({"g3": {"inputs": {"legacy_boats": [{"repr_type": "has space"}]}}},
             "g3.inputs.legacy_boats[0].repr_type"),                                     # 不正な型名値
            ({"g3": {"capture": {"errors": ["weird_error"]}}}, "g3.capture.errors"),     # 未知ラベル
            ({"foo": {"errors": ["x"]}}, "foo.errors"),                                  # 未知パスの errors
            ({"message": "leak"}, "message"),                                            # message は None のみ
        ]
        for obj, expected in bad_cases:
            got = GI.unclassified_output_keys(obj)
            self.assertIn(expected, got, (obj, got))
        self.assertEqual(GI.unclassified_output_keys({"error": "race_not_in_programs"}), [])  # 対照：許可ラベル

    def test_result_sorted_and_deduped(self):
        got = GI.unclassified_output_keys({"b_error": "x", "a_error": "y", "error": "BadClass"})
        self.assertEqual(got, sorted(got))
        self.assertEqual(len(got), len(set(got)))


class ScanOutputTreeTests(unittest.TestCase):
    """G-30：collect 前ゲート scan_output_tree。正常木・no_target_races の整合・生本文混入・SHA 不一致。"""

    def _mk_out(self, eids):
        d = Path(tempfile.mkdtemp(prefix="g3_scan_"))
        self.addCleanup(shutil.rmtree, d, True)
        recs = [{"eval_id": e, "g3": {"outcome": "compared"}} for e in eids]
        rec_bytes = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs).encode("utf-8")
        (d / "records.jsonl").write_bytes(rec_bytes)
        races = d / "races"
        races.mkdir()
        nonce = "a" * 32
        for e in eids:
            (races / f"{e}.{nonce}.json").write_text(
                json.dumps({"eval_id": e, "nonce": nonce}, ensure_ascii=False), encoding="utf-8")
        summary = {
            "race_date": "20261107", "races": len(eids), "outcomes": {"compared": len(eids)},
            "child_failures": 0, "started_at": "t0", "finished_at": "t1",
            "records_count": len(eids), "eval_ids": list(eids),
            "no_target_races": len(eids) == 0, "all_children_failed": False, "no_odds_count": 0,
            "records_sha256": hashlib.sha256(rec_bytes).hexdigest(),
            "environment": {"python": "3.x", "packages": {"p": "1.0"},
                            "code_sha256": {"f": "a" * 64}, "fan_files": {"g": None}},
            "inputs_sha256": {"programs_raw": "b" * 64, "odds": "b" * 64},
        }
        self._write_summary(d, summary)
        return d, summary

    def _write_summary(self, d, summary):
        (d / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")

    def test_valid_tree_ok(self):
        d, _ = self._mk_out(["20261107_01_01", "20261107_01_02"])
        r = GI.scan_output_tree(d)
        self.assertTrue(r["ok"], r["problems"])
        self.assertEqual(r["problems"], [])

    def test_no_target_races_ok(self):
        d, _ = self._mk_out([])
        r = GI.scan_output_tree(d)
        self.assertTrue(r["ok"], r["problems"])

    def test_no_target_races_inconsistent_detected(self):
        d, s = self._mk_out(["20261107_01_01"])
        s["no_target_races"] = True                       # eval_ids が非空なのに True は不整合
        self._write_summary(d, s)
        r = GI.scan_output_tree(d)
        self.assertFalse(r["ok"])
        self.assertIn("summary.no_target_races:inconsistent", r["problems"])

    def test_raw_body_in_record_detected(self):
        d, s = self._mk_out(["20261107_01_01"])
        bad = {"eval_id": "20261107_01_01", "child_stderr_tail": "boom SECRET"}
        rb = (json.dumps(bad, ensure_ascii=False) + "\n").encode("utf-8")
        (d / "records.jsonl").write_bytes(rb)
        s["records_sha256"] = hashlib.sha256(rb).hexdigest()   # SHA は一致させ、未分類だけを残す
        self._write_summary(d, s)
        r = GI.scan_output_tree(d)
        self.assertFalse(r["ok"])
        self.assertIn("records.jsonl[0].child_stderr_tail", r["problems"])

    def test_records_sha_mismatch_detected(self):
        d, s = self._mk_out(["20261107_01_01", "20261107_01_02"])
        s["records_sha256"] = "0" * 64
        self._write_summary(d, s)
        r = GI.scan_output_tree(d)
        self.assertFalse(r["ok"])
        self.assertIn("summary.records_sha256:mismatch", r["problems"])


class FreetextPathRulesTests(unittest.TestCase):
    """B（F-10 受入ブロッカー修正）：Legacy 整形文字列をパス限定・上限付きで許可し、逸脱は拒否。"""

    def test_accepts_real_shaped_values(self):
        ok = [
            {"evaluation": {"upset_result": {"detail": {"荒れ確率": "30.0%", "モデルVersion": "v1",
                                                        "等級フィルタ": "適用"}}}},
            {"g3": {"capture": {"return_value": {"message": "見送り: 資金条件",
                                                 "excluded": [{"combo": "1-2-3", "reason": "オッズ低"}]}}}},
            {"g3": {"capture": {"return_value": {"message": None, "excluded": []}}}},
        ]
        for o in ok:
            self.assertEqual(GI.unclassified_output_keys(o), [], o)

    def test_detail_rejects_bad_shapes(self):
        def base(d):
            return {"evaluation": {"upset_result": {"detail": d}}}
        bad = [
            base({"x": 1}),                              # 非 str 値
            base({"x": {"y": "z"}}),                     # ネスト
            base("raw"),                                 # 自由文スカラー
            base({"x": "y" * 300}),                      # 値長超過
            base({f"k{i}": "v" for i in range(17)}),     # 要素数超過
        ]
        for o in bad:
            self.assertIn("evaluation.upset_result.detail", GI.unclassified_output_keys(o), o)

    def test_message_rejects_bad(self):
        def rv(m):
            return {"g3": {"capture": {"return_value": {"message": m, "excluded": []}}}}
        self.assertIn("g3.capture.return_value.message", GI.unclassified_output_keys(rv({"a": 1})))
        self.assertIn("g3.capture.return_value.message", GI.unclassified_output_keys(rv("x" * 1100)))
        self.assertEqual(GI.unclassified_output_keys(rv("x" * 1024)), [])   # 上限ちょうどは許可

    def test_excluded_rejects_bad(self):
        def rv(ex):
            return {"g3": {"capture": {"return_value": {"message": None, "excluded": ex}}}}
        cases = [
            [{"combo": "1-2", "reason": 123}],                        # reason 型違い
            [{"combo": "1-2"}],                                       # reason 欠落
            [{"combo": "1-2", "reason": "x", "k": 1}],                # キー追加
            [{"combo": "1-2", "reason": "y" * 300}],                  # reason 長超過
            [{"combo": {"child_stderr_tail": "RAW"}, "reason": "x"}],  # combo dict 密輸
            [{"combo": "x" * 40, "reason": "x"}],                     # combo 長超過
            [{"combo": "1-2", "reason": "x"}] * 300,                  # リスト長超過
            "notalist",                                               # list でない
        ]
        for ex in cases:
            self.assertIn("g3.capture.return_value.excluded", GI.unclassified_output_keys(rv(ex)), ex)

    def test_other_detail_message_reason_paths_still_none_only(self):
        # 3パス以外の detail/message は従来どおり None のみ、reason も従来どおり拒否（検証力を弱めない）
        self.assertIn("message", GI.unclassified_output_keys({"message": "x"}))
        self.assertIn("detail", GI.unclassified_output_keys({"detail": {"a": "b"}}))
        self.assertIn("foo.reason", GI.unclassified_output_keys({"foo": {"reason": "x"}}))

    def test_raw_and_exception_handling_unchanged(self):
        self.assertEqual(
            GI.unclassified_output_keys({"child_stderr_tail": {"redacted": True, "len": 3, "lines": 1}}), [])
        self.assertIn("child_stderr_tail", GI.unclassified_output_keys({"child_stderr_tail": "raw"}))
        self.assertIn("error", GI.unclassified_output_keys({"error": "RuntimeError"}))


class LegacyContextKeysSyncTests(unittest.TestCase):
    """本人判断1 の複製 _LEGACY_CONTEXT_KEYS が producer 側 LEGACY_CONTEXT_KEYS と一致し続けることを保証。"""

    def test_in_sync(self):
        from actions import g3_entrypoint as G
        self.assertEqual(set(GI._LEGACY_CONTEXT_KEYS), set(G.LEGACY_CONTEXT_KEYS))


class RealShapedRecordDetectorTests(unittest.TestCase):
    """実 producer 形状（entry._evaluation_record／entrypoint capture・upset.detail・x_buyscore return）を
    模したフルレコードが検出器を通過することを確認。Stage14 で実捕捉 fixture に差し替える前提の暫定正本。"""

    def _real_shaped_record(self):
        # 現実の COMPARED レコードに寄せた形（kind/status 実値、passthrough=False＋buy/excluded、
        # danger_breakdown・weather dict を含む）。実捕捉 fixture は Stage14/CI で差し替える。
        return {
            "kind": "g3_stage3_diagnostic_race", "eval_id": "20261107_01_01", "race_date": "20261107",
            "stage": "done", "error": None,
            "evaluation": {
                "eval_id": "20261107_01_01", "upset_score": 3.2, "race_type": "一般",
                "win_probs": {"1": 0.4, "2": 0.2},
                "upset_result": {
                    "upset_score": 3.2, "target_lanes": [2, 3], "lane_probs": {"1": 0.4, "2": 0.2},
                    "danger_score": 1.2, "venue_name": "住之江",
                    "danger_breakdown": {"weighted": 1.2, "worse_count": 2, "worse_total": 5, "kind": "a"},
                    "detail": {"荒れ確率": "30.0%", "1号艇確率": "40.0%(rank1)",
                               "モデルVersion": "asahi-v4.0-unified-engine"},
                },
                "sha256": "a" * 64,
            },
            "g3": {
                "outcome": "compared", "exception": None,
                "capture": {
                    "status": "captured", "call_count": 1, "return_count": 1, "errors": ["cfg_not_captured"],
                    "return_value": {
                        "buy": [{"combo": "1-2-3", "amount": 100, "buyscore": 7.5, "kelly": 0.1,
                                 "is_korogashi": False, "is_ippatsu": True, "why_bet": "高期待値"}],
                        "passthrough": False, "passthrough_reason": None,
                        "message": "買い目: 1-2-3 ◎ / 資金通常", "capital_reason": "通常",
                        "excluded": [{"combo": "1-4-5", "reason": "オッズ低"}]},
                    "return_value_sha256": "b" * 64,
                },
                "conditions": {"c6_context": {"ok": True, "check_detail": {"diff_keys": ["upset_score"]}}},
                "evaluate_bets_args": {"weather_type": "WeatherInfo",
                                       "weather": {"wind_speed": 3.0, "wind_direction": "N",
                                                   "wave_height": 2.0, "weather": "晴"}},
                "legacy": {"passthrough": False, "passthrough_reason": None, "capital_reason": "通常",
                           "skip_reason": None},
            },
            "message": None, "detail": None,
        }

    def test_full_record_passes_detector(self):
        self.assertEqual(GI.unclassified_output_keys(self._real_shaped_record()), [])


if __name__ == "__main__":
    unittest.main()
