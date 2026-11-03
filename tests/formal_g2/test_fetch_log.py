"""正式G2：評価ジョブのログ原本の2段階取得（fetch_log）。通信はすべて差し替え、外部に出ない。

実行（リポジトリ直下で）：
  PYTHONPATH="rebuild:." python -P -B -m unittest discover -s tests/formal_g2 -t tests/formal_g2
"""

import hashlib
import json
import os
import socket
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

from formal_g2.aggregate import fetch_log as fl
from formal_g2.aggregate import main as agg
from test_aggregate import BASELINE_RAW, PERMS, Q8_LOG_HEAD, _race, _write_run

TOKEN = "ghs_FAKE_TOKEN_FOR_TEST_0123456789"
SIG_QUERY = "sv=2025-11-05&sig=SECRET_SIGNATURE_VALUE&se=2026-11-03T13%3A00%3A00Z"
LOCATION = f"https://productionresultssa4.blob.core.windows.net/actions-results/x/logs/job.txt?{SIG_QUERY}"
# ANSI エスケープ・CRLF・BOM を含む原本（そのまま保存されること）
RAW_LOG = ("﻿" + Q8_LOG_HEAD).encode("utf-8") + b"2026-11-02T06:40:20Z \x1b[36;1mecho x\x1b[0m\r\n"


class FakeResp:
    def __init__(self, status, headers=None, body=b"", raise_on_read=None):
        self.status = status
        self._headers = headers or {}
        self._body = body
        self._pos = 0
        self._raise = raise_on_read

    def getheader(self, name):
        return self._headers.get(name)

    def read(self, n=-1):
        if self._raise:
            raise self._raise
        if n is None or n < 0:
            out, self._pos = self._body[self._pos:], len(self._body)
            return out
        out = self._body[self._pos:self._pos + n]
        self._pos += len(out)
        return out


class FakeConn:
    def __init__(self, host, log, responses):
        self.host, self.log, self.responses = host, log, responses

    def request(self, method, target, headers=None):
        self.log.append({"host": self.host, "method": method, "target": target, "headers": dict(headers or {})})

    def getresponse(self):
        r = self.responses[self.host]
        if isinstance(r, BaseException):
            raise r
        return r

    def close(self):
        pass


def _factory(responses, log):
    return lambda host, timeout: FakeConn(host, log, responses)


def _ok_responses(location=LOCATION, body=RAW_LOG, dl_status=200):
    return {"api.github.com": FakeResp(302, {"Location": location}),
            "productionresultssa4.blob.core.windows.net": FakeResp(dl_status, body=body)}


class FetchTests(unittest.TestCase):
    def _fetch(self, responses, **kw):
        log = []
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "job.log")
            res = fl.fetch_job_log("sinrai74/my-app", "125930182041", TOKEN, out,
                                   conn_factory=_factory(responses, log), **kw)
            data = None
            if os.path.exists(out):
                with open(out, "rb") as f:
                    data = f.read()
        return res, data, log

    def test_302_and_valid_location_saves_raw_bytes(self):
        res, data, log = self._fetch(_ok_responses())
        self.assertTrue(res["ok"], res)
        self.assertEqual(data, RAW_LOG)  # ANSI・CRLF・BOM を含めてそのまま
        self.assertEqual(res["bytes"], len(RAW_LOG))
        self.assertEqual(res["sha256"], hashlib.sha256(RAW_LOG).hexdigest())
        self.assertEqual((res["api_status"], res["download_status"]), (302, 200))
        self.assertEqual(res["download_host"], "productionresultssa4.blob.core.windows.net")
        # 1回目だけ認証付き。署名付き URL には Authorization を付けない
        self.assertEqual(log[0]["host"], "api.github.com")
        self.assertEqual(log[0]["target"], "/repos/sinrai74/my-app/actions/jobs/125930182041/logs")
        self.assertEqual(log[0]["headers"]["Authorization"], f"Bearer {TOKEN}")
        self.assertNotIn("Authorization", log[1]["headers"])
        self.assertTrue(log[1]["target"].endswith(SIG_QUERY))
        self.assertEqual(len(log), 2)

    def test_location_problems(self):
        cases = {
            "missing": None, "empty": "  ", "http": LOCATION.replace("https://", "http://"),
            "ftp": "ftp://productionresultssa4.blob.core.windows.net/x",
            "host": f"https://evil.example.com/x?{SIG_QUERY}",
            "suffix_trick": f"https://blob.core.windows.net.evil.example/x?{SIG_QUERY}",
            "userinfo": f"https://u:p@productionresultssa4.blob.core.windows.net/x?{SIG_QUERY}",
            "port": f"https://productionresultssa4.blob.core.windows.net:8443/x?{SIG_QUERY}",
        }
        for name, loc in cases.items():
            with self.subTest(name=name):
                res, data, log = self._fetch(_ok_responses(location=loc))
                self.assertFalse(res["ok"])
                self.assertIsNone(data)
                self.assertEqual(len(log), 1)  # 2段目に進まない
                self.assertNotIn("SECRET_SIGNATURE_VALUE", json.dumps(res))

    def test_api_status_not_302(self):
        for st in (200, 301, 401, 403, 404, 410, 500):
            with self.subTest(status=st):
                res, data, log = self._fetch({"api.github.com": FakeResp(st, {"Location": LOCATION})})
                self.assertFalse(res["ok"])
                self.assertEqual(res["api_status"], st)
                self.assertEqual(len(log), 1)

    def test_download_errors(self):
        for st in (403, 404, 500):
            with self.subTest(status=st):
                res, data, _ = self._fetch(_ok_responses(dl_status=st))
                self.assertFalse(res["ok"])
                self.assertEqual(res["download_status"], st)
                self.assertIsNone(data)

    def test_timeout(self):
        for where in ("api.github.com", "productionresultssa4.blob.core.windows.net"):
            with self.subTest(where=where):
                r = _ok_responses()
                r[where] = socket.timeout("timed out")
                res, data, _ = self._fetch(r)
                self.assertFalse(res["ok"])
                self.assertEqual(res["error"], "timeout")
                self.assertIsNone(data)
        r = _ok_responses()
        r["productionresultssa4.blob.core.windows.net"] = FakeResp(200, raise_on_read=TimeoutError())
        res, data, _ = self._fetch(r)
        self.assertEqual(res["error"], "timeout")

    def test_empty_and_oversize(self):
        res, data, _ = self._fetch(_ok_responses(body=b""))
        self.assertFalse(res["ok"])
        self.assertEqual(res["error"], "downloaded log is empty")
        res, data, _ = self._fetch(_ok_responses(), max_bytes=10)
        self.assertFalse(res["ok"])
        self.assertIsNone(data)

    def test_no_token_or_bad_ids(self):
        for repo, jid, tok in (("sinrai74/my-app", "1", ""), ("bad repo", "1", TOKEN), ("a/b", "1; rm", TOKEN)):
            with self.subTest(repo=repo, jid=jid):
                log = []
                with tempfile.TemporaryDirectory() as tmp:
                    res = fl.fetch_job_log(repo, jid, tok, os.path.join(tmp, "x"),
                                           conn_factory=_factory(_ok_responses(), log))
                self.assertFalse(res["ok"])
                self.assertEqual(log, [])

    def test_cli_result_and_stdout_do_not_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            out, result = os.path.join(tmp, "job.log"), os.path.join(tmp, "fetch.json")
            for responses in (_ok_responses(), _ok_responses(dl_status=403),
                              _ok_responses(location=f"https://evil.example.com/x?{SIG_QUERY}")):
                buf = StringIO()
                with mock.patch.dict(os.environ, {"GH_TOKEN": TOKEN}), \
                        mock.patch.object(fl, "_https_conn", _factory(responses, [])), \
                        redirect_stdout(buf):
                    self.assertEqual(fl.main(["--repo", "sinrai74/my-app", "--job-id", "125930182041",
                                              "--out", out, "--result", result]), 0)
                with open(result, encoding="utf-8") as f:
                    text = buf.getvalue() + f.read()
                self.assertNotIn(TOKEN, text)
                self.assertNotIn("SECRET_SIGNATURE_VALUE", text)
                self.assertNotIn("sig=", text)


class AggregateWithFetchFailureTests(unittest.TestCase):
    """取得に失敗した実行：環境変数の値で代わりにせず、不適格・終了コード 3。"""

    def test_failure_is_not_replaced_by_env_and_exit_3(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], log_fetch_ok=False)
            meta_p = os.path.join(base, "1-1", "meta.json")
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
            meta["log_fetch_error"] = ("download status 403 (expected 200) from productionresultssa4.blob.core.windows.net"
                                       " (api_status=302, download_status=403, host=productionresultssa4.blob.core.windows.net)")
            with open(meta_p, "w", encoding="utf-8") as f:
                json.dump(meta, f)
            bpath, perms = os.path.join(tmp, "b.json"), os.path.join(tmp, "p.json")
            with open(bpath, "wb") as f:
                f.write(BASELINE_RAW)
            with open(perms, "w", encoding="utf-8") as f:
                json.dump(PERMS, f)
            out = os.path.join(tmp, "out")
            buf = StringIO()
            with redirect_stdout(buf):
                rc = agg.main(["--artifacts-dir", base, "--first-date", "20990101", "--last-date", "20990101",
                               "--out", out])
            self.assertEqual(rc, agg.EXIT_VERIFICATION_FAILED)
            with open(os.path.join(out, "aggregate_result.json"), encoding="utf-8") as f:
                res = json.load(f)
            run = res["runs"][0]
            self.assertIsNone(run["runner_image"]["log"])
            self.assertEqual(run["runner_image"]["env_observed"]["ImageVersion"], "20261026.354.1")
            self.assertFalse(run["verified"])
            self.assertNotIn("sig=", json.dumps(res))


class WorkflowUsesTwoStageFetchTests(unittest.TestCase):
    def test_no_gh_api_for_logs_and_no_redirect_following(self):
        root = os.path.realpath(os.getcwd())
        with open(os.path.join(root, ".github/workflows/formal_g2_aggregate.yml"), encoding="utf-8") as f:
            text = f.read()
        code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
        self.assertNotIn("/logs\"", code)
        self.assertNotIn("--location-trusted", code)
        self.assertNotIn("curl", code)
        self.assertIn("python -P -B -m formal_g2.aggregate.fetch_log", code)

    def test_fetch_log_uses_stdlib_only_and_no_redirect_handler(self):
        root = os.path.realpath(os.getcwd())
        with open(os.path.join(root, "rebuild/formal_g2/aggregate/fetch_log.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("urllib.request", src)  # 自動でリダイレクトを追う仕組みを使わない
        self.assertNotIn("import requests", src)


if __name__ == "__main__":
    unittest.main()
