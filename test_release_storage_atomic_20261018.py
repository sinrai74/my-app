"""
x_release_storage の保存手順（原子性・統合・途中失敗からの回復）の検証。

GitHub Releases API をメモリ上の偽サーバで置き換え、各 API 呼び出しの失敗・
実行の異常終了（途中で例外が出て以降の処理が行われない）を注入する。
ネットワークには接続しない。
"""
import hashlib
import json
import os
import re

import pytest
import requests

import x_release_storage as xrs

API = xrs.API_BASE
UP = xrs.UPLOAD_BASE


class Crash(BaseException):
    """実行の異常終了（プロセスがその場で止まる）を表す。except Exception では捕まらない。"""


class Resp:
    def __init__(self, status, body=None, content=b""):
        self.status_code = status
        self._body = body
        self.content = content
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        return self._body

    def iter_content(self, chunk_size=65536):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]


class FakeGH:
    def __init__(self):
        self.assets = {}  # id -> {"name", "data"}
        self.next_id = 100
        self.faults = []  # [method, regex, action, remaining]
        self.calls = []

    # ── テスト用の操作 ──
    def put(self, name, data: bytes):
        self.next_id += 1
        self.assets[self.next_id] = {"name": name, "data": data}

    def names(self):
        return sorted(a["name"] for a in self.assets.values())

    def get_data(self, name):
        for a in self.assets.values():
            if a["name"] == name:
                return a["data"]
        return None

    def fault(self, method, pattern, action="error", times=1):
        self.faults.append([method, re.compile(pattern), action, times])

    def _check_fault(self, method, url):
        for f in self.faults:
            if f[0] == method and f[3] > 0 and f[1].search(url):
                f[3] -= 1
                if f[2] == "crash":
                    raise Crash(f"{method} {url}")
                if f[2] == "neterr":
                    raise requests.ConnectionError("injected")
                if f[2] == "badsize":
                    return "badsize"
                return Resp(500, {"message": "injected"})
        return None

    def _asset_json(self, i):
        a = self.assets[i]
        return {
            "id": i, "name": a["name"], "size": len(a["data"]),
            "url": f"{API}/repos/o/r/releases/assets/{i}",
            "digest": "sha256:" + hashlib.sha256(a["data"]).hexdigest(),
        }

    # ── requests の代わり ──
    def get(self, url, headers=None, timeout=None, stream=False, **kw):
        self.calls.append(("GET", url))
        f = self._check_fault("GET", url)
        if isinstance(f, Resp):
            return f
        if url.endswith("/releases/tags/data-store"):
            return Resp(200, {"id": 1, "assets": [self._asset_json(i) for i in sorted(self.assets)]})
        m = re.search(r"/releases/assets/(\d+)$", url)
        if m and int(m.group(1)) in self.assets:
            return Resp(200, content=self.assets[int(m.group(1))]["data"])
        return Resp(404, {})

    def post(self, url, headers=None, data=None, json=None, timeout=None, **kw):
        self.calls.append(("POST", url))
        f = self._check_fault("POST", url)
        if isinstance(f, Resp):
            return f
        m = re.search(r"/assets\?name=(.+)$", url)
        if m:
            name = m.group(1)
            if name in [a["name"] for a in self.assets.values()]:
                return Resp(422, {"message": "already_exists"})
            self.put(name, data)
            body = self._asset_json(self.next_id)
            if f == "badsize":
                body["size"] = body["size"] + 1
            return Resp(201, body)
        return Resp(404, {})

    def delete(self, url, headers=None, timeout=None, **kw):
        self.calls.append(("DELETE", url))
        f = self._check_fault("DELETE", url)
        if isinstance(f, Resp):
            return f
        m = re.search(r"/releases/assets/(\d+)$", url)
        if m and int(m.group(1)) in self.assets:
            del self.assets[int(m.group(1))]
            return Resp(204)
        return Resp(404, {})

    def patch(self, url, headers=None, json=None, timeout=None, **kw):
        self.calls.append(("PATCH", url))
        f = self._check_fault("PATCH", url)
        if isinstance(f, Resp):
            return f
        m = re.search(r"/releases/assets/(\d+)$", url)
        if not m or int(m.group(1)) not in self.assets:
            return Resp(404, {})
        new = json["name"]
        if new in [a["name"] for a in self.assets.values()]:
            return Resp(422, {"message": "already_exists"})
        self.assets[int(m.group(1))]["name"] = new
        return Resp(200, self._asset_json(int(m.group(1))))


@pytest.fixture
def gh(monkeypatch, tmp_path):
    fake = FakeGH()
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setenv("GITHUB_REPOSITORY", "o/r")
    for m in ("get", "post", "delete", "patch"):
        monkeypatch.setattr(xrs.requests, m, getattr(fake, m))
    monkeypatch.setattr(xrs.time, "sleep", lambda s: None)
    monkeypatch.chdir(tmp_path)
    return fake


def _w(path, text):
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def _r(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


LOG = "buyscore_log.jsonl"
A = '{"date":"20261014","venue":"丸亀","race":"4","n":1}\n'
B = '{"date":"20261014","venue":"丸亀","race":"4","n":2}\n'
C = '{"date":"20261014","venue":"丸亀","race":"5","n":3}\n'


def _lines(b: bytes):
    return [ln for ln in b.decode("utf-8").splitlines() if ln.strip()]


# ════════════════════════════════════════════════════════════
# 通常の保存
# ════════════════════════════════════════════════════════════

def test_first_upload_creates_main_only(gh):
    _w(LOG, A)
    assert xrs.upload_file(LOG, LOG)
    assert gh.names() == [LOG]
    assert gh.get_data(LOG) == A.encode()


def test_append_rotates_bak(gh):
    gh.put(LOG, A.encode())
    _w(LOG, A + B)
    assert xrs.upload_file(LOG, LOG)
    assert gh.names() == [LOG, LOG + ".bak"]
    assert gh.get_data(LOG) == (A + B).encode()
    assert gh.get_data(LOG + ".bak") == A.encode()


def test_no_new_content_is_noop(gh):
    gh.put(LOG, (A + B).encode())
    _w(LOG, A + B)
    n = len(gh.calls)
    assert xrs.upload_file(LOG, LOG)
    assert not any(c[0] in ("POST", "DELETE", "PATCH") for c in gh.calls[n:])


# ════════════════════════════════════════════════════════════
# 古いローカル・ダウンロード失敗・別の実行の保存（後勝ちで消えない）
# ════════════════════════════════════════════════════════════

def test_stale_local_does_not_drop_remote_rows(gh):
    gh.put(LOG, (A + B).encode())   # 別の実行が B を保存済み
    _w(LOG, A + C)                  # こちらは A の時点の版に C を追記
    assert xrs.upload_file(LOG, LOG)
    assert _lines(gh.get_data(LOG)) == [A.strip(), B.strip(), C.strip()]


def test_download_failed_then_upload_keeps_history(gh):
    gh.put(LOG, (A + B).encode())
    gh.fault("GET", r"/releases/assets/\d+$")          # 実行開始時のダウンロードだけ失敗
    assert not xrs.download_file(LOG, LOG)
    _w(LOG, C)                                          # 空から1件だけ書いた状態（I5）
    assert xrs.upload_file(LOG, LOG)
    assert _lines(gh.get_data(LOG)) == [A.strip(), B.strip(), C.strip()]


def test_remote_unreadable_at_upload_refuses(gh):
    gh.put(LOG, (A + B).encode())
    _w(LOG, C)
    gh.fault("GET", r"/releases/assets/\d+$", times=5)
    assert not xrs.upload_file(LOG, LOG)
    assert gh.names() == [LOG]
    assert gh.get_data(LOG) == (A + B).encode()


def test_interleaved_runs_both_kept(gh):
    gh.put(LOG, A.encode())
    d1, d2 = "r1", "r2"
    os.makedirs(d1); os.makedirs(d2)
    assert xrs.download_file(LOG, f"{d1}/{LOG}")
    assert xrs.download_file(LOG, f"{d2}/{LOG}")
    with open(f"{d1}/{LOG}", "a", encoding="utf-8") as f:
        f.write(B)
    with open(f"{d2}/{LOG}", "a", encoding="utf-8") as f:
        f.write(C)
    assert xrs.upload_file(f"{d1}/{LOG}", LOG)
    assert xrs.upload_file(f"{d2}/{LOG}", LOG)
    assert _lines(gh.get_data(LOG)) == [A.strip(), B.strip(), C.strip()]


# ════════════════════════════════════════════════════════════
# 各手順での失敗：本体は元のまま（または回復可能）、履歴は失われない
# ════════════════════════════════════════════════════════════

FAIL_POINTS = [
    ("POST", r"name=buyscore_log\.jsonl\.new", "error"),     # .new のアップロード
    ("POST", r"name=buyscore_log\.jsonl\.new", "badsize"),   # 検証の不一致
    ("POST", r"name=buyscore_log\.jsonl\.new", "neterr"),
    ("DELETE", r"/releases/assets/\d+$", "error"),           # 旧 .bak の削除
    ("PATCH", r"/releases/assets/\d+$", "error"),            # 本体 → .bak
]


@pytest.mark.parametrize("method,pat,action", FAIL_POINTS)
def test_failure_keeps_main_intact(gh, method, pat, action):
    gh.put(LOG, (A + B).encode())
    gh.put(LOG + ".bak", A.encode())
    _w(LOG, A + B + C)
    gh.fault(method, pat, action)
    assert not xrs.upload_file(LOG, LOG)
    assert gh.get_data(LOG) == (A + B).encode()
    assert LOG + ".new" not in gh.names()
    # 次の実行は通常どおり保存できる
    assert xrs.upload_file(LOG, LOG)
    assert _lines(gh.get_data(LOG)) == [A.strip(), B.strip(), C.strip()]


def test_promote_failure_leaves_new_and_recovers(gh):
    gh.put(LOG, (A + B).encode())
    _w(LOG, A + B + C)
    # 本体→.bak は成功、.new→本体 は2回とも失敗
    gh.fault("PATCH", r"/releases/assets/\d+$", "error", times=0)
    calls = {"n": 0}
    orig = gh.patch

    def patch(url, **kw):
        calls["n"] += 1
        if calls["n"] >= 2:
            return Resp(500, {})
        return orig(url, **kw)

    gh.patch = patch
    xrs.requests.patch = patch
    assert not xrs.upload_file(LOG, LOG)
    assert LOG not in gh.names()
    assert sorted(gh.names()) == [LOG + ".bak", LOG + ".new"]
    # 読み込みは .new（最新）を読む
    assert xrs.download_file(LOG, "read.jsonl")
    assert _r("read.jsonl") == A + B + C
    # 次の保存で回復する
    xrs.requests.patch = orig
    with open("read.jsonl", "a", encoding="utf-8") as f:
        f.write('{"n":4}\n')
    assert xrs.upload_file("read.jsonl", LOG)
    assert LOG + ".new" not in gh.names()
    assert _lines(gh.get_data(LOG))[-1] == '{"n":4}'


@pytest.mark.parametrize("method,pat", [
    ("POST", r"name=buyscore_log\.jsonl\.new"),
    ("DELETE", r"/releases/assets/\d+$"),
    ("PATCH", r"/releases/assets/\d+$"),
])
def test_crash_at_any_step_loses_no_committed_data(gh, method, pat):
    gh.put(LOG, (A + B).encode())
    gh.put(LOG + ".bak", A.encode())
    _w(LOG, A + B + C)
    gh.fault(method, pat, "crash")
    with pytest.raises(Crash):
        xrs.upload_file(LOG, LOG)
    # 新しい実行環境で読む
    assert xrs.download_file(LOG, "fresh.jsonl")
    got = _lines(open("fresh.jsonl", "rb").read())
    assert A.strip() in got and B.strip() in got
    # そこへ追記して保存すれば、C を含めて回復する（C は異常終了した実行の分）
    with open("fresh.jsonl", "a", encoding="utf-8") as f:
        f.write(C)
    assert xrs.upload_file("fresh.jsonl", LOG)
    final = _lines(gh.get_data(LOG))
    assert final.count(C.strip()) == 1
    assert {A.strip(), B.strip(), C.strip()} <= set(final)
    assert LOG + ".new" not in gh.names()


def test_crash_between_bak_and_promote(gh):
    """本体を .bak にした直後に停止（本体が無い唯一の状態）"""
    gh.put(LOG, (A + B).encode())
    _w(LOG, A + B + C)
    n = {"i": 0}
    orig = gh.patch

    def patch(url, **kw):
        n["i"] += 1
        if n["i"] == 2:
            raise Crash("after main->bak")
        return orig(url, **kw)

    xrs.requests.patch = patch
    with pytest.raises(Crash):
        xrs.upload_file(LOG, LOG)
    xrs.requests.patch = orig
    assert LOG not in gh.names()
    assert xrs.download_file(LOG, "fresh.jsonl")
    assert _r("fresh.jsonl") == A + B + C


def test_legacy_gap_state_reads_bak(gh):
    """旧方式の保存の途中（本体削除後、再アップロード前）＝ 2026-10-14 10:59 の状態（E4）"""
    gh.put(LOG + ".bak", (A + B).encode())
    assert xrs.download_file(LOG, LOG)
    assert _r(LOG) == A + B
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(C)
    assert xrs.upload_file(LOG, LOG)
    assert _lines(gh.get_data(LOG)) == [A.strip(), B.strip(), C.strip()]


def test_legacy_gap_state_fresh_local_still_merges_bak(gh):
    gh.put(LOG + ".bak", (A + B).encode())
    _w(LOG, C)
    assert xrs.upload_file(LOG, LOG)
    assert _lines(gh.get_data(LOG)) == [A.strip(), B.strip(), C.strip()]


# ════════════════════════════════════════════════════════════
# 追記型でないファイル（全置換）
# ════════════════════════════════════════════════════════════

def test_non_append_file_replaces_and_keeps_bak(gh):
    gh.put("daily_stats.json", b'{"v":1}')
    _w("daily_stats.json", '{"v":2}')
    assert xrs.upload_file("daily_stats.json", "daily_stats.json")
    assert gh.get_data("daily_stats.json") == b'{"v":2}'
    assert gh.get_data("daily_stats.json.bak") == b'{"v":1}'


def test_non_append_file_gap_state_not_read_from_bak(gh):
    gh.put("daily_stats.json.bak", b'{"v":1}')
    assert not xrs.download_file("daily_stats.json", "daily_stats.json")


# ════════════════════════════════════════════════════════════
# hit_record.csv（行の統合・ヘッダ不一致）
# ════════════════════════════════════════════════════════════

HDR = "date,venue_num,race,pred_combo,hit\r\n"
R1 = "20261017,1,1,1-2-3,0\r\n"
R2 = "20261017,1,2,1-3-2,1\r\n"
R3 = "20261018,2,1,2-1-3,0\r\n"


def test_hit_record_merge_keeps_remote_rows(gh):
    gh.put("hit_record.csv", (HDR + R1 + R2).encode())
    _w("hit_record.csv", HDR + R1 + R3)
    assert xrs.upload_file("hit_record.csv", "hit_record.csv")
    assert gh.get_data("hit_record.csv").decode() == HDR + R1 + R2 + R3


def test_hit_record_header_mismatch_refuses(gh):
    gh.put("hit_record.csv", (HDR + R1).encode())
    _w("hit_record.csv", "date,venue_num,race\r\n20261018,2,1\r\n")
    assert not xrs.upload_file("hit_record.csv", "hit_record.csv")
    assert gh.get_data("hit_record.csv") == (HDR + R1).encode()


def test_hit_record_append_with_wrong_width_refuses(gh):
    gh.put("hit_record.csv", (HDR + R1).encode())
    _w("hit_record.csv", HDR + R1 + "20261018,2,1,2-1-3,0,EXTRA\r\n")
    assert not xrs.upload_file("hit_record.csv", "hit_record.csv")
    assert gh.get_data("hit_record.csv") == (HDR + R1).encode()


def test_hit_record_merge_false_replaces(gh):
    gh.put("hit_record.csv", (HDR + R1 + R2).encode())
    _w("hit_record.csv", HDR + R1)
    assert xrs.upload_file("hit_record.csv", "hit_record.csv", merge=False)
    assert gh.get_data("hit_record.csv") == (HDR + R1).encode()


# ════════════════════════════════════════════════════════════
# sent ファイル（同じ key は先に保存された行を残す）
# ════════════════════════════════════════════════════════════

def test_sent_key_first(gh):
    k1a = json.dumps({"key": "20261018_1_1", "combo": "1-2-3"})
    k1b = json.dumps({"key": "20261018_1_1", "combo": "3-2-1"})
    k2 = json.dumps({"key": "20261018_1_2", "combo": "1-3-2"})
    gh.put("sent_20261018.txt", k1a.encode())
    _w("sent_20261018.txt", "\n".join([k1b, k2]))
    assert xrs.upload_file("sent_20261018.txt", "sent_20261018.txt")
    got = gh.get_data("sent_20261018.txt").decode().split("\n")
    assert got == [k1a, k2]


def test_merge_kind_registry():
    assert xrs.merge_kind("hit_record.csv") == "csv_rows"
    assert xrs.merge_kind("buyscore_log.jsonl") == "lines"
    assert xrs.merge_kind("sent_20261018.txt") == "sent_key_first"
    assert xrs.merge_kind("daily_stats.json") is None
    assert xrs.merge_kind("motor_history.csv") is None
    assert xrs.merge_kind("hit_record_legacy44_20260704.csv") is None
