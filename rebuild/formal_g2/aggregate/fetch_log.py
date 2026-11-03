"""正式G2：評価ジョブのログ原本の取得（M7-3。集約側だけで使う）。

GitHub REST API「Download job logs for a workflow run」を2段階で取得する。
  1. API に認証付きで GET し、302 と Location（署名付き URL）を受け取る。リダイレクトは追わない。
  2. Location に、Authorization ヘッダーを付けずに GET する。200 で、空でないことを確かめる。
- 原本のバイト列をそのまま保存する（改行変換・ANSI 除去・文字コード変換をしない）。
- 署名付き URL の問い合わせ部分とトークンは、記録にもエラー文にも出さない。
- 標準ライブラリだけを使い、評価コード・Legacy のコードは読み込まない。

実行（集約 workflow から）：
  GH_TOKEN=... python -P -B -m formal_g2.aggregate.fetch_log \
      --repo OWNER/REPO --job-id ID --out job.log --result fetch.json
  結果（fetch.json）：{"ok", "error", "api_status", "download_status", "download_host",
                      "bytes", "sha256"}。終了コードは常に 0（成否は ok で表し、集約が判定する）。
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import socket
import sys
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

API_HOST = "api.github.com"
# 署名付き URL として受け付けるホスト（GitHub Actions のログ・結果の保存先）。これ以外は失敗にする。
ALLOWED_HOST_RE = re.compile(r"(?:[a-z0-9-]+\.)+(?:blob\.core\.windows\.net|actions\.githubusercontent\.com)")
TIMEOUT_SECONDS = 60
MAX_BYTES = 64 * 1024 * 1024
USER_AGENT = "formal-g2-aggregate"

ConnFactory = Callable[[str, float], Any]


def _https_conn(host: str, timeout: float) -> http.client.HTTPSConnection:
    return http.client.HTTPSConnection(host, 443, timeout=timeout)


class FetchError(Exception):
    """記録してよい文だけを持つ（URL の問い合わせ部分・トークンを含めない）。"""


def check_location(location: Optional[str]) -> tuple[str, str]:
    """Location を確かめ、(host, path?query) を返す。問題があれば FetchError。"""
    if not location or not location.strip():
        raise FetchError("redirect Location missing or empty")
    parts = urlsplit(location.strip())
    if parts.scheme != "https":
        raise FetchError(f"redirect Location scheme not https: {parts.scheme or '(none)'}")
    host = (parts.hostname or "").lower()
    if parts.username is not None or parts.password is not None:
        raise FetchError("redirect Location has userinfo")
    if parts.port not in (None, 443):
        raise FetchError(f"redirect Location port not allowed: {parts.port}")
    if not ALLOWED_HOST_RE.fullmatch(host):
        raise FetchError(f"redirect Location host not allowed: {host or '(none)'}")
    target = parts.path or "/"
    if parts.query:
        target += "?" + parts.query
    return host, target


def fetch_job_log(repo: str, job_id: str, token: str, out_path: str,
                  conn_factory: Optional[ConnFactory] = None, timeout: float = TIMEOUT_SECONDS,
                  max_bytes: int = MAX_BYTES) -> dict[str, Any]:
    conn_factory = conn_factory or _https_conn
    res: dict[str, Any] = {"ok": False, "error": None, "api_status": None, "download_status": None,
                           "download_host": None, "bytes": None, "sha256": None}
    try:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo or ""):
            raise FetchError("invalid repo")
        if not re.fullmatch(r"\d+", job_id or ""):
            raise FetchError("invalid job id")
        if not token:
            raise FetchError("token unavailable")
        # 1. API（認証付き）。リダイレクトは追わない。
        conn = conn_factory(API_HOST, timeout)
        try:
            conn.request("GET", f"/repos/{repo}/actions/jobs/{job_id}/logs", headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": USER_AGENT,
            })
            resp = conn.getresponse()
            res["api_status"] = resp.status
            location = resp.getheader("Location")
            resp.read()
        finally:
            conn.close()
        if resp.status != 302:
            raise FetchError(f"API status {resp.status} (expected 302)")
        host, target = check_location(location)
        res["download_host"] = host
        # 2. 署名付き URL（Authorization を付けない）
        conn = conn_factory(host, timeout)
        try:
            conn.request("GET", target, headers={"User-Agent": USER_AGENT})
            resp = conn.getresponse()
            res["download_status"] = resp.status
            if resp.status != 200:
                resp.read()
                raise FetchError(f"download status {resp.status} (expected 200) from {host}")
            chunks, total = [], 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                total += len(chunk)
                if total > max_bytes:
                    raise FetchError(f"log larger than {max_bytes} bytes")
                chunks.append(chunk)
        finally:
            conn.close()
        blob = b"".join(chunks)
        if not blob:
            raise FetchError("downloaded log is empty")
        with open(out_path, "wb") as f:  # 原本のバイト列をそのまま保存する
            f.write(blob)
        res.update(ok=True, bytes=len(blob), sha256=hashlib.sha256(blob).hexdigest())
    except FetchError as exc:
        res["error"] = str(exc)
    except (socket.timeout, TimeoutError):
        res["error"] = "timeout"
    except (OSError, http.client.HTTPException) as exc:
        res["error"] = f"network error: {type(exc).__name__}"
    if not res["ok"] and os.path.exists(out_path):
        os.remove(out_path)
    return res


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="formal_g2.aggregate.fetch_log")
    p.add_argument("--repo", required=True)
    p.add_argument("--job-id", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--result", required=True)
    args = p.parse_args(argv)
    res = fetch_job_log(args.repo, args.job_id, os.environ.get("GH_TOKEN", ""), args.out)
    with open(args.result, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, sort_keys=True)
    print(f"fetch_log: job={args.job_id} ok={res['ok']} api_status={res['api_status']} "
          f"download_status={res['download_status']} host={res['download_host']} "
          f"bytes={res['bytes']} sha256={res['sha256']} error={res['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
