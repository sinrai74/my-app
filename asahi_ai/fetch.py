"""公式 B/K ファイルの取得と原本の保存（書き換えない）。

- 1件ずつ、前の取得から INTERVAL 秒以上あけて取りに行く。
- 通信の失敗は1回だけ再試行する。HTTP の 404 は「ファイルが無い」として記録する（開催が無い日を含む）。
- 原本はそのまま保存する。同じ名前ですでにあり、SHA-256 が違えば例外（上書きしない）。
- 取得のたびに manifest（raw_files の1行）を返す。
"""
from __future__ import annotations

import datetime as dt
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from . import PARSER_VERSION, archive
from .layout import LAYOUT_VERSION

BASE = "http://www1.mbrace.or.jp/od2"
UA = "Mozilla/5.0 (compatible; asahi-ai-data/1)"
INTERVAL = 3.0


class RawConflict(Exception):
    pass


def url_for(kind: str, d: dt.date) -> str:
    return f"{BASE}/{kind}/{d:%Y%m}/{kind.lower()}{d:%y%m%d}.lzh"


def _default_opener(url: str, timeout: float):
    """返り値：(HTTP の状態, 本体, Last-Modified の値 or None)。"""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read(), r.headers.get("Last-Modified")


class Fetcher:
    def __init__(self, raw_dir: Path, opener: Callable | None = None,
                 interval: float = INTERVAL, clock=time.time, sleep=time.sleep):
        self.raw_dir = Path(raw_dir)
        self.opener = opener or _default_opener
        self.interval = interval
        self.clock, self.sleep = clock, sleep
        self._last = None

    def _wait(self):
        if self._last is not None:
            gap = self.interval - (self.clock() - self._last)
            if gap > 0:
                self.sleep(gap)

    def _open(self, url):
        """返り値：(状態, 本体, Last-Modified, エラー)。通信の失敗と 5xx・429 は1回だけ再試行する。"""
        status, err = None, "unreachable"
        for attempt in (1, 2):
            self._wait()
            try:
                r = self.opener(url, 60)
                status, data = r[0], r[1]
                lm = r[2] if len(r) > 2 else None
                return status, data, lm, None
            except urllib.error.HTTPError as e:
                status, err = e.code, f"HTTP {e.code}: {e.reason}"
                if not (e.code >= 500 or e.code == 429):
                    return status, None, None, err
            except Exception as e:  # noqa: BLE001  通信の失敗
                status, err = None, f"{type(e).__name__}: {e}"
            finally:
                self._last = self.clock()
        return status, None, None, err

    def path_for(self, kind: str, d: dt.date) -> Path:
        return self.raw_dir / kind / f"{d:%Y}" / f"{kind.lower()}{d:%y%m%d}.lzh"

    def get(self, kind: str, d: dt.date) -> tuple[dict, bytes | None]:
        """取得して保存する。返り値は (manifest の1行, LZH のバイト列 or None)。"""
        url = url_for(kind, d)
        fetched_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        status, data, last_modified, err = self._open(url)
        rec = {
            "kind": kind, "date": d.isoformat(), "source_url": url, "fetched_at": fetched_at,
            "http_status": status, "last_modified": last_modified, "error": err, "lzh_sha256": None, "lzh_size": None,
            "member_name": None, "txt_sha256": None, "txt_size": None, "crc_ok": None,
            "capture_status": None, "layout_version": LAYOUT_VERSION, "parser_version": PARSER_VERSION,
        }
        if data is None:
            rec["capture_status"] = "not_found" if status == 404 else "fetch_failed"
            return rec, None
        rec["lzh_sha256"], rec["lzh_size"] = archive.sha256(data), len(data)
        try:
            m = archive.extract_single(data)
            rec.update(member_name=m.name, txt_sha256=archive.sha256(m.data), txt_size=len(m.data), crc_ok=True)
        except archive.ArchiveError as e:
            rec.update(crc_ok=False, error=str(e))
            # 壊れたファイルは原本の場所に置かない（別の場所に残し、次の実行で取り直す）
            bad = self.raw_dir / "_bad" / kind / f"{kind.lower()}{d:%y%m%d}.{rec['lzh_sha256'][:12]}.lzh"
            bad.parent.mkdir(parents=True, exist_ok=True)
            bad.write_bytes(data)
            rec["capture_status"] = "bad_archive"
            return rec, None
        p = self.path_for(kind, d)
        if p.exists():
            old = p.read_bytes()
            if archive.sha256(old) != rec["lzh_sha256"]:
                rec["capture_status"] = "conflict"
                raise RawConflict(f"{p} の既存の原本と内容が違います（上書きしない）: {json.dumps(rec, ensure_ascii=False)}")
            rec["capture_status"] = "unchanged"
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(data)
            tmp.replace(p)
            rec["capture_status"] = "saved"
        return rec, data
