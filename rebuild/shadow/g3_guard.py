"""
正式G3（段階4-2）：排他のための入口ガード（GPT最終判断⑤・doc 17 §3）。

G3 run は G2 instrument を import・起動しない（統合プロファイラは採らない）。
入口で既存プロファイラ（sys/threading）が無いことを明示アサートする。これは
shadow.g3_capture の existing_profiler による fail-closed に加えた早期検出で、
G2 instrument（sys.setprofile / threading.setprofile を使う）との同一 run 併用を防ぐ。

正式G2 の評価側 baseline 42ファイル・instrument.py は変更しない新規ファイル。
"""

from __future__ import annotations

import sys
import threading
from typing import Any


class G3ProfilerPresent(RuntimeError):
    """G3 入口で既存プロファイラが検出された（排他違反の早期検出）。"""


def profiler_state() -> dict[str, bool]:
    return {"sys": sys.getprofile() is not None, "threading": threading.getprofile() is not None}


def assert_no_profiler() -> dict[str, bool]:
    """既存プロファイラが無いことを確かめる。あれば G3ProfilerPresent を送出（fail-closed）。"""
    st = profiler_state()
    if st["sys"] or st["threading"]:
        raise G3ProfilerPresent(f"profiler already present at G3 entry: {st}")
    return st
