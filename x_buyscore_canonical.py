"""
x_buyscore_canonical.py ── buyscore_log.jsonl の「1レース1件」化（候補の比較用）

notify は実行のたびに締切前の全レースを評価し、そのたびに購入ログへ1件追記する
（docs/incidents/2026-10-14_buyscore-log-duplicate-and-nonatomic-upload.md §2.1）。
どの記録をそのレースの判定とみなすかは未決定（同 §5）のため、本モジュールは
候補ごとの選び方を提供するだけで、どこからも呼ばれていない。

記録には時刻が無い。保存は追記と、Release 上の最新版との統合（x_release_storage）で
行われるため、ファイル内の順序＝保存された順序とみなす。
"""
from __future__ import annotations

import json
from typing import Iterable

POLICIES = ("first", "last", "last_pass")


def race_key(rec: dict) -> tuple:
    return (str(rec.get("date", "")), str(rec.get("venue", "")), str(rec.get("race", "")))


def read_jsonl(lines: Iterable[str]) -> tuple[list[dict], int]:
    """JSON として読めない行（途中で切れた行など）は数えて除外する。"""
    out, bad = [], 0
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            bad += 1
            continue
        if isinstance(obj, dict):
            out.append(obj)
        else:
            bad += 1
    return out, bad


def select(records: list[dict], policy: str) -> list[dict]:
    """
    first     : そのレースで最初に保存された記録
    last      : 最後に保存された記録（締切前の最後の判定。ただし時刻が無いため
                「締切前であること」は記録からは確かめられない）
    last_pass : 購入判定を通過した（passthrough_reason が None の）記録のうち最後のもの。
                通過した記録が無ければ last と同じ
    返す順序は、各レースが最初に現れた順。
    """
    if policy not in POLICIES:
        raise ValueError(policy)
    order: list[tuple] = []
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        k = race_key(r)
        if k not in groups:
            groups[k] = []
            order.append(k)
        groups[k].append(r)
    out = []
    for k in order:
        g = groups[k]
        if policy == "first":
            out.append(g[0])
        elif policy == "last":
            out.append(g[-1])
        else:
            passed = [r for r in g if r.get("passthrough_reason") is None]
            out.append(passed[-1] if passed else g[-1])
    return out


def summarize(records: list[dict]) -> dict:
    """候補ごとの件数の違いを見るための集計（analyze_buyscore_log と同じ数え方）。"""
    res = {"raw_records": len(records), "races": len({race_key(r) for r in records})}
    for p in POLICIES:
        sel = select(records, p)
        res[p] = {
            "n_races": len(sel),
            "buy_count": sum(1 for r in sel if not r.get("passthrough_reason")),
            "passthrough_count": sum(1 for r in sel if r.get("passthrough_reason")),
        }
    return res
