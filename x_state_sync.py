#!/usr/bin/env python3
"""
x_state_sync.py ── notify の実行の前後で sent_YYYYMMDD.txt を Release に保存・復元する

【背景】
notify_arashi._check_yesterday_results() は sent_{前日}.txt を読んで hit_record.csv に
結果を記録する。GitHub Actions の実行環境は毎回使い捨てで、sent ファイルを次の実行へ
引き継ぐ処理が無いため、前日分は常に「ファイル不存在」で処理を抜け（return_route=A）、
hit_record.csv が一度も作られていなかった（凍結タグ freeze-v1-baseline でも同じ）。

本スクリプトは notify_arashi.py を変更せず、workflow の前後の手順として動く。
  restore : 実行の前に、Release から sent_{前日}.txt を取り戻す。
            前日の送信分がすべて hit_record.csv に記録済みなら取り戻さない
            （取り戻すと notify が前日の全レースの結果を毎回取り直すため）。
  save    : 実行の後に、ローカルの sent_*.txt を Release へ保存する。
            保存は x_release_storage.upload_file（同じ key は先に保存された行を残す統合）。
            restore で取り戻したまま変わっていないファイルは保存しない。

当日分（sent_{当日}.txt）は実行の前に取り戻さない。取り戻すと、notify の
送信済みスキップ（notify_arashi.py L3601〜）と1回の実行内の件数制限の結果が変わるため。

失敗しても notify の実行は止めない（終了コードは常に 0。失敗は ::warning:: で表示）。
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone

import x_release_storage as xrs

log = logging.getLogger("x_state_sync")
STATE_FILE = ".state_sync.json"
JST = timezone(timedelta(hours=9))


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _warn(msg: str) -> None:
    log.warning(msg)
    print(f"::warning::{msg}")


def _race_date(arg: str | None) -> str:
    if arg:
        return arg
    return datetime.now(JST).strftime("%Y%m%d")


def _sent_keys(path: str, date: str) -> set[tuple]:
    """sent ファイルのうち、notify が hit_record に記録する対象の (date, venue_num, race, combo)。"""
    keys = set()
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except ValueError:
                obj = {"key": ln}
            parts = str(obj.get("key", "")).replace("_ex", "").split("_")
            if len(parts) != 3:
                continue  # 送信対象0件のマーカーなど（notify も読み飛ばす）
            _, venue_num, race = parts
            keys.add((date, str(venue_num), str(race), obj.get("combo", "")))
    return keys


def _recorded_keys(path: str, date: str) -> set[tuple]:
    keys = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("date") == date:
                keys.add((row.get("date", ""), str(row.get("venue_num", "")),
                          str(row.get("race", "")), row.get("pred_combo", "")))
    return keys


def restore(race_date: str) -> dict:
    y = (datetime.strptime(race_date, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")
    name = f"sent_{y}.txt"
    state = {"race_date": race_date, "restored": {}}
    if os.path.exists(name):
        log.info("%s はローカルにあるため取り戻さない", name)
        return state
    with tempfile.TemporaryDirectory() as td:
        tmp_sent = os.path.join(td, name)
        if not xrs.download_file(name, tmp_sent):
            log.info("%s は Release に無い（前日の記録なし、または初回）", name)
            return state
        want = _sent_keys(tmp_sent, y)
        if not want:
            log.info("%s に記録対象のエントリが無い（マーカーのみ等）。取り戻さない", name)
            return state
        tmp_hit = os.path.join(td, "hit_record.csv")
        if want and xrs.download_file("hit_record.csv", tmp_hit):
            missing = want - _recorded_keys(tmp_hit, y)
            if not missing:
                log.info("%s の %d 件はすべて hit_record.csv に記録済み。取り戻さない", name, len(want))
                state["skipped_complete"] = name
                return state
            log.info("%s: 未記録 %d / %d 件", name, len(missing), len(want))
        os.replace(tmp_sent, name)
    state["restored"][name] = _sha(name)
    log.info("%s を取り戻した", name)
    return state


def save() -> int:
    state = {}
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, ValueError):
            state = {}
    restored = state.get("restored", {})
    failed = 0
    for path in sorted(glob.glob("sent_[0-9]*.txt")):
        name = os.path.basename(path)
        if not xrs.merge_kind(name):
            continue
        if restored.get(name) == _sha(path):
            log.info("%s は取り戻したときから変わっていない。保存しない", name)
            continue
        if xrs.upload_file(path, name):
            log.info("%s を保存した", name)
        else:
            failed += 1
            _warn(f"[state_sync] {name} の Release への保存に失敗（次の実行で再度保存を試みる）")
    return failed


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["restore", "save"])
    p.add_argument("--date", default=None)
    a = p.parse_args(argv)
    try:
        if a.action == "restore":
            st = restore(_race_date(a.date))
            with open(STATE_FILE, "w", encoding="utf-8") as f:
                json.dump(st, f, ensure_ascii=False)
        else:
            save()
    except Exception as e:  # notify の実行は止めない
        _warn(f"[state_sync] {a.action} が失敗: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
