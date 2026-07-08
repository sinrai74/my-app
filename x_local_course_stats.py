#!/usr/bin/env python3
"""
x_local_course_stats.py  ── 当地コース別成績 独立モジュール

【独立モジュール】既存Bot（notify_arashi.py・x_ranking.py・x_asahi_scoring.py等）
には一切組み込まれていない。既存コードへの統合は行わず、
本モジュール単体で完結する（design.md 参照）。

【生成するファイル】
  k_race_history.csv       Kファイルから抽出した「選手×レース」の生履歴（永続蓄積）
  local_course_stats.csv   選手×場×コース別の集計結果（k_race_historyから毎回再生成）
  k_race_history_progress.json  初回一括構築の進捗（中断・再開用）

【使い方】
  # 初回一括構築（2023年〜現在まで、指定フォルダ内の全Kファイルを処理）
  python x_local_course_stats.py --init --k-dir /path/to/k_files

  # 毎日の更新（前日分のKファイル1つを追記）
  python x_local_course_stats.py --daily --k-file /path/to/K250708.TXT

  # 集計のみ再実行（k_race_history.csv は変更せず local_course_stats.csv だけ再生成）
  python x_local_course_stats.py --rebuild-only
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from pathlib import Path

from x_kfile_race_parser import parse_k_race_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("x_local_course_stats")

K_RACE_HISTORY_CSV   = "k_race_history.csv"
LOCAL_COURSE_STATS_CSV = "local_course_stats.csv"
PROGRESS_FILE        = "k_race_history_progress.json"

HISTORY_FIELDNAMES = [
    "date", "venue_code", "venue_name", "race_no",
    "racer_no", "racer_name", "boat_no", "course", "order",
    "motor_no", "boat_equip_no", "exhibition_time", "start_timing",
    "race_time", "source_file",
]

STATS_FIELDNAMES = [
    "racer_no", "venue_code", "venue_name", "course",
    "starts", "first", "second", "third", "fourth", "fifth", "sixth",
    "first_rate", "top2_rate", "top3_rate", "last_updated",
]


# ════════════════════════════════════════════════════════════
# 進捗管理（初回一括構築の中断・再開用）
# ════════════════════════════════════════════════════════════

def _load_progress() -> dict:
    if not os.path.exists(PROGRESS_FILE):
        return {"processed_files": []}
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"processed_files": []}


def _save_progress(progress: dict) -> None:
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


# ════════════════════════════════════════════════════════════
# 履歴CSVへの追記（重複除去つき）
# ════════════════════════════════════════════════════════════

def _dedup_key(rec: dict) -> tuple:
    return (rec["date"], rec["venue_code"], rec["race_no"], rec["racer_no"])


def _load_existing_keys() -> set:
    """k_race_history.csv に既に存在するレコードのキー集合を返す（重複防止用）。"""
    keys: set = set()
    if not os.path.exists(K_RACE_HISTORY_CSV):
        return keys
    with open(K_RACE_HISTORY_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                keys.add((int(row["date"]), row["venue_code"], int(row["race_no"]), row["racer_no"]))
            except (KeyError, ValueError):
                continue
    return keys


def append_to_history(records: list[dict]) -> int:
    """
    records を k_race_history.csv に重複除去して追記する。
    実際に追記された件数を返す。
    """
    if not records:
        return 0

    existing_keys = _load_existing_keys()
    new_records = [r for r in records if _dedup_key(r) not in existing_keys]

    if not new_records:
        log.info("[履歴] 追記対象なし（すべて既存レコードと重複）")
        return 0

    write_header = not os.path.exists(K_RACE_HISTORY_CSV)
    with open(K_RACE_HISTORY_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(new_records)

    log.info("[履歴] %d件追記（%d件は重複のためスキップ）", len(new_records), len(records) - len(new_records))
    return len(new_records)


# ════════════════════════════════════════════════════════════
# 集計（local_course_stats.csv の再生成）
# ════════════════════════════════════════════════════════════

def rebuild_local_course_stats(today: int | None = None) -> int:
    """
    k_race_history.csv 全体を読み込み、選手×場×コース別に集計して
    local_course_stats.csv を洗い替え生成する。
    生成した行数を返す。

    【差分更新ではなく毎回全体再集計する理由】design.md 4節参照。
    データ量が将来大きく育った場合はこの関数の設計を見直すこと。
    """
    if not os.path.exists(K_RACE_HISTORY_CSV):
        log.warning("[集計] %s が存在しません", K_RACE_HISTORY_CSV)
        return 0

    # (racer_no, venue_code, course) -> {1: count, 2: count, ..., 6: count}
    agg: dict[tuple, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    venue_names: dict[str, str] = {}

    with open(K_RACE_HISTORY_CSV, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            order_raw = row.get("order", "")
            if not order_raw.isdigit():
                # フライング等の異常着順は「出走」としては数えるが、着順集計には使わない
                # （出走数の扱いは要件次第で調整可能。ここでは異常着順は starts にのみ加算する）
                pass
            try:
                course = int(row["course"])
                racer_no = row["racer_no"]
                venue_code = row["venue_code"]
            except (KeyError, ValueError):
                continue
            if not (1 <= course <= 6):
                continue

            venue_names[venue_code] = row.get("venue_name", f"場{venue_code}")
            key = (racer_no, venue_code, course)
            agg[key]["starts"] = agg[key].get("starts", 0) + 1
            if order_raw.isdigit():
                place = int(order_raw)
                if 1 <= place <= 6:
                    agg[key][place] = agg[key].get(place, 0) + 1

    today = today or 0
    rows_out = []
    for (racer_no, venue_code, course), counts in agg.items():
        starts = counts.get("starts", 0)
        if starts == 0:
            continue
        first  = counts.get(1, 0)
        second = counts.get(2, 0)
        third  = counts.get(3, 0)
        fourth = counts.get(4, 0)
        fifth  = counts.get(5, 0)
        sixth  = counts.get(6, 0)
        rows_out.append({
            "racer_no":   racer_no,
            "venue_code": venue_code,
            "venue_name": venue_names.get(venue_code, f"場{venue_code}"),
            "course":     course,
            "starts":     starts,
            "first":      first,
            "second":     second,
            "third":      third,
            "fourth":     fourth,
            "fifth":      fifth,
            "sixth":      sixth,
            "first_rate": round(first / starts * 100, 1),
            "top2_rate":  round((first + second) / starts * 100, 1),
            "top3_rate":  round((first + second + third) / starts * 100, 1),
            "last_updated": today,
        })

    tmp_path = LOCAL_COURSE_STATS_CSV + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STATS_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows_out)
    os.replace(tmp_path, LOCAL_COURSE_STATS_CSV)

    log.info("[集計] %s を再生成しました（%d行）", LOCAL_COURSE_STATS_CSV, len(rows_out))
    return len(rows_out)


# ════════════════════════════════════════════════════════════
# 初回一括構築
# ════════════════════════════════════════════════════════════

def init_build(k_dir: str, today: int | None = None) -> None:
    """指定フォルダ内の全Kファイルを日付順に処理し、履歴を一括構築する。"""
    k_dir_path = Path(k_dir)
    if not k_dir_path.is_dir():
        log.error("[初回構築] フォルダが見つかりません: %s", k_dir)
        return

    all_files = sorted(k_dir_path.glob("K*.TXT")) + sorted(k_dir_path.glob("K*.txt"))
    all_files = sorted(set(all_files))
    if not all_files:
        log.warning("[初回構築] Kファイルが見つかりません: %s", k_dir)
        return

    progress = _load_progress()
    processed = set(progress.get("processed_files", []))
    remaining = [f for f in all_files if f.name not in processed]

    log.info("[初回構築] 対象ファイル数=%d（処理済み=%d、残り=%d）",
              len(all_files), len(processed), len(remaining))

    total_appended = 0
    for i, fp in enumerate(remaining, 1):
        records = parse_k_race_file(fp)
        appended = append_to_history(records)
        total_appended += appended
        processed.add(fp.name)

        # 【中断・再開対応】一定件数ごとに進捗を保存する
        if i % 50 == 0 or i == len(remaining):
            progress["processed_files"] = sorted(processed)
            _save_progress(progress)
            log.info("[初回構築] 進捗 %d/%d ファイル処理済み", i, len(remaining))

    log.info("[初回構築] 完了。累計追記件数=%d", total_appended)

    # 全ファイル処理が終わってから最後に1回だけ集計する
    rebuild_local_course_stats(today=today)


# ════════════════════════════════════════════════════════════
# 毎日の更新
# ════════════════════════════════════════════════════════════

def daily_update(k_file: str, today: int | None = None) -> None:
    """前日分のKファイル1つを処理し、履歴に追記後、集計を再生成する。"""
    if not os.path.exists(k_file):
        log.error("[日次更新] ファイルが見つかりません: %s", k_file)
        return
    records = parse_k_race_file(k_file)
    append_to_history(records)
    rebuild_local_course_stats(today=today)


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="当地コース別成績 独立モジュール")
    parser.add_argument("--init", action="store_true", help="初回一括構築モード")
    parser.add_argument("--daily", action="store_true", help="日次更新モード")
    parser.add_argument("--rebuild-only", action="store_true", help="集計のみ再実行")
    parser.add_argument("--k-dir", type=str, help="--init 用: Kファイルが格納されたフォルダ")
    parser.add_argument("--k-file", type=str, help="--daily 用: 処理する単一のKファイル")
    parser.add_argument("--today", type=int, default=None, help="last_updated に記録する日付(YYYYMMDD)")
    args = parser.parse_args()

    if args.init:
        if not args.k_dir:
            parser.error("--init には --k-dir の指定が必須です")
        init_build(args.k_dir, today=args.today)
    elif args.daily:
        if not args.k_file:
            parser.error("--daily には --k-file の指定が必須です")
        daily_update(args.k_file, today=args.today)
    elif args.rebuild_only:
        rebuild_local_course_stats(today=args.today)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
