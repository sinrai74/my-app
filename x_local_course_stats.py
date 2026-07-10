#!/usr/bin/env python3
"""
x_local_course_stats.py  ── 当地コース別成績 独立モジュール

【独立モジュール】既存Bot（notify_arashi.py・x_ranking.py・x_asahi_scoring.py等）
には一切組み込まれていない。既存コードへの統合は行わず、
本モジュール単体で完結する（local_course_stats_design.md 参照）。

【生成するファイル】
  k_race_history.csv       Kファイルから抽出した「選手×レース」の生履歴（永続蓄積・唯一の正）
  local_course_stats.csv   選手×場×コース別の集計結果（k_race_historyから毎回再生成）
  k_race_history_progress.json  初回一括構築の進捗（中断・再開用）
  .k_race_history_schema_version  実ファイルのスキーマバージョン記録

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
import time
from collections import defaultdict, Counter
from pathlib import Path

from x_kfile_race_parser import parse_k_race_file_with_stats
import k_race_history_schema as _schema
import k_race_history_integrity as _integrity

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("x_local_course_stats")

K_RACE_HISTORY_CSV   = "k_race_history.csv"
LOCAL_COURSE_STATS_CSV = "local_course_stats.csv"
PROGRESS_FILE        = "k_race_history_progress.json"

# 【単一の真実の源】列定義は k_race_history_schema.py が正。
# ここではハードコードせず、常にスキーマモジュールの現在バージョンを参照する。
HISTORY_FIELDNAMES = _schema.current_columns()

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


def append_to_history(records: list[dict], existing_keys: set | None = None) -> dict:
    """
    records を k_race_history.csv に重複除去して追記する。

    existing_keys を渡した場合はディスクからの再読み込みを行わず、
    渡されたset を直接更新する（3年分などファイル数が多い一括構築で
    「ファイルごとに全件再読込」というO(n^2)の劣化を避けるため）。
    渡さなかった場合は従来通り自前で読み込む（後方互換）。

    書き込み前に必ずスキーマチェックを行う（将来列が増えた場合、
    既存の古いスキーマのファイルへ新しい列構成で書き込んでしまう
    事故を防ぐため）。判定不能な場合はサイレントに続行せず、
    追記を中止する。

    戻り値: {"appended": int, "duplicates": int, "fatal": bool}
    """
    if not records:
        return {"appended": 0, "duplicates": 0, "fatal": False}

    schema_status = _schema.ensure_schema(K_RACE_HISTORY_CSV, auto=True)
    if schema_status.get("fatal"):
        log.error(
            "[履歴] k_race_history.csv のスキーマ状態が安全に判定できません: %s "
            "→ 追記を中止します。`python k_race_history_schema.py` 相当の手動確認が必要です。",
            schema_status.get("reason", "詳細不明"),
        )
        return {"appended": 0, "duplicates": 0, "fatal": True}
    if schema_status.get("migration"):
        log.info("[履歴] k_race_history.csv をマイグレーションしました: %s",
                  schema_status["migration"].get("reason"))

    if existing_keys is None:
        existing_keys = _load_existing_keys()
    new_records = [r for r in records if _dedup_key(r) not in existing_keys]
    duplicates = len(records) - len(new_records)

    if not new_records:
        log.info("[履歴] 追記対象なし（すべて既存レコードと重複）")
        return {"appended": 0, "duplicates": duplicates, "fatal": False}

    write_header = not os.path.exists(K_RACE_HISTORY_CSV)
    with open(K_RACE_HISTORY_CSV, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(new_records)

    # existing_keys を呼び出し元と共有している場合、次のファイル処理でも
    # 今回追記した分が「既存」として認識されるよう、ここで更新しておく。
    for r in new_records:
        existing_keys.add(_dedup_key(r))

    log.info("[履歴] %d件追記（%d件は重複のためスキップ）", len(new_records), duplicates)
    return {"appended": len(new_records), "duplicates": duplicates, "fatal": False}


# ════════════════════════════════════════════════════════════
# 集計（local_course_stats.csv の再生成）
# ════════════════════════════════════════════════════════════

def rebuild_local_course_stats(today: int | None = None) -> int:
    """
    k_race_history.csv 全体を読み込み、選手×場×コース別に集計して
    local_course_stats.csv を洗い替え生成する。
    生成した行数を返す。

    【履歴DBが唯一の正】この関数は k_race_history.csv から毎回
    まるごと再計算する。local_course_stats.csv 自体は手で編集
    しないこと（次回実行時に無条件で上書きされる）。

    【差分更新ではなく毎回全体再集計する理由】design.md 4節参照。
    データ量が将来大きく育った場合はこの関数の設計を見直すこと。
    """
    if not os.path.exists(K_RACE_HISTORY_CSV):
        log.warning("[集計] %s が存在しません", K_RACE_HISTORY_CSV)
        return 0

    schema_status = _schema.ensure_schema(K_RACE_HISTORY_CSV, auto=True)
    if schema_status.get("fatal"):
        log.error(
            "[集計] k_race_history.csv のスキーマ状態が安全に判定できません: %s → 集計を中止します。",
            schema_status.get("reason", "詳細不明"),
        )
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


def print_local_course_stats_summary(csv_file: str = LOCAL_COURSE_STATS_CSV) -> dict:
    """
    Phase2向け: local_course_stats.csv 生成後の統計サマリーを表示する。
    選手数・開催場数・コース別件数・平均/最大/最小出走数を出力する。
    """
    if not os.path.exists(csv_file):
        log.warning("[Phase2] %s が存在しません", csv_file)
        return {}

    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    racers = set(r["racer_no"] for r in rows)
    venues = set(r["venue_code"] for r in rows)
    course_counts = Counter(r["course"] for r in rows)
    starts_list = [int(r["starts"]) for r in rows if r.get("starts", "").isdigit()]

    summary = {
        "total_rows": len(rows),
        "racer_count": len(racers),
        "venue_count": len(venues),
        "course_counts": dict(sorted(course_counts.items())),
        "avg_starts": round(sum(starts_list) / len(starts_list), 2) if starts_list else 0,
        "max_starts": max(starts_list) if starts_list else 0,
        "min_starts": min(starts_list) if starts_list else 0,
    }

    print("=" * 60)
    print(" local_course_stats.csv 統計サマリー（Phase2）")
    print("=" * 60)
    print(f" 総行数（選手×場×コース） : {summary['total_rows']}")
    print(f" 選手数（ユニーク）        : {summary['racer_count']}")
    print(f" 開催場数（ユニーク）      : {summary['venue_count']}")
    print(" コース別件数              :")
    for course, count in summary["course_counts"].items():
        print(f"   {course}コース: {count}件")
    print(f" 平均出走数                : {summary['avg_starts']}")
    print(f" 最大出走数                : {summary['max_starts']}")
    print(f" 最小出走数                : {summary['min_starts']}")
    print("=" * 60)

    return summary


# ════════════════════════════════════════════════════════════
# 品質レポート
# ════════════════════════════════════════════════════════════

def _new_report() -> dict:
    """処理開始時に呼ぶ。品質レポート集計用の初期状態を返す。"""
    return {
        "files_total": 0,
        "files_processed": 0,
        "files_error": 0,
        "races_seen": set(),      # (date, venue_code, race_no)
        "racers_seen": set(),     # racer_no
        "records_generated": 0,   # append_to_history で実際に追記された件数
        "duplicates_skipped": 0,
        "skipped_invalid_course": 0,  # 欠場等でコース情報が読み取れず除外した件数
        "parse_errors": 0,
        "venue_counts": defaultdict(int),  # venue_name -> 抽出レコード数
        "fatal": False,
        "start_time": time.time(),
    }


def _process_one_file(fp: "str | Path", report: dict, existing_keys: set) -> None:
    """1つのKファイルを処理し、report を書き換える（副作用のみ、戻り値なし）。"""
    report["files_total"] += 1
    try:
        records, stats = parse_k_race_file_with_stats(fp)
    except Exception as e:
        report["files_error"] += 1
        log.error("[品質] ファイル処理中にエラー: %s (%s)", fp, e)
        return

    report["files_processed"] += 1
    report["races_seen"] |= stats["races_seen"]
    report["racers_seen"] |= stats["racers_seen"]
    report["skipped_invalid_course"] += stats["skipped_invalid_course"]
    report["parse_errors"] += stats["parse_errors"]
    for r in records:
        report["venue_counts"][r["venue_name"]] += 1

    result = append_to_history(records, existing_keys=existing_keys)
    report["records_generated"] += result["appended"]
    report["duplicates_skipped"] += result["duplicates"]
    if result["fatal"]:
        report["fatal"] = True


def print_quality_report(report: dict, integrity_result: "dict | None" = None) -> bool:
    """
    品質レポートを表示する。「正常終了」判定結果（bool）を返す。
    正常終了の条件: files_error == 0, fatal でない, 整合性チェックがある場合は ok であること。
    """
    elapsed = time.time() - report["start_time"]
    is_ok = (
        report["files_error"] == 0
        and not report["fatal"]
        and (integrity_result is None or integrity_result.get("ok", True))
    )

    print("=" * 60)
    print(" k_race_history.csv 品質レポート")
    print("=" * 60)
    print(f" 処理ファイル数     : {report['files_total']}（成功 {report['files_processed']} / 失敗 {report['files_error']}）")
    print(f" 処理レース数       : {len(report['races_seen'])}")
    print(f" 処理選手数（延べ） : {len(report['racers_seen'])}")
    print(f" 生成レコード数     : {report['records_generated']}")
    print(f" 重複スキップ件数   : {report['duplicates_skipped']}")
    print(f" 欠場スキップ件数   : {report['skipped_invalid_course']}")
    print(f" パース失敗件数     : {report['parse_errors']}")
    print(f" エラー件数（ファイル単位）: {report['files_error']}")
    print(" 開催場別件数       :")
    for venue, count in sorted(report["venue_counts"].items(), key=lambda kv: -kv[1]):
        print(f"   {venue}: {count}件")
    print(f" 処理時間           : {elapsed:.2f}秒")
    if integrity_result is not None:
        print(f" データ整合性チェック: {'OK' if integrity_result.get('ok') else '異常あり（詳細はログ参照）'}")
    print("=" * 60)
    print(f" 結果: {'✅ 正常終了' if is_ok else '⚠️ 異常終了（ログを確認してください）'}")
    print("=" * 60)
    return is_ok


# ════════════════════════════════════════════════════════════
# 初回一括構築
# ════════════════════════════════════════════════════════════

def init_build(k_dir: str, today: int | None = None) -> dict:
    """
    指定フォルダ内の全Kファイルを日付順に処理し、履歴を一括構築する。
    最後に品質レポートを表示する。レポート辞書を返す。
    """
    k_dir_path = Path(k_dir)
    if not k_dir_path.is_dir():
        log.error("[初回構築] フォルダが見つかりません: %s", k_dir)
        return {}

    all_files = sorted(k_dir_path.glob("K*.TXT")) + sorted(k_dir_path.glob("K*.txt"))
    all_files = sorted(set(all_files))
    if not all_files:
        log.warning("[初回構築] Kファイルが見つかりません: %s", k_dir)
        return {}

    progress = _load_progress()
    processed = set(progress.get("processed_files", []))
    remaining = [f for f in all_files if f.name not in processed]

    log.info("[初回構築] 対象ファイル数=%d（処理済み=%d、残り=%d）",
              len(all_files), len(processed), len(remaining))

    report = _new_report()
    existing_keys = _load_existing_keys()  # ループの外で1回だけ読み込む（O(n^2)防止）

    for i, fp in enumerate(remaining, 1):
        _process_one_file(fp, report, existing_keys)
        processed.add(fp.name)

        # 【中断・再開対応】一定件数ごとに進捗を保存する
        if i % 50 == 0 or i == len(remaining):
            progress["processed_files"] = sorted(processed)
            _save_progress(progress)
            log.info("[初回構築] 進捗 %d/%d ファイル処理済み", i, len(remaining))

    log.info("[初回構築] 完了。累計追記件数=%d", report["records_generated"])

    # 全ファイル処理が終わってから最後に1回だけ集計する
    rebuild_local_course_stats(today=today)
    print_local_course_stats_summary()

    integrity_result = _integrity.validate_history(K_RACE_HISTORY_CSV)
    print_quality_report(report, integrity_result)
    return report


# ════════════════════════════════════════════════════════════
# 毎日の更新
# ════════════════════════════════════════════════════════════

def daily_update(k_file: str, today: int | None = None) -> dict:
    """前日分のKファイル1つを処理し、履歴に追記後、集計を再生成する。品質レポートを返す。"""
    if not os.path.exists(k_file):
        log.error("[日次更新] ファイルが見つかりません: %s", k_file)
        return {}

    report = _new_report()
    existing_keys = _load_existing_keys()
    _process_one_file(k_file, report, existing_keys)

    rebuild_local_course_stats(today=today)

    integrity_result = _integrity.validate_history(K_RACE_HISTORY_CSV)
    print_quality_report(report, integrity_result)
    return report


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
        print_local_course_stats_summary()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
