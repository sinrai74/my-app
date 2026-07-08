#!/usr/bin/env python3
"""
k_race_history_schema.py  ── k_race_history.csv スキーマ管理・マイグレーション

【独立モジュール】既存Bot（hit_record.csv用の csv_schema.py / migration.py 等）
とは完全に独立している。import依存も一切ない。k_race_history.csv 専用。

────────────────────────────────────────────────────────────
スキーマを変更するときの手順（将来の自分へ）
────────────────────────────────────────────────────────────
1. CURRENT_SCHEMA_VERSION をインクリメントする（例: 1 → 2）。
2. SCHEMA_COLUMNS に新バージョンの完全な列リストを追加する。
3. NEW_COLUMN_DEFAULTS に「新しく増えた列」の初期値だけ追加する。
4. MIGRATIONS に (旧→新) の変換関数を1つ追加する。
   → k_race_history.csv を手作業で編集する運用は禁止。
     必ず ensure_schema() 経由のマイグレーションで移行する。

k_race_history.csv は「唯一の正（Single Source of Truth）」であり、
local_course_stats.csv はそこから毎回まるごと再生成される派生データ
に過ぎない（本ファイルのスキーマ変更が集計ロジック側に影響しても、
再生成すれば必ず最新の列定義に追従する）。
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
from datetime import datetime, timedelta, timezone

log = logging.getLogger("k_race_history_schema")

JST = timezone(timedelta(hours=9))

K_RACE_HISTORY_CSV = "k_race_history.csv"
VERSION_FILE = ".k_race_history_schema_version"

# ════════════════════════════════════════════════════════════
# 現在の期待スキーマバージョン
# ════════════════════════════════════════════════════════════
CURRENT_SCHEMA_VERSION = 1

# ════════════════════════════════════════════════════════════
# バージョン別 完全列定義
# ════════════════════════════════════════════════════════════
# Version 1: 初期スキーマ（Kファイル選手別詳細行の基本項目）
_COLUMNS_V1 = [
    "date", "venue_code", "venue_name", "race_no",
    "racer_no", "racer_name", "boat_no", "course", "order",
    "motor_no", "boat_equip_no", "exhibition_time", "start_timing",
    "race_time", "source_file",
]

SCHEMA_COLUMNS: dict[int, list[str]] = {
    1: _COLUMNS_V1,
}

# ════════════════════════════════════════════════════════════
# 新規追加列の初期値（旧データを変換するときの補完値）
# ════════════════════════════════════════════════════════════
# 「あるバージョンで初めて登場した列」の初期値をここで定義する。
# 例（将来 Version2 で "weather" 列を追加する場合）:
#   NEW_COLUMN_DEFAULTS["weather"] = ""
NEW_COLUMN_DEFAULTS: dict[str, str] = {
    # v1 → v2 以降、ここに追記していく
}


def get_columns(version: int) -> list[str]:
    """指定バージョンの完全列リストを返す。"""
    if version not in SCHEMA_COLUMNS:
        raise ValueError(f"未知のスキーマバージョン: {version}")
    return list(SCHEMA_COLUMNS[version])


def current_columns() -> list[str]:
    """現在の期待スキーマの完全列リストを返す。"""
    return get_columns(CURRENT_SCHEMA_VERSION)


def detect_version_from_header(header: list[str]) -> "int | None":
    """CSVヘッダー(列名リスト)からスキーマバージョンを推定する。完全一致するもののみ。"""
    for version, cols in SCHEMA_COLUMNS.items():
        if header == cols:
            return version
    return None


# ════════════════════════════════════════════════════════════
# マイグレーション登録表（チェーン）
# ════════════════════════════════════════════════════════════
# (from_version, to_version): 変換関数
# 将来 Version2 を追加する際は、ここに以下のように1行足す:
#   MIGRATIONS[(1, 2)] = migrate_v1_to_v2
MIGRATIONS: dict[tuple[int, int], "callable"] = {}


# ════════════════════════════════════════════════════════════
# バージョン記録ファイルの読み書き
# ════════════════════════════════════════════════════════════

def read_recorded_version() -> "int | None":
    if not os.path.exists(VERSION_FILE):
        return None
    try:
        with open(VERSION_FILE, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except (ValueError, OSError):
        return None


def write_recorded_version(version: int) -> None:
    with open(VERSION_FILE, "w", encoding="utf-8") as f:
        f.write(str(version))


def read_csv_header(csv_file: str = K_RACE_HISTORY_CSV) -> "list[str] | None":
    if not os.path.exists(csv_file):
        return None
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return None


def detect_current_file_version(csv_file: str = K_RACE_HISTORY_CSV) -> "int | None":
    header = read_csv_header(csv_file)
    if header is not None:
        v = detect_version_from_header(header)
        if v is not None:
            return v
    return read_recorded_version()


# ════════════════════════════════════════════════════════════
# 起動時チェック
# ════════════════════════════════════════════════════════════

def check_schema(csv_file: str = K_RACE_HISTORY_CSV) -> dict:
    """
    現在の状態と、マイグレーションが必要かどうかを返す。
    戻り値: {"exists", "current_version", "expected_version",
             "needs_migration", "reason"}
    """
    expected = CURRENT_SCHEMA_VERSION

    if not os.path.exists(csv_file):
        return {
            "exists": False, "current_version": None, "expected_version": expected,
            "needs_migration": False,
            "reason": f"{csv_file} が存在しません（初回書き込み時に v{expected} で作成されます）",
        }

    current = detect_current_file_version(csv_file)

    if current is None:
        return {
            "exists": True, "current_version": None, "expected_version": expected,
            "needs_migration": True,
            "reason": "スキーマバージョンを判定できません（未知のヘッダー）。手動確認が必要です。",
        }

    if current == expected:
        return {
            "exists": True, "current_version": current, "expected_version": expected,
            "needs_migration": False, "reason": f"スキーマは最新(v{expected})です。",
        }

    if current < expected:
        return {
            "exists": True, "current_version": current, "expected_version": expected,
            "needs_migration": True,
            "reason": f"スキーマが古い(v{current} < v{expected})。マイグレーションが必要です。",
        }

    return {
        "exists": True, "current_version": current, "expected_version": expected,
        "needs_migration": False,
        "reason": f"ファイル(v{current})がコード(v{expected})より新しいです。コードを更新してください。",
    }


# ════════════════════════════════════════════════════════════
# バックアップ
# ════════════════════════════════════════════════════════════

def make_backup(csv_file: str = K_RACE_HISTORY_CSV) -> str:
    """k_race_history_backup_YYYYMMDD_HHMMSS.csv を作成し、パスを返す。"""
    ts = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(csv_file))[0]
    backup_name = f"{base}_backup_{ts}.csv"
    backup_path = os.path.join(os.path.dirname(os.path.abspath(csv_file)) or ".", backup_name)
    shutil.copy2(csv_file, backup_path)
    return backup_path


# ════════════════════════════════════════════════════════════
# マイグレーション実行
# ════════════════════════════════════════════════════════════

def _build_migration_path(from_v: int, to_v: int) -> list[tuple[int, int]]:
    path = []
    v = from_v
    while v < to_v:
        step = (v, v + 1)
        if step not in MIGRATIONS:
            raise RuntimeError(f"マイグレーション {step[0]}→{step[1]} が未登録です")
        path.append(step)
        v += 1
    return path


def _read_rows(csv_file: str) -> list[dict]:
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows_atomic(csv_file: str, columns: list[str], rows: list[dict]) -> None:
    dir_ = os.path.dirname(os.path.abspath(csv_file)) or "."
    tmp_path = os.path.join(dir_, os.path.basename(csv_file) + ".tmp")
    try:
        with open(tmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c, "") for c in columns})
        os.replace(tmp_path, csv_file)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def run_migration(csv_file: str = K_RACE_HISTORY_CSV, target_version: int = CURRENT_SCHEMA_VERSION) -> dict:
    """k_race_history.csv を target_version まで段階マイグレーションする。"""
    status = check_schema(csv_file)

    if not status["exists"]:
        return {"performed": False, "reason": status["reason"]}

    from_v = status["current_version"]
    if from_v is None:
        return {"performed": False, "reason": "現在のスキーマバージョンを判定できないため中止しました。"}

    if from_v >= target_version:
        return {"performed": False, "reason": f"変換不要（現在 v{from_v} ≧ 目標 v{target_version}）"}

    path = _build_migration_path(from_v, target_version)
    rows = _read_rows(csv_file)
    old_columns = get_columns(from_v)
    new_columns = get_columns(target_version)
    added_columns = [c for c in new_columns if c not in old_columns]

    for (a, b) in path:
        rows = MIGRATIONS[(a, b)](rows)

    backup_path = make_backup(csv_file)
    try:
        _write_rows_atomic(csv_file, new_columns, rows)
        write_recorded_version(target_version)
    except Exception as e:
        shutil.copy2(backup_path, csv_file)
        return {"performed": False, "reason": f"変換失敗: {e}（バックアップから復元しました）"}

    return {
        "performed": True, "total_records": len(rows),
        "old_version": from_v, "new_version": target_version,
        "added_columns": added_columns, "backup_path": backup_path,
        "reason": "マイグレーション成功。",
    }


def ensure_schema(csv_file: str = K_RACE_HISTORY_CSV, auto: bool = True) -> dict:
    """
    起動時に呼び出す想定。スキーマが最新なら何もしない。古ければ自動マイグレーション。
    バージョン判定不能（未知のヘッダー）の場合は fatal=True を返す
    （呼び出し側はサイレントに動作を続けず、必要なら処理を止めること）。
    """
    status = check_schema(csv_file)
    status["fatal"] = False

    if not status["needs_migration"] and status["exists"] and status["current_version"] == CURRENT_SCHEMA_VERSION:
        write_recorded_version(CURRENT_SCHEMA_VERSION)
        return status

    if status["needs_migration"]:
        if status["current_version"] is None:
            status["fatal"] = True
            return status
        if auto:
            status["migration"] = run_migration(csv_file)
            if not status["migration"]["performed"]:
                status["fatal"] = True

    return status
