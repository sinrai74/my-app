#!/usr/bin/env python3
"""
schema_version.py  ── hit_record.csv のスキーマバージョン管理

hit_record.csv 自体には「バージョン列」を持たせない（全行に冗長な値が入り、
DictReader の挙動も汚すため）。代わりに、同じディレクトリの隠しファイル
`.hit_record_schema_version` に現在のファイルのスキーマバージョンを1行で
記録する。

起動時チェックの流れ:
  1. 実ファイル hit_record.csv のヘッダーからバージョンを推定
  2. .hit_record_schema_version の記録値と突き合わせ
  3. 期待バージョン(CURRENT_SCHEMA_VERSION)と一致するか判定
"""

from __future__ import annotations

import csv
import os

from csv_schema import (
    CURRENT_SCHEMA_VERSION,
    detect_version_from_header,
)

HIT_RECORD_CSV = "hit_record.csv"
VERSION_FILE = ".hit_record_schema_version"


def read_recorded_version(version_file: str = VERSION_FILE) -> int | None:
    """.hit_record_schema_version に記録されたバージョンを返す（なければNone）。"""
    if not os.path.exists(version_file):
        return None
    try:
        with open(version_file, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def write_recorded_version(version: int, version_file: str = VERSION_FILE) -> None:
    """スキーマバージョンを .hit_record_schema_version に記録する。"""
    with open(version_file, "w", encoding="utf-8") as f:
        f.write(str(version))


def read_csv_header(csv_file: str = HIT_RECORD_CSV) -> list[str] | None:
    """CSVの1行目(ヘッダー)を列名リストで返す。ファイルがなければNone。"""
    if not os.path.exists(csv_file):
        return None
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return None


def detect_current_file_version(csv_file: str = HIT_RECORD_CSV) -> int | None:
    """
    実ファイルの現在のスキーマバージョンを判定する。
    優先順位:
      1. ヘッダーが既知スキーマと完全一致 → そのバージョン
      2. 一致しないが .hit_record_schema_version がある → 記録値
      3. どちらも不明 → None
    """
    header = read_csv_header(csv_file)
    if header is not None:
        v = detect_version_from_header(header)
        if v is not None:
            return v
    # ヘッダーから判定できない場合は記録ファイルを信頼する
    return read_recorded_version()


def check_schema(csv_file: str = HIT_RECORD_CSV) -> dict:
    """
    起動時チェック用。現在の状態と、マイグレーションが必要かどうかを返す。

    戻り値:
        {
            "exists":          bool,   # CSVファイルが存在するか
            "current_version": int|None,
            "expected_version": int,
            "needs_migration": bool,
            "reason":          str,
        }
    """
    expected = CURRENT_SCHEMA_VERSION

    if not os.path.exists(csv_file):
        # ファイル自体がなければ、次回書き込み時に最新スキーマで作られる。
        return {
            "exists": False,
            "current_version": None,
            "expected_version": expected,
            "needs_migration": False,
            "reason": f"{csv_file} が存在しません（初回書き込み時に v{expected} で作成されます）",
        }

    current = detect_current_file_version(csv_file)

    if current is None:
        return {
            "exists": True,
            "current_version": None,
            "expected_version": expected,
            "needs_migration": True,
            "reason": "スキーマバージョンを判定できません（未知のヘッダー）。手動確認が必要です。",
        }

    if current == expected:
        return {
            "exists": True,
            "current_version": current,
            "expected_version": expected,
            "needs_migration": False,
            "reason": f"スキーマは最新(v{expected})です。",
        }

    if current < expected:
        return {
            "exists": True,
            "current_version": current,
            "expected_version": expected,
            "needs_migration": True,
            "reason": f"スキーマが古い(v{current} < v{expected})。マイグレーションが必要です。",
        }

    # current > expected: コードよりファイルが新しい（ダウングレード状況）
    return {
        "exists": True,
        "current_version": current,
        "expected_version": expected,
        "needs_migration": False,
        "reason": f"ファイル(v{current})がコード(v{expected})より新しいです。コードを更新してください。",
    }
