#!/usr/bin/env python3
"""
data_integrity.py  ── CSV読み込み時のデータ整合性チェック

hit_record.csv のような運用データCSVについて、読み込み時・変換後に
「列不足」「列順不一致」「重複列」「欠損列」「型異常」を検査し、
異常があればログへ出力する。サイレントに壊れたデータを使い続けることを防ぐ。

このモジュール単体では例外を投げて処理を止めない（呼び出し側が
ログを見て判断できるようにするため）。ただし check_startup() は
「起動を止めるべきレベルの異常」を bool で明示的に返す。
"""

from __future__ import annotations

import csv
import logging
import os
from typing import Optional

from csv_schema import CURRENT_SCHEMA_VERSION, get_columns

log = logging.getLogger("data_integrity")

# 型チェック対象の列（数値であるべき列）。空欄は許容する（未確定値のため）。
_NUMERIC_COLUMNS = {
    "date", "venue_num", "race", "night",
    "confidence", "pred_prob", "pred_ev", "pred_odds", "upset_score",
    "wind_speed", "wave",
    "payout", "hit", "profit", "n_bets", "cost",
    "purchased", "buyscore", "match_index",
    "feat_win_rate", "feat_avg_st", "feat_course_st_1c", "feat_course_rank_1c",
}


def _is_numeric_or_blank(value: str) -> bool:
    if value in (None, "", "None"):
        return True
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def validate_header(header: list[str], expected_version: int = CURRENT_SCHEMA_VERSION) -> dict:
    """
    ヘッダー(列名リスト)を検査する。

    戻り値:
        {
            "ok": bool,
            "missing_columns": list[str],   # 期待スキーマにあるが実ヘッダーにない列
            "extra_columns":   list[str],   # 実ヘッダーにあるが期待スキーマにない列
            "duplicate_columns": list[str], # ヘッダー内で重複している列名
            "order_mismatch": bool,         # 列は揃っているが順序が違う
        }
    """
    expected = get_columns(expected_version)

    seen = set()
    duplicates = []
    for c in header:
        if c in seen:
            duplicates.append(c)
        seen.add(c)

    missing = [c for c in expected if c not in header]
    extra = [c for c in header if c not in expected]
    order_mismatch = (not missing and not extra and not duplicates and header != expected)

    ok = not missing and not extra and not duplicates and not order_mismatch

    result = {
        "ok": ok,
        "missing_columns": missing,
        "extra_columns": extra,
        "duplicate_columns": duplicates,
        "order_mismatch": order_mismatch,
    }

    if missing:
        log.warning("[整合性] 欠損列を検出: %s", missing)
    if extra:
        log.warning("[整合性] 未知の列を検出: %s", extra)
    if duplicates:
        log.error("[整合性] 重複列を検出（データ破損の可能性）: %s", duplicates)
    if order_mismatch:
        log.info("[整合性] 列は揃っているが順序が期待スキーマと異なります")

    return result


def validate_rows(rows: list[dict], sample_size: Optional[int] = None) -> dict:
    """
    データ行を検査する（型異常のスポットチェック）。
    sample_size を指定すると先頭N件のみ検査する（大きなCSVでの高速化用）。

    戻り値:
        {
            "ok": bool,
            "type_errors": list[dict],  # [{"row": i, "column": c, "value": v}, ...]
            "checked_rows": int,
        }
    """
    target = rows if sample_size is None else rows[:sample_size]
    type_errors = []

    for i, row in enumerate(target):
        for col in _NUMERIC_COLUMNS:
            if col not in row:
                continue
            val = row[col]
            if not _is_numeric_or_blank(val):
                type_errors.append({"row": i, "column": col, "value": val})

    if type_errors:
        log.warning("[整合性] 型異常を %d件 検出（先頭5件: %s）", len(type_errors), type_errors[:5])

    return {
        "ok": not type_errors,
        "type_errors": type_errors,
        "checked_rows": len(target),
    }


def validate_csv_file(csv_file: str, expected_version: int = CURRENT_SCHEMA_VERSION,
                       sample_size: Optional[int] = 500) -> dict:
    """
    CSVファイル全体（ヘッダー＋データ）を検査する。読み込み時・マイグレーション後
    どちらからも呼べる統合エントリ。

    戻り値:
        {
            "ok": bool,
            "exists": bool,
            "header_check": dict | None,
            "row_check": dict | None,
            "total_rows": int,
        }
    """
    if not os.path.exists(csv_file):
        log.info("[整合性] %s が存在しないためチェックをスキップ", csv_file)
        return {"ok": True, "exists": False, "header_check": None, "row_check": None, "total_rows": 0}

    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            log.error("[整合性] %s は空ファイルです", csv_file)
            return {"ok": False, "exists": True, "header_check": None, "row_check": None, "total_rows": 0}

    header_check = validate_header(header, expected_version)

    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    row_check = validate_rows(rows, sample_size=sample_size)

    ok = header_check["ok"] and row_check["ok"]
    if ok:
        log.info("[整合性] %s: 異常なし（%d件）", csv_file, len(rows))

    return {
        "ok": ok,
        "exists": True,
        "header_check": header_check,
        "row_check": row_check,
        "total_rows": len(rows),
    }
