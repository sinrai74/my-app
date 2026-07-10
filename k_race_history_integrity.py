#!/usr/bin/env python3
"""
k_race_history_integrity.py  ── k_race_history.csv データ整合性チェック

【独立モジュール】既存Bot（data_integrity.py 等）とは完全に独立している。
import依存も一切ない。k_race_history.csv 専用の整合性検証を行う。

CSV生成（初回一括構築・日次更新）のたびに自動で以下を検証する:
  ・登録番号(racer_no)が空でないこと
  ・進入コース(course)が1〜6であること
  ・着順(order)が妥当（1〜6の数値、または既知の異常記号）であること
  ・展示タイム(exhibition_time)の形式（数値または空）
  ・ST(start_timing)の形式（数値または空）
  ・重複キー（date, venue_code, race_no, racer_no）が存在しないこと

異常があれば logging 経由でログに出力する（サイレントに握りつぶさない）。
"""

from __future__ import annotations

import csv
import logging
import os
from collections import Counter

log = logging.getLogger("k_race_history_integrity")

K_RACE_HISTORY_CSV = "k_race_history.csv"

# 着順として妥当な異常記号（欠場・フライング・出遅れ・失格等）。
# 実データで確認済みなのは "K0"（欠場）と "0"（下記）。他は将来の実例
# 発生に備えた一般的な想定値（PC-KYOTEIコード表・一般公開情報に基づく）。
#
# "0": レース不成立時に、フライングしなかった艇へ付与されるコード。
#      実データ確認済み（3年分構築時、K20230202.TXT 5R で発見）:
#        F 2 3475 ...  F0.01
#        F 3 4926 ...  F0.02
#        F 4 3726 ...  F0.04
#        F 5 3681 ...  F0.02
#        00 6 4950 ...  0.01      ← フライングしなかった6号艇の着順欄が "00"
#        レース不成立
#      x_kfile_race_parser.py の _parse_order() は "00".isdigit() が True
#      のため int("00") = 0 に変換する。パーサーのバグではなく、実データの
#      特殊コードを正しく拾えている。
_KNOWN_ABNORMAL_ORDERS = {"F", "L", "K", "S", "K0", "K1", "L0", "L1", "S0", "S1", "S2", "0"}


def _is_valid_order(raw: str) -> bool:
    raw = (raw or "").strip()
    # 既知の異常記号（"0"を含む）を先に判定する。
    # "0" は raw.isdigit() が True になるため、isdigit分岐を先に
    # 評価してしまうと 1<=int(raw)<=6 が False になり、
    # _KNOWN_ABNORMAL_ORDERS に "0" を追加しても絶対に参照されない
    # （早期returnで判定順序が壊れる）バグがあったため、判定順序を
    # 入れ替えてある。
    if raw in _KNOWN_ABNORMAL_ORDERS:
        return True
    if raw.isdigit():
        return 1 <= int(raw) <= 6
    return False


def _is_valid_float_or_blank(raw: str) -> bool:
    raw = (raw or "").strip()
    if raw == "":
        return True
    try:
        float(raw)
        return True
    except ValueError:
        return False


def validate_history(csv_file: str = K_RACE_HISTORY_CSV, sample_size: int = 5) -> dict:
    """
    k_race_history.csv 全体を検証する。

    戻り値: {
      "ok": bool, "exists": bool, "total_rows": int,
      "empty_racer_no": int, "invalid_course": int, "invalid_order": int,
      "invalid_exhibition_time": int, "invalid_start_timing": int,
      "duplicate_keys": int,
      "samples": {各項目名: [先頭sample_size件の行の主要キー]},
    }
    """
    result = {
        "ok": True, "exists": False, "total_rows": 0,
        "empty_racer_no": 0, "invalid_course": 0, "invalid_order": 0,
        "invalid_exhibition_time": 0, "invalid_start_timing": 0,
        "duplicate_keys": 0,
        "samples": {},
    }

    if not os.path.exists(csv_file):
        log.info("[整合性] %s が存在しないためチェックをスキップ", csv_file)
        return result

    result["exists"] = True
    key_counter: Counter = Counter()
    samples: dict[str, list] = {
        "empty_racer_no": [], "invalid_course": [], "invalid_order": [],
        "invalid_exhibition_time": [], "invalid_start_timing": [],
        "duplicate_keys": [],
    }

    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        for i, row in enumerate(csv.DictReader(f)):
            result["total_rows"] += 1
            summary = f"row={i} date={row.get('date')} venue={row.get('venue_code')} " \
                      f"race={row.get('race_no')} racer_no={row.get('racer_no')}"

            if not (row.get("racer_no") or "").strip():
                result["empty_racer_no"] += 1
                if len(samples["empty_racer_no"]) < sample_size:
                    samples["empty_racer_no"].append(summary)

            course_raw = (row.get("course") or "").strip()
            if not (course_raw.isdigit() and 1 <= int(course_raw) <= 6):
                result["invalid_course"] += 1
                if len(samples["invalid_course"]) < sample_size:
                    samples["invalid_course"].append(summary)

            if not _is_valid_order(row.get("order", "")):
                result["invalid_order"] += 1
                if len(samples["invalid_order"]) < sample_size:
                    samples["invalid_order"].append(summary)

            if not _is_valid_float_or_blank(row.get("exhibition_time", "")):
                result["invalid_exhibition_time"] += 1
                if len(samples["invalid_exhibition_time"]) < sample_size:
                    samples["invalid_exhibition_time"].append(summary)

            if not _is_valid_float_or_blank(row.get("start_timing", "")):
                result["invalid_start_timing"] += 1
                if len(samples["invalid_start_timing"]) < sample_size:
                    samples["invalid_start_timing"].append(summary)

            key = (row.get("date"), row.get("venue_code"), row.get("race_no"), row.get("racer_no"))
            key_counter[key] += 1

    dup_keys = [k for k, c in key_counter.items() if c > 1]
    result["duplicate_keys"] = len(dup_keys)
    for k in dup_keys[:sample_size]:
        samples["duplicate_keys"].append(
            f"date={k[0]} venue={k[1]} race={k[2]} racer_no={k[3]} (出現{key_counter[k]}回)"
        )

    result["samples"] = samples
    result["ok"] = all(
        result[k] == 0 for k in (
            "empty_racer_no", "invalid_course", "invalid_order",
            "invalid_exhibition_time", "invalid_start_timing", "duplicate_keys",
        )
    )

    if result["ok"]:
        log.info("[整合性] %s: 異常なし（%d行）", csv_file, result["total_rows"])
    else:
        for key in ("empty_racer_no", "invalid_course", "invalid_order",
                    "invalid_exhibition_time", "invalid_start_timing", "duplicate_keys"):
            if result[key] > 0:
                log.warning("[整合性] %s: %d件（例: %s）", key, result[key], samples[key][:sample_size])

    return result
