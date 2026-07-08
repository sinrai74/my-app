#!/usr/bin/env python3
"""
x_kfile_race_parser.py  ── Kファイル(テキスト形式) 選手別詳細行パーサー

【独立モジュール】既存Bot（notify_arashi.py・x_ranking.py等）には一切
組み込まれていない。既存の parsers.py / 添付スクリプトの parse_k_file()
は「3連単の結果行（払戻金計算用）」のみを抽出しており、選手別の
着順・艇番・進入コース・登録番号は取得していない。本モジュールは
その選手別詳細行を新たに解析する。

【重要な注意】
本パーサーが前提とする行フォーマットは、一般に公開されている競艇
公式サイトのテキスト形式Kファイルのレイアウトに基づく「初期実装」
であり、実際のファイルで検証されたものではない。
運用開始前に、必ず実際のKファイル1つ以上でパース結果を目視確認し、
必要に応じて _RACER_LINE_RE 等の正規表現・列位置を調整すること
（design.md の「9. 動作確認・検証方針」参照）。

【想定するKファイルの構造（テキスト形式）】
    21KBGN
                                ボートレース○○成績  20240115  1日目
                                                                1R  予選
    ------------------------------------------------------------------------------
     着  艇  登番  選手名           Ｍｏｔ  Ｂｏａｔ  展示   進入  ＳＴ    レースタイム
      1   3  4444  ヤマダ タロウ      22     45     6.78    3   .13     1.51.2
      2   1  3333  タナカ イチロウ    15     12     6.85    1   .15     1.52.0
      ...
    21KEND
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# ════════════════════════════════════════════════════════════
# 場コード対応表（PC-KYOTEIコード表・既存プロジェクトのVENUE_NAMESと同一）
# ════════════════════════════════════════════════════════════
VENUE_CODE_MAP: dict[str, str] = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川",
    "06": "浜名湖", "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国",
    "11": "びわこ", "12": "住之江", "13": "尼崎", "14": "鳴門", "15": "丸亀",
    "16": "児島", "17": "宮島", "18": "徳山", "19": "下関", "20": "若松",
    "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村",
}

# ── 正規表現定義（要・実データ検証。design.md 9節参照） ──────────
_VENUE_BGN_RE = re.compile(r"^(\d{2})KBGN$")
_VENUE_END_RE = re.compile(r"^\d{2}KEND$")
_DATE_RE      = re.compile(r"(\d{8})")
_RACE_NO_RE   = re.compile(r"(\d{1,2})Ｒ|(\d{1,2})R")

# 選手別詳細行: 着順(F/L/K/S等の記号も許容) 艇番 登番 選手名 モーターNo ボートNo
#              展示タイム 進入コース ST レースタイム
# 全角スペースを半角に正規化してからマッチさせる前提。
_RACER_LINE_RE = re.compile(
    r"""^\s*
    (?P<order>\d{1,2}|F|L|K|S|K0|K1|L0|L1|S0|S1|S2)\s+   # 着順（数字 or 異常記号）
    (?P<boat_no>[1-6])\s+                                  # 艇番
    (?P<racer_no>\d{4})\s+                                 # 登録番号
    (?P<racer_name>\S+(?:\s\S+)?)\s+                       # 選手名（姓 名の間に空白1つを許容）
    (?P<motor_no>\d{1,3})\s+                               # モーターNo
    (?P<boat_equip_no>\d{1,3})\s+                          # ボート番号
    (?P<exhibition_time>\d\.\d{2})\s+                      # 展示タイム
    (?P<course>[1-6])\s+                                   # 進入コース
    \.?(?P<start_timing>\d{2})\s+                          # ST（.13 → 0.13）
    (?P<race_time>[\d.]+)\s*$                              # レースタイム
    """,
    re.VERBOSE,
)

_ZEN2HAN = str.maketrans("０１２３４５６７８９．－", "0123456789.-")


def _normalize(line: str) -> str:
    """全角数字・記号を半角に変換する（Kファイルは全角混在のことがあるため）。"""
    return line.translate(_ZEN2HAN)


def _parse_order(raw: str) -> "int | str":
    """着順を int（1-6）または異常着順記号（文字列）に変換する。"""
    if raw.isdigit():
        return int(raw)
    return raw  # F, L, K, S, K0, K1, L0, L1, S0, S1, S2 等はそのまま文字列で保持


def parse_k_race_file(filepath: "str | Path") -> list[dict]:
    """
    Kファイル(テキスト形式)から選手別詳細行を抽出し、
    「選手1人 × 1レース」を1レコードとするリストを返す。

    戻り値の各要素（k_local_course_stats.csv の設計と対応）:
      {
        "date": int, "venue_code": str, "venue_name": str, "race_no": int,
        "racer_no": str, "racer_name": str, "boat_no": int, "course": int,
        "order": int|str, "motor_no": int|None, "boat_equip_no": int|None,
        "exhibition_time": float|None, "start_timing": float|None,
        "race_time": str|None, "source_file": str,
      }

    パースできなかった行は静かにスキップする（他レコードの取得を止めない）。
    """
    filepath = Path(filepath)
    try:
        with open(filepath, "rb") as f:
            raw = f.read()
    except OSError:
        return []

    text = raw.decode("shift_jis", errors="replace")
    lines = text.split("\r\n") if "\r\n" in text else text.split("\n")

    records: list[dict] = []
    venue_code: Optional[str] = None
    current_date: Optional[int] = None
    current_race_no: Optional[int] = None

    for raw_line in lines:
        line = _normalize(raw_line)
        stripped = line.strip()

        m_bgn = _VENUE_BGN_RE.match(stripped)
        if m_bgn:
            venue_code = m_bgn.group(1)
            current_date = None
            current_race_no = None
            continue

        if _VENUE_END_RE.match(stripped):
            venue_code = None
            current_date = None
            current_race_no = None
            continue

        if venue_code is None:
            continue

        # 開催日（8桁の日付）を含む行があれば更新
        m_date = _DATE_RE.search(stripped)
        if m_date and current_date is None:
            current_date = int(m_date.group(1))

        # レース番号（"1R" 等）を含む行があれば更新
        m_race = _RACE_NO_RE.search(stripped)
        if m_race:
            current_race_no = int(m_race.group(1) or m_race.group(2))

        # 選手別詳細行のパースを試みる
        m = _RACER_LINE_RE.match(line)
        if not m or current_date is None or current_race_no is None:
            continue

        try:
            records.append({
                "date":            current_date,
                "venue_code":      venue_code,
                "venue_name":      VENUE_CODE_MAP.get(venue_code, f"場{venue_code}"),
                "race_no":         current_race_no,
                "racer_no":        m.group("racer_no"),
                "racer_name":      m.group("racer_name"),
                "boat_no":         int(m.group("boat_no")),
                "course":          int(m.group("course")),
                "order":           _parse_order(m.group("order")),
                "motor_no":        int(m.group("motor_no")) if m.group("motor_no") else None,
                "boat_equip_no":   int(m.group("boat_equip_no")) if m.group("boat_equip_no") else None,
                "exhibition_time": float(m.group("exhibition_time")) if m.group("exhibition_time") else None,
                "start_timing":    float(f"0.{m.group('start_timing')}") if m.group("start_timing") else None,
                "race_time":       m.group("race_time"),
                "source_file":     filepath.name,
            })
        except (ValueError, TypeError):
            continue

    return records


def parse_k_race_files(filepaths: list["str | Path"]) -> list[dict]:
    """複数のKファイルをまとめてパースする（初回一括構築で使用）。"""
    all_records: list[dict] = []
    for fp in filepaths:
        all_records.extend(parse_k_race_file(fp))
    return all_records
