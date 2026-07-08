#!/usr/bin/env python3
"""
x_kfile_race_parser.py  ── Kファイル(テキスト形式) 選手別詳細行パーサー

【独立モジュール】既存Bot（notify_arashi.py・x_ranking.py等）には一切
組み込まれていない。既存の parsers.py / 添付スクリプトの parse_k_file()
は「3連単の結果行（払戻金計算用）」のみを抽出しており、選手別の
着順・艇番・進入コース・登録番号は取得していない。本モジュールは
その選手別詳細行を新たに解析する。

【v2: 実データ検証済み】
実際のKファイル（K260707.TXT、2026/7/7開催・12場分・169KB）で
構造を検証し、固定長カラム方式でパースするよう全面的に書き直した。
検証済みの事実:
  ・1ファイルに複数場（今回は12場）のデータが場コード単位で連続する
    （"24KBGN" 〜 次の "20KBGN" 〜 ... という区切り）。
  ・各場のブロック内に「第N日 YYYY/ M/ D」形式の開催日情報がある。
  ・レース見出し行は "   1R       予選　　　　  ...  H1800m 晴 風 南 1m 波 1cm"
    のように行頭空白＋レース番号＋"R"で始まる。
  ・選手別詳細行は下記の固定長カラム（文字位置、全角込みでカウント）:
      [0:4]   着順（右詰め2桁、例 "  01"）
      [6]     艇番（1桁）
      [8:12]  登録番号（4桁）
      [13:21] 選手名（全角8文字。姓名の間に全角スペースを含む固定長エリア）
      [22:24] モーターNo（2桁）
      [27:29] ボートNo（2桁）
      [31:35] 展示タイム（例 "6.97"）
      [38]    進入コース（1桁）
      [43:47] スタートタイミング（例 "0.16"。欠場等で " . 1" のような
              異常値になることがある）
      [52:]   レースタイム（例 "1.53.4"。欠場等で " .  . " になる）
  ・着順が数字以外（フライング等）になるケースは今回のサンプルには
    含まれておらず未検証。数字でない場合は文字列としてそのまま保持する
    安全設計にしている。

【実データで確認した既知の制約】欠場（着順欄が "K0" 等の記号）の場合、
本来「展示タイム」の位置に "K ." のような記号が入り、文字数が通常
（4文字）と異なるため、それ以降の全カラム（進入コース・ST・レース
タイム）の位置がずれる。この場合 _is_racer_line() の進入コース判定
（1〜6の数字であること）で自然に弾かれ、当該レコードはスキップされる。
「当地コース別成績」は進入コース情報が必須のため、コース不明の欠場
レコードを集計対象外にするのは意図した安全動作である
（K260707.TXT・864件中1件がこのケースで、正しくスキップされることを
確認済み）。
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

_VENUE_BGN_RE = re.compile(r"^(\d{2})KBGN")
_VENUE_END_RE = re.compile(r"^\d{2}KEND")
# 開催日: "第 3日          2026/ 7/ 7" のような行から year/month/day を抽出
_DATE_RE = re.compile(r"(\d{4})/\s*(\d{1,2})/\s*(\d{1,2})")
# レース見出し: 行頭スペース＋レース番号＋"R"（例: "   1R       予選"）
_RACE_NO_RE = re.compile(r"^\s{0,4}(\d{1,2})R\s")

_ZEN2HAN = str.maketrans("０１２３４５６７８９．－", "0123456789.-")
_ZEN_SPACE = "\u3000"


def _normalize_digits(s: str) -> str:
    """全角数字・記号のみ半角に変換する（全角スペースはそのまま残す＝選手名解析用）。"""
    return s.translate(_ZEN2HAN)


def _clean_name(raw_name_field: str) -> str:
    """選手名フィールド（全角スペース区切りの固定長エリア）から表示名を復元する。"""
    return raw_name_field.replace(_ZEN_SPACE, "").strip()


def _parse_order(raw: str) -> "int | str":
    """
    着順フィールードを int（1-6）または異常着順記号（文字列）に変換する。
    実データでの異常着順（F/L/K/S等）の実例は未検証のため、数字でなければ
    そのまま strip した文字列を保持する安全設計にする。
    """
    raw = raw.strip()
    if raw.isdigit():
        return int(raw)
    return raw


def _parse_float_safe(raw: str) -> Optional[float]:
    """" .  . " のような欠損値パターンも考慮して float に変換する。失敗時は None。"""
    raw = raw.strip()
    if not raw or raw in (".", "0.", ".0"):
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _is_racer_line(line: str) -> bool:
    """
    選手別詳細行かどうかを、固定長カラム位置の値が妥当かで判定する。
    （行の長さが十分にあり、着順・艇番・登番の位置が数字であること）
    """
    if len(line) < 47:
        return False
    boat_no_field = line[6:7].strip()
    racer_no_field = line[8:12].strip()
    return boat_no_field.isdigit() and racer_no_field.isdigit() and len(racer_no_field) == 4


def parse_k_race_file(filepath: "str | Path") -> list[dict]:
    """
    Kファイル(テキスト形式)から選手別詳細行を抽出し、
    「選手1人 × 1レース」を1レコードとするリストを返す。

    1ファイルに複数場のデータが含まれる場合も対応する
    （"XXKBGN" 〜 次の場コードの間を1場ブロックとして扱う）。

    戻り値の各要素:
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

    for line in lines:
        stripped = line.strip()

        m_bgn = _VENUE_BGN_RE.match(_normalize_digits(stripped))
        if m_bgn:
            venue_code = m_bgn.group(1)
            current_date = None
            current_race_no = None
            continue

        if _VENUE_END_RE.match(_normalize_digits(stripped)):
            venue_code = None
            current_date = None
            current_race_no = None
            continue

        if venue_code is None:
            continue

        # 開催日（"第N日 YYYY/ M/ D" 形式）
        if current_date is None:
            m_date = _DATE_RE.search(_normalize_digits(line))
            if m_date:
                y, mo, d = m_date.groups()
                try:
                    current_date = int(f"{y}{int(mo):02d}{int(d):02d}")
                except ValueError:
                    pass

        # レース番号見出し（"   1R  ..." 形式）
        m_race = _RACE_NO_RE.match(_normalize_digits(line))
        if m_race:
            current_race_no = int(m_race.group(1))
            continue

        # 選手別詳細行のパース（固定長カラム方式）
        if current_date is None or current_race_no is None:
            continue
        if not _is_racer_line(line):
            continue

        try:
            order_raw   = line[0:4]
            boat_no     = int(line[6:7])
            racer_no    = line[8:12].strip()
            racer_name  = _clean_name(line[13:21])
            motor_no_s  = line[22:24].strip()
            boat_eq_s   = line[27:29].strip()
            ex_time_s   = line[31:35].strip()
            course_s    = line[38:39].strip()
            st_s        = line[43:47].strip()
            race_time_s = line[52:].strip()

            if not course_s.isdigit() or not (1 <= int(course_s) <= 6):
                continue

            records.append({
                "date":            current_date,
                "venue_code":      venue_code,
                "venue_name":      VENUE_CODE_MAP.get(venue_code, f"場{venue_code}"),
                "race_no":         current_race_no,
                "racer_no":        racer_no,
                "racer_name":      racer_name,
                "boat_no":         boat_no,
                "course":          int(course_s),
                "order":           _parse_order(order_raw),
                "motor_no":        int(motor_no_s) if motor_no_s.isdigit() else None,
                "boat_equip_no":   int(boat_eq_s) if boat_eq_s.isdigit() else None,
                "exhibition_time": _parse_float_safe(ex_time_s),
                "start_timing":    _parse_float_safe(st_s),
                "race_time":       race_time_s if race_time_s and race_time_s not in (".", ". .", " .  . ") else None,
                "source_file":     filepath.name,
            })
        except (ValueError, TypeError, IndexError):
            continue

    return records


def parse_k_race_files(filepaths: list["str | Path"]) -> list[dict]:
    """複数のKファイルをまとめてパースする（初回一括構築で使用）。"""
    all_records: list[dict] = []
    for fp in filepaths:
        all_records.extend(parse_k_race_file(fp))
    return all_records
