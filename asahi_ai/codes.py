"""着順コード・ST・タイムの値の分解。

着順コードの意味は第三者の DB による仮の分類（Project 05 §5・07・08）。公式の定義表は未確認。
原文（result_code_raw）を必ず残し、未知のコードは数値に変換しない。
"""
from __future__ import annotations

import re

FINISHED = {f"0{i}" for i in range(1, 7)}
KNOWN = FINISHED | {"F", "L0", "L1", "K0", "K1", "S0", "S1", "S2"}


def classify_result(code: str) -> dict:
    """result_code_raw → finish_position・result_class・responsibility（仮）。"""
    c = code.strip()
    if c in FINISHED:
        return {"finish_position": int(c), "result_class": "finished", "responsibility": None}
    if c == "F":
        return {"finish_position": None, "result_class": "flying", "responsibility": "responsible"}
    if c in ("L0", "L1"):
        cls = "late_start"
    elif c in ("K0", "K1"):
        cls = "absent"
    elif c in ("S0", "S1", "S2"):
        cls = "disqualified"
    else:
        return {"finish_position": None, "result_class": "unknown", "responsibility": None}
    resp = "not_responsible" if c.endswith("0") else "responsible"
    return {"finish_position": None, "result_class": cls, "responsibility": resp}


def parse_st(raw: str) -> tuple[str, float | None, bool]:
    """ST の欄 → (flag, value, ok)。flag は F・L・K または ''。F の値は負にする。"""
    s = raw.strip()
    if s == "":
        return "", None, False
    m = re.fullmatch(r"([FLK]?)\s*(\d?\.\d{2}|\.)", s)
    if not m:
        return "?", None, False
    flag, num = m.group(1), m.group(2)
    if num == ".":
        return flag, None, flag in ("K", "L")
    v = float(num)
    return flag, (-v if flag == "F" else v), True


def parse_exhibition(raw: str) -> tuple[float | None, bool]:
    s = raw.strip()
    if re.fullmatch(r"\d\.\d{2}", s):
        return float(s), True
    if s.replace(" ", "") in ("K.", "", "."):
        return None, True
    return None, False


def parse_race_time(raw: str) -> tuple[float | None, bool]:
    """'1.48.9' → 108.9 秒。'.  .' や空は None（着外・欠場など）。"""
    s = raw.strip()
    m = re.fullmatch(r"(\d)\.(\d{2})\.(\d)", s)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 10, True
    if s.replace(" ", "") in (".", "..", ""):
        return None, True
    return None, False


def parse_series(raw: str) -> list[str]:
    """今節成績（例 '2 262 3423'）→ 空白を除いた1文字ずつの並び（数字以外の記号も残す）。"""
    return [ch for ch in raw if not ch.isspace() and ch != "　"]
