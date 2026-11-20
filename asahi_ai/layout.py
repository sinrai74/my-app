"""B/K ファイルの固定長の列定義（layout_version つき）。

確認した範囲（Project 05・06・09）：2010・2013・2016・2019・2022 年の抜き出しと、
2023〜2026 年の 224 日分。位置は **cp932 のバイト単位**（全角は 2 バイト）。
"""
from __future__ import annotations

LAYOUT_VERSION = "bk-2010-v1"

# B（番組表）の選手行：79 バイト
B_RACER_LEN = 79
B_COLS = (
    (0, 1, "lane"),
    (2, 6, "racer_no"),
    (6, 14, "racer_name"),
    (14, 16, "age"),
    (16, 20, "branch"),
    (20, 22, "weight"),
    (22, 24, "class"),
    (25, 29, "nat_win"),
    (30, 35, "nat_2r"),
    (36, 40, "loc_win"),
    (41, 46, "loc_2r"),
    (46, 49, "motor_no"),
    (49, 55, "motor_2r"),
    (55, 58, "boat_no"),
    (58, 64, "boat_2r"),
    (64, 76, "series_raw"),
    (76, 79, "hayami"),
)

# K（競走成績）の選手行：66 バイト
K_RACER_LEN = 66
K_COLS = (
    (2, 4, "result_code_raw"),
    (6, 7, "lane"),
    (8, 12, "racer_no"),
    (13, 29, "racer_name"),
    (29, 32, "motor_no"),
    (32, 37, "boat_no"),
    (37, 43, "exhibition_raw"),
    (43, 47, "course_raw"),
    (47, 56, "st_raw"),
    (56, 66, "race_time_raw"),
)

VENUE_CODES = {f"{i:02d}" for i in range(1, 25)}

BET_TYPES = {
    "単勝": "win",
    "複勝": "place",
    "２連単": "exacta",
    "２連複": "quinella",
    "拡連複": "wide",
    "３連単": "trifecta",
    "３連複": "trio",
}

ZEN2HAN = str.maketrans("０１２３４５６７８９：", "0123456789:")
