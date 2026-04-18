"""
boat_api.py  ── BoatraceOpenAPI から出走表データを取得

エンドポイント:
  出走表: https://boatraceopenapi.github.io/programs/v2/YYYY/YYYYMMDD.json
  今日:   https://boatraceopenapi.github.io/programs/v2/today.json

JSONフィールド（確認済み）:
  programs[].race_stadium_number   … 場コード (int, 1〜24)
  programs[].race_number           … レース番号 (int, 1〜12)
  programs[].boats[].racer_boat_number          … 艇番 (1〜6)
  programs[].boats[].racer_name                 … 選手名
  programs[].boats[].racer_average_start_timing … 平均ST
  programs[].boats[].racer_national_top_1_percent … 全国勝率
  programs[].boats[].racer_assigned_motor_number  … モーターNo
  programs[].boats[].racer_assigned_motor_top_2_percent … モーター2着率
  programs[].boats[].racer_assigned_boat_number   … ボートNo
"""

from __future__ import annotations
import requests
from datetime import date

BASE_URL  = "https://boatraceopenapi.github.io/programs/v2"
HEADERS   = {"User-Agent": "Mozilla/5.0"}
TIMEOUT   = 8

# 場コード(int) → 場名
VENUE_NAMES: dict[int, str] = {
    1:"桐生", 2:"江戸川", 3:"戸田", 4:"平和島", 5:"多摩川",
    6:"浜名湖", 7:"蒲郡", 8:"常滑", 9:"津", 10:"三国",
    11:"びわこ", 12:"住之江", 13:"尼崎", 14:"鳴門", 15:"丸亀",
    16:"児島", 17:"宮島", 18:"徳山", 19:"下関", 20:"若松",
    21:"芦屋", 22:"福岡", 23:"唐津", 24:"大村",
}
VENUE_NAME_TO_NUM: dict[str, int] = {v: k for k, v in VENUE_NAMES.items()}


def _fetch_json(race_date: str) -> list[dict] | None:
    """
    race_date: "YYYYMMDD" または "today"
    返却: programs リスト or None
    """
    if race_date == "today":
        url = f"{BASE_URL}/today.json"
    else:
        year = race_date[:4]
        url  = f"{BASE_URL}/{year}/{race_date}.json"

    try:
        res = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        res.raise_for_status()
        return res.json().get("programs", [])
    except Exception as e:
        print(f"[boat_api] 取得失敗: {e}")
        return None


def fetch_race(
    race_date:   str,
    venue_code:  str | int,
    race_number: int,
) -> list[dict] | None:
    """
    特定の場・レースの6艇データを返す。

    Returns:
        [
          {
            "lane":     1,
            "name":     "山本 浩次",
            "win_rate": 4.6,      # 全国勝率
            "motor":    33.33,    # モーター2着率
            "start":    0.14,     # 平均ST
            "ex_time":  None,     # 展示タイム（出走表APIにはなし）
          }, ...
        ]
        取得失敗時は None
    """
    programs = _fetch_json(race_date)
    if programs is None:
        return None

    # venue_code を int に統一
    if isinstance(venue_code, str):
        vc_int = VENUE_NAME_TO_NUM.get(venue_code) or int(venue_code)
    else:
        vc_int = int(venue_code)

    # 該当レースを検索
    target = None
    for p in programs:
        if p["race_stadium_number"] == vc_int and p["race_number"] == race_number:
            target = p
            break

    if target is None:
        print(f"[boat_api] 該当レースなし: 場{vc_int} {race_number}R {race_date}")
        return None

    boats = []
    for b in target["boats"]:
        boats.append({
            "lane":     b["racer_boat_number"],
            "name":     b["racer_name"],
            "win_rate": b["racer_national_top_1_percent"],
            "motor":    b["racer_assigned_motor_top_2_percent"],
            "start":    b["racer_average_start_timing"],
            "ex_time":  None,   # 出走表APIにはなし（直前情報で更新）
            # 追加データ（フロントで参考表示用）
            "motor_no":    b["racer_assigned_motor_number"],
            "local_win":   b["racer_local_top_1_percent"],
            "motor_top3":  b["racer_assigned_motor_top_3_percent"],
        })

    print(f"[boat_api] 取得: 場{vc_int}({VENUE_NAMES.get(vc_int,'?')}) "
          f"{race_number}R {race_date} / {len(boats)}艇")
    return boats


def fetch_available_races(race_date: str) -> list[dict]:
    """
    その日の全開催レース一覧を返す。
    Returns: [{"venue_num": 2, "venue_name": "江戸川", "race_number": 1}, ...]
    """
    programs = _fetch_json(race_date)
    if not programs:
        return []

    races = []
    for p in programs:
        vn = p["race_stadium_number"]
        races.append({
            "venue_num":   vn,
            "venue_name":  VENUE_NAMES.get(vn, f"場{vn}"),
            "race_number": p["race_number"],
            "closed_at":   p.get("race_closed_at", ""),
        })
    return races