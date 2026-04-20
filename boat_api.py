"""
boat_api.py  ── BoatraceOpenAPI から出走表・直前情報データを取得

エンドポイント:
  出走表:   https://boatraceopenapi.github.io/programs/v2/YYYY/YYYYMMDD.json
  直前情報: https://boatraceopenapi.github.io/previews/v2/YYYY/YYYYMMDD.json

直前情報(previews)に含まれる当日データ:
  boats[].racer_exhibition_time          … 展示タイム
  boats[].racer_exhibition_start_timing  … 展示ST
  boats[].racer_tilt                     … チルト角
"""

from __future__ import annotations
import requests
from datetime import date

PROGRAMS_URL = "https://boatraceopenapi.github.io/programs/v2"
PREVIEWS_URL = "https://boatraceopenapi.github.io/previews/v2"
HEADERS      = {"User-Agent": "Mozilla/5.0"}
TIMEOUT      = 8

# 場コード(int) → 場名
VENUE_NAMES: dict[int, str] = {
    1:"桐生", 2:"戸田", 3:"江戸川", 4:"平和島", 5:"多摩川",
    6:"浜名湖", 7:"蒲郡", 8:"常滑", 9:"津", 10:"三国",
    11:"びわこ", 12:"住之江", 13:"尼崎", 14:"鳴門", 15:"丸亀",
    16:"児島", 17:"宮島", 18:"徳山", 19:"下関", 20:"若松",
    21:"芦屋", 22:"福岡", 23:"唐津", 24:"大村",
}
VENUE_NAME_TO_NUM: dict[str, int] = {v: k for k, v in VENUE_NAMES.items()}


def _fetch_programs(race_date: str) -> list[dict] | None:
    """出走表APIを取得"""
    url = (f"{PROGRAMS_URL}/today.json" if race_date == "today"
           else f"{PROGRAMS_URL}/{race_date[:4]}/{race_date}.json")
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json().get("programs", [])
    except Exception as e:
        print(f"[boat_api] 出走表取得失敗: {e}")
        return None


def _fetch_previews(race_date: str) -> list[dict] | None:
    """直前情報APIを取得（展示タイム・展示ST）"""
    url = (f"{PREVIEWS_URL}/today.json" if race_date == "today"
           else f"{PREVIEWS_URL}/{race_date[:4]}/{race_date}.json")
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json().get("previews", [])
    except Exception as e:
        print(f"[boat_api] 直前情報取得失敗（まだ公開前の可能性あり）: {e}")
        return None


def _to_vc_int(venue_code: str | int) -> int:
    if isinstance(venue_code, str):
        return VENUE_NAME_TO_NUM.get(venue_code) or int(venue_code)
    return int(venue_code)


def fetch_race(
    race_date:   str,
    venue_code:  str | int,
    race_number: int,
) -> list[dict] | None:
    """
    特定の場・レースの6艇データを返す。
    直前情報が公開済みなら展示タイム・展示STを自動取得して補完する。

    Returns:
        (boats, weather_info) のタプル
          boats: [
            {
              "lane":     1,
              "name":     "山本 浩次",
              "win_rate": 4.6,
              "motor":    33.33,
              "start":    0.14,     # 平均ST
              "ex_time":  6.72,     # 展示タイム（直前情報から）or None
              "ex_st":    0.13,     # 展示ST（直前情報から）or None
              "tilt":     0.0,      # チルト角
              "motor_no": 12,
              "local_win":3.2,
            }, ...
          ]
          weather_info: {
            "wind_speed":     3.0,    # 風速(m/s)
            "wind_direction": "追",   # 追/向/横
            "wave_height":    5,      # 波高(cm)
            "weather":        "晴",   # 天候
          }
        取得失敗時は (None, {})
    """
    programs = _fetch_programs(race_date)
    if programs is None:
        return None, {}

    vc_int = _to_vc_int(venue_code)

    # 出走表から基本データを取得
    target = next(
        (p for p in programs
         if p["race_stadium_number"] == vc_int and p["race_number"] == race_number),
        None
    )
    if target is None:
        print(f"[boat_api] 該当レースなし: 場{vc_int} {race_number}R {race_date}")
        return None, {}

    boats = {}
    for b in target["boats"]:
        # boatsの要素が辞書でない場合はスキップ
        if not isinstance(b, dict):
            print(f"[boat_api] programs boats の要素が dict でない: {type(b)} = {b}")
            continue
        lane = b.get("racer_boat_number") or b.get("racer_number")
        if not lane:
            continue
        boats[lane] = {
            "lane":      lane,
            "name":      b.get("racer_name", f"{lane}号艇"),
            "win_rate":  b.get("racer_national_top_1_percent", 0.0),
            "motor":     b.get("racer_assigned_motor_top_2_percent", 0.0),
            "start":     b.get("racer_average_start_timing", 0.18),
            "ex_time":   None,
            "ex_st":     None,
            "tilt":      None,
            "motor_no":  b.get("racer_assigned_motor_number", 0),
            "local_win": b.get("racer_local_top_1_percent", 0.0),
            "motor_top3":b.get("racer_assigned_motor_top_3_percent", 0.0),
        }

    # 直前情報（展示タイム・展示ST・風・波）を取得して補完
    weather_info = {}   # {"wind_speed":3.0, "wind_direction":"追", "wave_height":5}
    previews = _fetch_previews(race_date)
    if previews:
        prev = next(
            (p for p in previews
             if p.get("race_stadium_number") == vc_int
             and p.get("race_number") == race_number),
            None
        )
        if prev:
            # ── 艇別: 展示タイム・展示ST・チルト ──
            for b in prev.get("boats", []):
                # boatsの要素が辞書でない場合はスキップ
                if not isinstance(b, dict):
                    print(f"[boat_api] previews boats の要素が dict でない: {type(b)} = {b}")
                    continue
                lane = b.get("racer_boat_number")
                if lane in boats:
                    ex_time = b.get("racer_exhibition_time")
                    ex_st   = b.get("racer_exhibition_start_timing")
                    tilt    = b.get("racer_tilt")
                    if ex_time is not None: boats[lane]["ex_time"] = float(ex_time)
                    if ex_st   is not None: boats[lane]["ex_st"]   = float(ex_st)
                    if tilt    is not None: boats[lane]["tilt"]    = float(tilt)

            # ── 気象情報: 風速・風向・波高 ──
            # BoatraceOpenAPI previewsに含まれるフィールド
            # wind_speed, wind_direction, wave_height など
            ws  = prev.get("wind_speed")
            wd  = prev.get("wind_direction")      # 例: "追い風", "向かい風", "横風"
            wh  = prev.get("wave_height")         # 波高(cm)
            wdc = prev.get("weather_condition")   # "晴", "曇", "雨" など
            if ws  is not None: weather_info["wind_speed"]     = float(ws)
            if wd  is not None: weather_info["wind_direction"] = _convert_wind_dir(str(wd))
            if wh  is not None: weather_info["wave_height"]    = int(wh)
            if wdc is not None: weather_info["weather"]        = str(wdc)

            ex_count = sum(1 for b in boats.values() if b["ex_time"] is not None)
            print(f"[boat_api] 直前情報取得完了 "
                  f"展示タイム{ex_count}艇 / "
                  f"風:{weather_info.get('wind_speed','?')}m {weather_info.get('wind_direction','?')} / "
                  f"波:{weather_info.get('wave_height','?')}cm")
        else:
            print(f"[boat_api] 直前情報: 該当レースなし（まだ公開前）")

    result = [boats[lane] for lane in sorted(boats.keys())]
    return result, weather_info


def _convert_wind_dir(raw: str) -> str:
    """風向文字列をengine.py用に変換（追/向/横）"""
    if "追" in raw: return "追"
    if "向" in raw: return "向"
    if "横" in raw or "斜" in raw: return "横"
    return raw


def fetch_available_races(race_date: str) -> list[dict]:
    """その日の全開催レース一覧を返す"""
    programs = _fetch_programs(race_date)
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