"""
notify_arashi.py
────────────────────────────────────────────────────────────
全ボートレース場の最新データを取得し、
「1号艇以外が1着になりそうなレース（荒れ予測）」を
Gmail でメール通知する。

GitHub Actions から 15 分ごとに呼び出される。

依存ライブラリ:
    pip install requests beautifulsoup4
    ※ メール送信は標準ライブラリ (smtplib) のみで動作

環境変数（GitHub Secrets）:
    GMAIL_ADDRESS   … 送信元 Gmail アドレス（例: yourname@gmail.com）
    GMAIL_APP_PASS  … Gmail アプリパスワード（16桁）
"""

from __future__ import annotations

import logging
import os
import smtplib
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from email.mime.text import MIMEText
from typing import Optional

import requests

# ════════════════════════════════════════════════════════════
# ロギング設定
# ════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# ════════════════════════════════════════════════════════════
# 定数
# ════════════════════════════════════════════════════════════

LINE_NOTIFY_URL = "https://notify-api.line.me/api/notify"  # 未使用（互換性のため残存）

# ── Gmail 送信先 ──────────────────────────────────────────────
MAIL_TO = "bigkirinuki@gmail.com"   # 通知先メールアドレス

# BoatraceOpenAPI エンドポイント
PROGRAMS_URL = "https://boatraceopenapi.github.io/programs/v2"
PREVIEWS_URL = "https://boatraceopenapi.github.io/previews/v2"

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
HTTP_TIMEOUT = 10

# 場コード → 場名マップ（BoatraceOpenAPI の race_stadium_number に対応）
VENUE_NAMES: dict[int, str] = {
    1: "桐生",   2: "戸田",   3: "江戸川", 4: "平和島", 5: "多摩川",
    6: "浜名湖", 7: "蒲郡",   8: "常滑",   9: "津",    10: "三国",
    11: "びわこ",12: "住之江",13: "尼崎",  14: "鳴門", 15: "丸亀",
    16: "児島",  17: "宮島",  18: "徳山",  19: "下関", 20: "若松",
    21: "芦屋",  22: "福岡",  23: "唐津",  24: "大村",
}

# ── 荒れスコアリング閾値 ────────────────────────────────────
# スコアがこの値以上のレースのみ通知する
UPSET_SCORE_THRESHOLD = 4.0   # チューニング可能

# ── 各判定項目の配点 ─────────────────────────────────────────
SCORE_WEIGHTS = {
    "ex_time_rank":    1.5,   # 展示タイムで1号艇が3位以下
    "ex_st_slow":      1.5,   # 展示STが0.18以上（遅れ）
    "wind_strong":     1.5,   # 風速 5m/s 以上
    "wind_headon":     1.0,   # 向かい風（インに不利）
    "wave_high":       1.0,   # 波高 10cm 以上
    "win_rate_low":    1.5,   # 1号艇の全国勝率が相対的に低い
    "motor_bad":       1.0,   # 1号艇のモーター勝率が平均以下
    "ex_time_gap":     1.5,   # 展示タイムで2位以下との差が大きい
}

# ── 将来モデル差し込み口 ─────────────────────────────────────
# `MODEL_SCORER` が None でなければ score_by_model() を呼ぶ。
# 機械学習モデル導入時はここに predict 関数をセットするだけでよい。
#
# 例:
#   import pickle
#   from notify_arashi import MODEL_SCORER
#   MODEL_SCORER = pickle.load(open("model_all.pkl","rb"))["win"].predict_proba
MODEL_SCORER = None   # type: ignore[assignment]


# ════════════════════════════════════════════════════════════
# データクラス
# ════════════════════════════════════════════════════════════

@dataclass
class BoatInfo:
    """1艇分の情報"""
    lane:      int
    name:      str       = "不明"
    win_rate:  float     = 0.0   # 全国勝率
    motor:     float     = 0.0   # モーター2連率
    avg_st:    float     = 0.18  # 平均スタートタイミング
    ex_time:   Optional[float] = None   # 展示タイム（直前情報）
    ex_st:     Optional[float] = None   # 展示ST（直前情報）
    tilt:      Optional[float] = None   # チルト角


@dataclass
class WeatherInfo:
    """気象情報"""
    wind_speed:     Optional[float] = None   # 風速 (m/s)
    wind_direction: Optional[str]   = None   # "追" / "向" / "横"
    wave_height:    Optional[int]   = None   # 波高 (cm)
    weather:        Optional[str]   = None   # "晴" / "曇" / "雨"


@dataclass
class RaceResult:
    """1レース分の判定結果"""
    venue_name:  str
    venue_num:   int
    race_number: int
    boats:       list[BoatInfo]
    weather:     WeatherInfo
    upset_score: float          = 0.0
    score_detail: dict          = field(default_factory=dict)
    target_lanes: list[int]     = field(default_factory=list)  # 狙い艇番


# ════════════════════════════════════════════════════════════
# API 取得レイヤー
# ════════════════════════════════════════════════════════════

def _safe_get(url: str) -> Optional[dict]:
    """JSON を取得して返す。失敗時は None。"""
    try:
        r = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        log.warning("HTTP 取得失敗: %s  url=%s", e, url)
        return None


# ── MLモデル読み込み ──────────────────────────────────────
import gzip, pickle as _pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

_ML_MODELS = None
_ML_FEATURE_COLS = None

def _load_ml_model():
    global _ML_MODELS, _ML_FEATURE_COLS
    if _ML_MODELS is not None:
        return True
    try:
        with gzip.open("model_all.pkl", "rb") as _f:
            _data = _pickle.load(_f)
        _ML_MODELS = _data["models"]
        _ML_FEATURE_COLS = _data["feature_cols"]
        log.info("[ML] モデル読み込み成功 特徴量=%d", len(_ML_FEATURE_COLS))
        return True
    except Exception as e:
        log.warning("[ML] モデル読み込み失敗: %s", e)
        return False


def _predict_win_prob(boats: list) -> dict[int, float]:
    if not _load_ml_model():
        return {}
    try:
        rows = []
        for b in boats:
            row = {f: np.nan for f in _ML_FEATURE_COLS}
            row["艇番"]    = b.lane
            row["全国勝率"] = b.win_rate
            row["モーター2率"] = b.motor
            row["平均ST"]   = b.avg_st
            row["当地勝率"] = getattr(b, "local_win", np.nan)
            row["全国2率"]  = getattr(b, "win_rate2", np.nan)
            row["当地2率"]  = getattr(b, "local_win2", np.nan)
            if b.ex_time:
                row["展示タイム"] = b.ex_time if "展示タイム" in _ML_FEATURE_COLS else np.nan
            if b.ex_st:
                row["展示ST"] = b.ex_st if "展示ST" in _ML_FEATURE_COLS else np.nan
            rows.append(row)
        df = pd.DataFrame(rows)
        for col in ["全国勝率","全国2率","当地勝率","当地2率","モーター2率"]:
            if f"{col}_IN順位" in _ML_FEATURE_COLS:
                df[f"{col}_IN順位"] = df[col].fillna(0).rank(ascending=False).astype(float)
            if f"{col}_IN偏差" in _ML_FEATURE_COLS:
                df[f"{col}_IN偏差"] = df[col] - df[col].mean()
            if f"{col}_MAX差" in _ML_FEATURE_COLS:
                df[f"{col}_MAX差"] = df[col].max() - df[col]
            if f"{col}_MIN差" in _ML_FEATURE_COLS:
                df[f"{col}_MIN差"] = df[col] - df[col].min()
        if "平均ST_IN順位" in _ML_FEATURE_COLS:
            df["平均ST_IN順位"] = df["平均ST"].fillna(0.18).rank(ascending=True).astype(float)
        X = df[_ML_FEATURE_COLS].fillna(df[_ML_FEATURE_COLS].median())
        probs = _ML_MODELS["win"].predict_proba(X)[:, 1]
        total = sum(probs)
        if total > 0:
            probs = probs / total  # 正規化
        return {b.lane: float(p) for b, p in zip(boats, probs)}
    except Exception as e:
        log.warning("[ML] 予測失敗: %s", e)
        return {}


def _scrape_beforeinfo_bulk(race_list: list, race_date: str) -> dict:
    """
    Playwrightを1回だけ起動して複数レースの展示タイムと気象データをまとめて取得。
    race_list: [(venue_num, race_number), ...]
    戻り値: {(venue_num, race_number): {"boats": {lane: {...}}, "weather": {...}}}
    """
    results = {}
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            for venue_num, race_number in race_list:
                url = (f"https://www.boatrace.jp/owpc/pc/race/beforeinfo"
                       f"?rno={race_number}&jcd={str(venue_num).zfill(2)}&hd={race_date}")
                try:
                    page.goto(url, wait_until="networkidle", timeout=20000)
                    tables = page.query_selector_all("table")
                    if len(tables) < 2:
                        continue

                    # ── 気象データ（Table[0]）──────────────────────
                    weather_data = {}
                    if len(tables) >= 1:
                        t0_rows = tables[0].query_selector_all("tr")
                        for row in t0_rows:
                            cells = [td.inner_text().strip() for td in row.query_selector_all("td")]
                            # 風速・風向・波高を探す
                            for i, cell in enumerate(cells):
                                if "m" in cell and len(cell) <= 6:
                                    try:
                                        ws = float(cell.replace("m", "").strip())
                                        if 0 <= ws <= 30:
                                            weather_data["wind_speed"] = ws
                                    except ValueError:
                                        pass
                                if "cm" in cell:
                                    try:
                                        wh = int(cell.replace("cm", "").strip())
                                        weather_data["wave_height"] = wh
                                    except ValueError:
                                        pass
                                if cell in ("向かい風", "追い風", "横風", "向", "追", "横"):
                                    wd = "向" if "向" in cell else "追" if "追" in cell else "横"
                                    weather_data["wind_direction"] = wd

                    # ── 艇別データ（Table[1]）──────────────────────
                    boats = {}
                    rows = tables[1].query_selector_all("tr")
                    for i, row in enumerate(rows):
                        cells = [td.inner_text().strip() for td in row.query_selector_all("td")]
                        if cells and cells[0].isdigit() and 1 <= int(cells[0]) <= 6:
                            lane = int(cells[0])
                            try:
                                ex_time = float(cells[4]) if len(cells) > 4 and cells[4] else None
                                tilt    = float(cells[5]) if len(cells) > 5 and cells[5] else None
                            except ValueError:
                                ex_time = tilt = None
                            ex_st = None
                            for j in range(i+1, min(i+5, len(rows))):
                                sc = [td.inner_text().strip() for td in rows[j].query_selector_all("td")]
                                if len(sc) >= 3 and sc[1] == "ST":
                                    try:
                                        ex_st = float("0"+sc[2]) if sc[2].startswith(".") else float(sc[2])
                                    except ValueError:
                                        pass
                                    break
                            boats[lane] = {"ex_time": ex_time, "ex_st": ex_st, "tilt": tilt}
                    if boats:
                        results[(venue_num, race_number)] = {
                            "boats": boats,
                            "weather": weather_data,
                        }
                except Exception as e:
                    log.warning("Playwright取得失敗: 場%s %sR %s", venue_num, race_number, e)
            browser.close()
    except Exception as e:
        log.warning("Playwright起動失敗: %s", e)
    return results


def _scrape_beforeinfo(race_no: int, venue_code: int, race_date: str) -> dict:
    """単体取得（後方互換用）- boatsデータのみ返す"""
    result = _scrape_beforeinfo_bulk([(venue_code, race_no)], race_date)
    entry = result.get((venue_code, race_no), {})
    return entry.get("boats", {})


def fetch_programs(race_date: str) -> list[dict]:
    """出走表を取得して programs リストを返す。失敗時は []。"""
    url = f"{PROGRAMS_URL}/{race_date[:4]}/{race_date}.json"
    data = _safe_get(url)
    if data is None:
        return []
    programs = data.get("programs", [])
    log.info("出走表取得: %d レース (date=%s)", len(programs), race_date)
    return programs


def fetch_previews(race_date: str) -> list[dict]:
    """直前情報を取得して previews リストを返す。未公開時は []。"""
    url = f"{PREVIEWS_URL}/{race_date[:4]}/{race_date}.json"
    data = _safe_get(url)
    if data is None:
        log.info("直前情報: まだ公開前の可能性あり (date=%s)", race_date)
        return []
    previews = data.get("previews", [])
    log.info("直前情報取得: %d レース (date=%s)", len(previews), race_date)
    return previews


# ════════════════════════════════════════════════════════════
# データ組み立て
# ════════════════════════════════════════════════════════════

def _extract_boats_from_program(program: dict) -> list[BoatInfo]:
    """出走表エントリから BoatInfo リストを作成する。"""
    boats: list[BoatInfo] = []
    for b in program.get("boats", []):
        if not isinstance(b, dict):
            continue
        lane = b.get("racer_boat_number") or b.get("racer_number")
        if not lane:
            continue
        boats.append(BoatInfo(
            lane     = int(lane),
            name     = b.get("racer_name", f"{lane}号艇"),
            win_rate = float(b.get("racer_national_top_1_percent") or 0),
            motor    = float(b.get("racer_assigned_motor_top_2_percent") or 0),
            avg_st   = float(b.get("racer_average_start_timing") or 0.18),
        ))
    return sorted(boats, key=lambda x: x.lane)


def _apply_preview_to_boats(
    boats: list[BoatInfo],
    preview: dict,
) -> WeatherInfo:
    """
    直前情報を boats に上書きし、WeatherInfo を返す。
    preview が None でも安全に動く。
    """
    # ── 艇別: 展示タイム / 展示ST / チルト ──────────────────
    # boats は {"1": {...}, "2": {...}} 形式の辞書
    boats_dict = preview.get("boats", {})
    if isinstance(boats_dict, dict):
        pb_list = list(boats_dict.values())
    else:
        pb_list = boats_dict

    for pb in pb_list:
        if not isinstance(pb, dict):
            continue
        lane = pb.get("racer_boat_number")
        boat = next((b for b in boats if b.lane == lane), None)
        if boat is None:
            continue
        if pb.get("racer_exhibition_time") is not None:
            boat.ex_time = float(pb["racer_exhibition_time"])
        if pb.get("racer_start_timing") is not None:
            boat.ex_st = float(pb["racer_start_timing"])
        if pb.get("racer_tilt_adjustment") is not None:
            boat.tilt = float(pb["racer_tilt_adjustment"])

    # 気象情報（実際のキー名に修正）
    wd_num = preview.get("race_wind_direction_number")
    wd_str: Optional[str] = None
    if wd_num is not None:
        wd_num = int(wd_num)
        if wd_num in (1, 2):
            wd_str = "追"
        elif wd_num in (9, 10):
            wd_str = "向"
        else:
            wd_str = "横"

    ws  = preview.get("race_wind")
    wh  = preview.get("race_wave")
    wdc = preview.get("race_weather_number")
    weather_label = {1: "晴", 2: "曇", 3: "雨", 4: "雪"}.get(int(wdc), str(wdc)) if wdc is not None else None

    return WeatherInfo(
        wind_speed     = float(ws)  if ws  is not None else None,
        wind_direction = wd_str,
        wave_height    = int(wh)    if wh  is not None else None,
        weather        = weather_label,
    )


def build_race_data(
    programs: list[dict],
    previews: list[dict],
) -> list[tuple[int, int, list[BoatInfo], WeatherInfo]]:
    """
    出走表 + 直前情報を結合して (venue_num, race_number, boats, weather) のリストを返す。
    """
    # 直前情報をキー (venue_num, race_number) で引けるようにする
    preview_map: dict[tuple[int, int], dict] = {}
    for p in previews:
        key = (p.get("race_stadium_number"), p.get("race_number"))
        preview_map[key] = p

    results: list[tuple[int, int, list[BoatInfo], WeatherInfo]] = []
    for prog in programs:
        vn  = prog.get("race_stadium_number")
        rno = prog.get("race_number")
        if vn is None or rno is None:
            continue

        boats   = _extract_boats_from_program(prog)
        preview = preview_map.get((vn, rno), {})
        weather = _apply_preview_to_boats(boats, preview)
        closed_at = prog.get('race_closed_at', '')
        results.append((int(vn), int(rno), boats, weather, closed_at))

    log.info("レース組み立て完了: %d レース", len(results))
    return results


# ════════════════════════════════════════════════════════════
# 荒れスコアリング
# ════════════════════════════════════════════════════════════

def _score_exhibition_time(boat1: BoatInfo, all_boats: list[BoatInfo]) -> tuple[float, str]:
    """展示タイムで 1 号艇が何位かを評価する。"""
    times = [(b.lane, b.ex_time) for b in all_boats if b.ex_time is not None]
    if not times or boat1.ex_time is None:
        return 0.0, "展示タイム: データなし"

    # 展示タイムは速い（小さい）ほど良い
    sorted_lanes = [lane for lane, _ in sorted(times, key=lambda x: x[1])]
    rank = sorted_lanes.index(1) + 1   # 1着=1位

    detail = f"展示タイム: 1号艇 {boat1.ex_time:.2f}秒 ({rank}位/{len(times)}艇)"

    if rank == 1:
        return 0.0, detail                         # 最速 → 問題なし
    if rank == 2:
        return SCORE_WEIGHTS["ex_time_rank"] * 0.5, detail   # やや劣勢
    # 3位以下
    score = SCORE_WEIGHTS["ex_time_rank"]

    # タイム差が 0.05 秒以上なら追加点
    best_time = min(t for _, t in times)
    if boat1.ex_time - best_time >= 0.05:
        score += SCORE_WEIGHTS["ex_time_gap"]
        detail += f" ※最速比 +{boat1.ex_time - best_time:.2f}秒"

    return score, detail


def _score_exhibition_st(boat1: BoatInfo, all_boats: list[BoatInfo]) -> tuple[float, str]:
    """展示 ST で 1 号艇を評価する。"""
    sts = [(b.lane, b.ex_st) for b in all_boats if b.ex_st is not None]
    if not sts or boat1.ex_st is None:
        return 0.0, "展示ST: データなし"

    detail = f"展示ST: 1号艇 {boat1.ex_st:.2f}"
    score  = 0.0

    # ST が 0.18 以上（反応が遅い）
    if boat1.ex_st >= 0.18:
        score += SCORE_WEIGHTS["ex_st_slow"]
        detail += "（遅れ）"
    # ST が他艇より 0.03 以上遅い艇が 2 艇以上いる場合
    faster_count = sum(1 for _, st in sts if st < boat1.ex_st - 0.03)
    if faster_count >= 2:
        score += SCORE_WEIGHTS["ex_st_slow"] * 0.5
        detail += f" ※{faster_count}艇が大幅に速い"

    return score, detail


def _score_weather(weather: WeatherInfo) -> tuple[float, str]:
    """風速・風向・波高の条件から荒れスコアを計算する。"""
    score  = 0.0
    parts: list[str] = []

    # 風速
    if weather.wind_speed is not None:
        ws = weather.wind_speed
        if ws >= 5:
            score += SCORE_WEIGHTS["wind_strong"]
            parts.append(f"風速 {ws:.1f}m（強風）")
        elif ws >= 2:
            score += SCORE_WEIGHTS["wind_strong"] * 0.5
            parts.append(f"風速 {ws:.1f}m（やや強）")
        else:
            parts.append(f"風速 {ws:.1f}m")

    # 風向（向かい風 = インに不利）
    if weather.wind_direction == "向":
        score += SCORE_WEIGHTS["wind_headon"]
        parts.append("向かい風（イン不利）")
    elif weather.wind_direction == "追":
        parts.append("追い風")
    elif weather.wind_direction == "横":
        score += SCORE_WEIGHTS["wind_headon"] * 0.5
        parts.append("横風")

    # 波高
    if weather.wave_height is not None:
        wh = weather.wave_height
        if wh >= 4:
            score += SCORE_WEIGHTS["wave_high"] * 1.5
            parts.append(f"波 {wh}cm（荒れ）")
        elif wh >= 2:
            score += SCORE_WEIGHTS["wave_high"]
            parts.append(f"波 {wh}cm（やや高）")
        else:
            parts.append(f"波 {wh}cm")

    detail = " / ".join(parts) if parts else "気象データなし"
    return score, detail


def _score_performance(boat1: BoatInfo, all_boats: list[BoatInfo]) -> tuple[float, str]:
    """勝率・モーター率から 1 号艇の相対的な弱さを評価する。"""
    score  = 0.0
    parts: list[str] = []

    if len(all_boats) < 2:
        return 0.0, "選手データ不足"

    other_boats = [b for b in all_boats if b.lane != 1]

    # 全国勝率の場平均と比較
    avg_other_win_rate = sum(b.win_rate for b in other_boats) / len(other_boats)
    if boat1.win_rate < avg_other_win_rate * 0.85:
        score += SCORE_WEIGHTS["win_rate_low"]
        parts.append(
            f"1号艇勝率 {boat1.win_rate:.2f} < 他艇平均 {avg_other_win_rate:.2f}"
        )

    # モーター2連率
    avg_other_motor = sum(b.motor for b in other_boats) / len(other_boats)
    if boat1.motor < avg_other_motor * 0.90:
        score += SCORE_WEIGHTS["motor_bad"]
        parts.append(
            f"1号艇モーター {boat1.motor:.1f}% < 他艇平均 {avg_other_motor:.1f}%"
        )

    detail = " / ".join(parts) if parts else "選手スペック: 問題なし"
    return score, detail


def score_by_model(boats: list[BoatInfo], weather: WeatherInfo) -> Optional[float]:
    """
    将来の機械学習モデル差し込み口。
    MODEL_SCORER が設定されていれば呼び出してスコアを返す。
    現状は None を返す（ルールベースのみ使用）。

    モデル組み込み例:
        from features import build_feature_vector   # 既存 features.py を活用
        feat = build_feature_vector(boats, weather)
        prob_not_1 = MODEL_SCORER(feat)[0][0]  # 1号艇以外が1着の確率
        return prob_not_1 * 10  # スコールに変換
    """
    if MODEL_SCORER is None:
        return None
    # ── ここに ML スコアリングを実装 ─────────────────────────
    # feat = build_feature_vector_for_notify(boats, weather)
    # return float(MODEL_SCORER(feat)[0][0]) * 10
    return None


def calc_boat_score(
    boat,
    all_boats: list,
    weather,
) -> float:
    score = 0.0

    # 展示タイム
    ex_times = [b.ex_time for b in all_boats if b.ex_time is not None and b.ex_time > 0]
    if ex_times and boat.ex_time is not None and boat.ex_time > 0:
        best_ex = min(ex_times)
        diff = boat.ex_time - best_ex
        if diff <= 0.00:   ex_s = 2.0
        elif diff <= 0.02: ex_s = 1.5
        elif diff <= 0.05: ex_s = 1.0
        elif diff <= 0.08: ex_s = 0.3
        elif diff <= 0.12: ex_s = -0.5
        else:              ex_s = -1.5
        score += ex_s * 1.5

    # ST
    sts = [b.ex_st for b in all_boats if b.ex_st is not None]
    if sts and boat.ex_st is not None:
        avg_st = sum(sts) / len(sts)
        diff_st = boat.ex_st - avg_st
        if diff_st <= -0.05:   st_s = 2.0
        elif diff_st <= -0.03: st_s = 1.2
        elif diff_st <= -0.01: st_s = 0.5
        elif diff_st <= 0.01:  st_s = 0.0
        elif diff_st <= 0.03:  st_s = -0.5
        else:                  st_s = -1.5
        score += st_s * 1.3

        # 展開補正（2〜4号艇のまくり）
        boat1 = next((b for b in all_boats if b.lane == 1), None)
        if boat.lane in [2, 3, 4] and boat1 is not None and boat1.ex_st is not None:
            if boat.ex_st <= 0.13 and boat1.ex_st >= 0.18:
                score += 1.5
        # 2号艇差し補正
        if boat.lane == 2 and boat1 is not None and boat1.ex_st is not None:
            if abs(boat.ex_st - boat1.ex_st) <= 0.02:
                score += 1.0

    # モーター順位（同率対応）
    motors = sorted([b.motor for b in all_boats if b.motor is not None], reverse=True)
    if motors and boat.motor is not None:
        rank = sorted(motors, reverse=True).index(boat.motor) + 1
        if rank == 1:   m_s = 2.0
        elif rank == 2: m_s = 1.2
        elif rank == 3: m_s = 0.5
        elif rank == 4: m_s = -0.3
        elif rank == 5: m_s = -0.8
        else:           m_s = -1.5
        score += m_s * 1.2

    # コース補正（5・6号艇緩和）
    lane_weight = {1: 1.6, 2: 1.2, 3: 0.9, 4: 0.5, 5: -0.2, 6: -0.6}
    score += lane_weight.get(boat.lane, 0.0)

    # 風（強化）
    if weather and weather.wind_speed is not None and weather.wind_speed > 0:
        ws = weather.wind_speed
        wd = weather.wind_direction
        if wd == "向":
            score += ws * 0.6
        elif wd == "追":
            if boat.lane == 1:
                score += ws * 0.4
            else:
                score -= ws * 0.3
        else:
            score -= ws * 0.2

    # 波（強化）
    if weather and weather.wave_height is not None:
        wh = weather.wave_height
        if wh >= 5:
            score += (-2.0 if boat.lane == 1 else 1.0)
        elif wh >= 3:
            score += (-0.8 if boat.lane == 1 else 0.3)

    # 勝率
    if boat.win_rate is not None:
        score += (boat.win_rate - 5.0) * 0.7

    return score


def calculate_upset_score(
    boats: list,
    weather,
) -> tuple:
    import math

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    probs = []
    for b in boats:
        s = calc_boat_score(b, boats, weather)
        p = sigmoid(s / 3.0)
        probs.append((b.lane, p, s))

    probs.sort(key=lambda x: x[1], reverse=True)

    boat1_prob = next((p for lane, p, s in probs if lane == 1), 0.5)
    boat1_rank = next((i+1 for i, (lane, p, s) in enumerate(probs) if lane == 1), 1)

    # 荒れスコア
    upset_score = (1 - boat1_prob) * 10
    if probs[0][0] != 1:
        upset_score += 2.0

    # 狙い目（calc_boat_scoreベースで上位3艇、1号艇除く）
    target = [lane for lane, p, s in probs if lane != 1][:3]

    # 通知条件: 1号艇が1位でない
    top_lane, top_prob, top_score = probs[0]
    second_prob = probs[1][1] if len(probs) > 1 else 0.0

    if top_lane == 1:
        upset_score = 0.0

    detail = {
        "予測1位": f"{top_lane}号艇(確率{top_prob:.1%})",
        "1号艇確率": f"{boat1_prob:.1%}(rank{boat1_rank})",
        "展示": f"{next((b.ex_time for b in boats if b.lane==1), None)}秒",
        "展示ST": f"{next((b.ex_st for b in boats if b.lane==1), None)}",
    }

    return upset_score, detail, target

def _danger_label(score: float) -> str:
    if score >= 7.0:
        return "🔴 非常に高"
    elif score >= 5.0:
        return "🟠 高"
    elif score >= 3.5:
        return "🟡 中"
    else:
        return "🟢 低"


def build_message(result: RaceResult) -> tuple[str, str]:
    """
    メール件名と本文を生成して (subject, body) のタプルで返す。
    """
    boat1 = next((b for b in result.boats if b.lane == 1), None)
    w     = result.weather
    label = _danger_label(result.upset_score)

    # ── 件名 ─────────────────────────────────────────────────
    subject = (
        f"【荒れ検知】{result.venue_name} {result.race_number}R "
        f"{label} (score:{result.upset_score:.1f})"
    )

    # ── 本文 ─────────────────────────────────────────────────
    lines = [
        f"{'='*30}",
        f"【荒れ検知】{result.venue_name} {result.race_number}R",
        f"危険度: {label}  スコア: {result.upset_score:.1f}",
        f"{'='*30}",
        "",
    ]

    # 気象情報
    weather_parts: list[str] = []
    if w.wind_speed     is not None: weather_parts.append(f"風{w.wind_speed:.1f}m{w.wind_direction or ''}")
    if w.wave_height    is not None: weather_parts.append(f"波{w.wave_height}cm")
    if w.weather        is not None: weather_parts.append(w.weather)
    if weather_parts:
        lines.append("🌊 " + " / ".join(weather_parts))
    else:
        lines.append("🌊 気象データなし")

    lines.append("")

    # 全艇の展示タイム
    lines.append("── 展示タイム ──")
    sorted_boats = sorted(result.boats, key=lambda b: b.ex_time or 99)
    for b in sorted(result.boats, key=lambda b: b.lane):
        et = f"{b.ex_time:.2f}" if b.ex_time else "——"
        st = f"{b.ex_st:.2f}"   if b.ex_st   else "——"
        tl = f"チルト{b.tilt:+.1f}" if b.tilt is not None else ""
        rank = sorted_boats.index(b) + 1 if b.ex_time else "-"
        marker = "★" if b.lane == 1 else "  "
        lines.append(f"{marker}{b.lane}号艇 {b.name[:4]:4s} 展示{et}({rank}位) ST{st} {tl}")

    lines.append("")

    # 狙い目
    if result.target_lanes:
        tgt = "-".join(str(l) for l in result.target_lanes)
        lines.append(f"🎯 狙い: {tgt}-全")

    # スコア詳細
    lines += ["", "── スコア内訳 ──"]
    for key, val in result.score_detail.items():
        lines.append(f"  {key}: {val}")

    return subject, "\n".join(lines)


# ════════════════════════════════════════════════════════════
# Gmail 送信
# ════════════════════════════════════════════════════════════

def send_email(subject: str, body: str) -> bool:
    """
    Gmail SMTP (TLS) でメールを送信する。成功で True。

    必要な環境変数（GitHub Secrets に登録）:
        GMAIL_ADDRESS  … 送信元 Gmail アドレス
        GMAIL_APP_PASS … Gmail アプリパスワード（16桁）
                         ※ Google アカウント → セキュリティ →
                            2段階認証 ON → アプリパスワード で発行
    """
    gmail_address = os.getenv("GMAIL_ADDRESS", "")
    app_password  = os.getenv("GMAIL_APP_PASS", "")

    if not gmail_address or not app_password:
        log.error("環境変数 GMAIL_ADDRESS / GMAIL_APP_PASS が未設定です")
        return False

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"]    = gmail_address
    msg["To"]      = MAIL_TO

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(gmail_address, app_password)
            smtp.sendmail(gmail_address, MAIL_TO, msg.as_string())
        log.info("メール送信成功 → %s  件名: %s", MAIL_TO, subject)
        return True
    except smtplib.SMTPException as e:
        log.error("メール送信失敗: %s", e)
        return False


def send_summary_if_none(total_checked: int) -> None:
    """荒れレースが 1 件もない場合は何も送らない（スパム防止）。"""
    log.info("荒れ検知なし: %d レース確認済み → メール送信スキップ", total_checked)


# ════════════════════════════════════════════════════════════
# メイン処理
# ════════════════════════════════════════════════════════════

def run(race_date: Optional[str] = None) -> None:
    """
    メインエントリポイント。

    Args:
        race_date: "YYYYMMDD" 形式。None なら今日の日付を使う。
    """
    # Gmail の認証情報チェック（早期失敗）
    if not os.getenv("GMAIL_ADDRESS") or not os.getenv("GMAIL_APP_PASS"):
        log.error("環境変数 GMAIL_ADDRESS / GMAIL_APP_PASS が設定されていません")
        sys.exit(1)

    # 時間帯チェック（JST 8:20〜22:45 以外はスキップ）
    from datetime import datetime, timezone, timedelta
    JST = timezone(timedelta(hours=9))
    now_jst = datetime.now(JST)
    now_minutes = now_jst.hour * 60 + now_jst.minute
    start_minutes = 8 * 60 + 20   # 8:20
    end_minutes   = 22 * 60 + 45  # 22:45
    skip_filter = os.getenv("SKIP_TIME_FILTER", "0") == "1"
    if not skip_filter and not (start_minutes <= now_minutes <= end_minutes):
        log.info("時間帯外のためスキップ (JST %02d:%02d)", now_jst.hour, now_jst.minute)
        return

    if race_date is None:
        race_date = date.today().strftime("%Y%m%d")

    log.info("▶ 開始 date=%s  threshold=%.1f  送信先=%s",
             race_date, UPSET_SCORE_THRESHOLD, MAIL_TO)

    # ── データ取得 ────────────────────────────────────────────
    programs = fetch_programs(race_date)
    if not programs:
        log.warning("出走表が取得できませんでした（開催なし or API 障害）")
        return

    # 直前情報の取得は失敗しても続行（未公開レースがあるため）
    previews = fetch_previews(race_date)

    # ── レースデータ組み立て ──────────────────────────────────
    race_list = build_race_data(programs, previews)
    # 締切前のレースのみに絞る
    from datetime import datetime, timezone, timedelta
    JST = timezone(timedelta(hours=9))
    now = datetime.now(JST).replace(tzinfo=None)
    skip_filter = os.getenv("SKIP_TIME_FILTER", "0") == "1"
    filtered = []
    for item in race_list:
        closed_at = item[4] if len(item) > 4 else None
        if skip_filter:
            filtered.append(item)
        elif closed_at:
            try:
                closed_dt = datetime.strptime(closed_at, "%Y-%m-%d %H:%M:%S")
                if closed_dt > now:
                    filtered.append(item)
            except Exception:
                filtered.append(item)
        else:
            filtered.append(item)
    log.info("処理対象: %d レース（締切前: %d / 全体: %d）", len(filtered), len(filtered), len(race_list))
    race_list = filtered

    # ── Playwright一括取得（展示タイムがないレースをまとめて取得）──
    no_ex_races = [
        (vn, rno) for vn, rno, boats, *_ in race_list
        if not any(b.ex_time and b.ex_time > 0 for b in boats)
    ]
    if no_ex_races:
        log.info("Playwright一括取得: %d レース", len(no_ex_races))
        race_date_str = str(race_date).replace("-", "")
        pw_cache = _scrape_beforeinfo_bulk(no_ex_races, race_date_str)
        log.info("Playwright取得完了: %d レース", len(pw_cache))
    else:
        pw_cache = {}

    # ── 荒れ判定 & 通知 ──────────────────────────────────────
    notified = 0
    # 送信済みレースを記録（日付_場コード_レース番号）
    sent_file = f"sent_{race_date}.txt"
    try:
        with open(sent_file, "r") as sf:
            sent_set = set(sf.read().splitlines())
    except Exception:
        sent_set = set()
    for venue_num, race_number, boats, weather, *_ in race_list:
        # 展示タイムがない場合はPlaywrightで一括取得済みデータを使う
        ex_times = [b.ex_time for b in boats if b.ex_time is not None and b.ex_time > 0]
        if not ex_times:
            scraped = pw_cache.get((venue_num, race_number), {})
            if scraped:
                # 艇別データを補完
                boats_data = scraped.get("boats", {})
                for b in boats:
                    if b.lane in boats_data:
                        b.ex_time = boats_data[b.lane]["ex_time"]
                        b.ex_st   = boats_data[b.lane]["ex_st"]
                        b.tilt    = boats_data[b.lane]["tilt"]
                # 気象データを補完
                wd = scraped.get("weather", {})
                if wd:
                    from dataclasses import replace as dc_replace
                    weather = WeatherInfo(
                        wind_speed     = wd.get("wind_speed",     weather.wind_speed),
                        wind_direction = wd.get("wind_direction", weather.wind_direction),
                        wave_height    = wd.get("wave_height",    weather.wave_height),
                        weather        = weather.weather,
                    )
        # 展示タイムが取れなければスキップ
        ex_times = [b.ex_time for b in boats if b.ex_time is not None and b.ex_time > 0]
        if not ex_times:
            continue
        try:
            score, detail, target = calculate_upset_score(boats, weather)

            # ── MLモデルスコアを加算 ──────────────────────────────
            ml_probs = _predict_win_prob(boats)
            if ml_probs:
                prob_1 = ml_probs.get(1, 0.0)
                # 1号艇以外の最有力艇の確率と比較
                other_probs = {l: p for l, p in ml_probs.items() if l != 1}
                best_other_lane = max(other_probs, key=other_probs.get) if other_probs else None
                best_other_prob = other_probs.get(best_other_lane, 0.0)
                # 1号艇より他艇が強い場合スコア加算
                if best_other_prob > prob_1 * 0.5:
                    ml_score = best_other_prob * 6.0  # 最大6点
                    score += ml_score
                    detail["MLスコア"] = f"対抗{best_other_lane}号艇ML確率{best_other_prob:.2f} +{ml_score:.2f}点"
                    sorted_lanes = sorted(
                        [b.lane for b in boats if b.lane != 1],
                        key=lambda l: ml_probs.get(l, 0), reverse=True
                    )
                    target = sorted_lanes[:3]

            log.debug("スコア: %s %dR score=%.2f", VENUE_NAMES.get(venue_num,f"場{venue_num}"), race_number, score)
            if score < UPSET_SCORE_THRESHOLD:
                log.debug("スコア不足: %s %dR score=%.2f",
                          VENUE_NAMES.get(venue_num, f"場{venue_num}"), race_number, score)
                continue

            race_key = f"{race_date}_{venue_num}_{race_number}"
            if race_key in sent_set:
                log.debug("送信済みスキップ: %s %dR", VENUE_NAMES.get(venue_num, f"場{venue_num}"), race_number)
                continue

            result = RaceResult(
                venue_name   = VENUE_NAMES.get(venue_num, f"場{venue_num}"),
                venue_num    = venue_num,
                race_number  = race_number,
                boats        = boats,
                weather      = weather,
                upset_score  = score,
                score_detail = detail,
                target_lanes = target,
            )

            log.info(
                "荒れ検知: %s %dR score=%.2f target=%s",
                result.venue_name, result.race_number,
                result.upset_score, result.target_lanes,
            )

            subject, body = build_message(result)
            if send_email(subject, body):
                notified += 1
                # 送信済みに記録
                sent_set.add(race_key)
                try:
                    with open(sent_file, "w") as sf:
                        sf.write("\n".join(sent_set))
                    # GitHubにコミットして永続化
                    os.system('git config user.email "action@render.com"')
                    os.system('git config user.name "Render Bot"')
                    os.system(f"git add {sent_file}")
                    os.system(f'git commit -m "update sent races [skip ci]"')
                    # GITHUB_TOKENを使ってpush
                    gh_token = os.getenv("GITHUB_TOKEN", "")
                    gh_repo  = os.getenv("GITHUB_REPO", "sinrai74/my-app")
                    if gh_token:
                        remote = f"https://{gh_token}@github.com/{gh_repo}.git"
                        os.system(f"git push {remote} main")
                    else:
                        os.system("git push")
                except Exception as ge:
                    log.warning("sent_file保存失敗: %s", ge)

            # Gmail レート制限対策（連続送信を避ける）
            time.sleep(1.0)

        except Exception as e:
            log.error(
                "レース処理中に例外: %s %dR  error=%s",
                VENUE_NAMES.get(venue_num, f"場{venue_num}"),
                race_number, e,
                exc_info=True,
            )
            # 1 レースのエラーで全体を止めない

    if notified == 0:
        send_summary_if_none(len(race_list))

    log.info("▶ 完了 通知件数=%d / 確認レース=%d", notified, len(race_list))


# ════════════════════════════════════════════════════════════
# エントリポイント
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ボートレース荒れ検知 Gmail 通知")
    parser.add_argument(
        "--date", "-d",
        default=None,
        help="対象日付 YYYYMMDD（省略時: 今日）",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=UPSET_SCORE_THRESHOLD,
        help=f"荒れスコア閾値（デフォルト: {UPSET_SCORE_THRESHOLD}）",
    )
    args = parser.parse_args()

    UPSET_SCORE_THRESHOLD = args.threshold
    run(race_date=args.date)
