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
RESULTS_URL  = "https://boatraceopenapi.github.io/results/v2"

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
UPSET_SCORE_THRESHOLD = 5.0   # デフォルト閾値

# ── 場ごとの荒れスコア閾値 ────────────────────────────────────
# 🔥超荒れ(4.8): 江戸川・平和島・戸田
# 🌪️荒れやすい(4.8): 宮島・若松・下関・唐津
# ⚖️バランス(5.0): 住之江・尼崎・常滑・芦屋・福岡・大村・蒲郡・びわこ・徳山
# ⚖️やや強め(5.5): 多摩川・浜名湖・三国・鳴門・丸亀
# 💪イン超強い(6.0): 桐生・津・児島
VENUE_THRESHOLDS: dict[int, float] = {
    1:  6.0,  # 桐生   イン超強い
    2:  4.8,  # 戸田   超荒れ
    3:  4.8,  # 江戸川 超荒れ
    4:  4.8,  # 平和島 超荒れ
    5:  5.5,  # 多摩川 やや強め
    6:  5.5,  # 浜名湖 やや強め
    7:  5.0,  # 蒲郡   バランス
    8:  5.0,  # 常滑   バランス
    9:  6.0,  # 津     イン超強い
    10: 5.5,  # 三国   やや強め
    11: 5.0,  # びわこ バランス
    12: 5.0,  # 住之江 バランス
    13: 5.0,  # 尼崎   バランス
    14: 5.5,  # 鳴門   やや強め
    15: 5.5,  # 丸亀   やや強め
    16: 6.0,  # 児島   イン超強い
    17: 4.8,  # 宮島   荒れやすい
    18: 5.0,  # 徳山   バランス
    19: 4.8,  # 下関   荒れやすい
    20: 4.8,  # 若松   荒れやすい
    21: 5.0,  # 芦屋   バランス
    22: 5.0,  # 福岡   バランス
    23: 4.8,  # 唐津   荒れやすい
    24: 5.0,  # 大村   バランス
}

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
    racer_class: str     = ""    # 等級（A1/A2/B1/B2）


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
    target_lanes: list[int]     = field(default_factory=list)
    odds_map:    dict           = field(default_factory=dict)
    best_combo:  str            = ""
    best_ev:     float          = 0.0
    race_grade:  int            = 0
    recommended_bets: list      = field(default_factory=list)
    closed_at:   str            = ""  # 締切時刻


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


def _fetch_beforeinfo_api(venue_num: int, race_number: int, race_date: str) -> dict:
    """
    ① BoatraceOpenAPI previews から展示タイム・ST・気象を取得する（軽量・高速）。
    戻り値: {"boats": {lane: {ex_time, ex_st, tilt}}, "weather": {...}}
    展示タイムが1艇も取れなかった場合は {} を返す。
    """
    url = f"{PREVIEWS_URL}/{race_date[:4]}/{race_date}.json"
    data = _safe_get(url)
    if not data:
        return {}

    previews = data.get("previews", [])
    target = next(
        (p for p in previews
         if p.get("race_stadium_number") == venue_num
         and p.get("race_number") == race_number),
        None
    )
    if target is None:
        return {}

    boats: dict[int, dict] = {}
    boats_raw = target.get("boats", {})
    pb_list = list(boats_raw.values()) if isinstance(boats_raw, dict) else boats_raw
    for pb in pb_list:
        if not isinstance(pb, dict):
            continue
        lane = pb.get("racer_boat_number")
        if not lane:
            continue
        ex_time = pb.get("racer_exhibition_time")
        ex_st   = pb.get("racer_start_timing")
        tilt    = pb.get("racer_tilt_adjustment")
        boats[lane] = {
            "ex_time": float(ex_time) if ex_time and float(ex_time) > 0 else None,
            "ex_st":   float(ex_st)   if ex_st   is not None else None,
            "tilt":    float(tilt)    if tilt    is not None else None,
        }

    # 展示タイムが1艇も取れていなければ「まだ公開前」
    if not any(v["ex_time"] for v in boats.values()):
        return {}

    # 気象情報
    wd_num = target.get("race_wind_direction_number")
    wd_str = None
    if wd_num is not None:
        n = int(wd_num)
        wd_str = "追" if n in (1, 2) else "向" if n in (9, 10) else "横"
    ws  = target.get("race_wind")
    wh  = target.get("race_wave")
    wdc = target.get("race_weather_number")

    return {
        "boats": boats,
        "weather": {
            "wind_speed":     float(ws)  if ws  is not None else None,
            "wind_direction": wd_str,
            "wave_height":    int(wh)    if wh  is not None else None,
            "weather_label":  {1:"晴",2:"曇",3:"雨",4:"雪"}.get(int(wdc)) if wdc is not None else None,
        },
    }


def _scrape_beforeinfo_bs4(venue_num: int, race_number: int, race_date: str) -> dict:
    """
    ② fallback: requests + BeautifulSoup で公式サイトの直前情報ページを取得。
    Playwright 不要・軽量。
    戻り値: {"boats": {lane: {...}}, "weather": {...}}
    """
    import re as _re
    from bs4 import BeautifulSoup as _BS

    url = (f"https://www.boatrace.jp/owpc/pc/race/beforeinfo"
           f"?rno={race_number}&jcd={str(venue_num).zfill(2)}&hd={race_date}")
    try:
        r = requests.get(url, headers=HTTP_HEADERS, timeout=6)   # 6秒で打ち切り
        r.raise_for_status()
    except requests.Timeout:
        log.warning("[fallback] タイムアウト(6s): 場%d %dR → スキップ", venue_num, race_number)
        return {}
    except requests.RequestException as e:
        log.warning("[fallback] 直前情報スクレイピング失敗: %s", e)
        return {}

    soup = _BS(r.content, "html.parser")
    boats: dict[int, dict] = {}
    weather_data: dict = {}

    # ── 気象情報（テーブル内テキストから抽出）────────────────
    for txt in soup.get_text(separator=" ").split():
        m_ws = _re.match(r"^(\d+\.?\d*)m$", txt)
        m_wh = _re.match(r"^(\d+)cm$", txt)
        if m_ws:
            try:
                ws = float(m_ws.group(1))
                if 0 <= ws <= 30:
                    weather_data["wind_speed"] = ws
            except ValueError:
                pass
        if m_wh:
            try:
                weather_data["wave_height"] = int(m_wh.group(1))
            except ValueError:
                pass
    for kw, val in [("向かい風","向"),("追い風","追"),("横風","横")]:
        if kw in soup.get_text():
            weather_data["wind_direction"] = val
            break

    # ── 展示タイム・ST（tbody tr を走査）─────────────────────
    for table in soup.find_all("table"):
        rows = table.select("tbody tr")
        for i, tr in enumerate(rows):
            tds = [td.get_text(strip=True) for td in tr.find_all("td")]
            if not tds or not tds[0].isdigit():
                continue
            if not (1 <= int(tds[0]) <= 6):
                continue
            lane = int(tds[0])
            ex_time = ex_st = tilt = None
            try:
                for col in tds[1:]:
                    val = col.replace(",", "")
                    fv = float(val)
                    if 6.0 <= fv <= 7.5 and ex_time is None:
                        ex_time = fv
                    elif -1.0 <= fv <= 1.0 and tilt is None:
                        tilt = fv
            except ValueError:
                pass
            # ST は次数行に "ST" + 数値のパターン
            for j in range(i + 1, min(i + 5, len(rows))):
                sc = [td.get_text(strip=True) for td in rows[j].find_all("td")]
                if "ST" in sc:
                    idx = sc.index("ST")
                    for k in range(idx + 1, len(sc)):
                        try:
                            sv = sc[k]
                            sv = "0" + sv if sv.startswith(".") else sv
                            ex_st = float(sv)
                            break
                        except ValueError:
                            pass
                    break

            boats[lane] = {"ex_time": ex_time, "ex_st": ex_st, "tilt": tilt}

    if not any(v["ex_time"] for v in boats.values()):
        log.debug("[fallback] 展示タイムなし: 場%d %dR", venue_num, race_number)
        return {}

    return {"boats": boats, "weather": weather_data}


def _get_beforeinfo(venue_num: int, race_number: int, race_date: str) -> dict:
    """
    API優先・fallback付きで直前情報を取得する。

    ① BoatraceOpenAPI previews（軽量・高速）
       → 展示タイムが取れたら即返す
    ② requests + BeautifulSoup（fallback）
       → Playwright 不使用で軽量
    ③ 両方ダメ → {} を返す（展示タイムなしで続行）
    """
    # ① API
    result = _fetch_beforeinfo_api(venue_num, race_number, race_date)
    if result:
        log.debug("[API] 直前情報取得: 場%d %dR", venue_num, race_number)
        return result

    # ② fallback（BS4スクレイピング）
    log.info("[fallback] API取得失敗 → BS4スクレイピング: 場%d %dR", venue_num, race_number)
    result = _scrape_beforeinfo_bs4(venue_num, race_number, race_date)
    if result:
        return result

    # ③ 両方失敗
    log.debug("[fallback] 展示タイム取得不可: 場%d %dR（展示前or障害）", venue_num, race_number)
    return {}


def _get_beforeinfo_bulk(race_list: list, race_date: str) -> dict:
    """
    複数レースの直前情報を一括取得する。
    race_list: [(venue_num, race_number), ...]
    戻り値: {(venue_num, race_number): {"boats": {...}, "weather": {...}}}

    まず previews API を1回だけ叩いて全レース分を取得し、
    取れなかったレースだけ個別 fallback を呼ぶ。
    """
    results: dict = {}

    # ── ① previews API を1回だけ叩く ──────────────────────────
    url = f"{PREVIEWS_URL}/{race_date[:4]}/{race_date}.json"
    api_data = _safe_get(url)
    preview_map: dict[tuple, dict] = {}
    if api_data:
        for p in api_data.get("previews", []):
            key = (p.get("race_stadium_number"), p.get("race_number"))
            preview_map[key] = p

    # ── ② 各レースを処理 ──────────────────────────────────────
    fallback_targets = []
    for venue_num, race_number in race_list:
        key = (venue_num, race_number)
        preview = preview_map.get(key)
        if preview is None:
            fallback_targets.append(key)
            continue

        # APIデータから展示タイムを組み立て
        boats: dict[int, dict] = {}
        boats_raw = preview.get("boats", {})
        pb_list = list(boats_raw.values()) if isinstance(boats_raw, dict) else boats_raw
        for pb in pb_list:
            if not isinstance(pb, dict):
                continue
            lane = pb.get("racer_boat_number")
            if not lane:
                continue
            ex_time = pb.get("racer_exhibition_time")
            ex_st   = pb.get("racer_start_timing")
            tilt    = pb.get("racer_tilt_adjustment")
            boats[lane] = {
                "ex_time": float(ex_time) if ex_time and float(ex_time) > 0 else None,
                "ex_st":   float(ex_st)   if ex_st   is not None else None,
                "tilt":    float(tilt)    if tilt    is not None else None,
            }

        if not any(v["ex_time"] for v in boats.values()):
            # 展示タイム未公開 → fallback
            fallback_targets.append(key)
            continue

        # 気象情報
        wd_num = preview.get("race_wind_direction_number")
        wd_str = None
        if wd_num is not None:
            n = int(wd_num)
            wd_str = "追" if n in (1, 2) else "向" if n in (9, 10) else "横"
        ws  = preview.get("race_wind")
        wh  = preview.get("race_wave")
        wdc = preview.get("race_weather_number")

        results[key] = {
            "boats": boats,
            "weather": {
                "wind_speed":     float(ws) if ws is not None else None,
                "wind_direction": wd_str,
                "wave_height":    int(wh)   if wh is not None else None,
                "weather_label":  {1:"晴",2:"曇",3:"雨",4:"雪"}.get(int(wdc)) if wdc else None,
            },
        }

    api_count = len(results)

    # ── ③ fallback: 個別 BS4 スクレイピング ───────────────────
    if fallback_targets:
        log.info("[fallback] BS4スクレイピング対象: %d レース", len(fallback_targets))
        for idx, (venue_num, race_number) in enumerate(fallback_targets, 1):
            log.info("[fallback] %d/%d 取得中: 場%d %dR",
                     idx, len(fallback_targets), venue_num, race_number)
            r = _scrape_beforeinfo_bs4(venue_num, race_number, race_date)
            if r:
                results[(venue_num, race_number)] = r
                log.info("[fallback] ✅ 取得成功: 場%d %dR", venue_num, race_number)
            else:
                log.info("[fallback] ⬜ 取得失敗（展示前）: 場%d %dR", venue_num, race_number)
            if idx < len(fallback_targets):
                time.sleep(0.3)   # 最終レース後はsleepしない

    log.info("[直前情報] API=%d件 / fallback=%d件 / 計=%d件",
             api_count, len(results) - api_count, len(results))
    return results


# ── 後方互換ラッパー（既存コードから呼ばれる箇所用）──────────────
def _scrape_beforeinfo_bulk(race_list: list, race_date: str) -> dict:
    """後方互換: _get_beforeinfo_bulk に委譲"""
    return _get_beforeinfo_bulk(race_list, race_date)


def _scrape_beforeinfo(race_no: int, venue_code: int, race_date: str) -> dict:
    """後方互換: 単体取得"""
    result = _get_beforeinfo_bulk([(venue_code, race_no)], race_date)
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
            lane         = int(lane),
            name         = b.get("racer_name", f"{lane}号艇"),
            win_rate     = float(b.get("racer_national_top_1_percent") or 0),
            motor        = float(b.get("racer_assigned_motor_top_2_percent") or 0),
            avg_st       = float(b.get("racer_average_start_timing") or 0.18),
            racer_class  = str(b.get("racer_class") or b.get("racer_grade") or ""),
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
            val = float(pb["racer_exhibition_time"])
            boat.ex_time = val if val > 0 else None  # 0は未公開
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
        closed_at  = prog.get('race_closed_at', '')
        race_grade = prog.get('race_grade_number', 0) or 0
        results.append((int(vn), int(rno), boats, weather, closed_at, int(race_grade)))

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
    race_grade: int = 0,
) -> tuple:
    import math

    def logit(p: float) -> float:
        p = max(0.001, min(0.999, p))
        return math.log(p / (1 - p))

    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def add_effect(base_prob: float, effect: float) -> float:
        """確率空間で効果を加算（確率が崩れない）"""
        return sigmoid(logit(base_prob) + effect)

    # ── 各艇のベース確率計算 ──────────────────────────────────
    raw_scores = []
    for b in boats:
        s = calc_boat_score(b, boats, weather)
        raw_scores.append((b.lane, s))

    # softmaxで確率に変換（合計1になる）
    max_s = max(s for _, s in raw_scores)
    exp_scores = [(lane, math.exp(s - max_s)) for lane, s in raw_scores]
    total = sum(e for _, e in exp_scores)
    lane_probs = {lane: e / total for lane, e in exp_scores}

    boat1_prob = lane_probs.get(1, 1/6)
    boat1 = next((b for b in boats if b.lane == 1), None)
    boat2 = next((b for b in boats if b.lane == 2), None)

    # 1号艇以外の最有力艇
    other_probs = {l: p for l, p in lane_probs.items() if l != 1}
    best_other_lane = max(other_probs, key=other_probs.get) if other_probs else 2
    best_other_prob = other_probs.get(best_other_lane, 0.0)

    # 荒れ確率の初期値 = 1号艇以外が勝つ確率
    upset_prob = 1.0 - boat1_prob

    # ── ①1号艇の弱さ（確率空間で加算）──────────────────────────
    if boat1:
        # ST系（1つにまとめる）
        st_risk = 0.0
        if boat1.avg_st > 0.17:
            st_risk += 0.3
        if boat1.ex_st is not None and boat1.ex_st > 0.16:
            st_risk += 0.3
        if st_risk > 0:
            upset_prob = add_effect(upset_prob, st_risk)

        # 勝率系（1つにまとめる）
        wr_risk = 0.0
        if boat1.win_rate < 5.0:
            wr_risk += 0.5
        elif boat1.win_rate < 5.5:
            wr_risk += 0.2
        if wr_risk > 0:
            upset_prob = add_effect(upset_prob, wr_risk)

        # 装備系（1つにまとめる）
        eq_risk = 0.0
        if boat1.motor < 35.0:
            eq_risk += 0.2
        if boat1.racer_class in ("B1", "B2"):
            eq_risk += 0.3
        if eq_risk > 0:
            upset_prob = add_effect(upset_prob, eq_risk)

    # ── ②2号艇の強さ（MLベース確率で比較）────────────────────
    boat2_prob = lane_probs.get(2, 0.0)
    if boat2 and boat2_prob > boat1_prob:
        effect = (boat2_prob - boat1_prob) * 2.0
        upset_prob = add_effect(upset_prob, effect)

    # ── ③展示系特徴量 ────────────────────────────────────────
    ex_all = [(b.lane, b.ex_time) for b in boats if b.ex_time and b.ex_time > 0]
    st_all = [(b.lane, b.ex_st) for b in boats if b.ex_st is not None]

    # 展示順位の分散（バラけ＝荒れ）
    if len(ex_all) >= 4:
        import statistics
        ex_times_vals = [t for _, t in ex_all]
        ex_std = statistics.stdev(ex_times_vals)
        if ex_std > 0.05:
            upset_prob = add_effect(upset_prob, ex_std * 3.0)

    # STの分散（バラけ＝事故）
    if len(st_all) >= 4:
        st_vals = [s for _, s in st_all]
        st_std = statistics.stdev(st_vals)
        if st_std > 0.04:
            upset_prob = add_effect(upset_prob, st_std * 2.0)

    # 展示隊形（外が速い＝まくり）
    inner_times = [t for l, t in ex_all if l in [1,2,3]]
    outer_times = [t for l, t in ex_all if l in [4,5,6]]
    inner_avg = outer_avg = None
    if inner_times and outer_times:
        inner_avg = sum(inner_times) / len(inner_times)
        outer_avg = sum(outer_times) / len(outer_times)
        if outer_avg < inner_avg:
            effect = (inner_avg - outer_avg) * 20.0
            upset_prob = add_effect(upset_prob, min(effect, 0.5))

    # ── ④トリガー（足すのではなく倍率）─────────────────────
    trigger = 0
    if boat1 and boat1.ex_st and boat1.ex_st > 0.18:
        trigger += 1
    if boat2 and boat2.ex_st and boat2.ex_st < 0.13:
        trigger += 1
    if inner_avg and outer_avg and outer_avg < inner_avg:
        trigger += 1
    if trigger >= 2:
        upset_prob = min(upset_prob * 1.2, 0.95)  # 倍率で

    # ── ⑤天候補正（倍率）────────────────────────────────────
    if weather:
        if weather.wind_speed and weather.wind_speed >= 5.0:
            wind_mult = 1.0 + min((weather.wind_speed - 5.0) * 0.02, 0.08)
            upset_prob = min(upset_prob * wind_mult, 0.95)
        if weather.weather and '雨' in weather.weather:
            upset_prob = min(upset_prob * 1.03, 0.95)
        if weather.wave_height and weather.wave_height >= 15:
            wave_mult = 1.0 + min((weather.wave_height - 15) * 0.003, 0.06)
            upset_prob = min(upset_prob * wave_mult, 0.95)

    # ── ⑥等級差（1号艇B級×他にA1）────────────────────────────
    if boat1 and boat1.racer_class in ("B1", "B2"):
        a1_others = [b for b in boats if b.lane != 1 and b.racer_class == "A1"]
        if a1_others:
            upset_prob = add_effect(upset_prob, 0.4)

    # ── ⑦レース種別補正 ──────────────────────────────────────
    grade_effects = {0: 0.0, 1: -0.1, 2: -0.2, 3: -0.3, 4: -0.5}
    upset_prob = add_effect(upset_prob, grade_effects.get(race_grade, 0.0))
    upset_prob = max(0.0, min(upset_prob, 0.95))

    # スコアは確率×10（0〜9.5の範囲）
    upset_score = upset_prob * 10.0

    # 通知条件：1号艇が65%超で最有力ならスコア0
    if boat1_prob > 0.65 and best_other_prob < boat1_prob:
        upset_score = 0.0

    # 狙い目（確率上位3艇、1号艇除く）
    target = sorted(other_probs, key=other_probs.get, reverse=True)[:3]

    # boat1_rankをprobs形式で計算
    sorted_lanes = sorted(lane_probs, key=lane_probs.get, reverse=True)
    boat1_rank = sorted_lanes.index(1) + 1 if 1 in sorted_lanes else 6

    grade_names = {0: '一般', 1: 'G3', 2: 'G2', 3: 'G1', 4: 'SG'}
    detail = {
        "荒れ確率":   f"{upset_prob:.1%}",
        "1号艇確率":  f"{boat1_prob:.1%}(rank{boat1_rank})",
        "最有力":     f"{best_other_lane}号艇({best_other_prob:.1%})",
        "展示":       f"{next((b.ex_time for b in boats if b.lane==1), None)}秒",
        "レース種別": grade_names.get(race_grade, f'grade{race_grade}'),
    }

    return upset_score, detail, target


def _generate_patterns(target_lanes: list[int], upset_score: float) -> dict[str, list]:
    """
    全三連単120通りを生成する。EVランキングで評価するため
    荒れスコアによる事前フィルタは行わない。
    target_lanes / upset_score は後段の参照用に引数として残す。
    """
    from itertools import permutations
    all_combos = [f"{a}-{b}-{c}" for a, b, c in permutations(range(1, 7), 3)]
    return {"all": all_combos}


def _evaluate_bets(
    patterns: dict[str, list],
    ml_probs: dict[int, float],
    odds_map: dict[str, float],
    bankroll: int = 10000,
    target_lanes: list[int] | None = None,
    has_exhibition: bool = True,
    boats: list | None = None,
) -> list[dict]:
    """
    全三連単120通りをEV順でランキングして推奨買い目を返す。

    パラメータ設定:
      EV_MIN   = 1.28 (低倍) / 1.38 (中倍) / 1.48 (高倍)
      MIN_PROB = 0.020
      MAX_ODDS = 80  （超万舟は的中率崩壊するため除外）
      MAX_BETS = 5   （買い目を絞る）
    """
    # ── ベースパラメータ ──────────────────────────────────────
    EV_LOW   = 1.28   # オッズ<40
    EV_MID   = 1.38   # オッズ40〜80
    EV_HIGH  = 1.48   # オッズ>80（万舟）※使わないがフォールバック用
    MIN_PROB = 0.020
    MAX_ODDS = 80     # これ以上は的中率崩壊 → 除外
    MAX_BETS = 5      # 買い目数上限

    # 展示なし → さらに厳しく（精度低下分を閾値に反映）
    if not has_exhibition:
        EV_LOW   += 0.20
        EV_MID   += 0.20
        MIN_PROB += 0.008

    # ── 展示タイム順位ボーナス（STより安定）─────────────────
    ex_rank_bonus = {1: 1.20, 2: 1.10, 3: 1.00, 4: 0.92, 5: 0.85, 6: 0.78}
    ex_rank: dict[int, int] = {}
    if boats and has_exhibition:
        ex_data = [(b.lane, b.ex_time) for b in boats
                   if b.ex_time and b.ex_time > 0]
        ex_data.sort(key=lambda x: x[1])   # タイム昇順（速い=1位）
        ex_rank = {lane: rank + 1 for rank, (lane, _) in enumerate(ex_data)}

    # ── 4カド攻め補正（4コースが1号艇に迫る場合）────────────
    lane4_boost = 0.0
    if ml_probs:
        p1 = ml_probs.get(1, 0.0)
        p4 = ml_probs.get(4, 0.0)
        if p4 > p1 * 0.9:
            lane4_boost = 1.2   # 荒れスコア+1.2相当の買い意欲

    # ── 三連単確率計算（Luce + 各種補正）────────────────────
    def trifecta_prob(a: int, b: int, c: int) -> float:
        pa = ml_probs.get(a, 0.001)
        rem1 = {l: p for l, p in ml_probs.items() if l != a}
        t1_  = sum(rem1.values()) or 1
        pb   = rem1.get(b, 0.001) / t1_
        rem2 = {l: p for l, p in rem1.items() if l != b}
        t2_  = sum(rem2.values()) or 1
        pc   = rem2.get(c, 0.001) / t2_

        prob = pa * pb * pc

        # 1号艇2着残り補正（荒れでも2着1号艇は多い）
        if a != 1 and b == 1:
            prob *= 1.15

        # 1残り荒れ補正（1号艇1着でオッズ15以上 = 意外な荒れ）
        if a == 1:
            odds_val = odds_map.get(f"{a}-{b}-{c}", 0)
            if odds_val >= 15:
                prob *= 1.12

        # 6号艇1着は現実的に少ない → 減点
        if a == 6:
            prob *= 0.70

        # コース補正を2・3着にも適用（展開反映）
        course_bonus = {1: 1.25, 2: 1.10, 3: 1.05, 4: 0.95, 5: 0.85, 6: 0.75}
        prob *= course_bonus.get(b, 1.0)
        prob *= course_bonus.get(c, 1.0)

        # 4カド補正
        if a == 4 and lane4_boost > 0:
            prob *= (1.0 + lane4_boost * 0.05)

        # 展示タイム順位ボーナス（1着艇に適用）
        if a in ex_rank:
            prob *= ex_rank_bonus.get(ex_rank[a], 1.0)

        return prob

    # パターンラベル付与
    tgt = target_lanes or []
    t1  = tgt[0] if tgt else None

    def _label(a: int, b: int, c: int) -> str:
        if a == 1:                            return "本命軸"
        if a == 4 and lane4_boost > 0:        return "4カド"
        if a in [5, 6]:                       return "万舟"
        if t1 and a == t1 and b == 1:         return "差し"
        if t1 and a == t1 and b != 1:         return "まくり"
        return "本命崩れ"

    # ── 全通り評価 ────────────────────────────────────────────
    from itertools import permutations
    candidates = []
    for a, b, c in permutations(range(1, 7), 3):
        combo = f"{a}-{b}-{c}"
        odds  = odds_map.get(combo, 0)

        # オッズ上限（超万舟は除外）
        if odds <= 0 or odds > MAX_ODDS:
            continue

        prob = trifecta_prob(a, b, c)

        # prob下限
        if prob < MIN_PROB:
            continue

        # オッズ帯別EV閾値
        ev_threshold = EV_MID if odds >= 40 else EV_LOW
        ev = prob * odds
        if ev < ev_threshold:
            continue

        candidates.append({
            "combo":   combo,
            "pattern": _label(a, b, c),
            "prob":    round(prob, 5),
            "odds":    odds,
            "ev":      round(ev, 3),
        })

    # EV降順 → 上位MAX_BETSのみ
    candidates.sort(key=lambda x: x["ev"], reverse=True)
    top = candidates[:MAX_BETS]

    # ケリー基準で賭け金計算
    for t in top:
        b_val = t["odds"] - 1
        if b_val <= 0:
            t["amount"] = 0
            continue
        kelly = (t["prob"] * t["odds"] - 1) / b_val
        kelly = max(0.0, min(kelly, 0.15))
        t["amount"] = int(bankroll * kelly / 100) * 100

    return top


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
    w     = result.weather
    label = _danger_label(result.upset_score)

    # ── 件名 ─────────────────────────────────────────────────
    top_ev_str = ""
    if result.recommended_bets:
        top_ev_str = f" EV:{result.recommended_bets[0]['ev']:.2f}"

    subject = (
        f"【荒れ検知】{result.venue_name} {result.race_number}R "
        f"{label} (score:{result.upset_score:.1f}{top_ev_str})"
    )

    # ── 本文（LINEとメール共通・コンパクト版）────────────────
    grade_names = {0: '', 1: '🏆G3', 2: '🏆G2', 3: '🏆G1', 4: '🏆SG'}
    grade_label = grade_names.get(result.race_grade, '')

    # 締切時刻をJSTで表示
    closed_str = ""
    if result.closed_at:
        try:
            from datetime import datetime as _dt
            closed_dt = _dt.strptime(result.closed_at, "%Y-%m-%d %H:%M:%S")
            closed_str = f" 締切{closed_dt.strftime('%H:%M')}"
        except Exception:
            pass

    lines = [
        f"【荒れ検知】{result.venue_name} {result.race_number}R {grade_label}{closed_str}".strip(),
        f"危険度: {label}  スコア: {result.upset_score:.1f}",
        "",
    ]

    # 気象情報
    weather_parts: list[str] = []
    if w.wind_speed  is not None: weather_parts.append(f"風{w.wind_speed:.1f}m{w.wind_direction or ''}")
    if w.wave_height is not None: weather_parts.append(f"波{w.wave_height}cm")
    if w.weather     is not None: weather_parts.append(w.weather)
    if weather_parts:
        lines.append("🌊 " + " / ".join(weather_parts))

    lines.append("")

    # 展示タイム（コンパクト版）
    sorted_boats = sorted(result.boats, key=lambda b: b.ex_time or 99)
    lines.append("📊 展示タイム")
    for b in sorted(result.boats, key=lambda b: b.lane):
        et   = f"{b.ex_time:.2f}" if b.ex_time else "——"
        st   = f"{b.ex_st:.2f}"   if b.ex_st is not None else "——"
        rank = sorted_boats.index(b) + 1 if b.ex_time else "-"
        marker = "★" if b.lane == 1 else "  "
        # STマイナス（フライング注意）
        f_warn = "⚠F" if b.ex_st is not None and b.ex_st < 0 else ""
        cls = f"[{b.racer_class}]" if b.racer_class else ""
        lines.append(f"{marker}{b.lane}:{b.name[:3]}{cls} {et}({rank}位) ST{st}{f_warn}")

    lines.append("")

    # 狙い目
    if result.target_lanes:
        tgt = "-".join(str(l) for l in result.target_lanes)
        lines.append(f"🎯 狙い: {tgt}-全")

    # オッズ（現実的なもののみ・上位3点）
    if result.odds_map and result.target_lanes:
        shown_odds = []
        for t1 in result.target_lanes[:2]:
            for t2 in [l for l in result.target_lanes if l != t1][:2]:
                for t3 in [l for l in range(1,7) if l != t1 and l != t2]:
                    combo = f"{t1}-{t2}-{t3}"
                    odds = result.odds_map.get(combo, 0)
                    if 0 < odds <= 300:
                        shown_odds.append((combo, odds))
        shown_odds.sort(key=lambda x: x[1])
        if shown_odds:
            lines.append("💴 " + "  ".join(f"{c}:{o:.0f}倍" for c, o in shown_odds[:4]))

    # 推奨買い目（EV順ランキング）
    if result.recommended_bets:
        lines.append("💰 EV上位買い目（期待値順）")
        pattern_jp = {
            "本命軸": "本命軸", "本命崩れ": "本命崩れ",
            "差し": "差し", "まくり": "まくり", "万舟": "万舟",
            # 旧パターン名後方互換
            "nami":"本命崩れ","makuri":"まくり","sashi":"差し","ana":"万舟","safe":"保険",
        }
        for bet in result.recommended_bets[:8]:
            pname = pattern_jp.get(bet["pattern"], bet["pattern"])
            amt   = f" ¥{bet['amount']:,}" if bet.get("amount", 0) > 0 else ""
            ev_bar = "★" * min(int(bet["ev"] * 2), 5)
            lines.append(
                f"  {bet['combo']}  {bet['odds']:.0f}倍 "
                f"EV:{bet['ev']:.2f}{ev_bar}  [{pname}]{amt}"
            )
    elif result.best_combo:
        odds_val = result.odds_map.get(result.best_combo, 0)
        lines.append(f"💰 推奨: {result.best_combo} ({odds_val:.0f}倍 EV:{result.best_ev:.2f})")

    return subject, "\n".join(lines)


# ════════════════════════════════════════════════════════════
# Gmail 送信
# ════════════════════════════════════════════════════════════

def send_line(body: str) -> bool:
    """
    LINE Messaging API でプッシュ通知を送信する。成功で True。

    必要な環境変数:
        LINE_TOKEN   … チャネルアクセストークン
        LINE_USER_ID … 送信先ユーザーID（Uから始まる）
    """
    token   = os.getenv("LINE_TOKEN", "")
    user_id = os.getenv("LINE_USER_ID", "")

    if not token or not user_id:
        log.debug("LINE_TOKEN / LINE_USER_ID 未設定 → LINE通知スキップ")
        return False

    try:
        import urllib.request, json as _json
        payload = _json.dumps({
            "to": user_id,
            "messages": [{"type": "text", "text": body[:2000]}]
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.line.me/v2/bot/message/push",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as res:
            if res.status == 200:
                log.info("LINE送信成功")
                return True
            else:
                log.error("LINE送信失敗: status=%d", res.status)
                return False
    except Exception as e:
        log.error("LINE送信エラー: %s", e)
        return False


def send_notification(subject: str, body: str) -> bool:
    """メールとLINEの両方に通知する。どちらか一方でも成功すればTrue。"""
    line_ok  = send_line(f"{subject}\n\n{body}")
    email_ok = send_email(subject, body)
    return line_ok or email_ok


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

    # ── 二重実行防止 ──────────────────────────────────────────
    lock_file = "notify_running.lock"
    if os.path.exists(lock_file):
        # ロックファイルが古い（10分以上）場合は削除して続行
        import time as _time
        lock_age = _time.time() - os.path.getmtime(lock_file)
        if lock_age < 600:
            log.info("別のプロセスが実行中のためスキップ (age=%.0fs)", lock_age)
            return
        else:
            os.remove(lock_file)
    # ロックファイル作成
    open(lock_file, "w").close()
    try:
        _run_main(race_date)
    finally:
        if os.path.exists(lock_file):
            os.remove(lock_file)


def _run_main(race_date: str | None = None) -> None:
    """メイン処理本体"""
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

    # ── 直前情報一括取得（展示タイムなし・締切15分以内のみ）──────
    # API優先 → 失敗時のみ BS4 fallback（Playwright不使用）
    pw_targets = []
    for item in race_list:
        vn, rno, boats = item[0], item[1], item[2]
        closed_at = item[4] if len(item) > 4 else ""
        if any(b.ex_time and b.ex_time > 0 for b in boats):
            continue  # 既に展示タイムあり
        if closed_at:
            try:
                closed_dt = datetime.strptime(closed_at, "%Y-%m-%d %H:%M:%S")
                minutes_to_close = (closed_dt - now).total_seconds() / 60
                if 0 < minutes_to_close <= 15:
                    pw_targets.append((vn, rno))
            except Exception:
                pass

    pw_cache = {}
    if pw_targets:
        log.info("直前情報取得: %d レース（締切15分以内）", len(pw_targets))
        race_date_str = str(race_date).replace("-", "")
        pw_cache = _get_beforeinfo_bulk(pw_targets, race_date_str)
        log.info("直前情報取得完了: %d レース", len(pw_cache))

        # 取得できたデータをboatsに反映
        for item in race_list:
            vn, rno, boats, weather = item[0], item[1], item[2], item[3]
            scraped = pw_cache.get((vn, rno), {})
            if not scraped:
                continue
            boats_data = scraped.get("boats", {})
            for b in boats:
                if b.lane in boats_data:
                    bd = boats_data[b.lane]
                    b.ex_time = bd.get("ex_time")
                    b.ex_st   = bd.get("ex_st")
                    b.tilt    = bd.get("tilt")
            wd = scraped.get("weather", {})
            if wd:
                from dataclasses import replace as _replace
                new_weather = WeatherInfo(
                    wind_speed     = wd.get("wind_speed",     weather.wind_speed),
                    wind_direction = wd.get("wind_direction", weather.wind_direction),
                    wave_height    = wd.get("wave_height",    weather.wave_height),
                    weather        = weather.weather,
                )
                # weatherを更新（tupleなので直接変更できないためrace_listを再構築）
                idx = race_list.index(item)
                race_list[idx] = (vn, rno, boats, new_weather) + item[4:]

    # ── 荒れ判定 & 通知 ──────────────────────────────────────
    notified = 0
    # 送信済みレースを記録（日付_場コード_レース番号）
    sent_file = f"sent_{race_date}.txt"
    try:
        with open(sent_file, "r") as sf:
            sent_set = set(sf.read().splitlines())
    except Exception:
        sent_set = set()
    for venue_num, race_number, boats, weather, *rest in race_list:
        race_grade = rest[1] if len(rest) > 1 else 0
        closed_at  = rest[0] if len(rest) > 0 else ""

        # 展示タイムの有無を記録
        ex_times = [b.ex_time for b in boats if b.ex_time is not None and b.ex_time > 0]
        has_exhibition = len(ex_times) > 0

        # 展示タイムなしの場合は締切15分以内のレースのみ通知
        if not has_exhibition and closed_at:
            try:
                closed_dt = datetime.strptime(closed_at, "%Y-%m-%d %H:%M:%S")
                minutes_to_close = (closed_dt - now).total_seconds() / 60
                if minutes_to_close > 15:
                    continue
            except Exception:
                pass
        try:
            score, detail, target = calculate_upset_score(boats, weather, race_grade)

            # ── MLモデルスコアを加算 ──────────────────────────────
            ml_probs = _predict_win_prob(boats)
            if ml_probs:
                prob_1 = ml_probs.get(1, 0.0)
                other_probs = {l: p for l, p in ml_probs.items() if l != 1}
                best_other_lane = max(other_probs, key=other_probs.get) if other_probs else None
                best_other_prob = other_probs.get(best_other_lane, 0.0)
                if best_other_prob > prob_1:
                    ml_score = min(best_other_prob * 8.0, 4.0)  # 上限4.0点
                    score += ml_score
                    detail["MLスコア"] = f"対抗{best_other_lane}号艇ML確率{best_other_prob:.2f} +{ml_score:.2f}点"
                    sorted_lanes = sorted(
                        [b.lane for b in boats if b.lane != 1],
                        key=lambda l: ml_probs.get(l, 0), reverse=True
                    )
                    target = sorted_lanes[:2]   # 本当に来る艇は2艇まで
            else:
                log.warning("ML予測失敗: %s %dR → MLスコアなし",
                            VENUE_NAMES.get(venue_num, f"場{venue_num}"), race_number)

            log.debug("スコア: %s %dR score=%.2f", VENUE_NAMES.get(venue_num,f"場{venue_num}"), race_number, score)

            # ── 展示タイムなし → EV閾値・prob下限を厳しくする（_evaluate_bets内で対処）
            if not has_exhibition:
                detail["展示補正"] = "展示タイムなし（EV閾値+0.20・prob下限+0.008）"

            # ═══════════════════════════════════════════════════════
            # ▼ EV計算を先に行う（EVフィルタが主軸）
            # ═══════════════════════════════════════════════════════
            recommended = []
            odds_map: dict = {}

            if ml_probs:
                try:
                    from odds_fetch import fetch_odds
                    race_date_str = str(race_date).replace("-", "")
                    odds_map = fetch_odds(race_number, str(venue_num).zfill(2), race_date_str) or {}
                except Exception as oe:
                    log.debug("オッズ取得失敗: %s", oe)

                if odds_map:
                    # 全120通りのEVランキング（荒れスコアによる事前絞り込みなし）
                    patterns = _generate_patterns(target, score)
                    recommended = _evaluate_bets(
                        patterns, ml_probs, odds_map,
                        target_lanes=target,
                        has_exhibition=has_exhibition,
                        boats=boats,
                    )
                    detail["EV上位"] = (
                        f"{recommended[0]['combo']} EV={recommended[0]['ev']:.2f}"
                        if recommended else "なし"
                    )

            # ── 気象条件フィルタ（複合条件で荒れやすさを判定）──────
            ws = weather.wind_speed      if weather else None
            wh = weather.wave_height     if weather else None
            wd = weather.wind_direction  if weather else None

            weather_bonus = 0
            if ws is not None and ws >= 5.0:   weather_bonus += 1
            if wh is not None and wh >= 5:     weather_bonus += 1
            if wd == "向":                     weather_bonus += 1

            if weather_bonus < 2:
                log.debug("気象条件不足: %s %dR 風%.1f%s 波%s bonus=%d",
                          VENUE_NAMES.get(venue_num, f"場{venue_num}"), race_number,
                          ws or 0, wd or "-", wh or 0, weather_bonus)
                continue

            # ── ①EVがある → 通知（荒れスコアはサブフィルタ）────────
            # ── ②EVなし  → 荒れスコア閾値でフォールバック判定 ──────
            venue_threshold = VENUE_THRESHOLDS.get(venue_num, UPSET_SCORE_THRESHOLD)
            if not has_exhibition:
                venue_threshold += 1.5

            has_ev_signal = bool(recommended)

            if not has_ev_signal:
                # オッズ取得できなかった / EVなし → 荒れスコアのみで判断
                if score < venue_threshold:
                    log.debug("スコア不足(EVなし): %s %dR score=%.2f threshold=%.1f",
                              VENUE_NAMES.get(venue_num, f"場{venue_num}"), race_number,
                              score, venue_threshold)
                    continue
            else:
                # EVあり → 荒れスコアが最低ラインを超えていればOK（閾値を緩める）
                loose_threshold = venue_threshold * 0.6
                if score < loose_threshold:
                    log.debug("スコア不足(EV有): %s %dR score=%.2f loose_threshold=%.1f",
                              VENUE_NAMES.get(venue_num, f"場{venue_num}"), race_number,
                              score, loose_threshold)
                    continue

            # ── 送信済みチェック ──────────────────────────────────
            race_key    = f"{race_date}_{venue_num}_{race_number}"
            race_key_ex = f"{race_date}_{venue_num}_{race_number}_ex"

            if has_exhibition:
                # 展示タイムあり：展示ありキーで未送信なら通知（展示なし送信済みでも再通知）
                if race_key_ex in sent_set:
                    log.debug("送信済みスキップ(展示あり): %s %dR",
                              VENUE_NAMES.get(venue_num, f"場{venue_num}"), race_number)
                    continue
                notify_key = race_key_ex
            else:
                # 展示タイムなし：通常キーで未送信なら通知（展示あり送信済みならスキップ）
                if race_key in sent_set or race_key_ex in sent_set:
                    log.debug("送信済みスキップ(展示なし): %s %dR",
                              VENUE_NAMES.get(venue_num, f"場{venue_num}"), race_number)
                    continue
                notify_key = race_key

            result = RaceResult(
                venue_name   = VENUE_NAMES.get(venue_num, f"場{venue_num}"),
                venue_num    = venue_num,
                race_number  = race_number,
                boats        = boats,
                weather      = weather,
                upset_score  = score,
                score_detail = detail,
                target_lanes = target,
                race_grade   = race_grade,
                closed_at    = closed_at,
                odds_map     = odds_map,
                recommended_bets = recommended,
            )

            if recommended:
                best = recommended[0]
                result.best_combo = best["combo"]
                result.best_ev    = best["ev"]
                detail["推奨"] = (
                    f"{best['combo']} "
                    f"({best['odds']:.0f}倍 EV:{best['ev']:.2f} "
                    f"パターン:{best['pattern']})"
                )
                log.info("EV上位: %s %dR 推奨=%s EV=%.2f (%d点)",
                         result.venue_name, race_number,
                         best["combo"], best["ev"], len(recommended))

            log.info(
                "荒れ検知: %s %dR score=%.2f target=%s",
                result.venue_name, result.race_number,
                result.upset_score, result.target_lanes,
            )

            subject, body = build_message(result)
            if send_notification(subject, body):
                notified += 1
                # 送信済みに記録
                sent_set.add(notify_key)
                try:
                    with open(sent_file, "w") as sf:
                        sf.write("\n".join(sent_set))
                    # GitHub Actions上のみgit操作（ローカル実行時はスキップ）
                    gh_token = os.getenv("GITHUB_TOKEN", "")
                    gh_repo  = os.getenv("GITHUB_REPO", "sinrai74/my-app")
                    if gh_token:
                        os.system('git config user.email "action@render.com"')
                        os.system('git config user.name "Render Bot"')
                        os.system("git checkout main")
                        os.system(f"git add {sent_file}")
                        os.system(f'git commit -m "update sent races [skip ci]"')
                        remote = f"https://{gh_token}@github.com/{gh_repo}.git"
                        os.system(f"git push {remote} main")
                    else:
                        log.debug("ローカル実行: git push スキップ（sent_fileはローカルに保存済み）")
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

    # ── 前日の結果照合 ────────────────────────────────────────
    _check_yesterday_results(race_date)

    # ── 翌日予告通知（21:00〜21:10のみ）────────────────────────
    send_preview_notification()

    # ── 連敗アラートチェック ──────────────────────────────────
    _check_losing_streak()

    # ── 選手データ自動更新（毎月1日のみ）────────────────────────
    update_fan_files()


def send_preview_notification() -> None:
    """
    翌日の注目レース予告を夜21:00頃に送信する。
    出走表APIから翌日データを取得して荒れそうなレースをまとめて通知。
    """
    from datetime import datetime, timezone, timedelta
    JST = timezone(timedelta(hours=9))
    now_jst = datetime.now(JST)

    # 21:00〜21:10のみ実行
    if not (21 * 60 <= now_jst.hour * 60 + now_jst.minute <= 21 * 60 + 10):
        return

    tomorrow = (now_jst + timedelta(days=1)).strftime("%Y%m%d")
    log.info("翌日予告通知開始: %s", tomorrow)

    try:
        programs = fetch_programs(tomorrow)
        if not programs:
            return

        # 選手スペックだけで荒れ候補を選出（展示タイムはまだない）
        candidates = []
        for prog in programs:
            vn  = prog.get("race_stadium_number")
            rno = prog.get("race_number")
            boats_raw = prog.get("boats", [])
            if not boats_raw:
                continue

            boats = _extract_boats_from_program(prog)
            if not boats:
                continue

            boat1 = next((b for b in boats if b.lane == 1), None)
            if not boat1:
                continue

            others = [b for b in boats if b.lane != 1]
            avg_other_win = sum(b.win_rate for b in others) / len(others) if others else 0
            avg_other_motor = sum(b.motor for b in others) / len(others) if others else 0

            # 1号艇が他艇より明らかに劣る場合を候補に
            win_diff   = avg_other_win   - boat1.win_rate
            motor_diff = avg_other_motor - boat1.motor

            if win_diff > 1.0 or motor_diff > 10.0:
                score = win_diff * 0.5 + motor_diff * 0.1
                venue_name = VENUE_NAMES.get(int(vn), f"場{vn}")
                candidates.append((score, venue_name, rno, boat1, others))

        if not candidates:
            log.info("翌日予告: 候補なし")
            return

        candidates.sort(reverse=True)
        top = candidates[:5]

        lines = [f"📅 明日({tomorrow[:4]}/{tomorrow[4:6]}/{tomorrow[6:]})の注目レース", ""]
        for score, vname, rno, b1, others in top:
            best_other = max(others, key=lambda b: b.win_rate)
            lines.append(f"🎯 {vname} {rno}R")
            lines.append(f"  1号艇:{b1.name[:3]} 勝率{b1.win_rate:.2f} モーター{b1.motor:.1f}%")
            lines.append(f"  対抗:{best_other.lane}号艇 {best_other.name[:3]} 勝率{best_other.win_rate:.2f}")
            lines.append("")

        body = "\n".join(lines)
        send_notification("【明日の注目レース予告】", body)
        log.info("翌日予告通知送信: %d レース", len(top))

    except Exception as e:
        log.warning("翌日予告通知失敗: %s", e)
    """公式サイトから3連単結果を取得"""
    try:
        import requests as _req
        url = (f"https://www.boatrace.jp/owpc/pc/race/raceresult"
               f"?rno={race_number}&jcd={str(venue_num).zfill(2)}&hd={race_date}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.boatrace.jp/",
        }
        r = _req.get(url, headers=headers, timeout=8)
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(r.content, "html.parser")
        # 3連単結果を探す
        for td in soup.find_all("td"):
            text = td.get_text(strip=True)
            if re.match(r"^\d-\d-\d$", text):
                # 払戻金を探す
                tr = td.find_parent("tr")
                if tr:
                    cells = [c.get_text(strip=True) for c in tr.find_all("td")]
                    for cell in cells:
                        clean = cell.replace(",", "").replace("円", "")
                        if clean.isdigit() and int(clean) > 100:
                            return {"combo": text, "payout": int(clean)}
    except Exception as e:
        log.debug("結果取得失敗: %s", e)
    return None


def _check_yesterday_results(today_date: str) -> None:
    """前日の送信済みレースと結果を照合してCSVに記録"""
    try:
        from datetime import datetime, timedelta
        today = datetime.strptime(today_date, "%Y%m%d")
        yesterday = (today - timedelta(days=1)).strftime("%Y%m%d")
        sent_file = f"sent_{yesterday}.txt"

        if not os.path.exists(sent_file):
            return

        with open(sent_file, "r") as f:
            sent_keys = [l.strip() for l in f if l.strip()]

        if not sent_keys:
            return

        log.info("前日結果照合: %d レース", len(sent_keys))
        records = []
        for key in sent_keys:
            # キー形式: 20260509_12_6 or 20260509_12_6_ex
            parts = key.replace("_ex", "").split("_")
            if len(parts) != 3:
                continue
            _, venue_num, race_number = parts
            result = _fetch_race_result(int(venue_num), int(race_number), yesterday)
            records.append({
                "date": yesterday,
                "venue": VENUE_NAMES.get(int(venue_num), f"場{venue_num}"),
                "race": race_number,
                "result_combo": result["combo"] if result else "不明",
                "payout": result["payout"] if result else 0,
            })

        if not records:
            return

        # CSVに追記
        import csv
        csv_file = "hit_record.csv"
        write_header = not os.path.exists(csv_file)
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["date","venue","race","result_combo","payout"])
            if write_header:
                writer.writeheader()
            writer.writerows(records)

        # GitHub Actions上のみgit操作（ローカル実行時はスキップ）
        gh_token = os.getenv("GITHUB_TOKEN", "")
        gh_repo  = os.getenv("GITHUB_REPO", "sinrai74/my-app")
        if gh_token:
            os.system('git config user.email "action@render.com"')
            os.system('git config user.name "Render Bot"')
            os.system(f"git add {csv_file}")
            os.system(f'git commit -m "update hit record {yesterday} [skip ci]"')
            os.system(f"git push https://{gh_token}@github.com/{gh_repo}.git main")
        else:
            log.debug("ローカル実行: git push スキップ（hit_recordはローカルに保存済み）")

        log.info("結果照合完了: %s に記録", csv_file)

    except Exception as e:
        log.warning("結果照合失敗: %s", e)


def _fetch_race_result(venue_num: int, race_number: int, race_date: str) -> dict | None:
    """
    BoatraceOpenAPI から前日レース結果（3連単着順・払戻）を取得する。
    戻り値: {"combo": "2-3-1", "payout": 4520} or None
    """
    url = f"{RESULTS_URL}/{race_date[:4]}/{race_date}.json"
    data = _safe_get(url)
    if not data:
        return None

    for r in data.get("results", []):
        if (r.get("race_stadium_number") == venue_num
                and r.get("race_number") == race_number):
            # 3連単着順
            boats = r.get("boats", [])
            if isinstance(boats, dict):
                boats = list(boats.values())
            order = sorted(
                [b for b in boats if isinstance(b, dict) and b.get("racer_rank")],
                key=lambda b: b.get("racer_rank", 99)
            )
            if len(order) >= 3:
                combo = "-".join(str(b.get("racer_boat_number", "?")) for b in order[:3])
            else:
                combo = "不明"

            # 3連単払戻
            payout = 0
            for pay in r.get("payouts", []):
                if isinstance(pay, dict) and pay.get("bet_type") in ("3T", "三連単", 7):
                    try:
                        payout = int(pay.get("payout_amount", 0))
                    except (ValueError, TypeError):
                        pass
                    break

            return {"combo": combo, "payout": payout}

    return None


def _check_losing_streak() -> None:
    """
    連続外れが5回以上続いたらLINE/メールでアラートを送信。
    hit_record.csvから直近の結果を読んで判定する。
    """
    try:
        import csv
        csv_file = "hit_record.csv"
        if not os.path.exists(csv_file):
            return

        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        if len(rows) < 5:
            return

        # 直近10件の的中判定（通知した組み合わせが結果に含まれるか）
        recent = rows[-10:]
        miss_count = 0
        for row in reversed(recent):
            result_combo = row.get("result_combo", "不明")
            notified_combo = row.get("notified_combo", "")
            # 通知した狙い目の1着が結果の1着と一致するか
            if result_combo == "不明":
                continue
            result_1st = result_combo.split("-")[0] if "-" in result_combo else ""
            notified_1st = notified_combo.split("-")[0] if "-" in notified_combo else ""
            if result_1st and notified_1st and result_1st != notified_1st:
                miss_count += 1
            else:
                break  # 的中したら連敗ストップ

        if miss_count >= 5:
            msg = f"⚠️ 連敗アラート\n直近{miss_count}回連続で外れています。\nスコア閾値を上げることを検討してください。\n現在の閾値: {UPSET_SCORE_THRESHOLD}"
            send_notification("【連敗アラート】", msg)
            log.warning("連敗アラート送信: %d連敗", miss_count)

    except Exception as e:
        log.debug("連敗アラートエラー: %s", e)


def update_fan_files() -> None:
    """
    毎月1日に最新のfanファイルを自動ダウンロードする。
    """
    from datetime import datetime, timezone, timedelta
    JST = timezone(timedelta(hours=9))
    now_jst = datetime.now(JST)

    # 毎月1日の8:20〜8:30のみ実行
    if now_jst.day != 1 or now_jst.hour != 8 or now_jst.minute > 30:
        return

    log.info("fanファイル自動更新開始")
    try:
        year2 = now_jst.strftime("%y")
        month = now_jst.strftime("%m")
        fan_name = f"fan{year2}{month}.txt"
        url = f"https://www.boatrace.jp/owpc/pc/extra/data/{fan_name}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://www.boatrace.jp/",
        }
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 200 and len(r.content) > 10000:
            with open(fan_name, "wb") as f:
                f.write(r.content)

            gh_token = os.getenv("GITHUB_TOKEN", "")
            gh_repo  = os.getenv("GITHUB_REPO", "sinrai74/my-app")
            if gh_token:
                os.system('git config user.email "action@render.com"')
                os.system('git config user.name "Render Bot"')
                os.system(f"git add {fan_name}")
                os.system(f'git commit -m "update fan file {fan_name} [skip ci]"')
                os.system(f"git push https://{gh_token}@github.com/{gh_repo}.git main")
            log.info("fanファイル更新完了: %s", fan_name)
        else:
            log.warning("fanファイルダウンロード失敗: status=%d", r.status_code)
    except Exception as e:
        log.warning("fanファイル更新エラー: %s", e)


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
