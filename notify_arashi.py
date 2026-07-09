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

# 【朝刊AI】previews由来データを使わない新スコアリングエンジン
from x_asahi_scoring import (
    calculate_upset_score_v2,
    calc_danger_score_v2,
    calc_rank_probabilities_v2,
    get_model_version as _asahi_model_version,
)

# 【購入判定分離】BuyScoreによる「予想」と「購入」の分離エンジン
from x_buyscore import apply_buyscore as _apply_buyscore

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

# 【③地元選手】開催場→所属支部（全18支部・24場、公式の支部管轄表に基づく）。
# 選手のfanファイル上の「支部」がここに一致すれば「地元選手」と判定する。
# 東京(江戸川/平和島/多摩川)・愛知(蒲郡/常滑)・山口(徳山/下関)・
# 福岡(若松/芦屋/福岡)の4支部だけは複数場を管轄する。
# 【重要】この表は情報表示のみに使う。スコア・ランキングには一切影響しない
# （加点は行わない方針で合意済み。将来加点を検討する場合は別途Phase A/B相当の
# 効果検証を経てから導入すること）。
VENUE_BRANCH_MAP: dict[int, str] = {
    1: "群馬",   2: "埼玉",   3: "東京",   4: "東京",   5: "東京",
    6: "静岡",   7: "愛知",   8: "愛知",   9: "三重",  10: "福井",
    11: "滋賀",  12: "大阪",  13: "兵庫",  14: "徳島", 15: "香川",
    16: "岡山",  17: "広島",  18: "山口",  19: "山口", 20: "福岡",
    21: "福岡",  22: "福岡",  23: "佐賀",  24: "長崎",
}
# fanファイルの支部表記ゆれ対策（同じ支部を指す別名を正規化する）
_BRANCH_ALIASES: dict[str, str] = {
    "京滋": "滋賀",
}


def is_local_racer(venue_num: int, branch: str) -> bool:
    """
    選手の支部が、指定した開催場の管轄支部と一致するか（＝地元選手か）を返す。
    branchが空文字・未取得の場合は安全側でFalseを返す。
    """
    if not branch:
        return False
    branch = _BRANCH_ALIASES.get(branch, branch)
    home = VENUE_BRANCH_MAP.get(venue_num)
    return home is not None and branch == home

# ── 荒れスコアリング閾値 ────────────────────────────────────
# スコアがこの値以上のレースのみ通知する
UPSET_SCORE_THRESHOLD = 6.0   # デフォルト閾値

# ── 強制スキップ場 ────────────────────────────────────────────
# この場コードのレースは理由に関わらず通知しない
VENUE_FORCE_SKIP: set[int] = {18}   # 18=徳山

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
    avg_st:    float     = 0.18  # 平均スタートタイミング（全コース合算）
    ex_time:   Optional[float] = None   # 展示タイム（直前情報）
    ex_st:     Optional[float] = None   # 展示ST（直前情報）
    tilt:      Optional[float] = None   # チルト角
    racer_class: str     = ""    # 等級（A1/A2/B1/B2）
    racer_id:  str       = ""    # 登録番号（fanファイル照合用）
    branch:    str       = ""    # 【③地元選手】支部（fanファイルから取得、情報表示のみ・加点なし）
    # コース別平均ST（fanファイルから取得、index 0=1コース〜5=6コース）
    # 0.0はデータなし（そのコースの進入実績がない）
    course_st:   list    = None  # 各コースの平均ST [0.17, 0.19, ...]
    course_nyuko: list   = None  # 各コースの進入回数 [46, 28, ...]
    course_rank:  list   = None  # 各コースのST順位平均 [1.8, 2.7, ...]

    # 【Ver4対応・暫定】calc_danger_score_v2(Ver4)が参照するが、まだ
    # 実データを取り込めていない属性群。AttributeErrorでの即クラッシュを
    # 防ぐため安全なデフォルト値を設定しておく。この状態だと該当する
    # Ver4のサブ項目（当地勝率以外）は「差なし」＝0点寄与として扱われ、
    # スコア自体は壊れずに動くが、まだそれらの項目は実質機能していない。
    # TODO: fanファイルの追加パース（能力指数・コース別着順回数・F/L回数）
    #       が実装され次第、正しい値に置き換えること。
    local_win:  float    = 0.0   # 当地勝率（下のpost_initで実データがあれば上書き）
    ability_curr: Optional[float] = None   # 今期能力指数（未実装、常にNone）
    ability_prev: Optional[float] = None   # 前期能力指数（未実装、常にNone）
    # 【重要】以下はゼロ埋めしない。0.0/[0,0,...]で埋めると「実績が
    # 本当に0%だった」という意味になり、calc_danger_score_v2側の
    # 「閾値未満なら加点」という判定に誤って引っかかって、データ未実装
    # なのに危険度を不当に押し上げてしまう。Noneのままにしておけば、
    # 呼び出し側の `if boat1.course_win_rate and ...` 等のガード節で
    # 正しく「データなし＝この項目は評価対象外」としてスキップされる。
    course_place_counts: list = None       # 各コースの[1着,2着,...,6着]回数（未実装）
    course_place_rate:   list = None       # 各コースの複勝率（未実装）
    course_win_rate:      list = None      # 各コースの1着率（未実装）
    course_f_count:        list = None     # 各コースのフライング回数（未実装）
    course_l_count:         list = None    # 各コースの出遅れ回数（未実装）

    def __post_init__(self):
        if self.course_st is None:
            self.course_st = [0.0] * 6
        if self.course_nyuko is None:
            self.course_nyuko = [0] * 6
        if self.course_rank is None:
            self.course_rank = [0.0] * 6
        # course_place_counts / course_place_rate / course_win_rate /
        # course_f_count / course_l_count は意図的にゼロ埋めしない
        # （Noneのままにして「データなし」を正しく表現する。上のコメント参照）


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
    """
    【朝刊AI】previews由来の展示タイム・展示STは特徴量から除外する。
    model_all.pkl の特徴量セットに展示タイム・展示ST列があっても
    常に NaN（欠損）として扱い、中央値補完のみで対応する。
    """
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
            # 展示タイム・展示STは【朝刊AI】ポリシーにより特徴量から除外
            # （row[...]への代入自体を行わず、np.nan(欠損)のまま中央値補完させる）
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
        from datetime import datetime, timezone, timedelta
        today_str = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d")
        if race_date == today_str:
            data = _safe_get(f"{PREVIEWS_URL}/today.json")
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
    if not api_data:
        from datetime import datetime, timezone, timedelta
        today_str = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d")
        if race_date == today_str:
            api_data = _safe_get(f"{PREVIEWS_URL}/today.json")
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


# ════════════════════════════════════════════════════════════
# fanファイル パーサー（公式仕様書に基づくコース別ST取得）
# ════════════════════════════════════════════════════════════

import glob as _glob

_FAN_CACHE: dict[str, dict[str, dict]] = {}  # {filename: {racer_id: data}}


def _get_fan_file() -> str:
    """最新のfanファイルを返す（fan2607.txt → fan2606.txt の順で探す）"""
    files = sorted(_glob.glob("fan*.txt"), reverse=True)
    return files[0] if files else ""


def _load_fan_file(filepath: str) -> dict[str, dict]:
    """
    fanファイルをパースして {登録番号: コース別STデータ} の辞書を返す。
    公式仕様書に基づくバイトオフセット:
      byte 0-3:   登録番号
      byte 4-19:  名前漢字（16byte）
      byte 20-34: 名前カナ（15byte）
      byte 35-38: 支部（4byte、全角2文字。【③地元選手】判定に使用）
      byte 58-61: 全国勝率（確認用）
      byte 79-81: 全国平均ST（3桁、例: 016 → 0.16）
      byte 82〜:  コース別データ（1コース=13byte × 6コース）
                  各コース: 進入回数(3)+複勝率(4)+ST平均(3)+ST順位(3)
    """
    if filepath in _FAN_CACHE:
        return _FAN_CACHE[filepath]

    result: dict[str, dict] = {}
    try:
        with open(filepath, "rb") as f:
            raw_all = f.read()
    except OSError as e:
        log.warning("[fan] ファイル読み込み失敗: %s", e)
        return result

    for line_raw in raw_all.split(b"\n"):
        line_raw = line_raw.rstrip(b"\r")
        if len(line_raw) < 160:  # コース別データが揃う最小長
            continue
        try:
            racer_id = line_raw[0:4].decode("ascii").strip()
            if not racer_id.isdigit():
                continue

            # 支部（全角2文字、shift_jis/cp932でデコード。数値項目と違いASCIIでは読めない）
            try:
                branch = line_raw[35:39].decode("cp932", errors="replace").strip()
            except Exception:
                branch = ""

            # 全国平均ST（確認用）
            avg_st_raw = line_raw[79:82].decode("ascii", errors="replace")
            avg_st_global = int(avg_st_raw) / 100 if avg_st_raw.isdigit() else 0.18

            # コース別データ（1〜6コース）
            course_st    = [0.0] * 6
            course_nyuko = [0]   * 6
            course_rank  = [0.0] * 6

            for c in range(6):
                base = 82 + c * 13
                if base + 13 > len(line_raw):
                    break
                nyuko_raw   = line_raw[base:base+3].decode("ascii", errors="replace")
                st_raw      = line_raw[base+7:base+10].decode("ascii", errors="replace")
                rank_raw    = line_raw[base+10:base+13].decode("ascii", errors="replace")

                if nyuko_raw.isdigit() and st_raw.isdigit() and rank_raw.isdigit():
                    nyuko = int(nyuko_raw)
                    st    = int(st_raw) / 100   # 015 → 0.15
                    rank  = int(rank_raw) / 100  # 240 → 2.40
                    course_st[c]    = st if nyuko > 0 else 0.0
                    course_nyuko[c] = nyuko
                    course_rank[c]  = rank if nyuko > 0 else 0.0

            result[racer_id] = {
                "avg_st_global": avg_st_global,
                "branch":        branch,
                "course_st":     course_st,
                "course_nyuko":  course_nyuko,
                "course_rank":   course_rank,
            }
        except Exception:
            continue

    _FAN_CACHE[filepath] = result
    log.info("[fan] %s: %d選手ロード", filepath, len(result))
    return result


def get_course_st(racer_id: str) -> dict:
    """
    登録番号からコース別STデータを取得する。
    fanファイルがなければ空データを返す（graceful degradation）。
    """
    fan_file = _get_fan_file()
    if not fan_file:
        return {}
    fan_data = _load_fan_file(fan_file)
    return fan_data.get(str(racer_id), {})


def format_course_st_table(boats: list) -> str:
    """
    全選手のコース別ST一覧テキストを生成する（新聞・メール向け）。
    各選手について「担当コース(lane)でのST平均」と「6コース中の順位」を表示する。

    表示例:
    ┌─ コース別ST（参考:前期）─────────────────
    │ 1号艇 山口剛 [1コースST 0.17 / 全6コース中1位]
    │ 2号艇 田村美 [1コースST 0.18 / 全6コース中2位]  ← 1コース実績で比較
    └──────────────────────────────────────────
    """
    if not boats:
        return ""

    fan_file = _get_fan_file()
    if not fan_file:
        return ""  # fanファイルなし → 非表示

    lines_out = ["【コース別ST参考（前期実績）】"]
    has_data = False

    for boat in sorted(boats, key=lambda b: b.lane):
        # その選手の lane コース（進入予定コース）のST平均を表示
        # ただし出走表上の lane と実際の進入コースは異なる場合があるため
        # 全6コースのSTを表示して比較できるようにする
        lane_idx = boat.lane - 1  # 0-indexed

        if not boat.racer_id or not any(v > 0 for v in boat.course_st):
            continue

        has_data = True

        # 1コース固定でのST（「1コースST」比較が要望なので全選手1コースSTも表示）
        st_1c = boat.course_st[0] if boat.course_nyuko[0] > 0 else None
        st_lane = boat.course_st[lane_idx] if boat.course_nyuko[lane_idx] > 0 else None

        # 全選手の1コースSTをリストアップして順位を計算
        parts = [f"{boat.lane}号艇 {boat.name}"]
        if st_lane is not None:
            parts.append(f"{boat.lane}コースST:{st_lane:.2f}")
        if st_1c is not None and boat.lane != 1:
            parts.append(f"(1コース:{st_1c:.2f})")
        if boat.course_nyuko[lane_idx] > 0:
            parts.append(f"進入{boat.course_nyuko[lane_idx]}回")

        lines_out.append("  " + "  ".join(parts))

    if not has_data:
        return ""

    return "\n".join(lines_out)


def format_course_st_ranking(boats: list) -> str:
    """
    6コース中の順位形式で表示するバージョン。
    各選手の担当コース(lane)のST平均を比較し、速い順に順位付けする。
    「6艇中何位か」をパッと見てわかる形式。

    出力例:
    【今日の1コースST速さ比較（前期実績）】
    1位 3号艇 田村美  1コースST 0.15（進入46回）
    2位 1号艇 山口剛  1コースST 0.17（進入38回）
    ...
    """
    if not boats:
        return ""

    fan_file = _get_fan_file()
    if not fan_file:
        return ""

    # 全選手の1コースSTを収集（1コース固定で比較）
    st_data = []
    for boat in boats:
        if not boat.racer_id:
            continue
        st_1c = boat.course_st[0]
        nyuko_1c = boat.course_nyuko[0]
        if nyuko_1c > 0:
            st_data.append((boat.lane, boat.name, st_1c, nyuko_1c))

    if not st_data:
        return ""

    # ST昇順（速い順）でソート
    st_data.sort(key=lambda x: x[2])

    lines_out = ["【1コースST速さ（前期実績・全選手比較）】"]
    for rank, (lane, name, st, nyuko) in enumerate(st_data, 1):
        bar = "●" * min(5, round(st * 20))  # ST 0.15 → 3個, 0.20 → 4個
        lines_out.append(f"  {rank}位 {lane}号艇 {name:4s}  ST:{st:.2f}  {bar}  （1コース進入{nyuko}回）")

    return "\n".join(lines_out)


def fetch_programs(race_date: str) -> list[dict]:
    """出走表を取得して programs リストを返す。失敗時は []。"""
    url = f"{PROGRAMS_URL}/{race_date[:4]}/{race_date}.json"
    data = _safe_get(url)
    if data is None:
        # today.jsonフォールバックは当日分のみ（翌日分取得時には使わない）
        from datetime import datetime, timezone, timedelta
        today_str = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d")
        if race_date == today_str:
            log.info("日付URL 404 → today.json にフォールバック (date=%s)", race_date)
            data = _safe_get(f"{PROGRAMS_URL}/today.json")
        else:
            log.warning("出走表が取得できませんでした（開催なし or API 障害）")
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
        from datetime import datetime, timezone, timedelta
        today_str = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d")
        if race_date == today_str:
            log.info("直前情報: 日付URL 404 → today.json にフォールバック (date=%s)", race_date)
            data = _safe_get(f"{PREVIEWS_URL}/today.json")
        else:
            log.info("直前情報: まだ公開前の可能性あり (date=%s)", race_date)
    if data is None:
        return []
    previews = data.get("previews", [])
    log.info("直前情報取得: %d レース (date=%s)", len(previews), race_date)
    return previews


# ════════════════════════════════════════════════════════════
# データ組み立て
# ════════════════════════════════════════════════════════════

def _extract_boats_from_program(program: dict) -> list[BoatInfo]:
    """
    出走表エントリから BoatInfo リストを作成する。
    fanファイルが存在する場合、登録番号でコース別ST・進入回数を補完する。
    """
    # fanファイルをロード（存在しなければ空辞書）
    fan_file = _get_fan_file()
    fan_data = _load_fan_file(fan_file) if fan_file else {}

    boats: list[BoatInfo] = []
    for b in program.get("boats", []):
        if not isinstance(b, dict):
            continue
        lane = b.get("racer_boat_number")
        if not lane:
            continue

        # 登録番号: BoatraceOpenAPI v2 では "racer_number" が登録番号（4桁）。
        # （racer_boat_number は艇番1-6、racer_number は選手登録番号で別物）
        racer_id_raw = str(b.get("racer_number") or "")
        racer_id = racer_id_raw.strip()

        # fanファイルからコース別ST・支部を取得
        fan_entry = fan_data.get(racer_id, {}) if racer_id else {}
        course_st    = fan_entry.get("course_st",    [0.0] * 6)
        course_nyuko = fan_entry.get("course_nyuko", [0]   * 6)
        course_rank  = fan_entry.get("course_rank",  [0.0] * 6)
        branch       = fan_entry.get("branch", "")

        api_avg_st = float(b.get("racer_average_start_timing") or 0.18)

        boats.append(BoatInfo(
            lane          = int(lane),
            name          = b.get("racer_name", f"{lane}号艇"),
            win_rate      = float(b.get("racer_national_top_1_percent") or 0),
            motor         = float(b.get("racer_assigned_motor_top_2_percent") or 0),
            avg_st        = api_avg_st,
            racer_class   = str(b.get("racer_class") or b.get("racer_grade") or ""),
            racer_id      = racer_id,
            branch        = branch,
            course_st     = course_st,
            course_nyuko  = course_nyuko,
            course_rank   = course_rank,
            # 【Ver4対応】当地勝率はBoatraceOpenAPIのプログラムデータに
            # 既に含まれているため、fanファイルの追加パースなしで実データ化できる。
            local_win     = float(b.get("racer_local_top_1_percent") or 0),
        ))
    return sorted(boats, key=lambda x: x.lane)


def _extract_preview_raw(preview: dict) -> dict:
    """
    【朝刊AI・教師データ用】previewから展示・気象の生データを抽出する。
    この結果は予想には使わず、hit_record.csv 等への保存を通じて
    Phase2 の特徴量重要度学習・精度検証にのみ用いる。
    """
    boats_raw: dict[int, dict] = {}
    boats_dict = preview.get("boats", {})
    pb_list = list(boats_dict.values()) if isinstance(boats_dict, dict) else boats_dict
    for pb in pb_list:
        if not isinstance(pb, dict):
            continue
        lane = pb.get("racer_boat_number")
        if lane is None:
            continue
        ex_time = pb.get("racer_exhibition_time")
        ex_st   = pb.get("racer_start_timing")
        boats_raw[lane] = {
            "ex_time": float(ex_time) if ex_time and float(ex_time) > 0 else None,
            "ex_st":   float(ex_st) if ex_st is not None else None,
            "tilt":    float(pb["racer_tilt_adjustment"]) if pb.get("racer_tilt_adjustment") is not None else None,
        }

    wd_num = preview.get("race_wind_direction_number")
    wd_str: Optional[str] = None
    if wd_num is not None:
        wd_num = int(wd_num)
        wd_str = "追" if wd_num in (1, 2) else "向" if wd_num in (9, 10) else "横"

    ws  = preview.get("race_wind")
    wh  = preview.get("race_wave")
    wdc = preview.get("race_weather_number")
    weather_label = {1: "晴", 2: "曇", 3: "雨", 4: "雪"}.get(int(wdc), str(wdc)) if wdc is not None else None

    return {
        "boats": boats_raw,
        "wind_speed":     float(ws) if ws is not None else None,
        "wind_direction": wd_str,
        "wave_height":    int(wh) if wh is not None else None,
        "weather":        weather_label,
    }


def _apply_preview_to_boats(
    boats: list[BoatInfo],
    preview: dict,
) -> WeatherInfo:
    """
    【非推奨・朝刊AIポリシーにより不使用】
    展示・気象を boats/WeatherInfo に反映する旧関数。
    予想には使わず、教師データ抽出には _extract_preview_raw を使う。
    互換性のためコードは残すが、build_race_data からは呼び出されない。
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
    出走表を組み立てる。
    【朝刊AI】previews（展示・気象）は予想用の boats/weather には反映しない。
    展示・気象の生データは _extract_preview_raw で別途抽出し、
    教師データ（hit_record.csv保存等）としてのみ利用する。
    予想に使う weather は常に空の WeatherInfo（全フィールドNone）。
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

        # 【朝刊AI】教師データ用: 展示・気象の生データを抽出してキャッシュに保存。
        # 予想には使わず、hit_record.csv保存等の学習・検証用途にのみ使う。
        if preview:
            try:
                _PREVIEW_RAW_CACHE[(int(vn), int(rno))] = _extract_preview_raw(preview)
            except Exception:
                pass

        # 予想用の weather は常に空（全フィールドNone）。
        # boats の ex_time/ex_st も _apply_preview_to_boats を呼ばないため常に None のまま。
        weather = WeatherInfo()

        closed_at  = prog.get('race_closed_at', '')
        race_grade = prog.get('race_grade_number', 0) or 0
        results.append((int(vn), int(rno), boats, weather, closed_at, int(race_grade)))

    log.info("レース組み立て完了: %d レース（【朝刊AI】展示・気象は予想に不使用、教師データとしてのみ保存）", len(results))
    return results


# 【朝刊AI・教師データ用】(venue_num, race_number) → 展示・気象の生データ
# build_race_data 実行時に埋まる。予想には使わず、結果照合時に hit_record.csv へ保存する。
_PREVIEW_RAW_CACHE: dict[tuple[int, int], dict] = {}


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
    """
    【非推奨・朝刊AIポリシーにより不使用】
    展示タイム・展示ST・気象データに依存する旧スコアリング関数。
    実際の予想には x_asahi_scoring.calc_boat_score_v2 を使用する
    （previews由来データを一切参照しない朝データのみのエンジン）。
    互換性のためコードは残すが、どこからも呼び出されない。
    """
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
        if wh >= 3:
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
    venue_num: int = 0,
    is_night: bool = False,
) -> tuple:
    """
    【非推奨・朝刊AIポリシーにより不使用】
    展示タイム・展示ST・気象データに依存する旧予想エンジン。
    実際の予想には x_asahi_scoring.calculate_upset_score_v2 を使用する
    （previews由来データを一切参照しない朝データのみのエンジン）。
    互換性のためコードは残すが、どこからも呼び出されない。
    """
    import math

    def logit(p: float) -> float:
        p = max(0.001, min(0.999, p))
        return math.log(p / (1 - p))

    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def add_effect(base_prob: float, effect: float) -> float:
        return sigmoid(logit(base_prob) + effect)

    # ── ④ 会場補正（イン強い場は荒れにくい）──────────────────
    # 値が大きい = 荒れやすい(+) / 小さい = イン強い(-)
    VENUE_UPSET_FACTOR: dict[int, float] = {
        1:  0.10,  # 桐生   荒れやすい
        2:  0.05,  # 戸田
        3:  0.15,  # 江戸川 超荒れ
        4:  0.10,  # 平和島 荒れやすい
        5:  0.00,  # 多摩川 標準
        6: -0.05,  # 浜名湖 やや安定
        7:  0.00,  # 蒲郡
        8: -0.05,  # 常滑
        9: -0.10,  # 津     イン強い
        10: 0.00,  # 三国
        11: 0.00,  # びわこ
        12:-0.05,  # 住之江 やや安定
        13: 0.00,  # 尼崎
        14: 0.00,  # 鳴門
        15: 0.00,  # 丸亀
        16:-0.10,  # 児島   イン強い
        17: 0.10,  # 宮島   荒れやすい
        18: 0.00,  # 徳山
        19: 0.05,  # 下関
        20: 0.05,  # 若松
        21: 0.00,  # 芦屋
        22: 0.00,  # 福岡
        23: 0.05,  # 唐津
        24:-0.15,  # 大村   超イン強い
    }

    # ── 各艇のベース確率計算 ──────────────────────────────────
    raw_scores = []
    for b in boats:
        s = calc_boat_score(b, boats, weather)
        raw_scores.append((b.lane, s))

    max_s = max(s for _, s in raw_scores)
    exp_scores = [(lane, math.exp(s - max_s)) for lane, s in raw_scores]
    total = sum(e for _, e in exp_scores)
    lane_probs = {lane: e / total for lane, e in exp_scores}

    # ── ③ 風向き補正（lane_probsに直接反映）──────────────────
    if weather and weather.wind_direction and weather.wind_speed:
        ws  = weather.wind_speed
        wd  = weather.wind_direction
        if wd == "向" and ws >= 3.0:
            # 向かい風: 1号艇不利
            factor = 1.0 - min(ws * 0.015, 0.15)
            lane_probs[1] = lane_probs.get(1, 1/6) * factor
        elif wd == "追" and ws >= 3.0:
            # 追い風: 1号艇有利
            factor = 1.0 + min(ws * 0.010, 0.10)
            lane_probs[1] = lane_probs.get(1, 1/6) * factor
        # 再正規化
        total_lp = sum(lane_probs.values())
        lane_probs = {l: p / total_lp for l, p in lane_probs.items()}

    boat1_prob = lane_probs.get(1, 1/6)
    boat1 = next((b for b in boats if b.lane == 1), None)
    boat2 = next((b for b in boats if b.lane == 2), None)

    other_probs = {l: p for l, p in lane_probs.items() if l != 1}
    best_other_lane = max(other_probs, key=other_probs.get) if other_probs else 2
    best_other_prob = other_probs.get(best_other_lane, 0.0)

    upset_prob = 1.0 - boat1_prob

    # ── ①1号艇の弱さ ─────────────────────────────────────────
    if boat1:
        st_risk = 0.0
        if boat1.avg_st > 0.17:    st_risk += 0.3
        if boat1.ex_st is not None and boat1.ex_st > 0.16: st_risk += 0.3
        if st_risk > 0:
            upset_prob = add_effect(upset_prob, st_risk)

        wr_risk = 0.0
        if boat1.win_rate < 5.0:   wr_risk += 0.5
        elif boat1.win_rate < 5.5: wr_risk += 0.2
        if wr_risk > 0:
            upset_prob = add_effect(upset_prob, wr_risk)

        eq_risk = 0.0
        if boat1.motor < 35.0:                         eq_risk += 0.2
        if boat1.racer_class in ("B1", "B2"):          eq_risk += 0.3
        if eq_risk > 0:
            upset_prob = add_effect(upset_prob, eq_risk)

    # ── ②2号艇の強さ ─────────────────────────────────────────
    boat2_prob = lane_probs.get(2, 0.0)
    if boat2 and boat2_prob > boat1_prob:
        upset_prob = add_effect(upset_prob, (boat2_prob - boat1_prob) * 2.0)

    # ── ③展示系特徴量 ─────────────────────────────────────────
    ex_all = [(b.lane, b.ex_time) for b in boats if b.ex_time and b.ex_time > 0]
    st_all = [(b.lane, b.ex_st)   for b in boats if b.ex_st is not None]

    if len(ex_all) >= 4:
        import statistics
        ex_std = statistics.stdev([t for _, t in ex_all])
        if ex_std > 0.05:
            upset_prob = add_effect(upset_prob, ex_std * 3.0)

    if len(st_all) >= 4:
        st_std = statistics.stdev([s for _, s in st_all])
        if st_std > 0.04:
            upset_prob = add_effect(upset_prob, st_std * 2.0)

    inner_times = [t for l, t in ex_all if l in [1,2,3]]
    outer_times = [t for l, t in ex_all if l in [4,5,6]]
    inner_avg = outer_avg = None
    if inner_times and outer_times:
        inner_avg = sum(inner_times) / len(inner_times)
        outer_avg = sum(outer_times) / len(outer_times)
        if outer_avg < inner_avg:
            upset_prob = add_effect(upset_prob, min((inner_avg - outer_avg) * 20.0, 0.5))

    # ── ⑤ 複合条件（条件が重なると急に荒れる）───────────────
    # 4カド展示1位 + 1号艇ST遅い + 風向かい → ×1.5
    boat4 = next((b for b in boats if b.lane == 4), None)
    ex_rank1 = None
    if ex_all:
        sorted_ex = sorted(ex_all, key=lambda x: x[1])
        ex_rank_map = {lane: rank+1 for rank, (lane,_) in enumerate(sorted_ex)}
        ex_rank1 = ex_rank_map.get(4)  # 4号艇の展示順位

    complex_trigger = 0
    if boat1 and boat1.ex_st and boat1.ex_st >= 0.18:        complex_trigger += 1
    if ex_rank1 == 1:                                         complex_trigger += 1
    if weather and weather.wind_direction == "向" and \
       weather.wind_speed and weather.wind_speed >= 5.0:      complex_trigger += 1
    if weather and weather.wave_height and weather.wave_height >= 5: complex_trigger += 1

    if complex_trigger >= 2:
        upset_prob = min(upset_prob * 1.5, 0.95)  # 複合条件→×1.5

    # 通常トリガー
    trigger = 0
    if boat1 and boat1.ex_st and boat1.ex_st > 0.18:        trigger += 1
    if boat2 and boat2.ex_st and boat2.ex_st < 0.13:        trigger += 1
    if inner_avg and outer_avg and outer_avg < inner_avg:   trigger += 1
    if trigger >= 2:
        upset_prob = min(upset_prob * 1.2, 0.95)

    # ── ⑤天候補正 ────────────────────────────────────────────
    if weather:
        if weather.wind_speed and weather.wind_speed >= 5.0:
            upset_prob = min(upset_prob * (1.0 + min((weather.wind_speed-5.0)*0.02, 0.08)), 0.95)
        if weather.weather and '雨' in weather.weather:
            upset_prob = min(upset_prob * 1.03, 0.95)
        if weather.wave_height and weather.wave_height >= 15:
            upset_prob = min(upset_prob * (1.0 + min((weather.wave_height-15)*0.003, 0.06)), 0.95)

    # ── ⑥等級差 ──────────────────────────────────────────────
    if boat1 and boat1.racer_class in ("B1", "B2"):
        if any(b.lane != 1 and b.racer_class == "A1" for b in boats):
            upset_prob = add_effect(upset_prob, 0.4)

    # ── ⑦レース種別補正 ──────────────────────────────────────
    grade_effects = {0: 0.0, 1: -0.1, 2: -0.2, 3: -0.3, 4: -0.5}
    upset_prob = add_effect(upset_prob, grade_effects.get(race_grade, 0.0))

    # ── ④ 会場補正を適用 ─────────────────────────────────────
    venue_factor = VENUE_UPSET_FACTOR.get(venue_num, 0.0)
    if venue_factor != 0.0:
        upset_prob = add_effect(upset_prob, venue_factor)

    # ── ⑥ ナイター補正（ナイターはイン強化傾向）─────────────
    if is_night:
        upset_prob = add_effect(upset_prob, -0.08)  # ×0.92相当

    upset_prob = max(0.0, min(upset_prob, 0.95))
    upset_score = upset_prob * 10.0

    # ── ③ 1号艇危険度AI（独立スコア）──────────────────────────
    # ST・展示・風・モーター・等級を統合した「1号艇が飛ぶ確率」専用スコア
    danger_score = 0.0
    if boat1:
        # ST危険度（平均・展示）
        if boat1.avg_st and boat1.avg_st >= 0.18:  danger_score += 1.5
        elif boat1.avg_st and boat1.avg_st >= 0.16: danger_score += 0.8
        if boat1.ex_st is not None:
            if boat1.ex_st >= 0.20:   danger_score += 2.0
            elif boat1.ex_st >= 0.17: danger_score += 1.0

        # 展示タイム危険度
        ex_list = sorted([b.ex_time for b in boats if b.ex_time and b.ex_time > 0])
        if ex_list and boat1.ex_time:
            rank1 = ex_list.index(boat1.ex_time) + 1 if boat1.ex_time in ex_list else len(ex_list)
            if rank1 >= 5:   danger_score += 2.0
            elif rank1 == 4: danger_score += 1.0
            elif rank1 == 3: danger_score += 0.5

        # モーター危険度
        motors = sorted([b.motor for b in boats if b.motor], reverse=True)
        if motors and boat1.motor:
            motor_rank = motors.index(boat1.motor) + 1
            if motor_rank >= 5:   danger_score += 1.5
            elif motor_rank == 4: danger_score += 0.8

        # 等級危険度
        if boat1.racer_class in ("B1", "B2"):
            danger_score += 1.5
            if any(b.lane != 1 and b.racer_class == "A1" for b in boats):
                danger_score += 1.0

        # 勝率危険度
        if boat1.win_rate and boat1.win_rate < 4.5: danger_score += 1.5
        elif boat1.win_rate and boat1.win_rate < 5.0: danger_score += 0.8

    # 風危険度
    if weather:
        if weather.wind_direction == "向" and weather.wind_speed:
            ws = weather.wind_speed
            if ws >= 7:   danger_score += 2.0
            elif ws >= 5: danger_score += 1.2
            elif ws >= 3: danger_score += 0.5
        if weather.wave_height and weather.wave_height >= 10:
            danger_score += 1.0

    # danger_score が高い場合は upset_score を強化
    if danger_score >= 6.0:
        upset_score = min(upset_score * 1.30, 9.5)
    elif danger_score >= 4.0:
        upset_score = min(upset_score * 1.15, 9.5)

    if boat1_prob > 0.65 and best_other_prob < boat1_prob:
        upset_score = 0.0

    # ── 1号艇等級フィルタ ─────────────────────────────────────
    # A1: 逃げ率が高いため通知しない（スコアを0にする）
    # A2: やや強いため-1.5のペナルティ
    grade_filter_note = ""
    if boat1 and boat1.racer_class == "A1":
        upset_score = 0.0
        grade_filter_note = "1号艇A1 → スキップ"
    elif boat1 and boat1.racer_class == "A2":
        upset_score = max(upset_score - 1.5, 0.0)
        grade_filter_note = "1号艇A2 → -1.5"

    target = sorted(other_probs, key=other_probs.get, reverse=True)[:2]

    sorted_lanes = sorted(lane_probs, key=lane_probs.get, reverse=True)
    boat1_rank = sorted_lanes.index(1) + 1 if 1 in sorted_lanes else 6

    grade_names = {0: '一般', 1: 'G3', 2: 'G2', 3: 'G1', 4: 'SG'}
    detail = {
        "荒れ確率":    f"{upset_prob:.1%}",
        "1号艇確率":   f"{boat1_prob:.1%}(rank{boat1_rank})",
        "1号艇危険度": f"{danger_score:.1f}",
        "最有力":      f"{best_other_lane}号艇({best_other_prob:.1%})",
        "展示":        f"{next((b.ex_time for b in boats if b.lane==1), None)}秒",
        "レース種別":  grade_names.get(race_grade, f'grade{race_grade}'),
        "複合条件":    f"{complex_trigger}個",
    }
    if grade_filter_note:
        detail["等級フィルタ"] = grade_filter_note

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


def _classify_race_type(
    ml_probs: dict[int, float],
    boats: list,
    weather,
    has_exhibition: bool,
    monster_lane: int,
    lane4_boost: float,
) -> str:
    """
    レースタイプを分類する。
    戻り値: "4カド型" / "イン逃げ型" / "1残り荒れ型" / "大荒れ型" / "モンスター型" / "混戦型"
    """
    p1 = ml_probs.get(1, 0.0)
    p4 = ml_probs.get(4, 0.0)
    sorted_p = sorted(ml_probs.values(), reverse=True)
    top_p = sorted_p[0] if sorted_p else 0

    # モンスター艇（展示で圧倒的）
    if monster_lane > 0 and monster_lane != 1:
        return "モンスター型"

    # 4カド（4号艇が1号艇に迫る）
    if lane4_boost > 0:
        return "4カド型"

    # イン逃げ（1号艇が圧倒的に強い）
    if p1 > 0.55:
        return "イン逃げ型"

    # 1残り荒れ（1号艇がそこそこ強いが荒れ要素あり）
    if p1 > 0.35:
        return "1残り荒れ型"

    # 大荒れ（全艇拮抗）
    if top_p < 0.28:
        return "大荒れ型"

    return "混戦型"


def _build_why_bet(
    combo: str,
    ml_probs: dict[int, float],
    boats: list,
    weather,
    has_exhibition: bool,
    monster_lane: int,
    lane4_boost: float,
    odds_dropped: list[str],
    upset_score: float,
) -> list[str]:
    """
    買い理由を言語化して返す。
    例: ["4カド", "向かい風5m", "1号艇ST遅れ0.22", "展示1位"]
    """
    reasons = []
    parts = combo.split("-")
    first = int(parts[0]) if parts else 0

    # 【朝刊AI】気象（風速・風向・波高）は previews 由来のため買い理由に使用しない

    # 1号艇弱さ（詳細理由、朝取得可能データのみ）
    if boats:
        boat1 = next((b for b in boats if b.lane == 1), None)
        if boat1:
            # 平均ST実績（展示STの代わりに朝データのavg_stを使用）
            if boat1.avg_st and boat1.avg_st >= 0.18:
                reasons.append(f"1号艇平均ST遅め{boat1.avg_st:.2f}")
            elif boat1.avg_st and boat1.avg_st >= 0.16:
                reasons.append(f"1号艇平均STやや遅め{boat1.avg_st:.2f}")
            if boat1.win_rate and boat1.win_rate < 5.0:
                reasons.append(f"1号艇勝率低({boat1.win_rate:.1f})")
            if boat1.racer_class in ("B1","B2"):
                reasons.append(f"1号艇{boat1.racer_class}級")
            # モーター
            motors = sorted([b.motor for b in boats if b.motor], reverse=True)
            if motors and boat1.motor:
                m_rank = motors.index(boat1.motor) + 1
                if m_rank >= 5:
                    reasons.append(f"1号艇モーター{m_rank}位")

    # 特殊パターン
    if monster_lane > 0 and first == monster_lane:
        reasons.append(f"モンスター{monster_lane}号艇")
    if lane4_boost > 0 and first == 4:
        reasons.append("4カド優勢")

    # 【朝刊AI】展示順位に基づく買い理由は previews 由来のため生成しない

    # オッズ急落
    if odds_dropped:
        reasons.append(f"急落{odds_dropped[0]}")

    # 荒れスコア
    if upset_score >= 7.0:
        reasons.append(f"超荒れ({upset_score:.1f})")
    elif upset_score >= 5.0:
        reasons.append(f"荒れ({upset_score:.1f})")

    return reasons if reasons else ["EV高値"]


def _evaluate_bets(
    patterns: dict[str, list],
    ml_probs: dict[int, float],
    odds_map: dict[str, float],
    bankroll: int = 10000,
    target_lanes: list[int] | None = None,
    has_exhibition: bool = True,
    boats: list | None = None,
    upset_score: float = 0.0,
    odds_dropped: list[str] | None = None,
    weather=None,
    venue_num: int = 0,
    race_number: int = 0,
    race_date: str = "",
    venue_name: str = "",
) -> list[dict]:
    """
    全三連単120通りを複合スコア順でランキングして推奨買い目を返す。

    スコア = EV×0.45 + prob×0.35 + upset_score正規化×0.20
    見送り条件:
      - 上位EVと2位EVの差が0.08未満（中途半端）
      - 軸候補なし（最強艇が2位の0.9倍未満）
    """
    # ── ベースパラメータ ──────────────────────────────────────
    EV_LOW   = 1.28
    EV_MID   = 1.38
    MIN_PROB = 0.020
    MAX_ODDS = 80
    MAX_BETS = 5
    MAX_FULL_FADE = 2   # 1号艇完全飛び（1を含まない買い目）の上限点数

    if not has_exhibition:
        EV_LOW   += 0.20
        EV_MID   += 0.20
        MIN_PROB += 0.008

    # ── 展示タイム順位ボーナス ────────────────────────────────
    ex_rank_bonus = {1: 1.20, 2: 1.10, 3: 1.00, 4: 0.92, 5: 0.85, 6: 0.78}
    ex_rank: dict[int, int] = {}
    if boats and has_exhibition:
        ex_data = [(b.lane, b.ex_time) for b in boats
                   if b.ex_time and b.ex_time > 0]
        ex_data.sort(key=lambda x: x[1])
        ex_rank = {lane: rank + 1 for rank, (lane, _) in enumerate(ex_data)}

    # ── モンスター艇検出（1艇だけ異常に強い）────────────────
    # 展示タイムが他より0.05秒以上速い = モンスター
    monster_lane = 0
    monster_boost = 1.0
    if boats and has_exhibition and ex_rank:
        ex_times = {b.lane: b.ex_time for b in boats
                    if b.ex_time and b.ex_time > 0}
        if len(ex_times) >= 2:
            sorted_ex = sorted(ex_times.items(), key=lambda x: x[1])
            best_lane, best_time = sorted_ex[0]
            second_time = sorted_ex[1][1]
            gap = second_time - best_time
            if gap >= 0.05:   # 0.05秒以上の差 = モンスター
                monster_lane  = best_lane
                monster_boost = 1.0 + gap * 8.0   # 0.05秒差 → ×1.4
                log.debug("モンスター艇検出: %d号艇 gap=%.3f boost=%.2f",
                          monster_lane, gap, monster_boost)

    # ── 4カド攻め補正 ─────────────────────────────────────────
    lane4_boost = 0.0
    if ml_probs:
        p1 = ml_probs.get(1, 0.0)
        p4 = ml_probs.get(4, 0.0)
        if p4 > p1 * 0.9:
            lane4_boost = 1.2

    # ── オッズ歪み検出（モデル順位 vs オッズ順位）────────────
    # モデルが高評価なのに市場オッズが高い = 過小評価 = バリュー
    model_rank: dict[int, int] = {}
    if ml_probs:
        sorted_ml = sorted(ml_probs.items(), key=lambda x: -x[1])
        model_rank = {lane: rank+1 for rank, (lane,_) in enumerate(sorted_ml)}

    # 1着オッズの市場ランク（オッズが低い=人気=市場ランク1）
    lane_avg_odds: dict[int, float] = {}
    for lane in range(1, 7):
        lane_odds = [v for k, v in odds_map.items()
                     if k.startswith(f"{lane}-") and v > 0]
        if lane_odds:
            lane_avg_odds[lane] = sum(lane_odds) / len(lane_odds)
    sorted_market = sorted(lane_avg_odds.items(), key=lambda x: x[1])
    market_rank = {lane: rank+1 for rank, (lane,_) in enumerate(sorted_market)}

    # ── 【朝刊AI】着順別確率（1着・2着・3着を個別評価）────────
    # _mc_trifecta_probs と trifecta_prob の両方から参照する共通データ。
    # 危険艇判定（1号艇の1着適性低下）は1着評価にのみ強く反映され、
    # 2着・3着評価への影響は最小限に抑えられている
    # （x_asahi_scoring.calc_rank_probabilities_v2 を参照）。
    if boats:
        _rank_probs = calc_rank_probabilities_v2(
            boats, context={"match_index": 0, "upset_score": upset_score}
        )
    else:
        _rank_probs = {"first": ml_probs, "second": ml_probs, "third": ml_probs,
                        "lane6_first_allowed": True}

    # ── 三連単確率計算 ────────────────────────────────────────
    # ════════════════════════════════════════════════════════
    # 展開シミュレーションエンジン
    # ════════════════════════════════════════════════════════

    def _detect_attack_lane(probs: dict[int, float]) -> int:
        """
        攻め艇を検出する。
        2〜4号艇の中でPL確率が最も高く、かつ1号艇より高い艇を攻め艇とする。
        攻め艇なしの場合は0を返す。
        """
        candidates = {l: p for l, p in probs.items()
                      if l in [2, 3, 4] and p > probs.get(1, 0)}
        if not candidates:
            return 0
        return max(candidates, key=candidates.get)

    def _calc_lane1_survive_prob(probs: dict[int, float], attack_lane: int) -> float:
        """
        1号艇の「逃げ切り確率」を計算する。
        攻め艇が強いほど下がり、1号艇STが遅いほど下がる。
        """
        p1     = probs.get(1, 1/6)
        p_att  = probs.get(attack_lane, 0) if attack_lane > 0 else 0
        # 基本逃げ確率は PL 確率から
        survive = p1

        # 攻め艇補正（攻め艇が強いほど逃げ確率が落ちる）
        if p_att > 0:
            attack_pressure = p_att / (p1 + p_att + 1e-6)
            survive *= (1.0 - attack_pressure * 0.4)

        # 1号艇ST補正（【朝刊AI】平均ST実績を使用、展示STは使わない）
        if boats:
            boat1 = next((b for b in boats if b.lane == 1), None)
            if boat1 and boat1.avg_st:
                if boat1.avg_st >= 0.18:   survive *= 0.75
                elif boat1.avg_st >= 0.16: survive *= 0.88
            # コース別ST実績（1コース、fanファイル）による補正
            if boat1 and boat1.course_nyuko and boat1.course_nyuko[0] > 0:
                course_st_1c = boat1.course_st[0]
                course_rank_1c = boat1.course_rank[0]
                if course_rank_1c and course_rank_1c >= 4: survive *= 0.80
                elif course_rank_1c and course_rank_1c >= 3: survive *= 0.92

        return min(max(survive, 0.01), 0.99)

    def _mc_trifecta_probs(
        probs: dict[int, float],
        n_sim: int = 5000,
    ) -> dict[str, float]:
        """
        ② Monte Carlo 三連単確率

        【朝刊AI・着順別評価】1着・2着・3着それぞれ独立した確率分布
        （calc_rank_probabilities_v2）を用いる。従来は1着確率(probs)を
        2着・3着にもそのまま流用していたが、これにより
          ・危険艇判定で1号艇の1着適性が下がっても、2着・3着では
            自然に舟券圏内へ残る
          ・6号艇は艇番の実決着率に基づき1着候補になりにくく、
            AI一致指数・荒れ度が高い場合のみ1着候補として残る
        という設計を実現する。

        n_sim 回レース仮想再生して三連単確率を推定する。
        1着→2着は展開補正（attack_lane / survive_prob）を組み込み、
        3着は独立した third_probs を用いる（1着確率の使い回しをしない）。
        """
        import random as _rand
        _rand.seed(None)   # ランダムシード

        if not boats:
            # boats がない場合は従来通り単一分布にフォールバック
            rank_probs = {"first": probs, "second": probs, "third": probs}
        else:
            # 【朝刊AI】_evaluate_bets 冒頭で計算済みの共通データを再利用する
            rank_probs = _rank_probs

        first_probs  = rank_probs["first"]
        second_probs_base = rank_probs["second"]
        third_probs  = rank_probs["third"]

        attack_lane  = _detect_attack_lane(probs)
        survive_prob = _calc_lane1_survive_prob(probs, attack_lane)

        lanes  = list(first_probs.keys())
        result_count: dict[str, int] = {}

        for _ in range(n_sim):
            # 1着（着順別1着確率分布を使用。6号艇は特別条件を満たさない限り抑制済み）
            first = _rand.choices(lanes, weights=[first_probs[l] for l in lanes])[0]

            # 2着（着順別2着確率分布をベースに、展開補正を適用）
            sec_probs = _expansion_second_probs(second_probs_base, first, attack_lane, survive_prob)
            sec_lanes = [l for l in lanes if l != first]
            sec_w     = [sec_probs.get(l, 0.001) for l in sec_lanes]

            if sum(sec_w) <= 0:
                continue
            second = _rand.choices(sec_lanes, weights=sec_w)[0]

            # 3着（着順別3着確率分布を使用。1着確率の使い回しをしない）
            thi_lanes = [l for l in lanes if l != first and l != second]
            thi_w     = [third_probs.get(l, 0.001) for l in thi_lanes]
            if not thi_lanes or sum(thi_w) <= 0:
                continue
            third = _rand.choices(thi_lanes, weights=thi_w)[0]

            key = f"{first}-{second}-{third}"
            result_count[key] = result_count.get(key, 0) + 1

        # 確率に変換
        total = sum(result_count.values()) or 1
        return {k: v / total for k, v in result_count.items()}

    # ── MCシミュレーション（展開補正あり） ────────────────────
    # 【朝刊AI】has_exhibition は展示データ不使用ポリシーにより常に False
    # だが、MCシミュレーション自体は展示の有無に関係なく実行すべきなので
    # ここでは has_exhibition によるガードを行わない（旧実装のバグ修正）。
    mc_probs: dict[str, float] = {}
    if boats:
        mc_probs = _mc_trifecta_probs(ml_probs, n_sim=3000)

    # ════════════════════════════════════════════════════════
    # ① 確率キャリブレーション補正
    # ════════════════════════════════════════════════════════

    def _load_calibration_table(csv_file: str = "hit_record.csv") -> list[tuple]:
        """
        hit_record.csv から確率帯別の実測的中率テーブルを作成する。
        戻り値: [(pred_mid, actual_rate), ...]  isotonic回帰の近似
        """
        import csv as _csv
        if not os.path.exists(csv_file):
            return []
        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                rows = list(_csv.DictReader(f))
            valid = [r for r in rows
                     if r.get("pred_prob") and r.get("hit") not in ("",None,"-1")
                     and r.get("pred_combo")]  # 通知済みレースのみ
            if len(valid) < 20:
                return []
            bands = [(0,0.02),(0.02,0.03),(0.03,0.05),(0.05,0.08),(0.08,1.0)]
            table = []
            for lo, hi in bands:
                seg = [r for r in valid if lo <= float(r["pred_prob"]) < hi]
                if len(seg) < 3:
                    continue
                pred_mid = sum(float(r["pred_prob"]) for r in seg) / len(seg)
                actual   = sum(int(r["hit"] or 0)   for r in seg) / len(seg)
                table.append((pred_mid, actual))
            return table
        except Exception:
            return []

    def _calibrate_prob(raw_prob: float, calib_table: list[tuple]) -> float:
        """
        キャリブレーションテーブルで確率を補正する（線形補間）。
        テーブルがない場合はそのまま返す。
        """
        if not calib_table or len(calib_table) < 2:
            return raw_prob
        # 最近傍補間
        if raw_prob <= calib_table[0][0]:
            ratio = calib_table[0][1] / max(calib_table[0][0], 1e-6)
            return raw_prob * ratio
        if raw_prob >= calib_table[-1][0]:
            ratio = calib_table[-1][1] / max(calib_table[-1][0], 1e-6)
            return min(raw_prob * ratio, 0.99)
        for i in range(len(calib_table)-1):
            x0, y0 = calib_table[i]
            x1, y1 = calib_table[i+1]
            if x0 <= raw_prob <= x1:
                t = (raw_prob - x0) / max(x1 - x0, 1e-9)
                return y0 + t * (y1 - y0)
        return raw_prob

    # キャリブレーションテーブルをロード
    calib_table = _load_calibration_table()
    if calib_table:
        log.debug("キャリブレーション補正: %d帯", len(calib_table))

    # ════════════════════════════════════════════════════════
    # ② 市場融合モデル（オッズ → 市場確率 → モデルとブレンド）
    # ════════════════════════════════════════════════════════

    def _market_implied_probs(
        odds_map: dict[str, float],
        takeout_rate: float = 0.25,  # 競艇の控除率は約25%
    ) -> dict[str, float]:
        """
        三連単オッズから市場の暗示確率を計算する。
        implied_prob = 1 / odds * (1 - takeout_rate) で近似。
        """
        if not odds_map:
            return {}
        implied = {}
        for combo, odds in odds_map.items():
            if odds > 0:
                implied[combo] = (1.0 / odds) * (1.0 - takeout_rate)
        # 合計が1になるよう正規化
        total = sum(implied.values()) or 1.0
        return {k: v / total for k, v in implied.items()}

    market_implied = _market_implied_probs(odds_map)

    # ════════════════════════════════════════════════════════
    # ③ シナリオ別Monte Carlo
    # ════════════════════════════════════════════════════════

    def _scenario_mc(
        probs: dict[int, float],
        scenario: str,   # "4攻め" / "1逃げ" / "差し"
        n_sim: int = 1500,
    ) -> dict[str, float]:
        """シナリオに応じた展開重みで三連単確率を推定する。"""
        import random as _rand

        # シナリオに応じた1着確率の歪み
        adj = dict(probs)
        if scenario == "4攻め":
            adj[4] = adj.get(4, 0) * 2.0
            adj[1] = adj.get(1, 0) * 0.6
        elif scenario == "1逃げ":
            adj[1] = adj.get(1, 0) * 1.8
            for l in [2,3,4]: adj[l] = adj.get(l,0) * 0.7
        elif scenario == "差し":
            adj[2] = adj.get(2, 0) * 1.5
            adj[3] = adj.get(3, 0) * 1.3
            adj[1] = adj.get(1, 0) * 0.7

        total = sum(adj.values()) or 1.0
        adj = {l: p/total for l, p in adj.items()}
        lanes = list(adj.keys())

        result_count: dict[str, int] = {}
        for _ in range(n_sim):
            first = _rand.choices(lanes, weights=[adj[l] for l in lanes])[0]
            rem   = {l: probs[l] for l in lanes if l != first}
            t1    = sum(rem.values()) or 1
            rem2  = {l: p/t1 for l, p in rem.items()}
            sec_l = list(rem2.keys())
            second = _rand.choices(sec_l, weights=[rem2[l] for l in sec_l])[0]
            thi_l = [l for l in lanes if l != first and l != second]
            if not thi_l: continue
            thi_w = [probs.get(l, 0.001) for l in thi_l]
            third = _rand.choices(thi_l, weights=thi_w)[0]
            k = f"{first}-{second}-{third}"
            result_count[k] = result_count.get(k, 0) + 1

        total_sim = sum(result_count.values()) or 1
        return {k: v/total_sim for k, v in result_count.items()}

    # シナリオ確率を計算してブレンド
    scenario_probs: dict[str, float] = {}
    if has_exhibition and boats and ml_probs:
        import random as _rnd; _rnd.seed(42)
        sc_4  = _scenario_mc(ml_probs, "4攻め",  1000)
        sc_1  = _scenario_mc(ml_probs, "1逃げ",  1000)
        sc_sa = _scenario_mc(ml_probs, "差し",   1000)
        # 荒れスコアに応じてシナリオ重みを変える
        # 荒れ強い → 4攻め・差し優先 / 荒れ弱い → 1逃げ優先
        w_4  = min(upset_score / 10.0, 0.5)
        w_sa = min(upset_score / 15.0, 0.3)
        w_1  = max(1.0 - w_4 - w_sa, 0.2)
        total_w = w_4 + w_1 + w_sa
        w_4 /= total_w; w_1 /= total_w; w_sa /= total_w
        for combo in set(list(sc_4.keys()) + list(sc_1.keys()) + list(sc_sa.keys())):
            scenario_probs[combo] = (
                sc_4.get(combo,0) * w_4 +
                sc_1.get(combo,0) * w_1 +
                sc_sa.get(combo,0) * w_sa
            )

    # ════════════════════════════════════════════════════════
    # ⑤ 隊列崩れ率（formation_break_prob）
    # ════════════════════════════════════════════════════════

    def _calc_formation_break_prob(
        boats: list,
        probs: dict[int, float],
        weather,
    ) -> float:
        """
        「123/456」の標準隊形が崩れる確率を推定する。
        1・2・3号艇以外が1着になる確率を基本として、
        気象・展示隊形を加味して補正する。
        """
        outer_prob = sum(probs.get(l, 0) for l in [4, 5, 6])

        # 展示タイムで外側が速い場合は崩れやすい
        ex_inner = [b.ex_time for b in boats if b.lane in [1,2,3] and b.ex_time]
        ex_outer = [b.ex_time for b in boats if b.lane in [4,5,6] and b.ex_time]
        if ex_inner and ex_outer:
            avg_in  = sum(ex_inner) / len(ex_inner)
            avg_out = sum(ex_outer) / len(ex_outer)
            if avg_out < avg_in:
                outer_prob = min(outer_prob * 1.3, 0.95)

        # 気象補正
        if weather:
            ws = weather.wind_speed or 0
            if weather.wind_direction == "向" and ws >= 5:
                outer_prob = min(outer_prob * 1.2, 0.95)
            if weather.wave_height and weather.wave_height >= 5:
                outer_prob = min(outer_prob * 1.1, 0.95)

        return round(outer_prob, 3)

    formation_break = _calc_formation_break_prob(boats or [], ml_probs, weather)
    log.debug("隊列崩れ率: %.0f%%", formation_break * 100)

    # ════════════════════════════════════════════════════════
    # 最終確率計算: PL × MC × シナリオ × 市場 のブレンド
    # ════════════════════════════════════════════════════════

    # ════════════════════════════════════════════════════════
    # ① レジーム判定（今日の環境を4種類に分類）
    # ════════════════════════════════════════════════════════

    def _detect_regime(
        weather,
        upset_score: float,
        is_night: bool,
    ) -> str:
        """
        レースの「環境レジーム」を判定する。
        戻り値:
          "extreme_wind"  … 風速7m以上 or 波高10cm以上
          "rough"         … 荒れスコア6以上 or 風5m以上
          "night_stable"  … ナイター + 荒れスコア低い
          "calm"          … それ以外（標準）
        """
        ws = weather.wind_speed  if weather else 0.0
        wh = weather.wave_height if weather else 0
        ws = ws or 0.0
        wh = wh or 0

        if ws >= 7.0 or wh >= 10:
            return "extreme_wind"
        if upset_score >= 6.0 or ws >= 3.0:
            return "rough"
        if is_night and upset_score < 4.0:
            return "night_stable"
        return "calm"

    NIGHT_VENUES = {4, 6, 12, 17, 20, 21, 22, 23, 24}
    is_night_local = (boats[0].lane if boats else 0) and False  # fallback
    # upset_score はこのスコープでは _evaluate_bets 引数から取得
    regime = _detect_regime(weather, upset_score, False)

    # ════════════════════════════════════════════════════════
    # ② 動的ウェイト（レジームに応じてブレンド比率を調整）
    # ════════════════════════════════════════════════════════

    REGIME_WEIGHTS = {
        # (PL,  MC,   SC,   市場)
        "extreme_wind":  (0.20, 0.35, 0.30, 0.15),
        "rough":         (0.25, 0.30, 0.30, 0.15),
        "night_stable":  (0.40, 0.15, 0.15, 0.30),
        "calm":          (0.30, 0.25, 0.25, 0.20),
    }
    w_pl, w_mc, w_sc, w_mkt = REGIME_WEIGHTS.get(regime, (0.30, 0.25, 0.25, 0.20))

    # 展示なし → PLと市場のみ
    if not mc_probs or not scenario_probs:
        w_pl, w_mc, w_sc, w_mkt = 0.55, 0.0, 0.0, 0.45
    # 市場データなし → PLとMCとシナリオのみ
    elif not market_implied:
        w_pl, w_mc, w_sc, w_mkt = 0.40, 0.30, 0.30, 0.0

    log.debug("レジーム: %s → PL=%.2f MC=%.2f SC=%.2f 市場=%.2f",
              regime, w_pl, w_mc, w_sc, w_mkt)

    # ════════════════════════════════════════════════════════
    # ③ モデル間 disagreement（PL vs シナリオの意見の差）
    # ════════════════════════════════════════════════════════

    def _calc_disagreement(
        pl_map: dict[str, float],
        sc_map: dict[str, float],
        top_n: int = 10,
    ) -> float:
        """
        PLモデルとシナリオMCの上位TOP_N候補の重複率から
        disagreementスコアを計算する（0.0〜1.0）。
        1.0 = 完全不一致（危険） / 0.0 = 完全一致（安全）
        """
        if not pl_map or not sc_map:
            return 0.0
        pl_top  = set(k for k,_ in sorted(pl_map.items(),  key=lambda x:-x[1])[:top_n])
        sc_top  = set(k for k,_ in sorted(sc_map.items(),  key=lambda x:-x[1])[:top_n])
        overlap = len(pl_top & sc_top) / top_n
        return round(1.0 - overlap, 3)

    # PL確率マップ（全120通り）
    def _pl_map_all(probs: dict[int, float]) -> dict[str, float]:
        from itertools import permutations as _p
        result = {}
        for a, b, c in _p(probs.keys(), 3):
            pa = probs[a]
            rem1 = {l: p for l, p in probs.items() if l != a}
            t1 = sum(rem1.values()) or 1
            pb = rem1.get(b, 0.001) / t1
            rem2 = {l: p for l, p in rem1.items() if l != b}
            t2 = sum(rem2.values()) or 1
            pc = rem2.get(c, 0.001) / t2
            result[f"{a}-{b}-{c}"] = pa * pb * pc
        return result

    pl_all = _pl_map_all(ml_probs)
    disagreement = _calc_disagreement(pl_all, scenario_probs)

    # disagreementが高い = モデル間で意見が割れている = 不確実性高い
    if disagreement >= 0.70:
        log.debug("モデル間disagreement高: %.2f → bet縮小対象", disagreement)

    # ════════════════════════════════════════════════════════
    # uncertainty（エントロピーで確率分布の不確実性を測定）
    # ════════════════════════════════════════════════════════

    def _calc_entropy(probs: dict[str, float], top_n: int = 20) -> float:
        """
        三連単確率分布のエントロピーを計算する。
        高エントロピー = 「どれが来るか分からない」 = 買いにくい
        """
        import math as _math
        top = sorted(probs.values(), reverse=True)[:top_n]
        total = sum(top) or 1.0
        norm  = [p / total for p in top if p > 0]
        return -sum(p * _math.log(p + 1e-9) for p in norm)

    entropy = _calc_entropy(pl_all) if pl_all else 3.0
    # entropy の上限は log(120) ≈ 4.79（完全一様分布）
    # 正規化: 0.0（一点集中）〜 1.0（完全ランダム）
    import math as _math
    entropy_norm = min(entropy / _math.log(120), 1.0)

    log.debug("uncertainty: entropy=%.3f (normalized=%.3f), disagreement=%.3f",
              entropy, entropy_norm, disagreement)

    # ════════════════════════════════════════════════════════
    # レース品質スコア（disagreement + uncertainty を加味）
    # ════════════════════════════════════════════════════════
    # quality が低い or disagreement が高い → bet_score を下げる
    quality_penalty = (
        entropy_norm    * 0.20 +   # 高エントロピー = 不確か
        disagreement    * 0.15     # 高disagreement = 意見割れ
    )
    log.debug("品質ペナルティ: %.3f (regime=%s)", quality_penalty, regime)

    def trifecta_prob(a: int, b: int, c: int) -> float:
        """
        PL・MC・シナリオ・市場確率をレジーム対応の動的ウェイトでブレンドし
        キャリブレーション補正を適用した最終三連単確率

        【朝刊AI・着順別評価】PL(Plackett-Luce)確率も1着・2着・3着の
        独立した確率分布(_rank_probs)を用いる。従来は1着確率(ml_probs)を
        2着・3着の計算にも使い回していたため、危険艇判定や艇番バイアスが
        本来1着にしか影響すべきでないのに2着・3着まで連動して歪んでいた。
        """
        # ── PL確率（Luce モデル、着順別の独立分布を使用）────────
        first_p, second_p, third_p = _rank_probs["first"], _rank_probs["second"], _rank_probs["third"]
        pa = first_p.get(a, 0.001)
        rem1 = {l: p for l, p in second_p.items() if l != a}
        t1_  = sum(rem1.values()) or 1
        pb   = rem1.get(b, 0.001) / t1_
        rem2 = {l: p for l, p in third_p.items() if l != a and l != b}
        t2_  = sum(rem2.values()) or 1
        pc   = rem2.get(c, 0.001) / t2_
        pl_prob = pa * pb * pc

        combo = f"{a}-{b}-{c}"

        mc_prob  = mc_probs.get(combo, pl_prob)
        sc_prob  = scenario_probs.get(combo, pl_prob)
        mkt_prob = market_implied.get(combo, pl_prob)

        # ── ②動的ウェイトでブレンド ──────────────────────────
        prob = (pl_prob  * w_pl +
                mc_prob  * w_mc +
                sc_prob  * w_sc +
                mkt_prob * w_mkt)

        # ── ①キャリブレーション補正 ──────────────────────────
        prob = _calibrate_prob(prob, calib_table)

        # ── 個別補正 ──────────────────────────────────────────
        # 【朝刊AI】6号艇1着の抑制は _rank_probs["first"] 側
        # （calc_rank_probabilities_v2 の特別条件フィルタ）で既に
        # 反映済みのため、ここでの重複ペナルティ(旧: prob *= 0.70)は行わない。
        # 艇番バイアス(course_bonus)も同様に着順別分布へ既に織り込み済みのため、
        # 2着・3着へ一律の艇番バイアスをかける処理（旧実装）は廃止する。
        if a != 1 and b == 1:
            prob *= 1.15
        if a == 1 and odds_map.get(combo, 0) >= 15:
            prob *= 1.12
        if a == 4 and lane4_boost > 0:
            prob *= (1.0 + lane4_boost * 0.05)
        if a == monster_lane and monster_boost > 1.0:
            prob *= monster_boost
        if a in ex_rank:
            prob *= ex_rank_bonus.get(ex_rank[a], 1.0)
        if a in model_rank and a in market_rank:
            value_gap = market_rank[a] - model_rank[a]
            if value_gap >= 2:
                prob *= (1.0 + value_gap * 0.04)

        return max(prob, 1e-6)

    # ── 軸候補チェック（実力差なしレースは見送り）────────────
    if ml_probs:
        sorted_probs = sorted(ml_probs.values(), reverse=True)
        if len(sorted_probs) >= 2:
            top_p, second_p = sorted_probs[0], sorted_probs[1]
            if top_p < second_p * 1.03:   # 旧: 1.08（厳しすぎた）
                # 軸なし = 運ゲー → スキップ
                log.debug("軸候補なし（実力差不足）→ 見送り: top=%.3f 2nd=%.3f",
                          top_p, second_p)
                return []

    # パターンラベル
    tgt = target_lanes or []
    t1  = tgt[0] if tgt else None

    def _label(a: int, b: int, c: int) -> str:
        if a == monster_lane and monster_boost > 1.0: return "モンスター"
        if a == 1:                                    return "本命軸"
        if a == 4 and lane4_boost > 0:               return "4カド"
        if a in [5, 6]:                               return "万舟"
        if t1 and a == t1 and b == 1:                return "差し"
        if t1 and a == t1 and b != 1:                return "まくり"
        return "本命崩れ"

    # ── 全通り評価 ────────────────────────────────────────────
    from itertools import permutations
    candidates = []
    for a, b, c in permutations(range(1, 7), 3):
        combo = f"{a}-{b}-{c}"
        odds  = odds_map.get(combo, 0)
        if odds <= 0 or odds > MAX_ODDS:
            continue

        prob = trifecta_prob(a, b, c)
        if prob < MIN_PROB:
            continue

        ev_threshold = EV_MID if odds >= 40 else EV_LOW
        ev = prob * odds

        # ─────────────────────────────────────────────────────────
        # 1号艇2・3着残り補正（実戦向け）
        # ─────────────────────────────────────────────────────────
        parts = combo.split("-")
        first, second, third = map(int, parts)
        boat1 = next((b for b in boats if b.lane == 1), None) if boats else None
        if boat1 and first != 1 and 1 in (second, third):
            remain_bonus = 1.0
            if boat1.win_rate and boat1.win_rate >= 5.0:
                remain_bonus += 0.10
            if boat1.racer_class == "A1":
                remain_bonus += 0.20
            elif boat1.racer_class == "A2":
                remain_bonus += 0.10
            if boat1.ex_st is not None and boat1.ex_st <= 0.18:
                remain_bonus += 0.10
            if has_exhibition and boat1.ex_time:
                ex_sorted = sorted(
                    [b.ex_time for b in boats if b.ex_time and b.ex_time > 0]
                )
                if boat1.ex_time in ex_sorted:
                    rank1 = ex_sorted.index(boat1.ex_time) + 1
                    if rank1 <= 3:
                        remain_bonus += 0.10
            ev   *= remain_bonus
            prob *= remain_bonus

        # 1号艇絡み（2・3着）はEV閾値を緩和
        effective_threshold = ev_threshold
        parts_check = combo.split("-")
        if parts_check[0] != "1" and "1" in parts_check[1:]:
            effective_threshold *= 0.85
        if ev < effective_threshold:
            continue

        # ── 複合スコア（EVを単独使用しない）────────────────
        ev_norm    = min(ev / 3.0, 1.0)          # 3.0を上限として正規化
        prob_norm  = min(prob / 0.10, 1.0)        # 10%を上限として正規化
        score_norm = min(upset_score / 10.0, 1.0) # 荒れスコア正規化
        composite  = ev_norm * 0.45 + prob_norm * 0.35 + score_norm * 0.20

        candidates.append({
            "combo":     combo,
            "pattern":   _label(a, b, c),
            "prob":      round(prob, 5),
            "odds":      odds,
            "ev":        round(ev, 3),
            "composite": round(composite, 4),
        })

    if not candidates:
        return []

    # ── 見送り条件①: 最高probが低すぎる ──────────────────────
    max_prob_val = max(c["prob"] for c in candidates)
    if max_prob_val < 0.025:
        log.debug("最高prob不足 → 見送り: max_prob=%.4f", max_prob_val)
        return []

    # ── 見送り条件②: disagreement高すぎる（モデル意見割れ）──
    if disagreement >= 0.90:   # 旧: 0.80
        log.debug("disagreement高 → 見送り: %.2f", disagreement)
        return []

    # ── 見送り条件③: entropy高すぎる（完全ランダム）──────────
    if entropy_norm >= 0.92:   # 旧: 0.85
        log.debug("uncertainty高 → 見送り: entropy_norm=%.2f", entropy_norm)
        return []

    # ── 見送り条件④（中途半端レースを消す）──────────────────
    candidates.sort(key=lambda x: x["composite"], reverse=True)
    if len(candidates) >= 2:
        top_ev    = candidates[0]["ev"]
        second_ev = candidates[1]["ev"]
        if top_ev - second_ev < 0.03:   # 旧: 0.08
            log.debug("EV差不足（中途半端）→ 見送り: top=%.3f 2nd=%.3f",
                      top_ev, second_ev)
            return []

    # ── 同型制限（同じ1着艇は最大2点）→ 上位MAX_BETS ────────
    head_count: dict[int, int] = {}
    top = []
    for c in candidates:
        head = int(c["combo"].split("-")[0])
        if head_count.get(head, 0) >= 2:
            continue
        head_count[head] = head_count.get(head, 0) + 1
        top.append(c)
        if len(top) >= MAX_BETS:
            break

    # ── 1号艇完全飛び上限制御（MAX_FULL_FADE）─────────────────
    fade_count = 0
    filtered_top = []
    for c in top:
        parts = c["combo"].split("-")
        has_boat1 = "1" in parts
        if not has_boat1:
            if fade_count >= MAX_FULL_FADE:
                continue
            fade_count += 1
        filtered_top.append(c)
    top = filtered_top

    # ── ベット制御（信頼度連動3段階）+ why_bet / race_type / confidence 付与 ──
    race_type = _classify_race_type(
        ml_probs, boats or [], weather, has_exhibition, monster_lane, lane4_boost
    )

    # ドローダウン状態を読んでベット倍率を決める
    dd_multiplier = _get_bet_multiplier()

    for t in top:
        # ── bet_score（quality_penaltyを減算）──────────────────
        ev_n      = min(t["ev"] / 3.0, 1.0)
        prob_n    = min(t["prob"] / 0.10, 1.0)
        conf_n    = t.get("composite", 0)

        # 市場逆張りスコア
        first_lane = int(t["combo"].split("-")[0])
        mgap = 0.0
        if first_lane in model_rank and first_lane in market_rank:
            raw_gap = market_rank[first_lane] - model_rank[first_lane]
            mgap = min(max(raw_gap / 5.0, 0.0), 1.0)

        bet_score = (ev_n * 0.35 + prob_n * 0.35 + conf_n * 0.15 + mgap * 0.15
                     - quality_penalty)   # ③ uncertainty/disagreementで減点
        bet_score = max(bet_score, 0.0)

        # ── 3段階ベット金額 ─────────────────────────────────────
        if bet_score >= 0.9:
            base_bet = 3000
        elif bet_score >= 0.75:
            base_bet = 1500
        else:
            base_bet = 500

        # ③ uncertainty連動ベットサイジング: bet *= (1 - uncertainty * 0.5)
        # uncertainty=0.0 → 等倍 / uncertainty=0.8 → ×0.6
        uncertainty_factor = max(1.0 - entropy_norm * 0.5, 0.40)

        # disagreement連動: 意見割れが多いほど縮小
        disagreement_factor = max(1.0 - disagreement * 0.30, 0.50)

        # ドローダウン時縮小
        final_amount = int(
            base_bet * dd_multiplier * uncertainty_factor * disagreement_factor
            / 100
        ) * 100
        t["amount"]   = final_amount
        t["bet_score"] = round(bet_score, 3)

        # why_bet / race_type / confidence
        t["why_bet"] = _build_why_bet(
            t["combo"], ml_probs, boats or [], weather,
            has_exhibition, monster_lane, lane4_boost,
            odds_dropped or [], upset_score,
        )
        t["race_type"]    = race_type
        t["confidence"]   = t.get("composite", 0)
        t["regime"]       = regime
        t["disagreement"] = disagreement
        t["uncertainty"]  = round(entropy_norm, 3)
        t["formation_break"] = formation_break

    # ════════════════════════════════════════════════════════
    # 【購入判定分離】BuyScoreエンジンによる最終購入判定
    # ════════════════════════════════════════════════════════
    # ここまでの top は「予想」（評価済み候補）に過ぎない。
    # 実際に「購入」するかどうかは、BuyScore・AI一致指数・EV・
    # 危険度・市場乖離・信頼度を総合評価する x_buyscore.apply_buyscore
    # に委ねる。見送りと判定された場合、top は空リストではなく
    # "評価はしたが購入しなかった" 記録として _run_main 側に伝わるよう、
    # skip_info を持つ特別なリストとして返す。
    if not top:
        return top

    # AI一致指数は新聞生成時にしか確定しない値のため、この時点では
    # upset_score から近似する（x_note_report.py と同じ近似式）。
    approx_match_index = min(100, upset_score * 10.5)

    # 市場とAIのモデル順位の乖離（順位差ベース、x_post.py と同じロジック）
    market_gap = 0.0
    if model_rank and market_rank:
        best_lane = min(model_rank, key=model_rank.get)  # モデル1位の艇
        if best_lane in market_rank:
            rank_diff = market_rank[best_lane] - model_rank[best_lane]
            market_gap = round(rank_diff / 5.0, 3)

    buyscore_context = {
        "match_index":        approx_match_index,
        "match_index_approx": True,
        "race_type":          race_type,
        "has_exhibition":     has_exhibition,
        "market_gap":         market_gap,
        "upset_score":        upset_score,
        "ex_rank_1st":        0,   # 【朝刊AI】展示は使用しないため常に0
    }

    try:
        buy_result = _apply_buyscore(
            top, buyscore_context,
            venue=venue_name, race=str(race_number), date=race_date,
        )
    except Exception as e:
        log.warning("[購入判定] BuyScoreエンジン失敗（フォールバックで従来通り購入扱い）: %s", e)
        return top

    if buy_result["passthrough"]:
        log.info("[購入判定] 見送り: %s %sR 理由=%s",
                 venue_name, race_number, buy_result["passthrough_reason"])
        # 「予想はしたが購入しなかった」ことが後段で分かるよう、
        # 元の候補に見送り情報を付与した特別なリストを返す。
        # amount=0 のため実際の投資は発生しない。
        skipped = []
        for t in top[:1]:   # 最有力候補のみ記録（見送り理由の参照用）
            t = dict(t)
            t["purchased"] = False
            t["skip_reason"] = buy_result["passthrough_reason"]
            t["buyscore"] = t.get("buyscore", 0)
            t["amount"] = 0
            skipped.append(t)
        return skipped

    # 購入判定されたレース: BuyScoreでランク付けされた候補（最大4点）を採用
    purchased = []
    for c in buy_result["buy"]:
        c = dict(c)
        c["purchased"] = True
        c["skip_reason"] = None
        c["match_index"] = approx_match_index  # 購入履歴保存用（AI一致指数の近似値）
        purchased.append(c)

    log.info("[購入判定] 購入: %s %sR %d点 (BuyScore最高=%.0f)",
             venue_name, race_number, len(purchased),
             max((c.get("buyscore", 0) for c in purchased), default=0))

    return purchased


def _get_bet_multiplier() -> float:
    """
    hit_record.csv の直近成績に基づいてベット倍率を返す。
    通常: 1.0  /  連敗5〜9: 0.5  /  連敗10+: 0.0（停止）
    日次損失-10000円超: 0.0（停止）
    """
    try:
        import csv as _csv
        csv_file = "hit_record.csv"
        if not os.path.exists(csv_file):
            return 1.0
        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        if not rows:
            return 1.0

        # 連敗チェック（通知済みレースのみ）
        streak = 0
        for r in reversed(rows):
            if not r.get("pred_combo"):
                continue  # 通知していないレースは除外
            if r.get("hit") in ("", None, "-1"):
                continue
            if int(r.get("hit", 0) or 0) == 0:
                streak += 1
            else:
                break
        if streak >= 10:
            log.warning("連敗10回超 → ベット停止 (streak=%d)", streak)
            return 0.0
        if streak >= 5:
            log.info("連敗%d回 → ベット50%%縮小", streak)
            return 0.5

        # 日次損失チェック
        from datetime import datetime, timezone, timedelta
        today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d")
        today_rows = [r for r in rows if str(r.get("date","")).replace("-","") == today]
        daily_profit = sum(int(r.get("profit", -100) or -100) for r in today_rows)
        if daily_profit <= -10000:
            log.warning("日次損失%d円 → ベット停止", daily_profit)
            return 0.0
        return 1.0
    except Exception:
        return 1.0


def _check_race_quality(
    boats: list,
    weather,
    ml_probs: dict[int, float],
    upset_score: float,
    has_exhibition: bool,
) -> tuple[float, str]:
    """
    レース品質スコアを計算する（0.0〜1.0）。
    低品質レース（0.3未満）はスキップ推奨。

    品質の要素:
      - 展示データの充実度
      - 荒れスコアの高さ
      - 軸候補の明確さ（最有力と2番手の差）
      - 気象条件の荒れ適性

    戻り値: (quality_score, skip_reason)
    """
    quality = 0.0
    reasons = []

    # 展示データの充実度（0〜0.25）
    if has_exhibition:
        ex_count = sum(1 for b in boats if b.ex_time and b.ex_time > 0)
        quality += min(ex_count / 6.0, 1.0) * 0.25
    else:
        reasons.append("展示なし")

    # 荒れスコア（0〜0.30）
    if upset_score >= 6.0:
        quality += 0.30
    elif upset_score >= 4.0:
        quality += 0.15
    else:
        reasons.append(f"荒れ低({upset_score:.1f})")

    # 軸候補の明確さ（0〜0.25）
    if ml_probs:
        sorted_p = sorted(ml_probs.values(), reverse=True)
        if len(sorted_p) >= 2:
            gap = sorted_p[0] - sorted_p[1]
            if gap >= 0.15:
                quality += 0.25
            elif gap >= 0.08:
                quality += 0.12
            else:
                reasons.append("軸不明")

    # 【朝刊AI】気象適性は previews 由来のため使用しない（weather は常に空）
    if weather:
        ws = weather.wind_speed or 0
        wh = weather.wave_height or 0
        wd = weather.wind_direction or ""
        if (ws >= 5 and wd == "向") or wh >= 3:
            quality += 0.20
        elif ws >= 3 or wh >= 3:
            quality += 0.10

    quality = min(quality, 1.0)
    return quality, (" / ".join(reasons) if reasons else "")


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
        f"[v2.9]【荒れ検知】{result.venue_name} {result.race_number}R "
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

    # 1号艇危険度AI
    danger = result.score_detail.get("1号艇危険度", "")
    if danger:
        try:
            d_val  = float(danger)
            d_bar  = "🔴" * min(int(d_val / 2), 4)
            lines.append(f"⚠️ 1号艇危険度: {d_val:.0f}点 {d_bar}")
        except ValueError:
            pass

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

    # オッズ急落アラート
    if result.score_detail.get("オッズ急落"):
        lines.append(f"📉 オッズ急落: {result.score_detail['オッズ急落']}")

    # 推奨買い目（EV順ランキング）
    if result.recommended_bets:
        best0 = result.recommended_bets[0]
        # レースタイプ + レジーム
        regime_label = {
            "extreme_wind": "🌪️ 極端荒天",
            "rough":        "🌊 荒れ環境",
            "night_stable": "🌙 ナイター安定",
            "calm":         "☀️ 標準",
        }
        type_str   = best0.get("race_type", "")
        regime_str = regime_label.get(best0.get("regime",""), "")
        if type_str or regime_str:
            lines.append(f"🏁 {type_str}  {regime_str}".strip())
        # 買い理由
        if best0.get("why_bet"):
            lines.append("📌 " + " / ".join(best0["why_bet"][:4]))

        lines.append("💰 EV上位買い目（期待値順）")
        pattern_jp = {
            "本命軸": "本命軸", "本命崩れ": "本命崩れ",
            "差し": "差し", "まくり": "まくり", "万舟": "万舟",
            # 旧パターン名後方互換
            "nami":"本命崩れ","makuri":"まくり","sashi":"差し","ana":"万舟","safe":"保険",
        }
        for bet in result.recommended_bets[:8]:
            pname    = pattern_jp.get(bet["pattern"], bet["pattern"])
            amt      = f" ¥{bet['amount']:,}" if bet.get("amount", 0) > 0 else " (停止中)"
            ev_bar   = "★" * min(int(bet["ev"] * 2), 5)
            bs       = f" 信頼{bet.get('bet_score',0):.2f}" if bet.get("bet_score") else ""
            lines.append(
                f"  {bet['combo']}  {bet['odds']:.0f}倍 "
                f"EV:{bet['ev']:.2f}{ev_bar}  [{pname}]{bs}{amt}"
            )
    elif result.best_combo:
        odds_val = result.odds_map.get(result.best_combo, 0)
        lines.append(f"💰 推奨: {result.best_combo} ({odds_val:.0f}倍 EV:{result.best_ev:.2f})")

    # フッター（不確実性・disagreeが高い場合のみ表示）
    if result.recommended_bets:
        b0 = result.recommended_bets[0]
        unc  = b0.get("uncertainty", 0)
        disc = b0.get("disagreement", 0)
        if unc >= 0.60 or disc >= 0.50:
            lines.append(f"⚡ 不確実性:{unc:.2f} 意見割れ:{disc:.2f} → 少額推奨")

    return subject, "\n".join(lines)


# ════════════════════════════════════════════════════════════
# Gmail 送信
# ════════════════════════════════════════════════════════════

def send_line(body: str) -> bool:
    """
    LINE Notify でプッシュ通知を送信する。成功で True。

    必要な環境変数:
        LINE_NOTIFY_TOKEN … LINE Notifyのトークン
    """
    token = os.getenv("LINE_NOTIFY_TOKEN", "")
    if not token:
        log.debug("LINE_NOTIFY_TOKEN 未設定 → LINE通知スキップ")
        return False

    try:
        import urllib.request, urllib.parse
        data = urllib.parse.urlencode({"message": body[:1000]}).encode("utf-8")
        req  = urllib.request.Request(
            "https://notify-api.line.me/api/notify",
            data=data,
            headers={"Authorization": f"Bearer {token}"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as res:
            if res.status == 200:
                log.info("LINE通知送信成功")
                return True
            else:
                log.error("LINE通知失敗: status=%d", res.status)
                return False
    except Exception as e:
        log.error("LINE通知エラー: %s", e)
        return False


# ════════════════════════════════════════════════════════════
# ランキングフィルタ（ranking_filter.json 参照）
# ════════════════════════════════════════════════════════════

def _load_ranking_filter() -> set[tuple[int, int]]:
    """
    ranking_filter.json から通知許可レースのセットを返す。
    ファイルが存在しない・読み込み失敗時は None を返す（全許可）。
    """
    path = "ranking_filter.json"
    if not os.path.exists(path):
        return None   # ファイルなし → 制限なし（初日対応）
    try:
        import json as _json
        with open(path, "r", encoding="utf-8") as f:
            data = _json.load(f)
        allowed = set()
        for a in data.get("allowed", []):
            vn  = a.get("venue_num")
            rno = a.get("race")
            if vn is not None and rno is not None:
                allowed.add((int(vn), int(rno)))
        return allowed
    except Exception as e:
        log.warning("[フィルタ] 読み込み失敗: %s → 制限なし", e)
        return None


def _is_ranking_target(venue_num: int, race_number: int) -> bool:
    """
    このレースがランキングに含まれているか確認する。
    ranking_filter.json がない場合は True（全許可）を返す。
    """
    allowed = _load_ranking_filter()
    if allowed is None:
        return True   # ファイルなし → 全許可
    return (venue_num, race_number) in allowed


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
    pw_cache = {}
    log.info("直前情報取得: スキップ")
    if False:
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

    pw_targets = []
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
    # 捨て条件を事前ロード（ログから自動抽出）
    _skip_conds = _load_skip_conditions("hit_record.csv")
    if _skip_conds:
        log.info("捨て条件: %d件 (%s)", len(_skip_conds),
                 " / ".join(f"{c['key']}={c['val']}" for c in _skip_conds[:3]))

    # ── ④ 異常日検知 ──────────────────────────────────────────
    # 今日の環境が過去平均から著しくズレていないかチェック
    all_wind_speeds = [item[3].wind_speed for item in race_list
                       if len(item) > 3 and item[3] and item[3].wind_speed]
    all_waves       = [item[3].wave_height for item in race_list
                       if len(item) > 3 and item[3] and item[3].wave_height]
    anomaly_flags: list[str] = []
    if all_wind_speeds:
        avg_ws = sum(all_wind_speeds) / len(all_wind_speeds)
        if avg_ws >= 7.0:
            anomaly_flags.append(f"全場平均風速{avg_ws:.1f}m（異常強風）")
    if all_waves:
        avg_wh = sum(all_waves) / len(all_waves)
        if avg_wh >= 8:
            anomaly_flags.append(f"全場平均波高{avg_wh:.0f}cm（高波）")
    # 展示なしレース率
    no_ex_rate = sum(1 for item in race_list
                     if not any(b.ex_time for b in item[2])) / max(len(race_list), 1)
    if no_ex_rate >= 0.6:
        anomaly_flags.append(f"展示なし率{no_ex_rate:.0%}（APIデータ不足？）")

    if anomaly_flags:
        log.warning("⚠️ 異常日検知: %s", " / ".join(anomaly_flags))

    # ── ⑤ system_confidence ───────────────────────────────────
    # システム全体の今日の自信度（0.0〜1.0）
    system_conf = 1.0
    if anomaly_flags:
        system_conf -= 0.15 * len(anomaly_flags)
    if no_ex_rate >= 0.5:
        system_conf -= 0.20
    system_conf = max(system_conf, 0.20)
    if system_conf < 0.60:
        log.warning("⚠️ system_confidence低: %.2f → 全体的にベット縮小", system_conf)

    # ── exposure管理（型・レジーム別の偏り制限）────────────────
    # MAX_STYLE_EXPOSURE: 同じ race_type の通知は全体の35%まで
    # MAX_CLUSTER_EXPOSURE: 同じ cluster（レジーム×venue）の通知は3件まで
    MAX_STYLE_EXPOSURE  = 0.60
    MAX_CLUSTER_BETS    = 5
    style_count:   dict[str, int] = {}   # race_type → count
    cluster_count: dict[str, int] = {}   # "regime_venue" → count
    total_notified  = 0
    MAX_DAILY_BETS  = 9999  # 上限なし

    sent_file = f"sent_{race_date}.txt"
    try:
        import json as _json
        sent_set = set()
        with open(sent_file, "r", encoding="utf-8") as sf:
            for _line in sf:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _obj = _json.loads(_line)
                    sent_set.add(_obj.get("key", ""))
                except Exception:
                    sent_set.add(_line)
    except Exception:
        sent_set = set()

    # ── オッズ並列取得（締切前・捨て条件除外済みの候補のみ）────
    from concurrent.futures import ThreadPoolExecutor, as_completed
    _race_date_str = str(race_date).replace("-", "")

    # 締切前かつ捨て条件外のレースだけ絞る
    _pre_targets = []
    for item in race_list:
        vn, rno = item[0], item[1]
        closed_at_pre = item[4] if len(item) > 4 else ""
        if closed_at_pre and not skip_filter:
            try:
                _cdt = datetime.strptime(closed_at_pre, "%Y-%m-%d %H:%M:%S")
                if (_cdt - now).total_seconds() < 0:
                    continue   # 締切済み
            except Exception:
                pass
        _vname = VENUE_NAMES.get(vn, f"場{vn}")
        _skip_pre, _ = _should_skip_by_condition(_vname, "", "", _skip_conds)
        if _skip_pre:
            continue
        _pre_targets.append((vn, rno))

    try:
        from odds_fetch import fetch_odds as _fetch_odds

        def _fetch_one(args):
            vn, rno = args
            try:
                return (vn, rno), (_fetch_odds(rno, str(vn).zfill(2), _race_date_str) or {})
            except Exception:
                return (vn, rno), {}

        log.info("オッズ並列取得: %d レース (8スレッド) ← 事前フィルタ済み %d→%d件",
                 len(_pre_targets), len(race_list), len(_pre_targets))
        _prefetched_odds: dict[tuple, dict] = {}
        with ThreadPoolExecutor(max_workers=8) as _ex:
            for future in as_completed(_ex.submit(_fetch_one, t) for t in _pre_targets):
                key, res = future.result()
                _prefetched_odds[key] = res
        hit = sum(1 for v in _prefetched_odds.values() if v)
        log.info("オッズ取得完了: %d/%d レース取得成功", hit, len(_pre_targets))
    except Exception as _oe:
        log.warning("オッズ並列取得失敗: %s → 逐次取得にフォールバック", _oe)
        _prefetched_odds = {}

    for venue_num, race_number, boats, weather, *rest in race_list:
        race_grade = rest[1] if len(rest) > 1 else 0
        closed_at  = rest[0] if len(rest) > 0 else ""

        # 【朝刊AI】展示データ(previews由来)は予想に使用しない。
        # has_exhibition は常に False に固定し、下流の既存フォールバック
        # ロジック（展示なし時の保守的な処理）が常時適用されるようにする。
        # これにより「展示取得を待って通知を遅らせる」処理も不要になる
        # （旧: 展示タイムなしの場合は締切15分以内のみ通知 → 廃止）。
        has_exhibition = False
        # ── 強制スキップ場 ────────────────────────────────────────
        if venue_num in VENUE_FORCE_SKIP:
            log.debug("強制スキップ: %s %dR",
                      VENUE_NAMES.get(venue_num, f"場{venue_num}"), race_number)
            continue

        try:
            # ナイター場判定（場コード: 4平和島 6浜名湖 12住之江 17宮島 20若松 21芦屋 22福岡 23唐津 24大村）
            NIGHT_VENUES = {4, 6, 12, 17, 20, 21, 22, 23, 24}
            is_night = venue_num in NIGHT_VENUES

            # 【朝刊AI】previews由来（展示・気象）データを使わない新エンジンで計算
            score, detail, target = calculate_upset_score_v2(
                boats, race_grade,
                venue_num=venue_num,
                is_night=is_night,
            )

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

            # ── 捨て条件チェック（ログから自動抽出した負け条件）──
            # 【朝刊AI】wind_dir は previews 由来のため weather は常に空。
            # wind_dir 条件は実質的に発火しなくなる（venue/race_type条件は有効）。
            venue_name_str = VENUE_NAMES.get(venue_num, f"場{venue_num}")
            wd_str = weather.wind_direction if weather else ""
            rt_str = detail.get("race_type", "")   # calculate_upset_scoreから取得できれば
            should_skip, skip_reason = _should_skip_by_condition(
                venue_name_str, wd_str, rt_str, _skip_conds
            )
            if should_skip:
                log.debug("捨て条件スキップ: %s %dR %s",
                          venue_name_str, race_number, skip_reason)
                continue

            # ── 展示タイムなし → EV閾値・prob下限を厳しくする（_evaluate_bets内で対処）
            if not has_exhibition:
                detail["展示補正"] = "展示タイムなし（EV閾値+0.20・prob下限+0.008）"

            # ═══════════════════════════════════════════════════════
            # ▼ EV計算を先に行う（EVフィルタが主軸）
            # ═══════════════════════════════════════════════════════
            recommended = []
            odds_map: dict = {}
            dropped: list[str] = []

            if ml_probs:
                try:
                    # 並列取得済みキャッシュから取得（なければ逐次フォールバック）
                    odds_map = _prefetched_odds.get((venue_num, race_number), {})
                    if not odds_map:
                        from odds_fetch import fetch_odds
                        _rds = str(race_date).replace("-", "")
                        odds_map = fetch_odds(race_number, str(venue_num).zfill(2), _rds) or {}
                except Exception as oe:
                    log.debug("オッズ取得失敗: %s", oe)

                if odds_map:
                    # ── 多要素 confidence 計算 ──────────────────────
                    multi_conf = _calc_multi_confidence(
                        ml_probs, score, has_exhibition, weather, boats, odds_map
                    )
                    detail["confidence"] = f"{multi_conf:.3f}"

                    # ── オッズ急落チェック ─────────────────────────
                    dropped = _check_odds_drop(venue_num, race_number, odds_map)
                    if dropped:
                        detail["オッズ急落"] = " / ".join(dropped[:3])
                        log.info("オッズ急落: %s %dR %s",
                                 VENUE_NAMES.get(venue_num, f"場{venue_num}"),
                                 race_number, dropped[:3])

                    # 全120通りのEVランキング
                    patterns = _generate_patterns(target, score)
                    recommended = _evaluate_bets(
                        patterns, ml_probs, odds_map,
                        target_lanes=target,
                        has_exhibition=has_exhibition,
                        boats=boats,
                        upset_score=score,
                        odds_dropped=dropped if odds_map else [],
                        weather=weather,
                        venue_num=venue_num,
                        race_number=race_number,
                        race_date=race_date,
                        venue_name=VENUE_NAMES.get(venue_num, f"場{venue_num}"),
                    )
                    detail["EV上位"] = (
                        f"{recommended[0]['combo']} EV={recommended[0]['ev']:.2f}"
                        if recommended else "なし"
                    )
                    # 隊列崩れ率（recommendedから取得）
                    if recommended:
                        fb = recommended[0].get("formation_break", 0)
                        if fb >= 0.40:
                            detail["隊列崩れ率"] = f"{fb:.0%} ⚠️"
                        elif fb >= 0.25:
                            detail["隊列崩れ率"] = f"{fb:.0%}"

            # ── レース品質スコアによる選別 ──────────────────────
            race_quality, quality_skip_reason = _check_race_quality(
                boats, weather, ml_probs or {}, score, has_exhibition
            )
            detail["品質スコア"] = f"{race_quality:.2f}"
            if race_quality < 0.15:   # 旧: 0.30（厳しすぎた）
                log.debug("品質不足: %s %dR quality=%.2f (%s)",
                          VENUE_NAMES.get(venue_num, f"場{venue_num}"), race_number,
                          race_quality, quality_skip_reason)
                continue

            # 【朝刊AI】気象条件フィルタは previews 由来データに依存するため廃止。
            # weather は常に空のため、このフィルタを残すと全レースが
            # スキップされてしまう。朝データのみの品質スコア(_check_race_quality)
            # による選別のみで判定する。

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

            # 【購入判定分離】recommended は「予想」の全候補を含みうるが、
            # 実際に通知・記録するのは purchased=True（BuyScoreで購入判定
            # されたレース）のみ。purchased=False は「評価はしたが購入
            # しなかった」レースであり、別途 skip_log.jsonl に記録する。
            is_purchased = bool(recommended) and recommended[0].get("purchased", True)

            if recommended and not is_purchased:
                try:
                    import json as _json
                    skip_entry = {
                        "date": race_date, "venue": VENUE_NAMES.get(venue_num, f"場{venue_num}"),
                        "venue_num": venue_num, "race": race_number,
                        "upset_score": round(score, 3),
                        "best_combo": recommended[0].get("combo", ""),
                        "best_ev": recommended[0].get("ev", 0),
                        "buyscore": recommended[0].get("buyscore", 0),
                        "skip_reason": recommended[0].get("skip_reason", ""),
                        "model_version": _asahi_model_version(),
                    }
                    with open("skip_log.jsonl", "a", encoding="utf-8") as sf:
                        sf.write(_json.dumps(skip_entry, ensure_ascii=False) + "\n")
                except Exception as _se:
                    log.debug("見送りログ保存失敗: %s", _se)
                log.info("[見送り] %s %dR 理由=%s",
                         VENUE_NAMES.get(venue_num, f"場{venue_num}"), race_number,
                         recommended[0].get("skip_reason", ""))

            # best は購入・見送りに関わらず定義する（後段の _pred_entry 構築で
            # 「予想は出したが購入しなかった」場合も combo 等を参照するため）
            best = recommended[0] if recommended else {}

            if recommended and is_purchased:
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

                # ── 予測ログ保存（完全版・キャリブレーション用）────────
                try:
                    import json as _json
                    pred_file = f"pred_{race_date}_{venue_num}_{race_number}.json"
                    NIGHT_VENUES = {4, 6, 12, 17, 20, 21, 22, 23, 24}
                    # 【朝刊AI・教師データ用】展示・気象は予想には使わないが、
                    # Phase2の学習・検証のため _PREVIEW_RAW_CACHE から生データを記録する。
                    _preview_raw = _PREVIEW_RAW_CACHE.get((venue_num, race_number), {})
                    _pred_data = {
                        # レース識別
                        "race_id":    f"{race_date}_{venue_num}_{race_number}",
                        "venue":      VENUE_NAMES.get(venue_num, f"場{venue_num}"),
                        "venue_num":  venue_num,
                        "race":       race_number,
                        "night":      int(venue_num in NIGHT_VENUES),
                        # 推奨買い目（全点）
                        "buy":        [b["combo"] for b in recommended],
                        # ベスト買い目
                        "combo":      best["combo"],
                        "odds":       best["odds"],
                        "prob":       best["prob"],
                        "ev":         best["ev"],
                        "composite":  best.get("composite", 0),
                        "confidence": best.get("confidence", 0),
                        # 買い理由 / レースタイプ / レジーム
                        "why_bet":      best.get("why_bet", []),
                        "race_type":    best.get("race_type", ""),
                        "regime":       best.get("regime", ""),
                        "disagreement": best.get("disagreement", 0),
                        "uncertainty":  best.get("uncertainty", 0),
                        # システム自信度
                        "system_conf":  round(system_conf, 3),
                        # モデルスコア（朝刊AI: previews由来データを使わずに算出）
                        "upset_score": round(score, 3),
                        "model_version": _asahi_model_version(),
                        # 【教師データ】展示・気象の生データ（予想には未使用、Phase2学習用）
                        "wind_speed": _preview_raw.get("wind_speed"),
                        "wind_dir":   _preview_raw.get("wind_direction"),
                        "wave":       _preview_raw.get("wave_height"),
                        "weather":    _preview_raw.get("weather"),
                        # オッズ急落
                        "odds_dropped": dropped[:3] if dropped else [],
                        # 結果（後で埋める）
                        "result":     "",
                        "hit":        -1,
                        "profit":     0,
                    }
                    with open(pred_file, "w", encoding="utf-8") as pf:
                        _json.dump(_pred_data, pf, ensure_ascii=False)
                except Exception as _pe:
                    log.debug("予測ログ保存失敗: %s", _pe)

            # 【朝刊AI】通知直前の気象・展示再取得は行わない。
            # 新聞・危険艇・万舟等は朝1回生成した内容で固定し、
            # 送信直前にデータを再取得・再計算することはポリシー上禁止する。

            log.info(
                "荒れ検知: %s %dR score=%.2f target=%s",
                result.venue_name, result.race_number,
                result.upset_score, result.target_lanes,
            )

            # 買い目なしの場合は通知スキップ（的中判定もできないため）
            if not recommended:
                log.debug("買い目なしスキップ: %s %dR (EVフィルタで全除外)",
                         result.venue_name, race_number)
                continue

            subject, body = build_message(result)

            # ── ① exposure チェック（型・クラスター偏り制限）──────
            race_type_str = recommended[0].get("race_type", "不明") if recommended else "不明"
            regime_str    = recommended[0].get("regime", "calm")     if recommended else "calm"
            cluster_key   = f"{regime_str}_{venue_num}"

            # 同じraceTypeが多すぎる
            if total_notified > 0:
                # race_type=不明のときはexposure制限をスキップ（分類できないため）
                if race_type_str != "不明":
                    style_ratio = style_count.get(race_type_str, 0) / total_notified
                    if style_ratio >= MAX_STYLE_EXPOSURE:
                        log.info("exposure制限(型): %s %dR 型=%s (%.0f%%超)",
                                 result.venue_name, race_number, race_type_str, MAX_STYLE_EXPOSURE*100)
                        continue

            # 同じクラスターが多すぎる
            if cluster_count.get(cluster_key, 0) >= MAX_CLUSTER_BETS:
                log.info("exposure制限(cluster): %s %dR cluster=%s (%d件超)",
                         result.venue_name, race_number, cluster_key, MAX_CLUSTER_BETS)
                continue

            # 1日の上限
            if total_notified >= MAX_DAILY_BETS:
                log.info("1日上限到達: %d件", MAX_DAILY_BETS)
                break

            # ── ランキングフィルタチェック ──────────────────────────
            _ranking_ok = _is_ranking_target(venue_num, race_number)
            if not _ranking_ok:
                log.info(
                    "ランキング外スキップ: %s %dR（メール送信なし・データ蓄積のみ）",
                    result.venue_name, race_number,
                )
                # sent_*.txt に予測データを保存（結果照合・hit_record.csv 記録のため）
                try:
                    import json as _json
                    _best = recommended[0] if recommended else {}
                    _is_purchased_pre = bool(recommended) and _best.get("purchased", True)
                    _pred_entry_pre = {
                        "key":         notify_key,
                        "combo":       _best.get("combo", ""),
                        "buy":         [b["combo"] for b in recommended] if (recommended and _is_purchased_pre) else [],
                        "buy_amounts": [b.get("amount", 100) for b in recommended] if (recommended and _is_purchased_pre) else [],
                        "odds":        _best.get("odds", 0),
                        "prob":        _best.get("prob", 0),
                        "ev":          _best.get("ev", 0),
                        "confidence":  _best.get("confidence", 0),
                        "why_bet":     _best.get("why_bet", []),
                        "race_type":   _best.get("race_type", ""),
                        "upset_score": round(score, 3),
                        "venue":       VENUE_NAMES.get(venue_num, f"場{venue_num}"),
                        "venue_num":   venue_num,
                        "race":        race_number,
                        "night":       int(venue_num in {4,6,12,17,20,21,22,23,24}),
                        # 【購入判定分離】
                        "purchased":   _is_purchased_pre,
                        "buyscore":    _best.get("buyscore", 0),
                        "match_index": _best.get("match_index", 0),
                        "skip_reason": _best.get("skip_reason", "") if not _is_purchased_pre else "",
                        # 【教師データ】予想には未使用、Phase2学習用の生データ
                        "wind_speed":  _PREVIEW_RAW_CACHE.get((venue_num, race_number), {}).get("wind_speed"),
                        "wind_dir":    _PREVIEW_RAW_CACHE.get((venue_num, race_number), {}).get("wind_direction"),
                        "wave":        _PREVIEW_RAW_CACHE.get((venue_num, race_number), {}).get("wave_height"),
                        "model_version": _asahi_model_version(),
                        "feat_win_rate":       (detail.get("_features") or {}).get("win_rate"),
                        "feat_motor":          (detail.get("_features") or {}).get("motor"),
                        "feat_avg_st":         (detail.get("_features") or {}).get("avg_st"),
                        "feat_racer_class":    (detail.get("_features") or {}).get("racer_class"),
                        "feat_course_st_1c":   (detail.get("_features") or {}).get("course_st_1c"),
                        "feat_course_rank_1c": (detail.get("_features") or {}).get("course_rank_1c"),
                        "feat_danger_breakdown": _json.dumps(
                            (detail.get("_features") or {}).get("danger_breakdown") or {},
                            ensure_ascii=False,
                        ),
                        "ranking_skip": True,   # ランキング外フラグ
                    }
                    _sl, _ks = [], set()
                    if os.path.exists(sent_file):
                        with open(sent_file, "r", encoding="utf-8") as sf:
                            for _ln in sf:
                                _ln = _ln.strip()
                                if not _ln: continue
                                try:
                                    _ks.add(_json.loads(_ln).get("key",""))
                                    _sl.append(_ln)
                                except Exception:
                                    _ks.add(_ln); _sl.append(_ln)
                    if notify_key not in _ks:
                        _sl.append(_json.dumps(_pred_entry_pre, ensure_ascii=False))
                    with open(sent_file, "w", encoding="utf-8") as sf:
                        sf.write("\n".join(_sl))
                except Exception as _pe:
                    log.warning("ランキング外データ蓄積失敗: %s", _pe)
                continue

            # ── メール通知は停止（転がし専用エンジンに移行）────────
            # send_notification はスキップし、予測データの蓄積のみ継続する
            log.debug(
                "通知スキップ（蓄積のみ）: %s %dR",
                VENUE_NAMES.get(venue_num, f"場{venue_num}"), race_number,
            )
            notified += 1
            total_notified += 1
            # exposure更新
            style_count[race_type_str]  = style_count.get(race_type_str, 0) + 1
            cluster_count[cluster_key]  = cluster_count.get(cluster_key, 0) + 1
            # 送信済みに記録（照合・hit_record.csv のために必要）
            sent_set.add(notify_key)
            # 予測データをJSON Lines形式でsent_fileに保存
            import json as _json
            _pred_entry = {
                "key":        notify_key,
                "combo":      best.get("combo", "") if recommended else "",
                "buy":        [b["combo"] for b in recommended] if (recommended and is_purchased) else [],
                "buy_amounts": [b.get("amount", 100) for b in recommended] if (recommended and is_purchased) else [],
                "odds":       best.get("odds", 0)   if recommended else 0,
                "prob":       best.get("prob", 0)   if recommended else 0,
                "ev":         best.get("ev", 0)     if recommended else 0,
                "confidence": best.get("confidence", 0) if recommended else 0,
                "why_bet":    best.get("why_bet", [])   if recommended else [],
                "race_type":  best.get("race_type", "") if recommended else "",
                "upset_score": round(score, 3),
                "venue":      VENUE_NAMES.get(venue_num, f"場{venue_num}"),
                "venue_num":  venue_num,
                "race":       race_number,
                "night":      int(venue_num in {4,6,12,17,20,21,22,23,24}),
                # 【購入判定分離】予想は出したが実際に購入したかどうかを区別する。
                # purchased=False の場合、cost=0（投資額なし）となり、
                # 回収率・ROI集計から自動的に除外される。
                "purchased":   is_purchased,
                "buyscore":    best.get("buyscore", 0) if recommended else 0,
                "match_index": best.get("match_index", 0) if recommended else 0,
                "skip_reason": best.get("skip_reason", "") if (recommended and not is_purchased) else "",
                # 【教師データ】予想には未使用、Phase2学習用の生データ
                "wind_speed": _PREVIEW_RAW_CACHE.get((venue_num, race_number), {}).get("wind_speed"),
                "wind_dir":   _PREVIEW_RAW_CACHE.get((venue_num, race_number), {}).get("wind_direction"),
                "wave":       _PREVIEW_RAW_CACHE.get((venue_num, race_number), {}).get("wave_height"),
                "model_version": _asahi_model_version(),
                # 【朝刊AI特徴量】実際に予想に使った各特徴量の値（Phase2重み学習用）
                "feat_win_rate":       (detail.get("_features") or {}).get("win_rate"),
                "feat_motor":          (detail.get("_features") or {}).get("motor"),
                "feat_avg_st":         (detail.get("_features") or {}).get("avg_st"),
                "feat_racer_class":    (detail.get("_features") or {}).get("racer_class"),
                "feat_course_st_1c":   (detail.get("_features") or {}).get("course_st_1c"),
                "feat_course_rank_1c": (detail.get("_features") or {}).get("course_rank_1c"),
                "feat_danger_breakdown": _json.dumps(
                    (detail.get("_features") or {}).get("danger_breakdown") or {},
                    ensure_ascii=False,
                ),
            }
            try:
                _sent_lines = []
                _keys_seen = set()
                if os.path.exists(sent_file):
                    with open(sent_file, "r", encoding="utf-8") as sf:
                        for _line in sf:
                            _line = _line.strip()
                            if not _line:
                                continue
                            try:
                                _obj = _json.loads(_line)
                                _keys_seen.add(_obj.get("key",""))
                                _sent_lines.append(_line)
                            except Exception:
                                _keys_seen.add(_line)
                                _sent_lines.append(_line)
                if notify_key not in _keys_seen:
                    _sent_lines.append(_json.dumps(_pred_entry, ensure_ascii=False))
                with open(sent_file, "w", encoding="utf-8") as sf:
                    sf.write("\n".join(_sent_lines))
            except Exception as ge:
                log.warning("sent_file保存失敗: %s", ge)

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
        # send_notification("【明日の注目レース予告】", body)  # 通知停止（転がし専用に移行）
        log.info("翌日予告（蓄積のみ、通知なし）: %d レース", len(top))

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


def _load_skip_conditions(csv_file: str = "hit_record.csv") -> list[dict]:
    """
    hit_record.csv から ROI < 0.5 かつ n >= 5 の条件を捨て条件として返す。
    """
    import csv as _csv
    if not os.path.exists(csv_file):
        return []
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        valid = [r for r in rows
                 if r.get("hit") not in ("", None, "-1")
                 and r.get("pred_combo")]  # 通知済みレースのみ
        valid_hit = [r for r in valid if int(r.get("hit", 0) or 0) == 1]
        if len(valid_hit) < 20:
            return []
        SKIP_KEYS = ["venue", "wind_dir", "race_type"]
        skip_conds = []
        for key in SKIP_KEYS:
            for val in set(str(r.get(key,"") or "") for r in valid):
                if not val:
                    continue
                subset = [r for r in valid if str(r.get(key,"") or "") == val]
                if len(subset) < 5:
                    continue
                def _c(r):
                    try:
                        c = r.get("cost")
                        if c not in (None, ""): return int(c)
                        return max(1, int(r.get("n_bets", 1) or 1)) * 100
                    except: return 100
                def _p(r):
                    try:
                        p = r.get("profit")
                        if p not in (None, ""): return int(p)
                    except: pass
                    pay = int(r.get("payout",0) or 0) if int(r.get("hit",0) or 0) else 0
                    return pay - _c(r)
                _cost = sum(_c(r) for r in subset)
                _prof = sum(_p(r) for r in subset)
                roi = (_cost + _prof) / _cost if _cost > 0 else 0
                if roi < 0.50:
                    skip_conds.append({"key":key,"val":val,"roi":round(roi,3),"n":len(subset)})
                    log.info("捨て条件登録: %s=%s (ROI=%.2f n=%d)", key, val, roi, len(subset))
        return skip_conds
    except Exception as e:
        log.debug("捨て条件読み込み失敗: %s", e)
        return []


def _should_skip_by_condition(
    venue: str, wind_dir: str, race_type: str,
    skip_conds: list[dict],
) -> tuple[bool, str]:
    """捨て条件に該当するかチェック。戻り値: (skip, reason)"""
    checks = {"venue": venue, "wind_dir": wind_dir, "race_type": race_type}
    for cond in skip_conds:
        if checks.get(cond["key"]) == cond["val"]:
            return True, f"{cond['key']}={cond['val']}(ROI={cond['roi']:.2f})"
    return False, ""


def _analyze_loss_reason(pred_combo: str, result_combo: str) -> str:
    """外れたレースの敗因を自動分類する。"""
    if not result_combo or result_combo == "不明":
        return "結果不明"
    rf = result_combo.split("-")[0] if "-" in result_combo else "?"
    pf = pred_combo.split("-")[0]   if pred_combo and "-" in pred_combo else "?"
    if rf in ["5","6"]:    return "万舟"
    if rf == "1" and pf != "1": return "1残り"
    if rf != pf:           return "展開違い"
    return "着順違い（頭当たり）"


def _calc_real_ev(csv_file: str = "hit_record.csv") -> dict:
    """理論EVと実測ROIのギャップを条件別に返す。"""
    import csv as _csv
    if not os.path.exists(csv_file):
        return {}
    try:
        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        valid = [r for r in rows
                 if r.get("hit") not in ("", None, "-1")
                 and r.get("pred_ev") and r.get("pred_combo")]
        if not valid:
            return {}
        def _c(r):
            try:
                c = r.get("cost")
                if c not in (None, ""): return int(c)
                return max(1, int(r.get("n_bets", 1) or 1)) * 100
            except: return 100
        def _p(r):
            try:
                p = r.get("profit")
                if p not in (None, ""): return int(p)
            except: pass
            pay = int(r.get("payout",0) or 0) if int(r.get("hit",0) or 0) else 0
            return pay - _c(r)
        result = {}
        def _ev_stats(subset, label):
            if len(subset) < 3: return
            theory = sum(float(r.get("pred_ev",0) or 0) for r in subset)/len(subset)
            cost   = sum(_c(r) for r in subset)
            prof   = sum(_p(r) for r in subset)
            real   = (cost + prof) / cost if cost > 0 else 0
            result[label] = {"n":len(subset),"理論EV":round(theory,3),
                             "実測ROI":round(real,3),"ギャップ":round(real-theory,3)}
        _ev_stats(valid, "全体")
        for lo,hi,lb in [(1.0,1.5,"EV1.0-1.5"),(1.5,2.0,"EV1.5-2.0"),(2.0,9.9,"EV2.0+")]:
            _ev_stats([r for r in valid if lo <= float(r.get("pred_ev",0) or 0) < hi], lb)
        for rt in set(r.get("race_type","") for r in valid if r.get("race_type")):
            _ev_stats([r for r in valid if r.get("race_type")==rt], f"型:{rt}")
        return result
    except Exception as e:
        log.debug("実測EV失敗: %s", e); return {}


def _calc_multi_confidence(
    ml_probs: dict[int, float],
    upset_score: float,
    has_exhibition: bool,
    weather,
    boats: list,
    odds_map: dict,
) -> float:
    """
    多要素 confidence スコア（0.0〜1.0）
    A. モデル合意度（軸の明確さ）× 0.50
    B. データ品質（展示・STの充実度）× 0.25
    C. 市場ギャップ（過小評価されてる艇あり）× 0.25
    """
    # A. モデル合意度
    a = 0.0
    if ml_probs:
        sp = sorted(ml_probs.values(), reverse=True)
        if len(sp) >= 2:
            a = min((sp[0]-sp[1]) / 0.30, 1.0) * 0.40
        a += min(upset_score / 10.0, 1.0) * 0.10

    # B. データ品質
    b = 0.0
    if has_exhibition:
        ex = sum(1 for boat in boats if boat.ex_time and boat.ex_time > 0)
        st = sum(1 for boat in boats if boat.ex_st is not None)
        b = min(ex/6.0, 1.0)*0.15 + min(st/6.0, 1.0)*0.10
    else:
        b = 0.05

    # C. 市場ギャップ
    c = 0.0
    if ml_probs and odds_map:
        sorted_ml = sorted(ml_probs.items(), key=lambda x: -x[1])
        mr = {lane: rank+1 for rank,(lane,_) in enumerate(sorted_ml)}
        la = {}
        for lane in range(1,7):
            lo_ = [v for k,v in odds_map.items() if k.startswith(f"{lane}-") and v>0]
            if lo_: la[lane] = sum(lo_)/len(lo_)
        if la:
            sm = sorted(la.items(), key=lambda x: x[1])
            mkt = {lane: rank+1 for rank,(lane,_) in enumerate(sm)}
            mg  = max((mkt.get(l,3)-mr.get(l,3)) for l in ml_probs)
            c   = min(max(mg,0)/5.0, 1.0) * 0.25

    return round(min(a+b+c, 1.0), 3)


def _print_dashboard(csv_file: str = "hit_record.csv") -> None:
    """
    ① 回収率ダッシュボード
    場別・風向き別・レースタイプ別ROI、実測EV、捨て条件、外れ分析を一括表示。
    """
    import csv as _csv
    from collections import defaultdict, Counter

    print(f"\n{'█'*55}")
    print(f"  📊 回収率ダッシュボード")
    print(f"{'█'*55}")

    if not os.path.exists(csv_file):
        print("  hit_record.csv が見つかりません")
        return

    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))
    valid = [r for r in rows
             if r.get("hit") not in ("", None, "-1")
             and r.get("pred_combo")]  # 通知済みレースのみ
    if len(valid) < 5:
        print(f"  データ不足: {len(valid)}件"); return

    def _cost(r):
        # costカラム優先、なければn_bets*100、それもなければ100
        try:
            c = r.get("cost")
            if c not in (None, ""): return int(c)
            return max(1, int(r.get("n_bets", 1) or 1)) * 100
        except: return 100
    def _profit(r):
        # profitカラム優先（実額計算済み）
        try:
            p = r.get("profit")
            if p not in (None, ""): return int(p)
        except: pass
        # フォールバック: payout - cost
        pay = int(r.get("payout",0) or 0) if int(r.get("hit",0) or 0) else 0
        return pay - _cost(r)
    total_bet  = sum(_cost(r) for r in valid)
    total_prof = sum(_profit(r) for r in valid)
    total_pay  = total_bet + total_prof
    total_roi  = total_pay / total_bet if total_bet > 0 else 0
    total_hit  = sum(int(r.get("hit",0) or 0) for r in valid)
    flag = "🟢 黒字" if total_roi>=1.0 else ("🟡 接戦" if total_roi>=0.8 else "🔴 赤字")

    print(f"\n  総合: ROI={total_roi:.3f} {flag}")
    print(f"  {len(valid)}件 / 的中{total_hit}件({total_hit/len(valid):.1%}) / {total_prof:+,}円")

    def _show_breakdown(key: str, title: str) -> None:
        stats: dict = defaultdict(lambda: {"n":0,"prof":0,"hit":0,"cost":0})
        for r in valid:
            v = str(r.get(key,"") or "不明")
            stats[v]["n"]    += 1
            stats[v]["hit"]  += int(r.get("hit",0) or 0)
            stats[v]["cost"] += _cost(r)
            stats[v]["prof"] += _profit(r)
        rows_ = [(v,s["n"],s["hit"],(s["cost"]+s["prof"])/max(s["cost"],1),s["prof"])
                 for v,s in stats.items() if s["n"]>=3]
        rows_.sort(key=lambda x:-x[3])
        print(f"\n  ── {title} ──")
        print(f"  {'':12} {'n':>4}  {'的中':>6}  {'ROI':>6}  {'損益':>9}")
        for v,n,h,roi,prof in rows_[:6]:
            f_ = "🟢" if roi>=1.0 else ("🟡" if roi>=0.7 else "🔴")
            print(f"  {str(v):<12} {n:>4}  {h:>2}({h/n:.0%})  {roi:>5.2f}  {prof:>+,}円 {f_}")

    _show_breakdown("venue",     "場別")
    _show_breakdown("wind_dir",  "風向き別")
    _show_breakdown("race_type", "レースタイプ別")
    _show_breakdown("night",     "ナイター別")

    # 実測EV
    real_ev = _calc_real_ev(csv_file)
    if real_ev:
        print(f"\n  ── 理論EV vs 実測ROI ──")
        print(f"  {'条件':<12} {'n':>4}  {'理論EV':>7}  {'実測':>7}  {'ギャップ':>9}")
        for label,v in real_ev.items():
            gf = "⚠️ " if v["ギャップ"]<-0.2 else ("✅" if v["ギャップ"]>-0.05 else "  ")
            print(f"  {label:<12} {v['n']:>4}  {v['理論EV']:>7.3f}  {v['実測ROI']:>7.3f}  {v['ギャップ']:>+.3f} {gf}")

    # 捨て条件
    skip_conds = _load_skip_conditions(csv_file)
    if skip_conds:
        print(f"\n  ── 捨て条件（自動スキップ対象） ──")
        for c in skip_conds:
            print(f"  ⚠️  {c['key']}={c['val']}  ROI={c['roi']:.2f}  n={c['n']}")

    # 外れ分析
    misses = [r for r in valid if int(r.get("hit",0) or 0)==0
              and r.get("result_combo") and r.get("result_combo")!="不明"]
    if misses:
        reasons = Counter(_analyze_loss_reason(r.get("pred_combo",""), r.get("result_combo",""))
                          for r in misses)
        print(f"\n  ── 外れ分析（{len(misses)}件） ──")
        for reason,cnt in reasons.most_common():
            pct = cnt/len(misses)
            bar = "█" * int(pct*20)
            print(f"  {reason:<20} {cnt:>3}件({pct:.0%}) {bar}")

    print(f"\n{'█'*55}")


def _auto_extract_patterns(csv_file: str = "hit_record.csv") -> None:
    """
    ① 勝ちパターン自動抽出エンジン
    hit_record.csv のログから条件の組み合わせごとにROIを集計し、
    「人間が気づかない勝ちパターン」を自動発見する。

    分析軸:
      - 単一条件（場, 風向き, ナイター, レースタイプ, 1着艇）
      - 2条件の組み合わせ（場×風向き, レースタイプ×ナイター など）
    """
    import csv as _csv
    from itertools import combinations
    from collections import defaultdict

    if not os.path.exists(csv_file):
        print("hit_record.csv が見つかりません")
        return

    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))

    valid = [r for r in rows
             if r.get("hit") not in ("", None, "-1")
             and r.get("pred_combo")]  # 通知済みレースのみ
    if len(valid) < 10:
        print(f"データ不足: {len(valid)}件（10件以上必要）")
        return

    # ── 各レコードに特徴量を付与 ────────────────────────────
    def _features(r: dict) -> dict:
        combo = r.get("pred_combo","")
        first = combo.split("-")[0] if "-" in combo else "?"
        ws    = float(r.get("wind_speed",0) or 0)
        return {
            "場":          r.get("venue","不明"),
            "風向き":       r.get("wind_dir","不明") or "不明",
            "ナイター":     "ナイター" if str(r.get("night","0")) == "1" else "昼間",
            "レースタイプ": r.get("race_type","不明") or "不明",
            "1着予測艇":    f"{first}号艇",
            "風速帯":       "強風5m+" if ws >= 5 else ("中風3-5m" if ws >= 3 else "弱風"),
            "波高帯":       "高波5cm+" if int(r.get("wave",0) or 0) >= 5 else "低波",
        }

    feat_rows = [(_features(r), r) for r in valid]

    def _c(r):
        try:
            c = r.get("cost")
            if c not in (None, ""): return int(c)
            return max(1, int(r.get("n_bets", 1) or 1)) * 100
        except: return 100
    def _p(r):
        try:
            p = r.get("profit")
            if p not in (None, ""): return int(p)
        except: pass
        pay = int(r.get("payout",0) or 0) if int(r.get("hit",0) or 0) else 0
        return pay - _c(r)
    def _roi_stats(subset: list) -> tuple:
        if not subset:
            return 0, 0, 0, 0
        n     = len(subset)
        hits  = sum(int(r.get("hit",0) or 0) for r in subset)
        cost  = sum(_c(r) for r in subset)
        prof  = sum(_p(r) for r in subset)
        roi   = (cost + prof) / cost if cost > 0 else 0
        return n, hits, roi, prof

    feat_keys = list(list(_features(valid[0]).keys()))

    print(f"\n{'='*60}")
    print(f"  勝ちパターン自動抽出  ({len(valid)}件)")
    print(f"{'='*60}")

    all_patterns = []

    # ── 単一条件 ────────────────────────────────────────────
    for key in feat_keys:
        vals = set(f[key] for f, _ in feat_rows)
        for val in vals:
            subset = [r for f, r in feat_rows if f[key] == val]
            n, hits, roi, profit = _roi_stats(subset)
            if n < 3:
                continue
            all_patterns.append({
                "条件":   f"{key}={val}",
                "n":      n,
                "hit":    hits,
                "roi":    roi,
                "profit": profit,
            })

    # ── 2条件の組み合わせ ────────────────────────────────────
    for k1, k2 in combinations(feat_keys, 2):
        vals1 = set(f[k1] for f, _ in feat_rows)
        vals2 = set(f[k2] for f, _ in feat_rows)
        for v1 in vals1:
            for v2 in vals2:
                subset = [r for f, r in feat_rows if f[k1] == v1 and f[k2] == v2]
                n, hits, roi, profit = _roi_stats(subset)
                if n < 3:
                    continue
                all_patterns.append({
                    "条件":   f"{k1}={v1} & {k2}={v2}",
                    "n":      n,
                    "hit":    hits,
                    "roi":    roi,
                    "profit": profit,
                })

    # ── ROI上位（勝ちパターン）と下位（負けパターン）を表示 ──
    all_patterns.sort(key=lambda x: -x["roi"])

    print(f"\n🟢 勝ちパターン TOP10（ROI高順）")
    print(f"  {'条件':<35} {'n':>4} {'的中':>4} {'ROI':>6} {'利益':>8}")
    print(f"  {'-'*60}")
    for p in all_patterns[:10]:
        hr = p["hit"] / p["n"] if p["n"] > 0 else 0
        print(f"  {p['条件']:<35} {p['n']:>4} {p['hit']:>3}({hr:.0%}) {p['roi']:>5.2f} {p['profit']:>+,}円")

    print(f"\n🔴 負けパターン TOP5（ROI低順）")
    print(f"  {'条件':<35} {'n':>4} {'的中':>4} {'ROI':>6} {'利益':>8}")
    print(f"  {'-'*60}")
    worst = [p for p in reversed(all_patterns) if p["n"] >= 5][:5]
    for p in worst:
        hr = p["hit"] / p["n"] if p["n"] > 0 else 0
        print(f"  {p['条件']:<35} {p['n']:>4} {p['hit']:>3}({hr:.0%}) {p['roi']:>5.2f} {p['profit']:>+,}円")

    # ── 発見した「高ROI条件」をスキップ推奨として表示 ──────
    skip_conds = [p for p in all_patterns if p["roi"] < 0.5 and p["n"] >= 5]
    if skip_conds:
        print(f"\n⚠️  スキップ推奨条件（ROI<0.5 かつ n≥5）")
        for p in skip_conds[:3]:
            print(f"  → {p['条件']} (n={p['n']}, ROI={p['roi']:.2f})")

    print(f"\n{'='*60}")


def _monte_carlo_simulation(
    csv_file: str = "hit_record.csv",
    n_days: int = 1000,
    n_sim: int = 2000,
    bets_per_day: float | None = None,
) -> None:
    """
    ④ Monte Carlo シミュレーション
    hit_record.csv の実績から1日あたりのベット分布を推定し、
    n_days日間 × n_sim回シミュレーションして長期資金リスクを評価する。

    表示:
      - 最大連敗の分布
      - 最大ドローダウンの分布
      - 月次回収率の分布（中央値・10%ile・90%ile）
      - 破産率（資金-50%以上）
    """
    import csv as _csv
    import random
    import statistics

    if not os.path.exists(csv_file):
        print("hit_record.csv が見つかりません")
        return

    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))

    valid = [r for r in rows
             if r.get("hit") not in ("", None, "-1")
             and r.get("pred_combo")]  # 通知済みレースのみ
    if len(valid) < 10:
        print(f"データ不足: {len(valid)}件")
        return

    # 実績から P(hit) と平均払戻を推定
    hits   = [r for r in valid if int(r.get("hit",0) or 0) == 1]
    payouts = [int(r.get("payout",0) or 0) for r in hits]
    n_bets = len(valid)
    hit_rate = len(hits) / n_bets if n_bets > 0 else 0.05
    avg_payout = statistics.mean(payouts) if payouts else 3000
    bet_unit = 500   # 1ベット単位（円）

    # 1日あたりのベット数（実績から推定 or 引数）
    if bets_per_day is None:
        # sent_*.txt から日数を推定
        import glob
        sent_files = glob.glob("sent_*.txt")
        bets_per_day = max(1.0, n_bets / max(len(sent_files), 1))

    print(f"\n{'='*55}")
    print(f"  Monte Carlo シミュレーション")
    print(f"  実績: {n_bets}件 / 的中率{hit_rate:.1%} / 平均払戻¥{avg_payout:,.0f}")
    print(f"  1日{bets_per_day:.1f}ベット × {n_days}日 × {n_sim}回シミュレーション")
    print(f"{'='*55}")

    random.seed(42)
    results = {
        "final_profit": [],
        "max_dd":       [],
        "max_streak":   [],
        "monthly_rois": [],
        "bankrupt":     0,
    }

    INITIAL_CAPITAL = 100000   # 初期資金10万円

    for _ in range(n_sim):
        capital    = INITIAL_CAPITAL
        peak       = INITIAL_CAPITAL
        max_dd     = 0
        streak     = 0
        max_streak = 0
        monthly    = []
        month_start = capital

        for day in range(n_days):
            # 月末処理
            if day > 0 and day % 30 == 0:
                roi = capital / month_start if month_start > 0 else 0
                monthly.append(roi)
                month_start = capital

            # 1日のベット数（ポアソン分布でランダム化）
            n_today = max(0, int(random.gauss(bets_per_day, bets_per_day ** 0.5)))

            for _ in range(n_today):
                # ベット
                capital -= bet_unit

                # 的中判定
                if random.random() < hit_rate:
                    pay = max(100, int(random.gauss(avg_payout, avg_payout * 0.3)))
                    capital += pay
                    streak  = 0
                else:
                    streak     += 1
                    max_streak  = max(max_streak, streak)

                # ドローダウン更新
                peak   = max(peak, capital)
                max_dd = max(max_dd, peak - capital)

                # 破産判定（初期資金の50%以下）
                if capital <= INITIAL_CAPITAL * 0.5:
                    results["bankrupt"] += 1
                    capital = INITIAL_CAPITAL * 0.5  # リセット
                    peak    = capital
                    break

        results["final_profit"].append(capital - INITIAL_CAPITAL)
        results["max_dd"].append(max_dd)
        results["max_streak"].append(max_streak)
        if monthly:
            results["monthly_rois"].extend(monthly)

    # ── 結果表示 ─────────────────────────────────────────────
    def pct(lst, p):
        lst_s = sorted(lst)
        return lst_s[int(len(lst_s) * p / 100)]

    fp = results["final_profit"]
    dd = results["max_dd"]
    ms = results["max_streak"]

    print(f"\n── {n_days}日後の損益（{n_sim}回） ──")
    print(f"  中央値:  {pct(fp,50):>+,}円")
    print(f"  10%ile:  {pct(fp,10):>+,}円  （悪いケース）")
    print(f"  90%ile:  {pct(fp,90):>+,}円  （良いケース）")
    print(f"  プラス:  {sum(1 for x in fp if x>0)/n_sim:.0%}")

    print(f"\n── 最大ドローダウン ──")
    print(f"  中央値:  -{pct(dd,50):,}円")
    print(f"  90%ile:  -{pct(dd,90):,}円  （最悪ケースの10%）")

    print(f"\n── 最大連敗 ──")
    print(f"  中央値:  {pct(ms,50):.0f}連敗")
    print(f"  90%ile:  {pct(ms,90):.0f}連敗")

    if results["monthly_rois"]:
        mr = results["monthly_rois"]
        print(f"\n── 月次ROI分布 ──")
        print(f"  中央値:  {pct(mr,50):.3f}")
        print(f"  10%ile:  {pct(mr,10):.3f}  （悪い月の10%）")
        print(f"  90%ile:  {pct(mr,90):.3f}  （良い月の10%）")
        print(f"  ROI>1.0の月: {sum(1 for x in mr if x>1.0)/len(mr):.0%}")

    bankrupt_rate = results["bankrupt"] / n_sim
    flag = "🔴 要注意" if bankrupt_rate > 0.1 else ("🟡 注意" if bankrupt_rate > 0.05 else "🟢 安全")
    print(f"\n── 破産率（資金-50%以上） ──")
    print(f"  {bankrupt_rate:.1%}  {flag}")

    # ── ⑤ tail risk（CVaR / Expected Shortfall）────────────────
    # 最悪の10%シナリオの平均損失
    fp_sorted = sorted(results["final_profit"])
    n_tail    = max(1, int(n_sim * 0.10))
    cvar_10   = sum(fp_sorted[:n_tail]) / n_tail
    n_tail5   = max(1, int(n_sim * 0.05))
    cvar_5    = sum(fp_sorted[:n_tail5]) / n_tail5
    cvar_flag = "🔴 危険" if cvar_10 < -30000 else ("🟡 注意" if cvar_10 < -15000 else "🟢 許容")
    print(f"\n── Tail Risk（CVaR）──")
    print(f"  CVaR(10%):  {cvar_10:>+,}円  ← 最悪10%の平均損益 {cvar_flag}")
    print(f"  CVaR( 5%):  {cvar_5:>+,}円  ← 最悪5%の平均損益")
    print(f"  ※ これが許容できる損失か確認してください")

    print(f"\n{'='*55}")


def _run_stats_analysis(csv_file: str = "hit_record.csv") -> None:
    """
    hit_record.csv から統計分析を表示する。
    - 月次回収率
    - 最大連敗 / ドローダウン
    - レースタイプ別成績
    - 勝ちパターン（why_bet）
    """
    import csv as _csv
    from collections import defaultdict, Counter

    if not os.path.exists(csv_file):
        print("hit_record.csv が見つかりません")
        return

    with open(csv_file, "r", encoding="utf-8") as f:
        rows = list(_csv.DictReader(f))

    valid = [r for r in rows if r.get("hit") not in ("", None, "-1")]
    if len(valid) < 5:
        print(f"データ不足: {len(valid)}件")
        return

    print(f"\n{'='*55}")
    print(f"  統計分析  ({len(valid)}件)")
    print(f"{'='*55}")

    # ── 月次回収率 ───────────────────────────────────────────
    monthly: dict[str, dict] = defaultdict(lambda: {"bet":0,"ret":0,"n":0,"hit":0})
    for r in valid:
        m = str(r.get("date",""))[:6]
        payout  = int(r.get("payout", 0) or 0)
        hit     = int(r.get("hit", 0) or 0)
        monthly[m]["bet"] += 100
        monthly[m]["ret"] += payout if hit else 0
        monthly[m]["n"]   += 1
        monthly[m]["hit"] += hit

    print("\n── 月次回収率 ──")
    print(f"  {'月':<8} {'ROI':>6}  {'的中':>6}  {'件数':>5}")
    for m in sorted(monthly):
        v   = monthly[m]
        roi = v["ret"] / v["bet"] if v["bet"] > 0 else 0
        flag = "🟢" if roi >= 1.0 else ("🟡" if roi >= 0.7 else "🔴")
        print(f"  {m}  {roi:>5.2f}  {v['hit']:>3}/{v['n']:<3}  {flag}")

    # ── 最大連敗 / ドローダウン ──────────────────────────────
    profits = [int(r.get("profit", -100) or -100) for r in valid]
    max_streak = cur_streak = 0
    peak = cum = 0
    max_dd = 0
    for p in profits:
        if p < 0:
            cur_streak += 1
            max_streak = max(max_streak, cur_streak)
        else:
            cur_streak = 0
        cum += p
        peak = max(peak, cum)
        max_dd = max(max_dd, peak - cum)

    total_bet = len(valid) * 100
    total_ret = sum(int(r.get("payout",0) or 0) for r in valid if int(r.get("hit",0) or 0))
    roi_total = total_ret / total_bet if total_bet > 0 else 0
    total_profit = total_ret - total_bet

    print(f"\n── 総合成績 ──")
    print(f"  総ROI:    {roi_total:.3f}")
    print(f"  総利益:   {total_profit:+,}円")
    print(f"  最大連敗: {max_streak}連敗")
    print(f"  最大DD:   -{max_dd:,}円")

    # ── Sharpe的評価（利益の安定性）────────────────────────────
    import statistics as _stat
    if len(profits) >= 5:
        avg_p  = _stat.mean(profits)
        std_p  = _stat.stdev(profits) if len(profits) > 1 else 1
        sharpe = avg_p / std_p if std_p > 0 else 0
        # 月次ごとに集計して月次シャープも計算
        monthly_p: dict[str, list] = {}
        for r in valid:
            m = str(r.get("date",""))[:6]
            monthly_p.setdefault(m, []).append(int(r.get("profit",-100) or -100))
        monthly_rois = []
        for m, ps in monthly_p.items():
            bet_ = len(ps) * 100
            ret_ = sum(p + 100 for p in ps if p > 0)
            monthly_rois.append(ret_ / bet_ if bet_ > 0 else 0)
        monthly_sharpe = 0.0
        if len(monthly_rois) >= 3:
            mr_mean = _stat.mean(monthly_rois)
            mr_std  = _stat.stdev(monthly_rois)
            monthly_sharpe = (mr_mean - 1.0) / mr_std if mr_std > 0 else 0

        sharpe_flag = "🟢 安定" if sharpe > 0.1 else ("🟡 不安定" if sharpe > -0.1 else "🔴 不安定")
        print(f"\n── Sharpe的評価 ──")
        print(f"  1ベット平均利益: {avg_p:+.0f}円  標準偏差: {std_p:.0f}円")
        print(f"  Sharpeレシオ:   {sharpe:+.3f}  {sharpe_flag}")
        if len(monthly_rois) >= 3:
            print(f"  月次Sharpe:     {monthly_sharpe:+.3f}  (月次ROI平均={_stat.mean(monthly_rois):.3f})")

    # ── レースタイプ別 ───────────────────────────────────────
    type_stats: dict[str, dict] = defaultdict(lambda: {"n":0,"hit":0,"profit":0})
    for r in valid:
        rt = r.get("race_type", "不明") or "不明"
        type_stats[rt]["n"]      += 1
        type_stats[rt]["hit"]    += int(r.get("hit", 0) or 0)
        type_stats[rt]["profit"] += int(r.get("profit", -100) or -100)

    print(f"\n── レースタイプ別成績 ──")
    for rt, v in sorted(type_stats.items(), key=lambda x: -x[1]["profit"]):
        hr  = v["hit"] / v["n"] if v["n"] > 0 else 0
        roi = (v["profit"] + v["n"]*100) / (v["n"]*100) if v["n"] > 0 else 0
        print(f"  {rt:<12} n={v['n']:>3}  的中率={hr:.1%}  ROI={roi:.2f}  利益{v['profit']:+,}円")

    # ── 勝ちパターン（why_bet） ──────────────────────────────
    hits   = [r for r in valid if int(r.get("hit",0) or 0) == 1]
    misses = [r for r in valid if int(r.get("hit",0) or 0) == 0]

    if hits:
        print(f"\n── 勝ちパターン ({len(hits)}件) ──")
        why_counter: Counter = Counter()
        for r in hits:
            for w in (r.get("why_bet","") or "").split("|"):
                if w.strip():
                    why_counter[w.strip()] += 1
        for reason, cnt in why_counter.most_common(8):
            pct = cnt / len(hits)
            print(f"  {reason:<20} {cnt}件 ({pct:.0%})")

    if misses and len(misses) >= 3:
        print(f"\n── 負けパターン ({len(misses)}件) ──")
        why_miss: Counter = Counter()
        for r in misses:
            for w in (r.get("why_bet","") or "").split("|"):
                if w.strip():
                    why_miss[w.strip()] += 1
        for reason, cnt in why_miss.most_common(5):
            pct = cnt / len(misses)
            print(f"  {reason:<20} {cnt}件 ({pct:.0%})")

    # ── 条件別回収率（場 / 風向き / ナイター / レースタイプ）──────
    print(f"\n── 条件別回収率 ──")

    def _cond_roi(rows_: list, key: str, label: str) -> None:
        vals = sorted(set(str(r.get(key,"") or "不明") for r in rows_))
        results_ = []
        for v in vals:
            seg = [r for r in rows_ if str(r.get(key,"") or "不明") == v]
            if len(seg) < 3:
                continue
            h_   = sum(int(r.get("hit",0) or 0) for r in seg)
            pay_ = sum(int(r.get("payout",0) or 0) for r in seg if int(r.get("hit",0) or 0))
            roi_ = pay_ / (len(seg)*100) if seg else 0
            results_.append((v, len(seg), h_, roi_, pay_ - len(seg)*100))
        results_.sort(key=lambda x: -x[3])
        if not results_:
            return
        print(f"  [{label}]")
        for v, n, h, roi_, prof_ in results_[:6]:
            flag = "🟢" if roi_ >= 1.0 else ("🟡" if roi_ >= 0.7 else "🔴")
            print(f"    {str(v):<12} n={n:>3} 的中{h:>2}  ROI={roi_:.2f}  {prof_:+,}円 {flag}")

    _cond_roi(valid, "venue",     "場別")
    _cond_roi(valid, "wind_dir",  "風向き別")
    _cond_roi(valid, "night",     "ナイター")
    _cond_roi(valid, "race_type", "レースタイプ別")

    # 1着予測艇別
    first_rows = []
    for r in valid:
        combo = r.get("pred_combo","")
        if combo and "-" in combo:
            first_rows.append({**r, "_first": combo.split("-")[0]})
    if first_rows:
        _cond_roi(first_rows, "_first", "1着予測艇別")

    print(f"\n{'='*55}")


def _run_calibration_check(csv_file: str = "hit_record.csv") -> None:
    """
    hit_record.csv から予測確率のキャリブレーションを確認する。
    予測prob群ごとの実測的中率を表示。
    """
    try:
        import csv
        if not os.path.exists(csv_file):
            return
        with open(csv_file, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        valid = [r for r in rows
                 if r.get("pred_prob") and r.get("hit") != "" and r.get("hit") is not None]
        if len(valid) < 10:
            return

        # prob帯別に集計
        bands = [
            (0.000, 0.020, "<2%"),
            (0.020, 0.030, "2-3%"),
            (0.030, 0.050, "3-5%"),
            (0.050, 0.080, "5-8%"),
            (0.080, 1.000, "8%+"),
        ]
        log.info("── キャリブレーション分析 (%d件) ──", len(valid))
        for lo, hi, label in bands:
            seg = [r for r in valid
                   if lo <= float(r["pred_prob"]) < hi]
            if not seg:
                continue
            actual_hit_rate = sum(int(r["hit"]) for r in seg) / len(seg)
            avg_pred        = sum(float(r["pred_prob"]) for r in seg) / len(seg)
            total_profit    = sum(int(r.get("profit", 0)) for r in seg)
            roi = (sum(int(r.get("payout",0)) for r in seg) /
                   (len(seg) * 100)) if seg else 0
            gap = actual_hit_rate - avg_pred
            flag = "⚠️ 過大予測" if gap < -0.01 else ("✅ 良好" if abs(gap) < 0.005 else "📈 過小予測")
            log.info("  prob%s: n=%d 予測%.1f%% 実測%.1f%% ROI=%.2f 利益%+d円 %s",
                     label, len(seg),
                     avg_pred*100, actual_hit_rate*100,
                     roi, total_profit, flag)

        # キャリブレーション補正提案
        overfit = [l for lo,hi,l in bands
                   if any(abs(float(r["pred_prob"])-(lo+hi)/2) < (hi-lo)/2
                          for r in valid
                          if lo <= float(r["pred_prob"]) < hi)]
        calib_rows = []
        for lo, hi, label in bands:
            seg = [r for r in valid if lo <= float(r["pred_prob"]) < hi]
            if len(seg) < 3: continue
            avg_pred = sum(float(r["pred_prob"]) for r in seg) / len(seg)
            actual   = sum(int(r["hit"] or 0) for r in seg) / len(seg)
            ratio    = actual / avg_pred if avg_pred > 0 else 1.0
            calib_rows.append((label, len(seg), avg_pred, actual, ratio))

        log.info("── キャリブレーション補正テーブル ──")
        log.info("  %s  %s  %s  %s  %s", "帯".ljust(6), "n".rjust(4),
                 "予測".rjust(7), "実測".rjust(7), "補正率".rjust(7))
        for label, n, pred, actual, ratio in calib_rows:
            flag = "⚠️ 過大" if ratio < 0.7 else ("📈 過小" if ratio > 1.3 else "✅")
            log.info("  %s  %s  %s  %s  %s  %s",
                     label.ljust(6), str(n).rjust(4),
                     f"{pred:.3f}".rjust(7), f"{actual:.3f}".rjust(7),
                     f"×{ratio:.2f}".rjust(7), flag)
        if hits:
            log.info("── 勝ちパターン (%d件) ──", len(hits))
            from collections import Counter
            wind_cnt  = Counter(r.get("wind_dir","不明") for r in hits)
            venue_cnt = Counter(r.get("venue","不明")    for r in hits)
            night_cnt = Counter(r.get("night","0")       for r in hits)
            log.info("  風向き: %s", dict(wind_cnt.most_common(3)))
            log.info("  場: %s",    dict(venue_cnt.most_common(5)))
            log.info("  ナイター: %s", dict(night_cnt))

    except Exception as e:
        log.debug("キャリブレーション分析失敗: %s", e)


# ── オッズキャッシュ（急落監視用）────────────────────────────
_odds_cache: dict[str, dict[str, float]] = {}   # key: "venue_race", value: odds_map


def _check_odds_drop(
    venue_num: int, race_number: int,
    new_odds: dict[str, float],
    drop_threshold: float = 0.15,
) -> list[str]:
    """
    前回オッズと比較して15%以上急落した組み合わせを返す。
    急落 = 市場が急に注目 = 情報あり = 信頼度UP。
    """
    key = f"{venue_num}_{race_number}"
    prev = _odds_cache.get(key, {})
    dropped = []
    for combo, new_o in new_odds.items():
        old_o = prev.get(combo, 0)
        if old_o > 0 and new_o < old_o * (1 - drop_threshold):
            drop_rate = (old_o - new_o) / old_o
            dropped.append(f"{combo}({old_o:.0f}→{new_o:.0f} -{drop_rate:.0%})")
    _odds_cache[key] = new_odds.copy()
    return dropped


def _check_yesterday_results(today_date: str) -> None:
    """前日の送信済みレースと結果を照合してCSVに記録"""
    try:
        from datetime import datetime, timedelta
        today = datetime.strptime(today_date, "%Y%m%d")
        yesterday = (today - timedelta(days=1)).strftime("%Y%m%d")
        sent_file = f"sent_{yesterday}.txt"

        if not os.path.exists(sent_file):
            return

        import json as _json
        sent_entries = []
        with open(sent_file, "r", encoding="utf-8") as f:
            for _line in f:
                _line = _line.strip()
                if not _line:
                    continue
                try:
                    _obj = _json.loads(_line)
                    sent_entries.append(_obj)
                except Exception:
                    # 旧形式（キーのみ）
                    sent_entries.append({"key": _line})

        if not sent_entries:
            return

        log.info("前日結果照合: %d レース", len(sent_entries))
        records = []
        for entry in sent_entries:
            key = entry.get("key", "")
            parts = key.replace("_ex", "").split("_")
            if len(parts) != 3:
                continue
            _, venue_num, race_number = parts
            result = _fetch_race_result(int(venue_num), int(race_number), yesterday)

            # sent_fileのJSONから直接読み込み（pred_*.jsonは不要）
            pd_data: dict = entry

            pred_combo  = pd_data.get("combo", "")        # ベスト1点
            buy_list    = pd_data.get("buy", [])          # 全買い目リスト
            buy_amounts = pd_data.get("buy_amounts", [])  # 各買い目の実ベット額
            pred_prob   = pd_data.get("prob", "")
            pred_ev     = pd_data.get("ev", "")
            pred_odds   = pd_data.get("odds", 0)
            n_bets      = max(1, len(buy_list)) if buy_list else 1

            result_combo = result["combo"] if result else "不明"
            payout       = result["payout"] if result else 0  # 100円あたりの払戻

            # 全買い目のどれかが当たれば的中（見送りでも「当たっていたか」は分析用に判定する）
            all_combos = buy_list if buy_list else ([pred_combo] if pred_combo else [])
            hit        = 1 if result_combo != "不明" and result_combo in all_combos else 0

            # 【購入判定分離】見送り(purchased=False)の場合は実際に投資していないため
            # cost・profit を必ず0にする。的中判定(hit)自体は「見送りが正しかったか」の
            # 分析用に残すが、回収率・ROI集計には含めない（cost=0で自動的に除外される）。
            is_purchased = pd_data.get("purchased", True)  # 旧データ互換のためデフォルトTrue

            if not is_purchased:
                cost   = 0
                profit = 0
            elif buy_amounts and len(buy_amounts) == len(buy_list):
                total_cost = sum(int(a) for a in buy_amounts)
                # 当たった買い目の金額を特定
                hit_amount = 0
                if hit:
                    for _c, _a in zip(buy_list, buy_amounts):
                        if _c == result_combo:
                            hit_amount = int(_a)
                            break
                # 払戻 = (払戻倍率/100) × 当たり金額
                hit_return = int(payout * hit_amount / 100) if hit else 0
                cost   = total_cost
                profit = (hit_return - total_cost) if hit else -total_cost
            else:
                total_cost = n_bets * 100
                hit_return = payout if hit else 0
                cost   = total_cost
                profit = (hit_return - total_cost) if hit else -total_cost

            # pred_fileに結果を書き戻す（ファイルが存在する場合のみ）
            pred_file = f"pred_{yesterday}_{venue_num}_{race_number}.json"
            if pd_data and os.path.exists(pred_file):
                try:
                    pd_data.update({
                        "result":  result_combo,
                        "hit":     hit,
                        "profit":  profit,
                        "payout":  payout,
                    })
                    with open(pred_file, "w", encoding="utf-8") as pf:
                        _json.dump(pd_data, pf, ensure_ascii=False)
                except Exception:
                    pass

            records.append({
                "date":        yesterday,
                "venue":       pd_data.get("venue", VENUE_NAMES.get(int(venue_num), f"場{venue_num}")),
                "venue_num":   venue_num,
                "race":        race_number,
                "night":       pd_data.get("night", 0),
                "race_type":   pd_data.get("race_type", ""),
                "why_bet":     "|".join(pd_data.get("why_bet", [])),
                "confidence":  pd_data.get("confidence", ""),
                "pred_combo":  pred_combo,
                "pred_prob":   pred_prob,
                "pred_ev":     pred_ev,
                "pred_odds":   pred_odds,
                "upset_score": pd_data.get("upset_score", ""),
                # 【教師データ】展示・気象は予想には未使用、Phase2学習用にのみ保存
                "wind_speed":  pd_data.get("wind_speed", ""),
                "wind_dir":    pd_data.get("wind_dir", ""),
                "wave":        pd_data.get("wave", ""),
                "result_combo": result_combo,
                "payout":      payout,
                "hit":         hit,
                "profit":      profit,
                "n_bets":      n_bets,
                "cost":        cost,
                # 【購入判定分離】BuyScoreによる購入判定の記録
                "purchased":   int(is_purchased),
                "buyscore":    pd_data.get("buyscore", 0),
                "match_index": pd_data.get("match_index", 0),
                "skip_reason": pd_data.get("skip_reason", ""),
                # 【朝刊AI】使用モデルVersion・特徴量（Phase2重み学習用）
                "model_version":         pd_data.get("model_version", ""),
                "feat_win_rate":         pd_data.get("feat_win_rate", ""),
                "feat_motor":            pd_data.get("feat_motor", ""),
                "feat_avg_st":           pd_data.get("feat_avg_st", ""),
                "feat_racer_class":      pd_data.get("feat_racer_class", ""),
                "feat_course_st_1c":     pd_data.get("feat_course_st_1c", ""),
                "feat_course_rank_1c":   pd_data.get("feat_course_rank_1c", ""),
                "feat_danger_breakdown": pd_data.get("feat_danger_breakdown", ""),
            })

        if not records:
            return

        import csv
        csv_file = "hit_record.csv"
        fieldnames = [
            "date","venue","venue_num","race","night",
            "race_type","why_bet","confidence",
            "pred_combo","pred_prob","pred_ev","pred_odds","upset_score",
            "wind_speed","wind_dir","wave",
            "result_combo","payout","hit","profit","n_bets","cost",
            "purchased","buyscore","match_index","skip_reason",
            "model_version",
            "feat_win_rate","feat_motor","feat_avg_st","feat_racer_class",
            "feat_course_st_1c","feat_course_rank_1c","feat_danger_breakdown",
        ]

        # ── 重複書き込み防止 ──────────────────────────────────────
        # _check_yesterday_results() は run() から15分ごとに呼ばれるため、
        # 重複チェックなしで追記すると同一レコードが際限なく増殖する。
        # date + venue_num + race + pred_combo を一意キーとして、
        # CSVに既に存在する行はスキップする。
        existing_keys: set[tuple] = set()
        if os.path.exists(csv_file):
            try:
                with open(csv_file, "r", encoding="utf-8") as _ef:
                    for _row in csv.DictReader(_ef):
                        existing_keys.add((
                            _row.get("date",""),
                            str(_row.get("venue_num","")),
                            str(_row.get("race","")),
                            _row.get("pred_combo",""),
                        ))
            except Exception:
                pass

        new_records = [
            r for r in records
            if (
                str(r.get("date","")),
                str(r.get("venue_num","")),
                str(r.get("race","")),
                str(r.get("pred_combo","")),
            ) not in existing_keys
        ]

        if not new_records:
            log.debug("[hit_record] 新規レコードなし（全件既記録済み）")
            return

        write_header = not os.path.exists(csv_file)
        with open(csv_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(new_records)
        log.info("[hit_record] %d件追記（%d件はスキップ）", len(new_records), len(records) - len(new_records))

        # ── キャリブレーション簡易分析 ───────────────────────────
        _run_calibration_check(csv_file)

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
        # today.jsonにフォールバック（当日分のみ）
        from datetime import datetime, timezone, timedelta
        today_str = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d")
        if race_date == today_str:
            data = _safe_get(f"{RESULTS_URL}/today.json")
    if not data:
        return None

    for r in data.get("results", []):
        if (r.get("race_stadium_number") == venue_num
                and r.get("race_number") == race_number):
            # 3連単結果・払戻（payouts.trifecta から直接取得）
            combo  = "不明"
            payout = 0
            payouts = r.get("payouts", {})
            if isinstance(payouts, dict):
                trifecta = payouts.get("trifecta", [])
                if isinstance(trifecta, list) and trifecta:
                    try:
                        combo  = trifecta[0].get("combination", "不明")
                        payout = int(trifecta[0].get("payout", 0))
                    except (ValueError, TypeError):
                        pass

            # trifecta取得できなかった場合はboatsからフォールバック
            if combo == "不明":
                boats = r.get("boats", [])
                if isinstance(boats, dict):
                    boats = list(boats.values())
                order = sorted(
                    [b for b in boats if isinstance(b, dict) and b.get("racer_place_number")],
                    key=lambda b: b.get("racer_place_number", 99)
                )
                if len(order) >= 3:
                    combo = "-".join(str(b.get("racer_boat_number", "?")) for b in order[:3])

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
            # send_notification("【連敗アラート】", msg)  # 通知停止（転がし専用に移行）
            log.warning("連敗アラート（通知なし）: %d連敗 ※ログのみ", miss_count)

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
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="回収率ダッシュボード（場別・風向別・実測EV・捨て条件・外れ分析）",
    )
    parser.add_argument(
        "--patterns",
        action="store_true",
        help="勝ちパターン自動抽出（ログから条件×ROIを分析）",
    )
    parser.add_argument(
        "--monte-carlo",
        action="store_true",
        help="Monte Carloシミュレーション（長期資金リスク評価）",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="hit_record.csv のドローダウン・月次回収率・勝ちパターン分析を表示",
    )
    parser.add_argument(
        "--calibration",
        action="store_true",
        help="hit_record.csv のキャリブレーション分析を表示して終了",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="15分ごとに繰り返し実行（ローカル常駐モード）",
    )
    args = parser.parse_args()

    UPSET_SCORE_THRESHOLD = args.threshold

    if args.dashboard:
        _print_dashboard("hit_record.csv")
        import sys; sys.exit(0)

    if args.patterns:
        _auto_extract_patterns("hit_record.csv")
        import sys; sys.exit(0)

    if getattr(args, "monte_carlo", False):
        _monte_carlo_simulation("hit_record.csv")
        import sys; sys.exit(0)

    if args.stats:
        _run_stats_analysis("hit_record.csv")
        import sys; sys.exit(0)

    if args.calibration:
        _run_calibration_check("hit_record.csv")
        import sys; sys.exit(0)

    if args.loop:
        # ── ローカル常駐モード ────────────────────────────────
        import time as _time
        log.info("常駐モード起動（15分ごとに実行・Ctrl+Cで停止）")
        while True:
            try:
                run(race_date=args.date)
            except Exception as e:
                log.error("実行エラー: %s", e)
            _time.sleep(15 * 60)   # 15分待機
    else:
        run(race_date=args.date)
