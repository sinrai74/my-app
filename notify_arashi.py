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
UPSET_SCORE_THRESHOLD = 1.5   # チューニング可能

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
    for pb in preview.get("boats", []):
        if not isinstance(pb, dict):
            continue
        lane = pb.get("racer_boat_number")
        boat = next((b for b in boats if b.lane == lane), None)
        if boat is None:
            continue
        if pb.get("racer_exhibition_time") is not None:
            boat.ex_time = float(pb["racer_exhibition_time"])
        if pb.get("racer_exhibition_start_timing") is not None:
            boat.ex_st = float(pb["racer_exhibition_start_timing"])
        if pb.get("racer_tilt") is not None:
            boat.tilt = float(pb["racer_tilt"])

    # ── 気象情報 ─────────────────────────────────────────────
    wd_raw = preview.get("wind_direction")
    wd_str: Optional[str] = None
    if wd_raw is not None:
        raw = str(wd_raw)
        if "追" in raw:
            wd_str = "追"
        elif "向" in raw:
            wd_str = "向"
        elif "横" in raw or "斜" in raw:
            wd_str = "横"
        else:
            wd_str = raw

    ws  = preview.get("wind_speed")
    wh  = preview.get("wave_height")
    wdc = preview.get("weather_condition")

    return WeatherInfo(
        wind_speed     = float(ws)  if ws  is not None else None,
        wind_direction = wd_str,
        wave_height    = int(wh)    if wh  is not None else None,
        weather        = str(wdc)   if wdc is not None else None,
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

    # ── デバッグ: 最初の1レースのJSONキー構造をそのまま出力 ──────
    if programs:
        p0 = programs[0]
        log.info("[DEBUG] programs[0] のトップレベルキー: %s", list(p0.keys()))
        boats0 = p0.get("boats", [])
        if boats0 and isinstance(boats0[0], dict):
            log.info("[DEBUG] programs[0].boats[0] のキー: %s", list(boats0[0].keys()))
            log.info("[DEBUG] programs[0].boats[0] の値: %s", boats0[0])
    if previews:
        v0 = previews[0]
        log.info("[DEBUG] previews[0] のトップレベルキー: %s", list(v0.keys()))
        pboats0 = v0.get("boats", [])
        if pboats0 and isinstance(pboats0[0], dict):
            log.info("[DEBUG] previews[0].boats[0] のキー: %s", list(pboats0[0].keys()))
            log.info("[DEBUG] previews[0].boats[0] の値: %s", pboats0[0])
        log.info("[DEBUG] previews[0] の気象系キー: wind_speed=%s, wind_direction=%s, wave_height=%s, weather_condition=%s",
                 v0.get("wind_speed"), v0.get("wind_direction"),
                 v0.get("wave_height"), v0.get("weather_condition"))
    # ── デバッグここまで ─────────────────────────────────────────

    results: list[tuple[int, int, list[BoatInfo], WeatherInfo]] = []
    for prog in programs:
        vn  = prog.get("race_stadium_number")
        rno = prog.get("race_number")
        if vn is None or rno is None:
            continue

        boats   = _extract_boats_from_program(prog)
        preview = preview_map.get((vn, rno), {})
        weather = _apply_preview_to_boats(boats, preview)
        results.append((int(vn), int(rno), boats, weather))

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


def calculate_upset_score(
    boats:   list[BoatInfo],
    weather: WeatherInfo,
) -> tuple[float, dict[str, str], list[int]]:
    """
    荒れスコアを計算して (score, detail_dict, target_lanes) を返す。

    target_lanes: 狙える艇番のリスト（スコアが高い上位 3 艇）
    """
    boat1 = next((b for b in boats if b.lane == 1), None)
    if boat1 is None:
        return 0.0, {"error": "1号艇データなし"}, []

    detail: dict[str, str] = {}
    total_score = 0.0

    # ── ルールベーススコアリング ──────────────────────────────
    s, d = _score_exhibition_time(boat1, boats)
    total_score += s
    detail["展示タイム"] = d

    s, d = _score_exhibition_st(boat1, boats)
    total_score += s
    detail["展示ST"] = d

    s, d = _score_weather(weather)
    total_score += s
    detail["気象"] = d

    s, d = _score_performance(boat1, boats)
    total_score += s
    detail["選手スペック"] = d

    # ── ML モデルスコアの上乗せ（将来実装）──────────────────
    ml_score = score_by_model(boats, weather)
    if ml_score is not None:
        total_score = total_score * 0.6 + ml_score * 0.4
        detail["MLスコア"] = f"{ml_score:.2f}"

    # ── 狙い艇を選定（1号艇を除く上位スコア艇）──────────────
    target_lanes = _select_target_lanes(boat1, boats)

    return round(total_score, 2), detail, target_lanes


def _select_target_lanes(boat1: BoatInfo, all_boats: list[BoatInfo]) -> list[int]:
    """
    1号艇以外で最も有望な艇を選ぶ。
    展示タイム・ST・勝率の複合スコアで上位3艇を返す。
    """
    others = [b for b in all_boats if b.lane != 1]
    if not others:
        return []

    # 各艇を簡易スコアリング（高いほど良い）
    scored: list[tuple[int, float]] = []
    for b in others:
        s = 0.0
        s += b.win_rate * 10             # 全国勝率
        s += b.motor   * 5               # モーター2連率
        if b.ex_time is not None:
            # 展示タイムは速い（小さい）ほど高スコア
            s += max(0, (7.00 - b.ex_time) * 20)
        if b.ex_st is not None:
            # STは小さい（速い）ほど高スコア
            s += max(0, (0.20 - b.ex_st) * 50)
        scored.append((b.lane, s))

    scored.sort(key=lambda x: -x[1])
    return [lane for lane, _ in scored[:3]]


# ════════════════════════════════════════════════════════════
# 通知メッセージ生成
# ════════════════════════════════════════════════════════════

def _danger_label(score: float) -> str:
    if score >= 7.0:
        return "🔴 非常に高"
    if score >= 5.0:
        return "🟠 高"
    if score >= 3.0:
        return "🟡 中"
    return "🟢 低"


def build_message(result: RaceResult) -> tuple[str, str]:
    """
    メール件名と本文を生成して (subject, body) のタプルで返す。
    """
    boat1 = next((b for b in result.boats if b.lane == 1), None)
    w     = result.weather

    # ── 件名 ─────────────────────────────────────────────────
    subject = (
        f"【荒れ検知】{result.venue_name} {result.race_number}R "
        f"1号艇危険度: {_danger_label(result.upset_score)}"
    )

    # ── 本文 ─────────────────────────────────────────────────
    lines = [
        f"【荒れ検知】{result.venue_name} {result.race_number}R",
        "",
    ]

    # 気象情報
    weather_parts: list[str] = []
    if w.wind_speed     is not None: weather_parts.append(f"風: {w.wind_speed:.1f}m {w.wind_direction or ''}")
    if w.wave_height    is not None: weather_parts.append(f"波: {w.wave_height}cm")
    if w.weather        is not None: weather_parts.append(f"天候: {w.weather}")
    if weather_parts:
        lines.append("🌊 " + " / ".join(weather_parts))

    # 1号艇の展示情報
    if boat1 is not None:
        et_str = f"{boat1.ex_time:.2f}" if boat1.ex_time is not None else "—"
        st_str = f"{boat1.ex_st:.2f}"   if boat1.ex_st   is not None else "—"
        lines.append(f"🚤 1号艇 {boat1.name}: 展示 {et_str}秒 / ST {st_str}")

    # 危険度
    lines.append(f"⚠️  1号艇危険度: {_danger_label(result.upset_score)} (スコア: {result.upset_score})")

    # 狙い目
    if result.target_lanes:
        tgt = "-".join(str(l) for l in result.target_lanes)
        lines.append(
            f"🎯 狙い: {result.target_lanes[0]}-{tgt[2:]}-全"
            if len(result.target_lanes) >= 2
            else f"🎯 狙い: {result.target_lanes[0]}軸"
        )

    # スコア詳細
    lines += ["", "── スコア詳細 ──"]
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
    log.info("処理対象: %d レース", len(race_list))

    # ── 荒れ判定 & 通知 ──────────────────────────────────────
    notified = 0
    for venue_num, race_number, boats, weather in race_list:
        try:
            score, detail, target = calculate_upset_score(boats, weather)

            if score < UPSET_SCORE_THRESHOLD:
                # デバッグ: 最初の場の1〜3Rはスコア詳細をINFOで出力
                if race_number <= 3 and venue_num == race_list[0][0]:
                    log.info(
                        "[DEBUG] スコア詳細 %s %dR score=%.2f | %s",
                        VENUE_NAMES.get(venue_num, f"場{venue_num}"),
                        race_number, score,
                        " | ".join(f"{k}:{v}" for k, v in detail.items()),
                    )
                else:
                    log.debug(
                        "スコア不足スキップ: %s %dR score=%.2f",
                        VENUE_NAMES.get(venue_num, f"場{venue_num}"),
                        race_number, score,
                    )
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
