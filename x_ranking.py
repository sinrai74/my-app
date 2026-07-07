#!/usr/bin/env python3
"""
x_ranking.py  ── 競艇 X 自動投稿用ランキング生成

Usage:
    python x_ranking.py --generate             # 今日のランキング生成
    python x_ranking.py --generate --date 20260627
    python x_ranking.py --update-history       # 前日のモーター履歴を蓄積
    python x_ranking.py --update-history --date 20260626
    python x_ranking.py --generate --output today_tweets.txt
    python x_ranking.py --generate --split     # 4ファイルに分けて出力

notify_arashi.py の fetch_programs / fetch_previews / _extract_boats_from_program /
VENUE_NAMES / _safe_get / RESULTS_URL を再利用する。
【朝刊AI】previews由来（展示・気象）データはスコア計算に使用せず、
x_asahi_scoring.py の共通エンジンで朝データのみから評価する。
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import statistics
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

# ── notify_arashi.py から必要な関数・定数を再利用 ─────────────────────────
from notify_arashi import (
    VENUE_NAMES,
    RESULTS_URL,
    _safe_get,
    fetch_programs,
    fetch_previews,
    _extract_boats_from_program,
    _extract_preview_raw,
    _PREVIEW_RAW_CACHE,
    BoatInfo,
    WeatherInfo,
)

# 【朝刊AI】共通スコアリングエンジン（previews由来データ不使用）
from x_asahi_scoring import (
    calc_danger_score_v2,
    calculate_upset_score_v2,
    calc_rank_index_v2,
    select_featured_boats,
    load_asahi_config,
    get_model_version as _asahi_model_version,
)
from x_venue_stats import classify_water_type

import x_release_storage

# ════════════════════════════════════════════════════════════
# 定数・設定
# ════════════════════════════════════════════════════════════

JST = timezone(timedelta(hours=9))
log = logging.getLogger("x_ranking")

MOTOR_HISTORY = "motor_history.csv"
RANKING_CACHE = "ranking_cache.json"

MOTOR_HISTORY_FIELDS = [
    "date", "venue_num", "venue", "motor_no", "racer_no",
    "racer_name", "lane", "place", "ex_time", "start_timing", "race_number",
]

DIVIDER = "=" * 50


# ════════════════════════════════════════════════════════════
# ユーティリティ
# ════════════════════════════════════════════════════════════

def _today_jst() -> str:
    return datetime.now(JST).strftime("%Y%m%d")


def _yesterday_jst() -> str:
    return (datetime.now(JST) - timedelta(days=1)).strftime("%Y%m%d")


def _boat_motor_no(boat_raw: dict) -> Optional[int]:
    """出走表の boat 辞書からモーター番号を取得する"""
    val = boat_raw.get("motor_number") or boat_raw.get("racer_motor_number")
    try:
        return int(val) if val is not None else None
    except (ValueError, TypeError):
        return None


def _boat_racer_no(boat_raw: dict) -> str:
    return str(
        boat_raw.get("racer_number")
        or boat_raw.get("racer_registration_number")
        or ""
    )


def _is_valid_ex(val) -> bool:
    try:
        return float(val) > 0
    except (ValueError, TypeError):
        return False


# ════════════════════════════════════════════════════════════
# モーター履歴管理
# ════════════════════════════════════════════════════════════

def load_motor_history() -> dict[tuple[int, int], list[dict]]:
    """motor_history.csv を読み込んで {(venue_num, motor_no): [rows]} を返す"""
    x_release_storage.download_file(MOTOR_HISTORY, MOTOR_HISTORY)
    history: dict[tuple[int, int], list[dict]] = {}
    if not os.path.exists(MOTOR_HISTORY):
        log.info("[履歴] %s が見つかりません（初回起動）", MOTOR_HISTORY)
        return history
    with open(MOTOR_HISTORY, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                key = (int(row["venue_num"]), int(row["motor_no"]))
            except (KeyError, ValueError):
                continue
            history.setdefault(key, []).append(row)
    log.info("[履歴] 読み込み: %d モーター", len(history))
    return history


def update_motor_history(race_date: str) -> int:
    """
    指定日の結果データからモーター履歴を motor_history.csv に追記する。
    追記件数を返す。
    """
    # 追記の前に、Releaseに保存されている最新版をローカルへ反映しておく
    # （GitHub Actionsのジョブ実行環境は毎回使い捨てのため、これがないと
    #  過去の蓄積データが失われ、追記のたびに空ファイルへの1回分だけになる）。
    x_release_storage.download_file(MOTOR_HISTORY, MOTOR_HISTORY)

    # 結果 API 取得
    url = f"{RESULTS_URL}/{race_date[:4]}/{race_date}.json"
    data = _safe_get(url)
    if not data:
        today_str = _today_jst()
        if race_date == today_str:
            data = _safe_get(f"{RESULTS_URL}/today.json")
    if not data:
        log.warning("[履歴] 結果取得失敗: %s", race_date)
        return 0

    # 出走表からモーター番号・選手番号を引く
    programs = fetch_programs(race_date)
    prog_map: dict[tuple[int, int], dict[int, dict]] = {}
    for p in programs:
        vn  = p.get("race_stadium_number")
        rno = p.get("race_number")
        if vn is None or rno is None:
            continue
        boat_dict: dict[int, dict] = {}
        for b in p.get("boats", []):
            if not isinstance(b, dict):
                continue
            bn = b.get("racer_boat_number")
            if bn is not None:
                boat_dict[int(bn)] = b
        prog_map[(int(vn), int(rno))] = boat_dict

    # 直前情報から展示タイムを引く
    previews = fetch_previews(race_date)
    prev_ex: dict[tuple[int, int], dict[int, Optional[float]]] = {}
    for pv in previews:
        vn  = pv.get("race_stadium_number")
        rno = pv.get("race_number")
        if vn is None or rno is None:
            continue
        ex_map: dict[int, Optional[float]] = {}
        boats_raw = pv.get("boats", {})
        pb_list = list(boats_raw.values()) if isinstance(boats_raw, dict) else boats_raw
        for pb in pb_list:
            if not isinstance(pb, dict):
                continue
            bn = pb.get("racer_boat_number")
            et = pb.get("racer_exhibition_time")
            if bn is not None:
                try:
                    ex_map[int(bn)] = float(et) if et and float(et) > 0 else None
                except (ValueError, TypeError):
                    ex_map[int(bn)] = None
        prev_ex[(int(vn), int(rno))] = ex_map

    # 結果データを走査してレコード構築
    rows: list[dict] = []
    for r in data.get("results", []):
        vn  = r.get("race_stadium_number")
        rno = r.get("race_number")
        if vn is None or rno is None:
            continue
        vn, rno = int(vn), int(rno)
        venue = VENUE_NAMES.get(vn, f"場{vn}")
        prog_boats = prog_map.get((vn, rno), {})
        ex_map     = prev_ex.get((vn, rno), {})

        for b in r.get("boats", []):
            if not isinstance(b, dict):
                continue
            boat_no = b.get("racer_boat_number")
            place   = b.get("racer_place_number")   # v2.8 修正済みフィールド名
            lane    = b.get("racer_course_number", boat_no)
            st      = b.get("racer_start_timing")

            if boat_no is None or place is None:
                continue
            boat_no = int(boat_no)

            prog_b     = prog_boats.get(boat_no, {})
            motor_no   = _boat_motor_no(prog_b)
            racer_name = b.get("racer_name") or prog_b.get("racer_name", "")
            racer_no   = _boat_racer_no(b) or _boat_racer_no(prog_b)
            ex_time    = ex_map.get(boat_no)

            if motor_no is None:
                continue  # モーター番号不明はスキップ

            rows.append({
                "date":         race_date,
                "venue_num":    vn,
                "venue":        venue,
                "motor_no":     motor_no,
                "racer_no":     racer_no,
                "racer_name":   racer_name,
                "lane":         int(lane) if lane is not None else "",
                "place":        int(place),
                "ex_time":      ex_time if ex_time is not None else "",
                "start_timing": float(st) if st is not None else "",
                "race_number":  rno,
            })

    if not rows:
        log.info("[履歴] 追記レコードなし: %s", race_date)
        return 0

    file_exists = os.path.exists(MOTOR_HISTORY)
    with open(MOTOR_HISTORY, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MOTOR_HISTORY_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    log.info("[履歴] 追記: %d件 (%s)", len(rows), race_date)

    if not x_release_storage.upload_file(MOTOR_HISTORY, MOTOR_HISTORY):
        log.warning("[履歴] %s のRelease反映に失敗しました（ローカルには保存済み）", MOTOR_HISTORY)

    return len(rows)


# ════════════════════════════════════════════════════════════
# ① 危険な1号艇ランキング
# ════════════════════════════════════════════════════════════

def _calc_danger_breakdown(
    boat1: Optional[BoatInfo],
    all_boats: list[BoatInfo],
    weather: Optional[WeatherInfo] = None,
    venue: Optional[str] = None,
) -> dict:
    """
    危険度スコアの内訳を辞書で返す（【朝刊AI】共通スコアリングエンジン使用）。
    x_asahi_scoring.calc_danger_score_v2 に委譲し、朝データのみで算出する。
    weather 引数は互換性のため残すが使用しない（previews由来のため）。
    venue（場名）を渡すと Ver4 の場補正（venue_unfavorable）が有効になる。
    note レポートのスコア内訳表示・AIコメント生成に使用。

    戻り値: calc_danger_score_v2 の breakdown（各項目 {"weighted":, "kind":,
            (relative項目のみ)"worse_count":, "worse_total":} のネスト辞書）に
            "total" キーを加えたもの。
    """
    if not boat1:
        return {"total": 0}

    total, raw_breakdown = calc_danger_score_v2(boat1, all_boats, venue=venue)
    result = {"total": round(total)}
    result.update(raw_breakdown)
    return result


def calc_danger_score(
    boat1: Optional[BoatInfo],
    all_boats: list[BoatInfo],
    weather: Optional[WeatherInfo] = None,
) -> int:
    """1号艇の危険度スコアを 0〜100 で返す（【朝刊AI】共通エンジン使用）"""
    return _calc_danger_breakdown(boat1, all_boats, weather)["total"]


def _score_to_rank(score: int) -> str:
    """スコアをS/A/Bランクに変換"""
    if score >= 80: return "🔴 S 危険"
    if score >= 60: return "🟠 A 注意"
    return "🟡 B やや危険"


def _score_to_rank_short(score: int) -> str:
    """画像・テキスト用の短いランク表示"""
    if score >= 80: return "🔴S"
    if score >= 60: return "🟠A"
    return "🟡B"


_DANGER_REASON_LABELS = {
    "win_rate":       "全国勝率",
    "local_win":      "当地勝率",
    "avg_st":         "平均ST",
    "motor":          "モーター",
    "racer_class":    "級別",
    "ability_trend":  "能力指数推移",
    "course_rentai":  "進入コース2連対率",
    "win_rate_low":       "全国勝率(絶対的に低調)",
    "local_win_low":      "当地勝率(絶対的に低調)",
    "motor_bad":          "モーター(絶対的に低調)",
    "avg_st_slow":        "平均ST(絶対的に遅い)",
    "course1_place_low":  "1コース1着率(前期実績)",
    "f_risk":             "1コースFリスク",
    "venue_unfavorable":  "当地水面(場×コース実績)",
}

# solo（単体評価）項目のうち、汎用文言「〜が基準未満」より具体的な
# 説明の方が分かりやすいものは個別に文言を用意する。
_SOLO_REASON_TEMPLATES = {
    "f_risk":            "1コースでのフライング率が高い",
    "venue_unfavorable": "当地の1コース実績が全国平均より低調",
}


def _danger_reason(boat1: Optional[BoatInfo], all_boats: list[BoatInfo], breakdown: Optional[dict] = None) -> str:
    """
    危険理由を数値で説明する文字列を生成する（calc_danger_score_v2 の
    breakdown＝「1号艇より優れている艇数」をそのまま文章化する）。
    例: "全国勝率で4艇に劣る（5艇中5位）" / "モーターで3艇に劣る（5艇中4位）"
    """
    if not boat1:
        return "不明"
    breakdown = breakdown or {}
    items = [(k, v) for k, v in breakdown.items() if isinstance(v, dict) and v.get("weighted", 0) > 0]
    # 加点が大きい順（＝危険度への寄与が大きい順）に並べる
    items.sort(key=lambda kv: -kv[1]["weighted"])

    reasons: list[str] = []
    for key, v in items:
        label = _DANGER_REASON_LABELS.get(key, key)
        if v.get("kind") == "relative":
            worse = v["worse_count"]
            total_other = v["worse_total"]
            rank = worse + 1  # 1号艇含めた順位（1号艇より優れている艇数+1位）
            reasons.append(f"{label}で{worse}艇に劣る（{total_other + 1}艇中{rank}位）")
        elif key in _SOLO_REASON_TEMPLATES:
            reasons.append(_SOLO_REASON_TEMPLATES[key])
        else:
            reasons.append(f"{label}が基準未満")

    return " / ".join(reasons) if reasons else "総合判定"


# ══════════════════════════════════════════════════════
# B. AI評価軸★を算出する関数
# ══════════════════════════════════════════════════════

def _calc_stars(boat1: Optional[BoatInfo], all_boats: list[BoatInfo]) -> dict[str, str]:
    """
    B. AI評価軸を★1〜5で返す
    {
      "ST":   "★★★★☆",
      "機力":  "★★☆☆☆",
      "近況":  "★★★☆☆",
      "相手":  "★★★★★",
    }
    ★が多い = 1号艇が危ない（飛びやすい）
    """
    def stars(n: int) -> str:
        n = max(0, min(5, n))
        return "★" * n + "☆" * (5 - n)

    if not boat1:
        return {"ST": stars(3), "機力": stars(3), "近況": stars(3), "相手": stars(3)}

    # ST危険度 (遅いほど★多、【朝刊AI】平均ST実績・コース別ST実績を使用)
    avg_st = boat1.avg_st or 0.17
    course_st_1c = (
        boat1.course_st[0]
        if boat1.course_st and boat1.course_nyuko and boat1.course_nyuko[0] > 0
        else avg_st
    )
    st_score = (
        5 if avg_st >= 0.20 else
        4 if avg_st >= 0.18 else
        3 if avg_st >= 0.17 else
        2 if avg_st >= 0.16 else 1
    )
    if course_st_1c >= 0.20: st_score = min(5, st_score + 1)

    # 機力（モーター2連率が低いほど★多）
    m2 = boat1.motor or 32.0
    motor_score = (
        5 if m2 <= 20 else
        4 if m2 <= 25 else
        3 if m2 <= 30 else
        2 if m2 <= 35 else 1
    )

    # 近況（勝率・等級が低いほど★多）
    wr = boat1.win_rate or 5.0
    cls = boat1.racer_class or "A2"
    grade_penalty = {"A1": 0, "A2": 1, "B1": 2, "B2": 3}.get(cls, 1)
    recent_score = (
        5 if wr < 4.0 else
        4 if wr < 4.5 else
        3 if wr < 5.0 else
        2 if wr < 5.5 else 1
    )
    recent_score = min(5, recent_score + grade_penalty)

    # 相手強度（強い相手が多いほど★多）
    others = [b for b in all_boats if b.lane != 1]
    a1_count = sum(1 for b in others if b.racer_class == "A1")
    best_rival_wr = max((b.win_rate or 0) for b in others) if others else 0
    rival_score = (
        5 if a1_count >= 3 else
        4 if a1_count >= 2 else
        3 if best_rival_wr > wr + 1.5 else
        2 if best_rival_wr > wr + 0.5 else 1
    )

    return {
        "ST":  stars(st_score),
        "機力": stars(motor_score),
        "近況": stars(recent_score),
        "相手": stars(rival_score),
    }


# ════════════════════════════════════════════════════════════
# ② 激走モーターランキング
# ════════════════════════════════════════════════════════════

def _place_score(place: int) -> int:
    return {1: 100, 2: 70, 3: 40, 4: 15, 5: 5, 6: 0}.get(int(place), 0)


def calc_hot_motor(
    venue_num: int,
    motor_no: int,
    motor_2rate: Optional[float],
    history: dict[tuple[int, int], list[dict]],
) -> Optional[int]:
    """
    激走モーター指数を 0〜100 で返す。データ不足（<10走）は None。

    【朝刊AI: データ区分の明記】
    ここで使う `history`（motor_history.csv）は、過去に開催済みの
    レースで既に確定した実績データ（着順・展示タイム等）である。
    未来情報を一切含まないため、当日朝の時点でも安定して参照でき、
    特徴量として使用してよい（過去実績データ）。

    一方、当日これから行われるレースの previews API 由来データ
    （当日の展示タイム・展示ST・展示順位・風速・風向・波高など、
    リアルタイムデータ）は、本関数はもちろん朝刊AI全体で一切使用しない。
    過去実績データと当日リアルタイムデータを混在させないこと。
    """
    rows = history.get((venue_num, motor_no), [])
    if len(rows) < 10:
        return None

    recent10 = rows[-10:]
    recent20 = rows[-20:] if len(rows) >= 20 else rows

    # 直近10走・20走の着順スコア平均
    r10 = sum(_place_score(r["place"]) for r in recent10) / len(recent10)
    r20 = sum(_place_score(r["place"]) for r in recent20) / len(recent20)

    # 【過去実績データ】展示タイム偏差（直近10走 vs 全履歴平均、いずれも確定済みの過去実績）
    ex10  = [float(r["ex_time"]) for r in recent10 if _is_valid_ex(r.get("ex_time"))]
    ex_all = [float(r["ex_time"]) for r in rows    if _is_valid_ex(r.get("ex_time"))]
    if ex10 and ex_all:
        avg_all    = sum(ex_all) / len(ex_all)
        avg_recent = sum(ex10)   / len(ex10)
        ex_dev = min(100, max(0, (avg_all - avg_recent) * 500 + 50))
    else:
        ex_dev = 50

    # 公式2連率との乖離
    r10_2rate = sum(1 for r in recent10 if int(r["place"]) <= 2) / len(recent10) * 100
    official  = motor_2rate or 32.0
    gap = min(100, max(0, (r10_2rate - official) * 2 + 50))

    total = r10 * 0.35 + r20 * 0.20 + ex_dev * 0.25 + gap * 0.20
    return round(min(100, max(0, total)))


# ════════════════════════════════════════════════════════════
# ③ 万舟警報ランキング
# ════════════════════════════════════════════════════════════

def calc_upset_index(
    boats: list[BoatInfo],
    weather: Optional[WeatherInfo] = None,
) -> int:
    """
    万舟警報指数を 0〜100 で返す（【朝刊AI】展示差・気象リスクを除外）。
    weather 引数は互換性のため残すが使用しない（previews由来のため）。
    展示差・気象リスク分(30%)を除外した分、朝データのみの4項目に
    比例配分して重みを再正規化する（元合計75% → 100%へ ×1.333）。
    """
    if len(boats) < 4:
        return 0

    boat1 = next((b for b in boats if b.lane == 1), None)
    if not boat1:
        return 0

    # イン信頼度逆転
    wr1 = boat1.win_rate or 5.0
    in_score = min(100, max(0, 100 - wr1 * 10))
    if boat1.racer_class in ("B1", "B2"):
        in_score = min(100, in_score + 20)

    # 実力接近度（6艇の勝率σ）
    win_rates = [b.win_rate or 4.5 for b in boats]
    sigma = statistics.stdev(win_rates) if len(win_rates) > 1 else 1.0
    if   sigma < 0.5: close_score = 100
    elif sigma < 0.8: close_score = 70
    elif sigma < 1.0: close_score = 40
    else:             close_score = 10

    # STばらつき（【朝刊AI】平均ST実績のみ使用）
    sts = [b.avg_st or 0.17 for b in boats]
    st_sigma = statistics.stdev(sts) if len(sts) > 1 else 0.02
    if   st_sigma >= 0.04: st_var = 100
    elif st_sigma >= 0.03: st_var = 60
    else:                  st_var = 20

    # モーター格差（BoatInfo.motor = 2連率）
    motors    = [b.motor or 32.0 for b in boats]
    motor_gap = max(motors) - min(motors)
    if   motor_gap >= 20: mg_score = 100
    elif motor_gap >= 15: mg_score = 70
    elif motor_gap >= 10: mg_score = 40
    else:                 mg_score = 10

    # 【朝刊AI】展示差・気象リスクは previews 由来のため使用しない。
    # 元の重み(展示15%+気象10%=25%)を残り4項目へ比例配分して100%に正規化。
    total = (
        in_score    * 0.333 +
        close_score * 0.267 +
        st_var      * 0.200 +
        mg_score    * 0.200
    )
    return round(min(100, max(0, total)))



# ════════════════════════════════════════════════════════════
# ③ 万舟警報: キー選手ピックアップ
# ════════════════════════════════════════════════════════════

def _upset_reasons(boats: list, boat1) -> list[str]:
    """
    ③ 万舟警報: 「荒れる理由」を最大3つ返す。
    画像・テキスト両方で使う。
    例: ["🔥 5号艇が展示1位(6.68)", "🔥 1号艇より高勝率が2人", "🔥 モーター41%の強機あり"]
    """
    others = [b for b in boats if b.lane != 1]
    if not others:
        return ["データなし"]

    reasons: list[str] = []

    # 【朝刊AI】コース別ST実績: 各艇の実際の進入コース(自艇の艇番)における
    # ST実績で比較する。全艇を1コースのST実績(course_st[0])で比べるのは、
    # 2〜6号艇にとって無関係なコースの実績を見ることになり誤り。
    # 1号艇はcourse_st[0]（1コース実績）、2号艇はcourse_st[1]（2コース実績）
    # …というように、各艇の lane に対応するインデックスを使う。
    def _own_course_st(b) -> Optional[float]:
        idx = b.lane - 1
        if b.course_nyuko and 0 <= idx < 6 and b.course_nyuko[idx] > 0 and b.course_st[idx] > 0:
            return b.course_st[idx]
        return None

    course_st_valid = [b for b in boats if _own_course_st(b) is not None]
    if course_st_valid:
        all_sorted = sorted(course_st_valid, key=lambda b: _own_course_st(b))
        fastest = all_sorted[0]
        if fastest.lane != 1:
            reasons.append(f"🔥 {fastest.lane}号艇が進入コースST実績{_own_course_st(fastest):.2f}で最速")
        elif len(all_sorted) > 1 and all_sorted[1].lane != 1:
            second = all_sorted[1]
            reasons.append(f"🔥 {second.lane}号艇が進入コースST実績{_own_course_st(second):.2f}で2番手")

    # 1号艇より勝率上の選手
    wr1 = boat1.win_rate or 0 if boat1 else 0
    stronger = [b for b in others if (b.win_rate or 0) > wr1]
    if stronger:
        best = max(stronger, key=lambda b: b.win_rate or 0)
        if len(stronger) >= 2:
            reasons.append(f"🔥 1号艇より高勝率が{len(stronger)}人(最高{best.win_rate:.1f})")
        else:
            reasons.append(f"🔥 {best.lane}号艇が勝率{best.win_rate:.1f}で上回る")

    # 高モーター艇
    high_motor = [b for b in others if (b.motor or 0) >= 40]
    if high_motor:
        best_m = max(high_motor, key=lambda b: b.motor or 0)
        reasons.append(f"🔥 {best_m.lane}号艇モーター{best_m.motor:.0f}%の強機")

    # 1号艇のST遅れ（【朝刊AI】平均ST実績・コース別ST実績を使用）
    if boat1 and boat1.avg_st and boat1.avg_st >= 0.19:
        course_st_1c = (
            boat1.course_st[0]
            if boat1.course_st and boat1.course_nyuko and boat1.course_nyuko[0] > 0
            else None
        )
        st_str = f"🔥 1号艇ST{boat1.avg_st:.2f}（遅れリスク）"
        if course_st_1c and course_st_1c >= 0.19:
            st_str = f"🔥 1号艇ST{boat1.avg_st:.2f}/1コース実績{course_st_1c:.2f}（遅れリスク）"
        reasons.append(st_str)

    # モーター格差
    motors = [b.motor or 32 for b in boats]
    if max(motors) - min(motors) >= 20:
        reasons.append(f"🔥 モーター格差{max(motors)-min(motors):.0f}pt（荒れやすい）")

    return reasons[:3] if reasons else ["🔥 複合要因で荒れ判定"]


def _pick_key_racer(boats: list, boat1) -> tuple:
    """後方互換用: upset_reasonsの結果をキー選手形式に変換"""
    others = [b for b in boats if b.lane != 1]
    if not others:
        return ("不明", "データなし")

    # 【朝刊AI】コース別ST実績: 各艇の実際の進入コース(自艇の艇番)における
    # ST実績を展示の代替として使用する（全艇を1コース実績で比べない）。
    def _own_course_st_kr(b) -> Optional[float]:
        idx = b.lane - 1
        if b.course_nyuko and 0 <= idx < 6 and b.course_nyuko[idx] > 0 and b.course_st[idx] > 0:
            return b.course_st[idx]
        return None

    course_st_valid = [b for b in others if _own_course_st_kr(b) is not None]
    ex_best  = min(course_st_valid, key=lambda b: _own_course_st_kr(b)) if course_st_valid else None
    wr_best  = max(others, key=lambda b: b.win_rate or 0)
    motor_best = max(others, key=lambda b: b.motor or 0)

    votes: dict = {}
    for c in [ex_best, wr_best, motor_best]:
        if c:
            votes[c.lane] = votes.get(c.lane, 0) + 1
    best_lane = max(votes, key=lambda k: (votes[k], k))
    key_boat  = next((b for b in others if b.lane == best_lane), others[0])

    reasons_raw = _upset_reasons(boats, boat1)
    reason_str  = " / ".join(r.replace("🔥 ", "") for r in reasons_raw)
    return (f"{key_boat.lane}号艇 {key_boat.name}", reason_str)

# ════════════════════════════════════════════════════════════
# ④ 覚醒モーターランキング
# ════════════════════════════════════════════════════════════

def calc_awakening(
    venue_num: int,
    motor_no: int,
    history: dict[tuple[int, int], list[dict]],
) -> Optional[int]:
    """
    覚醒モーター指数を 0〜100 で返す。データ不足（<20走）は None。

    【朝刊AI: データ区分の明記】
    calc_hot_motor と同様、`history`（motor_history.csv）は過去に
    確定済みのレース実績データであり、未来情報を含まないため使用する
    （過去実績データ）。当日これから行われるレースの previews API
    由来データ（当日リアルタイムの展示タイム・展示ST等）は一切使用しない。
    """
    rows = history.get((venue_num, motor_no), [])
    if len(rows) < 20:
        return None

    recent10 = rows[-10:]
    recent50 = rows[-50:] if len(rows) >= 50 else rows
    recent5  = rows[-5:]
    recent20 = rows[-20:]

    # 短期トレンド（着順平均: 小さいほど良い）
    avg10 = sum(int(r["place"]) for r in recent10) / len(recent10)
    avg50 = sum(int(r["place"]) for r in recent50) / len(recent50)
    trend = min(100, max(0, (avg50 - avg10) * 30 + 50))

    # 勝ち上がり速度
    rate10 = sum(1 for r in recent10 if int(r["place"]) <= 2) / len(recent10)
    rate50 = sum(1 for r in recent50 if int(r["place"]) <= 2) / len(recent50)
    climb  = min(100, max(0, (rate10 - rate50) * 500 + 50))

    # 【過去実績データ】展示改善度（速く＝小さくなるほど改善。いずれも確定済みの過去実績）
    ex5  = [float(r["ex_time"]) for r in recent5  if _is_valid_ex(r.get("ex_time"))]
    ex20 = [float(r["ex_time"]) for r in recent20 if _is_valid_ex(r.get("ex_time"))]
    if ex5 and ex20:
        avg_ex5  = sum(ex5)  / len(ex5)
        avg_ex20 = sum(ex20) / len(ex20)
        ex_improve = min(100, max(0, (avg_ex20 - avg_ex5) * 1000 + 50))
    else:
        ex_improve = 50

    total = trend * 0.40 + climb * 0.30 + ex_improve * 0.30
    return round(min(100, max(0, total)))


# ════════════════════════════════════════════════════════════
# メイン: 全ランキング生成
# ════════════════════════════════════════════════════════════

def generate_all_rankings(race_date: Optional[str] = None) -> dict:
    """全4ランキングを生成して辞書で返す"""
    if not race_date:
        race_date = _today_jst()

    programs = fetch_programs(race_date)
    previews = fetch_previews(race_date)

    if not programs:
        log.warning("[ランキング] 出走表なし: %s", race_date)
        return {
            "date": race_date,
            "danger_boat1": [], "hot_motor": [],
            "manshuu_alert": [], "awakening_motor": [],
        }

    # preview を (venue_num, race_number) で引けるようにする
    preview_map: dict[tuple, dict] = {}
    for pv in previews:
        key = (pv.get("race_stadium_number"), pv.get("race_number"))
        preview_map[key] = pv

    history = load_motor_history()

    danger_list: list[dict] = []
    upset_list:  list[dict] = []
    motor_seen:  dict[tuple, dict] = {}   # (venue_num, motor_no) → info

    for prog in programs:
        vn  = prog.get("race_stadium_number")
        rno = prog.get("race_number")
        if vn is None or rno is None:
            continue
        vn, rno = int(vn), int(rno)
        venue_name = VENUE_NAMES.get(vn, f"場{vn}")

        boats = _extract_boats_from_program(prog)
        if not boats:
            continue

        # 【朝刊AI】previews（展示・気象）は予想には使わない。
        # 生データは教師データとしてキャッシュに保存するのみ。
        preview = preview_map.get((vn, rno))
        if preview:
            try:
                _PREVIEW_RAW_CACHE[(vn, rno)] = _extract_preview_raw(preview)
            except Exception:
                pass
        weather: Optional[WeatherInfo] = WeatherInfo()

        boat1 = next((b for b in boats if b.lane == 1), None)

        # ── ① 危険な1号艇（【朝刊AI】共通エンジンで算出、Ver4: 場補正込み）────
        breakdown  = _calc_danger_breakdown(boat1, boats, weather, venue=venue_name)
        d_score    = breakdown["total"]
        if d_score >= 40:
            # コース別ST情報をまとめる（新聞表示・ST順位計算用）
            boats_course_st = []
            for b in boats:
                boats_course_st.append({
                    "lane":        b.lane,
                    "name":        b.name,
                    "racer_id":    b.racer_id,
                    "course_st":   b.course_st,
                    "course_nyuko": b.course_nyuko,
                    "course_rank": b.course_rank,
                    "avg_st":      b.avg_st,
                    # 【朝刊AI】ex_st は previews 由来のため使用しない（常にNone）
                })

            # 1号艇の1コースSTが全6艇中何位か計算
            # （1コース以外の選手も「1コースでのST」で比較）
            st_1c_list = [
                (b.lane, b.course_st[0])
                for b in boats
                if b.course_nyuko[0] > 0
            ]
            if st_1c_list and boat1 and boat1.course_nyuko[0] > 0:
                st_1c_sorted = sorted(st_1c_list, key=lambda x: x[1])
                boat1_1c_rank = next(
                    (i+1 for i, (lane, _) in enumerate(st_1c_sorted) if lane == 1), None
                )
            else:
                boat1_1c_rank = None

            # ── 上位進出指数（0-100、危険艇速報・買い目生成・万舟警報で共通、Ver4: 場補正込み） ──
            rank_index = calc_rank_index_v2(boats, venue=venue_name)
            featured_boats = select_featured_boats(boats, rank_index, d_score)

            # ── Ver4: 水面タイプ（実データ自動算出。データ不足時は
            #    著名な場のみ参考値、それ以外は「標準」） ──
            water_type = classify_water_type(venue_name)

            danger_list.append({
                "venue":        venue_name,
                "venue_num":    vn,
                "race":         rno,
                "score":        d_score,
                "racer":        boat1.name if boat1 else "?",
                "racer_class":  boat1.racer_class if boat1 else "",
                "win_rate":     boat1.win_rate    if boat1 else 0,
                "motor":        boat1.motor       if boat1 else 0,
                "avg_st":       boat1.avg_st      if boat1 else 0,
                # 【朝刊AI】ex_time は previews 由来のため保存しない
                "reason":       _danger_reason(boat1, boats, breakdown),
                "stars":        _calc_stars(boat1, boats),
                "breakdown":    breakdown,
                # コース別ST情報（新聞の危険艇速報で表示）
                "boats_course_st":  boats_course_st,
                "boat1_1c_st":      boat1.course_st[0] if boat1 and boat1.course_nyuko[0] > 0 else None,
                "boat1_1c_rank":    boat1_1c_rank,      # 6艇中の1コースST順位
                "boat1_1c_nyuko":   boat1.course_nyuko[0] if boat1 else 0,
                # 上位進出指数（各艇: top1/top2/top3、0-100）と注目選手
                "rank_index":     rank_index,
                "featured_boats": featured_boats,
                # 【Ver4】水面タイプ
                "water_type": water_type,
            })

        # ── ③ 万舟警報 ───────────────────────────────────
        u_score = calc_upset_index(boats, weather)
        if u_score >= 40:
            key_racer, key_reason = _pick_key_racer(boats, boat1)
            upset_list.append({
                "venue":      venue_name,
                "venue_num":  vn,
                "race":       rno,
                "score":      u_score,
                "key_racer":  key_racer,
                "key_reason": key_reason,
            })

        # ── ②④ 用のモーター情報収集 ──────────────────────
        for b_raw in prog.get("boats", []):
            if not isinstance(b_raw, dict):
                continue
            motor_no = _boat_motor_no(b_raw)
            if motor_no is None:
                continue
            key = (vn, motor_no)
            if key not in motor_seen:
                m2rate_raw = b_raw.get("racer_assigned_motor_top_2_percent")
                try:
                    m2rate: Optional[float] = float(m2rate_raw) if m2rate_raw else None
                except (ValueError, TypeError):
                    m2rate = None
                motor_seen[key] = {
                    "venue":      venue_name,
                    "venue_num":  vn,
                    "motor_no":   motor_no,
                    "motor_2rate": m2rate,
                }

    # ── ② 激走モーター ───────────────────────────────────
    hot_list: list[dict] = []
    for (vn, mno), info in motor_seen.items():
        score = calc_hot_motor(vn, mno, info["motor_2rate"], history)
        if score is not None and score >= 50:
            rows = history.get((vn, mno), [])
            recent5_places = [str(int(r["place"])) for r in rows[-5:]]
            recent10_2rate = (
                sum(1 for r in rows[-10:] if int(r["place"]) <= 2) / min(len(rows), 10) * 100
                if rows else 0
            )
            official_2rate = info["motor_2rate"] or 32.0
            hot_list.append({
                **info,
                "score":           score,
                "recent5":         "-".join(recent5_places) if recent5_places else "---",
                "recent10_2rate":  round(recent10_2rate, 1),
                "official_2rate":  official_2rate,
                "gap":             round(recent10_2rate - official_2rate, 1),
            })

    # ── ④ 覚醒モーター ───────────────────────────────────
    awake_list: list[dict] = []
    for (vn, mno), info in motor_seen.items():
        score = calc_awakening(vn, mno, history)
        if score is not None and score >= 50:
            rows = history.get((vn, mno), [])
            recent10 = rows[-10:]
            recent10_places = [str(int(r["place"])) for r in recent10]
            # 直近展示タイム平均
            ex10 = [float(r["ex_time"]) for r in recent10 if _is_valid_ex(r.get("ex_time"))]
            ex_avg = round(sum(ex10) / len(ex10), 2) if ex10 else None
            # 直近10走 vs 前半10走の2連率比較
            old10 = rows[-20:-10] if len(rows) >= 20 else []
            old_2rate = (sum(1 for r in old10 if int(r["place"]) <= 2) / len(old10) * 100) if old10 else None
            new_2rate = (sum(1 for r in recent10 if int(r["place"]) <= 2) / len(recent10) * 100) if recent10 else None
            awake_list.append({
                **info,
                "score":         score,
                "recent10":      "-".join(recent10_places) if recent10_places else "---",
                "ex_avg":        ex_avg,
                "old_2rate":     round(old_2rate, 1) if old_2rate is not None else None,
                "new_2rate":     round(new_2rate,  1) if new_2rate  is not None else None,
            })

    result = {
        "date":            race_date,
        "danger_boat1":    sorted(danger_list, key=lambda x: -x["score"])[:10],
        "hot_motor":       sorted(hot_list,    key=lambda x: -x["score"])[:20],
        "manshuu_alert":   sorted(upset_list,  key=lambda x: -x["score"])[:10],
        "awakening_motor": sorted(awake_list,  key=lambda x: -x["score"])[:10],
        # note用: スコア40以上の全レース（ランク問わず）
        "all_danger":      sorted(danger_list, key=lambda x: -x["score"]),
        "all_manshuu":     sorted(upset_list,  key=lambda x: -x["score"]),
    }

    # ranking_cache.json に保存
    try:
        with open(RANKING_CACHE, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        log.info("[キャッシュ] 保存: %s", RANKING_CACHE)
    except OSError as e:
        log.warning("[キャッシュ] 保存失敗: %s", e)

    # ── ranking_filter.json に通知対象レースを保存 ──────────────
    # notify_arashi.py がこれを参照して送信可否を判断する
    try:
        allowed: list[dict] = []
        for d in result["danger_boat1"]:
            allowed.append({
                "venue_num": d["venue_num"],
                "race":      d["race"],
                "reason":    "danger",
                "venue":     d["venue"],
            })
        for u in result["manshuu_alert"]:
            key = (u["venue_num"], u["race"])
            if not any(a["venue_num"] == key[0] and a["race"] == key[1] for a in allowed):
                allowed.append({
                    "venue_num": u["venue_num"],
                    "race":      u["race"],
                    "reason":    "manshuu",
                    "venue":     u["venue"],
                })
        # 激走モーターは場単位なのでその場の全レースを許可
        hot_venues = set(m["venue_num"] for m in result["hot_motor"])
        for prog_vn in hot_venues:
            # その場の全レース番号を追加（1〜12R）
            for rno in range(1, 13):
                key = (prog_vn, rno)
                if not any(a["venue_num"] == key[0] and a["race"] == key[1] for a in allowed):
                    vname = next((m["venue"] for m in result["hot_motor"] if m["venue_num"] == prog_vn), f"場{prog_vn}")
                    allowed.append({
                        "venue_num": prog_vn,
                        "race":      rno,
                        "reason":    "hot_motor",
                        "venue":     vname,
                    })

        filter_data = {
            "date":    race_date,
            "allowed": allowed,
        }
        with open("ranking_filter.json", "w", encoding="utf-8") as f:
            json.dump(filter_data, f, ensure_ascii=False, indent=2)
        log.info("[フィルタ] 保存: ranking_filter.json (%d件)", len(allowed))
    except OSError as e:
        log.warning("[フィルタ] 保存失敗: %s", e)

    return result


# ════════════════════════════════════════════════════════════
# 投稿テキスト生成
# ════════════════════════════════════════════════════════════

def format_danger_tweet(data: dict) -> str:
    """⑥ 危険な1号艇: ワースト1位を主役にした投稿文"""
    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    items = data.get("danger_boat1", [])

    if not items:
        return f"⚠️【{date_str} 危険な1号艇】⚠️\n\n本日は該当レースなし\n#競艇 #ボートレース"

    top = items[0]
    rank = _score_to_rank(top["score"])

    stars = top.get("stars", {})
    lines = [
        f"🚨 AI危険艇速報 {date_str}",
        "",
        f"【{rank}】{top['venue']}{top['race']}R",
        f"{top['racer']}",
        f"▶ {top['reason']}",
        "",
    ]
    if stars:
        lines += [
            "── AI評価 ──",
            f"ST　　{stars.get('ST',  '★★★☆☆')}",
            f"機力　{stars.get('機力', '★★★☆☆')}",
            f"近況　{stars.get('近況', '★★★☆☆')}",
            f"相手　{stars.get('相手', '★★★☆☆')}",
            "",
        ]

    if len(items) > 1:
        lines.append("── 他の注目レース ──")
        for d in items[1:6]:
            r = _score_to_rank_short(d["score"])
            lines.append(f"{r} {d['venue']}{d['race']}R {d['racer']}")
        lines.append("")

    # 昨日の答え合わせを差し込む
    try:
        from x_verification import get_yesterday_summary, format_yesterday_oneliner
        yday = format_yesterday_oneliner(get_yesterday_summary())
        if yday:
            lines += ["", "─" * 20, yday, "─" * 20]
    except Exception:
        pass

    lines += [
        "あなたが今日気になるレースはどこですか？💬",
        "#競艇 #ボートレース #競艇予想 #1号艇 #荒れ予想",
    ]
    return "\n".join(lines)


def format_hot_motor_tweet(data: dict) -> str:
    """⑥ 激走モーター: 直近走行と上昇率を表示"""
    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    items = data.get("hot_motor", [])

    if not items:
        return (f"🔥【{date_str} 激走モーターTOP10】🔥\n\n"
                "データ蓄積中...\n#競艇 #ボートレース #モーター")

    lines = [f"🔥 AI激走モーター {date_str}", ""]

    for i, m in enumerate(items[:10], 1):
        recent = m.get("recent5", "---")
        gap    = m.get("gap", 0)
        gap_str = f"+{gap:.0f}%" if gap > 0 else f"{gap:.0f}%"
        lines.append(f"{i}位 {m['venue']}{m['motor_no']}号機")
        lines.append(f"   直近5走: {recent}  公式比{gap_str}")

    lines += [
        "",
        "公式2連率を大きく上回る激走モーター🔧",
        "今節乗っている選手に注目！",
        "",
        "あなたのお気に入りモーターはありましたか？💬",
        "#競艇 #ボートレース #モーター #競艇予想",
    ]
    return "\n".join(lines)


def format_manshuu_tweet(data: dict) -> str:
    """⑥ 万舟警報: 荒れる理由を前面に出した投稿文"""
    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    items = data.get("manshuu_alert", [])

    if not items:
        return (f"🚨【{date_str} 万舟警報】🚨\n\n本日は該当なし\n#競艇 #ボートレース")

    top = items[0]
    rank = _score_to_rank(top["score"])
    reasons = top.get("key_reason", "").split(" / ")

    lines = [
        f"💰 AI万舟警報 {date_str}",
        "",
        f"【{rank}】{top['venue']}{top['race']}R",
        "",
    ]
    for r in reasons[:3]:
        lines.append(r if r.startswith("🔥") else f"🔥 {r}")
    lines.append("")

    if len(items) > 1:
        lines.append("── 他の警戒レース ──")
        for u in items[1:5]:
            r = _score_to_rank_short(u["score"])
            lines.append(f"{r} {u['venue']}{u['race']}R")
        lines.append("")

    lines += [
        "高配当が出そうなレースに注目💰",
        "どのレースが気になりますか？💬",
        "#競艇 #ボートレース #万舟 #荒れ予想 #穴予想",
    ]
    return "\n".join(lines)


def format_awakening_tweet(data: dict) -> str:
    """⑥ 覚醒モーター: 急変の証拠を数字で示す"""
    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    items = data.get("awakening_motor", [])

    if not items:
        return (f"⚡【{date_str} 覚醒モーターTOP10】⚡\n\n"
                "データ蓄積中...\n#競艇 #ボートレース #モーター")

    lines = [f"⚡ AI覚醒モーター {date_str}", ""]

    for i, a in enumerate(items[:10], 1):
        recent  = a.get("recent10", "---")
        old_r   = a.get("old_2rate")
        new_r   = a.get("new_2rate")
        ex_avg  = a.get("ex_avg")
        lines.append(f"{i}位 {a['venue']}{a['motor_no']}号機")
        detail = []
        if old_r is not None and new_r is not None:
            detail.append(f"2連率 {old_r:.0f}%→{new_r:.0f}%")
        if ex_avg:
            detail.append(f"展示平均{ex_avg:.2f}秒")
        lines.append(f"   直近10走: {recent}")
        if detail:
            lines.append(f"   {'  '.join(detail)}")

    lines += [
        "",
        "急に仕上がってきたモーターは狙い目📈",
        "",
        "どのモーターが気になりましたか？💬",
        "#競艇 #ボートレース #モーター #覚醒 #競艇予想",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# エントリポイント
# ════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="競艇 X 自動投稿用ランキング生成")
    parser.add_argument("--date",           help="対象日 YYYYMMDD（省略時は今日）")
    parser.add_argument("--update-history", action="store_true",
                        help="モーター履歴を蓄積（通常は前日分）")
    parser.add_argument("--generate",       action="store_true",
                        help="ランキング生成＋ツイートテキスト出力")
    parser.add_argument("--output",         default="today_tweets.txt",
                        help="ツイートテキストをまとめて出力するファイル名（デフォルト: today_tweets.txt）")
    parser.add_argument("--split",          action="store_true",
                        help="4種類のツイートを別ファイルに分けて出力")
    args = parser.parse_args()

    race_date = args.date or _today_jst()

    # ── モーター履歴蓄積 ──────────────────────────────────────
    if args.update_history:
        target_date = args.date or _yesterday_jst()
        n = update_motor_history(target_date)
        log.info("[履歴更新] %s: %d件追記", target_date, n)

    # ── ランキング生成 ────────────────────────────────────────
    if args.generate:
        data = generate_all_rankings(race_date)

        tweets = [
            format_danger_tweet(data),
            format_hot_motor_tweet(data),
            format_manshuu_tweet(data),
            format_awakening_tweet(data),
        ]
        split_names = [
            "danger_tweet.txt",
            "hot_motor_tweet.txt",
            "manshuu_tweet.txt",
            "awakening_tweet.txt",
        ]

        if args.split:
            for name, text in zip(split_names, tweets):
                with open(name, "w", encoding="utf-8") as f:
                    f.write(text + "\n")
                log.info("[出力] %s", name)
        else:
            combined = ("\n" + DIVIDER + "\n").join(tweets)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(combined + "\n")
            log.info("[出力] %s", args.output)

        # 標準出力にも表示
        for text in tweets:
            print(DIVIDER)
            print(text)
        print(DIVIDER)

        log.info(
            "[完了] 危険1号艇:%d件 激走モーター:%d件 万舟警報:%d件 覚醒モーター:%d件",
            len(data["danger_boat1"]),
            len(data["hot_motor"]),
            len(data["manshuu_alert"]),
            len(data["awakening_motor"]),
        )


if __name__ == "__main__":
    main()
