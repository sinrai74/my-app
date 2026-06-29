#!/usr/bin/env python3
"""
x_korogashi.py  ── AI転がし専用エンジン

「2・3・4号艇が1着になる可能性が高いレース」だけを抽出し、
転がしチャレンジに最適なレースをスコアリングする。

notify_arashi.py の fetch_programs / fetch_previews /
_extract_boats_from_program / _apply_preview_to_boats /
VENUE_NAMES / BoatInfo / WeatherInfo を再利用。

Usage:
    python x_korogashi.py --generate              # 今日の転がし候補生成
    python x_korogashi.py --generate --date 20260629
    python x_korogashi.py --generate --dry-run    # 保存のみ（メール送信なし）
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import smtplib
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from notify_arashi import (
    VENUE_NAMES,
    fetch_programs,
    fetch_previews,
    _extract_boats_from_program,
    _apply_preview_to_boats,
    BoatInfo,
    WeatherInfo,
)

log = logging.getLogger("x_korogashi")

JST    = timezone(timedelta(hours=9))
MAIL_TO = "bigkirinuki@gmail.com"
KOROGASHI_CACHE = "korogashi_cache.json"

# ────────────────────────────────────────────────────────────
# 転がし適性スコアの閾値
# ────────────────────────────────────────────────────────────
THRESHOLD_BUY    = 85   # 85以上 → 購入
THRESHOLD_WATCH  = 70   # 70-84  → 注意
# 70未満 → 見送り

# 対象艇番（1号艇は原則外す）
TARGET_LANES = {2, 3, 4}
ALLOW_LANE_5_6_EV_MIN = 1.6  # 5・6号艇を許容する最低期待値

# スコアリング重み（合計100点）
WEIGHTS = {
    "ex_time":       18,   # 展示タイム（レース内順位）
    "avg_st":        14,   # 平均ST（速さ）
    "motor":         14,   # モーター2連率
    "recent":        12,   # 直近成績（勝率ベース）
    "course_win":    10,   # コース勝率（2-4コースのイン逃げ対抗力）
    "in_trust":      10,   # 1号艇信頼度の逆数（飛びやすさ）
    "rival":          8,   # 相手関係（自艇 vs 他艇）
    "odds_ev":        8,   # 期待値（オッズ × 確率）
    "weather":        6,   # 気象リスク（低いほど良い）
}


# ════════════════════════════════════════════════════════════
# スコアリング計算
# ════════════════════════════════════════════════════════════

@dataclass
class KorogashiScore:
    """1艇の転がし適性スコア内訳"""
    venue:        str
    venue_num:    int
    race:         int
    lane:         int
    racer_name:   str
    racer_class:  str
    win_rate:     float
    motor:        float
    avg_st:       float
    ex_time:      Optional[float]
    # スコア内訳
    sc_ex_time:   float = 0.0
    sc_avg_st:    float = 0.0
    sc_motor:     float = 0.0
    sc_recent:    float = 0.0
    sc_course_win:float = 0.0
    sc_in_trust:  float = 0.0
    sc_rival:     float = 0.0
    sc_odds_ev:   float = 0.0
    sc_weather:   float = 0.0
    # 集計
    reliability:  float = 0.0   # AI信頼度（100点満点）
    payout_score: float = 0.0   # 配当期待（100点満点）
    fitness:      float = 0.0   # 転がし適性（100点満点）
    # 購入判定
    verdict:      str   = ""    # "購入" / "注意" / "見送り"
    # 期待値
    odds_est:     float = 0.0   # 想定オッズ
    ev:           float = 0.0   # 期待値
    # 危険要因
    dangers:      list  = field(default_factory=list)
    # 理由テキスト
    reason:       str   = ""


def _rank_in_list(value: float, values: list[float],
                  reverse: bool = False) -> int:
    """value が values の中で何位か（1=最良）"""
    if not values:
        return 1
    if reverse:
        sorted_v = sorted(values)          # 小さい方が良い
    else:
        sorted_v = sorted(values, reverse=True)  # 大きい方が良い
    try:
        return sorted_v.index(value) + 1
    except ValueError:
        return len(values)


def _score_rank(rank: int, n: int) -> float:
    """順位をスコア化（1位=100点, n位=0点）"""
    if n <= 1:
        return 100.0
    return max(0.0, 100.0 * (n - rank) / (n - 1))


def calc_korogashi_score(
    target: BoatInfo,
    all_boats: list[BoatInfo],
    weather: Optional[WeatherInfo],
    venue: str,
    venue_num: int,
    race: int,
    odds_map: dict[str, float],
    history: dict,  # motor_history
) -> KorogashiScore:
    """
    1艇の転がし適性スコアを計算する。
    """
    n = len(all_boats)
    boat1 = next((b for b in all_boats if b.lane == 1), None)

    ks = KorogashiScore(
        venue=venue, venue_num=venue_num, race=race,
        lane=target.lane, racer_name=target.name,
        racer_class=target.racer_class,
        win_rate=target.win_rate or 0.0,
        motor=target.motor or 0.0,
        avg_st=target.avg_st or 0.18,
        ex_time=target.ex_time,
    )

    # ── ① 展示タイム ─────────────────────────────────────
    ex_times = [b.ex_time for b in all_boats if b.ex_time and b.ex_time > 0]
    if target.ex_time and target.ex_time > 0 and ex_times:
        rank = _rank_in_list(target.ex_time, ex_times, reverse=True)  # 小さい=速い
        ks.sc_ex_time = _score_rank(rank, len(ex_times))
    else:
        ks.sc_ex_time = 50.0

    # ── ② 平均ST ─────────────────────────────────────────
    avg_sts = [b.avg_st for b in all_boats if b.avg_st and b.avg_st > 0]
    if avg_sts:
        rank = _rank_in_list(target.avg_st or 0.18, avg_sts, reverse=True)
        ks.sc_avg_st = _score_rank(rank, len(avg_sts))
    else:
        ks.sc_avg_st = 50.0

    # ── ③ モーター2連率 ──────────────────────────────────
    motors = [b.motor for b in all_boats if b.motor and b.motor > 0]
    if motors:
        rank = _rank_in_list(target.motor or 32.0, motors)
        ks.sc_motor = _score_rank(rank, len(motors))
    else:
        ks.sc_motor = 50.0

    # 激走モーター補正（motor_historyがあれば）
    mh_key = (venue_num, target.lane)
    if history:
        rows = list(history.values())[0] if history else []
        # motor_noが特定できない場合はスキップ
        pass

    # ── ④ 直近成績（勝率ベース） ────────────────────────
    win_rates = [b.win_rate for b in all_boats if b.win_rate and b.win_rate > 0]
    if win_rates:
        rank = _rank_in_list(target.win_rate or 0.0, win_rates)
        ks.sc_recent = _score_rank(rank, len(win_rates))
    else:
        ks.sc_recent = 50.0

    # ── ⑤ コース勝率（コース別イン逃げ対抗力） ──────────
    # 2コース: 差し/捲り率が高い → 好評価
    # 3コース: 捲り得意なら好評価
    # 4コース: 展示が良ければ十分
    lane = target.lane
    course_bonus = {2: 0.85, 3: 0.75, 4: 0.70, 5: 0.60, 6: 0.50}.get(lane, 0.50)
    # 等級補正
    grade_bonus = {"A1": 1.0, "A2": 0.9, "B1": 0.75, "B2": 0.60}.get(
        target.racer_class, 0.80)
    ks.sc_course_win = course_bonus * grade_bonus * 100.0

    # ── ⑥ 1号艇信頼度の逆数（イン信頼度が低い=転がしに有利）
    if boat1:
        in_trust = 0.0
        # 1号艇がA1・高勝率・ST安定 → インが堅い → 転がし不利
        if boat1.racer_class == "A1":
            in_trust += 40
        elif boat1.racer_class == "A2":
            in_trust += 25
        if boat1.win_rate and boat1.win_rate >= 6.0:
            in_trust += 30
        elif boat1.win_rate and boat1.win_rate >= 5.0:
            in_trust += 15
        if boat1.avg_st and boat1.avg_st <= 0.15:
            in_trust += 20
        if boat1.motor and boat1.motor >= 40:
            in_trust += 10
        # 信頼度が高い = 転がしに不利 → 逆転させる
        ks.sc_in_trust = max(0.0, 100.0 - min(100.0, in_trust))
    else:
        ks.sc_in_trust = 50.0

    # ── ⑦ 相手関係 ──────────────────────────────────────
    others = [b for b in all_boats if b.lane != target.lane and b.lane != 1]
    if others:
        my_wr = target.win_rate or 0.0
        better = sum(1 for b in others if (b.win_rate or 0) > my_wr)
        ks.sc_rival = max(0.0, 100.0 - better * 15)
    else:
        ks.sc_rival = 80.0

    # ── ⑧ 期待値（想定オッズ×確率） ────────────────────
    # 単勝オッズをodds_mapから推定（三連単から逆算）
    my_wins = {k: v for k, v in odds_map.items()
               if k.startswith(f"{lane}-") and v > 0}
    if my_wins:
        avg_odds = sum(my_wins.values()) / len(my_wins)
        # 単勝オッズ推定（三連単平均÷6）
        tansho_est = max(1.1, avg_odds / 6.0)
        # 確率推定（単勝オッズから）
        prob_est = min(0.5, 1.0 / tansho_est)
        ev = tansho_est * prob_est
        ks.odds_est = round(tansho_est, 1)
        ks.ev = round(ev, 3)
        # EV スコア化（EV1.2以上=100点, EV0.8以下=0点）
        ks.sc_odds_ev = min(100.0, max(0.0, (ev - 0.8) / 0.4 * 100))
    else:
        # オッズデータなし → コース別推定
        lane_odds = {2: 3.5, 3: 5.0, 4: 6.5, 5: 10.0, 6: 15.0}
        ks.odds_est = lane_odds.get(lane, 5.0)
        prob_est = 1.0 / ks.odds_est
        ks.ev = round(ks.odds_est * prob_est, 3)
        ks.sc_odds_ev = 50.0

    # ── ⑨ 気象リスク ────────────────────────────────────
    if weather:
        ws = weather.wind_speed or 0
        wh = weather.wave_height or 0
        wd = weather.wind_direction or ""
        risk = 0
        if ws >= 5: risk += 40
        elif ws >= 3: risk += 20
        if wh >= 10: risk += 30
        elif wh >= 5: risk += 15
        if "向" in wd: risk += 20
        ks.sc_weather = max(0.0, 100.0 - risk)
    else:
        ks.sc_weather = 80.0

    # ── 危険要因チェック ─────────────────────────────────
    dangers = []
    if target.ex_time and ex_times:
        ex_rank = _rank_in_list(target.ex_time, ex_times, reverse=True)
        if ex_rank >= n - 1:
            dangers.append("展示ワースト")
    if target.avg_st and target.avg_st >= 0.20:
        dangers.append(f"ST遅め({target.avg_st:.2f})")
    if target.motor and target.motor <= 25:
        dangers.append(f"モーター低({target.motor:.0f}%)")
    if boat1 and boat1.racer_class == "A1" and (boat1.win_rate or 0) >= 6.0:
        dangers.append("1号艇強A1")
    if weather and (weather.wind_speed or 0) >= 5:
        dangers.append(f"強風({weather.wind_speed:.0f}m/s)")
    if weather and (weather.wave_height or 0) >= 10:
        dangers.append(f"波高({weather.wave_height}cm)")
    ks.dangers = dangers

    # ── 総合スコア計算 ───────────────────────────────────
    W = WEIGHTS
    raw = (
        ks.sc_ex_time    * W["ex_time"]    +
        ks.sc_avg_st     * W["avg_st"]     +
        ks.sc_motor      * W["motor"]      +
        ks.sc_recent     * W["recent"]     +
        ks.sc_course_win * W["course_win"] +
        ks.sc_in_trust   * W["in_trust"]   +
        ks.sc_rival      * W["rival"]      +
        ks.sc_odds_ev    * W["odds_ev"]    +
        ks.sc_weather    * W["weather"]
    ) / sum(W.values())

    ks.reliability  = round(raw, 1)
    # 配当期待スコア（オッズ重視）
    ks.payout_score = round(
        ks.sc_odds_ev * 0.5 + ks.sc_in_trust * 0.3 + ks.sc_ex_time * 0.2, 1)
    # 転がし適性（信頼度と配当の調和平均）
    r = ks.reliability / 100
    p = ks.payout_score / 100
    if r + p > 0:
        harmonic = 2 * r * p / (r + p)
    else:
        harmonic = 0
    ks.fitness = round(harmonic * 100, 1)

    # 判定
    if ks.fitness >= THRESHOLD_BUY:
        ks.verdict = "購入"
    elif ks.fitness >= THRESHOLD_WATCH:
        ks.verdict = "注意"
    else:
        ks.verdict = "見送り"

    # 理由テキスト
    positives = []
    if ks.sc_ex_time >= 70:
        positives.append(f"展示{ex_times.index(target.ex_time)+1 if target.ex_time in ex_times else '?'}位")
    if ks.sc_motor >= 70:
        positives.append(f"モーター{target.motor:.0f}%")
    if ks.sc_avg_st >= 70:
        positives.append(f"ST{target.avg_st:.2f}")
    if ks.sc_in_trust >= 70:
        positives.append("1号艇弱め")
    if target.racer_class in ("A1", "A2"):
        positives.append(target.racer_class)
    ks.reason = " / ".join(positives) if positives else "総合判定"

    return ks


# ════════════════════════════════════════════════════════════
# 全レーススキャン
# ════════════════════════════════════════════════════════════

def scan_all_races(
    race_date: Optional[str] = None,
    history: dict = {},
) -> list[KorogashiScore]:
    """
    全レースをスキャンして転がし候補スコアのリストを返す。
    対象は2・3・4号艇（条件次第で5・6号艇も含む）。
    """
    if not race_date:
        race_date = datetime.now(JST).strftime("%Y%m%d")

    programs = fetch_programs(race_date)
    previews  = fetch_previews(race_date)

    if not programs:
        log.warning("[転がし] 出走表なし: %s", race_date)
        return []

    preview_map: dict[tuple, dict] = {}
    for pv in previews:
        key = (pv.get("race_stadium_number"), pv.get("race_number"))
        preview_map[key] = pv

    results: list[KorogashiScore] = []

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

        preview = preview_map.get((vn, rno))
        weather: Optional[WeatherInfo] = None
        if preview:
            weather = _apply_preview_to_boats(boats, preview)

        # オッズmap（あれば利用）
        odds_map: dict[str, float] = {}

        boat1 = next((b for b in boats if b.lane == 1), None)

        for b in boats:
            lane = b.lane

            # 対象艇判定
            if lane in TARGET_LANES:
                pass  # 対象
            elif lane in {5, 6}:
                # 5・6号艇: 展示1位かつ1号艇が弱い場合のみ候補
                ex_valid = [x.ex_time for x in boats if x.ex_time and x.ex_time > 0]
                if not ex_valid:
                    continue
                is_fastest = b.ex_time and b.ex_time == min(ex_valid)
                boat1_weak = boat1 and (
                    boat1.racer_class in ("B1", "B2") or
                    (boat1.win_rate or 0) < 4.5
                )
                if not (is_fastest and boat1_weak):
                    continue
            else:
                continue  # 1号艇はスキップ

            ks = calc_korogashi_score(
                target=b,
                all_boats=boats,
                weather=weather,
                venue=venue_name,
                venue_num=vn,
                race=rno,
                odds_map=odds_map,
                history=history,
            )

            # 最低EVチェック（5・6号艇）
            if lane in {5, 6} and ks.ev < ALLOW_LANE_5_6_EV_MIN:
                continue

            results.append(ks)

    # 転がし適性降順でソート
    results.sort(key=lambda x: -x.fitness)
    return results


# ════════════════════════════════════════════════════════════
# 日次判定（今日は買いか見送りか）
# ════════════════════════════════════════════════════════════

def daily_verdict(scores: list[KorogashiScore]) -> dict:
    """
    今日の転がし総合判定を返す。
    {
      "verdict":      "購入" / "注意" / "見送り",
      "reason":       str,
      "top1":         KorogashiScore | None,
      "top10":        list[KorogashiScore],
      "buy_count":    int,
      "watch_count":  int,
    }
    """
    if not scores:
        return {
            "verdict": "見送り",
            "reason":  "本日は対象レースなし",
            "top1": None, "top10": [],
            "buy_count": 0, "watch_count": 0,
        }

    buy_list   = [s for s in scores if s.verdict == "購入"]
    watch_list = [s for s in scores if s.verdict == "注意"]
    top10      = scores[:10]
    top1       = scores[0] if scores else None

    # 見送り判定
    if not buy_list and not watch_list:
        verdict = "見送り"
        reason  = f"全{len(scores)}候補中、転がし適性70超えなし"
    elif top1 and top1.fitness < THRESHOLD_WATCH:
        verdict = "見送り"
        reason  = f"最高適性スコア{top1.fitness}（基準{THRESHOLD_WATCH}未満）"
    elif top1 and top1.fitness < THRESHOLD_BUY:
        verdict = "注意"
        reason  = f"適性スコア{top1.fitness}（購入基準{THRESHOLD_BUY}未満）"
    else:
        verdict = "購入"
        reason  = (
            f"転がし適性{top1.fitness}点 {top1.venue}{top1.race}R "
            f"{top1.lane}号艇 {top1.racer_name} が最有力"
        )

    return {
        "verdict":     verdict,
        "reason":      reason,
        "top1":        top1,
        "top10":       top10,
        "buy_count":   len(buy_list),
        "watch_count": len(watch_list),
    }


# ════════════════════════════════════════════════════════════
# JSON出力
# ════════════════════════════════════════════════════════════

def scores_to_dict(scores: list[KorogashiScore]) -> list[dict]:
    out = []
    for s in scores:
        out.append({
            "venue":        s.venue,
            "venue_num":    s.venue_num,
            "race":         s.race,
            "lane":         s.lane,
            "racer_name":   s.racer_name,
            "racer_class":  s.racer_class,
            "win_rate":     s.win_rate,
            "motor":        s.motor,
            "avg_st":       s.avg_st,
            "ex_time":      s.ex_time,
            "reliability":  s.reliability,
            "payout_score": s.payout_score,
            "fitness":      s.fitness,
            "verdict":      s.verdict,
            "odds_est":     s.odds_est,
            "ev":           s.ev,
            "dangers":      s.dangers,
            "reason":       s.reason,
            "score_detail": {
                "ex_time":    round(s.sc_ex_time, 1),
                "avg_st":     round(s.sc_avg_st, 1),
                "motor":      round(s.sc_motor, 1),
                "recent":     round(s.sc_recent, 1),
                "course_win": round(s.sc_course_win, 1),
                "in_trust":   round(s.sc_in_trust, 1),
                "rival":      round(s.sc_rival, 1),
                "odds_ev":    round(s.sc_odds_ev, 1),
                "weather":    round(s.sc_weather, 1),
            },
        })
    return out


def save_cache(result: dict, path: str = KOROGASHI_CACHE) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        log.info("[キャッシュ] 保存: %s", path)
    except OSError as e:
        log.warning("[キャッシュ] 保存失敗: %s", e)


# ════════════════════════════════════════════════════════════
# テキスト生成（X投稿用）
# ════════════════════════════════════════════════════════════

def format_daily_tweet(result: dict, date_str: str) -> str:
    """毎朝の転がし候補投稿テキスト"""
    date_disp = f"{date_str[4:6]}/{date_str[6:8]}"
    verdict   = result["verdict"]
    top1      = result.get("top1")
    top10     = result.get("top10", [])

    if verdict == "見送り":
        return (
            f"🤖【{date_disp} AI転がし日記】\n\n"
            f"本日の判定：⛔ 見送り\n\n"
            f"{result['reason']}\n\n"
            "期待値が低い日は無理に買いません。\n"
            "転がしは長期戦です。\n\n"
            "#AI転がし日記 #競艇 #ボートレース"
        )

    emoji = "✅" if verdict == "購入" else "⚠️"
    lines = [
        f"🤖【{date_disp} AI転がし日記】",
        "",
        f"本日の判定：{emoji} {verdict}",
        "",
    ]

    if top1:
        lines += [
            f"🏆 本日のチャレンジレース",
            f"  {top1.venue}{top1.race}R  {top1.lane}号艇 {top1.racer_name}",
            f"  転がし適性 {top1.fitness}点",
            f"  AI信頼度 {top1.reliability}点",
            f"  想定オッズ {top1.odds_est}倍",
            "",
        ]
        if top1.dangers:
            lines.append(f"  ⚠️ リスク: {' / '.join(top1.dangers)}")
            lines.append("")

    if len(top10) > 1:
        lines.append("── 転がし候補ランキング ──")
        for i, s in enumerate(top10[:5], 1):
            verdict_mark = "✅" if s.verdict == "購入" else "⚠️" if s.verdict == "注意" else "❌"
            lines.append(
                f"{i}位 {s.venue}{s.race}R {s.lane}号艇 "
                f"{verdict_mark}適性{s.fitness} / {s.odds_est}倍"
            )
        lines.append("")

    lines += [
        f"購入候補: {result['buy_count']}件  注意: {result['watch_count']}件",
        "",
        "結果は夜に報告します📊",
        "#AI転がし日記 #競艇 #ボートレース #転がし",
    ]
    return "\n".join(lines)


def format_result_tweet(result: dict, date_str: str,
                        hit: bool, profit: int, balance: int) -> str:
    """夜の結果報告テキスト"""
    date_disp = f"{date_str[4:6]}/{date_str[6:8]}"
    top1 = result.get("top1")

    if result["verdict"] == "見送り":
        return (
            f"📊【{date_disp} AI転がし日記 結果】\n\n"
            "本日は見送り判定のため購入なし。\n"
            f"残高に変動なし。¥{balance:,}\n\n"
            "#AI転がし日記 #競艇"
        )

    race_str = f"{top1.venue}{top1.race}R {top1.lane}号艇" if top1 else "---"
    result_mark = "✅ 的中！" if hit else "❌ 外れ"
    profit_str  = f"+¥{profit:,}" if profit >= 0 else f"▲¥{abs(profit):,}"

    return (
        f"📊【{date_disp} AI転がし日記 結果】\n\n"
        f"{race_str}\n"
        f"{result_mark}  {profit_str}\n\n"
        f"残高: ¥{balance:,}\n\n"
        f"AIコメント: {top1.reason if top1 else '---'}\n\n"
        "#AI転がし日記 #競艇 #ボートレース #転がし"
    )


# ════════════════════════════════════════════════════════════
# メール送信
# ════════════════════════════════════════════════════════════

def send_korogashi_mail(
    result: dict,
    date_str: str,
    dry_run: bool = False,
) -> bool:
    date_disp = f"{date_str[4:6]}/{date_str[6:8]}"
    verdict   = result["verdict"]
    top10     = result.get("top10", [])

    tweet = format_daily_tweet(result, date_str)

    # メール本文（詳細版）
    lines = [
        f"=== AI転がし日記 {date_disp} ===",
        f"判定: {verdict}  理由: {result['reason']}",
        "",
        "── 転がし候補 TOP10 ──",
        f"{'順位':<4}{'レース':<10}{'本命':<8}{'適性':<6}{'信頼度':<6}{'配当期待':<8}{'想定オッズ':<10}{'判定':<6}理由",
        "-" * 80,
    ]
    for i, s in enumerate(top10, 1):
        lines.append(
            f"{i:<4}{s.venue}{s.race}R{'':<3}{s.lane}号艇{s.racer_name:<5}"
            f"{s.fitness:<6.1f}{s.reliability:<6.1f}{s.payout_score:<8.1f}"
            f"{s.odds_est:<10.1f}{s.verdict:<6}{s.reason}"
        )
    lines += [
        "",
        "── スコア内訳（1位） ──",
    ]
    if top10:
        t = top10[0]
        d = t.score_detail if hasattr(t, 'score_detail') else {}
        lines += [
            f"  展示タイム: {t.sc_ex_time:.1f}pt",
            f"  平均ST:     {t.sc_avg_st:.1f}pt",
            f"  モーター:   {t.sc_motor:.1f}pt",
            f"  直近成績:   {t.sc_recent:.1f}pt",
            f"  コース勝率: {t.sc_course_win:.1f}pt",
            f"  イン信頼度: {t.sc_in_trust:.1f}pt（逆転）",
            f"  相手関係:   {t.sc_rival:.1f}pt",
            f"  期待値:     {t.sc_odds_ev:.1f}pt",
            f"  気象:       {t.sc_weather:.1f}pt",
            f"  危険要因:   {' / '.join(t.dangers) if t.dangers else 'なし'}",
        ]
    lines += ["", "── X投稿用テキスト ──", "", tweet]

    subject = f"🤖 AI転がし日記 {date_disp} [{verdict}] 候補{len(top10)}件"
    body    = "\n".join(lines)

    if dry_run:
        print("=" * 60)
        print(f"[DRY RUN] 件名: {subject}")
        print(body)
        print("=" * 60)
        return True

    gmail_address = os.getenv("GMAIL_ADDRESS", "")
    app_password  = os.getenv("GMAIL_APP_PASS", "")
    if not gmail_address or not app_password:
        log.error("GMAIL_ADDRESS / GMAIL_APP_PASS が未設定")
        return False

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"]    = gmail_address
    msg["To"]      = MAIL_TO
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo(); smtp.starttls()
            smtp.login(gmail_address, app_password)
            smtp.sendmail(gmail_address, MAIL_TO, msg.as_string())
        log.info("[送信] 成功: %s", subject)
        return True
    except smtplib.SMTPException as e:
        log.error("[送信] 失敗: %s", e)
        return False


# ════════════════════════════════════════════════════════════
# エントリポイント
# ════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="AI転がし専用エンジン")
    parser.add_argument("--generate", action="store_true", help="転がし候補を生成")
    parser.add_argument("--date",     help="対象日 YYYYMMDD（省略時は今日）")
    parser.add_argument("--dry-run",  action="store_true", help="送信せず表示のみ")
    parser.add_argument("--top",      type=int, default=10, help="表示件数（デフォルト10）")
    args = parser.parse_args()

    race_date = args.date or datetime.now(JST).strftime("%Y%m%d")

    if args.generate:
        # motor_history があれば読み込む
        history = {}
        try:
            from x_ranking import load_motor_history
            history = load_motor_history()
        except Exception:
            pass

        log.info("[転がし] スキャン開始: %s", race_date)
        scores  = scan_all_races(race_date, history)
        log.info("[転がし] %d件スキャン完了", len(scores))

        result  = daily_verdict(scores)

        # JSON保存
        cache_data = {
            "date":        race_date,
            "verdict":     result["verdict"],
            "reason":      result["reason"],
            "buy_count":   result["buy_count"],
            "watch_count": result["watch_count"],
            "top10":       scores_to_dict(result["top10"]),
            "top1":        scores_to_dict([result["top1"]])[0] if result["top1"] else None,
        }
        save_cache(cache_data)

        # 標準出力
        tweet = format_daily_tweet(result, race_date)
        print(tweet)
        print()
        log.info(
            "[完了] 判定: %s  購入候補: %d件  注意: %d件",
            result["verdict"], result["buy_count"], result["watch_count"],
        )

        # メール送信
        ok = send_korogashi_mail(result, race_date, dry_run=args.dry_run)
        sys.exit(0 if ok else 1)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
