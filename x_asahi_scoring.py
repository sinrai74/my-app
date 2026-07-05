#!/usr/bin/env python3
"""
x_asahi_scoring.py  ── 朝刊AI (Asahi AI) スコアリングエンジン Ver4

previews API由来のデータ（展示タイム・展示ST・風速・風向・波高）を
一切使用せず、朝時点で確定する以下のデータのみでスコアを算出する。

  ・全国勝率 / 当地勝率 (win_rate / local_win)
  ・モーター2連率 (motor)
  ・平均ST実績 (avg_st)
  ・級別・級別推移・能力指数推移 (racer_class, class_prev*, ability_*)
  ・コース別ST実績・ST順位実績・1着率・2連対率・3連対率・平均着順
    （course_st, course_rank, course_win_rate, course_place_counts / fanファイル）
  ・コース別F率・L率（フライング・出遅れリスク / fanファイル）
  ・開催場×コース別統計（場補正、x_venue_stats.py で実データから自動算出）
  ・レースグレード (race_grade)
  ・ナイター場かどうか (is_night)

重み・閾値は asahi_config.json で一元管理し、コードに固定値を書かない。
展示・気象データは本モジュールの対象外（教師データとしては別途
hit_record.csv に保存し、Phase2の学習・検証にのみ用いる）。

【Ver4での変更】危険艇速報・買い目生成(BuyScore経由)・万舟警報・新聞・
AI実績ページのすべてが本モジュールの calc_danger_score_v2 /
calc_rank_index_v2 を単一の評価基盤として参照する。
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Optional

from x_venue_stats import (
    compute_venue_course_stats,
    get_corrected_venue_course_stat,
    get_venue_course_factor,
    classify_water_type,
)

log = logging.getLogger("x_asahi_scoring")

ASAHI_CONFIG_FILE = "asahi_config.json"

# 会場番号 → 場名（notify_arashi.py の VENUE_NAMES と同一。
# 循環import回避のためここにも定義する。場統計(x_venue_stats)の
# キーは場名文字列のため、venue_num からの変換に使う。）
VENUE_NAMES: dict[int, str] = {
    1: "桐生",   2: "戸田",   3: "江戸川", 4: "平和島", 5: "多摩川",
    6: "浜名湖", 7: "蒲郡",   8: "常滑",   9: "津",    10: "三国",
    11: "びわこ",12: "住之江",13: "尼崎",  14: "鳴門", 15: "丸亀",
    16: "児島",  17: "宮島",  18: "徳山",  19: "下関", 20: "若松",
    21: "芦屋",  22: "福岡",  23: "唐津",  24: "大村",
}

# 会場ごとの統計的な荒れやすさ（朝時点で分かる過去統計。previews由来ではない）
VENUE_UPSET_FACTOR: dict[int, float] = {
    1:  0.10, 2:  0.05, 3:  0.15, 4:  0.10, 5:  0.00, 6: -0.05,
    7:  0.00, 8: -0.05, 9: -0.10, 10: 0.00, 11: 0.00, 12:-0.05,
    13: 0.00, 14: 0.00, 15: 0.00, 16:-0.10, 17: 0.10, 18: 0.00,
    19: 0.05, 20: 0.05, 21: 0.00, 22: 0.00, 23: 0.05, 24:-0.15,
}

# レースグレードによる荒れ補正（グレードが高いほど手堅くなる傾向）
GRADE_EFFECTS: dict[int, float] = {0: 0.0, 1: -0.1, 2: -0.2, 3: -0.3, 4: -0.5}


# ════════════════════════════════════════════════════════════
# config 読み込み
# ════════════════════════════════════════════════════════════

_CONFIG_CACHE: Optional[dict] = None


def load_asahi_config(force_reload: bool = False) -> dict:
    """asahi_config.json を読み込む。存在しない場合は最低限のデフォルトを返す。"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None and not force_reload:
        return _CONFIG_CACHE

    defaults = {
        "model_version": "asahi-v4.0-unified-engine",
        "danger_score": {
            "total_scale": 100,
            "relative_weights": {
                "win_rate":      {"per_boat": 4.0, "max_weight": 20},
                "local_win":     {"per_boat": 2.0, "max_weight": 10},
                "avg_st":        {"per_boat": 2.4, "max_weight": 12},
                "motor":         {"per_boat": 3.2, "max_weight": 16},
                "racer_class":   {"per_boat": 2.0, "max_weight": 10},
                "ability_trend": {"per_boat": 1.6, "max_weight": 8},
                "course_rentai": {"per_boat": 2.4, "max_weight": 12},
            },
            "solo_weights": {
                "win_rate_low":      {"weight": 2},
                "local_win_low":     {"weight": 1},
                "motor_bad":         {"weight": 2},
                "avg_st_slow":       {"weight": 2},
                "course1_place_low": {"weight": 1},
                "f_risk":            {"weight": 2},
                "venue_unfavorable": {"weight": 2},
            },
            "solo_thresholds": {
                "win_rate_low_abs": 5.0, "local_win_low_abs": 5.0,
                "motor_bad_abs": 33.0, "avg_st_slow_abs": 0.17,
                "course1_place_low_abs": 45.0,
                "f_risk_rate_abs": 8.0, "venue_unfavorable_factor_abs": 0.85,
            },
            "class_rank": {"A1": 4, "A2": 3, "B1": 2, "B2": 1, "": 0},
            "thresholds": {
                "win_rate_low_ratio": 0.85, "motor_bad_ratio": 0.90,
                "avg_st_slow_1": 0.18, "avg_st_slow_2": 0.16,
                "course_st_slow_1": 0.18, "course_st_slow_2": 0.16,
                "course_rank_bad_1": 4.0, "course_rank_bad_2": 3.0,
            },
        },
        "upset_prob": {
            "effects": {
                "st_risk_avg_st": {"value": 0.30}, "st_risk_course_st": {"value": 0.25},
                "win_rate_risk_1": {"value": 0.50}, "win_rate_risk_2": {"value": 0.20},
                "motor_risk": {"value": 0.20}, "class_risk": {"value": 0.30},
                "boat2_strength_mult": {"value": 2.0}, "class_gap_effect": {"value": 0.40},
            },
            "night_factor": -0.08,
        },
        "boat_relative_score": {
            "weights": {
                "avg_st_component":    {"weight": 1.5},
                "course_st_component": {"weight": 1.3},
                "motor_component":     {"weight": 1.2},
                "win_rate_component":  {"weight": 0.7},
                "lane_weight": {"1": 1.6, "2": 1.2, "3": 0.9, "4": 0.5, "5": -0.2, "6": -0.6},
            },
        },
        "lane_rank_scores": {
            "lane_base_rate": {
                "1": {"first": 55, "second": 16, "third": 12},
                "2": {"first": 14, "second": 17, "third": 14},
                "3": {"first": 12, "second": 16, "third": 15},
                "4": {"first": 10, "second": 17, "third": 16},
                "5": {"first": 6,  "second": 14, "third": 17},
                "6": {"first": 3,  "second": 10, "third": 13},
            },
            "first_place": {
                "base_rate_weight": 1.0, "win_rate_weight": 0.9, "local_win_weight": 0.3,
                "motor_weight": 0.7, "avg_st_weight": 0.7,
                "course_st_weight": 0.5, "course_perf_weight": 0.3, "class_weight": 0.4,
                "ability_trend_weight": 0.3, "venue_factor_weight": 3.0, "f_risk_weight": -0.5,
                "danger_penalty_weight": 1.0,
            },
            "second_place": {
                "base_rate_weight": 1.0, "win_rate_weight": 0.5, "local_win_weight": 0.2,
                "motor_weight": 0.5, "avg_st_weight": 0.2,
                "course_st_weight": 0.3, "course_perf_weight": 0.2, "class_weight": 0.2,
                "ability_trend_weight": 0.2, "venue_factor_weight": 1.5, "f_risk_weight": -0.2,
                "danger_penalty_weight": 0.15,
            },
            "third_place": {
                "base_rate_weight": 1.0, "win_rate_weight": 0.3, "local_win_weight": 0.1,
                "motor_weight": 0.4, "avg_st_weight": 0.1,
                "course_st_weight": 0.2, "course_perf_weight": 0.1, "class_weight": 0.1,
                "ability_trend_weight": 0.1, "venue_factor_weight": 1.0, "f_risk_weight": -0.1,
                "danger_penalty_weight": 0.0,
            },
            "lane6_first_place_conditions": {
                "min_match_index": 85, "min_upset_score": 8.5, "normal_suppress_ratio": 0.35,
            },
        },
        "featured_racers": {
            "max_count": 3,
            "marks": ["◎", "○", "▲"],
            "exclude_boat1_threshold": 40,
            "min_composite_threshold": 25.0,
            "composite_weights": {
                "top1_weight": 0.5, "top2_weight": 0.3, "top3_weight": 0.2,
            },
        },
    }

    if not os.path.exists(ASAHI_CONFIG_FILE):
        log.warning("[朝刊AI] %s が見つかりません → デフォルト設定を使用", ASAHI_CONFIG_FILE)
        _CONFIG_CACHE = defaults
        return defaults

    try:
        with open(ASAHI_CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # デフォルトとマージ（欠けているキーを補完）
        for section, val in defaults.items():
            if section not in data:
                data[section] = val
        _CONFIG_CACHE = data
        return data
    except Exception as e:
        log.warning("[朝刊AI] config読み込み失敗: %s → デフォルト使用", e)
        _CONFIG_CACHE = defaults
        return defaults


def get_model_version() -> str:
    return load_asahi_config().get("model_version", "unknown")


# ════════════════════════════════════════════════════════════
# ① 危険艇速報用: danger_score（100点満点、朝データのみ）
# ════════════════════════════════════════════════════════════

def calc_danger_score_v2(
    boat1, all_boats: list, config: Optional[dict] = None, venue: Optional[str] = None,
) -> tuple[float, dict]:
    """
    1号艇の危険度を100点満点で算出する（朝取得可能データのみ）Ver4。

    【設計】相対評価を中心とし、絶対評価（単体）を補助的に加える。
      ・relative: 7指標（全国勝率・当地勝率・平均ST・モーター・級別・
        能力指数推移・進入コース別2連対率）について「1号艇より優れている
        他艇の数」に応じて加点する（per_boat × 艇数）。
      ・solo: 1号艇単体の値が絶対的に低い場合の小さな加点
        （F率リスク・場不利を含む）。
    venue（場名）を渡すと、進入コース別2連対率に場補正
    （x_venue_stats.get_venue_course_factor）がかかる。渡さない場合は
    場補正なし（全国統計のみ）で算出する。

    戻り値: (score, breakdown)
      breakdown: {
        "win_rate": {"weighted": 12.3, "worse_count": 4, "worse_total": 5, "kind": "relative"},
        ...
        "win_rate_low": {"weighted": 2.0, "kind": "solo"},
        ...
      }
      各項目に "weighted"（加点） と、relative項目には "worse_count"/"worse_total"
      （劣っている艇数／比較対象艇数）を含む。新聞のAI理由文生成に使う。
    """
    cfg = config or load_asahi_config()
    ds_cfg = cfg["danger_score"]
    RW = {k: v for k, v in ds_cfg["relative_weights"].items() if k != "_comment"}
    SW = {k: v for k, v in ds_cfg["solo_weights"].items() if k != "_comment"}
    STH = ds_cfg["solo_thresholds"]
    class_rank = ds_cfg.get("class_rank", {"A1": 4, "A2": 3, "B1": 2, "B2": 1, "": 0})

    breakdown: dict[str, dict] = {}

    if boat1 is None or len(all_boats) < 2:
        return 0.0, breakdown

    other_boats = [b for b in all_boats if b.lane != 1]
    if not other_boats:
        return 0.0, breakdown

    n_other = len(other_boats)
    venue_stats = compute_venue_course_stats() if venue else None

    def _relative_item(key: str, boat1_val: float, other_vals: list[float], higher_is_better: bool = True) -> None:
        """1号艇より優れている他艇数を数え、per_boat×艇数を加点する。"""
        w = RW.get(key)
        if not w:
            return
        if higher_is_better:
            worse_count = sum(1 for v in other_vals if v > boat1_val)
        else:
            # ST等、値が小さいほど良い指標
            worse_count = sum(1 for v in other_vals if v < boat1_val)
        weighted = round(min(w["max_weight"], w["per_boat"] * worse_count), 2)
        breakdown[key] = {
            "weighted": weighted,
            "worse_count": worse_count,
            "worse_total": len(other_vals),
            "kind": "relative",
        }

    # ── ①全国勝率（相対） ─────────────────────────────────────
    _relative_item("win_rate", boat1.win_rate or 0.0,
                    [b.win_rate or 0.0 for b in other_boats], higher_is_better=True)

    # ── ②当地勝率（相対） ─────────────────────────────────────
    _relative_item("local_win", boat1.local_win or 0.0,
                    [b.local_win or 0.0 for b in other_boats], higher_is_better=True)

    # ── ③平均ST実績（相対、小さいほど良い） ───────────────────
    _relative_item("avg_st", boat1.avg_st or 0.18,
                    [b.avg_st or 0.18 for b in other_boats], higher_is_better=False)

    # ── ④モーター2連率（相対） ─────────────────────────────────
    _relative_item("motor", boat1.motor or 0.0,
                    [b.motor or 0.0 for b in other_boats], higher_is_better=True)

    # ── ⑤級別（相対） ─────────────────────────────────────────
    boat1_class_rank = class_rank.get(boat1.racer_class or "", 0)
    other_class_ranks = [class_rank.get(b.racer_class or "", 0) for b in other_boats]
    _relative_item("racer_class", boat1_class_rank, other_class_ranks, higher_is_better=True)

    # ── ⑥【Ver4新規】能力指数推移（相対、今期-前期の伸び） ──────
    def _ability_trend(b) -> float:
        if b.ability_curr and b.ability_prev:
            return b.ability_curr - b.ability_prev
        return 0.0
    _relative_item("ability_trend", _ability_trend(boat1),
                    [_ability_trend(b) for b in other_boats], higher_is_better=True)

    # ── ⑦【Ver4新規】進入コース別2連対率（相対、信頼度補正・場補正込み） ──
    def _course_rentai(b) -> float:
        lane_idx = b.lane - 1
        if not (b.course_nyuko and 0 <= lane_idx < 6 and b.course_nyuko[lane_idx] > 0):
            return 0.0
        counts = b.course_place_counts[lane_idx] if b.course_place_counts else None
        nyuko = b.course_nyuko[lane_idx]
        if counts and sum(counts) > 0:
            rentai2 = (counts[0] + counts[1]) / nyuko * 100
        else:
            # 着順回数まで取得できていない場合は複勝率（補助情報）で近似
            rentai2 = b.course_place_rate[lane_idx] if b.course_place_rate else 0.0
        if venue and venue_stats:
            factor = get_venue_course_factor(venue, b.lane, venue_stats, cfg.get("_venue_cfg"))["factor"]
            rentai2 *= factor
        return rentai2
    _relative_item("course_rentai", _course_rentai(boat1),
                    [_course_rentai(b) for b in other_boats], higher_is_better=True)

    # ── 単体評価（絶対値が低い場合の小さな加点） ─────────────
    def _solo_item(key: str, cond: bool) -> None:
        w = SW.get(key)
        if not w or not cond:
            return
        breakdown[key] = {"weighted": float(w["weight"]), "kind": "solo"}

    _solo_item("win_rate_low",  (boat1.win_rate or 0.0) < STH["win_rate_low_abs"])
    _solo_item("local_win_low", (boat1.local_win or 0.0) < STH["local_win_low_abs"])
    _solo_item("motor_bad",     (boat1.motor or 0.0) < STH["motor_bad_abs"])
    _solo_item("avg_st_slow",   (boat1.avg_st or 0.0) >= STH["avg_st_slow_abs"])

    # 1コース1着率（fanファイル由来、展示STは使用しない＝前期実績のみ）
    # 【調査結果】fanファイルのコース別「着順回数」ブロック(1着回数)から
    # 1着回数÷進入回数で真の1着率を算出できることを確認済み（複勝率は代替不要）。
    course1_win = None
    if boat1.course_win_rate and boat1.course_nyuko and boat1.course_nyuko[0] > 0:
        course1_win = boat1.course_win_rate[0]
    if course1_win is not None:
        _solo_item("course1_place_low", course1_win < STH["course1_place_low_abs"])

    # ── 【Ver4新規】1コースFリスク（単体） ────────────────────
    if boat1.course_nyuko and boat1.course_nyuko[0] > 0 and boat1.course_f_count:
        f_rate = boat1.course_f_count[0] / boat1.course_nyuko[0] * 100
        _solo_item("f_risk", f_rate >= STH["f_risk_rate_abs"])

    # ── 【Ver4新規】場不利（単体、選手能力×場特性） ─────────────
    if venue:
        vf = get_venue_course_factor(venue, 1, venue_stats)
        _solo_item("venue_unfavorable", vf["factor"] < STH["venue_unfavorable_factor_abs"])

    total = round(sum(v["weighted"] for v in breakdown.values()), 2)
    total = min(total, ds_cfg["total_scale"])
    return total, breakdown


# ════════════════════════════════════════════════════════════
# ② 各艇の相対スコア（万舟警報の確率計算の土台、朝データのみ）
# ════════════════════════════════════════════════════════════

def calc_boat_score_v2(boat, all_boats: list, config: Optional[dict] = None) -> float:
    """
    1艇の相対的な強さスコアを算出する（展示・気象は一切使わない）。
    calculate_upset_score_v2 の内部で各艇の確率化に使う。
    """
    cfg = config or load_asahi_config()
    brs = cfg["boat_relative_score"]["weights"]
    score = 0.0

    # 平均ST実績（速いほど加点。基準0.17秒を中心に線形）
    if boat.avg_st:
        avg_st_score = (0.17 - boat.avg_st) * 10.0  # 0.17より速ければ+、遅ければ-
        score += avg_st_score * brs["avg_st_component"]["weight"]

    # コース別ST実績（進入コースでの実績、あれば使う）
    lane_idx = boat.lane - 1
    if boat.course_nyuko and 0 <= lane_idx < 6 and boat.course_nyuko[lane_idx] > 0:
        c_st = boat.course_st[lane_idx]
        c_st_score = (0.17 - c_st) * 10.0
        score += c_st_score * brs["course_st_component"]["weight"]

    # モーター2連率（全艇平均比）
    motors = [b.motor for b in all_boats if b.motor is not None]
    if motors and boat.motor is not None:
        avg_motor = sum(motors) / len(motors)
        if avg_motor > 0:
            motor_score = (boat.motor - avg_motor) / avg_motor * 5.0
            score += motor_score * brs["motor_component"]["weight"]

    # 全国勝率
    if boat.win_rate is not None:
        score += (boat.win_rate - 5.0) * brs["win_rate_component"]["weight"]

    # コース（進入枠）による基礎優位性
    lane_weight = brs.get("lane_weight", {})
    score += lane_weight.get(str(boat.lane), 0.0)

    return score


# ════════════════════════════════════════════════════════════
# ③ 荒れ確率（万舟警報の核心ロジック、朝データのみ）
# ════════════════════════════════════════════════════════════

def calculate_upset_score_v2(
    boats: list,
    race_grade: int = 0,
    venue_num: int = 0,
    is_night: bool = False,
    config: Optional[dict] = None,
) -> tuple[float, dict, list[int]]:
    """
    朝データのみで「1号艇が飛ぶ確率」を計算する。
    戻り値は既存の calculate_upset_score と同じ形式:
      (upset_score, detail_dict, target_lanes)
    展示・気象データは一切参照しない。
    """
    cfg = config or load_asahi_config()
    up_cfg = cfg["upset_prob"]
    EFF = {k: v["value"] for k, v in up_cfg["effects"].items()}

    def logit(p: float) -> float:
        p = max(0.001, min(0.999, p))
        return math.log(p / (1 - p))

    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    def add_effect(base_prob: float, effect: float) -> float:
        return sigmoid(logit(base_prob) + effect)

    # ── 各艇のベース確率計算（展示・気象なし） ─────────────
    raw_scores = [(b.lane, calc_boat_score_v2(b, boats, cfg)) for b in boats]
    max_s = max(s for _, s in raw_scores)
    exp_scores = [(lane, math.exp(s - max_s)) for lane, s in raw_scores]
    total = sum(e for _, e in exp_scores)
    lane_probs = {lane: e / total for lane, e in exp_scores}

    boat1_prob = lane_probs.get(1, 1 / 6)
    boat1 = next((b for b in boats if b.lane == 1), None)
    boat2 = next((b for b in boats if b.lane == 2), None)

    other_probs = {l: p for l, p in lane_probs.items() if l != 1}
    best_other_lane = max(other_probs, key=other_probs.get) if other_probs else 2
    best_other_prob = other_probs.get(best_other_lane, 0.0)

    upset_prob = 1.0 - boat1_prob
    venue_name = VENUE_NAMES.get(venue_num)
    danger_score, danger_breakdown = calc_danger_score_v2(boat1, boats, cfg, venue=venue_name)

    # ── ①1号艇の弱さ（平均ST実績・コース別ST実績） ────────
    if boat1:
        st_risk = 0.0
        if boat1.avg_st and boat1.avg_st > 0.17:
            st_risk += EFF["st_risk_avg_st"]
        lane_idx = boat1.lane - 1
        if boat1.course_nyuko and boat1.course_nyuko[0] > 0 and boat1.course_st[0] > 0.16:
            st_risk += EFF["st_risk_course_st"]
        if st_risk > 0:
            upset_prob = add_effect(upset_prob, st_risk)

        wr_risk = 0.0
        if boat1.win_rate < 5.0:   wr_risk += EFF["win_rate_risk_1"]
        elif boat1.win_rate < 5.5: wr_risk += EFF["win_rate_risk_2"]
        if wr_risk > 0:
            upset_prob = add_effect(upset_prob, wr_risk)

        eq_risk = 0.0
        if boat1.motor < 35.0:                eq_risk += EFF["motor_risk"]
        if boat1.racer_class in ("B1", "B2"): eq_risk += EFF["class_risk"]
        if eq_risk > 0:
            upset_prob = add_effect(upset_prob, eq_risk)

    # ── ②2号艇の強さ ─────────────────────────────────────────
    boat2_prob = lane_probs.get(2, 0.0)
    if boat2 and boat2_prob > boat1_prob:
        upset_prob = add_effect(upset_prob, (boat2_prob - boat1_prob) * EFF["boat2_strength_mult"])

    # ── ③会場・グレード・ナイター補正（統計的傾向、朝データ）─
    venue_factor = VENUE_UPSET_FACTOR.get(venue_num, 0.0)
    if venue_factor != 0.0:
        upset_prob = add_effect(upset_prob, venue_factor)

    if is_night:
        upset_prob = add_effect(upset_prob, up_cfg.get("night_factor", -0.08))

    # ── ④等級差 ──────────────────────────────────────────────
    if boat1 and boat1.racer_class in ("B1", "B2"):
        if any(b.lane != 1 and b.racer_class == "A1" for b in boats):
            upset_prob = add_effect(upset_prob, EFF["class_gap_effect"])

    # ── ⑤レース種別補正 ──────────────────────────────────────
    upset_prob = add_effect(upset_prob, GRADE_EFFECTS.get(race_grade, 0.0))

    upset_prob = max(0.0, min(upset_prob, 0.95))
    upset_score = upset_prob * 10.0

    # danger_score が高い場合は upset_score を強化
    if danger_score >= 60:
        upset_score = min(upset_score * 1.30, 9.5)
    elif danger_score >= 40:
        upset_score = min(upset_score * 1.15, 9.5)

    if boat1_prob > 0.65 and best_other_prob < boat1_prob:
        upset_score = 0.0

    # ── 1号艇等級フィルタ ─────────────────────────────────────
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
        "荒れ確率":     f"{upset_prob:.1%}",
        "1号艇確率":    f"{boat1_prob:.1%}(rank{boat1_rank})",
        "1号艇危険度":  f"{danger_score:.1f}",
        "最有力":       f"{best_other_lane}号艇({best_other_prob:.1%})",
        "レース種別":   grade_names.get(race_grade, f'grade{race_grade}'),
        "モデルVersion": cfg.get("model_version", "unknown"),
    }
    if grade_filter_note:
        detail["等級フィルタ"] = grade_filter_note

    # 学習・検証用に特徴量内訳も detail に埋め込む（hit_record.csv保存時に利用）
    # ⑦統一: 上位進出指数・注目選手も同じ boats・config から算出し、
    # 危険艇速報・買い目生成・note新聞ですべて同じ値を参照する。
    rank_index_ctx = {"match_index": min(100, upset_score * 10.5), "upset_score": upset_score}
    rank_index = calc_rank_index_v2(boats, rank_index_ctx, cfg, venue=venue_name)
    featured_boats = select_featured_boats(boats, rank_index, danger_score, cfg)

    # ── Ver4追加: 水面タイプ・場補正・能力推移・F率/L率・コース連対率・
    #    サンプル数信頼度（いずれも1号艇について。hit_record.csv保存用） ──
    venue_water_type = ""
    venue_factor_val = ""
    course_sample_confidence = ""
    if venue_name:
        wt = classify_water_type(venue_name)
        venue_water_type = wt["label"]
        vf = get_venue_course_factor(venue_name, 1)
        venue_factor_val = vf["factor"]
        course_sample_confidence = vf["samples"]

    ability_trend_1c = ""
    if boat1 and boat1.ability_curr and boat1.ability_prev:
        ability_trend_1c = round(boat1.ability_curr - boat1.ability_prev, 2)

    course_f_rate_1c = ""
    course_l_rate_1c = ""
    course_rentai2_1c = ""
    if boat1 and boat1.course_nyuko and boat1.course_nyuko[0] > 0:
        nyuko0 = boat1.course_nyuko[0]
        if boat1.course_f_count:
            course_f_rate_1c = round(boat1.course_f_count[0] / nyuko0 * 100, 1)
        if boat1.course_l_count:
            course_l_rate_1c = round(boat1.course_l_count[0] / nyuko0 * 100, 1)
        if boat1.course_place_counts and sum(boat1.course_place_counts[0]) > 0:
            counts0 = boat1.course_place_counts[0]
            course_rentai2_1c = round((counts0[0] + counts0[1]) / nyuko0 * 100, 1)

    detail["_features"] = {
        "win_rate":      boat1.win_rate if boat1 else None,
        "motor":         boat1.motor if boat1 else None,
        "avg_st":        boat1.avg_st if boat1 else None,
        "racer_class":   boat1.racer_class if boat1 else None,
        "course_st_1c":  boat1.course_st[0] if boat1 and boat1.course_nyuko and boat1.course_nyuko[0] > 0 else None,
        "course_rank_1c": boat1.course_rank[0] if boat1 and boat1.course_nyuko and boat1.course_nyuko[0] > 0 else None,
        "danger_breakdown": danger_breakdown,
        "danger_score_v3": danger_score,
        "rank_index": rank_index,
        "featured_boats": featured_boats,
        "model_version": cfg.get("model_version", "unknown"),
        # Ver4
        "venue_water_type": venue_water_type,
        "venue_factor": venue_factor_val,
        "ability_trend": ability_trend_1c,
        "course_f_rate_1c": course_f_rate_1c,
        "course_l_rate_1c": course_l_rate_1c,
        "course_rentai2_1c": course_rentai2_1c,
        "course_sample_confidence": course_sample_confidence,
    }

    return upset_score, detail, target


# ════════════════════════════════════════════════════════════
# ④ 着順別評価（1着・2着・3着を個別に算出する）
# ════════════════════════════════════════════════════════════
#
# 【設計背景】
# 従来は単一のスコア(calc_boat_score_v2)をsoftmax化した1つの確率分布を
# 1着・2着・3着すべてに流用していたため、以下の歪みが生じていた:
#   ・危険艇判定で1号艇の1着適性を下げると、2着・3着適性まで連動して
#     下がってしまい、1号艇が舟券圏内(2-3着)からも過度に排除される
#   ・6号艇は艇番による基礎優位性が低いだけなのに、モーター・勝率が
#     良いと1着候補になりやすく、実際の competitin における6号艇の
#     低い1着率と乖離する
#
# 本関数群は、艇番別の実決着率（lane_base_rate）を土台に、
# 1着＝勝ち切る能力、2着＝連対能力、3着＝舟券絡み率という
# 異なる評価軸で独立にスコアを算出する。
# 危険艇判定（1号艇の1着適性低下）は1着評価にのみ強く反映し、
# 2着・3着評価への影響は danger_penalty_weight で個別に抑制する。

def calc_lane_rank_scores_v2(
    boat,
    all_boats: list,
    config: Optional[dict] = None,
    venue: Optional[str] = None,
) -> dict:
    """
    1艇について、1着適性・2着適性・3着適性を個別に算出する。Ver4。

    戻り値: {
      "first": float, "second": float, "third": float,   # 相対スコア（従来通り）
      "contributions": {
        "first":  {"win_rate": x, "local_win": x, "avg_st": x, "motor": x,
                    "class": x, "course_perf": x, "ability_trend": x,
                    "venue_factor": x, "f_risk": x, "other": x},
        "second": {...}, "third": {...}
      }
    }
    contributions は各特徴量が最終スコアへ与えた加点（寄与度）。
    将来の特徴量分析・重み学習（Phase2以降）のため hit_record.csv に保存する。
    venue（場名）を渡すと場補正（x_venue_stats）が寄与度に反映される。
    """
    cfg = config or load_asahi_config()
    lrs = cfg["lane_rank_scores"]
    base_rates = lrs["lane_base_rate"].get(
        str(boat.lane), {"first": 10, "second": 15, "third": 15}
    )

    # 全艇平均比で個別指標を評価する（相対評価にすることで艇番間の
    # スケールのブレを吸収する）
    win_rates  = [b.win_rate  for b in all_boats if b.win_rate  is not None]
    local_wins = [b.local_win for b in all_boats if b.local_win is not None]
    motors     = [b.motor     for b in all_boats if b.motor     is not None]
    avg_win_rate  = sum(win_rates)  / len(win_rates)  if win_rates  else 5.0
    avg_local_win = sum(local_wins) / len(local_wins) if local_wins else 5.0
    avg_motor     = sum(motors)     / len(motors)     if motors    else 33.0

    win_rate_diff  = (boat.win_rate  - avg_win_rate)  if boat.win_rate  is not None else 0.0
    local_win_diff = (boat.local_win - avg_local_win) if boat.local_win is not None else 0.0
    motor_diff     = (boat.motor     - avg_motor)     if boat.motor     is not None else 0.0
    # 平均ST実績: 0.17秒を基準に、速いほどプラスに寄与する
    avg_st_diff = (0.17 - boat.avg_st) * 10.0 if boat.avg_st else 0.0

    # コース別ST実績（進入コースでの実績、あれば使う。fanファイル由来）
    lane_idx = boat.lane - 1
    course_st_diff = 0.0
    if boat.course_nyuko and 0 <= lane_idx < 6 and boat.course_nyuko[lane_idx] > 0:
        course_st_diff = (0.17 - boat.course_st[lane_idx]) * 10.0

    # コース別1着率実績（全国平均(lane_base_rateのfirst%)比）
    course_perf_diff = 0.0
    if boat.course_nyuko and 0 <= lane_idx < 6 and boat.course_nyuko[lane_idx] > 0 \
            and boat.course_win_rate:
        national_base = lrs["lane_base_rate"].get(str(boat.lane), {}).get("first", 0.0)
        course_perf_diff = (boat.course_win_rate[lane_idx] - national_base) / 10.0

    # 級別（全艇平均比）
    class_rank_map = cfg.get("danger_score", {}).get("class_rank", {"A1": 4, "A2": 3, "B1": 2, "B2": 1, "": 0})
    class_ranks = [class_rank_map.get(b.racer_class or "", 0) for b in all_boats]
    avg_class_rank = sum(class_ranks) / len(class_ranks) if class_ranks else 0.0
    class_diff = class_rank_map.get(boat.racer_class or "", 0) - avg_class_rank

    # 【Ver4新規】能力指数推移（全艇平均比、今期-前期の伸び）
    def _ability_trend(b) -> float:
        if b.ability_curr and b.ability_prev:
            return b.ability_curr - b.ability_prev
        return 0.0
    ability_trends = [_ability_trend(b) for b in all_boats]
    avg_ability_trend = sum(ability_trends) / len(ability_trends) if ability_trends else 0.0
    ability_trend_diff = _ability_trend(boat) - avg_ability_trend

    # 【Ver4新規】場補正（選手能力×場特性。1.0=全国平均並み、それとの差分）
    venue_factor_diff = 0.0
    if venue and boat.course_nyuko and 0 <= lane_idx < 6 and boat.course_nyuko[lane_idx] > 0:
        vf = get_venue_course_factor(venue, boat.lane)
        venue_factor_diff = vf["factor"] - 1.0

    # 【Ver4新規】Fリスク（進入コースでのフライング率%。高いほどマイナス寄与）
    f_rate = 0.0
    if boat.course_nyuko and 0 <= lane_idx < 6 and boat.course_nyuko[lane_idx] > 0 and boat.course_f_count:
        f_rate = boat.course_f_count[lane_idx] / boat.course_nyuko[lane_idx] * 100

    def _rank_score_with_contrib(rank_key: str, base_rate: float) -> tuple[float, dict]:
        w = lrs[rank_key]
        base = base_rate * w["base_rate_weight"]
        contrib = {
            "base_rate":     round(base, 3),
            "win_rate":      round(win_rate_diff  * w["win_rate_weight"], 3),
            "local_win":     round(local_win_diff * w.get("local_win_weight", 0.0), 3),
            "avg_st":        round(avg_st_diff    * w["avg_st_weight"], 3),
            "motor":         round(motor_diff     * w["motor_weight"], 3),
            "class":         round(class_diff     * w.get("class_weight", 0.0), 3),
            # 【寄与度カテゴリ】コース実績寄与＝コース別1着率(fanファイル)、
            # その他特徴量寄与＝コース別ST実績（avg_stと別軸のためこちらに分類）
            "course_perf":   round(course_perf_diff * w.get("course_perf_weight", 0.0), 3),
            "ability_trend": round(ability_trend_diff * w.get("ability_trend_weight", 0.0), 3),
            "venue_factor":  round(venue_factor_diff  * w.get("venue_factor_weight", 0.0), 3),
            "f_risk":        round(f_rate / 10.0 * w.get("f_risk_weight", 0.0), 3),
            "other":         round(course_st_diff   * w.get("course_st_weight", 0.0), 3),
        }
        score = sum(contrib.values())
        return score, contrib

    first_score,  first_contrib  = _rank_score_with_contrib("first_place",  base_rates["first"])
    second_score, second_contrib = _rank_score_with_contrib("second_place", base_rates["second"])
    third_score,  third_contrib  = _rank_score_with_contrib("third_place",  base_rates["third"])

    # ── 危険艇判定（1号艇のみ）を1着評価に強く反映し、2着・3着への
    #    影響は各着順ごとの danger_penalty_weight で個別に抑制する ──
    if boat.lane == 1:
        danger_score, _ = calc_danger_score_v2(boat, all_boats, cfg, venue=venue)
        # danger_score は0-100点。スコアスケール(概ね数十点)に合わせて
        # 1/10した値をペナルティ基準量とする。
        penalty_unit = danger_score / 10.0
        first_penalty  = penalty_unit * lrs["first_place"]["danger_penalty_weight"]
        second_penalty = penalty_unit * lrs["second_place"]["danger_penalty_weight"]
        third_penalty  = penalty_unit * lrs["third_place"]["danger_penalty_weight"]
        first_score  -= first_penalty
        second_score -= second_penalty
        third_score  -= third_penalty
        first_contrib["danger_penalty"]  = round(-first_penalty, 3)
        second_contrib["danger_penalty"] = round(-second_penalty, 3)
        third_contrib["danger_penalty"]  = round(-third_penalty, 3)

    return {
        "first":  max(0.1, first_score),
        "second": max(0.1, second_score),
        "third":  max(0.1, third_score),
        "contributions": {
            "first": first_contrib, "second": second_contrib, "third": third_contrib,
        },
    }


def calc_rank_probabilities_v2(
    boats: list,
    context: Optional[dict] = None,
    config: Optional[dict] = None,
    venue: Optional[str] = None,
) -> dict:
    """
    レース全体について、1着・2着・3着それぞれの確率分布を個別に算出する。

    context: {"match_index": float, "upset_score": float} を渡すと、
             6号艇1着の特別条件判定に使う（省略時は特別条件なしとして
             通常の減衰を適用する＝6号艇1着を控えめに評価する）。
    venue: 場名を渡すと場補正（x_venue_stats）が寄与度・スコアに反映される。

    戻り値: {
      "first":  {lane: prob, ...},   # 1着確率分布（6号艇は条件次第で減衰）
      "second": {lane: prob, ...},   # 2着確率分布
      "third":  {lane: prob, ...},   # 3着確率分布
      "lane6_first_allowed": bool,   # 6号艇1着の特別条件を満たしたか
    }
    """
    cfg = config or load_asahi_config()
    context = context or {}
    lrs = cfg["lane_rank_scores"]

    scores = {b.lane: calc_lane_rank_scores_v2(b, boats, cfg, venue=venue) for b in boats}

    def _softmax_normalize(score_map: dict) -> dict:
        total = sum(score_map.values())
        if total <= 0:
            n = len(score_map) or 1
            return {l: 1.0 / n for l in score_map}
        return {l: s / total for l, s in score_map.items()}

    # ── 6号艇1着の特別条件判定 ──────────────────────────────
    lane6_cond = lrs["lane6_first_place_conditions"]
    match_index = context.get("match_index", 0)
    upset_score = context.get("upset_score", 0)
    lane6_allowed = (
        match_index >= lane6_cond["min_match_index"]
        or upset_score >= lane6_cond["min_upset_score"]
    )

    first_raw = {lane: s["first"] for lane, s in scores.items()}
    if not lane6_allowed and 6 in first_raw:
        # 特別条件を満たさない通常時は、6号艇の1着スコアを大幅に減衰させる
        # （完全除外はしない＝万が一の展開まで確率0にはしない）
        first_raw[6] *= lane6_cond["normal_suppress_ratio"]

    first_probs  = _softmax_normalize(first_raw)
    second_probs = _softmax_normalize({lane: s["second"] for lane, s in scores.items()})
    third_probs  = _softmax_normalize({lane: s["third"]  for lane, s in scores.items()})

    return {
        "first": first_probs, "second": second_probs, "third": third_probs,
        "lane6_first_allowed": lane6_allowed,
    }


# ════════════════════════════════════════════════════════════
# ⑤ 上位進出指数（0-100） ── 危険艇速報・買い目生成・万舟警報で共通利用
# ════════════════════════════════════════════════════════════
#
# 展示データは使用しない。全国勝率・当地勝率・平均ST・モーター勝率・
# 級別・コース実績（fanファイル）・過去成績のみで構成する
# calc_rank_probabilities_v2 の確率分布をそのまま 0-100 の指数に変換する。
# 1着指数＝1着確率、2着以内指数＝1着+2着確率、3着以内指数＝1着+2着+3着確率。

def calc_rank_index_v2(
    boats: list,
    context: Optional[dict] = None,
    config: Optional[dict] = None,
    venue: Optional[str] = None,
) -> dict[int, dict]:
    """
    全艇について「上位進出指数」を0-100で算出する。Ver4。
    戻り値: {lane: {"top1": float, "top2": float, "top3": float,
                     "contributions": {...}}, ...}
      top1 = 1着指数（1着確率×100）
      top2 = 2着以内指数（1着+2着確率×100）
      top3 = 3着以内指数（1着+2着+3着確率×100）
      contributions = calc_lane_rank_scores_v2 が返す各特徴量の寄与度
        （全国勝率・当地勝率・平均ST・モーター・級別・コース実績・
          能力推移・場補正・Fリスク・その他）。
        将来の特徴量分析・重み学習のため hit_record.csv に保存する。
    venue: 場名を渡すと場補正（x_venue_stats）が寄与度に反映される。
    """
    cfg = config or load_asahi_config()
    probs = calc_rank_probabilities_v2(boats, context, cfg, venue=venue)

    # 寄与度は calc_lane_rank_scores_v2 から直接取得する
    # （calc_rank_probabilities_v2 の内部でも同じ関数を呼んでいるが、
    #  寄与度まではそちらの戻り値に含めていないため、ここで個別に再取得する）
    result: dict[int, dict] = {}
    for b in boats:
        lane = b.lane
        p1 = probs["first"].get(lane, 0.0)
        p2 = probs["second"].get(lane, 0.0)
        p3 = probs["third"].get(lane, 0.0)
        lane_scores = calc_lane_rank_scores_v2(b, boats, cfg, venue=venue)
        result[lane] = {
            "top1": round(p1 * 100, 1),
            "top2": round((p1 + p2) * 100, 1),
            "top3": round((p1 + p2 + p3) * 100, 1),
            "contributions": lane_scores.get("contributions", {}),
        }
    return result


# ════════════════════════════════════════════════════════════
# ⑥ 注目選手選定 ── 危険な1号艇の代わりに狙うべき艇を提示する
# ════════════════════════════════════════════════════════════

def select_featured_boats(
    boats: list,
    rank_index: dict[int, dict],
    danger_score: float,
    config: Optional[dict] = None,
) -> list[dict]:
    """
    上位進出指数から「注目選手」を最大N艇選ぶ（configで人数・マーク変更可）。

    2つの独立した閾値で判定する（④要件: 危険度だけでなく注目選手自身の
    指数も考慮する）:
      ・exclude_boat1_threshold: 1号艇を候補から除外するかどうかの判定
        （danger_score がこれ以上なら1号艇は代わりの艇を探す対象から外れる）
      ・min_composite_threshold: 注目選手として表示してよい最低ライン
        （危険度に関わらず、この指数を満たさない艇には無理に◎○▲を付けない）
    そのため、1号艇が危険でなくても強い挑戦者がいれば表示されるし、
    1号艇が危険でも代替候補全員が指数不足なら注目選手は0〜少数になる。

    戻り値: [{"lane": int, "name": str, "mark": "◎", "composite": float,
              "top1": float, "top2": float, "top3": float}, ...]
    """
    cfg = config or load_asahi_config()
    fr_cfg = cfg.get("featured_racers", {})
    max_count = fr_cfg.get("max_count", 3)
    marks = fr_cfg.get("marks", ["◎", "○", "▲"])
    exclude_th = fr_cfg.get("exclude_boat1_threshold", 40)
    min_composite = fr_cfg.get("min_composite_threshold", 0.0)
    cw = fr_cfg.get("composite_weights", {"top1_weight": 0.5, "top2_weight": 0.3, "top3_weight": 0.2})

    candidates = boats
    if danger_score >= exclude_th:
        candidates = [b for b in boats if b.lane != 1]

    scored = []
    for b in candidates:
        idx = rank_index.get(b.lane, {"top1": 0.0, "top2": 0.0, "top3": 0.0})
        composite = (
            idx["top1"] * cw.get("top1_weight", 0.5)
            + idx["top2"] * cw.get("top2_weight", 0.3)
            + idx["top3"] * cw.get("top3_weight", 0.2)
        )
        if composite < min_composite:
            continue  # 指数不足の艇には無理に◎○▲を付けない
        scored.append({
            "lane": b.lane, "name": b.name, "composite": round(composite, 1),
            "top1": idx["top1"], "top2": idx["top2"], "top3": idx["top3"],
        })

    scored.sort(key=lambda x: -x["composite"])
    top_n = scored[:max_count]
    for i, item in enumerate(top_n):
        item["mark"] = marks[i] if i < len(marks) else ""
    return top_n


# ════════════════════════════════════════════════════════════
# 共通データ構造 ── Ver4評価エンジンの統一インターフェース
# ════════════════════════════════════════════════════════════
#
# 【設計目的】
# 危険艇速報・買い目生成(BuyScore)・万舟警報・新聞・AI実績ページの
# すべてが「同じ評価基盤・同じ特徴量」を使うことを保証するため、
# calc_danger_score_v2 / calc_rank_index_v2 / select_featured_boats /
# x_venue_stats の全出力を1回の呼び出しでまとめて取得できる関数を
# ここに用意する。
#
# 【今回のスコープ】
# 本関数はこの統一構造を「提供する」ところまでを実装する。
# x_buyscore.py（BuyScore本体の判定ロジック・重み・閾値）は今回
# 一切変更しない。Ver4評価エンジン単独での効果検証を優先するため。
#
# 【将来の接続方法】
# 次フェーズでBuyScoreをVer4評価エンジンへ移行する際は、
# calc_buyscore() の呼び出し前に本関数を呼び、戻り値の
# "contributions_first"（1着適性の寄与度）や "danger_breakdown"
# （危険度内訳）を calc_buyscore() の入力特徴量として渡すだけで
# 接続できる。BuyScore側のスコア合成式・閾値には触れずに済む
# ように、本関数は「素材（特徴量・寄与度）」を渡すに留め、
# BuyScore固有の重み付け判断は一切行わない。

def build_race_evaluation_v4(
    boats: list,
    venue: Optional[str] = None,
    race_grade: int = 0,
    venue_num: int = 0,
    is_night: bool = False,
    config: Optional[dict] = None,
) -> dict:
    """
    Ver4評価エンジンの全出力を1つの共通データ構造にまとめて返す。

    危険艇速報・新聞・AI実績ページ・学習用CSV出力は、個別に
    calc_danger_score_v2 等を呼ぶのではなく、本関数の戻り値を
    参照することで「同じ評価基盤」を保証する。

    戻り値:
      {
        "model_version": str,
        "venue": str, "venue_num": int,
        "boat1": {"lane": 1, "name": str, "racer_class": str, ...},

        # 危険度（1号艇）
        "danger_score": float,              # 0-100
        "danger_breakdown": dict,           # calc_danger_score_v2 の内訳
        "water_type": {"type": str, "label": str, "course1_win_rate": float,
                        "source": str, "samples": int},
        "venue_factor_1c": dict,            # get_venue_course_factor(venue, 1) の結果

        # 上位進出指数（全艇）
        "rank_index": {lane: {"top1":, "top2":, "top3":, "contributions": {...}}, ...},

        # 注目選手
        "featured_boats": [{"lane":, "name":, "mark":, "composite":, ...}, ...],

        # 荒れ確率（万舟警報・従来のupset_score系と同一の計算式）
        "upset_score": float,
        "upset_detail": dict,

        # Ver4追加の1号艇個別特徴量（学習用CSV・BuyScore接続用の「素材」）
        "boat1_features": {
          "ability_trend": float|None,       # 今期-前期 能力指数
          "course_f_rate_1c": float|None,    # 1コースF率(%)
          "course_l_rate_1c": float|None,    # 1コースL率(%)
          "course_rentai2_1c": float|None,   # 1コース2連対率(%)
          "course_sample_confidence": int,   # 1コース進入回数（信頼度の目安）
        },
      }
    """
    cfg = config or load_asahi_config()
    boat1 = next((b for b in boats if b.lane == 1), None)

    upset_score, upset_detail, _target = calculate_upset_score_v2(
        boats, race_grade=race_grade, venue_num=venue_num, is_night=is_night, config=cfg,
    )
    feats = upset_detail.get("_features", {})

    water_type = classify_water_type(venue) if venue else {
        "type": "unknown", "label": "不明", "course1_win_rate": 0.0,
        "source": "no_venue", "samples": 0,
    }
    venue_factor_1c = get_venue_course_factor(venue, 1) if venue else {
        "factor": 1.0, "venue_win_rate": 0.0, "national_win_rate": 0.0,
        "samples": 0, "water_type": "unknown",
    }

    return {
        "model_version": cfg.get("model_version", "unknown"),
        "venue": venue or "", "venue_num": venue_num,
        "boat1": {
            "lane": 1,
            "name": boat1.name if boat1 else "",
            "racer_class": boat1.racer_class if boat1 else "",
        },
        "danger_score": feats.get("danger_score_v3", 0.0),
        "danger_breakdown": feats.get("danger_breakdown", {}),
        "water_type": water_type,
        "venue_factor_1c": venue_factor_1c,
        "rank_index": feats.get("rank_index", {}),
        "featured_boats": feats.get("featured_boats", []),
        "upset_score": upset_score,
        "upset_detail": upset_detail,
        "boat1_features": {
            "ability_trend":            feats.get("ability_trend"),
            "course_f_rate_1c":         feats.get("course_f_rate_1c"),
            "course_l_rate_1c":         feats.get("course_l_rate_1c"),
            "course_rentai2_1c":        feats.get("course_rentai2_1c"),
            "course_sample_confidence": feats.get("course_sample_confidence"),
        },
    }
