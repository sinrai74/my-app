#!/usr/bin/env python3
"""
x_asahi_scoring.py  ── 朝刊AI (Asahi AI) スコアリングエンジン

previews API由来のデータ（展示タイム・展示ST・風速・風向・波高）を
一切使用せず、朝時点で確定する以下のデータのみでスコアを算出する。

  ・全国勝率 (win_rate)
  ・モーター2連率 (motor)
  ・平均ST実績 (avg_st)
  ・級別 (racer_class)
  ・コース別ST実績・ST順位実績 (course_st, course_rank / fanファイル)
  ・開催場の統計的な荒れやすさ (venue_num)
  ・レースグレード (race_grade)
  ・ナイター場かどうか (is_night)

重み・閾値は asahi_config.json で一元管理し、コードに固定値を書かない。
展示・気象データは本モジュールの対象外（教師データとしては別途
hit_record.csv に保存し、Phase2の学習・検証にのみ用いる）。
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Optional

log = logging.getLogger("x_asahi_scoring")

ASAHI_CONFIG_FILE = "asahi_config.json"

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
        "model_version": "asahi-v1.0-phase1-default",
        "danger_score": {
            "total_scale": 100,
            "weights": {
                "win_rate_low":    {"weight": 28},
                "motor_bad":       {"weight": 22},
                "avg_st_slow":     {"weight": 18},
                "class_gap":       {"weight": 14},
                "course_st_slow":  {"weight": 10},
                "course_rank_bad": {"weight": 8},
            },
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

def calc_danger_score_v2(boat1, all_boats: list, config: Optional[dict] = None) -> tuple[float, dict]:
    """
    1号艇の危険度を100点満点で算出する（朝取得可能データのみ）。
    戻り値: (score, breakdown)
      breakdown: {"win_rate_low": 12.3, "motor_bad": 0.0, ...} 各項目の実加点
    """
    cfg = config or load_asahi_config()
    ds_cfg = cfg["danger_score"]
    W = {k: v["weight"] for k, v in ds_cfg["weights"].items()}
    TH = ds_cfg["thresholds"]

    breakdown: dict[str, float] = {k: 0.0 for k in W}

    if boat1 is None or len(all_boats) < 2:
        return 0.0, breakdown

    other_boats = [b for b in all_boats if b.lane != 1]
    if not other_boats:
        return 0.0, breakdown

    # ── 全国勝率 ──────────────────────────────────────────────
    avg_other_wr = sum(b.win_rate for b in other_boats) / len(other_boats)
    if avg_other_wr > 0 and boat1.win_rate < avg_other_wr * TH["win_rate_low_ratio"]:
        # 差が大きいほど加点（比率ベースで最大配点まで滑らかに増加）
        ratio = 1.0 - (boat1.win_rate / avg_other_wr) if avg_other_wr > 0 else 0
        breakdown["win_rate_low"] = round(min(W["win_rate_low"], W["win_rate_low"] * (ratio / (1 - TH["win_rate_low_ratio"]))), 2)

    # ── モーター2連率 ────────────────────────────────────────
    avg_other_motor = sum(b.motor for b in other_boats) / len(other_boats)
    if avg_other_motor > 0 and boat1.motor < avg_other_motor * TH["motor_bad_ratio"]:
        ratio = 1.0 - (boat1.motor / avg_other_motor) if avg_other_motor > 0 else 0
        breakdown["motor_bad"] = round(min(W["motor_bad"], W["motor_bad"] * (ratio / (1 - TH["motor_bad_ratio"]))), 2)

    # ── 平均ST実績 ───────────────────────────────────────────
    if boat1.avg_st:
        if boat1.avg_st >= TH["avg_st_slow_1"]:
            breakdown["avg_st_slow"] = W["avg_st_slow"]
        elif boat1.avg_st >= TH["avg_st_slow_2"]:
            breakdown["avg_st_slow"] = round(W["avg_st_slow"] * 0.55, 2)

    # ── 級別ギャップ ─────────────────────────────────────────
    if boat1.racer_class in ("B1", "B2"):
        base = W["class_gap"] * 0.6
        if any(b.lane != 1 and b.racer_class == "A1" for b in other_boats):
            base = W["class_gap"]
        breakdown["class_gap"] = round(base, 2)

    # ── コース別ST実績（fanファイル、1コース） ──────────────
    course_st_1c = boat1.course_st[0] if boat1.course_st and boat1.course_nyuko and boat1.course_nyuko[0] > 0 else None
    if course_st_1c is not None:
        if course_st_1c >= TH["course_st_slow_1"]:
            breakdown["course_st_slow"] = W["course_st_slow"]
        elif course_st_1c >= TH["course_st_slow_2"]:
            breakdown["course_st_slow"] = round(W["course_st_slow"] * 0.5, 2)

    # ── コース別ST順位実績（fanファイル、1コース） ──────────
    course_rank_1c = boat1.course_rank[0] if boat1.course_rank and boat1.course_nyuko and boat1.course_nyuko[0] > 0 else None
    if course_rank_1c is not None and course_rank_1c > 0:
        if course_rank_1c >= TH["course_rank_bad_1"]:
            breakdown["course_rank_bad"] = W["course_rank_bad"]
        elif course_rank_1c >= TH["course_rank_bad_2"]:
            breakdown["course_rank_bad"] = round(W["course_rank_bad"] * 0.5, 2)

    total = round(sum(breakdown.values()), 2)
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
    danger_score, danger_breakdown = calc_danger_score_v2(boat1, boats, cfg)

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
    detail["_features"] = {
        "win_rate":      boat1.win_rate if boat1 else None,
        "motor":         boat1.motor if boat1 else None,
        "avg_st":        boat1.avg_st if boat1 else None,
        "racer_class":   boat1.racer_class if boat1 else None,
        "course_st_1c":  boat1.course_st[0] if boat1 and boat1.course_nyuko and boat1.course_nyuko[0] > 0 else None,
        "course_rank_1c": boat1.course_rank[0] if boat1 and boat1.course_nyuko and boat1.course_nyuko[0] > 0 else None,
        "danger_breakdown": danger_breakdown,
        "model_version": cfg.get("model_version", "unknown"),
    }

    return upset_score, detail, target
