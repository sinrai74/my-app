"""
UpsetScore算出（L2 Core・純関数）: 「1号艇が飛ぶ確率」の10点満点スコア。

移植元: x_asahi_scoring.py calc_boat_score_v2（L361-L399）および
calculate_upset_score_v2 の本体（L405-L536。rank_index/featured_boats/_features
の組み立てはStep4-4/4-5で移植し、本モジュールの範囲外）。
基準コミット: freeze-v1-baseline（Golden生成時source_commit=3a7f9c3dd7c628255285aefbe5b3e03978ec93b3）。

Feature Freeze厳守（Step4計画書C2）:
  logit/sigmoidの0.001-0.999クランプ、softmaxのオーバーフロー対策（max減算）、
  各リスクの閾値・truthyゲート、0.95上限、danger連動ブースト、安心レース
  ゼロ化、1号艇等級フィルタ、detail文字列の書式（f"{p:.1%}"等）まで
  すべて忠実に再現する。是正・改善はしない。

責務・依存（⑤5.3・⑩）:
  - 純関数のみ（config・providerは注入。ファイル・API・時刻アクセスなし）
  - 依存は標準ライブラリ＋core内部（core.danger）のみ。storage層依存禁止
  - danger算出は core.danger.calc_danger_score を同一引数で内部呼び出しする
    （移植元がcalc_danger_score_v2を内部で呼ぶ構造と同一）

定数（VENUE_NAMES / VENUE_UPSET_FACTOR / GRADE_EFFECTS）は移植元
x_asahi_scoring.py L50-L67 から機械転記した（手作業での再構成禁止）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from core.danger import Boat, CourseFactorProvider, calc_danger_score

# ── 移植元 L50-L58 機械転記 ──
VENUE_NAMES: dict[int, str] = {
    1: "桐生",   2: "戸田",   3: "江戸川", 4: "平和島", 5: "多摩川",
    6: "浜名湖", 7: "蒲郡",   8: "常滑",   9: "津",    10: "三国",
    11: "びわこ", 12: "住之江", 13: "尼崎",  14: "鳴門", 15: "丸亀",
    16: "児島",  17: "宮島",  18: "徳山",  19: "下関", 20: "若松",
    21: "芦屋",  22: "福岡",  23: "唐津",  24: "大村",
}

# ── 移植元 L59-L66 機械転記 ──
VENUE_UPSET_FACTOR: dict[int, float] = {
    1:  0.10, 2:  0.05, 3:  0.15, 4:  0.10, 5:  0.00, 6: -0.05,
    7:  0.00, 8: -0.05, 9: -0.10, 10: 0.00, 11: 0.00, 12: -0.05,
    13: 0.00, 14: 0.00, 15: 0.00, 16: -0.10, 17: 0.10, 18: 0.00,
    19: 0.05, 20: 0.05, 21: 0.00, 22: 0.00, 23: 0.05, 24: -0.15,
}

# ── 移植元 L67 機械転記 ──
GRADE_EFFECTS: dict[int, float] = {0: 0.0, 1: -0.1, 2: -0.2, 3: -0.3, 4: -0.5}

_GRADE_NAMES: dict[int, str] = {0: "一般", 1: "G3", 2: "G2", 3: "G1", 4: "SG"}


def _logit(p: float) -> float:
    p = max(0.001, min(0.999, p))
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _add_effect(base_prob: float, effect: float) -> float:
    return _sigmoid(_logit(base_prob) + effect)


def calc_boat_score(boat: Boat, all_boats: Sequence[Boat], config: dict) -> float:
    """1艇の相対的な強さスコア（移植元 calc_boat_score_v2 L361-L399）。"""
    brs = config["boat_relative_score"]["weights"]
    score = 0.0

    # 平均ST実績（0.17秒を中心に線形。truthyゲート＝移植元と同一）
    if boat.get("avg_st"):
        avg_st_score = (0.17 - boat["avg_st"]) * 10.0
        score += avg_st_score * brs["avg_st_component"]["weight"]

    # コース別ST実績
    lane_idx = (boat.get("lane") or 0) - 1
    nyuko = boat.get("course_nyuko")
    if nyuko and 0 <= lane_idx < 6 and nyuko[lane_idx] > 0:
        c_st = boat["course_st"][lane_idx]
        c_st_score = (0.17 - c_st) * 10.0
        score += c_st_score * brs["course_st_component"]["weight"]

    # モーター2連率（全艇平均比。is not None判定＝移植元と同一）
    motors = [b.get("motor") for b in all_boats if b.get("motor") is not None]
    if motors and boat.get("motor") is not None:
        avg_motor = sum(motors) / len(motors)
        if avg_motor > 0:
            motor_score = (boat["motor"] - avg_motor) / avg_motor * 5.0
            score += motor_score * brs["motor_component"]["weight"]

    # 全国勝率
    if boat.get("win_rate") is not None:
        score += (boat["win_rate"] - 5.0) * brs["win_rate_component"]["weight"]

    # コース（進入枠）による基礎優位性
    lane_weight = brs.get("lane_weight", {})
    score += lane_weight.get(str(boat.get("lane")), 0.0)

    return score


@dataclass(frozen=True)
class UpsetResult:
    """upset算出の結果と、後続段（rank/featured/_features＝Step4-4/4-5）が
    参照する中間値。移植元では1関数内のローカル変数だったものを型として公開する
    （分離が旧出力を変えないことはGolden回帰で保証する＝C3方式）。

    Legacy: calculate_upset_score_v2（x_asahi_scoring.py L405-L536）
    Freeze Commit: 3a7f9c3dd7c628255285aefbe5b3e03978ec93b3

    Consumed by: RankEngine（Step4-4: rank_index/featured_boats/match_indexの
    算出が lane_probs・danger_score・upset_score 等の中間値を参照する）
    """

    upset_score: float
    detail: dict[str, Any]  # 移植元detailのコア部（荒れ確率〜等級フィルタ）
    target_lanes: tuple[int, ...]
    lane_probs: dict[int, float]
    boat1_prob: float
    boat1_rank: int
    best_other_lane: int
    best_other_prob: float
    danger_score: float
    danger_breakdown: dict[str, Any]
    venue_name: Optional[str]


def calculate_upset_score(
    boats: Sequence[Boat],
    config: dict,
    race_grade: int = 0,
    venue_num: int = 0,
    is_night: bool = False,
    venue_stats: Optional[CourseFactorProvider] = None,
) -> UpsetResult:
    """荒れ確率スコアの算出（移植元 calculate_upset_score_v2 L405-L536）。"""
    up_cfg = config["upset_prob"]
    eff = {k: v["value"] for k, v in up_cfg["effects"].items()}

    # ── 各艇のベース確率（softmax・max減算＝移植元と同一） ──
    raw_scores = [(b.get("lane"), calc_boat_score(b, boats, config)) for b in boats]
    max_s = max(s for _, s in raw_scores)
    exp_scores = [(lane, math.exp(s - max_s)) for lane, s in raw_scores]
    total = sum(e for _, e in exp_scores)
    lane_probs = {lane: e / total for lane, e in exp_scores}

    boat1_prob = lane_probs.get(1, 1 / 6)
    boat1 = next((b for b in boats if b.get("lane") == 1), None)
    boat2 = next((b for b in boats if b.get("lane") == 2), None)

    other_probs = {lane: p for lane, p in lane_probs.items() if lane != 1}
    best_other_lane = max(other_probs, key=other_probs.get) if other_probs else 2
    best_other_prob = other_probs.get(best_other_lane, 0.0)

    upset_prob = 1.0 - boat1_prob
    venue_name = VENUE_NAMES.get(venue_num)
    danger_score, danger_breakdown = calc_danger_score(
        boat1, boats, config, venue_name=venue_name, venue_stats=venue_stats
    )

    # ── ①1号艇の弱さ（ST・勝率・機力/級別） ──
    if boat1:
        st_risk = 0.0
        if boat1.get("avg_st") and boat1["avg_st"] > 0.17:
            st_risk += eff["st_risk_avg_st"]
        nyuko = boat1.get("course_nyuko")
        # 移植元はlane_idxを算出しつつ添字[0]を直接使う。その挙動を保存する
        if nyuko and nyuko[0] > 0 and boat1["course_st"][0] > 0.16:
            st_risk += eff["st_risk_course_st"]
        if st_risk > 0:
            upset_prob = _add_effect(upset_prob, st_risk)

        wr_risk = 0.0
        if boat1["win_rate"] < 5.0:
            wr_risk += eff["win_rate_risk_1"]
        elif boat1["win_rate"] < 5.5:
            wr_risk += eff["win_rate_risk_2"]
        if wr_risk > 0:
            upset_prob = _add_effect(upset_prob, wr_risk)

        eq_risk = 0.0
        if boat1["motor"] < 35.0:
            eq_risk += eff["motor_risk"]
        if boat1.get("racer_class") in ("B1", "B2"):
            eq_risk += eff["class_risk"]
        if eq_risk > 0:
            upset_prob = _add_effect(upset_prob, eq_risk)

    # ── ②2号艇の強さ ──
    boat2_prob = lane_probs.get(2, 0.0)
    if boat2 and boat2_prob > boat1_prob:
        upset_prob = _add_effect(
            upset_prob, (boat2_prob - boat1_prob) * eff["boat2_strength_mult"]
        )

    # ── ③会場・ナイター補正 ──
    venue_factor = VENUE_UPSET_FACTOR.get(venue_num, 0.0)
    if venue_factor != 0.0:
        upset_prob = _add_effect(upset_prob, venue_factor)
    if is_night:
        upset_prob = _add_effect(upset_prob, up_cfg.get("night_factor", -0.08))

    # ── ④等級差 ──
    if boat1 and boat1.get("racer_class") in ("B1", "B2"):
        if any(b.get("lane") != 1 and b.get("racer_class") == "A1" for b in boats):
            upset_prob = _add_effect(upset_prob, eff["class_gap_effect"])

    # ── ⑤レース種別補正 ──
    upset_prob = _add_effect(upset_prob, GRADE_EFFECTS.get(race_grade, 0.0))

    upset_prob = max(0.0, min(upset_prob, 0.95))
    upset_score = upset_prob * 10.0

    # danger連動ブースト（閾値・倍率・9.5上限＝移植元と同一）
    if danger_score >= 60:
        upset_score = min(upset_score * 1.30, 9.5)
    elif danger_score >= 40:
        upset_score = min(upset_score * 1.15, 9.5)

    # 安心レースのゼロ化
    if boat1_prob > 0.65 and best_other_prob < boat1_prob:
        upset_score = 0.0

    # ── 1号艇等級フィルタ ──
    grade_filter_note = ""
    if boat1 and boat1.get("racer_class") == "A1":
        upset_score = 0.0
        grade_filter_note = "1号艇A1 → スキップ"
    elif boat1 and boat1.get("racer_class") == "A2":
        upset_score = max(upset_score - 1.5, 0.0)
        grade_filter_note = "1号艇A2 → -1.5"

    target = sorted(other_probs, key=other_probs.get, reverse=True)[:2]

    sorted_lanes = sorted(lane_probs, key=lane_probs.get, reverse=True)
    boat1_rank = sorted_lanes.index(1) + 1 if 1 in sorted_lanes else 6

    detail: dict[str, Any] = {
        "荒れ確率":     f"{upset_prob:.1%}",
        "1号艇確率":    f"{boat1_prob:.1%}(rank{boat1_rank})",
        "1号艇危険度":  f"{danger_score:.1f}",
        "最有力":       f"{best_other_lane}号艇({best_other_prob:.1%})",
        "レース種別":   _GRADE_NAMES.get(race_grade, f"grade{race_grade}"),
        "モデルVersion": config.get("model_version", "unknown"),
    }
    if grade_filter_note:
        detail["等級フィルタ"] = grade_filter_note

    return UpsetResult(
        upset_score=upset_score,
        detail=detail,
        target_lanes=tuple(target),
        lane_probs=lane_probs,
        boat1_prob=boat1_prob,
        boat1_rank=boat1_rank,
        best_other_lane=best_other_lane,
        best_other_prob=best_other_prob,
        danger_score=danger_score,
        danger_breakdown=danger_breakdown,
        venue_name=venue_name,
    )
