"""
上位進出指数・注目選手・match_index（L2 Core・純関数）。

移植元（すべて x_asahi_scoring.py、freeze-v1-baseline）:
  - calc_lane_rank_scores_v2   L613-L746 → calc_lane_rank_scores
  - calc_rank_probabilities_v2 L747-L804 → calc_rank_probabilities
  - calc_rank_index_v2         L815-L859 → calc_rank_index
  - select_featured_boats      L860-L939 → select_featured_boats
  - match_index式              L536（rank_index_ctx） → compute_match_index
基準コミット: 3a7f9c3dd7c628255285aefbe5b3e03978ec93b3

Feature Freeze厳守（Step4計画書C2）:
  丸め（寄与度round3・指数round1）、スコア＝丸め済み寄与度の合計、
  max(0.1)フロア、danger減点の適用順、6号艇1着の特別条件と減衰、
  正規化（単純比率・total<=0で等分）、featuredの閾値・並び・マーク付与を
  すべて忠実に再現する。是正・改善はしない。
  なお calc_rank_index は移植元と同様に calc_lane_rank_scores を
  艇ごとに二重計算する（probabilities内部と寄与度取得の2回。
  最適化は行わない＝挙動保存）。

責務・依存（⑤5.3・⑩）:
  - 純関数のみ（configは呼び出し側がfreeze configを渡す。移植元の
    load_asahi_configフォールバックはcore純粋性のため持たない）
  - 依存は標準ライブラリ＋core内部（core.danger）のみ。storage層依存禁止
  - 場補正はCourseFactorProviderを注入。venue_name指定時にvenue_statsが
    Noneの場合は場補正なし（venue未指定と同挙動）として扱う
  - Danger / Upset / FeatureBuilder のコードは変更しない（dangerは
    移植元と同一引数で内部呼び出しのみ）
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from core.danger import Boat, CourseFactorProvider, calc_danger_score

# legacy: class_rank既定値（移植元L672と同一）
_DEFAULT_CLASS_RANK: dict[str, int] = {"A1": 4, "A2": 3, "B1": 2, "B2": 1, "": 0}

# legacy: lane_base_rate未定義レーンの既定値（移植元L638と同一）
_DEFAULT_BASE_RATES: dict[str, float] = {"first": 10, "second": 15, "third": 15}


def compute_match_index(upset_score: float) -> float:
    """match_index（0-100）。移植元L536のrank_index_ctx構築式の機械転記。"""
    return min(100, upset_score * 10.5)


def calc_lane_rank_scores(
    boat: Boat,
    all_boats: Sequence[Boat],
    config: dict,
    venue_name: Optional[str] = None,
    venue_stats: Optional[CourseFactorProvider] = None,
) -> dict:
    """1艇の1着・2着・3着適性と寄与度（移植元 calc_lane_rank_scores_v2）。"""
    lrs = config["lane_rank_scores"]
    base_rates = lrs["lane_base_rate"].get(str(boat.get("lane")), _DEFAULT_BASE_RATES)

    # 全艇平均比（is not None フィルタ＝移植元と同一）
    win_rates = [b.get("win_rate") for b in all_boats if b.get("win_rate") is not None]
    local_wins = [b.get("local_win") for b in all_boats if b.get("local_win") is not None]
    motors = [b.get("motor") for b in all_boats if b.get("motor") is not None]
    avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 5.0
    avg_local_win = sum(local_wins) / len(local_wins) if local_wins else 5.0
    avg_motor = sum(motors) / len(motors) if motors else 33.0

    win_rate_diff = (
        (boat["win_rate"] - avg_win_rate) if boat.get("win_rate") is not None else 0.0
    )
    local_win_diff = (
        (boat["local_win"] - avg_local_win) if boat.get("local_win") is not None else 0.0
    )
    motor_diff = (boat["motor"] - avg_motor) if boat.get("motor") is not None else 0.0
    # 平均ST（truthyゲート＝移植元と同一。0.0は欠損扱い）
    avg_st_diff = (0.17 - boat["avg_st"]) * 10.0 if boat.get("avg_st") else 0.0

    lane_idx = (boat.get("lane") or 0) - 1
    nyuko = boat.get("course_nyuko")
    in_lane = bool(nyuko) and 0 <= lane_idx < 6 and nyuko[lane_idx] > 0

    course_st_diff = 0.0
    if in_lane:
        course_st_diff = (0.17 - boat["course_st"][lane_idx]) * 10.0

    course_perf_diff = 0.0
    if in_lane and boat.get("course_win_rate"):
        national_base = (
            lrs["lane_base_rate"].get(str(boat.get("lane")), {}).get("first", 0.0)
        )
        course_perf_diff = (boat["course_win_rate"][lane_idx] - national_base) / 10.0

    class_rank_map = config.get("danger_score", {}).get("class_rank", _DEFAULT_CLASS_RANK)
    class_ranks = [class_rank_map.get(b.get("racer_class") or "", 0) for b in all_boats]
    avg_class_rank = sum(class_ranks) / len(class_ranks) if class_ranks else 0.0
    class_diff = class_rank_map.get(boat.get("racer_class") or "", 0) - avg_class_rank

    def _ability_trend(b: Boat) -> float:
        if b.get("ability_curr") and b.get("ability_prev"):
            return b["ability_curr"] - b["ability_prev"]
        return 0.0

    ability_trends = [_ability_trend(b) for b in all_boats]
    avg_ability_trend = (
        sum(ability_trends) / len(ability_trends) if ability_trends else 0.0
    )
    ability_trend_diff = _ability_trend(boat) - avg_ability_trend

    venue_factor_diff = 0.0
    if venue_name and venue_stats and in_lane:
        vf = venue_stats.get_venue_course_factor(venue_name, boat.get("lane"))
        venue_factor_diff = vf["factor"] - 1.0

    f_rate = 0.0
    if in_lane and boat.get("course_f_count"):
        f_rate = boat["course_f_count"][lane_idx] / nyuko[lane_idx] * 100

    def _rank_score_with_contrib(rank_key: str, base_rate: float) -> tuple[float, dict]:
        w = lrs[rank_key]
        base = base_rate * w["base_rate_weight"]
        contrib = {
            "base_rate": round(base, 3),
            "win_rate": round(win_rate_diff * w["win_rate_weight"], 3),
            "local_win": round(local_win_diff * w.get("local_win_weight", 0.0), 3),
            "avg_st": round(avg_st_diff * w["avg_st_weight"], 3),
            "motor": round(motor_diff * w["motor_weight"], 3),
            "class": round(class_diff * w.get("class_weight", 0.0), 3),
            "course_perf": round(course_perf_diff * w.get("course_perf_weight", 0.0), 3),
            "ability_trend": round(
                ability_trend_diff * w.get("ability_trend_weight", 0.0), 3
            ),
            "venue_factor": round(
                venue_factor_diff * w.get("venue_factor_weight", 0.0), 3
            ),
            "f_risk": round(f_rate / 10.0 * w.get("f_risk_weight", 0.0), 3),
            "other": round(course_st_diff * w.get("course_st_weight", 0.0), 3),
        }
        score = sum(contrib.values())
        return score, contrib

    first_score, first_contrib = _rank_score_with_contrib(
        "first_place", base_rates["first"]
    )
    second_score, second_contrib = _rank_score_with_contrib(
        "second_place", base_rates["second"]
    )
    third_score, third_contrib = _rank_score_with_contrib(
        "third_place", base_rates["third"]
    )

    # 危険艇判定（1号艇のみ）: dangerを内部呼び出しし各着順の重みで減点
    if boat.get("lane") == 1:
        danger_score, _ = calc_danger_score(
            boat, all_boats, config, venue_name=venue_name, venue_stats=venue_stats
        )
        penalty_unit = danger_score / 10.0
        first_penalty = penalty_unit * lrs["first_place"]["danger_penalty_weight"]
        second_penalty = penalty_unit * lrs["second_place"]["danger_penalty_weight"]
        third_penalty = penalty_unit * lrs["third_place"]["danger_penalty_weight"]
        first_score -= first_penalty
        second_score -= second_penalty
        third_score -= third_penalty
        first_contrib["danger_penalty"] = round(-first_penalty, 3)
        second_contrib["danger_penalty"] = round(-second_penalty, 3)
        third_contrib["danger_penalty"] = round(-third_penalty, 3)

    return {
        "first": max(0.1, first_score),
        "second": max(0.1, second_score),
        "third": max(0.1, third_score),
        "contributions": {
            "first": first_contrib,
            "second": second_contrib,
            "third": third_contrib,
        },
    }


def calc_rank_probabilities(
    boats: Sequence[Boat],
    context: Optional[dict] = None,
    config: Optional[dict] = None,
    venue_name: Optional[str] = None,
    venue_stats: Optional[CourseFactorProvider] = None,
) -> dict:
    """1着・2着・3着の確率分布（移植元 calc_rank_probabilities_v2）。"""
    cfg = config
    context = context or {}
    lrs = cfg["lane_rank_scores"]

    scores = {
        b.get("lane"): calc_lane_rank_scores(
            b, boats, cfg, venue_name=venue_name, venue_stats=venue_stats
        )
        for b in boats
    }

    def _softmax_normalize(score_map: dict) -> dict:
        total = sum(score_map.values())
        if total <= 0:
            n = len(score_map) or 1
            return {lane: 1.0 / n for lane in score_map}
        return {lane: s / total for lane, s in score_map.items()}

    lane6_cond = lrs["lane6_first_place_conditions"]
    match_index = context.get("match_index", 0)
    upset_score = context.get("upset_score", 0)
    lane6_allowed = (
        match_index >= lane6_cond["min_match_index"]
        or upset_score >= lane6_cond["min_upset_score"]
    )

    first_raw = {lane: s["first"] for lane, s in scores.items()}
    if not lane6_allowed and 6 in first_raw:
        first_raw[6] *= lane6_cond["normal_suppress_ratio"]

    first_probs = _softmax_normalize(first_raw)
    second_probs = _softmax_normalize({lane: s["second"] for lane, s in scores.items()})
    third_probs = _softmax_normalize({lane: s["third"] for lane, s in scores.items()})

    return {
        "first": first_probs,
        "second": second_probs,
        "third": third_probs,
        "lane6_first_allowed": lane6_allowed,
    }


def calc_rank_index(
    boats: Sequence[Boat],
    context: Optional[dict] = None,
    config: Optional[dict] = None,
    venue_name: Optional[str] = None,
    venue_stats: Optional[CourseFactorProvider] = None,
) -> dict[int, dict]:
    """全艇の上位進出指数0-100（移植元 calc_rank_index_v2）。

    移植元と同様、寄与度取得のためcalc_lane_rank_scoresを艇ごとに
    再計算する（probabilities内部との二重計算＝挙動保存）。
    """
    cfg = config
    probs = calc_rank_probabilities(
        boats, context, cfg, venue_name=venue_name, venue_stats=venue_stats
    )

    result: dict[int, dict] = {}
    for b in boats:
        lane = b.get("lane")
        p1 = probs["first"].get(lane, 0.0)
        p2 = probs["second"].get(lane, 0.0)
        p3 = probs["third"].get(lane, 0.0)
        lane_scores = calc_lane_rank_scores(
            b, boats, cfg, venue_name=venue_name, venue_stats=venue_stats
        )
        result[lane] = {
            "top1": round(p1 * 100, 1),
            "top2": round((p1 + p2) * 100, 1),
            "top3": round((p1 + p2 + p3) * 100, 1),
            "contributions": lane_scores.get("contributions", {}),
        }
    return result


def select_featured_boats(
    boats: Sequence[Boat],
    rank_index: dict[int, dict],
    danger_score: float,
    config: Optional[dict] = None,
) -> list[dict[str, Any]]:
    """上位進出指数から注目選手を最大N艇選ぶ（移植元 select_featured_boats）。"""
    cfg = config
    fr_cfg = cfg.get("featured_racers", {})
    max_count = fr_cfg.get("max_count", 3)
    marks = fr_cfg.get("marks", ["◎", "○", "▲"])
    exclude_th = fr_cfg.get("exclude_boat1_threshold", 40)
    min_composite = fr_cfg.get("min_composite_threshold", 0.0)
    cw = fr_cfg.get(
        "composite_weights",
        {"top1_weight": 0.5, "top2_weight": 0.3, "top3_weight": 0.2},
    )

    candidates = boats
    if danger_score >= exclude_th:
        candidates = [b for b in boats if b.get("lane") != 1]

    scored = []
    for b in candidates:
        idx = rank_index.get(b.get("lane"), {"top1": 0.0, "top2": 0.0, "top3": 0.0})
        composite = (
            idx["top1"] * cw.get("top1_weight", 0.5)
            + idx["top2"] * cw.get("top2_weight", 0.3)
            + idx["top3"] * cw.get("top3_weight", 0.2)
        )
        if composite < min_composite:
            continue
        scored.append(
            {
                "lane": b.get("lane"),
                "name": b.get("name"),
                "composite": round(composite, 1),
                "top1": idx["top1"],
                "top2": idx["top2"],
                "top3": idx["top3"],
            }
        )

    scored.sort(key=lambda x: -x["composite"])
    top_n = scored[:max_count]
    for i, item in enumerate(top_n):
        item["mark"] = marks[i] if i < len(marks) else ""
    return top_n
