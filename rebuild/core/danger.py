"""
DangerScore算出（L2 Core・純関数）: 1号艇の危険度100点満点。

移植元: x_asahi_scoring.py calc_danger_score_v2（L210-L360）。
基準コミット: freeze-v1-baseline（Golden生成時source_commit=3a7f9c3dd7c628255285aefbe5b3e03978ec93b3）。

Feature Freeze厳守（Step4計画書C2）:
  旧ロジックの出力を正とし、or演算子によるデフォルト値（0がfalsyになる挙動）、
  丸め、ゲート条件、上限クリップをすべて忠実に再現する。是正・改善はしない。

責務・依存（⑤5.3・⑩）:
  - 純関数のみ。config読込・ファイル・API・時刻へアクセスしない
    （configは呼び出し側がdictで渡す。venue補正はProvider注入）
  - 依存は標準ライブラリのみ（modelsにも依存しない: 入力はVer4互換の
    boat属性Mapping。FeatureInputs.boatsと同一形式）
  - storage層への依存禁止（Step4-2指示）

venue補正の注入:
  legacyは x_venue_stats.get_venue_course_factor を直接呼ぶが、本移植では
  CourseFactorProvider（構造的Protocol）として注入する。features層の
  VenueStatsProviderと互換のシグネチャ（同一のFake/アダプタを共用できる）。
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence

Boat = Mapping[str, Any]


class CourseFactorProvider(Protocol):
    """場×コース補正の供給者（venue指定時のみ使用）。"""

    def get_venue_course_factor(self, venue_name: str, course: int) -> Mapping[str, Any]:
        """少なくとも {"factor": float} を含むdictを返す。"""
        ...


# legacy: ds_cfg.get("class_rank", {...}) の既定値（移植元L241と同一）
_DEFAULT_CLASS_RANK: dict[str, int] = {"A1": 4, "A2": 3, "B1": 2, "B2": 1, "": 0}


def calc_danger_score(
    boat1: Optional[Boat],
    all_boats: Sequence[Boat],
    config: dict,
    venue_name: Optional[str] = None,
    venue_stats: Optional[CourseFactorProvider] = None,
) -> tuple[float, dict]:
    """1号艇の危険度と内訳を返す（移植元と同一の戻り値契約）。

    breakdown: {key: {"weighted": float, ("worse_count"/"worse_total"),
                      "kind": "relative"|"solo"}}
    """
    ds_cfg = config["danger_score"]
    rw = {k: v for k, v in ds_cfg["relative_weights"].items() if k != "_comment"}
    sw = {k: v for k, v in ds_cfg["solo_weights"].items() if k != "_comment"}
    sth = ds_cfg["solo_thresholds"]
    class_rank = ds_cfg.get("class_rank", _DEFAULT_CLASS_RANK)

    breakdown: dict[str, dict] = {}

    if boat1 is None or len(all_boats) < 2:
        return 0.0, breakdown
    other_boats = [b for b in all_boats if b.get("lane") != 1]
    if not other_boats:
        return 0.0, breakdown

    def _relative_item(
        key: str,
        boat1_val: float,
        other_vals: list[float],
        higher_is_better: bool = True,
    ) -> None:
        weight = rw.get(key)
        if not weight:
            return
        if higher_is_better:
            worse_count = sum(1 for v in other_vals if v > boat1_val)
        else:
            worse_count = sum(1 for v in other_vals if v < boat1_val)
        weighted = round(min(weight["max_weight"], weight["per_boat"] * worse_count), 2)
        breakdown[key] = {
            "weighted": weighted,
            "worse_count": worse_count,
            "worse_total": len(other_vals),
            "kind": "relative",
        }

    # ①全国勝率（相対）
    _relative_item(
        "win_rate",
        boat1.get("win_rate") or 0.0,
        [(b.get("win_rate") or 0.0) for b in other_boats],
        higher_is_better=True,
    )
    # ②当地勝率（相対）
    _relative_item(
        "local_win",
        boat1.get("local_win") or 0.0,
        [(b.get("local_win") or 0.0) for b in other_boats],
        higher_is_better=True,
    )
    # ③平均ST（相対・小さいほど良い。欠損は0.18で補完＝移植元と同一）
    _relative_item(
        "avg_st",
        boat1.get("avg_st") or 0.18,
        [(b.get("avg_st") or 0.18) for b in other_boats],
        higher_is_better=False,
    )
    # ④モーター2連率（相対）
    _relative_item(
        "motor",
        boat1.get("motor") or 0.0,
        [(b.get("motor") or 0.0) for b in other_boats],
        higher_is_better=True,
    )
    # ⑤級別（相対）
    boat1_class_rank = class_rank.get(boat1.get("racer_class") or "", 0)
    other_class_ranks = [class_rank.get(b.get("racer_class") or "", 0) for b in other_boats]
    _relative_item("racer_class", boat1_class_rank, other_class_ranks, higher_is_better=True)

    # ⑥能力指数推移（相対。curr/prevのどちらかがfalsyなら0.0＝移植元の挙動を保存）
    def _ability_trend(b: Boat) -> float:
        if b.get("ability_curr") and b.get("ability_prev"):
            return b["ability_curr"] - b["ability_prev"]
        return 0.0

    _relative_item(
        "ability_trend",
        _ability_trend(boat1),
        [_ability_trend(b) for b in other_boats],
        higher_is_better=True,
    )

    # ⑦進入コース別2連対率（相対・場補正込み）
    def _course_rentai(b: Boat) -> float:
        lane = b.get("lane") or 0
        lane_idx = lane - 1
        nyuko = b.get("course_nyuko")
        if not (nyuko and 0 <= lane_idx < 6 and nyuko[lane_idx] > 0):
            return 0.0
        place_counts = b.get("course_place_counts")
        counts = place_counts[lane_idx] if place_counts else None
        nyuko_n = nyuko[lane_idx]
        if counts and sum(counts) > 0:
            rentai2 = (counts[0] + counts[1]) / nyuko_n * 100
        else:
            # 着順回数未取得時は複勝率で近似（移植元と同一）
            place_rate = b.get("course_place_rate")
            rentai2 = place_rate[lane_idx] if place_rate else 0.0
        if venue_name and venue_stats:
            factor = venue_stats.get_venue_course_factor(venue_name, lane)["factor"]
            rentai2 *= factor
        return rentai2

    _relative_item(
        "course_rentai",
        _course_rentai(boat1),
        [_course_rentai(b) for b in other_boats],
        higher_is_better=True,
    )

    # ── 単体評価（絶対値ベースの小さな加点） ──
    def _solo_item(key: str, cond: bool) -> None:
        weight = sw.get(key)
        if not weight or not cond:
            return
        breakdown[key] = {"weighted": float(weight["weight"]), "kind": "solo"}

    _solo_item("win_rate_low", (boat1.get("win_rate") or 0.0) < sth["win_rate_low_abs"])
    _solo_item("local_win_low", (boat1.get("local_win") or 0.0) < sth["local_win_low_abs"])
    _solo_item("motor_bad", (boat1.get("motor") or 0.0) < sth["motor_bad_abs"])
    _solo_item("avg_st_slow", (boat1.get("avg_st") or 0.0) >= sth["avg_st_slow_abs"])

    # 1コース1着率（gateと値の取り方は移植元と同一）
    course1_win = None
    course_win_rate = boat1.get("course_win_rate")
    nyuko1 = boat1.get("course_nyuko")
    if course_win_rate and nyuko1 and nyuko1[0] > 0:
        course1_win = course_win_rate[0]
    if course1_win is not None:
        _solo_item("course1_place_low", course1_win < sth["course1_place_low_abs"])

    # 1コースFリスク
    f_counts = boat1.get("course_f_count")
    if nyuko1 and nyuko1[0] > 0 and f_counts:
        f_rate = f_counts[0] / nyuko1[0] * 100
        _solo_item("f_risk", f_rate >= sth["f_risk_rate_abs"])

    # 場不利（venue指定時のみ）
    if venue_name and venue_stats is not None:
        vf = venue_stats.get_venue_course_factor(venue_name, 1)
        _solo_item("venue_unfavorable", vf["factor"] < sth["venue_unfavorable_factor_abs"])

    total = round(sum(v["weighted"] for v in breakdown.values()), 2)
    total = min(total, ds_cfg["total_scale"])
    return total, breakdown
