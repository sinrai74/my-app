"""
正式G3（B案）：ML 加点・target 入れ替えの純関数。

Legacy の規則（notify_arashi.py run() 3463〜3476行目、ループ内のコード）と同じ:
  prob_1 = ml_probs.get(1, 0.0)
  1号艇以外の最大確率 best_other_prob が prob_1 を超える（>）とき
    bonus = min(best_other_prob * 8.0, 4.0)、score += bonus
    target = boats の並び（1号艇以外）を ml_probs.get(lane, 0) の降順に
             並べた上位2艇（同値は boats の並び順。sorted は安定）
  ml_probs が空のときは何もしない。

注意: Legacy は best_other_lane を other_probs の max（キー順の先勝ち）で選ぶが、
加点量は best_other_prob（最大値）だけに依存するため、結果は同じ。

core/__init__.py からは import しない（正式G2 の評価側 baseline の42ファイルを
変更しないため）。
"""

from __future__ import annotations

from typing import Mapping, Sequence

from models.g3_input import MlBonusResult


def apply_ml_bonus(
    upset_score: float,
    target_lanes: Sequence[int],
    win_probs: Mapping[int, float],
    lane_order: Sequence[int],
) -> MlBonusResult:
    target_pre = tuple(int(x) for x in target_lanes)
    if not win_probs:
        return MlBonusResult(
            score_post=float(upset_score), target_post=target_pre,
            applied=False, bonus=0.0,
        )
    prob_1 = win_probs.get(1, 0.0)
    other_probs = {lane: p for lane, p in win_probs.items() if lane != 1}
    best_other_prob = max(other_probs.values()) if other_probs else 0.0
    if not best_other_prob > prob_1:
        return MlBonusResult(
            score_post=float(upset_score), target_post=target_pre,
            applied=False, bonus=0.0,
        )
    bonus = min(best_other_prob * 8.0, 4.0)
    sorted_lanes = sorted(
        [lane for lane in lane_order if lane != 1],
        key=lambda lane: win_probs.get(lane, 0),
        reverse=True,
    )
    return MlBonusResult(
        score_post=float(upset_score) + bonus,
        target_post=tuple(int(x) for x in sorted_lanes[:2]),
        applied=True,
        bonus=float(bonus),
    )
