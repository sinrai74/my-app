"""
正式G3（段階4-2）：100レース連続一致の集約（連続カウント）。

GPT最終判断（2026-11-18）・doc 17 §1③ の規則:
  - 単位は1レース1件の BuyAssessment 比較（pred_combo 正本により1レース=1件）。
  - 比較成立・一致：+1
  - 比較成立・不一致：連続数を0にリセット（件数には含める）
  - 比較不能・skip・no_prediction・no_odds：件数に含めず、連続数を0にリセット
  - investigation は正式カウントに含めない（連続に影響しない）
  - 100（target）到達後の追加一致は達成条件に加えない
  - 複数日通算（既定）：permission=開催日単位・集約=run 横断（doc 17 §2-2、G2 と整合）

注（残論点、doc 17 §5）：
  - 同層（評価ロジック層）で両者が見送りかつ理由一致のレースは「比較成立・一致」(+1)。
    資金管理・金額0円層の見送りは g3_compare で INCOMPARABLE になり「比較不能」として扱う。
    GPT③の「skip は非カウント」は、この比較不能（層違い）として実現される。
    もし「同層一致の見送りも一切カウントしない」を採るなら本人・GPT の確定が要る。

正式G2 の評価側 baseline 42ファイル・Legacy・instrument は変更しない新規ファイル。
shadow/__init__.py からは import しない。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from core.g3_compare import INCOMPARABLE, MATCH, MISMATCH

OUTCOME_COMPARED = "compared"        # g3_entrypoint.OUTCOME_COMPARED と同じ
MODE_INVESTIGATION = "investigation"
MODE_FORMAL = "formal"

# per_race の効果ラベル
EFFECT_EXCLUDED_INVESTIGATION = "excluded_investigation"
EFFECT_MATCH_PLUS1 = "match_plus1"
EFFECT_MATCH_AFTER_ACHIEVED = "match_after_achieved"  # 到達後の一致（達成条件に加えない）
EFFECT_RESET_MISMATCH = "reset_mismatch"
EFFECT_RESET_INCOMPARABLE = "reset_incomparable"
EFFECT_RESET_NON_COMPARED = "reset_non_compared"      # no_prediction/no_odds など


def count_consecutive(records: Sequence[Mapping[str, Any]], target: int = 100) -> dict[str, Any]:
    """順序どおりの G3 レース記録から連続一致を数える。

    各 record は少なくとも outcome・mode を持ち、outcome=="compared" のときは race_judge
    （match／mismatch／incomparable）を持つ。records は時系列（複数日通算なら日・run をまたいで
    連結した順）で渡す。investigation は mode=="investigation" で除外する。
    """
    streak = 0
    best = 0
    achieved = False
    achieved_at: Any = None
    compared_count = 0          # 比較成立（match または mismatch）の件数
    per_race: list[dict[str, Any]] = []

    for i, r in enumerate(records):
        mode = r.get("mode")
        if mode == MODE_INVESTIGATION:
            per_race.append({"index": i, "effect": EFFECT_EXCLUDED_INVESTIGATION, "streak_after": streak})
            continue

        outcome = r.get("outcome")
        if outcome != OUTCOME_COMPARED:
            streak = 0
            per_race.append({"index": i, "effect": EFFECT_RESET_NON_COMPARED,
                             "outcome": outcome, "streak_after": 0})
            continue

        judge = r.get("race_judge")
        if judge == MATCH:
            compared_count += 1
            if achieved:
                per_race.append({"index": i, "effect": EFFECT_MATCH_AFTER_ACHIEVED, "streak_after": streak})
            else:
                streak += 1
                best = max(best, streak)
                if streak >= target:
                    achieved = True
                    achieved_at = i
                per_race.append({"index": i, "effect": EFFECT_MATCH_PLUS1, "streak_after": streak})
        elif judge == MISMATCH:
            compared_count += 1
            streak = 0
            per_race.append({"index": i, "effect": EFFECT_RESET_MISMATCH, "streak_after": 0})
        else:  # INCOMPARABLE（比較不能）：件数に含めず、リセット
            streak = 0
            per_race.append({"index": i, "effect": EFFECT_RESET_INCOMPARABLE, "streak_after": 0})

    return {
        "target": target,
        "streak_current": streak,
        "best_streak": best,
        "achieved": achieved,
        "achieved_at_index": achieved_at,
        "compared_count": compared_count,
        "total_records": len(records),
        "per_race": per_race,
    }
