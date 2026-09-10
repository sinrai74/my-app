"""evaluation段のdiffがGo/No-Go streakをリセットしないことの検証。

方針（承認済み）: evaluation段（upset_score/race_type）はShadowで観測・
diff記録するが、Go/No-Goのstreak必須判定からは外す。
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from shadow.aggregator import ShadowAggregator, to_staged_result


@dataclass
class _FakeRunResult:
    eval_id: str
    diffs: list = field(default_factory=list)


def _eval_diff(eval_id="e1", field_path="$.evaluation.upset_score"):
    return {"eval_id": eval_id, "field_path": field_path, "legacy": 8.0, "rebuild": 7.0}


def _buy_diff(eval_id="e1"):
    return {"eval_id": eval_id, "field_path": "$.buy_decision.n_bets",
            "legacy": 2, "rebuild": 0}


def _race_diff(eval_id="e1"):
    return {"eval_id": eval_id, "field_path": "$.race.venue_num",
            "legacy": 12, "rebuild": 13}


class TestEvaluationExcludedFromStreak(unittest.TestCase):
    def test_evaluation_only_diff_is_all_matched(self):
        # evaluation差だけ → streak判定上はall_matched（一致扱い）
        staged = to_staged_result(_FakeRunResult("e1", [_eval_diff()]))
        self.assertTrue(staged.all_matched)
        # ただしdiffsには全件残る（観測用）
        self.assertEqual(len(staged.diffs), 1)

    def test_race_type_diff_also_excluded(self):
        staged = to_staged_result(
            _FakeRunResult("e1", [_eval_diff(field_path="$.evaluation.race_type")]))
        self.assertTrue(staged.all_matched)
        self.assertEqual(len(staged.diffs), 1)

    def test_buy_decision_diff_breaks_streak(self):
        # buy_decision差 → streakを壊す（従来通り）
        staged = to_staged_result(_FakeRunResult("e1", [_buy_diff()]))
        self.assertFalse(staged.all_matched)

    def test_race_diff_breaks_streak(self):
        staged = to_staged_result(_FakeRunResult("e1", [_race_diff()]))
        self.assertFalse(staged.all_matched)

    def test_mixed_eval_and_buy_breaks_streak(self):
        # evaluation差 + buy_decision差 → buy_decision差でstreakは壊れる
        staged = to_staged_result(
            _FakeRunResult("e1", [_eval_diff(), _buy_diff()]))
        self.assertFalse(staged.all_matched)
        # diffsは両方残る
        self.assertEqual(len(staged.diffs), 2)

    def test_no_diff_is_matched(self):
        staged = to_staged_result(_FakeRunResult("e1", []))
        self.assertTrue(staged.all_matched)


class TestAggregatorStreakWithEvaluationDiffs(unittest.TestCase):
    def test_streak_not_reset_by_evaluation_diffs(self):
        agg = ShadowAggregator(required=3)
        # 3レース連続で evaluation差のみ → streak は3まで伸びる
        for i in range(3):
            agg.record(_FakeRunResult(f"e{i}", [_eval_diff(eval_id=f"e{i}")]))
        aggregate = agg.aggregate()
        self.assertEqual(aggregate.current_streak, 3)
        self.assertEqual(aggregate.matched_races, 3)
        self.assertEqual(aggregate.diff_races, 0)
        self.assertEqual(aggregate.broken_at, [])

    def test_streak_reset_by_buy_diff_amid_eval_diffs(self):
        agg = ShadowAggregator(required=3)
        agg.record(_FakeRunResult("e0", [_eval_diff(eval_id="e0")]))
        agg.record(_FakeRunResult("e1", [_buy_diff(eval_id="e1")]))  # breaks
        agg.record(_FakeRunResult("e2", [_eval_diff(eval_id="e2")]))
        aggregate = agg.aggregate()
        self.assertEqual(aggregate.current_streak, 1)  # only e2 after reset
        self.assertIn("e1", aggregate.broken_at)


if __name__ == "__main__":
    unittest.main()
