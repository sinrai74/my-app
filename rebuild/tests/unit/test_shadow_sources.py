"""
Step6-2 取得元確定（必須①②③）のテスト。

確定した取得元:
  ① RenderResult      : Legacy生成HTMLのbyte読み取り（再実行・戻り値dict比較なし）
  ② NotificationRequest: build_message() の戻り値(subject, body)のみ
  ③ FeatureSet        : sent_*.txtのfeat_*のみ（FeatureBuilder再実行なし）

Legacyは読む/呼ぶだけ（フック・print・logging・一時ファイル追加なし）。
"""

from __future__ import annotations

import os
import tempfile
import unittest

from shadow.legacy_notification import (
    LegacyNotificationView,
    build_notification_view,
)
from shadow.legacy_output import compare_output_bytes, read_output_bytes
from shadow.legacy_source import LegacySentRecord, load_sent_records


# ---------------- ③ FeatureSet: sent_*.txtのfeat_*のみ ----------------


_SENT_WITH_FEATURES = (
    '{"key": "20260704_1_1", "combo": "6-1-3", "prob": 0.04591, "ev": 3.03, '
    '"odds": 66.0, "confidence": 0.8007, "race_type": "1残り荒れ型", '
    '"upset_score": 9.5, "venue": "桐生", "venue_num": 1, "race": 1, '
    '"night": 0, "model_version": "asahi-v1.0-phase1", '
    '"feat_win_rate": 3.71, "feat_motor": 28.81, "feat_avg_st": 0.17, '
    '"feat_racer_class": 2, "feat_course_st_1c": 0.16, '
    '"feat_course_rank_1c": 2.4, "feat_danger_breakdown": "{}"}\n'
)

_SENT_WITHOUT_FEATURES = (
    '{"key": "20260628_4_12", "combo": "2-1-3", "prob": 0.03422, "ev": 2.053, '
    '"odds": 60.0, "confidence": 0.6177, "race_type": "1残り荒れ型", '
    '"upset_score": 9.5, "venue": "平和島", "venue_num": 4, "race": 12, '
    '"night": 1}\n'
)


def _write_sent(content: str) -> str:
    d = tempfile.mkdtemp()
    path = os.path.join(d, "sent_20260704.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


class TestFeatureView(unittest.TestCase):
    def test_returns_only_existing_feat_keys(self) -> None:
        records = load_sent_records(_write_sent(_SENT_WITH_FEATURES))
        view = records["20260704_01_01"].feature_view()
        self.assertEqual(view["feat_win_rate"], 3.71)
        self.assertEqual(view["feat_motor"], 28.81)
        self.assertEqual(view["feat_course_rank_1c"], 2.4)
        # feat_*以外は含まれない（combo等は別ビューの責務）
        self.assertNotIn("combo", view)
        self.assertNotIn("upset_score", view)

    def test_all_seven_feature_keys_defined(self) -> None:
        """実データ集計で確認した7種が定義されていること。"""
        self.assertEqual(
            LegacySentRecord.FEATURE_KEYS,
            ("feat_win_rate", "feat_motor", "feat_avg_st", "feat_racer_class",
             "feat_course_st_1c", "feat_course_rank_1c", "feat_danger_breakdown"),
        )

    def test_missing_features_not_filled(self) -> None:
        """feat_*が無い行では空dict（仮値で埋めない）。"""
        records = load_sent_records(_write_sent(_SENT_WITHOUT_FEATURES))
        view = records["20260628_04_12"].feature_view()
        self.assertEqual(view, {})

    def test_missing_feature_keys_recorded(self) -> None:
        """Legacy取得元なしのキーが記録用に列挙されること。"""
        records = load_sent_records(_write_sent(_SENT_WITHOUT_FEATURES))
        missing = records["20260628_04_12"].missing_feature_keys()
        self.assertEqual(len(missing), 7)
        self.assertIn("feat_win_rate", missing)

    def test_partial_features(self) -> None:
        """一部のみ存在する場合、存在するものだけ返す。"""
        partial = (
            '{"key": "20260704_1_2", "feat_win_rate": 5.5, "feat_motor": 33.3}\n'
        )
        records = load_sent_records(_write_sent(partial))
        view = records["20260704_01_02"].feature_view()
        self.assertEqual(view, {"feat_win_rate": 5.5, "feat_motor": 33.3})
        self.assertEqual(len(records["20260704_01_02"].missing_feature_keys()), 5)


# ---------------- ① RenderResult: HTML byte比較 ----------------


class TestLegacyOutputBytes(unittest.TestCase):
    def test_read_output_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.html")
            with open(path, "wb") as f:
                f.write(b"<html>x</html>")
            self.assertEqual(read_output_bytes(path), b"<html>x</html>")

    def test_missing_file_returns_none(self) -> None:
        self.assertIsNone(read_output_bytes("/no/such/file.html"))

    def test_byte_match_no_diff(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "legacy.html")
            p2 = os.path.join(d, "rebuild.html")
            for p in (p1, p2):
                with open(p, "wb") as f:
                    f.write(b"<html>same</html>")
            self.assertEqual(compare_output_bytes("e1", p1, p2), [])

    def test_byte_mismatch_reports_diff(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "legacy.html")
            p2 = os.path.join(d, "rebuild.html")
            with open(p1, "wb") as f:
                f.write(b"<html>legacy</html>")
            with open(p2, "wb") as f:
                f.write(b"<html>rebuild-different</html>")
            diffs = compare_output_bytes("e1", p1, p2)
            self.assertEqual(len(diffs), 1)
            self.assertEqual(
                set(diffs[0].keys()),
                {"eval_id", "field_path", "legacy", "rebuild"},
            )
            self.assertEqual(diffs[0]["field_path"], "$.output.bytes")

    def test_missing_file_is_not_a_diff(self) -> None:
        """Legacy取得元なしは差分にしない（誤検出防止）。"""
        with tempfile.TemporaryDirectory() as d:
            p1 = os.path.join(d, "exists.html")
            with open(p1, "wb") as f:
                f.write(b"x")
            self.assertEqual(
                compare_output_bytes("e1", p1, os.path.join(d, "missing.html")), []
            )


# ---------------- ② NotificationRequest: build_message結果 ----------------


class TestLegacyNotificationView(unittest.TestCase):
    def test_wraps_build_message(self) -> None:
        calls = []

        def _fake_build_message(result):
            calls.append(result)
            return ("【件名】桐生1R", "本文テキスト")

        view = build_notification_view(
            result={"venue": "桐生"}, build_message=_fake_build_message
        )
        self.assertIsInstance(view, LegacyNotificationView)
        self.assertEqual(view.subject, "【件名】桐生1R")
        self.assertEqual(view.body, "本文テキスト")
        self.assertEqual(len(calls), 1)

    def test_only_subject_and_body_held(self) -> None:
        """比較対象はsubject/bodyのみ（送信API情報を持たない）。"""
        view = build_notification_view(
            result=None, build_message=lambda r: ("s", "b")
        )
        self.assertEqual(set(LegacyNotificationView.__slots__), {"subject", "body"})
        self.assertFalse(hasattr(view, "destination"))
        self.assertFalse(hasattr(view, "channel"))

    def test_equality_for_comparison(self) -> None:
        v1 = build_notification_view(None, lambda r: ("s", "b"))
        v2 = build_notification_view(None, lambda r: ("s", "b"))
        v3 = build_notification_view(None, lambda r: ("s", "different"))
        self.assertEqual(v1, v2)
        self.assertNotEqual(v1, v3)

    def test_non_tuple_raises(self) -> None:
        """2要素タプル以外は補完せず例外（デフォルト補完禁止）。"""
        with self.assertRaises(ValueError):
            build_notification_view(None, lambda r: "not a tuple")

    def test_wrong_length_tuple_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_notification_view(None, lambda r: ("a", "b", "c"))

    def test_does_not_call_send_apis(self) -> None:
        """送信APIを呼び出すコードが無いこと（実送信0件の担保）。"""
        import ast

        import shadow.legacy_notification as mod

        with open(mod.__file__, encoding="utf-8") as f:
            tree = ast.parse(f.read())
        # AST上の関数呼び出し名を収集（docstring内の言及は対象外）
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        for send_api in ("send_email", "send_line", "send_notification"):
            self.assertNotIn(send_api, called)


# ---------------- Shadow比較順序の確認（7段階固定） ----------------


class TestStageOrderFixed(unittest.TestCase):
    def test_order_matches_step6_2_spec(self) -> None:
        from shadow.staged_comparator import STAGE_ORDER

        self.assertEqual(
            STAGE_ORDER,
            ("race", "feature_set", "evaluation", "prediction",
             "buy_assessment", "output", "notification_request"),
        )


if __name__ == "__main__":
    unittest.main()


# ---------------- Warning対応: SKIPPED記録とBuyAssessment Pending ----------------


class TestSkippedStagesLogged(unittest.TestCase):
    """スキップ段は単に飛ばさず理由を記録すること。"""

    def test_skip_reason_recorded(self) -> None:
        from shadow.staged_comparator import compare_staged

        legacy = {"race": {"v": 1}, "evaluation": {"u": 9.5}}
        rebuild = {"race": {"v": 1}, "evaluation": {"u": 9.5}}
        result = compare_staged("e1", legacy, rebuild)
        # buy_assessmentは両方に無い→SKIPPEDとして理由が残る
        self.assertIn("buy_assessment", result.skipped_stages)
        self.assertIn("取得元なし", result.skipped_stages["buy_assessment"])

    def test_skip_reason_distinguishes_side(self) -> None:
        from shadow.staged_comparator import compare_staged

        result = compare_staged(
            "e1", {"race": {"v": 1}}, {"race": {"v": 1}, "prediction": {"c": "x"}}
        )
        self.assertIn("legacy側", result.skipped_stages["prediction"])

    def test_summary_lines_show_skipped(self) -> None:
        from shadow.staged_comparator import compare_staged

        legacy = {"race": {"v": 1}}
        rebuild = {"race": {"v": 1}}
        lines = compare_staged("e1", legacy, rebuild).summary_lines()
        joined = "\n".join(lines)
        self.assertIn("race", joined)
        self.assertIn("✔", joined)
        self.assertIn("SKIPPED", joined)
        self.assertIn("buy_assessment", joined)

    def test_summary_lines_show_all_seven_stages(self) -> None:
        from shadow.staged_comparator import STAGE_ORDER, compare_staged

        lines = compare_staged("e1", {}, {}).summary_lines()
        self.assertEqual(len(lines), len(STAGE_ORDER))

    def test_all_matched_true_even_with_skips(self) -> None:
        """スキップは差分ではない（一致判定を妨げない）。"""
        from shadow.staged_comparator import compare_staged

        result = compare_staged("e1", {"race": {"v": 1}}, {"race": {"v": 1}})
        self.assertTrue(result.all_matched)
        self.assertTrue(len(result.skipped_stages) > 0)


class TestBuyAssessmentPending(unittest.TestCase):
    """BuyAssessmentは「一致」ではなく「取得元未確定=Pending」扱い。"""

    def test_pending_recorded_when_source_unavailable(self) -> None:
        from shadow.go_no_go import GoNoGoCriteria, evaluate_go_no_go

        result = evaluate_go_no_go(GoNoGoCriteria(
            golden_100_percent=True, shadow_consecutive_matches=100,
        ))
        self.assertEqual(result.decision, "GO")
        self.assertTrue(any("BuyAssessment" in p for p in result.pendings))
        self.assertTrue(any("取得元未確定" in p for p in result.pendings))

    def test_pending_does_not_block_go(self) -> None:
        """取得元未確定でもGO判定を妨げない（判定対象外）。"""
        from shadow.go_no_go import GoNoGoCriteria, evaluate_go_no_go

        result = evaluate_go_no_go(GoNoGoCriteria(
            golden_100_percent=True, shadow_consecutive_matches=100,
            buy_assessment_source_available=False, buy_assessment_match=False,
        ))
        self.assertEqual(result.decision, "GO")

    def test_enabled_source_is_judged(self) -> None:
        """取得元確定後は判定対象になり、不一致ならNO_GO。"""
        from shadow.go_no_go import GoNoGoCriteria, evaluate_go_no_go

        result = evaluate_go_no_go(GoNoGoCriteria(
            golden_100_percent=True, shadow_consecutive_matches=100,
            buy_assessment_source_available=True, buy_assessment_match=False,
        ))
        self.assertEqual(result.decision, "NO_GO")
        self.assertTrue(any("BuyAssessment" in r for r in result.reasons))
        # 有効化後はPendingに載らない
        self.assertFalse(any("BuyAssessment" in p for p in result.pendings))

    def test_enabled_and_matched_is_go(self) -> None:
        from shadow.go_no_go import GoNoGoCriteria, evaluate_go_no_go

        result = evaluate_go_no_go(GoNoGoCriteria(
            golden_100_percent=True, shadow_consecutive_matches=100,
            buy_assessment_source_available=True, buy_assessment_match=True,
        ))
        self.assertEqual(result.decision, "GO")
        self.assertEqual(result.pendings, ())


# ---------------- 集計サマリー（100レース並走後の分析用） ----------------


class TestShadowSummary(unittest.TestCase):
    def _matched(self, eval_id, stages, skipped=None):
        from shadow.staged_comparator import StagedResult

        r = StagedResult(eval_id=eval_id, matched_stages=list(stages))
        r.skipped_stages = skipped or {}
        return r

    def _diff(self, eval_id, matched, stopped_at, field_path):
        from shadow.staged_comparator import StagedResult

        r = StagedResult(eval_id=eval_id, matched_stages=list(matched))
        r.stopped_at = stopped_at
        r.diffs = [{
            "eval_id": eval_id, "field_path": field_path,
            "legacy": 1.0, "rebuild": 2.0,
        }]
        return r

    def test_counts_processed_races(self) -> None:
        from shadow.staged_comparator import ShadowSummary

        s = ShadowSummary()
        for i in range(100):
            s.record(self._matched(f"e{i}", ["race"]))
        self.assertEqual(s.processed_races, 100)

    def test_counts_matched_per_stage(self) -> None:
        from shadow.staged_comparator import ShadowSummary

        s = ShadowSummary()
        for i in range(5):
            s.record(self._matched(f"e{i}", ["race", "evaluation"]))
        self.assertEqual(s.matched_per_stage["race"], 5)
        self.assertEqual(s.matched_per_stage["evaluation"], 5)

    def test_counts_skipped_per_stage(self) -> None:
        from shadow.staged_comparator import ShadowSummary

        s = ShadowSummary()
        for i in range(3):
            s.record(self._matched(
                f"e{i}", ["race"], {"buy_assessment": "取得元なし"}
            ))
        self.assertEqual(s.skipped_per_stage["buy_assessment"], 3)

    def test_counts_diff_per_stage(self) -> None:
        from shadow.staged_comparator import ShadowSummary

        s = ShadowSummary()
        s.record(self._diff("x1", ["race"], "prediction", "$.prediction.p"))
        s.record(self._diff("x2", ["race"], "prediction", "$.prediction.p"))
        self.assertEqual(s.diff_per_stage["prediction"], 2)

    def test_first_mismatch_recorded(self) -> None:
        from shadow.staged_comparator import ShadowSummary

        s = ShadowSummary()
        s.record(self._matched("ok1", ["race"]))
        s.record(self._diff("bad1", ["race"], "prediction", "$.prediction.prob"))
        s.record(self._diff("bad2", ["race"], "evaluation", "$.evaluation.danger"))
        self.assertIsNotNone(s.first_mismatch)
        # 最初の差分のみ保持（後続で上書きしない）
        self.assertEqual(s.first_mismatch["eval_id"], "bad1")
        self.assertEqual(s.first_mismatch["stage"], "prediction")
        self.assertEqual(s.first_mismatch["field_path"], "$.prediction.prob")

    def test_first_mismatch_none_when_all_matched(self) -> None:
        from shadow.staged_comparator import ShadowSummary

        s = ShadowSummary()
        for i in range(10):
            s.record(self._matched(f"e{i}", ["race"]))
        self.assertIsNone(s.first_mismatch)

    def test_summary_lines_format(self) -> None:
        from shadow.staged_comparator import ShadowSummary

        s = ShadowSummary()
        for i in range(97):
            s.record(self._matched(
                f"e{i}", ["race", "prediction"],
                {"buy_assessment": "取得元なし"},
            ))
        s.record(self._diff("bad", ["race"], "prediction", "$.prediction.prob"))
        lines = s.summary_lines()
        joined = "\n".join(lines)
        self.assertIn("Processed races: 98", joined)
        self.assertIn("race", joined)
        self.assertIn("First mismatch:", joined)
        self.assertIn("bad", joined)

    def test_pending_stage_shown_as_pending(self) -> None:
        """一度も比較されずスキップのみの段はPendingと表示。"""
        from shadow.staged_comparator import ShadowSummary

        s = ShadowSummary()
        for i in range(100):
            s.record(self._matched(
                f"e{i}", ["race"], {"buy_assessment": "取得元なし"}
            ))
        joined = "\n".join(s.summary_lines())
        self.assertIn("buy_assessment", joined)
        self.assertIn("Pending (skipped: 100)", joined)


# ---------------- decision_line（表示専用・判定ロジック不変） ----------------


class TestDecisionLine(unittest.TestCase):
    def _criteria(self, **over):
        from shadow.go_no_go import GoNoGoCriteria

        base = dict(golden_100_percent=True, shadow_consecutive_matches=100)
        base.update(over)
        return GoNoGoCriteria(**base)

    def test_go_with_pending(self) -> None:
        from shadow.go_no_go import evaluate_go_no_go

        result = evaluate_go_no_go(self._criteria())
        self.assertEqual(
            result.decision_line(),
            "GO (Pending: BuyAssessment source unavailable)",
        )

    def test_go_without_pending(self) -> None:
        from shadow.go_no_go import evaluate_go_no_go

        result = evaluate_go_no_go(self._criteria(
            buy_assessment_source_available=True, buy_assessment_match=True,
        ))
        self.assertEqual(result.decision_line(), "GO")

    def test_no_go(self) -> None:
        from shadow.go_no_go import evaluate_go_no_go

        result = evaluate_go_no_go(self._criteria(golden_100_percent=False))
        self.assertEqual(result.decision_line(), "NO_GO")

    def test_no_go_with_pending_shows_no_go_only(self) -> None:
        """NO_GO時はPendingを併記しない（判定が主）。"""
        from shadow.go_no_go import evaluate_go_no_go

        result = evaluate_go_no_go(self._criteria(shadow_consecutive_matches=0))
        self.assertEqual(result.decision_line(), "NO_GO")

    def test_decision_field_unchanged_by_display(self) -> None:
        """decision_lineはdecisionフィールドを変更しない（表示専用）。"""
        from shadow.go_no_go import evaluate_go_no_go

        result = evaluate_go_no_go(self._criteria())
        before = result.decision
        result.decision_line()
        self.assertEqual(result.decision, before)
        self.assertEqual(result.decision, "GO")
