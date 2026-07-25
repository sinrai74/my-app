"""
shadow/comparator.py と shadow/go_no_go.py のテスト（Step5-6レビュー反映）。

確認項目:
  Serializer利用 / dataclass利用 / __dict__利用 / __slots__利用 /
  JSON比較 / 差分検出 / Go / No-Go。
"""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass

from shadow.comparator import compare, to_comparable
from shadow.go_no_go import GoNoGoCriteria, evaluate_go_no_go

from models.evaluation import FeatureSet, Prediction, RaceEvaluation


def _evaluation(eval_id="20260704_12_05", **over) -> RaceEvaluation:
    base = dict(
        eval_id=eval_id, race_date="20260704", venue_num=12, venue_name="住之江",
        race_number=5, is_night=True, engine_name="ver4", engine_version="4.0.0",
        feature_schema_version=1,
        features=FeatureSet(
            eval_id=eval_id, feature_schema_version=1, built_at="t",
            boat_features={1: {}}, race_features={}, local_features=None,
            missing_keys=(),
        ),
        model_version="m", evaluated_at="t", danger_score=10.0,
        danger_breakdown={}, upset_score=5.0, upset_reasons=(),
        rank_index={}, featured_boats=None, win_probs=None, race_type="",
        match_index=52.5,
    )
    base.update(over)
    return RaceEvaluation(**base)


def _prediction() -> Prediction:
    return Prediction(
        eval_id="20260704_12_05", pred_combo="1-2-3", pred_prob=0.06,
        pred_ev=2.0, pred_odds=35.0, confidence=0.5, why_bet="x", patterns=(),
    )


class TestToComparableSerializer(unittest.TestCase):
    """段階1: 既存Serializer利用。"""

    def test_race_evaluation_uses_serializer(self) -> None:
        d = to_comparable(_evaluation())
        self.assertIsInstance(d, dict)
        self.assertEqual(d["eval_id"], "20260704_12_05")
        self.assertIn("danger_score", d)

    def test_prediction_uses_serializer(self) -> None:
        d = to_comparable(_prediction())
        self.assertIsInstance(d, dict)
        self.assertEqual(d["pred_combo"], "1-2-3")


class TestToComparableDataclass(unittest.TestCase):
    """段階2: dataclass → asdict。"""

    def test_plain_dataclass(self) -> None:
        @dataclass
        class _Sample:
            a: int
            b: str

        d = to_comparable(_Sample(a=1, b="x"))
        self.assertEqual(d, {"a": 1, "b": "x"})

    def test_nested_dataclass(self) -> None:
        @dataclass
        class _Inner:
            v: int

        @dataclass
        class _Outer:
            inner: _Inner

        d = to_comparable(_Outer(inner=_Inner(v=5)))
        self.assertEqual(d, {"inner": {"v": 5}})


class TestToComparableDict(unittest.TestCase):
    """段階3: __dict__を持つ通常オブジェクト。"""

    def test_plain_object_with_dict(self) -> None:
        class _Obj:
            def __init__(self):
                self.x = 1
                self.y = "z"

        d = to_comparable(_Obj())
        self.assertEqual(d, {"x": 1, "y": "z"})


class TestToComparableSlots(unittest.TestCase):
    """段階4: __dict__が無く__slots__を持つオブジェクト（汎用対応）。"""

    def test_slots_object(self) -> None:
        class _Slotted:
            __slots__ = ("a", "b")

            def __init__(self, a, b):
                self.a = a
                self.b = b

        d = to_comparable(_Slotted(a=1, b=2))
        self.assertEqual(d, {"a": 1, "b": 2})

    def test_render_result_slots(self) -> None:
        """RenderResult（__slots__）が型別ifなしで変換されること。"""
        from output.renderers import RenderResult

        d = to_comparable(RenderResult(output_path="/x.html", summary={"n": 1}))
        self.assertEqual(d, {"output_path": "/x.html", "summary": {"n": 1}})

    def test_notification_request_slots(self) -> None:
        """NotificationRequest（__slots__）が型別ifなしで変換されること。"""
        from notification.notifiers import NotificationRequest
        from output.renderers import RenderResult

        req = NotificationRequest(
            render_result=RenderResult("/x.html", {"n": 1}), channel="mail",
            destination="a@b.com", title="件名", attachment_path=None,
        )
        d = to_comparable(req)
        self.assertEqual(d["channel"], "mail")
        self.assertEqual(d["destination"], "a@b.com")
        self.assertEqual(d["title"], "件名")
        self.assertEqual(d["render_result"], {"output_path": "/x.html",
                                              "summary": {"n": 1}})

    def test_inherited_slots(self) -> None:
        """継承階層の__slots__も汎用列挙されること。"""
        class _Base:
            __slots__ = ("a",)

            def __init__(self):
                self.a = 1

        class _Child(_Base):
            __slots__ = ("b",)

            def __init__(self):
                super().__init__()
                self.b = 2

        d = to_comparable(_Child())
        self.assertEqual(d, {"a": 1, "b": 2})


class TestJsonComparable(unittest.TestCase):
    """JSON比較: to_comparableの結果がJSON化可能であること。"""

    def test_render_result_json_serializable(self) -> None:
        from output.renderers import RenderResult

        d = to_comparable(RenderResult("/x.html", {"n": 1}))
        # 例外なくJSON化できる（__slots__非対応だと以前はTypeErrorだった）
        self.assertEqual(json.loads(json.dumps(d)), {"output_path": "/x.html",
                                                     "summary": {"n": 1}})

    def test_evaluation_json_serializable(self) -> None:
        d = to_comparable(_evaluation())
        json.dumps(d, ensure_ascii=False)  # 例外が出ないこと


class TestCompareDiff(unittest.TestCase):
    """差分検出（eval_id/field_path/legacy/rebuild形式）。"""

    def test_no_diff_when_equal(self) -> None:
        self.assertEqual(compare("e1", _evaluation(), _evaluation()), [])

    def test_diff_detected(self) -> None:
        diffs = compare("e1", _evaluation(danger_score=10.0),
                        _evaluation(danger_score=99.0), path="$.eval")
        self.assertTrue(any(d["field_path"] == "$.eval.danger_score" for d in diffs))
        target = [d for d in diffs if d["field_path"] == "$.eval.danger_score"][0]
        self.assertEqual(set(target.keys()),
                         {"eval_id", "field_path", "legacy", "rebuild"})
        self.assertEqual(target["legacy"], 10.0)
        self.assertEqual(target["rebuild"], 99.0)

    def test_slots_object_diff(self) -> None:
        from output.renderers import RenderResult

        diffs = compare(
            "e1",
            RenderResult("/a.html", {}), RenderResult("/b.html", {}),
            path="$.output",
        )
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["field_path"], "$.output.output_path")
        self.assertEqual(diffs[0]["legacy"], "/a.html")
        self.assertEqual(diffs[0]["rebuild"], "/b.html")


class TestGoNoGo(unittest.TestCase):
    """Go / No-Go（整理後も判定内容不変）。"""

    def test_go(self) -> None:
        criteria = GoNoGoCriteria(
            golden_100_percent=True, shadow_consecutive_matches=100,
            output_byte_match=True, notification_request_match=True,
            shadow_real_sends=0, feature_freeze_intact=True, all_tests_passed=True,
        )
        self.assertEqual(evaluate_go_no_go(criteria).decision, "GO")

    def test_no_go_single_reason(self) -> None:
        result = evaluate_go_no_go(GoNoGoCriteria(
            golden_100_percent=True, shadow_consecutive_matches=99))
        self.assertEqual(result.decision, "NO_GO")
        self.assertTrue(any("G2/G3" in r for r in result.reasons))

    def test_no_go_all_reasons_listed(self) -> None:
        result = evaluate_go_no_go(GoNoGoCriteria(
            golden_100_percent=False, shadow_consecutive_matches=0,
            output_byte_match=False, notification_request_match=False,
            shadow_real_sends=2, feature_freeze_intact=False,
            all_tests_passed=False,
        ))
        self.assertEqual(result.decision, "NO_GO")
        # 全7条件が未充足 → 7理由すべて列挙
        self.assertEqual(len(result.reasons), 7)

    def test_no_go_real_send(self) -> None:
        result = evaluate_go_no_go(GoNoGoCriteria(
            golden_100_percent=True, shadow_consecutive_matches=100,
            shadow_real_sends=1))
        self.assertTrue(any("G6" in r for r in result.reasons))


if __name__ == "__main__":
    unittest.main()
