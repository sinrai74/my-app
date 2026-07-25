"""
Step6-1（Shadow実体結線）のテスト。

対象:
  - shadow/legacy_source.py（sent_*.txt読み取り・Legacy無変更）
  - shadow/prediction_provider.py（_evaluate_betsラップ）
  - shadow/staged_comparator.py（段階比較・100連続一致）

実装範囲の注記（Step6-1レビュー反映）:
  Legacy実データとの実比較は Race / RaceEvaluation / Prediction /
  BuyAssessment の4段階のみ（取得元=sent_*.txt）。
  FeatureSet / RenderResult / NotificationRequest は比較器が対応済みだが、
  Legacy側取得元が未確定のため実比較はStep6-2で実装する。

Legacy実コードへ接続せず、_evaluate_betsはFakeをDIする。
"""

from __future__ import annotations

import os
import tempfile
import unittest

from shadow.legacy_source import (
    LegacySentRecord,
    get_legacy_record,
    load_sent_records,
)
from shadow.prediction_provider import (
    LegacyPredictionProvider,
    PredictionContext,
    _default_result_mapper,
)
from shadow.staged_comparator import (
    STAGE_ORDER,
    ConsecutiveMatchCounter,
    StagedResult,
    compare_staged,
)

from models.evaluation import FeatureSet, Prediction, RaceEvaluation


# ---------------- legacy_source ----------------


_SENT_SAMPLE = (
    '{"key": "__no_bets_evaluated__", "checked_races": 180}\n'
    '{"key": "20260628_4_12", "combo": "2-1-3", "buy": ["2-1-3", "4-1-2"], '
    '"odds": 60.0, "prob": 0.03422, "ev": 2.053, "confidence": 0.6177, '
    '"race_type": "1残り荒れ型", "upset_score": 9.5, "venue": "平和島", '
    '"venue_num": 4, "race": 12, "night": 1}\n'
    '{"key": "20260704_1_1", "combo": "6-1-3", "buy": ["6-1-3"], '
    '"odds": 66.0, "prob": 0.04591, "ev": 3.03, "confidence": 0.8007, '
    '"race_type": "1残り荒れ型", "upset_score": 9.5, "venue": "桐生", '
    '"venue_num": 1, "race": 1, "night": 0}\n'
)


class TestLoadSentRecords(unittest.TestCase):
    def _write(self, content):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "sent_20260628.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def test_skips_meta_rows(self) -> None:
        records = load_sent_records(self._write(_SENT_SAMPLE))
        # __no_bets_evaluated__ はスキップ、2レースのみ
        self.assertEqual(set(records.keys()), {"20260628_04_12", "20260704_01_01"})

    def test_missing_file_returns_empty(self) -> None:
        self.assertEqual(load_sent_records("/no/such/file.txt"), {})

    def test_empty_lines_skipped(self) -> None:
        records = load_sent_records(self._write("\n\n" + _SENT_SAMPLE + "\n"))
        self.assertEqual(len(records), 2)

    def test_prediction_view(self) -> None:
        records = load_sent_records(self._write(_SENT_SAMPLE))
        view = records["20260628_04_12"].prediction_view()
        self.assertEqual(view["pred_combo"], "2-1-3")
        self.assertEqual(view["pred_odds"], 60.0)
        self.assertEqual(view["pred_ev"], 2.053)

    def test_race_view(self) -> None:
        records = load_sent_records(self._write(_SENT_SAMPLE))
        view = records["20260628_04_12"].race_view()
        self.assertEqual(view["venue_num"], 4)
        self.assertEqual(view["race_number"], 12)
        self.assertTrue(view["is_night"])

    def test_evaluation_view_only_present_keys(self) -> None:
        records = load_sent_records(self._write(_SENT_SAMPLE))
        view = records["20260628_04_12"].evaluation_view()
        self.assertEqual(view["upset_score"], 9.5)
        self.assertEqual(view["race_type"], "1残り荒れ型")
        # danger_score_v3はサンプルに無い→キーが無い（補完しない）
        self.assertNotIn("danger_score", view)

    def test_buy_view_only_present_keys(self) -> None:
        records = load_sent_records(self._write(_SENT_SAMPLE))
        # sampleにbuyscore等が無い→空dict（仮値補完しない）
        self.assertEqual(records["20260628_04_12"].buy_view(), {})

    def test_get_legacy_record(self) -> None:
        path = self._write(_SENT_SAMPLE)
        rec = get_legacy_record(path, "20260704_01_01")
        self.assertIsInstance(rec, LegacySentRecord)
        self.assertEqual(rec.prediction_view()["pred_combo"], "6-1-3")


# ---------------- prediction_provider ----------------


def _evaluation(eval_id="20260704_01_01") -> RaceEvaluation:
    return RaceEvaluation(
        eval_id=eval_id, race_date="20260704", venue_num=1, venue_name="桐生",
        race_number=1, is_night=False, engine_name="ver4", engine_version="4.0.0",
        feature_schema_version=1,
        features=FeatureSet(
            eval_id=eval_id, feature_schema_version=1, built_at="t",
            boat_features={1: {}}, race_features={}, local_features=None,
            missing_keys=(),
        ),
        model_version="m", evaluated_at="t", danger_score=10.0,
        danger_breakdown={}, upset_score=9.5, upset_reasons=(),
        rank_index={}, featured_boats=None, win_probs=None, race_type="",
        match_index=52.5,
    )


class TestLegacyPredictionProvider(unittest.TestCase):
    def _context_resolver(self, evaluation):
        return PredictionContext(
            patterns={"1-2-3": []}, ml_probs={1: 0.5}, odds_map={"6-1-3": 66.0},
        )

    def test_wraps_evaluate_bets(self) -> None:
        captured = {}

        def _fake_evaluate_bets(**kwargs):
            captured.update(kwargs)
            return {"combo": "6-1-3", "prob": 0.04591, "ev": 3.03, "odds": 66.0,
                    "confidence": 0.8007, "buy": ["6-1-3"], "why_bet": ["x"]}

        provider = LegacyPredictionProvider(
            context_resolver=self._context_resolver,
            evaluate_bets=_fake_evaluate_bets,
        )
        prediction = provider.provide(_evaluation(), {})
        self.assertIsInstance(prediction, Prediction)
        self.assertEqual(prediction.pred_combo, "6-1-3")
        self.assertEqual(prediction.pred_ev, 3.03)
        # eval由来の識別子が_evaluate_betsへ渡る
        self.assertEqual(captured["venue_num"], 1)
        self.assertEqual(captured["race_number"], 1)
        self.assertEqual(captured["race_date"], "20260704")

    def test_result_mapper_reads_only(self) -> None:
        prediction = _default_result_mapper(
            {"combo": "1-2-3", "prob": 0.06, "ev": 2.0, "odds": 35.0,
             "confidence": 0.5},
            _evaluation(),
        )
        self.assertEqual(prediction.pred_combo, "1-2-3")
        self.assertEqual(prediction.pred_prob, 0.06)

    def test_non_dict_result_raises(self) -> None:
        with self.assertRaises(ValueError):
            _default_result_mapper(("tuple", "form"), _evaluation())

    def test_none_result_raises(self) -> None:
        """None戻り値は仮値で埋めず例外（デフォルト補完禁止）。"""
        with self.assertRaises(ValueError):
            _default_result_mapper(None, _evaluation())

    def test_missing_required_key_raises(self) -> None:
        """必須キー欠損時は0.0等で補完せず例外（比較不能として停止）。"""
        with self.assertRaises(ValueError) as ctx:
            _default_result_mapper(
                {"combo": "1-2-3", "prob": 0.06},  # ev/odds/confidence欠損
                _evaluation(),
            )
        self.assertIn("missing required keys", str(ctx.exception))

    def test_no_default_value_supplied(self) -> None:
        """空dictでも0.0を入れずに例外を送出すること。"""
        with self.assertRaises(ValueError):
            _default_result_mapper({}, _evaluation())


# ---------------- staged_comparator ----------------


class TestStagedCompare(unittest.TestCase):
    def test_all_stages_match(self) -> None:
        legacy = {"race": {"venue_num": 1}, "evaluation": {"upset_score": 9.5}}
        rebuild = {"race": {"venue_num": 1}, "evaluation": {"upset_score": 9.5}}
        result = compare_staged("e1", legacy, rebuild)
        self.assertTrue(result.all_matched)
        self.assertIsNone(result.stopped_at)
        self.assertIn("race", result.matched_stages)

    def test_stops_at_first_diff(self) -> None:
        # raceが違えば後段（evaluation）は見ない
        legacy = {"race": {"venue_num": 1}, "evaluation": {"upset_score": 9.5}}
        rebuild = {"race": {"venue_num": 2}, "evaluation": {"upset_score": 0.0}}
        result = compare_staged("e1", legacy, rebuild)
        self.assertFalse(result.all_matched)
        self.assertEqual(result.stopped_at, "race")
        # evaluationの差分は含まれない（前段で停止）
        self.assertTrue(all("race" in d["field_path"] for d in result.diffs))

    def test_stage_order_respected(self) -> None:
        # evaluationのみ差分→raceは一致してからevaluationで止まる
        legacy = {"race": {"v": 1}, "evaluation": {"upset_score": 9.5}}
        rebuild = {"race": {"v": 1}, "evaluation": {"upset_score": 1.0}}
        result = compare_staged("e1", legacy, rebuild)
        self.assertEqual(result.stopped_at, "evaluation")
        self.assertIn("race", result.matched_stages)

    def test_missing_stage_skipped(self) -> None:
        # 片方に無い段はスキップ（差分ではない）
        legacy = {"race": {"v": 1}}
        rebuild = {"race": {"v": 1}, "evaluation": {"upset_score": 9.5}}
        result = compare_staged("e1", legacy, rebuild)
        self.assertTrue(result.all_matched)

    def test_stage_order_constant(self) -> None:
        self.assertEqual(
            STAGE_ORDER,
            ("race", "feature_set", "evaluation", "prediction",
             "buy_assessment", "output", "notification_request"),
        )


class TestConsecutiveMatchCounter(unittest.TestCase):
    def _matched(self, eval_id="e"):
        return StagedResult(eval_id=eval_id, matched_stages=["race"], stopped_at=None)

    def _diff(self, eval_id="e"):
        r = StagedResult(eval_id=eval_id)
        r.stopped_at = "race"
        return r

    def test_counts_consecutive(self) -> None:
        counter = ConsecutiveMatchCounter(required=3)
        for _ in range(3):
            counter.record(self._matched())
        self.assertTrue(counter.satisfied)
        self.assertEqual(counter.current_streak, 3)

    def test_diff_resets_streak(self) -> None:
        counter = ConsecutiveMatchCounter(required=100)
        for _ in range(50):
            counter.record(self._matched())
        counter.record(self._diff("bad"))  # 1件差分→0へ
        self.assertEqual(counter.current_streak, 0)
        self.assertEqual(counter.max_streak, 50)
        self.assertIn("bad", counter.broken_at)
        self.assertFalse(counter.satisfied)

    def test_100_consecutive_required_not_100_of_105(self) -> None:
        """105件中100一致でも、途中で差分があれば連続ではない。"""
        counter = ConsecutiveMatchCounter(required=100)
        for _ in range(99):
            counter.record(self._matched())
        counter.record(self._diff())  # 100件目で差分
        for _ in range(5):
            counter.record(self._matched())
        # 累計104一致だが連続は5→未達
        self.assertEqual(counter.current_streak, 5)
        self.assertFalse(counter.satisfied)

    def test_satisfied_at_exactly_required(self) -> None:
        counter = ConsecutiveMatchCounter(required=100)
        for _ in range(100):
            counter.record(self._matched())
        self.assertTrue(counter.satisfied)
        self.assertEqual(counter.total_seen, 100)


if __name__ == "__main__":
    unittest.main()


# ---------------- ③ Shadow比較: 各段階を実モデルで検証 ----------------


from output.renderers import RenderResult
from notification.notifiers import NotificationRequest
from core.buyscore import BuyAssessment
from models.race import Race, RaceEntry


def _race_obj(venue_num=1) -> Race:
    return Race(
        race_date="20260704", venue_num=venue_num, venue_name="桐生",
        race_number=1, close_time="10:00", is_night=False,
        entries=(RaceEntry(
            lane=1, racer_no="4001", racer_name="A", racer_class="A1",
            win_rate=6.5, place_rate=0.0, motor_no=0, motor_rate2=38.0,
            avg_st=0.16,
        ),),
        grade="一般",
    )


def _feature_set_obj(eval_id="20260704_01_01") -> FeatureSet:
    return FeatureSet(
        eval_id=eval_id, feature_schema_version=1, built_at="t",
        boat_features={1: {}}, race_features={}, local_features=None,
        missing_keys=(),
    )


def _buy_assessment_obj(eval_id="20260704_01_01", **over) -> BuyAssessment:
    base = dict(eval_id=eval_id, buyscore=72.0, investment_type="穴狙い",
                kelly_fraction=0.01, skip_reason=None, config_version="t")
    base.update(over)
    return BuyAssessment(**base)


def _render_result_obj(path="/out/public.html") -> RenderResult:
    return RenderResult(output_path=path, summary={"n": 1})


def _notification_request_obj(channel="mail") -> NotificationRequest:
    return NotificationRequest(
        render_result=_render_result_obj(), channel=channel,
        destination="a@b.com", title="件名", attachment_path=None,
    )


class TestStagedCompareEachStageRealModels(unittest.TestCase):
    """比較器（compare_staged）が7段階の各モデル型を扱えることの検証。

    重要（Step6-1レビュー反映・実装範囲の正確な記述）:
      本クラスはlegacy側・rebuild側の双方を合成オブジェクトで用意し、
      「比較器が各段階の型を正しく比較・停止できるか」を検証するもの。
      Legacy実データとの実比較ではない。

    Legacy実データ比較の現状:
      - 実比較済み（sent_*.txtから取得）: Race / RaceEvaluation /
        Prediction / BuyAssessment
      - 比較器のみ（Legacy取得元が未確定・Step6-2で実装）:
        FeatureSet / RenderResult / NotificationRequest
    """

    def test_race_stage_match_and_diff(self) -> None:
        legacy = {"race": _race_obj(venue_num=1)}
        rebuild_match = {"race": _race_obj(venue_num=1)}
        rebuild_diff = {"race": _race_obj(venue_num=2)}
        self.assertTrue(compare_staged("e1", legacy, rebuild_match).all_matched)
        r = compare_staged("e1", legacy, rebuild_diff)
        self.assertEqual(r.stopped_at, "race")

    def test_feature_set_stage_match_and_diff(self) -> None:
        legacy = {"race": _race_obj(), "feature_set": _feature_set_obj()}
        rebuild_match = {"race": _race_obj(), "feature_set": _feature_set_obj()}
        rebuild_diff = {
            "race": _race_obj(),
            "feature_set": _feature_set_obj().__class__(
                eval_id="20260704_01_01", feature_schema_version=2,  # 差分
                built_at="t", boat_features={1: {}}, race_features={},
                local_features=None, missing_keys=(),
            ),
        }
        self.assertTrue(compare_staged("e1", legacy, rebuild_match).all_matched)
        r = compare_staged("e1", legacy, rebuild_diff)
        self.assertEqual(r.stopped_at, "feature_set")

    def test_race_evaluation_stage_match_and_diff(self) -> None:
        eval_a = _evaluation(eval_id="e1")
        eval_b = RaceEvaluation(
            eval_id="e1", race_date="20260704", venue_num=1, venue_name="桐生",
            race_number=1, is_night=False, engine_name="ver4",
            engine_version="4.0.0", feature_schema_version=1,
            features=_feature_set_obj(eval_id="e1"),
            model_version="m", evaluated_at="t", danger_score=99.0,  # 差分
            danger_breakdown={}, upset_score=9.5, upset_reasons=(),
            rank_index={}, featured_boats=None, win_probs=None, race_type="",
            match_index=52.5,
        )
        legacy = {"evaluation": eval_a}
        rebuild_match = {"evaluation": _evaluation(eval_id="e1")}
        rebuild_diff = {"evaluation": eval_b}
        self.assertTrue(compare_staged("e1", legacy, rebuild_match).all_matched)
        r = compare_staged("e1", legacy, rebuild_diff)
        self.assertEqual(r.stopped_at, "evaluation")

    def test_prediction_stage_match_and_diff(self) -> None:
        p1 = Prediction(eval_id="e1", pred_combo="1-2-3", pred_prob=0.06,
                        pred_ev=2.0, pred_odds=35.0, confidence=0.5,
                        why_bet="x", patterns=())
        p2 = Prediction(eval_id="e1", pred_combo="3-2-1", pred_prob=0.06,
                        pred_ev=2.0, pred_odds=35.0, confidence=0.5,
                        why_bet="x", patterns=())
        legacy = {"prediction": p1}
        self.assertTrue(
            compare_staged("e1", legacy, {"prediction": p1}).all_matched
        )
        r = compare_staged("e1", legacy, {"prediction": p2})
        self.assertEqual(r.stopped_at, "prediction")

    def test_buy_assessment_stage_match_and_diff(self) -> None:
        legacy = {"buy_assessment": _buy_assessment_obj()}
        rebuild_match = {"buy_assessment": _buy_assessment_obj()}
        rebuild_diff = {"buy_assessment": _buy_assessment_obj(buyscore=1.0)}
        self.assertTrue(compare_staged("e1", legacy, rebuild_match).all_matched)
        r = compare_staged("e1", legacy, rebuild_diff)
        self.assertEqual(r.stopped_at, "buy_assessment")

    def test_render_result_stage_match_and_diff(self) -> None:
        legacy = {"output": _render_result_obj("/a.html")}
        rebuild_match = {"output": _render_result_obj("/a.html")}
        rebuild_diff = {"output": _render_result_obj("/b.html")}
        self.assertTrue(compare_staged("e1", legacy, rebuild_match).all_matched)
        r = compare_staged("e1", legacy, rebuild_diff)
        self.assertEqual(r.stopped_at, "output")

    def test_notification_request_stage_match_and_diff(self) -> None:
        legacy = {"notification_request": _notification_request_obj("mail")}
        rebuild_match = {"notification_request": _notification_request_obj("mail")}
        rebuild_diff = {"notification_request": _notification_request_obj("line")}
        self.assertTrue(compare_staged("e1", legacy, rebuild_match).all_matched)
        r = compare_staged("e1", legacy, rebuild_diff)
        self.assertEqual(r.stopped_at, "notification_request")

    def test_full_seven_stage_order_all_match(self) -> None:
        """7段階すべて一致すればall_matchedがTrueで、matched_stagesが順序通り。"""
        legacy = {
            "race": _race_obj(), "feature_set": _feature_set_obj(),
            "evaluation": _evaluation(eval_id="e1"),
            "prediction": Prediction(eval_id="e1", pred_combo="1-2-3",
                                     pred_prob=0.06, pred_ev=2.0, pred_odds=35.0,
                                     confidence=0.5, why_bet="x", patterns=()),
            "buy_assessment": _buy_assessment_obj(eval_id="e1"),
            "output": _render_result_obj(),
            "notification_request": _notification_request_obj(),
        }
        rebuild = dict(legacy)  # 完全一致
        result = compare_staged("e1", legacy, rebuild)
        self.assertTrue(result.all_matched)
        self.assertEqual(list(result.matched_stages), list(STAGE_ORDER))

    def test_no_correction_applied_on_diff(self) -> None:
        """差分検出時、値は補正されず記録されるだけ（rebuild値は変更されない）。"""
        legacy = {"race": _race_obj(venue_num=1)}
        rebuild_obj = _race_obj(venue_num=2)
        rebuild = {"race": rebuild_obj}
        compare_staged("e1", legacy, rebuild)
        # 比較後もrebuild_obj自体は不変（補正されていない）
        self.assertEqual(rebuild["race"].venue_num, 2)
        self.assertIs(rebuild["race"], rebuild_obj)


# ---------------- ⑤ shadow_run.yml 存在確認・既存Workflow無変更確認 ----------------


class TestWorkflowFiles(unittest.TestCase):
    """shadow_run.yml追加確認と既存Workflow無変更確認。"""

    def _repo_root(self):
        # tests/unit/ から3階層上がリポジトリルート想定
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(here, "..", "..", ".."))

    def test_shadow_run_yml_exists(self) -> None:
        path = os.path.join(self._repo_root(), "shadow_run.yml")
        if not os.path.exists(path):
            self.skipTest("shadow_run.yml not deployed at repo root in this env")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn("shadow-run", content)
        self.assertIn("USE_REBUILD_PIPELINE", content)

    def test_shadow_run_yml_does_not_modify_notify_arashi(self) -> None:
        """shadow_run.ymlがnotify_arashi.ymlを書き換える処理を持たないこと。

        コメントでの言及（分離方針の説明）は許容し、実際にファイルへ
        書き込む/編集する操作（cp/mv/sed/echo >>notify_arashi等）が
        無いことを確認する。
        """
        path = os.path.join(self._repo_root(), "shadow_run.yml")
        if not os.path.exists(path):
            self.skipTest("shadow_run.yml not deployed at repo root in this env")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        forbidden_ops = (
            "> notify_arashi.yml", ">> notify_arashi.yml",
            "cp notify_arashi", "mv notify_arashi", "sed", "rm notify_arashi",
        )
        for op in forbidden_ops:
            self.assertNotIn(op, content)

    def test_existing_notify_arashi_yml_unchanged_markers(self) -> None:
        """既存notify_arashi.ymlのcronスケジュールが維持されていること
        （本Step6-1で当該ファイルを一切編集していないことの傍証）。
        """
        path = os.path.join(self._repo_root(), "notify_arashi.yml")
        if not os.path.exists(path):
            self.skipTest("notify_arashi.yml not present in this environment")
        with open(path, encoding="utf-8") as f:
            content = f.read()
        self.assertIn('cron: "0,15,30,45 23 * * *"', content)
