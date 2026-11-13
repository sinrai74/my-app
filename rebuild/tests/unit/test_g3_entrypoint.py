"""正式G3（B案）：1レースの組み立て（07 §8・§9）。偽の Legacy で区分と同一性の条件を確かめる。"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from actions import g3_entrypoint as G
from core.upset import UpsetResult
from shadow.g3_capture import ApplyBuyscoreCapture
from tests.unit import g3_fake_legacy as F
from tests.unit import g3_support as S


def _upset(score=5.0, target=(2, 3)):
    return UpsetResult(upset_score=score, detail={}, target_lanes=tuple(target), lane_probs={},
                       boat1_prob=0.5, boat1_rank=1, best_other_lane=2, best_other_prob=0.2,
                       danger_score=10.0, danger_breakdown={}, venue_name=None)


def _boats():
    return [SimpleNamespace(lane=l) for l in (1, 2, 3, 4, 5, 6)]


def _gen_patterns(target, score):
    combos = ["1-2-3", "1-3-2", "2-1-3"]
    return {"combos": combos, "probs": {"1-2-3": 0.09, "1-3-2": 0.07, "2-1-3": 0.05},
            "unc": {c: 0.3 for c in combos}, "race_type": "イン逃げ型", "_target": list(target)}


ODDS = {"1-2-3": 30.0, "1-3-2": 25.0, "2-1-3": 40.0}


class EntrypointTests(unittest.TestCase):
    def setUp(self):
        F.FAKE_CFG.clear()
        F.FAKE_CFG.update(S.load_freeze_config())
        for k in F.MODE:
            F.MODE[k] = False

    def _run(self, evaluation=None, upset=None, odds=ODDS, **kw):
        ev = evaluation or S.evaluation(upset_score=5.0, win_probs={1: 0.2, 2: 0.1, 3: 0.3, 4: 0.25})
        return G.evaluate_race_g3(
            evaluation=ev, upset=upset or _upset(), legacy_boats=_boats(), odds_map=odds,
            weather=object(), buy_config=F.FAKE_CFG, evaluate_bets=F.evaluate_bets,
            generate_patterns=_gen_patterns,
            capture_factory=lambda: ApplyBuyscoreCapture(target=F.apply_buyscore), **kw)

    def test_compared_and_all_conditions_ok(self):
        rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_COMPARED, rec["conditions"])
        self.assertTrue(all(v["ok"] for v in rec["conditions"].values()))
        self.assertTrue(rec["match"]["all"], rec)
        self.assertTrue(rec["ml_bonus"]["applied"])
        self.assertEqual(rec["target"]["rebuild_target_post"], [3, 4])
        self.assertEqual(rec["capture"]["call_context"]["has_exhibition"], False)

    def test_no_prediction(self):
        rec = self._run(evaluation=S.evaluation(upset_score=5.0, win_probs={}))
        self.assertEqual(rec["outcome"], G.OUTCOME_NO_PREDICTION)
        rec2 = self._run(evaluation=S.evaluation(upset_score=5.0, win_probs=None))
        self.assertEqual(rec2["outcome"], G.OUTCOME_NO_PREDICTION)

    def test_no_odds(self):
        self.assertEqual(self._run(odds={})["outcome"], G.OUTCOME_NO_ODDS)

    def test_condition1_defect(self):
        rec = self._run(upset=_upset(score=4.0))
        self.assertEqual(rec["outcome"], G.OUTCOME_INPUT_DEFECT)
        self.assertFalse(rec["conditions"]["c1_upset_pre"]["ok"])

    def test_legacy_exception(self):
        F.MODE["raise"] = True
        self.assertEqual(self._run()["outcome"], G.OUTCOME_LEGACY_EXCEPTION)

    def test_capture_failed(self):
        F.MODE["call_twice"] = True
        self.assertEqual(self._run()["outcome"], G.OUTCOME_CAPTURE_FAILED)

    def test_prediction_drift(self):
        F.MODE["zero_amount_top"] = True
        rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_PREDICTION_DRIFT, rec.get("legacy"))

    def test_target_is_recorded_not_judged(self):
        rec = self._run(legacy_target=[5, 6])
        self.assertEqual(rec["outcome"], G.OUTCOME_COMPARED)
        self.assertEqual(rec["target"]["legacy_target"], [5, 6])
        self.assertNotEqual(rec["target"]["legacy_target"], rec["target"]["rebuild_target_post"])

    # ---- 監査 F-06：条件ごとに「1つだけずらした場合」 ----
    def test_condition6_upset_mismatch(self):
        F.MODE["ctx_upset_offset"] = 0.5
        rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_INPUT_DEFECT)
        self.assertFalse(rec["conditions"]["c6_context"]["ok"])

    def test_condition6_flags_exhibition_leak(self):
        F.MODE["force_exhibition_true"] = True
        rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_INPUT_DEFECT)
        self.assertFalse(rec["conditions"]["c6_flags"]["ok"])

    def test_condition7_duplicate_combo(self):
        F.MODE["dup_combo"] = True
        rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_INPUT_DEFECT)
        self.assertFalse(rec["conditions"]["c7_pred_combo_once"]["ok"])

    def test_t13_capture_vs_return_mismatch(self):
        F.MODE["alter_return_unc"] = True
        rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_INPUT_DEFECT)
        self.assertFalse(rec["conditions"]["t13_capture_vs_return"]["ok"])

    def test_no_candidate(self):
        rec = self._run(odds={"9-9-9": 10.0})
        self.assertEqual(rec["outcome"], G.OUTCOME_NO_CANDIDATE)

    def test_evaluate_bets_exception_is_classified(self):
        F.MODE["raise_outside"] = True
        rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_LEGACY_EVALUATE_BETS_EXCEPTION)
        self.assertEqual(rec["exception"], "KeyError")

    def test_cfg_not_captured_is_capture_failed(self):
        F.MODE["no_cfg"] = True
        ev = S.evaluation(upset_score=5.0, win_probs={1: 0.2, 2: 0.1, 3: 0.3, 4: 0.25})
        rec = G.evaluate_race_g3(
            evaluation=ev, upset=_upset(), legacy_boats=_boats(), odds_map=ODDS, weather=object(),
            buy_config=F.FAKE_CFG, evaluate_bets=F.evaluate_bets, generate_patterns=_gen_patterns,
            capture_factory=lambda: ApplyBuyscoreCapture(target=F.apply_buyscore_nocfg))
        self.assertEqual(rec["outcome"], G.OUTCOME_CAPTURE_FAILED)
        self.assertIn("cfg_not_captured", rec["capture"]["errors"])

    def test_config_is_recorded(self):
        rec = self._run(run_id="r1", executed_at="2026-11-13T00:00:00Z")
        self.assertEqual(rec["outcome"], G.OUTCOME_COMPARED)
        self.assertEqual(rec["config"]["source"], "captured_legacy_cfg")
        self.assertTrue(rec["config"]["matches_freeze"])  # FAKE_CFG は freeze 版
        self.assertTrue(rec["config"]["buy_config_equal"])
        self.assertEqual((rec["run_id"], rec["executed_at"]), ("r1", "2026-11-13T00:00:00Z"))
        self.assertEqual(rec["evaluate_bets_call_count"], 1)
        self.assertIsNotNone(rec["capture"]["return_candidates_sha256"])

    def test_config_difference_is_visible(self):
        other = dict(F.FAKE_CFG)
        other["_version"] = "changed"
        ev = S.evaluation(upset_score=5.0, win_probs={1: 0.2, 2: 0.1, 3: 0.3, 4: 0.25})
        rec = G.evaluate_race_g3(
            evaluation=ev, upset=_upset(), legacy_boats=_boats(), odds_map=ODDS, weather=object(),
            buy_config=other, evaluate_bets=F.evaluate_bets, generate_patterns=_gen_patterns,
            capture_factory=lambda: ApplyBuyscoreCapture(target=F.apply_buyscore))
        self.assertFalse(rec["config"]["buy_config_equal"])

    def test_condition1_has_priority_over_no_prediction(self):
        rec = self._run(evaluation=S.evaluation(upset_score=5.0, win_probs={}), upset=_upset(score=4.0))
        self.assertEqual(rec["outcome"], G.OUTCOME_INPUT_DEFECT)


    # ---- 再監査 N-02：残りの同一性の条件の否定テスト ----
    def test_condition6_match_index_only(self):
        F.MODE["ctx_match_offset"] = 1.0
        rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_INPUT_DEFECT)
        self.assertFalse(rec["conditions"]["c6_context"]["ok"])
        self.assertIn("match_index", rec["conditions"]["c6_context"]["detail"])

    def test_condition6_flags_approx_not_true(self):
        F.MODE["approx_false"] = True
        rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_INPUT_DEFECT)
        self.assertFalse(rec["conditions"]["c6_flags"]["ok"])

    def test_condition7b_prediction_values(self):
        F.MODE["alter_return_prob"] = True
        rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_INPUT_DEFECT)
        self.assertFalse(rec["conditions"]["c7b_prediction_values"]["ok"])

    def test_context_keys(self):
        F.MODE["extra_ctx_key"] = True
        rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_INPUT_DEFECT)
        self.assertFalse(rec["conditions"]["context_keys"]["ok"])

    # ---- 再監査 N-03：比較結果そのものを守るテスト ----
    def test_match_false_when_legacy_value_differs(self):
        F.MODE["legacy_kelly_offset"] = 0.5
        rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_COMPARED)
        self.assertFalse(rec["match"]["kelly_fraction"])
        self.assertTrue(rec["match"]["buyscore"])
        self.assertFalse(rec["match"]["all"])

    def test_skip_case_matches(self):
        ev = S.evaluation(upset_score=5.0, win_probs={1: 0.2, 2: 0.1, 3: 0.3, 4: 0.25})

        def low_patterns(target, score):
            p = _gen_patterns(target, score)
            p["probs"] = {c: 0.005 for c in p["combos"]}
            return p
        rec = G.evaluate_race_g3(
            evaluation=ev, upset=_upset(), legacy_boats=_boats(), odds_map=ODDS, weather=object(),
            buy_config=F.FAKE_CFG, evaluate_bets=F.evaluate_bets, generate_patterns=low_patterns,
            capture_factory=lambda: ApplyBuyscoreCapture(target=F.apply_buyscore))
        self.assertEqual(rec["outcome"], G.OUTCOME_COMPARED, rec["conditions"])
        self.assertIs(rec["legacy"]["passthrough"], True)
        self.assertIsNotNone(rec["rebuild"]["skip_reason"])
        self.assertEqual(rec["rebuild"]["skip_reason"], rec["legacy"]["passthrough_reason"])
        self.assertTrue(rec["match"]["all"])

    def test_matches_freeze_false_and_none(self):
        changed = dict(F.FAKE_CFG)
        changed["_version"] = "changed"
        F.FAKE_CFG.clear()
        F.FAKE_CFG.update(changed)
        rec = self._run()
        self.assertIs(rec["config"]["matches_freeze"], False)
        self.assertEqual(rec["config"]["version"], "changed")
        original = G._freeze_config
        G._freeze_config = lambda: None
        try:
            rec2 = self._run()
        finally:
            G._freeze_config = original
        self.assertIsNone(rec2["config"]["matches_freeze"])

    # ---- 再監査 N-04：加点後の target・score から patterns を作る配線 ----
    def test_patterns_built_from_post_bonus_values(self):
        calls = []

        def recording(target, score):
            calls.append((list(target), score))
            return _gen_patterns(target, score)
        ev = S.evaluation(upset_score=5.0, win_probs={1: 0.2, 2: 0.1, 3: 0.3, 4: 0.25})
        rec = G.evaluate_race_g3(
            evaluation=ev, upset=_upset(), legacy_boats=_boats(), odds_map=ODDS, weather=object(),
            buy_config=F.FAKE_CFG, evaluate_bets=F.evaluate_bets, generate_patterns=recording,
            capture_factory=lambda: ApplyBuyscoreCapture(target=F.apply_buyscore))
        self.assertTrue(rec["ml_bonus"]["applied"])
        self.assertEqual(calls, [([3, 4], rec["ml_bonus"]["score_post"])])
        self.assertNotEqual(rec["ml_bonus"]["score_post"], 5.0)


if __name__ == "__main__":
    unittest.main()
