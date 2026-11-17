"""正式G3（B案）：1レースの組み立て（07 §8・§9）。偽の Legacy で区分と同一性の条件を確かめる。"""

from __future__ import annotations

import json
import os
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
        kw.setdefault("isolation_factory", S.iso_factory())
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
        self.assertEqual(rec["exception"], "unexpected_error")  # G-22：区分は outcome が保持

    def test_cfg_not_captured_is_capture_failed(self):
        F.MODE["no_cfg"] = True
        ev = S.evaluation(upset_score=5.0, win_probs={1: 0.2, 2: 0.1, 3: 0.3, 4: 0.25})
        rec = G.evaluate_race_g3(
            evaluation=ev, upset=_upset(), legacy_boats=_boats(), odds_map=ODDS, weather=object(),
            buy_config=F.FAKE_CFG, evaluate_bets=F.evaluate_bets, generate_patterns=_gen_patterns,
            capture_factory=lambda: ApplyBuyscoreCapture(target=F.apply_buyscore_nocfg),
            isolation_factory=S.iso_factory())
        self.assertEqual(rec["outcome"], G.OUTCOME_CAPTURE_FAILED)
        self.assertIn("cfg_not_captured", rec["capture"]["errors"])

    def test_config_is_recorded(self):
        rec = self._run(run_id="r1", executed_at="2026-11-13T00:00:00Z")
        self.assertEqual(rec["outcome"], G.OUTCOME_COMPARED)
        self.assertEqual(rec["config"]["source"], "workdir_file")
        self.assertEqual(rec["config"]["source_detail"]["source_file_sha256"], S.CONFIG_SHA256)
        self.assertTrue(rec["config"]["source_detail"]["reload_equal"])
        self.assertEqual(rec["config"]["body"], S.load_freeze_config())
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
            capture_factory=lambda: ApplyBuyscoreCapture(target=F.apply_buyscore),
            isolation_factory=S.iso_factory())
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
        self.assertIn("match_index", rec["conditions"]["c6_context"]["check_detail"]["diff_keys"])  # G-27 追従

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
            capture_factory=lambda: ApplyBuyscoreCapture(target=F.apply_buyscore),
            isolation_factory=S.iso_factory())
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
        seeds = {"buyscore_config.json": S.config_bytes(changed)}
        rec = self._run(isolation_factory=S.iso_factory(seeds))
        self.assertEqual(rec["outcome"], G.OUTCOME_COMPARED, rec["config"])
        self.assertIs(rec["config"]["matches_freeze"], False)
        self.assertEqual(rec["config"]["version"], "changed")
        original = G._freeze_config
        G._freeze_config = lambda: None
        try:
            rec2 = self._run(isolation_factory=S.iso_factory(seeds))
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
            capture_factory=lambda: ApplyBuyscoreCapture(target=F.apply_buyscore),
            isolation_factory=S.iso_factory())
        self.assertTrue(rec["ml_bonus"]["applied"])
        self.assertEqual(calls, [([3, 4], rec["ml_bonus"]["score_post"])])
        self.assertNotEqual(rec["ml_bonus"]["score_post"], 5.0)


class _IsoHelpers:
    """偽の Legacy で evaluate_race_g3 を通す共通の手順。"""

    def setUp(self):
        F.FAKE_CFG.clear()
        F.FAKE_CFG.update(S.load_freeze_config())
        for k in F.MODE:
            F.MODE[k] = False
        self.calls = []

    def _run(self, **kw):
        kw.setdefault("isolation_factory", S.iso_factory())
        ev = S.evaluation(upset_score=5.0, win_probs={1: 0.2, 2: 0.1, 3: 0.3, 4: 0.25})

        def counting_evaluate_bets(**k):
            self.calls.append(1)
            return F.evaluate_bets(**k)
        return G.evaluate_race_g3(
            evaluation=ev, upset=_upset(), legacy_boats=_boats(), odds_map=ODDS,
            weather=object(), buy_config=F.FAKE_CFG, evaluate_bets=counting_evaluate_bets,
            generate_patterns=_gen_patterns,
            capture_factory=lambda: ApplyBuyscoreCapture(target=F.apply_buyscore), **kw)

    def _defaults_in_isolation(self, seeds):
        import x_buyscore
        from shadow.g3_isolation import G3Isolation
        with G3Isolation(seeds) as iso:
            self.assertTrue(iso.established, iso.errors)
            cfg = x_buyscore.load_config()
        return cfg



class IsolationAndEvidenceTests(_IsoHelpers, unittest.TestCase):
    """N-01・N-05・N-06：境界と、記録の欄（偽の Legacy）。"""

    def test_isolation_not_established_does_not_call_legacy(self):
        rec = self._run(isolation_factory=S.iso_factory({"../outside.json": b"{}"}))
        self.assertEqual(rec["outcome"], G.OUTCOME_ISOLATION_NOT_ESTABLISHED)
        self.assertFalse(rec["isolation"]["established"])
        self.assertEqual(self.calls, [])
        self.assertEqual(rec["evaluate_bets_call_count"], 0)
        self.assertIsNone(rec["match"])

    def _assert_violation(self, kind):
        rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_ISOLATION_VIOLATION)
        self.assertFalse(rec["isolation"]["verified"])
        self.assertTrue(rec["isolation"]["established"])
        self.assertIn(kind, [e["kind"] for e in rec["isolation"]["blocked_events"]])
        self.assertIsNone(rec["match"])
        self.assertIsNone(rec["comparison"])
        return rec

    def test_network_attempt_is_violation(self):
        F.MODE["net_attempt"] = True
        rec = self._assert_violation("network")
        self.assertGreaterEqual(rec["isolation"]["network_attempts_observed"], 1)

    def test_os_system_is_violation(self):
        F.MODE["os_system"] = True
        self._assert_violation("process")

    def test_write_outside_is_violation(self):
        F.MODE["write_outside"] = True
        self._assert_violation("write_outside")

    def test_config_source_unknown_is_not_compared(self):
        changed = dict(F.FAKE_CFG)
        changed["_version"] = "not_the_seeded_file"
        F.FAKE_CFG.clear()
        F.FAKE_CFG.update(changed)
        rec = self._run()  # 作業フォルダには freeze 版を置く：Legacy が使った設定と読み直しが一致しない
        self.assertEqual(rec["outcome"], G.OUTCOME_CONFIG_SOURCE_UNKNOWN)
        self.assertEqual(rec["config"]["source"], "unknown")
        self.assertFalse(rec["config"]["source_detail"]["reload_equal"])
        self.assertIsNone(rec["match"])

    def test_config_source_builtin_defaults(self):
        cfg = self._defaults_in_isolation({})
        F.FAKE_CFG.clear()
        F.FAKE_CFG.update(cfg)
        rec = self._run(isolation_factory=S.iso_factory({}))
        self.assertEqual(rec["config"]["source"], "builtin_defaults", rec["config"]["source_detail"])
        self.assertIsNone(rec["config"]["source_detail"]["source_file_sha256"])

    def test_config_source_default_template(self):
        seeds = {"buyscore_config.default.json": S.REPO_DIR / "buyscore_config.default.json"}
        cfg = self._defaults_in_isolation(seeds)
        F.FAKE_CFG.clear()
        F.FAKE_CFG.update(cfg)
        rec = self._run(isolation_factory=S.iso_factory(seeds))
        self.assertEqual(rec["config"]["source"], "default_template_copy", rec["config"]["source_detail"])
        self.assertEqual(rec["config"]["source_detail"]["source_file_sha256"],
                         rec["isolation"]["seeded"]["buyscore_config.default.json"])

    def test_evidence_fields_with_token_env(self):
        with S.fake_token_env():
            rec = self._run(run_id="r-ev", executed_at="2026-11-13T00:00:00+09:00")
            self.assertEqual(os.environ["GITHUB_TOKEN"], S.FAKE_TOKEN_ENV["GITHUB_TOKEN"])  # 終了後は戻る
        self.assertEqual(rec["outcome"], G.OUTCOME_COMPARED, rec["conditions"])
        iso = rec["isolation"]
        self.assertTrue(iso["verified"])
        self.assertIn("GITHUB_TOKEN", iso["env_removed_names"])
        self.assertIn("GH_TOKEN", iso["env_removed_names"])
        self.assertEqual(iso["network_attempts_observed"], 0)
        # 値（トークン）は記録に入らない
        self.assertNotIn(S.FAKE_TOKEN_ENV["GITHUB_TOKEN"], json.dumps(rec, ensure_ascii=False, default=str))
        self.assertEqual(rec["sequence"], ["ml_bonus", "generate_patterns", "evaluate_bets"])
        self.assertEqual(rec["ml_bonus"]["score_pre"], 5.0)
        self.assertEqual(rec["ml_bonus"]["target_pre"], [2, 3])
        self.assertEqual(rec["patterns"]["input_target"], list(rec["ml_bonus"]["target_post"]))
        self.assertEqual(rec["patterns"]["input_score"], rec["ml_bonus"]["score_post"])
        self.assertEqual(rec["win_probs"], {"1": 0.2, "2": 0.1, "3": 0.3, "4": 0.25})
        self.assertIsNotNone(rec["prediction"]["pred_combo"])
        self.assertEqual(rec["race_type"]["legacy_context"], "イン逃げ型")
        self.assertIn("rebuild_input_context", rec["race_type"])
        call_ids = rec["candidates"]["call"]
        self.assertEqual([c["combo"] for c in call_ids],
                         [c["combo"] for c in rec["capture"]["call_candidates"]])
        self.assertTrue(all(len(c["sha256"]) == 64 for c in call_ids))
        self.assertEqual([c["item"] for c in rec["comparison"]],
                         ["buyscore", "kelly_fraction", "skip_reason", "investment_type"])
        for c in rec["comparison"]:
            self.assertIs(rec["match"][c["item"]], c["rebuild"] == c["legacy"])
        self.assertEqual((rec["run_id"], rec["executed_at"]), ("r-ev", "2026-11-13T00:00:00+09:00"))


class ConfigSourceAndHardeningTests(_IsoHelpers, unittest.TestCase):
    """A-02・A-03・A-07：設定の出どころの完全一致、自己試験の失敗、記録の欄。"""

    def _cfg_from(self, seeds):
        cfg = self._defaults_in_isolation(seeds)
        F.FAKE_CFG.clear()
        F.FAKE_CFG.update(cfg)
        return cfg

    def test_broken_config_file_is_unknown(self):
        seeds = {"buyscore_config.json": b"{broken"}
        self._cfg_from(seeds)  # Legacy は壊れたファイルを読めず、組み込みの既定値を使う
        rec = self._run(isolation_factory=S.iso_factory(seeds))
        self.assertEqual(rec["outcome"], G.OUTCOME_CONFIG_SOURCE_UNKNOWN)
        self.assertEqual(rec["config"]["source_detail"]["reason"], "config_file_invalid")
        self.assertIsNone(rec["config"]["source_detail"]["source_file_sha256"])

    def test_partial_config_file_is_unknown(self):
        seeds = {"buyscore_config.json": b'{"_version": "partial-only"}'}
        cfg = self._cfg_from(seeds)
        self.assertEqual(cfg.get("_version"), "partial-only")  # 既定値で補われた設定
        rec = self._run(isolation_factory=S.iso_factory(seeds))
        self.assertEqual(rec["outcome"], G.OUTCOME_CONFIG_SOURCE_UNKNOWN)
        self.assertEqual(rec["config"]["source_detail"]["reason"], "config_file_not_equal_captured")
        self.assertIs(rec["config"]["source_detail"]["file_equals_captured"], False)

    def test_config_file_changed_during_run_is_unknown(self):
        F.MODE["modify_config"] = True
        rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_CONFIG_SOURCE_UNKNOWN)
        self.assertEqual(rec["config"]["source_detail"]["reason"], "config_file_changed")

    def test_exact_config_file_is_workdir_file(self):
        rec = self._run()
        self.assertEqual(rec["config"]["source"], "workdir_file")
        self.assertIs(rec["config"]["source_detail"]["file_equals_captured"], True)
        self.assertIsNone(rec["config"]["source_detail"]["reason"])

    def test_reload_failure_is_unknown(self):                      # M20
        """ファイルと写し取った設定が一致しても、境界の中での読み直しが失敗すれば unknown。"""
        from unittest import mock
        import x_buyscore
        with mock.patch.object(x_buyscore, "load_config", side_effect=RuntimeError("reload")):
            rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_CONFIG_SOURCE_UNKNOWN)
        self.assertEqual(rec["config"]["source_detail"]["reason"], "reload_mismatch")
        self.assertEqual(rec["config"]["source_detail"]["reload_error"], "unexpected_error")  # G-25 追従

    def test_self_test_failure_does_not_call_legacy(self):
        from unittest import mock
        from shadow import g3_isolation as GI
        with mock.patch.object(GI, "_NETWORK_EVENTS", frozenset()):  # 層2 のネットワークを効かなくする
            rec = self._run()
        self.assertEqual(rec["outcome"], G.OUTCOME_ISOLATION_NOT_ESTABLISHED)
        self.assertFalse(rec["isolation"]["self_test"]["socket_connect"])
        self.assertEqual(self.calls, [])
        self.assertIsNone(rec["match"])

    def test_inputs_and_flags_recorded(self):
        rec = self._run()
        inp = rec["inputs"]
        self.assertEqual(inp["lane_order"], [b.lane for b in _boats()])
        self.assertEqual(inp["odds_map_count"], len(ODDS))
        self.assertEqual(inp["odds_map"], dict(ODDS))
        self.assertEqual(len(inp["odds_map_sha256"]), 64)
        self.assertEqual(len(inp["legacy_boats"]), len(_boats()))
        self.assertEqual(rec["patterns"]["body"], json.loads(json.dumps(
            _gen_patterns(rec["patterns"]["input_target"], rec["patterns"]["input_score"]),
            ensure_ascii=False, default=str)))
        self.assertEqual(rec["target"]["status"], "legacy_target_not_provided")
        self.assertIs(rec["race_type"]["equal"],
                      rec["race_type"]["legacy_context"] == rec["race_type"]["rebuild_input_context"])


if __name__ == "__main__":
    unittest.main()
