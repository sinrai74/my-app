"""正式G3（B案）：入力の追加ありで、assess_with_input の4項目が Legacy x_buyscore の純関数と一致すること（07 §5・§6、08 §3）。

期待値は Legacy の純関数をテストの中で直接呼んで作る（Rebuild の側から作らない）。
apply_buyscore は呼ばない。設定は golden の freeze 版だけを使う。
"""

from __future__ import annotations

import copy
import random
import unittest

import x_buyscore as L  # Legacy（読み込むだけ）

from core.exceptions import ValidationError
from core.g3_assess import assess_with_input, build_input_candidate, build_input_context
from models.g3_input import BuyScoreInput, G3InputValidationError, MlBonusResult
from tests.unit import g3_support as S


def _legacy_four(candidate: dict, context: dict, cfg: dict) -> tuple:
    cand = copy.deepcopy(candidate)
    ctx = copy.deepcopy(context)
    score = L.calc_buyscore(cand, ctx, cfg)
    kelly = L.kelly_fraction(cand.get("prob", 0), cand.get("odds", 0), score, cfg)
    scored = dict(cand)
    scored["buyscore"] = score
    skip = L.check_passthrough([scored], ctx, cfg)
    inv = L.investment_type(score, cand.get("ev", 0.0), cand.get("odds", 0.0), ctx.get("race_type", "混戦"))
    return (score, inv, kelly, skip)


def _case_objects(k: dict):
    ev = S.evaluation(race_type=k["race_type"])
    pr = S.prediction(pred_prob=k["pred_prob"], pred_ev=k["pred_ev"], pred_odds=k["pred_odds"])
    bonus = MlBonusResult(score_post=k["score_post"], target_post=(2, 3), applied=False, bonus=0.0)
    bi = BuyScoreInput(
        eval_id=ev.eval_id, pred_combo=pr.pred_combo, uncertainty=k["uncertainty"],
        disagreement=k["disagreement"], composite=k["composite"],
        has_exhibition=k["has_exhibition"], market_gap=k["market_gap"],
        match_index_approx=k["match_index_approx"],
    )
    return ev, pr, bonus, bi


def _rebuild_four(ev, pr, cfg, bonus, bi) -> tuple:
    r = assess_with_input(ev, pr, cfg, bonus, bi)
    return (r.buyscore, r.investment_type, r.kelly_fraction, r.skip_reason)


class LegacyComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = S.load_freeze_config()

    def _compare(self, k: dict) -> tuple:
        ev, pr, bonus, bi = _case_objects(k)
        cand = build_input_candidate(pr, bi)
        ctx = build_input_context(ev, bonus, bi)
        return _rebuild_four(ev, pr, self.cfg, bonus, bi), _legacy_four(cand, ctx, self.cfg)

    def test_random_10000_deterministic(self):
        rng1, rng2 = random.Random(S.SEED_LEGACY), random.Random(S.SEED_LEGACY)
        cases = [S.random_legacy_case(rng1) for _ in range(S.N_RANDOM)]
        again = [S.random_legacy_case(rng2) for _ in range(S.N_RANDOM)]
        self.assertEqual(S.cases_sha256(cases), S.cases_sha256(again))
        failures = []
        for i, k in enumerate(cases):
            a, b = self._compare(k)
            if a != b:
                failures.append({"i": i, "input": k, "rebuild": a, "legacy": b})
        if failures:
            path = S.save_failures("legacy_comparison", failures)
            self.fail(f"{len(failures)} mismatches (seed={S.SEED_LEGACY}); saved {path}")

    def test_boundaries(self):
        base = dict(pred_prob=0.08, pred_ev=2.0, pred_odds=35.0, uncertainty=0.5,
                    disagreement=0.2, composite=0.6, has_exhibition=True, market_gap=0.0,
                    match_index_approx=True, score_post=6.0, race_type="")
        variants = []
        for v in (0.749, 0.75, 0.751):
            variants.append({"uncertainty": v})
        for v in (0.799, 0.80, 0.801):
            variants.append({"disagreement": v})
        for v in (-0.5, 0.0, 0.5):
            variants.append({"market_gap": v})
        for sp in (3.80, 3.81, 3.82):  # match_index 39.9〜40.1 付近
            for approx in (True, False):
                variants.append({"score_post": sp, "match_index_approx": approx})
        for comp in (0.0, 0.39, 0.40, 0.41):
            variants.append({"has_exhibition": False, "composite": comp})
        for ev_ in (1.27, 1.28, 1.29):
            variants.append({"pred_ev": ev_})
        for rt in S.RACE_TYPES:
            variants.append({"race_type": rt})
        for ex in (True, False):
            variants.append({"has_exhibition": ex})
        for p in (0.0, 0.2, 0.5):
            variants.append({"pred_prob": p})
        for v in variants:
            k = dict(base, **v)
            a, b = self._compare(k)
            self.assertEqual(a, b, v)

    def test_buyscore_threshold_neighbourhood(self):
        # buyscore が 60 の前後に来る入力を走査し、両者が一致すること
        found = set()
        for i in range(400):
            k = dict(pred_prob=0.02 + i * 0.0005, pred_ev=1.6, pred_odds=40.0, uncertainty=0.4,
                     disagreement=0.1, composite=0.6, has_exhibition=True, market_gap=0.1,
                     match_index_approx=True, score_post=5.0, race_type="")
            a, b = self._compare(k)
            self.assertEqual(a, b, k)
            found.add(a[0] >= 60)
        self.assertEqual(found, {True, False}, "60 をまたぐ入力が生成されていない")


class InputValidationTests(unittest.TestCase):
    def test_buy_input_rejects_bad_values(self):
        good = dict(eval_id="e", pred_combo="1-2-3", uncertainty=0.5, disagreement=0.5,
                    composite=0.5, has_exhibition=False, market_gap=0.0, match_index_approx=True)
        bad = [("uncertainty", None), ("uncertainty", 1.5), ("disagreement", -0.1),
               ("composite", "x"), ("has_exhibition", 0), ("market_gap", float("nan")),
               ("match_index_approx", None), ("eval_id", ""), ("pred_combo", None)]
        BuyScoreInput(**good)
        for key, val in bad:
            with self.assertRaises(G3InputValidationError, msg=key):
                BuyScoreInput(**dict(good, **{key: val}))

    def test_bonus_rejects_bad_values(self):
        MlBonusResult(score_post=5.0, target_post=(2, 3), applied=True, bonus=1.0)
        for kw in (dict(score_post=-1.0), dict(bonus=4.1), dict(applied=False, bonus=1.0),
                   dict(target_post=[2, 3])):
            base = dict(score_post=5.0, target_post=(2, 3), applied=True, bonus=1.0)
            base.update(kw)
            with self.assertRaises(G3InputValidationError, msg=str(kw)):
                MlBonusResult(**base)

    def test_mismatched_ids_raise(self):
        cfg = S.load_freeze_config()
        ev, pr = S.evaluation(), S.prediction()
        bonus = MlBonusResult(score_post=5.0, target_post=(2, 3), applied=False, bonus=0.0)
        bi = BuyScoreInput(eval_id="other", pred_combo=pr.pred_combo, uncertainty=0.5, disagreement=0.5,
                           composite=0.5, has_exhibition=False, market_gap=0.0, match_index_approx=True)
        with self.assertRaises(ValidationError):
            assess_with_input(ev, pr, cfg, bonus, bi)
        bi2 = BuyScoreInput(eval_id=ev.eval_id, pred_combo="9-9-9", uncertainty=0.5, disagreement=0.5,
                            composite=0.5, has_exhibition=False, market_gap=0.0, match_index_approx=True)
        with self.assertRaises(ValidationError):
            assess_with_input(ev, pr, cfg, bonus, bi2)

    def test_approx_true_suppresses_match_index_skip(self):
        cfg = S.load_freeze_config()
        ev = S.evaluation()
        pr = S.prediction(pred_prob=0.2, pred_ev=3.0, pred_odds=20.0)
        bonus = MlBonusResult(score_post=1.0, target_post=(2, 3), applied=False, bonus=0.0)  # match_index 10.5
        mk = lambda approx: BuyScoreInput(eval_id=ev.eval_id, pred_combo=pr.pred_combo, uncertainty=0.1,
                                          disagreement=0.1, composite=0.9, has_exhibition=True,
                                          market_gap=0.5, match_index_approx=approx)
        r_true = assess_with_input(ev, pr, cfg, bonus, mk(True))
        r_false = assess_with_input(ev, pr, cfg, bonus, mk(False))
        self.assertFalse((r_true.skip_reason or "").startswith("AI一致指数不足"))
        if r_false.buyscore >= cfg["thresholds"]["buyscore_min"] and pr.pred_ev >= cfg["thresholds"]["ev_min"]:
            self.assertTrue((r_false.skip_reason or "").startswith("AI一致指数不足"))


if __name__ == "__main__":
    unittest.main()
