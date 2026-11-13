"""正式G3（B案）：入力の追加なしで、assess_equivalent と既存 DefaultBuyEngine.assess が同値であること（07 §4・§6）。"""

from __future__ import annotations

import random
import unittest
from dataclasses import astuple

from core.buyscore import DefaultBuyEngine
from core.exceptions import ValidationError
from core.g3_assess import assess_equivalent
from tests.unit import g3_support as S


class _FixedKelly:
    def __call__(self, prob, odds, buyscore, cfg):
        return 0.123


def _run_both(ev, pr, cfg, kelly=None):
    """両者を実行し、(結果 or 例外の型名) を返す。"""
    out = []
    for fn in (lambda: DefaultBuyEngine(kelly_strategy=kelly).assess(ev, pr, cfg),
               lambda: assess_equivalent(ev, pr, cfg, kelly_strategy=kelly)):
        try:
            out.append(("ok", astuple(fn())))
        except Exception as exc:  # noqa: BLE001
            out.append(("exc", type(exc).__name__))
    return out


class GoldenManifestTests(unittest.TestCase):
    def test_manifest_hashes_match_fixed_files(self):
        pairs = S.verify_golden_manifest()
        self.assertEqual(len(pairs), 70)


class EquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = S.load_freeze_config()

    def _assert_same(self, ev, pr, cfg, kelly=None, label=""):
        a, b = _run_both(ev, pr, cfg, kelly)
        self.assertEqual(a, b, label)

    def test_e1_golden_70(self):
        for inp, _exp in S.verify_golden_manifest():
            c, x = inp["candidate"], inp["context"]
            ev = S.evaluation(match_index=x["match_index"], race_type=x["race_type"])
            pr = S.prediction(pred_combo=c["combo"], pred_prob=c["prob"],
                              pred_ev=c["ev"], pred_odds=c["odds"])
            self._assert_same(ev, pr, self.cfg, label=c["combo"])

    def test_e2_random_10000_deterministic(self):
        cases1 = [S.random_equiv_case(r) for r in [random.Random(S.SEED_EQUIV)] for _ in range(S.N_RANDOM)]
        rng2 = random.Random(S.SEED_EQUIV)
        cases2 = [S.random_equiv_case(rng2) for _ in range(S.N_RANDOM)]
        self.assertEqual(S.cases_sha256(cases1), S.cases_sha256(cases2))
        failures = []
        for i, k in enumerate(cases1):
            ev = S.evaluation(match_index=k["match_index"], upset_score=k["upset_score"],
                              race_type=k["race_type"])
            pr = S.prediction(pred_prob=k["pred_prob"], pred_ev=k["pred_ev"], pred_odds=k["pred_odds"])
            a, b = _run_both(ev, pr, self.cfg)
            if a != b:
                failures.append({"i": i, "input": k, "existing": a, "new": b})
        if failures:
            path = S.save_failures("e2_equivalence", failures)
            self.fail(f"{len(failures)} mismatches (seed={S.SEED_EQUIV}); saved {path}")

    def test_e3_boundaries(self):
        for bs_ev in (1.27, 1.28, 1.29):
            for mi in (39.9, 40.0, 40.1):
                for odds in (1.0, 29.9, 30.0, 30.1, 999.0):
                    for prob in (0.0, 0.01, 0.06, 0.5, 1.0):
                        ev = S.evaluation(match_index=mi)
                        pr = S.prediction(pred_prob=prob, pred_ev=bs_ev, pred_odds=odds)
                        self._assert_same(ev, pr, self.cfg, label=f"{bs_ev},{mi},{odds},{prob}")

    def test_exceptions(self):
        ev, pr = S.evaluation(), S.prediction()
        for e_, p_ in ((None, pr), (ev, None), (ev, S.prediction(eval_id="x")),
                       (ev, S.prediction(pred_prob=None)), (ev, S.prediction(pred_ev=None)),
                       (ev, S.prediction(pred_odds=None))):
            a, b = _run_both(e_, p_, self.cfg)
            self.assertEqual(a, b)
            self.assertEqual(a, ("exc", ValidationError.__name__))

    def test_match_index_none(self):
        self._assert_same(S.evaluation(match_index=None), S.prediction(), self.cfg)

    def test_config_variants(self):
        no_version = {k: v for k, v in self.cfg.items() if k != "_version"}
        missing_kelly = {k: v for k, v in self.cfg.items() if k != "kelly"}
        for cfg in (self.cfg, no_version, missing_kelly):
            self._assert_same(S.evaluation(), S.prediction(), cfg)

    def test_kelly_injection(self):
        self._assert_same(S.evaluation(), S.prediction(), self.cfg, kelly=_FixedKelly())


if __name__ == "__main__":
    unittest.main()
