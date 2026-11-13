"""正式G3（B案）：写し取り（04 §4 の T1〜T11、T13 は組み立てのテストで確認）。"""

from __future__ import annotations

import sys
import unittest

from shadow.g3_capture import (
    STATUS_CAPTURE_FAILED, STATUS_CAPTURED, STATUS_LEGACY_EXCEPTION, STATUS_NOT_CALLED,
    ApplyBuyscoreCapture,
)
from tests.unit import g3_fake_legacy as F
from tests.unit import g3_support as S


def _patterns():
    combos = ["1-2-3", "1-3-2", "2-1-3"]
    return {"combos": combos, "probs": {c: 0.08 for c in combos}, "unc": {c: 0.3 for c in combos}}


def _odds():
    return {"1-2-3": 30.0, "1-3-2": 25.0, "2-1-3": 40.0}


class CaptureTests(unittest.TestCase):
    def setUp(self):
        F.FAKE_CFG.clear()
        F.FAKE_CFG.update(S.load_freeze_config())
        for k in F.MODE:
            F.MODE[k] = False

    def _run(self, patterns=None, odds=None):
        cap = ApplyBuyscoreCapture(target=F.apply_buyscore)
        with cap:
            res = F.evaluate_bets(patterns or _patterns(), {1: 0.3}, odds if odds is not None else _odds(),
                                  upset_score=5.0, has_exhibition=False)
        return cap, res

    def test_t1_t2_captured_and_no_legacy_scores_at_call(self):
        cap, _ = self._run()
        self.assertEqual(cap.status, STATUS_CAPTURED)
        self.assertEqual(sorted(cap.call_context), sorted(
            ["match_index", "match_index_approx", "race_type", "has_exhibition", "market_gap",
             "upset_score", "ex_rank_1st"]))
        self.assertTrue(all("buyscore" not in c and "kelly" not in c for c in cap.call_candidates))
        self.assertTrue(all("buyscore" in c for c in cap.return_candidates))
        self.assertEqual(cap.return_cfg, F.FAKE_CFG)

    def test_t3_copy_is_not_affected_by_mutation(self):
        cap, _ = self._run()
        self.assertEqual([c["combo"] for c in cap.call_candidates], ["1-2-3", "1-3-2", "2-1-3"])

    def test_t4_result_unchanged_by_capture(self):
        _, with_cap = self._run()
        without = F.evaluate_bets(_patterns(), {1: 0.3}, _odds(), upset_score=5.0, has_exhibition=False)
        self.assertEqual(with_cap, without)

    def test_t5_not_called(self):
        cap, res = self._run(odds={})
        self.assertEqual(res, [])
        self.assertEqual(cap.status, STATUS_NOT_CALLED)

    def test_t6_legacy_exception(self):
        F.MODE["raise"] = True
        cap, res = self._run()
        self.assertEqual(cap.status, STATUS_LEGACY_EXCEPTION)
        self.assertEqual([c["combo"] for c in res], ["1-2-3", "1-3-2", "2-1-3"])

    def test_t7_error_inside_observer_does_not_stop_legacy(self):
        class Boom:
            def __deepcopy__(self, memo):
                raise RuntimeError("boom")

        cap = ApplyBuyscoreCapture(target=F.apply_buyscore)
        with cap:
            res = F.apply_buyscore([{"combo": "1-2-3", "prob": 0.1, "ev": 2.0, "odds": 20.0}],
                                   {"match_index": 50, "bad": Boom()})
        self.assertEqual(cap.status, STATUS_CAPTURE_FAILED)
        self.assertIn("passthrough", res)  # Legacy 側の処理は止まらず、戻り値を返す

    def test_t8_called_twice(self):
        F.MODE["call_twice"] = True
        cap, _ = self._run()
        self.assertEqual(cap.status, STATUS_CAPTURE_FAILED)

    def test_t9_same_name_other_file_is_ignored(self):
        ns: dict = {}
        code = compile("def apply_buyscore(candidates, context):\n    return {'passthrough': True}\n",
                       "/tmp/other_module.py", "exec")
        exec(code, ns)
        cap = ApplyBuyscoreCapture(target=F.apply_buyscore)
        with cap:
            ns["apply_buyscore"]([], {})
        self.assertEqual(cap.status, STATUS_NOT_CALLED)

    def test_t10_existing_profiler(self):
        def other(frame, event, arg):
            return None
        sys.setprofile(other)
        try:
            cap = ApplyBuyscoreCapture(target=F.apply_buyscore)
            with cap:
                F.evaluate_bets(_patterns(), {1: 0.3}, _odds(), upset_score=5.0)
            self.assertEqual(cap.status, STATUS_CAPTURE_FAILED)
            self.assertIs(sys.getprofile(), other)
        finally:
            sys.setprofile(None)

    def test_profiler_removed_after_exit(self):
        self._run()
        self.assertIsNone(sys.getprofile())

    def test_default_target_is_real_x_buyscore(self):
        import x_buyscore
        cap = ApplyBuyscoreCapture()
        self.assertEqual(cap._key[1], "apply_buyscore")
        self.assertEqual(cap._key[2], x_buyscore.apply_buyscore.__code__.co_firstlineno)


if __name__ == "__main__":
    unittest.main()
