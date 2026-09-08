import unittest

from models.evaluation import LegacyPurchaseResult
from shadow.prediction_provider import _EvaluateBetsCapture


def _bet(combo, purchased=True, amount=500):
    return {"combo": combo, "prob": 0.1, "ev": 1.5, "odds": 15.0,
            "confidence": 0.5, "purchased": purchased, "amount": amount}


def _make_capture(fake_result):
    calls = {"count": 0}

    def _fake_evaluate_bets(**kwargs):
        calls["count"] += 1
        return fake_result

    capture = _EvaluateBetsCapture(evaluate_bets=_fake_evaluate_bets)
    return capture, calls


class TestEvaluateBetsCapture(unittest.TestCase):
    def _call(self, capture, fake_result=None, **extra_kwargs):
        kwargs = dict(race_date="20260101", venue_num=1, race_number=5)
        kwargs.update(extra_kwargs)
        return capture(**kwargs)

    def test_single_call_returns_index0_for_prediction(self):
        capture, _ = _make_capture([_bet("1-2-3"), _bet("1-3-2", amount=300)])
        result = self._call(capture)
        self.assertEqual(result["combo"], "1-2-3")

    def test_full_list_available_via_last_purchase_result(self):
        capture, _ = _make_capture([_bet("1-2-3"), _bet("1-3-2", amount=300)])
        self._call(capture)
        pr = capture.last_purchase_result("20260101_01_05")
        self.assertIsInstance(pr, LegacyPurchaseResult)
        self.assertEqual(len(pr.purchases), 2)
        self.assertEqual(pr.purchases[0]["combo"], "1-2-3")
        self.assertEqual(pr.purchases[1]["combo"], "1-3-2")

    def test_only_one_evaluate_bets_call_per_capture(self):
        capture, calls = _make_capture([_bet("1-2-3")])
        capture(race_date="20260101", venue_num=1, race_number=5)
        self.assertEqual(calls["count"], 1)

    def test_empty_list_raises_and_no_stale_purchase_result(self):
        capture, _ = _make_capture([])
        with self.assertRaises(ValueError):
            self._call(capture)

    def test_skip_single_item_preserved_in_purchase_result(self):
        skip_item = _bet("1-2-3", purchased=False, amount=0)
        skip_item["skip_reason"] = "BuyScore不足"
        capture, _ = _make_capture([skip_item])
        self._call(capture)
        pr = capture.last_purchase_result("20260101_01_05")
        self.assertEqual(len(pr.purchases), 1)
        self.assertFalse(pr.purchases[0]["purchased"])

    def test_mismatched_eval_id_raises(self):
        capture, _ = _make_capture([_bet("1-2-3")])
        self._call(capture)
        with self.assertRaises(ValueError):
            capture.last_purchase_result("20260102_09_03")

    def test_purchase_result_before_any_call_raises(self):
        capture = _EvaluateBetsCapture()
        with self.assertRaises(ValueError):
            capture.last_purchase_result("20260101_01_05")


if __name__ == "__main__":
    unittest.main()
