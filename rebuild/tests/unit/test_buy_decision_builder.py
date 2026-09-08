import unittest

from core.buyscore import BuyAssessment
from models.evaluation import LegacyPurchaseResult
from pipelines.buy_decision_builder import DefaultBuyDecisionBuilder


def _assessment(**over):
    base = dict(
        eval_id="e1", buyscore=70.0, investment_type="堅実",
        kelly_fraction=0.05, skip_reason=None, config_version="t-1.0",
    )
    base.update(over)
    return BuyAssessment(**base)


def _purchase_result(candidates, eval_id="e1"):
    return LegacyPurchaseResult(eval_id=eval_id, purchases=tuple(candidates))


class TestBuyDecisionBuilder(unittest.TestCase):
    def setUp(self):
        self.builder = DefaultBuyDecisionBuilder()

    def test_case_a_skip(self):
        pr = _purchase_result([{
            "combo": "1-2-3", "purchased": False,
            "skip_reason": "BuyScore不足", "amount": 0,
        }])
        assess = _assessment(skip_reason="BuyScore不足")
        d = self.builder.build(pr, assess)
        self.assertFalse(d.purchased)
        self.assertEqual(d.n_bets, 0)
        self.assertEqual(d.cost, 0)
        self.assertEqual(d.purchased_combos, ())
        self.assertEqual(d.purchased_amounts, ())
        self.assertEqual(d.skip_reason, "BuyScore不足")

    def test_case_b_single_purchase(self):
        pr = _purchase_result([{"combo": "1-2-3", "purchased": True, "amount": 500}])
        d = self.builder.build(pr, _assessment())
        self.assertTrue(d.purchased)
        self.assertEqual(d.n_bets, 1)
        self.assertEqual(d.cost, 500)
        self.assertEqual(d.purchased_combos, ("1-2-3",))
        self.assertEqual(d.purchased_amounts, (500,))
        self.assertIsNone(d.skip_reason)

    def test_case_c_multi_purchase(self):
        pr = _purchase_result([
            {"combo": "1-2-3", "purchased": True, "amount": 500},
            {"combo": "1-3-2", "purchased": True, "amount": 300},
        ])
        d = self.builder.build(pr, _assessment())
        self.assertEqual(d.n_bets, 2)
        self.assertEqual(d.cost, 800)
        self.assertEqual(d.purchased_combos, ("1-2-3", "1-3-2"))
        self.assertEqual(d.purchased_amounts, (500, 300))

    def test_case_d_max_points(self):
        pr = _purchase_result([
            {"combo": f"1-{i}-x", "purchased": True, "amount": 100}
            for i in range(2, 6)
        ])
        d = self.builder.build(pr, _assessment())
        self.assertEqual(d.n_bets, 4)
        self.assertEqual(d.cost, 400)

    def test_case_e_varying_amounts_and_index_correspondence(self):
        pr = _purchase_result([
            {"combo": "1-2-3", "purchased": True, "amount": 1000},
            {"combo": "1-2-4", "purchased": True, "amount": 200},
            {"combo": "1-2-5", "purchased": True, "amount": 700},
        ])
        d = self.builder.build(pr, _assessment())
        self.assertEqual(d.purchased_combos, ("1-2-3", "1-2-4", "1-2-5"))
        self.assertEqual(d.purchased_amounts, (1000, 200, 700))
        for combo, amount in zip(d.purchased_combos, d.purchased_amounts):
            self.assertIn(combo, {"1-2-3": 1000, "1-2-4": 200, "1-2-5": 700})
            self.assertEqual(
                {"1-2-3": 1000, "1-2-4": 200, "1-2-5": 700}[combo], amount
            )
        self.assertEqual(d.cost, 1900)

    def test_case_f_multiplier_already_reflected_upstream(self):
        pr = _purchase_result([{"combo": "1-2-3", "purchased": True, "amount": 250}])
        d = self.builder.build(pr, _assessment())
        self.assertEqual(d.cost, 250)

    def test_case_g_no_candidates(self):
        pr = _purchase_result([])
        d = self.builder.build(pr, _assessment(skip_reason=None))
        self.assertFalse(d.purchased)
        self.assertEqual(d.n_bets, 0)
        self.assertEqual(d.cost, 0)

    def test_eval_id_mismatch_raises(self):
        pr = _purchase_result(
            [{"combo": "1-2-3", "purchased": True, "amount": 100}], eval_id="e1")
        assess = _assessment(eval_id="e2")
        with self.assertRaises(ValueError):
            self.builder.build(pr, assess)

    def test_assessment_fields_passthrough(self):
        pr = _purchase_result([{"combo": "1-2-3", "purchased": True, "amount": 100}])
        assess = _assessment(buyscore=88.0, investment_type="穴狙い",
                              kelly_fraction=0.12, config_version="v9")
        d = self.builder.build(pr, assess)
        self.assertEqual(d.buyscore, 88.0)
        self.assertEqual(d.investment_type, "穴狙い")
        self.assertEqual(d.kelly_fraction, 0.12)
        self.assertEqual(d.config_version, "v9")

    def test_order_preserved_matches_legacy_buyscore_descending(self):
        # Legacy側はbuyscore降順で並んでいる前提。Builderは並び替えず
        # そのままの順序を維持する（順序に意味があるため）。
        pr = _purchase_result([
            {"combo": "best", "purchased": True, "amount": 900, "buyscore": 95},
            {"combo": "second", "purchased": True, "amount": 100, "buyscore": 60},
        ])
        d = self.builder.build(pr, _assessment())
        self.assertEqual(d.purchased_combos[0], "best")
        self.assertEqual(d.purchased_combos[1], "second")


if __name__ == "__main__":
    unittest.main()
