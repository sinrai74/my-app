"""Stage 2: Legacy側購入ビュー・legacy_values_builder拡張の単体テスト。"""

from __future__ import annotations

import unittest

from shadow.legacy_source import LegacySentRecord
from shadow.legacy_values_builder import build_legacy_values


def _record(raw: dict) -> LegacySentRecord:
    return LegacySentRecord(eval_id="20260704_12_05", raw=raw)


class TestPurchaseView(unittest.TestCase):
    def test_multi_purchase(self):
        rec = _record({
            "buy": ["1-2-3", "1-3-2", "2-1-3"],
            "buy_amounts": [300, 200, 100],
        })
        v = rec.purchase_view()
        self.assertEqual(v["purchased_combos"], ["1-2-3", "1-3-2", "2-1-3"])
        self.assertEqual(v["purchased_amounts"], [300, 200, 100])
        self.assertEqual(v["n_bets"], 3)
        self.assertEqual(v["cost"], 600)

    def test_single_purchase(self):
        rec = _record({"buy": ["1-2-3"], "buy_amounts": [500]})
        v = rec.purchase_view()
        self.assertEqual(v["n_bets"], 1)
        self.assertEqual(v["cost"], 500)
        self.assertEqual(v["purchased_combos"], ["1-2-3"])
        self.assertEqual(v["purchased_amounts"], [500])

    def test_index_correspondence_preserved(self):
        rec = _record({
            "buy": ["a", "b", "c", "d"],
            "buy_amounts": [1, 2, 3, 4],
        })
        v = rec.purchase_view()
        for i, combo in enumerate(v["purchased_combos"]):
            self.assertEqual(v["purchased_amounts"][i], i + 1)

    def test_order_preserved_not_sorted(self):
        # buyscore降順の順序を維持する（sortしない）。
        rec = _record({"buy": ["9-9-9", "1-1-1"], "buy_amounts": [100, 900]})
        v = rec.purchase_view()
        self.assertEqual(v["purchased_combos"], ["9-9-9", "1-1-1"])
        self.assertEqual(v["purchased_amounts"], [100, 900])

    def test_empty_buy_list(self):
        rec = _record({"buy": [], "buy_amounts": []})
        v = rec.purchase_view()
        self.assertEqual(v["n_bets"], 0)
        self.assertEqual(v["cost"], 0)

    def test_no_buy_keys_returns_none(self):
        rec = _record({"combo": "1-2-3", "prob": 0.1})
        self.assertIsNone(rec.purchase_view())

    def test_cost_field_name_matches_buydecision(self):
        # Rebuild BuyDecision.cost と同名のキーを使う（total_costではない）。
        rec = _record({"buy": ["1-2-3"], "buy_amounts": [500]})
        v = rec.purchase_view()
        self.assertIn("cost", v)
        self.assertNotIn("total_cost", v)


class TestBuildLegacyValues(unittest.TestCase):
    def _records(self, raw):
        return {"20260704_12_05": _record(raw)}

    def test_race_always_present(self):
        recs = self._records({"venue_num": 12, "race": 5, "venue": "住之江"})
        v = build_legacy_values(recs, "20260704_12_05")
        self.assertIn("race", v)

    def test_buy_decision_added_when_buy_present(self):
        recs = self._records({
            "venue_num": 12, "race": 5,
            "buy": ["1-2-3", "1-3-2"], "buy_amounts": [300, 200],
        })
        v = build_legacy_values(recs, "20260704_12_05")
        self.assertIn("buy_decision", v)
        self.assertEqual(v["buy_decision"]["n_bets"], 2)
        self.assertEqual(v["buy_decision"]["cost"], 500)

    def test_buy_decision_absent_when_no_buy(self):
        recs = self._records({"venue_num": 12, "race": 5})
        v = build_legacy_values(recs, "20260704_12_05")
        self.assertNotIn("buy_decision", v)

    def test_evaluation_added_when_present(self):
        recs = self._records({
            "venue_num": 12, "race": 5,
            "upset_score": 42.0, "race_type": "混戦",
        })
        v = build_legacy_values(recs, "20260704_12_05")
        self.assertIn("evaluation", v)
        self.assertEqual(v["evaluation"]["upset_score"], 42.0)

    def test_evaluation_absent_when_empty(self):
        recs = self._records({"venue_num": 12, "race": 5})
        v = build_legacy_values(recs, "20260704_12_05")
        self.assertNotIn("evaluation", v)

    def test_missing_eval_id_returns_none(self):
        recs = self._records({"venue_num": 12, "race": 5})
        self.assertIsNone(build_legacy_values(recs, "99999999_99_99"))


if __name__ == "__main__":
    unittest.main()
