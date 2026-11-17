"""段階4-2：g3_compare（4項目比較判定）の回帰テスト。"""
import unittest

from core.g3_compare import (
    DEFAULT_COMPARE_POLICY,
    INCOMPARABLE,
    LAYER_CAPITAL,
    LAYER_EVAL_LOGIC,
    LAYER_NONE,
    LAYER_ZERO_AMOUNT,
    MATCH,
    MISMATCH,
    ComparePolicy,
    classify_legacy_skip_layer,
    compare_four_items,
    compare_number,
    compare_skip_reason,
    judge_race,
)


class CompareNumberTests(unittest.TestCase):
    def test_exact_equal_default(self):
        self.assertEqual(compare_number(60.0, 60.0, None, None), MATCH)
        self.assertEqual(compare_number(60.0, 60.1, None, None), MISMATCH)

    def test_none_or_nonfinite_incomparable(self):
        self.assertEqual(compare_number(None, 1.0, None, None), INCOMPARABLE)
        self.assertEqual(compare_number(1.0, None, None, None), INCOMPARABLE)
        self.assertEqual(compare_number(float("nan"), 1.0, None, None), INCOMPARABLE)
        self.assertEqual(compare_number(True, 1.0, None, None), INCOMPARABLE)  # bool は数値扱いしない

    def test_round_and_tol(self):
        self.assertEqual(compare_number(0.12345, 0.12346, 4, None), MATCH)   # 4桁で丸め一致
        self.assertEqual(compare_number(60.04, 60.0, None, 0.05), MATCH)     # 許容差内
        self.assertEqual(compare_number(60.2, 60.0, None, 0.05), MISMATCH)


class SkipLayerTests(unittest.TestCase):
    def test_classify_layers(self):
        self.assertEqual(classify_legacy_skip_layer(False, None, None), LAYER_NONE)
        self.assertEqual(classify_legacy_skip_layer(True, "資金管理による見送り", "資金管理による見送り"),
                         LAYER_CAPITAL)
        self.assertEqual(classify_legacy_skip_layer(True, "ベット金額が0円のため見送り（資金管理の多重判定による）", "別の資金理由"),
                         LAYER_ZERO_AMOUNT)
        self.assertEqual(classify_legacy_skip_layer(True, "BuyScore不足(55)", "資金管理の別理由"),
                         LAYER_EVAL_LOGIC)
        # passthrough=True だが理由 None は保守側で capital（Rebuild と別層＝比較不能になる）
        self.assertEqual(classify_legacy_skip_layer(True, None, None), LAYER_CAPITAL)

    def test_compare_skip_same_layer(self):
        # 両方買い（None）
        self.assertEqual(compare_skip_reason(None, LAYER_NONE, None), MATCH)
        # 片方だけ見送り
        self.assertEqual(compare_skip_reason("BuyScore不足(55)", LAYER_NONE, None), MISMATCH)
        self.assertEqual(compare_skip_reason(None, LAYER_EVAL_LOGIC, "BuyScore不足(55)"), MISMATCH)
        # 同層・同文字列
        self.assertEqual(compare_skip_reason("BuyScore不足(55)", LAYER_EVAL_LOGIC, "BuyScore不足(55)"), MATCH)
        self.assertEqual(compare_skip_reason("BuyScore不足(55)", LAYER_EVAL_LOGIC, "期待値不足(EV1.20)"), MISMATCH)

    def test_compare_skip_cross_layer_incomparable(self):
        # Legacy が資金管理/金額0円層 → Rebuild（none/eval_logic）と層が異なる → 比較不能
        self.assertEqual(compare_skip_reason(None, LAYER_CAPITAL, "資金管理"), INCOMPARABLE)
        self.assertEqual(compare_skip_reason("BuyScore不足(55)", LAYER_CAPITAL, "資金管理"), INCOMPARABLE)
        self.assertEqual(compare_skip_reason(None, LAYER_ZERO_AMOUNT, "ベット金額が0円…"), INCOMPARABLE)


class FourItemsTests(unittest.TestCase):
    def _buy(self, **kw):
        base = dict(
            rebuild_buyscore=72.0, legacy_buyscore=72.0,
            rebuild_kelly=0.1234, legacy_kelly=0.1234,
            rebuild_investment_type="堅実", legacy_investment_type="堅実",
            rebuild_skip_reason=None,
            legacy_passthrough=False, legacy_passthrough_reason=None, legacy_capital_reason="資金理由",
        )
        base.update(kw)
        return compare_four_items(**base)

    def test_all_match_buy(self):
        r = self._buy()
        self.assertEqual(r["race_judge"], MATCH)
        self.assertTrue(all(v == MATCH for v in r["items"].values()))

    def test_numeric_mismatch(self):
        r = self._buy(legacy_buyscore=71.9)
        self.assertEqual(r["items"]["buyscore"], MISMATCH)
        self.assertEqual(r["race_judge"], MISMATCH)

    def test_cross_layer_skip_incomparable(self):
        # Legacy 資金管理見送り、Rebuild 買い → skip_reason 比較不能 → race 比較不能
        r = self._buy(legacy_passthrough=True, legacy_passthrough_reason="資金理由",
                      legacy_capital_reason="資金理由")
        self.assertEqual(r["items"]["skip_reason"], INCOMPARABLE)
        self.assertEqual(r["race_judge"], INCOMPARABLE)
        self.assertEqual(r["skip_layer"]["legacy"], LAYER_CAPITAL)

    def test_same_layer_skip_match(self):
        # 両者 eval_logic 見送り・同文字列 → 一致
        r = self._buy(rebuild_skip_reason="BuyScore不足(55)",
                      legacy_passthrough=True, legacy_passthrough_reason="BuyScore不足(55)",
                      legacy_capital_reason="別の資金理由")
        self.assertEqual(r["items"]["skip_reason"], MATCH)
        self.assertEqual(r["race_judge"], MATCH)

    def test_policy_none_numbers_incomparable(self):
        r = self._buy(policy=None)
        self.assertEqual(r["items"]["buyscore"], INCOMPARABLE)
        self.assertEqual(r["items"]["kelly_fraction"], INCOMPARABLE)
        self.assertEqual(r["race_judge"], INCOMPARABLE)
        self.assertFalse(r["compare_policy_set"])

    def test_investment_type_none_incomparable(self):
        r = self._buy(legacy_investment_type=None)
        self.assertEqual(r["items"]["investment_type"], INCOMPARABLE)
        self.assertEqual(r["race_judge"], INCOMPARABLE)


class JudgeRaceTests(unittest.TestCase):
    def test_priority(self):
        self.assertEqual(judge_race({"a": MATCH, "b": MATCH}), MATCH)
        self.assertEqual(judge_race({"a": MATCH, "b": MISMATCH}), MISMATCH)
        self.assertEqual(judge_race({"a": MISMATCH, "b": INCOMPARABLE}), INCOMPARABLE)


if __name__ == "__main__":
    unittest.main()
