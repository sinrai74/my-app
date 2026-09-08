import dataclasses
import unittest

from models.evaluation import BuyDecision
from storage.mappers.hit_record_mapper import (
    ALL_COLUMNS,
    LEGACY_COLUMNS,
    HitRecordCsvMapper,
)
from storage.serializers.evaluation_serializer import BuyDecisionSerializer
from tests.unit.test_mappers_hit_record import _full_record


def _decision(**over):
    base = dict(
        eval_id="e1", purchased=True, buyscore=80.0, investment_type="堅実",
        n_bets=2, cost=800, kelly_fraction=0.08, config_version="t-1.0",
        skip_reason=None, purchased_combos=("1-2-3", "1-3-2"),
        purchased_amounts=(500, 300),
    )
    base.update(over)
    return BuyDecision(**base)


class TestBuyDecisionSerializerNewFields(unittest.TestCase):
    def test_roundtrip_preserves_combos_and_amounts_order(self):
        model = _decision()
        restored = BuyDecisionSerializer.from_dict(
            BuyDecisionSerializer.to_dict(model)
        )
        self.assertEqual(restored.purchased_combos, ("1-2-3", "1-3-2"))
        self.assertEqual(restored.purchased_amounts, (500, 300))
        self.assertEqual(restored, model)

    def test_old_format_dict_without_new_fields_defaults_to_empty(self):
        model = _decision()
        old_style = BuyDecisionSerializer.to_dict(model)
        del old_style["purchased_combos"]
        del old_style["purchased_amounts"]
        restored = BuyDecisionSerializer.from_dict(old_style)
        self.assertEqual(restored.purchased_combos, ())
        self.assertEqual(restored.purchased_amounts, ())

    def test_skip_case_empty_tuples_roundtrip(self):
        model = _decision(
            purchased=False, n_bets=0, cost=0, skip_reason="BuyScore不足",
            purchased_combos=(), purchased_amounts=(),
        )
        restored = BuyDecisionSerializer.from_dict(
            BuyDecisionSerializer.to_dict(model)
        )
        self.assertEqual(restored.purchased_combos, ())
        self.assertEqual(restored.purchased_amounts, ())


class TestCsvSchemaCompatibility(unittest.TestCase):
    def test_legacy_44_columns_unchanged(self):
        # 44互換列自体（列名・順序・件数）は今回変更していないことを保証する。
        self.assertEqual(len(LEGACY_COLUMNS), 44)

    def test_new_extension_columns_appended_not_inserted(self):
        self.assertIn("purchased_combos_json", ALL_COLUMNS)
        self.assertIn("purchased_amounts_json", ALL_COLUMNS)
        # 新拡張列は既存拡張列より後ろ（末尾側）にあること＝挿入ではなく追加。
        idx_patterns = ALL_COLUMNS.index("patterns_json")
        idx_combos = ALL_COLUMNS.index("purchased_combos_json")
        idx_amounts = ALL_COLUMNS.index("purchased_amounts_json")
        self.assertLess(idx_patterns, idx_combos)
        self.assertLess(idx_combos, idx_amounts)


class TestCsvRoundtripNewFields(unittest.TestCase):
    def test_new_record_csv_roundtrip_preserves_combos_and_amounts(self):
        record = _full_record()
        record = dataclasses.replace(
            record,
            buy_decision=dataclasses.replace(
                record.buy_decision,
                purchased_combos=("1-2-3", "1-3-2", "2-1-3"),
                purchased_amounts=(500, 300, 100),
            ),
        )
        row = HitRecordCsvMapper.to_row(record)
        restored = HitRecordCsvMapper.from_row(row)
        self.assertEqual(
            restored.buy_decision.purchased_combos, ("1-2-3", "1-3-2", "2-1-3")
        )
        self.assertEqual(
            restored.buy_decision.purchased_amounts, (500, 300, 100)
        )
        self.assertEqual(restored, record)

    def test_legacy_row_without_new_columns_restores_empty_tuples(self):
        record = _full_record()
        row = HitRecordCsvMapper.to_row(record)
        # レガシー行を模擬: 新拡張列を削除（旧CSVには存在しない）。
        del row["purchased_combos_json"]
        del row["purchased_amounts_json"]
        restored = HitRecordCsvMapper.from_row(row)
        self.assertEqual(restored.buy_decision.purchased_combos, ())
        self.assertEqual(restored.buy_decision.purchased_amounts, ())


if __name__ == "__main__":
    unittest.main()
