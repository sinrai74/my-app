"""
Step2-1共通基盤（storage/mappers/row_types.py・tests/helpers.py）の単体テスト。

Step2実装計画書 §3（CSV互換戦略の正規化規則）・§4（1e-6比較）に対応。
標準ライブラリ unittest のみを使用する。
"""

from __future__ import annotations

import unittest

from storage.mappers.row_types import (
    format_bool01,
    format_optional,
    parse_bool01,
    parse_optional_float,
    parse_optional_int,
    parse_optional_str,
)
from tests.helpers import FLOAT_TOLERANCE, assert_row_equal, floats_equal


class TestParseOptionalStr(unittest.TestCase):
    def test_empty_string_becomes_none(self) -> None:
        """現行CSVの空欄（空文字）はNoneに正規化されること（計画書§3-3）。"""
        self.assertIsNone(parse_optional_str(""))

    def test_none_stays_none(self) -> None:
        self.assertIsNone(parse_optional_str(None))

    def test_normal_string(self) -> None:
        self.assertEqual(parse_optional_str("住之江"), "住之江")

    def test_numeric_input_becomes_str(self) -> None:
        """csvモジュール以外の経路で数値が来た場合も文字列化されること。"""
        self.assertEqual(parse_optional_str(123), "123")


class TestParseOptionalInt(unittest.TestCase):
    def test_empty_string_becomes_none(self) -> None:
        self.assertIsNone(parse_optional_int(""))

    def test_none_stays_none(self) -> None:
        self.assertIsNone(parse_optional_int(None))

    def test_str_digits(self) -> None:
        self.assertEqual(parse_optional_int("38320"), 38320)

    def test_int_passthrough(self) -> None:
        self.assertEqual(parse_optional_int(90), 90)

    def test_float_notation_raises(self) -> None:
        """整数列へのfloat表記（"1.0"）は型ゆらぎとして拒否されること。"""
        with self.assertRaises(ValueError):
            parse_optional_int("1.0")

    def test_bool_raises(self) -> None:
        """整数列へのbool混入は拒否されること（黙認による型ゆらぎ防止）。"""
        with self.assertRaises(ValueError):
            parse_optional_int(True)


class TestParseOptionalFloat(unittest.TestCase):
    def test_empty_string_becomes_none(self) -> None:
        self.assertIsNone(parse_optional_float(""))

    def test_none_stays_none(self) -> None:
        self.assertIsNone(parse_optional_float(None))

    def test_str_float(self) -> None:
        self.assertEqual(parse_optional_float("0.04591"), 0.04591)

    def test_int_input_becomes_float(self) -> None:
        result = parse_optional_float(3)
        self.assertIsInstance(result, float)
        self.assertEqual(result, 3.0)

    def test_bool_raises(self) -> None:
        with self.assertRaises(ValueError):
            parse_optional_float(False)


class TestParseBool01(unittest.TestCase):
    def test_str_zero_one(self) -> None:
        self.assertIs(parse_bool01("0"), False)
        self.assertIs(parse_bool01("1"), True)

    def test_int_zero_one(self) -> None:
        self.assertIs(parse_bool01(0), False)
        self.assertIs(parse_bool01(1), True)

    def test_bool_passthrough(self) -> None:
        self.assertIs(parse_bool01(True), True)

    def test_empty_and_none_become_none(self) -> None:
        self.assertIsNone(parse_bool01(""))
        self.assertIsNone(parse_bool01(None))

    def test_invalid_values_raise(self) -> None:
        """0/1以外の値は黙認せず拒否されること。"""
        with self.assertRaises(ValueError):
            parse_bool01("2")
        with self.assertRaises(ValueError):
            parse_bool01("yes")
        with self.assertRaises(ValueError):
            parse_bool01(2)


class TestFormatters(unittest.TestCase):
    def test_format_optional_none_becomes_empty(self) -> None:
        """to_row側でNoneは空文字になること（現行CSVの見た目維持、計画書§3-3）。"""
        self.assertEqual(format_optional(None), "")

    def test_format_optional_value_passthrough(self) -> None:
        """None以外の値は丸め・変換せずそのまま返ること（計画書§10リスク3）。"""
        self.assertEqual(format_optional(0.04591), 0.04591)
        self.assertEqual(format_optional("住之江"), "住之江")
        self.assertEqual(format_optional(0), 0)

    def test_format_bool01(self) -> None:
        self.assertEqual(format_bool01(True), 1)
        self.assertEqual(format_bool01(False), 0)
        self.assertEqual(format_bool01(None), "")

    def test_parse_format_roundtrip_bool(self) -> None:
        """bool01の往復（parse -> format -> parse）が同値になること。"""
        for original in (True, False, None):
            formatted = format_bool01(original)
            self.assertIs(parse_bool01(formatted), original)


class TestFloatsEqual(unittest.TestCase):
    def test_within_tolerance(self) -> None:
        self.assertTrue(floats_equal(1.0, 1.0 + 1e-7))

    def test_outside_tolerance(self) -> None:
        self.assertFalse(floats_equal(1.0, 1.0 + 1e-5))

    def test_exact(self) -> None:
        self.assertTrue(floats_equal(0.04591, 0.04591))

    def test_nan_never_equal(self) -> None:
        self.assertFalse(floats_equal(float("nan"), float("nan")))

    def test_default_tolerance_is_1e_6(self) -> None:
        """既定許容誤差が設計確定値1e-6であること。"""
        self.assertEqual(FLOAT_TOLERANCE, 1e-6)


class TestAssertRowEqual(unittest.TestCase):
    def test_identical_rows_pass(self) -> None:
        row = {"date": "20260704", "payout": 38320, "pred_prob": 0.04591, "hit": 0}
        assert_row_equal(self, row, dict(row))

    def test_float_within_tolerance_passes(self) -> None:
        a = {"pred_prob": 0.045910000}
        b = {"pred_prob": 0.045910001}
        assert_row_equal(self, a, b)

    def test_float_outside_tolerance_fails(self) -> None:
        a = {"pred_prob": 0.04591}
        b = {"pred_prob": 0.04701}
        with self.assertRaises(AssertionError):
            assert_row_equal(self, a, b)

    def test_int_float_numeric_equality_passes(self) -> None:
        """int 0 と float 0.0 は数値として等しければ許容されること。"""
        assert_row_equal(self, {"cost": 0}, {"cost": 0.0})

    def test_missing_key_fails(self) -> None:
        with self.assertRaises(AssertionError):
            assert_row_equal(self, {"a": 1, "b": 2}, {"a": 1})

    def test_extra_key_fails(self) -> None:
        with self.assertRaises(AssertionError):
            assert_row_equal(self, {"a": 1}, {"a": 1, "b": 2})

    def test_string_mismatch_fails(self) -> None:
        with self.assertRaises(AssertionError):
            assert_row_equal(self, {"venue": "住之江"}, {"venue": "桐生"})

    def test_bool_vs_int_is_not_numeric_equal(self) -> None:
        """boolとintの混同は不一致とすること（True == 1 の同値を許さない）。"""
        with self.assertRaises(AssertionError):
            assert_row_equal(self, {"hit": True}, {"hit": 1})

    def test_none_vs_empty_string_fails(self) -> None:
        """Noneと空文字は別値として区別されること（正規化はMapperの責務）。"""
        with self.assertRaises(AssertionError):
            assert_row_equal(self, {"skip_reason": None}, {"skip_reason": ""})


if __name__ == "__main__":
    unittest.main()
