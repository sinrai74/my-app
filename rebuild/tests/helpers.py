"""
テスト共通ヘルパー。

Step2実装計画書 §4（往復戦略）に基づく。float比較の1e-6許容は
このモジュールに1箇所集約し、各テストでの重複実装を禁止する。
標準ライブラリのみ使用。
"""

from __future__ import annotations

import math
import unittest
from typing import Any

# 浮動小数比較の既定許容誤差（設計書⑭・Step2計画書§1.2で確定した値）
FLOAT_TOLERANCE: float = 1e-6


def floats_equal(a: float, b: float, tol: float = FLOAT_TOLERANCE) -> bool:
    """2つのfloatが許容誤差tol以内で等しいか。NaNどうしは等しいとみなさない。"""
    if math.isnan(a) or math.isnan(b):
        return False
    return abs(a - b) <= tol


def assert_row_equal(
    case: unittest.TestCase,
    expected: dict[str, Any],
    actual: dict[str, Any],
    tol: float = FLOAT_TOLERANCE,
) -> None:
    """2つのRow（dict[str, Any]）の同一性を検証する。

    - キー集合の完全一致を要求する（欠落・過剰の双方を検出）
    - float値（bool除く）は許容誤差tolで比較する
    - int⇔floatの型差は数値として等しければ許容する（例: 0 と 0.0）
      ただしboolとintの混同は不一致とする
    - それ以外の型は == で比較する
    失敗時は不一致の列名を含むメッセージでAssertionErrorを送出する。
    """
    expected_keys = set(expected.keys())
    actual_keys = set(actual.keys())
    case.assertEqual(
        expected_keys,
        actual_keys,
        msg=(
            f"Row keys differ. missing={sorted(expected_keys - actual_keys)} "
            f"extra={sorted(actual_keys - expected_keys)}"
        ),
    )
    for key in sorted(expected_keys):
        e = expected[key]
        a = actual[key]
        if _is_number(e) and _is_number(a):
            case.assertTrue(
                floats_equal(float(e), float(a), tol),
                msg=f"column {key!r}: expected {e!r}, actual {a!r} (tol={tol})",
            )
        else:
            # Python の True == 1 / False == 0 仕様による bool⇔int の
            # すり抜けを防ぐ（bool と非boolの混在は型不一致として扱う）
            if isinstance(e, bool) != isinstance(a, bool):
                case.fail(
                    f"column {key!r}: bool/non-bool mismatch. "
                    f"expected {e!r} ({type(e).__name__}), "
                    f"actual {a!r} ({type(a).__name__})"
                )
            case.assertEqual(e, a, msg=f"column {key!r}: expected {e!r}, actual {a!r}")


def _is_number(value: Any) -> bool:
    """数値比較の対象か（boolは数値扱いしない）。"""
    return isinstance(value, (int, float)) and not isinstance(value, bool)
