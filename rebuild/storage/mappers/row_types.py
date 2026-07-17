"""
Row型エイリアスとCSV値正規化の共通部品。

Step2実装計画書 §1.2（Mapper共通インターフェース）・§3（CSV互換戦略）に基づく。

Normalization Rules
- "" <-> None
- bool <-> 0/1
- float comparison tolerance is handled outside this module (see tests/helpers.py)
- Invalid values raise ValueError
- No silent coercion

正規化規則（計画書§3の3・4を実装）:
- 現行CSVの空欄（空文字）は from_row 側で None に正規化する
- to_row 側で None は空文字として出力する（現行CSVと同じ見た目を維持）
- bool列は 0/1 の整数表記と相互変換する
- 値の丸めは行わない（float比較の1e-6許容はテストヘルパー側の責務）

依存: 標準ライブラリのみ。ファイルI/Oなし。例外クラス（ParseError等）の導入は
Step2-3（HitRecordCsvMapper）まで行わず、本モジュールの各関数は不正入力に対して
Python標準の ValueError をそのまま送出する。
"""

from __future__ import annotations

from typing import Any, Optional

# Mapperが扱う「行」の型。キー=列名、値=プリミティブ（str/int/float/bool/None）。
# ネスト構造（dict/tuple）はJSON文字列列として格納する（計画書§1.2）。
Row = dict[str, Any]

# CSV上で「値なし」を表す表記。現行hit_record.csv等の空欄に一致させる。
EMPTY: str = ""


def parse_optional_str(value: Any) -> Optional[str]:
    """CSVセル値をOptional[str]へ正規化する。空文字・NoneはNone。"""
    if value is None:
        return None
    text = str(value)
    if text == EMPTY:
        return None
    return text


def parse_optional_int(value: Any) -> Optional[int]:
    """CSVセル値をOptional[int]へ正規化する。空文字・NoneはNone。

    現行CSVでは整数列に "38320" のような文字列が入る。float表記の整数
    （"1.0" 等）は許容しない（型ゆらぎの黙認を避けるため、ValueErrorとする）。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # bool は int のサブクラスだが、整数列への bool 混入は型ゆらぎとして拒否する
        raise ValueError(f"bool value is not allowed for int column: {value!r}")
    if isinstance(value, int):
        return value
    text = str(value)
    if text == EMPTY:
        return None
    return int(text)


def parse_optional_float(value: Any) -> Optional[float]:
    """CSVセル値をOptional[float]へ正規化する。空文字・NoneはNone。"""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"bool value is not allowed for float column: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    if text == EMPTY:
        return None
    return float(text)


def parse_bool01(value: Any) -> Optional[bool]:
    """0/1表記のCSVセル値をOptional[bool]へ正規化する。

    許容入力: 0, 1, "0", "1", True, False, None, 空文字。
    それ以外（"2", "yes" 等）はValueError（黙認による型ゆらぎ拡大を防ぐ）。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"bool01 column accepts only 0/1: {value!r}")
    text = str(value)
    if text == EMPTY:
        return None
    if text == "0":
        return False
    if text == "1":
        return True
    raise ValueError(f"bool01 column accepts only '0'/'1'/'': {value!r}")


def format_optional(value: Any) -> Any:
    """to_row側の出力正規化。Noneは空文字、それ以外はそのまま返す。

    値の丸め・文字列化の強制は行わない（現行CSVの書式維持は
    Repository層の書込処理とテスト側の数値比較で扱う。計画書§10リスク3）。
    """
    if value is None:
        return EMPTY
    return value


def format_bool01(value: Optional[bool]) -> Any:
    """to_row側のbool出力。Trueは1、Falseは0、Noneは空文字。"""
    if value is None:
        return EMPTY
    return 1 if value else 0
