"""
core層の例外（Step4-6エラー方針対応）。

storage.exceptions（ParseError等）はL3の例外であり、core（L2）は⑩の依存ルール上
storageをimportできないため、core専用のValidationErrorをここに定義する。
サイレント補正の禁止: 必須データの不足は本例外で明示的に失敗させる。
"""

from __future__ import annotations


class ValidationError(ValueError):
    """入力・設定の必須要素が不足している場合に送出する。"""
