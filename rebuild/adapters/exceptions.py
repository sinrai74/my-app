"""
Provider専用例外（Step5-1）。

既存例外（storage.exceptions / core.exceptions）は変更禁止のため、
adapters層専用の例外をここに定義する。Legacyの取得失敗・不正データを
Provider境界でこの例外へ正規化する（サイレント失敗の禁止）。
"""

from __future__ import annotations


class ProviderError(Exception):
    """Providerによるデータ取得・変換の失敗を表す。"""
