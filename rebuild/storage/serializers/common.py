"""
Serializer層の共通部品（スキーマ版数・必須キー取得）。

設計書 v1.1.6 ④、Step2実装計画書 §2 に基づく。
"""

from __future__ import annotations

from typing import Any

from storage.exceptions import ParseError

SCHEMA_VERSION: int = 1


def require_key(data: dict[str, Any], key: str) -> Any:
    """必須キーの取得。欠落はParseError（④: 書き手は必須キーを省略しない）。"""
    if key not in data:
        raise ParseError(f"required key missing: {key!r}")
    return data[key]
