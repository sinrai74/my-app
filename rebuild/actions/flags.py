"""
切替フラグ（Step5-6・指示7）: フラグまたはDI切替のみで行う。

Legacyコードは書き換えない。環境変数 USE_REBUILD_PIPELINE の真偽値を
読むだけ（判定・計算はしない。単純な読み取りと変換）。
"""

from __future__ import annotations

import os


def use_rebuild_pipeline(env: dict | None = None) -> bool:
    """USE_REBUILD_PIPELINE 環境変数を読む（既定False=Legacy継続）。"""
    source = env if env is not None else os.environ
    value = source.get("USE_REBUILD_PIPELINE", "")
    return value.strip().lower() in ("1", "true", "yes", "on")
