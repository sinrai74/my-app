"""
Legacy Output取得（Step6-2・必須①確定方針）: 生成済みHTMLをbyte読み取り。

方針（レビュー確定）:
  - Legacyが生成した実HTMLファイルを読むだけ
  - rebuildが生成した実HTMLファイルを読むだけ
  - byte一致のみ確認する
  - Rendererの再実行・戻り値dictの比較はしない（禁止）

Legacyは一切変更しない（ファイルを読むだけ）。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def read_output_bytes(path: str) -> Optional[bytes]:
    """生成済み成果物をbyte読み取りする（存在しなければNone）。

    Noneは「Legacy取得元なし」を意味し、比較側でスキップされる
    （欠損を差分として誤検出しない）。
    """
    file_path = Path(path)
    if not file_path.exists():
        log.info("Legacy output not found path=%s", path)
        return None
    return file_path.read_bytes()


def compare_output_bytes(
    eval_id: str, legacy_path: str, rebuild_path: str
) -> list[dict]:
    """2つの成果物のbyte一致を確認し、差分を標準形式で返す。

    一致すれば空リスト。片方でも存在しなければ空リスト（比較不能=
    「Legacy取得元なし」として扱い、差分にはしない）。
    差分内容の補正・自動変換はしない（違いを記録するだけ）。
    """
    legacy_bytes = read_output_bytes(legacy_path)
    rebuild_bytes = read_output_bytes(rebuild_path)
    if legacy_bytes is None or rebuild_bytes is None:
        return []
    if legacy_bytes == rebuild_bytes:
        return []
    return [{
        "eval_id": eval_id,
        "field_path": "$.output.bytes",
        "legacy": f"<{len(legacy_bytes)} bytes>",
        "rebuild": f"<{len(rebuild_bytes)} bytes>",
    }]
