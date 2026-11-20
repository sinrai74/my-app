"""LZH の解凍（既存の lzh_extract を変更せずに使い、CRC-16 の照合を加える）。

既存の lzh_extract は CRC を照合しない。ここでヘッダーから CRC を読み、解凍結果と照らす。
"""
from __future__ import annotations

import hashlib
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import lzh_extract  # noqa: E402  既存モジュール（変更しない）


class ArchiveError(Exception):
    pass


@dataclass(frozen=True)
class Member:
    name: str
    method: str
    data: bytes
    crc_expected: int
    crc_actual: int


def crc16(data: bytes) -> int:
    """LHA の CRC-16（多項式 0xA001、初期値 0）。"""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _header_crc(data: bytes) -> tuple[str, int]:
    """先頭のエントリのヘッダーから、方式と CRC を読む（レベル 0・1・2）。"""
    if len(data) < 27:
        raise ArchiveError("ファイルが短すぎます")
    method = data[2:7].decode("ascii", errors="replace")
    level = data[20]
    if level in (0, 1):
        fn_len = data[21]
        return method, struct.unpack_from("<H", data, 22 + fn_len)[0]
    if level == 2:
        return method, struct.unpack_from("<H", data, 21)[0]
    raise ArchiveError(f"未対応のヘッダーレベル {level}")


def extract_single(data: bytes) -> Member:
    """メンバーがちょうど1つのアーカイブを解凍し、CRC を照合する。合わなければ例外。"""
    method, crc_expected = _header_crc(data)
    if method != "-lh5-":
        raise ArchiveError(f"想定外の圧縮方式 {method}")
    try:
        files = lzh_extract.extract_all(data)
    except lzh_extract.BadLzhFile as e:
        raise ArchiveError(f"解凍できません: {e}") from e
    if len(files) != 1:
        raise ArchiveError(f"メンバーが1つではありません: {sorted(files)}")
    name, body = next(iter(files.items()))
    actual = crc16(body)
    if actual != crc_expected:
        raise ArchiveError(f"CRC が合いません（ヘッダー {crc_expected:04x}、実際 {actual:04x}）")
    return Member(name, method, body, crc_expected, actual)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
