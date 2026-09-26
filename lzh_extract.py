#!/usr/bin/env python3
"""
lzh_extract.py  ── 純粋Python実装 LZH(LHa)アーカイブ展開モジュール

【背景】boatrace.jp が配布するKファイルのアーカイブ形式はLZH（LHa 2.x,
lh5圧縮）である。PyPI上の既存ライブラリ（lhafile・pylhasa等）はいずれも
C拡張（lzhlib）を含み、Windows環境でのインストールには
Microsoft Visual C++ Build Tools が必要になる。この依存を避けるため、
lh5/lh6/lh7の展開アルゴリズムを純粋Pythonで実装した。

【実装の正確性】アルゴリズムは既存の lhafile パッケージ（BSDライセンス、
Hidekazu Ohnishi氏作、PyPI公開）が使用しているCライブラリ lzhlib.c の
ロジック（ビットストリーム処理・canonical huffman構築・LZSS展開）を
そのまま参照してPythonに移植した。実際のKファイル（k260707.lzh、
lh5圧縮・34,991バイト）で展開結果が元のK260707.TXT（169,038バイト）と
MD5完全一致することを確認済み。

【対応形式】-lh5-, -lh6-, -lh7- （LZSS+動的ハフマン符号）
           -lh0- （無圧縮）
未対応: -lh1- 〜 -lh4-, -lhd-（ディレクトリ）等の旧式・特殊形式。
Kファイルの配布形式が -lh5- である限り問題にならない想定。
"""

from __future__ import annotations

import struct
from datetime import datetime
from typing import Optional


class BadLzhFile(Exception):
    """LZHファイルとして解釈できない、または未対応の形式の場合に送出する。"""
    pass


# ════════════════════════════════════════════════════════════
# 圧縮方式ごとのパラメータ（lzhlib.c と同一の値）
# ════════════════════════════════════════════════════════════
_COMPRESS_PARAMS = {
    b"-lh5-": {"dic_size": 8192,  "dic_bit": 13, "dispos_bit": 14, "dis_bit": 4},
    b"-lh6-": {"dic_size": 32768, "dic_bit": 15, "dispos_bit": 16, "dis_bit": 5},
    b"-lh7-": {"dic_size": 65536, "dic_bit": 16, "dispos_bit": 17, "dis_bit": 5},
}


# ════════════════════════════════════════════════════════════
# ビットストリームリーダー（MSBファースト）
# ════════════════════════════════════════════════════════════

class _BitReader:
    """
    32bit相当のキャッシュにバイト列を読み込み、MSB側からnビットずつ
    取り出す。lzhlib.c の bit_stream_reader を模した実装。
    データ終端を超えて読もうとした場合は 0 で埋める（元実装と同じ挙動）。
    """

    def __init__(self, data: bytes):
        self._data = data
        self._byte_pos = 0
        self._cache = 0
        self._cache_bits = 0
        self._fill()

    def _fill(self) -> None:
        while self._cache_bits <= 24 and self._byte_pos < len(self._data):
            self._cache = (self._cache << 8) | self._data[self._byte_pos]
            self._byte_pos += 1
            self._cache_bits += 8

    def pre_fetch(self, n: int) -> int:
        """消費せずに上位nビットを覗き見る。"""
        if n <= 0:
            return 0
        if self._cache_bits < n:
            self._fill()
        if self._cache_bits < n:
            # データ終端付近: 不足分は0で埋める
            shift = n - self._cache_bits
            return (self._cache << shift) & ((1 << n) - 1)
        return (self._cache >> (self._cache_bits - n)) & ((1 << n) - 1)

    def fetch(self, n: int) -> int:
        """上位nビットを取り出して消費する。"""
        if n <= 0:
            return 0
        val = self.pre_fetch(n)
        consume = min(n, self._cache_bits)
        self._cache_bits -= consume
        if self._cache_bits > 0:
            self._cache &= (1 << self._cache_bits) - 1
        else:
            self._cache = 0
        return val


# ════════════════════════════════════════════════════════════
# Canonical Huffman デコーダー
# ════════════════════════════════════════════════════════════

class _HuffmanDecoder:
    """
    ビット長のリスト（各シンボルの符号長）から canonical huffman 符号を
    再構築し、高速ルックアップテーブルでデコードする。
    lzhlib.c の bit_length_table / bit_pattern_table / huffman_decoder を
    1クラスにまとめた実装。
    """

    def __init__(self, bit_lengths: list[int]):
        bit_max = max(bit_lengths) if bit_lengths else 0
        if bit_max == 0 or bit_max > 16 or not bit_lengths:
            raise BadLzhFile("ビット長テーブルが不正です")

        n = len(bit_lengths)
        freq_table = [0] * (bit_max + 1)
        for bl in bit_lengths:
            if bl != 0:
                freq_table[bl] += 1

        start_pattern = [0] * (bit_max + 1)
        weight = [0] * (bit_max + 1)
        ptn = 0
        w = 1 << (bit_max - 1)
        for i in range(1, bit_max + 1):
            start_pattern[i] = ptn
            weight[i] = w
            ptn += w * freq_table[i]
            w >>= 1
        if ptn > (1 << bit_max):
            raise BadLzhFile("Huffman符号テーブルの構築に失敗しました")

        pattern_table = [0] * n
        for i in range(n):
            bl = bit_lengths[i]
            if bl == 0:
                pattern_table[i] = 0
                continue
            ptn = start_pattern[bl]
            pattern_table[i] = ptn >> (bit_max - bl)
            start_pattern[bl] += weight[bl]

        # ルックアップテーブル本体: 各エントリは (bit_len << 11) | symbol
        table_size = 1 << bit_max
        blen_code = [0] * table_size
        for i in range(n):
            bl = bit_lengths[i]
            if bl == 0:
                continue
            ptn = pattern_table[i] << (bit_max - bl)
            blen_code[ptn] = (bl << 11) | i

        if bit_max == 1 and blen_code[1] == 0:
            blen_code[0] &= 0x1FF

        # 「穴埋め」: エントリが0のインデックスは直前の非ゼロ値で埋める
        # （canonical huffmanの性質上、これで正しいprefixマッチになる）
        last = blen_code[0]
        for i in range(1, table_size):
            if blen_code[i] == 0:
                blen_code[i] = last
            else:
                last = blen_code[i]

        self._blen_code = blen_code
        self._bit_max = bit_max

    def decode(self, reader: _BitReader) -> int:
        bits = reader.pre_fetch(self._bit_max)
        packed = self._blen_code[bits]
        bit_len = packed >> 11
        symbol = packed & 0x1FF
        reader.fetch(bit_len)
        return symbol


# ════════════════════════════════════════════════════════════
# lh5/lh6/lh7 展開本体
# ════════════════════════════════════════════════════════════

def _decode_unary7(reader: _BitReader) -> int:
    """3bit読み、7ならその後1が続く限り読み進める特殊unary符号。"""
    code = reader.fetch(3)
    if code == 7:
        while reader.fetch(1) == 1:
            code += 1
    return code


def _decode_bitlen19(reader: _BitReader) -> list[int]:
    """19要素の「ビット長のビット長」テーブルを復元する。"""
    table = [0] * 19
    size = reader.fetch(5)
    if size > 19:
        raise BadLzhFile("ビット長テーブルのサイズが不正です")
    if size == 0:
        leaf = reader.fetch(5)
        table[leaf] = 1
        return table

    i = 0
    while i < size:
        c = _decode_unary7(reader)
        table[i] = c
        i += 1
        if i == 3:
            nmax = reader.fetch(2)
            while nmax > 0:
                table[i] = 0
                i += 1
                nmax -= 1
    return table


def _decode_bitlen510(reader: _BitReader, bitlen_decoder: _HuffmanDecoder) -> list[int]:
    """510要素のリテラル/長さ符号のビット長テーブルを復元する。"""
    table = [0] * 510
    n = reader.fetch(9)
    if n == 0:
        leaf = reader.fetch(9)
        table[leaf] = 1
        return table

    i = 0
    while i < n:
        code = bitlen_decoder.decode(reader)
        if code == 0:
            table[i] = 0
            i += 1
        elif code == 1:
            c = reader.fetch(4) + 3
            i += c
        elif code == 2:
            c = reader.fetch(9) + 20
            i += c
        else:
            table[i] = code - 2
            i += 1
    return table


def _decode_bitlen_distance(reader: _BitReader, dispos_bit: int, dis_bit: int) -> list[int]:
    """距離符号のビット長テーブルを復元する。"""
    table = [0] * (dispos_bit + 1)
    size = reader.fetch(dis_bit)
    if size == 0:
        leaf = reader.fetch(dis_bit)
        table[leaf] = 1
        return table

    i = 0
    while i < size:
        table[i] = _decode_unary7(reader)
        i += 1
    return table


def _decode_lzss_block(compressed: bytes, out_size: int, dic_size: int,
                        dispos_bit: int, dis_bit: int) -> bytes:
    """
    LZSS+動的ハフマン符号（lh5/lh6/lh7共通）の展開本体。
    lzhlib.c の LZHDecodeSession_do_next を1関数にまとめた実装。
    """
    reader = _BitReader(compressed)
    out = bytearray()
    dic_buf = bytearray(dic_size)
    dic_pos = 0
    dic_mask = dic_size - 1
    block_size = 0
    literal_decoder: Optional[_HuffmanDecoder] = None
    distance_decoder: Optional[_HuffmanDecoder] = None

    while len(out) < out_size:
        if block_size <= 0:
            block_size = reader.fetch(16)
            if block_size == 0 and len(out) > 0:
                # ブロック終端かつ既にデータがあれば終了とみなす
                # （元実装ではfetchが-1=EOFのときのみ終了するが、
                #  Python版はEOF後0埋めのため出力サイズで判定する）
                break

            bitlen19 = _decode_bitlen19(reader)
            bitlen_decoder = _HuffmanDecoder(bitlen19)

            bitlen510 = _decode_bitlen510(reader, bitlen_decoder)
            literal_decoder = _HuffmanDecoder(bitlen510)

            bitlen_dist = _decode_bitlen_distance(reader, dispos_bit, dis_bit)
            distance_decoder = _HuffmanDecoder(bitlen_dist)

        code = literal_decoder.decode(reader)
        block_size -= 1

        if code < 256:
            dic_buf[dic_pos] = code
            out.append(code)
            dic_pos = (dic_pos + 1) & dic_mask
            continue

        mlen = code - 256 + 3
        bitl = distance_decoder.decode(reader)
        if bitl == 0:
            dist = 1
        else:
            dist = reader.fetch(bitl - 1)
            dist += (1 << (bitl - 1))
            dist += 1

        src_pos = (dic_pos - dist) & dic_mask
        for _ in range(mlen):
            b = dic_buf[src_pos]
            dic_buf[dic_pos] = b
            out.append(b)
            dic_pos = (dic_pos + 1) & dic_mask
            src_pos = (src_pos + 1) & dic_mask

    return bytes(out[:out_size])


def _decode_lh0(compressed: bytes, out_size: int) -> bytes:
    """無圧縮形式。"""
    return compressed[:out_size]


def decode_compressed(compress_type: bytes, compressed: bytes, out_size: int) -> bytes:
    """
    compress_type（例: b"-lh5-"）に応じて展開する。
    """
    if compress_type == b"-lh0-":
        return _decode_lh0(compressed, out_size)
    params = _COMPRESS_PARAMS.get(compress_type)
    if params is None:
        raise BadLzhFile(f"未対応の圧縮方式です: {compress_type!r}")
    return _decode_lzss_block(
        compressed, out_size,
        dic_size=params["dic_size"],
        dispos_bit=params["dispos_bit"],
        dis_bit=params["dis_bit"],
    )


# ════════════════════════════════════════════════════════════
# LHAヘッダー解析（level 0 / 1 / 2 対応）
# ════════════════════════════════════════════════════════════

class LhaEntry:
    """LZHアーカイブ内の1エントリ（ファイル）の情報。"""
    def __init__(self, filename: str, compress_type: bytes,
                 compress_size: int, file_size: int, data: bytes):
        self.filename = filename
        self.compress_type = compress_type
        self.compress_size = compress_size
        self.file_size = file_size
        self.data = data  # 展開済みバイト列


def _parse_one_header(data: bytes, pos: int) -> "tuple[Optional[LhaEntry], int]":
    """
    1エントリ分のヘッダー＋データを読み、(LhaEntry, 次のヘッダー位置) を返す。
    ファイル終端（ヘッダーサイズ0）に達したら (None, pos) を返す。
    """
    if pos >= len(data):
        return None, pos
    header_size = data[pos]
    if header_size == 0:
        return None, pos

    if pos + 2 > len(data):
        raise BadLzhFile("ヘッダーが壊れています（サイズ不足）")

    # header_size(1) + checksum(1) + signature(5) までで7バイト
    if pos + 22 > len(data):
        raise BadLzhFile("ヘッダーが壊れています（本体不足）")

    _, _, signature = struct.unpack_from("<BB5s", data, pos)
    os_level_byte = data[pos + 20]

    if os_level_byte not in (0, 1, 2):
        raise BadLzhFile(f"未対応のヘッダーレベルです: {os_level_byte}")

    if os_level_byte in (0, 1):
        (hsize, checksum, sig, skip_size, file_size, modify_time,
         reserved, os_level, filename_length) = struct.unpack_from("<BB5sII4sBBB", data, pos)
        cursor = pos + 22
        filename_bytes = data[cursor:cursor + filename_length]
        cursor += filename_length
        # CRC (2 bytes) - level0/1はここに配置される
        cursor += 2
        ext_header_size = 0
        if os_level == 1:
            # level1拡張ヘッダーサイズはheader_sizeとの差分から算出する必要があるが
            # Kファイル（level0想定）では通常ここに到達しない。安全のため
            # 最小限のみ対応する。
            extra_size = hsize - (5 + 4 + 4 + 2 + 2 + 1 + 1 + 1 + filename_length + 2 + 1 + 2)
            if extra_size > 0:
                cursor += 1  # os_identifier
                cursor += extra_size
                ext_header_size = struct.unpack_from("<H", data, cursor)[0]
                cursor += 2
        compress_size = skip_size
        file_offset = cursor
    else:  # os_level_byte == 2
        (all_header_size, sig, compress_size, file_size, modify_time,
         reserved, os_level, crc, os_identifier, ext_header_size) = \
            struct.unpack_from("<H5sIIIBBHBH", data, pos)
        cursor = pos + 26
        filename_bytes = b""
        file_offset = cursor

    # 拡張ヘッダーの読み飛ばし（ファイル名拡張ヘッダー等）
    while ext_header_size != 0:
        if file_offset + ext_header_size > len(data):
            raise BadLzhFile("拡張ヘッダーが壊れています")
        ext_type = data[file_offset]
        body = data[file_offset + 1: file_offset + ext_header_size - 2]
        next_size = struct.unpack_from("<H", data, file_offset + ext_header_size - 2)[0]
        if ext_type == 0x01:  # ファイル名拡張ヘッダー
            filename_bytes = body
        file_offset += ext_header_size
        ext_header_size = next_size

    if file_offset + compress_size > len(data):
        raise BadLzhFile("圧縮データが不足しています")

    compressed = data[file_offset: file_offset + compress_size]
    filename = filename_bytes.decode("shift_jis", errors="replace")

    decoded = decode_compressed(signature, compressed, file_size)

    entry = LhaEntry(filename, signature, compress_size, file_size, decoded)
    next_pos = file_offset + compress_size
    return entry, next_pos


def extract_all(data: bytes) -> "dict[str, bytes]":
    """
    LZHアーカイブ（バイト列）内の全エントリを展開し、
    {ファイル名: 展開済みバイト列} の辞書を返す。
    """
    result: dict[str, bytes] = {}
    pos = 0
    while pos < len(data):
        entry, next_pos = _parse_one_header(data, pos)
        if entry is None:
            break
        result[entry.filename] = entry.data
        if next_pos <= pos:
            break
        pos = next_pos
    return result
