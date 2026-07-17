"""
IdempotencyStore / CacheStore の単体テスト。

設計書 v1.1.8 ⑥、Step3計画書 §2・§6、Step3-4指示（責務境界）に対応。
tempfileによる実I/Oテスト。標準ライブラリ unittest のみを使用する。
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from storage.cache import CacheStore
from storage.exceptions import ParseError
from storage.idempotency import IdempotencyStore


class TestIdempotencyStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "idempotency.jsonl"

    def test_unrecorded_key_returns_false(self) -> None:
        store = IdempotencyStore(self.path)
        self.assertFalse(store.is_recorded("arashi_mail", "20260714:eid"))

    def test_record_then_is_recorded(self) -> None:
        store = IdempotencyStore(self.path)
        store.record("arashi_mail", "20260714:eid")
        self.assertTrue(store.is_recorded("arashi_mail", "20260714:eid"))

    def test_same_channel_different_key_independent(self) -> None:
        store = IdempotencyStore(self.path)
        store.record("arashi_mail", "k1")
        self.assertFalse(store.is_recorded("arashi_mail", "k2"))

    def test_same_key_different_channel_independent(self) -> None:
        """(channel, message_key)の組で独立に管理されること。"""
        store = IdempotencyStore(self.path)
        store.record("arashi_mail", "k1")
        self.assertFalse(store.is_recorded("ranking_mail", "k1"))
        self.assertTrue(store.is_recorded("arashi_mail", "k1"))

    def test_record_is_idempotent_no_duplicate_lines(self) -> None:
        """同一キーの二重recordで行が増えないこと（append-only・重複回避）。"""
        store = IdempotencyStore(self.path)
        store.record("c", "k")
        store.record("c", "k")
        lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln]
        self.assertEqual(len(lines), 1)

    def test_broken_json_line_raises_parse_error(self) -> None:
        self.path.write_text('{"channel": \n', encoding="utf-8")
        with self.assertRaises(ParseError):
            IdempotencyStore(self.path).is_recorded("c", "k")

    def test_missing_field_raises_parse_error(self) -> None:
        self.path.write_text(json.dumps({"channel": "c"}) + "\n", encoding="utf-8")
        with self.assertRaises(ParseError):
            IdempotencyStore(self.path).is_recorded("c", "k")

    def test_persists_across_instances(self) -> None:
        """別インスタンスからも記録が読めること（ファイル永続化）。"""
        IdempotencyStore(self.path).record("c", "k")
        self.assertTrue(IdempotencyStore(self.path).is_recorded("c", "k"))


class TestCacheStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "cache.json"

    def test_missing_file_returns_none(self) -> None:
        self.assertIsNone(CacheStore(self.path).load_if_valid("20260714"))

    def test_save_and_load_same_day(self) -> None:
        store = CacheStore(self.path)
        store.save({"odds": {"1-2-3": 12.5}}, cached_date="20260714")
        loaded = store.load_if_valid("20260714")
        self.assertEqual(loaded, {"odds": {"1-2-3": 12.5}})

    def test_different_day_returns_none(self) -> None:
        """当日以外のキャッシュはNone（破棄扱い）。再生成はしない。"""
        store = CacheStore(self.path)
        store.save({"x": 1}, cached_date="20260713")
        self.assertIsNone(store.load_if_valid("20260714"))

    def test_schema_version_mismatch_returns_none(self) -> None:
        CacheStore(self.path, schema_version=1).save({"x": 1}, cached_date="20260714")
        self.assertIsNone(
            CacheStore(self.path, schema_version=2).load_if_valid("20260714")
        )

    def test_broken_json_raises_parse_error(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ParseError):
            CacheStore(self.path).load_if_valid("20260714")

    def test_missing_key_raises_parse_error(self) -> None:
        self.path.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")
        with self.assertRaises(ParseError):
            CacheStore(self.path).load_if_valid("20260714")

    def test_save_overwrites(self) -> None:
        store = CacheStore(self.path)
        store.save({"v": 1}, cached_date="20260714")
        store.save({"v": 2}, cached_date="20260714")
        self.assertEqual(store.load_if_valid("20260714"), {"v": 2})


if __name__ == "__main__":
    unittest.main()
