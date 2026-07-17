"""
Golden Dataset 回帰テスト: HitRecordCsvMapperの実データ往復検証。

設計書 v1.1.6 ⑭（テスト戦略: hit_record CSV一致 44互換列で100%・浮動小数1e-6）、
Step2実装計画書 §5（ゴールデンデータ回帰戦略）に対応。

固定入力: tests/regression/golden/hit_record_golden_100.csv
- freeze-v1-baseline 時点の hit_record.csv から、承認済み選定ルール
  （20260704全90レース＋20260703をvenue_num昇順race_number昇順で先頭10レース）
  で抽出した100行。**このファイルの変更は禁止**（変更時は設計書⑳20.3の承認記録）。

検証内容: 各行について from_row -> to_row を実行し、44互換列を比較する。
- 数値化可能なセル: float比較（許容1e-6、tests/helpers.pyに集約）
- JSON列（feat_danger_breakdown / rank_index_json / featured_boats_json）:
  文字列書式ではなく構造（json.loads結果）で比較する（Step2-3承認方針。
  json.dumpsの空白書式差を偽陽性にしないため）
- その他: 文字列完全一致
"""

from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path

from storage.mappers.hit_record_mapper import LEGACY_COLUMNS, HitRecordCsvMapper
from tests.helpers import assert_row_equal

GOLDEN_CSV = Path(__file__).parent / "golden" / "hit_record_golden_100.csv"

# 固定時に記録したSHA-256（Step2-4承認時の基準値）。CSVが意図せず書き換えられた
# 場合に即検出する（Should-A対応）。正当な更新時は設計書⑳20.3の承認記録とともに
# この値も更新する。
EXPECTED_SHA256 = "b1a629e9ef2c2ae676010553b02196c9b9e8301d643a34b92b4f91b212aab11f"

# 構造比較の対象とするJSON列
JSON_COLUMNS: frozenset[str] = frozenset(
    {"feat_danger_breakdown", "rank_index_json", "featured_boats_json"}
)


def _load_golden_rows() -> list[dict]:
    with open(GOLDEN_CSV, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_cell(value) -> object:
    """比較用正規化: 空はそのまま、数値化可能セルはfloat、それ以外は文字列。"""
    if value is None or value == "":
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


class TestGoldenHitRecordRoundtrip(unittest.TestCase):
    def test_golden_file_hash_is_unchanged(self) -> None:
        """ゴールデンCSVのSHA-256が固定時の基準値と一致すること（改変検出）。"""
        digest = hashlib.sha256(GOLDEN_CSV.read_bytes()).hexdigest()
        self.assertEqual(
            digest,
            EXPECTED_SHA256,
            msg="golden CSV has been modified. 正当な更新なら承認記録とともに基準値を更新すること",
        )

    def test_golden_file_is_fixed_at_100_rows(self) -> None:
        """ゴールデンセットが100行・44列で固定されていること。"""
        rows = _load_golden_rows()
        self.assertEqual(len(rows), 100)
        self.assertEqual(tuple(rows[0].keys()), LEGACY_COLUMNS)

    def test_golden_selection_rule(self) -> None:
        """承認済み選定ルール（20260704=90行、20260703=10行）どおりであること。"""
        rows = _load_golden_rows()
        by_date: dict[str, int] = {}
        for r in rows:
            by_date[r["date"]] = by_date.get(r["date"], 0) + 1
        self.assertEqual(by_date, {"20260704": 90, "20260703": 10})

    def test_all_100_rows_roundtrip_preserves_legacy_columns(self) -> None:
        """⑭一致基準: 全100行の from_row -> to_row で44互換列が100%保全されること。

        1行でも不一致ならテスト失敗（失敗行のeval_id・列名をメッセージに含める）。
        """
        rows = _load_golden_rows()
        for index, original in enumerate(rows):
            with self.subTest(row=index, race=f"{original['date']}_{original['venue_num']}_{original['race']}"):
                record = HitRecordCsvMapper.from_row(original)
                output = HitRecordCsvMapper.to_row(record)

                expected: dict = {}
                actual: dict = {}
                for col in LEGACY_COLUMNS:
                    if col in JSON_COLUMNS:
                        expected[col] = _parse_json_or_empty(original[col])
                        actual[col] = _parse_json_or_empty(output[col])
                    else:
                        expected[col] = _normalize_cell(original[col])
                        actual[col] = _normalize_cell(output[col])
                assert_row_equal(self, expected, actual)

    def test_roundtrip_twice_is_stable(self) -> None:
        """二重往復（row->model->row->model->row）でも出力が安定していること。"""
        rows = _load_golden_rows()
        sample = rows[0]
        first = HitRecordCsvMapper.to_row(HitRecordCsvMapper.from_row(sample))
        second = HitRecordCsvMapper.to_row(HitRecordCsvMapper.from_row(first))
        self.assertEqual(first, second)


def _parse_json_or_empty(text) -> object:
    if text is None or text == "":
        return ""
    return json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
