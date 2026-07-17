"""
Repository層（storage/repositories/）の単体テスト。

設計書 v1.1.6 ⑥、Step2-6承認スコープ・追加確認事項に対応。
tempfileによる実I/Oテスト。標準ライブラリ unittest のみを使用する。
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from models.evaluation import BuyDecision, FeatureSet, Prediction, RaceEvaluation
from models.race import Weather
from models.record import HitRecord, RaceResult
from storage.exceptions import ParseError, StorageError
from storage.mappers.hit_record_mapper import ALL_COLUMNS, LEGACY_COLUMNS
from storage.repositories.evaluation_repository import EvaluationRepository
from storage.repositories.hit_record_repository import HitRecordCsvRepository


def _evaluation(eval_id: str = "20260713_12_05") -> RaceEvaluation:
    date, venue, race = eval_id.split("_")
    return RaceEvaluation(
        eval_id=eval_id,
        race_date=date,
        venue_num=int(venue),
        venue_name="住之江",
        race_number=int(race),
        is_night=False,
        engine_name="ver4",
        engine_version="4.2.0",
        feature_schema_version=1,
        model_version="v20260701",
        evaluated_at="2026-07-13T06:05:00+09:00",
        danger_score=78.5,
        danger_breakdown={"win_rate_low": 28},
        upset_score=42.0,
        upset_reasons=("1号艇平均ST遅め",),
        rank_index={"1": 0.80},
        featured_boats={"featured": [1, 4]},
        win_probs={1: 0.42},
        race_type="本命",
        match_index=44.1,
        features=FeatureSet(
            eval_id=eval_id,
            feature_schema_version=1,
            built_at="2026-07-13T06:00:00+09:00",
            boat_features={1: {"win_rate": 6.50}},
            race_features={"venue_factor": 1.02},
        ),
    )


def _hit_record(eval_id: str = "20260713_12_05") -> HitRecord:
    return HitRecord(
        eval_id=eval_id,
        evaluation=_evaluation(eval_id),
        prediction=Prediction(
            eval_id=eval_id,
            pred_combo="1-2-3",
            pred_prob=0.185,
            pred_ev=1.42,
            pred_odds=7.7,
            confidence=0.72,
            why_bet="1号艇信頼度高",
            patterns=({"combo": "1-2-3"},),
        ),
        buy_decision=BuyDecision(
            eval_id=eval_id,
            purchased=True,
            buyscore=68.5,
            investment_type="通常",
            n_bets=3,
            cost=900,
            kelly_fraction=0.05,
            config_version="12",
        ),
        result=RaceResult(
            eval_id=eval_id, result_combo="1-3-2", payout=1970, hit=True, profit=1070
        ),
        weather=Weather(wind_speed_mps=3.0, wind_direction="横", wave_height_cm=3),
    )


class TestEvaluationRepository(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "evaluations" / "20260713.jsonl"

    def test_missing_file_loads_empty(self) -> None:
        """存在しないファイル -> 空リスト（新しい日付の正常状態）。"""
        repo = EvaluationRepository(self.path)
        self.assertEqual(repo.load_all(), [])

    def test_empty_file_loads_empty(self) -> None:
        self.path.parent.mkdir(parents=True)
        self.path.write_text("", encoding="utf-8")
        self.assertEqual(EvaluationRepository(self.path).load_all(), [])

    def test_append_and_load_roundtrip(self) -> None:
        repo = EvaluationRepository(self.path)
        e1 = _evaluation("20260713_12_05")
        e2 = _evaluation("20260713_12_06")
        repo.append(e1)
        repo.append(e2)
        loaded = repo.load_all()
        self.assertEqual(loaded, [e1, e2])

    def test_duplicate_eval_id_raises_storage_error(self) -> None:
        """追加確認事項: 同一eval_idの追記は一意性違反としてStorageError。"""
        repo = EvaluationRepository(self.path)
        repo.append(_evaluation("20260713_12_05"))
        with self.assertRaises(StorageError):
            repo.append(_evaluation("20260713_12_05"))

    def test_broken_json_line_raises_parse_error_with_line_no(self) -> None:
        self.path.parent.mkdir(parents=True)
        self.path.write_text('{"broken": \n', encoding="utf-8")
        with self.assertRaises(ParseError) as ctx:
            EvaluationRepository(self.path).load_all()
        self.assertIn("line 1", str(ctx.exception))

    def test_find_by_eval_id(self) -> None:
        """案P: source_eval_id -> RaceEvaluation の解決地点。"""
        repo = EvaluationRepository(self.path)
        target = _evaluation("20260713_12_06")
        repo.append(_evaluation("20260713_12_05"))
        repo.append(target)
        self.assertEqual(repo.find_by_eval_id("20260713_12_06"), target)
        self.assertIsNone(repo.find_by_eval_id("20990101_01_01"))


class TestHitRecordCsvRepository(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "hit_record.csv"

    def test_missing_file_reads_empty(self) -> None:
        self.assertEqual(HitRecordCsvRepository(self.path).read_all(), [])

    def test_write_read_rewrite_reversibility(self) -> None:
        """追加確認事項: 読み込み→書き込み→再読み込みで可逆であること。"""
        repo = HitRecordCsvRepository(self.path)
        records = [_hit_record("20260713_12_05"), _hit_record("20260713_12_06")]
        repo.write_all(records)
        first = repo.read_all()
        self.assertEqual(first, records)
        repo.write_all(first)
        second = repo.read_all()
        self.assertEqual(second, records)

    def test_append_to_new_and_existing_file(self) -> None:
        repo = HitRecordCsvRepository(self.path)
        repo.append(_hit_record("20260713_12_05"))
        repo.append(_hit_record("20260713_12_06"))
        self.assertEqual(len(repo.read_all()), 2)

    def test_append_to_legacy_header_raises_storage_error(self) -> None:
        """レガシー44列ファイルへの追記は列破壊防止のためStorageError。"""
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(LEGACY_COLUMNS)
        with self.assertRaises(StorageError):
            HitRecordCsvRepository(self.path).append(_hit_record())

    def test_unexpected_header_raises_storage_error(self) -> None:
        """追加確認事項: ヘッダー破損（列欠落）を握りつぶさないこと。"""
        broken_header = list(LEGACY_COLUMNS[:-1])  # 1列欠落
        with open(self.path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(broken_header)
        with self.assertRaises(StorageError):
            HitRecordCsvRepository(self.path).read_all()

    def test_legacy_format_file_is_readable(self) -> None:
        """レガシー44列形式のファイルを読み込めること（新旧経路の分離）。"""
        golden = Path("tests/regression/golden/hit_record_golden_100.csv")
        repo = HitRecordCsvRepository(golden)
        records = repo.read_all()
        self.assertEqual(len(records), 100)
        self.assertEqual(records[0].evaluation.engine_name, "legacy")

    def test_legacy_to_extended_conversion_preserves_models(self) -> None:
        """Golden回帰のRepository版: レガシー読込→新形式書込→再読込で
        モデルが同値であること（Repository経由の可逆性）。
        """
        golden = Path("tests/regression/golden/hit_record_golden_100.csv")
        originals = HitRecordCsvRepository(golden).read_all()
        repo = HitRecordCsvRepository(self.path)
        repo.write_all(originals)
        # 新形式ヘッダーで書き出されていること
        with open(self.path, encoding="utf-8", newline="") as f:
            header = tuple(next(csv.reader(f)))
        self.assertEqual(header, ALL_COLUMNS)
        reloaded = repo.read_all()
        self.assertEqual(reloaded, originals)


if __name__ == "__main__":
    unittest.main()
