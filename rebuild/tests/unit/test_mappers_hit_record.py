"""
HitRecordCsvMapper（storage/mappers/hit_record_mapper.py）の単体テスト。

設計書 v1.1.5 ③3.8・⑩Mapper層ルール・⑫、Step2計画書§1.4・§4に対応。
標準ライブラリ unittest のみを使用する。
"""

from __future__ import annotations

import unittest

from models.evaluation import BuyDecision, FeatureSet, Prediction, RaceEvaluation
from models.race import Weather
from models.record import HitRecord, RaceResult
from storage.exceptions import ParseError
from storage.mappers.hit_record_mapper import (
    ALL_COLUMNS,
    EXTENSION_COLUMNS,
    FEAT_COLUMN_TO_FEATURE_KEY,
    LEGACY_COLUMNS,
    WATER_TYPE_CODE_TO_LABEL,
    WATER_TYPE_LABEL_TO_CODE,
    HitRecordCsvMapper,
)
from tests.helpers import assert_row_equal


def _feature_set() -> FeatureSet:
    return FeatureSet(
        eval_id="20260713_12_05",
        feature_schema_version=1,
        built_at="2026-07-13T06:00:00+09:00",
        boat_features={
            1: {"win_rate": 6.50, "motor_rate2": 38.5, "avg_st": 0.16},
            2: {"win_rate": 5.20},
        },
        race_features={"venue_factor": 1.02, "venue_water_type_code": 2.0},
    )


def _full_record() -> HitRecord:
    """全フィールドが埋まった新世代のHitRecord（拡張列往復の検証用）。"""
    evaluation = RaceEvaluation(
        eval_id="20260713_12_05",
        race_date="20260713",
        venue_num=12,
        venue_name="住之江",
        race_number=5,
        is_night=False,
        engine_name="ver4",
        engine_version="4.2.0",
        feature_schema_version=1,
        model_version="v20260701",
        evaluated_at="2026-07-13T06:05:00+09:00",
        danger_score=78.5,
        danger_breakdown={"win_rate_low": 28, "motor_bad": 0.0},
        upset_score=42.0,
        upset_reasons=("1号艇平均ST遅め",),
        rank_index={"1": 0.80},
        featured_boats={"featured": [1, 4]},
        win_probs={1: 0.42, 2: 0.18},
        race_type="本命",
        match_index=44.1,
        features=_feature_set(),
    )
    prediction = Prediction(
        eval_id="20260713_12_05",
        pred_combo="1-2-3",
        pred_prob=0.185,
        pred_ev=1.42,
        pred_odds=7.7,
        confidence=0.72,
        why_bet="1号艇信頼度高",
        patterns=({"combo": "1-2-3", "prob": 0.185},),
    )
    buy = BuyDecision(
        eval_id="20260713_12_05",
        purchased=True,
        buyscore=68.5,
        investment_type="通常",
        n_bets=3,
        cost=900,
        kelly_fraction=0.05,
        config_version="12",
        skip_reason=None,
    )
    result = RaceResult(
        eval_id="20260713_12_05",
        result_combo="1-3-2",
        payout=1970,
        hit=True,
        profit=1070,
    )
    weather = Weather(wind_speed_mps=3.0, wind_direction="横", wave_height_cm=3)
    return HitRecord(
        eval_id="20260713_12_05",
        evaluation=evaluation,
        prediction=prediction,
        buy_decision=buy,
        result=result,
        weather=weather,
    )


def _legacy_row() -> dict:
    """実データ（20260704桐生1R）を模したレガシー44列行。"""
    return {
        "date": "20260704",
        "venue": "桐生",
        "venue_num": "1",
        "race": "1",
        "night": "0",
        "race_type": "1殴り荒れ型",
        "why_bet": "1号艇平均ST遅め0.17|超荒れ(9.5)",
        "confidence": "0.8007",
        "pred_combo": "6-1-3",
        "pred_prob": "0.04591",
        "pred_ev": "3.03",
        "pred_odds": "66.0",
        "upset_score": "9.5",
        "wind_speed": "0.0",
        "wind_dir": "",
        "wave": "0",
        "result_combo": "3-2-4",
        "payout": "38320",
        "hit": "0",
        "profit": "0",
        "n_bets": "1",
        "cost": "0",
        "purchased": "0",
        "buyscore": "",
        "match_index": "0",
        "skip_reason": "retroactive_fix:amount_zero_ghost_purchase",
        "model_version": "asahi-v1.0-phase1",
        "feat_win_rate": "3.71",
        "feat_motor": "28.81",
        "feat_avg_st": "0.17",
        "feat_racer_class": "",
        "feat_course_st_1c": "0.17",
        "feat_course_rank_1c": "3.3",
        "feat_danger_breakdown": '{"win_rate_low": 28, "motor_bad": 0.0}',
        "danger_score_v3": "",
        "rank_index_json": "",
        "featured_boats_json": "",
        "venue_water_type": "",
        "venue_factor": "",
        "ability_trend": "",
        "course_f_rate_1c": "",
        "course_l_rate_1c": "",
        "course_rentai2_1c": "",
        "course_sample_confidence": "",
    }


class TestConstants(unittest.TestCase):
    def test_legacy_columns_count_is_44(self) -> None:
        self.assertEqual(len(LEGACY_COLUMNS), 44)

    def test_all_columns_are_legacy_plus_extension(self) -> None:
        self.assertEqual(ALL_COLUMNS, LEGACY_COLUMNS + EXTENSION_COLUMNS)
        self.assertEqual(len(set(ALL_COLUMNS)), len(ALL_COLUMNS))

    def test_feat_mapping_is_bijective(self) -> None:
        """追加ルール2: feat_*列⇔FeatureSetキー対応が全単射であること。"""
        cols = list(FEAT_COLUMN_TO_FEATURE_KEY.keys())
        keys = list(FEAT_COLUMN_TO_FEATURE_KEY.values())
        self.assertEqual(len(cols), len(set(cols)))
        self.assertEqual(len(keys), len(set(keys)))
        for col in cols:
            self.assertIn(col, LEGACY_COLUMNS)

    def test_water_type_table_is_bijective(self) -> None:
        self.assertEqual(
            len(WATER_TYPE_LABEL_TO_CODE), len(WATER_TYPE_CODE_TO_LABEL)
        )
        for label, code in WATER_TYPE_LABEL_TO_CODE.items():
            self.assertEqual(WATER_TYPE_CODE_TO_LABEL[code], label)


class TestFullModelRoundtrip(unittest.TestCase):
    def test_model_to_row_to_model_identity(self) -> None:
        """不変条件: from_row(to_row(m)) == m（全フィールド埋め・完全一致）。"""
        record = _full_record()
        restored = HitRecordCsvMapper.from_row(HitRecordCsvMapper.to_row(record))
        self.assertEqual(record, restored)

    def test_to_row_emits_all_columns(self) -> None:
        row = HitRecordCsvMapper.to_row(_full_record())
        self.assertEqual(set(row.keys()), set(ALL_COLUMNS))

    def test_roundtrip_with_none_result_and_none_weather(self) -> None:
        """結果未判明・気象未取得のHitRecordも完全往復すること。"""
        base = _full_record()
        record = HitRecord(
            eval_id=base.eval_id,
            evaluation=base.evaluation,
            prediction=base.prediction,
            buy_decision=base.buy_decision,
            result=None,
            weather=None,
        )
        restored = HitRecordCsvMapper.from_row(HitRecordCsvMapper.to_row(record))
        self.assertEqual(record, restored)
        self.assertIsNone(restored.result)
        self.assertIsNone(restored.weather)


class TestLegacyRowRoundtrip(unittest.TestCase):
    def test_from_row_restores_legacy_row(self) -> None:
        record = HitRecordCsvMapper.from_row(_legacy_row())
        self.assertEqual(record.eval_id, "20260704_01_01")
        self.assertIsNone(record.evaluation.danger_score)
        self.assertIsNone(record.buy_decision.buyscore)
        self.assertEqual(record.evaluation.engine_name, "legacy")
        self.assertEqual(record.evaluation.upset_reasons, ())
        self.assertEqual(record.prediction.patterns, ())
        self.assertEqual(record.evaluation.match_index, 0.0)
        # v1.1.6: kelly_fractionはレガシー行でNone（0.0注入しない）
        self.assertIsNone(record.buy_decision.kelly_fraction)
        # wind_dirのみ空欄 -> Weatherは構築され、wind_direction=""
        self.assertIsNotNone(record.weather)
        self.assertEqual(record.weather.wind_direction, "")
        # 空欄feat列はmissing_keysへ（racer_class_score等）
        self.assertIn("racer_class_score", record.evaluation.features.missing_keys)
        self.assertEqual(
            record.evaluation.features.boat_features[1]["motor_rate2"], 28.81
        )

    def test_legacy_row_to_model_to_row_preserves_legacy_columns(self) -> None:
        """追加ルール3: レガシー行の往復で44互換列が保全されること（1e-6許容）。"""
        original = _legacy_row()
        record = HitRecordCsvMapper.from_row(original)
        row = HitRecordCsvMapper.to_row(record)
        # 44互換列のみ抽出し、数値化可能セルは数値で比較（書式差は不一致にしない）
        legacy_out = {c: _normalize(row[c]) for c in LEGACY_COLUMNS if c != "feat_danger_breakdown"}
        legacy_in = {c: _normalize(original[c]) for c in LEGACY_COLUMNS if c != "feat_danger_breakdown"}
        assert_row_equal(self, legacy_in, legacy_out)
        # JSON列は文字列書式でなく構造で比較する
        import json

        self.assertEqual(
            json.loads(original["feat_danger_breakdown"]),
            json.loads(row["feat_danger_breakdown"]),
        )


def _normalize(value):
    """比較用正規化: 数値化可能なセルはfloatへ、それ以外は文字列のまま。"""
    if value is None or value == "":
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


class TestParseErrors(unittest.TestCase):
    def test_missing_legacy_column_raises(self) -> None:
        """追加ルール4: 44列の欠落（ヘッダーレベル）はParseError。"""
        row = _legacy_row()
        del row["upset_score"]
        with self.assertRaises(ParseError):
            HitRecordCsvMapper.from_row(row)

    def test_unknown_column_raises(self) -> None:
        """未知列の黙殺禁止（設計書⑩）: タイポ・列名変更を検知すること。"""
        row = _legacy_row()
        row["upset_socre_typo"] = "1.0"
        with self.assertRaises(ParseError):
            HitRecordCsvMapper.from_row(row)

    def test_required_cell_empty_raises(self) -> None:
        """必須セル（pred_prob等、Optional対象外）の空欄はParseError。"""
        row = _legacy_row()
        row["pred_prob"] = ""
        with self.assertRaises(ParseError):
            HitRecordCsvMapper.from_row(row)

    def test_broken_json_raises(self) -> None:
        row = _legacy_row()
        row["feat_danger_breakdown"] = '{"broken": '
        with self.assertRaises(ParseError):
            HitRecordCsvMapper.from_row(row)

    def test_partial_result_columns_raise(self) -> None:
        """結果4列の部分空欄は破損としてParseError。"""
        row = _legacy_row()
        row["payout"] = ""
        with self.assertRaises(ParseError):
            HitRecordCsvMapper.from_row(row)

    def test_partial_weather_raises(self) -> None:
        """wind_speed/waveの片側だけ空欄はParseError（wind_dirのみ空欄は許容）。"""
        row = _legacy_row()
        row["wind_speed"] = ""
        with self.assertRaises(ParseError):
            HitRecordCsvMapper.from_row(row)

    def test_unknown_water_type_label_raises(self) -> None:
        row = _legacy_row()
        row["venue_water_type"] = "未知の水面"
        with self.assertRaises(ParseError):
            HitRecordCsvMapper.from_row(row)

    def test_eval_id_mismatch_raises(self) -> None:
        """拡張列eval_idと導出値の不一致は破損としてParseError。"""
        row = _legacy_row()
        row["eval_id"] = "20990101_99_99"
        with self.assertRaises(ParseError):
            HitRecordCsvMapper.from_row(row)


class TestAllEmptyWeather(unittest.TestCase):
    def test_all_empty_weather_restores_none(self) -> None:
        row = _legacy_row()
        row["wind_speed"] = ""
        row["wind_dir"] = ""
        row["wave"] = ""
        record = HitRecordCsvMapper.from_row(row)
        self.assertIsNone(record.weather)


if __name__ == "__main__":
    unittest.main()
