"""
HitRecordCsvMapper: HitRecord（集約モデル） ⇔ hit_record.csv行 の変換。

設計書 Phase0.5 v1.1.5 ③3.8、⑩Mapper層ルール、⑫（ParseError）、
Step2実装計画書 §1.4 に基づく。

HitRecordCsvMapper Design Rules（設計書⑩）:
- This mapper is the only mapper allowed to aggregate multiple domain models.
- No business logic is allowed.
- Missing required columns raise ParseError.
- Unknown columns must not be silently ignored unless explicitly designated
  as forward-compatible extension columns.

Additional Rules（Step2-3承認）:
1. LEGACY_COLUMNS は実CSVヘッダーから機械転記した（手作業での再構成禁止）。
2. feat_*列⇔FeatureSetキーの対応は全単射（FEAT_COLUMN_TO_FEATURE_KEY）。
3. 往復検証必須（レガシー44列を1e-6許容で保全）。
4. 必須「列」の欠落（ヘッダーレベル）はParseError。空欄セルはNone受理
   （v1.1.4正式化: レガシーデータの未記録表現）。

LEGACY_DEFAULTS の適用は拡張メタデータのみ。評価結果（スコア・指標）への
既定値注入は禁止（v1.1.4）。コレクション型（upset_reasons, patterns）の
未記録は空タプル()で表現する（v1.1.5）。
"""

from __future__ import annotations

import json
from typing import Any, Optional

from models.evaluation import BuyDecision, FeatureSet, Prediction, RaceEvaluation
from models.race import Weather
from models.record import HitRecord, RaceResult
from storage.exceptions import ParseError
from storage.mappers.row_types import (
    Row,
    format_bool01,
    format_optional,
    parse_bool01,
    parse_optional_float,
    parse_optional_int,
    parse_optional_str,
)

# ── 44互換列（実hit_record.csvヘッダーから機械転記。順序変更・手編集禁止） ──
LEGACY_COLUMNS: tuple[str, ...] = (
    "date",
    "venue",
    "venue_num",
    "race",
    "night",
    "race_type",
    "why_bet",
    "confidence",
    "pred_combo",
    "pred_prob",
    "pred_ev",
    "pred_odds",
    "upset_score",
    "wind_speed",
    "wind_dir",
    "wave",
    "result_combo",
    "payout",
    "hit",
    "profit",
    "n_bets",
    "cost",
    "purchased",
    "buyscore",
    "match_index",
    "skip_reason",
    "model_version",
    "feat_win_rate",
    "feat_motor",
    "feat_avg_st",
    "feat_racer_class",
    "feat_course_st_1c",
    "feat_course_rank_1c",
    "feat_danger_breakdown",
    "danger_score_v3",
    "rank_index_json",
    "featured_boats_json",
    "venue_water_type",
    "venue_factor",
    "ability_trend",
    "course_f_rate_1c",
    "course_l_rate_1c",
    "course_rentai2_1c",
    "course_sample_confidence",
)

# ── 前方互換の拡張列（44列の末尾に追加。設計書④のCSV後方互換ルール準拠） ──
EXTENSION_COLUMNS: tuple[str, ...] = (
    "eval_id",
    "engine_name",
    "engine_version",
    "feature_schema_version",
    "evaluated_at",
    "upset_reasons_json",
    "win_probs_json",
    "investment_type",
    "kelly_fraction",
    "config_version",
    "patterns_json",
    "features_json",
)

ALL_COLUMNS: tuple[str, ...] = LEGACY_COLUMNS + EXTENSION_COLUMNS
_KNOWN_COLUMNS: frozenset[str] = frozenset(ALL_COLUMNS)

# ── レガシー行（拡張列なし）からの復元時に用いる補完定数。 ──
# 適用対象は拡張メタデータのみ（v1.1.4: 評価結果への既定値注入は禁止）。
LEGACY_DEFAULTS: dict[str, Any] = {
    "engine_name": "legacy",
    "engine_version": "0.0.0",
    "feature_schema_version": 0,
    "evaluated_at": "",  # 当時の評価時刻は記録されていない
    "investment_type": "",  # 投資タイプの概念導入前
    "config_version": "",
    "built_at": "",  # FeatureSet再構築時のビルド時刻（記録なし）
}

# ── feat_*列 ⇔ FeatureSet.boat_features[1] キー対応（全単射・Step2-3で確定） ──
# 旧コード突合済み: x_asahi_scoring.py L569のdetail["_features"]は全て1号艇の値。
FEAT_COLUMN_TO_FEATURE_KEY: dict[str, str] = {
    "feat_win_rate": "win_rate",
    "feat_motor": "motor_rate2",
    "feat_avg_st": "avg_st",
    "feat_racer_class": "racer_class_score",
    "feat_course_st_1c": "course_st_1c",
    "feat_course_rank_1c": "course_rank_1c",
    "ability_trend": "ability_trend",
    "course_f_rate_1c": "course_f_rate_1c",
    "course_l_rate_1c": "course_l_rate_1c",
    "course_rentai2_1c": "course_rentai2_1c",
    "course_sample_confidence": "course_sample_confidence",
}

# ── venue_water_type ラベル⇔コード変換表（全単射） ──
# 転記元: x_venue_stats.py L72-77 water_type_labels のキー集合。
WATER_TYPE_LABEL_TO_CODE: dict[str, float] = {
    "super_in": 1.0,
    "in_favorable": 2.0,
    "standard": 3.0,
    "rough": 4.0,
}
WATER_TYPE_CODE_TO_LABEL: dict[float, str] = {
    v: k for k, v in WATER_TYPE_LABEL_TO_CODE.items()
}


class HitRecordCsvMapper:
    """HitRecord ⇔ Row の純粋変換（唯一の集約Mapper）。

    不変条件:
    - from_row(to_row(m)) == m（完全往復。float許容1e-6はテスト側で吸収）
    - レガシー行（44列のみ）の from_row -> to_row で44互換列を保全
    Mapperはステートレス。ファイルI/O・パス解決・環境変数参照を行わない。

    前提条件: RaceEvaluation.feature_schema_version と
    RaceEvaluation.features.feature_schema_version は一致していること
    （from_rowは常にFeatureSet側の値を採用するため、不一致のモデルは
    往復同一性が成立しない。生成側=EvaluationEngineが一致を保証する）。
    """

    # ==================== to_row ====================

    @staticmethod
    def to_row(record: HitRecord) -> Row:
        """HitRecordをCSV行（44互換列＋拡張列）へ変換する。"""
        ev = record.evaluation
        pr = record.prediction
        bd = record.buy_decision
        rs = record.result
        wt = record.weather

        row: Row = {
            # --- 44互換列 ---
            "date": ev.race_date,
            "venue": ev.venue_name,
            "venue_num": ev.venue_num,
            "race": ev.race_number,
            "night": format_bool01(ev.is_night),
            "race_type": ev.race_type,
            "why_bet": pr.why_bet,
            "confidence": pr.confidence,
            "pred_combo": pr.pred_combo,
            "pred_prob": pr.pred_prob,
            "pred_ev": pr.pred_ev,
            "pred_odds": pr.pred_odds,
            "upset_score": ev.upset_score,
            "wind_speed": format_optional(wt.wind_speed_mps if wt else None),
            "wind_dir": format_optional(wt.wind_direction if wt else None),
            "wave": format_optional(wt.wave_height_cm if wt else None),
            "result_combo": format_optional(rs.result_combo if rs else None),
            "payout": format_optional(rs.payout if rs else None),
            "hit": format_bool01(rs.hit if rs else None),
            "profit": format_optional(rs.profit if rs else None),
            "n_bets": bd.n_bets,
            "cost": bd.cost,
            "purchased": format_bool01(bd.purchased),
            "buyscore": format_optional(bd.buyscore),
            "match_index": format_optional(ev.match_index),
            "skip_reason": format_optional(bd.skip_reason),
            "model_version": format_optional(ev.model_version),
            "feat_danger_breakdown": _dumps_or_empty(ev.danger_breakdown),
            "danger_score_v3": format_optional(ev.danger_score),
            "rank_index_json": _dumps_or_empty(ev.rank_index),
            "featured_boats_json": _dumps_or_empty(ev.featured_boats),
            # --- 拡張列 ---
            "eval_id": record.eval_id,
            "engine_name": ev.engine_name,
            "engine_version": ev.engine_version,
            "feature_schema_version": ev.feature_schema_version,
            "evaluated_at": ev.evaluated_at,
            "upset_reasons_json": json.dumps(list(ev.upset_reasons), ensure_ascii=False),
            "win_probs_json": (
                json.dumps({str(k): v for k, v in ev.win_probs.items()}, ensure_ascii=False)
                if ev.win_probs is not None
                else ""
            ),
            "investment_type": bd.investment_type,
            "kelly_fraction": bd.kelly_fraction,
            "config_version": bd.config_version,
            "patterns_json": json.dumps(list(pr.patterns), ensure_ascii=False),
            "features_json": json.dumps(ev.features.to_dict(), ensure_ascii=False),
        }

        # feat_*列と当地系列（FeatureSetのboat1・raceキーから導出）
        boat1 = ev.features.boat_features.get(1, {})
        for col, key in FEAT_COLUMN_TO_FEATURE_KEY.items():
            row[col] = format_optional(boat1.get(key))
        race_feats = ev.features.race_features
        row["venue_factor"] = format_optional(race_feats.get("venue_factor"))
        code = race_feats.get("venue_water_type_code")
        if code is None:
            row["venue_water_type"] = ""
        elif code in WATER_TYPE_CODE_TO_LABEL:
            row["venue_water_type"] = WATER_TYPE_CODE_TO_LABEL[code]
        else:
            raise ParseError(
                f"unknown venue_water_type_code: {code!r} (eval_id={record.eval_id})"
            )
        return row

    # ==================== from_row ====================

    @staticmethod
    def from_row(row: Row) -> HitRecord:
        """CSV行からHitRecordを復元する。

        - 44互換列のいずれかが「列として」欠落 -> ParseError
        - 既知列以外の未知列が存在 -> ParseError（黙殺禁止）
        - 44互換列の空欄セル -> Noneとして受理（レガシー未記録表現）
        - 拡張列は前方互換の任意列（欠落は正常。LEGACY_DEFAULTSで補完）
        """
        _validate_columns(row)

        date = _req_str(row, "date")
        venue = _req_str(row, "venue")
        venue_num = _req_int(row, "venue_num")
        race_number = _req_int(row, "race")
        eval_id_derived = f"{date}_{venue_num:02d}_{race_number:02d}"
        eval_id = parse_optional_str(row.get("eval_id")) or eval_id_derived
        if eval_id != eval_id_derived:
            raise ParseError(
                f"eval_id mismatch: column={eval_id!r} derived={eval_id_derived!r}"
            )

        features = _restore_feature_set(row, eval_id)
        evaluation = RaceEvaluation(
            eval_id=eval_id,
            race_date=date,
            venue_num=venue_num,
            venue_name=venue,
            race_number=race_number,
            is_night=_req_bool01(row, "night"),
            engine_name=parse_optional_str(row.get("engine_name"))
            or LEGACY_DEFAULTS["engine_name"],
            engine_version=parse_optional_str(row.get("engine_version"))
            or LEGACY_DEFAULTS["engine_version"],
            feature_schema_version=features.feature_schema_version,
            model_version=_opt_str(row, "model_version"),
            evaluated_at=_ext_str(row, "evaluated_at"),
            danger_score=_opt_float(row, "danger_score_v3"),
            danger_breakdown=_loads_or_none(row, "feat_danger_breakdown"),
            upset_score=_req_float(row, "upset_score"),
            upset_reasons=_ext_json_tuple(row, "upset_reasons_json"),
            rank_index=_loads_or_none(row, "rank_index_json"),
            featured_boats=_loads_or_none(row, "featured_boats_json"),
            win_probs=_ext_win_probs(row),
            race_type=_req_str(row, "race_type"),
            match_index=_opt_float(row, "match_index"),
            features=features,
        )
        prediction = Prediction(
            eval_id=eval_id,
            pred_combo=_req_str(row, "pred_combo"),
            pred_prob=_req_float(row, "pred_prob"),
            pred_ev=_req_float(row, "pred_ev"),
            pred_odds=_req_float(row, "pred_odds"),
            confidence=_req_float(row, "confidence"),
            why_bet=_req_str(row, "why_bet"),
            patterns=_ext_json_tuple(row, "patterns_json"),
        )
        buy_decision = BuyDecision(
            eval_id=eval_id,
            purchased=_req_bool01(row, "purchased"),
            buyscore=_opt_float(row, "buyscore"),
            investment_type=_ext_str(row, "investment_type"),
            n_bets=_req_int(row, "n_bets"),
            cost=_req_int(row, "cost"),
            kelly_fraction=_ext_opt_float(row, "kelly_fraction"),
            config_version=_ext_str(row, "config_version"),
            skip_reason=_opt_str(row, "skip_reason"),
        )
        return HitRecord(
            eval_id=eval_id,
            evaluation=evaluation,
            prediction=prediction,
            buy_decision=buy_decision,
            result=_restore_result(row, eval_id),
            weather=_restore_weather(row),
        )


# ==================== 内部ヘルパー（純粋関数） ====================


def _validate_columns(row: Row) -> None:
    missing = [c for c in LEGACY_COLUMNS if c not in row]
    if missing:
        raise ParseError(f"missing required legacy columns: {missing}")
    unknown = sorted(set(row.keys()) - _KNOWN_COLUMNS)
    if unknown:
        raise ParseError(f"unknown columns (must not be silently ignored): {unknown}")


def _opt_float(row: Row, col: str) -> Optional[float]:
    try:
        return parse_optional_float(row[col])
    except ValueError as exc:
        raise ParseError(f"column {col!r}: {exc}") from exc


def _opt_str(row: Row, col: str) -> Optional[str]:
    return parse_optional_str(row[col])


def _req_float(row: Row, col: str) -> float:
    value = _opt_float(row, col)
    if value is None:
        raise ParseError(f"column {col!r}: required cell is empty")
    return value


def _req_int(row: Row, col: str) -> int:
    try:
        value = parse_optional_int(row[col])
    except ValueError as exc:
        raise ParseError(f"column {col!r}: {exc}") from exc
    if value is None:
        raise ParseError(f"column {col!r}: required cell is empty")
    return value


def _req_str(row: Row, col: str) -> str:
    value = parse_optional_str(row[col])
    if value is None:
        raise ParseError(f"column {col!r}: required cell is empty")
    return value


def _req_bool01(row: Row, col: str) -> bool:
    try:
        value = parse_bool01(row[col])
    except ValueError as exc:
        raise ParseError(f"column {col!r}: {exc}") from exc
    if value is None:
        raise ParseError(f"column {col!r}: required cell is empty")
    return value


def _dumps_or_empty(obj: Optional[dict[str, Any]]) -> str:
    return json.dumps(obj, ensure_ascii=False) if obj is not None else ""


def _loads_or_none(row: Row, col: str) -> Optional[dict[str, Any]]:
    text = parse_optional_str(row[col])
    if text is None:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"column {col!r}: broken JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ParseError(f"column {col!r}: expected JSON object, got {type(value).__name__}")
    return value


def _ext_str(row: Row, col: str) -> str:
    """拡張列のstr取得。列欠落時はLEGACY_DEFAULTSで補完（空文字は値として尊重）。"""
    if col not in row:
        return LEGACY_DEFAULTS[col]
    value = row[col]
    return str(value) if value is not None else LEGACY_DEFAULTS[col]


def _ext_opt_float(row: Row, col: str) -> Optional[float]:
    """拡張列のOptional[float]取得。列欠落・空欄はNone（v1.1.6: 未計算の表現）。"""
    if col not in row:
        return None
    try:
        return parse_optional_float(row[col])
    except ValueError as exc:
        raise ParseError(f"column {col!r}: {exc}") from exc


def _ext_json_tuple(row: Row, col: str) -> tuple:
    """拡張JSON配列列をタプルへ。列欠落・空欄は空タプル（v1.1.5コレクション方針）。"""
    text = parse_optional_str(row.get(col))
    if text is None:
        return ()
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"column {col!r}: broken JSON: {exc}") from exc
    if not isinstance(value, list):
        raise ParseError(f"column {col!r}: expected JSON array, got {type(value).__name__}")
    return tuple(value)


def _ext_win_probs(row: Row) -> Optional[dict[int, float]]:
    text = parse_optional_str(row.get("win_probs_json"))
    if text is None:
        return None
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"column 'win_probs_json': broken JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ParseError("column 'win_probs_json': expected JSON object")
    try:
        return {int(k): float(v) for k, v in raw.items()}
    except (ValueError, TypeError) as exc:
        raise ParseError(f"column 'win_probs_json': invalid entries: {exc}") from exc


def _restore_feature_set(row: Row, eval_id: str) -> FeatureSet:
    """FeatureSetの復元。拡張列features_jsonが最優先、無ければfeat_*列から再構築。"""
    text = parse_optional_str(row.get("features_json"))
    if text is not None:
        try:
            return FeatureSet.from_dict(json.loads(text))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise ParseError(f"column 'features_json': broken: {exc}") from exc

    boat1: dict[str, float] = {}
    missing: list[str] = []
    for col, key in FEAT_COLUMN_TO_FEATURE_KEY.items():
        value = _opt_float(row, col)
        if value is None:
            missing.append(key)
        else:
            boat1[key] = value
    race_feats: dict[str, float] = {}
    venue_factor = _opt_float(row, "venue_factor")
    if venue_factor is None:
        missing.append("venue_factor")
    else:
        race_feats["venue_factor"] = venue_factor
    label = parse_optional_str(row["venue_water_type"])
    if label is None:
        missing.append("venue_water_type_code")
    elif label in WATER_TYPE_LABEL_TO_CODE:
        race_feats["venue_water_type_code"] = WATER_TYPE_LABEL_TO_CODE[label]
    else:
        raise ParseError(f"column 'venue_water_type': unknown label {label!r}")
    return FeatureSet(
        eval_id=eval_id,
        feature_schema_version=LEGACY_DEFAULTS["feature_schema_version"],
        built_at=LEGACY_DEFAULTS["built_at"],
        boat_features={1: boat1},
        race_features=race_feats,
        local_features=None,
        missing_keys=tuple(missing),
    )


def _restore_result(row: Row, eval_id: str) -> Optional[RaceResult]:
    """結果4列の復元。全て空欄->None（結果未判明）。部分的な空欄->ParseError。"""
    cells = ("result_combo", "payout", "hit", "profit")
    emptiness = {c: parse_optional_str(row[c]) is None for c in cells}
    if all(emptiness.values()):
        return None
    if any(emptiness.values()):
        raise ParseError(
            f"result columns partially empty (data corruption suspected): {emptiness}"
        )
    return RaceResult(
        eval_id=eval_id,
        result_combo=_req_str(row, "result_combo"),
        payout=_req_int(row, "payout"),
        hit=_req_bool01(row, "hit"),
        profit=_req_int(row, "profit"),
    )


def _restore_weather(row: Row) -> Optional[Weather]:
    """気象3列の復元。全て空欄->None。風速/波高の片側だけ空欄->ParseError。

    wind_dirのみ空欄は許容する（実データ87/255行で確認済みのパターン。
    wind_direction=""として復元する）。
    """
    speed_text = parse_optional_str(row["wind_speed"])
    dir_text = parse_optional_str(row["wind_dir"])
    wave_text = parse_optional_str(row["wave"])
    if speed_text is None and dir_text is None and wave_text is None:
        return None
    if speed_text is None or wave_text is None:
        raise ParseError(
            "weather columns partially empty: wind_speed/wave must be present together"
        )
    return Weather(
        wind_speed_mps=_req_float(row, "wind_speed"),
        wind_direction=dir_text if dir_text is not None else "",
        wave_height_cm=_req_int(row, "wave"),
    )
