"""
FeatureBuilder（L1）: Race＋出走データ -> FeatureSet の構築。

設計書 v1.1.8 ③3.3・⑤5.1（FeatureSetがAIエンジンへの唯一の入力）、
Step4計画書 §2・§4・C3 に基づく。

移植元: x_asahi_scoring.py calculate_upset_score_v2 内の
detail["_features"] 構築ブロック（L540-L590）。
基準コミット: freeze-v1-baseline（Golden生成時source_commit=3a7f9c3dd7c628255285aefbe5b3e03978ec93b3）。
**旧コードの出力を正とし、丸め・ゲート条件（0値をfalsyとして欠損扱いにする等の
挙動を含む）を1e-6以内で忠実に再現する（C2）。改善・是正はしない。**

責務境界:
- 本モジュールは純粋計算のみ。ファイル・API・時刻へ直接アクセスしない（⑤5.3）
- venue統計（水面タイプ・場コース補正）は計算済みの値をVenueStatsProvider経由で
  受け取る（DI）。統計の算出自体はL1の責務ではない（現行はx_venue_stats、
  新系統ではStep5でproviderアダプタを結線する）
- boats はVer4互換のBoatInfo属性投影（BOAT_ATTRS 17属性のdict）として受け取る。
  RaceEntryモデルとの統合はレガシー移行完了後の課題（Step5以降で提案）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence

from models.evaluation import FeatureSet
from models.race import Race

# venue_water_type の type キー -> コード変換表。
# storage.mappers.hit_record_mapper.WATER_TYPE_LABEL_TO_CODE と同一内容を維持する
# （L1はstorage層へ依存できないため独立定義。同期はユニットテストで機械検証する）。
WATER_TYPE_TYPE_TO_CODE: dict[str, float] = {
    "super_in": 1.0,
    "in_favorable": 2.0,
    "standard": 3.0,
    "rough": 4.0,
}

# FeatureSet boat_features[1] のキー定義順（missing_keysの決定的順序にも使用）。
# storage.mappers.hit_record_mapper.FEAT_COLUMN_TO_FEATURE_KEY の値集合と
# 一致すること（同期はユニットテストで機械検証する）。
BOAT1_FEATURE_KEYS: tuple[str, ...] = (
    "win_rate",
    "motor_rate2",
    "avg_st",
    "racer_class_score",
    "course_st_1c",
    "course_rank_1c",
    "ability_trend",
    "course_f_rate_1c",
    "course_l_rate_1c",
    "course_rentai2_1c",
    "course_sample_confidence",
)
RACE_FEATURE_KEYS: tuple[str, ...] = ("venue_factor", "venue_water_type_code")

FEATURE_SCHEMA_VERSION: int = 1


class VenueStatsProvider(Protocol):
    """venue統計の供給者（DI）。現行x_venue_statsと同一の戻り値形状。"""

    def classify_water_type(self, venue_name: str) -> Mapping[str, Any]:
        """水面タイプ分類。少なくとも {"type": str, "label": str} を含む。"""
        ...

    def get_venue_course_factor(self, venue_name: str, course: int) -> Mapping[str, Any]:
        """場×コース補正。少なくとも {"factor": float, "samples": int} を含む。"""
        ...


@dataclass(frozen=True)
class FeatureInputs:
    """FeatureBuilderへの出走データ入力（Ver4互換のBoatInfo属性投影）。

    boats: 各艇のBOAT_ATTRS属性dict（lane, win_rate, motor, avg_st, racer_class,
    ability_curr, ability_prev, course_nyuko, course_st, course_rank,
    course_f_count, course_l_count, course_place_counts 等）。
    golden_wrapper.pyのGolden Inputと同一形式。
    """

    boats: tuple[Mapping[str, Any], ...]


class FeatureBuilder(Protocol):
    """FeatureSet構築の抽象（設計書⑤5.1）。"""

    def build(self, race: Race, inputs: FeatureInputs, built_at: str) -> FeatureSet:
        ...


# 将来拡張メモ（Should-3・実装は変更しない）:
# 現在buildはFeatureSetを直接返すが、構築診断（欠損理由・入力品質警告等）を
# 伴わせたくなった場合は、FeatureSetを内包する FeatureBuilderResult 型を導入して
# 戻り値を包む拡張が可能（Protocolの戻り値変更は設計書改訂を伴う承認事項）。
class DefaultFeatureBuilder:
    """レガシー_features構築の忠実移植（Ver4互換）。"""

    def __init__(self, venue_stats: VenueStatsProvider) -> None:
        self._venue_stats = venue_stats

    def build(self, race: Race, inputs: FeatureInputs, built_at: str) -> FeatureSet:
        boat1 = _find_boat1(inputs.boats)
        boat_feats: dict[str, float] = {}
        race_feats: dict[str, float] = {}
        missing: list[str] = []

        # ── 1号艇 基本3属性（移植元: _features win_rate/motor/avg_st） ──
        _put(boat_feats, missing, "win_rate", _num(boat1.get("win_rate")) if boat1 else None)
        _put(boat_feats, missing, "motor_rate2", _num(boat1.get("motor")) if boat1 else None)
        _put(boat_feats, missing, "avg_st", _num(boat1.get("avg_st")) if boat1 else None)

        # ── racer_class_score ──
        # レガシー_features["racer_class"]は級別文字列（例:"A1"）であり数値特徴として
        # 定義されていない（実CSVでも全行空欄）。数値化仕様はfreeze時点で存在しない
        # ため、常に欠損として扱う（推測実装の禁止・C2）。
        _put(boat_feats, missing, "racer_class_score", None)

        # ── コース別（1コース）: nyuko[0]>0 ゲート（移植元と同一条件） ──
        nyuko = boat1.get("course_nyuko") if boat1 else None
        gate_1c = bool(boat1) and bool(nyuko) and nyuko[0] > 0
        course_st = boat1.get("course_st") if boat1 else None
        course_rank = boat1.get("course_rank") if boat1 else None
        _put(boat_feats, missing, "course_st_1c",
             _num(course_st[0]) if gate_1c and course_st else None)
        _put(boat_feats, missing, "course_rank_1c",
             _num(course_rank[0]) if gate_1c and course_rank else None)

        # ── 能力推移: curr/prevがともにtruthy（0.0は欠損になる挙動を保存） ──
        ability_trend: Optional[float] = None
        if boat1 and boat1.get("ability_curr") and boat1.get("ability_prev"):
            ability_trend = round(
                float(boat1["ability_curr"]) - float(boat1["ability_prev"]), 2
            )
        _put(boat_feats, missing, "ability_trend", ability_trend)

        # ── F率/L率/2連対率（1コース）: 移植元の各truthyゲートを保存 ──
        f_rate = l_rate = rentai2 = None
        if gate_1c:
            nyuko0 = nyuko[0]
            f_counts = boat1.get("course_f_count")
            l_counts = boat1.get("course_l_count")
            place_counts = boat1.get("course_place_counts")
            if f_counts:
                f_rate = round(f_counts[0] / nyuko0 * 100, 1)
            if l_counts:
                l_rate = round(l_counts[0] / nyuko0 * 100, 1)
            if place_counts and sum(place_counts[0]) > 0:
                counts0 = place_counts[0]
                rentai2 = round((counts0[0] + counts0[1]) / nyuko0 * 100, 1)
        _put(boat_feats, missing, "course_f_rate_1c", f_rate)
        _put(boat_feats, missing, "course_l_rate_1c", l_rate)
        _put(boat_feats, missing, "course_rentai2_1c", rentai2)

        # ── venue系: venue_nameが真のときのみ算出（移植元と同一ゲート） ──
        sample_confidence: Optional[float] = None
        venue_factor: Optional[float] = None
        water_code: Optional[float] = None
        if race.venue_name:
            water = self._venue_stats.classify_water_type(race.venue_name)
            water_type_key = str(water.get("type", ""))
            if water_type_key in WATER_TYPE_TYPE_TO_CODE:
                water_code = WATER_TYPE_TYPE_TO_CODE[water_type_key]
            factor = self._venue_stats.get_venue_course_factor(race.venue_name, 1)
            venue_factor = _num(factor.get("factor"))
            sample_confidence = _num(factor.get("samples"))
        _put(boat_feats, missing, "course_sample_confidence", sample_confidence)
        _put(race_feats, missing, "venue_factor", venue_factor)
        _put(race_feats, missing, "venue_water_type_code", water_code)

        return FeatureSet(
            eval_id=race.eval_id,
            feature_schema_version=FEATURE_SCHEMA_VERSION,
            built_at=built_at,
            boat_features={1: boat_feats},
            race_features=race_feats,
            local_features=None,
            missing_keys=tuple(missing),
        )


def create_feature_builder(venue_stats: VenueStatsProvider) -> FeatureBuilder:
    """DI配線用ファクトリ（Step4-1スコープ。エンジン結線はStep4-5以降）。"""
    return DefaultFeatureBuilder(venue_stats)


# ==================== 内部ヘルパー（純粋関数） ====================


def _find_boat1(boats: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for boat in boats:
        if boat.get("lane") == 1:
            return boat
    return None


def _num(value: Any) -> Optional[float]:
    """数値化。None・空文字は欠損（None）。それ以外はfloatへ。"""
    if value is None or value == "":
        return None
    return float(value)


def _put(
    target: dict[str, float],
    missing: list[str],
    key: str,
    value: Optional[float],
) -> None:
    """値があればtargetへ、無ければmissing_keysへ（FeatureSetの欠損契約③3.3）。"""
    if value is None:
        missing.append(key)
    else:
        target[key] = value
