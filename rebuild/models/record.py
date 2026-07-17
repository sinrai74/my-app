"""
Design Spec: §3.7-3.11, 3.14-3.15 実績・学習系モデル群

Implements the immutable domain model.
No business logic.
No persistence.

対象: RaceResult, HitRecord, MotorHistory, VenueStatistics,
      LocalCourseStats, LearningData, VerificationResult

依存: 標準ライブラリのみ（dataclasses, typing）。
HitRecordは現行hit_record.csvの44列互換をモデル自体では表現しない
（案A'採用: RaceEvaluation・Prediction・BuyDecision・RaceResultを
型で保持する集約モデルとする）。CSVとの相互変換・44互換列一致検証は
Mapper/Serializer層（Step2以降）の責務とする（設計書⑤責務分離）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from models.evaluation import BuyDecision, FeatureSet, Prediction, RaceEvaluation
from models.race import Weather


@dataclass(frozen=True)
class RaceResult:
    """
    Design Spec: §3.7 RaceResult

    Implements the immutable domain model.
    No business logic.
    No persistence.
    """

    eval_id: str
    result_combo: str
    payout: int
    hit: bool
    profit: int


@dataclass(frozen=True)
class HitRecord:
    """
    Design Spec: §3.8 HitRecord

    Implements the immutable domain model.
    No business logic.
    No persistence.

    RaceEvaluation＋Prediction＋BuyDecision＋RaceResultの結合ビュー（集約モデル、案A'）。
    現行hit_record.csvの44列スキーマとの互換性はこのモデルでは表現しない。
    列単位の変換・⑭の一致基準（44互換列100%一致）検証はCSV Mapper側のテストで保証する。

    RaceResultは結果判明前は存在しないため任意項目とする
    （設計書⑥保存層責務: BuyDecision確定時に追記、結果判明時に結果列のみ更新）。

    weather（v1.1.3追加・案W-B）:
    Weather is stored in HitRecord only for historical record reconstruction
    and legacy CSV compatibility. It is not the source of truth for evaluation.
    RaceEvaluation remains independent from Weather.
    """

    eval_id: str
    evaluation: RaceEvaluation
    prediction: Prediction
    buy_decision: BuyDecision
    result: Optional[RaceResult] = None
    weather: Optional[Weather] = None


@dataclass(frozen=True)
class MotorHistory:
    """
    Design Spec: §3.9 MotorHistory

    Implements the immutable domain model.
    No business logic.
    No persistence.

    現行motor_history.csvの11列（date, venue_num, venue, motor_no, racer_no,
    racer_name, lane, place, ex_time, start_timing, race_number）と1:1対応。
    """

    date: str
    venue_num: int
    venue: str
    motor_no: int
    racer_no: str
    racer_name: str
    lane: int
    place: int
    ex_time: float
    start_timing: float
    race_number: int


@dataclass(frozen=True)
class VenueStatistics:
    """
    Design Spec: §3.10 VenueStatistics

    Implements the immutable domain model.
    No business logic.
    No persistence.
    """

    venue_num: int
    water_type: str
    course_stats: dict[str, Any]
    venue_factor: float
    updated_at: str  # ISO8601 JST


@dataclass(frozen=True)
class LocalCourseStats:
    """
    Design Spec: §3.11 LocalCourseStats

    Implements the immutable domain model.
    No business logic.
    No persistence.

    現行local_course_stats.csvの15列と1:1対応
    （racer_no, venue_code, venue_name, course, starts, first..sixth,
    first_rate, top2_rate, top3_rate, last_updated）。
    """

    racer_no: str
    venue_code: int
    venue_name: str
    course: int
    starts: int
    first: int
    second: int
    third: int
    fourth: int
    fifth: int
    sixth: int
    first_rate: float
    top2_rate: float
    top3_rate: float
    last_updated: str  # YYYYMMDD


@dataclass(frozen=True)
class LearningData:
    """
    Design Spec: §3.15 LearningData

    Implements the immutable domain model.
    No business logic.
    No persistence.

    RaceEvaluation.features ＋ RaceResult を結合した特徴量行（学習用の集約モデル）。
    features はFeatureSet型そのものを保持する（案X採用: Step1の「ドメインモデルは
    型安全なオブジェクトを保持し、永続化・エクスポート形式に依存しない」方針の維持）。
    CSVへの展開・フラット化はMapper/Serializer層（Step2以降）の責務とする。
    """

    eval_id: str
    features: FeatureSet
    race_result: RaceResult
    engine_version: str
    feature_schema_version: int


@dataclass(frozen=True)
class VerificationResult:
    """
    Design Spec: §3.14 VerificationResult

    Implements the immutable domain model.
    No business logic.
    No persistence.
    """

    race_date: str
    n_races: int
    n_purchased: int
    n_hit: int
    total_cost: int
    total_payout: int
    roi: float
    by_race_type: dict[str, Any]
    by_rank: dict[str, Any]
    engine_version: str
