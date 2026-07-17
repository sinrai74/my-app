"""
Race系データモデル。

設計書 Phase0.5 v1.1 ③3.1（Race）・3.2（RaceEntry）に対応する。
FANファイル由来のコース別成績（CourseStats）、天候（Weather）、
オッズ（OddsSnapshot）を含む。

依存: 標準ライブラリのみ（dataclasses, typing）。
設計書⑤5.3の禁止事項により、本モジュールはファイルI/O・ネットワーク・
現在時刻取得を行わない（値を保持するデータ構造のみを定義する）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CourseStats:
    """
    Design Spec: §3.2 RaceEntry.course_stats

    FANファイル由来の、選手のコース別成績。
    1コース〜6コースぶんの進入回数・複勝率・平均ST・平均スタート順位を保持する。

    項目名はFANファイル仕様書の「Nコース進入回数」等に対応する
    （設計書⑪ 命名規則: CSV/JSON同様 snake_case、単位を含める）。
    """

    course: int  # 1-6
    entry_count: int  # 進入回数
    place_rate: float  # 複勝率（%、小数点以下1桁）
    avg_start_timing: float  # 平均スタートタイミング（小数点以下2桁）
    avg_start_rank: float  # 平均スタート順位（小数点以下2桁）


@dataclass(frozen=True)
class Weather:
    """
    Design Spec: §3.1 Race.weather

    レース時点の気象情報。
    「朝刊のみ」ポリシー（Phase0.5 ⑨引き継ぐべき資産17）により、
    本モデルの値はAIエンジン（core/）の評価には使用しない。
    保持のみ行う（表示・記録用途）。
    """

    wind_speed_mps: float
    wind_direction: str
    wave_height_cm: int
    temperature_celsius: Optional[float] = None
    water_temperature_celsius: Optional[float] = None


@dataclass(frozen=True)
class OddsSnapshot:
    """
    Design Spec: §3.1 Race (関連: data/odds.pyが取得するオッズ情報)

    ある時点でのオッズのスナップショット。

    3連単オッズ等、data/odds.py（現行 odds_fetch.py 後継）が取得する
    オッズ情報を保持する。取得時刻はスナップショットごとに固定する
    （設計書⑤5.3: coreは現在時刻を取得しない。オッズ取得時刻はここに記録し、
    core側へは引数として渡す）。
    """

    eval_id: str  # 対象レース "{race_date}_{venue_num:02d}_{race_number:02d}"
    fetched_at: str  # ISO8601 JST
    trifecta_odds: dict[str, float] = field(default_factory=dict)
    # trifecta_odds のキーは組番文字列（例 "1-2-3"）、値はオッズ


@dataclass(frozen=True)
class RaceEntry:
    """
    Design Spec: §3.2 RaceEntry

    出走艇1艇分の情報。
    """

    lane: int  # 枠番 1-6
    racer_no: str
    racer_name: str
    racer_class: str  # A1/A2/B1/B2 等
    win_rate: float
    place_rate: float
    motor_no: int
    motor_rate2: float  # モーター2連率
    avg_st: float  # 平均スタートタイミング
    branch: Optional[str] = None  # 支部
    hometown: Optional[str] = None  # 出身地
    boat_no: Optional[int] = None
    boat_rate2: Optional[float] = None
    course_stats: Optional[tuple[CourseStats, ...]] = None
    # course_stats は1〜6コース分のCourseStatsのタプル（任意・FANファイル未取得時はNone）


@dataclass(frozen=True)
class Race:
    """
    Design Spec: §3.1 Race

    レース基本情報。全レイヤーの起点となるデータモデル。
    features/feature_builder（L1）がこのモデルとMotorHistory等の履歴データから
    FeatureSetを構築し、それがAIエンジン（L2）への唯一の入力となる
    （設計書③3.3 FeatureSet、⑤5.1）。
    """

    race_date: str  # 開催日 YYYYMMDD
    venue_num: int  # 場コード 1-24
    venue_name: str
    race_number: int  # 1-12
    close_time: str  # 締切時刻 HH:MM
    is_night: bool  # ナイター
    entries: tuple[RaceEntry, ...]  # 6艇
    grade: Optional[str] = None  # SG/G1/G2/G3/一般
    weather: Optional[Weather] = None

    @property
    def eval_id(self) -> str:
        """
        設計書③3.4 RaceEvaluation.eval_id と同一の採番規則。
        "{race_date}_{venue_num:02d}_{race_number:02d}"
        """
        return f"{self.race_date}_{self.venue_num:02d}_{self.race_number:02d}"
