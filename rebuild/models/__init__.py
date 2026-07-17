"""
競艇AIプラットフォーム 共通データモデル定義。

Phase0.5設計固定書 v1.1 ③（データモデル固定）に基づく。
このパッケージは標準ライブラリのみに依存する（設計書⑩ 依存ルール）。

Step1-1: Race系モデル（Race, RaceEntry, CourseStats, Weather, OddsSnapshot）
Step1-2: Feature/Evaluation系モデル（FeatureSet, RaceEvaluation, Prediction, BuyDecision）
Step1-3: 実績・学習系モデル（RaceResult, HitRecord, MotorHistory, VenueStatistics,
          LocalCourseStats, LearningData, VerificationResult）
Step1-4: 出力・KPI系モデル（RankingEntry, NewsArticle, SystemMetrics）
"""

from models.evaluation import BuyDecision, FeatureSet, Prediction, RaceEvaluation
from models.output import NewsArticle, RankingEntry, SystemMetrics
from models.race import (
    CourseStats,
    OddsSnapshot,
    Race,
    RaceEntry,
    Weather,
)
from models.record import (
    HitRecord,
    LearningData,
    LocalCourseStats,
    MotorHistory,
    RaceResult,
    VenueStatistics,
    VerificationResult,
)

__all__ = [
    "BuyDecision",
    "CourseStats",
    "FeatureSet",
    "HitRecord",
    "LearningData",
    "LocalCourseStats",
    "MotorHistory",
    "NewsArticle",
    "OddsSnapshot",
    "Prediction",
    "Race",
    "RaceEntry",
    "RaceEvaluation",
    "RaceResult",
    "RankingEntry",
    "SystemMetrics",
    "VenueStatistics",
    "VerificationResult",
    "Weather",
]
