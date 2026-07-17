"""
Serializer層: ドメインモデル ⇔ JSON互換dict（ネスト許容）の純粋変換。

設計書 v1.1.6 ④・⑩、Step2実装計画書 §2 に基づく（5ファイル構成は計画書§7で確定）。
Step2-5で追加。
"""

from storage.serializers.common import SCHEMA_VERSION
from storage.serializers.evaluation_serializer import (
    BuyDecisionSerializer,
    PredictionSerializer,
    RaceEvaluationSerializer,
    RaceResultSerializer,
)
from storage.serializers.metrics_serializer import SystemMetricsSerializer
from storage.serializers.news_serializer import NewsArticleSerializer
from storage.serializers.ranking_serializer import RankingEntrySerializer
from storage.serializers.verification_serializer import VerificationResultSerializer

__all__ = [
    "SCHEMA_VERSION",
    "BuyDecisionSerializer",
    "NewsArticleSerializer",
    "PredictionSerializer",
    "RaceEvaluationSerializer",
    "RaceResultSerializer",
    "RankingEntrySerializer",
    "SystemMetricsSerializer",
    "VerificationResultSerializer",
]
