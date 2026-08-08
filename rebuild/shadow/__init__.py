"""
shadow層（Step5-6）: Legacy/Rebuild並走比較・実送信抑止・Go/No-Go判定。

責務: Shadow運用を支える結線コード（比較・NullNotifier・判定集約）。
禁止: 判定結果に応じたスコア補正・既存ロジック修正。差分は原因調査対象
  であり、コードでの補正は行わない。
"""

from shadow.aggregator import (
    ShadowAggregate,
    ShadowAggregator,
    to_staged_result,
)
from shadow.comparator import compare, to_comparable
from shadow.go_no_go import GoNoGoCriteria, GoNoGoResult, evaluate_go_no_go
from shadow.legacy_notification import (
    LegacyNotificationView,
    build_notification_view,
)
from shadow.legacy_output import compare_output_bytes, read_output_bytes
from shadow.legacy_source import LegacySentRecord, load_sent_records
from shadow.legacy_values_builder import build_legacy_values
from shadow.notifier import NullNotifier
from shadow.prediction_provider import (
    LegacyPredictionProvider,
    PredictionContext,
)
from shadow.runner import ShadowRunner
from shadow.staged_comparator import (
    ConsecutiveMatchCounter,
    ShadowSummary,
    StagedResult,
    compare_staged,
)

__all__ = [
    "ConsecutiveMatchCounter",
    "GoNoGoCriteria",
    "GoNoGoResult",
    "LegacyNotificationView",
    "LegacyPredictionProvider",
    "LegacySentRecord",
    "NullNotifier",
    "PredictionContext",
    "ShadowAggregate",
    "ShadowAggregator",
    "ShadowRunner",
    "ShadowSummary",
    "StagedResult",
    "build_legacy_values",
    "build_notification_view",
    "compare",
    "compare_output_bytes",
    "compare_staged",
    "evaluate_go_no_go",
    "load_sent_records",
    "read_output_bytes",
    "to_comparable",
    "to_staged_result",
]
