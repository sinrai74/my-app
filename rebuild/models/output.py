"""
Design Spec: §3.12-3.13, 3.16 出力・KPI系モデル群

Implements the immutable domain model.
No business logic.
No persistence.

対象: RankingEntry, NewsArticle, SystemMetrics

依存: 標準ライブラリのみ（dataclasses, typing）。
RankingEntryはRaceEvaluationの集約モデルではなく、「ランキング結果が
どのRaceEvaluationを元に生成されたか」を識別するための軽量な参照モデルである
（案P採用）。RaceEvaluationオブジェクトの取得・解決はRepository/Mapper層
（Step2以降）の責務とし、本モデルはsource_eval_id（str）のみを保持する。
HitRecord（§3.8、複数ドメインモデルを束ねる集約モデル）とは役割が異なる。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class RankingEntry:
    """
    Design Spec: §3.12 RankingEntry

    Implements the immutable domain model.
    No business logic.
    No persistence.

    日次ランキング（危険艇/激走モーター/覚醒モーター/万舟）1件。
    source_eval_idは対象RaceEvaluationへの参照IDであり、RaceEvaluation型
    そのものは保持しない（案P: 軽量な参照モデルとしての設計）。
    独自再計算は行わない前提のモデルであり、scoreはsource_eval_id先の
    RaceEvaluationから抽出された値をそのまま保持する。
    """

    ranking_type: str  # danger / hot_motor / awakening / manshuu
    race_date: str
    venue_name: str
    race_number: int
    score: float
    rank_label: str  # S/A/B等
    subject: dict[str, Any]  # 選手 or モーターの識別情報
    reasons: tuple[str, ...]
    ai_comment: str
    source_eval_id: str  # 対象RaceEvaluationへの参照ID（型は保持しない）


@dataclass(frozen=True)
class NewsArticle:
    """
    Design Spec: §3.13 NewsArticle

    Implements the immutable domain model.
    No business logic.
    No persistence.

    AI新聞。sectionsは保存済みデータ（evaluations/rankings/verification）から
    表示層（output/）が構成した節の集合を保持する。本モデル自体は
    集計・抽出ロジックを持たない（設計書⑦表示層責務: 判定を一切行わない）。
    """

    race_date: str
    sections: dict[str, Any]  # 注目レース/注目選手/警報/実績サマリー
    generated_at: str  # ISO8601 JST
    brand_version: str


@dataclass(frozen=True)
class SystemMetrics:
    """
    Design Spec: §3.16 SystemMetrics

    Implements the immutable domain model.
    No business logic.
    No persistence.

    ジョブ実行ごとに1レコード。AIの成績ではなくシステム自体の健康状態を記録する
    （設計書⑲）。counters/rates/errorsは計測値の辞書として保持し、
    集計・閾値判定ロジック（WARNING/ERROR判定等）は本モデルに含めない
    （判定はservices層の責務、設計書⑲19.3）。
    ai_summaryはVerificationResultからの転記のみを想定し、本モデル内では
    再計算しない（設計書①理念1.3: 評価は一度だけ）。
    """

    metrics_id: str  # "{date}_{job}_{run_id}"
    race_date: str
    job_name: str
    run_id: str
    started_at: str
    finished_at: str
    duration_seconds: float
    status: str  # success / partial / failed
    counters: dict[str, Any]
    rates: dict[str, Any]
    errors: dict[str, Any]
    schema_version: int
    ai_summary: Optional[dict[str, Any]] = None
