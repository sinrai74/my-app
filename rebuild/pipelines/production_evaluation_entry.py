"""Production 評価・永続化入口（CLI・薄いwrapper）。

目的: 実装済みの RaceEvaluation 実行間再利用（run_one_race）と、評価直後の
Durable保存を、実際の実行入口から利用可能にする。公開HTML出力・メール等の
実送信・Workflowによる自動運転はこの入口の対象外。

構成（既存部品の組み合わせのみ）:
  - TARGET_RACES: 既存 parse_target_races()（生成はしない）
  - EvaluationRepository / DurableEvaluationStore: 同一 evaluations/{date}.jsonl
    （Phase0.5 §3.4/§④ の保存先。通常実行の正本＝checkout 済み JSONL・C-1）。
    書込は append_durably（A）。1回の実行＝1日（TARGET_RACES の日付混在は入力エラー）。
  - Releases tag: data-store-v2（Step3 S6。現行 data-store へは書き込まない）
  - S4: config/pipeline.json「締切前分数」で評価前の締切判定、
    config/delivery.json「1日投稿数上限」＋日次通知カウンタで通知前の上限
  - Output: 空 Renderer（出力先パスを新設しない）
  - Notification: NullNotifier のみ（実送信不能）
  - 実行: run_production_day(persist=True, evaluation_repository=...)

環境変数: TARGET_RACES（必須）、GITHUB_TOKEN / GITHUB_REPOSITORY（既存
build_github_evaluation_store の要件）。JSONLパスは実行時カレントディレクトリ
からの相対パス。
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

_JST = timezone(timedelta(hours=9))

from actions.config_loader import (
    load_daily_notification_limit,
    load_deadline_minutes,
)
from actions.production_entrypoint import (
    build_evaluation_repository,
    build_github_evaluation_store,
    build_github_notification_counter,
    build_production_bundle,
    run_production_day,
)
from actions.config_loader import ConfigError
from actions.metrics_reporter import (
    JOB_NAME,
    RELEASE_TAG as METRICS_RELEASE_TAG,
    MetricsReporter,
    build_errors,
)
from actions.wiring import PipelineBundle

log = logging.getLogger(__name__)

RELEASE_TAG = "data-store-v2"  # Step3 S6（確定）


def evaluations_path_for(race_date: str) -> str:
    """評価JSONLの保存先（Phase0.5 §3.4 / §④: evaluations/{date}.jsonl）。"""
    return f"evaluations/{race_date}.jsonl"


def single_race_date(target_races: list[tuple[str, int, int]]) -> str:
    """TARGET_RACES が1日分であることを確認し、その YYYYMMDD を返す。

    空・日付混在は ValueError（評価処理を開始しない）。
    """
    dates = sorted({race_date for race_date, _v, _r in target_races})
    if not dates:
        raise ValueError("TARGET_RACES is empty; one race date is required")
    if len(dates) > 1:
        raise ValueError(
            f"TARGET_RACES must contain a single race date per run; got {dates}"
        )
    return dates[0]


def build_evaluation_only_bundle(
    durable_store,
    *,
    production_bundle_factory: Callable[..., PipelineBundle] = build_production_bundle,
) -> PipelineBundle:
    """評価・永続化専用 bundle を既存部品から組む（実送信不能・出力なし）。

    evaluation / buy は build_production_bundle(durable_store=...) のものを
    そのまま使い、output は空 Renderer、notification は NullNotifier のみに
    差し替える（Shadow の build_bundle と同じ構成要素。既存層は無改変）。
    """
    from notification.service import NotificationService
    from pipelines.notification_pipeline import NotificationPipeline
    from pipelines.output_pipeline import OutputPipeline
    from shadow.notifier import NullNotifier

    production = production_bundle_factory(durable_store=durable_store)
    return PipelineBundle(
        race_source=production.race_source,
        evaluation_pipeline=production.evaluation_pipeline,
        buy_pipeline=production.buy_pipeline,
        output_pipeline=OutputPipeline({}),
        notification_pipeline=NotificationPipeline(
            NotificationService({"mail": NullNotifier("mail")})
        ),
    )


def _no_output_paths(race_date: str, venue_num: int, race_number: int) -> dict:
    """空 Renderer に対応する空の output_paths（出力先パスは作らない）。"""
    return {}


def run_entry(
    target_races: list[tuple[str, int, int]],
    *,
    env: Mapping[str, str] | None = None,
    store_factory: Callable[..., Any] = build_github_evaluation_store,
    repository_factory: Callable[..., Any] = build_evaluation_repository,
    bundle_factory: Callable[..., PipelineBundle] = build_evaluation_only_bundle,
    day_runner: Callable[..., dict] = run_production_day,
    counter_factory: Callable[..., Any] = build_github_notification_counter,
    deadline_minutes_loader: Callable[[], int] = load_deadline_minutes,
    daily_limit_loader: Callable[[], int] = load_daily_notification_limit,
    now_provider: Callable[[], Any] | None = None,
) -> dict:
    """対象日の同一JSONLで Store / Repository を組み、run_production_day を呼ぶ。"""
    jsonl_path = evaluations_path_for(single_race_date(target_races))
    store = store_factory(jsonl_path, tag=RELEASE_TAG, env=env)
    repository = repository_factory(jsonl_path)
    bundle = bundle_factory(store)
    deadline_minutes = deadline_minutes_loader()
    daily_limit = daily_limit_loader()
    counter = counter_factory(env=env)
    extra = {} if now_provider is None else {"now_provider": now_provider}
    return day_runner(
        target_races,
        _no_output_paths,
        bundle=bundle,
        persist=True,
        evaluation_repository=repository,
        deadline_minutes=deadline_minutes,
        daily_notification_limit=daily_limit,
        notification_counter=counter,
        **extra,
    )


SNAPSHOT_PATH = "system_metrics.json"
MONTHLY_DIR = "metrics"


def build_metrics_reporter(
    env: Mapping[str, str] | None = None,
) -> MetricsReporter:
    """S6: MetricsStore（既存・無変更）と ReleaseClient を結線するだけ。

    Metrics は git管理外・Releasesのみ（S6の設計判断）のため commit はしない。
    認証情報が無い場合は Release退避なしのReporterを返す（保存は行う）。
    """
    import os as _os
    from pathlib import Path as _Path

    from storage.clients.github_release_client import GithubReleaseClient
    from storage.metrics_store import MetricsStore

    source = env if env is not None else _os.environ
    store = MetricsStore(_Path(SNAPSHOT_PATH), _Path(MONTHLY_DIR))
    token = source.get("GITHUB_TOKEN", "")
    repository = source.get("GITHUB_REPOSITORY", "")
    release = None
    if token and "/" in repository:
        owner, repo = repository.split("/", 1)
        release = GithubReleaseClient(
            owner=owner, repo=repo, tag=METRICS_RELEASE_TAG, token=token
        )
    else:
        log.warning(
            "Metrics release skipped: GITHUB_TOKEN/GITHUB_REPOSITORY not set"
        )
    return MetricsReporter(
        store, monthly_dir=MONTHLY_DIR, release=release, job_name=JOB_NAME
    )


def emit_metrics(
    *,
    race_date: str,
    run_id: str,
    started_at: datetime,
    finished_at: datetime,
    status: str,
    counters: Mapping[str, Any],
    errors: Mapping[str, Any],
    reporter_factory: Callable[..., MetricsReporter] = build_metrics_reporter,
    env: Mapping[str, str] | None = None,
) -> None:
    """SystemMetrics を1件出力する。失敗しても呼び出し元へ例外を流さない。"""
    try:
        reporter = reporter_factory(env=env)
        metrics = reporter.build(
            race_date=race_date,
            run_id=run_id,
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            duration_seconds=(finished_at - started_at).total_seconds(),
            status=status,
            counters=counters,
            errors=errors,
        )
        reporter.report(metrics)
        log.info(
            "SystemMetrics written metrics_id=%s status=%s",
            metrics.metrics_id, status,
        )
    except Exception as exc:  # noqa: BLE001 計測の失敗でジョブを変えない（⑥欠損許容）
        log.warning("SystemMetrics output failed: %s", exc)


def main(argv: list[str] | None = None) -> int:
    """CLIエントリ（入力取得・組み立て・呼び出し・終了コードのみ）。

    TARGET_RACES の不備（空・日付混在を含む）/ 認証設定の不備は終了コード1。
    実行完了時は 0
    （Shadow main と同じ規約。status はログに出す）。
    """
    parser = argparse.ArgumentParser(
        description=(
            "Production evaluation/persistence entry: reuse saved "
            "RaceEvaluation or evaluate and persist it (no output, no sending). "
            "Reads TARGET_RACES, GITHUB_TOKEN, GITHUB_REPOSITORY from env."
        )
    )
    parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    from actions.shadow_entrypoint import parse_target_races

    try:
        races = parse_target_races(os.environ.get("TARGET_RACES"))
        race_date = single_race_date(races)
    except ValueError as exc:
        # race_date 未確定（Production処理開始前）はMetrics対象外
        log.error("TARGET_RACES error: %s", exc)
        return 1

    # ここから「Production処理開始」（S6の明示仕様: single_race_date 完了後）。
    # 以降は途中で例外終了しても SystemMetrics を出す（status=failed）。
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    started_at = datetime.now(_JST)
    report: dict | None = None
    metrics_status = "failed"
    metrics_errors: dict[str, Any] = {}
    try:
        try:
            try:
                report = run_entry(races)
                metrics_status = report["status"]
                metrics_errors = build_errors(report["errors"])
            except BaseException as exc:  # noqa: BLE001 Metrics出力後に再送出
                metrics_errors = build_errors(exception=exc)
                raise
        except (ValueError, ConfigError) as exc:
            # 既存の終了コード規約（設定不備は1）を維持する
            log.error(
                "Production evaluation entry configuration error: %s", exc
            )
            return 1
    finally:
        if run_id:
            counters: dict[str, Any] = {"races_found": len(races)}
            if report is not None:
                counters["races_evaluated"] = report["races_evaluated"]
                counters["races_skipped"] = report["s4_excluded_count"]
                counters["records_written"] = report["records_written"]
            emit_metrics(
                race_date=race_date,
                run_id=run_id,
                started_at=started_at,
                finished_at=datetime.now(_JST),
                status=metrics_status,
                counters=counters,
                errors=metrics_errors,
            )
        else:
            log.info("SystemMetrics skipped: GITHUB_RUN_ID is not set")

    reused = sum(1 for r in report["results"] if r.get("evaluation_reused"))
    log.info(
        "Production evaluation entry done status=%s total=%d success=%d "
        "failure=%d reused=%d",
        report["status"], report["total"], report["success_count"],
        report["failure_count"], reused,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
