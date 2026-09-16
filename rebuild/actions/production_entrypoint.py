"""
Production driver（薄い結線のみ・新しい業務ロジックを作らない）:
既存部品を組み合わせて、1レースについて
  入力 → Evaluation(persist) → Buy → Output(PublicHtmlRenderer)
       → NotificationRequest(mail) → Notification(MailNotifier)
を駆動する。

厳守:
  - 新しいPipeline / 新しい評価・購入ロジックは作らない。
  - RaceEvaluation は1レースにつき1回だけ生成し、同一インスタンスを
    Buy / Output / Notification へ渡す（Evaluate Once, Publish Everywhere）。
    Output/Notification で再評価しない。
  - Legacy / Feature Freeze対象 / 各モデル / 既存Pipelineの責務は変更しない。
  - 実送信は行わない（本モジュールは送信可否を判定しない）。
    実運用切替（USE_REBUILD_PIPELINE=True）はGO判定後にのみ許可される
    （Step5-0）。本driverはコード経路の結線までであり、切替はしない。

テスト容易性:
  run_one_race(bundle, ...) は PipelineBundle を受け取り、Fake bundle を
  注入して実送信なしで検証できる。本番組立は assemble_production_bundle()。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

from actions.wiring import PipelineBundle
from pipelines.notification_request_builder import build_mail_notification_request

log = logging.getLogger(__name__)


def run_one_race(
    bundle: PipelineBundle,
    race_date: str,
    venue_num: int,
    race_number: int,
    output_paths: Mapping[str, str],
    *,
    persist: bool = True,
) -> dict[str, Any]:
    """1レースを本番経路で処理する（結線のみ）。

    順序（Evaluate Once）:
      1. evaluation = evaluation_pipeline.evaluate_race(..., persist=persist)
         ※ RaceEvaluation はここで1回だけ生成・保存される
      2. buy_assessment / buy_decision = buy_pipeline から同一 evaluation で取得
      3. render_results = output_pipeline.render_all(...)   （PublicHtmlRenderer等）
      4. requests = 各 render_result → build_mail_notification_request（mail・固定title）
      5. notification_pipeline.send_all(requests)

    戻り値は各段の結果（テスト・監査用）。値の加工・再評価はしない。
    """
    # 1. RaceEvaluation を1回だけ生成（persist=True で DurableStore へ保存）
    evaluation = bundle.evaluation_pipeline.evaluate_race(
        race_date, venue_num, race_number, persist=persist
    )
    log.info("Production evaluate done eval_id=%s", evaluation.eval_id)

    # 2. Buy: 同一 evaluation を渡す（再評価しない）
    #    DI（purchase_result_source / decision_builder）が揃っていれば
    #    assess_and_decide で BuyAssessment と BuyDecision を1回で得る。
    #    未注入構成では assess_race のみ（BuyDecisionはNone）。
    try:
        buy_assessment, buy_decision = (
            bundle.buy_pipeline.assess_and_decide(evaluation)
        )
    except ValueError:
        buy_assessment = bundle.buy_pipeline.assess_race(evaluation)
        buy_decision = None

    # 3. Output: 同一 evaluation の日付で描画（Rendererは再評価しない）
    render_results = bundle.output_pipeline.render_all(
        evaluation.race_date, dict(output_paths)
    )

    # 4. RenderResult → NotificationRequest（mail・固定title・destination未設定）
    requests = [
        build_mail_notification_request(render_result)
        for render_result in render_results.values()
    ]

    # 5. Notification: 送信は Notifier の責務。driverは requests を渡すだけ。
    notification_results = bundle.notification_pipeline.send_all(requests)

    return {
        "eval_id": evaluation.eval_id,
        "evaluation": evaluation,
        "buy_assessment": buy_assessment,
        "buy_decision": buy_decision,
        "render_results": render_results,
        "requests": requests,
        "notification_results": notification_results,
    }


def build_local_evaluation_store(jsonl_path: str = "evaluations/production.jsonl"):
    """ローカルのみで完結する DurableEvaluationStore を構築する（送信・push なし）。

    本番の DurableEvaluationStore は EvaluationRepository（ローカルJSONL）＋
    ReleaseClient（GitHub Release）＋ GitClient（commit/push）を要するが、
    ここでは Release/Git を no-op クライアントに差し替え、**ローカルファイルへの
    追記だけ**を行う store を返す。ネットワーク・GitHub認証・実push は発生しない。

    本番運用で GitHub Release / commit まで行いたい場合は、実行環境側で
    実クライアントを注入した DurableEvaluationStore を build_production_bundle の
    durable_store 引数へ渡すこと（本関数はローカル永続の最小構成）。
    """
    from pathlib import Path

    from storage.durability import DurableEvaluationStore
    from storage.repositories.evaluation_repository import EvaluationRepository

    class _NoopReleaseClient:
        def upload_asset(self, file_path, asset_name): pass
        def download_asset(self, asset_name, dest_path): pass
        def list_assets(self): return []
        def delete_asset(self, asset_name): pass

    class _NoopGitClient:
        def commit_and_push(self, paths, message): pass

    repository = EvaluationRepository(Path(jsonl_path))
    return DurableEvaluationStore(
        repository=repository, release=_NoopReleaseClient(), git=_NoopGitClient()
    )


def build_github_evaluation_store(
    jsonl_path: str = "evaluations/production.jsonl",
    *,
    repo_dir: str = ".",
    tag: str = "data-store",
    env: "Mapping[str, str] | None" = None,
):
    """本番用 DurableEvaluationStore を既存部品の結線だけで構成する。

    既存部品をそのまま接続する（新しいPersistence設計は作らない）:
      - EvaluationRepository（ローカルJSONL）
      - GithubReleaseClient（owner/repo/tag/token）
      - SubprocessGitClient（repo_dir）
      - DurableEvaluationStore（上記3つを束ねる）

    Credentials（token・repo）はハードコードせず、環境変数から取得する
    （既存workflow notify_arashi.yml と同じ GITHUB_TOKEN / GITHUB_REPOSITORY
    を利用）。env 引数は主にテスト用（未指定時は os.environ）。
    GITHUB_REPOSITORY は "owner/repo" 形式（GitHub Actions標準）を分解する。

    本関数は「構成」のみ。GitHub Releasesへの実push・実アップロードは
    append_durably 呼び出し時に初めて発生し、本関数自体はネットワークに触れない。
    必要な認証が env に無い場合は ValueError（穴埋め・仮値は使わない）。
    """
    import os
    from pathlib import Path

    from storage.clients.git_client import SubprocessGitClient
    from storage.clients.github_release_client import GithubReleaseClient
    from storage.durability import DurableEvaluationStore
    from storage.repositories.evaluation_repository import EvaluationRepository

    source = env if env is not None else os.environ
    token = source.get("GITHUB_TOKEN", "")
    repository = source.get("GITHUB_REPOSITORY", "")  # "owner/repo"
    if not token or "/" not in repository:
        raise ValueError(
            "build_github_evaluation_store requires GITHUB_TOKEN and "
            "GITHUB_REPOSITORY (owner/repo) in the environment; "
            "credentials are never hardcoded"
        )
    owner, repo = repository.split("/", 1)

    release = GithubReleaseClient(owner=owner, repo=repo, tag=tag, token=token)
    git = SubprocessGitClient(repo_dir=Path(repo_dir))
    repo_store = EvaluationRepository(Path(jsonl_path))
    return DurableEvaluationStore(repository=repo_store, release=release, git=git)


def build_production_bundle(
    eval_config: dict | None = None,
    buy_config: dict | None = None,
    durable_store=None,
) -> PipelineBundle:
    """本番用 PipelineBundle を構築する（Output=PublicHtmlRenderer, 通知=実MailNotifier）。

    重い結線（Provider / Engine / Prediction / evaluation・buy パイプライン）は
    既存の shadow_entrypoint.build_bundle をそのまま再利用する（Legツールは
    import して呼ぶだけ・無改変。build_bundle を変更しない）。build_bundle は
    Shadow用に NullNotifier / 空Renderer / durable_store=None を組むため、本関数
    では、その evaluation_pipeline / buy_pipeline を流用しつつ、output_pipeline と
    notification_pipeline のみ本番用へ差し替えた PipelineBundle を新規に束ねる。

    durable_store を渡した場合は、base.evaluation_pipeline と同一の内部部品
    （race_source / feature_builder / engine / now_provider / config）を用いて
    EvaluationPipeline を再構築し、durable_store を注入する（build_bundle は
    durable_store=None のため、persist=True で保存するには本経路が必要）。
    build_bundle の重い結線を複製せず、その公開済みパイプラインの部品を再利用
    するだけ（新しい業務ロジックは作らない）。

    注意: 本関数は「組み立て」のみ。実送信・USE_REBUILD_PIPELINE=True への切替は
    行わない（Step5-0によりGO判定後にのみ許可）。
    """
    from actions.shadow_entrypoint import build_bundle
    from pipelines.evaluation_pipeline import EvaluationPipeline
    from pipelines.notification_pipeline import NotificationPipeline
    from pipelines.output_pipeline import OutputPipeline

    # 1. 既存 build_bundle で重い結線を正しく構築（無改変・そのまま呼ぶ）
    base = build_bundle(eval_config=eval_config, buy_config=buy_config)

    # 2. durable_store 指定時は、base の EvaluationPipeline と同一部品で
    #    durable_store を注入した EvaluationPipeline を再構築する。
    #    未指定時は base の evaluation_pipeline をそのまま流用（persist=False運用）。
    if durable_store is not None:
        base_eval = base.evaluation_pipeline
        evaluation_pipeline = EvaluationPipeline(
            race_source=base_eval._race_source,
            feature_builder=base_eval._feature_builder,
            engine=base_eval._engine,
            now_provider=base_eval._now_provider,
            config=base_eval._config,
            durable_store=durable_store,
        )
    else:
        evaluation_pipeline = base.evaluation_pipeline

    # 3. output / notification のみ本番用へ差し替え。
    return PipelineBundle(
        evaluation_pipeline=evaluation_pipeline,
        buy_pipeline=base.buy_pipeline,
        output_pipeline=OutputPipeline(production_output_renderers()),
        notification_pipeline=NotificationPipeline(
            production_notification_service()
        ),
    )


def run_production_race(
    race_date: str,
    venue_num: int,
    race_number: int,
    output_paths: Mapping[str, str],
    *,
    bundle: PipelineBundle | None = None,
    persist: bool = False,
) -> dict[str, Any]:
    """本番起動入口: bundleを構築（または注入）して1レースを処理する。

    bundle 未指定時は build_production_bundle() で本番bundleを構築する。
    persist は既定 False（durable_store を注入していない bundle で
    persist=True にすると EvaluationPipeline が ValueError を送出するため）。
    ローカル永続を行う場合は build_production_bundle(durable_store=
    build_local_evaluation_store()) で bundle を構築し persist=True を渡すこと。
    テストでは Fake bundle を注入して実送信なしで検証する。
    実送信・USE_REBUILD_PIPELINE=True への切替はしない（GO後限定）。
    """
    active = bundle if bundle is not None else build_production_bundle()
    return run_one_race(
        active, race_date, venue_num, race_number, output_paths, persist=persist
    )


def run_production_day(
    target_races: "list[tuple[str, int, int]]",
    output_paths_for: "Callable[[str, int, int], Mapping[str, str]]",
    *,
    bundle: PipelineBundle | None = None,
    persist: bool = False,
) -> "list[dict[str, Any]]":
    """複数レースを本番経路で順に処理する（orchestration・結線のみ）。

    target_races: (race_date, venue_num, race_number) のリスト。
      shadow_entrypoint.parse_target_races で環境変数から解析した形式と同一。
    output_paths_for: レースごとの出力パス dict を返す関数。

    bundle を1つ構築して全レースで使い回す（評価・購入・出力・通知は各レース
    ごとに実行）。Evaluate Once は run_one_race 内で1レース1回保証される。
    実送信・切替はしない（run_production_race と同じ制約）。
    各レースの結果 dict をリストで返す（監査・テスト用）。
    """
    active = bundle if bundle is not None else build_production_bundle()
    results: list[dict[str, Any]] = []
    for race_date, venue_num, race_number in target_races:
        paths = output_paths_for(race_date, venue_num, race_number)
        results.append(
            run_one_race(
                active, race_date, venue_num, race_number, paths, persist=persist
            )
        )
    return results


def production_output_renderers() -> dict:
    """本番Output構成（PublicHtmlRenderer単独）を返す。

    確定方針: 本番Rendererは PublicHtmlRenderer。OutputPipeline へ渡す
    {名前: Renderer} を返すだけ（新しいRendererは作らない）。
    """
    from output.renderers import PublicHtmlRenderer

    return {"public": PublicHtmlRenderer("public")}


def production_notification_service():
    """本番Notification構成（Mail単独・実MailNotifier）を返す。

    確定方針: 通知は Mail 単独。NotificationService に実 MailNotifier を登録
    する（NullNotifierは使わない＝本番向け）。宛先は MailNotifier →
    notify_arashi.send_email() の内部設定へ委譲するため destination は不要。

    注意: これは「登録」だけである。実際の送信可否・実運用切替
    （USE_REBUILD_PIPELINE=True）はGO判定後にのみ許可される（Step5-0）。
    本関数は本番用Notifierを組み込む結線を提供するに留まる。
    """
    from notification.notifiers import MailNotifier
    from notification.service import NotificationService

    return NotificationService({"mail": MailNotifier()})
