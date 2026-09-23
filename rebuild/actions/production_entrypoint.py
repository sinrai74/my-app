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
from notification.body_formatter import format_race_body, format_race_subject
from pipelines.notification_request_builder import (
    build_mail_notification_request,
    build_per_race_mail_request,
)
from storage.exceptions import StorageError

log = logging.getLogger(__name__)

# Legacy _evaluate_bets が空listを返したときに PredictionProvider が送出する
# ValueError のメッセージ先頭（Shadowと同じ識別方法。契約側は変更しない）。
_INCOMPARABLE_ERROR_PREFIX = "_evaluate_bets returned an empty list"


def _find_saved_evaluation(evaluation_repository, eval_id: str):
    """保存済み RaceEvaluation を eval_id で1件だけ取得する（無ければ None）。

    既存 EvaluationRepository.load_all() を読むだけ（Repositoryは無改変）。
    - JSONL破損は load_all の ParseError をそのまま送出する（未評価とみなさない）。
    - 同一 eval_id の重複レコードは正常状態ではないため、どれかを選択・修復せず
      StorageError とする（find_by_eval_id の先頭採用は使わない）。
    """
    matches = [
        ev for ev in evaluation_repository.load_all() if ev.eval_id == eval_id
    ]
    if len(matches) > 1:
        raise StorageError(
            f"duplicate eval_id records in evaluations store: {eval_id} "
            f"(count={len(matches)}); not selecting or repairing"
        )
    return matches[0] if matches else None


def run_one_race(
    bundle: PipelineBundle,
    race_date: str,
    venue_num: int,
    race_number: int,
    output_paths: Mapping[str, str],
    *,
    persist: bool = True,
    evaluation_repository=None,
) -> dict[str, Any]:
    """1レースを本番経路で処理する（結線のみ）。

    順序（Evaluate Once）:
      1. evaluation を1回だけ得る:
         - evaluation_repository 注入時: eval_id で保存済み評価を確認し、
           あれば再利用（evaluate_race を呼ばない）。無ければ
           evaluation_pipeline.evaluate_race(..., persist=True) で評価し、
           評価直後に durable_store.append_durably で保存する。
         - 未注入時: 従来どおり evaluate_race(..., persist=persist)。
      2. buy_assessment / buy_decision = buy_pipeline から同一 evaluation で取得
      3. render_results = output_pipeline.render_all(...)   （PublicHtmlRenderer等）
      4. requests = 各 render_result → build_mail_notification_request（mail・固定title）
      5. notification_pipeline.send_all(requests)

    evaluation_repository は bundle の durable_store が書き込む JSONL と同一パスを
    読むこと（通常実行の正本＝checkout された evaluations JSONL）。
    戻り値は各段の結果（テスト・監査用）。値の加工・再評価はしない。
    """
    # 1. RaceEvaluation を1回だけ得る（再利用 or 初回評価＋評価直後保存）
    evaluation = None
    reused = False
    if evaluation_repository is not None:
        if not persist:
            raise ValueError(
                "evaluation_repository requires persist=True "
                "(unevaluated races must be saved immediately after evaluation)"
            )
        # eval_id 採番規則（Phase0.5 L156 / evaluation_pipeline と同一）
        eval_id = f"{race_date}_{venue_num:02d}_{race_number:02d}"
        evaluation = _find_saved_evaluation(evaluation_repository, eval_id)
        reused = evaluation is not None
    if evaluation is None:
        evaluation = bundle.evaluation_pipeline.evaluate_race(
            race_date, venue_num, race_number, persist=persist
        )
    log.info(
        "Production evaluation ready eval_id=%s reused=%s",
        evaluation.eval_id, reused,
    )

    # 2. Buy: 同一 evaluation を渡す（再評価しない）
    #    per-race通知本文（案C）は Prediction を要するため assess_decide_predict
    #    で prediction も受け取る（_evaluate_bets呼び出しは1回のまま・Buyロジック
    #    不変）。未注入構成では assess_race のみ（BuyDecision/predictionはNone）。
    prediction = None
    try:
        buy_assessment, buy_decision, prediction = (
            bundle.buy_pipeline.assess_decide_predict(evaluation)
        )
    except ValueError:
        buy_assessment = bundle.buy_pipeline.assess_race(evaluation)
        buy_decision = None

    # 3. Output: 同一 evaluation の日付で描画（Rendererは再評価しない）
    render_results = bundle.output_pipeline.render_all(
        evaluation.race_date, dict(output_paths)
    )

    # 4. per-race NotificationRequest を組み立てる。
    #    purchased=False（見送り）は通知しない（ユーザー確定）＝requestを作らない。
    #    本文・件名は formatter が evaluation/prediction/decision の確定値を
    #    整形して用意する（本層は評価・計算をしない）。
    requests = []
    if (buy_decision is not None and prediction is not None
            and buy_decision.purchased):
        render_result = next(iter(render_results.values())) if render_results else None
        if render_result is not None:
            body = format_race_body(evaluation, prediction, buy_decision)
            subject = format_race_subject(evaluation)
            requests.append(
                build_per_race_mail_request(
                    render_result,
                    subject=subject,
                    body=body,
                    race_date=evaluation.race_date,
                    venue_num=evaluation.venue_num,
                    race_number=evaluation.race_number,
                )
            )

    # 5. Notification: 送信は Notifier の責務。driverは requests を渡すだけ。
    notification_results = bundle.notification_pipeline.send_all(requests)

    return {
        "eval_id": evaluation.eval_id,
        "evaluation": evaluation,
        "evaluation_reused": reused,
        "buy_assessment": buy_assessment,
        "buy_decision": buy_decision,
        "prediction": prediction,
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


def build_evaluation_repository(jsonl_path: str = "evaluations/production.jsonl"):
    """保存済み評価の確認用に EvaluationRepository を返す（既存クラスを生成するだけ）。

    run_one_race / run_production_day の evaluation_repository に渡す。
    durable_store（build_local_evaluation_store / build_github_evaluation_store）と
    同じ jsonl_path を指定すること（通常実行の正本＝checkout された JSONL）。
    """
    from pathlib import Path

    from storage.repositories.evaluation_repository import EvaluationRepository

    return EvaluationRepository(Path(jsonl_path))


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
    evaluation_repository=None,
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
        active, race_date, venue_num, race_number, output_paths,
        persist=persist, evaluation_repository=evaluation_repository,
    )


def run_production_day(
    target_races: "list[tuple[str, int, int]]",
    output_paths_for: "Callable[[str, int, int], Mapping[str, str]]",
    *,
    bundle: PipelineBundle | None = None,
    persist: bool = False,
    evaluation_repository=None,
) -> "dict[str, Any]":
    """複数レースを本番経路で順に処理する（orchestration＋ジョブレベル
    フェイルセーフ）。

    target_races: (race_date, venue_num, race_number) のリスト。
      shadow_entrypoint.parse_target_races で環境変数から解析した形式と同一。
    output_paths_for: レースごとの出力パス dict を返す関数。

    ジョブレベルのフェイルセーフ（Phase0.5設計固定書 §590-593 確定仕様）:
      - 1レースの処理失敗は記録してスキップし、ジョブ全体は続行する
        （現行 `|| echo 続行` 方針の構造化）。
      - 部分成功の定義: 成功率 80%以上 = partial(WARNING継続) /
        80%未満 = failed(ERROR)。全レース失敗も failed。全レース成功は success。
        判定結果は status（success/partial/failed）として返す。
      - Legacy `_evaluate_bets` が空listを返したレース（買い目候補なし＝Legacy
        戻り値の異常。見送りとは別事象）は incomparable として別枠に記録し、
        通常の失敗（errors/failure_count）には計上しない。Shadowの
        「比較不能」(Step6-2c-12・案A)と同じ事象を、Production側の責務
        （status判定）に合わせて扱う。成功率の分母からも除外する。
      - status は services/orchestration 層で判定する（SystemMetricsモデルは
        判定ロジックを持たない・§19.3）。本関数がその判定を担う。

    ※対象別のネットワークリトライ（GitHub/Gmail等・§585）は各クライアントの
      RetryPolicyが担当する。本関数のフェイルセーフはレース単位のスキップ継続
      とジョブ status 判定であり、_evaluate_bets等の再実行はしない。

    実送信・切替はしない（run_production_race と同じ制約）。

    Returns:
        {
          "status": "success"|"partial"|"failed",
          "total": <対象レース数>,
          "success_count": <成功数>,
          "failure_count": <失敗数>,
          "success_rate": <0.0-1.0>,  # success_count / (total - incomparable_count)
          "incomparable_count": <比較不能数>,
          "results": [<成功レースのrun_one_race結果dict>, ...],
          "errors": [{"race": "date_venue_race", "error": "..."}, ...],
          "incomparable": [{"race": "date_venue_race", "reason": "..."}, ...],
        }
    """
    active = bundle if bundle is not None else build_production_bundle()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    incomparable: list[dict[str, str]] = []
    total = len(target_races)

    for race_date, venue_num, race_number in target_races:
        race_tag = f"{race_date}_{venue_num}_{race_number}"
        try:
            paths = output_paths_for(race_date, venue_num, race_number)
            results.append(
                run_one_race(
                    active, race_date, venue_num, race_number, paths,
                    persist=persist,
                    evaluation_repository=evaluation_repository,
                )
            )
        except ValueError as exc:
            if not str(exc).startswith(_INCOMPARABLE_ERROR_PREFIX):
                log.warning(
                    "Production race failed (skipped, job continues) race=%s error=%s",
                    race_tag, exc,
                )
                errors.append(
                    {"race": race_tag, "error": f"{type(exc).__name__}: {exc}"}
                )
                continue
            log.warning(
                "Production race incomparable (skipped, job continues) race=%s "
                "reason=%s", race_tag, exc,
            )
            incomparable.append({"race": race_tag, "reason": str(exc)})
        except Exception as exc:  # noqa: BLE001 レース単位で記録してスキップ・続行
            log.warning(
                "Production race failed (skipped, job continues) race=%s error=%s",
                race_tag, exc,
            )
            errors.append({"race": race_tag, "error": f"{type(exc).__name__}: {exc}"})

    success_count = len(results)
    failure_count = len(errors)
    incomparable_count = len(incomparable)
    # 成功率の分母は incomparable を除外した対象数（ユーザー確定）
    comparable_total = total - incomparable_count
    success_rate = (success_count / comparable_total) if comparable_total > 0 else 0.0

    # status 判定（§592）: 全成功=success / 80%以上=partial / 80%未満=failed
    # 全件incomparable（分母0かつincomparableあり）は failed（ユーザー確定）。
    # 対象0件（total=0）の扱いは従来どおり変更しない。
    if incomparable_count > 0 and comparable_total <= 0:
        status = "failed"
    elif failure_count == 0:
        status = "success"
    elif success_rate >= 0.8:
        status = "partial"
    else:
        status = "failed"

    log.info(
        "Production day end total=%d success=%d failure=%d incomparable=%d "
        "rate=%.3f status=%s",
        total, success_count, failure_count, incomparable_count,
        success_rate, status,
    )
    return {
        "status": status,
        "total": total,
        "success_count": success_count,
        "failure_count": failure_count,
        "incomparable_count": incomparable_count,
        "success_rate": success_rate,
        "results": results,
        "errors": errors,
        "incomparable": incomparable,
    }


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
