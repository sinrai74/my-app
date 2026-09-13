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
from typing import Any, Mapping

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


def build_production_bundle(
    eval_config: dict | None = None, buy_config: dict | None = None
) -> PipelineBundle:
    """本番用 PipelineBundle を構築する（Output=PublicHtmlRenderer, 通知=実MailNotifier）。

    重い結線（Provider / Engine / Prediction / DurableStore / evaluation・buy
    パイプライン）は既存の shadow_entrypoint.build_bundle をそのまま再利用する
    （Legツールは import して呼ぶだけ・無改変。build_bundle を変更しない）。
    build_bundle は Shadow用に NullNotifier / 空Renderer を組むため、本関数では
    その evaluation_pipeline / buy_pipeline を流用しつつ、output_pipeline と
    notification_pipeline のみ本番用（PublicHtmlRenderer / 実MailNotifier）へ
    差し替えた PipelineBundle を新規に束ねて返す。

    PipelineBundle は frozen dataclass のため、evaluation/buy を流用し
    output/notification を差し替えた新しい PipelineBundle を生成する。
    既存 Pipeline クラスの責務は変更しない（コンストラクタへ渡すだけ）。

    注意: 本関数は「組み立て」のみ。実送信・USE_REBUILD_PIPELINE=True への切替は
    行わない（Step5-0によりGO判定後にのみ許可）。
    """
    from actions.shadow_entrypoint import build_bundle
    from pipelines.notification_pipeline import NotificationPipeline
    from pipelines.output_pipeline import OutputPipeline

    # 1. 既存 build_bundle で重い結線を正しく構築（無改変・そのまま呼ぶ）
    base = build_bundle(eval_config=eval_config, buy_config=buy_config)

    # 2. evaluation / buy はそのまま流用。output / notification のみ本番用へ差し替え。
    return PipelineBundle(
        evaluation_pipeline=base.evaluation_pipeline,
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
    persist: bool = True,
) -> dict[str, Any]:
    """本番起動入口: bundleを構築（または注入）して1レースを処理する。

    bundle 未指定時は build_production_bundle() で本番bundleを構築する。
    テストでは Fake bundle を注入して実送信なしで検証する。
    実送信・USE_REBUILD_PIPELINE=True への切替はしない（GO後限定）。
    """
    active = bundle if bundle is not None else build_production_bundle()
    return run_one_race(
        active, race_date, venue_num, race_number, output_paths, persist=persist
    )


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
