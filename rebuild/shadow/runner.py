"""
ShadowRunner（Step5-6）: Pipelineを最後まで実行し、legacy値と比較するだけ。

責務:
  - PipelineBundleを使ってrebuild側を実行（Evaluation→Buy→Output→
    Notification）。NotificationはNullNotifier必須（実送信を物理的に禁止）。
  - 呼び出し側が用意した legacy_values（辞書: 対象名→legacyオブジェクト）
    と rebuild側の結果を shadow.comparator.compare で突き合わせる。

禁止: 差分の補正・スコア変更・既存ロジック修正。差分はそのまま報告する。

比較対象（指示）: Race / FeatureSet / RaceEvaluation / Prediction /
  BuyAssessment / Output(RenderResult) / NotificationRequest。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from actions.wiring import PipelineBundle
from notification.notifiers import NotificationRequest
from shadow.comparator import compare
from shadow.notifier import NullNotifier

log = logging.getLogger(__name__)


@dataclass
class ShadowRunResult:
    """1レース分のShadow実行結果（rebuild側の各段の値＋比較差分）。"""

    eval_id: str
    race: Any
    feature_set: Any
    evaluation: Any
    buy_assessment: Any
    render_results: dict
    notification_requests: list
    diffs: list[dict]


class ShadowRunner:
    """rebuild側をPipelineBundle経由で実行し、legacy値と比較するだけ。"""

    def __init__(self, bundle: PipelineBundle) -> None:
        self._bundle = bundle

    def run_and_compare(
        self,
        race_date: str,
        venue_num: int,
        race_number: int,
        output_paths: Mapping[str, str],
        legacy_values: Optional[Mapping[str, Any]] = None,
        notification_requests: Optional[list[NotificationRequest]] = None,
    ) -> ShadowRunResult:
        """Pipelineを最後まで実行し、legacy_valuesとの差分を集める。

        legacy_values のキー: "race" / "feature_set" / "evaluation" /
          "prediction"（BuyPipeline内部生成のため直接は取得できない場合あり）/
          "buy_assessment" / "render_results" / "notification_requests"
        いずれも省略可（未指定の対象は比較しない＝呼び出し側の用意次第）。
        """
        eval_id = f"{race_date}_{venue_num:02d}_{race_number:02d}"
        log.info("Shadow compare start eval_id=%s", eval_id)

        # ---- rebuild実行（Pipelineは「呼ぶだけ」。ShadowRunnerも計算しない） ----
        race = self._bundle.evaluation_pipeline._race_source.resolve_race(
            race_date, venue_num, race_number
        )
        evaluation = self._bundle.evaluation_pipeline.evaluate_race(
            race_date, venue_num, race_number, persist=False
        )
        buy_assessment = self._bundle.buy_pipeline.assess_race(evaluation)
        render_results = self._bundle.output_pipeline.render_all(
            race_date, output_paths
        )

        # Notificationは実送信禁止。NullNotifier以外が登録されていたら停止する
        # （サイレントに実送信させない＝指示3・6の担保）。
        self._assert_all_null_notifiers()
        sent_requests = notification_requests or []
        notification_results = self._bundle.notification_pipeline.send_all(
            sent_requests
        )
        for result in notification_results:
            if result.sent:
                raise RuntimeError(
                    f"Shadow real-send detected: eval_id={eval_id} "
                    f"channel={result.channel} (must be 0 in Shadow)"
                )

        # ---- 比較（差分は原因調査対象。ここで補正はしない） ----
        diffs: list[dict] = []
        legacy_values = legacy_values or {}
        pairs = (
            ("race", race), ("evaluation", evaluation),
            ("buy_assessment", buy_assessment),
        )
        for name, rebuild_obj in pairs:
            if name in legacy_values:
                diffs.extend(
                    compare(eval_id, legacy_values[name], rebuild_obj,
                            path=f"$.{name}")
                )
        if "render_results" in legacy_values:
            diffs.extend(
                compare(eval_id, legacy_values["render_results"], render_results,
                        path="$.render_results")
            )
        if "notification_requests" in legacy_values:
            diffs.extend(
                compare(eval_id, legacy_values["notification_requests"],
                        sent_requests, path="$.notification_requests")
            )

        log.info(
            "Shadow compare end eval_id=%s diff_count=%d",
            eval_id, len(diffs),
        )
        return ShadowRunResult(
            eval_id=eval_id, race=race, feature_set=None, evaluation=evaluation,
            buy_assessment=buy_assessment, render_results=render_results,
            notification_requests=sent_requests, diffs=diffs,
        )

    def _assert_all_null_notifiers(self) -> None:
        notifiers = self._bundle.notification_pipeline._service._notifiers
        for channel, notifier in notifiers.items():
            if not isinstance(notifier, NullNotifier):
                raise RuntimeError(
                    f"Shadow requires NullNotifier for channel={channel!r}, "
                    f"got {type(notifier).__name__} (real send risk)"
                )
