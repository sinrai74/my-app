"""
段階的Shadow比較（Step6-1）: 指定順で止めながら比較する。

順序（Step6-1レビュー確定）:
  Race → FeatureSet → RaceEvaluation → Prediction → BuyAssessment →
  Output → NotificationRequest
前段に差分があれば後段は比較しない（Raceが違う状態でPredictionだけ
合わせても意味がないため）。比較そのものは shadow.comparator.compare を
再利用する（新しい比較方式は追加しない）。

実装範囲（Step6-1レビュー反映・正確な現状）:
  本モジュールは7段階すべてを比較する機能を持つが、Legacy実データからの
  取得が実装済みなのは以下4段階のみ（取得元: sent_*.txt）:
    Race / RaceEvaluation / Prediction / BuyAssessment
  以下3段階はLegacy側の取得元が未確定のため、Step6-2で取得元を確定した
  うえで実比較を実装する（現時点では比較器が型を扱えることのみ検証済み）:
    FeatureSet / RenderResult(Output) / NotificationRequest
  compare_stagedは両側に存在する段のみ比較するため、未取得の段は
  自動的にスキップされる（欠損を差分として誤検出しない）。

Shadowは「違う」と記録するだけ。補正・特例・Legacyへ合わせる処理はしない。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from shadow.comparator import compare

log = logging.getLogger(__name__)

# 比較順序（前段が一致した場合のみ次段へ進む）
STAGE_ORDER: tuple[str, ...] = (
    "race", "feature_set", "evaluation", "prediction",
    "buy_assessment", "output", "notification_request",
)


@dataclass
class StagedResult:
    """段階比較の結果。どの段で止まったか・差分内容・スキップ理由を保持する。"""

    eval_id: str
    matched_stages: list[str] = field(default_factory=list)
    stopped_at: Optional[str] = None
    diffs: list[dict] = field(default_factory=list)
    # 段名 -> スキップ理由（"取得元なし" 等）。単に飛ばさず理由を必ず残す。
    skipped_stages: dict[str, str] = field(default_factory=dict)

    @property
    def all_matched(self) -> bool:
        return self.stopped_at is None

    def summary_lines(self) -> list[str]:
        """段階ごとの結果を人が読める形で返す（Shadowログ用）。

        例:
          race                 ✔
          feature_set          ✔
          evaluation           ✔
          prediction           ✔
          buy_assessment       SKIPPED（取得元なし）
          output               ✔
          notification_request ✔
        """
        lines: list[str] = []
        for stage in STAGE_ORDER:
            if stage in self.skipped_stages:
                lines.append(f"{stage:<20} SKIPPED（{self.skipped_stages[stage]}）")
            elif stage in self.matched_stages:
                lines.append(f"{stage:<20} ✔")
            elif stage == self.stopped_at:
                lines.append(f"{stage:<20} DIFF（{len(self.diffs)}件で停止）")
            else:
                lines.append(f"{stage:<20} NOT_REACHED")
        return lines


def compare_staged(
    eval_id: str,
    legacy: dict[str, Any],
    rebuild: dict[str, Any],
) -> StagedResult:
    """STAGE_ORDER順に比較し、最初に差分が出た段で停止する。

    legacy/rebuild: 段名（"race"等）→ 比較対象オブジェクト の辞書。
      両方に存在する段のみ比較する。片方に無い段はスキップするが、
      「取得元なし」等の理由を skipped_stages へ必ず記録する
      （単に飛ばさず、比較していない事実をログへ残すため）。
    """
    result = StagedResult(eval_id=eval_id)
    for stage in STAGE_ORDER:
        in_legacy = stage in legacy
        in_rebuild = stage in rebuild
        if not in_legacy or not in_rebuild:
            if not in_legacy and not in_rebuild:
                reason = "取得元なし（legacy/rebuild両方）"
            elif not in_legacy:
                reason = "取得元なし（legacy側）"
            else:
                reason = "取得元なし（rebuild側）"
            result.skipped_stages[stage] = reason
            log.info(
                "Shadow staged compare SKIPPED eval_id=%s stage=%s reason=%s",
                eval_id, stage, reason,
            )
            continue
        stage_diffs = compare(
            eval_id, legacy[stage], rebuild[stage], path=f"$.{stage}"
        )
        if stage_diffs:
            result.diffs.extend(stage_diffs)
            result.stopped_at = stage
            log.info(
                "Shadow staged compare stopped eval_id=%s stage=%s diffs=%d",
                eval_id, stage, len(stage_diffs),
            )
            return result
        result.matched_stages.append(stage)
        log.info("Shadow staged compare matched eval_id=%s stage=%s", eval_id, stage)
    log.info(
        "Shadow staged compare finished eval_id=%s matched=%d skipped=%d",
        eval_id, len(result.matched_stages), len(result.skipped_stages),
    )
    return result


@dataclass
class ConsecutiveMatchCounter:
    """100レース連続一致を数える（Step6-1レビュー(d)）。

    「連続」の定義: Shadow対象100レース連続で全段一致。途中で1件でも
    差分が出たら0から数え直し（100/105ではなく100連続）。
    """

    required: int = 100
    current_streak: int = 0
    max_streak: int = 0
    total_seen: int = 0
    broken_at: list[str] = field(default_factory=list)

    def record(self, result: StagedResult) -> None:
        self.total_seen += 1
        if result.all_matched:
            self.current_streak += 1
            self.max_streak = max(self.max_streak, self.current_streak)
        else:
            # 差分が出たら0へリセット（数え直し）
            self.broken_at.append(result.eval_id)
            self.current_streak = 0

    @property
    def satisfied(self) -> bool:
        """現在の連続一致数が必要数に達しているか。"""
        return self.current_streak >= self.required


@dataclass
class ShadowSummary:
    """Shadow並走の集計（100レース流した後の分析用）。

    段階ごとの一致数・スキップ数・最初の差分を集計する。
    集計するだけで補正・判定はしない（Shadowの責務内）。
    """

    processed_races: int = 0
    matched_per_stage: dict[str, int] = field(default_factory=dict)
    skipped_per_stage: dict[str, int] = field(default_factory=dict)
    diff_per_stage: dict[str, int] = field(default_factory=dict)
    first_mismatch: Optional[dict] = None

    def record(self, result: StagedResult) -> None:
        """1レース分の段階比較結果を集計へ加える。"""
        self.processed_races += 1
        for stage in result.matched_stages:
            self.matched_per_stage[stage] = self.matched_per_stage.get(stage, 0) + 1
        for stage in result.skipped_stages:
            self.skipped_per_stage[stage] = self.skipped_per_stage.get(stage, 0) + 1
        if result.stopped_at is not None:
            stage = result.stopped_at
            self.diff_per_stage[stage] = self.diff_per_stage.get(stage, 0) + 1
            if self.first_mismatch is None and result.diffs:
                head = result.diffs[0]
                self.first_mismatch = {
                    "eval_id": result.eval_id,
                    "stage": stage,
                    "field_path": head.get("field_path"),
                    "legacy": head.get("legacy"),
                    "rebuild": head.get("rebuild"),
                }

    def summary_lines(self) -> list[str]:
        """集計結果を人が読める形で返す（Shadow運用の分析用）。

        例:
          Processed races: 100

          race                 matched: 100
          feature_set          matched: 98  diff: 2
          ...
          buy_assessment       Pending (skipped: 100)

          First mismatch:
            eval_id: 20260704_12_05
            stage: prediction
            field: $.prediction.pred_prob
        """
        lines = [f"Processed races: {self.processed_races}", ""]
        for stage in STAGE_ORDER:
            matched = self.matched_per_stage.get(stage, 0)
            skipped = self.skipped_per_stage.get(stage, 0)
            diffs = self.diff_per_stage.get(stage, 0)
            if skipped and not matched and not diffs:
                lines.append(f"{stage:<20} Pending (skipped: {skipped})")
                continue
            parts = [f"matched: {matched}"]
            if diffs:
                parts.append(f"diff: {diffs}")
            if skipped:
                parts.append(f"skipped: {skipped}")
            lines.append(f"{stage:<20} " + "  ".join(parts))
        if self.first_mismatch is not None:
            lines.extend([
                "",
                "First mismatch:",
                f"  eval_id: {self.first_mismatch['eval_id']}",
                f"  stage: {self.first_mismatch['stage']}",
                f"  field: {self.first_mismatch['field_path']}",
                f"  legacy: {self.first_mismatch['legacy']}",
                f"  rebuild: {self.first_mismatch['rebuild']}",
            ])
        return lines
