"""
shadow/aggregator.py（Step6-2c-10）: Shadow実行結果の集計。

責務:
  - ShadowRunner.run_and_compare() の結果（ShadowRunResult）を
    ConsecutiveMatchCounter が扱える StagedResult へ変換する。
  - 複数レース分の実行結果を記録し、連続一致数（streak）を集約する。
  - GoNoGoCriteria へ渡す shadow_consecutive_matches を提供する。

禁止:
  - 差分の補正・スコア変更・既存ロジック修正。
  - runner.py / staged_comparator.py / go_no_go.py 本体の変更
    （本モジュールは既存実装を「呼ぶだけ」の結線層）。

設計根拠（Step6-2c-10 設計判断）:
  D-1: 既存比較ロジック（staged_comparator.py / runner.py）を変更せず、
       本ファイルに変換・集計責務を置く。
  D-2: Shadow専用Adapter方式。ShadowRunResult → StagedResult へ変換する。
       matched_stages / skipped_stages は今回は空で保持する
       （ShadowRunResult は段別の一致・スキップ情報を持たないため、
        推測で埋めない）。
  D-3: diffs の内容解析は行わない。diffs の有無のみで一致を判定する。
       diffs 空 → stopped_at=None（all_matched=True）
       diffs あり → stopped_at="diff"（all_matched=False）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from shadow.staged_comparator import ConsecutiveMatchCounter, StagedResult

log = logging.getLogger(__name__)

# diffs があるときに StagedResult.stopped_at へ入れる値。
# D-3により diffs の内容解析はしないため、段名ではなく固定値を用いる
# （「どの段で止まったか」は ShadowRunResult からは判定できない）。
DIFF_SENTINEL = "diff"


def to_staged_result(run_result: Any) -> StagedResult:
    """ShadowRunResult を StagedResult へ変換する（D-2）。

    ShadowRunResult は段別の一致・スキップ情報を持たないため、
    matched_stages / skipped_stages は空のまま返す（仮値で埋めない）。
    一致判定は diffs の有無のみで行う（D-3）。
    """
    diffs = list(run_result.diffs)
    return StagedResult(
        eval_id=run_result.eval_id,
        matched_stages=[],
        stopped_at=None if not diffs else DIFF_SENTINEL,
        diffs=diffs,
        skipped_stages={},
    )


@dataclass
class ShadowAggregate:
    """複数レース分のShadow実行結果の集約値（1実行内で完結）。"""

    total_races: int = 0
    matched_races: int = 0
    diff_races: int = 0
    current_streak: int = 0
    max_streak: int = 0
    broken_at: list[str] = field(default_factory=list)
    eval_ids: list[str] = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        """人が読める形で集約結果を返す（Shadowログ用）。"""
        lines = [
            f"total_races          {self.total_races}",
            f"matched_races        {self.matched_races}",
            f"diff_races           {self.diff_races}",
            f"current_streak       {self.current_streak}",
            f"max_streak           {self.max_streak}",
        ]
        if self.broken_at:
            lines.append(f"broken_at            {', '.join(self.broken_at)}")
        return lines


class ShadowAggregator:
    """Shadow実行結果を記録し、連続一致数を集約する（1実行内で完結）。

    D-5（GitHub Releasesでのstreak永続化）は本Stepの対象外。
    100レースを1実行で処理し、その実行内で streak を算出する方式のため、
    実行間の永続化は行わない。
    """

    def __init__(self, required: int = 100) -> None:
        self._counter = ConsecutiveMatchCounter(required=required)
        self._eval_ids: list[str] = []
        self._matched = 0
        self._diff = 0

    def record(self, run_result: Any) -> StagedResult:
        """1レース分のShadow実行結果を記録する。変換後のStagedResultを返す。"""
        staged = to_staged_result(run_result)
        self._counter.record(staged)
        self._eval_ids.append(staged.eval_id)
        if staged.all_matched:
            self._matched += 1
        else:
            self._diff += 1
        log.info(
            "Shadow aggregate record eval_id=%s matched=%s streak=%d",
            staged.eval_id, staged.all_matched, self._counter.current_streak,
        )
        return staged

    @property
    def consecutive_matches(self) -> int:
        """GoNoGoCriteria.shadow_consecutive_matches へ渡す値。"""
        return self._counter.current_streak

    @property
    def satisfied(self) -> bool:
        """required件の連続一致に到達しているか。"""
        return self._counter.satisfied

    @property
    def counter(self) -> ConsecutiveMatchCounter:
        return self._counter

    def aggregate(self) -> ShadowAggregate:
        """集約結果を返す。"""
        return ShadowAggregate(
            total_races=self._counter.total_seen,
            matched_races=self._matched,
            diff_races=self._diff,
            current_streak=self._counter.current_streak,
            max_streak=self._counter.max_streak,
            broken_at=list(self._counter.broken_at),
            eval_ids=list(self._eval_ids),
        )
