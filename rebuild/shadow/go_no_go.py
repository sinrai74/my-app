"""
Go/No-Go判定（Step5-6）: Step5-0で定義した数値基準を集約し判定するだけ。

判定式そのものは新規ロジックではなく、Step5-0計画書のG1-G8基準の
機械的な集約（すべてtrue/条件充足ならGO）。スコア・閾値・評価式には
一切触れない（AIロジックではなく運用判定）。

整理（レビュー反映）: 各条件の判定式とNG理由文を _CHECKS 定数へ集約し、
evaluate_go_no_go は1つのループで全条件を評価してNG理由を列挙する。
判定内容・数値・仕様は変更していない。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class GoNoGoCriteria:
    """Step5-0 G1-G8に対応する入力値（すべて呼び出し側が計測して渡す）。"""

    golden_100_percent: bool          # G1
    shadow_consecutive_matches: int   # G2/G3の連続一致数
    shadow_required_matches: int = 100
    output_byte_match: bool = True    # G5
    notification_request_match: bool = True
    shadow_real_sends: int = 0        # G6: 0でなければならない
    feature_freeze_intact: bool = True  # G8
    all_tests_passed: bool = True     # 全テスト成功
    # BuyAssessment比較の状態（Step6-2レビュー反映）。
    # sent_*.txtにbuyscoreが記録されていない（実データ集計で0件）ため、
    # 現状は取得元未確定＝Pending。「一致」と称さない。
    # 取得元が確定したらTrueにして有効化し、buy_assessment_matchで判定する。
    buy_assessment_source_available: bool = False
    buy_assessment_match: bool = False


@dataclass(frozen=True)
class GoNoGoResult:
    decision: str  # "GO" or "NO_GO"
    reasons: tuple[str, ...]  # NO_GO時の未充足条件一覧
    # 取得元未確定などで判定対象外の項目（"一致"と称さず記録する）
    pendings: tuple[str, ...] = ()

    def decision_line(self) -> str:
        """人向けの1行表示を返す（表示専用）。

        Returns a human-readable summary. This method does NOT affect the
        GO/NO_GO decision logic — 判定は evaluate_go_no_go が行い、本メソッドは
        既にある decision / pendings を整形して返すだけである。
        判定値そのものが必要な場合は decision フィールドを参照すること。

        出力例:
          GO
          GO (Pending: BuyAssessment source unavailable)
          NO_GO
        """
        if self.decision != "GO" or not self.pendings:
            return self.decision
        labels = ", ".join(_pending_label(p) for p in self.pendings)
        return f"{self.decision} (Pending: {labels})"


def _pending_label(pending: str) -> str:
    """Pending説明文から表示用の短いラベルを取り出す（表示専用）。

    説明文は "BuyAssessment: 取得元未確定のため..." の形式のため、
    先頭のコロンまでを項目名として使う（判定には一切関与しない）。
    """
    name = pending.split(":", 1)[0].strip()
    return f"{name} source unavailable" if name else pending


# 判定チェック定数（充足判定関数, NG理由生成関数）の集約点。
# 判定内容・数値はStep5-0のG1-G8から不変。
_CHECKS: tuple[tuple[Callable[[GoNoGoCriteria], bool],
                     Callable[[GoNoGoCriteria], str]], ...] = (
    (lambda c: c.golden_100_percent,
     lambda c: "G1: Golden回帰が100%一致していない"),
    (lambda c: c.shadow_consecutive_matches >= c.shadow_required_matches,
     lambda c: (f"G2/G3: Shadow連続一致が{c.shadow_consecutive_matches}件"
                f"（必要{c.shadow_required_matches}件）")),
    (lambda c: c.output_byte_match,
     lambda c: "G5: OutputのHTML/CSV/TXT byte一致が崩れている"),
    (lambda c: c.notification_request_match,
     lambda c: "NotificationRequestの一致が崩れている"),
    (lambda c: c.shadow_real_sends == 0,
     lambda c: f"G6: Shadow中に実送信が{c.shadow_real_sends}件検出された"),
    (lambda c: c.feature_freeze_intact,
     lambda c: "G8: Feature Freeze対象に変更が検出された"),
    (lambda c: c.all_tests_passed,
     lambda c: "全テストが成功していない"),
    # BuyAssessmentは取得元が確定している場合のみ判定対象。
    # 未確定時はPending扱いとし、GO/NO_GOの判定材料にしない。
    (lambda c: (not c.buy_assessment_source_available) or c.buy_assessment_match,
     lambda c: "BuyAssessmentの一致が崩れている"),
)

# Pending判定（判定対象外であることを記録するための定数）
_PENDINGS: tuple[tuple[Any, Any], ...] = (
    (lambda c: not c.buy_assessment_source_available,
     lambda c: ("BuyAssessment: 取得元未確定のため判定対象外"
                "（sent_*.txtにbuyscore記録なし。取得元確定後に有効化）")),
)


def evaluate_go_no_go(criteria: GoNoGoCriteria) -> GoNoGoResult:
    """全条件充足でGO、1つでも未充足ならNO_GO（Step5-0 G1-G8の機械判定）。

    判定ロジックは _CHECKS を1ループで評価するだけ（数値・仕様は不変）。
    NG理由はすべて列挙する。
    """
    reasons = [
        reason(criteria)
        for predicate, reason in _CHECKS
        if not predicate(criteria)
    ]
    pendings = tuple(
        note(criteria)
        for predicate, note in _PENDINGS
        if predicate(criteria)
    )
    if reasons:
        return GoNoGoResult(
            decision="NO_GO", reasons=tuple(reasons), pendings=pendings
        )
    return GoNoGoResult(decision="GO", reasons=(), pendings=pendings)
