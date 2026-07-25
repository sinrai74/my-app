"""
Legacy Notification取得（Step6-2・必須②確定方針）: build_message()をラップ。

方針（レビュー確定・案A）:
  比較点は
      result → build_message() → (subject, body)
  ここまで。PredictionProviderが _evaluate_bets をラップしたのと同じ思想。

  比較対象は subject / body の一致のみ。
  send_email() / send_line() / send_notification() の呼び出し自体は
  比較対象にしない（Legacy変更禁止・Notification停止中・実送信禁止の
  3条件を同時に満たすため）。

禁止事項の遵守:
  - Legacyへのフック/print/logging/一時ファイル追加なし
  - build_message は import して呼ぶだけ（分解・再実装・コピーなし）
  - 送信APIは呼ばない（実送信0件を維持）
  - subject/body の補正・整形・デフォルト補完なし
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

# Legacy build_message のシグネチャ: (result: RaceResult) -> tuple[str, str]
LegacyBuildMessage = Callable[[Any], tuple]


class LegacyNotificationView:
    """build_message() の戻り値（subject, body）だけを保持する。"""

    __slots__ = ("subject", "body")

    def __init__(self, subject: str, body: str) -> None:
        self.subject = subject
        self.body = body

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, LegacyNotificationView)
            and self.subject == other.subject
            and self.body == other.body
        )

    def __repr__(self) -> str:
        return f"LegacyNotificationView(subject={self.subject!r})"


def build_notification_view(
    result: Any, build_message: Optional[LegacyBuildMessage] = None
) -> LegacyNotificationView:
    """Legacy build_message を呼び、(subject, body) を読み取るだけ。

    戻り値が (str, str) のタプルでない場合は補完せず例外
    （デフォルト補完禁止・サイレント推測禁止）。
    """
    func = build_message
    if func is None:
        from notify_arashi import build_message as func  # 遅延import・無改変

    raw = func(result)
    if not isinstance(raw, tuple) or len(raw) != 2:
        raise ValueError(
            "legacy build_message did not return a 2-tuple "
            f"(got type={type(raw).__name__}); no default value is supplied"
        )
    subject, body = raw
    return LegacyNotificationView(subject=subject, body=body)
