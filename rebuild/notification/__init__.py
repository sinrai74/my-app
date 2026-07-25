"""
notification層（Step5-5）: RenderResultを既存Notifierへ渡して送信するだけ。

責務（Step5-5指示）:
  RenderResult受け取り → Notifier呼び出し → 送信結果返却。
  判定・計算・補正・HTML/CSV/TXT生成・RenderResult編集・通知本文変更はしない。
  既存の送信処理（send_email/send_line等）はimportして呼ぶだけ（再実装禁止）。

入力: RenderResult のみ（RaceEvaluation/Prediction/BuyAssessment/Coreモデルは
  参照しない）。

依存: notification → output(RenderResult) / Legacy送信関数(import利用)。
  Storage/Release/DurableStore/Core/Pipelineには依存しない。
"""

from notification.notifiers import (
    DiscordNotifier,
    LineNotifier,
    MailNotifier,
    NotificationRequest,
    NotificationResult,
    Notifier,
    XNotifier,
)
from notification.service import NotificationService

__all__ = [
    "DiscordNotifier",
    "LineNotifier",
    "MailNotifier",
    "NotificationRequest",
    "NotificationResult",
    "NotificationService",
    "Notifier",
    "XNotifier",
]
