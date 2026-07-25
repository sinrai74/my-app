"""
NotificationService（Step5-5）: channel名でNotifierを選び送信するだけ。

責務: 登録済みNotifier（Protocol）の中からrequest.channelのものを選び、
  notify()を呼んで結果を返す。判定・計算・本文生成はしない。
禁止: RenderResult編集・通知内容変更・保存・Release・DurableStore。

具象Notifierは外から注入する（Serviceはnewしない）。
"""

from __future__ import annotations

import logging
import time
from typing import Mapping

from notification.notifiers import (
    NotificationRequest,
    NotificationResult,
    Notifier,
)

log = logging.getLogger(__name__)


class NotificationService:
    """channel → Notifier のディスパッチ（送信するだけ）。"""

    def __init__(self, notifiers: Mapping[str, Notifier]) -> None:
        # channel名 -> Notifier。具象は注入（Serviceはnewしない）
        self._notifiers = dict(notifiers)

    def send(self, request: NotificationRequest) -> NotificationResult:
        start = time.monotonic()
        log.info("Notification start channel=%s", request.channel)

        notifier = self._notifiers.get(request.channel)
        if notifier is None:
            raise ValueError(
                f"no notifier registered for channel={request.channel!r} "
                f"(available: {sorted(self._notifiers)})"
            )
        log.info("Notifier selected channel=%s", request.channel)

        log.info("Notification send start channel=%s", request.channel)
        result = notifier.notify(request)
        log.info(
            "Notification send end channel=%s sent=%s elapsed=%.3fs",
            request.channel, result.sent, time.monotonic() - start,
        )
        return result
