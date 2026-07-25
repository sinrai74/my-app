"""
NotificationPipeline（Step5-5・必要最小限）: 複数のNotificationRequestを
NotificationServiceへ順に渡すだけの結線。

責務: request列をserviceへ順に送り、結果を集める。
禁止: 判定・計算・RenderResult編集・通知内容変更・保存・Release。

依存: pipelines → notification(Protocol/Service)。具象は外から注入。
"""

from __future__ import annotations

import logging
import time
from typing import Sequence

from notification.notifiers import NotificationRequest, NotificationResult
from notification.service import NotificationService

log = logging.getLogger(__name__)


class NotificationPipeline:
    """NotificationRequest群の結線（送信するだけ）。"""

    def __init__(self, service: NotificationService) -> None:
        self._service = service

    def send_all(
        self, requests: Sequence[NotificationRequest]
    ) -> list[NotificationResult]:
        start = time.monotonic()
        log.info("NotificationPipeline start count=%d", len(requests))
        results = [self._service.send(req) for req in requests]
        log.info("NotificationPipeline end elapsed=%.3fs",
                 time.monotonic() - start)
        return results
