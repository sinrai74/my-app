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
from typing import Mapping, Optional, Protocol

from notification.notifiers import (
    NotificationRequest,
    NotificationResult,
    Notifier,
)

log = logging.getLogger(__name__)

# message_key の job 固定値（ユーザー確定・§454準拠）
_JOB = "arashi"


class _IdempotencyStore(Protocol):
    """NotificationServiceが使う冪等ストアの部分Protocol。"""

    def is_recorded(self, channel: str, message_key: str) -> bool: ...
    def record(self, channel: str, message_key: str) -> None: ...


def build_message_key(request: NotificationRequest) -> Optional[str]:
    """message_key を組み立てる（§454規約）: {channel}:{job}:{race_date}:{対象ID}

    対象ID = {venue_num:02d}_{race_number:02d}（ユーザー確定）。
    race識別子（race_date/venue_num/race_number）が揃わない場合は None
    （＝冪等管理対象外。従来どおり送信）。
    """
    if (request.race_date is None or request.venue_num is None
            or request.race_number is None):
        return None
    target_id = f"{request.venue_num:02d}_{request.race_number:02d}"
    return f"{request.channel}:{_JOB}:{request.race_date}:{target_id}"


class NotificationService:
    """channel → Notifier のディスパッチ（送信するだけ）。

    idempotency_store を注入した場合、送信直前に冪等チェックを行う（L5 services
    層で一元管理・§453/§422）:
      is_recorded(channel, message_key) が True → 送信スキップ
      False → notify() 実行 → 成功時のみ record(channel, message_key)
      送信失敗時は record しない（次回再送のため・§453）
    未注入時は従来どおり単純ディスパッチ（後方互換）。
    """

    def __init__(
        self,
        notifiers: Mapping[str, Notifier],
        idempotency_store: Optional[_IdempotencyStore] = None,
    ) -> None:
        # channel名 -> Notifier。具象は注入（Serviceはnewしない）
        self._notifiers = dict(notifiers)
        self._idempotency = idempotency_store

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

        # 冪等チェック（L5 services層・送信直前・§453/§422）
        message_key = None
        if self._idempotency is not None:
            message_key = build_message_key(request)
            if message_key is not None and self._idempotency.is_recorded(
                request.channel, message_key
            ):
                log.info(
                    "Notification skipped (already recorded) channel=%s key=%s",
                    request.channel, message_key,
                )
                return NotificationResult(
                    request.channel, sent=False, detail="idempotent-skip"
                )

        log.info("Notification send start channel=%s", request.channel)
        result = notifier.notify(request)
        log.info(
            "Notification send end channel=%s sent=%s elapsed=%.3fs",
            request.channel, result.sent, time.monotonic() - start,
        )

        # 送信成功時のみ記録（失敗時はrecordしない・次回再送のため・§453）
        if (self._idempotency is not None and message_key is not None
                and result.sent):
            self._idempotency.record(request.channel, message_key)
            log.info(
                "Notification recorded channel=%s key=%s",
                request.channel, message_key,
            )

        return result
