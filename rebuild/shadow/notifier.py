"""
NullNotifier（Step5-6・指示3）: Shadow中の実送信を物理的に禁止するNotifier。

どのchannelにも登録できる。notify()は外部APIを一切呼ばず、requestを
記録して常に sent=False の結果を返す。呼び出し回数はテスト・監査で
「実送信0件」を確認するために公開する。
"""

from __future__ import annotations

from notification.notifiers import NotificationRequest, NotificationResult


class NullNotifier:
    """全チャネル共通のno-op Notifier。実送信を一切行わない。"""

    def __init__(self, channel: str = "shadow") -> None:
        self.channel = channel
        self.received: list[NotificationRequest] = []

    def notify(self, request: NotificationRequest) -> NotificationResult:
        # 外部送信APIは一切呼ばない。記録のみ。
        self.received.append(request)
        return NotificationResult(request.channel, sent=False, detail="shadow-noop")

    @property
    def call_count(self) -> int:
        return len(self.received)
