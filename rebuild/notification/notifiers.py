"""
Notifier（Step5-5）: 既存の送信処理を呼ぶだけのラッパー群。

各Notifierは RenderResult＋宛先情報（NotificationRequest）を受け取り、
既存のLegacy送信関数（send_email/send_line等）をimportして呼ぶ。
通知本文の生成・編集・判定はしない（送信するだけ）。

NotificationRequest は「Notifierへの入力」を表す不変データ。
Shadow比較の対象（RenderResult/通知先/添付対象/タイトル/種別）はこの
オブジェクトで表現され、送信API自体は比較しない（指示準拠）。
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Protocol

from output.renderers import RenderResult

log = logging.getLogger(__name__)


class NotificationRequest:
    """Notifierへの入力（不変）。Notificationが扱う唯一のデータ。

    - render_result: Output層の成果物（本層は中身を編集しない）
    - channel: 種別（"mail"/"line"/"discord"/"x"）
    - destination: 通知先（メールアドレス/チャンネル等。Noneは既存既定を使う）
    - title: タイトル（メール件名等。本文生成はしない・呼び出し側が用意）
    - attachment_path: 添付対象パス（Noneなら添付なし）
    """

    __slots__ = ("render_result", "channel", "destination", "title",
                 "attachment_path")

    def __init__(
        self,
        render_result: RenderResult,
        channel: str,
        destination: Optional[str] = None,
        title: Optional[str] = None,
        attachment_path: Optional[str] = None,
    ) -> None:
        self.render_result = render_result
        self.channel = channel
        self.destination = destination
        self.title = title
        self.attachment_path = attachment_path

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, NotificationRequest)
            and self.render_result == other.render_result
            and self.channel == other.channel
            and self.destination == other.destination
            and self.title == other.title
            and self.attachment_path == other.attachment_path
        )

    def __repr__(self) -> str:
        return (
            f"NotificationRequest(channel={self.channel!r}, "
            f"destination={self.destination!r}, title={self.title!r}, "
            f"attachment_path={self.attachment_path!r})"
        )


class NotificationResult:
    """送信結果（不変）。"""

    __slots__ = ("channel", "sent", "detail")

    def __init__(self, channel: str, sent: bool, detail: str = "") -> None:
        self.channel = channel
        self.sent = sent
        self.detail = detail

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, NotificationResult)
            and self.channel == other.channel
            and self.sent == other.sent
            and self.detail == other.detail
        )

    def __repr__(self) -> str:
        return f"NotificationResult(channel={self.channel!r}, sent={self.sent})"


class Notifier(Protocol):
    """通知の抽象。RenderResultを含むRequestを受け、送信結果を返す。

    禁止: 判定・計算・HTML/CSV/TXT生成・RenderResult編集・本文変更。
    """

    channel: str

    def notify(self, request: NotificationRequest) -> NotificationResult: ...


# Legacy送信関数の型
_SubjectBodySender = Callable[[str, str], bool]  # send_email(subject, body)
_BodySender = Callable[[str], bool]              # send_line(body)


class MailNotifier:
    """send_email(subject, body) のラッパー。

    body には RenderResult の成果物パスまたはsummaryを渡す方針だが、
    「本文生成」はしない。ここではrequest.titleを件名、
    render_resultのoutput_pathを本文の実体位置として既存関数へ渡すだけ。
    実際の本文組み立てが必要な場合は呼び出し側（Step5-6の配線）が
    request.title/attachment_pathを用意する（本層は生成しない）。
    """

    channel = "mail"

    def __init__(self, sender: Optional[_SubjectBodySender] = None) -> None:
        self._sender = sender

    def notify(self, request: NotificationRequest) -> NotificationResult:
        sender = self._sender
        if sender is None:
            from notify_arashi import send_email as sender  # 遅延import・無改変
        subject = request.title or ""
        # 本文は生成しない。成果物パスをそのまま渡す（既存関数の引数仕様に委ねる）
        body = request.render_result.output_path
        ok = bool(sender(subject, body))
        return NotificationResult(self.channel, ok)


class LineNotifier:
    """send_line(body) のラッパー。"""

    channel = "line"

    def __init__(self, sender: Optional[_BodySender] = None) -> None:
        self._sender = sender

    def notify(self, request: NotificationRequest) -> NotificationResult:
        sender = self._sender
        if sender is None:
            from notify_arashi import send_line as sender  # 遅延import・無改変
        body = request.render_result.output_path
        ok = bool(sender(body))
        return NotificationResult(self.channel, ok)


class DiscordNotifier:
    """Discord送信のラッパー。

    既存にDiscord専用の単純送信関数が確認できないため、送信実体は
    senderとしてDIで受ける設計にとどめる（実体の再実装はしない）。
    未注入で呼ばれた場合は明示的に失敗する（サイレント成功にしない）。
    """

    channel = "discord"

    def __init__(self, sender: Optional[_BodySender] = None) -> None:
        self._sender = sender

    def notify(self, request: NotificationRequest) -> NotificationResult:
        if self._sender is None:
            raise NotImplementedError(
                "DiscordNotifier requires an injected sender "
                "(no legacy discord sender wired). Provide it at DI time."
            )
        body = request.render_result.output_path
        ok = bool(self._sender(body))
        return NotificationResult(self.channel, ok)


class XNotifier:
    """X投稿のラッパー。

    既存のX投稿はx_post.post_from_ranking等で構成が複雑なため、
    送信実体はsenderとしてDIで受ける（再実装・引数書き換えはしない）。
    未注入時は明示的に失敗する。
    """

    channel = "x"

    def __init__(self, sender: Optional[_BodySender] = None) -> None:
        self._sender = sender

    def notify(self, request: NotificationRequest) -> NotificationResult:
        if self._sender is None:
            raise NotImplementedError(
                "XNotifier requires an injected sender "
                "(legacy x_post has a complex API; wire it at DI time)."
            )
        body = request.render_result.output_path
        ok = bool(self._sender(body))
        return NotificationResult(self.channel, ok)
