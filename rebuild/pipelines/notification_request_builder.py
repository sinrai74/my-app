"""
NotificationRequestビルダー（Production driver向け・結線のみ）:
RenderResult → NotificationRequest を組み立てるだけ。

方針（ユーザー確定）:
  - channel = "mail" 固定（Production driverはMail単独）
  - title   = 固定文字列（Legacyの件名生成 build_message は RaceResult 型依存で
    Rebuildから直接再利用できないため、Production通知用の新規固定値を用いる。
    Legacyの件名を再現しようとしない）
  - destination = Production側では設定しない（None）。宛先は既存
    MailNotifier → notify_arashi.send_email() の内部設定へ委譲する
    （現MailNotifierは NotificationRequest.destination を参照しない構造を維持）
  - render_result = OutputPipeline の RenderResult をそのまま渡す
  - attachment_path = 既存構造上安全に決められる場合のみ設定。
    本ビルダーでは安全に一意に決められないため None（推測で設定しない）

本モジュールは NotificationRequest / RenderResult / MailNotifier を変更しない。
値の生成・加工・評価はしない（固定値の付与と受け渡しのみ）。
"""

from __future__ import annotations

from typing import Optional

from notification.notifiers import NotificationRequest
from output.renderers import RenderResult

# Production通知用の固定件名（新規固定値・Legacy件名の再現ではない）
PRODUCTION_MAIL_TITLE = "【競艇AI】本日の予想結果"
PRODUCTION_MAIL_CHANNEL = "mail"


def build_mail_notification_request(
    render_result: RenderResult,
    *,
    title: str = PRODUCTION_MAIL_TITLE,
    attachment_path: Optional[str] = None,
) -> NotificationRequest:
    """RenderResult から mail 向け NotificationRequest を組み立てる。

    - channel は "mail" 固定
    - destination は設定しない（None・send_email内部設定へ委譲）
    - title は固定文字列（既定 PRODUCTION_MAIL_TITLE）
    - attachment_path は既定 None（安全に一意に決められないため設定しない）
    """
    return NotificationRequest(
        render_result=render_result,
        channel=PRODUCTION_MAIL_CHANNEL,
        destination=None,
        title=title,
        attachment_path=attachment_path,
    )
