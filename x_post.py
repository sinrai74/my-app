#!/usr/bin/env python3
"""
x_post.py  ── ランキング結果を Gmail で送信し、X投稿用テキストも出力する

依存: 標準ライブラリのみ（smtplib）+ Pillow（画像添付）

環境変数（notify_arashi.py と共通、GitHub Secrets に登録済み）:
    GMAIL_ADDRESS  … 送信元 Gmail アドレス
    GMAIL_APP_PASS … Gmail アプリパスワード（16桁）

Usage:
    python x_post.py --input ranking_cache.json --with-image
    python x_post.py --input ranking_cache.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
import sys
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

log = logging.getLogger("x_post")

# ── 送信先（notify_arashi.py と同じ） ────────────────────────
MAIL_TO = "bigkirinuki@gmail.com"

# ランキング種類 → 画像ファイル名
_IMAGE_MAP: dict[str, str] = {
    "danger":    "danger.png",
    "hot":       "hot_motor.png",
    "manshuu":   ["manshuu_1.png", "manshuu_2.png"],
    "awakening": "awakening.png",
}

# ツイート用ハッシュタグ（手動投稿時にコピペしやすいよう末尾に付ける）
_HASHTAGS: dict[str, str] = {
    "danger":    "#競艇 #ボートレース #競艇予想 #1号艇 #荒れ予想",
    "hot":       "#競艇 #ボートレース #モーター #競艇予想",
    "manshuu":   "#競艇 #ボートレース #万舟 #荒れ予想 #穴予想",
    "awakening": "#競艇 #ボートレース #モーター #覚醒 #競艇予想",
}


# ════════════════════════════════════════════════════════════
# テキスト生成（x_ranking.py と同じ内容）
# ════════════════════════════════════════════════════════════

def _format_danger(data: dict) -> str:
    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    lines = [f"⚠️【{date_str} 危険な1号艇TOP10】⚠️", ""]
    for i, d in enumerate(data["danger_boat1"][:10], 1):
        emoji = "🔴" if d["score"] >= 80 else "🟡" if d["score"] >= 60 else "🟢"
        lines.append(f"{i}位 {d['venue']}{d['race']}R {emoji}")
        lines.append(f"   {d['racer']}（{d['reason']}）")
    lines += ["", "1号艇が飛ぶ可能性が高いレースです🏁",
              _HASHTAGS["danger"]]
    return "\n".join(lines)


def _format_hot(data: dict) -> str:
    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    lines = [f"🔥【{date_str} 激走モーターTOP10】🔥", ""]
    for i, m in enumerate(data["hot_motor"][:10], 1):
        lines.append(f"{i}位 {m['venue']}{m['motor_no']}号機")
    lines += ["", "数字以上に出ているモーター🔧", _HASHTAGS["hot"]]
    return "\n".join(lines)


def _format_manshuu(data: dict) -> str:
    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    lines = [f"🚨【{date_str} 万舟警報】🚨", ""]
    for i, u in enumerate(data["manshuu_alert"][:10], 1):
        emoji = "🔴" if u["score"] >= 80 else "🟡" if u["score"] >= 60 else "🟢"
        lines.append(f"{i}位 {u['venue']}{u['race']}R {emoji}")
    lines += ["", "高配当が出そうなレース💰", _HASHTAGS["manshuu"]]
    return "\n".join(lines)


def _format_awakening(data: dict) -> str:
    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    lines = [f"⚡【{date_str} 覚醒モーターTOP10】⚡", ""]
    for i, a in enumerate(data["awakening_motor"][:10], 1):
        lines.append(f"{i}位 {a['venue']}{a['motor_no']}号機")
    lines += ["", "最近急に伸びているモーター📈", _HASHTAGS["awakening"]]
    return "\n".join(lines)


FORMATTERS = {
    "danger":    _format_danger,
    "hot":       _format_hot,
    "manshuu":   _format_manshuu,
    "awakening": _format_awakening,
}

TITLES = {
    "danger":    "危険な1号艇TOP10",
    "hot":       "激走モーターTOP10",
    "manshuu":   "万舟警報TOP10",
    "awakening": "覚醒モーターTOP10",
}


# ════════════════════════════════════════════════════════════
# Gmail 送信
# ════════════════════════════════════════════════════════════

def _send_email(
    subject: str,
    body: str,
    image_paths: list[str],
    dry_run: bool = False,
) -> bool:
    """
    Gmail SMTP で画像添付メールを送信する。
    dry_run=True の場合は内容を表示するだけで送信しない。
    """
    if dry_run:
        sep = "─" * 60
        print(sep)
        print(f"[DRY RUN] 件名: {subject}")
        print(f"[DRY RUN] 添付: {image_paths}")
        print(body)
        print(sep)
        return True

    gmail_address = os.getenv("GMAIL_ADDRESS", "")
    app_password  = os.getenv("GMAIL_APP_PASS", "")

    if not gmail_address or not app_password:
        log.error("環境変数 GMAIL_ADDRESS / GMAIL_APP_PASS が未設定です")
        return False

    # MIMEMultipart で本文＋画像添付
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"]    = gmail_address
    msg["To"]      = MAIL_TO

    msg.attach(MIMEText(body, "plain", "utf-8"))

    # 画像を添付
    for path in image_paths:
        if not os.path.exists(path):
            log.warning("[添付] ファイルなし: %s → スキップ", path)
            continue
        with open(path, "rb") as f:
            part = MIMEBase("image", "png")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment",
                        filename=os.path.basename(path))
        msg.attach(part)
        log.info("[添付] %s", path)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(gmail_address, app_password)
            smtp.sendmail(gmail_address, MAIL_TO, msg.as_string())
        log.info("[送信] 成功 → %s  件名: %s", MAIL_TO, subject)
        return True
    except smtplib.SMTPException as e:
        log.error("[送信] 失敗: %s", e)
        return False


# ════════════════════════════════════════════════════════════
# メイン処理
# ════════════════════════════════════════════════════════════

def post_from_ranking(
    data: dict,
    types: Optional[list[str]] = None,
    with_image: bool = True,
    dry_run: bool = False,
) -> dict[str, bool]:
    """
    ランキングデータから全4種を1通のメールにまとめて送信する。
    戻り値: {"mail": 成功/失敗}
    """
    target_types = types or list(FORMATTERS.keys())
    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"

    # ── 本文: 4種類を縦に並べる ──────────────────────────────
    sep = "\n" + "=" * 50 + "\n"
    sections = []
    for t in target_types:
        if t in FORMATTERS:
            sections.append(FORMATTERS[t](data))
    body = sep.join(sections)

    # ── X手動投稿用の案内を末尾に追加 ────────────────────────
    body += "\n\n" + "=" * 50
    body += "\n【X手動投稿用】\n上記4ブロックをそれぞれXにコピペしてください。\n画像は添付ファイルを使用してください。"

    subject = f"[競艇ランキング] {date_str} 本日の4大ランキング"

    # ── 添付画像を収集 ───────────────────────────────────────
    image_paths: list[str] = []
    if with_image:
        for t in target_types:
            paths = _IMAGE_MAP.get(t, "")
            if isinstance(paths, list):
                image_paths.extend(paths)
            elif paths:
                image_paths.append(paths)

    ok = _send_email(subject, body, image_paths, dry_run=dry_run)
    return {"mail": ok}


# ════════════════════════════════════════════════════════════
# エントリポイント
# ════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="競艇ランキング Gmail 送信スクリプト")
    parser.add_argument("--input",      default="ranking_cache.json",
                        help="ランキング JSON ファイル（デフォルト: ranking_cache.json）")
    parser.add_argument("--with-image", action="store_true",
                        help="画像を添付して送信（x_image.py で事前生成が必要）")
    parser.add_argument("--dry-run",    action="store_true",
                        help="実際には送信せず内容を表示する")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        log.error("ランキングファイルが見つかりません: %s", args.input)
        log.error("先に `python x_ranking.py --generate` を実行してください")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = post_from_ranking(
        data,
        with_image = args.with_image,
        dry_run    = args.dry_run,
    )

    if results.get("mail"):
        log.info("[完了] メール送信成功")
    else:
        log.error("[完了] メール送信失敗")
        sys.exit(1)


if __name__ == "__main__":
    main()
