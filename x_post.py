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

def _get_yesterday_oneliner() -> str:
    """前日サマリーを取得して1〜3行テキストを返す。失敗時は空文字"""
    try:
        from x_verification import get_yesterday_summary, format_yesterday_oneliner
        summary = get_yesterday_summary()
        return format_yesterday_oneliner(summary)
    except Exception:
        return ""


def _score_to_rank(score: int) -> str:
    if score >= 80: return "🔴 S 危険"
    if score >= 60: return "🟠 A 注意"
    return "🟡 B やや危険"

def _score_to_rank_short(score: int) -> str:
    if score >= 80: return "🔴S"
    if score >= 60: return "🟠A"
    return "🟡B"

def _format_danger(data: dict) -> str:
    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    items = data.get("danger_boat1", [])
    if not items:
        return f"⚠️【{date_str} 危険な1号艇】⚠️\n\n本日は該当レースなし\n{_HASHTAGS['danger']}"
    top = items[0]
    rank = _score_to_rank(top["score"])
    stars = top.get("stars", {})
    lines = [
        f"🚨 AI危険艇速報 {date_str}", "",
        f"【{rank}】{top['venue']}{top['race']}R",
        f"{top['racer']}",
        f"▶ {top['reason']}", "",
    ]
    if stars:
        lines += [
            "── AI評価 ──",
            f"ST　　{stars.get('ST',  '★★★☆☆')}",
            f"機力　{stars.get('機力', '★★★☆☆')}",
            f"近況　{stars.get('近況', '★★★☆☆')}",
            f"相手　{stars.get('相手', '★★★☆☆')}",
            "",
        ]
    if len(items) > 1:
        lines.append("── 他の注目レース ──")
        for d in items[1:6]:
            r = _score_to_rank_short(d["score"])
            lines.append(f"{r} {d['venue']}{d['race']}R {d['racer']}")
        lines.append("")
    # 昨日の答え合わせを差し込む
    yesterday_line = _get_yesterday_oneliner()
    if yesterday_line:
        lines += ["", "─" * 20, yesterday_line, "─" * 20]

    lines += ["あなたが今日気になるレースはどこですか？💬", _HASHTAGS["danger"]]
    return "\n".join(lines)


def _format_hot(data: dict) -> str:
    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    items = data.get("hot_motor", [])
    if not items:
        return f"🔥【{date_str} 激走モーターTOP10】🔥\n\nデータ蓄積中...\n{_HASHTAGS['hot']}"
    lines = [f"🔥 AI激走モーター {date_str}", ""]
    for i, m in enumerate(items[:10], 1):
        recent = m.get("recent5", "---")
        gap    = m.get("gap", 0)
        gap_str = f"+{gap:.0f}%" if gap > 0 else f"{gap:.0f}%"
        lines.append(f"{i}位 {m['venue']}{m['motor_no']}号機")
        lines.append(f"   直近5走: {recent}  公式比{gap_str}")
    lines += ["", "公式2連率を上回る激走モーター🔧", "あなたのお気に入りはありましたか？💬", _HASHTAGS["hot"]]
    return "\n".join(lines)


def _format_manshuu(data: dict) -> str:
    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    items = data.get("manshuu_alert", [])
    if not items:
        return f"🚨【{date_str} 万舟警報】🚨\n\n本日は該当なし\n{_HASHTAGS['manshuu']}"
    top = items[0]
    rank = _score_to_rank(top["score"])
    reasons = top.get("key_reason", "").split(" / ")
    lines = [f"💰 AI万舟警報 {date_str}", "", f"【{rank}】{top['venue']}{top['race']}R", ""]
    for r in reasons[:3]:
        lines.append(r if r.startswith("🔥") else f"🔥 {r}")
    lines.append("")
    if len(items) > 1:
        lines.append("── 他の警戒レース ──")
        for u in items[1:5]:
            r = _score_to_rank_short(u["score"])
            lines.append(f"{r} {u['venue']}{u['race']}R")
        lines.append("")
    lines += ["高配当が出そうなレースに注目💰", "どのレースが気になりますか？💬", _HASHTAGS["manshuu"]]
    return "\n".join(lines)


def _format_awakening(data: dict) -> str:
    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    items = data.get("awakening_motor", [])
    if not items:
        return f"⚡【{date_str} 覚醒モーターTOP10】⚡\n\nデータ蓄積中...\n{_HASHTAGS['awakening']}"
    lines = [f"⚡ AI覚醒モーター {date_str}", ""]
    for i, a in enumerate(items[:10], 1):
        recent = a.get("recent10", "---")
        old_r  = a.get("old_2rate")
        new_r  = a.get("new_2rate")
        ex_avg = a.get("ex_avg")
        lines.append(f"{i}位 {a['venue']}{a['motor_no']}号機")
        lines.append(f"   直近10走: {recent}")
        detail = []
        if old_r is not None and new_r is not None:
            detail.append(f"2連率 {old_r:.0f}%→{new_r:.0f}%")
        if ex_avg:
            detail.append(f"展示平均{ex_avg:.2f}秒")
        if detail:
            lines.append(f"   {'  '.join(detail)}")
    lines += ["", "急に仕上がってきたモーターは狙い目📈", "どのモーターが気になりましたか？💬", _HASHTAGS["awakening"]]
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
# 覚醒モーター: 条件チェック（90以上が1件以上あるときだけ送信）
# ════════════════════════════════════════════════════════════

AWAKENING_THRESHOLD = 90   # この指数以上のモーターがあるときだけ送信

def _should_send_awakening(data: dict) -> bool:
    """覚醒度90以上が1件以上あれば True"""
    items = data.get("awakening_motor", [])
    return any(m.get("score", 0) >= AWAKENING_THRESHOLD for m in items)


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

    # ── 件名：種類数に応じて変える ───────────────────────────
    type_labels = {
        "danger":    "危険な1号艇",
        "hot":       "激走モーター",
        "manshuu":   "万舟警報",
        "awakening": "覚醒モーター",
    }
    if len(target_types) == 1:
        subject = f"[競艇AI] {date_str} {type_labels.get(target_types[0], target_types[0])}"
    else:
        subject = f"[競艇ランキング] {date_str} 本日のAIランキング"

    # ── 本文: 指定種類を縦に並べる ───────────────────────────
    sep = "\n" + "=" * 50 + "\n"
    sections = []
    for t in target_types:
        if t in FORMATTERS:
            sections.append(FORMATTERS[t](data))
    body = sep.join(sections)
    body += "\n\n" + "=" * 50

    # ── X投稿候補ブロックを追加 ──────────────────────────────
    try:
        from x_post_text import danger_post, manshuu_post

        # 危険艇データ
        danger_items  = data.get("danger_boat1", [])
        manshuu_items = data.get("manshuu_alert", [])

        s_count   = sum(1 for d in danger_items if d.get("score", 0) >= 80)
        top_danger_venue = danger_items[0].get("venue", "") if danger_items else ""
        top_danger_race  = str(danger_items[0].get("race", "")) if danger_items else ""
        top_manshuu_venue = manshuu_items[0].get("venue", "") if manshuu_items else ""
        top_manshuu_race  = str(manshuu_items[0].get("race", "")) if manshuu_items else ""
        top_manshuu_match = int(manshuu_items[0].get("score", 0)) if manshuu_items else 0

        if "danger" in target_types:
            body += danger_post(
                s_count   = s_count,
                top_venue = top_danger_venue,
                top_race  = top_danger_race,
            )
        if "manshuu" in target_types:
            body += manshuu_post(
                count     = len(manshuu_items),
                top_venue = top_manshuu_venue,
                top_race  = top_manshuu_race,
                top_match = top_manshuu_match,
            )
    except Exception as _e:
        log.error("[X投稿] 生成失敗: %s", _e, exc_info=True)
        body += "\n【X投稿用】上記をXにコピペしてください。"

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
    parser.add_argument("--type",       choices=list(_IMAGE_MAP.keys()), action="append",
                        dest="types",   help="送信する種類（複数指定可: danger / hot / manshuu / awakening）")
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
        types      = args.types,          # None のとき全種類を送信
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
