#!/usr/bin/env python3
"""
x_results_page.py  ── ⑧ AI実績ページ 生成・送信

毎日21時、各ブランド（危険艇・万舟・転がし・激走・覚醒）ごとの
成功率・的中率・改善点コメントをまとめたHTMLページを生成し、
メール添付で送信する。

Usage:
    python x_results_page.py              # 今日の実績ページを生成
    python x_results_page.py --date 20260628
    python x_results_page.py --dry-run    # 送信せず保存のみ
"""

from __future__ import annotations

import argparse
import logging
import os
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

from x_verification import (
    load_today_records,
    aggregate,
    aggregate_by_rank,
    _yesterday_jst,
)

log = logging.getLogger("x_results")

JST     = timezone(timedelta(hours=9))
MAIL_TO = "bigkirinuki@gmail.com"


# ════════════════════════════════════════════════════════════
# 改善点コメント生成
# ════════════════════════════════════════════════════════════

def _improvement_comment_danger(rank_data: dict, agg: dict) -> str:
    s = rank_data.get("S", {})
    a = rank_data.get("A", {})
    if s.get("total", 0) >= 3 and s.get("rate", 0) < 60:
        return "Sランクの的中率が想定より低め。展示タイムの重み付けを見直す予定です。"
    if a.get("total", 0) >= 3 and a.get("rate", 0) > s.get("rate", 0):
        return "Aランクの方がSランクより成績が良い傾向。閾値の再調整を検討中です。"
    if agg.get("boat1_flew_rate", 0) >= 70:
        return "1号艇のイン逃げ失敗率が高水準で推移。判定ロジックは順調です。"
    return "順調にデータが蓄積されています。"


def _improvement_comment_manshuu(rank_data: dict, agg: dict) -> str:
    s = rank_data.get("S", {})
    if s.get("total", 0) >= 3 and s.get("rate", 0) >= 50:
        return "Sランクの万舟的中率が良好。荒れ指数の精度は高い状態です。"
    if agg.get("manshuu_count", 0) == 0:
        return "本日は万舟該当なし。低調な日でした。"
    return "万舟発生件数を継続的に検証中です。"


def _improvement_comment_generic(label: str) -> str:
    return f"{label}は実績データ蓄積中です。結果照合の仕組みを準備中のため、今後反映していきます。"


# ════════════════════════════════════════════════════════════
# HTML生成
# ════════════════════════════════════════════════════════════

def generate_results_html(date_str: str, output_path: str) -> dict:
    """
    実績ページHTMLを生成し、生成に使ったサマリーdictを返す
    （メール本文・テキスト版にも再利用するため）
    """
    records   = load_today_records(date_str)
    agg       = aggregate(records)
    rank_data = aggregate_by_rank(records) if records else {
        "danger": {"S": {"total":0,"hit":0,"rate":0.0}, "A": {"total":0,"hit":0,"rate":0.0}, "B": {"total":0,"hit":0,"rate":0.0}},
        "manshuu": {"S": {"total":0,"hit":0,"rate":0.0}, "A": {"total":0,"hit":0,"rate":0.0}, "B": {"total":0,"hit":0,"rate":0.0}},
    }

    date_disp = f"{date_str[4:6]}/{date_str[6:8]}"
    now_str   = datetime.now(JST).strftime("%H:%M")

    def rank_table(rank_data_cat: dict) -> str:
        rows = ""
        for rank in ["S", "A", "B"]:
            d = rank_data_cat.get(rank, {"total": 0, "hit": 0, "rate": 0.0})
            if d["total"] == 0:
                continue
            color = {"S": "#ef5350", "A": "#ffa726", "B": "#66bb6a"}[rank]
            rows += f"""
<div class="rank-bar-row">
  <span class="rank-badge" style="background:{color}">{rank}</span>
  <span class="rank-detail">{d['total']}件中{d['hit']}件的中</span>
  <div class="rank-bar-bg"><div class="rank-bar-fill" style="width:{d['rate']}%;background:{color}"></div></div>
  <span class="rank-rate">{d['rate']}%</span>
</div>"""
        return rows or '<p class="no-data">本日は対象データなし</p>'

    danger_table  = rank_table(rank_data["danger"])
    manshuu_table = rank_table(rank_data["manshuu"])

    danger_comment  = _improvement_comment_danger(rank_data["danger"], agg)
    manshuu_comment = _improvement_comment_manshuu(rank_data["manshuu"], agg)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI実績ページ {date_disp}</title>
<style>
:root{{--bg:#0d0d1a;--card:#16162a;--border:#252540;--text:#e8e8f0;--gray:#8888aa;--accent:#4fc3f7;}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Hiragino Sans','Noto Sans JP',sans-serif;
  line-height:1.7;padding:12px;max-width:780px;margin:0 auto}}
.header{{text-align:center;border:2px solid var(--accent);border-radius:12px;
  padding:24px 16px;margin-bottom:20px;background:var(--card)}}
.header h1{{font-size:1.8em;color:var(--accent)}}
.header .meta{{color:var(--gray);margin-top:6px;font-size:.85em}}
.brand-section{{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:18px;margin-bottom:16px}}
.brand-section h2{{font-size:1.15em;margin-bottom:12px;padding-bottom:10px;
  border-bottom:1px solid var(--border)}}
.rank-bar-row{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
.rank-badge{{width:24px;height:24px;border-radius:6px;color:#fff;font-weight:bold;
  display:flex;align-items:center;justify-content:center;font-size:.85em;flex-shrink:0}}
.rank-detail{{font-size:.82em;color:var(--gray);min-width:110px}}
.rank-bar-bg{{flex:1;background:#252540;border-radius:4px;height:14px;overflow:hidden}}
.rank-bar-fill{{height:100%;border-radius:4px}}
.rank-rate{{font-size:.85em;font-weight:bold;min-width:42px;text-align:right}}
.brand-comment{{background:#14142a;border-left:3px solid var(--accent);
  border-radius:6px;padding:10px 14px;margin-top:12px;color:#bbb;font-size:.85em}}
.no-data{{color:var(--gray);font-size:.88em;padding:8px 0}}
.footer{{text-align:center;color:#444;font-size:.78em;padding:20px 0}}
</style>
</head>
<body>

<div class="header">
  <h1>📊 AI実績ページ</h1>
  <div class="meta">{date_disp}分　生成: {now_str}</div>
</div>

<div class="brand-section">
  <h2>🚨 危険艇速報</h2>
  {danger_table}
  <div class="brand-comment">💬 {danger_comment}</div>
</div>

<div class="brand-section">
  <h2>💰 万舟警報</h2>
  {manshuu_table}
  <div class="brand-comment">💬 {manshuu_comment}</div>
</div>

<div class="brand-section">
  <h2>🎯 転がし候補</h2>
  <p class="no-data">結果照合の仕組みを準備中です</p>
  <div class="brand-comment">💬 {_improvement_comment_generic("転がし候補")}</div>
</div>

<div class="brand-section">
  <h2>⚡ 激走モーター</h2>
  <p class="no-data">結果照合の仕組みを準備中です</p>
  <div class="brand-comment">💬 {_improvement_comment_generic("激走モーター")}</div>
</div>

<div class="brand-section">
  <h2>📈 覚醒モーター</h2>
  <p class="no-data">結果照合の仕組みを準備中です</p>
  <div class="brand-comment">💬 {_improvement_comment_generic("覚醒モーター")}</div>
</div>

<div class="footer">
  AI競艇新聞 実績ページ | 全レース機械学習分析<br>
  ※本ページはデータ分析結果であり、的中を保証するものではありません。
</div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("[実績] HTML保存: %s", output_path)

    return {"agg": agg, "rank_data": rank_data}


# ════════════════════════════════════════════════════════════
# メール送信
# ════════════════════════════════════════════════════════════

def send_results_page(html_path: str, date_str: str, summary: dict,
                      dry_run: bool = False) -> bool:
    date_disp = f"{date_str[4:6]}/{date_str[6:8]}"
    subject   = f"📊 AI実績ページ {date_disp}"

    rank_data = summary.get("rank_data", {})
    danger_s  = rank_data.get("danger", {}).get("S", {})
    manshuu_s = rank_data.get("manshuu", {}).get("S", {})

    body = (
        f"AI実績ページ {date_disp} を添付します。\n\n"
        f"危険艇Sランク: {danger_s.get('total',0)}件中{danger_s.get('hit',0)}件的中（{danger_s.get('rate',0)}%）\n"
        f"万舟Sランク: {manshuu_s.get('total',0)}件中{manshuu_s.get('hit',0)}件的中（{manshuu_s.get('rate',0)}%）\n\n"
        "詳細は添付のHTMLをご確認ください。\n"
    )

    if dry_run:
        print("=" * 60)
        print(f"[DRY RUN] 件名: {subject}")
        print(body)
        print("=" * 60)
        return True

    gmail_address = os.getenv("GMAIL_ADDRESS", "")
    app_password  = os.getenv("GMAIL_APP_PASS", "")
    if not gmail_address or not app_password:
        log.error("GMAIL_ADDRESS / GMAIL_APP_PASS が未設定")
        return False

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"]    = gmail_address
    msg["To"]      = MAIL_TO
    msg.attach(MIMEText(body, "plain", "utf-8"))

    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            part = MIMEBase("text", "html")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment",
                        filename=os.path.basename(html_path))
        msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo(); smtp.starttls()
            smtp.login(gmail_address, app_password)
            smtp.sendmail(gmail_address, MAIL_TO, msg.as_string())
        log.info("[送信] 成功: %s", subject)
        return True
    except smtplib.SMTPException as e:
        log.error("[送信] 失敗: %s", e)
        return False


# ════════════════════════════════════════════════════════════
# エントリポイント
# ════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="AI実績ページ 生成・送信")
    parser.add_argument("--date",    help="対象日 YYYYMMDD（省略時は前日）")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    date_str = args.date or _yesterday_jst()
    html_path = "results.html"

    summary = generate_results_html(date_str, html_path)
    ok = send_results_page(html_path, date_str, summary, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
