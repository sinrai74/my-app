#!/usr/bin/env python3
"""
x_results_public.py  ── AI実績ページ【公開用】

目的: ユーザーへ「AIの実績」「透明性」「信頼性」を伝える。
X・noteでの公開を前提に、見やすさを最優先にシンプルな内容とする。
外れた予想も隠さず掲載する。

出力: ai_result_public.html / ai_result_public.pdf

集計ロジックは x_results_common.py を利用し、本ファイルは
表示（HTML生成）のみを担当する。
"""

from __future__ import annotations

import logging
import os
import smtplib
from datetime import datetime, timedelta, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from x_brand_config import AI_VERSION, SYSTEM_NAME
from x_results_common import (
    collect_all_periods, load_daily_stats, load_daily_stats_range,
    calc_brand_results, calc_brand_results_range,
    calc_korogashi_results, calc_hot_motor_results, calc_awakening_results,
    calc_overall_roi, find_mvp_prediction, find_close_misses,
    generate_ai_comment_for_miss,
)

log = logging.getLogger("x_results_public")

JST = timezone(timedelta(hours=9))
MAIL_TO = "bigkirinuki@gmail.com"


# ════════════════════════════════════════════════════════════
# 表示パーツ
# ════════════════════════════════════════════════════════════

def _brand_row_html(icon: str, label: str, result: dict) -> str:
    """1ブランド分の的中数・的中率の行HTML"""
    if not result.get("has_data", result.get("data_available", False)):
        reason = result.get("reason", "本日データなし")
        return f"""
<div class="brand-row">
  <span class="br-icon">{icon}</span>
  <span class="br-label">{label}</span>
  <span class="br-value no-data">{reason}</span>
</div>"""
    checked = result.get("checked", result.get("listed", 0))
    hit     = result.get("hit", 0)
    rate    = result.get("rate", 0.0)
    return f"""
<div class="brand-row">
  <span class="br-icon">{icon}</span>
  <span class="br-label">{label}</span>
  <span class="br-value">{hit}/{checked}件的中（{rate}%）</span>
</div>"""


def _mvp_html(mvp: dict | None) -> str:
    if not mvp:
        return '<p class="no-data">本日は的中データがありませんでした</p>'
    return f"""
<div class="mvp-card">
  <div class="mvp-race">{mvp['venue']}{mvp['race']}R</div>
  <div class="mvp-combo">🎯 {mvp['pred_combo']} → 的中！</div>
  <div class="mvp-payout">払戻 {mvp['payout']:,}円</div>
</div>"""


def _close_misses_html(misses: list[dict]) -> str:
    if not misses:
        return '<p class="no-data">本日は対象データがありませんでした</p>'
    rows = ""
    for m in misses:
        comment = generate_ai_comment_for_miss(m)
        rows += f"""
<div class="miss-card">
  <div class="miss-race">{m.get('venue','')}{m.get('race','')}R</div>
  <div class="miss-combo">予想 {m.get('pred_combo','')} → 結果 {m.get('result_combo','')}</div>
  <div class="miss-comment">🤖 {comment}</div>
</div>"""
    return rows


# ════════════════════════════════════════════════════════════
# メイン生成関数
# ════════════════════════════════════════════════════════════

def generate_public_html(date_str: str, output_path: str) -> dict:
    """公開用実績ページHTMLを生成し、サマリーdictを返す"""
    periods = collect_all_periods(date_str)
    today_records = periods["today"]["records"]
    daily_stats   = load_daily_stats(date_str)

    date_disp = f"{date_str[4:6]}/{date_str[6:8]}"
    now_str   = datetime.now(JST).strftime("%H:%M")

    # ── ブランド別的中率 ──────────────────────────────────
    danger_r    = calc_brand_results(today_records, daily_stats, "danger")
    manshuu_r   = calc_brand_results(today_records, daily_stats, "manshuu")
    korogashi_r = calc_korogashi_results()
    hot_r       = calc_hot_motor_results()
    awake_r     = calc_awakening_results()

    # ── 回収率・ROI ───────────────────────────────────────
    roi_today = calc_overall_roi(today_records)
    daily_stats_7d  = load_daily_stats_range(date_str, 7)
    daily_stats_30d = load_daily_stats_range(date_str, 30)
    roi_7d  = calc_overall_roi(periods["d7"]["records"])
    roi_30d = calc_overall_roi(periods["d30"]["records"])

    # ── MVP・惜しかった予想 ──────────────────────────────
    mvp = find_mvp_prediction(today_records)
    close_misses = find_close_misses(today_records, limit=3)

    races_analyzed = danger_r.get("listed", 0) + manshuu_r.get("listed", 0)

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>AI実績ページ {date_disp}</title>
<style>
:root {{
  --bg: #0d0d1a; --card: #1a1a2e; --border: #2a2a44;
  --accent: #4fc3f7; --text: #e8e8f0; --gray: #8a8aa0;
  --good: #66bb6a; --bad: #ef5350;
}}
body {{background:var(--bg);color:var(--text);font-family:'Hiragino Sans','Meiryo',sans-serif;
  max-width:680px;margin:0 auto;padding:16px;line-height:1.6;}}
.header {{text-align:center;padding:20px 0;border-bottom:2px solid var(--accent);margin-bottom:20px;}}
.header h1 {{font-size:1.6em;color:var(--accent);}}
.header .meta {{color:var(--gray);font-size:.85em;margin-top:6px;}}
.section {{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:16px;margin-bottom:16px;}}
.section h2 {{font-size:1.05em;color:var(--accent);margin-bottom:10px;}}
.brand-row {{display:flex;align-items:center;gap:10px;padding:8px 0;
  border-bottom:1px solid var(--border);}}
.brand-row:last-child {{border-bottom:none;}}
.br-icon {{font-size:1.2em;}}
.br-label {{min-width:90px;font-weight:bold;}}
.br-value {{color:var(--good);flex:1;text-align:right;}}
.br-value.no-data {{color:var(--gray);font-size:.85em;}}
.roi-grid {{display:grid;grid-template-columns:1fr 1fr;gap:10px;}}
.roi-cell {{background:#12121e;border-radius:8px;padding:12px;text-align:center;}}
.roi-num {{font-size:1.4em;font-weight:bold;color:var(--accent);}}
.roi-label {{font-size:.78em;color:var(--gray);margin-top:4px;}}
.profit-highlight {{text-align:center;font-size:1.3em;font-weight:bold;
  color:var(--good);margin-top:12px;padding:10px;background:#12121e;border-radius:8px;}}
.roi-note {{font-size:.78em;color:var(--gray);margin-top:10px;line-height:1.5;}}
.mvp-card {{background:linear-gradient(135deg,#1a3a1a,#0d0d1a);border:1px solid var(--good);
  border-radius:8px;padding:14px;text-align:center;}}
.mvp-race {{font-size:1.1em;font-weight:bold;}}
.mvp-combo {{color:var(--good);font-size:1.2em;margin:6px 0;}}
.mvp-payout {{color:var(--accent);font-weight:bold;}}
.miss-card {{background:#12121e;border-radius:8px;padding:10px 14px;margin-bottom:8px;
  border-left:3px solid var(--bad);}}
.miss-race {{font-weight:bold;}}
.miss-combo {{color:var(--gray);font-size:.9em;margin:4px 0;}}
.miss-comment {{color:var(--text);font-size:.9em;}}
.no-data {{color:var(--gray);font-size:.88em;}}
.footer {{text-align:center;color:var(--gray);font-size:.85em;padding:20px 0;}}
.footer-comment {{background:var(--card);border-radius:8px;padding:14px;text-align:center;
  color:var(--accent);margin-top:16px;}}
</style>
</head>
<body>

<div class="header">
  <h1>📊 AI実績ページ</h1>
  <div class="meta">{date_disp}分　生成: {now_str}　解析{races_analyzed}レース</div>
</div>

<div class="section">
  <h2>🎯 ブランド別的中実績</h2>
  {_brand_row_html("🚨", "危険艇速報", danger_r)}
  {_brand_row_html("💰", "万舟警報", manshuu_r)}
  {_brand_row_html("🎯", "転がし候補", korogashi_r)}
  {_brand_row_html("⚡", "激走モーター", hot_r)}
  {_brand_row_html("📈", "覚醒モーター", awake_r)}
</div>

<div class="section">
  <h2>💰 購入実績（本日）</h2>
  <div class="roi-grid">
    <div class="roi-cell"><div class="roi-num">{roi_today['n_races']}</div><div class="roi-label">購入レース数</div></div>
    <div class="roi-cell"><div class="roi-num">{roi_today['n_bets']}</div><div class="roi-label">購入点数</div></div>
    <div class="roi-cell"><div class="roi-num">{roi_today['total_cost']:,}円</div><div class="roi-label">総投資金額</div></div>
    <div class="roi-cell"><div class="roi-num">{roi_today['total_return']:,}円</div><div class="roi-label">総払戻金</div></div>
  </div>
  <div class="profit-highlight" style="color:{'var(--good)' if (roi_today['total_return'] - roi_today['total_cost']) >= 0 else 'var(--bad)'}">
    利益: {(roi_today['total_return'] - roi_today['total_cost']):+,}円
  </div>
</div>

<div class="section">
  <h2>💹 回収率・ROI（購入対象のみ集計）</h2>
  <div class="roi-grid">
    <div class="roi-cell"><div class="roi-num">{roi_today['recovery_rate']}%</div><div class="roi-label">本日 回収率</div></div>
    <div class="roi-cell"><div class="roi-num">{roi_today['roi']:+.1f}%</div><div class="roi-label">本日 ROI</div></div>
    <div class="roi-cell"><div class="roi-num">{roi_7d['recovery_rate']}%</div><div class="roi-label">7日平均 回収率</div></div>
    <div class="roi-cell"><div class="roi-num">{roi_30d['recovery_rate']}%</div><div class="roi-label">30日平均 回収率</div></div>
  </div>
  <p class="roi-note">※ AIが見送った（購入しなかった）レースは集計に含みません。実際に購入した場合の成績のみを反映しています。</p>
</div>

<div class="section">
  <h2>🏆 昨日のMVP予想</h2>
  {_mvp_html(mvp)}
</div>

<div class="section">
  <h2>😅 昨日の惜しかった予想</h2>
  {_close_misses_html(close_misses)}
</div>

<div class="footer-comment">
  🤖 本日も改善を続けます。
</div>

<div class="footer">
  {SYSTEM_NAME} | AI Version: {AI_VERSION}
</div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("[公開用実績] HTML保存: %s", output_path)

    return {
        "date_str": date_str, "date_disp": date_disp,
        "danger": danger_r, "manshuu": manshuu_r,
        "roi_today": roi_today, "roi_7d": roi_7d, "roi_30d": roi_30d,
        "mvp": mvp, "close_misses": close_misses,
    }


def generate_public_pdf(html_path: str, pdf_path: str) -> bool:
    """公開用HTMLをPDFに変換する（weasyprint使用）"""
    try:
        from weasyprint import HTML
        HTML(filename=html_path).write_pdf(pdf_path)
        log.info("[公開用実績] PDF保存: %s", pdf_path)
        return True
    except ImportError:
        log.warning("[公開用実績] weasyprint未インストール、PDF生成をスキップ")
        return False
    except Exception as e:
        log.warning("[公開用実績] PDF生成失敗: %s", e)
        return False


def send_public_page(html_path: str, pdf_path: str | None, date_str: str,
                     summary: dict, dry_run: bool = False) -> bool:
    """公開用実績ページをメール送信する"""
    date_disp = summary["date_disp"]
    subject = f"📊 【公開用】AI実績ページ {date_disp}"

    danger = summary["danger"]
    manshuu = summary["manshuu"]
    body = (
        f"AI実績ページ（公開用）{date_disp} を添付します。\n\n"
        f"危険艇: {danger.get('hit',0)}/{danger.get('checked',0)}件的中（{danger.get('rate',0)}%）\n"
        f"万舟:   {manshuu.get('hit',0)}/{manshuu.get('checked',0)}件的中（{manshuu.get('rate',0)}%）\n"
        f"本日回収率: {summary['roi_today']['recovery_rate']}%\n\n"
        "X・noteでの公開用にご利用ください。\n"
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
        log.error("環境変数 GMAIL_ADDRESS / GMAIL_APP_PASS が未設定です")
        return False

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = MAIL_TO
    msg.attach(MIMEText(body, "plain", "utf-8"))

    for path in [html_path, pdf_path]:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment", filename=os.path.basename(path))
            msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(gmail_address, app_password)
            smtp.sendmail(gmail_address, MAIL_TO, msg.as_string())
        log.info("[公開用実績] 送信成功 → %s", MAIL_TO)
        return True
    except smtplib.SMTPException as e:
        log.error("[公開用実績] 送信失敗: %s", e)
        return False
