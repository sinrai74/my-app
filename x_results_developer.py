#!/usr/bin/env python3
"""
x_results_developer.py  ── AI実績ページ【開発用】

目的: AI改善専用の詳細分析レポート。一般公開しない。
特徴量分析・BuyScore分析・外れ理由ランキング・改善候補・前日比較・
学習履歴など、「AI改善に必要な情報」のみを掲載する。

出力: ai_result_developer.html

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
    collect_all_periods, load_daily_stats,
    calc_brand_dev_stats, analyze_features, rank_improvement_candidates,
    classify_miss_reasons, analyze_buyscore_log, get_learning_history,
    calc_day_over_day_comparison,
    analyze_boat_number_bias, generate_improvement_suggestions,
)

log = logging.getLogger("x_results_developer")

JST = timezone(timedelta(hours=9))
DEV_MAIL_TO = "bigkirinuki@gmail.com"   # 開発者宛（一般公開しない）


# ════════════════════════════════════════════════════════════
# 表示パーツ
# ════════════════════════════════════════════════════════════

def _brand_dev_row_html(icon: str, label: str, stats: dict) -> str:
    if not stats.get("has_data", stats.get("data_available", False)):
        reason = stats.get("reason", "データなし")
        return f"""
<tr><td>{icon} {label}</td><td colspan="4" class="no-data">{reason}</td></tr>"""
    checked = stats.get("checked", stats.get("listed", 0))
    hit     = stats.get("hit", 0)
    rate    = stats.get("rate", 0.0)
    recovery = stats.get("recovery_rate", "-")
    roi      = stats.get("roi", "-")
    return f"""
<tr>
  <td>{icon} {label}</td>
  <td>{checked}件</td>
  <td>{hit}件</td>
  <td>{rate}%</td>
  <td>回収{recovery}% / ROI{roi if roi=='-' else f'{roi:+.1f}%'}</td>
</tr>"""


def _feature_analysis_html(analysis: dict) -> str:
    if not analysis.get("data_available"):
        return f'<p class="no-data">{analysis.get("reason", "データ蓄積中")}</p>'
    rows = ""
    for name, stats in analysis.get("features", {}).items():
        corr = stats.get("correlation_with_hit")
        corr_str = f"{corr:+.3f}" if corr is not None else "算出不可"
        rows += f"""
<tr>
  <td>{name}</td><td>{stats['mean']}</td><td>{stats['variance']}</td>
  <td>{stats['n']}件</td><td>{corr_str}</td>
</tr>"""
    return f"""
<table class="dev-table">
  <thead><tr><th>特徴量</th><th>平均値</th><th>分散</th><th>サンプル数</th><th>的中との相関</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<p class="dev-note">n={analysis['n_samples']}件で算出。相関は -1〜+1（絶対値が大きいほど的中との関連が強い）。</p>"""


def _improvement_candidates_html(candidates: list[dict]) -> str:
    if not candidates:
        return '<p class="no-data">学習データ蓄積中のため未算出</p>'
    rows = ""
    for i, c in enumerate(candidates, 1):
        rows += f'<div class="cand-row">{i}位　{c["feature"]}　相関: {c["correlation"]:+.3f}</div>'
    return rows


def _miss_reasons_html(classification: dict) -> str:
    if classification.get("total", 0) == 0:
        return '<p class="no-data">対象データなし</p>'
    rows = ""
    for reason, count in classification.get("categories", {}).items():
        pct = round(count / classification["total"] * 100, 1)
        rows += f'<div class="miss-reason-row">{reason}: {count}件（{pct}%）</div>'
    return rows


def _boat_bias_html(bias: dict) -> str:
    """予想1着艇番 vs 実際1着艇番の分布比較テーブル"""
    if not bias.get("data_available"):
        return f'<p class="no-data">{bias.get("reason", "データ不足")}</p>'
    rows = ""
    for lane in [str(i) for i in range(1, 7)]:
        pred = bias["pred_distribution"][lane]
        actual = bias["actual_distribution"][lane]
        diff = bias["bias"][lane]
        sign = "+" if diff >= 0 else ""
        flag = ""
        if diff >= 3.0:
            flag = " ⚠️過大評価"
        elif diff <= -3.0:
            flag = " ⚠️過小評価"
        rows += f"""
<tr><td>{lane}号艇</td><td>{pred}%</td><td>{actual}%</td><td>{sign}{diff}pt{flag}</td></tr>"""
    return f"""
<table class="dev-table">
  <thead><tr><th>艇番</th><th>予想1着率</th><th>実際1着率</th><th>差分</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
<p class="dev-note">n={bias['n']}件。差分がプラスに大きいほど「予想で1着に選び過ぎ（過大評価）」、
マイナスに大きいほど「実際は来ているのに予想で選べていない（過小評価）」ことを示す。</p>"""


def _improvement_suggestions_html(suggestions: list[str]) -> str:
    """統計ベースで自動生成された「本日の改善候補」"""
    if not suggestions:
        return '<p class="no-data">改善候補を生成するのに十分な外れデータがありません</p>'
    return "".join(f'<div class="suggestion-row">・{s}</div>' for s in suggestions)


def _buyscore_analysis_html(analysis: dict) -> str:
    if not analysis.get("data_available"):
        return f'<p class="no-data">{analysis.get("reason", "データなし")}</p>'
    dist_rows = "".join(
        f'<div class="dist-row">{band}: {count}件</div>'
        for band, count in analysis.get("score_distribution", {}).items()
    )
    return f"""
<div class="buyscore-summary">
  <div>対象レース数: {analysis['n_races']}件</div>
  <div>購入: {analysis['buy_count']}件　見送り: {analysis['passthrough_count']}件（{analysis['passthrough_rate']}%）</div>
</div>
<div class="score-dist">
  <h4>BuyScore分布</h4>
  {dist_rows}
</div>"""


def _learning_history_html(history: dict) -> str:
    parts = []
    bt = history.get("buyscore_tuning")
    if bt:
        parts.append(f"<div>BuyScore重み学習: {bt['last_tuned']}（{bt['last_samples']}件で学習）</div>")
    else:
        parts.append("<div>BuyScore重み学習: 未実行（サンプル不足）</div>")

    am = history.get("asahi_model")
    if am:
        parts.append(f"<div>朝刊AIモデル: {am['model_version']}（Phase{am['phase']}） - {am.get('note','')}</div>")
    return "".join(f'<p class="hist-row">{p}</p>' for p in parts) if False else "".join(parts)


def _comparison_html(comparison: dict) -> str:
    rows = ""
    labels = {"danger": "🚨 危険艇", "manshuu": "💰 万舟"}
    for brand, data in comparison.items():
        y = data.get("yesterday")
        t = data.get("today")
        d = data.get("diff")
        if y is None or t is None:
            rows += f'<div class="comp-row">{labels.get(brand,brand)}: データ不足</div>'
            continue
        sign = "+" if d is not None and d >= 0 else ""
        rows += f'<div class="comp-row">{labels.get(brand,brand)}: 昨日{y}% → 今日{t}%（{sign}{d}%）</div>'
    return rows


# ════════════════════════════════════════════════════════════
# メイン生成関数
# ════════════════════════════════════════════════════════════

def generate_developer_html(date_str: str, output_path: str) -> dict:
    """開発用実績レポートHTMLを生成し、サマリーdictを返す"""
    periods = collect_all_periods(date_str)
    today_records = periods["today"]["records"]
    daily_stats   = load_daily_stats(date_str)

    date_disp = f"{date_str[4:6]}/{date_str[6:8]}"
    now_str   = datetime.now(JST).strftime("%Y/%m/%d %H:%M")

    brand_stats  = calc_brand_dev_stats(today_records, daily_stats)
    feat_analysis = analyze_features(periods["d30"]["records"])
    improve_candidates = rank_improvement_candidates(feat_analysis)
    miss_classification = classify_miss_reasons(today_records)
    boat_bias = analyze_boat_number_bias(today_records)
    improvement_suggestions = generate_improvement_suggestions(miss_classification, boat_bias, top_n=3)
    buyscore_analysis = analyze_buyscore_log(days=7)
    learning_history = get_learning_history()
    comparison = calc_day_over_day_comparison(date_str)

    asahi_version = (learning_history.get("asahi_model") or {}).get("model_version", "unknown")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<title>AI分析レポート（開発用）{date_disp}</title>
<style>
body {{background:#0a0a12;color:#d0d0e0;font-family:'Courier New',monospace;
  max-width:900px;margin:0 auto;padding:16px;font-size:.92em;line-height:1.6;}}
h1 {{color:#00e5ff;font-size:1.4em;border-bottom:2px solid #00e5ff;padding-bottom:8px;}}
h2 {{color:#ffab40;font-size:1.05em;margin-top:24px;}}
h4 {{color:#8a8aa0;font-size:.9em;margin:10px 0 4px;}}
.meta-box {{background:#14141f;border:1px solid #333;border-radius:6px;padding:10px;
  margin-bottom:16px;font-size:.85em;color:#999;}}
.dev-table {{width:100%;border-collapse:collapse;margin:8px 0;}}
.dev-table th, .dev-table td {{border:1px solid #333;padding:6px 10px;text-align:left;}}
.dev-table th {{background:#1a1a2e;color:#00e5ff;}}
.no-data {{color:#666;font-style:italic;}}
.cand-row, .miss-reason-row, .dist-row, .comp-row, .suggestion-row {{padding:4px 0;border-bottom:1px solid #222;}}
.suggestion-row {{color:#69f0ae;}}
.dev-note {{color:#666;font-size:.8em;}}
.buyscore-summary {{background:#14141f;padding:10px;border-radius:6px;margin-bottom:10px;}}
</style>
</head>
<body>

<h1>🔧 AI分析レポート（開発用・非公開）</h1>
<div class="meta-box">
  AI Version: {AI_VERSION}　|　モデルVersion: {asahi_version}　|　生成日時: {now_str}　|　対象日: {date_disp}
</div>

<h2>📊 ブランド別詳細</h2>
<table class="dev-table">
  <thead><tr><th>ブランド</th><th>件数</th><th>的中</th><th>的中率</th><th>回収率/ROI</th></tr></thead>
  <tbody>
    {_brand_dev_row_html("🚨", "危険艇", brand_stats["danger"])}
    {_brand_dev_row_html("💰", "万舟", brand_stats["manshuu"])}
    {_brand_dev_row_html("🎯", "転がし", brand_stats["korogashi"])}
    {_brand_dev_row_html("⚡", "激走モーター", brand_stats["hot_motor"])}
    {_brand_dev_row_html("📈", "覚醒モーター", brand_stats["awakening_motor"])}
  </tbody>
</table>

<h2>📈 前日比較</h2>
{_comparison_html(comparison)}

<h2>🧬 特徴量分析（過去30日）</h2>
{_feature_analysis_html(feat_analysis)}

<h2>💡 改善候補 TOP5（的中との相関が強い特徴量）</h2>
{_improvement_candidates_html(improve_candidates)}

<h2>🎲 BuyScore分析（過去7日）</h2>
{_buyscore_analysis_html(buyscore_analysis)}

<h2>❌ 外れ理由ランキング（本日）</h2>
{_miss_reasons_html(miss_classification)}

<h2>🚤 艇番評価バイアス分析（本日）</h2>
{_boat_bias_html(boat_bias)}

<h2>🎯 本日の改善候補</h2>
{_improvement_suggestions_html(improvement_suggestions)}

<h2>🎓 学習履歴</h2>
{_learning_history_html(learning_history)}

<p style="color:#444;font-size:.8em;margin-top:30px;">
本レポートは開発・AI改善専用です。一般公開しないでください。
</p>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("[開発用レポート] HTML保存: %s", output_path)

    return {
        "date_str": date_str, "date_disp": date_disp,
        "brand_stats": brand_stats, "feat_analysis": feat_analysis,
        "improve_candidates": improve_candidates,
        "miss_classification": miss_classification,
        "boat_bias": boat_bias,
        "improvement_suggestions": improvement_suggestions,
        "buyscore_analysis": buyscore_analysis,
        "learning_history": learning_history,
        "comparison": comparison,
    }


def send_developer_report(html_path: str, date_str: str, summary: dict,
                          dry_run: bool = False) -> bool:
    """
    開発用レポートを送信する（デフォルトでは呼び出さない想定。
    一般公開しない前提のため、必要な場合のみ明示的に呼ぶこと）。
    """
    date_disp = summary["date_disp"]
    subject = f"🔧 【開発用・非公開】AI分析レポート {date_disp}"
    body = f"AI分析レポート（開発用）{date_disp} を添付します。\n本レポートは非公開・開発専用です。\n"

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
    msg["To"] = DEV_MAIL_TO
    msg.attach(MIMEText(body, "plain", "utf-8"))

    if os.path.exists(html_path):
        with open(html_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment", filename=os.path.basename(html_path))
        msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(gmail_address, app_password)
            smtp.sendmail(gmail_address, DEV_MAIL_TO, msg.as_string())
        log.info("[開発用レポート] 送信成功 → %s", DEV_MAIL_TO)
        return True
    except smtplib.SMTPException as e:
        log.error("[開発用レポート] 送信失敗: %s", e)
        return False
