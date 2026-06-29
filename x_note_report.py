#!/usr/bin/env python3
"""
x_note_report.py  ── note有料コンテンツ用 AI競艇新聞 生成

出力物（毎朝7時 → メール添付）:
  note.html          ← noteエディタへ貼り付け（スマホ対応）
  newspaper.pdf      ← 保存・印刷用（PDF添付）
  danger.csv         ← 危険な1号艇 全データ
  manshuu.csv        ← 万舟警報 全データ
  motor.csv          ← 激走モーター データ
  awake.csv          ← 覚醒モーター データ

Usage:
    python x_note_report.py              # 生成 + メール送信
    python x_note_report.py --dry-run    # 生成のみ（送信なし）
    python x_note_report.py --date 20260628
"""

from __future__ import annotations

import argparse
import csv
import json
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

log = logging.getLogger("x_note")

JST           = timezone(timedelta(hours=9))
MAIL_TO       = "bigkirinuki@gmail.com"
RANKING_CACHE = "ranking_cache.json"


# ════════════════════════════════════════════════════════════
# AI編集部コメント（総評）生成
# ════════════════════════════════════════════════════════════

def _generate_editorial(data: dict) -> dict:
    """
    全データを俯瞰して「今日の総評」を生成する。
    戻り値:
      {
        "headline": "最注目レース名",
        "reason":   "複合理由テキスト",
        "summary":  "全体サマリーテキスト",
        "s_danger": int,  "s_manshuu": int,
        "awake_count": int, "hot_count": int,
      }
    """
    all_danger  = data.get("all_danger",  data.get("danger_boat1", []))
    all_manshuu = data.get("all_manshuu", data.get("manshuu_alert", []))
    hot_motor   = data.get("hot_motor", [])
    awake_motor = data.get("awakening_motor", [])

    s_danger  = [d for d in all_danger  if d.get("score", 0) >= 80]
    s_manshuu = [u for u in all_manshuu if u.get("score", 0) >= 80]

    # 最注目レース: 危険Sランク + 万舟SランクのW一致を優先
    headline_race = None
    headline_reasons: list[str] = []

    danger_keys  = {(d["venue_num"], d["race"]): d for d in all_danger}
    manshuu_keys = {(u["venue_num"], u["race"]): u for u in all_manshuu}

    # ダブル一致レースを探す
    double = [(k, v) for k, v in danger_keys.items() if k in manshuu_keys]
    double_s = [(k, v) for k, v in double
                if v.get("score", 0) >= 80 and manshuu_keys[k].get("score", 0) >= 80]

    if double_s:
        k, d = double_s[0]
        headline_race = f"{d['venue']}{d['race']}R"
        headline_reasons = [
            f"危険な1号艇 Sランク（危険度{d['score']}）",
            f"万舟警報 Sランク（荒れ指数{manshuu_keys[k]['score']}）",
            "この2つが重なった唯一のレース",
        ]
    elif double:
        k, d = double[0]
        headline_race = f"{d['venue']}{d['race']}R"
        headline_reasons = [
            f"危険な1号艇（危険度{d['score']}）",
            f"万舟警報（荒れ指数{manshuu_keys[k]['score']}）",
        ]
    elif s_danger:
        d = s_danger[0]
        headline_race = f"{d['venue']}{d['race']}R"
        headline_reasons = [
            f"危険な1号艇 Sランク最高危険度{d['score']}",
            f"{d.get('racer', '?')}（{d.get('reason', '')}）",
        ]
    elif all_danger:
        d = all_danger[0]
        headline_race = f"{d['venue']}{d['race']}R"
        headline_reasons = [f"本日最高危険度{d['score']}"]

    # 全体サマリー
    summary_parts = []
    if s_danger:
        summary_parts.append(f"危険な1号艇 Sランク{len(s_danger)}件")
    if s_manshuu:
        summary_parts.append(f"万舟警報 Sランク{len(s_manshuu)}件")
    if hot_motor:
        summary_parts.append(f"激走モーター{len(hot_motor)}件")
    if awake_motor:
        summary_parts.append(f"覚醒モーター{len(awake_motor)}件")
    summary = "・".join(summary_parts) if summary_parts else "本日の注目データをまとめました"

    return {
        "headline":    headline_race or "本日の注目レース",
        "reasons":     headline_reasons,
        "summary":     summary,
        "s_danger":    len(s_danger),
        "s_manshuu":   len(s_manshuu),
        "hot_count":   len(hot_motor),
        "awake_count": len(awake_motor),
        "all_danger_count":  len(all_danger),
        "all_manshuu_count": len(all_manshuu),
    }


# ════════════════════════════════════════════════════════════
# レースごとのAIコメント（1〜2行）
# ════════════════════════════════════════════════════════════

def _race_comment(d: dict) -> str:
    bd = d.get("breakdown", {})
    items = [
        ("ST",   bd.get("st",    (0,0))[1], "平均ST遅れリスク"),
        ("展示",  bd.get("ex",    (0,0))[1], "展示タイム低調"),
        ("機力",  bd.get("motor", (0,0))[1], "モーター力不足"),
        ("等級",  bd.get("grade", (0,0))[1], "格下等級"),
        ("勝率",  bd.get("wr",    (0,0))[1], "勝率が低調"),
        ("相手",  bd.get("rival", (0,0))[1], "強力な対抗馬"),
    ]
    top = sorted(items, key=lambda x: -x[1])[:2]
    parts = [label for label, val, _ in top if val >= 8]
    if not parts:
        return "総合的な危険判定"
    return " + ".join(parts) + "が主因"


def _manshuu_comment(u: dict) -> str:
    reasons = u.get("key_reason", "").split(" / ")
    clean = [r.replace("🔥 ", "").replace("🔥", "").strip() for r in reasons if r.strip()]
    return "、".join(clean[:2]) if clean else "複合要因で荒れ判定"


# ════════════════════════════════════════════════════════════
# CSV生成
# ════════════════════════════════════════════════════════════

def generate_csvs(data: dict, prefix: str = "") -> list[str]:
    """4種類のCSVを生成してパスのリストを返す"""
    paths: list[str] = []
    date_str = data.get("date", "")

    # ── danger.csv ──────────────────────────────────────────
    path = f"{prefix}danger.csv"
    all_danger = data.get("all_danger", data.get("danger_boat1", []))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["日付","会場","レース","危険度","ランク","選手名","等級",
                    "勝率","モーター2連率","平均ST","展示タイム","判定理由","AIコメント"])
        for d in all_danger:
            score = d.get("score", 0)
            rank  = "S" if score >= 80 else "A" if score >= 60 else "B"
            w.writerow([
                date_str, d.get("venue",""), d.get("race",""),
                score, rank, d.get("racer",""), d.get("racer_class",""),
                d.get("win_rate",""), d.get("motor",""),
                d.get("avg_st",""), d.get("ex_time",""),
                d.get("reason",""), _race_comment(d),
            ])
    paths.append(path)

    # ── manshuu.csv ─────────────────────────────────────────
    path = f"{prefix}manshuu.csv"
    all_manshuu = data.get("all_manshuu", data.get("manshuu_alert", []))
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["日付","会場","レース","荒れ指数","ランク","注目選手","荒れる理由","AIコメント"])
        for u in all_manshuu:
            score = u.get("score", 0)
            rank  = "S" if score >= 80 else "A" if score >= 60 else "B"
            w.writerow([
                date_str, u.get("venue",""), u.get("race",""),
                score, rank, u.get("key_racer",""),
                u.get("key_reason",""), _manshuu_comment(u),
            ])
    paths.append(path)

    # ── motor.csv ────────────────────────────────────────────
    path = f"{prefix}motor.csv"
    hot_motor = data.get("hot_motor", [])
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["日付","会場","モーター番号","激走指数","直近5走","公式2連率","実績2連率","公式比"])
        for m in hot_motor:
            w.writerow([
                date_str, m.get("venue",""), m.get("motor_no",""),
                m.get("score",""), m.get("recent5","---"),
                m.get("official_2rate",""), m.get("recent10_2rate",""),
                m.get("gap",""),
            ])
    paths.append(path)

    # ── awake.csv ────────────────────────────────────────────
    path = f"{prefix}awake.csv"
    awake_motor = data.get("awakening_motor", [])
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["日付","会場","モーター番号","覚醒指数","直近10走","旧2連率","新2連率","展示平均"])
        for a in awake_motor:
            w.writerow([
                date_str, a.get("venue",""), a.get("motor_no",""),
                a.get("score",""), a.get("recent10","---"),
                a.get("old_2rate",""), a.get("new_2rate",""),
                a.get("ex_avg",""),
            ])
    paths.append(path)

    log.info("[CSV] %d件生成", len(paths))
    return paths


# ════════════════════════════════════════════════════════════
# HTML生成（noteコピペ用・スマホ対応）
# ════════════════════════════════════════════════════════════

def generate_html(data: dict, output_path: str) -> None:
    date_str  = data.get("date", "")
    date_disp = f"{date_str[4:6]}/{date_str[6:8]}" if len(date_str) >= 8 else ""
    now_str   = datetime.now(JST).strftime("%H:%M")

    all_danger  = data.get("all_danger",  data.get("danger_boat1", []))
    all_manshuu = data.get("all_manshuu", data.get("manshuu_alert", []))
    hot_motor   = data.get("hot_motor", [])
    awake_motor = data.get("awakening_motor", [])
    editorial   = _generate_editorial(data)

    s_d = len([d for d in all_danger  if d.get("score",0) >= 80])
    a_d = len([d for d in all_danger  if 60 <= d.get("score",0) < 80])
    b_d = len([d for d in all_danger  if 40 <= d.get("score",0) < 60])
    s_m = len([u for u in all_manshuu if u.get("score",0) >= 80])

    def rank_label(score):
        if score >= 80: return "🔴 Sランク"
        if score >= 60: return "🟠 Aランク"
        return "🟡 Bランク"

    def danger_rows():
        rows = ""
        for d in all_danger:
            score = d.get("score", 0)
            comment = _race_comment(d)
            bd = d.get("breakdown", {})
            bar_items = [
                ("ST",   bd.get("st",    (0,0))[1], "#ef5350"),
                ("展示",  bd.get("ex",    (0,0))[1], "#ff7043"),
                ("機力",  bd.get("motor", (0,0))[1], "#ffa726"),
                ("等級",  bd.get("grade", (0,0))[1], "#ab47bc"),
                ("勝率",  bd.get("wr",    (0,0))[1], "#42a5f5"),
                ("相手",  bd.get("rival", (0,0))[1], "#26a69a"),
            ]
            bars = "".join(
                f'<div class="bi"><span class="bl">{l}</span>'
                f'<div class="bb"><div class="bf" style="width:{int(v/20*100)}%;background:{c}"></div></div>'
                f'<span class="bv">{v:.0f}pt</span></div>'
                for l, v, c in bar_items
            )
            rank_cls = "s" if score >= 80 else "a" if score >= 60 else "b"
            rows += f"""
<div class="race-card {rank_cls}">
  <div class="rc-header">
    <span class="badge {rank_cls}">{rank_label(score)}</span>
    <strong class="rc-name">{d.get('venue','')}{d.get('race','')}R</strong>
    <span class="rc-racer">{d.get('racer','?')} {d.get('racer_class','')}</span>
  </div>
  <div class="rc-reason">{d.get('reason','')}</div>
  <div class="rc-comment">💬 {comment}</div>
  <div class="breakdown">{bars}</div>
</div>"""
        return rows

    def manshuu_rows():
        rows = ""
        for u in all_manshuu:
            score = u.get("score", 0)
            rank_cls = "s" if score >= 80 else "a" if score >= 60 else "b"
            reasons = u.get("key_reason","").split(" / ")
            reason_html = "".join(f"<div class='mr'>{r}</div>" for r in reasons if r.strip())
            rows += f"""
<div class="race-card {rank_cls}">
  <div class="rc-header">
    <span class="badge {rank_cls}">{rank_label(score)}</span>
    <strong class="rc-name">{u.get('venue','')}{u.get('race','')}R</strong>
    <span class="rc-racer">注目: {u.get('key_racer','')}</span>
  </div>
  {reason_html}
  <div class="rc-comment">💬 {_manshuu_comment(u)}</div>
</div>"""
        return rows

    def motor_table(items, cols):
        if not items:
            return '<p class="no-data">データ蓄積中（数日後に表示されます）</p>'
        header = "".join(f"<th>{c}</th>" for c in cols[0])
        rows = ""
        for i, m in enumerate(items, 1):
            cells = "".join(f"<td>{m.get(k,'')}</td>" for k in cols[1])
            rows += f"<tr><td>{i}</td>{cells}</tr>"
        return f'<table class="mt"><thead><tr><th>#</th>{header}</tr></thead><tbody>{rows}</tbody></table>'

    hot_cols  = (["会場","モーター","直近5走","公式比"],
                 ["venue","motor_no","recent5","gap"])
    awake_cols = (["会場","モーター","直近10走","2連率変化","展示平均"],
                  ["venue","motor_no","recent10","old_2rate","ex_avg"])

    # 昨日の実績を取得
    yesterday_block = ""
    try:
        from x_verification import get_yesterday_summary
        yday = get_yesterday_summary()
        if yday:
            profit = yday.get("profit", 0)
            p_mark = "✅ 利益" if profit >= 0 else "❌ 損失"
            yesterday_block = f"""
<section id="yesterday">
  <h2>📊 昨日のAI成績</h2>
  <div class="yday-grid">
    <div class="yc"><div class="yn">{yday.get('total_notified',0)}</div><div class="yl">通知レース</div></div>
    <div class="yc"><div class="yn">{yday.get('boat1_flew_rate',0):.0f}%</div><div class="yl">イン逃げ失敗率</div></div>
    <div class="yc"><div class="yn">{p_mark}<br>{profit:+,}円</div><div class="yl">損益</div></div>
  </div>
</section>"""
    except Exception:
        pass

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI競艇新聞 {date_disp}</title>
<style>
:root{{
  --bg:#0d0d1a;--card:#16162a;--border:#252540;
  --text:#e8e8f0;--gray:#8888aa;--accent:#4fc3f7;
  --s:#ef5350;--a:#ffa726;--b:#66bb6a;--hot:#ff7043;--awake:#00bcd4;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--text);font-family:'Hiragino Sans','Noto Sans JP',sans-serif;
  line-height:1.7;padding:12px;max-width:780px;margin:0 auto}}
/* 表紙 */
.cover{{text-align:center;border:2px solid var(--accent);border-radius:12px;
  padding:28px 16px;margin-bottom:20px;background:var(--card)}}
.cover h1{{font-size:2em;color:var(--accent);letter-spacing:.1em}}
.cover .date{{color:var(--gray);margin:6px 0 14px;font-size:.9em}}
.kpi-grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-top:14px}}
@media(min-width:480px){{.kpi-grid{{grid-template-columns:repeat(4,1fr)}}}}
.kpi{{background:#1a1a30;border-radius:8px;padding:12px 8px;text-align:center}}
.kpi .kn{{font-size:1.8em;font-weight:bold}}
.kpi .kl{{font-size:.75em;color:var(--gray);margin-top:4px}}
.kn.s{{color:var(--s)}}.kn.hot{{color:var(--hot)}}.kn.awake{{color:var(--awake)}}
/* 総評 */
.editorial{{background:#0e1e0e;border:1px solid #2a4a2a;border-radius:10px;
  padding:18px;margin-bottom:20px}}
.editorial h2{{color:#81c784;margin-bottom:10px}}
.editorial .headline{{font-size:1.2em;font-weight:bold;color:#fff;margin-bottom:8px}}
.editorial .reasons li{{color:#aaa;font-size:.9em;margin:4px 0 4px 16px}}
.editorial .summary{{color:#aaa;font-size:.88em;margin-top:10px;padding-top:10px;
  border-top:1px solid #2a4a2a}}
/* セクション */
section{{margin-bottom:24px}}
section h2{{font-size:1.2em;color:var(--accent);padding:10px 0;
  border-bottom:1px solid var(--border);margin-bottom:12px}}
/* レースカード */
.race-card{{background:var(--card);border-radius:8px;padding:14px;
  margin-bottom:10px;border-left:4px solid var(--border)}}
.race-card.s{{border-left-color:var(--s)}}
.race-card.a{{border-left-color:var(--a)}}
.race-card.b{{border-left-color:var(--b)}}
.rc-header{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px}}
.badge{{padding:2px 8px;border-radius:4px;font-size:.78em;font-weight:bold}}
.badge.s{{background:var(--s);color:#fff}}
.badge.a{{background:var(--a);color:#000}}
.badge.b{{background:var(--b);color:#000}}
.rc-name{{font-size:1.05em}}
.rc-racer{{color:var(--gray);font-size:.88em}}
.rc-reason{{color:#bbb;font-size:.85em;margin:4px 0}}
.rc-comment{{color:var(--accent);font-size:.85em;margin-top:4px}}
.mr{{color:#bbb;font-size:.85em;margin:3px 0}}
/* スコアバー */
.breakdown{{margin-top:8px;display:grid;grid-template-columns:repeat(3,1fr);gap:4px}}
@media(min-width:480px){{.breakdown{{grid-template-columns:repeat(6,1fr)}}}}
.bi{{display:flex;flex-direction:column;gap:2px}}
.bl{{font-size:.7em;color:var(--gray);text-align:center}}
.bb{{background:#252540;border-radius:2px;height:8px;overflow:hidden}}
.bf{{height:100%;border-radius:2px;transition:width .3s}}
.bv{{font-size:.7em;color:var(--gray);text-align:center}}
/* 昨日の実績 */
#yesterday{{background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:16px;margin-bottom:20px}}
#yesterday h2{{color:var(--accent);margin-bottom:12px;font-size:1.1em}}
.yday-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;text-align:center}}
.yc .yn{{font-size:1.3em;font-weight:bold;color:#fff}}
.yc .yl{{font-size:.78em;color:var(--gray);margin-top:3px}}
/* モーターテーブル */
.mt{{width:100%;border-collapse:collapse;font-size:.85em}}
.mt th{{background:#1a1a30;color:var(--gray);padding:6px 8px;text-align:left;font-weight:normal}}
.mt td{{padding:6px 8px;border-bottom:1px solid var(--border)}}
.no-data{{color:var(--gray);font-size:.9em;padding:12px 0}}
/* ダウンロードセクション */
.dl-section{{background:#0e1a2e;border:1px solid #1a3a5a;border-radius:10px;padding:18px;margin-bottom:20px}}
.dl-section h2{{color:#4fc3f7;margin-bottom:12px;font-size:1.1em}}
.dl-list{{list-style:none;padding:0}}
.dl-list li{{padding:8px 0;border-bottom:1px solid #1a3a5a;color:#aaa;font-size:.9em}}
.dl-list li:last-child{{border-bottom:none}}
/* フッター */
.footer{{text-align:center;color:#444;font-size:.78em;padding:20px 0}}
</style>
</head>
<body>

<!-- ① 表紙 -->
<div class="cover">
  <h1>📰 AI競艇新聞</h1>
  <div class="date">{date_disp}発行　{now_str}生成　毎朝7時更新</div>
  <div class="kpi-grid">
    <div class="kpi"><div class="kn s">{s_d}</div><div class="kl">危険Sランク</div></div>
    <div class="kpi"><div class="kn s">{s_m}</div><div class="kl">万舟Sランク</div></div>
    <div class="kpi"><div class="kn hot">{len(hot_motor)}</div><div class="kl">激走モーター</div></div>
    <div class="kpi"><div class="kn awake">{len(awake_motor)}</div><div class="kl">覚醒モーター</div></div>
  </div>
</div>

<!-- ② 今日の総評（AIコメント） -->
<div class="editorial">
  <h2>🤖 AI編集部 今日の総評</h2>
  <div class="headline">本日最注目：{editorial['headline']}</div>
  <ul class="reasons">
    {''.join(f"<li>{r}</li>" for r in editorial['reasons'])}
  </ul>
  <div class="summary">📊 {editorial['summary']}</div>
</div>

<!-- ③ 危険な1号艇 -->
<section id="danger">
  <h2>⚠️ 危険な1号艇　全{len(all_danger)}レース</h2>
  {danger_rows()}
</section>

<!-- ④ 万舟警報 -->
<section id="manshuu">
  <h2>💰 万舟警報　全{len(all_manshuu)}レース</h2>
  {manshuu_rows()}
</section>

<!-- ⑤ 激走モーター -->
<section id="motor">
  <h2>🔥 激走モーター　全{len(hot_motor)}件</h2>
  {motor_table(hot_motor, hot_cols)}
</section>

<!-- ⑥ 覚醒モーター -->
<section id="awake">
  <h2>⚡ 覚醒モーター　全{len(awake_motor)}件</h2>
  {motor_table(awake_motor, awake_cols)}
</section>

<!-- ⑦ 昨日の検証 -->
{yesterday_block}

<!-- ⑧⑨ ダウンロード案内 -->
<div class="dl-section">
  <h2>📥 データダウンロード（購入者特典）</h2>
  <ul class="dl-list">
    <li>📄 newspaper.pdf　── 印刷・保存用PDF版（全データ収録）</li>
    <li>📊 danger.csv　── 危険な1号艇 全{len(all_danger)}レースの詳細データ</li>
    <li>📊 manshuu.csv　── 万舟警報 全{len(all_manshuu)}レースの詳細データ</li>
    <li>📊 motor.csv　── 激走モーター データ一覧</li>
    <li>📊 awake.csv　── 覚醒モーター データ一覧</li>
    <li style="color:#666;font-size:.82em">※ 本記事購入後、メッセージよりデータをリクエストしてください</li>
  </ul>
</div>

<div class="footer">
  AI競艇新聞 | 全レース機械学習分析 | 毎日更新<br>
  ※本レポートはデータ分析結果であり、的中を保証するものではありません。
</div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("[HTML] 保存: %s (%d bytes)", output_path, len(html))


# ════════════════════════════════════════════════════════════
# PDF生成
# ════════════════════════════════════════════════════════════

def generate_pdf(html_path: str, pdf_path: str) -> bool:
    try:
        from weasyprint import HTML
        HTML(filename=html_path).write_pdf(pdf_path)
        log.info("[PDF] 保存(weasyprint): %s", pdf_path)
        return True
    except ImportError:
        pass
    except Exception as e:
        log.warning("[PDF] weasyprint失敗: %s", e)

    try:
        import pdfkit
        pdfkit.from_file(html_path, pdf_path, options={"encoding":"utf-8","quiet":""})
        log.info("[PDF] 保存(pdfkit): %s", pdf_path)
        return True
    except Exception as e:
        log.warning("[PDF] pdfkit失敗: %s", e)

    # Pillowフォールバック（ダークテーマ・構造的レンダリング）
    try:
        from PIL import Image, ImageDraw, ImageFont
        import re as _re, json as _json

        # ── フォント取得 ─────────────────────────────────────
        def _gf(size, bold=False):
            candidates = (
                ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                 "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
                 "C:/Windows/Fonts/meiryob.ttc"] if bold else
                ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                 "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
                 "C:/Windows/Fonts/meiryo.ttc"]
            )
            for p in candidates:
                if os.path.exists(p):
                    try: return ImageFont.truetype(p, size)
                    except: pass
            return ImageFont.load_default()

        # ── HTMLからデータを再取得 ────────────────────────────
        # ranking_cache.json が隣にあれば使う
        cache_path = os.path.join(os.path.dirname(html_path), "ranking_cache.json")
        if os.path.exists(cache_path):
            with open(cache_path, encoding="utf-8") as f:
                data = _json.load(f)
        else:
            data = {}

        all_danger  = data.get("all_danger",  data.get("danger_boat1", []))
        all_manshuu = data.get("all_manshuu", data.get("manshuu_alert", []))
        hot_motor   = data.get("hot_motor", [])
        awake_motor = data.get("awakening_motor", [])
        date_str_d  = data.get("date", "")
        date_disp   = f"{date_str_d[4:6]}/{date_str_d[6:8]}" if len(date_str_d) >= 8 else ""

        # ── カラー定義 ────────────────────────────────────────
        C_BG     = (13,  13,  26)
        C_CARD   = (22,  22,  42)
        C_WHITE  = (232, 232, 240)
        C_GRAY   = (136, 136, 170)
        C_ACCENT = ( 79, 195, 247)
        C_S      = (239,  83,  80)
        C_A      = (255, 167,  38)
        C_B      = (102, 187, 106)
        C_SEP    = ( 37,  37,  64)

        W = 1200
        fhd  = _gf(38, bold=True)
        fbig = _gf(28, bold=True)
        fmd  = _gf(22, bold=True)
        fsm  = _gf(18)
        fxs  = _gf(15)

        def txt_w(text, font):
            try: return font.getbbox(text)[2] - font.getbbox(text)[0]
            except: return len(text) * 14

        def rank_color(score):
            if score >= 80: return C_S
            if score >= 60: return C_A
            return C_B

        def rank_label(score):
            if score >= 80: return "S"
            if score >= 60: return "A"
            return "B"

        # ── 描画ブロックをリストで構築 ────────────────────────
        # (type, *args) 形式
        blocks = []  # (y_cost, draw_fn)

        # ページ幅固定で各ブロックの高さを計算してから一気に描く

        # セクション描画用のリスト形式
        # 最終的にブロックの合計高さを計算してImageを作る

        draw_commands = []  # (y, fn) リスト

        y = [0]  # mutable

        def add_gap(h):
            y[0] += h

        def draw_header(draw):
            # ヘッダー背景
            draw.rectangle([0, y[0], W, y[0]+80], fill=(26,35,126))
            draw.rectangle([0, y[0], 6, y[0]+80], fill=C_ACCENT)
            # タイトル
            t = f"AI競艇新聞  {date_disp}"
            draw.text((W//2 - txt_w(t, fhd)//2, y[0]+12), t, font=fhd, fill=C_WHITE)
            sub = "全レースAI分析 | 毎朝7時更新"
            draw.text((W//2 - txt_w(sub, fxs)//2, y[0]+56), sub, font=fxs, fill=C_GRAY)
            y[0] += 100

        def draw_section_title(draw, title, color=C_ACCENT):
            draw.rectangle([0, y[0], W, y[0]+48], fill=C_CARD)
            draw.rectangle([0, y[0], 4, y[0]+48], fill=color)
            draw.text((16, y[0]+10), title, font=fmd, fill=color)
            y[0] += 58

        def draw_danger_row(draw, d):
            score = d.get("score", 0)
            rc    = rank_color(score)
            rl    = rank_label(score)
            # 行背景
            draw.rectangle([0, y[0], W, y[0]+62], fill=C_CARD)
            draw.rectangle([0, y[0], 4, y[0]+62], fill=rc)
            # ランクバッジ
            draw.rectangle([12, y[0]+14, 54, y[0]+48], fill=rc)
            draw.text((18, y[0]+16), rl, font=fmd, fill=C_WHITE)
            # レース名
            race_t = f"{d.get('venue','')}{d.get('race','')}R"
            draw.text((64, y[0]+10), race_t, font=fbig, fill=C_WHITE)
            # 選手名・等級
            racer_t = f"{d.get('racer','?')} {d.get('racer_class','')}"
            draw.text((64, y[0]+38), racer_t, font=fxs, fill=C_GRAY)
            # 理由（右寄せ）
            reason = d.get("reason", "")[:40]
            draw.text((320, y[0]+10), reason, font=fxs, fill=C_A)
            # セパレータ
            draw.rectangle([0, y[0]+62, W, y[0]+63], fill=C_SEP)
            y[0] += 64

        def draw_manshuu_row(draw, u):
            score = u.get("score", 0)
            rc    = rank_color(score)
            rl    = rank_label(score)
            draw.rectangle([0, y[0], W, y[0]+62], fill=C_CARD)
            draw.rectangle([0, y[0], 4, y[0]+62], fill=rc)
            draw.rectangle([12, y[0]+14, 54, y[0]+48], fill=rc)
            draw.text((18, y[0]+16), rl, font=fmd, fill=C_WHITE)
            race_t = f"{u.get('venue','')}{u.get('race','')}R"
            draw.text((64, y[0]+10), race_t, font=fbig, fill=C_WHITE)
            key_racer = u.get("key_racer", "")
            draw.text((64, y[0]+38), f"注目: {key_racer}", font=fxs, fill=C_GRAY)
            # 荒れ理由
            reason_raw = u.get("key_reason", "")
            reason_clean = _re.sub(r'🔥\s*', '', reason_raw)[:45]
            draw.text((320, y[0]+10), reason_clean, font=fxs, fill=C_A)
            draw.rectangle([0, y[0]+62, W, y[0]+63], fill=C_SEP)
            y[0] += 64

        def draw_motor_row(draw, m, is_awake=False):
            draw.rectangle([0, y[0], W, y[0]+46], fill=C_CARD)
            venue_t = f"{m.get('venue','')} {m.get('motor_no','')}号機"
            draw.text((16, y[0]+4), venue_t, font=fmd, fill=C_WHITE)
            if is_awake:
                detail = f"直近10走: {m.get('recent10','---')}  {m.get('old_2rate','')}%→{m.get('new_2rate','')}%  展示{m.get('ex_avg','')}秒"
            else:
                gap = m.get("gap", "")
                gs = f"+{gap:.0f}%" if isinstance(gap,(int,float)) and gap > 0 else f"{gap}"
                detail = f"直近5走: {m.get('recent5','---')}  公式比{gs}"
            draw.text((16, y[0]+28), detail, font=fxs, fill=C_GRAY)
            draw.rectangle([0, y[0]+46, W, y[0]+47], fill=C_SEP)
            y[0] += 48

        # ── 高さ計算（描画なしで一巡）───────────────────────
        # ヘッダー: 100
        # 危険 セクションタイトル: 58、各行: 64
        # 万舟 セクションタイトル: 58、各行: 64
        # 激走 セクションタイトル: 58、各行: 48
        # 覚醒 セクションタイトル: 58、各行: 48
        # フッター: 60、余白: 30

        total_h = (
            100 + 20 +
            58 + len(all_danger)  * 64 + 20 +
            58 + len(all_manshuu) * 64 + 20 +
            58 + len(hot_motor)   * 48 + 20 +
            58 + len(awake_motor) * 48 + 60
        )
        total_h = max(1600, total_h)

        # ── 実際に描画 ────────────────────────────────────────
        img  = Image.new("RGB", (W, total_h), C_BG)
        draw = ImageDraw.Draw(img)

        y[0] = 0
        draw_header(draw)
        add_gap(20)

        draw_section_title(draw, f"⚠ 危険な1号艇  全{len(all_danger)}レース", C_S)
        for d in all_danger:
            draw_danger_row(draw, d)
        add_gap(20)

        draw_section_title(draw, f"¥ 万舟警報  全{len(all_manshuu)}レース", C_A)
        for u in all_manshuu:
            draw_manshuu_row(draw, u)
        add_gap(20)

        draw_section_title(draw, f"激走モーター  全{len(hot_motor)}件", (255,112,67))
        for m in hot_motor:
            draw_motor_row(draw, m, is_awake=False)
        add_gap(20)

        draw_section_title(draw, f"覚醒モーター  全{len(awake_motor)}件", (0,188,212))
        for a in awake_motor:
            draw_motor_row(draw, a, is_awake=True)

        # フッター
        draw.rectangle([0, total_h-50, W, total_h], fill=(20,20,40))
        ft = "AI競艇新聞 | 全レース機械学習分析 | 毎日更新"
        draw.text((W//2 - txt_w(ft, fxs)//2, total_h-34), ft, font=fxs, fill=C_GRAY)

        img.save(pdf_path, "PDF", resolution=120)
        log.info("[PDF] 保存(Pillow dark): %s", pdf_path)
        return True
    except Exception as e:
        log.warning("[PDF] Pillow失敗: %s", e)

    return False


# ════════════════════════════════════════════════════════════
# メール送信
# ════════════════════════════════════════════════════════════

def send_note_report(
    html_path: str,
    pdf_path: Optional[str],
    csv_paths: list[str],
    date_str: str,
    dry_run: bool = False,
) -> bool:
    date_disp = f"{date_str[4:6]}/{date_str[6:8]}"
    subject   = f"📰 AI競艇新聞 {date_disp}（note用）"
    body = (
        f"AI競艇新聞 {date_disp} を生成しました。\n\n"
        "【添付ファイル一覧】\n"
        f"  note.html     ← noteエディタにコピペ\n"
        f"  newspaper.pdf ← 印刷・保存用PDF\n"
        f"  danger.csv    ← 危険な1号艇 全データ\n"
        f"  manshuu.csv   ← 万舟警報 全データ\n"
        f"  motor.csv     ← 激走モーター データ\n"
        f"  awake.csv     ← 覚醒モーター データ\n\n"
        "【noteへの貼り付け手順】\n"
        "  1. note.html をブラウザで開く\n"
        "  2. Ctrl+A → Ctrl+C でコピー\n"
        "  3. noteエディタに貼り付け\n"
        "  4. newspaper.pdf を記事末尾に添付\n"
    )

    if dry_run:
        print("=" * 60)
        print(f"[DRY RUN] 件名: {subject}")
        print(body)
        files = [html_path]
        if pdf_path: files.append(pdf_path)
        files.extend(csv_paths)
        print(f"添付: {files}")
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

    attach_files = [(html_path, "text", "html")]
    if pdf_path and os.path.exists(pdf_path):
        attach_files.append((pdf_path, "application", "pdf"))
    for p in csv_paths:
        if os.path.exists(p):
            attach_files.append((p, "text", "csv"))

    for path, mime_type, subtype in attach_files:
        with open(path, "rb") as f:
            part = MIMEBase(mime_type, subtype)
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment",
                        filename=os.path.basename(path))
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

    parser = argparse.ArgumentParser(description="AI競艇新聞 生成・送信")
    parser.add_argument("--date",      help="対象日 YYYYMMDD")
    parser.add_argument("--input",     default=RANKING_CACHE)
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--html-only", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        log.error("ランキングファイルなし: %s", args.input)
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    date_str  = args.date or data.get("date", datetime.now(JST).strftime("%Y%m%d"))
    html_path = "note.html"
    pdf_path  = "newspaper.pdf"

    generate_html(data, html_path)

    csv_paths = generate_csvs(data)

    if not args.html_only:
        ok = generate_pdf(html_path, pdf_path)
        if not ok:
            pdf_path = None
    else:
        pdf_path = None

    ok = send_note_report(html_path, pdf_path, csv_paths,
                          date_str, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
