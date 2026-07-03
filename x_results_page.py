#!/usr/bin/env python3
"""
x_results_page.py  ── ⑭ AI実績ページ 完全刷新版

今日 / 7日 / 30日 / 累計 の4期間でブランド別成功率を集計し、
⑮AI信頼度・⑯改善ログ・⑰外れ分析・⑱AI総評を統合したHTMLページを
生成・メール送信する。

Usage:
    python x_results_page.py              # 前日分の実績ページを生成
    python x_results_page.py --date 20260628
    python x_results_page.py --dry-run    # 送信せず保存のみ
"""

from __future__ import annotations

import argparse
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

from x_brand_config import (
    BRANDS, brand_icon, brand_name, AI_VERSION, SYSTEM_NAME,
    rank_color, trust_rank_of,
)
from x_verification import (
    load_today_records, load_records_range,
    aggregate, aggregate_by_rank,
    calc_brand_trust, analyze_misses, generate_daily_review,
    trend_vs_previous, _yesterday_jst, _safe_float,
)
from x_improvement_log import format_log_html, format_log_text

log = logging.getLogger("x_results")

JST     = timezone(timedelta(hours=9))
MAIL_TO = "bigkirinuki@gmail.com"


# ════════════════════════════════════════════════════════════
# 期間別データ集計
# ════════════════════════════════════════════════════════════

def collect_all_periods(end_date: str) -> dict:
    """
    ⑭ 今日/7日/30日/累計の4期間すべてを集計してまとめて返す。
    """
    records_1d  = load_today_records(end_date)
    records_7d  = load_records_range(days=7,  end_date=end_date)
    records_30d = load_records_range(days=30, end_date=end_date)
    records_all = load_records_range(days=None, end_date=end_date)

    periods = {}
    for label, records in [
        ("today", records_1d), ("d7", records_7d),
        ("d30", records_30d), ("all", records_all),
    ]:
        agg = aggregate(records)
        rank_data = aggregate_by_rank(records)
        periods[label] = {
            "records": records,
            "agg": agg,
            "rank_data": rank_data,
        }

    return periods


# ════════════════════════════════════════════════════════════
# HTML生成パーツ
# ════════════════════════════════════════════════════════════

def _rank_bar_html(rank_data_cat: dict) -> str:
    """S/A/B/Cランクのバー表示HTML"""
    rows = ""
    for rank in ["S", "A", "B", "C"]:
        d = rank_data_cat.get(rank, {"total": 0, "hit": 0, "rate": 0.0})
        if d["total"] == 0:
            continue
        color = rank_color(rank)
        rows += f"""
<div class="rank-bar-row">
  <span class="rank-badge" style="background:{color}">{rank}</span>
  <span class="rank-detail">{d['total']}件中{d['hit']}件的中</span>
  <div class="rank-bar-bg"><div class="rank-bar-fill" style="width:{d['rate']}%;background:{color}"></div></div>
  <span class="rank-rate">{d['rate']}%</span>
</div>"""
    return rows or '<p class="no-data">対象データなし</p>'


def _period_summary_html(label: str, period_data: dict, cumulative_rank_data: dict,
                         daily_stats: dict = None) -> str:
    """
    期間タブ1つ分のサマリーHTML（今日/7日/30日/累計共通）。
    危険艇: 掲載X件中Y件的中形式。
    万舟: 掲載X件中、万舟Y件 / 高配当Y件 / 中穴Y件 の内訳形式。
    高配当は独立ブランドとして廃止し万舟の内訳として表示。
    """
    agg = period_data["agg"]
    rank_data = period_data["rank_data"]
    records = period_data.get("records", [])
    if daily_stats is None:
        daily_stats = {}

    if agg["total_notified"] == 0:
        return f'<div class="period-panel" data-period="{label}"><p class="no-data">対象データなし</p></div>'

    profit_mark = "✅" if agg["total_profit"] >= 0 else "❌"

    def _brand_block_danger() -> str:
        result = _calc_brand_results(records, daily_stats, "danger")
        cat_data = rank_data.get("danger", {})
        cum = cumulative_rank_data.get("danger", {})
        cum_total = sum(d["total"] for d in cum.values())
        cum_hit   = sum(d["hit"]   for d in cum.values())
        cum_rate  = round(cum_hit / cum_total * 100, 1) if cum_total > 0 else 0.0

        if result["has_data"]:
            period_text = f'掲載{result["listed"]}件中{result["hit"]}件的中（{result["rate"]}%）'
        else:
            total = sum(d["total"] for d in cat_data.values())
            hit   = sum(d["hit"]   for d in cat_data.values())
            rate  = round(hit / total * 100, 1) if total > 0 else 0.0
            period_text = f'{rate}%（{total}件中{hit}件）'

        return f"""
<div class="period-brand">
  <h3>🚨 危険艇</h3>
  <div class="period-vs-cumulative">
    <span class="pvc-period">この期間: {period_text}</span>
    <span class="pvc-cumulative">累計: {cum_rate}%（{cum_total}件中{cum_hit}件）</span>
  </div>
  <details class="rank-detail-toggle">
    <summary>ランク別内訳（S/A/B/C）</summary>
    {_rank_bar_html(cat_data)}
  </details>
</div>"""

    def _brand_block_manshuu() -> str:
        m_result  = _calc_brand_results(records, daily_stats, "manshuu")
        h_result  = _calc_brand_results(records, daily_stats, "hot_high")
        k_result  = _calc_brand_results(records, daily_stats, "korogashi")
        listed    = daily_stats.get("manshuu", {}).get("count", 0)
        cum = cumulative_rank_data.get("manshuu", {})
        cum_total = sum(d["total"] for d in cum.values())
        cum_hit   = sum(d["hit"]   for d in cum.values())
        cum_rate  = round(cum_hit / cum_total * 100, 1) if cum_total > 0 else 0.0

        if m_result["has_data"] or h_result["has_data"]:
            parts = []
            if m_result["has_data"]:
                parts.append(f'上位10件中{m_result["hit"]}件万舟（{m_result["rate"]}%）')
            if h_result["has_data"]:
                parts.append(f'{listed}件中{h_result["hit"]}件高配当（{h_result["rate"]}%）')
            if k_result["has_data"]:
                parts.append(f'{listed}件中{k_result["hit"]}件中穴（{k_result["rate"]}%）')
            period_text = f'掲載{listed}件　' + "　".join(parts) if parts else f'掲載{listed}件'
        else:
            cat_data = rank_data.get("manshuu", {})
            total = sum(d["total"] for d in cat_data.values())
            hit   = sum(d["hit"]   for d in cat_data.values())
            rate  = round(hit / total * 100, 1) if total > 0 else 0.0
            period_text = f'{rate}%（{total}件中{hit}件）'

        cat_data = rank_data.get("manshuu", {})
        return f"""
<div class="period-brand">
  <h3>💰 万舟</h3>
  <div class="period-vs-cumulative">
    <span class="pvc-period">この期間: {period_text}</span>
    <span class="pvc-cumulative">累計: {cum_rate}%（{cum_total}件中{cum_hit}件）</span>
  </div>
  <details class="rank-detail-toggle">
    <summary>ランク別内訳（S/A/B/C）</summary>
    {_rank_bar_html(cat_data)}
  </details>
</div>"""

    blocks = _brand_block_danger() + _brand_block_manshuu()

    return f"""
<div class="period-panel" data-period="{label}">
  <div class="period-kpi-grid">
    <div class="pk-cell"><div class="pk-num">{agg['total_notified']}</div><div class="pk-label">通知数</div></div>
    <div class="pk-cell"><div class="pk-num">{agg['hit_rate']}%</div><div class="pk-label">的中率</div></div>
    <div class="pk-cell"><div class="pk-num">{profit_mark}{agg['total_profit']:+,}</div><div class="pk-label">損益(円)</div></div>
    <div class="pk-cell"><div class="pk-num">{agg['roi']:+.1f}%</div><div class="pk-label">ROI</div></div>
  </div>
  {blocks}
</div>"""


def _trust_badge_html(records_30d: list, daily_stats_range: dict) -> str:
    """
    ⑮ AI信頼度バッジHTML（30日実績・新聞掲載件数ベース）。
    危険艇: 過去30日の「掲載X件中Y件的中」を合算して表示。
    万舟:   過去30日の「掲載X件　上位10件中Y件万舟／高配当Y件／中穴Y件」を合算表示。
    高配当は独立ブランドとして廃止し、万舟の内訳として表示する。
    daily_stats.json が無い日・列が無い旧データは自動的に母数から除外される
    （_calc_brand_results の has_data 判定による）。
    """
    color_map = {"A+": "#ef5350", "A": "#ffa726", "B": "#ffee58", "C": "#42a5f5"}
    rows = ""

    def _trust_of(rate: float) -> str:
        if rate >= 65: return "A+"
        if rate >= 55: return "A"
        if rate >= 45: return "B"
        return "C"

    # 危険艇（過去30日の掲載件数ベース合算）
    danger_result = _calc_brand_results_range(records_30d, daily_stats_range, "danger")
    if danger_result["has_data"]:
        trust_d = _trust_of(danger_result["rate"])
        color   = color_map.get(trust_d, "#999")
        detail  = f'掲載{danger_result["listed"]}件中{danger_result["hit"]}件的中（{danger_result["rate"]}%・30日実績）'
    else:
        trust_d = "C"
        color   = color_map.get(trust_d, "#999")
        detail  = "30日実績データなし"
    rows += f"""
<div class="trust-row">
  <span class="trust-icon">🚨</span>
  <span class="trust-label">危険艇</span>
  <span class="trust-badge" style="background:{color}">{trust_d}</span>
  <span class="trust-detail">{detail}</span>
</div>"""

    # 万舟（過去30日、上位10件の的中検証 + 全掲載件数からの高配当・中穴内訳）
    manshuu_result   = _calc_brand_results_range(records_30d, daily_stats_range, "manshuu")
    hot_result       = _calc_brand_results_range(records_30d, daily_stats_range, "hot_high")
    korogashi_result = _calc_brand_results_range(records_30d, daily_stats_range, "korogashi")
    listed_total = sum(
        v.get("manshuu", {}).get("count", 0) for v in daily_stats_range.values()
    )

    if manshuu_result["has_data"] or hot_result["has_data"]:
        trust_m = _trust_of(manshuu_result["rate"] if manshuu_result["has_data"] else hot_result["rate"])
        color_m = color_map.get(trust_m, "#999")
        top10_text = f'上位10件中{manshuu_result["hit"]}件万舟（{manshuu_result["rate"]}%）' if manshuu_result["has_data"] else ""
        hot_text   = f'高配当{hot_result["hit"]}件' if hot_result["has_data"] else ""
        koro_text  = f'中穴{korogashi_result["hit"]}件' if korogashi_result["has_data"] else ""
        parts = [x for x in [top10_text, hot_text, koro_text] if x]
        detail_m = f'掲載{listed_total}件（30日累計）　' + "　".join(parts) if parts else f'掲載{listed_total}件（30日累計）'
    else:
        trust_m = "C"
        color_m = color_map.get(trust_m, "#999")
        detail_m = "30日実績データなし"

    rows += f"""
<div class="trust-row">
  <span class="trust-icon">💰</span>
  <span class="trust-label">万舟</span>
  <span class="trust-badge" style="background:{color_m}">{trust_m}</span>
  <span class="trust-detail">{detail_m}</span>
</div>"""

    return rows or '<p class="no-data">対象データなし</p>'


def _miss_analysis_html(miss: dict) -> str:
    """⑰ 外れ分析HTML"""
    if not miss["top_misses"]:
        return '<p class="no-data">外れデータなし</p>'
    rows = ""
    for m in miss["top_misses"]:
        rows += f"""
<div class="miss-row">
  <div class="miss-race">{m['venue']}{m['race']}R　予想{m['pred']} → 結果{m['result']}</div>
  <div class="miss-reason">原因: {m['reason']}</div>
</div>"""
    return f'<div class="miss-list">{rows}</div><p class="miss-total">本日の外れ総数: {miss["total_missed"]}件</p>'


# ════════════════════════════════════════════════════════════
# HTML生成
# ════════════════════════════════════════════════════════════

def _load_daily_stats(date_str: str) -> dict:
    """
    daily_stats.json から指定日の掲載件数・レース一覧を返す。
    ファイルがない・日付がない場合は空dictを返す。
    """
    if not os.path.exists("daily_stats.json"):
        return {}
    try:
        with open("daily_stats.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(date_str, {})
    except Exception:
        return {}


def _calc_brand_results(
    records: list[dict],
    daily_stats: dict,
    brand: str,
) -> dict:
    """
    「掲載X件中、Y件が条件達成」を計算して返す。

    brand: "danger" / "manshuu" / "hot_high" / "korogashi"
    daily_stats: _load_daily_stats() の戻り値（掲載レース一覧）

    戻り値:
      {
        "listed":  20,   # 掲載件数（daily_stats から）
        "checked": 10,   # 検証対象件数（万舟は top10 のみ）
        "hit":     4,    # 条件達成件数
        "rate":    20.0, # 達成率（checked 基準）
        "has_data": bool,
      }
    """
    # 掲載レースのキーセット
    if brand == "manshuu":
        # 万舟の的中検証は上位10件のみ
        listed_count = daily_stats.get("manshuu", {}).get("count", 0)
        check_races  = {
            (str(r["venue_num"]), str(r["race"]))
            for r in daily_stats.get("manshuu", {}).get("top10", [])
        }
    elif brand == "danger":
        listed_count = daily_stats.get("danger", {}).get("count", 0)
        check_races  = {
            (str(r["venue_num"]), str(r["race"]))
            for r in daily_stats.get("danger", {}).get("races", [])
        }
    else:
        # hot_high / korogashi: 万舟掲載20件全てが対象
        listed_count = daily_stats.get("manshuu", {}).get("count", 0)
        check_races  = {
            (str(r["venue_num"]), str(r["race"]))
            for r in daily_stats.get("manshuu", {}).get("races", [])
        }

    if not check_races or not records:
        return {"listed": listed_count, "checked": 0, "hit": 0, "rate": 0.0, "has_data": False}

    checked = 0
    hit     = 0
    for r in records:
        key = (str(r.get("venue_num", "")), str(r.get("race", "")))
        if key not in check_races:
            continue
        result_combo = r.get("result_combo", "")
        if not (result_combo and "-" in result_combo):
            continue
        payout = float(r.get("payout") or 0)
        checked += 1
        if brand == "danger":
            if result_combo.split("-")[0].strip() != "1":
                hit += 1
        elif brand == "manshuu":
            if payout > 10000:
                hit += 1
        elif brand == "hot_high":
            if payout > 5000:
                hit += 1
        elif brand == "korogashi":
            if payout > 2700:
                hit += 1

    rate = round(hit / checked * 100, 1) if checked > 0 else 0.0
    return {
        "listed":   listed_count,
        "checked":  checked,
        "hit":      hit,
        "rate":     rate,
        "has_data": checked > 0,
    }


def _load_daily_stats_range(end_date: str, days: int) -> dict:
    """
    daily_stats.json から end_date を含む過去 days 日間の全日付分を返す。
    戻り値: {date_str: daily_stats_entry, ...}
    """
    if not os.path.exists("daily_stats.json"):
        return {}
    try:
        with open("daily_stats.json", "r", encoding="utf-8") as f:
            all_data = json.load(f)
    except Exception:
        return {}

    end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=days - 1)
    start_str = start_dt.strftime("%Y%m%d")

    return {d: v for d, v in all_data.items() if start_str <= d <= end_date}


def _calc_brand_results_range(
    records: list[dict],
    daily_stats_range: dict,
    brand: str,
) -> dict:
    """
    複数日分の daily_stats（_load_daily_stats_range の戻り値）を使って
    「期間内の掲載件数合計 X件中、Y件が条件達成」を計算する（30日実績用）。

    records は対象期間の hit_record.csv レコード（date列を含む）。
    日付ごとに掲載レース一覧が異なるため、日付単位で照合してから合算する。

    戻り値は _calc_brand_results と同じ形式。
    """
    if not daily_stats_range:
        return {"listed": 0, "checked": 0, "hit": 0, "rate": 0.0, "has_data": False}

    # 日付ごとにレコードをグルーピング
    records_by_date: dict[str, list[dict]] = {}
    for r in records:
        d = str(r.get("date", "")).replace("-", "")
        records_by_date.setdefault(d, []).append(r)

    total_listed = 0
    total_checked = 0
    total_hit = 0

    for date_str, daily_stats in daily_stats_range.items():
        day_records = records_by_date.get(date_str, [])
        result = _calc_brand_results(day_records, daily_stats, brand)
        total_listed  += result["listed"]
        total_checked += result["checked"]
        total_hit     += result["hit"]

    rate = round(total_hit / total_checked * 100, 1) if total_checked > 0 else 0.0
    return {
        "listed":   total_listed,
        "checked":  total_checked,
        "hit":      total_hit,
        "rate":     rate,
        "has_data": total_checked > 0,
    }


def _calc_ranking_payouts(records: list[dict], daily_stats: dict) -> list[dict]:
    """
    「📊今日のAIランキング」（AI一致指数トップ10）の各レースについて、
    実際の払戻・的中結果を hit_record.csv と突合して返す。

    daily_stats["ranking"] は _save_daily_stats（x_note_report.py）が
    AI一致指数降順で保存した上位10件（venue_num・race・venue・match_index）。

    戻り値: [{"venue":, "race":, "match_index":, "payout": int|None,
              "result_combo": str, "has_data": bool}, ...]
             match_index 降順（= 元の順位）のまま返す。
    """
    ranking = daily_stats.get("ranking", [])
    if not ranking:
        return []

    # (venue_num, race) → レコード のマップを作る
    record_map: dict[tuple, dict] = {}
    for r in records:
        key = (str(r.get("venue_num", "")), str(r.get("race", "")))
        record_map[key] = r

    result = []
    for item in ranking:
        key = (str(item.get("venue_num", "")), str(item.get("race", "")))
        rec = record_map.get(key)

        payout = None
        result_combo = ""
        has_data = False
        if rec:
            result_combo = rec.get("result_combo", "")
            if result_combo and "-" in result_combo:
                payout = int(_safe_float(rec.get("payout")))
                has_data = True

        result.append({
            "venue":        item.get("venue", ""),
            "race":         item.get("race", ""),
            "match_index":  item.get("match_index", 0),
            "payout":       payout,
            "result_combo": result_combo,
            "has_data":     has_data,
        })

    return result


def _brand_expectation_summary_html(records: list[dict], daily_stats: dict) -> str:
    """
    📊 本日のAI成績サマリー（ご要望の5項目）
    ① 危険艇達成率（掲載20件中、1号艇以外1着の割合）
    ② 万舟達成率（掲載上位10件中、payout>10,000円の割合）
    ③ 高配当達成率（掲載20件中、payout>5,000円の割合）
    ④ 各マークの期待値（役割達成率）一覧
    ⑤ 今日のAIランキング（AI一致指数トップ10）の払戻一覧
    """
    danger_result = _calc_brand_results(records, daily_stats, "danger")
    manshuu_result = _calc_brand_results(records, daily_stats, "manshuu")
    hot_result = _calc_brand_results(records, daily_stats, "hot_high")

    def _rate_text(result: dict, label: str) -> str:
        if result["has_data"]:
            return f'{label}: 掲載{result["listed"]}件中{result["hit"]}件達成（{result["rate"]}%）'
        return f'{label}: データなし'

    summary_rows = "".join([
        f'<div class="expect-row"><span class="expect-icon">🚨</span>'
        f'<span class="expect-label">危険艇</span>'
        f'<span class="expect-detail">{_rate_text(danger_result, "1号艇以外1着")}</span></div>',
        f'<div class="expect-row"><span class="expect-icon">💰</span>'
        f'<span class="expect-label">万舟</span>'
        f'<span class="expect-detail">{_rate_text(manshuu_result, "1万円超")}</span></div>',
        f'<div class="expect-row"><span class="expect-icon">🎯</span>'
        f'<span class="expect-label">転がし候補</span>'
        f'<span class="expect-detail">準備中（データ整備後に対応）</span></div>',
        f'<div class="expect-row"><span class="expect-icon">⚡</span>'
        f'<span class="expect-label">激走モーター</span>'
        f'<span class="expect-detail">対象外（的中判定なし）</span></div>',
        f'<div class="expect-row"><span class="expect-icon">📈</span>'
        f'<span class="expect-label">覚醒モーター</span>'
        f'<span class="expect-detail">対象外（的中判定なし）</span></div>',
    ])

    # ③高配当は独立ブロックとしても表示（万舟20件からの内訳）
    high_payout_block = (
        f'<div class="expect-row"><span class="expect-icon">🔥</span>'
        f'<span class="expect-label">高配当</span>'
        f'<span class="expect-detail">{_rate_text(hot_result, "5千円超")}</span></div>'
    )

    # ⑤ AIランキング払戻一覧
    ranking_list = _calc_ranking_payouts(records, daily_stats)
    if ranking_list:
        ranking_rows = ""
        for i, item in enumerate(ranking_list, 1):
            if item["has_data"]:
                payout_text = f'{item["payout"]:,}円'
                combo_text  = item["result_combo"]
            else:
                payout_text = "結果待ち"
                combo_text  = "-"
            ranking_rows += (
                f'<tr><td>{i}位</td><td>{item["venue"]}{item["race"]}R</td>'
                f'<td>{item["match_index"]}</td><td>{combo_text}</td>'
                f'<td>{payout_text}</td></tr>'
            )
        ranking_html = (
            '<table class="ranking-payout-table">'
            '<thead><tr><th>順位</th><th>レース</th><th>一致指数</th><th>結果</th><th>払戻</th></tr></thead>'
            f'<tbody>{ranking_rows}</tbody></table>'
        )
    else:
        ranking_html = '<p class="no-data">AIランキングデータなし</p>'

    return f"""
<div class="brand-expectation-section">
  <h2>📊 本日のAI成績サマリー</h2>
  <div class="expectation-grid">
    {summary_rows}
    {high_payout_block}
  </div>
  <h3>📊 今日のAIランキング 払戻一覧</h3>
  {ranking_html}
</div>"""



def generate_results_html(date_str: str, output_path: str) -> dict:
    """実績ページHTMLを生成し、サマリーdictを返す"""
    periods = collect_all_periods(date_str)
    miss_analysis = analyze_misses(periods["today"]["records"])
    trust = calc_brand_trust(periods["d30"]["records"])
    daily_stats = _load_daily_stats(date_str)
    daily_stats_range_30d = _load_daily_stats_range(date_str, 30)
    brand_expectation_html = _brand_expectation_summary_html(periods["today"]["records"], daily_stats)
    daily_review = generate_daily_review(
        periods["today"]["agg"], periods["today"]["rank_data"], miss_analysis,
    )

    date_disp = f"{date_str[4:6]}/{date_str[6:8]}"
    now_str   = datetime.now(JST).strftime("%H:%M")

    # 前日比・7日比トレンド
    today_rate = periods["today"]["agg"]["hit_rate"]
    d7_rate    = periods["d7"]["agg"]["hit_rate"]
    d30_rate   = periods["d30"]["agg"]["hit_rate"]
    trend_7d   = trend_vs_previous(today_rate, d7_rate)
    trend_30d  = trend_vs_previous(today_rate, d30_rate)

    period_tabs_html = "".join(
        _period_summary_html(label, periods[label], periods["all"]["rank_data"], daily_stats)
        for label in ["today", "d7", "d30", "all"]
    )

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
/* AI総評 */
.review-box{{background:#0e1e0e;border:1px solid #2a4a2a;border-radius:10px;
  padding:16px;margin-bottom:20px}}
.review-box h2{{color:#81c784;font-size:1.05em;margin-bottom:8px}}
/* 本日のAI成績サマリー */
.brand-expectation-section{{background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:16px;margin-bottom:20px}}
.brand-expectation-section h2{{font-size:1.05em;color:var(--accent);margin-bottom:10px}}
.brand-expectation-section h3{{font-size:.92em;color:var(--gray);margin:14px 0 8px}}
.expectation-grid{{display:flex;flex-direction:column;gap:6px}}
.expect-row{{display:flex;align-items:center;gap:8px;font-size:.85em;
  padding:6px 8px;background:#12121e;border-radius:6px}}
.expect-icon{{font-size:1.1em}}
.expect-label{{min-width:70px;font-weight:bold;color:#cde}}
.expect-detail{{color:var(--gray);flex:1}}
.ranking-payout-table{{width:100%;border-collapse:collapse;font-size:.82em}}
.ranking-payout-table th,.ranking-payout-table td{{padding:5px 8px;
  border-bottom:1px solid #333;text-align:left}}
.ranking-payout-table thead th{{color:var(--gray);font-weight:normal}}
.review-text{{color:#cde;font-size:.92em;line-height:1.7}}
/* タブ切り替え（CSS単体で動くラジオ+ラベル方式。JSがあれば併用して同期させる） */
.tab-radio{{position:absolute;opacity:0;pointer-events:none;width:0;height:0}}
.tab-bar{{display:flex;gap:6px;margin-bottom:14px}}
.tab-btn{{flex:1;background:var(--card);border:1px solid var(--border);border-radius:8px;
  padding:10px 4px;text-align:center;color:var(--gray);font-size:.85em;cursor:pointer;
  user-select:none}}
.tab-radio:checked + .tab-btn{{background:var(--accent);color:#0d0d1a;font-weight:bold;border-color:var(--accent)}}
.tab-btn.active{{background:var(--accent);color:#0d0d1a;font-weight:bold;border-color:var(--accent)}}
/* ラジオ(#tab-xxx)がcheckedのとき、対応するlabel[for=tab-xxx]をハイライトする。
   ラジオとlabelの間に.tab-barが挟まるため、~結合子でlabelを直接指定する
   （+結合子は直後の要素にしか届かないため使えない）。 */
#tab-today:checked ~ .tab-bar label[for="tab-today"],
#tab-d7:checked    ~ .tab-bar label[for="tab-d7"],
#tab-d30:checked   ~ .tab-bar label[for="tab-d30"],
#tab-all:checked   ~ .tab-bar label[for="tab-all"]{{
  background:var(--accent);color:#0d0d1a;font-weight:bold;border-color:var(--accent);
}}
.period-panel{{display:none;background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:16px;margin-bottom:20px}}
.period-panel.active{{display:block}}
/* ラジオが checked のとき、対応する data-period のパネルだけを表示する。
   ラジオ→パネルの間に他要素が挟まるため、隣接兄弟(~)結合子で #period-panels 内の
   該当パネルを individually ターゲットする。 */
#tab-today:checked ~ #period-panels .period-panel[data-period="today"],
#tab-d7:checked    ~ #period-panels .period-panel[data-period="d7"],
#tab-d30:checked   ~ #period-panels .period-panel[data-period="d30"],
#tab-all:checked   ~ #period-panels .period-panel[data-period="all"]{{display:block}}
/* ラジオが1つでもcheckedなら、JS用の.activeクラスによる表示指定より
   ラジオ駆動の表示を優先させたいので、非対象パネルは明示的に隠す */
#tab-today:checked ~ #period-panels .period-panel:not([data-period="today"]),
#tab-d7:checked    ~ #period-panels .period-panel:not([data-period="d7"]),
#tab-d30:checked   ~ #period-panels .period-panel:not([data-period="d30"]),
#tab-all:checked   ~ #period-panels .period-panel:not([data-period="all"]){{display:none}}
.period-kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:16px}}
.pk-cell{{background:#1a1a30;border-radius:6px;padding:10px 4px;text-align:center}}
.pk-num{{font-size:1.15em;font-weight:bold;color:#fff}}
.pk-label{{font-size:.68em;color:var(--gray);margin-top:3px}}
.period-brand{{margin-bottom:14px}}
.period-brand h3{{font-size:.95em;color:var(--accent);margin-bottom:8px}}
.period-vs-cumulative{{display:flex;justify-content:space-between;gap:8px;
  font-size:.76em;color:var(--gray);margin-bottom:6px;flex-wrap:wrap}}
.pvc-period{{color:#cfe;font-weight:bold}}
.pvc-cumulative{{color:var(--gray)}}
.rank-detail-toggle summary{{cursor:pointer;font-size:.78em;color:var(--accent);
  padding:4px 0;user-select:none}}
.rank-detail-toggle summary::-webkit-details-marker{{color:var(--accent)}}
.rank-detail-toggle[open] summary{{margin-bottom:6px}}
/* ランクバー */
.rank-bar-row{{display:flex;align-items:center;gap:8px;margin-bottom:6px}}
.rank-badge{{width:22px;height:22px;border-radius:5px;color:#fff;font-weight:bold;
  display:flex;align-items:center;justify-content:center;font-size:.78em;flex-shrink:0}}
.rank-detail{{font-size:.76em;color:var(--gray);min-width:100px}}
.rank-bar-bg{{flex:1;background:#252540;border-radius:4px;height:12px;overflow:hidden}}
.rank-bar-fill{{height:100%;border-radius:4px}}
.rank-rate{{font-size:.8em;font-weight:bold;min-width:38px;text-align:right}}
/* 信頼度 */
.trust-section{{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:16px;margin-bottom:20px}}
.trust-section h2{{font-size:1.05em;color:var(--accent);margin-bottom:12px}}
.trust-row{{display:flex;align-items:center;gap:10px;margin-bottom:8px}}
.trust-icon{{font-size:1.1em}}
.trust-label{{min-width:48px;font-size:.88em}}
.trust-badge{{width:32px;height:24px;border-radius:5px;color:#fff;font-weight:bold;
  display:flex;align-items:center;justify-content:center;font-size:.82em}}
.trust-detail{{color:var(--gray);font-size:.78em}}
/* 外れ分析 */
.miss-section{{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:16px;margin-bottom:20px}}
.miss-section h2{{font-size:1.05em;color:#ff8a80;margin-bottom:12px}}
.miss-row{{background:#1a1a30;border-radius:6px;padding:10px 12px;margin-bottom:8px}}
.miss-race{{font-size:.88em;color:#fff;margin-bottom:4px}}
.miss-reason{{font-size:.78em;color:var(--gray)}}
.miss-total{{color:var(--gray);font-size:.82em;margin-top:8px}}
/* 改善ログ */
.log-section{{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:16px;margin-bottom:20px}}
.log-section h2{{font-size:1.05em;color:#ce93d8;margin-bottom:12px}}
.log-entry{{background:#1a1a30;border-radius:6px;padding:10px 12px;margin-bottom:8px;
  border-left:3px solid #ab47bc}}
.log-header{{display:flex;justify-content:space-between;margin-bottom:4px}}
.log-version{{color:#ce93d8;font-weight:bold;font-size:.85em}}
.log-date{{color:var(--gray);font-size:.78em}}
.log-content{{color:#ddd;font-size:.88em;margin-bottom:3px}}
.log-reason{{color:var(--gray);font-size:.78em}}
.log-rate{{color:#81c784;font-size:.8em;display:block;margin-top:4px}}
.no-data{{color:var(--gray);font-size:.88em;padding:8px 0}}
.footer{{text-align:center;color:#444;font-size:.78em;padding:20px 0;
  border-top:1px solid var(--border);margin-top:10px}}
.footer-meta{{color:#3a3a3a;font-size:.85em;margin-top:6px}}
.verification-notice{{font-size:.78em;color:var(--gray);background:#12121e;
  border-radius:6px;padding:8px 12px;margin-top:10px;line-height:1.5}}
</style>
</head>
<body>

<div class="header">
  <h1>📊 AI実績ページ</h1>
  <div class="meta">{date_disp}分　生成: {now_str}</div>
  <div class="verification-notice">本検証は朝版AI競艇新聞（{date_disp}発行分）の予想を基準に集計しています。当日の途中経過・速報の反映は行っていません。</div>
</div>

<!-- ⑱ AI総評 -->
<div class="review-box">
  <h2>🤖 AI総評</h2>
  <div class="review-text">{daily_review}</div>
</div>

<!-- 📊 本日のAI成績サマリー（危険艇・万舟・高配当・各マーク達成率・AIランキング払戻） -->
{brand_expectation_html}

<!-- ⑭ 期間タブ（ラジオ+ラベルでCSS単体でも切替可能。JSがあれば併用同期）
     ラジオは #period-panels と同じ階層の兄弟に置き、~結合子でパネルを参照する。
     ラベルの見た目強調は隣接の.tab-radio:checked+.tab-btnではなく、
     JSでの.activeクラス同期、および:checkedの兄弟参照(下記CSS)で行う。 -->
<input type="radio" name="period-tab" id="tab-today" class="tab-radio" checked>
<input type="radio" name="period-tab" id="tab-d7"    class="tab-radio">
<input type="radio" name="period-tab" id="tab-d30"   class="tab-radio">
<input type="radio" name="period-tab" id="tab-all"   class="tab-radio">
<div class="tab-bar">
  <label class="tab-btn active" data-tab="today" for="tab-today">今日</label>
  <label class="tab-btn" data-tab="d7" for="tab-d7">7日</label>
  <label class="tab-btn" data-tab="d30" for="tab-d30">30日</label>
  <label class="tab-btn" data-tab="all" for="tab-all">累計</label>
</div>
<div id="period-panels">
{period_tabs_html}
</div>

<!-- ⑮ AI信頼度 -->
<div class="trust-section">
  <h2>🏅 AI信頼度（30日実績）</h2>
  {_trust_badge_html(periods["d30"]["records"], daily_stats_range_30d)}
</div>

<!-- ⑰ 外れ分析 -->
<div class="miss-section">
  <h2>📉 外れ分析（本日）</h2>
  {_miss_analysis_html(miss_analysis)}
</div>

<!-- ⑯ 改善ログ -->
<div class="log-section">
  <h2>🔧 AI改善ログ</h2>
  {format_log_html(limit=5)}
</div>

<div class="footer">
  AI競艇データメディア | 実績ページ | {SYSTEM_NAME}
  <div class="footer-meta">AI Version: {AI_VERSION}　|　対象日: {date_disp}</div>
</div>

<script>
document.querySelectorAll('.tab-btn').forEach(function(btn) {{
  btn.addEventListener('click', function() {{
    var tab = btn.getAttribute('data-tab');
    document.querySelectorAll('.tab-btn').forEach(function(b) {{ b.classList.remove('active'); }});
    document.querySelectorAll('.period-panel').forEach(function(p) {{ p.classList.remove('active'); }});
    btn.classList.add('active');
    var panel = document.querySelector('.period-panel[data-period="' + tab + '"]');
    if (panel) {{ panel.classList.add('active'); }}
    var radio = document.getElementById('tab-' + tab);
    if (radio) {{ radio.checked = true; }}
  }});
}});
document.querySelector('.period-panel[data-period="today"]').classList.add('active');
</script>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("[実績] HTML保存: %s", output_path)

    return {
        "periods": periods,
        "trust": trust,
        "miss_analysis": miss_analysis,
        "daily_review": daily_review,
        "trend_7d": trend_7d,
        "trend_30d": trend_30d,
    }


# ════════════════════════════════════════════════════════════
# メール送信
# ════════════════════════════════════════════════════════════

def send_results_page(html_path: str, date_str: str, summary: dict,
                      dry_run: bool = False) -> bool:
    date_disp = f"{date_str[4:6]}/{date_str[6:8]}"
    subject   = f"📊 AI実績ページ {date_disp}"

    today_agg = summary["periods"]["today"]["agg"]
    trust     = summary["trust"]

    # X投稿候補テキストを生成
    try:
        from x_post_text import results_post
        danger_trust = trust.get("danger", {})
        x_post_block = results_post(
            date_str    = date_str,
            hit_rate    = today_agg.get("hit_rate", 0.0),
            danger_hit  = danger_trust.get("hit",   0),
            danger_total= danger_trust.get("total", 0),
            profit      = today_agg.get("total_profit", 0),
        )
    except Exception as e:
        log.error("[X投稿] 生成失敗（フォールバックを使用）: %s", e, exc_info=True)
        sep = "━━━━━━━━━━━━━━"
        x_post_block = (
            f"\n{sep}\n📱 X投稿候補\n{sep}\n"
            f"📊AI実績ページを公開しました。\n毎日検証して精度を高めています。\n"
            f"#競艇 #ボートレース #競艇予想\n{sep}"
        )

    body = (
        f"AI実績ページ {date_disp} を添付します。\n"
        f"本検証は朝版AI競艇新聞（{date_disp}発行分）の予想を基準に集計しています。"
        f"当日の途中経過・速報の反映は行っていません。\n\n"
        f"【AI総評】\n{summary['daily_review']}\n\n"
        f"今日の的中率: {today_agg['hit_rate']}%\n"
        f"7日トレンド: {summary['trend_7d']['arrow']} {summary['trend_7d']['text']}\n"
        f"30日トレンド: {summary['trend_30d']['arrow']} {summary['trend_30d']['text']}\n\n"
        f"危険艇信頼度: {trust['danger']['trust']}（{trust['danger']['rate']}%）\n"
        f"万舟信頼度: {trust['manshuu']['trust']}（{trust['manshuu']['rate']}%）\n\n"
        "詳細は添付のHTMLをご確認ください。\n"
    ) + x_post_block

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
