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
import math
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
    BRANDS, BRAND_ICONS, BRAND_NAMES, BRAND_SHORT, BRAND_COLOR,
    brand_icon, brand_name, brand_short, brand_color,
    RANK_THRESHOLDS, rank_of, rank_color, rank_color_of_score, rank_label_with_emoji,
    MATCH_INDEX_POINTS, MATCH_INDEX_RANK_BONUS_S, MATCH_INDEX_RANK_BONUS_A,
    VENUE_CONDITION_WEIGHTS,
    HOT_HIGH_THRESHOLD_FALLBACK, HOT_HIGH_RATIO,
    AI_VERSION, SYSTEM_NAME,
    stars, heat_emoji,
)

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
# ② 今日のAI編集部コメント（新聞冒頭・100〜200文字）
# ════════════════════════════════════════════════════════════

def _generate_editor_note(data: dict, conditions: list) -> str:
    """
    全レース・全開催場を俯瞰し、新聞冒頭に載せる「今日の特徴」
    コメントを100〜200文字程度で自動生成する。
    """
    all_danger  = data.get("all_danger",  data.get("danger_boat1", []))
    all_manshuu = data.get("all_manshuu", data.get("manshuu_alert", []))

    s_danger  = [d for d in all_danger  if d.get("score", 0) >= 80]
    s_manshuu = [u for u in all_manshuu if u.get("score", 0) >= 80]

    lines: list[str] = []

    # 開催場特徴（コンディション上位2件・下位1件）
    if conditions:
        top = conditions[:2]
        for c in top:
            if c.get("manshuu_count", 0) >= 2 and (c.get("manshuu_avg") or 0) >= 60:
                lines.append(f"{c['venue']}は荒れ期待が高い")
            elif c.get("danger_count", 0) >= 2 and (c.get("danger_avg") or 0) >= 60:
                lines.append(f"{c['venue']}は1号艇に不安あり")
            elif c.get("korogashi_avg") is not None and c["korogashi_avg"] >= 80:
                lines.append(f"{c['venue']}は転がし向き")
            else:
                lines.append(f"{c['venue']}は総合的に面白い一日")

        # イン優勢な場（危険艇が少ない場）を1つ
        calm = [c for c in conditions if c.get("danger_count", 0) == 0 and c.get("manshuu_count", 0) <= 1]
        if calm:
            lines.append(f"{calm[0]['venue']}はイン優勢")

    if s_danger:
        lines.append(f"危険艇Sランク{len(s_danger)}件")
    if s_manshuu:
        lines.append(f"万舟Sランク{len(s_manshuu)}件")

    if not lines:
        return "本日は際立った特徴は少なめですが、全レースをAIが分析しています。"

    return "・".join(lines[:5]) + "。"


# ════════════════════════════════════════════════════════════
# 各ブランドセクション冒頭のAIコメント生成
# ════════════════════════════════════════════════════════════

def _section_comment_danger(all_danger: list) -> str:
    if not all_danger:
        return "本日は対象レースがありません。"
    s_count = sum(1 for d in all_danger if rank_of(d.get("score", 0)) == "S")
    a_count = sum(1 for d in all_danger if rank_of(d.get("score", 0)) == "A")
    avg_score = sum(d.get("score", 0) for d in all_danger) / len(all_danger)
    venues  = list(dict.fromkeys(d.get("venue","") for d in all_danger[:5]))
    venue_txt = "・".join(venues[:3]) if venues else ""

    lines = [f"本日は危険艇{len(all_danger)}件（Sランク{s_count}件・Aランク{a_count}件）を抽出。平均危険度は{avg_score:.1f}点です。"]
    if venue_txt:
        lines.append(f"{venue_txt}で1号艇の信頼度低下が目立ち、2〜4号艇に注目です。")
    return " ".join(lines)


def _section_comment_manshuu(all_manshuu: list) -> str:
    if not all_manshuu:
        return "本日は対象レースがありません。"
    s_count = sum(1 for u in all_manshuu if rank_of(u.get("score", 0)) == "S")
    a_count = sum(1 for u in all_manshuu if rank_of(u.get("score", 0)) == "A")
    avg_score = sum(u.get("score", 0) for u in all_manshuu) / len(all_manshuu)
    venues  = list(dict.fromkeys(u.get("venue","") for u in all_manshuu[:5]))
    venue_txt = "・".join(venues[:3]) if venues else ""

    lines = [f"本日は万舟候補{len(all_manshuu)}件（Sランク{s_count}件・Aランク{a_count}件）。平均荒れ指数は{avg_score:.1f}点です。"]
    if venue_txt:
        lines.append(f"特に{venue_txt}で高配当期待値が高くなっています。")
    return " ".join(lines)


def _section_comment_hot_high(manshuu_list: list) -> str:
    cutoff = _hot_high_threshold(manshuu_list)
    high = [u for u in manshuu_list if u.get("score", 0) >= cutoff]
    if not high:
        return "本日は高配当期待レースがありません。"
    venues = list(dict.fromkeys(u.get("venue","") for u in high[:5]))
    venue_txt = "・".join(venues[:3])
    return f"荒れ指数{cutoff:.0f}以上の高配当期待レースが{len(high)}件。{venue_txt}は特に要チェックです。"


def _section_comment_motor(hot_motor: list, label: str) -> str:
    if not hot_motor:
        return "本日はデータ蓄積中です（数日後に表示されます）。"
    venues = list(dict.fromkeys(m.get("venue","") for m in hot_motor[:5]))
    venue_txt = "・".join(venues[:3])
    return f"{label}が{len(hot_motor)}件検出されました。{venue_txt}のモーターに注目です。"


def _section_comment_korogashi(kdata: dict) -> str:
    top10 = kdata.get("top10", [])
    buy = [s for s in top10 if s.get("verdict") == "購入"]
    if not top10:
        return "本日は転がし候補の算出データがありません。"
    if not buy:
        return f"本日は転がし適性70未満が多く、見送り寄りの判定です（候補{len(top10)}件中）。"
    venues = list(dict.fromkeys(s.get("venue","") for s in buy[:3]))
    venue_txt = "・".join(venues)
    return f"購入判定レースが{len(buy)}件あります。{venue_txt}が本日の有力候補です。"


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

# ════════════════════════════════════════════════════════════
# ブランド・ランク・一致指数の定数は x_brand_config.py に集約済み
# （BRAND_ICONS, BRAND_NAMES, MATCH_INDEX_POINTS, VENUE_CONDITION_WEIGHTS 等）
# ════════════════════════════════════════════════════════════

# AI総合注目度（PICKセクション用の加重平均、一致指数とは別軸）の重み
OVERALL_WEIGHTS = {
    "danger":    0.25,
    "manshuu":   0.25,
    "korogashi": 0.20,
    "motor_hot": 0.15,
    "motor_awk": 0.10,
    "hot_high":  0.05,
}

HOT_HIGH_THRESHOLD = HOT_HIGH_THRESHOLD_FALLBACK   # 互換用エイリアス


def _calc_match_index(brands: list[str], raw_scores: dict) -> float:
    """
    ① AI一致指数を算出する（加点方式・100点満点）。
    各ブランドの掲載で配点を加算し、そのブランドのスコアが
    Sランク(80+)なら1.15倍、Aランク(60+)なら1.05倍のボーナスを掛ける。
    激走/覚醒は会場単位データのため、掲載されていれば満額加点する。
    """
    total = 0.0
    for b in brands:
        base = MATCH_INDEX_POINTS.get(b, 0)
        if base == 0:
            continue
        score = raw_scores.get(b)
        if score is not None and score >= 80:
            base *= MATCH_INDEX_RANK_BONUS_S
        elif score is not None and score >= 60:
            base *= MATCH_INDEX_RANK_BONUS_A
        total += base
    return round(min(100, total), 1)


# ════════════════════════════════════════════════════════════
# ② AI開催場ヒートマップ／コンディション指数（開催場別）
# 重みは x_brand_config.VENUE_CONDITION_WEIGHTS を使用
# ════════════════════════════════════════════════════════════

def calc_venue_conditions(data: dict) -> list[dict]:
    """
    開催場ごとに5項目（危険艇・万舟・転がし・激走・覚醒）を個別採点し、
    さらに総合の「AIコンディション指数」を算出する。
    要件②: 開催場ヒートマップの土台データ。

    戻り値: [{
      "venue": "江戸川",
      "score": 92, "stars": "★★★★★",
      # 5項目それぞれの 0-100点スコアと★・信号表示
      "items": {
        "danger":    {"score": 85, "stars": "★★★★☆", "heat": "🟢", "count": 3},
        "manshuu":   {"score": 70, "stars": "★★★★☆", "heat": "🟢", "count": 2},
        "korogashi": {"score": 0,  "stars": "☆☆☆☆☆", "heat": "🔴", "count": 0},
        "motor_hot": {"score": 75, "stars": "★★★★☆", "heat": "🟢", "count": 1},
        "motor_awk": {"score": 0,  "stars": "☆☆☆☆☆", "heat": "🔴", "count": 0},
      },
      # 互換用（旧コード参照のため残す）
      "danger_avg":, "manshuu_avg":, "korogashi_avg":,
      "motor_hot": bool, "motor_awk": bool,
      "danger_count":, "manshuu_count":,
    }, ...]  スコア降順
    """
    all_danger  = data.get("all_danger",  data.get("danger_boat1", []))
    all_manshuu = data.get("all_manshuu", data.get("manshuu_alert", []))
    hot_motor   = data.get("hot_motor", [])
    awake_motor = data.get("awakening_motor", [])
    kdata       = _load_korogashi_cache()
    korogashi_top10 = [s for s in kdata.get("top10", []) if s.get("verdict") in ("購入", "注意")]

    venues: set[str] = set()
    for d in all_danger:  venues.add(d.get("venue",""))
    for u in all_manshuu: venues.add(u.get("venue",""))
    for m in hot_motor:   venues.add(m.get("venue",""))
    for a in awake_motor: venues.add(a.get("venue",""))
    for s in korogashi_top10: venues.add(s.get("venue",""))
    venues.discard("")

    results: list[dict] = []
    W = VENUE_CONDITION_WEIGHTS

    for venue in venues:
        d_list = [d.get("score", 0) for d in all_danger  if d.get("venue") == venue]
        m_list = [u.get("score", 0) for u in all_manshuu if u.get("venue") == venue]
        k_list = [s.get("fitness", 0) for s in korogashi_top10 if s.get("venue") == venue]
        hot_list = [m for m in hot_motor   if m.get("venue") == venue]
        awk_list = [a for a in awake_motor if a.get("venue") == venue]
        has_hot  = len(hot_list) > 0
        has_awk  = len(awk_list) > 0

        d_avg = sum(d_list) / len(d_list) if d_list else 0
        m_avg = sum(m_list) / len(m_list) if m_list else 0
        k_avg = sum(k_list) / len(k_list) if k_list else 0
        # 激走/覚醒はレース単位スコアがないため、件数に応じて0-100点を疑似算出
        hot_s = min(100, 60 + len(hot_list) * 10) if has_hot else 0
        awk_s = min(100, 60 + len(awk_list) * 10) if has_awk else 0

        items = {
            "danger":    {"score": round(d_avg, 1),   "stars": _stars(d_avg),   "heat": heat_emoji(d_avg),   "count": len(d_list)},
            "manshuu":   {"score": round(m_avg, 1),   "stars": _stars(m_avg),   "heat": heat_emoji(m_avg),   "count": len(m_list)},
            "korogashi": {"score": round(k_avg, 1),   "stars": _stars(k_avg),   "heat": heat_emoji(k_avg),   "count": len(k_list)},
            "motor_hot": {"score": round(hot_s, 1),   "stars": _stars(hot_s),   "heat": heat_emoji(hot_s),   "count": len(hot_list)},
            "motor_awk": {"score": round(awk_s, 1),   "stars": _stars(awk_s),   "heat": heat_emoji(awk_s),   "count": len(awk_list)},
        }

        # 掲載されている指標の重みだけで正規化（フェアな比較のため）
        present_weights = []
        if d_list: present_weights.append(("danger", d_avg))
        if m_list: present_weights.append(("manshuu", m_avg))
        if k_list: present_weights.append(("korogashi", k_avg))
        if has_hot: present_weights.append(("motor_hot", hot_s))
        if has_awk: present_weights.append(("motor_awk", awk_s))

        if not present_weights:
            continue

        total_w = sum(W[k] for k, _ in present_weights)
        raw = sum(W[k] * v for k, v in present_weights)
        score = round(min(100, raw / total_w), 1) if total_w > 0 else 0

        results.append({
            "venue":         venue,
            "score":         score,
            "stars":         _stars(score),
            "heat":          heat_emoji(score),
            "items":         items,
            # 互換用フィールド（既存コードが参照しているため残す）
            "danger_avg":    round(d_avg, 1) if d_list else None,
            "manshuu_avg":   round(m_avg, 1) if m_list else None,
            "korogashi_avg": round(k_avg, 1) if k_list else None,
            "motor_hot":     has_hot,
            "motor_awk":     has_awk,
            "danger_count":  len(d_list),
            "manshuu_count": len(m_list),
        })

    results.sort(key=lambda x: -x["score"])
    return results


# ════════════════════════════════════════════════════════════
# ① 今日のダッシュボード（新聞最初のサマリー）
# ════════════════════════════════════════════════════════════

def build_dashboard(data: dict, conditions: list) -> dict:
    """
    要件①: 新聞冒頭に表示する今日の全体サマリーを集計する。
    """
    all_danger  = data.get("all_danger",  data.get("danger_boat1", []))
    all_manshuu = data.get("all_manshuu", data.get("manshuu_alert", []))
    hot_motor   = data.get("hot_motor", [])
    awake_motor = data.get("awakening_motor", [])
    kdata       = _load_korogashi_cache()
    korogashi_buy = [s for s in kdata.get("top10", []) if s.get("verdict") == "購入"]

    sorted_index, brand_counts, race_scores = _build_race_index(data)

    venues_analyzed = set()
    for d in all_danger:  venues_analyzed.add(d.get("venue",""))
    for u in all_manshuu: venues_analyzed.add(u.get("venue",""))
    venues_analyzed.discard("")

    races_analyzed = set()
    for d in all_danger:  races_analyzed.add((d.get("venue",""), d.get("race","")))
    for u in all_manshuu: races_analyzed.add((u.get("venue",""), u.get("race","")))

    s_danger_count  = sum(1 for d in all_danger  if rank_of(d.get("score",0)) == "S")
    s_manshuu_count = sum(1 for u in all_manshuu if rank_of(u.get("score",0)) == "S")
    high_match_count = sum(1 for s in race_scores.values() if s["match_index"] >= 95)

    # 高配当期待件数（動的閾値）
    cutoff = _hot_high_threshold(all_manshuu)
    hot_high_count = sum(1 for u in all_manshuu if u.get("score", 0) >= cutoff)

    best_venue  = conditions[0]["venue"] if conditions else None    # 最も期待
    worst_venue = None   # 最も堅そう（コンディション低い＝荒れにくい）
    rough_venue = None   # 最も荒れそう（万舟スコア最高）

    if conditions:
        worst_venue = min(conditions, key=lambda c: c["score"])["venue"]
        manshuu_sorted = sorted(
            [c for c in conditions if c.get("manshuu_count", 0) > 0],
            key=lambda c: -(c.get("manshuu_avg") or 0),
        )
        rough_venue = manshuu_sorted[0]["venue"] if manshuu_sorted else best_venue

    return {
        "venues_analyzed":    len(venues_analyzed),
        "races_analyzed":     len(races_analyzed),
        "danger_s_count":     s_danger_count,
        "manshuu_s_count":    s_manshuu_count,
        "korogashi_count":    len(korogashi_buy),
        "hot_high_count":     hot_high_count,
        "motor_hot_count":    len(hot_motor),
        "motor_awk_count":    len(awake_motor),
        "high_match_count":   high_match_count,
        "best_venue":         best_venue,
        "rough_venue":        rough_venue,
        "calm_venue":         worst_venue,
    }


def _render_dashboard_section(data: dict, dashboard: dict) -> str:
    """① 今日のダッシュボード セクションHTMLを生成する"""
    d = dashboard
    venue_html = ""
    if d["best_venue"]:
        venue_html += f'<div class="dash-venue-row"><span class="dv-label">⭐ 最も期待</span><span class="dv-venue">{d["best_venue"]}</span></div>'
    if d["rough_venue"]:
        venue_html += f'<div class="dash-venue-row"><span class="dv-label">🌊 最も荒れそう</span><span class="dv-venue">{d["rough_venue"]}</span></div>'
    if d["calm_venue"]:
        venue_html += f'<div class="dash-venue-row"><span class="dv-label">🛡️ 最も堅そう</span><span class="dv-venue">{d["calm_venue"]}</span></div>'

    return f"""
<section id="dashboard">
  <h2>📋 今日のダッシュボード</h2>
  <div class="dash-grid">
    <div class="dash-cell"><div class="dc-num">{d['venues_analyzed']}</div><div class="dc-label">解析開催場数</div></div>
    <div class="dash-cell"><div class="dc-num">{d['races_analyzed']}</div><div class="dc-label">解析レース数</div></div>
    <div class="dash-cell s"><div class="dc-num">{d['danger_s_count']}</div><div class="dc-label">🚨危険艇S</div></div>
    <div class="dash-cell s"><div class="dc-num">{d['manshuu_s_count']}</div><div class="dc-label">💰万舟S</div></div>
    <div class="dash-cell"><div class="dc-num">{d['korogashi_count']}</div><div class="dc-label">🎯転がし候補</div></div>
    <div class="dash-cell"><div class="dc-num">{d['hot_high_count']}</div><div class="dc-label">🔥高配当期待</div></div>
    <div class="dash-cell"><div class="dc-num">{d['motor_hot_count']}</div><div class="dc-label">⚡激走モーター</div></div>
    <div class="dash-cell"><div class="dc-num">{d['motor_awk_count']}</div><div class="dc-label">📈覚醒モーター</div></div>
    <div class="dash-cell hi"><div class="dc-num">{d['high_match_count']}</div><div class="dc-label">一致指数95+</div></div>
  </div>
  <div class="dash-venues">
    {venue_html}
  </div>
</section>"""


def _render_condition_section(data: dict) -> str:
    """
    ② 本日のAI開催場ヒートマップ セクション（開催場別）
    要件②: 危険艇・万舟・転がし・激走・覚醒の5項目を一覧表で色分け表示する。
    """
    conditions = calc_venue_conditions(data)
    if not conditions:
        return ""

    item_keys = ["danger", "manshuu", "korogashi", "motor_hot", "motor_awk"]

    header_cells = "".join(
        f'<th>{brand_icon(k)}{brand_short(k)}</th>' for k in item_keys
    )

    rows = ""
    for c in conditions:
        cells = ""
        for k in item_keys:
            it = c["items"][k]
            if it["count"] == 0:
                cells += '<td class="hm-cell empty">－</td>'
            else:
                cells += (
                    f'<td class="hm-cell" title="{it["score"]}点・{it["count"]}件">'
                    f'<span class="hm-heat">{it["heat"]}</span>'
                    f'<span class="hm-num">{it["score"]:.0f}</span>'
                    f'</td>'
                )
        rows += f"""
<tr class="hm-row">
  <td class="hm-venue">{c['venue']}</td>
  <td class="hm-total"><span class="hm-stars">{c['stars']}</span></td>
  {cells}
</tr>"""

    return f"""
<section id="condition">
  <h2>📈 本日のAI開催場ヒートマップ</h2>
  <div class="section-comment">開催場ごとの今日の面白さを5項目で採点しました。🟢が多い開催場ほど見どころが多い一日です。</div>
  <div class="heatmap-wrap">
    <table class="heatmap-table">
      <thead><tr><th>開催場</th><th>総合</th>{header_cells}</tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </div>
</section>"""


def _hot_high_threshold(manshuu_list: list) -> float:
    """
    万舟リストから高配当期待の動的閾値を算出する。
    上位 HOT_HIGH_RATIO 件のスコアを基準にする（最低3件は確保）。
    データが少ない・分散がない場合は HOT_HIGH_THRESHOLD にフォールバック。
    """
    if not manshuu_list:
        return HOT_HIGH_THRESHOLD
    scores = sorted((u.get("score", 0) for u in manshuu_list), reverse=True)
    n = max(3, math.ceil(len(scores) * HOT_HIGH_RATIO))
    n = min(n, len(scores))
    cutoff = scores[n - 1]
    # 全件が同スコア帯などで閾値が低すぎる場合は固定値と高い方を採用
    return max(cutoff, min(HOT_HIGH_THRESHOLD, scores[0]))


# ════════════════════════════════════════════════════════════
# ⑦ ブランドページ統一テンプレート（掲載条件・注意点）
# ════════════════════════════════════════════════════════════
# 各ブランドの「掲載条件」「注意点」を一元管理する。
# 新しいブランドを追加する場合はここに1エントリ追加するだけでよい。

BRAND_CRITERIA: dict[str, dict] = {
    "danger": {
        "condition": "危険度スコア40点以上の1号艇を抽出（展示・ST・モーター・等級・勝率・相手関係の6指標で算出）",
        "caution":   "展示タイム未確定の早い時間帯は判定の確度が下がる場合があります。最新の展示情報は締切直前にご確認ください。",
    },
    "manshuu": {
        "condition": "荒れ指数40点以上のレースを抽出（イン信頼度・実力接近度・ST分散・モーター格差・展示差・気象の6指標で算出）",
        "caution":   "荒れ指数はあくまで統計的傾向であり、決まり手や進入の急変までは予測できません。",
    },
    "hot_high": {
        "condition": "万舟警報の中でも荒れ指数が上位30%（または80点以上）のレースのみを抽出",
        "caution":   "高配当期待は的中率より払戻額の大きさに焦点を当てた指標です。点数を絞った勝負には不向きな場合があります。",
    },
    "motor_hot": {
        "condition": "直近5走の公式2連率を上回るモーターを検出（モーター履歴データが10走以上蓄積された会場のみ対象）",
        "caution":   "整備状況やコース替わりにより、当日の調子が履歴と異なるケースがあります。",
    },
    "motor_awk": {
        "condition": "直近10走の2連率が、それ以前10走から大きく上昇しているモーターを検出（20走以上の履歴が必要）",
        "caution":   "覚醒は一時的な好調の場合もあり、継続性を保証するものではありません。",
    },
    "korogashi": {
        "condition": "2〜4号艇を中心に、展示・ST・モーター・直近成績・コース勝率・イン信頼度・相手関係・期待値・気象の9指標で転がし適性を算出（適性70未満は見送り）",
        "caution":   "転がし企画は連敗リスクを伴います。無理のない範囲でお楽しみください。",
    },
    "racer": {
        "condition": "本日の危険艇判定上位5名を、勝率・モーター・直近成績の観点でピックアップ",
        "caution":   "選手のコンディションは直前情報でも変動します。",
    },
}


def _render_brand_header(brand_key: str, count_label: str, comment: str) -> str:
    """
    要件⑦: 全ブランド共通のページヘッダー（タイトル・AIコメント・掲載条件・注意点）を生成する。
    各ブランドセクションの先頭で呼び出す。
    """
    icon = brand_icon(brand_key)
    name = brand_name(brand_key)
    criteria = BRAND_CRITERIA.get(brand_key, {})
    condition = criteria.get("condition", "")
    caution   = criteria.get("caution", "")

    return f"""
  <h2>{icon} {name}　{count_label}</h2>
  <div class="section-comment">{comment}</div>
  <div class="brand-meta">
    <div class="bm-row"><span class="bm-label">📋 掲載条件</span><span class="bm-text">{condition}</span></div>
    <div class="bm-row caution"><span class="bm-label">⚠️ 注意点</span><span class="bm-text">{caution}</span></div>
  </div>"""


def _race_anchor(venue: str, race) -> str:
    """会場名・レース番号からHTMLアンカーIDを生成する"""
    import re as _re
    safe_venue = _re.sub(r'[^\w]', '', str(venue))
    return f"race-{safe_venue}-{race}"


def _load_korogashi_cache() -> dict:
    """korogashi_cache.json を安全に読み込む"""
    try:
        if os.path.exists("korogashi_cache.json"):
            with open("korogashi_cache.json", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _build_race_index(data: dict) -> tuple:
    """
    全レースのブランド掲載状況とAIスコアを集計する。
    戻り値: (sorted_index, brand_counts, race_scores)
      sorted_index: [(venue, race, [brand_key, ...]), ...]  会場→レース番号昇順
      brand_counts: {"danger": 18, "manshuu": 10, ...}
      race_scores:  {(venue, race): {
          "overall":     重み付き加重平均スコア（既存・引き続き内部利用）,
          "match_index": AI一致指数（① 加点方式・100点満点）,
          "danger":, "manshuu":, "korogashi":, "hot_high":,
          "motor_hot":, "motor_awk":, "brand_count":,
      }}
    """
    race_brands: dict[tuple, list[str]] = {}
    race_raw: dict[tuple, dict] = {}   # 各カテゴリの生スコアを保持

    def _add(venue, race, brand_key, score=None):
        key = (venue, race)
        race_brands.setdefault(key, [])
        if brand_key not in race_brands[key]:
            race_brands[key].append(brand_key)
        race_raw.setdefault(key, {})
        if score is not None:
            race_raw[key][brand_key] = score

    all_danger  = data.get("all_danger",  data.get("danger_boat1", []))
    all_manshuu = data.get("all_manshuu", data.get("manshuu_alert", []))

    for d in all_danger:
        _add(d.get("venue",""), d.get("race",""), "danger", d.get("score", 0))

    for u in all_manshuu:
        venue, race = u.get("venue",""), u.get("race","")
        score = u.get("score", 0)
        _add(venue, race, "manshuu", score)
        # 高配当期待: 万舟スコアが閾値以上のものを別ブランドとしても掲載
        if score >= HOT_HIGH_THRESHOLD:
            _add(venue, race, "hot_high", score)

    # 転がし候補（korogashi_cache.json があれば統合）
    kdata = _load_korogashi_cache()
    for s in kdata.get("top10", []):
        if s.get("verdict") in ("購入", "注意"):
            _add(s.get("venue",""), s.get("race",""), "korogashi", s.get("fitness", 0))

    # 会場名→開催場コード順、レース番号昇順でソート
    def _sort_key(item):
        (venue, race), brands = item
        try:
            race_num = int(race)
        except (ValueError, TypeError):
            race_num = 0
        return (venue, race_num)

    sorted_items = sorted(race_brands.items(), key=_sort_key)
    sorted_index = [(v, r, brands) for (v, r), brands in sorted_items]

    # ── AIスコアの算出 ──────────────────────────────────────
    race_scores: dict[tuple, dict] = {}
    for key, brands in race_brands.items():
        raw = race_raw.get(key, {})
        W   = OVERALL_WEIGHTS

        # 各指標を0-100に正規化（なければ0扱い、重みは「掲載なし」を考慮）
        danger_s    = raw.get("danger", 0)
        manshuu_s   = raw.get("manshuu", 0)
        korogashi_s = raw.get("korogashi", 0)
        hot_high_s  = raw.get("hot_high", 0)
        # 激走/覚醒は会場単位データのためレース単位スコアはなし→掲載有無のみ反映
        motor_hot_present = any(
            m.get("venue","") == key[0]
            for m in data.get("hot_motor", [])
        )
        motor_awk_present = any(
            m.get("venue","") == key[0]
            for m in data.get("awakening_motor", [])
        )
        motor_hot_s = 70 if motor_hot_present and "motor_hot" in brands else 0
        motor_awk_s = 70 if motor_awk_present and "motor_awk" in brands else 0

        overall = (
            danger_s    * W["danger"]    +
            manshuu_s   * W["manshuu"]   +
            korogashi_s * W["korogashi"] +
            motor_hot_s * W["motor_hot"] +
            motor_awk_s * W["motor_awk"] +
            hot_high_s  * W["hot_high"]
        )
        # 重み合計に対して正規化（満点ブランドが揃っていなくても公平に）
        total_w = sum(W[b] for b in brands if b in W) or 1.0
        # 掲載されているブランドの重みだけで正規化すると相対比較しやすい
        norm_overall = overall / total_w if total_w > 0 else overall

        # ── ① AI一致指数（加点方式）────────────────────────
        match_index = _calc_match_index(brands, raw)

        race_scores[key] = {
            "overall":     round(min(100, norm_overall), 1),
            "match_index": match_index,
            "danger":      danger_s,
            "manshuu":     manshuu_s,
            "korogashi":   korogashi_s,
            "hot_high":    hot_high_s,
            "motor_hot":   motor_hot_s if motor_hot_present else None,
            "motor_awk":   motor_awk_s if motor_awk_present else None,
            "brand_count": len(brands),
        }

    # ブランド別件数
    brand_counts = {
        "danger":    len(all_danger),
        "manshuu":   len(all_manshuu),
        "hot_high":  sum(1 for u in all_manshuu if u.get("score", 0) >= HOT_HIGH_THRESHOLD),
        "motor_hot": len(data.get("hot_motor", [])),
        "motor_awk": len(data.get("awakening_motor", [])),
        "korogashi": sum(1 for s in kdata.get("top10", []) if s.get("verdict") in ("購入", "注意")),
    }

    return sorted_index, brand_counts, race_scores


def _anchor_for(venue, race, brands) -> str:
    """互換用: 優先順位 danger > manshuu でジャンプ先アンカーを1つ返す"""
    base = _race_anchor(venue, race)
    if "danger" in brands:
        return base
    if "manshuu" in brands or "hot_high" in brands:
        return base + "-manshuu"
    return base


def _anchor_for_brand(venue, race, brand: str) -> str:
    """
    指定ブランド単体のジャンプ先アンカーを返す。
    danger      → レース詳細カード（危険艇セクション）
    manshuu     → レース詳細カード（万舟セクション）
    hot_high    → 高配当期待セクション内の該当リンク
                  （実体は万舟カードと同じだが、専用アンカーを振って
                   高配当期待セクションへ直接ジャンプできるようにする）
    motor_hot   → 激走モーターセクションの先頭（レース単位データがないため）
    motor_awk   → 覚醒モーターセクションの先頭
    korogashi   → 転がし候補セクションの先頭
    """
    base = _race_anchor(venue, race)
    if brand == "danger":
        return base
    if brand == "manshuu":
        return base + "-manshuu"
    if brand == "hot_high":
        return base + "-manshuu"   # 高配当期待は万舟詳細カードと同じ実体
    if brand == "motor_hot":
        return "motor"
    if brand == "motor_awk":
        return "awake"
    if brand == "korogashi":
        return "korogashi"
    return base


def _stars(score: float) -> str:
    """0-100点を★5段階表示に変換（x_brand_config.stars のエイリアス）"""
    return stars(score, max_score=100)


def _brand_badge_html(brands: list[str]) -> str:
    """レース詳細カードのタイトル右上に表示するブランドバッジ"""
    icons = "".join(
        f'<span class="brand-badge" title="{BRAND_NAMES.get(b,b)}">{BRAND_ICONS.get(b,"")}</span>'
        for b in brands
    )
    return f'<span class="brand-badges">{icons}</span>' if icons else ""


def _render_index_section(data: dict) -> str:
    """📖 本日のレースインデックス セクションのHTMLを生成する"""
    sorted_index, brand_counts, race_scores = _build_race_index(data)

    if not sorted_index:
        return ""

    rows_html = ""
    for venue, race, brands in sorted_index:
        # 各ブランドアイコンを個別にクリック可能にする
        # （危険艇＋万舟の両方が付いているレースは、どちらのアイコンを押しても
        #   対応するセクションへ別々にジャンプできる）
        icon_links = "".join(
            f'<a href="#{_anchor_for_brand(venue, race, b)}" '
            f'class="idx-icon-link" title="{BRAND_NAMES.get(b,b)}">'
            f'{BRAND_ICONS.get(b,"")}</a>'
            for b in brands
        )
        rows_html += f"""
<div class="idx-row">
  <span class="idx-race">{venue}{race}R</span>
  <span class="idx-icons">{icon_links}</span>
</div>"""

    count_rows = ""
    for key in ["danger", "manshuu", "hot_high", "motor_hot", "motor_awk", "korogashi"]:
        cnt = brand_counts.get(key, 0)
        if cnt == 0:
            continue
        icon = BRAND_ICONS.get(key, "")
        name = BRAND_NAMES.get(key, key)
        count_rows += f'<div class="idx-count-row"><span>{icon} {name}</span><span class="idx-count-num">{cnt}件</span></div>'

    return f"""
<section id="race-index">
  <h2>📖 本日のレースINDEX</h2>
  <div class="index-counts top">
    {count_rows}
  </div>
  <div class="index-grid">
    {rows_html}
  </div>
</section>"""


def _race_detail(venue: str, race, data: dict) -> Optional[dict]:
    """指定レースの詳細データ（危険艇/万舟/転がし）を1つにまとめて返す"""
    all_danger  = data.get("all_danger",  data.get("danger_boat1", []))
    all_manshuu = data.get("all_manshuu", data.get("manshuu_alert", []))
    kdata       = _load_korogashi_cache()

    d = next((x for x in all_danger  if x.get("venue")==venue and str(x.get("race"))==str(race)), None)
    u = next((x for x in all_manshuu if x.get("venue")==venue and str(x.get("race"))==str(race)), None)
    k = next((x for x in kdata.get("top10", [])
              if x.get("venue")==venue and str(x.get("race"))==str(race)), None)
    if not (d or u or k):
        return None
    return {"danger": d, "manshuu": u, "korogashi": k}


def _render_pickup_section(data: dict) -> str:
    """⑤ 今日のAI編集部PICK（AI一致指数1位のレース）"""
    sorted_index, brand_counts, race_scores = _build_race_index(data)
    if not race_scores:
        return ""

    best_key = max(race_scores, key=lambda k: race_scores[k]["match_index"])
    venue, race = best_key
    s = race_scores[best_key]
    brands = next((b for v, r, b in sorted_index if (v, r) == best_key), [])

    detail = _race_detail(venue, race, data) or {}
    d, u, k = detail.get("danger"), detail.get("manshuu"), detail.get("korogashi")

    badges = _brand_badge_html(brands)
    stars  = _stars(s["match_index"])

    # 注目ポイント（3〜5行）
    points = []
    if d:
        points.append(f"🚨 危険艇: {d.get('reason','')}")
    if u:
        reasons = u.get("key_reason","").split(" / ")
        clean = [r.replace('🔥 ','').replace('🔥','').strip() for r in reasons if r.strip()]
        if clean:
            points.append(f"💰 万舟: {clean[0]}")
    if k:
        points.append(f"🎯 転がし: 適性{k.get('fitness','-')}点・{k.get('lane','-')}号艇 {k.get('racer_name','')}")
    if s.get("motor_hot"):
        points.append("⚡ 激走モーター対象会場")
    if s.get("motor_awk"):
        points.append("📈 覚醒モーター対象会場")

    points_html = "".join(f"<li>{p}</li>" for p in points[:5])

    # 一致ブランド数による総合コメント生成（200文字程度）
    brand_names_jp = {
        "danger": "危険艇", "manshuu": "万舟", "korogashi": "転がし",
        "motor_hot": "激走モーター", "motor_awk": "覚醒モーター", "hot_high": "高配当期待",
    }
    matched_names = [brand_names_jp.get(b, b) for b in brands if b in brand_names_jp]
    match_text = "・".join(matched_names)

    condition_parts = []
    if d and d.get("score", 0) >= 80: condition_parts.append("危険艇Sランク")
    if u and u.get("score", 0) >= 80: condition_parts.append("万舟Sランク")
    if k and k.get("fitness", 0) >= 90: condition_parts.append("転がし適性90以上")
    if s.get("motor_hot"): condition_parts.append("激走モーター対象")
    if s.get("motor_awk"): condition_parts.append("覚醒モーター対象")
    cond_text = "・".join(condition_parts) if condition_parts else "複数指標で高評価"

    ai_comment = (
        f"{venue}{race}Rは本日唯一、{match_text}（{len(brands)}指標）が一致したレースです。"
        f"{cond_text}という条件が重なっており、AI一致指数{s['match_index']}点・"
        f"期待値スコア{s['overall']}点はいずれも本日最高クラス。"
        f"複数のAIが同じ方向を示している点で、本日最も注目すべき1レースです。"
    )

    anchor = _anchor_for(venue, race, brands)

    return f"""
<section id="pickup">
  <h2>⭐ 今日のAI編集部PICK</h2>
  <div class="pickup-card">
    <a href="#{anchor}" class="pickup-race">{venue}{race}R</a>
    <div class="pickup-score">
      <span class="pickup-num">AI一致指数 {s['match_index']}</span>
      <span class="pickup-stars">{stars}</span>
    </div>
    <div class="pickup-score sub">
      <span class="pickup-num-sub">期待値スコア {s['overall']}</span>
    </div>
    <div class="pickup-badges">{badges}</div>
    <div class="pickup-comment">
      <strong>【AIコメント】</strong>
      <p>{ai_comment}</p>
    </div>
    <div class="pickup-points">
      <strong>【注目ポイント】</strong>
      <ul>{points_html}</ul>
    </div>
  </div>
</section>"""


def _render_ranking_section(data: dict) -> str:
    """④ 今日のAIランキング（AI一致指数ベース・新聞最初に掲載）"""
    sorted_index, brand_counts, race_scores = _build_race_index(data)
    if not race_scores:
        return ""

    ranked = sorted(race_scores.items(), key=lambda kv: -kv[1]["match_index"])[:10]

    rows = ""
    for i, (key, s) in enumerate(ranked, 1):
        venue, race = key
        brands = next((b for v, r, b in sorted_index if (v, r) == key), [])
        anchor = _anchor_for(venue, race, brands)
        icons  = "".join(BRAND_ICONS.get(b, "") for b in brands)
        medal  = {1: "🥇", 2: "🥈", 3: "🥉"}.get(i, f"{i}")
        rows += f"""
<a href="#{anchor}" class="rank-row">
  <span class="rank-medal">{medal}</span>
  <span class="rank-race">{venue}{race}R</span>
  <span class="rank-icons">{icons}</span>
  <span class="rank-score">一致指数{s['match_index']}</span>
</a>"""

    return f"""
<section id="ai-ranking">
  <h2>📊 今日のAIランキング</h2>
  <div class="section-comment">各AIブランドの掲載状況からAI一致指数を算出し、ランキング化しました。複数のAIが同じレースを推すほど上位に来ます。</div>
  <div class="rank-list">
    {rows}
  </div>
</section>"""


def _render_top5_section(data: dict) -> str:
    """🏆 本日の注目レースTOP5 セクション（AI一致指数ベース）"""
    sorted_index, brand_counts, race_scores = _build_race_index(data)
    if not race_scores:
        return ""

    ranked = sorted(race_scores.items(), key=lambda kv: -kv[1]["match_index"])[:5]

    rows = ""
    for i, (key, s) in enumerate(ranked, 1):
        venue, race = key
        brands = next((b for v, r, b in sorted_index if (v, r) == key), [])
        anchor = _anchor_for(venue, race, brands)
        icons  = " ".join(BRAND_ICONS.get(b, "") for b in brands)
        rows += f"""
<a href="#{anchor}" class="top5-row">
  <span class="top5-rank">{i}</span>
  <span class="top5-race">{venue}{race}R</span>
  <span class="top5-icons">{icons}</span>
  <span class="top5-score">一致指数{s['match_index']}</span>
</a>"""

    return f"""
<section id="top5">
  <h2>🏆 本日の注目レースTOP5</h2>
  <div class="top5-list">
    {rows}
  </div>
</section>"""


def generate_html(data: dict, output_path: str) -> None:
    date_str  = data.get("date", "")
    date_disp = f"{date_str[4:6]}/{date_str[6:8]}" if len(date_str) >= 8 else ""
    now_str   = datetime.now(JST).strftime("%H:%M")

    all_danger  = data.get("all_danger",  data.get("danger_boat1", []))
    all_manshuu = data.get("all_manshuu", data.get("manshuu_alert", []))
    hot_motor   = data.get("hot_motor", [])
    awake_motor = data.get("awakening_motor", [])
    editorial   = _generate_editorial(data)

    # 新聞本文に「掲載」するレース数は見やすさのため上限を設ける。
    # INDEX/イチオシ/TOP5/CSVは all_danger / all_manshuu（全件）を使い続ける。
    DANGER_DISPLAY_LIMIT  = 20
    MANSHUU_DISPLAY_LIMIT = 10
    danger_display  = sorted(all_danger,  key=lambda x: -x.get("score", 0))[:DANGER_DISPLAY_LIMIT]
    manshuu_display = sorted(all_manshuu, key=lambda x: -x.get("score", 0))[:MANSHUU_DISPLAY_LIMIT]

    # 新セクション用データ（INDEX/イチオシ/TOP5で共有・全件ベース）
    race_index_data    = _build_race_index(data)   # (sorted_index, brand_counts, race_scores)
    venue_conditions    = calc_venue_conditions(data)
    dashboard_data      = build_dashboard(data, venue_conditions)
    dashboard_section   = _render_dashboard_section(data, dashboard_data)
    editor_note         = _generate_editor_note(data, venue_conditions)
    condition_section   = _render_condition_section(data)
    ranking_section      = _render_ranking_section(data)
    pickup_section  = _render_pickup_section(data)
    top5_section    = _render_top5_section(data)
    index_section   = _render_index_section(data)

    # 転がし候補データ・高配当期待（動的閾値、新聞表示分のmanshuu_displayベース）
    korogashi_data    = _load_korogashi_cache()
    hot_high_cutoff   = _hot_high_threshold(manshuu_display)
    high_payout       = [u for u in manshuu_display if u.get("score", 0) >= hot_high_cutoff]

    s_d = len([d for d in all_danger  if rank_of(d.get("score",0)) == "S"])
    a_d = len([d for d in all_danger  if rank_of(d.get("score",0)) == "A"])
    b_d = len([d for d in all_danger  if rank_of(d.get("score",0)) == "B"])
    c_d = len([d for d in all_danger  if rank_of(d.get("score",0)) == "C"])
    s_m = len([u for u in all_manshuu if rank_of(u.get("score",0)) == "S"])

    def rank_label(score):
        return rank_label_with_emoji(score) + "ランク"

    def danger_rows():
        rows = ""
        for d in danger_display:
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
            rank_cls = rank_of(score).lower()
            _anchor = _race_anchor(d.get('venue',''), d.get('race',''))
            _brands = next((b for v, r, b in race_index_data[0]
                           if v == d.get('venue','') and str(r) == str(d.get('race',''))), ["danger"])
            _badges = _brand_badge_html(_brands)
            rows += f"""
<div class="race-card {rank_cls}" id="{_anchor}">
  <div class="rc-header">
    <span class="badge {rank_cls}">{rank_label(score)}</span>
    <strong class="rc-name">{d.get('venue','')}{d.get('race','')}R</strong>
    <span class="rc-racer">{d.get('racer','?')} {d.get('racer_class','')}</span>
    {_badges}
  </div>
  <div class="rc-reason">{d.get('reason','')}</div>
  <div class="rc-comment">💬 {comment}</div>
  <div class="breakdown">{bars}</div>
</div>"""
        return rows

    def manshuu_rows():
        rows = ""
        for u in manshuu_display:
            score = u.get("score", 0)
            rank_cls = rank_of(score).lower()
            reasons = u.get("key_reason","").split(" / ")
            reason_html = "".join(f"<div class='mr'>{r}</div>" for r in reasons if r.strip())
            _anchor_m = _race_anchor(u.get('venue',''), u.get('race','')) + "-manshuu"
            _brands = next((b for v, r, b in race_index_data[0]
                           if v == u.get('venue','') and str(r) == str(u.get('race',''))), ["manshuu"])
            _badges = _brand_badge_html(_brands)
            rows += f"""
<div class="race-card {rank_cls}" id="{_anchor_m}">
  <div class="rc-header">
    <span class="badge {rank_cls}">{rank_label(score)}</span>
    <strong class="rc-name">{u.get('venue','')}{u.get('race','')}R</strong>
    <span class="rc-racer">注目: {u.get('key_racer','')}</span>
    {_badges}
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
  --s:#ef5350;--a:#ffa726;--b:#ffee58;--c:#42a5f5;--hot:#ff7043;--awake:#00bcd4;
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
/* レースインデックス */
#race-index{{background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:18px;margin-bottom:20px}}
#race-index h2{{border-bottom:none;padding-top:0}}
.index-grid{{display:flex;flex-direction:column;gap:2px;margin:12px 0 16px}}
.idx-row{{display:flex;justify-content:space-between;align-items:center;
  padding:9px 12px;border-radius:6px;background:#1a1a30}}
.idx-race{{font-weight:bold;font-size:.92em}}
.idx-icons{{display:inline-flex;gap:6px}}
.idx-icon-link{{display:inline-block;font-size:1.1em;text-decoration:none;
  padding:2px 4px;border-radius:5px;transition:background .15s,transform .1s}}
.idx-icon-link:hover{{background:#2a2a4a;transform:scale(1.15)}}
.idx-icon-link:active{{background:#33335a}}
.index-counts{{border-top:1px solid var(--border);padding-top:12px;
  display:flex;flex-direction:column;gap:6px}}
.idx-count-row{{display:flex;justify-content:space-between;font-size:.88em;color:#bbb}}
.idx-count-num{{color:var(--accent);font-weight:bold}}
.index-counts.top{{margin-bottom:16px;padding-top:0;border-top:none;
  background:#1a1a30;border-radius:8px;padding:12px}}
/* AIイチオシ */
#pickup{{margin-bottom:20px}}
#pickup h2{{border-bottom:none;padding-top:0}}
.pickup-card{{background:linear-gradient(135deg,#1a1a30,#1e1e3a);
  border:2px solid #ffd54f;border-radius:14px;padding:20px;
  box-shadow:0 0 20px rgba(255,213,79,.15)}}
.pickup-race{{display:block;font-size:1.6em;font-weight:bold;color:#fff;
  text-decoration:none;margin-bottom:8px}}
.pickup-race:hover{{color:#ffd54f}}
.pickup-score{{display:flex;align-items:center;gap:12px;margin-bottom:10px}}
.pickup-score.sub{{margin-bottom:14px}}
.pickup-num{{font-size:1.1em;color:#ffd54f;font-weight:bold}}
.pickup-num-sub{{font-size:.92em;color:#90caf9}}
.pickup-stars{{font-size:1.1em;color:#ffd54f}}
.pickup-badges{{margin-bottom:14px}}
.pickup-comment{{background:#14142a;border-radius:8px;padding:12px;margin-bottom:12px}}
.pickup-comment strong{{color:#ffd54f;font-size:.9em}}
.pickup-comment p{{color:#ccc;font-size:.92em;margin-top:6px}}
.pickup-points strong{{color:#81c784;font-size:.9em}}
.pickup-points ul{{list-style:none;margin-top:8px}}
.pickup-points li{{color:#bbb;font-size:.88em;padding:4px 0 4px 4px;
  border-left:2px solid #2a4a2a;padding-left:10px;margin-bottom:4px}}
/* 編集部コメント */
.editorial{{background:#0e1e0e;border:1px solid #2a4a2a;border-radius:10px;
  padding:18px;margin-bottom:20px}}
.editorial h2{{color:#81c784;border-bottom:none;padding-top:0;margin-bottom:10px}}
.editor-note{{color:#cde;font-size:.95em;line-height:1.7}}
/* ① 今日のダッシュボード */
#dashboard{{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:18px;margin-bottom:20px}}
#dashboard h2{{border-bottom:none;padding-top:0;margin-bottom:14px}}
.dash-grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}}
@media(min-width:480px){{.dash-grid{{grid-template-columns:repeat(3,1fr)}}}}
.dash-cell{{background:#1a1a30;border-radius:8px;padding:12px 6px;text-align:center}}
.dash-cell.s{{border:1px solid var(--s)}}
.dash-cell.hi{{border:1px solid #ffd54f}}
.dc-num{{font-size:1.5em;font-weight:bold;color:#fff}}
.dash-cell.s .dc-num{{color:var(--s)}}
.dash-cell.hi .dc-num{{color:#ffd54f}}
.dc-label{{font-size:.72em;color:var(--gray);margin-top:4px}}
.dash-venues{{display:flex;flex-direction:column;gap:6px;border-top:1px solid var(--border);
  padding-top:12px}}
.dash-venue-row{{display:flex;justify-content:space-between;align-items:center;
  background:#1a1a30;border-radius:6px;padding:9px 12px}}
.dv-label{{color:var(--gray);font-size:.85em}}
.dv-venue{{font-weight:bold;color:var(--accent)}}
/* ② AI開催場ヒートマップ */
#condition{{margin-bottom:20px}}
#condition h2{{border-bottom:none;padding-top:0}}
.heatmap-wrap{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
.heatmap-table{{width:100%;border-collapse:collapse;font-size:.82em;min-width:480px}}
.heatmap-table th{{background:#1a1a30;color:var(--gray);padding:8px 6px;
  text-align:center;font-weight:normal;white-space:nowrap}}
.heatmap-table th:first-child{{text-align:left;padding-left:10px}}
.hm-row td{{padding:8px 6px;border-bottom:1px solid var(--border);text-align:center}}
.hm-venue{{font-weight:bold;text-align:left !important;padding-left:10px !important;
  white-space:nowrap}}
.hm-total{{min-width:70px}}
.hm-stars{{color:#ffd54f;font-size:.95em}}
.hm-cell{{white-space:nowrap}}
.hm-cell.empty{{color:#3a3a3a}}
.hm-heat{{margin-right:3px}}
.hm-num{{color:var(--gray);font-size:.88em}}
/* AIランキング */
#ai-ranking{{margin-bottom:20px}}
#ai-ranking h2{{border-bottom:none;padding-top:0}}
.rank-list{{display:flex;flex-direction:column;gap:6px}}
.rank-row{{display:flex;align-items:center;gap:10px;background:var(--card);
  border-radius:8px;padding:12px;text-decoration:none;color:var(--text);
  border-left:3px solid #ffd54f}}
.rank-row:hover{{background:#1e1e3a}}
.rank-medal{{font-size:1.1em;font-weight:bold;width:28px;text-align:center;color:#ffd54f}}
.rank-race{{font-weight:bold;flex:1}}
.rank-icons{{font-size:.95em;letter-spacing:1px}}
.rank-score{{font-size:.8em;color:var(--gray)}}
/* TOP5 */
#top5{{margin-bottom:20px}}
#top5 h2{{border-bottom:none;padding-top:0}}
.top5-list{{display:flex;flex-direction:column;gap:6px}}
.top5-row{{display:flex;align-items:center;gap:10px;background:var(--card);
  border-radius:8px;padding:12px;text-decoration:none;color:var(--text);
  border-left:3px solid var(--accent)}}
.top5-row:hover{{background:#1e1e3a}}
.top5-rank{{font-size:1.3em;font-weight:bold;color:var(--accent);width:24px}}
.top5-race{{font-weight:bold;flex:1}}
.top5-icons{{font-size:.95em;letter-spacing:1px}}
.top5-score{{font-size:.82em;color:var(--gray)}}
/* ブランドバッジ */
.brand-badges{{margin-left:auto;display:inline-flex;gap:2px}}
.brand-badge{{font-size:.95em}}
/* セクションコメント */
.section-comment{{background:#14142a;border-left:3px solid var(--accent);
  border-radius:6px;padding:10px 14px;margin-bottom:14px;
  color:#bbb;font-size:.87em;line-height:1.6}}
/* ⑦ ブランド共通: 掲載条件・注意点 */
.brand-meta{{display:flex;flex-direction:column;gap:6px;margin-bottom:16px}}
.bm-row{{display:flex;gap:8px;background:#14142a;border-radius:6px;
  padding:8px 12px;font-size:.8em;line-height:1.5}}
.bm-row.caution{{background:#1e1810}}
.bm-label{{color:var(--gray);white-space:nowrap;flex-shrink:0}}
.bm-row.caution .bm-label{{color:#ffb74d}}
.bm-text{{color:#999}}
/* 高配当期待・転がしリンク */
.hp-link,.kr-link{{display:block;background:var(--card);border-radius:6px;
  padding:10px 14px;margin-bottom:6px;color:var(--text);text-decoration:none;
  font-size:.9em;border-left:3px solid var(--hot)}}
.kr-link{{border-left-color:#26a69a}}
.hp-link:hover,.kr-link:hover{{background:#1e1e3a}}
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
.race-card.c{{border-left-color:var(--c)}}
.rc-header{{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:6px}}
.badge{{padding:2px 8px;border-radius:4px;font-size:.78em;font-weight:bold}}
.badge.s{{background:var(--s);color:#fff}}
.badge.a{{background:var(--a);color:#000}}
.badge.b{{background:var(--b);color:#000}}
.badge.c{{background:var(--c);color:#fff}}
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
.footer{{text-align:center;color:#444;font-size:.78em;padding:20px 0;
  border-top:1px solid var(--border);margin-top:10px}}
.footer-line{{color:#666;margin-bottom:6px}}
.footer-meta{{color:#444;font-size:.92em;margin-bottom:6px}}
.footer-page{{color:#3a3a3a;font-size:.85em;margin-bottom:4px}}
.footer-note{{color:#3a3a3a;font-size:.85em}}
/* レース検索（⑪） */
.search-box{{margin-bottom:16px}}
.search-box input{{width:100%;padding:12px 14px;border-radius:8px;
  border:1px solid var(--border);background:var(--card);color:var(--text);
  font-size:.95em}}
.search-box input::placeholder{{color:var(--gray)}}
.search-box input:focus{{outline:none;border-color:var(--accent)}}
.search-count{{color:var(--accent);font-size:.82em;margin-top:6px;padding-left:4px}}
/* 目次（⑪） */
.toc{{background:var(--card);border:1px solid var(--border);border-radius:10px;
  padding:16px 18px;margin-bottom:20px}}
.toc h2{{font-size:1.05em;color:var(--accent);margin-bottom:10px}}
.toc ul{{list-style:none;display:grid;grid-template-columns:repeat(2,1fr);gap:6px}}
.toc li a{{color:#bbb;text-decoration:none;font-size:.85em;display:block;
  padding:6px 8px;border-radius:6px;transition:background .15s}}
.toc li a:hover{{background:#1e1e3a;color:var(--accent)}}
/* トップへ戻る（⑪） */
#back-to-top{{position:fixed;bottom:24px;right:18px;width:44px;height:44px;
  border-radius:50%;background:var(--accent);color:#0d0d1a;display:none;
  align-items:center;justify-content:center;text-decoration:none;
  font-size:1.1em;font-weight:bold;box-shadow:0 2px 10px rgba(0,0,0,.4);z-index:100}}
</style>
</head>
<body id="top">

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

<!-- ⑪ レース検索 -->
<div class="search-box">
  <input type="text" id="race-search" placeholder="🔍 会場名・レース番号で検索（例: 三国10）"
         oninput="filterRaces(this.value)">
  <div id="search-result-count" class="search-count"></div>
</div>

<!-- ⑪ 目次 -->
<nav class="toc">
  <h2>📑 目次</h2>
  <ul>
    <li><a href="#condition">📈 AIコンディション指数</a></li>
    <li><a href="#ai-ranking">📊 今日のAIランキング</a></li>
    <li><a href="#pickup">⭐ 今日のAI編集部PICK</a></li>
    <li><a href="#top5">🏆 注目レースTOP5</a></li>
    <li><a href="#race-index">📖 レースINDEX</a></li>
    <li><a href="#dashboard">📋 今日のダッシュボード</a></li>
    <li><a href="#condition">📈 開催場ヒートマップ</a></li>
    <li><a href="#danger">🚨 AI危険艇速報</a></li>
    <li><a href="#manshuu">💰 AI万舟警報</a></li>
    <li><a href="#hot-high">🔥 AI高配当期待</a></li>
    <li><a href="#motor">⚡ AI激走モーター</a></li>
    <li><a href="#awake">📈 AI覚醒モーター</a></li>
    <li><a href="#korogashi">🎯 AI転がし候補</a></li>
    <li><a href="#racer">👤 今日の注目レーサー</a></li>
    <li><a href="#yesterday">📊 昨日の検証</a></li>
  </ul>
</nav>

<!-- ① 今日のダッシュボード -->
{dashboard_section}

<!-- ② 今日のAI編集部コメント -->
<div class="editorial">
  <h2>🤖 AI編集部コメント</h2>
  <div class="editor-note">{editor_note}</div>
</div>

<!-- ③ AI開催場ヒートマップ -->
{condition_section}

<!-- ④ 今日のAIランキング -->
{ranking_section}

<!-- ⑤ 今日のAI編集部PICK -->
{pickup_section}

<!-- ⑥ 本日の注目レースTOP5 -->
{top5_section}

<!-- ⑦ 本日のレースINDEX -->
{index_section}

<!-- ⑧ 危険な1号艇 -->
<section id="danger">
  {_render_brand_header("danger", f"掲載{len(danger_display)}件（全{len(all_danger)}件中）", _section_comment_danger(all_danger))}
  {danger_rows()}
</section>

<!-- ⑦ 万舟警報 -->
<section id="manshuu">
  {_render_brand_header("manshuu", f"掲載{len(manshuu_display)}件（全{len(all_manshuu)}件中）", _section_comment_manshuu(all_manshuu))}
  {manshuu_rows()}
</section>

<!-- ⑧ 高配当期待 -->
<section id="hot-high">
  {_render_brand_header("hot_high", f"{len(high_payout)}レース", _section_comment_hot_high(manshuu_display))}
  {''.join(f'<a href="#{_race_anchor(u.get("venue",""), u.get("race",""))}-manshuu" class="hp-link">{u.get("venue","")}{u.get("race","")}R　荒れ指数{u.get("score",0)}</a>' for u in high_payout) or '<p class="no-data">本日は対象レースなし</p>'}
</section>

<!-- ⑨ 激走モーター -->
<section id="motor">
  {_render_brand_header("motor_hot", f"全{len(hot_motor)}件", _section_comment_motor(hot_motor, "激走モーター"))}
  {motor_table(hot_motor, hot_cols)}
</section>

<!-- ⑩ 覚醒モーター -->
<section id="awake">
  {_render_brand_header("motor_awk", f"全{len(awake_motor)}件", _section_comment_motor(awake_motor, "覚醒モーター"))}
  {motor_table(awake_motor, awake_cols)}
</section>

<!-- ⑪ 転がし候補 -->
<section id="korogashi">
  {_render_brand_header("korogashi", "", _section_comment_korogashi(korogashi_data))}
  {''.join(
      f'<a href="#" class="kr-link">{s.get("venue","")}{s.get("race","")}R　'
      f'{s.get("lane","")}号艇 {s.get("racer_name","")}　適性{s.get("fitness","")}点　[{s.get("verdict","")}]</a>'
      for s in korogashi_data.get("top10", [])
  ) or '<p class="no-data">本日のデータはまだありません</p>'}
</section>

<!-- ⑫ 今日の注目レーサー -->
<section id="racer">
  {_render_brand_header("racer", "", "勝率・モーター・直近成績から本日特に注目すべき選手をピックアップしています。")}
  {''.join(
      f'<div class="race-card a"><div class="rc-header">'
      f'<strong class="rc-name">{d.get("racer","?")}</strong>'
      f'<span class="rc-racer">{d.get("venue","")}{d.get("race","")}R {d.get("racer_class","")}</span>'
      f'</div><div class="rc-reason">{d.get("reason","")}</div></div>'
      for d in sorted(all_danger, key=lambda x: -x.get("score",0))[:5]
  ) or '<p class="no-data">本日のデータはまだありません</p>'}
</section>

<!-- ⑬ 昨日の検証 -->
{yesterday_block}

<!-- ダウンロード案内 -->
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
  <div class="footer-line">AI競艇新聞 | 全レース機械学習分析 | 毎日更新</div>
  <div class="footer-meta">
    生成日時: {date_disp} {now_str}　|　AI Engine v3.0　|
    総データ件数: {len(all_danger)+len(all_manshuu)+len(hot_motor)+len(awake_motor)}件　|
    対象開催場数: {len(venue_conditions)}場
  </div>
  <div class="footer-page">1 / 1 ページ</div>
  <div class="footer-note">※本レポートはデータ分析結果であり、的中を保証するものではありません。</div>
</div>

<!-- ⑪ ページトップへ戻る -->
<a href="#top" id="back-to-top" title="トップへ戻る">▲</a>

<script>
function filterRaces(keyword) {{
  var kw = keyword.trim().toLowerCase();
  var rows = document.querySelectorAll('.idx-row');
  var visibleCount = 0;
  rows.forEach(function(row) {{
    var raceText = row.querySelector('.idx-race');
    var text = raceText ? raceText.textContent.toLowerCase() : '';
    if (kw === '' || text.indexOf(kw) !== -1) {{
      row.style.display = '';
      visibleCount++;
    }} else {{
      row.style.display = 'none';
    }}
  }});
  var counter = document.getElementById('search-result-count');
  if (kw === '') {{
    counter.textContent = '';
  }} else {{
    counter.textContent = visibleCount + '件ヒット';
  }}
}}

window.addEventListener('scroll', function() {{
  var btn = document.getElementById('back-to-top');
  if (window.scrollY > 400) {{
    btn.style.display = 'flex';
  }} else {{
    btn.style.display = 'none';
  }}
}});
</script>

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

        def _hex_to_rgb(hex_color: str) -> tuple:
            h = hex_color.lstrip("#")
            return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

        C_C = (66, 165, 245)   # Cランク用カラー（青）

        def _pdf_rank_color(score):
            rank = rank_of(score)
            return {"S": C_S, "A": C_A, "B": C_B, "C": C_C}.get(rank, C_B)

        def _pdf_rank_label(score):
            return rank_of(score)

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
            rc    = _pdf_rank_color(score)
            rl    = _pdf_rank_label(score)
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
            rc    = _pdf_rank_color(score)
            rl    = _pdf_rank_label(score)
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
        # 危険 セクションタイトル: 58、各行: 64（表示上限20件）
        # 万舟 セクションタイトル: 58、各行: 64（表示上限10件）
        # 激走 セクションタイトル: 58、各行: 48
        # 覚醒 セクションタイトル: 58、各行: 48
        # フッター: 60、余白: 30

        _danger_count_disp  = min(len(all_danger), 20)
        _manshuu_count_disp = min(len(all_manshuu), 10)

        total_h = (
            100 + 20 +
            58 + _danger_count_disp  * 64 + 20 +
            58 + _manshuu_count_disp * 64 + 20 +
            58 + len(hot_motor)   * 48 + 20 +
            58 + len(awake_motor) * 48 + 60
        )
        total_h = max(1600, total_h)

        # ── 実際に描画（表示は新聞本文と同じ上限を適用） ─────
        _danger_disp  = sorted(all_danger,  key=lambda x: -x.get("score", 0))[:20]
        _manshuu_disp = sorted(all_manshuu, key=lambda x: -x.get("score", 0))[:10]

        img  = Image.new("RGB", (W, total_h), C_BG)
        draw = ImageDraw.Draw(img)

        y[0] = 0
        draw_header(draw)
        add_gap(20)

        draw_section_title(draw, f"⚠ 危険な1号艇  掲載{len(_danger_disp)}件（全{len(all_danger)}件中）", C_S)
        for d in _danger_disp:
            draw_danger_row(draw, d)
        add_gap(20)

        draw_section_title(draw, f"¥ 万舟警報  掲載{len(_manshuu_disp)}件（全{len(all_manshuu)}件中）", C_A)
        for u in _manshuu_disp:
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
# ⑬ note用Markdown生成（目次付き）
# ════════════════════════════════════════════════════════════

def generate_markdown(data: dict, output_path: str) -> None:
    """
    ⑬ note貼り付け用のMarkdown版を生成する。
    HTML版と同じデータソースを使い、目次付きのテキスト形式で出力する。
    """
    date_str  = data.get("date", "")
    date_disp = f"{date_str[4:6]}/{date_str[6:8]}" if len(date_str) >= 8 else ""

    all_danger  = data.get("all_danger",  data.get("danger_boat1", []))
    all_manshuu = data.get("all_manshuu", data.get("manshuu_alert", []))
    hot_motor   = data.get("hot_motor", [])
    awake_motor = data.get("awakening_motor", [])

    danger_display  = sorted(all_danger,  key=lambda x: -x.get("score", 0))[:20]
    manshuu_display = sorted(all_manshuu, key=lambda x: -x.get("score", 0))[:10]

    venue_conditions = calc_venue_conditions(data)
    dashboard = build_dashboard(data, venue_conditions)
    editor_note = _generate_editor_note(data, venue_conditions)

    lines = [
        f"# 📰 AI競艇新聞 {date_disp}",
        "",
        f"*{SYSTEM_NAME} | AI Version: {AI_VERSION}*",
        "",
        "## 📑 目次",
        "",
        "- [今日のダッシュボード](#dashboard)",
        "- [AI編集部コメント](#comment)",
        "- [開催場ヒートマップ](#heatmap)",
        "- [危険な1号艇](#danger)",
        "- [万舟警報](#manshuu)",
        "- [激走モーター](#motor-hot)",
        "- [覚醒モーター](#motor-awk)",
        "",
        "---",
        "",
        "## <a id=\"dashboard\"></a>📋 今日のダッシュボード",
        "",
        f"- 解析開催場数: **{dashboard['venues_analyzed']}場**",
        f"- 解析レース数: **{dashboard['races_analyzed']}レース**",
        f"- 危険艇Sランク: **{dashboard['danger_s_count']}件**",
        f"- 万舟Sランク: **{dashboard['manshuu_s_count']}件**",
        f"- 転がし候補: **{dashboard['korogashi_count']}件**",
        f"- 高配当期待: **{dashboard['hot_high_count']}件**",
    ]
    if dashboard["best_venue"]:
        lines.append(f"- ⭐ 最も期待: **{dashboard['best_venue']}**")
    if dashboard["rough_venue"]:
        lines.append(f"- 🌊 最も荒れそう: **{dashboard['rough_venue']}**")
    if dashboard["calm_venue"]:
        lines.append(f"- 🛡️ 最も堅そう: **{dashboard['calm_venue']}**")

    lines += [
        "",
        "---",
        "",
        "## <a id=\"comment\"></a>🤖 AI編集部コメント",
        "",
        editor_note,
        "",
        "---",
        "",
        "## <a id=\"heatmap\"></a>📈 開催場ヒートマップ",
        "",
        "| 開催場 | 総合 | 🚨危険艇 | 💰万舟 | 🎯転がし | ⚡激走 | 📈覚醒 |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in venue_conditions:
        items = c["items"]
        def _cell(k):
            it = items[k]
            return f"{it['heat']}{it['score']:.0f}" if it["count"] > 0 else "－"
        lines.append(
            f"| {c['venue']} | {c['stars']} | {_cell('danger')} | {_cell('manshuu')} | "
            f"{_cell('korogashi')} | {_cell('motor_hot')} | {_cell('motor_awk')} |"
        )

    lines += [
        "",
        "---",
        "",
        f"## <a id=\"danger\"></a>🚨 AI危険艇速報　掲載{len(danger_display)}件（全{len(all_danger)}件中）",
        "",
        _section_comment_danger(all_danger),
        "",
    ]
    for d in danger_display:
        rank = rank_of(d.get("score", 0))
        lines.append(f"### {rank}ランク {d.get('venue','')}{d.get('race','')}R　{d.get('racer','?')}")
        lines.append(f"- 理由: {d.get('reason','')}")
        lines.append("")

    lines += [
        "---",
        "",
        f"## <a id=\"manshuu\"></a>💰 AI万舟警報　掲載{len(manshuu_display)}件（全{len(all_manshuu)}件中）",
        "",
        _section_comment_manshuu(all_manshuu),
        "",
    ]
    for u in manshuu_display:
        rank = rank_of(u.get("score", 0))
        lines.append(f"### {rank}ランク {u.get('venue','')}{u.get('race','')}R")
        lines.append(f"- 注目: {u.get('key_racer','')}")
        lines.append(f"- 理由: {u.get('key_reason','').replace(chr(0x1F525)+' ', '')}")
        lines.append("")

    lines += [
        "---",
        "",
        f"## <a id=\"motor-hot\"></a>⚡ AI激走モーター　全{len(hot_motor)}件",
        "",
        _section_comment_motor(hot_motor, "激走モーター"),
        "",
    ]
    for m in hot_motor:
        lines.append(f"- {m.get('venue','')} {m.get('motor_no','')}号機（直近5走: {m.get('recent5','---')}）")

    lines += [
        "",
        "---",
        "",
        f"## <a id=\"motor-awk\"></a>📈 AI覚醒モーター　全{len(awake_motor)}件",
        "",
        _section_comment_motor(awake_motor, "覚醒モーター"),
        "",
    ]
    for a in awake_motor:
        lines.append(f"- {a.get('venue','')} {a.get('motor_no','')}号機（直近10走: {a.get('recent10','---')}）")

    lines += [
        "",
        "---",
        "",
        "*本レポートはデータ分析結果であり、的中を保証するものではありません。*",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info("[Markdown] 保存: %s", output_path)


# ════════════════════════════════════════════════════════════
# メール送信
# ════════════════════════════════════════════════════════════

def send_note_report(
    html_path: str,
    pdf_path: Optional[str],
    csv_paths: list[str],
    date_str: str,
    dry_run: bool = False,
    md_path: Optional[str] = None,
) -> bool:
    date_disp = f"{date_str[4:6]}/{date_str[6:8]}"
    subject   = f"📰 AI競艇新聞 {date_disp}（note用）"
    body = (
        f"AI競艇新聞 {date_disp} を生成しました。\n\n"
        "【添付ファイル一覧】\n"
        f"  note.html     ← noteエディタにコピペ\n"
        f"  newspaper.pdf ← 印刷・保存用PDF\n"
        f"  note.md       ← Markdown版（目次付き）\n"
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
        if md_path:  files.append(md_path)
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
    if md_path and os.path.exists(md_path):
        attach_files.append((md_path, "text", "markdown"))
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
# ⑨ X向け投稿生成（ブランドごとの興味喚起テキスト）
# ════════════════════════════════════════════════════════════

def generate_x_teaser(data: dict) -> str:
    """
    AI一致指数95以上のレースを核に、Xで投稿しやすい
    興味喚起テキストを生成する。「続きは新聞へ」で着地させる。
    """
    sorted_index, brand_counts, race_scores = _build_race_index(data)
    if not race_scores:
        return "本日はデータ準備中です。\n\n続きは新聞へ📰"

    high_match = sorted(
        [(k, v) for k, v in race_scores.items() if v["match_index"] >= 95],
        key=lambda kv: -kv[1]["match_index"],
    )

    brand_names_jp = {
        "danger": "危険艇", "manshuu": "万舟", "korogashi": "転がし",
        "motor_hot": "激走", "motor_awk": "覚醒", "hot_high": "高配当期待",
    }

    lines = ["━━━━━━━━━━━━━━", ""]

    if high_match:
        lines.append(f"今日はAI一致指数95以上が{len(high_match)}レースあります。")
        lines.append("")
        top_key, top_s = high_match[0]
        venue, race = top_key
        brands = next((b for v, r, b in sorted_index if (v, r) == top_key), [])
        matched = "・".join(brand_names_jp.get(b, b) for b in brands if b in brand_names_jp)
        lines.append(f"特に{venue}{race}Rは")
        lines.append(f"{matched}")
        lines.append(f"の{len(brands)}指標一致。")
    else:
        # 一致指数トップのレースを代わりに紹介
        top_key = max(race_scores, key=lambda k: race_scores[k]["match_index"])
        top_s   = race_scores[top_key]
        venue, race = top_key
        brands = next((b for v, r, b in sorted_index if (v, r) == top_key), [])
        matched = "・".join(brand_names_jp.get(b, b) for b in brands if b in brand_names_jp)
        lines.append(f"本日の最有力候補は{venue}{race}R。")
        lines.append(f"{matched}が一致（一致指数{top_s['match_index']}）。")

    lines += ["", "━━━━━━━━━━━━━━", "", "続きは新聞へ📰", "",
              "#競艇 #ボートレース #AI予想 #競艇新聞"]
    return "\n".join(lines)


def _brand_summary_facts(data: dict, brand: str) -> dict:
    """ブランドの要約事実（件数・最注目レース等）をまとめて返す内部ヘルパー"""
    all_danger  = data.get("all_danger",  data.get("danger_boat1", []))
    all_manshuu = data.get("all_manshuu", data.get("manshuu_alert", []))
    hot_motor   = data.get("hot_motor", [])
    awake_motor = data.get("awakening_motor", [])
    kdata       = _load_korogashi_cache()

    facts = {"brand": brand, "count": 0, "s_count": 0, "top": None}

    if brand == "danger":
        facts["count"]   = len(all_danger)
        facts["s_count"] = sum(1 for d in all_danger if rank_of(d.get("score",0)) == "S")
        if all_danger:
            facts["top"] = max(all_danger, key=lambda x: x.get("score", 0))
    elif brand == "manshuu":
        facts["count"]   = len(all_manshuu)
        facts["s_count"] = sum(1 for u in all_manshuu if rank_of(u.get("score",0)) == "S")
        if all_manshuu:
            facts["top"] = max(all_manshuu, key=lambda x: x.get("score", 0))
    elif brand == "motor_hot":
        facts["count"] = len(hot_motor)
    elif brand == "motor_awk":
        facts["count"] = len(awake_motor)
    elif brand == "korogashi":
        buy = [s for s in kdata.get("top10", []) if s.get("verdict") == "購入"]
        facts["count"] = len(buy)
        if buy:
            facts["top"] = buy[0]

    return facts


def generate_brand_teaser(data: dict, brand: str, style: str = "normal") -> str:
    """
    ⑪ 指定ブランドの興味喚起テキストを生成する。
    style: "normal"(通常) / "buzz"(バズ狙い) / "quote"(引用用) / "reply"(返信用)
    """
    facts = _brand_summary_facts(data, brand)
    icon  = brand_icon(brand)
    name  = brand_name(brand)
    top   = facts["top"]

    top_venue = top.get("venue","") if top else ""
    top_race  = top.get("race","")  if top else ""
    top_score = top.get("score", top.get("fitness", 0)) if top else 0

    if style == "normal":
        lines = ["━━━━━━━━━━━━━━", "", f"{icon} {name}", ""]
        lines.append(f"本日はSランク{facts['s_count']}件を含む{facts['count']}件を抽出。" if facts["s_count"] else f"本日{facts['count']}件を検出。")
        if top:
            lines.append(f"最注目は{top_venue}{top_race}R（スコア{top_score}）。")
        lines += ["", "━━━━━━━━━━━━━━", "", "続きは新聞へ📰",
                  "#競艇 #ボートレース #AI予想"]
        return "\n".join(lines)

    elif style == "buzz":
        # 数字をフックにした「クリックしたくなる」文章＋質問
        lines = [f"{icon} 今日の{name}、"]
        if facts["s_count"] >= 3:
            lines[0] += f"Sランクが{facts['s_count']}件も…"
        elif top:
            lines[0] += f"{top_venue}{top_race}Rがヤバい。"
        else:
            lines[0] += f"{facts['count']}件を検出。"
        lines.append("")
        if top:
            lines.append(f"スコア{top_score}は今月でも上位クラス。")
        lines.append("")
        lines.append("あなたはどのレースが気になりますか？👀")
        lines.append("")
        lines.append("#競艇 #ボートレース #AI予想")
        return "\n".join(lines)

    elif style == "quote":
        # 引用リツイート用（短文、新聞リンクの引用を想定）
        if top:
            return (
                f"{icon} {top_venue}{top_race}R、AIスコア{top_score}。\n"
                f"{name}の本日No.1はこのレース。\n"
                "詳細は新聞で📰"
            )
        return f"{icon} 本日の{name}は{facts['count']}件。詳細は新聞で📰"

    elif style == "reply":
        # 返信・リプライ用（短く、フレンドリー）
        if top:
            return f"今日だと{top_venue}{top_race}Rが{name}の本命です！スコア{top_score}でした🙆"
        return f"今日は{name}が{facts['count']}件出てます！よければ新聞もチェックしてみてください📰"

    return generate_brand_teaser(data, brand, style="normal")


def generate_brand_teasers_all_styles(data: dict, brand: str) -> dict:
    """⑪ 1ブランドにつき4種類（通常/バズ/引用/返信）すべてを生成する"""
    return {
        "normal": generate_brand_teaser(data, brand, "normal"),
        "buzz":   generate_brand_teaser(data, brand, "buzz"),
        "quote":  generate_brand_teaser(data, brand, "quote"),
        "reply":  generate_brand_teaser(data, brand, "reply"),
    }


# ════════════════════════════════════════════════════════════
# ⑩ インタラクション投稿（アンケート）生成
# ════════════════════════════════════════════════════════════

def generate_poll_text(data: dict, variant: str = "race") -> dict:
    """
    ⑫ Xアンケート投稿用のテキストを生成する。
    variant: "race"(一致指数TOP3レース) / "venue"(コンディションTOP3開催場) / "korogashi"(転がし候補)
    戻り値: {"question": str, "options": [str, str, str, str]}
    """
    circled = ["①", "②", "③"]

    if variant == "venue":
        conditions = calc_venue_conditions(data)
        if not conditions:
            return {"question": "今日最も期待する開催場は？",
                    "options": ["①データ準備中", "②データ準備中", "③データ準備中", "④その他"]}
        options = [f"{circled[i]}{c['venue']}" for i, c in enumerate(conditions[:3])]
        while len(options) < 3:
            options.append(f"{circled[len(options)]}該当なし")
        options.append("④その他")
        return {"question": "今日最も期待する開催場は？", "options": options}

    if variant == "korogashi":
        kdata = _load_korogashi_cache()
        buy = [s for s in kdata.get("top10", []) if s.get("verdict") in ("購入", "注意")][:3]
        if not buy:
            return {"question": "今日の転がしチャレンジ、どのレースが気になる？",
                    "options": ["①データ準備中", "②データ準備中", "③データ準備中", "④見送りでいい"]}
        options = [f"{circled[i]}{s.get('venue','')}{s.get('race','')}R" for i, s in enumerate(buy)]
        while len(options) < 3:
            options.append(f"{circled[len(options)]}該当なし")
        options.append("④見送りでいい")
        return {"question": "今日の転がしチャレンジ、どのレースが気になる？", "options": options}

    # デフォルト: variant == "race"
    sorted_index, brand_counts, race_scores = _build_race_index(data)
    if not race_scores:
        return {
            "question": "今日最も気になるレースは？",
            "options": ["①データ準備中", "②データ準備中", "③データ準備中", "④その他"],
        }

    ranked = sorted(race_scores.items(), key=lambda kv: -kv[1]["match_index"])[:3]
    options = []
    for i, (key, s) in enumerate(ranked):
        venue, race = key
        options.append(f"{circled[i]}{venue}{race}R")
    while len(options) < 3:
        options.append(f"{circled[len(options)]}該当なし")
    options.append("④その他")

    return {
        "question": "今日最も気になるレースは？",
        "options": options,
    }


def format_poll_tweet(data: dict, variant: str = "race") -> str:
    """アンケート投稿のX用テキストを整形する"""
    poll = generate_poll_text(data, variant=variant)
    lines = [poll["question"], ""]
    lines.extend(poll["options"])
    lines += ["", "#競艇 #ボートレース #AI予想"]
    return "\n".join(lines)


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
    md_path   = "note.md"

    generate_html(data, html_path)
    generate_markdown(data, md_path)

    csv_paths = generate_csvs(data)

    if not args.html_only:
        ok = generate_pdf(html_path, pdf_path)
        if not ok:
            pdf_path = None
    else:
        pdf_path = None

    ok = send_note_report(html_path, pdf_path, csv_paths,
                          date_str, dry_run=args.dry_run, md_path=md_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
