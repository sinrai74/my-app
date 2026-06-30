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
# 各ブランドセクション冒頭のAIコメント生成
# ════════════════════════════════════════════════════════════

def _section_comment_danger(all_danger: list) -> str:
    if not all_danger:
        return "本日は対象レースがありません。"
    s_count = sum(1 for d in all_danger if d.get("score", 0) >= 80)
    venues  = list(dict.fromkeys(d.get("venue","") for d in all_danger[:5]))
    venue_txt = "・".join(venues[:3]) if venues else ""
    lines = [f"本日は危険艇Sランクが{s_count}件あります。"] if s_count else [f"本日は危険艇候補が{len(all_danger)}件あります。"]
    if venue_txt:
        lines.append(f"{venue_txt}で1号艇の信頼度低下が目立ちます。2〜4号艇に注目です。")
    return " ".join(lines)


def _section_comment_manshuu(all_manshuu: list) -> str:
    if not all_manshuu:
        return "本日は対象レースがありません。"
    s_count = sum(1 for u in all_manshuu if u.get("score", 0) >= 80)
    venues  = list(dict.fromkeys(u.get("venue","") for u in all_manshuu[:5]))
    venue_txt = "・".join(venues[:3]) if venues else ""
    lines = [f"本日は万舟候補が{len(all_manshuu)}件あります。"]
    if s_count:
        lines.append(f"うちSランクは{s_count}件。")
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

# ── ブランドアイコン定義 ──────────────────────────────────
# 要件のアイコン配分: 🚨危険艇 💰万舟 🔥高配当期待 ⚡激走 📈覚醒 🎯転がし 👤注目レーサー
BRAND_ICONS = {
    "danger":    "🚨",   # AI危険艇速報
    "manshuu":   "💰",   # AI万舟警報
    "hot_high":  "🔥",   # AI高配当期待（万舟Sランク＝荒れ指数80以上）
    "motor_hot": "⚡",   # AI激走モーター
    "motor_awk": "📈",   # AI覚醒モーター
    "korogashi": "🎯",   # AI転がし候補
    "racer":     "👤",   # 注目レーサー
}
BRAND_NAMES = {
    "danger":    "AI危険艇速報",
    "manshuu":   "AI万舟警報",
    "hot_high":  "AI高配当期待",
    "motor_hot": "AI激走モーター",
    "motor_awk": "AI覚醒モーター",
    "korogashi": "AI転がし候補",
    "racer":     "今日の注目レーサー",
}

# AI総合注目度の重み（要件指定）
OVERALL_WEIGHTS = {
    "danger":    0.25,
    "manshuu":   0.25,
    "korogashi": 0.20,
    "motor_hot": 0.15,
    "motor_awk": 0.10,
    "hot_high":  0.05,
}

HOT_HIGH_THRESHOLD = 80   # 万舟Sランク＝高配当期待の閾値（フォールバック用固定値）
HOT_HIGH_RATIO     = 0.3 # 万舟TOP10のうち上位30%を高配当期待とする


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
    全レースのブランド掲載状況とAI総合注目度を集計する。
    戻り値: (sorted_index, brand_counts, race_scores)
      sorted_index: [(venue, race, [brand_key, ...]), ...]  会場→レース番号昇順
      brand_counts: {"danger": 18, "manshuu": 10, ...}
      race_scores:  {(venue, race): {"overall": 98, "danger": 85, "manshuu": 82,
                                      "korogashi": 90, "motor_hot": None, "motor_awk": None,
                                      "hot_high": True/False}}
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

    # ── AI総合注目度の算出 ──────────────────────────────────
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

        race_scores[key] = {
            "overall":    round(min(100, norm_overall), 1),
            "danger":     danger_s,
            "manshuu":    manshuu_s,
            "korogashi":  korogashi_s,
            "hot_high":   hot_high_s,
            "motor_hot":  motor_hot_s if motor_hot_present else None,
            "motor_awk":  motor_awk_s if motor_awk_present else None,
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
    """優先順位: danger > manshuu の順でジャンプ先アンカーを決める"""
    base = _race_anchor(venue, race)
    if "danger" in brands:
        return base
    if "manshuu" in brands or "hot_high" in brands:
        return base + "-manshuu"
    return base


def _stars(score: float) -> str:
    """0-100点を★5段階表示に変換"""
    n = max(0, min(5, round(score / 20)))
    return "★" * n + "☆" * (5 - n)


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
        anchor = _anchor_for(venue, race, brands)
        icons  = " ".join(BRAND_ICONS.get(b, "") for b in brands)
        rows_html += f"""
<a href="#{anchor}" class="idx-row">
  <span class="idx-race">{venue}{race}R</span>
  <span class="idx-icons">{icons}</span>
</a>"""

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
    """⭐ 本日のAIイチオシ セクション（総合注目度1位のレース）"""
    sorted_index, brand_counts, race_scores = _build_race_index(data)
    if not race_scores:
        return ""

    best_key = max(race_scores, key=lambda k: race_scores[k]["overall"])
    venue, race = best_key
    s = race_scores[best_key]
    brands = next((b for v, r, b in sorted_index if (v, r) == best_key), [])

    detail = _race_detail(venue, race, data) or {}
    d, u, k = detail.get("danger"), detail.get("manshuu"), detail.get("korogashi")

    badges = _brand_badge_html(brands)
    stars  = _stars(s["overall"])

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

    # 総合コメント生成
    condition_parts = []
    if d and d.get("score", 0) >= 80: condition_parts.append("危険艇Sランク")
    if u and u.get("score", 0) >= 80: condition_parts.append("万舟Sランク")
    if k and k.get("fitness", 0) >= 90: condition_parts.append("転がし適性90以上")
    if s.get("motor_hot"): condition_parts.append("激走モーター対象")
    if s.get("motor_awk"): condition_parts.append("覚醒モーター対象")

    cond_text = "・".join(condition_parts) if condition_parts else "複数指標で高評価"
    ai_comment = (
        f"本日唯一、{cond_text}の条件が重なりました。"
        f"AI総合注目度{s['overall']}点は本日最高評価です。"
    )

    anchor = _anchor_for(venue, race, brands)

    return f"""
<section id="pickup">
  <h2>⭐ 本日のAIイチオシ</h2>
  <div class="pickup-card">
    <a href="#{anchor}" class="pickup-race">{venue}{race}R</a>
    <div class="pickup-score">
      <span class="pickup-num">AI総合注目度 {s['overall']}</span>
      <span class="pickup-stars">{stars}</span>
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


def _render_top5_section(data: dict) -> str:
    """🏆 本日の注目レースTOP5 セクション"""
    sorted_index, brand_counts, race_scores = _build_race_index(data)
    if not race_scores:
        return ""

    ranked = sorted(race_scores.items(), key=lambda kv: -kv[1]["overall"])[:5]

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
  <span class="top5-score">注目度{s['overall']}</span>
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
    race_index_data = _build_race_index(data)   # (sorted_index, brand_counts, race_scores)
    pickup_section  = _render_pickup_section(data)
    top5_section    = _render_top5_section(data)
    index_section   = _render_index_section(data)

    # 転がし候補データ・高配当期待（動的閾値、新聞表示分のmanshuu_displayベース）
    korogashi_data    = _load_korogashi_cache()
    hot_high_cutoff   = _hot_high_threshold(manshuu_display)
    high_payout       = [u for u in manshuu_display if u.get("score", 0) >= hot_high_cutoff]

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
            rank_cls = "s" if score >= 80 else "a" if score >= 60 else "b"
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
            rank_cls = "s" if score >= 80 else "a" if score >= 60 else "b"
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
/* レースインデックス */
#race-index{{background:var(--card);border:1px solid var(--border);
  border-radius:10px;padding:18px;margin-bottom:20px}}
#race-index h2{{border-bottom:none;padding-top:0}}
.index-grid{{display:flex;flex-direction:column;gap:2px;margin:12px 0 16px}}
.idx-row{{display:flex;justify-content:space-between;align-items:center;
  padding:9px 12px;border-radius:6px;background:#1a1a30;
  text-decoration:none;color:var(--text);transition:background .15s}}
.idx-row:hover,.idx-row:active{{background:#22224a}}
.idx-race{{font-weight:bold;font-size:.92em}}
.idx-icons{{font-size:1em;letter-spacing:2px}}
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
.pickup-num{{font-size:1.1em;color:#ffd54f;font-weight:bold}}
.pickup-stars{{font-size:1.1em;color:#ffd54f}}
.pickup-badges{{margin-bottom:14px}}
.pickup-comment{{background:#14142a;border-radius:8px;padding:12px;margin-bottom:12px}}
.pickup-comment strong{{color:#ffd54f;font-size:.9em}}
.pickup-comment p{{color:#ccc;font-size:.92em;margin-top:6px}}
.pickup-points strong{{color:#81c784;font-size:.9em}}
.pickup-points ul{{list-style:none;margin-top:8px}}
.pickup-points li{{color:#bbb;font-size:.88em;padding:4px 0 4px 4px;
  border-left:2px solid #2a4a2a;padding-left:10px;margin-bottom:4px}}
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
.footer{{text-align:center;color:#444;font-size:.78em;padding:20px 0;
  border-top:1px solid var(--border);margin-top:10px}}
.footer-line{{color:#666;margin-bottom:6px}}
.footer-meta{{color:#444;font-size:.92em;margin-bottom:6px}}
.footer-note{{color:#3a3a3a;font-size:.85em}}
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

<!-- ③ 本日のAIイチオシ -->
{pickup_section}

<!-- ④ 本日の注目レースTOP5 -->
{top5_section}

<!-- ⑤ 本日のレースINDEX -->
{index_section}

<!-- ⑥ 危険な1号艇 -->
<section id="danger">
  <h2>🚨 AI危険艇速報　掲載{len(danger_display)}件（全{len(all_danger)}件中）</h2>
  <div class="section-comment">{_section_comment_danger(all_danger)}</div>
  {danger_rows()}
</section>

<!-- ⑦ 万舟警報 -->
<section id="manshuu">
  <h2>💰 AI万舟警報　掲載{len(manshuu_display)}件（全{len(all_manshuu)}件中）</h2>
  <div class="section-comment">{_section_comment_manshuu(all_manshuu)}</div>
  {manshuu_rows()}
</section>

<!-- ⑧ 高配当期待 -->
<section id="hot-high">
  <h2>🔥 AI高配当期待　{len(high_payout)}レース</h2>
  <div class="section-comment">{_section_comment_hot_high(manshuu_display)}</div>
  {''.join(f'<a href="#{_race_anchor(u.get("venue",""), u.get("race",""))}-manshuu" class="hp-link">{u.get("venue","")}{u.get("race","")}R　荒れ指数{u.get("score",0)}</a>' for u in high_payout) or '<p class="no-data">本日は対象レースなし</p>'}
</section>

<!-- ⑨ 激走モーター -->
<section id="motor">
  <h2>⚡ AI激走モーター　全{len(hot_motor)}件</h2>
  <div class="section-comment">{_section_comment_motor(hot_motor, "激走モーター")}</div>
  {motor_table(hot_motor, hot_cols)}
</section>

<!-- ⑩ 覚醒モーター -->
<section id="awake">
  <h2>📈 AI覚醒モーター　全{len(awake_motor)}件</h2>
  <div class="section-comment">{_section_comment_motor(awake_motor, "覚醒モーター")}</div>
  {motor_table(awake_motor, awake_cols)}
</section>

<!-- ⑪ 転がし候補 -->
<section id="korogashi">
  <h2>🎯 AI転がし候補</h2>
  <div class="section-comment">{_section_comment_korogashi(korogashi_data)}</div>
  {''.join(
      f'<a href="#" class="kr-link">{s.get("venue","")}{s.get("race","")}R　'
      f'{s.get("lane","")}号艇 {s.get("racer_name","")}　適性{s.get("fitness","")}点　[{s.get("verdict","")}]</a>'
      for s in korogashi_data.get("top10", [])
  ) or '<p class="no-data">本日のデータはまだありません</p>'}
</section>

<!-- ⑫ 今日の注目レーサー -->
<section id="racer">
  <h2>👤 今日の注目レーサー</h2>
  <div class="section-comment">勝率・モーター・直近成績から本日特に注目すべき選手をピックアップしています。</div>
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
    総データ件数: {len(all_danger)+len(all_manshuu)+len(hot_motor)+len(awake_motor)}件
  </div>
  <div class="footer-note">※本レポートはデータ分析結果であり、的中を保証するものではありません。</div>
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
