#!/usr/bin/env python3
"""
x_verification.py  ── 本日のAI予測結果を集計してメール送信する

hit_record.csv から当日分を読み込み、以下を集計：
  - 危険な1号艇：何件通知 → 何件飛んだか
  - 収支：投資額・回収額・ROI
  - 万舟警報：万舟発生件数（払戻3万円以上）
  - 連続好成績 / 連敗情報

Usage:
    python x_verification.py                  # 今日の結果を集計して送信
    python x_verification.py --date 20260628  # 指定日
    python x_verification.py --dry-run        # 送信せず表示のみ
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

log = logging.getLogger("x_verification")

JST      = timezone(timedelta(hours=9))
MAIL_TO  = "bigkirinuki@gmail.com"
HIT_CSV  = "hit_record.csv"

# 万舟の定義：払戻3万円以上
MANSHUU_PAYOUT = 30000


# ════════════════════════════════════════════════════════════
# データ読み込み
# ════════════════════════════════════════════════════════════

def _today_jst() -> str:
    return datetime.now(JST).strftime("%Y%m%d")


def load_today_records(target_date: Optional[str] = None) -> list[dict]:
    """hit_record.csv から指定日のレコードを返す"""
    date_str = target_date or _today_jst()
    if not os.path.exists(HIT_CSV):
        log.warning("[検証] %s が見つかりません", HIT_CSV)
        return []

    records: list[dict] = []
    with open(HIT_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_date = row.get("date", "").replace("-", "")
            if row_date == date_str:
                records.append(row)

    log.info("[検証] %s: %d件読み込み", date_str, len(records))
    return records


# ════════════════════════════════════════════════════════════
# 集計
# ════════════════════════════════════════════════════════════

def _safe_float(val, default: float = 0.0) -> float:
    try:
        return float(val) if val not in (None, "", "None") else default
    except (ValueError, TypeError):
        return default


def _safe_int(val, default: int = 0) -> int:
    try:
        return int(float(val)) if val not in (None, "", "None") else default
    except (ValueError, TypeError):
        return default


def aggregate(records: list[dict]) -> dict:
    """レコードを集計して辞書で返す"""

    # 通知済みレース（pred_combo がある＝実際に通知したもの）のみ対象
    notified = [r for r in records if r.get("pred_combo", "")]

    total_notified = len(notified)
    if total_notified == 0:
        return {
            "total_notified": 0,
            "total_hit": 0,
            "hit_rate": 0.0,
            "total_cost": 0,
            "total_profit": 0,
            "roi": 0.0,
            "boat1_flew": 0,
            "boat1_total": 0,
            "boat1_flew_rate": 0.0,
            "manshuu_count": 0,
            "manshuu_races": [],
            "best_race": None,
            "worst_loss": 0,
            "streak_win": 0,
            "streak_lose": 0,
        }

    total_hit    = sum(1 for r in notified if _safe_int(r.get("hit")) == 1)
    total_cost   = sum(_safe_float(r.get("cost"))   for r in notified)
    total_profit = sum(_safe_float(r.get("profit")) for r in notified)
    roi = (total_profit / total_cost * 100) if total_cost > 0 else 0.0
    hit_rate = (total_hit / total_notified * 100) if total_notified > 0 else 0.0

    # 1号艇が飛んだかどうか（result_combo の1着が1号艇でない）
    boat1_total = 0
    boat1_flew  = 0
    for r in notified:
        result = r.get("result_combo", "")
        if result and "-" in result:
            boat1_total += 1
            first_boat = result.split("-")[0].strip()
            if first_boat != "1":
                boat1_flew += 1
    boat1_flew_rate = (boat1_flew / boat1_total * 100) if boat1_total > 0 else 0.0

    # 万舟（払戻3万以上）
    manshuu_races: list[dict] = []
    for r in notified:
        payout = _safe_float(r.get("payout"))
        if payout >= MANSHUU_PAYOUT:
            manshuu_races.append({
                "venue":  r.get("venue", ""),
                "race":   r.get("race", ""),
                "combo":  r.get("result_combo", ""),
                "payout": int(payout),
                "hit":    _safe_int(r.get("hit")),
            })

    # 最高払戻レース
    hit_records = [r for r in notified if _safe_int(r.get("hit")) == 1]
    best_race = None
    if hit_records:
        best = max(hit_records, key=lambda r: _safe_float(r.get("payout")))
        best_race = {
            "venue":  best.get("venue", ""),
            "race":   best.get("race", ""),
            "combo":  best.get("pred_combo", ""),
            "payout": int(_safe_float(best.get("payout"))),
            "profit": int(_safe_float(best.get("profit"))),
        }

    # 最大損失
    worst_loss = min((_safe_float(r.get("profit")) for r in notified), default=0)

    # 連勝・連敗
    streak_win = streak_lose = 0
    current_win = current_lose = 0
    for r in notified:
        if _safe_int(r.get("hit")) == 1:
            current_win  += 1
            current_lose  = 0
        else:
            current_lose += 1
            current_win   = 0
        streak_win  = max(streak_win,  current_win)
        streak_lose = max(streak_lose, current_lose)

    return {
        "total_notified":  total_notified,
        "total_hit":       total_hit,
        "hit_rate":        round(hit_rate, 1),
        "total_cost":      int(total_cost),
        "total_profit":    int(total_profit),
        "roi":             round(roi, 1),
        "boat1_flew":      boat1_flew,
        "boat1_total":     boat1_total,
        "boat1_flew_rate": round(boat1_flew_rate, 1),
        "manshuu_count":   len(manshuu_races),
        "manshuu_races":   manshuu_races,
        "best_race":       best_race,
        "worst_loss":      int(worst_loss),
        "streak_win":      streak_win,
        "streak_lose":     streak_lose,
    }


# ════════════════════════════════════════════════════════════
# テキスト生成
# ════════════════════════════════════════════════════════════

def _star(value: float, thresholds: list[float], reverse: bool = False) -> str:
    """数値を★表示に変換。reverse=Trueは値が小さいほど良い"""
    stars = 5
    for t in thresholds:
        if (value >= t and not reverse) or (value <= t and reverse):
            break
        stars -= 1
    filled  = "★" * max(0, min(5, stars))
    empty   = "☆" * (5 - len(filled))
    return filled + empty


def build_verification_text(agg: dict, date_str: str, records: Optional[list] = None) -> str:
    """X投稿用のテキストを生成する"""
    date_disp = f"{date_str[4:6]}/{date_str[6:8]}"

    if agg["total_notified"] == 0:
        return (
            f"📊【{date_disp} AI成績】📊\n\n"
            "本日は通知なし（対象レースなし）\n\n"
            "#競艇 #ボートレース #AI予想"
        )

    # 収支の絵文字
    profit = agg["total_profit"]
    if   profit > 0:   profit_emoji = "✅"
    elif profit == 0:  profit_emoji = "➖"
    else:              profit_emoji = "❌"

    # ★評価
    hit_stars    = _star(agg["hit_rate"],        [60, 50, 40, 30, 0])
    flew_stars   = _star(agg["boat1_flew_rate"], [70, 60, 50, 40, 0])
    roi_stars    = _star(agg["roi"],             [20, 10, 0, -10, -999])

    lines = [
        f"📊【{date_disp} AIの本日成績】📊",
        "",
        "━━ 危険な1号艇 ━━",
        f"通知レース数: {agg['total_notified']}件",
        f"的中: {agg['total_hit']}件  的中率: {agg['hit_rate']}%",
        f"1号艇が飛んだ率: {agg['boat1_flew_rate']}%  {flew_stars}",
        "",
        "━━ 収支 ━━",
        f"{profit_emoji} 投資: {agg['total_cost']:,}円",
        f"{profit_emoji} 回収: {agg['total_cost'] + profit:,}円",
        f"{profit_emoji} 損益: {profit:+,}円  ROI: {agg['roi']:+.1f}%",
        f"精度: {roi_stars}",
        "",
    ]

    # 万舟警報結果
    if agg["manshuu_count"] > 0:
        lines.append("━━ 万舟警報 ━━")
        lines.append(f"万舟発生: {agg['manshuu_count']}件 🎯")
        for m in agg["manshuu_races"][:3]:
            hit_mark = "✅" if m["hit"] else "❌"
            lines.append(f"  {hit_mark} {m['venue']}{m['race']}R {m['combo']} {m['payout']:,}円")
        lines.append("")

    # 最高的中
    if agg["best_race"]:
        b = agg["best_race"]
        lines += [
            "━━ 本日最高払戻 ━━",
            f"🏆 {b['venue']}{b['race']}R {b['combo']}",
            f"   払戻 {b['payout']:,}円 / 損益 {b['profit']:+,}円",
            "",
        ]

    # 連勝・連敗
    if agg["streak_win"] >= 3:
        lines.append(f"🔥 本日{agg['streak_win']}連勝の時間帯あり！")
    if agg["streak_lose"] >= 5:
        lines.append(f"⚠️ 最大{agg['streak_lose']}連敗あり（慎重に）")

    # 外れ公開（信頼性向上）
    if records:
        notified_all = [r for r in records if r.get("pred_combo", "")]
        missed_all   = [r for r in notified_all if _safe_int(r.get("hit")) == 0]
        if missed_all:
            lines += ["━━ 外れたレース（公開） ━━"]
            for r in missed_all[:3]:
                lines.append(
                    f"❌ {r.get('venue','')}{r.get('race','')}R"
                    f" 予想{r.get('pred_combo','-')} → 結果{r.get('result_combo','-')}"
                )
            if len(missed_all) > 3:
                lines.append(f"   他{len(missed_all)-3}件")
            lines += ["次回の精度向上に活用します🔧", ""]

    lines += [
        "",
        "明日も全レース分析します💪",
        "今日のAI予想はどうでしたか？💬",
        "#競艇 #ボートレース #AI予想 #競艇予想 #検証",
    ]
    return "\n".join(lines)


def build_mail_body(agg: dict, tweet_text: str, date_str: str) -> str:
    """メール本文（詳細版）を生成する"""
    date_disp = f"{date_str[4:6]}/{date_str[6:8]}"
    lines = [
        f"=== {date_disp} AI予測 検証レポート ===",
        "",
        tweet_text,
        "",
        "=" * 50,
        "【詳細データ】",
        f"通知レース数:     {agg['total_notified']}件",
        f"的中:             {agg['total_hit']}件",
        f"的中率:           {agg['hit_rate']}%",
        f"1号艇飛び:        {agg['boat1_flew']}/{agg['boat1_total']}件 ({agg['boat1_flew_rate']}%)",
        f"投資額:           {agg['total_cost']:,}円",
        f"損益:             {agg['total_profit']:+,}円",
        f"ROI:              {agg['roi']:+.1f}%",
        f"万舟発生:         {agg['manshuu_count']}件",
        f"最大連勝:         {agg['streak_win']}",
        f"最大連敗:         {agg['streak_lose']}",
        "",
        "=" * 50,
        "▶ X投稿用テキスト（コピペ用）",
        "",
        tweet_text,
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# メール送信
# ════════════════════════════════════════════════════════════

def send_verification(
    agg: dict,
    date_str: str,
    dry_run: bool = False,
    records: Optional[list] = None,
) -> bool:
    date_disp  = f"{date_str[4:6]}/{date_str[6:8]}"
    tweet_text = build_verification_text(agg, date_str, records=records)
    body       = build_mail_body(agg, tweet_text, date_str)
    subject    = (
        f"📊 [{date_disp}] AI成績 "
        f"的中{agg['total_hit']}/{agg['total_notified']}件 "
        f"損益{agg['total_profit']:+,}円"
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

# ════════════════════════════════════════════════════════════
# 外部から呼び出せる前日サマリー取得API
# ════════════════════════════════════════════════════════════

def get_yesterday_summary() -> dict:
    """
    前日の実績サマリーを辞書で返す。
    x_post.py の朝投稿（危険な1号艇）に「昨日の答え合わせ」として埋め込む。
    hit_record.csv がない・前日データがない場合は空辞書を返す。
    """
    yesterday = (datetime.now(JST) - timedelta(days=1)).strftime("%Y%m%d")
    records   = load_today_records(yesterday)
    if not records:
        return {}
    agg = aggregate(records)
    if agg["total_notified"] == 0:
        return {}

    # 外れたレース（hit=0 かつ pred_combo あり）
    notified = [r for r in records if r.get("pred_combo", "")]
    missed   = [r for r in notified if _safe_int(r.get("hit")) == 0]
    missed_summary = []
    for r in missed[:3]:   # 最大3件
        missed_summary.append(f"{r.get('venue','')}{r.get('race','')}R"
                               f"（結果{r.get('result_combo','-')}）")

    date_disp = f"{yesterday[4:6]}/{yesterday[6:8]}"
    return {
        "date":           yesterday,
        "date_disp":      date_disp,
        "total_notified": agg["total_notified"],
        "total_hit":      agg["total_hit"],
        "hit_rate":       agg["hit_rate"],
        "boat1_flew":     agg["boat1_flew"],
        "boat1_total":    agg["boat1_total"],
        "boat1_flew_rate":agg["boat1_flew_rate"],
        "profit":         agg["total_profit"],
        "roi":            agg["roi"],
        "manshuu_count":  agg["manshuu_count"],
        "missed_summary": missed_summary,
    }


def format_yesterday_oneliner(summary: dict) -> str:
    """
    朝投稿に差し込む「昨日の答え合わせ」1〜3行テキストを返す。
    例:
      📌 昨日(06/27)の答え合わせ
      危険艇 10件中7件がイン逃げ失敗（70%）
      詳しい結果は21時のAIレポートで公開します📊
    """
    if not summary:
        return ""
    flew  = summary["boat1_flew"]
    total = summary["boat1_total"]
    rate  = summary["boat1_flew_rate"]
    date  = summary["date_disp"]
    lines = [
        f"📌 昨日({date})の答え合わせ",
        f"危険艇 {total}件中{flew}件がイン逃げ失敗（{rate:.0f}%）",
    ]
    if summary["manshuu_count"] > 0:
        lines.append(f"万舟も{summary['manshuu_count']}件発生！")
    lines.append("詳しい結果は21時のAIレポートで公開します📊")
    return "\n".join(lines)


def format_missed_detail(summary: dict) -> str:
    """
    外れレースの詳細テキスト（21時検証投稿用）
    """
    if not summary or not summary.get("missed_summary"):
        return ""
    lines = ["── 外れたレース ──"]
    for m in summary["missed_summary"]:
        lines.append(f"❌ {m}")
    lines.append("（次回の精度向上に活用します）")
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="AI予測 検証レポート送信")
    parser.add_argument("--date",    help="対象日 YYYYMMDD（省略時は今日）")
    parser.add_argument("--dry-run", action="store_true", help="送信せず表示のみ")
    args = parser.parse_args()

    date_str = args.date or _today_jst()
    records  = load_today_records(date_str)
    agg      = aggregate(records)

    log.info(
        "[集計] 通知%d件 的中%d件 損益%+d円 万舟%d件",
        agg["total_notified"], agg["total_hit"],
        agg["total_profit"],   agg["manshuu_count"],
    )

    ok = send_verification(agg, date_str, dry_run=args.dry_run, records=records)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
