#!/usr/bin/env python3
"""
x_weekly_report.py  ── 週次AIレポート画像生成＋メール送信

毎週日曜21時に hit_record.csv から過去7日分を集計し、
プロフィール固定用の画像とXコピペ用テキストをメール送信する。

Usage:
    python x_weekly_report.py              # 今週（過去7日）
    python x_weekly_report.py --dry-run    # 送信せず表示のみ
    python x_weekly_report.py --weeks 2    # 過去14日
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email.mime.base import MIMEBase
from email import encoders
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

log = logging.getLogger("x_weekly")

JST      = timezone(timedelta(hours=9))
MAIL_TO  = "bigkirinuki@gmail.com"
HIT_CSV  = "hit_record.csv"
MANSHUU_PAYOUT = 30000


# ════════════════════════════════════════════════════════════
# データ集計
# ════════════════════════════════════════════════════════════

def _safe_float(val, d=0.0):
    try: return float(val) if val not in (None,"","None") else d
    except: return d

def _safe_int(val, d=0):
    try: return int(float(val)) if val not in (None,"","None") else d
    except: return d


def load_week_records(weeks: int = 1) -> list[dict]:
    """過去 weeks*7 日分のレコードを返す"""
    if not os.path.exists(HIT_CSV):
        return []
    now   = datetime.now(JST)
    since = (now - timedelta(days=weeks * 7)).strftime("%Y%m%d")
    records = []
    with open(HIT_CSV, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row_date = row.get("date", "").replace("-", "")
            if row_date >= since:
                records.append(row)
    log.info("[週次] %d件読み込み（%s以降）", len(records), since)
    return records


def aggregate_weekly(records: list[dict]) -> dict:
    """週次集計"""
    notified = [r for r in records if r.get("pred_combo", "")]
    if not notified:
        return {
            "total": 0, "hit": 0, "hit_rate": 0.0,
            "cost": 0, "profit": 0, "roi": 0.0,
            "boat1_flew": 0, "boat1_total": 0, "boat1_flew_rate": 0.0,
            "manshuu_hit": 0, "manshuu_total": 0, "manshuu_rate": 0.0,
            "best_payout": 0, "best_race": "",
            "daily": {},
        }

    total   = len(notified)
    hit     = sum(1 for r in notified if _safe_int(r.get("hit")) == 1)
    cost    = sum(_safe_float(r.get("cost"))   for r in notified)
    profit  = sum(_safe_float(r.get("profit")) for r in notified)
    roi     = (profit / cost * 100) if cost > 0 else 0.0

    # 1号艇飛び率
    b1_total = b1_flew = 0
    for r in notified:
        res = r.get("result_combo", "")
        if res and "-" in res:
            b1_total += 1
            if res.split("-")[0].strip() != "1":
                b1_flew += 1
    b1_rate = (b1_flew / b1_total * 100) if b1_total > 0 else 0.0

    # 万舟ヒット（3万以上のレースが警報対象だったか）
    manshuu_total = sum(1 for r in records if _safe_float(r.get("payout")) >= MANSHUU_PAYOUT)
    manshuu_hit   = sum(1 for r in notified
                        if _safe_int(r.get("hit")) == 1
                        and _safe_float(r.get("payout")) >= MANSHUU_PAYOUT)
    m_rate = (manshuu_hit / manshuu_total * 100) if manshuu_total > 0 else 0.0

    # 最高払戻
    hit_records = [r for r in notified if _safe_int(r.get("hit")) == 1]
    best_payout = 0
    best_race   = ""
    if hit_records:
        best = max(hit_records, key=lambda r: _safe_float(r.get("payout")))
        best_payout = int(_safe_float(best.get("payout")))
        best_race   = f"{best.get('venue','')}{best.get('race','')}R {best.get('result_combo','')}"

    # 日別成績
    daily: dict[str, dict] = {}
    for r in notified:
        d = r.get("date", "").replace("-", "")[:8]
        if d not in daily:
            daily[d] = {"hit": 0, "total": 0, "profit": 0}
        daily[d]["total"]  += 1
        daily[d]["hit"]    += _safe_int(r.get("hit"))
        daily[d]["profit"] += _safe_float(r.get("profit"))

    return {
        "total": total, "hit": hit, "hit_rate": round(hit / total * 100, 1),
        "cost": int(cost), "profit": int(profit), "roi": round(roi, 1),
        "boat1_flew": b1_flew, "boat1_total": b1_total,
        "boat1_flew_rate": round(b1_rate, 1),
        "manshuu_hit": manshuu_hit, "manshuu_total": manshuu_total,
        "manshuu_rate": round(m_rate, 1),
        "best_payout": best_payout, "best_race": best_race,
        "daily": daily,
    }


# ════════════════════════════════════════════════════════════
# 画像生成
# ════════════════════════════════════════════════════════════

def generate_weekly_image(agg: dict, output_path: str, weeks: int = 1) -> None:
    from PIL import Image, ImageDraw

    W, H = 1200, 800
    C_BG     = (18, 18, 18)
    C_HEADER = (26, 35, 126)
    C_WHITE  = (255, 255, 255)
    C_GRAY   = (160, 160, 160)
    C_GREEN  = (76, 175, 80)
    C_RED    = (244, 67, 54)
    C_YELLOW = (255, 193, 7)
    C_ACCENT = (255, 152, 0)
    C_CYAN   = (0, 188, 212)
    C_SECTION= (35, 35, 60)

    def get_font(size, bold=False):
        from PIL import ImageFont
        candidates = (
            ["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
             "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
             "C:/Windows/Fonts/meiryob.ttc"]
            if bold else
            ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
             "C:/Windows/Fonts/meiryo.ttc"]
        )
        for p in candidates:
            if os.path.exists(p):
                try: return ImageFont.truetype(p, size)
                except: pass
        return ImageFont.load_default()

    def text_center(draw, text, y, font, color, width=W):
        try: tw = font.getbbox(text)[2] - font.getbbox(text)[0]
        except: tw = len(text) * (font.size if hasattr(font, 'size') else 12)
        draw.text((int((width - tw) / 2), y), text, font=font, fill=color)

    img  = Image.new("RGB", (W, H), C_BG)
    draw = ImageDraw.Draw(img)

    fhd  = get_font(38, bold=True)
    fsub = get_font(22)
    fbig = get_font(52, bold=True)
    fmd  = get_font(26, bold=True)
    fsm  = get_font(20)
    fxs  = get_font(17)

    # ヘッダー
    draw.rectangle([0, 0, W, 90], fill=C_HEADER)
    draw.rectangle([0, 0, 6, 90], fill=C_ACCENT)
    now = datetime.now(JST)
    since_date = (now - timedelta(days=weeks*7)).strftime("%m/%d")
    to_date    = now.strftime("%m/%d")
    text_center(draw, f"AI週間レポート  {since_date}〜{to_date}", 18, fhd, C_WHITE)
    text_center(draw, f"過去{weeks*7}日間の予測精度レポート", 58, fsub, C_GRAY)

    # ══ 3列グリッドレイアウト（座標を列ごとに完全独立） ══
    COL  = W // 3          # 各列の幅 = 400px
    TOP  = 105             # グリッド開始y
    ROW1 = 240             # 上段の高さ
    ROW2 = 120             # 下段の高さ
    GAP  = 8               # セル間隔

    def cell(x, y, w, h, title):
        """セル背景＋左アクセントバー＋タイトル描画"""
        draw.rectangle([x+GAP, y, x+w-GAP, y+h], fill=C_SECTION)
        draw.rectangle([x+GAP, y, x+GAP+4, y+h], fill=C_ACCENT)
        draw.text((x+GAP+14, y+8), title, font=fsm, fill=C_GRAY)

    def tc(x, y, w, text, font, color):
        """列内センタリングテキスト"""
        try: tw = font.getbbox(text)[2] - font.getbbox(text)[0]
        except: tw = len(text) * 14
        draw.text((x + int((w - tw) / 2), y), text, font=font, fill=color)

    hit_color  = C_GREEN if agg["hit_rate"]  >= 60 else C_YELLOW if agg["hit_rate"]  >= 40 else C_RED
    flew_color = C_GREEN if agg["boat1_flew_rate"] >= 60 else C_YELLOW
    profit     = agg["profit"]
    sign       = "+" if profit >= 0 else ""
    p_color    = C_GREEN if profit >= 0 else C_RED
    roi_color  = C_GREEN if agg["roi"] >= 0 else C_RED
    m_color    = C_GREEN if agg["manshuu_rate"] >= 50 else C_YELLOW if agg["manshuu_rate"] >= 30 else C_RED

    # ── 列0 上段: 危険な1号艇 ────────────────────────────
    X, Y, CW = 0, TOP, COL
    cell(X, Y, CW, ROW1, "危険な1号艇")
    tc(X, Y+42,  CW, str(agg["total"]),       fbig, C_WHITE)
    tc(X, Y+96,  CW, "抽出レース数",           fsm,  C_GRAY)
    tc(X, Y+122, CW, f"{agg['hit']}件",        fmd,  C_WHITE)
    tc(X, Y+154, CW, "的中",                   fsm,  C_GRAY)
    tc(X, Y+176, CW, f"{agg['hit_rate']}%",    fbig, hit_color)
    tc(X, Y+228, CW, "的中率",                 fsm,  C_GRAY)

    # ── 列0 下段: イン逃げ失敗率 ─────────────────────────
    Y2 = TOP + ROW1 + GAP
    cell(X, Y2, CW, ROW2, "イン逃げ失敗率")
    tc(X, Y2+38, CW, f"{agg['boat1_flew_rate']}%", fbig, flew_color)
    tc(X, Y2+92, CW, f"{agg['boat1_flew']}件 / {agg['boat1_total']}件", fsm, C_GRAY)

    # ── 列1 上段: 収支 ────────────────────────────────────
    X = COL; Y = TOP
    cell(X, Y, CW, ROW1, "収支")
    tc(X, Y+42,  CW, f"{sign}{profit:,}円",       fbig, p_color)
    tc(X, Y+96,  CW, "週間損益",                  fsm,  C_GRAY)
    tc(X, Y+122, CW, f"ROI {sign}{agg['roi']:.1f}%", fmd, roi_color)
    tc(X, Y+158, CW, f"投資 {agg['cost']:,}円",   fsm,  C_WHITE)
    tc(X, Y+186, CW, f"回収 {agg['cost']+profit:,}円", fsm, C_WHITE)

    # ── 列1 下段: 週間最高払戻 ───────────────────────────
    Y2 = TOP + ROW1 + GAP
    cell(X, Y2, CW, ROW2, "週間最高払戻")
    tc(X, Y2+38, CW, f"{agg['best_payout']:,}円", fmd, C_YELLOW)
    br = agg["best_race"]
    if len(br) > 16: br = br[:14] + "…"
    tc(X, Y2+78, CW, br, fxs, C_GRAY)

    # ── 列2 上段: 万舟警報 ───────────────────────────────
    X = COL*2; Y = TOP
    cell(X, Y, CW, ROW1, "万舟警報")
    tc(X, Y+42,  CW, str(agg["manshuu_total"]),      fbig, C_WHITE)
    tc(X, Y+96,  CW, "万舟発生レース数",             fsm,  C_GRAY)
    tc(X, Y+122, CW, f"{agg['manshuu_hit']}件 的中", fmd,  C_WHITE)
    tc(X, Y+176, CW, f"{agg['manshuu_rate']:.0f}%",  fbig, m_color)
    tc(X, Y+228, CW, "ヒット率",                     fsm,  C_GRAY)

    # ── 列2 下段: 日別成績バー ───────────────────────────
    Y2 = TOP + ROW1 + GAP
    cell(X, Y2, CW, ROW2, "日別成績")
    daily = agg["daily"]
    if daily:
        keys = sorted(daily.keys())[-7:]
        n    = len(keys)
        bw   = (CW - GAP*2 - 16) // max(n, 1)
        bx   = X + GAP + 8
        base = Y2 + ROW2 - 20
        for dk in keys:
            d    = daily[dk]
            rate = (d["hit"] / d["total"] * 100) if d["total"] > 0 else 0
            bh   = max(4, int(rate / 100 * (ROW2 - 50)))
            bc   = C_GREEN if rate >= 60 else C_YELLOW if rate >= 40 else C_RED
            draw.rectangle([bx, base-bh, bx+bw-4, base], fill=bc)
            draw.text((bx, base+2), dk[6:], font=fxs, fill=C_GRAY)
            bx  += bw

    # フッター
    footer_y = H - 45
    draw.rectangle([0, footer_y, W, H], fill=(35, 35, 35))
    text_center(draw,
        "AIが全レースを毎日分析 | 競艇荒れ検知Bot",
        footer_y + 12, fxs, C_GRAY)

    img.save(output_path)
    log.info("[週次] 画像保存: %s", output_path)


# ════════════════════════════════════════════════════════════
# テキスト生成
# ════════════════════════════════════════════════════════════

def build_weekly_tweet(agg: dict, weeks: int = 1) -> str:
    now        = datetime.now(JST)
    since_date = (now - timedelta(days=weeks*7)).strftime("%m/%d")
    to_date    = now.strftime("%m/%d")
    profit     = agg["profit"]
    sign       = "+" if profit >= 0 else ""
    p_emoji    = "✅" if profit >= 0 else "❌"

    lines = [
        f"📊【AI週間レポート {since_date}〜{to_date}】",
        "",
        "━━ 危険な1号艇 ━━",
        f"抽出: {agg['total']}レース",
        f"的中: {agg['hit']}件 / 的中率: {agg['hit_rate']}%",
        f"イン逃げ失敗: {agg['boat1_flew_rate']}%",
        "",
        "━━ 万舟警報 ━━",
        f"万舟発生: {agg['manshuu_total']}件",
        f"ヒット率: {agg['manshuu_rate']:.0f}%",
        "",
        "━━ 収支 ━━",
        f"{p_emoji} {sign}{profit:,}円 （ROI {sign}{agg['roi']:.1f}%）",
    ]

    if agg["best_payout"] > 0:
        lines += ["", f"🏆 週間最高: {agg['best_payout']:,}円", f"   {agg['best_race']}"]

    lines += [
        "",
        "毎日8時・11時・21時に投稿中📲",
        "このAIの精度はどう思いますか？💬",
        "#競艇 #ボートレース #AI予想 #競艇予想 #週間レポート",
    ]
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# メール送信
# ════════════════════════════════════════════════════════════

def send_weekly(agg: dict, image_path: str, weeks: int = 1,
                dry_run: bool = False) -> bool:
    now       = datetime.now(JST)
    to_date   = now.strftime("%m/%d")
    tweet     = build_weekly_tweet(agg, weeks)
    subject   = f"📊 AI週間レポート {to_date} 的中率{agg['hit_rate']}% 損益{agg['profit']:+,}円"
    body      = tweet + "\n\n" + "="*50 + "\n▶ X投稿用テキスト（コピペ用）\n\n" + tweet

    if dry_run:
        print("="*60)
        print(f"[DRY RUN] 件名: {subject}")
        print(body)
        print("="*60)
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

    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            part = MIMEBase("image", "png")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", "attachment",
                        filename=os.path.basename(image_path))
        msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo(); smtp.starttls()
            smtp.login(gmail_address, app_password)
            smtp.sendmail(gmail_address, MAIL_TO, msg.as_string())
        log.info("[週次] 送信成功: %s", subject)
        return True
    except smtplib.SMTPException as e:
        log.error("[週次] 送信失敗: %s", e)
        return False


# ════════════════════════════════════════════════════════════
# エントリポイント
# ════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        stream=sys.stdout)

    parser = argparse.ArgumentParser(description="AI週次レポート")
    parser.add_argument("--weeks",   type=int, default=1, help="集計週数（デフォルト1）")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output",  default="weekly_report.png")
    args = parser.parse_args()

    records = load_week_records(args.weeks)
    agg     = aggregate_weekly(records)

    log.info("[集計] 通知%d件 的中%d件 損益%+d円", agg["total"], agg["hit"], agg["profit"])

    try:
        generate_weekly_image(agg, args.output, weeks=args.weeks)
    except Exception as e:
        log.warning("[週次] 画像生成失敗: %s", e)

    ok = send_weekly(agg, args.output, weeks=args.weeks, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
