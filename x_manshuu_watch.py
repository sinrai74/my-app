#!/usr/bin/env python3
"""
x_manshuu_watch.py  ── 万舟警報 イベント駆動監視スクリプト

「締切45分以内 かつ 万舟指数65以上」のレースを検知したらメール送信する。
GitHub Actions から1分ごとに呼ばれる（notify_arashi.yml の run_on_schedule と同居）。

Usage:
    python x_manshuu_watch.py                    # 監視 + 該当レースがあればメール送信
    python x_manshuu_watch.py --dry-run          # 送信せず内容を表示
    python x_manshuu_watch.py --threshold 60     # 指数閾値を変更（デフォルト65）
    python x_manshuu_watch.py --minutes 45       # 締切までの分数を変更（デフォルト45）
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

from notify_arashi import (
    VENUE_NAMES,
    _safe_get,
    fetch_programs,
    fetch_previews,
    _extract_boats_from_program,
    _apply_preview_to_boats,
    BoatInfo,
    WeatherInfo,
)
from x_ranking import (
    calc_upset_index,
    _upset_reasons,
    _score_to_rank,
    _score_to_rank_short,
)

log = logging.getLogger("x_manshuu_watch")

JST = timezone(timedelta(hours=9))

# ── 設定 ──────────────────────────────────────────────────────
DEFAULT_THRESHOLD = 65    # 万舟指数の最低ライン
DEFAULT_MINUTES   = 45    # 締切までの猶予（分）
SENT_FILE         = "manshuu_sent.json"   # 送信済みレースの記録
MAIL_TO           = "bigkirinuki@gmail.com"


# ════════════════════════════════════════════════════════════
# 締切時刻チェック
# ════════════════════════════════════════════════════════════

def _minutes_to_close(closed_at_str: str) -> Optional[float]:
    """締切まで何分か返す。過去または不明の場合は None"""
    if not closed_at_str:
        return None
    now = datetime.now(JST).replace(tzinfo=None)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            closed_dt = datetime.strptime(closed_at_str, fmt)
            diff = (closed_dt - now).total_seconds() / 60
            return diff if diff > 0 else None
        except ValueError:
            continue
    return None


# ════════════════════════════════════════════════════════════
# 送信済み管理
# ════════════════════════════════════════════════════════════

def _load_sent() -> set[str]:
    """当日送信済みレースのキーセットを返す"""
    today = datetime.now(JST).strftime("%Y%m%d")
    if not os.path.exists(SENT_FILE):
        return set()
    try:
        with open(SENT_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get(today, []))
    except (json.JSONDecodeError, OSError):
        return set()


def _save_sent(keys: list[str]) -> None:
    """送信済みキーを追記保存（当日分のみ保持）"""
    today = datetime.now(JST).strftime("%Y%m%d")
    existing: dict = {}
    if os.path.exists(SENT_FILE):
        try:
            with open(SENT_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # 当日分だけ保持（前日以前は削除）
    existing = {k: v for k, v in existing.items() if k >= today}
    existing.setdefault(today, [])
    for k in keys:
        if k not in existing[today]:
            existing[today].append(k)

    with open(SENT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


# ════════════════════════════════════════════════════════════
# 対象レース検出
# ════════════════════════════════════════════════════════════

def scan_races(
    threshold: int = DEFAULT_THRESHOLD,
    minutes: int   = DEFAULT_MINUTES,
    race_date: Optional[str] = None,
) -> list[dict]:
    """
    締切まで `minutes` 分以内 かつ 万舟指数 >= threshold のレースを返す。
    送信済みは除外する。
    """
    if not race_date:
        race_date = datetime.now(JST).strftime("%Y%m%d")

    programs = fetch_programs(race_date)
    previews = fetch_previews(race_date)

    if not programs:
        log.warning("[監視] 出走表なし: %s", race_date)
        return []

    preview_map: dict[tuple, dict] = {}
    for pv in previews:
        key = (pv.get("race_stadium_number"), pv.get("race_number"))
        preview_map[key] = pv

    sent = _load_sent()
    results: list[dict] = []

    for prog in programs:
        vn  = prog.get("race_stadium_number")
        rno = prog.get("race_number")
        if vn is None or rno is None:
            continue
        vn, rno = int(vn), int(rno)

        # 送信済みスキップ
        key = f"{vn}_{rno}"
        if key in sent:
            continue

        # 締切時刻チェック
        closed_at = prog.get("race_closed_at", "")
        mins = _minutes_to_close(closed_at)
        if mins is None or mins > minutes:
            continue   # 締切まだ先 or 既に終了

        venue_name = VENUE_NAMES.get(vn, f"場{vn}")
        boats = _extract_boats_from_program(prog)
        if not boats:
            continue

        preview = preview_map.get((vn, rno))
        weather: Optional[WeatherInfo] = None
        if preview:
            weather = _apply_preview_to_boats(boats, preview)

        boat1 = next((b for b in boats if b.lane == 1), None)
        score = calc_upset_index(boats, weather)

        if score < threshold:
            continue

        reasons = _upset_reasons(boats, boat1)

        results.append({
            "venue":      venue_name,
            "venue_num":  vn,
            "race":       rno,
            "score":      score,
            "rank":       _score_to_rank(score),
            "rank_short": _score_to_rank_short(score),
            "reasons":    reasons,
            "minutes":    round(mins, 1),
            "closed_at":  closed_at,
            "key":        key,
        })
        log.info("[検知] %s %dR 指数%d 締切%.0f分前", venue_name, rno, score, mins)

    # スコア降順
    results.sort(key=lambda x: -x["score"])
    return results


# ════════════════════════════════════════════════════════════
# メール送信
# ════════════════════════════════════════════════════════════

def _build_mail_body(races: list[dict]) -> str:
    now_str = datetime.now(JST).strftime("%H:%M")
    lines = [
        f"🚨 【{now_str}】万舟警報 {len(races)}件 🚨",
        "締切45分以内・万舟指数65以上のレースを検知しました",
        "",
    ]
    for r in races:
        lines += [
            f"{'='*40}",
            f"【{r['rank']}】{r['venue']}{r['race']}R",
            f"締切まで約{r['minutes']:.0f}分",
            "",
        ]
        for reason in r["reasons"]:
            lines.append(reason)
        lines.append("")

    lines += [
        "="*40,
        "▶ X投稿用テキスト（コピペ用）",
        "",
    ]
    for r in races:
        lines += [
            f"🚨 {r['venue']}{r['race']}R 万舟警報！",
            f"{r['rank']}",
        ]
        for reason in r["reasons"]:
            lines.append(reason)
        lines += [
            "",
            "締切前に要チェック💰",
            "#競艇 #ボートレース #万舟 #穴予想",
            "",
            "---",
            "",
        ]
    return "\n".join(lines)


def send_alert(races: list[dict], dry_run: bool = False) -> bool:
    """検知レースをメールで送信する"""
    now_str = datetime.now(JST).strftime("%m/%d %H:%M")
    subject = f"🚨 万舟警報 {len(races)}件 [{now_str}]"
    body    = _build_mail_body(races)

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

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="万舟警報 イベント駆動監視")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD,
                        help=f"万舟指数の閾値（デフォルト: {DEFAULT_THRESHOLD}）")
    parser.add_argument("--minutes",   type=int, default=DEFAULT_MINUTES,
                        help=f"締切までの分数（デフォルト: {DEFAULT_MINUTES}）")
    parser.add_argument("--date",      help="対象日 YYYYMMDD（省略時は今日）")
    parser.add_argument("--dry-run",   action="store_true", help="送信せず表示のみ")
    args = parser.parse_args()

    races = scan_races(
        threshold = args.threshold,
        minutes   = args.minutes,
        race_date = args.date,
    )

    if not races:
        log.info("[監視] 該当レースなし（指数%d以上 締切%d分以内）",
                 args.threshold, args.minutes)
        sys.exit(0)

    log.info("[監視] %d件検知", len(races))
    ok = send_alert(races, dry_run=args.dry_run)

    if ok and not args.dry_run:
        _save_sent([r["key"] for r in races])
        log.info("[記録] 送信済みに追加: %s", [r["key"] for r in races])

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
