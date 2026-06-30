#!/usr/bin/env python3
"""
x_improvement_log.py  ── ⑯ AI改善ログ管理

Version / 改善内容 / 変更理由 / 改善後成功率 を improvement_log.json に
全履歴保存し、実績ページ・note新聞から参照できるようにする。

Usage（Pythonから直接呼び出し、または手動でログ追加）:
    from x_improvement_log import add_log_entry, load_log

    add_log_entry(
        version="v3.1",
        content="危険艇スコアにモーター加点を追加",
        reason="モーター由来の外れが多かったため",
        success_rate_before=48.2,
        success_rate_after=55.6,
    )
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger("x_improvement_log")

JST      = timezone(timedelta(hours=9))
LOG_FILE = "improvement_log.json"


def load_log() -> list[dict]:
    """改善ログ全履歴を読み込む（新しい順）"""
    if not os.path.exists(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries", [])
        return sorted(entries, key=lambda e: e.get("date", ""), reverse=True)
    except (json.JSONDecodeError, OSError) as e:
        log.warning("[改善ログ] 読み込み失敗: %s", e)
        return []


def add_log_entry(
    version: str,
    content: str,
    reason: str,
    success_rate_before: Optional[float] = None,
    success_rate_after: Optional[float] = None,
    brand: str = "",
) -> bool:
    """
    ⑯ 改善ログに1エントリを追加する。
    version:              バージョン番号（例: "v3.1"）
    content:               改善内容
    reason:                 変更理由
    success_rate_before:   変更前の成功率(%)
    success_rate_after:    変更後の成功率(%)
    brand:                  対象ブランド（"danger" 等、空文字なら全体）
    """
    entries = []
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries = data.get("entries", [])
        except (json.JSONDecodeError, OSError):
            entries = []

    entry = {
        "date":    datetime.now(JST).strftime("%Y-%m-%d"),
        "version": version,
        "brand":   brand,
        "content": content,
        "reason":  reason,
        "success_rate_before": success_rate_before,
        "success_rate_after":  success_rate_after,
    }
    entries.append(entry)

    try:
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump({"entries": entries}, f, ensure_ascii=False, indent=2)
        log.info("[改善ログ] 追加: %s %s", version, content)
        return True
    except OSError as e:
        log.error("[改善ログ] 保存失敗: %s", e)
        return False


def format_log_html(limit: int = 10) -> str:
    """⑯ 改善ログをHTML表示用に整形する（新聞・実績ページから呼び出す）"""
    entries = load_log()[:limit]
    if not entries:
        return '<p class="no-data">改善ログはまだありません</p>'

    rows = ""
    for e in entries:
        before = e.get("success_rate_before")
        after  = e.get("success_rate_after")
        rate_text = ""
        if before is not None and after is not None:
            diff = round(after - before, 1)
            sign = "+" if diff >= 0 else ""
            rate_text = f'<span class="log-rate">{before}% → {after}%（{sign}{diff}pt）</span>'

        rows += f"""
<div class="log-entry">
  <div class="log-header">
    <span class="log-version">{e.get('version','')}</span>
    <span class="log-date">{e.get('date','')}</span>
  </div>
  <div class="log-content">{e.get('content','')}</div>
  <div class="log-reason">理由: {e.get('reason','')}</div>
  {rate_text}
</div>"""
    return f'<div class="log-list">{rows}</div>'


def format_log_text(limit: int = 5) -> str:
    """⑯ 改善ログをテキスト形式で整形する（メール・X投稿用）"""
    entries = load_log()[:limit]
    if not entries:
        return "改善ログはまだありません。"

    lines = []
    for e in entries:
        before = e.get("success_rate_before")
        after  = e.get("success_rate_after")
        rate_txt = ""
        if before is not None and after is not None:
            diff = round(after - before, 1)
            sign = "+" if diff >= 0 else ""
            rate_txt = f"（{before}%→{after}% {sign}{diff}pt）"
        lines.append(f"[{e.get('version','')}] {e.get('content','')}{rate_txt}")
    return "\n".join(lines)
