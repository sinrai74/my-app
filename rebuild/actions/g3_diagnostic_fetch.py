"""
正式G3（B案）段階3：診断の入力を取得する段（ネットワークを使う。Legacy の評価は呼ばない）。

- 出走表：formal_g2.programs.fix_programs（G2 と同じ取得。読み込んで使うだけで、G2 のファイルは変えない）。
- オッズ：odds_fetch.fetch_odds（3連単。取得した時点の値）。オッズの時点の扱いは未決定なので、
  取得の時刻と種類（kind）を記録するだけにする。
- 取得の段では、トークン・認証情報らしい環境変数があれば止まる（Release に触れうる状態で動かさない）。
- 出力：programs_raw.json（取得した生のバイト列）、odds.json、fetch_manifest.json（SHA-256）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

JST = timezone(timedelta(hours=9))
ODDS_KIND = "boatrace_jp_odds3t_as_published_at_fetch_time"


def sensitive_env_names() -> list[str]:
    """トークン・秘密・パスワードらしい名前（C-07：境界 shadow.g3_isolation と同じ名前の型＋既知の名前。
    GitHub が既定で置く GITHUB_SHA・RUNNER_OS などの名前は対象にしない）。"""
    from shadow.g3_isolation import env_name_must_refuse

    return sorted(k for k in os.environ if env_name_must_refuse(k))


def fetch(race_date: str, out: Path, *, fix_programs: Optional[Callable[[str], dict]] = None,
          fetch_odds: Optional[Callable[..., dict]] = None, only: Sequence[str] = (),
          sleep_sec: float = 0.5) -> dict[str, Any]:
    from actions.g3_diagnostic_entry import list_races

    if fix_programs is None:
        from formal_g2.programs import fix_programs as _fp

        fix_programs = _fp
    if fetch_odds is None:
        from odds_fetch import fetch_odds as _fo

        fetch_odds = _fo
    out.mkdir(parents=True, exist_ok=True)
    fixed = fix_programs(race_date)
    (out / "programs_raw.json").write_bytes(fixed["raw_bytes"])
    raw = json.loads(fixed["raw_bytes"])
    programs = raw["programs"] if isinstance(raw, dict) else raw
    races = list_races(race_date, programs)
    if only:
        races = [r for r in races if r["eval_id"] in set(only)]
    odds: dict[str, Any] = {}
    for r in races:
        fetched_at = datetime.now(JST).isoformat()
        try:
            o = fetch_odds(r["race_number"], str(r["venue_num"]).zfill(2), race_date)
            err = None if o else "empty"
        except Exception:  # noqa: BLE001 — G-21：例外クラス名を公開レコードに残さない
            o, err = {}, "unexpected_error"
        odds[r["eval_id"]] = {"fetched_at": fetched_at, "kind": ODDS_KIND, "count": len(o or {}),
                              "error": err, "race_closed_at": r.get("race_closed_at"), "odds": o or {}}
        if sleep_sec:
            time.sleep(sleep_sec)
    odds_doc = {"race_date": race_date, "races": odds}
    (out / "odds.json").write_text(json.dumps(odds_doc, ensure_ascii=False), encoding="utf-8")
    manifest = {
        "race_date": race_date,
        "programs_url": fixed.get("url"),
        "programs_status_code": fixed.get("status_code"),
        "programs_raw_sha256": hashlib.sha256(fixed["raw_bytes"]).hexdigest(),
        "programs_targets_sha256": fixed.get("targets_sha256"),
        "races": len(races),
        "odds_sha256": hashlib.sha256((out / "odds.json").read_bytes()).hexdigest(),
        "odds_kind": ODDS_KIND,
        "odds_errors": sum(1 for v in odds.values() if v["error"]),
        "fetched_at_first": min((v["fetched_at"] for v in odds.values()), default=None),
        "fetched_at_last": max((v["fetched_at"] for v in odds.values()), default=None),
    }
    (out / "fetch_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="fetch inputs for the G3 diagnostic")
    ap.add_argument("--race-date", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    names = sensitive_env_names()
    if names:
        print(f"sensitive environment variables are present ({len(names)}); refusing to run")
        return 1
    m = fetch(a.race_date, a.out.resolve(), only=[x for x in a.only.split(",") if x])
    print(json.dumps({k: m[k] for k in ("race_date", "races", "odds_errors", "programs_raw_sha256")}))
    return 0


if __name__ == "__main__":
    # F-10：ログ・print・例外の表示を含め、標準出力・標準エラーに出る文字を伏せ字にする（プロセスとして動くときだけ）
    from shadow.g3_isolation import install_redacting_streams

    install_redacting_streams()
    sys.exit(main())
