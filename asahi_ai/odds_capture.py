"""オッズの記録（公開リポジトリの Actions で動かす）。予測には使わない。Project 19。

対象：その日の予測（データリポジトリの predictions/）で、1号艇危険の上位20と荒れの上位20を合わせたレース。
各レースの締切の 10 分前と 3 分前に、公式の3連単オッズのページを1件ずつ取得する（取得の間隔は3秒以上）。
保存先：データリポジトリ（非公開）の odds/<年>/<日>/<race_id>_m10.json・_m3.json。取得ごとに commit・push する。

公開リポジトリのログには、レースの番号・予想の確率・買い目を出さない（件数と時刻だけ）。

  python -m asahi_ai.odds_capture --data DATA_DIR [--date YYYY-MM-DD] [--stop-after-min 330]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import time
from pathlib import Path

from . import odds

JST = dt.timezone(dt.timedelta(hours=9))
SNAPSHOTS = (("m10", 10), ("m3", 3))
TOP_N = 20
INTERVAL = 3.0


def targets(pred_path: Path, top_n: int = TOP_N) -> list[dict]:
    rows = [json.loads(l) for l in pred_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["danger"] = 1 - r["p_win"][0]
        r["upset"] = 1 - sum(sorted(r["tri"])[-20:])
    d = {r["race_id"] for r in sorted(rows, key=lambda r: -r["danger"])[:top_n]}
    u = {r["race_id"] for r in sorted(rows, key=lambda r: -r["upset"])[:top_n]}
    out = []
    for r in rows:
        if r["race_id"] in d | u and r.get("race_closed_at"):
            out.append({"race_id": r["race_id"], "closed_at": r["race_closed_at"],
                        "flags": sorted((["danger"] if r["race_id"] in d else []) + (["upset"] if r["race_id"] in u else []))})
    return sorted(out, key=lambda x: x["closed_at"])


def _git(data: Path, *args):
    subprocess.run(["git", "-C", str(data), *args], check=True, capture_output=True)


def _save_and_push(data: Path, path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    _git(data, "add", str(path.relative_to(data)))
    _git(data, "commit", "-q", "-m", f"odds: {path.name[:2]}…")
    for attempt in range(3):
        try:
            _git(data, "pull", "-q", "--rebase", "origin", "main")
            _git(data, "push", "-q", "origin", "HEAD:main")
            return
        except subprocess.CalledProcessError:
            time.sleep(5)
    raise RuntimeError("push に失敗")


def run(data: Path, day: dt.date, stop_after_min: float, sleep=time.sleep, now=lambda: dt.datetime.now(JST)) -> dict:
    t_start = now()
    pred = data / "predictions" / str(day.year) / f"{day.isoformat()}.jsonl"
    if not pred.exists():
        return {"status": "no_predictions"}
    tg = targets(pred)
    jobs = []
    for t in tg:
        closed = dt.datetime.fromisoformat(t["closed_at"]).replace(tzinfo=JST)
        for label, mins in SNAPSHOTS:
            jobs.append((closed - dt.timedelta(minutes=mins), label, t, closed))
    jobs.sort(key=lambda x: x[0])
    stats = {"targets": len(tg), "planned": len(jobs), "captured": 0, "parsed_120": 0, "failed": 0,
             "skipped_past": 0, "remaining": 0}
    last = 0.0
    for i, (at, label, t, closed) in enumerate(jobs):
        out = data / "odds" / str(day.year) / day.isoformat() / f"{t['race_id']}_{label}.json"
        if out.exists():
            continue
        if (now() - t_start).total_seconds() / 60 > stop_after_min:
            stats["remaining"] = sum(1 for j in jobs[i:] if not (data / "odds" / str(day.year) / day.isoformat()
                                                                   / f"{j[2]['race_id']}_{j[1]}.json").exists())
            break
        wait = (at - now()).total_seconds()
        if wait > 0:
            sleep(wait)
        if now() >= closed:
            stats["skipped_past"] += 1
            continue
        gap = INTERVAL - (time.time() - last)
        if gap > 0:
            sleep(gap)
        rid = t["race_id"]
        url = odds.url_for(int(rid[10:12]), rid[8:10], rid[:8])
        status, html, err = odds.fetch_html(url)
        last = time.time()
        captured_at = now().isoformat(timespec="seconds")
        rec = {"race_id": rid, "snapshot": label, "captured_at": captured_at, "race_closed_at": t["closed_at"],
               "minutes_before": round((closed - now()).total_seconds() / 60, 2), "flags": t["flags"],
               "source": "boatrace.jp odds3t", "http_status": status, "error": err,
               "html_sha256": hashlib.sha256(html).hexdigest() if html else None,
               "source_updated_text": odds.page_update_text(html) if html else None,
               "odds": None, "n_odds": 0, "capture_status": "fetch_failed"}
        if html:
            try:
                o = odds.parse_trifecta(html)
                rec.update(odds=o, n_odds=len(o), capture_status="ok")
                stats["parsed_120"] += len(o) == 120
            except odds.OddsTableError as e:
                rec.update(capture_status="parse_failed", error=str(e))
        stats["captured"] += rec["capture_status"] == "ok"
        stats["failed"] += rec["capture_status"] != "ok"
        _save_and_push(data, out, rec)
        print(f"ODDS {i + 1}/{len(jobs)} {label} status={rec['capture_status']} n={rec['n_odds']} "
              f"before={rec['minutes_before']}min", flush=True)
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--date")
    ap.add_argument("--stop-after-min", type=float, default=330)
    a = ap.parse_args(argv)
    day = dt.date.fromisoformat(a.date) if a.date else dt.datetime.now(JST).date()
    r = run(Path(a.data), day, a.stop_after_min)
    print("ODDS_DONE " + json.dumps(r), flush=True)
    with open("odds_result.json", "w") as f:
        json.dump(r, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
