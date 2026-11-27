"""オッズの記録（公開リポジトリの Actions で動かす）。予測には使わない。Project 19。

対象：その日の予測（データリポジトリの predictions/）で、1号艇危険の上位20と荒れの上位20を合わせたレース。
加えて、今の朝刊（AI新聞）の紙面に載ったレース（2026-11-27 本人の承認、Project 21）：危険艇の上位10・万舟警報の上位10・
AIランキングの上位10・一致指数が片方だけでAランク以上（27.0・29.6）のレース。AI新聞は朝刊の作成後に公開リポジトリの
ranking_cache.json・daily_stats.json に載るので、実行中に10分ごとに読み直して対象に足す（読むだけ。今の朝刊には触らない）。
各レースの締切の 10 分前と 3 分前に、公式の3連単オッズのページを1件ずつ取得する（取得の間隔は3秒以上）。
保存先：データリポジトリ（非公開）の odds/<年>/<日>/<race_id>_m10.json・_m3.json。取得ごとに commit・push する。
m3 のときは直前情報のページ（展示・気象）も1件取り、原本を beforeinfo/<年>/<日>/<race_id>_m3.html.gz に置く（直前のモデル用、Project 21）。

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
REFRESH_MIN = 10
BLOCK_AFTER = 5            # 拒否（403・429）がこの回数続いたら、その日の取得をやめる
BLOCK_CODES = (403, 429)
NEWS_BASE = "https://raw.githubusercontent.com/sinrai74/my-app/main/"
MI_POINT = 25.7   # x_brand_config.MATCH_INDEX_POINTS の危険艇・万舟（同じ値）


def _fetch_json(name: str):
    import urllib.request
    try:
        with urllib.request.urlopen(NEWS_BASE + name, timeout=20) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def news_races(day: dt.date, fetch=_fetch_json) -> dict[str, list[str]]:
    """AI新聞の紙面に載ったレース → 印。読めないとき・日付が違うときは空。"""
    ds = day.strftime("%Y%m%d")
    out: dict[str, set] = {}

    def add(vn, rno, flag):
        try:
            out.setdefault(f"{ds}{int(vn):02d}{int(rno):02d}", set()).add(flag)
        except (TypeError, ValueError):
            pass

    rc = fetch("ranking_cache.json")
    if isinstance(rc, dict) and str(rc.get("date")) == ds:
        for x in rc.get("danger_boat1", []) or []:
            add(x.get("venue_num"), x.get("race"), "news_danger")
        for x in rc.get("manshuu_alert", []) or []:
            add(x.get("venue_num"), x.get("race"), "news_manshuu")
        # 一致指数：危険艇・万舟の掲載ごとに 25.7 点、スコア 80 以上 ×1.15、60 以上 ×1.05（x_note_report._calc_match_index）
        brands: dict[tuple, dict] = {}
        for key, lst in (("danger", rc.get("all_danger", []) or []), ("manshuu", rc.get("all_manshuu", []) or [])):
            for x in lst:
                brands.setdefault((x.get("venue_num"), x.get("race")), {}).setdefault(key, x.get("score") or 0)
        for (vn, rno), b in brands.items():
            if len(b) == 1 and list(b.values())[0] >= 60:   # 片方だけで A 以上 → 27.0 または 29.6
                add(vn, rno, "news_mi27")
    st = fetch("daily_stats.json")
    if isinstance(st, dict) and isinstance(st.get(ds), dict):
        for x in (st[ds].get("ranking") or [])[:10]:
            add(x.get("venue_num"), x.get("race"), "news_rank")
    return {k: sorted(v) for k, v in out.items()}


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


def _push(data: Path, tries: int = 6, sleep=time.sleep) -> bool:
    """まだ送っていない commit を送る。ほかの実行と重なって失敗したら、間をあけてやり直す。"""
    for attempt in range(tries):
        try:
            _git(data, "pull", "-q", "--rebase", "origin", "main")
            _git(data, "push", "-q", "origin", "HEAD:main")
            return True
        except subprocess.CalledProcessError:
            subprocess.run(["git", "-C", str(data), "rebase", "--abort"], capture_output=True)
            sleep(min(5 * 2 ** attempt, 60))
    return False


BEFOREINFO = "https://www.boatrace.jp/owpc/pc/race/beforeinfo"


def _save_and_push(data: Path, path: Path, rec):
    """rec が bytes ならそのまま（直前情報のページの gzip）、dict なら JSON で保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rec, bytes):
        path.write_bytes(rec)
    else:
        path.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    _git(data, "add", str(path.relative_to(data)))
    _git(data, "commit", "-q", "-m", f"odds: {path.name[:2]}…")
    # 送れなくても止めない（commit は手元に残り、次に送れたときに一緒に送られる。最後にもう一度送る）
    if not _push(data):
        print("PUSH_DEFERRED", flush=True)


def run(data: Path, day: dt.date, stop_after_min: float, sleep=time.sleep, now=lambda: dt.datetime.now(JST),
        news=None) -> dict:
    t_start = now()
    pred = data / "predictions" / str(day.year) / f"{day.isoformat()}.jsonl"
    if not pred.exists():
        return {"status": "no_predictions"}
    blocked_mark = data / "odds" / str(day.year) / day.isoformat() / "_blocked.json"
    if blocked_mark.exists():   # 今日はすでに拒否されて止めた（続きの実行も取りにいかない）
        return {"status": "blocked_today", "remaining": 0}
    news = news or (lambda: news_races(day))
    closed_of = {}
    for l in pred.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            if r.get("race_closed_at"):
                closed_of[r["race_id"]] = r["race_closed_at"]
    tg = {t["race_id"]: t for t in targets(pred)}
    jobs: dict[tuple, tuple] = {}

    def merge_news():
        added = 0
        for rid, flags in news().items():
            if rid not in closed_of:
                continue
            if rid in tg:
                tg[rid]["flags"] = sorted(set(tg[rid]["flags"]) | set(flags))
            else:
                tg[rid] = {"race_id": rid, "closed_at": closed_of[rid], "flags": sorted(flags)}
                added += 1
        for t in tg.values():
            closed = dt.datetime.fromisoformat(t["closed_at"]).replace(tzinfo=JST)
            for label, mins in SNAPSHOTS:
                jobs.setdefault((t["race_id"], label), (closed - dt.timedelta(minutes=mins), label, t, closed))
        return added

    def out_of(j):
        return data / "odds" / str(day.year) / day.isoformat() / f"{j[2]['race_id']}_{j[1]}.json"

    stats = {"targets": 0, "news_added": merge_news(), "planned": 0, "captured": 0, "parsed_120": 0, "failed": 0,
             "skipped_past": 0, "remaining": 0}
    last_refresh = now()
    refused = 0
    done: set = set()
    last = 0.0
    limit = t_start + dt.timedelta(minutes=stop_after_min)
    while True:
        if now() - last_refresh >= dt.timedelta(minutes=REFRESH_MIN):
            stats["news_added"] += merge_news()
            last_refresh = now()
        pending = sorted((j for k, j in jobs.items() if k not in done and not out_of(j).exists()), key=lambda x: x[0])
        if not pending:
            break
        at, label, t, closed = pending[0]
        # 区切りの時刻より後に予定された取得は、待たずに次の実行へ回す（長く待つと実行の時間の上限を超えるため）
        if max(at, now()) > limit:
            stats["remaining"] = len(pending)
            break
        wait = (at - now()).total_seconds()
        if wait > 0:
            # 待つ間も AI新聞を読み直せるように、長い待ちは区切って寝る
            sleep(min(wait, REFRESH_MIN * 60))
            continue
        done.add((t["race_id"], label))
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
        refused = refused + 1 if status in BLOCK_CODES else 0
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
        _save_and_push(data, out_of((at, label, t, closed)), rec)
        if label == "m3" and status not in BLOCK_CODES:   # 直前情報（展示タイム・展示の進入と ST・気象）の原本。解析はあとで手元で行う（Project 21）
            bpath = data / "beforeinfo" / str(day.year) / day.isoformat() / f"{rid}_m3.html.gz"
            if not bpath.exists():
                sleep(max(0.0, INTERVAL - (time.time() - last)))
                bstatus, bhtml, _ = odds.fetch_html(f"{BEFOREINFO}?rno={int(rid[10:12])}&jcd={rid[8:10]}&hd={rid[:8]}")
                last = time.time()
                refused = refused + 1 if bstatus in BLOCK_CODES else 0
                if bhtml:
                    import gzip
                    _save_and_push(data, bpath, gzip.compress(bhtml, mtime=0))
                    stats["beforeinfo"] = stats.get("beforeinfo", 0) + 1
                else:
                    stats["beforeinfo_failed"] = stats.get("beforeinfo_failed", 0) + 1
        print(f"ODDS {len(done)}/{len(jobs)} {label} status={rec['capture_status']} n={rec['n_odds']} "
              f"before={rec['minutes_before']}min", flush=True)
        if refused >= BLOCK_AFTER:
            # 公式サイトに拒否されている。相手の意思に従い、今日はこれ以上取りにいかない（続きの実行も起動しない）
            _save_and_push(data, blocked_mark, {"blocked_at": now().isoformat(timespec="seconds"),
                                                "consecutive_refusals": refused, "last_status": status})
            stats["blocked"] = True
            print(f"BLOCKED consecutive_refusals={refused}", flush=True)
            break
    stats["targets"] = len(tg)
    stats["planned"] = len(jobs)
    return stats


def flush(data: Path) -> bool:
    """実行の最後に、送れていない commit を送る。"""
    try:
        ahead = subprocess.run(["git", "-C", str(data), "rev-list", "--count", "origin/main..HEAD"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        return False
    return ahead == "0" or _push(data, tries=8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--date")
    ap.add_argument("--stop-after-min", type=float, default=330)
    a = ap.parse_args(argv)
    day = dt.date.fromisoformat(a.date) if a.date else dt.datetime.now(JST).date()
    r = run(Path(a.data), day, a.stop_after_min)
    r["pushed"] = flush(Path(a.data))
    print("ODDS_DONE " + json.dumps(r), flush=True)
    with open("odds_result.json", "w") as f:
        json.dump(r, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
