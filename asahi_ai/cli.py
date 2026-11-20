"""コマンド。

  # 手元の原本から1日分を作る（取得しない）
  python -m asahi_ai.cli build-local --date 2023-01-01 --b b230101.lzh --k k230101.lzh --out OUT

  # 取得して作る（期間。1件ずつ3秒の間隔。原本は OUT/raw に保存）
  python -m asahi_ai.cli fetch-build --start 2026-11-01 --end 2026-11-03 --out OUT

日ごとの要約を1行の JSON（行頭 "DAY "）で標準出力に出す。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

from . import archive, build, fetch, parse_b, parse_k, store


def _summary(t: build.DayTables, extra: dict | None = None) -> str:
    levels = {}
    samples = {}
    for c in t.checks:
        levels.setdefault(c["level"], {}).setdefault(c["check"], 0)
        levels[c["level"]][c["check"]] += 1
        sm = samples.setdefault(c["check"], [])
        if len(sm) < 3:  # 各検査の例を3件まで（原因の確認用）
            sm.append({"target": c["target"], "detail": c["detail"], "expected": c["expected"], "actual": c["actual"]})
    return "DAY " + json.dumps({"date": t.date, "verdict": t.day_verdict, "counts": t.counts,
                                "checks": levels, "samples": samples, **(extra or {})},
                               ensure_ascii=False, default=str)


def process(d: dt.date, b_bytes: bytes | None, k_bytes: bytes | None,
            b_id: str | None = None, k_id: str | None = None) -> build.DayTables:
    bp = kp = None
    errs = []
    if b_bytes is not None:
        try:
            bp = parse_b.parse_b(archive.extract_single(b_bytes).data, d)
        except archive.ArchiveError as e:
            errs.append(("b_archive", str(e)))
    if k_bytes is not None:
        try:
            kp = parse_k.parse_k(archive.extract_single(k_bytes).data, d)
        except archive.ArchiveError as e:
            errs.append(("k_archive", str(e)))
    t = build.build_day(d, bp, kp, b_id, k_id)
    for name, msg in errs:
        build._check(t.checks, d.isoformat(), "stop", name, detail=msg)
        t.day_verdict = "stop"
    return t


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="asahi_ai")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("build-local")
    a.add_argument("--date", required=True)
    a.add_argument("--b")
    a.add_argument("--k")
    a.add_argument("--out", required=True)
    f = sub.add_parser("fetch-build")
    f.add_argument("--start")
    f.add_argument("--end")
    f.add_argument("--dates-file", help="1行に1日（YYYY-MM-DD）。--start/--end の代わり")
    f.add_argument("--out", required=True)
    f.add_argument("--interval", type=float, default=fetch.INTERVAL)
    bf = sub.add_parser("backfill", help="原本の取り込み（未処理の日だけ。表は作らず、原本・manifest・検査を保存）")
    bf.add_argument("--start", required=True)
    bf.add_argument("--end", required=True)
    bf.add_argument("--data", required=True, help="データリポジトリの作業ツリー")
    bf.add_argument("--max-days", type=int, default=60)
    bf.add_argument("--deadline-epoch", type=float, default=None, help="この時刻（UNIX 秒）を過ぎたら新しい日を始めない")
    bf.add_argument("--interval", type=float, default=fetch.INTERVAL)
    args = ap.parse_args(argv)

    if args.cmd == "backfill":
        return backfill(args)

    out = Path(args.out)
    if args.cmd == "build-local":
        d = dt.date.fromisoformat(args.date)
        bb = Path(args.b).read_bytes() if args.b else None
        kb = Path(args.k).read_bytes() if args.k else None
        t = process(d, bb, kb, archive.sha256(bb) if bb else None, archive.sha256(kb) if kb else None)
        store.write_day(out, t)
        print(_summary(t), flush=True)
        return 0 if t.day_verdict != "stop" else 2

    fe = fetch.Fetcher(out / "raw", interval=args.interval)
    if args.dates_file:
        days = [dt.date.fromisoformat(x.strip()) for x in Path(args.dates_file).read_text().split() if x.strip()]
    else:
        a0, a1 = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
        days = [a0 + dt.timedelta(days=i) for i in range((a1 - a0).days + 1)]
    worst = 0
    for d in days:
        recs = {}
        data = {}
        for kind in ("B", "K"):
            rec, raw = fe.get(kind, d)
            store.append_manifest(out, rec)
            recs[kind], data[kind] = rec, raw
        if all(recs[k]["capture_status"] == "not_found" for k in ("B", "K")):
            print("DAY " + json.dumps({"date": d.isoformat(), "verdict": "no_files"}), flush=True)
        else:
            t = process(d, data["B"], data["K"], recs["B"]["lzh_sha256"], recs["K"]["lzh_sha256"])
            for kind in ("B", "K"):
                if recs[kind]["capture_status"] in ("fetch_failed",):
                    build._check(t.checks, d.isoformat(), "stop", f"{kind.lower()}_fetch_failed",
                                 detail=recs[kind]["error"])
                    t.day_verdict = "stop"
            store.write_day(out, t)
            print(_summary(t, {"fetch": {k: [recs[k]["http_status"], recs[k]["capture_status"]] for k in recs}}),
                  flush=True)
            worst = max(worst, 2 if t.day_verdict == "stop" else 0)
    return worst


def backfill(args) -> int:
    """未処理の日（checks/<年>/<日>.jsonl が無い日）を古い順に max-days 日まで処理する。

    - 取得に失敗した日は checks を書かない（次の実行で取り直す）。
    - 原本の内容が前回と違えば RawConflict で止まる（上書きしない）。
    - 標準出力の最後に "BACKFILL {...}"（処理した日数・残り・stop の日）を出す。
    """
    import time

    data = Path(args.data)
    a0, a1 = dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end)
    todo = [a0 + dt.timedelta(days=i) for i in range((a1 - a0).days + 1)]
    todo = [d for d in todo if not store.checks_path(data, d.isoformat()).exists()]
    fe = fetch.Fetcher(data / "raw", interval=args.interval)
    done, stops, failed = 0, [], []
    for d in todo[: args.max_days]:
        if args.deadline_epoch and time.time() > args.deadline_epoch:
            break
        recs, raw = {}, {}
        for kind in ("B", "K"):
            recs[kind], raw[kind] = fe.get(kind, d)
            store.append_manifest(data, recs[kind])
        st = {k: recs[k]["capture_status"] for k in recs}
        if "fetch_failed" in st.values():
            failed.append(d.isoformat())
            print("DAY " + json.dumps({"date": d.isoformat(), "verdict": "fetch_failed", "fetch": st}), flush=True)
            continue
        if all(v == "not_found" for v in st.values()):
            store.write_no_files(data, d.isoformat())
            print("DAY " + json.dumps({"date": d.isoformat(), "verdict": "no_files"}), flush=True)
            done += 1
            continue
        t = process(d, raw["B"], raw["K"], recs["B"]["lzh_sha256"], recs["K"]["lzh_sha256"])
        store.write_day(data, t, tables=False)
        if t.day_verdict == "stop":
            stops.append(d.isoformat())
        print(_summary(t, {"fetch": st}), flush=True)
        done += 1
    remaining = len(todo) - done
    print("BACKFILL " + json.dumps({"done": done, "remaining": remaining, "stop_days": stops,
                                    "fetch_failed": failed}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
