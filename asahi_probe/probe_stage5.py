"""朝刊AI 新モデル 第5段階の追加調査（読み取り専用）。

1. 2006〜2022年の抜き出し：B/K の形式が 2023〜2026 の固定長定義と同じかを確かめる
2. motor_history との ST の不一致6件を、艇ごとに書き出す
3. 2026-02-01 の B を取り直す

リポジトリへの書き込み・通知はしない。公式ファイルは1件ずつ、3秒の間隔をあけて取得する。
結果はログに出す（1項目ごとに1行の JSON、行頭は "S5 "）。
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_stage4 as p4  # noqa: E402

YEARS = [2006, 2007, 2010, 2013, 2016, 2019, 2022]
ST_DAYS = ["2026-07-02", "2026-08-07", "2026-09-27", "2026-10-14", "2026-10-25", "2026-11-09"]
CLASSES = {"A1", "A2", "B1", "B2"}


def out(kind, obj):
    print("S5 " + json.dumps({"kind": kind, **obj}, ensure_ascii=False), flush=True)


_last = [0.0]


def get(kind, d):
    wait = p4.INTERVAL - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    t = time.time()
    url, status, data = p4.fetch(kind, d)
    _last[0] = time.time()
    if data is None and isinstance(status, str):  # 通信の失敗は1回だけ再試行
        time.sleep(p4.INTERVAL)
        url, status, data = p4.fetch(kind, d)
        _last[0] = time.time()
    return url, status, data, round(time.time() - t, 1)


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def block_line_stats(txt: bytes, tag: bytes):
    """場のブロック内の行の長さの分布と、先頭の場のブロックの原文（最大90行）。"""
    lens = collections.Counter()
    first_block, in_first, seen_block = [], False, False
    venue = None
    for raw in txt.split(b"\r\n"):
        if re.match(rb"^\d{2}" + tag + rb"BGN$", raw):
            venue = raw[:2]
            if not seen_block:
                in_first, seen_block = True, True
            continue
        if re.match(rb"^\d{2}" + tag + rb"END$", raw):
            venue = None
            in_first = False
            continue
        if venue is None:
            continue
        lens[len(raw)] += 1
        if in_first and len(first_block) < 90:
            first_block.append(raw.decode("cp932", errors="replace"))
    if not seen_block:  # 区切りの行が見つからない（形式が違う）ときは、ファイルの先頭を出す
        first_block = ["<NO_BLOCK_MARKER>"] + [x.decode("cp932", errors="replace") for x in txt.split(b"\r\n")[:90]]
    return dict(sorted(lens.items())), first_block


def check_b(rows):
    bad = collections.Counter()
    for r in rows:
        a, w = fnum(r["age"]), fnum(r["weight"])
        if a is None or not 15 <= a <= 80:
            bad["age"] += 1
        if w is None or not 35 <= w <= 80:
            bad["weight"] += 1
        if r["class"] not in CLASSES:
            bad["class"] += 1
        for c in ("nat_win", "loc_win"):
            v = fnum(r[c])
            if v is None or not 0 <= v <= 10:
                bad[c] += 1
        for c in ("nat_2r", "loc_2r", "motor_2r", "boat_2r"):
            v = fnum(r[c])
            if v is None or not 0 <= v <= 100:
                bad[c] += 1
        for c in ("motor_no", "boat_no"):
            if not re.fullmatch(r"\d{1,3}", r[c] or ""):
                bad[c] += 1
    return dict(bad)


def check_k(rows):
    bad = collections.Counter()
    codes = collections.Counter()
    for r in rows:
        codes[r["order_code"]] += 1
        ex = fnum(r["exhibition_time"])
        if r["exhibition_time"] and not r["exhibition_time"].startswith("K") and (ex is None or not 6.0 <= ex <= 8.0):
            bad["exhibition_time"] += 1
        if r["course"] and not re.fullmatch(r"[1-6]", r["course"]):
            bad["course"] += 1
        if r["st_value"] is None and r["st_prefix"] not in ("K", "L", ""):
            bad["st"] += 1
        if r["race_time"] and not re.fullmatch(r"\d\.\d{2}\.\d|\.\s*\.", r["race_time"]):
            bad["race_time"] += 1
        if not re.fullmatch(r"\d{1,3}", r["motor_no"] or ""):
            bad["motor_no"] += 1
    return dict(bad), dict(codes)


def payout_lines(txt: bytes, limit=40):
    """先頭の場の払戻の部分の原文（券種の語を含む行）。"""
    res = []
    for raw in txt.split(b"\r\n"):
        s = raw.decode("cp932", errors="replace")
        if re.search(r"単勝|複勝|２連単|２連複|拡連複|３連単|３連複|2連単|3連単", s):
            res.append(s)
            if len(res) >= limit:
                break
    return res


def probe_old_years():
    for y in YEARS:
        for m in (1, 7):
            base = dt.date(y, m, 15)
            for kind in ("B", "K"):
                got = None
                for off in range(4):  # 開催の無い日なら翌日以降を試す
                    d = base + dt.timedelta(days=off)
                    url, status, data, sec = get(kind, d)
                    if data:
                        got = (d, url, status, data, sec)
                        break
                    out("old_miss", {"kind": kind, "date": d.isoformat(), "status": status})
                if not got:
                    continue
                d, url, status, data, sec = got
                rec = {"kind": kind, "date": d.isoformat(), "status": status, "sec": sec,
                       "lzh_size": len(data)}
                try:
                    name, txt = p4.extract(data)
                except Exception as e:  # noqa: BLE001
                    rec["extract_error"] = f"{type(e).__name__}:{e}"
                    out("old_file", rec)
                    continue
                rec["member"] = name
                rec["txt_size"] = len(txt)
                rec["crlf"] = txt.count(b"\r\n")
                rec["lone_lf"] = txt.count(b"\n") - txt.count(b"\r\n")
                try:
                    txt.decode("cp932")
                    rec["cp932_ok"] = True
                except UnicodeDecodeError as e:
                    rec["cp932_ok"] = False
                    rec["cp932_err"] = str(e)[:120]
                tag = kind.encode()
                rec["venues"] = len(re.findall(rb"(?m)^\d{2}" + tag + rb"BGN\r?$", txt))
                rec["line_lens"], first = block_line_stats(txt, tag)
                if kind == "B":
                    rows, issues, deadlines, days = p4.parse_b(txt)
                    races = collections.Counter((r["venue"], r["race"]) for r in rows)
                    rec.update(rows=len(rows), races=len(races),
                               races_not6=sum(1 for v in races.values() if v != 6),
                               issues=collections.Counter(i["type"] for i in issues),
                               deadline_found=sum(1 for v in deadlines.values() if v),
                               deadline_total=len(deadlines),
                               meet_day_missing=sum(1 for v in days.values() if v is None),
                               bad_values=check_b(rows),
                               sample_rows=rows[:2])
                else:
                    rows, issues, conds = p4.parse_k(txt)
                    races = collections.Counter((r["venue"], r["race"]) for r in rows)
                    bad, codes = check_k(rows)
                    rec.update(rows=len(rows), races=len(races),
                               races_not6=sum(1 for v in races.values() if v != 6),
                               issues=collections.Counter(i["type"] for i in issues),
                               cond_ok=sum(1 for v in conds.values() if v.get("ok")),
                               cond_total=len(conds),
                               technique_found=sum(1 for v in conds.values() if v.get("technique")),
                               bad_values=bad, order_codes=codes,
                               st_prefixes=dict(collections.Counter(r["st_prefix"] for r in rows)),
                               sample_rows=rows[:2],
                               payout_sample=payout_lines(txt))
                rec["first_block"] = first
                out("old_file", rec)


def probe_st_mismatch():
    raw, dedup, mh = p4.load_motor_history()
    for ds in ST_DAYS:
        d = dt.date.fromisoformat(ds)
        url, status, data, sec = get("K", d)
        if not data:
            out("st_day", {"date": ds, "status": status})
            continue
        _, txt = p4.extract(data)
        rows, _, _ = p4.parse_k(txt)
        mhd = mh.get(ds.replace("-", ""))
        if mhd is None:
            out("st_day", {"date": ds, "mh": "absent"})
            continue
        idx = {}
        for r in mhd.dropna(subset=["venue_i", "race_i", "lane_i"]).itertuples():
            idx.setdefault((int(r.venue_i), int(r.race_i), int(r.lane_i)), []).append(r)
        n = 0
        for r in rows:
            if not re.fullmatch(r"\d", r["course"] or ""):
                continue
            ms = idx.get((int(r["venue"]), int(r["race"]), int(r["course"])))
            if not ms:
                continue
            m = ms[0]
            mst = fnum(m.start_timing)
            kst = r["st_value"]
            if mst is None or kst is None or round(mst, 2) != round(kst, 2):
                n += 1
                out("st_mismatch", {
                    "date": ds, "venue": r["venue"], "race": r["race"], "lane": r["lane"],
                    "course": r["course"], "racer_no": r["racer_no"], "order_code": r["order_code"],
                    "k_st_raw": r["st"], "k_st_value": kst, "mh_start_timing": m.start_timing,
                    "mh_place": m.place, "mh_dups": len(ms),
                    "mh_row": {k: str(getattr(m, k)) for k in m._fields if k not in ("Index",)},
                })
        out("st_day", {"date": ds, "mismatches": n, "k_rows": len(rows)})


def probe_b_20260201():
    d = dt.date(2026, 2, 1)
    url, status, data, sec = get("B", d)
    rec = {"date": d.isoformat(), "status": status, "sec": sec}
    if data:
        name, txt = p4.extract(data)
        rows, issues, deadlines, days = p4.parse_b(txt)
        races = collections.Counter((r["venue"], r["race"]) for r in rows)
        rec.update(member=name, lzh_sha256=__import__("hashlib").sha256(data).hexdigest(),
                   rows=len(rows), venues=len({r["venue"] for r in rows}), races=len(races),
                   races_not6=sum(1 for v in races.values() if v != 6),
                   issues=len(issues), bad_values=check_b(rows))
        url2, st2, kdata, _ = get("K", d)
        if kdata:
            _, ktxt = p4.extract(kdata)
            krows, _, _ = p4.parse_k(ktxt)
            kidx = {(r["venue"], r["race"], r["lane"]): r for r in krows}
            both = [(r, kidx.get((r["venue"], r["race"], r["lane"]))) for r in rows]
            rec["bk_pairs"] = sum(1 for _, k in both if k)
            rec["bk_racer_mismatch"] = sum(1 for b, k in both if k and b["racer_no"] != k["racer_no"])
            rec["bk_motor_mismatch"] = sum(1 for b, k in both if k and b["motor_no"].lstrip("0") != k["motor_no"].lstrip("0"))
            rec["b_not_in_k"] = sum(1 for _, k in both if not k)
    out("b_20260201", rec)


def main() -> int:
    t0 = time.time()
    probe_b_20260201()
    probe_st_mismatch()
    probe_old_years()
    out("done", {"minutes": round((time.time() - t0) / 60, 1)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
