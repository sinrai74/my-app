#!/usr/bin/env python3
"""
朝刊AI 新モデル 第4段階 調査 run（読み取り専用のデータ検証）

目的：公式 B（番組表）/ K（競走成績）ファイルを取得し、データ仕様草案（doc 05）の
固定長定義で読めるか、件数検査を満たすか、既存の motor_history.csv を作り直せるか、
旧パーサ parsers.parse_bangumi がどれだけ行を落としているかを確かめる。

- 本番データ・既存ファイルは変更しない。結果は標準出力（ジョブログ）と --out の下に書く。
- 取得は 3 秒間隔（サーバ負荷への配慮）。
- 調査専用。新システムの実装ではない。
"""
from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import hashlib
import io
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import lzh_extract  # noqa: E402

BASE = "http://www1.mbrace.or.jp/od2"
UA = "Mozilla/5.0 (compatible; asahi-ai-stage4-probe)"
INTERVAL = 3.0
ZEN = str.maketrans("０１２３４５６７８９：", "0123456789:")

# ---- doc 05 §3・§4 の固定長定義（バイト位置、cp932） ----
B_LEN = 79
B_COLS = [(0, 1, "lane"), (2, 6, "racer_no"), (6, 14, "name"), (14, 16, "age"), (16, 20, "branch"),
          (20, 22, "weight"), (22, 24, "class"), (25, 29, "nat_win"), (30, 35, "nat_2r"),
          (36, 40, "loc_win"), (41, 46, "loc_2r"), (46, 49, "motor_no"), (49, 55, "motor_2r"),
          (55, 58, "boat_no"), (58, 64, "boat_2r"), (64, 76, "series_raw"), (76, 79, "hayami")]
K_LEN = 66
K_COLS = [(2, 4, "order_code"), (6, 7, "lane"), (8, 12, "racer_no"), (13, 29, "name"),
          (29, 32, "motor_no"), (32, 37, "boat_no"), (37, 43, "exhibition_time"),
          (43, 47, "course"), (47, 56, "st"), (56, 66, "race_time")]
RE_B_RACER = re.compile(rb"^[1-6] \d{4}")
RE_K_RACER = re.compile(rb"^\s{2}[0-9A-Z ]{2}\s{2}[1-6] \d{4}")


def target_days() -> list[dt.date]:
    s = set()
    d = dt.date(2026, 6, 30)
    while d <= dt.date(2026, 11, 16):
        s.add(d)
        d += dt.timedelta(days=1)
    for y in (2023, 2024, 2025, 2026):
        for m in range(1, 13):
            for dd in (1, 15):
                x = dt.date(y, m, dd)
                if x <= dt.date(2026, 11, 16):
                    s.add(x)
    period = sorted(x for x in s if dt.date(2026, 6, 30) <= x <= dt.date(2026, 11, 16))
    others = sorted(x for x in s if not (dt.date(2026, 6, 30) <= x <= dt.date(2026, 11, 16)))
    return period + others


def fetch(kind: str, d: dt.date):
    url = f"{BASE}/{kind}/{d:%Y%m}/{kind.lower()}{d:%y%m%d}.lzh"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return url, r.status, r.read()
    except urllib.error.HTTPError as e:
        return url, e.code, None
    except Exception as e:  # noqa: BLE001
        return url, f"ERR:{type(e).__name__}", None


def extract(data: bytes) -> tuple[str, bytes]:
    files = lzh_extract.extract_all(data)
    if len(files) != 1:
        raise ValueError(f"members={list(files)}")
    return next(iter(files.items()))


def parse_b(txt: bytes):
    rows, issues, deadlines, days = [], [], {}, {}
    venue = race = None
    for raw in txt.split(b"\r\n"):
        m = re.match(rb"^(\d{2})BBGN$", raw)
        if m:
            venue, race = m.group(1).decode(), None
            days[venue] = None
            continue
        if re.match(rb"^\d{2}BEND$", raw):
            venue = None
            continue
        if venue is None:
            continue
        if RE_B_RACER.match(raw):
            if len(raw) != B_LEN:
                issues.append({"type": "b_line_len", "venue": venue, "race": race, "len": len(raw)})
                continue
            if race is None:
                issues.append({"type": "b_racer_before_race", "venue": venue})
                continue
            rec = {n: raw[a:b].decode("cp932").strip() for a, b, n in B_COLS}
            rec.update(venue=venue, race=race)
            rows.append(rec)
            continue
        s = raw.decode("cp932", errors="replace")
        mr = re.match(r"^[\s　]*([０-９\d]{1,2})Ｒ", s)
        if mr:
            race = int(mr.group(1).translate(ZEN))
            md = re.search(r"締切予定([０-９\d]{1,2})：([０-９\d]{2})", s)
            deadlines[(venue, race)] = (md.group(1).translate(ZEN) + ":" + md.group(2).translate(ZEN)) if md else None
            continue
        if days.get(venue) is None:
            mday = re.search(r"第[\s　]*([０-９\d]{1,2})[\s　]*日", s)
            if mday:
                days[venue] = int(mday.group(1).translate(ZEN))
            elif re.search(r"初[\s　]*日", s):
                days[venue] = 1
    return rows, issues, deadlines, days


def parse_st(s: str):
    s = s.strip()
    if not s or s.replace(" ", "") in (".", "K."):
        return None, s[:1] if s else ""
    prefix = s[0] if s[0].isalpha() else ""
    num = s[1:] if prefix else s
    try:
        v = float(num)
    except ValueError:
        return None, prefix or "?"
    return (-v if prefix == "F" else v), prefix


def parse_k(txt: bytes):
    rows, issues, conds = [], [], {}
    venue = race = None
    for raw in txt.split(b"\r\n"):
        m = re.match(rb"^(\d{2})KBGN$", raw)
        if m:
            venue, race = m.group(1).decode(), None
            continue
        if re.match(rb"^\d{2}KEND$", raw):
            venue = None
            continue
        if venue is None:
            continue
        if RE_K_RACER.match(raw):
            if len(raw) != K_LEN:
                issues.append({"type": "k_line_len", "venue": venue, "race": race, "len": len(raw)})
                continue
            if race is None:
                issues.append({"type": "k_racer_before_race", "venue": venue})
                continue
            rec = {n: raw[a:b].decode("cp932").strip() for a, b, n in K_COLS}
            rec["st_value"], rec["st_prefix"] = parse_st(rec["st"])
            rec.update(venue=venue, race=race)
            rows.append(rec)
            continue
        s = raw.decode("cp932", errors="replace")
        mh = re.match(r"^\s+(\d{1,2})R\s", s)
        if mh and re.search(r"H\d{3,4}m", s):
            race = int(mh.group(1))
            c = re.search(r"H(\d{3,4})m\s+(\S+)\s+風\s+(\S+)\s+(\d+)m\s+波\s+(\d+)cm", s.replace("　", " "))
            conds[(venue, race)] = {"ok": bool(c)}
            continue
        if race is not None and "ﾚｰｽﾀｲﾑ" in s:
            tail = s.split("ﾚｰｽﾀｲﾑ", 1)[1].replace("　", " ").strip()
            conds.setdefault((venue, race), {})["technique"] = tail or None
    return rows, issues, conds


def legacy_parse_bangumi_rows(txt: bytes, tmpdir: Path, name: str) -> tuple[int, list[str]]:
    import parsers  # 旧パーサ（比較用。変更しない）
    p = tmpdir / name
    p.write_bytes(txt)
    with contextlib.redirect_stdout(io.StringIO()):
        df = parsers.parse_bangumi(p)
    return len(df), sorted(df["場コード"].unique()) if len(df) else []


def load_motor_history():
    import pandas as pd
    mh = pd.read_csv(ROOT / "motor_history.csv", dtype=str)
    raw = len(mh)
    mh = mh.drop_duplicates()
    mh["venue_i"] = pd.to_numeric(mh["venue_num"], errors="coerce")
    mh["race_i"] = pd.to_numeric(mh["race_number"], errors="coerce")
    mh["lane_i"] = pd.to_numeric(mh["lane"], errors="coerce")
    return raw, len(mh), {d: g for d, g in mh.groupby("date")}


def compare_mh_day(krow, mhd):
    """その日の K（lane=進入コース）と motor_history を突き合わせる。"""
    st = collections.Counter()
    cp = collections.Counter()
    if mhd is None:
        st["mh_day_absent"] = 1
        return st, cp
    st["mh_rows"] = len(mhd)
    st["mh_lane_na"] = int(mhd["lane_i"].isna().sum())
    idx = {}
    for r in mhd.dropna(subset=["venue_i", "race_i", "lane_i"]).itertuples():
        idx[(int(r.venue_i), int(r.race_i), int(r.lane_i))] = r
    seen = set()
    for r in krow:
        if not re.fullmatch(r"\d", r["course"] or ""):
            continue
        key = (int(r["venue"]), int(r["race"]), int(r["course"]))
        m = idx.get(key)
        if m is None:
            st["k_only"] += 1
            continue
        seen.add(key)
        st["joined"] += 1
        st["racer_match"] += int(str(m.racer_no).lstrip("0") == r["racer_no"].lstrip("0"))
        try:
            st["motor_match"] += int(int(m.motor_no) == int(r["motor_no"]))
        except (TypeError, ValueError):
            pass

        def num(x):
            try:
                return round(float(x), 2)
            except (TypeError, ValueError):
                return None
        st["ex_time_match"] += int(num(m.ex_time) is not None and num(m.ex_time) == num(r["exhibition_time"]))
        st["st_match"] += int(num(m.start_timing) is not None and r["st_value"] is not None
                              and num(m.start_timing) == round(r["st_value"], 2))
        cp[f"{r['order_code']}->{m.place}"] += 1
    st["mh_only"] = len(idx) - len(seen)
    return st, cp


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--local", help="ローカルの lzh を読む（b/k + YYMMDD.lzh）。オフライン試験用")
    ap.add_argument("--days", help="YYYYMMDD をカンマ区切りで指定（試験用）")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    tmp = out / "tmp"
    tmp.mkdir(exist_ok=True)
    days = [dt.datetime.strptime(x, "%Y%m%d").date() for x in a.days.split(",")] if a.days else target_days()
    mh_raw, mh_dedup, mh_by_day = load_motor_history()
    print("MH " + json.dumps({"motor_history_rows_raw": mh_raw, "dedup": mh_dedup}), flush=True)

    S = collections.Counter()
    MH = collections.Counter()
    samples = collections.defaultdict(list)
    order_codes, st_prefixes, code_place = collections.Counter(), collections.Counter(), collections.Counter()
    max_digits = {"motor": 0, "boat": 0}
    perday_f = open(out / "per_day.jsonl", "w")

    def sample(key, v):
        if len(samples[key]) < 8:
            samples[key].append(v)

    for i, d in enumerate(days):
        rec = {"date": f"{d:%Y%m%d}", "issues": collections.Counter()}
        blobs = {}
        for kind in ("B", "K"):
            t0 = time.time()
            if a.local:
                pth = Path(a.local) / f"{kind.lower()}{d:%y%m%d}.lzh"
                status, data = (200 if pth.exists() else 404), (pth.read_bytes() if pth.exists() else None)
            else:
                _, status, data = fetch(kind, d)
            rec[f"{kind}_status"] = status
            rec[f"{kind}_sec"] = round(time.time() - t0, 1)
            S[f"http_{kind}_{status}"] += 1
            if not a.local:
                time.sleep(INTERVAL)
            if data is None:
                continue
            (out / "raw" / f"{kind.lower()}{d:%y%m%d}.lzh").write_bytes(data)
            rec[f"{kind}_sha"] = hashlib.sha256(data).hexdigest()[:16]
            try:
                blobs[kind] = extract(data)
            except Exception as e:  # noqa: BLE001
                rec["issues"][f"extract_fail_{kind}"] += 1
                sample(f"extract_fail_{kind}", f"{rec['date']}:{type(e).__name__}")
        if "B" in blobs and "K" in blobs:
            brow, biss, deadl, bdays = parse_b(blobs["B"][1])
            krow, kiss, kcond = parse_k(blobs["K"][1])
            for x in biss + kiss:
                rec["issues"][x["type"]] += 1
                sample(x["type"], {"date": rec["date"], **x})
            bv = collections.Counter(r["venue"] for r in brow)
            kv = collections.Counter(r["venue"] for r in krow)
            brr = collections.Counter((r["venue"], r["race"]) for r in brow)
            krr = collections.Counter((r["venue"], r["race"]) for r in krow)
            rec.update(b_rows=len(brow), k_rows=len(krow), b_venues=len(bv), k_venues=len(kv),
                       b_races=len(brr), k_races=len(krr))
            for key, n in brr.items():
                if n != 6:
                    rec["issues"]["b_race_not6"] += 1
                    sample("b_race_not6", [rec["date"], *key, n])
            for key, n in krr.items():
                if n != 6:
                    rec["issues"]["k_race_not6"] += 1
                    sample("k_race_not6", [rec["date"], *key, n])
            if set(bv) != set(kv):
                rec["issues"]["venue_set_mismatch"] += 1
                rec["venue_diff"] = sorted(set(bv) ^ set(kv))
            miss_k = sorted(set(brr) - set(krr))
            if miss_k:
                rec["issues"]["races_in_b_not_k"] += len(miss_k)
                rec["races_in_b_not_k"] = [f"{v}-{r}" for v, r in miss_k][:24]
            if set(krr) - set(brr):
                rec["issues"]["races_in_k_not_b"] += len(set(krr) - set(brr))
            rec["issues"]["deadline_missing"] += sum(1 for k in brr if not deadl.get(k))
            for v in bv:
                if not bdays.get(v):
                    rec["issues"]["meet_day_missing"] += 1
                    sample("meet_day_missing", [rec["date"], v])
            rec["issues"]["k_cond_unparsed"] += sum(1 for k in krr if not kcond.get(k, {}).get("ok"))
            rec["issues"]["k_technique_missing"] += sum(1 for k in krr if not kcond.get(k, {}).get("technique"))
            kidx = {(r["venue"], r["race"], r["lane"]): r for r in krow}
            for r in brow:
                kr = kidx.get((r["venue"], r["race"], r["lane"]))
                if kr is None:
                    rec["issues"]["b_entry_not_in_k"] += 1
                    continue
                if kr["order_code"] in ("K0", "K1"):
                    rec["issues"]["kesshou"] += 1
                    continue
                rec["issues"]["bk_pairs"] += 1
                if kr["racer_no"] != r["racer_no"]:
                    rec["issues"]["bk_racer_mismatch"] += 1
                    sample("bk_racer_mismatch", [rec["date"], r["venue"], r["race"], r["lane"], r["racer_no"], kr["racer_no"]])
                if kr["motor_no"].lstrip("0") != r["motor_no"].lstrip("0"):
                    rec["issues"]["bk_motor_mismatch"] += 1
                    sample("bk_motor_mismatch", [rec["date"], r["venue"], r["race"], r["lane"], r["motor_no"], kr["motor_no"]])
            for r in brow:
                max_digits["motor"] = max(max_digits["motor"], len(r["motor_no"]))
                max_digits["boat"] = max(max_digits["boat"], len(r["boat_no"]))
            for r in krow:
                order_codes[r["order_code"]] += 1
                st_prefixes[r["st_prefix"]] += 1
            try:
                n_old, v_old = legacy_parse_bangumi_rows(blobs["B"][1], tmp, blobs["B"][0])
                rec["legacy_rows"] = n_old
                if n_old < len(brow):
                    rec["legacy_lost_venues"] = sorted(set(bv) - set(v_old))
                    rec["issues"]["legacy_rows_lost"] += len(brow) - n_old
            except Exception as e:  # noqa: BLE001
                rec["issues"]["legacy_error"] += 1
                sample("legacy_error", f"{rec['date']}:{type(e).__name__}:{e}"[:200])
            if dt.date(2026, 6, 30) <= d <= dt.date(2026, 11, 16):
                mst, mcp = compare_mh_day(krow, mh_by_day.get(rec["date"]))
                rec["mh"] = dict(mst)
                MH.update(mst)
                code_place.update(mcp)
        rec["issues"] = dict(rec["issues"])
        S.update(rec["issues"])
        S["days_done"] += 1
        line = json.dumps(rec, ensure_ascii=False)
        perday_f.write(line + "\n")
        perday_f.flush()
        print(f"DAY {i + 1}/{len(days)} " + line, flush=True)

    perday_f.close()
    summary = {"days_requested": len(days), "counts": dict(S), "motor_history_check": dict(MH),
               "order_codes": dict(order_codes), "st_prefixes": dict(st_prefixes), "max_digits": max_digits,
               "k_code_to_mh_place": dict(code_place), "samples": dict(samples)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("=====ASAHI_PROBE_SUMMARY_BEGIN=====")
    print(json.dumps(summary, ensure_ascii=False))
    print("=====ASAHI_PROBE_SUMMARY_END=====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
