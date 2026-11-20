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
    return sorted(s)


def fetch(kind: str, d: dt.date):
    url = f"{BASE}/{kind}/{d:%Y%m}/{kind.lower()}{d:%y%m%d}.lzh"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
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
    mh_period = (dt.date(2026, 6, 30), dt.date(2026, 11, 16))

    S = collections.Counter()
    samples = collections.defaultdict(list)
    order_codes = collections.Counter()
    st_prefixes = collections.Counter()
    code_place = collections.Counter()
    max_digits = {"motor": 0, "boat": 0}
    per_day = []
    k_frames = []

    def sample(key, v):
        if len(samples[key]) < 8:
            samples[key].append(v)

    for i, d in enumerate(days):
        rec = {"date": f"{d:%Y%m%d}"}
        blobs = {}
        for kind in ("B", "K"):
            if a.local:
                p = Path(a.local) / f"{kind.lower()}{d:%y%m%d}.lzh"
                url, status, data = str(p), (200 if p.exists() else 404), (p.read_bytes() if p.exists() else None)
            else:
                url, status, data = fetch(kind, d)
                time.sleep(INTERVAL)
            rec[f"{kind}_status"] = status
            S[f"http_{kind}_{status}"] += 1
            if data is None:
                sample(f"missing_{kind}", rec["date"])
                continue
            (out / "raw" / f"{kind.lower()}{d:%y%m%d}.lzh").write_bytes(data)
            rec[f"{kind}_lzh_sha256"] = hashlib.sha256(data).hexdigest()
            try:
                name, txt = extract(data)
                blobs[kind] = (name, txt)
            except Exception as e:  # noqa: BLE001
                S[f"extract_fail_{kind}"] += 1
                sample(f"extract_fail_{kind}", f"{rec['date']}:{type(e).__name__}")
        if "B" not in blobs or "K" not in blobs:
            per_day.append(rec)
            continue

        brow, biss, deadl, bdays = parse_b(blobs["B"][1])
        krow, kiss, kcond = parse_k(blobs["K"][1])
        S["days_parsed"] += 1
        for x in biss + kiss:
            S[x["type"]] += 1
            sample(x["type"], {"date": rec["date"], **x})

        bv = collections.Counter(r["venue"] for r in brow)
        kv = collections.Counter(r["venue"] for r in krow)
        brr = collections.Counter((r["venue"], r["race"]) for r in brow)
        krr = collections.Counter((r["venue"], r["race"]) for r in krow)
        S["b_rows"] += len(brow)
        S["k_rows"] += len(krow)
        S["races_b"] += len(brr)
        S["races_k"] += len(krr)
        for key, n in brr.items():
            if n != 6:
                S["b_race_not6"] += 1
                sample("b_race_not6", [rec["date"], *key, n])
        for key, n in krr.items():
            if n != 6:
                S["k_race_not6"] += 1
                sample("k_race_not6", [rec["date"], *key, n])
        if set(bv) != set(kv):
            S["venue_set_mismatch"] += 1
            sample("venue_set_mismatch", [rec["date"], sorted(set(bv) ^ set(kv))])
        if set(brr) != set(krr):
            S["race_set_mismatch_days"] += 1
            sample("race_set_mismatch", [rec["date"], sorted(set(brr) ^ set(krr))[:6]])
        S["deadline_missing"] += sum(1 for k in brr if not deadl.get(k))
        S["meet_day_missing_venues"] += sum(1 for v in bv if not bdays.get(v))
        for v in bv:
            if not bdays.get(v):
                sample("meet_day_missing", [rec["date"], v])
        S["k_cond_unparsed"] += sum(1 for k in krr if not kcond.get(k, {}).get("ok"))
        S["k_technique_missing"] += sum(1 for k in krr if not kcond.get(k, {}).get("technique"))

        kidx = {(r["venue"], r["race"], r["lane"]): r for r in krow}
        for r in brow:
            kr = kidx.get((r["venue"], r["race"], r["lane"]))
            if kr is None:
                S["b_not_in_k"] += 1
                continue
            if kr["order_code"] in ("K0", "K1"):
                S["b_kesshou"] += 1
                continue
            S["bk_pairs"] += 1
            if kr["racer_no"] != r["racer_no"]:
                S["bk_racer_mismatch"] += 1
                sample("bk_racer_mismatch", [rec["date"], r["venue"], r["race"], r["lane"], r["racer_no"], kr["racer_no"]])
            if kr["motor_no"].lstrip("0") != r["motor_no"].lstrip("0"):
                S["bk_motor_mismatch"] += 1
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
                S["legacy_days_with_loss"] += 1
                S["legacy_rows_lost"] += len(brow) - n_old
                sample("legacy_loss", [rec["date"], len(brow), n_old, sorted(set(bv) - set(v_old))])
        except Exception as e:  # noqa: BLE001
            S["legacy_error"] += 1
            sample("legacy_error", f"{rec['date']}:{type(e).__name__}:{e}"[:200])

        if mh_period[0] <= d <= mh_period[1]:
            for r in krow:
                k_frames.append({**{k: r[k] for k in ("venue", "race", "lane", "racer_no", "motor_no", "exhibition_time",
                                                         "course", "order_code")},
                                 "st_value": r["st_value"], "date": rec["date"]})
        rec.update(b_rows=len(brow), k_rows=len(krow), venues=len(bv))
        per_day.append(rec)
        if i % 20 == 0:
            print(f"[{i + 1}/{len(days)}] {rec['date']} B={len(brow)} K={len(krow)} venues={len(bv)}", flush=True)

    # ---- motor_history の作り直し照合（lane = 進入コース） ----
    mh_result = {}
    try:
        import pandas as pd
        mh = pd.read_csv(ROOT / "motor_history.csv", dtype=str)
        mh_raw = len(mh)
        mh = mh.drop_duplicates()
        mh = mh[(mh["date"] >= f"{mh_period[0]:%Y%m%d}") & (mh["date"] <= f"{mh_period[1]:%Y%m%d}")]
        kd = pd.DataFrame(k_frames)
        days_k = set(kd["date"]) if len(kd) else set()
        mh = mh[mh["date"].isin(days_k)].copy()
        kd = kd[kd["course"].str.match(r"^\d$", na=False)].copy()
        kd["venue_i"] = kd["venue"].astype(int)
        kd["race_i"] = kd["race"].astype(int)
        kd["lane_i"] = kd["course"].astype(int)
        mh["venue_i"] = pd.to_numeric(mh["venue_num"], errors="coerce")
        mh["race_i"] = pd.to_numeric(mh["race_number"], errors="coerce")
        mh["lane_i"] = pd.to_numeric(mh["lane"], errors="coerce")
        mh_lane_na = int(mh["lane_i"].isna().sum())
        mhk = mh.dropna(subset=["venue_i", "race_i", "lane_i"]).copy()
        for c in ("venue_i", "race_i", "lane_i"):
            mhk[c] = mhk[c].astype(int)
        j = kd.merge(mhk, on=["date", "venue_i", "race_i", "lane_i"], how="outer", suffixes=("_k", "_m"), indicator=True)
        both = j[j["_merge"] == "both"]

        def numeq(a, b):
            return (pd.to_numeric(both[a], errors="coerce").round(2) == pd.to_numeric(both[b], errors="coerce").round(2))

        mh_result = {
            "motor_history_rows_raw": mh_raw,
            "compared_days": len(days_k),
            "k_rows_with_course": int(len(kd)),
            "mh_rows_on_those_days_dedup": int(len(mh)),
            "mh_rows_lane_na": mh_lane_na,
            "joined": int(len(both)),
            "k_only": int((j["_merge"] == "left_only").sum()),
            "mh_only": int((j["_merge"] == "right_only").sum()),
            "racer_match": int((both["racer_no_k"].astype(str).str.lstrip("0") == both["racer_no_m"].astype(str).str.lstrip("0")).sum()),
            "motor_match": int((pd.to_numeric(both["motor_no_k"], errors="coerce") == pd.to_numeric(both["motor_no_m"], errors="coerce")).sum()),
            "ex_time_match": int(numeq("exhibition_time", "ex_time").sum()),
            "st_match": int(numeq("st_value", "start_timing").sum()),
        }
        for code, place in zip(both["order_code"], both["place"]):
            code_place[f"{code}->{place}"] += 1
        ko = j[j["_merge"] == "left_only"][["date", "venue_i", "race_i", "lane_i"]].head(8).values.tolist()
        mo = j[j["_merge"] == "right_only"][["date", "venue_i", "race_i", "lane_i"]].head(8).values.tolist()
        samples["mh_k_only"] = [list(map(str, x)) for x in ko]
        samples["mh_mh_only"] = [list(map(str, x)) for x in mo]
    except Exception as e:  # noqa: BLE001
        mh_result = {"error": f"{type(e).__name__}: {e}"[:300]}

    summary = {
        "days_requested": len(days),
        "counts": dict(S),
        "order_codes": dict(order_codes),
        "st_prefixes": dict(st_prefixes),
        "max_digits": max_digits,
        "motor_history_check": mh_result,
        "k_code_to_mh_place": dict(code_place),
        "samples": dict(samples),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (out / "per_day.json").write_text(json.dumps(per_day, ensure_ascii=False))
    print("=====ASAHI_PROBE_SUMMARY_BEGIN=====")
    print(json.dumps(summary, ensure_ascii=False))
    print("=====ASAHI_PROBE_SUMMARY_END=====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
