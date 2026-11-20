"""1日分の B・K から、正規化テーブルと品質検査の結果を作る（Project 05 §6・§8）。

判定：
- stop：その日全体を使わない（列の定義の変化・B と K の不一致・キーの重複など）
- quarantined：そのレースだけ学習・評価から外す（6艇でない・未知のコード・払戻の不整合など）
- warn：記録だけする
- ok
"""
from __future__ import annotations

import collections
import datetime as dt
from dataclasses import dataclass, field

from . import PARSER_VERSION
from .layout import LAYOUT_VERSION
from .parse_b import BParse
from .parse_k import KParse


@dataclass
class DayTables:
    date: str
    day_verdict: str
    races: list[dict] = field(default_factory=list)
    entries: list[dict] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    race_conditions: list[dict] = field(default_factory=list)
    payouts: list[dict] = field(default_factory=list)
    checks: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)


def _check(checks, date, level, check, grain="day", target="", detail=None, expected=None, actual=None):
    checks.append({
        "date": date, "grain": grain, "target": target, "check": check, "level": level,
        "expected": expected, "actual": actual, "detail": detail,
        "parser_version": PARSER_VERSION, "layout_version": LAYOUT_VERSION,
    })


def build_day(date: dt.date, b: BParse | None, k: KParse | None,
              b_file_id: str | None = None, k_file_id: str | None = None) -> DayTables:
    ds = date.isoformat()
    checks: list[dict] = []
    t = DayTables(date=ds, day_verdict="ok")

    for src, p in (("B", b), ("K", k)):
        if p is None:
            _check(checks, ds, "stop" if src == "B" else "warn", f"{src.lower()}_source_unavailable",
                   detail=f"{src} の原本が無い")
            continue
        for i in p.issues:
            grain = "race" if i.get("race_id") else ("venue" if i.get("venue") else "day")
            tgt = i.get("race_id") or i.get("venue") or ""
            if i.get("lane") is not None:
                grain, tgt = "entry", f"{tgt}-{i['lane']}"
            _check(checks, ds, i["level"], i["check"], grain, tgt, detail=i.get("detail"))

    b_races = {r["race_id"]: r for r in (b.races if b else [])}
    k_races = {r["race_id"]: r for r in (k.races if k else [])}

    # キーの重複
    for name, rows, key in (("b_race", b.races if b else [], ("race_id",)),
                            ("b_entry", b.entries if b else [], ("race_id", "lane")),
                            ("k_race", k.races if k else [], ("race_id",)),
                            ("k_result", k.results if k else [], ("race_id", "lane"))):
        c = collections.Counter(tuple(r[x] for x in key) for r in rows)
        dups = [k_ for k_, n in c.items() if n > 1]
        if dups:
            _check(checks, ds, "stop", f"{name}_duplicate_key", detail=[list(d) for d in dups[:10]], actual=len(dups))

    b_ent = collections.defaultdict(dict)
    for e in (b.entries if b else []):
        b_ent[e["race_id"]][e["lane"]] = e
    k_res = collections.defaultdict(dict)
    for r in (k.results if k else []):
        k_res[r["race_id"]][r["lane"]] = r

    quarantined = {c["target"].split("-")[0] for c in checks if c["level"] == "quarantined" and c["target"]}

    # レースごとの件数・照合
    for rid in sorted(set(b_races) | set(k_races)):
        if rid in b_races and len(b_ent[rid]) != 6:
            _check(checks, ds, "quarantined", "b_race_not6", "race", rid, expected=6, actual=len(b_ent[rid]))
            quarantined.add(rid)
        if rid in k_races and len(k_res[rid]) != 6:
            _check(checks, ds, "quarantined", "k_race_not6", "race", rid, expected=6, actual=len(k_res[rid]))
            quarantined.add(rid)
        if b is not None and rid in k_races and rid not in b_races:
            _check(checks, ds, "stop", "k_race_not_in_b", "race", rid)
        if rid in b_races and rid in k_races:
            for lane, e in b_ent[rid].items():
                r = k_res[rid].get(lane)
                if r is None:
                    continue
                if r["racer_no"] != e["racer_no"] or r["motor_no"] != e["motor_no"]:
                    _check(checks, ds, "stop", "bk_entry_mismatch", "entry", f"{rid}-{lane}",
                           detail={"b": [e["racer_no"], e["motor_no"]], "k": [r["racer_no"], r["motor_no"]]})
        if rid in k_races and not k_races[rid]["winning_technique"]:
            _check(checks, ds, "warn", "k_technique_missing", "race", rid)

    # 払戻と着順の整合（3連単）。同着・特払い・3着までが揃わないときは照合しない（warn）
    pay = collections.defaultdict(list)
    for p in (k.payouts if k else []):
        pay[p["race_id"]].append(p)
    for rid in k_races:
        fin = sorted((r["finish_position"], r["lane"]) for r in k_res[rid].values() if r["finish_position"])
        pos = [f for f, _ in fin]
        tri = [p for p in pay[rid] if p["bet_type"] == "trifecta"]
        if len(pos) >= 3 and pos[:3] == [1, 2, 3] and all(pos.count(i) == 1 for i in (1, 2, 3)):
            expect = "-".join(str(l) for _, l in fin[:3])
            hits = [p["combination"] for p in tri if p["payout_kind"] == "hit"]
            if hits != [expect]:
                _check(checks, ds, "quarantined", "payout_trifecta_mismatch", "race", rid,
                       expected=expect, actual=hits)
                quarantined.add(rid)
            summ = k.summary_trifecta.get(rid)
            if summ and tri and (summ[0], summ[1]) != (tri[0]["combination"], tri[0]["payout_yen"]):
                _check(checks, ds, "quarantined", "payout_summary_mismatch", "race", rid,
                       expected=list(summ), actual=[tri[0]["combination"], tri[0]["payout_yen"]])
                quarantined.add(rid)
        else:
            _check(checks, ds, "warn", "payout_check_skipped", "race", rid, detail=pos)

    # races（B が正。状態を付ける）
    for rid, r in sorted(b_races.items()):
        if rid in k_races:
            status = "completed"
        elif k is None:
            status = "source_unavailable"
        else:
            status = "result_missing"
        t.races.append({**r, "race_status": status, "quarantined": rid in quarantined,
                        "b_file_id": b_file_id, "k_file_id": k_file_id if rid in k_races else None})
    if b is None:  # B が無い日は K だけのレースを記録（予測には使えない）
        for rid, r in sorted(k_races.items()):
            t.races.append({"race_id": rid, "date": ds, "venue": r["venue"], "race_no": r["race_no"],
                            "race_name": r["race_name_k"], "distance_m": r["distance_m"],
                            "deadline_time": None, "meet_day": None, "meet_final_day": None,
                            "race_status": "completed_no_program", "quarantined": rid in quarantined,
                            "b_file_id": None, "k_file_id": k_file_id})
    t.entries = [{**e, "b_file_id": b_file_id} for e in (b.entries if b else [])]
    t.results = [{**r, "k_file_id": k_file_id} for r in (k.results if k else [])]
    t.race_conditions = [{"race_id": r["race_id"], "weather": r["weather"], "wind_dir": r["wind_dir"],
                          "wind_speed_m": r["wind_speed_m"], "wave_cm": r["wave_cm"],
                          "winning_technique": r["winning_technique"], "k_file_id": k_file_id}
                         for r in (k.races if k else [])]
    t.payouts = [{**p, "k_file_id": k_file_id} for p in (k.payouts if k else [])]

    st = collections.Counter(r["race_status"] for r in t.races)
    t.counts = {
        "b_venues": len(b.venues) if b else None, "k_venues": len(k.venues) if k else None,
        "b_races": len(b_races), "k_races": len(k_races), "b_entries": len(t.entries),
        "k_results": len(t.results), "payouts": len(t.payouts),
        "absent": sum(1 for r in t.results if r["result_class"] == "absent"),
        "unknown_code": sum(1 for r in t.results if r["result_class"] == "unknown"),
        "quarantined_races": len(quarantined), **{f"status_{k_}": v for k_, v in st.items()},
    }
    levels = {c["level"] for c in checks}
    t.day_verdict = "stop" if "stop" in levels else ("quarantined" if "quarantined" in levels else "ok")
    t.checks = checks
    return t
