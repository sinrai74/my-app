"""朝の番組表（P4）。今の朝刊と同じ BoatraceOpenAPI の programs（公式の出走表を写した JSON）を使う。

2026-11-23 の調査（Project 18）：11/01〜11/22 の 19,440艇について、公式の B と比べた。
- 一致：選手・全国勝率・全国2率・モーター番号・モーター2率・ボート番号・ボート2率・年齢・距離・級
- 違う・無い：体重（朝は小数）、当地勝率・当地2率（約13% が違う）、今節成績・節日（無い）
  → これらはモデルに入れない（features.NOT_IN_MORNING）。
- 級：racer_class_number 1=A1、2=A2、3=B1、4=B2（全件で対応）。
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request

import pandas as pd

URL = "https://boatraceopenapi.github.io/programs/v2/{y}/{ymd}.json"
CLASS = {1: "A1", 2: "A2", 3: "B1", 4: "B2"}


def fetch(d: dt.date, timeout: float = 60) -> tuple[int | None, bytes | None, str | None, str | None]:
    """返り値：(HTTP の状態, 本体, Last-Modified, エラー)。"""
    url = URL.format(y=d.year, ymd=f"{d:%Y%m%d}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (asahi-ai-morning)"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), r.headers.get("Last-Modified"), None
    except urllib.error.HTTPError as e:
        return e.code, None, None, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return None, None, None, f"{type(e).__name__}: {e}"


def to_tables(body: bytes, d: dt.date) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    """JSON → (races, entries, issues)。entries の列は parse_b と同じ名前にそろえる。"""
    js = json.loads(body.decode("utf-8"))
    issues = []
    races, entries = [], []
    for p in js.get("programs", []):
        if str(p.get("race_date")) != d.isoformat():
            issues.append({"level": "stop", "check": "morning_date", "detail": p.get("race_date")})
            continue
        venue = f"{int(p['race_stadium_number']):02d}"
        no = int(p["race_number"])
        rid = f"{d:%Y%m%d}{venue}{no:02d}"
        closed = str(p.get("race_closed_at") or "")
        races.append({"race_id": rid, "date": d.isoformat(), "venue": venue, "race_no": no,
                      "distance_m": p.get("race_distance"), "deadline_time": closed[11:16] or None,
                      "race_closed_at": closed or None, "meet_day": None, "race_status": "scheduled",
                      "quarantined": False})
        boats = p.get("boats") or []
        if len(boats) != 6:
            issues.append({"level": "quarantined", "check": "morning_not6", "race_id": rid, "detail": len(boats)})
        for b in boats:
            cls = CLASS.get(b.get("racer_class_number"))
            if cls is None:
                issues.append({"level": "quarantined", "check": "morning_class", "race_id": rid,
                               "detail": b.get("racer_class_number")})
            entries.append({
                "race_id": rid, "date": d.isoformat(), "lane": int(b["racer_boat_number"]),
                "racer_no": f"{int(b['racer_number']):04d}", "age": b.get("racer_age"),
                "weight": b.get("racer_weight"), "class": cls,
                "nat_win": b.get("racer_national_top_1_percent"), "nat_2r": b.get("racer_national_top_2_percent"),
                "loc_win": b.get("racer_local_top_1_percent"), "loc_2r": b.get("racer_local_top_2_percent"),
                "motor_no": b.get("racer_assigned_motor_number"),
                "motor_2r": b.get("racer_assigned_motor_top_2_percent"),
                "boat_no": b.get("racer_assigned_boat_number"),
                "boat_2r": b.get("racer_assigned_boat_top_2_percent"),
                "series_orders": "",
            })
    return pd.DataFrame(races), pd.DataFrame(entries), issues


COMPARE = ["racer_no", "nat_win", "nat_2r", "motor_no", "motor_2r", "boat_no", "boat_2r", "age", "class"]


def compare_with_b(entries: pd.DataFrame, b_entries: list[dict]) -> dict:
    """朝の番組表と、後で公開された公式の B を比べる（モデルに入れる項目だけ）。"""
    bidx = {(e["race_id"], e["lane"]): e for e in b_entries}
    out = {"matched": 0, "missing_in_b": 0, "diff": {}, "examples": {}}
    for r in entries.to_dict("records"):
        e = bidx.get((r["race_id"], r["lane"]))
        if e is None:
            out["missing_in_b"] += 1
            continue
        out["matched"] += 1
        for k in COMPARE:
            x, y = r[k], e[k]
            try:
                same = (str(x) == str(y)) if k in ("racer_no", "class") else abs(float(x) - float(y)) < 1e-6
            except (TypeError, ValueError):
                same = False
            if not same:
                out["diff"][k] = out["diff"].get(k, 0) + 1
                out["examples"].setdefault(k, [r["race_id"], r["lane"], x, y])
    out["missing_in_morning"] = len(set(bidx) - {(r["race_id"], r["lane"]) for r in entries.to_dict("records")})
    return out
