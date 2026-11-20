"""正規化テーブルの書き出し（1日ずつ CSV。標準ライブラリだけ）。

置き場所：<out>/tables/<table>/<YYYY>/<YYYY-MM-DD>.csv、検査は <out>/checks/<YYYY>/<date>.jsonl、
manifest は <out>/manifest/<YYYY>.jsonl（追記）。P2 でまとめた形式（parquet 等）に変える余地を残す。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .build import DayTables

TABLES = ("races", "entries", "results", "race_conditions", "payouts")


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    cols: list[str] = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})
    tmp.replace(path)


def write_day(out: Path, t: DayTables):
    out = Path(out)
    y = t.date[:4]
    for name in TABLES:
        _write_csv(out / "tables" / name / y / f"{t.date}.csv", getattr(t, name))
    cp = out / "checks" / y / f"{t.date}.jsonl"
    cp.parent.mkdir(parents=True, exist_ok=True)
    with cp.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"date": t.date, "day_verdict": t.day_verdict, "counts": t.counts},
                           ensure_ascii=False) + "\n")
        for c in t.checks:
            f.write(json.dumps(c, ensure_ascii=False, default=str) + "\n")


def append_manifest(out: Path, rec: dict):
    p = Path(out) / "manifest" / f"{rec['date'][:4]}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
