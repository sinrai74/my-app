"""原本（データリポジトリの raw/）から表を作り直して読み込む。年ごとにキャッシュする。

正は原本。キャッシュはパーサの版（PARSER_VERSION）と原本の SHA-256 の一覧が一致するときだけ使う。
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd

from . import PARSER_VERSION, cli

TABLES = ("races", "entries", "results", "race_conditions", "payouts")


def _days_in(raw: Path, year: int) -> list[dt.date]:
    days = set()
    for kind in ("B", "K"):
        for p in (raw / kind / str(year)).glob(f"{kind.lower()}*.lzh"):
            days.add(dt.datetime.strptime(p.stem[1:], "%y%m%d").date())
    return sorted(days)


def _fingerprint(raw: Path, year: int) -> str:
    h = hashlib.sha256(PARSER_VERSION.encode())
    for kind in ("B", "K"):
        for p in sorted((raw / kind / str(year)).glob("*.lzh")):
            h.update(p.name.encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def build_year(raw: Path, year: int) -> dict:
    rows = {t: [] for t in TABLES}
    days = []
    for d in _days_in(raw, year):
        b = raw / "B" / str(year) / f"b{d:%y%m%d}.lzh"
        k = raw / "K" / str(year) / f"k{d:%y%m%d}.lzh"
        t = cli.process(d, b.read_bytes() if b.exists() else None, k.read_bytes() if k.exists() else None)
        days.append({"date": d.isoformat(), "day_verdict": t.day_verdict, **t.counts})
        for name in TABLES:
            for r in getattr(t, name):
                rows[name].append({**r, "date": d.isoformat()} if "date" not in r else r)
    out = {name: pd.DataFrame(rows[name]) for name in TABLES}
    out["days"] = pd.DataFrame(days)
    return out


def load(raw_dir: str | Path, years: list[int], cache_dir: str | Path | None = None) -> dict:
    """年をまたいで表を読み込む。date は datetime64 にそろえる。"""
    raw = Path(raw_dir)
    parts = []
    for y in years:
        data = None
        if cache_dir:
            cp = Path(cache_dir) / f"{y}.pkl"
            fp = _fingerprint(raw, y)
            if cp.exists():
                with cp.open("rb") as f:
                    cached = pickle.load(f)
                if cached.get("fingerprint") == fp:
                    data = cached["data"]
            if data is None:
                data = build_year(raw, y)
                cp.parent.mkdir(parents=True, exist_ok=True)
                with cp.open("wb") as f:
                    pickle.dump({"fingerprint": fp, "data": data}, f)
        else:
            data = build_year(raw, y)
        parts.append(data)
    out = {}
    for name in TABLES + ("days",):
        frames = [p[name] for p in parts if len(p[name])]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if "date" in df:
            df["date"] = pd.to_datetime(df["date"])
        out[name] = df
    return out


def summary(data: dict) -> dict:
    d = data["days"]
    return {
        "days": int(len(d)),
        "verdicts": d["day_verdict"].value_counts().to_dict() if len(d) else {},
        "races": int(len(data["races"])),
        "results": int(len(data["results"])),
        "race_status": data["races"]["race_status"].value_counts().to_dict() if len(data["races"]) else {},
    }


if __name__ == "__main__":
    import sys
    data = load(sys.argv[1], [int(y) for y in sys.argv[2].split(",")], sys.argv[3] if len(sys.argv) > 3 else None)
    print(json.dumps(summary(data), ensure_ascii=False, default=str))
