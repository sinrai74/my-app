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


_CODE_FILES = ("layout.py", "archive.py", "codes.py", "parse_b.py", "parse_k.py", "build.py", "cli.py", "dataset.py")


def code_fingerprint() -> str:
    """解析に関わるソースの SHA-256（版の番号の上げ忘れでも、古い表を使わないように）。"""
    h = hashlib.sha256(PARSER_VERSION.encode())
    base = Path(__file__).resolve().parent
    for name in _CODE_FILES:
        h.update(name.encode())
        h.update((base / name).read_bytes())
    return h.hexdigest()


def _fingerprint(raw: Path, year: int) -> str:
    h = hashlib.sha256(code_fingerprint().encode())
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


# 学習・評価で使う列だけに絞る（7年分を 7GB のメモリに収めるため）
SLIM = {
    "races": ["race_id", "date", "venue", "race_no", "distance_m", "meet_day", "deadline_time", "race_status",
              "quarantined"],
    "entries": ["race_id", "date", "lane", "racer_no", "age", "weight", "class", "nat_win", "nat_2r", "loc_win",
                "loc_2r", "motor_no", "motor_2r", "boat_no", "boat_2r", "series_orders"],
    "results": ["race_id", "date", "lane", "racer_no", "finish_position", "result_class", "course", "st_value",
                "st_flag", "exhibition_time", "motor_no"],
    "race_conditions": ["race_id", "weather", "wind_dir", "wind_speed_m", "wave_cm", "winning_technique"],
    "payouts": ["race_id", "bet_type", "combination", "payout_yen", "popularity", "payout_kind"],
}
CATEGORY: set[str] = set()  # 文字列の列はそのまま（category にすると groupby・map の挙動が変わるため）


def _slim(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if name not in SLIM or df.empty:
        return df
    df = df[[c for c in SLIM[name] if c in df.columns]].copy()
    if name == "payouts":  # 評価で使うのは3連単だけ
        df = df[df["bet_type"] == "trifecta"]
    for c in df.columns:
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
    return df


def load(raw_dir: str | Path, years: list[int], cache_dir: str | Path | None = None, slim: bool = False) -> dict:
    """年をまたいで表を読み込む。date は datetime64 にそろえる。slim=True で学習・評価に要る列だけにする。"""
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
        if slim:
            data = {k: _slim(k, v) for k, v in data.items()}
        parts.append(data)
    out = {}
    for name in TABLES + ("days",):
        frames = [p[name] for p in parts if len(p[name])]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if slim:
            for c in df.columns:
                if c in CATEGORY and df[c].dtype != "category":
                    df[c] = df[c].astype("category")
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
