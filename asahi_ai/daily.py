"""P4：毎朝の試験運転（記録だけ。公開・通知はしない）。Project 14・18。

  python -m asahi_ai.daily train   --data DATA --end 2026-11-22 --years 3
  python -m asahi_ai.daily morning --data DATA [--date 2026-11-24]
  python -m asahi_ai.daily score   --data DATA

DATA はデータリポジトリ（asahi-ai-data）の作業ツリー。書き込むもの：
- models/lag{1,2}/model.txt・meta.json
- programs/<年>/<日>.json（朝の番組表の原本）
- predictions/<年>/<日>.jsonl（1レース1行）・features/<年>/<日>.csv.gz（特徴量の値の写し）
- scores/<年>/<日>.json（結果との突き合わせ・公式 B との照合）
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import PARSER_VERSION, archive, dataset, features, fetch, model, morning, parse_b, parse_k
from .layout import LAYOUT_VERSION

JST = dt.timezone(dt.timedelta(hours=9))


def now_jst() -> dt.datetime:
    return dt.datetime.now(JST)


def _code_commit() -> str | None:
    return os.environ.get("ASAHI_CODE_COMMIT")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ---------------------------------------------------------------- 学習（週1回）
def train(data_dir: Path, end: dt.date, years: int, cache: Path | None, lags=(1, 2)) -> list[dict]:
    """end までのデータで lag ごとにモデルを作る。検証は最後の 90 日。"""
    import gc
    raw = data_dir / "raw"
    ys = list(range(end.year - years, end.year + 1))
    data = dataset.load(raw, ys, cache, slim=True)
    races = data["races"]
    usable = set(races[(races["race_status"] == "completed") & (~races["quarantined"].astype(bool))]["race_id"])
    stop_days = set(data["days"].loc[data["days"]["day_verdict"] == "stop", "date"].astype(str).str[:10])
    h = features.history(data)
    metas = []
    for lag in lags:
        parts = []
        for _, fy in features.build_years(data, lag, h):
            fy = fy[fy["race_id"].isin(usable) & ~fy["date"].dt.strftime("%Y-%m-%d").isin(stop_days)
                    & (fy["date"] <= pd.Timestamp(end))]
            parts.append(fy)
        f = pd.concat(parts, ignore_index=True)
        del parts
        gc.collect()
        cols = features.feature_columns(f)
        f[cols] = f[cols].astype("float32")
        v0 = pd.Timestamp(end) - pd.Timedelta(days=89)
        t0 = pd.Timestamp(end.replace(year=end.year - years + 1, month=1, day=1))  # 最初の年は集計の助走
        tr, va = f[(f["date"] >= t0) & (f["date"] < v0)], f[f["date"] >= v0]
        m = model.StrengthModel().fit(tr, va, cols)
        out = data_dir / "models" / f"lag{lag}"
        out.mkdir(parents=True, exist_ok=True)
        m.booster.save_model(str(out / "model.txt"), num_iteration=m.booster.best_iteration)
        txt = (out / "model.txt").read_bytes()
        meta = {"lag": lag, "features": cols, "temp": m.temp, "lam": list(m.lam),
                "train": [str(t0.date()), str((v0 - pd.Timedelta(days=1)).date())],
                "valid": [str(v0.date()), str(end)], "best_iter": int(m.booster.best_iteration),
                "model_sha256": _sha(txt), "code_commit": _code_commit(), "parser_version": PARSER_VERSION,
                "layout_version": LAYOUT_VERSION, "created_at": now_jst().isoformat(timespec="seconds"),
                "n_train_races": int(tr["race_id"].nunique())}
        (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        metas.append(meta)
        print("TRAIN " + json.dumps({k: v for k, v in meta.items() if k != "features"}, ensure_ascii=False), flush=True)
        del f, tr, va, m
        gc.collect()
    return metas


class LoadedModel(model.StrengthModel):
    @classmethod
    def load(cls, d: Path):
        import lightgbm as lgb
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        txt = (d / "model.txt").read_bytes()
        if _sha(txt) != meta["model_sha256"]:
            raise RuntimeError(f"{d} のモデルの SHA-256 が meta と違う")
        m = cls()
        m.booster = lgb.Booster(model_str=txt.decode("utf-8"))
        m.features, m.temp, m.lam, m.meta = meta["features"], meta["temp"], tuple(meta["lam"]), meta
        return m

    def score(self, df):
        return self.temp * self.booster.predict(df[self.features])


# ---------------------------------------------------------------- 毎朝の予測
def _ensure_k(data_dir: Path, d: dt.date) -> bool:
    """前日の K が手元に無ければ取りに行く（1件・3秒の間隔は Fetcher が守る）。"""
    fe = fetch.Fetcher(data_dir / "raw")
    if fe.path_for("K", d).exists():
        return True
    rec, raw = fe.get("K", d)
    from . import store
    store.append_manifest(data_dir, {**rec, "purpose": "morning"})
    return raw is not None


def run_morning(data_dir: Path, d: dt.date, cache: Path | None) -> dict:
    started = now_jst()
    out_pred = data_dir / "predictions" / str(d.year) / f"{d.isoformat()}.jsonl"
    if out_pred.exists():
        return {"date": d.isoformat(), "status": "already_done"}
    status, body, lm, err = morning.fetch(d)
    if body is None:
        return {"date": d.isoformat(), "status": "no_program", "http": status, "error": err}
    pdir = data_dir / "programs" / str(d.year)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / f"{d.isoformat()}.json").write_bytes(body)
    races, entries, issues = morning.to_tables(body, d)
    if races.empty or any(i["level"] == "stop" for i in issues):
        return {"date": d.isoformat(), "status": "program_stop", "issues": issues[:5]}
    k1 = _ensure_k(data_dir, d - dt.timedelta(days=1))
    lag = 1 if k1 else 2
    m = LoadedModel.load(data_dir / "models" / f"lag{lag}")
    # 過去の結果は、予測の時点で手元にあるものだけ（date ≤ d − lag）
    data = dataset.load(data_dir / "raw", list(range(d.year - 2, d.year + 1)), cache, slim=True)
    h = features.history(data)
    h = h[h["date"] <= pd.Timestamp(d - dt.timedelta(days=lag))]
    today = {"races": races.assign(date=pd.to_datetime(races["date"])),
             "entries": entries.assign(date=pd.to_datetime(entries["date"]))}
    f = features.program_features(today)
    f = pd.concat([f, features.window_features(f, h, lag)], axis=1).reset_index(drop=True)
    f = features._finish(f, pd.DataFrame(columns=["race_id", "lane", "finish_position", "result_class"]))
    missing = [c for c in m.features if c not in f.columns]
    if missing:
        return {"date": d.isoformat(), "status": "feature_mismatch", "missing": missing}
    f[m.features] = f[m.features].astype("float32")
    pr = m.predict(f)
    cutoff = now_jst()
    # 特徴量の写し
    fdir = data_dir / "features" / str(d.year)
    fdir.mkdir(parents=True, exist_ok=True)
    fcsv = f[["race_id", "lane", "racer_no"] + m.features].to_csv(index=False).encode("utf-8")
    (fdir / f"{d.isoformat()}.csv.gz").write_bytes(gzip.compress(fcsv, mtime=0))
    k_files = sorted(p.name for p in (data_dir / "raw" / "K").rglob("*.lzh")
                     if p.stem[1:] <= f"{d - dt.timedelta(days=lag):%y%m%d}" and p.stem[1:] >= f"{d - dt.timedelta(days=760):%y%m%d}")
    rows = []
    rinfo = races.set_index("race_id")
    for r in pr.to_dict("records"):
        rid = r["race_id"]
        p1 = [float(r[f"p_win_{i}"]) for i in range(1, 7)]
        tri = [float(x) for x in r["tri"]]
        rows.append({
            "race_id": rid, "deadline_time": rinfo.at[rid, "deadline_time"], "race_closed_at": rinfo.at[rid, "race_closed_at"],
            "prediction_cutoff_at": cutoff.isoformat(timespec="seconds"),
            "started_at": started.isoformat(timespec="seconds"), "lag_used": lag,
            "model_sha256": m.meta["model_sha256"], "model_train": m.meta["train"], "code_commit": _code_commit(),
            "parser_version": PARSER_VERSION, "layout_version": LAYOUT_VERSION,
            "program_source": "boatraceopenapi programs v2", "program_sha256": _sha(body), "program_last_modified": lm,
            "k_files_used": len(k_files), "k_latest": k_files[-1] if k_files else None,
            "feature_snapshot_sha256": _sha(fcsv),
            "p_win": p1, "tri": tri, "win_sum": float(sum(p1)), "tri_sum": float(sum(tri)),
            "sum_ok": bool(abs(sum(p1) - 1) < 1e-6 and abs(sum(tri) - 1) < 1e-6),
        })
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    out_pred.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
    return {"date": d.isoformat(), "status": "ok", "races": len(rows), "lag": lag,
            "sum_ok": all(x["sum_ok"] for x in rows), "issues": len(issues),
            "cutoff": cutoff.isoformat(timespec="seconds")}


# ---------------------------------------------------------------- 毎晩の採点
def score_day(data_dir: Path, d: dt.date) -> dict | None:
    pred_p = data_dir / "predictions" / str(d.year) / f"{d.isoformat()}.jsonl"
    out = data_dir / "scores" / str(d.year) / f"{d.isoformat()}.json"
    bpath = data_dir / "raw" / "B" / str(d.year) / f"b{d:%y%m%d}.lzh"
    kpath = data_dir / "raw" / "K" / str(d.year) / f"k{d:%y%m%d}.lzh"
    if not pred_p.exists() or out.exists() or not (bpath.exists() and kpath.exists()):
        return None
    preds = [json.loads(l) for l in pred_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    from . import build, evaluate
    b = parse_b.parse_b(archive.extract_single(bpath.read_bytes()).data, d)
    k = parse_k.parse_k(archive.extract_single(kpath.read_bytes()).data, d)
    t = build.build_day(d, b, k)
    # 朝の番組表と公式 B の照合
    body = (data_dir / "programs" / str(d.year) / f"{d.isoformat()}.json").read_bytes()
    _, entries, _ = morning.to_tables(body, d)
    cmp = morning.compare_with_b(entries, b.entries)
    # 予測の時刻が締切より後のレースは、採点から外す（事前の予想と言えない）
    races = pd.DataFrame(t.races)
    ok_status = set(races.loc[(races["race_status"] == "completed") & (~races["quarantined"]), "race_id"])
    rows, late = [], 0
    for p in preds:
        closed = p.get("race_closed_at")
        if closed and p["prediction_cutoff_at"][:19].replace("T", " ") >= closed:
            late += 1
            continue
        if p["race_id"] in ok_status:
            rows.append(p)
    res = pd.DataFrame(t.results)
    res["won"] = (res["finish_position"] == 1).astype(int)
    pr = pd.DataFrame({"race_id": [p["race_id"] for p in rows],
                       **{f"p_win_{i}": [p["p_win"][i - 1] for p in rows] for i in range(1, 7)},
                       "tri": [np.array(p["tri"]) for p in rows]})
    te = res[res["race_id"].isin(set(pr["race_id"]))]
    met = evaluate._metrics(pr, te, pd.DataFrame(t.payouts)) if len(pr) else {}
    met.pop("calibration", None)
    rec = {"date": d.isoformat(), "scored_races": len(rows), "excluded_after_deadline": late,
           "program_vs_b": cmp, "metrics": met, "scored_at": now_jst().isoformat(timespec="seconds")}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, ensure_ascii=False, default=str, indent=1), encoding="utf-8")
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="asahi_ai.daily")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--end")
    t.add_argument("--years", type=int, default=3)
    t.add_argument("--cache")
    mo = sub.add_parser("morning")
    mo.add_argument("--data", required=True)
    mo.add_argument("--date")
    mo.add_argument("--cache")
    sc = sub.add_parser("score")
    sc.add_argument("--data", required=True)
    a = ap.parse_args(argv)
    data = Path(a.data)
    today = now_jst().date()
    if a.cmd == "train":
        end = dt.date.fromisoformat(a.end) if a.end else today - dt.timedelta(days=1)
        train(data, end, a.years, Path(a.cache) if a.cache else None)
    elif a.cmd == "morning":
        d = dt.date.fromisoformat(a.date) if a.date else today
        r = run_morning(data, d, Path(a.cache) if a.cache else None)
        print("MORNING " + json.dumps(r, ensure_ascii=False, default=str), flush=True)
        return 0 if r["status"] in ("ok", "already_done", "no_program") else 2
    elif a.cmd == "score":
        for p in sorted((data / "predictions").rglob("*.jsonl")):
            r = score_day(data, dt.date.fromisoformat(p.stem))
            if r:
                print("SCORE " + json.dumps({k: v for k, v in r.items() if k != "program_vs_b"} |
                                            {"program_vs_b_diff": r["program_vs_b"]["diff"]}, ensure_ascii=False,
                                            default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
