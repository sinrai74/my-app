"""評価（Project 11 §5）。walk-forward で、新モデルと基準を同じ期間・同じ方法で採点する。

  python -m asahi_ai.evaluate --raw RAW_DIR --years 2020-2026 --cache CACHE --out report.json

- 学習：2021-01-01 〜 検証の開始の前日。検証（早期終了・λ の推定）：テスト開始の前の 90 日。テスト：1年（最後は直近まで）。
- 回収率は「確定配当での仮想購入」（各レースで確率最大の3連単を100円）。オッズは使わない。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from . import dataset, features, model


def _metrics(pred: pd.DataFrame, test: pd.DataFrame, payouts: pd.DataFrame) -> dict:
    t = test.sort_values(["race_id", "lane"])
    win = t[t["won"] == 1].drop_duplicates("race_id").set_index("race_id")["lane"]
    pr = pred.set_index("race_id")
    common = win.index.intersection(pr.index)
    pw = np.array([pr.at[r, f"p_win_{int(win[r])}"] for r in common])
    pmat = pr.loc[common, [f"p_win_{i}" for i in range(1, 7)]].to_numpy()
    top1 = (pmat.argmax(1) + 1 == win.loc[common].to_numpy()).mean()
    act = model.actual_trifecta(t)
    ac = act.index.intersection(pr.index)
    tri = np.stack(pr.loc[ac, "tri"].to_numpy())
    ai = act.loc[ac].to_numpy().astype(int)
    tri_ll = -np.log(np.clip(tri[np.arange(len(ai)), ai], 1e-12, 1)).mean()
    pick = tri.argmax(1)
    hit = (pick == ai).mean()
    # 較正（1着の確率、10区分）
    flat_p = pmat.ravel()
    flat_y = (np.arange(1, 7)[None, :] == win.loc[common].to_numpy()[:, None]).ravel()
    bins = np.clip((flat_p * 10).astype(int), 0, 9)
    calib = [{"bin": b, "n": int((bins == b).sum()), "pred": float(flat_p[bins == b].mean()),
              "actual": float(flat_y[bins == b].mean())} for b in range(10) if (bins == b).sum() > 0]
    # 確定配当での仮想購入（3連単の本命1点、100円）
    tp = payouts[(payouts["bet_type"] == "trifecta") & (payouts["payout_kind"] == "hit")]
    paymap = {(r, c): y for r, c, y in zip(tp["race_id"], tp["combination"], tp["payout_yen"])}
    all_r = pr.index.intersection(t["race_id"].unique())
    tri_all = np.stack(pr.loc[all_r, "tri"].to_numpy())
    picks = tri_all.argmax(1)
    ret = 0.0
    for r, k in zip(all_r, picks):
        a, b, c = model.PERMS[k] + 1
        ret += float(paymap.get((r, f"{a}-{b}-{c}"), 0) or 0)
    return {"races": int(len(common)), "win_logloss": float(-np.log(np.clip(pw, 1e-12, 1)).mean()),
            "top1_acc": float(top1), "tri_races": int(len(ai)), "tri_logloss": float(tri_ll),
            "tri_top_hit": float(hit), "roi_tri_top_pick": ret / (100 * len(all_r)) if len(all_r) else None,
            "bets": int(len(all_r)), "calibration": calib}


def _baseline_lane(f: pd.DataFrame) -> pd.DataFrame:
    d = f.sort_values(["race_id", "lane"]).copy()
    rate = d["vl365_win"].fillna(d.groupby("lane")["won"].transform("mean")).clip(1e-3, 1)
    d["score"] = np.log(rate)
    return model.race_probabilities(d, (1.0, 1.0))


def folds(dates: pd.Series, first_test: str, train_start: str = "2021-01-01", test_months: int = 12,
          valid_days: int = 90):
    last = dates.max()
    ts = pd.Timestamp(first_test)
    while ts <= last:
        te = min(ts + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), last)
        yield {"train": (pd.Timestamp(train_start), ts - pd.Timedelta(days=valid_days + 1)),
               "valid": (ts - pd.Timedelta(days=valid_days), ts - pd.Timedelta(days=1)), "test": (ts, te)}
        ts = ts + pd.DateOffset(months=test_months)


def run(raw: str, years: list[int], cache: str | None, first_test: str, lags=(1, 2),
        train_start: str = "2021-01-01", test_months: int = 12, valid_days: int = 90) -> dict:
    t0 = time.time()
    data = dataset.load(raw, years, cache)
    report = {"data": dataset.summary(data), "folds": []}
    h = features.history(data)
    races = data["races"]
    usable = races[(races["race_status"] == "completed") & (~races["quarantined"].astype(bool))]["race_id"]
    stop_days = set(data["days"].loc[data["days"]["day_verdict"] == "stop", "date"].astype(str).str[:10])
    for lag in lags:
        f = features.build(data, lag=lag, h=h)
        f = f[f["race_id"].isin(set(usable)) & ~f["date"].dt.strftime("%Y-%m-%d").isin(stop_days)]
        cols = features.feature_columns(f)
        prog_cols = [c for c in cols if not any(c.startswith(p) for p in ("r180", "r365", "rl365", "rv730", "m120", "vl365"))]
        for fd in folds(f["date"], first_test, train_start, test_months, valid_days):
            sel = lambda a, b: f[(f["date"] >= a) & (f["date"] <= b)]
            tr, va, te = sel(*fd["train"]), sel(*fd["valid"]), sel(*fd["test"])
            if min(len(tr), len(va), len(te)) == 0:
                continue
            row = {"lag": lag, **{k: [str(x.date()) for x in v] for k, v in fd.items()},
                   "n_train_races": int(tr["race_id"].nunique())}
            m = model.StrengthModel().fit(tr, va, cols)
            row["model"] = _metrics(m.predict(te[te["result_class"] != "absent"]), te, data["payouts"])
            row["model"]["lambda"] = m.lam
            row["model"]["best_iter"] = int(m.booster.best_iteration)
            imp = pd.Series(m.booster.feature_importance("gain"), index=cols).sort_values(ascending=False)
            row["model"]["top_features"] = {k: round(float(v), 1) for k, v in imp.head(15).items()}
            mb = model.StrengthModel().fit(tr, va, prog_cols)
            row["program_only"] = _metrics(mb.predict(te[te["result_class"] != "absent"]), te, data["payouts"])
            row["lane_baseline"] = _metrics(_baseline_lane(te[te["result_class"] != "absent"]), te, data["payouts"])
            report["folds"].append(row)
            print("FOLD " + json.dumps({k: (v if k not in ("model", "program_only", "lane_baseline") else
                                            {kk: vv for kk, vv in v.items() if kk != "calibration"})
                                        for k, v in row.items()}, ensure_ascii=False, default=str), flush=True)
    report["seconds"] = round(time.time() - t0)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--years", required=True, help="例 2020-2026")
    ap.add_argument("--cache")
    ap.add_argument("--first-test", default="2023-01-01")
    ap.add_argument("--train-start", default="2021-01-01")
    ap.add_argument("--test-months", type=int, default=12)
    ap.add_argument("--valid-days", type=int, default=90)
    ap.add_argument("--lags", default="1,2")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    y0, y1 = (int(x) for x in a.years.split("-"))
    rep = run(a.raw, list(range(y0, y1 + 1)), a.cache, a.first_test, tuple(int(x) for x in a.lags.split(",")),
              a.train_start, a.test_months, a.valid_days)
    s = json.dumps(rep, ensure_ascii=False, default=str, indent=1)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(s)
    print("REPORT_DONE", rep["seconds"])


if __name__ == "__main__":
    main()
