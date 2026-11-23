"""評価（Project 11 §5）。walk-forward で、新モデルと基準を同じ期間・同じ方法で採点する。

  python -m asahi_ai.evaluate --raw RAW_DIR --years 2020-2026 --cache CACHE --out report.json

- 学習：2021-01-01 〜 検証の開始の前日。検証（早期終了・λ の推定）：テスト開始の前の 90 日。テスト：1年（最後は直近まで）。
- 回収率は「確定配当での仮想購入」（各レースで確率最大の3連単を100円）。オッズは使わない。
  返還・不成立・特払いを反映する。予測は番組表の全艇で行い、欠場（結果の情報）は採点のときだけ使う。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import dataset, features, model


def _ticket_return(race_id: str, pick: tuple[int, int, int], tri_pay: dict, absent: set) -> float:
    """3連単1点（100円）の払戻（円）。確定配当での仮想購入。

    - 券種が不成立（全返還）：100
    - 買い目に欠場の艇がいる：返還 100
    - 的中：その配当
    - 特払い（的中が無いときの払戻）：特払いの額
    - それ以外：0
    """
    kinds = tri_pay.get(race_id, [])  # [(payout_kind, combination, 円), ...]
    if any(k == "不成立" for k, _, _ in kinds):
        return 100.0
    if any((race_id, lane) in absent for lane in pick):
        return 100.0
    combo = "-".join(str(x) for x in pick)
    for k, c, y in kinds:
        if k == "hit" and c == combo:
            return float(y or 0)
    for k, _, y in kinds:
        if k == "特払い":
            return float(y or 0)
    return 0.0


def _metrics(pred: pd.DataFrame, test: pd.DataFrame, payouts: pd.DataFrame) -> dict:
    """pred は番組表の全艇（欠場を含む）で作った予測。欠場は採点のときだけ扱う。"""
    t = test.sort_values(["race_id", "lane"])
    winners = t[t["won"] == 1].groupby("race_id")["lane"]
    win = winners.first()[winners.size() == 1]  # 1着の同着のレースは採点から除く（学習と同じ）
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
    hit = (tri.argmax(1) == ai).mean()
    flat_p = pmat.ravel()
    flat_y = (np.arange(1, 7)[None, :] == win.loc[common].to_numpy()[:, None]).ravel()
    bins = np.clip((flat_p * 10).astype(int), 0, 9)
    calib = [{"bin": b, "n": int((bins == b).sum()), "pred": float(flat_p[bins == b].mean()),
              "actual": float(flat_y[bins == b].mean())} for b in range(10) if (bins == b).sum() > 0]
    # 確定配当での仮想購入（3連単の本命1点、100円）。結果のあるレース（completed）すべて
    tp = payouts[payouts["bet_type"] == "trifecta"]
    tri_pay = {}
    for r, k, c, y in zip(tp["race_id"], tp["payout_kind"], tp["combination"], tp["payout_yen"]):
        tri_pay.setdefault(r, []).append((k, c, y))
    absent = set(zip(t.loc[t["result_class"] == "absent", "race_id"], t.loc[t["result_class"] == "absent", "lane"]))
    bet_races = pr.index.intersection(pd.Index(tri_pay.keys()))
    ret = 0.0
    for r in bet_races:
        k = int(np.argmax(pr.at[r, "tri"]))
        ret += _ticket_return(r, tuple(int(x) for x in model.PERMS[k] + 1), tri_pay, absent)
    return {"races": int(len(common)), "win_logloss": float(-np.log(np.clip(pw, 1e-12, 1)).mean()),
            "top1_acc": float(top1), "tri_races": int(len(ai)), "tri_logloss": float(tri_ll),
            "tri_top_hit": float(hit), "roi_tri_top_pick": ret / (100 * len(bet_races)) if len(bet_races) else None,
            "bets": int(len(bet_races)), "calibration": calib}


def _baseline_lane(te: pd.DataFrame, tr: pd.DataFrame) -> pd.DataFrame:
    """場×艇番の1着率（過去の窓）だけの基準。無いときは学習期間の艇番別の1着率で補う（テストの正解は使わない）。"""
    d = te.sort_values(["race_id", "lane"]).copy()
    lane_rate = tr[tr["result_class"].notna() & (tr["result_class"] != "absent")].groupby("lane")["won"].mean()
    rate = d["vl365_win"].where(d["vl365_n"] > 0).fillna(d["lane"].map(lane_rate)).fillna(1 / 6).clip(1e-3, 1)
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
    data = dataset.load(raw, years, cache, slim=True)
    report = {"data": dataset.summary(data), "folds": []}
    h = features.history(data)
    data["race_conditions"] = None  # 評価では使わない（メモリを空ける）
    races = data["races"]
    usable = set(races[(races["race_status"] == "completed") & (~races["quarantined"].astype(bool))]["race_id"])
    del races
    stop_days = set(data["days"].loc[data["days"]["day_verdict"] == "stop", "date"].astype(str).str[:10])
    import gc
    import pickle
    import tempfile
    payouts = data["payouts"]
    tmp = Path(tempfile.mkdtemp(prefix="asahi_feat_", dir=cache))
    # 第1段：lag ごと・年ごとに特徴量を作ってファイルに置く（元の表と同時にメモリに置かないため）
    for lag in lags:
        for y, fy in features.build_years(data, lag, h):
            fy = fy[fy["race_id"].isin(usable) & ~fy["date"].dt.strftime("%Y-%m-%d").isin(stop_days)]
            cols = features.feature_columns(fy)
            fy[cols] = fy[cols].astype("float32")
            with (tmp / f"lag{lag}_{y}.pkl").open("wb") as fh:
                pickle.dump(fy, fh, protocol=pickle.HIGHEST_PROTOCOL)
            del fy
            gc.collect()
    del data, h
    gc.collect()
    # 第2段：特徴量だけを読み込んで学習・評価
    for lag in lags:
        parts = []
        for p in sorted(tmp.glob(f"lag{lag}_*.pkl")):
            with p.open("rb") as fh:
                parts.append(pickle.load(fh))
        f = pd.concat(parts, ignore_index=True)
        del parts
        gc.collect()
        cols = features.feature_columns(f)
        # 番組表だけの基準：過去の結果から作った特徴量（窓の集計・休み明け）と、それを使ったレース内の比較を除く
        hist_prefix = tuple(name + "_" for name, _, _ in features.WINDOWS) + ("rest_days",)
        prog_cols = [c for c in cols if not c.startswith(hist_prefix)]
        for fd in folds(f["date"], first_test, train_start, test_months, valid_days):
            sel = lambda a, b: f[(f["date"] >= a) & (f["date"] <= b)]
            tr, va, te = sel(*fd["train"]), sel(*fd["valid"]), sel(*fd["test"])
            if min(len(tr), len(va), len(te)) == 0:
                continue
            row = {"lag": lag, **{k: [str(x.date()) for x in v] for k, v in fd.items()},
                   "n_train_races": int(tr["race_id"].nunique())}
            m = model.StrengthModel().fit(tr, va, cols)
            row["model"] = _metrics(m.predict(te), te, payouts)
            row["model"]["lambda"] = m.lam
            row["model"]["temp"] = m.temp
            row["model"]["best_iter"] = int(m.booster.best_iteration)
            imp = pd.Series(m.booster.feature_importance("gain"), index=cols).sort_values(ascending=False)
            row["model"]["top_features"] = {k: round(float(v), 1) for k, v in imp.head(15).items()}
            del m
            mb = model.StrengthModel().fit(tr, va, prog_cols)
            row["program_only"] = _metrics(mb.predict(te), te, payouts)
            del mb
            row["lane_baseline"] = _metrics(_baseline_lane(te, tr), te, payouts)
            report["folds"].append(row)
            print("FOLD " + json.dumps({k: (v if k not in ("model", "program_only", "lane_baseline") else
                                            {kk: vv for kk, vv in v.items() if kk != "calibration"})
                                        for k, v in row.items()}, ensure_ascii=False, default=str), flush=True)
            del tr, va, te
            gc.collect()
        del f
        gc.collect()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)
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
