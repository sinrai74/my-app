"""特徴量（Project 11 §3）。時点の規則（05 §7・11 §2）を守る。

レース日 D の特徴量に使う過去の結果（K）は、date ≤ D − lag の日のものだけ。
窓（window）つきの集計は「累積和の差」で求める（merge_asof、日付の後ろ向き一致）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_MAP = {"A1": 4, "A2": 3, "B1": 2, "B2": 1}
SUM_COLS = ["n", "win", "top2", "top3", "fin_sum", "st_sum", "st_sq", "st_n", "f_cnt", "late_cnt", "mae_cnt",
            "ex_rank_sum", "ex_n", "cdiff_sum", "cdiff_n"]

# (名前, キー, 窓の日数)
WINDOWS = [
    ("r30", ["racer_no"], 30),
    ("r180", ["racer_no"], 180),
    ("r365", ["racer_no"], 365),
    ("rl365", ["racer_no", "lane"], 365),
    ("rv730", ["racer_no", "venue"], 730),
    ("m60", ["venue", "motor_no"], 60),
    ("m120", ["venue", "motor_no"], 120),
    ("vl365", ["venue", "lane"], 365),
]


def history(data: dict) -> pd.DataFrame:
    """過去の結果の1艇1行（集計の元）。出走した艇だけ（欠場は除く）。"""
    races = data["races"]
    ok_days = set(data["days"].loc[data["days"]["day_verdict"] != "stop", "date"].astype(str).str[:10])
    rr = races[(races["race_status"] == "completed") & (~races["quarantined"].astype(bool))]
    rr = rr[rr["date"].dt.strftime("%Y-%m-%d").isin(ok_days)]  # stop の日は使わない（全日 stop なら何も使わない）
    h = data["results"].merge(rr[["race_id", "date", "venue"]], on="race_id", how="inner", suffixes=("_x", ""))
    if "date_x" in h:
        h = h.drop(columns=["date_x"])
    h = h[h["result_class"] != "absent"].copy()
    fp = pd.to_numeric(h["finish_position"], errors="coerce")
    h["n"] = 1
    h["win"] = (fp == 1).astype(int)
    h["top2"] = (fp <= 2).astype(int)
    h["top3"] = (fp <= 3).astype(int)
    h["fin_sum"] = fp.fillna(7)  # 着外は 7 として平均に入れる
    stv = pd.to_numeric(h["st_value"], errors="coerce")
    normal_st = (h["st_flag"].fillna("") == "") & stv.notna()
    h["st_sum"] = np.where(normal_st, stv, 0.0)
    h["st_sq"] = np.where(normal_st, stv ** 2, 0.0)
    h["st_n"] = normal_st.astype(int)
    h["f_cnt"] = (h["st_flag"] == "F").astype(int)
    h["late_cnt"] = (h["st_flag"] == "L").astype(int)
    ex = pd.to_numeric(h["exhibition_time"], errors="coerce")
    h["ex_rank"] = ex.groupby(h["race_id"]).rank(method="average")
    h["ex_rank_sum"] = h["ex_rank"].fillna(0)
    h["ex_n"] = h["ex_rank"].notna().astype(int)
    course = pd.to_numeric(h["course"], errors="coerce")
    h["cdiff_sum"] = (course - h["lane"]).fillna(0)
    h["mae_cnt"] = (course < h["lane"]).astype(int)  # 前付け（艇番より内のコースに入った）
    h["cdiff_n"] = course.notna().astype(int)
    h["motor_no"] = pd.to_numeric(h["motor_no"], errors="coerce")
    return h


def _cum(h: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    daily = h.groupby(keys + ["date"], sort=False)[SUM_COLS].sum().reset_index()
    daily = daily.sort_values(keys + ["date"])
    daily[SUM_COLS] = daily.groupby(keys, sort=False)[SUM_COLS].cumsum()
    return daily.sort_values("date")


def _asof(target: pd.DataFrame, cum: pd.DataFrame, keys: list[str], on: str) -> pd.DataFrame:
    """target の各行について、date ≤ target[on] の最新の累積値（無ければ 0）。"""
    t = target[keys + [on, "_row"]].dropna(subset=keys).sort_values(on).copy()
    c = cum.rename(columns={"date": on}).copy()
    for k in keys:  # キーの型をそろえる（文字列のキーと、数値のキー）
        if k in ("racer_no", "venue"):
            t[k], c[k] = t[k].astype(str), c[k].astype(str)
        else:
            t[k], c[k] = t[k].astype("float64"), c[k].astype("float64")
    t[on], c[on] = t[on].astype("datetime64[ns]"), c[on].astype("datetime64[ns]")
    m = pd.merge_asof(t, c, on=on, by=keys, direction="backward", allow_exact_matches=True)
    return m.set_index("_row")[SUM_COLS].reindex(target["_row"]).fillna(0).to_numpy()


def window_features(target: pd.DataFrame, h: pd.DataFrame, lag: int) -> pd.DataFrame:
    """target：1艇1行（race_id・date・venue・lane・racer_no・motor_no）。"""
    t = target.copy()
    t["_row"] = np.arange(len(t))
    t["_end"] = t["date"] - pd.Timedelta(days=lag)
    out = {}
    for name, keys, days in WINDOWS:
        cum = _cum(h, keys)
        t["_start"] = t["_end"] - pd.Timedelta(days=days)
        hi = _asof(t, cum, keys, "_end")
        lo = _asof(t, cum, keys, "_start")
        s = pd.DataFrame(hi - lo, columns=SUM_COLS)
        n = s["n"].replace(0, np.nan)
        out[f"{name}_n"] = s["n"]
        out[f"{name}_win"] = s["win"] / n
        out[f"{name}_top2"] = s["top2"] / n
        out[f"{name}_top3"] = s["top3"] / n
        if name in ("r30", "r180", "r365", "rv730", "rl365"):
            out[f"{name}_fin"] = s["fin_sum"] / n
        if name in ("r180", "r365"):
            stn = s["st_n"].replace(0, np.nan)
            mean = s["st_sum"] / stn
            out[f"{name}_st"] = mean
            out[f"{name}_st_sd"] = np.sqrt(np.clip(s["st_sq"] / stn - mean ** 2, 0, None))
            out[f"{name}_f"] = s["f_cnt"]
            out[f"{name}_late"] = s["late_cnt"]
            out[f"{name}_exrank"] = s["ex_rank_sum"] / s["ex_n"].replace(0, np.nan)
            out[f"{name}_cdiff"] = s["cdiff_sum"] / s["cdiff_n"].replace(0, np.nan)
            out[f"{name}_mae"] = s["mae_cnt"] / s["cdiff_n"].replace(0, np.nan)
        if name in ("m60", "m120"):
            out[f"{name}_exrank"] = s["ex_rank_sum"] / s["ex_n"].replace(0, np.nan)
    # 前に走った日からの日数（休み明け）。lag の日までの最後の出走日
    last = h[["racer_no", "date"]].drop_duplicates().rename(columns={"date": "_last"})
    last["_end"] = last["_last"]
    tt = t[["racer_no", "_end", "_row"]].dropna(subset=["racer_no"]).copy()
    tt["racer_no"], last["racer_no"] = tt["racer_no"].astype(str), last["racer_no"].astype(str)
    tt["_end"], last["_end"] = tt["_end"].astype("datetime64[ns]"), last["_end"].astype("datetime64[ns]")
    m = pd.merge_asof(tt.sort_values("_end"), last.sort_values("_end"), on="_end", by="racer_no", direction="backward")
    gap = (m.set_index("_row")["_end"] - m.set_index("_row")["_last"]).dt.days
    out["rest_days"] = gap.reindex(t["_row"]).to_numpy() + lag
    return pd.DataFrame(out, index=target.index)


def program_features(data: dict) -> pd.DataFrame:
    """番組表（B）から、レース日の朝に分かる項目。"""
    e = data["entries"].merge(
        data["races"][["race_id", "date", "venue", "race_no", "distance_m", "meet_day", "deadline_time"]],
        on="race_id", how="inner", suffixes=("_x", ""))
    if "date_x" in e:
        e = e.drop(columns=["date_x"])
    f = pd.DataFrame(index=e.index)
    for c in ["race_id", "date", "venue", "race_no", "lane", "racer_no", "motor_no"]:
        f[c] = e[c]
    f["motor_no"] = pd.to_numeric(f["motor_no"], errors="coerce")
    f["venue_code"] = pd.to_numeric(e["venue"], errors="coerce")
    f["cls"] = e["class"].map(CLASS_MAP)
    for c in ["age", "weight", "nat_win", "nat_2r", "loc_win", "loc_2r", "motor_2r", "boat_2r",
              "distance_m", "meet_day"]:
        f[c] = pd.to_numeric(e[c], errors="coerce")
    f["deadline_hour"] = pd.to_numeric(e["deadline_time"].astype(str).str[:2], errors="coerce")
    ser = e["series_orders"].fillna("").astype(str)
    digits = ser.apply(lambda s: [int(ch) for ch in s if ch.isdigit() and ch != "0"])
    f["ser_n"] = ser.str.len()
    f["ser_mean"] = digits.apply(lambda x: np.mean(x) if x else np.nan)
    f["ser_win"] = digits.apply(lambda x: sum(1 for v in x if v == 1))
    f["ser_bad"] = ser.apply(lambda s: sum(1 for ch in s if not ch.isdigit()))
    return f


RELATIVE = ["nat_win", "nat_2r", "loc_win", "motor_2r", "cls", "r30_top2", "r180_win", "r180_top2", "r180_st",
            "r180_fin", "rl365_top2", "r365_st", "m60_top2", "ser_mean", "rl365_win",
            "r180_exrank", "m120_top2"]


def build(data: dict, lag: int, h: pd.DataFrame | None = None) -> pd.DataFrame:
    """全レースの1艇1行の特徴量と正解。"""
    if h is None:
        h = history(data)
    f = program_features(data)
    w = window_features(f, h, lag)
    f = pd.concat([f, w], axis=1)
    for c in RELATIVE:
        g = f.groupby("race_id")[c]
        f[f"{c}_rank"] = g.rank(ascending=False, method="average")
        f[f"{c}_diff"] = f[c] - g.transform("mean")
    res = data["results"][["race_id", "lane", "finish_position", "result_class"]]
    f = f.merge(res, on=["race_id", "lane"], how="left")
    f["finish_position"] = pd.to_numeric(f["finish_position"], errors="coerce")
    f["won"] = (f["finish_position"] == 1).astype(int)
    return f


FEATURES_EXCLUDE = {"race_id", "date", "venue", "racer_no", "motor_no", "finish_position", "result_class", "won"}


def feature_columns(f: pd.DataFrame) -> list[str]:
    return [c for c in f.columns if c not in FEATURES_EXCLUDE]
