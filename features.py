"""
features.py  ── 特徴量エンジニアリング / 動的EV_MIN / バリュー判定 / 時間帯補正
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd

from config import (
    BASE_CONFIG, get_venue_config, NIGHT_RACE_VENUES, VALUE_THRESHOLD
)

# ════════════════════════════════════════════════════════════
# 動的 EV_MIN
# ════════════════════════════════════════════════════════════

def compute_dynamic_ev_min(
    k_df: pd.DataFrame,
    venue_code: str,
    base_ev_min: float,
    arashi_weight: float = 0.8,
    min_ev: float = 1.2,
    max_ev: float = 2.5,
) -> float:
    """
    過去結果から場ごとの「荒れ率」を計算し EV_MIN を動的調整する。

    荒れ率 = 払戻 >= 5000円のレース割合
    荒れやすい場 → EV_MIN を上げる（低オッズ本命軸を避ける）
    荒れにくい場 → EV_MIN を下げる（安定イン軸を積極的に狙う）
    """
    vc_df = k_df[k_df["場コード"] == str(venue_code).zfill(2)]
    if len(vc_df) < 10:
        return base_ev_min

    arashi_rate = (vc_df["払戻"].fillna(0) >= 5000).mean()
    delta       = (arashi_rate - 0.3) * arashi_weight
    dynamic_ev  = round(float(np.clip(base_ev_min + delta, min_ev, max_ev)), 3)

    cfg  = get_venue_config(venue_code)
    name = cfg.get("name", venue_code)
    avg  = vc_df["払戻"].clip(upper=50000).mean()
    print(f"  [{name}] 荒れ率={arashi_rate:.3f} 平均払戻={avg:.0f} "
          f"→ 動的EV_MIN: {base_ev_min} → {dynamic_ev}")
    return dynamic_ev


def compute_all_dynamic_ev(k_df: pd.DataFrame) -> dict[str, float]:
    """全場の動的 EV_MIN を一括計算"""
    results: dict[str, float] = {}
    for vc in k_df["場コード"].unique():
        cfg = get_venue_config(vc)
        results[vc] = compute_dynamic_ev_min(k_df, vc, cfg["EV_MIN"])
    return results


# ════════════════════════════════════════════════════════════
# バリューベット判定
# ════════════════════════════════════════════════════════════

def compute_implied_prob(odds: float) -> float:
    """控除率 25% 補正済みインプライドプロバビリティ"""
    if odds <= 1.0:
        return 1.0
    return (1.0 / odds) * 0.75


def is_value_bet(model_prob: float, odds: float,
                 threshold: float = VALUE_THRESHOLD) -> bool:
    """model_prob > implied_prob × threshold ならバリューベット"""
    if odds <= 1.0 or model_prob <= 0:
        return False
    return model_prob > compute_implied_prob(odds) * threshold


def compute_value_score(model_prob: float, odds: float) -> float:
    """バリュースコア = model_prob / implied_prob（1.0超でバリュー）"""
    implied = compute_implied_prob(odds)
    if implied <= 0:
        return 0.0
    return round(model_prob / implied, 3)


# ════════════════════════════════════════════════════════════
# 時間帯・気象補正
# ════════════════════════════════════════════════════════════

def is_night_race(venue_code: str, race_no: int) -> bool:
    return (str(venue_code).zfill(2) in NIGHT_RACE_VENUES) and (race_no >= 7)


def compute_time_correction(
    venue_code: str,
    race_no: int,
    node_day: int = 1,
    wind_speed: float = 0.0,
    wind_direction: str = "",
    tide_level: str = "",
    cfg: dict | None = None,
) -> dict:
    """
    会場×時間帯補正を計算。

    Returns dict with keys:
        night_flag, combined_in_adj, combined_outer_adj, ...
    """
    if cfg is None:
        cfg = get_venue_config(venue_code)

    night = is_night_race(venue_code, race_no)

    night_in_adj    = 1.04 if night else 1.00
    night_outer_adj = 0.95 if night else 1.00

    wind_outer = wind_in = 1.0
    if wind_speed > 0 and wind_direction:
        s = cfg.get("風補正感度", BASE_CONFIG["風補正感度"])
        if "追" in wind_direction:
            wind_outer = 1.0 + wind_speed * s
            wind_in    = 1.0 - wind_speed * s * 0.5
        elif "向" in wind_direction:
            wind_in    = 1.0 + wind_speed * s * 0.7
            wind_outer = 1.0 - wind_speed * s * 0.3
        elif "横" in wind_direction:
            wind_outer = 1.0 + wind_speed * s * 0.5

    tide_in = tide_outer = 1.0
    if tide_level:
        boost = cfg.get("干潮補正", BASE_CONFIG["干潮補正"])
        if "干" in tide_level:
            tide_in    = 1.0 + boost
            tide_outer = 1.0 - boost * 0.5
        elif "満" in tide_level:
            tide_in    = 1.0 - boost * 0.5
            tide_outer = 1.0 + boost

    return {
        "night_flag":         int(night),
        "night_in_adj":       round(night_in_adj,    3),
        "night_outer_adj":    round(night_outer_adj, 3),
        "wind_outer_boost":   round(wind_outer,      3),
        "wind_in_adj":        round(wind_in,         3),
        "tide_in_boost":      round(tide_in,         3),
        "tide_outer_adj":     round(tide_outer,      3),
        "combined_in_adj":    round(night_in_adj * wind_in * tide_in,       3),
        "combined_outer_adj": round(night_outer_adj * wind_outer * tide_outer, 3),
    }


# ════════════════════════════════════════════════════════════
# 今節成績パーサー
# ════════════════════════════════════════════════════════════

def _parse_today_results(s: str | None, node_day, race_no) -> dict:
    empty = {
        "今節出走数": 0, "今節F": 0, "今節L": 0, "今節K": 0,
        "今節連対率": 0.0, "今節平均着順": 3.5,
        "今節1着回数": 0, "今節直近3連対率": 0.0,
    }
    if not isinstance(s, str) or not s.strip():
        return empty
    text = s.strip()
    all_places = [int(d) for d in re.findall(r"[1-6]", text)]
    max_usable = (
        max(0, (int(node_day) - 1) * 12)
        if node_day and not (isinstance(node_day, float) and np.isnan(node_day))
        else 0
    )
    places = all_places[-max_usable:] if max_usable > 0 else []
    n  = len(places)
    t2 = sum(1 for p in places if p <= 2)
    r3 = places[-3:] if len(places) >= 3 else places
    return {
        "今節出走数":      n,
        "今節F":          text.count("F"),
        "今節L":          text.count("L"),
        "今節K":          text.count("K"),
        "今節連対率":     round(t2 / n, 3) if n > 0 else 0.0,
        "今節平均着順":   round(float(np.mean(places)), 2) if places else 3.5,
        "今節1着回数":    sum(1 for p in places if p == 1),
        "今節直近3連対率":round(sum(1 for p in r3 if p <= 2) / len(r3), 3) if r3 else 0.0,
    }


def estimate_odds(ninki_series: pd.Series) -> pd.Series:
    filled = ninki_series.fillna(10)
    return (np.exp(1.182 * np.log1p(filled) + 5.214) / 100).clip(lower=1.0)


# ════════════════════════════════════════════════════════════
# 特徴量エンジニアリング（全場統合）
# ════════════════════════════════════════════════════════════

FEATURE_COLS: list[str] = [
    "年齢","体重","級_数値","全国勝率","全国2率","当地勝率","当地2率",
    "今期能力指数","前期能力指数","平均ST","出走回数","優出率","優勝率",
    "モーター2率","ボート2率","艇番","予想進入","進入強度",
    "進入コースST","進入コース複勝率","進入コース進入割合","コース補正",
    "1コース複勝率","2コース複勝率","3コース複勝率","4コース複勝率","5コース複勝率","6コース複勝率",
    "今節出走数","今節F","今節L","今節K","今節連対率","今節平均着順","今節1着回数","今節直近3連対率",
    "人気_filled","推定オッズ","人気_IN順位","人気_逆数","人気差",
    "人気×勝率","人気×モーター","人気×能力指数",
    "勝率順位差","モーター順位差","能力順位差","ST順位差",
    "イン対抗指数","イン信頼度",
    "全国勝率_IN順位","全国2率_IN順位","当地勝率_IN順位","当地2率_IN順位",
    "今期能力指数_IN順位","平均ST_IN順位","モーター2率_IN順位","ボート2率_IN順位",
    "今節連対率_IN順位","進入コース複勝率_IN順位","当地スコア_IN順位",
    "全国勝率_IN偏差","全国勝率_MAX差","全国勝率_MIN差",
    "モーター2率_IN偏差","モーター2率_MAX差","モーター2率_MIN差",
    "今期能力指数_IN偏差","今期能力指数_MAX差","今期能力指数_MIN差",
    "当地勝率_IN偏差","当地勝率_MAX差","当地勝率_MIN差",
    "当地2率_IN偏差","当地2率_MAX差","当地2率_MIN差",
    "今節連対率_IN偏差","今節連対率_MAX差","今節連対率_MIN差",
    "進入コース複勝率_IN偏差","進入コース複勝率_MAX差","進入コース複勝率_MIN差",
    "モーター偏差","展開クラスタ","クラスタ別勝率",
    "ST優位度","ST不利度","イン崩壊指数","イン崩壊フラグ","過小評価フラグ",
    "荒れ指数","ST拮抗度","内有利度","外まくり力","インST","節日","初日フラグ",
    # 統合新規
    "当地スコア","モータースコア","インプライドプロブ",
    "ナイターフラグ","動的EV_MIN",
]


def engineer_features(
    df: pd.DataFrame,
    wind_speed: float = 0.0,
    wind_direction: str = "",
    tide_level: str = "",
    dynamic_ev_map: dict[str, float] | None = None,
) -> pd.DataFrame:
    """全場統合版特徴量エンジニアリング"""
    df = df.copy()
    df["級_数値"] = df["級"].map({"A1":4,"A2":3,"B1":2,"B2":1}).fillna(0)

    # 予想進入コース
    entry_cols = [f"{c}コース進入回数" for c in range(1, 7)]
    existing   = [c for c in entry_cols if c in df.columns]
    if existing:
        ec    = df[existing].fillna(0)
        total = ec.sum(axis=1)
        best  = ec.idxmax(axis=1).str.extract(r"(\d)コース").astype(float).squeeze()
        df["予想進入"] = np.where(total >= 5, best, df["艇番"]).astype(int)
    else:
        df["予想進入"] = df["艇番"].astype(int)
    df["進入強度"] = (df["艇番"] - df["予想進入"]).astype(int)

    # 進入コース別 ST / 複勝率 / 進入割合
    c_idx   = df["予想進入"].astype(int).clip(1, 6)
    st_vals = np.array([
        df.get(f"{c}コース平均ST", pd.Series(0, index=df.index)).fillna(0).values
        for c in range(1, 7)
    ])
    chosen_st       = st_vals[c_idx.values - 1, np.arange(len(df))]
    df["進入コースST"] = np.where(chosen_st > 0.05, chosen_st, df["平均ST"].fillna(0.18).values)

    w2_vals = np.array([
        df.get(f"{c}コース複勝率", pd.Series(0, index=df.index)).fillna(0).values
        for c in range(1, 7)
    ])
    df["進入コース複勝率"] = w2_vals[c_idx.values - 1, np.arange(len(df))]

    in_vals   = np.array([
        df.get(f"{c}コース進入回数", pd.Series(0, index=df.index)).fillna(0).values
        for c in range(1, 7)
    ])
    chosen_in = in_vals[c_idx.values - 1, np.arange(len(df))]
    total_in  = in_vals.sum(axis=0)
    df["進入コース進入割合"] = np.where(
        total_in > 0, np.round(chosen_in / total_in.clip(1), 3), 0.0)

    # ── 場ごとのコース補正（時間帯補正込み）──────────────────
    def _course_corr(row) -> float:
        vc  = row["場コード"]
        cfg = get_venue_config(vc)
        tc  = compute_time_correction(
            vc, row.get("レースNo", 6),
            node_day=int(row.get("節日", 1) or 1),
            wind_speed=wind_speed, wind_direction=wind_direction,
            tide_level=tide_level, cfg=cfg,
        )
        c    = int(row.get("予想進入", row.get("艇番", 1)))
        base = cfg.get(f"{c}コース補正", 1.0)
        if c == 1:
            return base * tc["combined_in_adj"]
        elif c >= 4:
            return base * tc["combined_outer_adj"]
        return base

    df["コース補正"] = df.apply(_course_corr, axis=1)

    # ナイターフラグ
    df["ナイターフラグ"] = df.apply(
        lambda r: int(is_night_race(r["場コード"], r.get("レースNo", 1))), axis=1)

    # 今節成績
    parsed = df.apply(
        lambda r: _parse_today_results(
            r["今節成績"],
            None if (isinstance(r.get("節日"), float) and np.isnan(r.get("節日", np.nan)))
                 else r.get("節日"),
            r.get("レースNo", 1),
        ), axis=1
    )
    for key in ["今節出走数","今節F","今節L","今節K","今節連対率","今節平均着順","今節1着回数","今節直近3連対率"]:
        df[key] = parsed.apply(lambda x: x.get(key, 0))

    grp_key = ["場コード", "レースNo"]
    G = df.groupby(grp_key)

    df["人気_filled"] = df["人気"].fillna(G["人気"].transform("median")).fillna(10)
    df["推定オッズ"]  = estimate_odds(df["人気_filled"])
    df["人気_IN順位"] = G["人気_filled"].transform(lambda x: x.rank(pct=True))
    df["人気差"]      = df["人気_filled"] - G["人気_filled"].transform("min")

    出走 = df["出走回数"].fillna(0)
    df["優出率"] = np.where(出走 > 0, df["優出回数"].fillna(0) / 出走, 0)
    df["優勝率"] = np.where(出走 > 0, df["優勝回数"].fillna(0) / 出走, 0)
    df["人気_逆数"]      = 1.0 / df["人気_filled"].clip(lower=1)
    df["人気×勝率"]     = df["人気_filled"] * df["全国勝率"].fillna(0)
    df["人気×モーター"] = df["人気_filled"] * df["モーター2率"].fillna(0)
    df["人気×能力指数"] = df["人気_filled"] * df["今期能力指数"].fillna(0)
    df["内有利度"]   = df["コース補正"] * (1 - df["進入コースST"].clip(0, 0.99))
    df["外まくり力"] = df["進入コース複勝率"] * (df["予想進入"] >= 3).astype(float)
    df["イン信頼度"] = (df["予想進入"] == 1).astype(float) * df["全国勝率"].fillna(0)

    # 当地スコア・モータースコア（場ごとの重み）
    def _lw(vc): return get_venue_config(vc).get("当地重み",  BASE_CONFIG["当地重み"])
    def _mw(vc): return get_venue_config(vc).get("モーター重み", BASE_CONFIG["モーター重み"])

    lw = df["場コード"].map(_lw).fillna(BASE_CONFIG["当地重み"])
    mw = df["場コード"].map(_mw).fillna(BASE_CONFIG["モーター重み"])
    df["当地スコア"]     = df["当地勝率"].fillna(0) * lw
    df["モータースコア"] = df["モーター2率"].fillna(0) * mw + df["ボート2率"].fillna(0) * 0.5

    # インプライドプロブ（この段階では推定オッズから）
    df["インプライドプロブ"] = df["推定オッズ"].apply(compute_implied_prob)

    # IN順位
    rank_specs = [
        ("全国勝率",False),("全国2率",False),("当地勝率",False),("当地2率",False),
        ("今期能力指数",False),("平均ST",True),("モーター2率",False),("ボート2率",False),
        ("今節連対率",False),("進入コース複勝率",False),("当地スコア",False),
    ]
    for col, asc in rank_specs:
        if col in df.columns:
            df[f"{col}_IN順位"] = G[col].transform(lambda x: x.rank(pct=True, ascending=asc))

    # 差分系
    for col in ["全国勝率","モーター2率","今期能力指数","当地勝率","当地2率","今節連対率","進入コース複勝率"]:
        if col in df.columns:
            gm = G[col].transform("mean")
            gx = G[col].transform("max")
            gn = G[col].transform("min")
            df[f"{col}_IN偏差"] = df[col] - gm
            df[f"{col}_MAX差"]  = gx - df[col]
            df[f"{col}_MIN差"]  = df[col] - gn

    df["モーター偏差"] = df["モーター2率"] - G["モーター2率"].transform("mean")
    df["荒れ指数"]     = G["全国勝率"].transform("std").fillna(0)
    df["ST拮抗度"]     = G["進入コースST"].transform("std").fillna(0)
    df["ST優位度"]     = G["進入コースST"].transform("min") - df["進入コースST"]
    df["ST不利度"]     = df["進入コースST"] - G["進入コースST"].transform("min")

    if "人気_IN順位" in df.columns:
        df["勝率順位差"]     = df["全国勝率_IN順位"]  - df["人気_IN順位"]
        df["モーター順位差"] = df["モーター2率_IN順位"] - df["人気_IN順位"]
        df["能力順位差"]     = df["今期能力指数_IN順位"] - df["人気_IN順位"]
    if "平均ST_IN順位" in df.columns:
        df["ST順位差"] = (1 - df["平均ST_IN順位"]) - df["人気_IN順位"]

    df["イン崩壊指数"] = (
        (df["予想進入"] == 1).astype(int)
        * (1 - df["進入コース複勝率"].clip(0, 1))
        * df["ST不利度"].clip(0, 0.1) * 10
    )
    df["イン崩壊フラグ"] = (
        (df["予想進入"] == 1)
        & ((1 - df["進入コース複勝率"].fillna(0)) > 0.55)
        & (df["ST不利度"] > 0.010)
    ).astype(int)
    df["過小評価フラグ"] = (
        (df.get("勝率順位差", pd.Series(0, index=df.index)).fillna(0) > 0.2)
        & (df["人気_IN順位"].fillna(0) > 0.6)
    ).astype(int)
    df["クラスタ別勝率"] = (
        df.groupby("展開クラスタ")["1着フラグ"].transform("mean").fillna(0.167)
        if "1着フラグ" in df.columns else 0.167
    ) if "展開クラスタ" in df.columns else 0.167

    df["展開クラスタ"] = np.where(
        (df["コース補正"] >= 1.35) & (df["進入コースST"] <= 0.16), 1,
        np.where((df["予想進入"].isin([2,3])) & (df["進入コース複勝率"] >= 0.35), 2,
        np.where((df["予想進入"] >= 4) & (df["進入コース複勝率"] >= 0.35), 3, 0))
    )

    in1 = df[df["予想進入"]==1].groupby(grp_key)["全国勝率"].first().rename("_in1")
    in2 = df[df["予想進入"]==2].groupby(grp_key)["全国勝率"].first().rename("_in2")
    df  = df.merge((in1 - in2).rename("_diff").reset_index(), on=grp_key, how="left")
    df["イン対抗指数"] = df["_diff"].fillna(0.0)
    df.drop(columns=["_diff"], inplace=True, errors="ignore")

    in_st = df[df["予想進入"]==1].groupby(grp_key)["進入コースST"].first().rename("_inST").reset_index()
    df = df.merge(in_st, on=grp_key, how="left")
    df["インST"] = df["_inST"].fillna(0.18)
    df.drop(columns=["_inST"], inplace=True, errors="ignore")

    df["節日"]       = pd.to_numeric(df["節日"], errors="coerce").fillna(1).astype(int)
    df["初日フラグ"] = (df["節日"] == 1).astype(int)

    # 動的 EV_MIN を列として付与
    if dynamic_ev_map:
        df["動的EV_MIN"] = df["場コード"].map(dynamic_ev_map).fillna(
            df["場コード"].map(lambda vc: get_venue_config(vc).get("EV_MIN", BASE_CONFIG["EV_MIN"])))
    else:
        df["動的EV_MIN"] = df["場コード"].map(
            lambda vc: get_venue_config(vc).get("EV_MIN", BASE_CONFIG["EV_MIN"]))

    return df


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """着順から学習用ターゲット列を生成"""
    df = df.copy()
    df["1着フラグ"]   = (df["着順"] == 1).astype("Int64")
    df["3着内フラグ"] = (df["着順"] <= 3).astype("Int64")
    df.loc[df["返還フラグ"] == 1, "払戻"] = None

    def _capped(r):
        cap = get_venue_config(r["場コード"]).get("PAY_CAP", BASE_CONFIG["PAY_CAP"])
        if r["1着フラグ"] == 1 and pd.notna(r["払戻"]):
            return min(float(r["払戻"]), cap)
        return 0.0 if r["返還フラグ"] == 0 else None

    df["期待値"]  = df.apply(_capped, axis=1)
    df["log払戻"] = df.apply(
        lambda r: np.log1p(
            min(r["払戻"], get_venue_config(r["場コード"]).get("PAY_CAP", BASE_CONFIG["PAY_CAP"]))
        ) if pd.notna(r.get("払戻")) and r.get("払戻", 0) > 0 else None,
        axis=1,
    )
    return df


def recalc_flags(df: pd.DataFrame) -> pd.DataFrame:
    """predict_all 後にフラグを最終計算"""
    df = df.copy()
    in_boat  = df["予想進入"] == 1
    low_win2 = df["進入コース複勝率"].fillna(0) < 0.40
    high_std = df.get("ST不利度", pd.Series(0, index=df.index)).fillna(0) > 0.005
    df["イン崩壊フラグ"]  = (in_boat & (low_win2 | high_std)).astype(int)
    df["過小評価フラグ"]  = (
        (df.get("勝率順位差", pd.Series(0, index=df.index)).fillna(0) > 0.2)
        & (df["人気_IN順位"].fillna(0) > 0.6)
    ).astype(int)
    df["ST狙い目フラグ"]  = (
        (df.get("ST順位差", pd.Series(0, index=df.index)).fillna(0) >= 0.20)
        & (df["予測_1着確率"].fillna(0) >= 0.15)
    ).astype(int)
    return df
