"""
models.py  ── モデル学習 / 予測 / TimeSeries CV
"""

from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import roc_auc_score

try:
    import lightgbm as lgb
    USE_LGB = True
    print("[LightGBM] 利用可能")
except ImportError:
    USE_LGB = False
    print("[LightGBM] 未インストール → HistGradientBoosting で代替")

from config import BASE_CONFIG, get_venue_config, VALUE_THRESHOLD
from features import (
    FEATURE_COLS, estimate_odds,
    compute_value_score, is_value_bet,
)

# ── 高配当学習設定 ─────────────────────────────────────────────
HIGH_PAYOUT_THRESHOLD: int   = 8000   # 払戻8000円以上を「高配当」とみなす
HIGH_PAYOUT_WEIGHT:    float = 5.0    # 高配当レースのサンプルウェイト倍率


# ════════════════════════════════════════════════════════════
# モデルファクトリ
# ════════════════════════════════════════════════════════════

def _make_clf(spw: float = 1.0):
    if USE_LGB:
        return lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=-1, num_leaves=31,
            subsample=0.78, colsample_bytree=0.78, min_child_samples=15,
            scale_pos_weight=spw, random_state=42, verbose=-1,
        )
    cw = {0: 1.0, 1: spw} if spw != 1.0 else None
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.04, max_leaf_nodes=31,
        min_samples_leaf=15, early_stopping=True, n_iter_no_change=30,
        validation_fraction=0.1, class_weight=cw, random_state=42,
    )


def _make_clf_cal(spw: float = 1.0) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(_make_clf(spw), method="isotonic", cv=2)


def _make_reg():
    if USE_LGB:
        return lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.04, max_depth=-1, num_leaves=31,
            subsample=0.78, colsample_bytree=0.78, min_child_samples=15,
            random_state=42, verbose=-1,
        )
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.04, max_leaf_nodes=31,
        min_samples_leaf=15, early_stopping=True, n_iter_no_change=30,
        validation_fraction=0.1, random_state=42,
    )


# ════════════════════════════════════════════════════════════
# 特徴量行列取得
# ════════════════════════════════════════════════════════════

def get_X(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    feats = [c for c in FEATURE_COLS if c in df.columns]
    X     = df[feats].copy().fillna(df[feats].median())
    return X, feats


# ════════════════════════════════════════════════════════════
# 学習
# ════════════════════════════════════════════════════════════

def fit_models(train: pd.DataFrame, add_noise: bool = True) -> dict:
    tr = train.copy()

    if add_noise and "人気_filled" in tr.columns:
        ns = tr["人気_filled"].std() * 0.1
        np.random.seed(42)
        tr["人気_filled"]   = (tr["人気_filled"] + np.random.normal(0, ns, len(tr))).clip(lower=1)
        tr["推定オッズ"]    = estimate_odds(tr["人気_filled"])
        tr["人気_逆数"]     = tr["人気_filled"].apply(lambda x: 1.0 / x if x > 0 else 0.0)
        tr["人気差"]        = tr.groupby(["場コード","レースNo"])["人気_filled"].transform(lambda x: x - x.min())
        tr["人気×勝率"]     = tr["人気_filled"] * tr["全国勝率"].fillna(0)
        tr["人気×モーター"] = tr["人気_filled"] * tr["モーター2率"].fillna(0)
        tr["人気×能力指数"] = tr["人気_filled"] * tr["今期能力指数"].fillna(0)

    X_tr, feats = get_X(tr)
    valid  = tr["返還フラグ"] == 0
    result = {"feature_names": feats}

    y_win  = tr.loc[valid, "1着フラグ"].astype(int)
    y_top3 = tr.loc[valid, "3着内フラグ"].astype(int)
    spw_win  = (y_win  == 0).sum() / max((y_win  == 1).sum(), 1)
    spw_top3 = (y_top3 == 0).sum() / max((y_top3 == 1).sum(), 1)

    # ── 高配当サンプルウェイト（8000円以上は5倍で学習）──────────
    race_max_pay = tr.groupby(["場コード","レースNo"])["払戻"].transform("max").fillna(0) \
                   if "払戻" in tr.columns else pd.Series(0, index=tr.index)
    is_hp = (race_max_pay >= HIGH_PAYOUT_THRESHOLD)
    sw = pd.Series(1.0, index=tr.index)
    sw[is_hp] = HIGH_PAYOUT_WEIGHT
    sw_valid  = sw[valid].values
    hp_cnt = int(is_hp[valid].sum())
    print(f"  学習: n={valid.sum()} / spw_win={spw_win:.1f} "
          f"/ 高配当(≥¥{HIGH_PAYOUT_THRESHOLD:,})レース: {hp_cnt//6}R (重み×{HIGH_PAYOUT_WEIGHT})")

    clf_win  = _make_clf_cal(spw_win);  clf_win.fit(X_tr[valid],  y_win,  sample_weight=sw_valid);  result["win"]  = clf_win
    clf_top3 = _make_clf_cal(spw_top3); clf_top3.fit(X_tr[valid], y_top3, sample_weight=sw_valid); result["top3"] = clf_top3

    reg_rank = _make_reg()
    reg_rank.fit(X_tr[valid], tr.loc[valid, "着順"].astype(float), sample_weight=sw_valid)
    result["rank"] = reg_rank

    win_mask = valid & (tr["1着フラグ"] == 1) & tr["log払戻"].notna()
    if win_mask.sum() >= 10:
        rp = _make_reg()
        rp.fit(X_tr[win_mask], tr.loc[win_mask, "log払戻"].astype(float),
               sample_weight=sw[win_mask].values)
        result["payout"] = rp
        print(f"  払戻モデル: {win_mask.sum()}件")
    else:
        result["payout"] = None

    if "推定オッズ" in tr.columns and "全国勝率_IN順位" in tr.columns:
        ev_mask  = valid & tr["推定オッズ"].notna()
        ev_proxy = tr["全国勝率_IN順位"].fillna(0.5) * tr["推定オッズ"].fillna(10) / 10
        y_ev     = (ev_proxy[ev_mask] > 1.0).astype(int)
        n1, n0   = y_ev.sum(), (y_ev == 0).sum()
        if n1 > 5 and n0 > 5:
            ce = _make_clf_cal(n0 / max(n1, 1))
            ce.fit(X_tr[ev_mask], y_ev)
            result["ev_clf"] = ce
        else:
            result["ev_clf"] = None
    else:
        result["ev_clf"] = None

    return result


# ════════════════════════════════════════════════════════════
# 予測
# ════════════════════════════════════════════════════════════

def predict_all(models: dict, df: pd.DataFrame) -> pd.DataFrame:
    X, _ = get_X(df)
    df   = df.copy()

    df["予測_1着確率"]      = models["win"].predict_proba(X)[:, 1]
    df["予測_3着内確率"]    = models["top3"].predict_proba(X)[:, 1]
    df["予測_着順"]         = models["rank"].predict(X).clip(1, 6)
    df["予測_EV_2段階"]     = (
        df["予測_1着確率"] * np.expm1(models["payout"].predict(X).clip(0)) / 100
        if models.get("payout")
        else df["予測_1着確率"] * df["推定オッズ"]
    )
    df["予測_EVフラグ確率"] = (
        models["ev_clf"].predict_proba(X)[:, 1]
        if models.get("ev_clf")
        else df["予測_1着確率"]
    )

    df["人気歪み補正"] = 1.0 + df["人気_IN順位"].fillna(0.5) * 0.15
    df["真期待値"]     = df["予測_1着確率"] * df["推定オッズ"] * df["人気歪み補正"]
    df["log真期待値"]  = df["予測_1着確率"] * np.log1p(df["推定オッズ"])
    df["予測不確実性"] = (
        df["予測_1着確率"]
        - df.groupby(["場コード","レースNo"])["予測_1着確率"].transform(
            lambda x: x.rank(method="min", ascending=True, pct=True)
        )
    ).abs()

    # バリュースコア（正式計算）
    df["バリュースコア"] = df.apply(
        lambda r: compute_value_score(r["予測_1着確率"], r["推定オッズ"]), axis=1)
    df["バリューフラグ"] = df.apply(
        lambda r: int(is_value_bet(r["予測_1着確率"], r["推定オッズ"])), axis=1)

    rank_s  = (7 - df["予測_着順"]) / 6
    ev_max  = df.groupby(["場コード","レースNo"])["予測_EV_2段階"].transform("max").clip(lower=0.001)
    val_max = df.groupby(["場コード","レースNo"])["バリュースコア"].transform("max").clip(0.001)
    loc_max = df.groupby(["場コード","レースNo"])["当地スコア"].transform("max").clip(0.001) \
              if "当地スコア" in df.columns else 1.0

    df["アンサンブルスコア"] = (
        df["予測_1着確率"]               * 0.28
        + df["予測_3着内確率"]           * 0.12
        + rank_s                         * 0.10
        + (df["予測_EV_2段階"] / ev_max) * 0.22
        + df["予測_EVフラグ確率"]        * 0.12
        + (df["バリュースコア"] / val_max) * 0.10
        + (df.get("当地スコア", pd.Series(0, index=df.index)).fillna(0) / loc_max) * 0.06
    )

    for col in ["予測_1着確率","真期待値","アンサンブルスコア"]:
        df[f"{col}_IN順位"] = df.groupby(["場コード","レースNo"])[col].transform(
            lambda x: x.rank(method="min", ascending=False).astype(int))

    # ── 高配当スコア・予測払戻・高配当フラグ（8000円基準）────────
    if models.get("payout"):
        df["予測払戻"] = np.expm1(models["payout"].predict(X).clip(0))
    else:
        df["予測払戻"] = df["推定オッズ"] * 100   # 推定払戻（円換算）
    df["高配当スコア"] = (df["予測_1着確率"] * df["予測払戻"] / HIGH_PAYOUT_THRESHOLD).round(4)
    df["高配当フラグ"] = (
        (df["予測払戻"] >= HIGH_PAYOUT_THRESHOLD) &
        (df["予測_1着確率"] >= 0.15)
    ).astype(int)

    return df


# ════════════════════════════════════════════════════════════
# TimeSeries CV
# ════════════════════════════════════════════════════════════

def timeseries_cv(df: pd.DataFrame, fast_mode: bool = True) -> dict:
    node_days  = sorted(df["節日"].unique())
    all_pred   = []
    fold_metrics = []
    eval_days  = node_days[-2:] if fast_mode and len(node_days) > 2 else node_days

    print(f"[TimeSeries CV] 節日: {node_days} / fast_mode={fast_mode}")

    for i, test_day in enumerate(node_days):
        train_days = [d for d in node_days if d < test_day]
        if not train_days:
            continue
        if fast_mode and test_day not in eval_days:
            continue
        train = df[df["節日"].isin(train_days)]
        test  = df[df["節日"] == test_day]
        if len(train[train["返還フラグ"] == 0]) < 30:
            continue
        print(f"  fold{i}: train=節日{train_days}({len(train)}行) → test=節日{test_day}({len(test)}行)")
        models_cv = fit_models(train)
        pred      = predict_all(models_cv, test)
        all_pred.append(pred)
        valid = pred[pred["返還フラグ"] == 0]
        try:
            auc = roc_auc_score(valid["1着フラグ"].astype(int), valid["予測_1着確率"])
            fold_metrics.append({"fold": i, "test_day": test_day, "n_test": len(valid), "AUC_win": round(auc, 4)})
        except Exception as e:
            fold_metrics.append({"fold": i, "test_day": test_day, "error": str(e)})

    pred_df    = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    metrics_df = pd.DataFrame(fold_metrics)
    print("\n[CV結果]")
    print(metrics_df.to_string(index=False))
    if "AUC_win" in metrics_df.columns:
        print(f"  平均AUC: {metrics_df['AUC_win'].mean():.4f}")

    print("\n[最終モデル] 全データで学習中...")
    final_models = fit_models(df)
    return {"pred_df": pred_df, "final_models": final_models, "metrics_df": metrics_df}


# ════════════════════════════════════════════════════════════
# モデル保存・読み込み
# ════════════════════════════════════════════════════════════

def save_model(models: dict, path: str | Path, extra: dict | None = None) -> None:
    payload = {"models": models, "feature_cols": FEATURE_COLS}
    if extra:
        payload.update(extra)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[保存] {path} ✅")


def load_model(path: str | Path) -> dict:
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[読込] {path} ✅")
    return data
