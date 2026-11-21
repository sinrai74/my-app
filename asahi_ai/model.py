"""モデル（Project 11 §4）。

- LightGBM で艇ごとの点数 s を学習する。目的関数はレース内の softmax（条件付きロジット）：
  1着の確率 p_i = exp(s_i) / Σ_j exp(s_j)。勾配 p_i − y_i、ヘッセ p_i(1 − p_i)。
- 2・3着と3連単は Plackett–Luce。2着の段は exp(λ2·s)、3着の段は exp(λ3·s) で選ぶ（Harville の偏りの補正）。
  λ2・λ3 は検証の期間で、3連単の対数尤度が最大になるように合わせる。
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

PERMS = np.array(list(itertools.permutations(range(6), 3)))  # 120 通り（艇のレース内の位置）


def _group_softmax(score: np.ndarray, group_idx: np.ndarray) -> np.ndarray:
    mx = pd.Series(score).groupby(group_idx).transform("max").to_numpy()
    e = np.exp(score - mx)
    return e / pd.Series(e).groupby(group_idx).transform("sum").to_numpy()


def make_objective(group_idx: np.ndarray):
    def obj(preds, dataset):
        y = dataset.get_label()
        p = _group_softmax(preds, group_idx)
        return p - y, np.maximum(p * (1 - p), 1e-6)
    return obj


def make_metric(group_idx: np.ndarray):
    def metric(preds, dataset):
        y = dataset.get_label()
        p = _group_softmax(preds, group_idx)
        ll = -np.log(np.clip(p[y == 1], 1e-12, 1)).mean()
        return "race_logloss", ll, False
    return metric


def _valid_races(df: pd.DataFrame) -> pd.DataFrame:
    """学習・評価に使う行：出走した艇（欠場を除く）で、1着がちょうど1艇のレース。"""
    d = df[df["result_class"].notna() & (df["result_class"] != "absent")]
    wins = d.groupby("race_id")["won"].transform("sum")
    return d[wins == 1].sort_values(["race_id", "lane"]).reset_index(drop=True)


class StrengthModel:
    def __init__(self, params: dict | None = None, num_rounds: int = 2000):
        self.params = {"learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 200,
                       "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
                       "lambda_l2": 1.0, "verbose": -1, "seed": 7, **(params or {})}
        self.num_rounds = num_rounds
        self.booster = None
        self.features: list[str] = []
        self.lam = (1.0, 1.0)
        self.temp = 1.0

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame, features: list[str]):
        import lightgbm as lgb
        tr, va = _valid_races(train), _valid_races(valid)
        self.features = features
        gtr = pd.factorize(tr["race_id"])[0]
        gva = pd.factorize(va["race_id"])[0]
        dtr = lgb.Dataset(tr[features], label=tr["won"], free_raw_data=False)
        dva = lgb.Dataset(va[features], label=va["won"], reference=dtr, free_raw_data=False)
        params = {**self.params, "objective": make_objective(gtr), "metric": "None"}
        # 検証の評価は検証の集合のグループで行う
        evals = {}

        def feval(preds, ds):
            if len(preds) == len(gva):
                return make_metric(gva)(preds, ds)
            return make_metric(gtr)(preds, ds)
        self.booster = lgb.train(params, dtr, num_boost_round=self.num_rounds, valid_sets=[dva],
                                 valid_names=["valid"], feval=feval,
                                 callbacks=[lgb.early_stopping(100, verbose=False), lgb.record_evaluation(evals)])
        raw = self.booster.predict(va[features], num_iteration=self.booster.best_iteration)
        self.temp = fit_temperature(raw, gva, va["won"].to_numpy())
        self.lam = fit_lambda(self.score(va), va)
        return self

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.temp * self.booster.predict(df[self.features], num_iteration=self.booster.best_iteration)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """df：1艇1行（同じ race_id に 6 行以内）。1着の確率と3連単120通りの確率。"""
        d = df.sort_values(["race_id", "lane"]).reset_index(drop=True)
        d["score"] = self.score(d)
        return race_probabilities(d, self.lam)


# 2着・3着の段の分母は「残っている艇の重みの和」。引き算（全体 − 1着 − 2着）は桁落ちするので使わない
_M2 = np.ones((120, 6))
_M2[np.arange(120), PERMS[:, 0]] = 0
_M3 = _M2.copy()
_M3[np.arange(120), PERMS[:, 1]] = 0


def _trifecta_matrix(s: np.ndarray, lam: tuple[float, float]) -> np.ndarray:
    """s：(レース数, 6) の点数（艇がいない所は −inf）。返り値：(レース数, 120) の確率。

    出走が3艇未満のレースは、3連単が成り立たないので全部 0（合計 0）。
    """
    a, b, c = PERMS[:, 0], PERMS[:, 1], PERMS[:, 2]
    finite = np.isfinite(s)

    def stage(weight):
        mx = np.max(np.where(finite, weight, -np.inf), axis=1, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        return np.where(finite, np.exp(np.where(finite, weight, 0.0) - mx), 0.0)
    w1, w2, w3 = stage(s), stage(lam[0] * s), stage(lam[1] * s)
    with np.errstate(divide="ignore", invalid="ignore"):
        p1 = w1[:, a] / w1.sum(1, keepdims=True)
        p2 = w2[:, b] / (w2 @ _M2.T)
        p3 = w3[:, c] / (w3 @ _M3.T)
        p = p1 * p2 * p3
    p = np.where(np.isfinite(p), p, 0.0)
    return p


def _to_matrix(d: pd.DataFrame, col: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    races = pd.unique(d["race_id"])
    idx = pd.Index(races).get_indexer(d["race_id"])
    m = np.full((len(races), 6), -np.inf)
    lanes = d["lane"].astype(int).to_numpy() - 1
    m[idx, lanes] = d[col].to_numpy()
    return m, list(races), idx


def race_probabilities(d: pd.DataFrame, lam: tuple[float, float]) -> pd.DataFrame:
    m, races, _ = _to_matrix(d, "score")
    p1 = np.exp(m - m.max(1, keepdims=True))
    p1 = p1 / p1.sum(1, keepdims=True)
    tri = _trifecta_matrix(m, lam)
    out = pd.DataFrame({"race_id": races})
    for i in range(6):
        out[f"p_win_{i + 1}"] = p1[:, i]
    out["tri"] = list(tri)
    out["tri_sum"] = tri.sum(1)
    out["win_sum"] = p1.sum(1)
    return out


def trifecta_index(order: tuple[int, int, int]) -> int:
    """(1着, 2着, 3着) の艇番 → 120 通りの何番目か。"""
    key = tuple(x - 1 for x in order)
    return int(np.where((PERMS == key).all(1))[0][0])


def actual_trifecta(d: pd.DataFrame) -> pd.Series:
    """race_id → 実際の3連単の位置（1〜3着が1艇ずつのレースだけ。それ以外は NaN）。"""
    fin = d[d["finish_position"].isin([1, 2, 3])].sort_values(["race_id", "finish_position"])
    g = fin.groupby("race_id")
    ok = g["finish_position"].apply(lambda x: list(x) == [1, 2, 3])
    lanes = g["lane"].apply(lambda x: tuple(int(v) for v in x))
    res = {r: trifecta_index(lanes[r]) for r in ok.index if ok[r]}
    return pd.Series(res, dtype=float)


def fit_lambda(score: np.ndarray, va: pd.DataFrame) -> tuple[float, float]:
    from scipy.optimize import minimize
    d = va.copy()
    d["score"] = score
    m, races, _ = _to_matrix(d, "score")
    act = actual_trifecta(d).reindex(races).to_numpy()
    keep = ~np.isnan(act)
    m, act = m[keep], act[keep].astype(int)

    def nll(x):
        p = _trifecta_matrix(m, (x[0], x[1]))[np.arange(len(act)), act]
        return -np.log(np.clip(p, 1e-12, 1)).mean()
    r = minimize(nll, x0=[0.8, 0.6], bounds=[(0.1, 2.0), (0.1, 2.0)], method="L-BFGS-B")
    return float(r.x[0]), float(r.x[1])


def fit_temperature(raw: np.ndarray, group_idx: np.ndarray, y: np.ndarray) -> float:
    """1着の確率の較正：点数に掛ける係数 a を、検証の期間のレース内の対数損失が最小になるように決める。"""
    from scipy.optimize import minimize_scalar

    def nll(a):
        p = _group_softmax(a * raw, group_idx)
        return -np.log(np.clip(p[y == 1], 1e-12, 1)).mean()
    r = minimize_scalar(nll, bounds=(0.3, 3.0), method="bounded")
    return float(r.x)
