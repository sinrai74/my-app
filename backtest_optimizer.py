"""
backtest_optimizer.py
─────────────────────────────────────────────────────────────
Plackett–Luce モデルのパラメータをバックテストで最適化する。

使い方:
    python backtest_optimizer.py --mode grid     # グリッドサーチ
    python backtest_optimizer.py --mode random   # ランダムサーチ（推奨）
    python backtest_optimizer.py --mode monthly  # 月別ROI分析

入力データ（hit_record.csv + 過去レースJSONキャッシュ）:
    races.json  ← 下記フォーマットで用意する
    [
      {
        "date": "20250501",
        "venue": 12,
        "race": 6,
        "boats": [
          {"lane":1, "win_rate":5.2, "motor":42.0, "ex_time":6.73,
           "ex_st":0.15, "avg_st":0.17, "racer_class":"A1"},
          ...
        ],
        "weather": {"wind_speed":2.0, "wind_direction":"向", "wave_height":3},
        "odds": {"1-2-3": 12.5, "2-1-3": 45.0, ...},
        "result": "2-3-1"
      },
      ...
    ]

依存: pip install scipy  (optionalだがランダムサーチで使う)
"""

from __future__ import annotations

import json
import math
import random
import argparse
import csv
import os
from dataclasses import dataclass
from itertools import permutations
from typing import Optional
from collections import defaultdict


# ════════════════════════════════════════════════════════════
# データクラス（notify_arashi.py と同一定義）
# ════════════════════════════════════════════════════════════

@dataclass
class BoatInfo:
    lane:        int
    win_rate:    float = 0.0
    motor:       float = 0.0
    avg_st:      float = 0.18
    ex_time:     Optional[float] = None
    ex_st:       Optional[float] = None
    racer_class: str = ""


@dataclass
class WeatherInfo:
    wind_speed:     Optional[float] = None
    wind_direction: Optional[str]   = None
    wave_height:    Optional[int]   = None


# ════════════════════════════════════════════════════════════
# Plackett–Luce スコア（パラメータ差し込み可能版）
# ════════════════════════════════════════════════════════════

DEFAULT_PARAMS = {
    # スコア係数
    "wr_weight":     0.15,   # 勝率の重み
    "motor_weight":  0.006,  # モーター2連率の重み
    "ex_weight":     3.0,    # 展示タイムの重み
    "st_weight":     4.0,    # 展示STの重み
    "makuri_bonus":  0.5,    # まくり補正（2-4号艇が早い＋1号艇遅い）
    "sashi_bonus":   0.3,    # 差し補正（2号艇がSTで拮抗）
    # コース補正（固定 — 実績ベースなのでサーチ対象外）
    "course_bias": {1: 0.60, 2: -0.13, 3: -0.28, 4: -0.31, 5: -0.44, 6: -0.51},
    # EVフィルタ
    "ev_th_upset":   1.1,    # 荒れスコア>=5 時のEV閾値（攻める）
    "ev_th_normal":  1.3,    # 荒れスコア<5  時のEV閾値（絞る）
    "upset_border":  5.0,    # 荒れ/固定の境界スコア
    # 資金管理
    "kelly_cap":     0.15,   # ケリー比率の上限
    "bankroll":      10000,  # 基準資金（円）
    "min_bets":      30,     # 有効判定の最低ベット数
}


def calc_boat_score(boat: BoatInfo, all_boats: list[BoatInfo],
                    weather: WeatherInfo, params: dict) -> float:
    score = 0.0

    # 勝率・モーター（平均差分）
    score += (boat.win_rate - 5.0) * params["wr_weight"]
    score += (boat.motor - 40.0)   * params["motor_weight"]

    # 展示タイム
    ex_times = [b.ex_time for b in all_boats if b.ex_time and b.ex_time > 0]
    if ex_times and boat.ex_time and boat.ex_time > 0:
        avg_ex = sum(ex_times) / len(ex_times)
        score += (avg_ex - boat.ex_time) * params["ex_weight"]

    # 展示ST
    sts = [b.ex_st for b in all_boats if b.ex_st is not None]
    if sts and boat.ex_st is not None:
        avg_st = sum(sts) / len(sts)
        score += (avg_st - boat.ex_st) * params["st_weight"]

        boat1 = next((b for b in all_boats if b.lane == 1), None)
        if boat.lane in [2, 3, 4] and boat1 and boat1.ex_st:
            if boat.ex_st <= 0.13 and boat1.ex_st >= 0.18:
                score += params["makuri_bonus"]
        if boat.lane == 2 and boat1 and boat1.ex_st:
            if abs(boat.ex_st - boat1.ex_st) <= 0.02:
                score += params["sashi_bonus"]

    # コース補正
    score += params["course_bias"].get(boat.lane, 0.0)

    # 風・波
    if weather:
        if weather.wind_speed and weather.wind_speed > 0:
            ws, wd = weather.wind_speed, weather.wind_direction or ""
            if wd == "向":
                score += ws * (0.03 if boat.lane >= 3 else -0.03)
            elif wd == "追":
                score += ws * (0.02 if boat.lane == 1 else -0.01)
        if weather.wave_height:
            wh = weather.wave_height
            if wh >= 5:   score += -0.5 if boat.lane == 1 else 0.15
            elif wh >= 3: score += -0.2 if boat.lane == 1 else 0.06

    # 等級
    grade_bonus = {"A1": 0.15, "A2": 0.05, "B1": -0.10, "B2": -0.25}
    score += grade_bonus.get(boat.racer_class, 0.0)

    return score


def pl_probs(boats: list[BoatInfo], weather: WeatherInfo,
             params: dict) -> dict[int, float]:
    """Plackett–Luce 1着確率を返す。"""
    raw = {b.lane: calc_boat_score(b, boats, weather, params) for b in boats}
    exp_s = {l: math.exp(s) for l, s in raw.items()}
    total = sum(exp_s.values()) or 1.0
    return {l: e / total for l, e in exp_s.items()}


def trifecta_probs_pl(lane_p: dict[int, float]) -> dict[str, float]:
    """Plackett–Luce で全120通りの三連単確率を計算。"""
    result = {}
    for a, b, c in permutations(lane_p.keys(), 3):
        pa = lane_p[a]
        rem1 = {l: p for l, p in lane_p.items() if l != a}
        t1   = sum(rem1.values()) or 1.0
        pb   = rem1.get(b, 0) / t1
        rem2 = {l: p for l, p in rem1.items() if l != b}
        t2   = sum(rem2.values()) or 1.0
        pc   = rem2.get(c, 0) / t2
        result[f"{a}-{b}-{c}"] = pa * pb * pc
    return result


def kelly_bet(prob: float, odds: float, bankroll: int,
              kelly_cap: float) -> int:
    """ケリー基準で賭け金（100円単位）を計算。"""
    if odds <= 1:
        return 0
    k = (prob * odds - 1) / (odds - 1)
    k = max(0.0, min(k, kelly_cap))
    return int(bankroll * k / 100) * 100


# ════════════════════════════════════════════════════════════
# シミュレーション本体
# ════════════════════════════════════════════════════════════

def simulate(races: list[dict], params: dict,
             fixed_bet: int = 100) -> dict:
    """
    races: レースリスト
    params: パラメータ辞書
    fixed_bet: 固定賭け金（ケリー無効時）

    returns: {
        "roi": float,          # 総回収率
        "total_bet": int,
        "total_return": int,
        "n_bets": int,         # 買い目数
        "n_hits": int,         # 的中数
        "monthly": {月: {"roi":..., "n_bets":...}}
    }
    """
    total_bet    = 0
    total_return = 0
    n_hits       = 0
    monthly      = defaultdict(lambda: {"bet": 0, "ret": 0, "n": 0})

    ev_th_upset  = params.get("ev_th_upset",  1.1)
    ev_th_normal = params.get("ev_th_normal", 1.3)
    upset_border = params.get("upset_border", 5.0)
    kelly_cap    = params.get("kelly_cap",    0.15)
    bankroll     = params.get("bankroll",     10000)
    use_kelly    = params.get("use_kelly",    False)

    for race in races:
        boats   = [BoatInfo(**b) for b in race["boats"]]
        weather = WeatherInfo(**race.get("weather", {}))
        odds_map: dict[str, float] = race.get("odds", {})
        result  = race.get("result", "")
        month   = str(race.get("date", ""))[:6]  # YYYYMM

        if not boats or not odds_map or not result:
            continue

        # Plackett–Luce 確率
        lp = pl_probs(boats, weather, params)
        tp = trifecta_probs_pl(lp)

        # 荒れスコア
        upset_score = (1.0 - lp.get(1, 1/6)) * 10.0
        ev_th = ev_th_upset if upset_score >= upset_border else ev_th_normal

        # EV計算 → 買い目抽出
        for combo, prob in tp.items():
            odds = odds_map.get(combo, 0)
            if odds <= 0 or odds > 500:
                continue
            ev = prob * odds
            if ev < ev_th:
                continue

            # 賭け金
            if use_kelly:
                bet = kelly_bet(prob, odds, bankroll, kelly_cap)
            else:
                bet = fixed_bet
            if bet <= 0:
                continue

            total_bet    += bet
            monthly[month]["bet"] += bet
            monthly[month]["n"]   += 1

            if combo == result:
                ret = int(bet * odds)
                total_return          += ret
                monthly[month]["ret"] += ret
                n_hits                += 1

    n_bets = sum(v["n"] for v in monthly.values())
    roi    = total_return / total_bet if total_bet > 0 else 0.0

    # 月別ROI
    monthly_roi = {}
    for m, v in sorted(monthly.items()):
        r = v["ret"] / v["bet"] if v["bet"] > 0 else 0.0
        monthly_roi[m] = {"roi": round(r, 4), "n_bets": v["n"],
                          "bet": v["bet"], "ret": v["ret"]}

    return {
        "roi":          round(roi, 4),
        "total_bet":    total_bet,
        "total_return": total_return,
        "n_bets":       n_bets,
        "n_hits":       n_hits,
        "monthly":      monthly_roi,
    }


# ════════════════════════════════════════════════════════════
# グリッドサーチ
# ════════════════════════════════════════════════════════════

def grid_search(train: list[dict], params_base: dict) -> dict:
    """
    ev_th_upset / ev_th_normal / st_weight / ex_weight の4軸を探索。
    """
    best_roi    = 0.0
    best_params = None
    results     = []

    grid = {
        "ev_th_upset":  [1.0, 1.1, 1.2, 1.3],
        "ev_th_normal": [1.2, 1.3, 1.4, 1.5],
        "st_weight":    [3.0, 4.0, 5.0, 6.0],
        "ex_weight":    [2.0, 3.0, 4.0, 5.0],
    }

    total = (len(grid["ev_th_upset"]) * len(grid["ev_th_normal"])
             * len(grid["st_weight"]) * len(grid["ex_weight"]))
    done = 0

    for ev_up in grid["ev_th_upset"]:
        for ev_no in grid["ev_th_normal"]:
            if ev_no <= ev_up:
                continue   # 攻め閾値が絞り閾値より高いのは無意味
            for st_w in grid["st_weight"]:
                for ex_w in grid["ex_weight"]:
                    params = {**params_base,
                              "ev_th_upset":  ev_up,
                              "ev_th_normal": ev_no,
                              "st_weight":    st_w,
                              "ex_weight":    ex_w}
                    res = simulate(train, params)
                    done += 1

                    # 最低ベット数チェック（過学習防止）
                    if res["n_bets"] < params_base.get("min_bets", 30):
                        continue

                    results.append({"params": params, **res})

                    if res["roi"] > best_roi:
                        best_roi    = res["roi"]
                        best_params = params
                        print(f"  [{done}/{total}] 新ベスト ROI={best_roi:.3f}  "
                              f"ev_up={ev_up} ev_no={ev_no} "
                              f"st={st_w} ex={ex_w}  "
                              f"n_bets={res['n_bets']}")

    return {"best_params": best_params, "best_roi": best_roi,
            "all_results": sorted(results, key=lambda x: -x["roi"])[:20]}


# ════════════════════════════════════════════════════════════
# ランダムサーチ（推奨）
# ════════════════════════════════════════════════════════════

def random_search(train: list[dict], params_base: dict,
                  n_iter: int = 300, seed: int = 42) -> dict:
    """
    ランダムサーチ。グリッドより少ない試行で広い空間を探索できる。
    """
    random.seed(seed)
    best_roi    = 0.0
    best_params = None
    results     = []

    for i in range(n_iter):
        params = {
            **params_base,
            "ev_th_upset":  random.uniform(0.9, 1.3),
            "ev_th_normal": random.uniform(1.2, 1.6),
            "st_weight":    random.uniform(2.0, 8.0),
            "ex_weight":    random.uniform(1.5, 6.0),
            "wr_weight":    random.uniform(0.08, 0.25),
            "motor_weight": random.uniform(0.003, 0.012),
            "makuri_bonus": random.uniform(0.2, 0.8),
            "upset_border": random.uniform(4.0, 6.5),
        }

        # ev_th_upset < ev_th_normal を強制
        if params["ev_th_normal"] <= params["ev_th_upset"]:
            params["ev_th_normal"] = params["ev_th_upset"] + 0.15

        res = simulate(train, params)

        if res["n_bets"] < params_base.get("min_bets", 30):
            continue

        results.append({"params": params, **res})

        if res["roi"] > best_roi:
            best_roi    = res["roi"]
            best_params = params
            print(f"  [{i+1}/{n_iter}] 新ベスト ROI={best_roi:.3f}  "
                  f"n_bets={res['n_bets']}  "
                  f"ev_up={params['ev_th_upset']:.2f} "
                  f"ev_no={params['ev_th_normal']:.2f} "
                  f"st={params['st_weight']:.1f} "
                  f"ex={params['ex_weight']:.1f}")

    return {"best_params": best_params, "best_roi": best_roi,
            "all_results": sorted(results, key=lambda x: -x["roi"])[:20]}


# ════════════════════════════════════════════════════════════
# 月別安定性分析
# ════════════════════════════════════════════════════════════

def monthly_stability(test: list[dict], params: dict) -> None:
    """
    最適パラメータをテストデータに適用して月別ROIを表示する。
    ROIがブレていないか（過学習していないか）を確認する。
    """
    res = simulate(test, params)
    print(f"\n{'='*55}")
    print(f"  テストデータ結果")
    print(f"  総ROI: {res['roi']:.3f}  "
          f"({res['total_return']:,}円 / {res['total_bet']:,}円)  "
          f"的中: {res['n_hits']} / {res['n_bets']}件")
    print(f"{'='*55}")
    print(f"  {'月':<8} {'ROI':>6}  {'ベット':>8}  {'回収':>8}  {'件数':>5}")
    print(f"  {'-'*45}")

    rois = []
    for m, v in res["monthly"].items():
        roi_str = f"{v['roi']:.3f}"
        flag    = "🔴" if v["roi"] < 0.7 else ("🟢" if v["roi"] >= 1.0 else "🟡")
        print(f"  {m}  {roi_str:>6}  "
              f"{v['bet']:>8,}円  {v['ret']:>8,}円  {v['n_bets']:>5}件  {flag}")
        rois.append(v["roi"])

    if rois:
        import statistics
        print(f"\n  月別ROI 平均={statistics.mean(rois):.3f}  "
              f"標準偏差={statistics.stdev(rois) if len(rois)>1 else 0:.3f}")
        print(f"  ブレが大きい（std>0.3）と過学習の可能性あり ⚠️")


# ════════════════════════════════════════════════════════════
# データ読み込み
# ════════════════════════════════════════════════════════════

def load_races(path: str) -> list[dict]:
    if not os.path.exists(path):
        print(f"⚠️  {path} が見つかりません。サンプルデータで実行します。")
        return _make_sample_races()
    with open(path, encoding="utf-8") as f:
        races = json.load(f)
    print(f"✅ {len(races)} レース読み込み完了: {path}")
    return races


def _make_sample_races(n: int = 500) -> list[dict]:
    """
    テスト用サンプルデータを生成する。
    実際の運用では races.json を用意して置き換える。
    """
    random.seed(0)
    lanes = list(range(1, 7))
    # コース実績ベースの1着確率（近似）
    true_probs = {1: 0.45, 2: 0.12, 3: 0.10, 4: 0.12, 5: 0.11, 6: 0.10}

    races = []
    dates = [f"2025{m:02d}{d:02d}"
             for m in range(1, 7) for d in range(1, 28)]

    for i in range(n):
        date = dates[i % len(dates)]
        boats = []
        for lane in lanes:
            boats.append({
                "lane":        lane,
                "win_rate":    random.gauss(5.0 + (1 if lane == 1 else 0), 1.2),
                "motor":       random.gauss(40.0, 8.0),
                "ex_time":     round(random.gauss(6.75, 0.06), 2),
                "ex_st":       round(random.gauss(0.17, 0.04), 2),
                "avg_st":      round(random.gauss(0.17, 0.02), 2),
                "racer_class": random.choice(["A1","A1","A2","B1"]),
            })
            boats[-1]["motor"] = max(10.0, boats[-1]["motor"])

        # 実際の着順（コース確率に基づくシミュレーション）
        result_1st = random.choices(lanes, weights=[true_probs[l] for l in lanes])[0]
        remaining  = [l for l in lanes if l != result_1st]
        result_2nd = random.choice(remaining)
        remaining2 = [l for l in remaining if l != result_2nd]
        result_3rd = random.choice(remaining2)
        result     = f"{result_1st}-{result_2nd}-{result_3rd}"

        # ダミーオッズ（理論値の70〜130%でランダム）
        odds: dict[str, float] = {}
        for a, b, c in permutations(lanes, 3):
            combo   = f"{a}-{b}-{c}"
            # 理論オッズ = 1 / (p1 * p2|残 * p3|残2) * 0.75（控除率）
            tp1 = true_probs[a]
            rem = {l: true_probs[l] for l in lanes if l != a}
            tp2 = true_probs[b] / sum(rem.values())
            rem2 = {l: true_probs[l] for l in rem if l != b}
            tp3  = true_probs[c] / sum(rem2.values())
            theo = 0.75 / (tp1 * tp2 * tp3)
            odds[combo] = round(theo * random.uniform(0.85, 1.20), 1)

        races.append({
            "date":    date,
            "venue":   random.randint(1, 24),
            "race":    random.randint(1, 12),
            "boats":   boats,
            "weather": {
                "wind_speed":     random.uniform(0, 6),
                "wind_direction": random.choice(["向", "追", "横", None]),
                "wave_height":    random.randint(0, 8),
            },
            "odds":   odds,
            "result": result,
        })

    print(f"  ⚙️  サンプルデータ {n} レース生成（races.json を用意すると実データで動作）")
    return races


# ════════════════════════════════════════════════════════════
# 結果保存
# ════════════════════════════════════════════════════════════

def save_results(search_result: dict, out_path: str = "best_params.json") -> None:
    bp = search_result["best_params"]
    if bp is None:
        print("⚠️  有効な結果が得られませんでした（min_bets 未達）")
        return

    output = {
        "best_roi":    search_result["best_roi"],
        "best_params": {k: v for k, v in bp.items() if k != "course_bias"},
        "top20": [
            {k: v for k, v in r.items() if k not in ("params", "monthly")}
            for r in search_result["all_results"]
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 結果保存: {out_path}")


# ════════════════════════════════════════════════════════════
# エントリポイント
# ════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Plackett–Luce バックテスト最適化")
    parser.add_argument("--mode", choices=["grid", "random", "monthly"],
                        default="random", help="探索モード")
    parser.add_argument("--data", default="races.json",
                        help="レースデータ JSON ファイル")
    parser.add_argument("--iter", type=int, default=300,
                        help="ランダムサーチの試行回数")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  default="best_params.json")
    parser.add_argument("--kelly", action="store_true",
                        help="ケリー基準で賭け金を最適化する")
    args = parser.parse_args()

    races = load_races(args.data)
    if not races:
        return

    # train:test = 70:30 で分割
    split = int(len(races) * 0.7)
    train = races[:split]
    test  = races[split:]
    print(f"  学習: {len(train)} レース  検証: {len(test)} レース")

    params_base = {**DEFAULT_PARAMS, "use_kelly": args.kelly}

    # ─── ベースライン（デフォルトパラメータ）──────────────────
    print("\n【ベースライン（デフォルトパラメータ）】")
    base_res = simulate(train, params_base)
    print(f"  ROI={base_res['roi']:.3f}  "
          f"n_bets={base_res['n_bets']}  "
          f"hits={base_res['n_hits']}")

    # ─── 探索 ────────────────────────────────────────────────
    if args.mode == "grid":
        print(f"\n【グリッドサーチ開始】")
        result = grid_search(train, params_base)

    elif args.mode == "random":
        print(f"\n【ランダムサーチ開始: {args.iter} 試行】")
        result = random_search(train, params_base,
                               n_iter=args.iter, seed=args.seed)

    elif args.mode == "monthly":
        print("\n【月別安定性分析（デフォルトパラメータ）】")
        monthly_stability(races, params_base)
        return

    # ─── 結果表示 ────────────────────────────────────────────
    bp = result["best_params"]
    if bp is None:
        print("有効な結果なし")
        return

    print(f"\n{'='*55}")
    print(f"  最適パラメータ  (学習データ ROI={result['best_roi']:.3f})")
    print(f"{'='*55}")
    for k, v in bp.items():
        if k == "course_bias": continue
        print(f"  {k:<18}: {v}")

    # ─── テストデータで検証 ──────────────────────────────────
    print("\n【テストデータ（未学習データ）での検証】")
    monthly_stability(test, bp)

    # ─── notify_arashi.py への反映コード ────────────────────
    print(f"\n【notify_arashi.py への反映 — calc_boat_score の重みをこの値に変更】")
    print(f"  wr_weight     = {bp.get('wr_weight',    0.15):.4f}")
    print(f"  motor_weight  = {bp.get('motor_weight', 0.006):.5f}")
    print(f"  ex_weight     = {bp.get('ex_weight',    3.0):.2f}   # (avg_ex - ex_time) × ?")
    print(f"  st_weight     = {bp.get('st_weight',    4.0):.2f}   # (avg_st - ex_st)  × ?")
    print(f"  makuri_bonus  = {bp.get('makuri_bonus', 0.5):.3f}")
    print(f"  ev_th_upset   = {bp.get('ev_th_upset',  1.1):.3f}   # 荒れ時のEV閾値")
    print(f"  ev_th_normal  = {bp.get('ev_th_normal', 1.3):.3f}   # 通常時のEV閾値")
    print(f"  upset_border  = {bp.get('upset_border', 5.0):.2f}   # 荒れ判定ボーダー")

    save_results(result, args.out)


if __name__ == "__main__":
    main()
