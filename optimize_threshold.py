"""
optimize_threshold.py  ── 場ごとの閾値を最適化するシミュレーター

使い方:
    python optimize_threshold.py

hit_record.csvが溜まったら実際の的中データで最適化できる。
データが少ない場合はスコア分布から推定する。
"""

from __future__ import annotations
import csv
import os
import numpy as np
from pathlib import Path
from itertools import groupby


# ── 場名→場コードのマップ ────────────────────────────────────
VENUE_NAME_TO_NUM = {
    "桐生":1,"戸田":2,"江戸川":3,"平和島":4,"多摩川":5,
    "浜名湖":6,"蒲郡":7,"常滑":8,"津":9,"三国":10,
    "びわこ":11,"住之江":12,"尼崎":13,"鳴門":14,"丸亀":15,
    "児島":16,"宮島":17,"徳山":18,"下関":19,"若松":20,
    "芦屋":21,"福岡":22,"唐津":23,"大村":24,
}


def simulate_threshold(threshold: float, races: list[dict]) -> dict:
    """指定閾値でシミュレーション。ROI・的中率・試行数を返す。"""
    total_bet = 0
    total_return = 0
    hits = 0

    for race in races:
        score = float(race.get("score", 0) or 0)
        if score < threshold:
            continue

        total_bet += 100  # 1点100円
        payout = int(race.get("payout", 0) or 0)
        if payout > 0:
            hits += 1
            total_return += payout  # 払戻金額（100円単位）

    if total_bet == 0:
        return {"roi": 0.0, "hit_rate": 0.0, "count": 0}

    roi = total_return / total_bet
    hit_rate = hits / (total_bet // 100)
    # スコア = ROI重視 + 的中率も少し考慮
    score_val = roi * 0.8 + hit_rate * 0.2

    return {
        "roi": round(roi, 3),
        "hit_rate": round(hit_rate, 3),
        "count": total_bet // 100,
        "hits": hits,
        "score": round(score_val, 3),
    }


def optimize_threshold(races: list[dict], min_count: int = 5) -> tuple[float, float, list]:
    """最適閾値を探索。"""
    thresholds = np.arange(3.5, 7.1, 0.1)
    best_threshold = 5.0
    best_score = 0.0
    results = []

    for th in thresholds:
        th = round(th, 1)
        sim = simulate_threshold(th, races)
        if sim["count"] < min_count:
            continue
        results.append((th, sim))
        if sim["score"] > best_score:
            best_score = sim["score"]
            best_threshold = th

    return best_threshold, best_score, results


def optimize_by_venue(races: list[dict]) -> dict:
    """場ごとに最適閾値を求める。"""
    venue_results = {}
    venues = set(r.get("venue", "") for r in races)

    for venue in venues:
        if not venue:
            continue
        subset = [r for r in races if r.get("venue") == venue]
        if len(subset) < 10:
            venue_results[venue] = {
                "threshold": 5.0,
                "roi": None,
                "count": len(subset),
                "note": "データ不足（10件未満）",
            }
            continue

        # 70/30 分割（過学習防止）
        split = int(len(subset) * 0.7)
        train = subset[:split]
        test  = subset[split:]

        best_th, best_score, _ = optimize_threshold(train, min_count=3)

        # テストで検証
        test_sim = simulate_threshold(best_th, test)

        venue_results[venue] = {
            "threshold": best_th,
            "train_score": round(best_score, 3),
            "test_roi": test_sim["roi"],
            "test_hit_rate": test_sim["hit_rate"],
            "test_count": test_sim["count"],
            "total_count": len(subset),
        }

    return venue_results


def load_hit_record(path: str = "hit_record.csv") -> list[dict]:
    """hit_record.csvを読み込む。"""
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def print_results(venue_results: dict) -> None:
    """結果を表示して最適な閾値設定を提案する。"""
    print("\n" + "="*60)
    print("場ごとの最適閾値シミュレーション結果")
    print("="*60)
    print(f"{'場名':10s} {'推奨閾値':8s} {'件数':6s} {'的中率':8s} {'ROI':8s} {'備考'}")
    print("-"*60)

    venue_thresholds = {}
    for venue, r in sorted(venue_results.items()):
        th     = r.get("threshold", 5.0)
        count  = r.get("total_count", 0)
        hr     = r.get("test_hit_rate")
        roi    = r.get("test_roi")
        note   = r.get("note", "")

        hr_str  = f"{hr:.1%}" if hr is not None else "N/A"
        roi_str = f"{roi:.2f}" if roi is not None else "N/A"

        print(f"{venue:10s} {th:8.1f} {count:6d} {hr_str:8s} {roi_str:8s} {note}")
        venue_thresholds[venue] = th

    print("\n" + "="*60)
    print("notify_arashi.py の VENUE_THRESHOLDS 更新案:")
    print("="*60)
    for venue, th in sorted(venue_thresholds.items()):
        num = VENUE_NAME_TO_NUM.get(venue, "?")
        print(f"    {num}: {th},  # {venue}")


def main():
    races = load_hit_record()

    if not races:
        print("hit_record.csvが見つかりません。")
        print("データが溜まったら再実行してください。")
        print("\n現在のデフォルト閾値設定を使用します。")
        return

    print(f"hit_record.csv: {len(races)}件のデータを読み込みました")

    # 全体最適化
    best_th, best_score, all_results = optimize_threshold(races)
    print(f"\n全体最適閾値: {best_th} (スコア: {best_score:.3f})")

    # 場ごと最適化
    venue_results = optimize_by_venue(races)
    print_results(venue_results)


if __name__ == "__main__":
    main()
