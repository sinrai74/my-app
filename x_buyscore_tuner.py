#!/usr/bin/env python3
"""
x_buyscore_tuner.py  ── BuyScore 重み自動チューニング（⑲）

過去30日の buyscore_log.jsonl と hit_record.csv を突合し、
各要素（EV・確率・AI一致指数など）と的中の相関から
最適な重みを推定して buyscore_config.json を更新する。

【設計方針】
  ・急激な変更は禁止: 1回の実行で ±max_change_per_run（デフォルト0.05）以内
  ・サンプル不足時は更新しない（min_samples: デフォルト50件）
  ・重みの総和は正の重みが1.0になるよう正規化する
  ・過学習防止のため regularization（L2正則化）を適用する

Usage:
    python x_buyscore_tuner.py           # 30日分で自動チューニング
    python x_buyscore_tuner.py --dry-run # 実際には保存しない（確認用）
    python x_buyscore_tuner.py --report  # 現在の重みと精度レポートのみ表示

実行タイミング:
    x_ranking.yml の verification ステップ（毎日21:00）に追加することを推奨。
    python x_buyscore_tuner.py || echo "チューニング失敗（スキップ）"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

log = logging.getLogger("x_buyscore_tuner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

JST            = timezone(timedelta(hours=9))
CONFIG_FILE    = "buyscore_config.json"
BUYSCORE_LOG   = "buyscore_log.jsonl"
HITLOG_FILE    = "hit_record.csv"
ANALYSIS_LOG   = "buyscore_analysis.json"  # ⑱ 自動分析結果


# ════════════════════════════════════════════════════════════
# ユーティリティ
# ════════════════════════════════════════════════════════════

def _safe_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        log.error("buyscore_config.json が見つかりません")
        return {}
    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(cfg: dict) -> None:
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    log.info("[チューニング] buyscore_config.json を更新しました")


# ════════════════════════════════════════════════════════════
# hit_record.csv の読み込み（重複除去付き）
# ════════════════════════════════════════════════════════════

def load_hit_records(lookback_days: int = 30) -> list[dict]:
    if not os.path.exists(HITLOG_FILE):
        return []
    cutoff = (datetime.now(JST) - timedelta(days=lookback_days)).strftime("%Y%m%d")
    seen: set[tuple] = set()
    records = []
    with open(HITLOG_FILE, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            date = row.get("date", "").replace("-", "")
            if date < cutoff:
                continue
            if row.get("hit") in ("", None, "-1"):
                continue
            if not row.get("pred_combo"):
                continue
            k = (date, str(row.get("venue_num","")),
                 str(row.get("race","")), row.get("pred_combo",""))
            if k not in seen:
                seen.add(k)
                records.append(row)
    return records


# ════════════════════════════════════════════════════════════
# buyscore_log.jsonl の読み込み
# ════════════════════════════════════════════════════════════

def load_buyscore_logs(lookback_days: int = 30) -> list[dict]:
    if not os.path.exists(BUYSCORE_LOG):
        return []
    cutoff = (datetime.now(JST) - timedelta(days=lookback_days)).strftime("%Y%m%d")
    records = []
    with open(BUYSCORE_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("date", "") >= cutoff:
                    records.append(r)
            except json.JSONDecodeError:
                continue
    return records


# ════════════════════════════════════════════════════════════
# buyscore_log と hit_record を突合して特徴量・的中を得る
# ════════════════════════════════════════════════════════════

def build_training_data(
    buyscore_logs: list[dict],
    hit_records: list[dict],
) -> list[dict]:
    """
    buyscore_log の各買い目エントリと hit_record を
    date + venue + race + combo で突合してラベル（hit）を付ける。
    """
    # hit_record を (date, race, combo) → hit でマッピング
    hit_map: dict[tuple, int] = {}
    for r in hit_records:
        k = (
            r.get("date", "").replace("-", ""),
            str(r.get("race", "")),
            r.get("pred_combo", ""),
        )
        hit_map[k] = int(r.get("hit") or 0)

    training = []
    for log_entry in buyscore_logs:
        date  = log_entry.get("date", "")
        race  = str(log_entry.get("race", ""))
        match_index = _safe_float(log_entry.get("match_index"))

        for c in log_entry.get("candidates", []):
            combo = c.get("combo", "")
            k = (date, race, combo)
            if k not in hit_map:
                continue  # 突合できないものはスキップ

            training.append({
                "hit":         hit_map[k],
                "ev":          _safe_float(c.get("ev")),
                "prob":        _safe_float(c.get("prob")),
                "match_index": match_index,
                "uncertainty": _safe_float(c.get("uncertainty")),
                "disagreement":_safe_float(c.get("disagreement")),
                "buyscore":    _safe_float(c.get("buyscore")),
                "kelly":       _safe_float(c.get("kelly")),
                "odds":        _safe_float(c.get("odds")),
            })

    return training


# ════════════════════════════════════════════════════════════
# 重み推定（相関ベース + L2正則化）
# ════════════════════════════════════════════════════════════

def estimate_weights(
    training: list[dict],
    current_weights: dict,
    max_change: float = 0.05,
    regularization: float = 0.1,
) -> dict:
    """
    各特徴量と hit の相関係数から重みの更新量を推定する。
    急激な変更防止のため max_change で変化量をクランプする。
    """
    features = ["ev", "prob", "match_index", "uncertainty", "disagreement"]
    hits   = [d["hit"] for d in training]
    n      = len(training)
    hit_mean = sum(hits) / n

    correlations: dict[str, float] = {}
    for feat in features:
        vals = [d.get(feat, 0.0) for d in training]
        v_mean = sum(vals) / n

        cov = sum((vals[i] - v_mean) * (hits[i] - hit_mean) for i in range(n)) / n
        std_v = math.sqrt(sum((v - v_mean)**2 for v in vals) / n + 1e-9)
        std_h = math.sqrt(sum((h - hit_mean)**2 for h in hits) / n + 1e-9)

        corr = cov / (std_v * std_h)
        # L2正則化（0に引き戻す力）
        corr = corr * (1.0 - regularization)
        correlations[feat] = round(corr, 4)

    log.info("[チューニング] 相関係数: %s", correlations)

    # 現在の重みに変化量を加算（クランプ付き）
    new_weights = dict(current_weights)
    for feat in features:
        corr = correlations.get(feat, 0.0)
        # 相関が正 → 重みを増やす / 負 → 減らす（ただし符号は仕様通りに維持）
        # uncertainty と disagreement は元々負なので符号反転
        sign = -1 if feat in ("uncertainty", "disagreement") else 1
        delta = corr * sign * 0.1  # 相関 × スケーリング
        delta = max(-max_change, min(max_change, delta))  # クランプ
        new_weights[feat] = round(current_weights.get(feat, 0.0) + delta, 4)

    # 更新した特徴量の正の重みだけを、更新前の合計に合わせて再正規化する。
    # （market_gap / calibration / race_type などチューニング対象外の重みは
    #  変更しない。毎回 ×0.85 のような減衰もかけない。）
    tuned_pos_before = sum(
        current_weights.get(f, 0.0) for f in features
        if current_weights.get(f, 0.0) > 0
    )
    tuned_pos_after = sum(
        new_weights.get(f, 0.0) for f in features
        if new_weights.get(f, 0.0) > 0
    )
    if tuned_pos_after > 0 and tuned_pos_before > 0:
        scale = tuned_pos_before / tuned_pos_after
        for f in features:
            if new_weights.get(f, 0.0) > 0:
                new_weights[f] = round(new_weights[f] * scale, 4)

    return new_weights


# ════════════════════════════════════════════════════════════
# ⑱ 自動分析レポート生成
# ════════════════════════════════════════════════════════════

def generate_analysis_report(training: list[dict], cfg: dict) -> dict:
    """
    どの条件で利益が出たか / 損失になったかを分析してレポートを返す（⑱）。
    """
    if not training:
        return {"error": "データ不足"}

    # BuyScore帯別の的中率
    score_bands = defaultdict(lambda: {"hit": 0, "total": 0})
    for d in training:
        bs = d.get("buyscore", 0)
        band = f"{int(bs // 10) * 10}-{int(bs // 10) * 10 + 9}"
        score_bands[band]["total"] += 1
        score_bands[band]["hit"]   += d["hit"]

    # オッズ帯別の的中率
    odds_bands = defaultdict(lambda: {"hit": 0, "total": 0})
    for d in training:
        odds = d.get("odds", 0)
        if odds <= 5:    band = "1-5"
        elif odds <= 12: band = "6-12"
        elif odds <= 25: band = "13-25"
        elif odds <= 40: band = "26-40"
        elif odds <= 80: band = "41-80"
        else:            band = "80+"
        odds_bands[band]["total"] += 1
        odds_bands[band]["hit"]   += d["hit"]

    # EV帯別
    ev_bands = defaultdict(lambda: {"hit": 0, "total": 0, "roi": 0.0})
    for d in training:
        ev = d.get("ev", 0)
        band = f"EV{int(ev * 10) / 10:.1f}-{int(ev * 10) / 10 + 0.2:.1f}"
        ev_bands[band]["total"] += 1
        ev_bands[band]["hit"]   += d["hit"]

    # 全体成績
    total = len(training)
    hits  = sum(d["hit"] for d in training)
    rate  = round(hits / total * 100, 1) if total else 0

    report = {
        "generated_at":    datetime.now(JST).isoformat(),
        "total_records":   total,
        "hit_count":       hits,
        "hit_rate":        rate,
        "score_bands":     {k: {
            "hit_rate": round(v["hit"] / v["total"] * 100, 1) if v["total"] else 0,
            **v
        } for k, v in sorted(score_bands.items())},
        "odds_bands":      {k: {
            "hit_rate": round(v["hit"] / v["total"] * 100, 1) if v["total"] else 0,
            **v
        } for k, v in sorted(odds_bands.items())},
        "current_weights": cfg.get("weights", {}),
    }
    return report


# ════════════════════════════════════════════════════════════
# メイン
# ════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="BuyScore 重み自動チューニング")
    parser.add_argument("--dry-run", action="store_true", help="保存しない（確認のみ）")
    parser.add_argument("--report",  action="store_true", help="分析レポートのみ表示")
    args = parser.parse_args()

    cfg = load_config()
    if not cfg:
        return

    tuning_cfg   = cfg.get("tuning", {})
    enabled      = tuning_cfg.get("enabled", True)
    lookback     = tuning_cfg.get("lookback_days", 30)
    max_change   = tuning_cfg.get("max_change_per_run", 0.05)
    min_samples  = tuning_cfg.get("min_samples", 50)
    regularization = tuning_cfg.get("regularization", 0.1)

    log.info("[チューニング] 過去%d日分のデータを読み込み中...", lookback)
    hit_records    = load_hit_records(lookback)
    buyscore_logs  = load_buyscore_logs(lookback)

    log.info("  hit_record: %d件 / buyscore_log: %dレース",
             len(hit_records), len(buyscore_logs))

    training = build_training_data(buyscore_logs, hit_records)
    log.info("  突合成功: %d件", len(training))

    # ── ⑱ 分析レポート生成（毎回） ──────────────────────────
    report = generate_analysis_report(training, cfg)
    try:
        with open(ANALYSIS_LOG, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        log.info("[分析] buyscore_analysis.json を保存しました")
    except Exception as e:
        log.warning("[分析] 保存失敗: %s", e)

    # レポートのみモード
    if args.report:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    # ── ⑲ 重み自動チューニング ──────────────────────────────
    if not enabled:
        log.info("[チューニング] 無効（config: tuning.enabled=false）")
        return

    if len(training) < min_samples:
        log.info(
            "[チューニング] サンプル不足（%d件 < %d件）→ 更新スキップ",
            len(training), min_samples,
        )
        return

    current_weights = cfg.get("weights", {})
    new_weights = estimate_weights(
        training, current_weights, max_change, regularization
    )

    log.info("[チューニング] 重み変化:")
    for k in new_weights:
        old_v = current_weights.get(k, 0.0)
        new_v = new_weights[k]
        diff  = new_v - old_v
        sign  = "+" if diff >= 0 else ""
        log.info("  %-15s: %.4f → %.4f (%s%.4f)", k, old_v, new_v, sign, diff)

    if args.dry_run:
        log.info("[チューニング] --dry-run モード: 保存しません")
        return

    cfg["weights"] = new_weights
    cfg["tuning"]["last_tuned"] = datetime.now(JST).isoformat()
    cfg["tuning"]["last_samples"] = len(training)
    save_config(cfg)


if __name__ == "__main__":
    main()
