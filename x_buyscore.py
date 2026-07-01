#!/usr/bin/env python3
"""
x_buyscore.py  ── BuyScore エンジン

_evaluate_bets() が算出した候補リストを受け取り、
BuyScore による統合評価・見送り判定・出力整形を行う。

notify_arashi.py の _evaluate_bets() 末尾から呼び出す:
    from x_buyscore import apply_buyscore
    return apply_buyscore(top, context)

【設計思想】
  「何を買うか」より「何を買わないか」を最重要視する。
  BuyScore が閾値未満なら自信を持って「見送り」を返す。
  すべての重み・閾値は buyscore_config.json で管理し、
  x_buyscore_tuner.py が30日実績から自動調整する。
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger("x_buyscore")

JST          = timezone(timedelta(hours=9))
CONFIG_FILE  = "buyscore_config.json"
HITLOG_FILE  = "hit_record.csv"


# ════════════════════════════════════════════════════════════
# 設定ロード
# ════════════════════════════════════════════════════════════

def load_config() -> dict:
    """buyscore_config.json を読み込む。ファイルがなければデフォルト値を返す。"""
    defaults = {
        "weights": {
            "ev": 0.35, "prob": 0.20, "match_index": 0.15,
            "market_gap": 0.10, "uncertainty": -0.10,
            "disagreement": -0.05, "calibration": 0.10, "race_type": 0.05,
        },
        "thresholds": {
            "buyscore_min": 60, "ev_min": 1.28, "match_index_min": 40,
            "uncertainty_max": 0.75, "disagreement_max": 0.80,
            "prob_min": 0.020, "odds_max": 80, "market_gap_max": 3.0,
        },
        "odds_band_bonus": {
            "1_5": -0.15, "6_12": 0.10, "13_25": 0.10,
            "26_40": 0.05, "41_80": -0.05, "80_over": -0.20,
        },
        "race_type_points": {
            "本命戦":  {"max_bets": 2, "label": "堅実"},
            "混戦":    {"max_bets": 4, "label": "期待値重視"},
            "超混戦":  {"max_bets": 6, "label": "穴狙い"},
            "荒れ戦":  {"max_bets": 5, "label": "穴狙い"},
        },
        "star_thresholds": {"5star": 90, "4star": 80, "3star": 70, "2star": 60, "1star": 0},
        "korogashi": {
            "buyscore_min": 75, "prob_min": 0.035, "odds_min": 6,
            "odds_max": 25, "uncertainty_max": 0.55, "match_index_min": 60,
        },
        "ippatsu": {"ev_min": 2.0, "odds_min": 30, "match_index_min": 40},
        "kelly": {
            "fraction": 0.25, "min_buyscore_full": 75,
            "half_below": 60, "zero_below": 50,
        },
        "capital_management": {
            "loss_streak_half": 5, "loss_streak_stop": 10,
            "daily_loss_stop": -10000, "profit_boost_after": 5000,
            "profit_boost_rate": 1.2, "max_dd_stop": -30000,
        },
        "tuning": {
            "enabled": True, "lookback_days": 30,
            "max_change_per_run": 0.05, "min_samples": 50, "regularization": 0.1,
        },
    }
    if not os.path.exists(CONFIG_FILE):
        return defaults
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # デフォルトにマージ（キーが欠けていても動作する）
        for k, v in defaults.items():
            if k not in data:
                data[k] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    data[k].setdefault(kk, vv)
        return data
    except Exception as e:
        log.warning("[BuyScore] config読み込み失敗: %s → デフォルト使用", e)
        return defaults


# ════════════════════════════════════════════════════════════
# BuyScore 計算
# ════════════════════════════════════════════════════════════

def _odds_band_bonus(odds: float, band_cfg: dict) -> float:
    """オッズ帯補正値を返す（⑤ オッズ帯補正）"""
    if odds <= 5:    return band_cfg.get("1_5",    0.0)
    if odds <= 12:   return band_cfg.get("6_12",   0.0)
    if odds <= 25:   return band_cfg.get("13_25",  0.0)
    if odds <= 40:   return band_cfg.get("26_40",  0.0)
    if odds <= 80:   return band_cfg.get("41_80",  0.0)
    return band_cfg.get("80_over", 0.0)


def _race_type_bonus(race_type: str) -> float:
    """レースタイプ補正の正規化値（④ レースタイプ別ロジック）。
    荒れ戦・本命戦は明確なタイプなので加点、混戦は0。"""
    bonus_map = {"本命戦": 0.05, "混戦": 0.0, "超混戦": 0.03, "荒れ戦": 0.05}
    return bonus_map.get(race_type, 0.0)


def calc_buyscore(
    candidate: dict,
    context: dict,
    cfg: dict,
) -> float:
    """
    1つの買い目候補に対してBuyScoreを計算する（0〜100点）。

    candidate: _evaluate_bets() が返す1件の辞書
        keys: combo, prob, ev, odds, composite, bet_score, uncertainty, disagreement, ...
    context: レース全体の情報
        keys: match_index, upset_score, race_type, has_exhibition, market_gap, ...
    cfg: load_config() の戻り値

    BuyScore = Σ(要素 × 重み) × 100 に各補正を加算して 0〜100 にクランプ
    """
    w   = cfg["weights"]
    thr = cfg["thresholds"]

    ev          = candidate.get("ev", 0.0)
    prob        = candidate.get("prob", 0.0)
    odds        = candidate.get("odds", 0.0)
    uncertainty = candidate.get("uncertainty", 0.5)
    disagreement = candidate.get("disagreement", 0.5)
    composite   = candidate.get("composite", 0.0)  # キャリブレーション後信頼度の近似

    match_index = context.get("match_index", 0.0)   # 0〜100
    race_type   = context.get("race_type", "混戦")
    market_gap  = context.get("market_gap", 0.0)    # 市場とAIのオッズ乖離率

    # ── 各要素を 0.0〜1.0 に正規化 ──────────────────────────
    # 分母は競艇3連単の現実的な優良水準に合わせる:
    #   EV 2.0 で満点相当（3.0は稀すぎて基準にすると全体が沈む）
    #   的中率 8% で満点相当（三連単では10%超は稀）
    ev_n          = min(ev / 2.0, 1.0)
    prob_n        = min(prob / 0.08, 1.0)
    match_n       = min(match_index / 100.0, 1.0)
    market_gap_n  = min(abs(market_gap) / thr.get("market_gap_max", 3.0), 1.0)
    uncertainty_n = min(uncertainty, 1.0)
    disagree_n    = min(disagreement, 1.0)
    calib_n       = composite  # すでに 0〜1

    # ── 重み付き合算 ─────────────────────────────────────────
    raw = (
        ev_n          * w.get("ev",           0.35)
        + prob_n        * w.get("prob",          0.20)
        + match_n       * w.get("match_index",   0.15)
        + market_gap_n  * w.get("market_gap",    0.10)
        + uncertainty_n * w.get("uncertainty",  -0.10)  # 負の重み
        + disagree_n    * w.get("disagreement", -0.05)  # 負の重み
        + calib_n       * w.get("calibration",   0.10)
    )

    # ── レースタイプ補正（重み race_type を通して1回だけ加算）──
    raw += w.get("race_type", 0.05) * _race_type_bonus(race_type)

    # ── オッズ帯補正 ──────────────────────────────────────────
    raw += _odds_band_bonus(odds, cfg.get("odds_band_bonus", {}))

    # ── 展示未取得ペナルティ ──────────────────────────────────
    if not context.get("has_exhibition", True):
        raw -= 0.08

    # 0〜100 にスケール・クランプ
    score = max(0.0, min(raw * 100.0, 100.0))
    return round(score, 1)


# ════════════════════════════════════════════════════════════
# 見送り判定
# ════════════════════════════════════════════════════════════

def check_passthrough(
    candidates: list[dict],
    context: dict,
    cfg: dict,
) -> Optional[str]:
    """
    見送りなら理由文字列を返す。買うなら None を返す。
    ② 見送り判定
    """
    thr = cfg["thresholds"]

    if not candidates:
        return "候補なし"

    best = candidates[0]
    best_score = best.get("buyscore", 0.0)
    ev          = best.get("ev", 0.0)
    uncertainty = best.get("uncertainty", 1.0)
    disagreement = best.get("disagreement", 1.0)
    match_index = context.get("match_index", 0.0)
    has_exhibition = context.get("has_exhibition", True)

    # BuyScore 不足
    if best_score < thr.get("buyscore_min", 60):
        return f"BuyScore不足({best_score:.0f})"

    # 期待値不足
    if ev < thr.get("ev_min", 1.28):
        return f"期待値不足(EV{ev:.2f})"

    # AI一致指数不足
    # ただし match_index が近似値（新聞側の正確な値が未確定）の場合は、
    # これ単独では見送りにしない（誤って全レース見送りになるのを防ぐ）。
    if match_index < thr.get("match_index_min", 40):
        if not context.get("match_index_approx", False):
            return f"AI一致指数不足({match_index:.0f})"

    # 不確実性過高
    if uncertainty > thr.get("uncertainty_max", 0.75):
        return f"不確実性高({uncertainty:.2f})"

    # 展示未取得 + 信頼度不足
    composite = best.get("composite", 0.0)
    if not has_exhibition and composite < 0.40:
        return f"展示なし+信頼度不足({composite:.2f})"

    # モデル間意見割れ
    if disagreement > thr.get("disagreement_max", 0.80):
        return f"モデル不一致({disagreement:.2f})"

    return None


# ════════════════════════════════════════════════════════════
# 星ランク・投資タイプ・リスク
# ════════════════════════════════════════════════════════════

def stars(buyscore: float, cfg: dict) -> str:
    """⑨ 信頼度ランク: ★★★★★ 表示"""
    thr = cfg.get("star_thresholds", {})
    if buyscore >= thr.get("5star", 90): return "★★★★★"
    if buyscore >= thr.get("4star", 80): return "★★★★☆"
    if buyscore >= thr.get("3star", 70): return "★★★☆☆"
    if buyscore >= thr.get("2star", 60): return "★★☆☆☆"
    return "★☆☆☆☆"


def investment_type(buyscore: float, ev: float, odds: float, race_type: str) -> str:
    """⑭ 投資タイプ分類"""
    if buyscore < 60:             return "見送り"
    if race_type == "本命戦":     return "堅実"
    if ev >= 2.0 and odds >= 30:  return "穴狙い"
    if ev >= 1.5:                 return "期待値重視"
    return "堅実"


def risk_level(uncertainty: float, disagreement: float, odds: float) -> str:
    """⑬ リスク表示"""
    score = uncertainty * 0.5 + disagreement * 0.3 + min(odds / 200, 0.2)
    if score >= 0.70: return "非常に高い"
    if score >= 0.50: return "高"
    if score >= 0.30: return "中"
    return "低"


# ════════════════════════════════════════════════════════════
# Kelly 改善
# ════════════════════════════════════════════════════════════

def kelly_fraction(
    prob: float,
    odds: float,
    buyscore: float,
    cfg: dict,
) -> float:
    """
    ⑦ BuyScore・信頼度を反映した Kelly 係数。
    低信頼なら半減、見送りなら 0 を返す。
    """
    kelly_cfg = cfg.get("kelly", {})
    zero_below = kelly_cfg.get("zero_below", 50)
    half_below = kelly_cfg.get("half_below", 60)
    fraction   = kelly_cfg.get("fraction", 0.25)

    if buyscore < zero_below:
        return 0.0

    # 基本 Kelly
    b = odds - 1.0
    q = 1.0 - prob
    raw_kelly = (b * prob - q) / b if b > 0 else 0.0
    if raw_kelly <= 0:
        return 0.0

    # フラクショナル Kelly
    k = raw_kelly * fraction

    # BuyScore による縮小
    if buyscore < half_below:
        k *= 0.5

    return round(max(k, 0.0), 4)


# ════════════════════════════════════════════════════════════
# 資金管理（⑧ 拡張版）
# ════════════════════════════════════════════════════════════

def get_bet_multiplier_extended(cfg: dict) -> tuple[float, str]:
    """
    ⑧ 拡張資金管理。
    戻り値: (multiplier, reason)
      multiplier: 0.0（停止）/ 0.5（縮小）/ 1.0（通常）/ 1.2（好調増加）
    """
    cm = cfg.get("capital_management", {})
    streak_half  = cm.get("loss_streak_half", 5)
    streak_stop  = cm.get("loss_streak_stop", 10)
    daily_stop   = cm.get("daily_loss_stop", -10000)
    profit_boost = cm.get("profit_boost_after", 5000)
    boost_rate   = cm.get("profit_boost_rate", 1.2)
    max_dd_stop  = cm.get("max_dd_stop", -30000)

    try:
        import csv as _csv
        if not os.path.exists(HITLOG_FILE):
            return 1.0, "通常"
        with open(HITLOG_FILE, "r", encoding="utf-8") as f:
            rows = list(_csv.DictReader(f))
        # 重複除去（x_verification と同じキー）
        seen: set[tuple] = set()
        unique = []
        for r in rows:
            k = (r.get("date",""), str(r.get("venue_num","")),
                 str(r.get("race","")), r.get("pred_combo",""))
            if k not in seen:
                seen.add(k)
                unique.append(r)
        rows = unique

        notified = [r for r in rows if r.get("pred_combo") and
                    r.get("hit") not in ("", None, "-1")]
        if not notified:
            return 1.0, "通常"

        # 連敗チェック
        streak = 0
        for r in reversed(notified):
            if int(r.get("hit", 0) or 0) == 0:
                streak += 1
            else:
                break
        if streak >= streak_stop:
            log.warning("連敗%d回 → ベット停止", streak)
            return 0.0, f"連敗{streak}回停止"
        if streak >= streak_half:
            log.info("連敗%d回 → ベット縮小", streak)
            return 0.5, f"連敗{streak}回縮小"

        # 日次損失チェック
        today = datetime.now(JST).strftime("%Y%m%d")
        today_rows = [r for r in rows if str(r.get("date","")).replace("-","") == today]
        daily_profit = sum(int(r.get("profit") or -100) for r in today_rows)
        if daily_profit <= daily_stop:
            log.warning("日次損失%d円 → 停止", daily_profit)
            return 0.0, f"日次損失{daily_profit}円停止"

        # 最大ドローダウンチェック
        cumulative = 0
        peak = 0
        max_dd = 0
        for r in notified:
            cumulative += int(r.get("profit") or -100)
            peak = max(peak, cumulative)
            max_dd = min(max_dd, cumulative - peak)
        if max_dd <= max_dd_stop:
            log.warning("最大DD%d円 → 停止", max_dd)
            return 0.0, f"最大DD{max_dd}円停止"

        # 好調時ベット増加（直近5戦が利益更新中）
        if len(notified) >= 5:
            recent5_profit = sum(int(r.get("profit") or 0) for r in notified[-5:])
            if recent5_profit >= profit_boost:
                return boost_rate, f"好調+{recent5_profit}円増加"

        return 1.0, "通常"

    except Exception as e:
        log.warning("[資金管理] 計算失敗: %s", e)
        return 1.0, "通常"


# ════════════════════════════════════════════════════════════
# 転がし・一撃判定
# ════════════════════════════════════════════════════════════

def check_korogashi(candidate: dict, context: dict, cfg: dict) -> bool:
    """⑮ 転がし向き判定"""
    kc = cfg.get("korogashi", {})
    return (
        candidate.get("buyscore", 0) >= kc.get("buyscore_min", 75)
        and candidate.get("prob", 0) >= kc.get("prob_min", 0.035)
        and kc.get("odds_min", 6) <= candidate.get("odds", 0) <= kc.get("odds_max", 25)
        and candidate.get("uncertainty", 1.0) <= kc.get("uncertainty_max", 0.55)
        and context.get("match_index", 0) >= kc.get("match_index_min", 60)
    )


def check_ippatsu(candidate: dict, context: dict, cfg: dict) -> bool:
    """⑯ 一撃向き（高配当）判定"""
    ic = cfg.get("ippatsu", {})
    return (
        candidate.get("ev", 0) >= ic.get("ev_min", 2.0)
        and candidate.get("odds", 0) >= ic.get("odds_min", 30)
        and context.get("match_index", 0) >= ic.get("match_index_min", 40)
    )


# ════════════════════════════════════════════════════════════
# ◎○▲△ ラベル付与（⑩ 最大4点に厳選）
# ════════════════════════════════════════════════════════════

RANK_LABELS = ["◎", "○", "▲", "△"]
RANK_NAMES  = ["本命", "対抗", "穴", "押さえ"]


def assign_rank_labels(candidates: list[dict], race_type: str, cfg: dict) -> list[dict]:
    """
    BuyScore 順に並んだ候補に ◎○▲△ を付ける。
    レースタイプで最大点数を決める（⑩）。
    """
    rt_cfg = cfg.get("race_type_points", {})
    max_bets = rt_cfg.get(race_type, {}).get("max_bets", 4)
    max_bets = min(max_bets, 4)  # 絶対上限は4点

    result = []
    for i, c in enumerate(candidates[:max_bets]):
        c = dict(c)
        c["rank_label"] = RANK_LABELS[i] if i < len(RANK_LABELS) else "△"
        c["rank_name"]  = RANK_NAMES[i]  if i < len(RANK_NAMES)  else "押さえ"
        result.append(c)
    return result


# ════════════════════════════════════════════════════════════
# 採用理由・除外理由テキスト生成
# ════════════════════════════════════════════════════════════

def format_buy_reason(c: dict, context: dict) -> str:
    """⑪ 採用理由テキスト"""
    parts = [
        f"EV{c.get('ev',0):.2f}",
        f"AI一致{context.get('match_index',0):.0f}",
    ]
    market_gap = context.get("market_gap", 0)
    if abs(market_gap) >= 0.05:
        sign = "+" if market_gap > 0 else ""
        parts.append(f"市場乖離{sign}{market_gap*100:.0f}%")
    ex_rank = context.get("ex_rank_1st", 0)
    if ex_rank == 1:
        parts.append("展示1位")
    if c.get("pattern") == "モンスター":
        parts.append("モーター最上位")
    return "　".join(parts)


def format_exclude_reason(c: dict, thr: dict) -> str:
    """⑫ 除外理由テキスト"""
    reasons = []
    if c.get("ev", 0) < thr.get("ev_min", 1.28):
        reasons.append("期待値不足")
    if c.get("buyscore", 100) < thr.get("buyscore_min", 60):
        reasons.append(f"BuyScore{c.get('buyscore',0):.0f}")
    match_index = c.get("_match_index", 0)
    if match_index < thr.get("match_index_min", 40):
        reasons.append(f"AI一致{match_index:.0f}")
    if not reasons:
        reasons.append("総合スコア不足")
    return "　".join(reasons)


# ════════════════════════════════════════════════════════════
# 見送りメッセージ生成
# ════════════════════════════════════════════════════════════

def format_passthrough_message(
    reason: str,
    best: Optional[dict],
    context: dict,
) -> str:
    """見送り時の出力テキスト（②）"""
    lines = ["━━━━━━━━━━━━"]
    lines.append("⛔ 見送り")
    lines.append(f"理由: {reason}")
    if best:
        lines.append(f"BuyScore {best.get('buyscore', 0):.0f}")
        lines.append(f"信頼度 {best.get('composite', 0)*100:.0f}%")
        lines.append(f"AI一致 {context.get('match_index', 0):.0f}")
    lines.append("━━━━━━━━━━━━")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# 最終出力フォーマット
# ════════════════════════════════════════════════════════════

def format_buy_message(
    ranked: list[dict],
    context: dict,
    capital_reason: str,
    cfg: dict,
) -> str:
    """㉑ 最終出力テキスト"""
    race_type  = context.get("race_type", "混戦")
    inv_type   = investment_type(
        ranked[0].get("buyscore", 0),
        ranked[0].get("ev", 0),
        ranked[0].get("odds", 0),
        race_type,
    ) if ranked else "見送り"

    lines = ["━━━━━━━━━━━━"]
    lines.append(f"【{inv_type}】 {race_type} {capital_reason}")

    for c in ranked:
        bs     = c.get("buyscore", 0)
        star   = stars(bs, cfg)
        label  = c.get("rank_label", "△")
        combo  = c.get("combo", "")
        ev     = c.get("ev", 0)
        odds   = c.get("odds", 0)
        amount = c.get("amount", 0)
        risk   = risk_level(
            c.get("uncertainty", 0.5),
            c.get("disagreement", 0.5),
            odds,
        )
        reason = format_buy_reason(c, context)

        lines.append(
            f"{label} {combo}  {star}  "
            f"EV{ev:.2f} / {odds:.0f}倍 / {amount}円"
        )
        lines.append(f"  [{reason}]  リスク:{risk}")

        # 転がし・一撃マーク
        if c.get("is_korogashi"):
            lines.append("  🎯 転がし候補")
        if c.get("is_ippatsu"):
            lines.append("  💥 一撃候補")

    lines.append("━━━━━━━━━━━━")
    return "\n".join(lines)


# ════════════════════════════════════════════════════════════
# 学習用ログ保存（⑰）
# ════════════════════════════════════════════════════════════

BUYSCORE_LOG = "buyscore_log.jsonl"


def save_buyscore_log(
    venue: str,
    race: str,
    date: str,
    candidates: list[dict],
    passthrough_reason: Optional[str],
    context: dict,
    capital_reason: str,
) -> None:
    """⑰ 予測・BuyScore・見送り理由などを JSONL に保存する"""
    record = {
        "date":               date,
        "venue":              venue,
        "race":               race,
        "passthrough_reason": passthrough_reason,
        "capital_reason":     capital_reason,
        "match_index":        context.get("match_index", 0),
        "race_type":          context.get("race_type", ""),
        "candidates": [
            {
                "combo":       c.get("combo"),
                "buyscore":    c.get("buyscore"),
                "ev":          c.get("ev"),
                "prob":        c.get("prob"),
                "odds":        c.get("odds"),
                "kelly":       c.get("kelly"),
                "amount":      c.get("amount"),
                "uncertainty": c.get("uncertainty"),
                "disagreement": c.get("disagreement"),
                "is_korogashi": c.get("is_korogashi", False),
                "is_ippatsu":   c.get("is_ippatsu", False),
                "rank_label":   c.get("rank_label", ""),
            }
            for c in candidates
        ],
    }
    try:
        with open(BUYSCORE_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning("[BuyScore] ログ保存失敗: %s", e)


# ════════════════════════════════════════════════════════════
# メインエントリ: apply_buyscore()
# ════════════════════════════════════════════════════════════

def apply_buyscore(
    candidates: list[dict],
    context: dict,
    venue: str = "",
    race: str = "",
    date: str = "",
) -> dict:
    """
    _evaluate_bets() の結果にBuyScoreを適用し、最終的な買い目または見送りを返す。

    context キー:
        match_index   : AI一致指数 (0〜100)
        race_type     : レースタイプ文字列
        has_exhibition: 展示データあり/なし
        market_gap    : 市場とAIのオッズ乖離率
        ex_rank_1st   : 展示1位の号艇番号
        upset_score   : 荒れスコア (0〜9.5)

    戻り値:
        {
            "buy":             list[dict],  # 最終買い目（見送りなら空）
            "passthrough":     bool,        # True = 見送り
            "passthrough_reason": str,
            "message":         str,         # 出力テキスト
            "capital_reason":  str,
        }
    """
    cfg = load_config()
    thr = cfg["thresholds"]

    # ── BuyScore を各候補に付与 ──────────────────────────────
    for c in candidates:
        c["buyscore"]    = calc_buyscore(c, context, cfg)
        c["stars"]       = stars(c["buyscore"], cfg)
        c["is_korogashi"] = check_korogashi(c, context, cfg)
        c["is_ippatsu"]   = check_ippatsu(c, context, cfg)
        c["kelly"]        = kelly_fraction(
            c.get("prob", 0), c.get("odds", 0), c["buyscore"], cfg
        )

    # BuyScore 降順に並べ替え
    candidates.sort(key=lambda x: x["buyscore"], reverse=True)

    # ── 資金管理チェック ─────────────────────────────────────
    multiplier, capital_reason = get_bet_multiplier_extended(cfg)
    if multiplier == 0.0:
        save_buyscore_log(venue, race, date, candidates, capital_reason, context, capital_reason)
        return {
            "buy": [],
            "passthrough": True,
            "passthrough_reason": capital_reason,
            "message": format_passthrough_message(capital_reason, candidates[0] if candidates else None, context),
            "capital_reason": capital_reason,
        }

    # ── 見送り判定 ───────────────────────────────────────────
    pass_reason = check_passthrough(candidates, context, cfg)
    if pass_reason:
        msg = format_passthrough_message(pass_reason, candidates[0] if candidates else None, context)
        save_buyscore_log(venue, race, date, candidates, pass_reason, context, capital_reason)
        return {
            "buy": [],
            "passthrough": True,
            "passthrough_reason": pass_reason,
            "message": msg,
            "capital_reason": capital_reason,
        }

    # ── ◎○▲△ ラベル付与・点数制限 ──────────────────────────
    race_type = context.get("race_type", "混戦")
    ranked    = assign_rank_labels(candidates, race_type, cfg)

    # ── ベット金額にドローダウン倍率を反映 ──────────────────
    for c in ranked:
        c["amount"] = int(c.get("amount", 500) * multiplier / 100) * 100

    # ── 除外された人気買い目の理由を生成（⑫）──────────────
    excluded = []
    for c in candidates:
        if c not in ranked and c.get("odds", 999) <= 15:  # 人気帯のみ対象
            c["_match_index"] = context.get("match_index", 0)
            excluded.append({
                "combo":  c.get("combo"),
                "reason": format_exclude_reason(c, thr),
            })

    # ── 最終メッセージ生成 ───────────────────────────────────
    msg = format_buy_message(ranked, context, capital_reason, cfg)

    # ── 学習用ログ保存 ───────────────────────────────────────
    save_buyscore_log(venue, race, date, candidates, None, context, capital_reason)

    return {
        "buy":                ranked,
        "passthrough":        False,
        "passthrough_reason": None,
        "message":            msg,
        "capital_reason":     capital_reason,
        "excluded":           excluded,
    }
