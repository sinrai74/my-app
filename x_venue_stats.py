#!/usr/bin/env python3
"""
x_venue_stats.py  ── 場別コース統計・水面タイプ分類（Ver4評価エンジン用）

【重要な設計方針】
24場×6コースの「1着率・2連対率・3連対率・平均ST」は競艇業界で半期ごとに
公表される統計だが、値は毎期変動し、かつ全24場×6コース分を裏付けを
持って正確に把握することはできない。したがって本モジュールは、
外部の統計値を決め打ちでコードに埋め込むのではなく、
本システム自身が蓄積した実データ（motor_history.csv / 将来的には
hit_record.csv の結果列）から場×コース別統計を都度算出する。

データが少ない場・コースはサンプル数補正（信頼度補正）で全国平均へ
縮約する。データが全く無い場合は「補正なし（全国平均相当）」として
扱い、存在しない数値を捏造しない。

水面タイプ分類（超イン水面／イン有利／標準／荒れ水面）も、
実データから算出した「その場の1コース1着率」を config の閾値と
比較して自動判定する。ごく一部の著名な場（徳山・大村など、
1コースが極端に有利なことが業界内で広く知られている場）については
コールドスタート用の参考初期値を venue_config.json に記載しているが、
これは実データが十分蓄積されるまでの暫定値であり、データが蓄積され
次第、実データによる自動算出が優先される。
"""

from __future__ import annotations

import csv
import json
import logging
import os
from collections import defaultdict
from typing import Optional

import x_release_storage

log = logging.getLogger("x_venue_stats")

VENUE_CONFIG_FILE = "venue_config.json"
MOTOR_HISTORY_FILE = "motor_history.csv"

_VENUE_STATS_CACHE: Optional[dict] = None
_VENUE_CONFIG_CACHE: Optional[dict] = None


# ════════════════════════════════════════════════════════════
# config 読み込み
# ════════════════════════════════════════════════════════════

def load_venue_config(force_reload: bool = False) -> dict:
    """venue_config.json を読み込む。存在しない場合は最低限のデフォルトを返す。"""
    global _VENUE_CONFIG_CACHE
    if _VENUE_CONFIG_CACHE is not None and not force_reload:
        return _VENUE_CONFIG_CACHE

    defaults = {
        "_comment": "場別統計・水面タイプ判定の設定。数値は実データ(motor_history.csv)から自動算出することを前提とし、ここでは閾値・補正強度のみを管理する。",
        "sample_size_correction": {
            "_comment": "サンプル数補正（信頼度補正）。ベイズ的シュリンケージ: "
                        "補正後 = (実測値×サンプル数 + 全国平均×prior_strength) / (サンプル数 + prior_strength)。"
                        "サンプル数が少ないほど全国平均に近づく。",
            "prior_strength": 20,
            "min_samples_for_full_trust": 100
        },
        "water_type_thresholds": {
            "_comment": "その場の実測1コース1着率(%)に基づく水面タイプ自動分類の閾値。",
            "super_in": 60.0,
            "in_favorable": 53.0,
            "standard_min": 45.0,
            "_below_standard_min_is": "rough"
        },
        "water_type_labels": {
            "super_in": "超イン水面",
            "in_favorable": "イン有利",
            "standard": "標準",
            "rough": "荒れ水面"
        },
        "cold_start_hints": {
            "_comment": "実データが十分蓄積されるまでの暫定的な参考値。業界内で広く知られる著名な傾向のみを記載（出典: 各種競艇統計コラム、2023-2025年時点の傾向）。データ蓄積後は自動算出値が優先される。",
            "徳山": "super_in", "大村": "super_in", "芦屋": "super_in",
            "江戸川": "rough", "平和島": "rough", "鳴門": "rough"
        }
    }

    if os.path.exists(VENUE_CONFIG_FILE):
        try:
            with open(VENUE_CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            for section, val in defaults.items():
                if section not in data:
                    data[section] = val
            _VENUE_CONFIG_CACHE = data
            return data
        except Exception as e:
            log.warning("[venue] config読み込み失敗、デフォルト使用: %s", e)

    _VENUE_CONFIG_CACHE = defaults
    return defaults


# ════════════════════════════════════════════════════════════
# 実データからの統計算出
# ════════════════════════════════════════════════════════════

def compute_venue_course_stats(
    history_file: str = MOTOR_HISTORY_FILE,
    force_reload: bool = False,
) -> dict:
    """
    motor_history.csv（date, venue, lane, place列を使用）から
    場×コース別の実測統計を算出する。

    戻り値: {
      "全国平均": {lane: {"win_rate": x, "rentai2_rate": x, "rentai3_rate": x,
                            "avg_finish": x, "samples": n}, ...},
      "桐生": {lane: {...}, ...},
      ...
    }
    win_rate等は%（0-100）。avg_finishは平均着順（1.0〜6.0、小さいほど良い）。
    samples はそのセルの実観測数（信頼度補正の入力に使う）。
    """
    global _VENUE_STATS_CACHE
    if _VENUE_STATS_CACHE is not None and not force_reload:
        return _VENUE_STATS_CACHE

    x_release_storage.download_file(history_file, history_file)

    if not os.path.exists(history_file):
        log.info("[venue] %s が存在しないため統計算出をスキップ", history_file)
        _VENUE_STATS_CACHE = {}
        return {}

    # venue -> lane -> place -> count
    counts: dict[str, dict[int, dict[int, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    with open(history_file, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            venue = row.get("venue", "").strip()
            try:
                lane = int(row.get("lane", 0))
                place = int(row.get("place", 0))
            except (ValueError, TypeError):
                continue
            if venue and 1 <= lane <= 6 and 1 <= place <= 6:
                counts[venue][lane][place] += 1

    def _stats_from_counts(place_counts: dict[int, int]) -> dict:
        total = sum(place_counts.values())
        if total == 0:
            return {"win_rate": 0.0, "rentai2_rate": 0.0, "rentai3_rate": 0.0, "avg_finish": 0.0, "samples": 0}
        win = place_counts.get(1, 0)
        rentai2 = win + place_counts.get(2, 0)
        rentai3 = rentai2 + place_counts.get(3, 0)
        avg_finish = sum(p * c for p, c in place_counts.items()) / total
        return {
            "win_rate":     round(win / total * 100, 1),
            "rentai2_rate": round(rentai2 / total * 100, 1),
            "rentai3_rate": round(rentai3 / total * 100, 1),
            "avg_finish":   round(avg_finish, 2),
            "samples":      total,
        }

    result: dict[str, dict] = {}

    # 全国平均（全場合算、コース別）
    national_counts: dict[int, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for venue, lanes in counts.items():
        for lane, place_counts in lanes.items():
            for place, c in place_counts.items():
                national_counts[lane][place] += c
    result["全国平均"] = {lane: _stats_from_counts(pc) for lane, pc in national_counts.items()}

    for venue, lanes in counts.items():
        result[venue] = {lane: _stats_from_counts(pc) for lane, pc in lanes.items()}

    _VENUE_STATS_CACHE = result
    log.info("[venue] %d場の統計を算出（元データ%s件）", len(result) - 1, history_file)
    return result


def get_corrected_venue_course_stat(
    venue: str,
    lane: int,
    metric: str,
    venue_stats: Optional[dict] = None,
    config: Optional[dict] = None,
) -> dict:
    """
    サンプル数補正（信頼度補正）を適用した場×コース統計値を1つ返す。
    実測値とサンプル数が少ない場合は全国平均へ縮約する。

    戻り値: {"value": float, "raw_value": float, "samples": int, "national_avg": float}
    """
    venue_stats = venue_stats if venue_stats is not None else compute_venue_course_stats()
    cfg = config or load_venue_config()
    prior_strength = cfg["sample_size_correction"]["prior_strength"]

    national = venue_stats.get("全国平均", {}).get(lane, {})
    national_avg = national.get(metric, 0.0)

    venue_entry = venue_stats.get(venue, {}).get(lane, {})
    raw_value = venue_entry.get(metric, 0.0)
    samples = venue_entry.get("samples", 0)

    if samples == 0:
        corrected = national_avg
    else:
        # ベイズ的シュリンケージ: サンプルが少ないほど全国平均に近づく
        corrected = (raw_value * samples + national_avg * prior_strength) / (samples + prior_strength)

    return {
        "value": round(corrected, 2),
        "raw_value": raw_value,
        "samples": samples,
        "national_avg": national_avg,
    }


# ════════════════════════════════════════════════════════════
# 水面タイプ分類
# ════════════════════════════════════════════════════════════

def classify_water_type(venue: str, venue_stats: Optional[dict] = None, config: Optional[dict] = None) -> dict:
    """
    その場の実測（信頼度補正後）1コース1着率から水面タイプを自動判定する。
    データが不十分な場合は cold_start_hints（著名な場のみの参考値）→
    それも無ければ「標準」を返す。

    戻り値: {"type": "super_in", "label": "超イン水面", "course1_win_rate": float,
              "source": "computed"|"cold_start_hint"|"default", "samples": int}
    """
    cfg = config or load_venue_config()
    venue_stats = venue_stats if venue_stats is not None else compute_venue_course_stats()
    th = cfg["water_type_thresholds"]
    labels = cfg["water_type_labels"]
    min_trust = cfg["sample_size_correction"]["min_samples_for_full_trust"]

    stat = get_corrected_venue_course_stat(venue, 1, "win_rate", venue_stats, cfg)

    if stat["samples"] >= min_trust:
        source = "computed"
        c1_win = stat["value"]
    elif venue in cfg.get("cold_start_hints", {}):
        # データ不足時は著名な場のみ参考値にフォールバック
        hint_type = cfg["cold_start_hints"][venue]
        return {
            "type": hint_type, "label": labels.get(hint_type, "標準"),
            "course1_win_rate": stat["value"], "source": "cold_start_hint",
            "samples": stat["samples"],
        }
    else:
        source = "computed_low_confidence" if stat["samples"] > 0 else "default"
        c1_win = stat["value"]

    if c1_win >= th["super_in"]:
        t = "super_in"
    elif c1_win >= th["in_favorable"]:
        t = "in_favorable"
    elif c1_win >= th["standard_min"]:
        t = "standard"
    else:
        t = "rough"

    return {
        "type": t, "label": labels.get(t, "標準"),
        "course1_win_rate": c1_win, "source": source, "samples": stat["samples"],
    }


def get_venue_course_factor(
    venue: str,
    lane: int,
    venue_stats: Optional[dict] = None,
    config: Optional[dict] = None,
) -> dict:
    """
    「選手能力 × 場特性」の場補正係数を算出する。
    その場・そのコースの信頼度補正済み1着率が全国平均に対してどれだけ
    高い/低いかを比率で返す（1.0 = 全国平均並み）。

    戻り値: {"factor": float, "venue_win_rate": float, "national_win_rate": float,
              "samples": int, "water_type": str}
    """
    venue_stats = venue_stats if venue_stats is not None else compute_venue_course_stats()
    cfg = config or load_venue_config()

    stat = get_corrected_venue_course_stat(venue, lane, "win_rate", venue_stats, cfg)
    national_avg = stat["national_avg"]
    factor = (stat["value"] / national_avg) if national_avg > 0 else 1.0

    water_type = classify_water_type(venue, venue_stats, cfg)

    return {
        "factor": round(factor, 3),
        "venue_win_rate": stat["value"],
        "national_win_rate": national_avg,
        "samples": stat["samples"],
        "water_type": water_type["type"],
    }
