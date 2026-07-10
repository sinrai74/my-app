#!/usr/bin/env python3
"""
x_training_export.py  ── 学習用特徴量CSV出力（Ver4評価エンジン）

hit_record.csv は「実績記録」（購入判定・回収率計算・実運用の追跡）を
目的としたスキーマであり、機械学習フレームワーク（LightGBM/CatBoost/
XGBoost等）にそのまま読み込むには不要な列（買い目文字列・払戻金など）
や、JSON文字列に折り畳まれた列（rank_index_json 等）が混在している。

本モジュールは hit_record.csv から「学習に使う特徴量」だけを抽出・
フラット化し、独立したCSV（training_dataset.csv）として出力する。

【設計方針】
・特徴量を後から追加しても学習コードを書き換えずに済むよう、
  列は「動的に決まる」構造にする（JSON列を展開して得られる列名は
  実行時のデータに応じて変わるため、出力時に一度全行をスキャンして
  列の和集合を確定してから書き出す＝欠けている特徴量はNaN/空欄）。
・目的変数（教師ラベル）候補として hit（的中）・profit（収支）も
  そのまま残す。学習コード側でどちらを使うかは自由に選べる。
・寄与度（contributions）は「1着適性」の内訳のみをフラット化する
  （2着・3着適性の寄与度も rank_index_json 内に残っているため、
  必要であれば同様の関数で追加抽出できる）。
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Optional

log = logging.getLogger("x_training_export")

import x_release_storage

HIT_RECORD_CSV = "hit_record.csv"
TRAINING_DATASET_CSV = "training_dataset.csv"

# hit_record.csv の列のうち、そのまま特徴量・教師ラベル候補として使う列
_DIRECT_FEATURE_COLUMNS = [
    "date", "venue", "venue_num", "race", "night", "race_type",
    "confidence", "pred_prob", "pred_ev", "pred_odds", "upset_score",
    "hit", "profit",  # 教師ラベル候補
    "buyscore", "match_index",
    "feat_win_rate", "feat_motor", "feat_avg_st", "feat_racer_class",
    "feat_course_st_1c", "feat_course_rank_1c",
    "danger_score_v3",
    "venue_water_type", "venue_factor", "ability_trend",
    "course_f_rate_1c", "course_l_rate_1c", "course_rentai2_1c",
    "course_sample_confidence",
]

# JSON文字列列 → フラット化して展開する対象
_JSON_COLUMNS = ["rank_index_json", "featured_boats_json", "feat_danger_breakdown"]


def _flatten_rank_index(raw_json: str) -> dict:
    """
    rank_index_json（{lane: {"top1":,"top2":,"top3":,"contributions":{...}}}）を
    "rank_1_top1", "rank_1_contrib_win_rate" のような列名にフラット化する。
    1号艇（危険艇速報の対象）を中心に、全艇分を展開する。
    """
    flat: dict = {}
    if not raw_json:
        return flat
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return flat

    for lane_key, idx in data.items():
        try:
            lane = int(lane_key)
        except (TypeError, ValueError):
            continue
        flat[f"rank_{lane}_top1"] = idx.get("top1")
        flat[f"rank_{lane}_top2"] = idx.get("top2")
        flat[f"rank_{lane}_top3"] = idx.get("top3")
        contrib = idx.get("contributions", {}).get("first", {})
        for feat_name, val in contrib.items():
            flat[f"rank_{lane}_contrib_{feat_name}"] = val
    return flat


def _flatten_danger_breakdown(raw_json: str) -> dict:
    """
    feat_danger_breakdown（{"win_rate": {"weighted":, "worse_count":, ...}, ...}）を
    "danger_win_rate_weighted", "danger_win_rate_worse_count" のようにフラット化する。
    """
    flat: dict = {}
    if not raw_json:
        return flat
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return flat

    for key, v in data.items():
        if not isinstance(v, dict):
            continue
        flat[f"danger_{key}_weighted"] = v.get("weighted")
        if v.get("kind") == "relative":
            flat[f"danger_{key}_worse_count"] = v.get("worse_count")
            flat[f"danger_{key}_worse_total"] = v.get("worse_total")


    return flat


def _flatten_featured_boats(raw_json: str) -> dict:
    """
    featured_boats_json（[{"lane":,"mark":,"composite":,...}, ...]）を
    "featured_1_lane", "featured_1_composite" のようにフラット化する
    （◎○▲の順で最大3件）。
    """
    flat: dict = {}
    if not raw_json:
        return flat
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError):
        return flat
    if not isinstance(data, list):
        return flat

    for i, item in enumerate(data, start=1):
        flat[f"featured_{i}_lane"] = item.get("lane")
        flat[f"featured_{i}_composite"] = item.get("composite")
        flat[f"featured_{i}_top1"] = item.get("top1")
    return flat


def build_training_row(hit_record_row: dict) -> dict:
    """
    hit_record.csv の1行から学習用特徴量の1行（フラット辞書）を作る。
    """
    row: dict = {}
    for col in _DIRECT_FEATURE_COLUMNS:
        row[col] = hit_record_row.get(col, "")

    row.update(_flatten_rank_index(hit_record_row.get("rank_index_json", "")))
    row.update(_flatten_danger_breakdown(hit_record_row.get("feat_danger_breakdown", "")))
    row.update(_flatten_featured_boats(hit_record_row.get("featured_boats_json", "")))

    return row


def export_training_dataset(
    hit_record_file: str = HIT_RECORD_CSV,
    output_file: str = TRAINING_DATASET_CSV,
) -> dict:
    """
    hit_record.csv 全体を読み込み、学習用にフラット化したCSVへ出力する。

    戻り値: {"exported": bool, "rows": int, "columns": int, "reason": str}
    """
    x_release_storage.download_file(hit_record_file, hit_record_file)
    if not os.path.exists(hit_record_file):
        return {"exported": False, "rows": 0, "columns": 0,
                 "reason": f"{hit_record_file} が存在しません"}

    with open(hit_record_file, "r", encoding="utf-8", newline="") as f:
        hit_rows = list(csv.DictReader(f))

    if not hit_rows:
        return {"exported": False, "rows": 0, "columns": 0, "reason": "hit_record.csv が空です"}

    training_rows = [build_training_row(r) for r in hit_rows]

    # 列の和集合を確定する（行ごとにJSON展開で列数が変わるため）。
    # _DIRECT_FEATURE_COLUMNS の並び順を保ちつつ、フラット化で増えた列は
    # 後ろにソートして追加する（実行のたびに列順が変わらないようにする）。
    all_keys: set = set()
    for row in training_rows:
        all_keys.update(row.keys())
    extra_keys = sorted(k for k in all_keys if k not in _DIRECT_FEATURE_COLUMNS)
    fieldnames = _DIRECT_FEATURE_COLUMNS + extra_keys

    tmp_path = output_file + ".tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in training_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    os.replace(tmp_path, output_file)

    log.info("[training_export] %s へ %d行 x %d列を出力しました",
              output_file, len(training_rows), len(fieldnames))

    return {
        "exported": True, "rows": len(training_rows), "columns": len(fieldnames),
        "reason": "成功",
    }


if __name__ == "__main__":
    result = export_training_dataset()
    print("=" * 60)
    print(" 学習用特徴量CSV出力")
    print("=" * 60)
    print(f" 出力           : {'成功' if result['exported'] else '失敗'}")
    print(f" 行数           : {result['rows']}")
    print(f" 列数           : {result['columns']}")
    print(f" 詳細           : {result['reason']}")
    print("=" * 60)
