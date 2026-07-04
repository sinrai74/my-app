#!/usr/bin/env python3
"""
csv_schema.py  ── hit_record.csv スキーマ定義（単一の真実の源）

このファイルは hit_record.csv の「あるべき列構成」をバージョンごとに定義する。
notify_arashi.py の書き込み、x_verification.py の読み込み、migration.py の
変換は、すべてここの定義を正とする。ハードコードされた列リストを各所に
散らばらせない（それがスキーマずれの原因になる）。

────────────────────────────────────────────────────────────
スキーマを変更するときの手順（将来の自分へ）
────────────────────────────────────────────────────────────
1. CURRENT_SCHEMA_VERSION をインクリメントする（例: 2 → 3）。
2. SCHEMA_COLUMNS に新バージョンの完全な列リストを追加する。
3. NEW_COLUMN_DEFAULTS に「新しく増えた列」の初期値だけ追加する。
4. migration.py の MIGRATIONS に (旧→新) の変換関数を1つ追加する。
   → CSVを手作業で編集する運用は禁止。必ずマイグレーションで移行する。
"""

from __future__ import annotations

# ════════════════════════════════════════════════════════════
# 現在の期待スキーマバージョン
# ════════════════════════════════════════════════════════════
# プログラムが「hit_record.csv はこのバージョンであるべき」と期待する値。
# 実ファイルのバージョン（.schema_version ファイルで管理）がこれより古ければ
# マイグレーションが必要と判定される。
CURRENT_SCHEMA_VERSION = 2

# ════════════════════════════════════════════════════════════
# バージョン別 完全列定義
# ════════════════════════════════════════════════════════════
# 各バージョンで hit_record.csv が持つべき列を、順序どおりに列挙する。
# 順序は書き込み時のカラム順（DictWriter fieldnames）と一致させること。

# Version 1: 「購入判定分離」導入前の旧スキーマ（22列）
_COLUMNS_V1 = [
    "date", "venue", "venue_num", "race", "night",
    "race_type", "why_bet", "confidence",
    "pred_combo", "pred_prob", "pred_ev", "pred_odds", "upset_score",
    "wind_speed", "wind_dir", "wave",
    "result_combo", "payout", "hit", "profit", "n_bets", "cost",
]

# Version 2: 購入判定分離(BuyScore) + Phase2学習用特徴量 を追加（34列）
# 追加列: purchased, buyscore, match_index, skip_reason,
#         model_version, feat_win_rate, feat_motor, feat_avg_st,
#         feat_racer_class, feat_course_st_1c, feat_course_rank_1c,
#         feat_danger_breakdown
_COLUMNS_V2 = _COLUMNS_V1 + [
    "purchased", "buyscore", "match_index", "skip_reason",
    "model_version",
    "feat_win_rate", "feat_motor", "feat_avg_st", "feat_racer_class",
    "feat_course_st_1c", "feat_course_rank_1c", "feat_danger_breakdown",
]

SCHEMA_COLUMNS: dict[int, list[str]] = {
    1: _COLUMNS_V1,
    2: _COLUMNS_V2,
}

# ════════════════════════════════════════════════════════════
# 新規追加列の初期値（旧データを変換するときの補完値）
# ════════════════════════════════════════════════════════════
# 「あるバージョンで初めて登場した列」の初期値をここで定義する。
# マイグレーション時、旧データにこの列がなければこの値で補完する。
#
# 【重要な設計判断】
#   purchased = "1"
#     旧データ(Version1)は「購入判定分離」導入前のため、記録されている
#     予想はすべて実際に賭けたものとみなす。よって購入扱い(1)で補完する。
#     これは x_results_common.calc_overall_roi() の
#     「旧データ(purchased列なし)は1扱い」というフォールバックと一致させる。
#   その他の列 = ""（空欄）
#     BuyScore・特徴量は旧データでは取得していないため、値なし(空欄)とする。
#     _safe_float() 等は空欄を 0/デフォルト として安全に扱えるため問題ない。
NEW_COLUMN_DEFAULTS: dict[str, str] = {
    "purchased":             "1",
    "buyscore":              "",
    "match_index":           "",
    "skip_reason":           "",
    "model_version":         "",
    "feat_win_rate":         "",
    "feat_motor":            "",
    "feat_avg_st":           "",
    "feat_racer_class":      "",
    "feat_course_st_1c":     "",
    "feat_course_rank_1c":   "",
    "feat_danger_breakdown": "",
}


def get_columns(version: int) -> list[str]:
    """指定バージョンの完全列リストを返す。"""
    if version not in SCHEMA_COLUMNS:
        raise ValueError(f"未知のスキーマバージョン: {version}")
    return list(SCHEMA_COLUMNS[version])


def current_columns() -> list[str]:
    """現在の期待スキーマの完全列リストを返す。"""
    return get_columns(CURRENT_SCHEMA_VERSION)


def detect_version_from_header(header: list[str]) -> int | None:
    """
    CSVヘッダー(列名リスト)からスキーマバージョンを推定する。
    完全一致するバージョンがあればその番号を、なければ None を返す。
    """
    for version, cols in SCHEMA_COLUMNS.items():
        if header == cols:
            return version
    return None
