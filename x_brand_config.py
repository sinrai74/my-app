#!/usr/bin/env python3
"""
x_brand_config.py  ── 全ブランド共通の定数・設定モジュール

要件㉓㉔（ブランド統一・保守性）に基づき、ブランド名・アイコン・配色・
ランク基準・重み付けなど、新聞/X投稿/実績ページ/note出力など
全コンポーネントで共有する定数をここに集約する。

ブランドを追加する場合はこのファイルの BRANDS 辞書に1エントリ
追加するだけで、新聞・INDEX・実績ページ等に自動反映されるよう設計する。
"""

from __future__ import annotations

# ════════════════════════════════════════════════════════════
# ブランド定義（要件㉓: シリーズ名統一）
# ════════════════════════════════════════════════════════════
# key はシステム内部で使うID。表示名・アイコン・カラーをここで一元管理する。

BRANDS: dict[str, dict] = {
    "danger": {
        "icon":  "🚨",
        "name":  "AI危険艇速報",
        "short": "危険艇",
        "color": "#ef5350",   # 赤系
    },
    "manshuu": {
        "icon":  "💰",
        "name":  "AI万舟警報",
        "short": "万舟",
        "color": "#ffa726",   # オレンジ系
    },
    "hot_high": {
        "icon":  "🔥",
        "name":  "AI高配当期待",
        "short": "高配当期待",
        "color": "#ff7043",
    },
    "motor_hot": {
        "icon":  "⚡",
        "name":  "AI激走モーター",
        "short": "激走モーター",
        "color": "#42a5f5",   # 青系
    },
    "motor_awk": {
        "icon":  "📈",
        "name":  "AI覚醒モーター",
        "short": "覚醒モーター",
        "color": "#00bcd4",   # シアン系
    },
    "korogashi": {
        "icon":  "🎯",
        "name":  "AI転がし研究",
        "short": "転がし候補",
        "color": "#26a69a",   # ティール系
    },
    "racer": {
        "icon":  "👤",
        "name":  "今日の注目レーサー",
        "short": "注目レーサー",
        "color": "#ab47bc",   # 紫系
    },
    "results": {
        "icon":  "📊",
        "name":  "AI実績",
        "short": "実績",
        "color": "#78909c",   # グレー系
    },
}


def brand_icon(key: str) -> str:
    return BRANDS.get(key, {}).get("icon", "")


def brand_name(key: str) -> str:
    return BRANDS.get(key, {}).get("name", key)


def brand_short(key: str) -> str:
    return BRANDS.get(key, {}).get("short", key)


def brand_color(key: str) -> str:
    return BRANDS.get(key, {}).get("color", "#999999")


# 互換用: 旧コードの BRAND_ICONS / BRAND_NAMES 形式でも参照できるようにする
BRAND_ICONS: dict[str, str] = {k: v["icon"]  for k, v in BRANDS.items()}
BRAND_NAMES: dict[str, str] = {k: v["name"]  for k, v in BRANDS.items()}
BRAND_SHORT: dict[str, str] = {k: v["short"] for k, v in BRANDS.items()}
BRAND_COLOR: dict[str, str] = {k: v["color"] for k, v in BRANDS.items()}


# ════════════════════════════════════════════════════════════
# ランク制（要件⑧: S/A/B/C 4段階に統一）
# ════════════════════════════════════════════════════════════
# 旧バージョンは S(80+)/A(60+)/B(40+) の3段階だったが、
# 要件⑧により S/A/B/C の4段階に統一する。

RANK_THRESHOLDS: list[tuple[str, float, str]] = [
    # (ランク名, 下限スコア, 表示色)
    ("S", 80, "#ef5350"),   # 赤
    ("A", 65, "#ffa726"),   # オレンジ
    ("B", 50, "#ffee58"),   # 黄
    ("C", 0,  "#42a5f5"),   # 青
]


def rank_of(score: float) -> str:
    """スコア(0-100)からランク名(S/A/B/C)を返す"""
    for name, lower, _ in RANK_THRESHOLDS:
        if score >= lower:
            return name
    return "C"


def rank_color(rank: str) -> str:
    """ランク名からカラーコードを返す"""
    for name, _, color in RANK_THRESHOLDS:
        if name == rank:
            return color
    return "#999999"


def rank_color_of_score(score: float) -> str:
    """スコアから直接カラーコードを返す（rank_of + rank_color のショートカット）"""
    return rank_color(rank_of(score))


def rank_label_with_emoji(score: float) -> str:
    """例: '🔴 S' のような表示用ラベルを返す"""
    rank = rank_of(score)
    emoji = {"S": "🔴", "A": "🟠", "B": "🟡", "C": "🔵"}.get(rank, "⚪")
    return f"{emoji} {rank}"


# ════════════════════════════════════════════════════════════
# AI一致指数の配点（要件③）
# ════════════════════════════════════════════════════════════

MATCH_INDEX_POINTS: dict[str, float] = {
    "danger":    20,
    "manshuu":   20,
    "korogashi": 20,
    "motor_hot": 15,
    "motor_awk": 15,
    "hot_high":  10,
}
MATCH_INDEX_RANK_BONUS_S = 1.15
MATCH_INDEX_RANK_BONUS_A = 1.05


# ════════════════════════════════════════════════════════════
# AI開催場コンディション指数の重み（要件②）
# ════════════════════════════════════════════════════════════

VENUE_CONDITION_WEIGHTS: dict[str, float] = {
    "danger":    0.30,
    "manshuu":   0.30,
    "korogashi": 0.20,
    "motor_hot": 0.10,
    "motor_awk": 0.10,
}


# ════════════════════════════════════════════════════════════
# 高配当期待の閾値設定
# ════════════════════════════════════════════════════════════

HOT_HIGH_THRESHOLD_FALLBACK = 80
HOT_HIGH_RATIO = 0.3   # 万舟掲載分の上位30%を高配当期待とする


# ════════════════════════════════════════════════════════════
# AI信頼度認定（要件⑮）
# ════════════════════════════════════════════════════════════
# ブランドごとの30日成功率からA+/A/B/Cの信頼度ランクを判定する

TRUST_THRESHOLDS: list[tuple[str, float]] = [
    ("A+", 65),
    ("A",  55),
    ("B",  45),
    ("C",  0),
]


def trust_rank_of(success_rate_30d: float) -> str:
    """30日成功率(%)からA+/A/B/Cの信頼度ランクを返す"""
    for name, lower in TRUST_THRESHOLDS:
        if success_rate_30d >= lower:
            return name
    return "C"


# ════════════════════════════════════════════════════════════
# システム共通メタ情報（要件㉑: 全ページ共通フッター）
# ════════════════════════════════════════════════════════════

AI_VERSION = "v3.1"
SYSTEM_NAME = "AI競艇データメディア"


# ════════════════════════════════════════════════════════════
# ブランド別「成功条件」の払戻閾値（円）
# ════════════════════════════════════════════════════════════
# 各ブランドの的中判定に使う払戻金額のしきい値。ここを唯一の基準とし、
# x_verification.py 等はこの定数を参照する（ハードコード禁止）。
#
#   危険艇: 1着に1号艇が来なかった場合に成功（金額基準なし。別ロジック）
#   万舟:   払戻が 10,000円 を超えたら成功
#   高配当: 払戻が  5,000円 を超えたら成功
#   中穴:   払戻が  2,700円 を超えたら成功

SUCCESS_PAYOUT_THRESHOLDS: dict[str, int] = {
    "manshuu":  10_000,
    "hot_high":  5_000,
    "korogashi": 2_700,   # 中穴
}


# ════════════════════════════════════════════════════════════
# ★表示ヘルパー
# ════════════════════════════════════════════════════════════

def stars(score: float, max_score: float = 100) -> str:
    """0-max_score点を★5段階表示に変換"""
    n = max(0, min(5, round(score / max_score * 5)))
    return "★" * n + "☆" * (5 - n)


def heat_emoji(score: float) -> str:
    """スコアを🟢🟡🔴の信号表示に変換（要件②）"""
    if score >= 70: return "🟢"
    if score >= 40: return "🟡"
    return "🔴"
