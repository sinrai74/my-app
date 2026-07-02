#!/usr/bin/env python3
"""
x_post_text.py  ── X投稿候補テキスト生成

各日次メール（危険艇・万舟・競艇新聞・実績ページ）の末尾に追加する
X投稿候補テキストを生成する。

・140文字以内
・通常・バズ・リプ の3種類
・テンプレートをランダムに切り替えて毎日変化
・ハッシュタグは最大4個
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))

TAGS_BASE = "#競艇 #ボートレース #競艇予想"
TAGS_NOTE = "#競艇 #ボートレース #競艇予想 #note"


def _today_seed() -> int:
    """日付ベースのシードで毎日違うテンプレートを選ぶ"""
    return int(datetime.now(JST).strftime("%Y%m%d"))


def _pick(lst: list, offset: int = 0) -> str:
    rng = random.Random(_today_seed() + offset)
    return rng.choice(lst)


def _wrap(text: str) -> str:
    """140文字を超えていれば警告付きで返す（念のため）"""
    if len(text) > 140:
        text = text[:137] + "…"
    return text


# ════════════════════════════════════════════════════════════
# 危険艇速報
# ════════════════════════════════════════════════════════════

def danger_post(
    s_count: int = 0,
    top_venue: str = "",
    top_race: str = "",
) -> str:
    """危険艇速報のX投稿候補3種を返す"""

    normals = [
        f"🚨今日の危険艇速報を公開しました。\nAIが全レースを解析しインが危険なレースを抽出。\n今日はSランク{s_count}件。気になるレースはありますか？\n{TAGS_BASE}",
        f"🚨AI危険艇速報 本日分を配信しました。\n1号艇が飛びそうなレースをスコア順に掲載。\nSランク{s_count}件を保存しておくと便利です。\n{TAGS_BASE}",
        f"🚨今日もAIが危険な1号艇を検出しました。\nSランク{s_count}件・全掲載{top_venue}等。\nどのレースが気になりますか？\n{TAGS_BASE}",
    ]

    buzzes = [
        f"今日はインが危ない日です⚠️\nAI一致Sランクは{s_count}レース。\n締切前に保存して見返してください🔖\n{TAGS_BASE}",
        f"1号艇が勝てないレースをAIが特定しました。\n今日のSランク{s_count}件。\nあなたはどのレースを狙いますか？\n{TAGS_BASE}",
        f"AIスコアが高いほど1号艇が来ない確率が上がります。\n今日のトップは{top_venue}{top_race}R。\n保存推奨です🔖\n{TAGS_BASE}",
    ]

    reps = [
        f"詳しいスコアや理由はAI競艇新聞に掲載しています👇\n{TAGS_BASE}",
        f"各レースの詳細分析はAI競艇新聞で公開中です👇\n{TAGS_BASE}",
        f"危険と判断した理由を新聞に全て書いています👇\n{TAGS_BASE}",
    ]

    return _format_posts(
        _wrap(_pick(normals, 0)),
        _wrap(_pick(buzzes, 1)),
        _wrap(_pick(reps, 2)),
    )


# ════════════════════════════════════════════════════════════
# 万舟警報
# ════════════════════════════════════════════════════════════

def manshuu_post(
    count: int = 0,
    top_venue: str = "",
    top_race: str = "",
    top_match: int = 0,
) -> str:
    """万舟警報のX投稿候補3種を返す"""

    normals = [
        f"💰AI万舟警報 本日分を公開しました。\n高配当が期待できるレースを厳選して掲載。\n今日の最注目は{top_venue}{top_race}R。保存推奨です。\n{TAGS_BASE}",
        f"💰今日の万舟候補{count}件を公開しました。\nAI一致指数最高は{top_match}。\n気になるレースはありますか？\n{TAGS_BASE}",
        f"💰万舟を狙うならAIを活用してください。\n本日{count}件を掲載。トップは{top_venue}{top_race}R。\n保存しておくと便利です🔖\n{TAGS_BASE}",
    ]

    buzzes = [
        f"今日の万舟候補は厳選{count}レース💰\n最注目は{top_venue}{top_race}R。\nあなたなら買いますか？\n{TAGS_BASE}",
        f"高配当を狙うなら今日の{top_venue}{top_race}Rに注目💰\nAI{count}件が一致しています。\n保存して締切前に確認を🔖\n{TAGS_BASE}",
        f"万舟は運ではなくデータで狙います💰\n今日AIが選んだ{count}件を公開。\n外れても全部公開します。信頼できますか？\n{TAGS_BASE}",
    ]

    reps = [
        f"各レースの詳しい理由はAI競艇新聞に掲載しています👇\n{TAGS_BASE}",
        f"万舟になる理由を新聞で全公開中です👇\n{TAGS_BASE}",
        f"なぜそのレースを選んだのか、新聞で解説しています👇\n{TAGS_BASE}",
    ]

    return _format_posts(
        _wrap(_pick(normals, 3)),
        _wrap(_pick(buzzes, 4)),
        _wrap(_pick(reps, 5)),
    )


# ════════════════════════════════════════════════════════════
# AI競艇新聞
# ════════════════════════════════════════════════════════════

def newspaper_post(
    danger_count: int = 0,
    manshuu_count: int = 0,
    top_venue: str = "",
    top_race: str = "",
    top_match: int = 0,
) -> str:
    """AI競艇新聞のX投稿候補3種を返す"""

    normals = [
        f"📰本日のAI競艇新聞を公開しました。\n危険艇{danger_count}件・万舟{manshuu_count}件・モーター情報を掲載。\nnoteで全文読めます。\n{TAGS_NOTE}",
        f"📰AI競艇新聞 {datetime.now(JST).strftime('%m/%d')}号を配信しました。\n全開催場・全レースをAIが解析。\n今日の全体像はこちらから。\n{TAGS_NOTE}",
        f"📰今日のAI競艇新聞を公開しました。\n危険艇・万舟・転がし・モーター情報を1本にまとめています。\nnoteで無料公開中です。\n{TAGS_NOTE}",
    ]

    buzzes = [
        f"今日AIが最も注目しているのは{top_venue}{top_race}R📰\nAI一致指数{top_match}で4ブランドが一致。\n詳細は新聞へ。\n{TAGS_NOTE}",
        f"AI{danger_count}件の危険艇・{manshuu_count}件の万舟を特定しました📰\n今日どこに資金を集中させるか、新聞で確認してください。\n{TAGS_NOTE}",
        f"毎朝AIが全レースを解析しています📰\n今日のトップレースは{top_venue}{top_race}R。\n読むだけで今日の全体像がわかります。\n{TAGS_NOTE}",
    ]

    reps = [
        f"読むだけで今日の全体像がわかります👇\nnoteで無料公開中です。\n{TAGS_NOTE}",
        f"危険艇から転がし候補まで全掲載しています👇\nnoteへどうぞ。\n{TAGS_NOTE}",
        f"全レース解析レポートはnoteで公開しています👇\n{TAGS_NOTE}",
    ]

    return _format_posts(
        _wrap(_pick(normals, 6)),
        _wrap(_pick(buzzes, 7)),
        _wrap(_pick(reps, 8)),
    )


# ════════════════════════════════════════════════════════════
# AI実績ページ
# ════════════════════════════════════════════════════════════

def results_post(
    date_str: str = "",
    hit_rate: float = 0.0,
    danger_hit: int = 0,
    danger_total: int = 0,
    profit: int = 0,
) -> str:
    """AI実績ページのX投稿候補3種を返す"""

    date_disp = f"{date_str[4:6]}/{date_str[6:8]}" if len(date_str) == 8 else "昨日"
    profit_str = f"+{profit:,}円" if profit >= 0 else f"{profit:,}円"
    danger_str = f"{danger_hit}/{danger_total}件" if danger_total > 0 else f"{hit_rate:.1f}%"

    normals = [
        f"📊{date_disp}のAI実績を公開しました。\n危険艇的中{danger_str}・損益{profit_str}。\n良かった予想も外れた予想も全て公開しています。\n{TAGS_BASE}",
        f"📊昨日のAI成績を公開しました。\n的中率{hit_rate:.1f}%・損益{profit_str}。\n改善点も全て載せています。\n{TAGS_BASE}",
        f"📊AI実績 {date_disp}分を公開しました。\n危険艇{danger_str}的中。\n毎日検証して精度を高めています。\n{TAGS_BASE}",
    ]

    buzzes = [
        f"外れも隠しません📊\n{date_disp}のAI成績はこちら。\n的中率{hit_rate:.1f}%・損益{profit_str}。今日はさらに改善済みです。\n{TAGS_BASE}",
        f"AIの成績を毎日公開しています📊\n{date_disp}：危険艇{danger_str}的中。\n信頼できるAIかどうか、数字で判断してください。\n{TAGS_BASE}",
        f"予想を売りません。全て無料で公開します📊\n{date_disp}の実績：的中率{hit_rate:.1f}%。\nあなたのAI選びの参考に。\n{TAGS_BASE}",
    ]

    reps = [
        f"毎日検証して精度を高めています👇\n実績ページで全データを公開中です。\n{TAGS_BASE}",
        f"外れた理由も全て公開しています👇\n改善を続けるAIです。\n{TAGS_BASE}",
        f"データに基づいて毎日改善しています👇\n実績は全公開です。\n{TAGS_BASE}",
    ]

    return _format_posts(
        _wrap(_pick(normals, 9)),
        _wrap(_pick(buzzes, 10)),
        _wrap(_pick(reps, 11)),
    )


# ════════════════════════════════════════════════════════════
# 共通フォーマット
# ════════════════════════════════════════════════════════════

def _format_posts(normal: str, buzz: str, rep: str) -> str:
    """3種の投稿をメール末尾用のブロックにフォーマットする"""
    sep = "━━━━━━━━━━━━━━"
    return "\n".join([
        "",
        sep,
        "📱 X投稿候補",
        sep,
        "【通常】",
        normal,
        sep,
        "【バズ】",
        buzz,
        sep,
        "【リプ・固定ポスト返信用】",
        rep,
        sep,
    ])


# ════════════════════════════════════════════════════════════
# 文字数チェック（デバッグ用）
# ════════════════════════════════════════════════════════════

def check_lengths(text: str) -> None:
    """生成されたブロック内の各投稿文字数を表示する"""
    import re
    sections = re.split(r'━+', text)
    for s in sections:
        s = s.strip()
        if s and not s.startswith("📱") and not s.startswith("【"):
            label = s.split("\n")[0][:10]
            body = s
            print(f"  {len(body):3d}文字: {label!r}")
