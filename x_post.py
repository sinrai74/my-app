#!/usr/bin/env python3
"""
x_post.py  ── X（旧Twitter）API 投稿スクリプト

依存:
    pip install tweepy

環境変数（GitHub Secrets に登録）:
    X_API_KEY          … API Key（Consumer Key）
    X_API_SECRET       … API Key Secret（Consumer Secret）
    X_ACCESS_TOKEN     … Access Token
    X_ACCESS_SECRET    … Access Token Secret

Usage:
    # ランキングJSONから自動投稿（画像付き）
    python x_post.py --input ranking_cache.json --with-image

    # テキストファイルから1ツイートずつ投稿（セパレータ区切り）
    python x_post.py --file today_tweets.txt

    # ランキング種類を指定して投稿
    python x_post.py --type danger    --input ranking_cache.json --with-image
    python x_post.py --type hot       --input ranking_cache.json --with-image
    python x_post.py --type manshuu   --input ranking_cache.json --with-image
    python x_post.py --type awakening --input ranking_cache.json --with-image

    # ドライラン（実際には投稿しない）
    python x_post.py --input ranking_cache.json --with-image --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

log = logging.getLogger("x_post")

# ════════════════════════════════════════════════════════════
# 環境変数から認証情報を取得
# ════════════════════════════════════════════════════════════

def _get_credentials() -> dict[str, str]:
    keys = ["X_API_KEY", "X_API_SECRET", "X_ACCESS_TOKEN", "X_ACCESS_SECRET"]
    creds: dict[str, str] = {}
    missing: list[str] = []
    for k in keys:
        val = os.environ.get(k, "")
        if not val:
            missing.append(k)
        creds[k] = val
    if missing:
        log.error("環境変数が設定されていません: %s", ", ".join(missing))
        sys.exit(1)
    return creds


# ════════════════════════════════════════════════════════════
# tweepy クライアント生成
# ════════════════════════════════════════════════════════════

def _build_client():
    """tweepy.Client（v2 API）を返す"""
    try:
        import tweepy
    except ImportError:
        log.error("tweepy がインストールされていません: pip install tweepy")
        sys.exit(1)

    creds = _get_credentials()
    client = tweepy.Client(
        consumer_key        = creds["X_API_KEY"],
        consumer_secret     = creds["X_API_SECRET"],
        access_token        = creds["X_ACCESS_TOKEN"],
        access_token_secret = creds["X_ACCESS_SECRET"],
    )
    return client


def _build_api_v1():
    """tweepy.API（v1.1 API, 画像アップロード用）を返す"""
    try:
        import tweepy
    except ImportError:
        log.error("tweepy がインストールされていません: pip install tweepy")
        sys.exit(1)

    creds = _get_credentials()
    auth = tweepy.OAuth1UserHandler(
        creds["X_API_KEY"],
        creds["X_API_SECRET"],
        creds["X_ACCESS_TOKEN"],
        creds["X_ACCESS_SECRET"],
    )
    return tweepy.API(auth)


# ════════════════════════════════════════════════════════════
# 画像アップロード（v1.1 media/upload）
# ════════════════════════════════════════════════════════════

def _upload_media(image_path: str) -> Optional[int]:
    """画像をアップロードして media_id を返す。失敗時は None"""
    if not os.path.exists(image_path):
        log.warning("[画像] ファイルが見つかりません: %s", image_path)
        return None
    try:
        api_v1 = _build_api_v1()
        media = api_v1.media_upload(filename=image_path)
        log.info("[画像] アップロード完了: %s → media_id=%s", image_path, media.media_id)
        return media.media_id
    except Exception as e:
        log.warning("[画像] アップロード失敗: %s", e)
        return None


# ════════════════════════════════════════════════════════════
# 投稿
# ════════════════════════════════════════════════════════════

# X の文字数制限（日本語は 1文字=1文字としてカウント、ただし URL は23文字固定）
MAX_TWEET_LEN = 280

TWEET_INTERVAL_SEC = 30   # 連続投稿の間隔（レート制限対策）

# ランキング種類 → (テキスト生成関数, 画像ファイル名)
_TWEET_MAP: dict[str, tuple[str, str]] = {
    "danger":    ("format_danger_tweet",    "danger.png"),
    "hot":       ("format_hot_motor_tweet", "hot_motor.png"),
    "manshuu":   ("format_manshuu_tweet",   "manshuu.png"),
    "awakening": ("format_awakening_tweet", "awakening.png"),
}


def _truncate(text: str, max_len: int = MAX_TWEET_LEN) -> str:
    """文字数超過時に末尾を省略する"""
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def post_tweet(
    text: str,
    image_path: Optional[str] = None,
    dry_run: bool = False,
) -> bool:
    """
    1ツイートを投稿する。
    dry_run=True の場合はテキストを出力するだけで投稿しない。
    成功時 True、失敗時 False を返す。
    """
    text = _truncate(text)

    if dry_run:
        sep = "─" * 60
        print(sep)
        print("[DRY RUN] 以下のツイートを投稿予定:")
        if image_path:
            print(f"[DRY RUN] 画像: {image_path}")
        print(text)
        print(sep)
        return True

    try:
        client = _build_client()
        media_ids: list[int] = []

        if image_path:
            media_id = _upload_media(image_path)
            if media_id:
                media_ids.append(media_id)

        kwargs: dict = {"text": text}
        if media_ids:
            kwargs["media_ids"] = media_ids

        response = client.create_tweet(**kwargs)
        tweet_id = response.data.get("id") if response.data else "?"
        log.info("[投稿] 成功: tweet_id=%s", tweet_id)
        return True

    except Exception as e:
        log.error("[投稿] 失敗: %s", e)
        return False


def post_from_ranking(
    data: dict,
    types: Optional[list[str]] = None,
    with_image: bool = True,
    dry_run: bool = False,
    interval: float = TWEET_INTERVAL_SEC,
) -> dict[str, bool]:
    """
    ranking_cache.json の data から指定種類のツイートを投稿する。
    types=None の場合は全4種を投稿する。
    戻り値: {type_key: 成功/失敗}
    """
    # x_ranking.py のフォーマット関数をインポート
    try:
        from x_ranking import (
            format_danger_tweet,
            format_hot_motor_tweet,
            format_manshuu_tweet,
            format_awakening_tweet,
        )
    except ImportError as e:
        log.error("x_ranking.py をインポートできません: %s", e)
        return {}

    formatter_map = {
        "danger":    format_danger_tweet,
        "hot":       format_hot_motor_tweet,
        "manshuu":   format_manshuu_tweet,
        "awakening": format_awakening_tweet,
    }

    target_types = types or list(_TWEET_MAP.keys())
    results: dict[str, bool] = {}

    for i, t in enumerate(target_types):
        if t not in formatter_map:
            log.warning("不明なタイプ: %s", t)
            results[t] = False
            continue

        text = formatter_map[t](data)
        image_path = _TWEET_MAP[t][1] if with_image else None

        # 画像ファイルが存在しない場合はテキストのみ投稿
        if image_path and not os.path.exists(image_path):
            log.info("[投稿] 画像なし（%s が見つからない）→ テキストのみ", image_path)
            image_path = None

        ok = post_tweet(text, image_path=image_path, dry_run=dry_run)
        results[t] = ok

        # 最後以外はインターバルを挟む
        if i < len(target_types) - 1:
            if not dry_run:
                log.info("[投稿] %d秒待機...", int(interval))
                time.sleep(interval)

    return results


def post_from_file(
    file_path: str,
    separator: str = "=" * 50,
    dry_run: bool = False,
    interval: float = TWEET_INTERVAL_SEC,
) -> list[bool]:
    """
    テキストファイル（セパレータ区切り）から複数ツイートを投稿する。
    戻り値: 各ツイートの成功/失敗リスト
    """
    if not os.path.exists(file_path):
        log.error("ファイルが見つかりません: %s", file_path)
        return []

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    tweets = [t.strip() for t in content.split(separator) if t.strip()]
    log.info("[ファイル] %d件のツイートを検出: %s", len(tweets), file_path)

    results: list[bool] = []
    for i, text in enumerate(tweets):
        ok = post_tweet(text, dry_run=dry_run)
        results.append(ok)
        if i < len(tweets) - 1 and not dry_run:
            log.info("[投稿] %d秒待機...", int(interval))
            time.sleep(interval)

    return results


# ════════════════════════════════════════════════════════════
# エントリポイント
# ════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="競艇 X 投稿スクリプト")
    parser.add_argument("--input",      default="ranking_cache.json",
                        help="ランキング JSON ファイル（デフォルト: ranking_cache.json）")
    parser.add_argument("--file",       help="テキストファイルから投稿（--input の代わりに使用）")
    parser.add_argument("--type",       choices=list(_TWEET_MAP.keys()), action="append",
                        dest="types",  help="投稿する種類（複数指定可）")
    parser.add_argument("--with-image", action="store_true",
                        help="画像を添付して投稿（x_image.py で事前生成が必要）")
    parser.add_argument("--dry-run",    action="store_true",
                        help="実際には投稿せず内容を表示する")
    parser.add_argument("--interval",   type=float, default=TWEET_INTERVAL_SEC,
                        help=f"連続投稿の間隔（秒）（デフォルト: {TWEET_INTERVAL_SEC}）")
    args = parser.parse_args()

    if args.file:
        # テキストファイルから投稿
        results = post_from_file(args.file, dry_run=args.dry_run, interval=args.interval)
        ok = sum(results)
        log.info("[完了] %d/%d件 投稿成功", ok, len(results))

    else:
        # ranking_cache.json から投稿
        if not os.path.exists(args.input):
            log.error("ランキングファイルが見つかりません: %s", args.input)
            log.error("先に `python x_ranking.py --generate` を実行してください")
            sys.exit(1)

        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

        results = post_from_ranking(
            data,
            types      = args.types,
            with_image = args.with_image,
            dry_run    = args.dry_run,
            interval   = args.interval,
        )
        ok = sum(results.values())
        log.info("[完了] %d/%d件 投稿成功: %s", ok, len(results), results)

        if not all(results.values()):
            sys.exit(1)


if __name__ == "__main__":
    main()
