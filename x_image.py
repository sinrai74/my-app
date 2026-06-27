#!/usr/bin/env python3
"""
x_image.py  ── 競艇 X 投稿用ランキング画像（PNG）生成

依存:
    pip install Pillow

Usage:
    python x_image.py --type danger   --input ranking_cache.json --output danger.png
    python x_image.py --type hot      --input ranking_cache.json --output hot.png
    python x_image.py --type manshuu  --input ranking_cache.json --output manshuu.png
    python x_image.py --type awakening --input ranking_cache.json --output awakening.png
    python x_image.py --all           --input ranking_cache.json   # 4枚まとめて生成

画像サイズ: 1200×675px（X/Twitter 推奨）
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

log = logging.getLogger("x_image")

# ════════════════════════════════════════════════════════════
# デザイン定数
# ════════════════════════════════════════════════════════════

IMG_W, IMG_H = 1200, 675

# カラーパレット
C_BG       = (18,  18,  18)    # ダーク背景 #121212
C_HEADER   = (26,  35, 126)    # 濃紺 #1a237e
C_WHITE    = (255, 255, 255)
C_GRAY     = (180, 180, 180)
C_RED      = (244,  67,  54)   # 危険度80+
C_YELLOW   = (255, 193,   7)   # 危険度60-79
C_GREEN    = ( 76, 175,  80)   # 危険度40-59
C_ACCENT   = (255, 152,   0)   # オレンジ強調
C_ROW_ODD  = ( 28,  28,  28)
C_ROW_EVEN = ( 38,  38,  38)
C_FOOTER   = ( 50,  50,  50)

HEADER_H = 90
FOOTER_H = 50
ROW_H    = 46


def _score_color(score: int) -> tuple[int, int, int]:
    if score >= 80:
        return C_RED
    elif score >= 60:
        return C_YELLOW
    else:
        return C_GREEN


def _score_emoji(score: int) -> str:
    if score >= 80:
        return "🔴"
    elif score >= 60:
        return "🟡"
    else:
        return "🟢"


# ════════════════════════════════════════════════════════════
# フォント取得
# ════════════════════════════════════════════════════════════

def _get_font(size: int, bold: bool = False):
    """
    日本語フォントを取得する。
    環境に応じて利用可能なフォントにフォールバックする。
    """
    from PIL import ImageFont

    # 優先順位: NotoSansCJK → IPAゴシック → Noto Sans JP → デフォルト
    candidates_bold = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Bold.otf",
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        # Windows
        "C:/Windows/Fonts/meiryob.ttc",
        "C:/Windows/Fonts/YuGothB.ttc",
        # macOS
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    ]
    candidates_regular = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "C:/Windows/Fonts/meiryo.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    ]
    candidates = candidates_bold if bold else candidates_regular
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    # フォールバック: Pillow デフォルト
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


# ════════════════════════════════════════════════════════════
# 描画ヘルパー
# ════════════════════════════════════════════════════════════

def _draw_rect(draw, x: int, y: int, w: int, h: int, color: tuple) -> None:
    draw.rectangle([x, y, x + w, y + h], fill=color)


def _draw_text_center(draw, text: str, y: int, font, color: tuple, width: int = IMG_W) -> None:
    try:
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
    except AttributeError:
        tw = font.getlength(text)
    x = int((width - tw) // 2)
    draw.text((x, y), text, font=font, fill=color)


def _draw_header(draw, title: str, date_str: str, font_large, font_small,
                 accent_color: tuple = C_WHITE) -> None:
    _draw_rect(draw, 0, 0, IMG_W, HEADER_H, C_HEADER)
    # ヘッダー左にアクセントバー
    _draw_rect(draw, 0, 0, 6, HEADER_H, accent_color)
    _draw_text_center(draw, title, 12, font_large, C_WHITE)
    _draw_text_center(draw, date_str, 58, font_small, C_GRAY)


def _draw_footer(draw, font_small) -> None:
    y = IMG_H - FOOTER_H
    _draw_rect(draw, 0, y, IMG_W, FOOTER_H, C_FOOTER)
    _draw_text_center(draw, "AI分析 by 競艇荒れ検知Bot  |  #競艇 #ボートレース #競艇予想",
                      y + 14, font_small, C_GRAY)


# ════════════════════════════════════════════════════════════
# 画像生成: ① 危険な1号艇
# ════════════════════════════════════════════════════════════

def generate_danger_image(data: dict, output_path: str) -> None:
    from PIL import Image, ImageDraw

    img  = Image.new("RGB", (IMG_W, IMG_H), C_BG)
    draw = ImageDraw.Draw(img)

    font_hd   = _get_font(36, bold=True)
    font_sub  = _get_font(22)
    font_row  = _get_font(26, bold=True)
    font_rsn  = _get_font(20)
    font_ft   = _get_font(18)

    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    _draw_header(draw, "本日の危険な1号艇 TOP10", date_str, font_hd, font_sub,
                 accent_color=C_RED)

    items = data.get("danger_boat1", [])
    y = HEADER_H + 4

    for i, d in enumerate(items[:10]):
        bg = C_ROW_ODD if i % 2 == 0 else C_ROW_EVEN
        _draw_rect(draw, 0, y, IMG_W, ROW_H, bg)

        score = d.get("score", 0)
        sc    = _score_color(score)

        # 順位
        draw.text((12, y + 10), f"{i+1:>2}", font=font_row, fill=C_GRAY)
        # スコアバー（左端）
        bar_w = int(score / 100 * 60)
        _draw_rect(draw, 50, y + 16, bar_w, 14, sc)
        # 危険度数値
        draw.text((118, y + 10), f"{score:3d}", font=font_row, fill=sc)
        # 場所・レース番号
        draw.text((175, y + 10), f"{d.get('venue','')}{d.get('race','')}R", font=font_row, fill=C_WHITE)
        # 選手名
        draw.text((380, y + 10), d.get("racer", "?"), font=font_row, fill=C_WHITE)
        # 理由
        draw.text((640, y + 14), d.get("reason", ""), font=font_rsn, fill=C_YELLOW)

        y += ROW_H

    # 残りをグレーで埋める
    remaining = 10 - len(items)
    for j in range(remaining):
        bg = C_ROW_ODD if (len(items) + j) % 2 == 0 else C_ROW_EVEN
        _draw_rect(draw, 0, y, IMG_W, ROW_H, bg)
        draw.text((12, y + 10), f"{len(items)+j+1:>2}", font=font_row, fill=(60, 60, 60))
        draw.text((175, y + 10), "---", font=font_row, fill=(60, 60, 60))
        y += ROW_H

    _draw_footer(draw, font_ft)
    img.save(output_path)
    log.info("[画像] 保存: %s", output_path)


# ════════════════════════════════════════════════════════════
# 画像生成: ② 激走モーター
# ════════════════════════════════════════════════════════════

def generate_hot_motor_image(data: dict, output_path: str) -> None:
    from PIL import Image, ImageDraw

    img  = Image.new("RGB", (IMG_W, IMG_H), C_BG)
    draw = ImageDraw.Draw(img)

    font_hd  = _get_font(36, bold=True)
    font_sub = _get_font(22)
    font_row = _get_font(26, bold=True)
    font_ft  = _get_font(18)

    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    _draw_header(draw, "激走モーター TOP10", date_str, font_hd, font_sub,
                 accent_color=C_ACCENT)

    items = data.get("hot_motor", [])
    y = HEADER_H + 4

    for i, m in enumerate(items[:10]):
        bg = C_ROW_ODD if i % 2 == 0 else C_ROW_EVEN
        _draw_rect(draw, 0, y, IMG_W, ROW_H, bg)

        score = m.get("score", 0)
        bar_w = int(score / 100 * 150)
        _draw_rect(draw, 50, y + 16, bar_w, 14, C_ACCENT)

        draw.text((12, y + 10), f"{i+1:>2}", font=font_row, fill=C_GRAY)
        draw.text((215, y + 10), f"指数 {score:3d}", font=font_row, fill=C_ACCENT)
        draw.text((380, y + 10),
                  f"{m.get('venue','')} {m.get('motor_no','')}号機",
                  font=font_row, fill=C_WHITE)
        y += ROW_H

    for j in range(10 - len(items)):
        bg = C_ROW_ODD if (len(items) + j) % 2 == 0 else C_ROW_EVEN
        _draw_rect(draw, 0, y, IMG_W, ROW_H, bg)
        draw.text((12, y + 10), f"{len(items)+j+1:>2}", font=font_row, fill=(60, 60, 60))
        draw.text((380, y + 10), "---", font=font_row, fill=(60, 60, 60))
        y += ROW_H

    _draw_footer(draw, font_ft)
    img.save(output_path)
    log.info("[画像] 保存: %s", output_path)


# ════════════════════════════════════════════════════════════
# 画像生成: ③ 万舟警報
# ════════════════════════════════════════════════════════════

def generate_manshuu_image(data: dict, output_path: str) -> None:
    from PIL import Image, ImageDraw

    img  = Image.new("RGB", (IMG_W, IMG_H), C_BG)
    draw = ImageDraw.Draw(img)

    font_hd  = _get_font(36, bold=True)
    font_sub = _get_font(22)
    font_row = _get_font(26, bold=True)
    font_ft  = _get_font(18)

    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    _draw_header(draw, "AI万舟警報 TOP10", date_str, font_hd, font_sub,
                 accent_color=C_RED)

    items = data.get("manshuu_alert", [])
    y = HEADER_H + 4

    for i, u in enumerate(items[:10]):
        bg = C_ROW_ODD if i % 2 == 0 else C_ROW_EVEN
        _draw_rect(draw, 0, y, IMG_W, ROW_H, bg)

        score = u.get("score", 0)
        sc    = _score_color(score)
        bar_w = int(score / 100 * 120)
        _draw_rect(draw, 50, y + 16, bar_w, 14, sc)

        draw.text((12, y + 10), f"{i+1:>2}", font=font_row, fill=C_GRAY)
        draw.text((178, y + 10), f"荒れ指数{score:3d}", font=font_row, fill=sc)
        draw.text((420, y + 10),
                  f"{u.get('venue','')}{u.get('race','')}R",
                  font=font_row, fill=C_WHITE)
        y += ROW_H

    for j in range(10 - len(items)):
        bg = C_ROW_ODD if (len(items) + j) % 2 == 0 else C_ROW_EVEN
        _draw_rect(draw, 0, y, IMG_W, ROW_H, bg)
        draw.text((12, y + 10), f"{len(items)+j+1:>2}", font=font_row, fill=(60, 60, 60))
        draw.text((420, y + 10), "---", font=font_row, fill=(60, 60, 60))
        y += ROW_H

    _draw_footer(draw, font_ft)
    img.save(output_path)
    log.info("[画像] 保存: %s", output_path)


# ════════════════════════════════════════════════════════════
# 画像生成: ④ 覚醒モーター
# ════════════════════════════════════════════════════════════

def generate_awakening_image(data: dict, output_path: str) -> None:
    from PIL import Image, ImageDraw

    img  = Image.new("RGB", (IMG_W, IMG_H), C_BG)
    draw = ImageDraw.Draw(img)

    font_hd  = _get_font(36, bold=True)
    font_sub = _get_font(22)
    font_row = _get_font(26, bold=True)
    font_ft  = _get_font(18)

    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    _draw_header(draw, "覚醒モーター TOP10", date_str, font_hd, font_sub,
                 accent_color=(0, 188, 212))

    items = data.get("awakening_motor", [])
    y = HEADER_H + 4

    for i, a in enumerate(items[:10]):
        bg = C_ROW_ODD if i % 2 == 0 else C_ROW_EVEN
        _draw_rect(draw, 0, y, IMG_W, ROW_H, bg)

        score = a.get("score", 0)
        bar_w = int(score / 100 * 150)
        _draw_rect(draw, 50, y + 16, bar_w, 14, (0, 188, 212))  # シアン

        draw.text((12, y + 10), f"{i+1:>2}", font=font_row, fill=C_GRAY)
        draw.text((215, y + 10), f"覚醒度{score:3d}", font=font_row, fill=(0, 229, 255))
        draw.text((380, y + 10),
                  f"{a.get('venue','')} {a.get('motor_no','')}号機",
                  font=font_row, fill=C_WHITE)
        y += ROW_H

    for j in range(10 - len(items)):
        bg = C_ROW_ODD if (len(items) + j) % 2 == 0 else C_ROW_EVEN
        _draw_rect(draw, 0, y, IMG_W, ROW_H, bg)
        draw.text((12, y + 10), f"{len(items)+j+1:>2}", font=font_row, fill=(60, 60, 60))
        draw.text((380, y + 10), "---", font=font_row, fill=(60, 60, 60))
        y += ROW_H

    _draw_footer(draw, font_ft)
    img.save(output_path)
    log.info("[画像] 保存: %s", output_path)


# ════════════════════════════════════════════════════════════
# まとめて生成
# ════════════════════════════════════════════════════════════

GENERATORS = {
    "danger":    (generate_danger_image,    "danger.png"),
    "hot":       (generate_hot_motor_image, "hot_motor.png"),
    "manshuu":   (generate_manshuu_image,   "manshuu.png"),
    "awakening": (generate_awakening_image, "awakening.png"),
}


def generate_all_images(data: dict, prefix: str = "") -> list[str]:
    """全4種の画像を生成し、出力ファイルパスのリストを返す"""
    paths: list[str] = []
    for key, (func, default_name) in GENERATORS.items():
        out = prefix + default_name if prefix else default_name
        func(data, out)
        paths.append(out)
    return paths


# ════════════════════════════════════════════════════════════
# エントリポイント
# ════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="競艇 X 投稿用ランキング画像生成")
    parser.add_argument("--input",  default="ranking_cache.json",
                        help="ランキング JSON ファイル（デフォルト: ranking_cache.json）")
    parser.add_argument("--type",   choices=list(GENERATORS.keys()),
                        help="生成する画像の種類（省略時は --all が必要）")
    parser.add_argument("--output", help="出力ファイル名（--type 指定時のみ有効）")
    parser.add_argument("--all",    action="store_true", help="全4種まとめて生成")
    parser.add_argument("--prefix", default="", help="--all 時の出力ファイル名プレフィックス")
    args = parser.parse_args()

    # データ読み込み
    if not os.path.exists(args.input):
        log.error("ランキングファイルが見つかりません: %s", args.input)
        log.error("先に `python x_ranking.py --generate` を実行してください")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    try:
        from PIL import Image  # noqa: F401  インポート確認
    except ImportError:
        log.error("Pillow がインストールされていません: pip install Pillow")
        sys.exit(1)

    if args.all:
        paths = generate_all_images(data, prefix=args.prefix)
        log.info("[完了] %d枚生成: %s", len(paths), ", ".join(paths))
    elif args.type:
        func, default_name = GENERATORS[args.type]
        out = args.output or default_name
        func(data, out)
        log.info("[完了] %s", out)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
