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
        "C:/Windows/Fonts/meiryo.ttc",
        "C:/Windows/Fonts/YuGothB.ttc",
        # macOS
        "/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc",
    ]
    candidates_regular = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJKjp-Regular.otf",
        "/usr/share/fonts/opentype/ipafont-gothic/ipag.ttf",
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "C:/Windows/Fonts/meiryob.ttc",
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
    x = (width - tw) // 2
    draw.text((x, y), text, font=font, fill=color)


def _draw_header(draw, title: str, date_str: str, font_large, font_small,
                 accent_color: tuple = C_WHITE) -> None:
    _draw_rect(draw, 0, 0, IMG_W, HEADER_H, C_HEADER)
    _draw_rect(draw, 0, 0, 6, HEADER_H, accent_color)
    _draw_text_center(draw, title, 12, font_large, C_WHITE)
    _draw_text_center(draw, date_str, 58, font_small, C_GRAY)


def _draw_footer(draw, font_small) -> None:
    y = IMG_H - FOOTER_H
    _draw_rect(draw, 0, y, IMG_W, FOOTER_H, C_FOOTER)


# ════════════════════════════════════════════════════════════
# 画像生成: ① 危険な1号艇
# ════════════════════════════════════════════════════════════

def generate_danger_image(data: dict, output_path: str) -> None:
    from PIL import Image, ImageDraw

    # TOP3は★付き2行、4位以降は1行
    TOP_ROWS   = 3
    ROW_H_TALL = 72   # TOP3の行高さ（2行分）
    ROW_H_SLIM = 46   # 4位以降の行高さ（1行）
    IMG_H2 = HEADER_H + ROW_H_TALL * TOP_ROWS + ROW_H_SLIM * 7 + FOOTER_H + 8

    img  = Image.new("RGB", (IMG_W, IMG_H2), C_BG)
    draw = ImageDraw.Draw(img)

    font_hd   = _get_font(36, bold=True)
    font_sub  = _get_font(22)
    font_row  = _get_font(24, bold=True)
    font_rsn  = _get_font(18)
    font_star = _get_font(17)
    font_ft   = _get_font(18)

    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    _draw_header(draw, "本日の危険な1号艇 TOP10", date_str, font_hd, font_sub,
                 accent_color=C_RED)

    items = data.get("danger_boat1", [])
    y = HEADER_H + 4

    for i, d in enumerate(items[:10]):
        is_top3  = (i < TOP_ROWS)
        row_h    = ROW_H_TALL if is_top3 else ROW_H_SLIM
        bg       = C_ROW_ODD if i % 2 == 0 else C_ROW_EVEN
        _draw_rect(draw, 0, y, IMG_W, row_h, bg)

        score    = d.get("score", 0)
        sc       = _score_color(score)
        rank_str = "S" if score >= 80 else "A" if score >= 60 else "B"

        # ── 1行目 ────────────────────────────────────────
        y1 = y + (6 if is_top3 else 10)
        draw.text((12,  y1), f"{i+1:>2}",  font=font_row, fill=C_GRAY)
        bar_w = int(score / 100 * 55)
        _draw_rect(draw, 48, y1 + 4, bar_w, 16, sc)
        draw.text((112, y1), rank_str, font=font_row, fill=sc)
        draw.text((148, y1),
                  f"{d.get('venue','')}{d.get('race','')}R",
                  font=font_row, fill=C_WHITE)
        draw.text((360, y1), d.get("racer", "?"), font=font_row, fill=C_WHITE)

        # ── 2行目: TOP3のみ ★ + 理由 ──────────────────
        if is_top3:
            stars = d.get("stars", {})
            y2 = y + 40
            if stars:
                star_text = (
                    f"ST {stars.get('ST','')}"
                    f"  機力 {stars.get('機力','')}"
                    f"  近況 {stars.get('近況','')}"
                    f"  相手 {stars.get('相手','')}"
                )
                draw.text((22, y2), star_text, font=font_star, fill=(180, 180, 255))
            else:
                reason = d.get("reason", "")
                draw.text((22, y2), reason, font=font_rsn, fill=C_YELLOW)
        else:
            # 4位以降は理由を1行目の右に表示
            reason = d.get("reason", "")
            if len(reason) > 30:
                reason = reason[:28] + "…"
            draw.text((560, y1), reason, font=font_rsn, fill=C_YELLOW)

        y += row_h

    # 残り枠
    for j in range(10 - len(items)):
        bg = C_ROW_ODD if (len(items) + j) % 2 == 0 else C_ROW_EVEN
        _draw_rect(draw, 0, y, IMG_W, ROW_H_SLIM, bg)
        draw.text((12, y + 10), f"{len(items)+j+1:>2}",
                  font=font_row, fill=(60, 60, 60))
        draw.text((175, y + 10), "---", font=font_row, fill=(60, 60, 60))
        y += ROW_H_SLIM

    _draw_rect(draw, 0, y, IMG_W, FOOTER_H, C_FOOTER)
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
    _draw_header(draw, "🔥 激走モーター TOP10", date_str, font_hd, font_sub)

    items = data.get("hot_motor", [])
    y = HEADER_H + 4

    for i, m in enumerate(items[:10]):
        bg = C_ROW_ODD if i % 2 == 0 else C_ROW_EVEN
        _draw_rect(draw, 0, y, IMG_W, ROW_H, bg)

        score = m.get("score", 0)
        bar_w = int(score / 100 * 150)
        _draw_rect(draw, 50, y + 16, bar_w, 14, C_ACCENT)

        draw.text((12, y + 10), f"{i+1:>2}", font=font_row, fill=C_GRAY)
        draw.text((210, y + 10),
                  f"{m.get('venue','')} {m.get('motor_no','')}号機",
                  font=font_row, fill=C_WHITE)
        races = m.get("races_today_str", "")
        if races:
            draw.text((900, y + 13), f"本日{races}", font=_get_font(18), fill=C_ACCENT)
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

def _draw_manshuu_block(data: dict, output_path: str,
                        start: int, end: int, title: str) -> None:
    """万舟警報を start〜end 番目（1始まり）で切り出して1枚の画像を生成する"""
    from PIL import Image, ImageDraw

    BLOCK_SIZE  = end - start + 1      # 表示件数（start〜end の件数）
    ROW_H2      = 76                   # 2行分の行高さ
    IMG_H2      = HEADER_H + ROW_H2 * BLOCK_SIZE + FOOTER_H + 8

    img  = Image.new("RGB", (IMG_W, IMG_H2), C_BG)
    draw = ImageDraw.Draw(img)

    font_hd  = _get_font(36, bold=True)
    font_sub = _get_font(22)
    font_row = _get_font(24, bold=True)
    font_key = _get_font(20)
    font_ft  = _get_font(18)

    date_str = f"{data['date'][4:6]}/{data['date'][6:8]}"
    _draw_header(draw, title, date_str, font_hd, font_sub, accent_color=C_RED)

    all_items = data.get("manshuu_alert", [])
    items = all_items[start - 1 : end]   # 1始まり → 0始まりに変換
    y = HEADER_H + 4

    for i, u in enumerate(items):
        rank = start + i   # 実際の順位（1〜10）
        bg = C_ROW_ODD if i % 2 == 0 else C_ROW_EVEN
        _draw_rect(draw, 0, y, IMG_W, ROW_H2, bg)

        score = u.get("score", 0)
        sc    = _score_color(score)
        bar_w = int(score / 100 * 100)
        _draw_rect(draw, 50, y + 10, bar_w, 12, sc)

        draw.text((12,  y + 6), f"{rank:>2}",  font=font_row, fill=C_GRAY)
        draw.text((175, y + 6),
                  f"{u.get('venue','')}{u.get('race','')}R",
                  font=font_row, fill=C_WHITE)

        key_racer  = u.get("key_racer",  "")
        key_reason = u.get("key_reason", "")
        key_text   = f"注目: {key_racer}  [{key_reason}]" if key_racer else ""
        draw.text((22, y + 44), key_text, font=font_key, fill=C_YELLOW)

        y += ROW_H2

    # データが足りない場合は空行で埋める
    for j in range(BLOCK_SIZE - len(items)):
        rank = end - (BLOCK_SIZE - len(items)) + j + 1
        bg = C_ROW_ODD if (len(items) + j) % 2 == 0 else C_ROW_EVEN
        _draw_rect(draw, 0, y, IMG_W, ROW_H2, bg)
        draw.text((12, y + 6), f"{rank:>2}", font=font_row, fill=(60, 60, 60))
        draw.text((380, y + 6), "---",       font=font_row, fill=(60, 60, 60))
        y += ROW_H2

    _draw_rect(draw, 0, y, IMG_W, FOOTER_H, C_FOOTER)
    img.save(output_path)
    log.info("[画像] 保存: %s", output_path)


def generate_manshuu_image(data: dict, output_path: str) -> None:
    """互換用: 1-5 と 6-10 の2枚を生成する"""
    base, ext = os.path.splitext(output_path)
    generate_manshuu_image_top5(data,  f"{base}_1{ext}")
    generate_manshuu_image_6to10(data, f"{base}_2{ext}")


def generate_manshuu_image_top5(data: dict, output_path: str) -> None:
    """万舟警報 1〜5位"""
    _draw_manshuu_block(data, output_path, start=1, end=5, title="万舟警報 TOP5")


def generate_manshuu_image_6to10(data: dict, output_path: str) -> None:
    """万舟警報 6〜10位"""
    _draw_manshuu_block(data, output_path, start=6, end=10, title="万舟警報 6〜10位")


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
    _draw_header(draw, "⚡ 覚醒モーター TOP10", date_str, font_hd, font_sub)

    items = data.get("awakening_motor", [])
    y = HEADER_H + 4

    for i, a in enumerate(items[:10]):
        bg = C_ROW_ODD if i % 2 == 0 else C_ROW_EVEN
        _draw_rect(draw, 0, y, IMG_W, ROW_H, bg)

        score = a.get("score", 0)
        bar_w = int(score / 100 * 150)
        _draw_rect(draw, 50, y + 16, bar_w, 14, (0, 188, 212))  # シアン

        draw.text((12, y + 10), f"{i+1:>2}", font=font_row, fill=C_GRAY)
        draw.text((210, y + 10),
                  f"{a.get('venue','')} {a.get('motor_no','')}号機",
                  font=font_row, fill=C_WHITE)
        races = a.get("races_today_str", "")
        if races:
            draw.text((900, y + 13), f"本日{races}", font=_get_font(18), fill=(0, 188, 212))
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
    "manshuu_top5":   (generate_manshuu_image_top5,   "manshuu_1.png"),
    "manshuu_6to10":  (generate_manshuu_image_6to10,  "manshuu_2.png"),
    "awakening": (generate_awakening_image, "awakening.png"),
}


def generate_all_images(data: dict, prefix: str = "") -> list[str]:
    """全5枚（危険1号艇/激走/万舟1-5/万舟6-10/覚醒）を生成し、パスのリストを返す"""
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
