#!/usr/bin/env python3
"""
x_note_report.py  ── note/有料コンテンツ用 AI競艇新聞 生成

ranking_cache.json から全レースデータを読み込み、
HTML（noteへのコピペ用）と PDF（添付・投稿用）を生成してメール送信する。

Usage:
    python x_note_report.py                    # 今日のレポートを生成
    python x_note_report.py --date 20260628
    python x_note_report.py --dry-run          # 送信せず保存のみ
    python x_note_report.py --html-only        # HTML のみ生成
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

log = logging.getLogger("x_note")

JST     = timezone(timedelta(hours=9))
MAIL_TO = "bigkirinuki@gmail.com"
RANKING_CACHE = "ranking_cache.json"


# ════════════════════════════════════════════════════════════
# AIコメント生成
# ════════════════════════════════════════════════════════════

def _ai_comment(d: dict) -> str:
    """危険な1号艇レースのAIコメントを1〜2行で生成"""
    bd   = d.get("breakdown", {})
    comments: list[str] = []

    # STが最大の危険因子
    st_w = bd.get("st", (0, 0))[1]
    ex_w = bd.get("ex", (0, 0))[1]
    if st_w >= 14:
        avg_st = d.get("avg_st", 0)
        comments.append(f"平均ST{avg_st:.2f}で遅れリスク大")
    if ex_w >= 14:
        ex_time = d.get("ex_time")
        if ex_time:
            comments.append(f"展示タイム{ex_time:.2f}秒（遅め）")

    # モーターが危険因子
    motor_w = bd.get("motor", (0, 0))[1]
    if motor_w >= 10:
        motor = d.get("motor", 0)
        comments.append(f"モーター2連率{motor:.0f}%で機力不足")

    # 等級・勝率
    grade_w = bd.get("grade", (0, 0))[1]
    wr_w    = bd.get("wr",    (0, 0))[1]
    cls     = d.get("racer_class", "")
    wr      = d.get("win_rate", 0)
    if grade_w >= 8 and cls in ("B1", "B2"):
        comments.append(f"{cls}級選手（格下）")
    elif wr_w >= 10 and wr > 0:
        comments.append(f"勝率{wr:.1f}で低調")

    # 相手が強い
    rival_w = bd.get("rival", (0, 0))[1]
    if rival_w >= 10:
        comments.append("強力な対抗馬あり")

    score  = d.get("score", 0)
    rank   = "S" if score >= 80 else "A" if score >= 60 else "B"
    prefix = {"S": "⚠️ 要注意！", "A": "注目レース", "B": "やや危険"}.get(rank, "")

    if comments:
        return f"{prefix}。{'、'.join(comments[:2])}。"
    return f"{prefix}。総合的な危険判定。"


def _manshuu_comment(u: dict) -> str:
    """万舟警報レースのAIコメントを生成"""
    key_reason = u.get("key_reason", "")
    reasons    = [r.replace("🔥 ", "").replace("🔥", "")
                  for r in key_reason.split(" / ") if r]
    score      = u.get("score", 0)
    rank       = "S" if score >= 80 else "A" if score >= 60 else "B"
    prefix     = {"S": "⚠️ 高配当期待！", "A": "万舟候補", "B": "やや荒れ"}.get(rank, "")
    if reasons:
        return f"{prefix}。{reasons[0]}。"
    return f"{prefix}。複合要因で荒れ判定。"


# ════════════════════════════════════════════════════════════
# HTML生成
# ════════════════════════════════════════════════════════════

def generate_html(data: dict, output_path: str) -> None:
    """note コピペ用 HTML を生成する"""
    date_str  = data.get("date", "")
    date_disp = f"{date_str[4:6]}/{date_str[6:8]}" if len(date_str) >= 8 else ""
    now_str   = datetime.now(JST).strftime("%H:%M")

    all_danger  = data.get("all_danger",  data.get("danger_boat1", []))
    all_manshuu = data.get("all_manshuu", data.get("manshuu_alert", []))
    hot_motor   = data.get("hot_motor", [])
    awake_motor = data.get("awakening_motor", [])

    # ランク集計
    s_danger = [d for d in all_danger  if d["score"] >= 80]
    a_danger = [d for d in all_danger  if 60 <= d["score"] < 80]
    b_danger = [d for d in all_danger  if 40 <= d["score"] < 60]

    def rank_badge(score: int) -> str:
        if score >= 80: return '<span class="badge s">S</span>'
        if score >= 60: return '<span class="badge a">A</span>'
        return '<span class="badge b">B</span>'

    def breakdown_bar(bd: dict) -> str:
        items = [
            ("ST",   bd.get("st",    (0,0))[1], "#ef5350"),
            ("展示",  bd.get("ex",    (0,0))[1], "#ff7043"),
            ("機力",  bd.get("motor", (0,0))[1], "#ffa726"),
            ("等級",  bd.get("grade", (0,0))[1], "#ab47bc"),
            ("勝率",  bd.get("wr",    (0,0))[1], "#42a5f5"),
            ("相手",  bd.get("rival", (0,0))[1], "#26a69a"),
        ]
        bars = ""
        for label, val, color in items:
            pct = int(val / 20 * 100)  # 最大weight=20
            bars += (
                f'<div class="bar-row">'
                f'<span class="bar-label">{label}</span>'
                f'<div class="bar-bg">'
                f'<div class="bar-fill" style="width:{pct}%;background:{color}"></div>'
                f'</div>'
                f'<span class="bar-val">{val:.1f}pt</span>'
                f'</div>'
            )
        return f'<div class="breakdown">{bars}</div>'

    # 危険な1号艇テーブル行
    danger_rows = ""
    for d in all_danger:
        bd      = d.get("breakdown", {})
        comment = _ai_comment(d)
        danger_rows += f"""
        <tr class="rank-{'s' if d['score']>=80 else 'a' if d['score']>=60 else 'b'}">
          <td>{rank_badge(d['score'])}</td>
          <td><strong>{d['venue']}{d['race']}R</strong></td>
          <td>{d['score']}</td>
          <td>{d.get('racer','?')}</td>
          <td class="reason">{d.get('reason','')}</td>
          <td class="comment">{comment}</td>
        </tr>
        <tr class="breakdown-row">
          <td colspan="6">{breakdown_bar(bd)}</td>
        </tr>"""

    # 万舟警報テーブル行
    manshuu_rows = ""
    for u in all_manshuu:
        comment = _manshuu_comment(u)
        key_reason = u.get("key_reason", "").replace("🔥 ", "🔥 ")
        manshuu_rows += f"""
        <tr class="rank-{'s' if u['score']>=80 else 'a' if u['score']>=60 else 'b'}">
          <td>{rank_badge(u['score'])}</td>
          <td><strong>{u['venue']}{u['race']}R</strong></td>
          <td>{u['score']}</td>
          <td colspan="2" class="reason">{key_reason}</td>
          <td class="comment">{comment}</td>
        </tr>"""

    # 激走・覚醒モーター
    def motor_section(items: list[dict], title: str, color: str) -> str:
        if not items:
            return f'<p style="color:#999">データ蓄積中（数日後に表示されます）</p>'
        rows = ""
        for i, m in enumerate(items[:20], 1):
            recent = m.get("recent5", m.get("recent10", "---"))
            gap    = m.get("gap", "")
            old_r  = m.get("old_2rate")
            new_r  = m.get("new_2rate")
            detail = f"直近: {recent}"
            if gap != "" and isinstance(gap, (int, float)):
                sign = "+" if gap > 0 else ""
                detail += f" / 公式比{sign}{gap:.0f}%"
            if old_r is not None and new_r is not None:
                detail += f" / {old_r:.0f}%→{new_r:.0f}%"
            rows += f'<tr><td>{i}</td><td>{m["venue"]}</td><td>{m["motor_no"]}号機</td><td>{detail}</td></tr>'
        return f'<table class="motor-table"><thead><tr><th>#</th><th>会場</th><th>モーター</th><th>データ</th></tr></thead><tbody>{rows}</tbody></table>'

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI競艇新聞 {date_disp}</title>
<style>
  :root {{
    --bg: #0f0f1a; --card: #1a1a2e; --border: #2a2a4a;
    --text: #e0e0e0; --gray: #888; --accent: #4fc3f7;
    --s: #ef5350; --a: #ffa726; --b: #66bb6a;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: 'Noto Sans JP', 'Hiragino Sans', sans-serif; padding: 16px; }}
  .newspaper-header {{ text-align: center; border: 2px solid var(--accent); padding: 20px; margin-bottom: 24px; background: var(--card); border-radius: 8px; }}
  .newspaper-header h1 {{ font-size: 2.2em; color: var(--accent); letter-spacing: 0.1em; }}
  .newspaper-header .meta {{ color: var(--gray); margin-top: 6px; font-size: 0.9em; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-bottom: 24px; }}
  .summary-card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 16px; text-align: center; }}
  .summary-card .num {{ font-size: 2.5em; font-weight: bold; }}
  .summary-card .label {{ color: var(--gray); font-size: 0.85em; margin-top: 4px; }}
  .summary-card.s .num {{ color: var(--s); }}
  .summary-card.a .num {{ color: var(--a); }}
  .summary-card.b .num {{ color: var(--b); }}
  .section {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }}
  .section h2 {{ font-size: 1.3em; color: var(--accent); border-bottom: 1px solid var(--border); padding-bottom: 10px; margin-bottom: 16px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9em; }}
  th {{ background: #1e1e3a; color: var(--gray); padding: 8px 10px; text-align: left; font-weight: normal; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr.rank-s td {{ border-left: 3px solid var(--s); }}
  tr.rank-a td {{ border-left: 3px solid var(--a); }}
  tr.rank-b td {{ border-left: 3px solid var(--b); }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-weight: bold; font-size: 0.9em; }}
  .badge.s {{ background: var(--s); color: #fff; }}
  .badge.a {{ background: var(--a); color: #000; }}
  .badge.b {{ background: var(--b); color: #000; }}
  .reason {{ color: #aaa; font-size: 0.85em; }}
  .comment {{ color: var(--accent); font-size: 0.85em; }}
  .breakdown-row td {{ background: #14142a; padding: 8px 10px; }}
  .breakdown {{ display: flex; flex-direction: column; gap: 4px; }}
  .bar-row {{ display: flex; align-items: center; gap: 8px; }}
  .bar-label {{ width: 32px; font-size: 0.75em; color: var(--gray); text-align: right; flex-shrink: 0; }}
  .bar-bg {{ flex: 1; background: #2a2a4a; border-radius: 3px; height: 10px; overflow: hidden; }}
  .bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
  .bar-val {{ width: 40px; font-size: 0.75em; color: var(--gray); text-align: right; flex-shrink: 0; }}
  .motor-table td {{ font-size: 0.88em; }}
  .hint-box {{ background: #1e2a1e; border: 1px solid #2a4a2a; border-radius: 6px; padding: 14px; margin-top: 16px; }}
  .hint-box h3 {{ color: var(--b); margin-bottom: 8px; }}
  .hint-box p {{ color: #aaa; font-size: 0.9em; line-height: 1.6; }}
  @media print {{
    body {{ background: #fff; color: #000; }}
    .badge.s {{ background: #e53935 !important; }}
  }}
</style>
</head>
<body>

<div class="newspaper-header">
  <h1>📰 AI競艇新聞</h1>
  <div class="meta">{date_disp} 発行　生成時刻: {now_str}　毎朝7時更新</div>
  <div class="meta" style="margin-top:6px;color:#4fc3f7">全レースAI分析 | データ駆動型競艇情報</div>
</div>

<!-- サマリー -->
<div class="summary-grid">
  <div class="summary-card s">
    <div class="num">{len(s_danger)}</div>
    <div class="label">Sランク（危険度80+）</div>
  </div>
  <div class="summary-card a">
    <div class="num">{len(a_danger)}</div>
    <div class="label">Aランク（危険度60-79）</div>
  </div>
  <div class="summary-card b">
    <div class="num">{len(b_danger)}</div>
    <div class="label">Bランク（危険度40-59）</div>
  </div>
</div>

<!-- 危険な1号艇 全レース -->
<div class="section">
  <h2>⚠️ 危険な1号艇 全{len(all_danger)}レース（スコア40以上）</h2>
  <table>
    <thead>
      <tr><th>ランク</th><th>レース</th><th>危険度</th><th>選手</th><th>判定理由</th><th>AIコメント</th></tr>
    </thead>
    <tbody>{danger_rows}</tbody>
  </table>

  <div class="hint-box">
    <h3>📌 買い方のヒント</h3>
    <p>
      <strong>Sランク（危険度80以上）</strong>：1号艇を頭固定にしない戦略が期待値を高める傾向があります。<br>
      <strong>Aランク（危険度60-79）</strong>：1号艇の連複を外した買い方が有効な場面が多いです。<br>
      <strong>Bランク（危険度40-59）</strong>：参考程度に。条件次第で変動します。<br>
      ※これは買い目の推奨ではなく、データの読み方の参考情報です。
    </p>
  </div>
</div>

<!-- 万舟警報 全レース -->
<div class="section">
  <h2>💰 万舟警報 全{len(all_manshuu)}レース</h2>
  <table>
    <thead>
      <tr><th>ランク</th><th>レース</th><th>荒れ指数</th><th colspan="2">荒れる理由</th><th>AIコメント</th></tr>
    </thead>
    <tbody>{manshuu_rows}</tbody>
  </table>
</div>

<!-- 激走モーター -->
<div class="section">
  <h2>🔥 激走モーター TOP20</h2>
  {motor_section(hot_motor, "激走モーター", "#ff7043")}
</div>

<!-- 覚醒モーター -->
<div class="section">
  <h2>⚡ 覚醒モーター TOP10</h2>
  {motor_section(awake_motor, "覚醒モーター", "#00bcd4")}
</div>

<!-- フッター -->
<div style="text-align:center;color:#444;font-size:0.8em;padding:20px 0;">
  AI競艇新聞 | 全レース機械学習分析 | 毎日更新<br>
  ※本レポートはデータ分析結果であり、的中を保証するものではありません。
</div>

</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    log.info("[note] HTML保存: %s (%d bytes)", output_path, len(html))


# ════════════════════════════════════════════════════════════
# PDF生成
# ════════════════════════════════════════════════════════════

def generate_pdf(html_path: str, pdf_path: str) -> bool:
    """
    HTML → PDF 変換。
    優先順位: weasyprint → pdfkit(wkhtmltopdf) → Pillow(簡易画像PDF)
    """
    # ① WeasyPrint（最高品質）
    try:
        from weasyprint import HTML
        HTML(filename=html_path).write_pdf(pdf_path)
        log.info("[note] PDF保存(weasyprint): %s", pdf_path)
        return True
    except ImportError:
        pass
    except Exception as e:
        log.warning("[note] weasyprint失敗: %s", e)

    # ② pdfkit + wkhtmltopdf
    try:
        import pdfkit
        pdfkit.from_file(html_path, pdf_path,
                         options={"encoding": "utf-8", "quiet": ""})
        log.info("[note] PDF保存(pdfkit): %s", pdf_path)
        return True
    except ImportError:
        pass
    except Exception as e:
        log.warning("[note] pdfkit失敗: %s", e)

    # ③ フォールバック: Pillow で簡易PDF（白背景テキスト）
    try:
        from PIL import Image, ImageDraw, ImageFont
        import io

        def _gf(size, bold=False):
            for p in (["/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
                        "C:/Windows/Fonts/meiryob.ttc"] if bold else
                       ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                        "C:/Windows/Fonts/truetype/fonts-japanese-gothic.ttf",
                        "C:/Windows/Fonts/meiryo.ttc"]):
                if os.path.exists(p):
                    try: return ImageFont.truetype(p, size)
                    except: pass
            return ImageFont.load_default()

        # HTMLからテキストを抽出して簡易レンダリング
        import re
        with open(html_path, encoding="utf-8") as f:
            html_content = f.read()
        text = re.sub(r'<[^>]+>', '', html_content)
        text = re.sub(r'\n{3,}', '\n\n', text).strip()
        lines = text.split('\n')[:120]

        W, LINE_H = 1200, 22
        H = max(1800, len(lines) * LINE_H + 100)
        img  = Image.new("RGB", (W, H), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        font = _gf(16)
        y = 40
        for line in lines:
            line = line.strip()
            if not line: y += 8; continue
            draw.text((40, y), line[:80], font=font, fill=(30, 30, 30))
            y += LINE_H

        # PIL でマルチページPDFに保存
        img.save(pdf_path, "PDF", resolution=150)
        log.info("[note] PDF保存(Pillow簡易): %s", pdf_path)
        return True
    except Exception as e:
        log.warning("[note] Pillow PDF失敗: %s", e)

    log.error("[note] PDF生成失敗（全手法が使用不可）")
    return False


# ════════════════════════════════════════════════════════════
# メール送信
# ════════════════════════════════════════════════════════════

def send_note_report(
    html_path: str,
    pdf_path: Optional[str],
    date_str: str,
    dry_run: bool = False,
) -> bool:
    date_disp = f"{date_str[4:6]}/{date_str[6:8]}"
    subject   = f"📰 AI競艇新聞 {date_disp}（note用レポート）"
    body      = (
        f"AI競艇新聞 {date_disp} を添付します。\n\n"
        f"【添付ファイル】\n"
        f"・note_report_{date_str}.html … noteエディタへのコピペ用\n"
        f"・note_report_{date_str}.pdf  … 画像投稿・プロフィール固定用\n\n"
        f"HTML はブラウザで開いて Ctrl+A → Ctrl+C でコピーし、\n"
        f"noteエディタに貼り付けてください。\n"
    )

    if dry_run:
        print("=" * 60)
        print(f"[DRY RUN] 件名: {subject}")
        print(body)
        print(f"添付: {html_path}, {pdf_path}")
        print("=" * 60)
        return True

    gmail_address = os.getenv("GMAIL_ADDRESS", "")
    app_password  = os.getenv("GMAIL_APP_PASS", "")
    if not gmail_address or not app_password:
        log.error("GMAIL_ADDRESS / GMAIL_APP_PASS が未設定")
        return False

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"]    = gmail_address
    msg["To"]      = MAIL_TO
    msg.attach(MIMEText(body, "plain", "utf-8"))

    for path, mime_type, subtype in [
        (html_path, "text",        "html"),
        (pdf_path,  "application", "pdf"),
    ]:
        if path and os.path.exists(path):
            with open(path, "rb") as f:
                part = MIMEBase(mime_type, subtype)
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", "attachment",
                            filename=os.path.basename(path))
            msg.attach(part)

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.ehlo(); smtp.starttls()
            smtp.login(gmail_address, app_password)
            smtp.sendmail(gmail_address, MAIL_TO, msg.as_string())
        log.info("[note] 送信成功: %s", subject)
        return True
    except smtplib.SMTPException as e:
        log.error("[note] 送信失敗: %s", e)
        return False


# ════════════════════════════════════════════════════════════
# エントリポイント
# ════════════════════════════════════════════════════════════

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(description="AI競艇新聞 生成・送信")
    parser.add_argument("--date",      help="対象日 YYYYMMDD（省略時は today）")
    parser.add_argument("--input",     default=RANKING_CACHE,
                        help=f"ランキングJSONファイル（デフォルト: {RANKING_CACHE}）")
    parser.add_argument("--dry-run",   action="store_true", help="送信せず保存のみ")
    parser.add_argument("--html-only", action="store_true", help="HTMLのみ生成（PDF不要）")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        log.error("ランキングファイルが見つかりません: %s", args.input)
        log.error("先に `python x_ranking.py --generate` を実行してください")
        sys.exit(1)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    date_str  = args.date or data.get("date", datetime.now(JST).strftime("%Y%m%d"))
    html_path = f"note_report_{date_str}.html"
    pdf_path  = f"note_report_{date_str}.pdf"

    # HTML 生成
    generate_html(data, html_path)

    # PDF 生成
    if not args.html_only:
        ok = generate_pdf(html_path, pdf_path)
        if not ok:
            pdf_path = None
    else:
        pdf_path = None

    # メール送信
    ok = send_note_report(html_path, pdf_path, date_str, dry_run=args.dry_run)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
