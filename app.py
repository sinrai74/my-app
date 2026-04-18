"""
app.py  ── 競艇予想API (Flask / Render対応)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import date

try:
    from odds_fetch import fetch_odds, _debug_html, _parse_v3
    import requests
    from bs4 import BeautifulSoup
    HAS_ODDS_FETCH = True
except ImportError:
    HAS_ODDS_FETCH = False

try:
    from config import get_venue_config, VENUE_NAME_MAP
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

app = Flask(__name__)
CORS(app)

VENUE_MAP_BUILTIN: dict[str, str] = {
    "桐生":"01","江戸川":"02","戸田":"03","平和島":"04","多摩川":"05",
    "浜名湖":"06","蒲郡":"07","常滑":"08","津":"09","三国":"10",
    "びわこ":"11","住之江":"12","尼崎":"13","鳴門":"14","丸亀":"15",
    "児島":"16","宮島":"17","徳山":"18","下関":"19","若松":"20",
    "芦屋":"21","福岡":"22","唐津":"23","大村":"24",
}
COURSE_BONUS_DEFAULT = {1: 1.5, 2: 0.8, 3: 0.4, 4: 0.2, 5: -0.2, 6: -0.5}


@app.route("/")
def home():
    try:
        with open("index.html", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h2>index.html が見つかりません</h2>", 404


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ════════════════════════════════════════════════════════════
# ★デバッグエンドポイント
# 使い方: GET /debug_odds?rno=1&jcd=04&hd=20260418
# → HTMLの構造解析結果とオッズ取得結果をJSONで返す
# ════════════════════════════════════════════════════════════

@app.route("/debug_odds")
def debug_odds():
    if not HAS_ODDS_FETCH:
        return jsonify({"error": "odds_fetch が読み込めません"}), 500

    race_no    = int(request.args.get("rno", 1))
    venue_code = request.args.get("jcd", "04").zfill(2)
    race_date  = request.args.get("hd", date.today().strftime("%Y%m%d"))

    BASE_URL = "https://www.boatrace.jp/owpc/pc/race/odds3t"
    HEADERS  = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.boatrace.jp/",
    }

    try:
        res  = requests.get(
            BASE_URL,
            params={"rno": race_no, "jcd": venue_code, "hd": race_date},
            headers=HEADERS, timeout=10
        )
        res.raise_for_status()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    soup   = BeautifulSoup(res.content, "html.parser")
    tables = soup.find_all("table")

    # テーブル構造の要約
    table_info = []
    for ti, table in enumerate(tables):
        rows = table.select("tbody tr")
        op   = table.find_all("td", class_="oddsPoint")
        sample_rows = []
        for tr in rows[:3]:
            tds = tr.find_all("td")
            sample_rows.append([
                {
                    "class": " ".join(td.get("class", [])),
                    "rowspan": td.get("rowspan"),
                    "text": td.get_text(strip=True)[:15],
                }
                for td in tds
            ])
        table_info.append({
            "index":       ti,
            "row_count":   len(rows),
            "oddsPoint_count": len(op),
            "sample_rows": sample_rows,
        })

    # oddsPoint 先頭30件
    odds_raw = [
        td.get_text(strip=True)
        for td in soup.select("td.oddsPoint")[:30]
    ]

    # 実際の取得結果
    odds_result = _parse_v3(soup)
    sample_odds = dict(list(odds_result.items())[:20])

    return jsonify({
        "url":          res.url,
        "status_code":  res.status_code,
        "table_count":  len(tables),
        "table_info":   table_info,
        "odds_raw_top30": odds_raw,
        "odds_count":   len(odds_result),
        "odds_sample":  sample_odds,
    })


# ════════════════════════════════════════════════════════════
# ヘルパー関数
# ════════════════════════════════════════════════════════════

def get_course_bonus(venue_code: str) -> dict[int, float]:
    if HAS_CONFIG:
        cfg = get_venue_config(venue_code)
        return {
            c: cfg.get(f"{c}コース補正", COURSE_BONUS_DEFAULT.get(c, 1.0))
            for c in range(1, 7)
        }
    return COURSE_BONUS_DEFAULT


def kelly_fraction(p: float, odds: float) -> float:
    b = odds - 1.0
    if b <= 0 or p <= 0:
        return 0.0
    return max(0.0, min((b * p - (1.0 - p)) / b, 1.0))


def calc_combo_score(boats, lane1, lane2, lane3, course_bonus):
    score = 0.0
    for b in boats:
        cb = course_bonus.get(b["lane"], 1.0)
        if b["lane"] == lane1:
            score += b["win_rate"] * 3.0 * cb
            score += b["motor"]    * 2.0
            score -= b["ex_time"]  * 10.0
            score -= b["start"]    * 5.0
        elif b["lane"] == lane2:
            score += b["win_rate"] * 2.0
            score += b["motor"]    * 1.5
        elif b["lane"] == lane3:
            score += b["win_rate"] * 1.0
            score += b["motor"]    * 1.0
    return score


def resolve_venue_code(venue_name, venue_code):
    if venue_code and str(venue_code).isdigit():
        return str(venue_code).zfill(2)
    if HAS_CONFIG and venue_name in VENUE_NAME_MAP:
        return VENUE_NAME_MAP[venue_name]
    return VENUE_MAP_BUILTIN.get(venue_name, "01")


# ════════════════════════════════════════════════════════════
# 予想API
# ════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "JSONのパースに失敗しました"}), 400

    boats      = data.get("boats", [])
    bankroll   = float(data.get("bankroll", 10000))
    venue_name = data.get("venue_name", "")
    venue_code = data.get("venue_code", "")
    race_no    = int(data.get("race_no", 1))
    race_date  = data.get("race_date", date.today().strftime("%Y%m%d"))

    if len(boats) != 6:
        return jsonify({"error": "boats は6艇必須です"}), 400

    required_keys = {"lane", "ex_time", "motor", "win_rate", "start"}
    for i, b in enumerate(boats):
        missing = required_keys - set(b.keys())
        if missing:
            return jsonify({"error": f"boats[{i}] にキーが不足: {missing}"}), 400
        for k in required_keys:
            try:
                float(b[k])
            except (TypeError, ValueError):
                return jsonify({"error": f"boats[{i}][{k}] が数値ではありません"}), 400

    vc           = resolve_venue_code(venue_name, venue_code)
    course_bonus = get_course_bonus(vc)

    odds_map: dict  = {}
    odds_source     = "暫定推定"

    if HAS_ODDS_FETCH and venue_name and race_no and race_date:
        try:
            odds_map = fetch_odds(race_no, vc, race_date) or {}
            if odds_map:
                odds_source = "公式サイト取得"
        except Exception as e:
            print(f"[app] fetch_odds 失敗: {e}")

    if not odds_map:
        odds_map = data.get("odds", {})
        if odds_map:
            odds_source = "フロント送信値"

    raw_patterns = []
    for i in range(1, 7):
        for j in range(1, 7):
            for k in range(1, 7):
                if i == j or j == k or i == k:
                    continue
                raw_patterns.append({
                    "combo": f"{i}-{j}-{k}",
                    "score": calc_combo_score(boats, i, j, k, course_bonus),
                    "lanes": [i, j, k],
                })

    score_min = min(p["score"] for p in raw_patterns)
    score_adj = [p["score"] - score_min + 1e-6 for p in raw_patterns]
    score_sum = sum(score_adj)

    all_patterns = []
    for p, adj in zip(raw_patterns, score_adj):
        combo = p["combo"]
        lanes = p["lanes"]
        prob  = adj / score_sum

        real_odds = float(odds_map.get(combo, 0.0))
        if real_odds <= 0:
            real_odds = (1.0 / max(prob, 1e-9)) * 0.75

        if odds_source == "公式サイト取得":
            true_ev = prob * real_odds
        else:
            true_ev = prob * real_odds * (1.0 + (1.0 / lanes[0]) * 0.15)

        kelly = kelly_fraction(prob, real_odds)
        bet   = (int(bankroll * kelly * 0.3) // 100) * 100

        if true_ev >= 2.0:
            verdict = "✅ 強く買い"
        elif true_ev >= 1.2:
            verdict = "⚠️ 買い"
        else:
            verdict = "🚫 見送り"

        all_patterns.append({
            "combo":   combo,
            "prob":    round(prob, 5),
            "odds":    round(real_odds, 2),
            "ev":      round(true_ev, 3),
            "bet":     bet,
            "verdict": verdict,
        })

    all_patterns.sort(key=lambda x: x["ev"], reverse=True)

    return jsonify({
        "status":      "ok",
        "venue_name":  venue_name,
        "venue_code":  vc,
        "race_no":     race_no,
        "race_date":   race_date,
        "odds_source": odds_source,
        "buy":         [p for p in all_patterns if p["ev"] >= 1.2][:10],
        "all":         all_patterns,
        "bankroll":    bankroll,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)