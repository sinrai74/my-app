"""
app.py  ── 競艇予想API (Flask / Render対応)
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import date

try:
    from boat_api import fetch_race, fetch_available_races
    HAS_BOAT_API = True
except ImportError:
    HAS_BOAT_API = False

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
COURSE_BONUS_DEFAULT = {1:1.5, 2:0.8, 3:0.4, 4:0.2, 5:-0.2, 6:-0.5}


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


@app.route("/races")
def races():
    if not HAS_BOAT_API:
        return jsonify({"error": "boat_api が読み込めません"}), 500
    race_date = request.args.get("date", date.today().strftime("%Y%m%d"))
    result    = fetch_available_races(race_date)
    if not result:
        return jsonify({"error": f"{race_date} のレースデータが取得できません"}), 404
    return jsonify({"date": race_date, "races": result})


@app.route("/boats")
def boats_endpoint():
    if not HAS_BOAT_API:
        return jsonify({"error": "boat_api が読み込めません"}), 500
    race_date  = request.args.get("date",  date.today().strftime("%Y%m%d"))
    venue_name = request.args.get("venue", "")
    race_no    = int(request.args.get("race", 1))
    boats = fetch_race(race_date, venue_name, race_no)
    if boats is None:
        return jsonify({"error": "データ取得失敗（まだ公開されていないか存在しないレース）"}), 404
    missing_ex = [b["lane"] for b in boats if b.get("ex_time") is None]
    if missing_ex:
        return jsonify({
            "error": "自動取得成功。展示タイムのみ手入力してください",
            "boats": boats, "missing_ex_time": missing_ex, "need_ex_time": True,
        }), 202
    return jsonify({"date": race_date, "venue": venue_name, "race_no": race_no, "boats": boats})


# ════════════════════════════════════════════════════════════
# ヘルパー
# ════════════════════════════════════════════════════════════

def get_course_bonus(venue_code: str) -> dict[int, float]:
    if HAS_CONFIG:
        cfg = get_venue_config(venue_code)
        return {c: cfg.get(f"{c}コース補正", COURSE_BONUS_DEFAULT.get(c, 1.0)) for c in range(1, 7)}
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


def resolve_venue_code(venue_name: str, venue_code: str) -> str:
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
    # ★フロントから送られてくる実オッズ（空なら暫定推定）
    odds_map   = data.get("odds", {})

    # ── バリデーション ──────────────────────────────────────
    if len(boats) != 6:
        return jsonify({"error": "boats は6艇必須です"}), 400
    required_keys = {"lane", "ex_time", "motor", "win_rate", "start"}
    for i, b in enumerate(boats):
        for k in required_keys - set(b.keys()):
            return jsonify({"error": f"boats[{i}] にキーが不足: {k}"}), 400
        for k in required_keys:
            try:
                if b[k] is None: raise ValueError
                float(b[k])
            except (TypeError, ValueError):
                return jsonify({"error": f"boats[{i}][{k}] が数値ではありません"}), 400

    vc           = resolve_venue_code(venue_name, venue_code)
    course_bonus = get_course_bonus(vc)

    # ── オッズソースを判定 ──────────────────────────────────
    real_odds_count = len(odds_map)
    if real_odds_count >= 60:
        odds_source = "実オッズ入力"
    elif real_odds_count > 0:
        odds_source = f"実オッズ一部入力({real_odds_count}点)"
    else:
        odds_source = "暫定推定"

    # ── 全120通りのスコア計算 ────────────────────────────────
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

    # ── softmax風に確率化 ────────────────────────────────────
    score_min = min(p["score"] for p in raw_patterns)
    score_adj = [p["score"] - score_min + 1e-6 for p in raw_patterns]
    score_sum = sum(score_adj)

    all_patterns = []
    for p, adj in zip(raw_patterns, score_adj):
        combo = p["combo"]
        lanes = p["lanes"]
        prob  = adj / score_sum  # 正規化された確率（合計1）

        # ★実オッズがあればそれを使う。なければ確率から推定
        if combo in odds_map and float(odds_map[combo]) > 0:
            real_odds  = float(odds_map[combo])
            # 実オッズがある場合はバイアス不要。純粋なEV = prob × odds
            true_ev    = prob * real_odds
        else:
            # 暫定推定: 控除率75%として implied odds を使う
            real_odds  = (1.0 / max(prob, 1e-9)) * 0.75
            # 暫定の場合は1着艇番バイアス補正を加える
            true_ev    = prob * real_odds * (1.0 + (1.0 / lanes[0]) * 0.15)

        kelly = kelly_fraction(prob, real_odds)
        bet   = (int(bankroll * kelly * 0.3) // 100) * 100

        if true_ev >= 2.0:   verdict = "✅ 強く買い"
        elif true_ev >= 1.2: verdict = "⚠️ 買い"
        else:                verdict = "🚫 見送り"

        all_patterns.append({
            "combo":        combo,
            "prob":         round(prob, 5),
            "odds":         round(real_odds, 2),
            "ev":           round(true_ev, 3),
            "bet":          bet,
            "verdict":      verdict,
            "odds_is_real": combo in odds_map,
        })

    all_patterns.sort(key=lambda x: x["ev"], reverse=True)

    return jsonify({
        "status":           "ok",
        "venue_name":       venue_name,
        "venue_code":       vc,
        "race_no":          race_no,
        "race_date":        race_date,
        "odds_source":      odds_source,
        "real_odds_count":  real_odds_count,
        "buy":              [p for p in all_patterns if p["ev"] >= 1.2][:10],
        "all":              all_patterns,
        "bankroll":         bankroll,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)