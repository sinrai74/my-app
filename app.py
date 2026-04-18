"""
app.py  ── 競艇予想API (Flask / Render対応)

新機能:
  GET  /races?date=YYYYMMDD  … その日の開催レース一覧
  GET  /boats?date=YYYYMMDD&venue=XX&race=N  … 6艇データ自動取得
  POST /predict  … 予想（boats自動取得 or 手動入力）
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import date

try:
    from boat_api import fetch_race, fetch_available_races, VENUE_NAME_TO_NUM
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


# ════════════════════════════════════════════════════════════
# ルート
# ════════════════════════════════════════════════════════════

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
# 開催レース一覧
# GET /races?date=YYYYMMDD  (省略時は今日)
# ════════════════════════════════════════════════════════════

@app.route("/races")
def races():
    if not HAS_BOAT_API:
        return jsonify({"error": "boat_api が読み込めません"}), 500

    race_date = request.args.get("date", date.today().strftime("%Y%m%d"))
    result    = fetch_available_races(race_date)

    if not result:
        return jsonify({"error": f"{race_date} のレースデータが取得できません（未開催または更新前）"}), 404

    return jsonify({"date": race_date, "races": result})


# ════════════════════════════════════════════════════════════
# 6艇データ自動取得
# GET /boats?date=YYYYMMDD&venue=江戸川&race=1
# ════════════════════════════════════════════════════════════

@app.route("/boats")
def boats_endpoint():
    if not HAS_BOAT_API:
        return jsonify({"error": "boat_api が読み込めません"}), 500

    race_date  = request.args.get("date",  date.today().strftime("%Y%m%d"))
    venue_name = request.args.get("venue", "")
    race_no    = int(request.args.get("race", 1))

    boats = fetch_race(race_date, venue_name, race_no)
    if boats is None:
        return jsonify({"error": "データ取得失敗（まだ公開されていないか、存在しないレース）"}), 404

    return jsonify({
        "date":       race_date,
        "venue":      venue_name,
        "race_no":    race_no,
        "boats":      boats,
        "note":       "ex_time（展示タイム）は出走表にはないため手動入力が必要です",
    })


# ════════════════════════════════════════════════════════════
# ヘルパー
# ════════════════════════════════════════════════════════════

def get_course_bonus(venue_code: str) -> dict[int, float]:
    if HAS_CONFIG:
        cfg = get_venue_config(venue_code)
        return {c: cfg.get(f"{c}コース補正", COURSE_BONUS_DEFAULT.get(c, 1.0))
                for c in range(1, 7)}
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

    # ── boats が空なら BoatraceOpenAPI から自動取得 ──────────
    auto_fetched = False
    if not boats and HAS_BOAT_API and venue_name and race_no:
        fetched = fetch_race(race_date, venue_name, race_no)
        if fetched:
            # ex_time が None の艇があるか確認
            missing_ex = [b["lane"] for b in fetched if b.get("ex_time") is None]
            if missing_ex:
                return jsonify({
                    "error":         "自動取得成功。展示タイムを入力してください",
                    "boats":         fetched,
                    "missing_ex_time": missing_ex,
                    "need_ex_time":  True,
                }), 202
            boats        = fetched
            auto_fetched = True

    # ── バリデーション ──────────────────────────────────────
    if len(boats) != 6:
        return jsonify({"error": "boats は6艇必須です（自動取得できない場合は手動入力）"}), 400

    required_keys = {"lane", "ex_time", "motor", "win_rate", "start"}
    for i, b in enumerate(boats):
        missing = required_keys - set(b.keys())
        if missing:
            return jsonify({"error": f"boats[{i}] にキーが不足: {missing}"}), 400
        for k in required_keys:
            try:
                if b[k] is None:
                    raise ValueError
                float(b[k])
            except (TypeError, ValueError):
                return jsonify({"error": f"boats[{i}][{k}] が数値ではありません"}), 400

    # ── 場別コース補正 ──────────────────────────────────────
    vc           = resolve_venue_code(venue_name, venue_code)
    course_bonus = get_course_bonus(vc)

    # ── オッズ（フロント送信値 or 暫定推定）───────────────────
    odds_map    = data.get("odds", {})
    odds_source = "フロント送信値" if odds_map else "暫定推定"

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

        true_ev = prob * real_odds * (1.0 + (1.0 / lanes[0]) * 0.15)
        kelly   = kelly_fraction(prob, real_odds)
        bet     = (int(bankroll * kelly * 0.3) // 100) * 100

        if true_ev >= 2.0:   verdict = "✅ 強く買い"
        elif true_ev >= 1.2: verdict = "⚠️ 買い"
        else:                verdict = "🚫 見送り"

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
        "status":       "ok",
        "venue_name":   venue_name,
        "venue_code":   vc,
        "race_no":      race_no,
        "race_date":    race_date,
        "auto_fetched": auto_fetched,
        "odds_source":  odds_source,
        "buy":          [p for p in all_patterns if p["ev"] >= 1.2][:10],
        "all":          all_patterns,
        "bankroll":     bankroll,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)