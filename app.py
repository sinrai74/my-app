"""
app.py  ── 競艇予想API (Flask / Render対応)

依存ファイル（同ディレクトリに置くこと）:
  - index.html    … フロントエンド
  - odds_fetch.py … 公式サイトからオッズ取得
  - config.py     … 場別設定（任意）
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import date

try:
    from odds_fetch import fetch_odds
    HAS_ODDS_FETCH = True
except ImportError:
    HAS_ODDS_FETCH = False

# config.py の場別設定を任意で読み込む
try:
    from config import get_venue_config, VENUE_NAME_MAP
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

app = Flask(__name__)
CORS(app)


# ════════════════════════════════════════════════════════════
# 場コード対応表（config.py がない環境用の内蔵版）
# ════════════════════════════════════════════════════════════

VENUE_MAP_BUILTIN: dict[str, str] = {
    "桐生":"01","江戸川":"02","戸田":"03","平和島":"04","多摩川":"05",
    "浜名湖":"06","蒲郡":"07","常滑":"08","津":"09","三国":"10",
    "びわこ":"11","住之江":"12","尼崎":"13","鳴門":"14","丸亀":"15",
    "児島":"16","宮島":"17","徳山":"18","下関":"19","若松":"20",
    "芦屋":"21","福岡":"22","唐津":"23","大村":"24",
}

# 場別コース補正（config.py がない場合のデフォルト値）
COURSE_BONUS_DEFAULT = {1: 1.5, 2: 0.8, 3: 0.4, 4: 0.2, 5: -0.2, 6: -0.5}


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
# ヘルパー関数
# ════════════════════════════════════════════════════════════

def get_course_bonus(venue_code: str) -> dict[int, float]:
    """場コードから場別コース補正を返す"""
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
    k = (b * p - (1.0 - p)) / b
    return max(0.0, min(k, 1.0))


def calc_combo_score(
    boats: list[dict],
    lane1: int, lane2: int, lane3: int,
    course_bonus: dict[int, float],
) -> float:
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
    """場名または場コードから2桁の場コードを返す"""
    if venue_code and venue_code.isdigit():
        return venue_code.zfill(2)
    if HAS_CONFIG and venue_name in VENUE_NAME_MAP:
        return VENUE_NAME_MAP[venue_name]
    if venue_name in VENUE_MAP_BUILTIN:
        return VENUE_MAP_BUILTIN[venue_name]
    return "01"  # デフォルト


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

    # ── バリデーション ──────────────────────────────────────
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

    # ── 場コード解決 ────────────────────────────────────────
    vc = resolve_venue_code(venue_name, venue_code)

    # ── 場別コース補正 ──────────────────────────────────────
    course_bonus = get_course_bonus(vc)

    # ── オッズ取得（公式サイト → フロント渡し → 暫定推定 の順）─
    odds_map: dict = {}
    odds_source = "暫定推定"

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

    # ── スコア → 確率（softmax風） ──────────────────────────
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

        # EV（場別補正なし: 公式オッズがあればバイアス不要）
        if odds_source == "公式サイト取得":
            true_ev = prob * real_odds
        else:
            popularity_bias = 1.0 + (1.0 / lanes[0]) * 0.15
            true_ev = prob * real_odds * popularity_bias

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
        "status":       "ok",
        "venue_name":   venue_name,
        "venue_code":   vc,
        "race_no":      race_no,
        "race_date":    race_date,
        "odds_source":  odds_source,    # ← どこからオッズを取ったか表示
        "buy":          [p for p in all_patterns if p["ev"] >= 1.2][:10],
        "all":          all_patterns,
        "bankroll":     bankroll,
    })


# ════════════════════════════════════════════════════════════
# 起動
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)