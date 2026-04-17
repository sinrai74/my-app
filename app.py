"""
app.py  ── 競艇予想API (Flask / Render対応)

依存ファイル（同ディレクトリに置くこと）:
  - index.html    … フロントエンド
  - odds_fetch.py … 外部オッズ取得（任意）
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# odds_fetch.py が存在する場合のみ import
try:
    from odds_fetch import fetch_odds
    HAS_ODDS_FETCH = True
except ImportError:
    HAS_ODDS_FETCH = False

app = Flask(__name__)
CORS(app)


# ════════════════════════════════════════════════════════════
# ルート
# ════════════════════════════════════════════════════════════

@app.route("/")
def home():
    try:
        with open("index.html", encoding="utf-8") as f:  # ← with文で確実にclose
            return f.read()
    except FileNotFoundError:
        return "<h2>index.html が見つかりません</h2>", 404


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# ════════════════════════════════════════════════════════════
# ヘルパー関数
# ════════════════════════════════════════════════════════════

def kelly_fraction(p: float, odds: float) -> float:
    """Kelly基準（フラクショナル）"""
    b = odds - 1.0
    if b <= 0 or p <= 0:
        return 0.0
    k = (b * p - (1.0 - p)) / b
    return max(0.0, min(k, 1.0))


def calc_combo_score(boats: list[dict], lane1: int, lane2: int, lane3: int) -> float:
    """
    3連単1通りのスコアを計算。
    lane1=1着, lane2=2着, lane3=3着 として各艇の能力を重み付け加算。
    """
    score = 0.0
    for b in boats:
        if b["lane"] == lane1:
            score += b["win_rate"] * 3.0
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


# ════════════════════════════════════════════════════════════
# 予想API
# ════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "JSONのパースに失敗しました"}), 400

    boats    = data.get("boats", [])
    bankroll = float(data.get("bankroll", 10000))

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

    # ── オッズ取得（外部 → フォールバック: リクエスト内 → 暫定値0）──
    odds_map: dict = {}
    if HAS_ODDS_FETCH:
        try:
            odds_map = fetch_odds() or {}
        except Exception:
            odds_map = {}
    if not odds_map:
        odds_map = data.get("odds", {})

    # ── 全120通りのスコア計算 ────────────────────────────────
    raw_patterns = []
    for i in range(1, 7):
        for j in range(1, 7):
            for k in range(1, 7):
                if i == j or j == k or i == k:
                    continue
                raw_patterns.append({
                    "combo": f"{i}-{j}-{k}",
                    "score": calc_combo_score(boats, i, j, k),
                    "lanes": [i, j, k],
                })

    # スコア → 確率（softmax風: 全非負にしてから正規化）
    score_min = min(p["score"] for p in raw_patterns)
    score_adj = [p["score"] - score_min + 1e-6 for p in raw_patterns]
    score_sum = sum(score_adj)

    all_patterns = []
    for p, adj in zip(raw_patterns, score_adj):
        combo = p["combo"]
        lanes = p["lanes"]
        prob  = adj / score_sum  # 正規化済み確率

        # 実オッズ（外部取得 or フロントから渡された値 or 確率逆数で暫定推定）
        real_odds = float(odds_map.get(combo, 0.0))
        if real_odds <= 0:
            real_odds = (1.0 / max(prob, 1e-9)) * 0.75

        # 真EV（人気薄バイアス補正）
        popularity_bias = 1.0 + (1.0 / lanes[0]) * 0.15
        true_ev = prob * real_odds * popularity_bias

        # Kelly × 0.3（保守的）→ 100円単位
        kelly = kelly_fraction(prob, real_odds)
        bet   = (int(bankroll * kelly * 0.3) // 100) * 100

        # 判定
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

    # EV降順ソート
    all_patterns.sort(key=lambda x: x["ev"], reverse=True)

    return jsonify({
        "status":   "ok",
        "buy":      [p for p in all_patterns if p["ev"] >= 1.2][:10],
        "all":      all_patterns,
        "bankroll": bankroll,
    })


# ════════════════════════════════════════════════════════════
# 起動
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)