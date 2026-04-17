from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# トップページ（HTML表示）
@app.route("/")
def home():
    return open("index.html", encoding="utf-8").read()

# ヘルスチェック
@app.route("/health")
def health():
    return "ok"

# 予想API
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    boats = data.get("boats", [])
    odds_map = data.get("odds", {})
    bankroll = data.get("bankroll", 10000)

    # エラー処理
    if len(boats) != 6:
        return jsonify({"error": "boats は6艇必須です"}), 400

    # 仮の確率生成（全120通り）
    prob_map = {}
    combos = []

    for i in range(1,7):
        for j in range(1,7):
            for k in range(1,7):
                if i != j and j != k and i != k:
                    combo = f"{i}-{j}-{k}"
                    prob = 1 / 120
                    prob_map[combo] = prob
                    combos.append(combo)

    all_patterns = []

    for combo in combos:
        prob = prob_map[combo]
        odds = odds_map.get(combo, 0)

        # ★EV計算（これが核心）
        ev = prob * odds

        k = kelly_fraction(prob, odds)
        bet = int(bankroll * k * 0.3)  # 0.3 = 安全係数

        all_patterns.append({
            "combo": combo,
            "prob": round(prob, 4),
            "odds": odds,
            "ev": round(ev, 3),
            "bet": bet,
            "verdict": "買い" if ev > 1 else "見送り"
        })

    # EV順に並べ替え
    all_patterns.sort(key=lambda x: x["ev"], reverse=True)

    return jsonify({
        "buy": all_patterns[:5],
        "all": all_patterns
    })

def kelly_fraction(p, odds):
    b = odds - 1
    q = 1 - p
    if b <= 0:
        return 0
    k = (b * p - q) / b
    return max(0, min(k, 1))

# 起動
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)