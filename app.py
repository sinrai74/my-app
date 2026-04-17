from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(**name**)
CORS(app)

@app.route("/")
def home():
return open("index.html", encoding="utf-8").read()

@app.route("/health")
def health():
return "ok"

# ===== Kelly関数 =====

def kelly_fraction(p, odds):
b = odds - 1
q = 1 - p
if b <= 0:
return 0
k = (b * p - q) / b
return max(0, min(k, 1))

# ===== 予想API =====

@app.route("/predict", methods=["POST"])
def predict():
data = request.get_json()

```
boats = data.get("boats", [])
odds_map = data.get("odds", {})
bankroll = data.get("bankroll", 10000)

if len(boats) != 6:
    return jsonify({"error": "boats は6艇必須です"}), 400

all_patterns = []

# ===== 全120通り =====
for i in range(1,7):
    for j in range(1,7):
        for k in range(1,7):
            if i != j and j != k and i != k:

                combo = f"{i}-{j}-{k}"
                lanes = [i, j, k]

                # ===== 確率スコア（完全版） =====
                score = 0

                for b in boats:
                    if b["lane"] == lanes[0]:  # 1着
                        score += b["win_rate"] * 3
                        score += b["motor"] * 2
                        score -= b["ex_time"] * 10
                        score -= b["start"] * 5

                    elif b["lane"] == lanes[1]:  # 2着
                        score += b["win_rate"] * 2
                        score += b["motor"] * 1.5

                    elif b["lane"] == lanes[2]:  # 3着
                        score += b["win_rate"] * 1
                        score += b["motor"] * 1

                prob = max(0.001, score / 300)

                odds = odds_map.get(combo, 0)

                # ===== EV =====
                ev = prob * odds

                # ===== Kelly =====
                kelly = kelly_fraction(prob, odds)
                bet = int(bankroll * kelly * 0.3)

                all_patterns.append({
                    "combo": combo,
                    "prob": round(prob, 4),
                    "odds": odds,
                    "ev": round(ev, 3),
                    "bet": bet,
                    "verdict": "買い" if ev > 1 else "見送り"
                })

# EV順
all_patterns.sort(key=lambda x: x["ev"], reverse=True)

return jsonify({
    "buy": all_patterns[:5],
    "all": all_patterns
})
```

# ===== 起動 =====

if **name** == "**main**":
port = int(os.environ.get("PORT", 10000))
app.run(host="0.0.0.0", port=port)
