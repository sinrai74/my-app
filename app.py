from itertools import permutations
import math
import os

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# コース補正（強化版）
COURSE_BONUS = {1: 1.5, 2: 0.8, 3: 0.4, 4: 0.2, 5: -0.2, 6: -0.5}

W_COURSE   = 1.5
W_EX_TIME  = 2.0
W_MOTOR    = 1.0
W_WIN_RATE = 1.0
W_START    = 1.5

EV_MIN     = 1.20
KELLY_FRAC = 0.25

def softmax(scores):
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]

def calc_score(boat):
    lane      = boat["lane"]
    ex_time   = boat["ex_time"]
    motor     = boat["motor"]
    win_rate  = boat["win_rate"]
    start     = boat["start"]

    course  = COURSE_BONUS.get(lane, 0.0) * W_COURSE
    ex_inv  = (1.0 / ex_time) ** 3 * 1000
    mot     = (motor / 100.0) * W_MOTOR
    wr      = (win_rate / 10.0) * W_WIN_RATE
    st_inv  = (1.0 / (start + 0.01)) * 0.2 * W_START

    return course + ex_inv + mot + wr + st_inv

def kelly_bet(prob, odds, bankroll):
    edge = prob * odds - 1.0
    if edge <= 0 or odds <= 1:
        return 0.0
    kelly = edge / (odds - 1.0)
    bet   = kelly * KELLY_FRAC * bankroll
    return max(0.0, min(bet, bankroll * 0.10))

def verdict(ev):
    if ev >= 2.0:
        return "✅ 強く買い"
    elif ev >= EV_MIN:
        return "⚠️ 買い"
    else:
        return "🚫 見送り"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data     = request.get_json(force=True)
        boats    = data.get("boats", [])
        bankroll = float(data.get("bankroll", 10000))

        odds_map = data.get("odds", {})

        if len(boats) != 6:
            return jsonify({"error": "boats は6艇必須です"}), 400

        scores = [calc_score(b) for b in boats]
        probs  = softmax(scores)

        boat_data = []
        for b, sc, pr in zip(boats, scores, probs):
            boat_data.append({
                "lane": b["lane"],
                "score": round(sc, 4),
                "win_prob": round(pr, 4),
            })

        sorted_boats = sorted(boat_data, key=lambda x: x["score"], reverse=True)
        top4_lanes   = [b["lane"] for b in sorted_boats[:4]]

        prob_map = {b["lane"]: b["win_prob"] for b in boat_data}

        picks = []
        for perm in permutations(top4_lanes, 3):
            p1, p2, p3 = perm

            p_win  = prob_map[p1]
            remain = 1.0 - p_win
            p_2nd  = prob_map[p2] / (remain + 1e-9) * remain
            remain2 = remain - p_2nd
            p_3rd  = prob_map[p3] / (remain2 + 1e-9) * remain2

            combo_prob = p_win * p_2nd * p_3rd

            combo_key = f"{p1}-{p2}-{p3}"

            implied_odds = (1.0 / max(combo_prob, 1e-9)) * 0.75
            real_odds = odds_map.get(combo_key, implied_odds)

            ev = real_odds * combo_prob

            bet = kelly_bet(combo_prob, real_odds, bankroll)
            bet = round(bet / 100) * 100

            picks.append({
                "combo": combo_key,
                "probability": round(combo_prob, 5),
                "odds": round(real_odds, 2),
                "ev": round(ev, 4),
                "bet": int(bet),
                "verdict": verdict(ev),
            })

        picks.sort(key=lambda x: x["ev"], reverse=True)

        return jsonify({
            "status": "ok",
            "buy": [p for p in picks if p["ev"] >= EV_MIN],
            "all": picks
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/")
def home():
    return send_file("index.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)