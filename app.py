"""
app.py  ── 競艇予想API (Flask / Render対応)

エンドポイント:
  GET  /                        … フロントエンド
  GET  /health                  … 死活監視
  GET  /races?date=YYYYMMDD     … 開催レース一覧
  GET  /boats?date=YYYYMMDD&venue=XX&race=N  … 6艇データ自動取得
  POST /predict                 … 予想
  POST /record                  … 結果を記録
  GET  /history                 … 記録一覧
  GET  /stats                   … 回収率・統計
  DELETE /record/<id>           … 記録削除
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from datetime import date, datetime
from pathlib import Path

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

# ── 記録ファイルパス（Renderの永続ストレージ or ローカル）──
RECORD_FILE = Path(os.environ.get("RECORD_PATH", "records.json"))

VENUE_MAP_BUILTIN: dict[str, str] = {
    "桐生":"01","江戸川":"02","戸田":"03","平和島":"04","多摩川":"05",
    "浜名湖":"06","蒲郡":"07","常滑":"08","津":"09","三国":"10",
    "びわこ":"11","住之江":"12","尼崎":"13","鳴門":"14","丸亀":"15",
    "児島":"16","宮島":"17","徳山":"18","下関":"19","若松":"20",
    "芦屋":"21","福岡":"22","唐津":"23","大村":"24",
}
COURSE_BONUS_DEFAULT = {1:1.5, 2:0.8, 3:0.4, 4:0.2, 5:-0.2, 6:-0.5}


# ════════════════════════════════════════════════════════════
# 記録ファイルの読み書き
# ════════════════════════════════════════════════════════════

def load_records() -> list[dict]:
    if not RECORD_FILE.exists():
        return []
    try:
        with open(RECORD_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_records(records: list[dict]) -> None:
    with open(RECORD_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def calc_stats(records: list[dict]) -> dict:
    """記録リストから統計を計算"""
    if not records:
        return {
            "total_races": 0, "total_bet": 0, "total_return": 0,
            "roi": 0.0, "hit_count": 0, "hit_rate": 0.0,
            "by_venue": {}, "by_month": {},
            "streak_win": 0, "streak_lose": 0,
        }

    total_bet    = sum(r.get("bet_amount", 0) for r in records)
    total_return = sum(r.get("return_amount", 0) for r in records)
    hit_count    = sum(1 for r in records if r.get("hit", False))
    n            = len(records)
    roi          = (total_return / total_bet * 100) if total_bet > 0 else 0.0

    # 場別集計
    by_venue: dict[str, dict] = {}
    for r in records:
        vn = r.get("venue_name", "不明")
        if vn not in by_venue:
            by_venue[vn] = {"races": 0, "bet": 0, "return": 0, "hits": 0}
        by_venue[vn]["races"]  += 1
        by_venue[vn]["bet"]    += r.get("bet_amount", 0)
        by_venue[vn]["return"] += r.get("return_amount", 0)
        by_venue[vn]["hits"]   += 1 if r.get("hit") else 0
    for vn in by_venue:
        b = by_venue[vn]["bet"]
        by_venue[vn]["roi"] = round(by_venue[vn]["return"] / b * 100, 1) if b > 0 else 0.0

    # 月別集計
    by_month: dict[str, dict] = {}
    for r in records:
        m = r.get("race_date", "")[:6]  # YYYYMM
        if not m:
            continue
        if m not in by_month:
            by_month[m] = {"races": 0, "bet": 0, "return": 0, "hits": 0}
        by_month[m]["races"]  += 1
        by_month[m]["bet"]    += r.get("bet_amount", 0)
        by_month[m]["return"] += r.get("return_amount", 0)
        by_month[m]["hits"]   += 1 if r.get("hit") else 0
    for m in by_month:
        b = by_month[m]["bet"]
        by_month[m]["roi"] = round(by_month[m]["return"] / b * 100, 1) if b > 0 else 0.0

    # 連勝・連敗ストリーク（最新から）
    streak_win = streak_lose = 0
    sorted_r = sorted(records, key=lambda x: x.get("recorded_at",""), reverse=True)
    for r in sorted_r:
        if r.get("hit"):
            if streak_lose == 0:
                streak_win += 1
            else:
                break
        else:
            if streak_win == 0:
                streak_lose += 1
            else:
                break

    return {
        "total_races":  n,
        "total_bet":    total_bet,
        "total_return": total_return,
        "roi":          round(roi, 1),
        "hit_count":    hit_count,
        "hit_rate":     round(hit_count / n * 100, 1) if n > 0 else 0.0,
        "profit":       total_return - total_bet,
        "by_venue":     by_venue,
        "by_month":     by_month,
        "streak_win":   streak_win,
        "streak_lose":  streak_lose,
    }


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
        return jsonify({"error": "データ取得失敗"}), 404
    missing_ex = [b["lane"] for b in boats if b.get("ex_time") is None]
    if missing_ex:
        return jsonify({
            "error": "展示タイムのみ手入力してください",
            "boats": boats, "missing_ex_time": missing_ex, "need_ex_time": True,
        }), 202
    return jsonify({"date": race_date, "venue": venue_name, "race_no": race_no, "boats": boats})


# ════════════════════════════════════════════════════════════
# 結果記録
# POST /record
# {
#   "race_date":     "20260418",
#   "venue_name":    "戸田",
#   "race_no":       3,
#   "combo":         "1-2-3",
#   "bet_amount":    1000,
#   "hit":           true,
#   "return_amount": 3590,   ← 的中なら払戻×(bet/100)、外れなら0
#   "odds":          35.9,
#   "ev":            1.23,
#   "memo":          "1号艇強め"
# }
# ════════════════════════════════════════════════════════════

@app.route("/record", methods=["POST"])
def add_record():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "JSONパース失敗"}), 400

    required = {"race_date", "venue_name", "race_no", "combo", "bet_amount", "hit"}
    missing  = required - set(data.keys())
    if missing:
        return jsonify({"error": f"必須フィールドが不足: {missing}"}), 400

    records = load_records()

    # IDは記録数+1（重複防止にタイムスタンプも使用）
    new_id = f"{len(records)+1:04d}_{datetime.now().strftime('%Y%m%d%H%M%S')}"

    hit           = bool(data["hit"])
    bet_amount    = int(data.get("bet_amount", 0))
    return_amount = int(data.get("return_amount", 0)) if hit else 0

    record = {
        "id":            new_id,
        "race_date":     str(data["race_date"]),
        "venue_name":    str(data["venue_name"]),
        "race_no":       int(data["race_no"]),
        "combo":         str(data["combo"]),
        "bet_amount":    bet_amount,
        "hit":           hit,
        "return_amount": return_amount,
        "profit":        return_amount - bet_amount,
        "odds":          float(data.get("odds", 0)),
        "ev":            float(data.get("ev", 0)),
        "memo":          str(data.get("memo", "")),
        "recorded_at":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    records.append(record)
    save_records(records)

    return jsonify({"status": "ok", "record": record, "total": len(records)})


# ════════════════════════════════════════════════════════════
# 記録一覧
# GET /history?limit=50&venue=戸田&date=20260418
# ════════════════════════════════════════════════════════════

@app.route("/history")
def history():
    records = load_records()

    # フィルタ
    venue = request.args.get("venue", "")
    dt    = request.args.get("date",  "")
    limit = int(request.args.get("limit", 100))

    if venue:
        records = [r for r in records if r.get("venue_name") == venue]
    if dt:
        records = [r for r in records if r.get("race_date", "").startswith(dt)]

    # 新しい順
    records = sorted(records, key=lambda x: x.get("recorded_at",""), reverse=True)[:limit]

    return jsonify({"records": records, "count": len(records)})


# ════════════════════════════════════════════════════════════
# 統計
# GET /stats
# ════════════════════════════════════════════════════════════

@app.route("/stats")
def stats():
    records = load_records()
    return jsonify(calc_stats(records))


# ════════════════════════════════════════════════════════════
# 記録削除
# DELETE /record/<id>
# ════════════════════════════════════════════════════════════

@app.route("/record/<record_id>", methods=["DELETE"])
def delete_record(record_id: str):
    records = load_records()
    before  = len(records)
    records = [r for r in records if r.get("id") != record_id]
    if len(records) == before:
        return jsonify({"error": "該当IDが見つかりません"}), 404
    save_records(records)
    return jsonify({"status": "ok", "deleted": record_id, "remaining": len(records)})


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
    odds_map   = data.get("odds", {})

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

    real_odds_count = len(odds_map)
    if real_odds_count >= 60:
        odds_source = "実オッズ入力"
    elif real_odds_count > 0:
        odds_source = f"実オッズ一部入力({real_odds_count}点)"
    else:
        odds_source = "暫定推定"

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

        if combo in odds_map and float(odds_map[combo]) > 0:
            real_odds = float(odds_map[combo])
            true_ev   = prob * real_odds
        else:
            real_odds = (1.0 / max(prob, 1e-9)) * 0.75
            true_ev   = prob * real_odds * (1.0 + (1.0 / lanes[0]) * 0.15)

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
        "status":          "ok",
        "venue_name":      venue_name,
        "venue_code":      vc,
        "race_no":         race_no,
        "race_date":       race_date,
        "odds_source":     odds_source,
        "real_odds_count": real_odds_count,
        "buy":             [p for p in all_patterns if p["ev"] >= 1.2][:10],
        "all":             all_patterns,
        "bankroll":        bankroll,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)