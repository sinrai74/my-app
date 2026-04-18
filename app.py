"""
app.py  ── 競艇予想API (Flask / Render + Turso対応)

DB: Turso (libsql) → 環境変数 TURSO_URL / TURSO_TOKEN で接続
    未設定の場合はローカルSQLite (records.db) にフォールバック
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from contextlib import contextmanager

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

# ════════════════════════════════════════════════════════════
# DB接続設定
#   TURSO_URL / TURSO_TOKEN が設定されていれば Turso を使用
#   未設定ならローカル SQLite にフォールバック
# ════════════════════════════════════════════════════════════
TURSO_URL   = os.environ.get("TURSO_URL",   "")
TURSO_TOKEN = os.environ.get("TURSO_TOKEN", "")
USE_TURSO   = bool(TURSO_URL and TURSO_TOKEN)
LOCAL_DB    = Path(os.environ.get("DB_PATH", "records.db"))
LEGACY_JSON = Path(os.environ.get("RECORD_PATH", "records.json"))

if USE_TURSO:
    import libsql_client
    print(f"[DB] Turso使用: {TURSO_URL}")
else:
    print(f"[DB] ローカルSQLite使用: {LOCAL_DB}")

VENUE_MAP_BUILTIN: dict[str, str] = {
    "桐生":"01","江戸川":"02","戸田":"03","平和島":"04","多摩川":"05",
    "浜名湖":"06","蒲郡":"07","常滑":"08","津":"09","三国":"10",
    "びわこ":"11","住之江":"12","尼崎":"13","鳴門":"14","丸亀":"15",
    "児島":"16","宮島":"17","徳山":"18","下関":"19","若松":"20",
    "芦屋":"21","福岡":"22","唐津":"23","大村":"24",
}
COURSE_BONUS_DEFAULT = {1:1.5, 2:0.8, 3:0.4, 4:0.2, 5:-0.2, 6:-0.5}

CREATE_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS records (
        id            TEXT PRIMARY KEY,
        race_date     TEXT NOT NULL,
        venue_name    TEXT NOT NULL,
        race_no       INTEGER NOT NULL,
        combo         TEXT NOT NULL,
        bet_amount    INTEGER NOT NULL DEFAULT 0,
        hit           INTEGER NOT NULL DEFAULT 0,
        return_amount INTEGER NOT NULL DEFAULT 0,
        profit        INTEGER NOT NULL DEFAULT 0,
        odds          REAL    NOT NULL DEFAULT 0,
        ev            REAL    NOT NULL DEFAULT 0,
        memo          TEXT    NOT NULL DEFAULT '',
        recorded_at   TEXT    NOT NULL
    )
"""


# ════════════════════════════════════════════════════════════
# DB操作の共通関数
# ════════════════════════════════════════════════════════════

def db_execute(sql: str, params: tuple = (), fetch: str = "none"):
    """
    Turso / ローカルSQLite を透過的に扱う実行関数。
    fetch: "none" | "one" | "all"
    """
    if USE_TURSO:
        import libsql_client
        with libsql_client.create_client_sync(
            url=TURSO_URL, auth_token=TURSO_TOKEN
        ) as client:
            # パラメータをlibsql形式に変換
            result = client.execute(sql, list(params))
            if fetch == "all":
                # ResultSet → list[dict]
                cols = [c.name for c in result.columns] if result.columns else []
                return [dict(zip(cols, row)) for row in result.rows]
            elif fetch == "one":
                cols = [c.name for c in result.columns] if result.columns else []
                if result.rows:
                    return dict(zip(cols, result.rows[0]))
                return None
            else:
                return result.rows_affected
    else:
        # ローカルSQLite
        con = sqlite3.connect(LOCAL_DB)
        con.row_factory = sqlite3.Row
        try:
            cur = con.execute(sql, params)
            con.commit()
            if fetch == "all":
                return [dict(r) for r in cur.fetchall()]
            elif fetch == "one":
                r = cur.fetchone()
                return dict(r) if r else None
            else:
                return cur.rowcount
        finally:
            con.close()


def init_db() -> None:
    """テーブルが存在しなければ作成"""
    if not USE_TURSO:
        LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
    db_execute(CREATE_TABLE_SQL)
    if USE_TURSO:
        # インデックスは個別に実行
        try:
            db_execute("CREATE INDEX IF NOT EXISTS idx_race_date  ON records(race_date)")
            db_execute("CREATE INDEX IF NOT EXISTS idx_venue_name ON records(venue_name)")
        except Exception:
            pass
    else:
        db_execute("CREATE INDEX IF NOT EXISTS idx_race_date  ON records(race_date)")
        db_execute("CREATE INDEX IF NOT EXISTS idx_venue_name ON records(venue_name)")
    print("[DB] 初期化完了")


# ════════════════════════════════════════════════════════════
# 統計計算
# ════════════════════════════════════════════════════════════

def calc_stats() -> dict:
    rows = db_execute("SELECT * FROM records", fetch="all") or []

    if not rows:
        return {
            "total_races": 0, "total_bet": 0, "total_return": 0,
            "roi": 0.0, "hit_count": 0, "hit_rate": 0.0, "profit": 0,
            "by_venue": {}, "by_month": {}, "streak_win": 0, "streak_lose": 0,
        }

    n            = len(rows)
    total_bet    = sum(r.get("bet_amount", 0)    for r in rows)
    total_return = sum(r.get("return_amount", 0) for r in rows)
    hit_count    = sum(1 for r in rows if r.get("hit"))
    profit       = total_return - total_bet
    roi          = round(total_return / total_bet * 100, 1) if total_bet > 0 else 0.0
    hit_rate     = round(hit_count / n * 100, 1) if n > 0 else 0.0

    # 場別集計
    by_venue: dict = {}
    for r in rows:
        vn = r.get("venue_name", "不明")
        if vn not in by_venue:
            by_venue[vn] = {"races":0, "bet":0, "return":0, "hits":0}
        by_venue[vn]["races"]  += 1
        by_venue[vn]["bet"]    += r.get("bet_amount", 0)
        by_venue[vn]["return"] += r.get("return_amount", 0)
        by_venue[vn]["hits"]   += 1 if r.get("hit") else 0
    for vn in by_venue:
        b = by_venue[vn]["bet"]
        by_venue[vn]["roi"] = round(by_venue[vn]["return"] / b * 100, 1) if b > 0 else 0.0

    # 月別集計
    by_month: dict = {}
    for r in rows:
        m = str(r.get("race_date", ""))[:6]
        if not m: continue
        if m not in by_month:
            by_month[m] = {"races":0, "bet":0, "return":0, "hits":0}
        by_month[m]["races"]  += 1
        by_month[m]["bet"]    += r.get("bet_amount", 0)
        by_month[m]["return"] += r.get("return_amount", 0)
        by_month[m]["hits"]   += 1 if r.get("hit") else 0
    for m in by_month:
        b = by_month[m]["bet"]
        by_month[m]["roi"] = round(by_month[m]["return"] / b * 100, 1) if b > 0 else 0.0

    # 連勝・連敗
    sorted_rows  = sorted(rows, key=lambda x: x.get("recorded_at",""), reverse=True)
    streak_win   = streak_lose = 0
    for r in sorted_rows[:20]:
        if r.get("hit"):
            if streak_lose == 0: streak_win += 1
            else: break
        else:
            if streak_win == 0: streak_lose += 1
            else: break

    return {
        "total_races": n, "total_bet": total_bet, "total_return": total_return,
        "roi": roi, "hit_count": hit_count, "hit_rate": hit_rate, "profit": profit,
        "by_venue": by_venue, "by_month": by_month,
        "streak_win": streak_win, "streak_lose": streak_lose,
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
    return jsonify({
        "status": "ok",
        "db_mode": "turso" if USE_TURSO else "local_sqlite",
        "db": TURSO_URL if USE_TURSO else str(LOCAL_DB),
    })


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

    hit           = bool(data["hit"])
    bet_amount    = int(data.get("bet_amount", 0))
    return_amount = int(data.get("return_amount", 0)) if hit else 0
    profit        = return_amount - bet_amount
    recorded_at   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_id        = datetime.now().strftime("%Y%m%d%H%M%S%f")

    db_execute("""
        INSERT INTO records
          (id, race_date, venue_name, race_no, combo,
           bet_amount, hit, return_amount, profit,
           odds, ev, memo, recorded_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        new_id, str(data["race_date"]), str(data["venue_name"]),
        int(data["race_no"]), str(data["combo"]),
        bet_amount, 1 if hit else 0,
        return_amount, profit,
        float(data.get("odds", 0)), float(data.get("ev", 0)),
        str(data.get("memo", "")), recorded_at,
    ))

    total_row = db_execute("SELECT COUNT(*) AS cnt FROM records", fetch="one")
    total     = total_row["cnt"] if total_row else 0

    return jsonify({
        "status": "ok",
        "record": {
            "id": new_id, "race_date": str(data["race_date"]),
            "venue_name": str(data["venue_name"]), "race_no": int(data["race_no"]),
            "combo": str(data["combo"]), "bet_amount": bet_amount,
            "hit": hit, "return_amount": return_amount, "profit": profit,
            "recorded_at": recorded_at,
        },
        "total": total,
    })


# ════════════════════════════════════════════════════════════
# 記録一覧
# ════════════════════════════════════════════════════════════

@app.route("/history")
def history():
    venue = request.args.get("venue", "")
    dt    = request.args.get("date",  "")
    limit = int(request.args.get("limit", 100))

    where  = []
    params: list = []
    if venue: where.append("venue_name = ?"); params.append(venue)
    if dt:    where.append("race_date LIKE ?"); params.append(dt + "%")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    params.append(limit)

    rows = db_execute(
        f"SELECT * FROM records {where_sql} ORDER BY recorded_at DESC LIMIT ?",
        tuple(params), fetch="all"
    ) or []

    for r in rows:
        r["hit"] = bool(r.get("hit"))

    return jsonify({"records": rows, "count": len(rows)})


# ════════════════════════════════════════════════════════════
# 統計
# ════════════════════════════════════════════════════════════

@app.route("/stats")
def stats():
    return jsonify(calc_stats())


# ════════════════════════════════════════════════════════════
# 記録削除
# ════════════════════════════════════════════════════════════

@app.route("/record/<record_id>", methods=["DELETE"])
def delete_record(record_id: str):
    affected = db_execute(
        "DELETE FROM records WHERE id = ?", (record_id,)
    )
    if not affected:
        return jsonify({"error": "該当IDが見つかりません"}), 404
    total_row = db_execute("SELECT COUNT(*) AS cnt FROM records", fetch="one")
    return jsonify({"status": "ok", "deleted": record_id,
                    "remaining": total_row["cnt"] if total_row else 0})


# ════════════════════════════════════════════════════════════
# records.json → Turso/SQLite 移行
# POST /migrate
# ════════════════════════════════════════════════════════════

@app.route("/migrate", methods=["POST"])
def migrate_from_json():
    if not LEGACY_JSON.exists():
        return jsonify({"message": "records.json が存在しないためスキップ"}), 200
    try:
        with open(LEGACY_JSON, encoding="utf-8") as f:
            old_records: list[dict] = json.load(f)
    except Exception as e:
        return jsonify({"error": f"records.json 読み込み失敗: {e}"}), 500

    inserted = skipped = 0
    for r in old_records:
        try:
            db_execute("""
                INSERT OR IGNORE INTO records
                  (id, race_date, venue_name, race_no, combo,
                   bet_amount, hit, return_amount, profit,
                   odds, ev, memo, recorded_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                r.get("id", datetime.now().strftime("%Y%m%d%H%M%S%f")),
                r.get("race_date",""), r.get("venue_name",""),
                int(r.get("race_no",0)), r.get("combo",""),
                int(r.get("bet_amount",0)), 1 if r.get("hit") else 0,
                int(r.get("return_amount",0)), int(r.get("profit",0)),
                float(r.get("odds",0)), float(r.get("ev",0)),
                r.get("memo",""),
                r.get("recorded_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            ))
            inserted += 1
        except Exception:
            skipped += 1

    backup = LEGACY_JSON.with_suffix(".json.bak")
    LEGACY_JSON.rename(backup)

    return jsonify({
        "status": "ok", "inserted": inserted, "skipped": skipped,
        "message": f"{inserted}件を移行しました。元ファイルは {backup} にバックアップしました。",
    })


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
    if real_odds_count >= 60:   odds_source = "実オッズ入力"
    elif real_odds_count > 0:   odds_source = f"実オッズ一部入力({real_odds_count}点)"
    else:                       odds_source = "暫定推定"

    raw_patterns = []
    for i in range(1, 7):
        for j in range(1, 7):
            for k in range(1, 7):
                if i == j or j == k or i == k: continue
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
        combo = p["combo"]; lanes = p["lanes"]
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
            "combo": combo, "prob": round(prob, 5),
            "odds": round(real_odds, 2), "ev": round(true_ev, 3),
            "bet": bet, "verdict": verdict, "odds_is_real": combo in odds_map,
        })

    all_patterns.sort(key=lambda x: x["ev"], reverse=True)

    return jsonify({
        "status": "ok", "venue_name": venue_name, "venue_code": vc,
        "race_no": race_no, "race_date": race_date,
        "odds_source": odds_source, "real_odds_count": real_odds_count,
        "buy": [p for p in all_patterns if p["ev"] >= 1.2][:10],
        "all": all_patterns, "bankroll": bankroll,
    })


# ════════════════════════════════════════════════════════════
# 起動
# ════════════════════════════════════════════════════════════

init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)