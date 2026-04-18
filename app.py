"""
app.py  ── 競艇予想API (Flask / Render + Turso HTTP API対応)

【⑤場別設定 完全反映】
  config.py の以下をすべて predict に組み込み済み:
    - コース補正（1〜6コース）
    - EV_MIN     … 場ごとに買い推奨の閾値が変わる（江戸川1.60など）
    - P_MIN      … 最低勝率確率フィルタ
    - Kelly_kf   … Kelly係数（場ごとに保守度が変わる）
    - Kelly_max  … ベット上限（資金に対する割合）
    - 荒れ閾値   … 荒れにくい場はスコア分散が小さくても買い推奨
    - ナイター補正… ナイター開催場は1コースSTが有利
    - モーター重み… 浜名湖など水面次第でモーター差が大きい場
    - 当地重み   … 地元選手の当地勝率を強調する場
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import sqlite3
import requests as http_requests
from datetime import date, datetime
from pathlib import Path

try:
    from boat_api import fetch_race, fetch_available_races
    HAS_BOAT_API = True
except ImportError:
    HAS_BOAT_API = False

try:
    from config import get_venue_config, VENUE_NAME_MAP, NIGHT_RACE_VENUES
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False

app = Flask(__name__)
CORS(app)

# ════════════════════════════════════════════════════════════
# DB設定
# ════════════════════════════════════════════════════════════
_TURSO_URL_RAW = os.environ.get("TURSO_URL",   "")
TURSO_TOKEN    = os.environ.get("TURSO_TOKEN", "")
LOCAL_DB       = Path(os.environ.get("DB_PATH", "records.db"))
LEGACY_JSON    = Path(os.environ.get("RECORD_PATH", "records.json"))

TURSO_URL = _TURSO_URL_RAW.replace("libsql://", "https://", 1) if _TURSO_URL_RAW else ""
USE_TURSO = bool(TURSO_URL and TURSO_TOKEN)
TURSO_API = f"{TURSO_URL}/v2/pipeline" if USE_TURSO else ""

if USE_TURSO:
    print(f"[DB] Turso HTTP API使用: {TURSO_URL}")
else:
    print(f"[DB] ローカルSQLite使用: {LOCAL_DB}")

# ════════════════════════════════════════════════════════════
# 場別設定（config.py がない場合のデフォルト）
# ════════════════════════════════════════════════════════════
VENUE_MAP_BUILTIN: dict[str, str] = {
    "桐生":"01","江戸川":"02","戸田":"03","平和島":"04","多摩川":"05",
    "浜名湖":"06","蒲郡":"07","常滑":"08","津":"09","三国":"10",
    "びわこ":"11","住之江":"12","尼崎":"13","鳴門":"14","丸亀":"15",
    "児島":"16","宮島":"17","徳山":"18","下関":"19","若松":"20",
    "芦屋":"21","福岡":"22","唐津":"23","大村":"24",
}

# config.py がない環境用のデフォルト設定
DEFAULT_CFG = {
    "EV_MIN": 1.50, "P_MIN": 0.24,
    "Kelly_kf": 0.50, "Kelly_max": 0.10,
    "荒れ閾値": 0.50, "ナイター補正": 1.00,
    "モーター重み": 1.50, "当地重み": 1.20,
    "1コース補正":1.30,"2コース補正":1.05,"3コース補正":1.00,
    "4コース補正":0.90,"5コース補正":0.85,"6コース補正":0.75,
}
NIGHT_VENUES_BUILTIN = {"04","06","12","17","20","21","22","23","24"}

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
# 場別設定取得
# ════════════════════════════════════════════════════════════

def get_cfg(venue_code: str) -> dict:
    """config.py があればそちらを使い、なければデフォルト値"""
    if HAS_CONFIG:
        return get_venue_config(venue_code)
    return DEFAULT_CFG.copy()


def is_night(venue_code: str, race_no: int) -> bool:
    nights = NIGHT_RACE_VENUES if HAS_CONFIG else NIGHT_VENUES_BUILTIN
    return (str(venue_code).zfill(2) in nights) and (race_no >= 7)


def get_course_bonus(cfg: dict, venue_code: str, race_no: int) -> dict[int, float]:
    """
    コース補正にナイター補正を乗算。
    ナイター×1コース → イン逃げがさらに有利になる。
    """
    night_adj = cfg.get("ナイター補正", 1.00)
    if is_night(venue_code, race_no) and night_adj != 1.00:
        # ナイターは1コースをさらに補正
        bonus = {
            c: cfg.get(f"{c}コース補正", DEFAULT_CFG[f"{c}コース補正"])
            for c in range(1, 7)
        }
        bonus[1] = bonus[1] * night_adj
        return bonus
    return {
        c: cfg.get(f"{c}コース補正", DEFAULT_CFG[f"{c}コース補正"])
        for c in range(1, 7)
    }


# ════════════════════════════════════════════════════════════
# Turso HTTP API ラッパー
# ════════════════════════════════════════════════════════════

def _turso_execute(sql: str, params: list = []) -> dict:
    payload = {
        "requests": [
            {
                "type": "execute",
                "stmt": {
                    "sql": sql,
                    "args": [
                        {"type": _turso_type(p), "value": str(p) if p is not None else None}
                        for p in params
                    ],
                }
            },
            {"type": "close"}
        ]
    }
    resp = http_requests.post(
        TURSO_API,
        headers={"Authorization": f"Bearer {TURSO_TOKEN}", "Content-Type": "application/json"},
        json=payload, timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["results"][0]


def _turso_type(val) -> str:
    if val is None:            return "null"
    if isinstance(val, bool):  return "integer"
    if isinstance(val, int):   return "integer"
    if isinstance(val, float): return "float"
    return "text"


def _turso_rows_to_dicts(result: dict) -> list[dict]:
    try:
        cols = [c["name"] for c in result["response"]["result"]["cols"]]
        rows = result["response"]["result"]["rows"]
        out  = []
        for row in rows:
            d = {}
            for col, cell in zip(cols, row):
                v = cell.get("value"); t = cell.get("type","text")
                if t=="integer" and v is not None: d[col] = int(v)
                elif t=="float" and v is not None: d[col] = float(v)
                else: d[col] = v
            out.append(d)
        return out
    except Exception:
        return []


def _turso_affected(result: dict) -> int:
    try:
        return result["response"]["result"]["affected_row_count"]
    except Exception:
        return 0


# ════════════════════════════════════════════════════════════
# DB操作
# ════════════════════════════════════════════════════════════

def db_execute(sql: str, params: tuple = (), fetch: str = "none"):
    if USE_TURSO:
        result = _turso_execute(sql, list(params))
        if fetch == "all":   return _turso_rows_to_dicts(result)
        elif fetch == "one":
            rows = _turso_rows_to_dicts(result)
            return rows[0] if rows else None
        else:                return _turso_affected(result)
    else:
        LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(LOCAL_DB)
        con.row_factory = sqlite3.Row
        try:
            cur = con.execute(sql, params); con.commit()
            if fetch == "all":  return [dict(r) for r in cur.fetchall()]
            elif fetch == "one":
                r = cur.fetchone()
                return dict(r) if r else None
            else: return cur.rowcount
        finally:
            con.close()


def init_db() -> None:
    db_execute(CREATE_TABLE_SQL)
    try:
        db_execute("CREATE INDEX IF NOT EXISTS idx_race_date  ON records(race_date)")
        db_execute("CREATE INDEX IF NOT EXISTS idx_venue_name ON records(venue_name)")
    except Exception:
        pass
    print("[DB] 初期化完了")


# ════════════════════════════════════════════════════════════
# 統計
# ════════════════════════════════════════════════════════════

def calc_stats() -> dict:
    rows = db_execute("SELECT * FROM records", fetch="all") or []
    if not rows:
        return {"total_races":0,"total_bet":0,"total_return":0,
                "roi":0.0,"hit_count":0,"hit_rate":0.0,"profit":0,
                "by_venue":{},"by_month":{},"streak_win":0,"streak_lose":0}

    n=len(rows); total_bet=sum(r.get("bet_amount",0) for r in rows)
    total_return=sum(r.get("return_amount",0) for r in rows)
    hit_count=sum(1 for r in rows if r.get("hit"))
    profit=total_return-total_bet
    roi=round(total_return/total_bet*100,1) if total_bet>0 else 0.0

    by_venue: dict = {}
    for r in rows:
        vn=r.get("venue_name","不明")
        if vn not in by_venue: by_venue[vn]={"races":0,"bet":0,"return":0,"hits":0}
        by_venue[vn]["races"]+=1; by_venue[vn]["bet"]+=r.get("bet_amount",0)
        by_venue[vn]["return"]+=r.get("return_amount",0)
        by_venue[vn]["hits"]+=1 if r.get("hit") else 0
    for vn in by_venue:
        b=by_venue[vn]["bet"]
        by_venue[vn]["roi"]=round(by_venue[vn]["return"]/b*100,1) if b>0 else 0.0

    by_month: dict = {}
    for r in rows:
        m=str(r.get("race_date",""))[:6]
        if not m: continue
        if m not in by_month: by_month[m]={"races":0,"bet":0,"return":0,"hits":0}
        by_month[m]["races"]+=1; by_month[m]["bet"]+=r.get("bet_amount",0)
        by_month[m]["return"]+=r.get("return_amount",0)
        by_month[m]["hits"]+=1 if r.get("hit") else 0
    for m in by_month:
        b=by_month[m]["bet"]
        by_month[m]["roi"]=round(by_month[m]["return"]/b*100,1) if b>0 else 0.0

    sorted_rows=sorted(rows, key=lambda x: x.get("recorded_at",""), reverse=True)
    streak_win=streak_lose=0
    for r in sorted_rows[:20]:
        if r.get("hit"):
            if streak_lose==0: streak_win+=1
            else: break
        else:
            if streak_win==0: streak_lose+=1
            else: break

    return {"total_races":n,"total_bet":total_bet,"total_return":total_return,
            "roi":roi,"hit_count":hit_count,
            "hit_rate":round(hit_count/n*100,1) if n>0 else 0.0,
            "profit":profit,"by_venue":by_venue,"by_month":by_month,
            "streak_win":streak_win,"streak_lose":streak_lose}


# ════════════════════════════════════════════════════════════
# ルート
# ════════════════════════════════════════════════════════════

@app.route("/")
def home():
    try:
        with open("index.html", encoding="utf-8") as f: return f.read()
    except FileNotFoundError:
        return "<h2>index.html が見つかりません</h2>", 404


@app.route("/health")
def health():
    return jsonify({"status":"ok",
                    "db_mode":"turso" if USE_TURSO else "local_sqlite",
                    "db":TURSO_URL if USE_TURSO else str(LOCAL_DB)})


@app.route("/races")
def races():
    if not HAS_BOAT_API: return jsonify({"error":"boat_api が読み込めません"}), 500
    race_date = request.args.get("date", date.today().strftime("%Y%m%d"))
    result    = fetch_available_races(race_date)
    if not result: return jsonify({"error":f"{race_date} のレースデータが取得できません"}), 404
    return jsonify({"date":race_date,"races":result})


@app.route("/boats")
def boats_endpoint():
    if not HAS_BOAT_API: return jsonify({"error":"boat_api が読み込めません"}), 500
    race_date  = request.args.get("date",  date.today().strftime("%Y%m%d"))
    venue_name = request.args.get("venue", "")
    race_no    = int(request.args.get("race", 1))
    boats = fetch_race(race_date, venue_name, race_no)
    if boats is None: return jsonify({"error":"データ取得失敗"}), 404
    missing_ex = [b["lane"] for b in boats if b.get("ex_time") is None]
    if missing_ex:
        return jsonify({"error":"展示タイムのみ手入力してください",
                        "boats":boats,"missing_ex_time":missing_ex,"need_ex_time":True}), 202
    return jsonify({"date":race_date,"venue":venue_name,"race_no":race_no,"boats":boats})


@app.route("/record", methods=["POST"])
def add_record():
    try: data = request.get_json(force=True)
    except Exception: return jsonify({"error":"JSONパース失敗"}), 400
    required = {"race_date","venue_name","race_no","combo","bet_amount","hit"}
    missing  = required - set(data.keys())
    if missing: return jsonify({"error":f"必須フィールドが不足: {missing}"}), 400
    hit=bool(data["hit"]); bet_amount=int(data.get("bet_amount",0))
    return_amount=int(data.get("return_amount",0)) if hit else 0
    profit=return_amount-bet_amount
    recorded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_id=datetime.now().strftime("%Y%m%d%H%M%S%f")
    db_execute("""INSERT INTO records
          (id,race_date,venue_name,race_no,combo,bet_amount,hit,return_amount,profit,odds,ev,memo,recorded_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (new_id,str(data["race_date"]),str(data["venue_name"]),int(data["race_no"]),str(data["combo"]),
         bet_amount,1 if hit else 0,return_amount,profit,
         float(data.get("odds",0)),float(data.get("ev",0)),str(data.get("memo","")),recorded_at))
    total_row=db_execute("SELECT COUNT(*) AS cnt FROM records",fetch="one")
    return jsonify({"status":"ok","record":{"id":new_id,"race_date":str(data["race_date"]),
        "venue_name":str(data["venue_name"]),"race_no":int(data["race_no"]),"combo":str(data["combo"]),
        "bet_amount":bet_amount,"hit":hit,"return_amount":return_amount,"profit":profit,
        "recorded_at":recorded_at},"total":total_row["cnt"] if total_row else 0})


@app.route("/history")
def history():
    venue=request.args.get("venue",""); dt=request.args.get("date","")
    limit=int(request.args.get("limit",100))
    where=[]; params: list=[]
    if venue: where.append("venue_name = ?"); params.append(venue)
    if dt:    where.append("race_date LIKE ?"); params.append(dt+"%")
    where_sql=("WHERE "+" AND ".join(where)) if where else ""
    params.append(limit)
    rows=db_execute(f"SELECT * FROM records {where_sql} ORDER BY recorded_at DESC LIMIT ?",
                    tuple(params),fetch="all") or []
    for r in rows: r["hit"]=bool(r.get("hit"))
    return jsonify({"records":rows,"count":len(rows)})


@app.route("/stats")
def stats():
    return jsonify(calc_stats())


@app.route("/record/<record_id>", methods=["DELETE"])
def delete_record(record_id: str):
    affected=db_execute("DELETE FROM records WHERE id = ?",(record_id,))
    if not affected: return jsonify({"error":"該当IDが見つかりません"}), 404
    total_row=db_execute("SELECT COUNT(*) AS cnt FROM records",fetch="one")
    return jsonify({"status":"ok","deleted":record_id,"remaining":total_row["cnt"] if total_row else 0})


@app.route("/migrate", methods=["POST"])
def migrate_from_json():
    if not LEGACY_JSON.exists(): return jsonify({"message":"records.json が存在しないためスキップ"}), 200
    try:
        with open(LEGACY_JSON,encoding="utf-8") as f: old_records=json.load(f)
    except Exception as e: return jsonify({"error":f"読み込み失敗: {e}"}), 500
    inserted=skipped=0
    for r in old_records:
        try:
            db_execute("""INSERT OR IGNORE INTO records
                (id,race_date,venue_name,race_no,combo,bet_amount,hit,return_amount,profit,odds,ev,memo,recorded_at)
              VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (r.get("id",datetime.now().strftime("%Y%m%d%H%M%S%f")),
                 r.get("race_date",""),r.get("venue_name",""),int(r.get("race_no",0)),r.get("combo",""),
                 int(r.get("bet_amount",0)),1 if r.get("hit") else 0,
                 int(r.get("return_amount",0)),int(r.get("profit",0)),
                 float(r.get("odds",0)),float(r.get("ev",0)),r.get("memo",""),
                 r.get("recorded_at",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))))
            inserted+=1
        except Exception: skipped+=1
    backup=LEGACY_JSON.with_suffix(".json.bak"); LEGACY_JSON.rename(backup)
    return jsonify({"status":"ok","inserted":inserted,"skipped":skipped,
                    "message":f"{inserted}件を移行しました。"})


# ════════════════════════════════════════════════════════════
# ヘルパー
# ════════════════════════════════════════════════════════════

def kelly_bet(prob: float, odds: float, bankroll: float, cfg: dict) -> int:
    """
    場別 Kelly_kf・Kelly_max・P_MIN を反映したベット額計算。
    Kelly_kf  … Kelly係数（小さいほど保守的、江戸川は0.45）
    Kelly_max … 1レースのベット上限（資金に対する割合）
    P_MIN     … 最低勝率確率フィルタ（これ未満なら0円）
    """
    p_min    = cfg.get("P_MIN",    0.24)
    kf       = cfg.get("Kelly_kf", 0.50)
    max_frac = cfg.get("Kelly_max",0.10)

    if prob < p_min: return 0

    b = odds - 1.0
    if b <= 0: return 0
    kelly_full = (b * prob - (1.0 - prob)) / b
    if kelly_full <= 0: return 0

    raw_bet = bankroll * kelly_full * kf
    capped  = min(raw_bet, bankroll * max_frac)
    return (int(capped) // 100) * 100


def calc_combo_score(boats, lane1, lane2, lane3, course_bonus, cfg: dict) -> float:
    """
    場別モーター重み・当地重みを反映したスコア計算。
    モーター重み … 浜名湖など水面でモーター差が大きい場で重み増
    当地重み     … 地元選手の当地勝率を強調する場（江戸川1.80など）
    """
    motor_w = cfg.get("モーター重み", 1.50)
    local_w = cfg.get("当地重み",    1.20)

    score = 0.0
    for b in boats:
        cb        = course_bonus.get(b["lane"], 1.0)
        motor_val = b["motor"]    * (motor_w / 1.50)  # デフォルト比で正規化
        local_val = b.get("local_win", 0) * (local_w / 1.20)

        if b["lane"] == lane1:
            score += b["win_rate"] * 3.0 * cb
            score += motor_val * 2.0
            score += local_val * 0.5          # ★当地勝率を1着スコアに加算
            score -= b["ex_time"] * 10.0
            score -= b["start"]   * 5.0
        elif b["lane"] == lane2:
            score += b["win_rate"] * 2.0
            score += motor_val * 1.5
        elif b["lane"] == lane3:
            score += b["win_rate"] * 1.0
            score += motor_val * 1.0
    return score


def resolve_venue_code(venue_name: str, venue_code: str) -> str:
    if venue_code and str(venue_code).isdigit():
        return str(venue_code).zfill(2)
    if HAS_CONFIG and venue_name in VENUE_NAME_MAP:
        return VENUE_NAME_MAP[venue_name]
    return VENUE_MAP_BUILTIN.get(venue_name, "01")


# ════════════════════════════════════════════════════════════
# 予想API（場別設定フル反映版）
# ════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error":"JSONのパースに失敗しました"}), 400

    boats      = data.get("boats", [])
    bankroll   = float(data.get("bankroll", 10000))
    venue_name = data.get("venue_name", "")
    venue_code = data.get("venue_code", "")
    race_no    = int(data.get("race_no", 1))
    race_date  = data.get("race_date", date.today().strftime("%Y%m%d"))
    odds_map   = data.get("odds", {})

    if len(boats) != 6:
        return jsonify({"error":"boats は6艇必須です"}), 400

    required_keys = {"lane","ex_time","motor","win_rate","start"}
    for i, b in enumerate(boats):
        for k in required_keys - set(b.keys()):
            return jsonify({"error":f"boats[{i}] にキーが不足: {k}"}), 400
        for k in required_keys:
            try:
                if b[k] is None: raise ValueError
                float(b[k])
            except (TypeError, ValueError):
                return jsonify({"error":f"boats[{i}][{k}] が数値ではありません"}), 400

    # ── 場別設定を取得 ──────────────────────────────────────
    vc           = resolve_venue_code(venue_name, venue_code)
    cfg          = get_cfg(vc)
    course_bonus = get_course_bonus(cfg, vc, race_no)

    # ── 場別パラメータ ──────────────────────────────────────
    ev_min   = cfg.get("EV_MIN",   1.50)   # 場ごとの買い推奨閾値
    is_night_race = is_night(vc, race_no)

    real_odds_count = len(odds_map)
    if real_odds_count >= 60:   odds_source = "実オッズ入力"
    elif real_odds_count > 0:   odds_source = f"実オッズ一部入力({real_odds_count}点)"
    else:                       odds_source = "暫定推定"

    # ── 全120通りのスコア計算 ────────────────────────────────
    raw_patterns = []
    for i in range(1,7):
        for j in range(1,7):
            for k in range(1,7):
                if i==j or j==k or i==k: continue
                raw_patterns.append({
                    "combo":f"{i}-{j}-{k}",
                    "score":calc_combo_score(boats,i,j,k,course_bonus,cfg),
                    "lanes":[i,j,k],
                })

    score_min = min(p["score"] for p in raw_patterns)
    score_adj = [p["score"]-score_min+1e-6 for p in raw_patterns]
    score_sum = sum(score_adj)

    all_patterns = []
    for p, adj in zip(raw_patterns, score_adj):
        combo=p["combo"]; lanes=p["lanes"]
        prob = adj / score_sum

        # オッズ
        if combo in odds_map and float(odds_map[combo]) > 0:
            real_odds = float(odds_map[combo])
            true_ev   = prob * real_odds
        else:
            real_odds = (1.0/max(prob,1e-9)) * 0.75
            true_ev   = prob * real_odds * (1.0 + (1.0/lanes[0]) * 0.15)

        # ★場別 Kelly でベット額計算
        bet = kelly_bet(prob, real_odds, bankroll, cfg)

        # ★場別 EV_MIN で判定
        if true_ev >= ev_min * 1.3:  verdict = "✅ 強く買い"
        elif true_ev >= ev_min:      verdict = "⚠️ 買い"
        else:                        verdict = "🚫 見送り"

        all_patterns.append({
            "combo":combo,"prob":round(prob,5),"odds":round(real_odds,2),
            "ev":round(true_ev,3),"bet":bet,"verdict":verdict,
            "odds_is_real":combo in odds_map,
        })

    all_patterns.sort(key=lambda x: x["ev"], reverse=True)

    # ★場別 EV_MIN で買い推奨フィルタ
    buy = [p for p in all_patterns if p["ev"] >= ev_min][:10]

    return jsonify({
        "status":"ok",
        "venue_name":venue_name,"venue_code":vc,
        "race_no":race_no,"race_date":race_date,
        "odds_source":odds_source,"real_odds_count":real_odds_count,
        # ★場別設定をレスポンスに含める（フロントで表示可能）
        "venue_config":{
            "ev_min":    ev_min,
            "p_min":     cfg.get("P_MIN",    0.24),
            "kelly_kf":  cfg.get("Kelly_kf", 0.50),
            "kelly_max": cfg.get("Kelly_max",0.10),
            "is_night":  is_night_race,
        },
        "buy":buy,"all":all_patterns,"bankroll":bankroll,
    })


# ════════════════════════════════════════════════════════════
# 起動
# ════════════════════════════════════════════════════════════

init_db()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)