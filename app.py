"""
app.py  ── 競艇予想API (Flask / Render対応)
高精度モデル（engine.py + model_all.pkl）をブラウザUIで使用。

エンドポイント:
  GET  /                   フロントエンド
  GET  /health             死活監視
  GET  /races?date=        開催レース一覧
  GET  /boats?date=&venue=&race=  6艇データ取得
  POST /predict_ml         高精度予想（engine.py使用）
  POST /record             結果記録
  GET  /history            記録一覧
  GET  /stats              回収率統計
  DELETE /record/<id>      記録削除
  POST /migrate            records.json → Turso移行
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, sqlite3, pickle, warnings
import requests as http_requests
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ── engine.py（競艇最新.txtから抽出した予想エンジン）──────────
try:
    import engine as E
    HAS_ENGINE = True
    print("[ENGINE] engine.py 読み込み成功")
except Exception as ex:
    HAS_ENGINE = False
    print(f"[ENGINE] engine.py 読み込み失敗: {ex}")

# ── BoatraceOpenAPI クライアント ────────────────────────────
try:
    from boat_api import fetch_race, fetch_available_races
    HAS_BOAT_API = True
except ImportError:
    HAS_BOAT_API = False

app = Flask(__name__)
CORS(app)

# ════════════════════════════════════════════════════════════
# model_all.pkl 読み込み（起動時に1回だけ）
# ════════════════════════════════════════════════════════════
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "model_all.pkl"))
_model_data   = None
_models       = None
_dynamic_ev   = {}
_feature_cols = []

def load_model():
    global _model_data, _models, _dynamic_ev, _feature_cols
    if not MODEL_PATH.exists():
        print(f"[MODEL] {MODEL_PATH} が見つかりません")
        return False
    try:
        with open(MODEL_PATH, "rb") as f:
            _model_data = pickle.load(f)
        _models       = _model_data["models"]
        _dynamic_ev   = _model_data.get("dynamic_ev_map", {})
        _feature_cols = _model_data.get("feature_cols", [])
        print(f"[MODEL] {MODEL_PATH} 読み込み完了 ({MODEL_PATH.stat().st_size//1024}KB)")
        return True
    except Exception as e:
        print(f"[MODEL] 読み込みエラー: {e}")
        return False

HAS_MODEL = load_model()

# ════════════════════════════════════════════════════════════
# DB設定（Turso HTTP API or ローカルSQLite）
# ════════════════════════════════════════════════════════════
_TURSO_URL_RAW = os.environ.get("TURSO_URL",   "")
TURSO_TOKEN    = os.environ.get("TURSO_TOKEN", "")
LOCAL_DB       = Path(os.environ.get("DB_PATH", "records.db"))
LEGACY_JSON    = Path(os.environ.get("RECORD_PATH", "records.json"))

TURSO_URL = _TURSO_URL_RAW.replace("libsql://", "https://", 1) if _TURSO_URL_RAW else ""
USE_TURSO = bool(TURSO_URL and TURSO_TOKEN)
TURSO_API = f"{TURSO_URL}/v2/pipeline" if USE_TURSO else ""

print(f"[DB] {'Turso: ' + TURSO_URL if USE_TURSO else 'SQLite: ' + str(LOCAL_DB)}")

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
)"""

VENUE_MAP: dict[str, str] = {
    "桐生":"01","江戸川":"02","戸田":"03","平和島":"04","多摩川":"05",
    "浜名湖":"06","蒲郡":"07","常滑":"08","津":"09","三国":"10",
    "びわこ":"11","住之江":"12","尼崎":"13","鳴門":"14","丸亀":"15",
    "児島":"16","宮島":"17","徳山":"18","下関":"19","若松":"20",
    "芦屋":"21","福岡":"22","唐津":"23","大村":"24",
}


# ════════════════════════════════════════════════════════════
# Turso / SQLite 共通DB操作
# ════════════════════════════════════════════════════════════

def _turso_type(val):
    if val is None:            return "null"
    if isinstance(val, bool):  return "integer"
    if isinstance(val, int):   return "integer"
    if isinstance(val, float): return "float"
    return "text"

def _turso_exec(sql, params=[]):
    payload = {"requests": [
        {"type":"execute","stmt":{"sql":sql,
         "args":[{"type":_turso_type(p),"value":str(p) if p is not None else None} for p in params]}},
        {"type":"close"}
    ]}
    r = http_requests.post(TURSO_API,
        headers={"Authorization":f"Bearer {TURSO_TOKEN}","Content-Type":"application/json"},
        json=payload, timeout=10)
    r.raise_for_status()
    return r.json()["results"][0]

def _turso_rows(result):
    try:
        cols = [c["name"] for c in result["response"]["result"]["cols"]]
        rows = result["response"]["result"]["rows"]
        out  = []
        for row in rows:
            d = {}
            for col, cell in zip(cols, row):
                v = cell.get("value"); t = cell.get("type","text")
                d[col] = int(v) if t=="integer" and v is not None else \
                         float(v) if t=="float" and v is not None else v
            out.append(d)
        return out
    except Exception:
        return []

def _turso_affected(result):
    try: return result["response"]["result"]["affected_row_count"]
    except Exception: return 0

def db_execute(sql, params=(), fetch="none"):
    if USE_TURSO:
        res = _turso_exec(sql, list(params))
        if fetch=="all":  return _turso_rows(res)
        if fetch=="one":  rows = _turso_rows(res); return rows[0] if rows else None
        return _turso_affected(res)
    else:
        LOCAL_DB.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(LOCAL_DB); con.row_factory = sqlite3.Row
        try:
            cur = con.execute(sql, params); con.commit()
            if fetch=="all":  return [dict(r) for r in cur.fetchall()]
            if fetch=="one":  r = cur.fetchone(); return dict(r) if r else None
            return cur.rowcount
        finally: con.close()

def init_db():
    db_execute(CREATE_TABLE_SQL)
    try:
        db_execute("CREATE INDEX IF NOT EXISTS idx_rd ON records(race_date)")
        db_execute("CREATE INDEX IF NOT EXISTS idx_vn ON records(venue_name)")
    except Exception: pass
    print("[DB] 初期化完了")

init_db()


# ════════════════════════════════════════════════════════════
# 統計計算
# ════════════════════════════════════════════════════════════

def calc_stats():
    rows = db_execute("SELECT * FROM records", fetch="all") or []
    if not rows:
        return {"total_races":0,"total_bet":0,"total_return":0,"roi":0.0,
                "hit_count":0,"hit_rate":0.0,"profit":0,"by_venue":{},"by_month":{},
                "streak_win":0,"streak_lose":0}
    n = len(rows)
    tb = sum(r.get("bet_amount",0) for r in rows)
    tr = sum(r.get("return_amount",0) for r in rows)
    hc = sum(1 for r in rows if r.get("hit"))
    by_venue: dict = {}
    for r in rows:
        vn = r.get("venue_name","不明")
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
    sr = sorted(rows,key=lambda x:x.get("recorded_at",""),reverse=True)
    sw=sl=0
    for r in sr[:20]:
        if r.get("hit"):
            if sl==0: sw+=1
            else: break
        else:
            if sw==0: sl+=1
            else: break
    return {"total_races":n,"total_bet":tb,"total_return":tr,
            "roi":round(tr/tb*100,1) if tb>0 else 0.0,"hit_count":hc,
            "hit_rate":round(hc/n*100,1) if n>0 else 0.0,"profit":tr-tb,
            "by_venue":by_venue,"by_month":by_month,"streak_win":sw,"streak_lose":sl}


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
    return jsonify({
        "status":    "ok",
        "model":     str(MODEL_PATH) if HAS_MODEL else "なし",
        "engine":    HAS_ENGINE,
        "db_mode":   "turso" if USE_TURSO else "sqlite",
    })

@app.route("/races")
def races():
    if not HAS_BOAT_API: return jsonify({"error":"boat_api なし"}), 500
    race_date = request.args.get("date", date.today().strftime("%Y%m%d"))
    result    = fetch_available_races(race_date)
    if not result: return jsonify({"error":"データなし"}), 404
    return jsonify({"date":race_date,"races":result})

@app.route("/boats")
def boats_endpoint():
    if not HAS_BOAT_API: return jsonify({"error":"boat_api なし"}), 500
    race_date  = request.args.get("date",  date.today().strftime("%Y%m%d"))
    venue_name = request.args.get("venue", "")
    race_no    = int(request.args.get("race", 1))
    boats = fetch_race(race_date, venue_name, race_no)
    if boats is None: return jsonify({"error":"取得失敗"}), 404
    missing = [b["lane"] for b in boats if b.get("ex_time") is None]
    if missing:
        return jsonify({"error":"展示タイムを入力してください","boats":boats,
                        "missing_ex_time":missing,"need_ex_time":True}), 202
    return jsonify({"date":race_date,"venue":venue_name,"race_no":race_no,"boats":boats})


# ════════════════════════════════════════════════════════════
# 高精度予想API（engine.py + model_all.pkl）
# POST /predict_ml
# {
#   "race_date": "20260419",
#   "venue_name": "戸田",
#   "race_no": 3,
#   "boats": [         ← 6艇の手入力データ（自動取得失敗時のフォールバック）
#     {"lane":1,"ex_time":6.72,"motor":35.2,"win_rate":6.5,"start":0.15,"local_win":3.2},
#     ...
#   ],
#   "bankroll": 10000,
#   "wind_speed": 0.0,
#   "wind_direction": "",
#   "tide_level": ""
# }
# ════════════════════════════════════════════════════════════

@app.route("/predict_ml", methods=["POST"])
def predict_ml():
    if not HAS_ENGINE:
        return jsonify({"error": "engine.py が読み込めません"}), 500
    if not HAS_MODEL:
        return jsonify({"error": f"model_all.pkl が見つかりません（{MODEL_PATH}）"}), 500

    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "JSONパース失敗"}), 400

    race_date      = data.get("race_date",  date.today().strftime("%Y%m%d"))
    venue_name     = data.get("venue_name", "")
    race_no        = int(data.get("race_no", 1))
    bankroll       = float(data.get("bankroll", 10000))
    wind_speed     = float(data.get("wind_speed", 0.0))
    wind_direction = str(data.get("wind_direction", ""))
    tide_level     = str(data.get("tide_level", ""))
    boats_input    = data.get("boats", [])  # 手動入力データ（任意）

    venue_code = VENUE_MAP.get(venue_name, "01")
    cfg        = E.get_venue_config(venue_code)

    # ── 番組表データを作成 ────────────────────────────────
    # boats_input がある場合はそれをDataFrameに変換
    # ない場合は BoatraceOpenAPIから取得（HAS_BOAT_API時）
    try:
        if boats_input and len(boats_input) == 6:
            # 手動入力データからDataFrameを構築
            recs = []
            for b in boats_input:
                recs.append({
                    "場コード": venue_code,
                    "場名":     venue_name,
                    "レースNo": race_no,
                    "開催日":   f"{race_date[:4]}-{race_date[4:6]}-{race_date[6:]}",
                    "節日":     1,
                    "艇番":     int(b["lane"]),
                    "登番":     str(b.get("player_no", "0000")),
                    "選手名":   str(b.get("name", f"{b['lane']}号艇")),
                    "年齢":     int(b.get("age", 30)),
                    "支部":     str(b.get("branch", "")),
                    "体重":     float(b.get("weight", 52)),
                    "級":       str(b.get("grade", "B1")),
                    "全国勝率": float(b.get("win_rate", 5.0)),
                    "全国2率":  float(b.get("win2_rate", 35.0)),
                    "当地勝率": float(b.get("local_win", 0.0)),
                    "当地2率":  float(b.get("local_win2", 0.0)),
                    "モーターNO":  int(b.get("motor_no", 1)),
                    "モーター2率": float(b.get("motor", 35.0)),
                    "ボートNO":   int(b.get("boat_no", 1)),
                    "ボート2率":  float(b.get("boat2_rate", 35.0)),
                    "今節成績":   str(b.get("today_results", "")),
                    # fan由来の列（なければデフォルト）
                    "勝率":         float(b.get("win_rate", 5.0)),
                    "複勝率":       float(b.get("win2_rate", 35.0)),
                    "平均ST":       float(b.get("start", 0.18)),
                    "今期能力指数": float(b.get("ability", 50.0)),
                    "前期能力指数": float(b.get("ability_prev", 50.0)),
                    "出走回数":     int(b.get("race_count", 100)),
                    "優出回数":     int(b.get("final_count", 5)),
                    "優勝回数":     int(b.get("win_count", 2)),
                    **{f"{c}コース進入回数": int(b.get(f"course_{c}_count", 10)) for c in range(1,7)},
                    **{f"{c}コース複勝率":   float(b.get(f"course_{c}_win2", 35.0)) for c in range(1,7)},
                    **{f"{c}コース平均ST":   float(b.get(f"course_{c}_st", 0.18))   for c in range(1,7)},
                })
            df_today = pd.DataFrame(recs)
        else:
            return jsonify({"error": "boats（6艇）が必要です"}), 400

        # 人気・払戻は予測時点では不明
        for col, dt in [("1着フラグ","Int64"),("3着内フラグ","Int64"),("着順","Int64")]:
            df_today[col] = pd.array([None]*len(df_today), dtype=dt)
        df_today["返還フラグ"] = 0
        df_today["人気"]       = np.nan
        df_today["払戻"]       = np.nan

        # 特徴量生成
        df_feat = E.engineer_features(
            df_today,
            wind_speed=wind_speed,
            wind_direction=wind_direction,
            tide_level=tide_level,
            dynamic_ev_map=_dynamic_ev,
        )

        # 予測実行
        pred_df = E.predict_all(_models, df_feat)
        pred_df = E.recalc_flags(pred_df)

        # 買い目生成
        picks_list = E.build_race_picks(pred_df)

        # 結果を整形してJSONに変換
        # 艇別詳細
        boat_details = []
        pred_sorted = pred_df.sort_values("アンサンブルスコア_IN順位")
        for _, row in pred_sorted.iterrows():
            boat_details.append({
                "lane":            int(row["艇番"]),
                "name":            str(row.get("選手名", "")),
                "win_prob":        round(float(row.get("予測_1着確率", 0)), 4),
                "ev":              round(float(row.get("真期待値", 0)), 3),
                "ensemble_rank":   int(row.get("アンサンブルスコア_IN順位", 6)),
                "value_score":     round(float(row.get("バリュースコア", 0)), 3),
                "value_flag":      bool(row.get("バリューフラグ", 0)),
                "in_collapse":     bool(row.get("イン崩壊フラグ", 0)),
                "undervalued":     bool(row.get("過小評価フラグ", 0)),
                "st_target":       bool(row.get("ST狙い目フラグ", 0)),
                "est_odds":        round(float(row.get("推定オッズ", 10)), 1),
                "course_bonus":    round(float(row.get("コース補正", 1.0)), 3),
                "night_flag":      bool(row.get("ナイターフラグ", 0)),
            })

        # 買い目
        picks_out = []
        for p in picks_list:
            if p.get("場コード") == venue_code and p.get("レースNo") == race_no:
                combos = [p.get(f"3連単{n}","") for n in ["①","②","③","④","⑤","⑥"] if p.get(f"3連単{n}","")]
                picks_out.append({
                    "verdict":    p.get("判定",""),
                    "axis_lane":  p.get("軸艇番",""),
                    "axis_name":  p.get("軸選手",""),
                    "axis_prob":  p.get("軸確率", 0),
                    "axis_ev":    p.get("軸真期待値", 0),
                    "value_score":p.get("バリュースコア", 0),
                    "dynamic_ev_min": p.get("動的EV_MIN", 0),
                    "combos":     combos,
                    "notes":      p.get("注意フラグ",""),
                    "is_night":   bool(p.get("ナイター","")),
                    "in_collapse":bool(p.get("イン崩壊","")),
                })

        # Kelly bet
        bets_out = []
        for pk in picks_out:
            if "✅" in pk["verdict"] and pk["axis_lane"]:
                ax = pk["axis_lane"]
                ax_row = pred_df[pred_df["艇番"] == ax]
                if len(ax_row):
                    p_win = float(ax_row.iloc[0].get("予測_1着確率", 0))
                    odds  = float(ax_row.iloc[0].get("推定オッズ", 10))
                    ev    = float(ax_row.iloc[0].get("真期待値", 0))
                    unc   = float(ax_row.iloc[0].get("予測不確実性", 0))
                    bet   = E.kelly_bet(p_win, odds, bankroll, venue_code, ev, unc)
                    bets_out.append({"lane": ax, "bet": int(bet)})

        ev_min = float(pred_df["動的EV_MIN"].iloc[0]) if "動的EV_MIN" in pred_df.columns else cfg.get("EV_MIN", 1.5)

        return jsonify({
            "status":       "ok",
            "venue_name":   venue_name,
            "venue_code":   venue_code,
            "race_no":      race_no,
            "race_date":    race_date,
            "dynamic_ev_min": round(ev_min, 3),
            "is_night":     bool(pred_df["ナイターフラグ"].iloc[0]) if "ナイターフラグ" in pred_df.columns else False,
            "boats":        boat_details,
            "picks":        picks_out,
            "bets":         bets_out,
            "bankroll":     bankroll,
            "venue_config": {
                "ev_min":   cfg.get("EV_MIN", 1.5),
                "kelly_kf": cfg.get("Kelly_kf", 0.5),
                "kelly_max":cfg.get("Kelly_max", 0.1),
            },
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ════════════════════════════════════════════════════════════
# 結果記録・履歴・統計・削除・移行
# ════════════════════════════════════════════════════════════

@app.route("/record", methods=["POST"])
def add_record():
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "リクエストボディがありません"}), 400
    except Exception as e:
        return jsonify({"error": f"JSONパース失敗: {e}"}), 400

    required = {"race_date", "venue_name", "race_no", "combo", "bet_amount", "hit"}
    missing  = required - set(data.keys())
    if missing:
        return jsonify({"error": f"必須フィールドが不足: {list(missing)}"}), 400

    try:
        hit         = bool(data["hit"])
        ba          = int(data.get("bet_amount", 0))
        ra          = int(data.get("return_amount", 0)) if hit else 0
        recorded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_id      = datetime.now().strftime("%Y%m%d%H%M%S%f")

        db_execute("""INSERT INTO records
            (id,race_date,venue_name,race_no,combo,bet_amount,hit,return_amount,profit,odds,ev,memo,recorded_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (new_id, str(data["race_date"]), str(data["venue_name"]),
             int(data["race_no"]), str(data["combo"]),
             ba, 1 if hit else 0, ra, ra - ba,
             float(data.get("odds", 0)), float(data.get("ev", 0)),
             str(data.get("memo", "")), recorded_at))

        total = db_execute("SELECT COUNT(*) AS cnt FROM records", fetch="one")
        return jsonify({"status": "ok", "total": total["cnt"] if total else 0})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/history")
def history():
    venue=request.args.get("venue",""); dt=request.args.get("date","")
    limit=int(request.args.get("limit",100))
    where=[]; params=[]
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
def delete_record(record_id):
    affected=db_execute("DELETE FROM records WHERE id = ?",(record_id,))
    if not affected: return jsonify({"error":"IDが見つかりません"}), 404
    total=db_execute("SELECT COUNT(*) AS cnt FROM records",fetch="one")
    return jsonify({"status":"ok","remaining":total["cnt"] if total else 0})

@app.route("/migrate", methods=["POST"])
def migrate_from_json():
    if not LEGACY_JSON.exists(): return jsonify({"message":"records.jsonなし"}), 200
    try:
        with open(LEGACY_JSON,encoding="utf-8") as f: old=json.load(f)
    except Exception as e: return jsonify({"error":str(e)}), 500
    ins=skp=0
    for r in old:
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
            ins+=1
        except Exception: skp+=1
    LEGACY_JSON.rename(LEGACY_JSON.with_suffix(".json.bak"))
    return jsonify({"status":"ok","inserted":ins,"skipped":skp})


# ════════════════════════════════════════════════════════════
# 起動
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)