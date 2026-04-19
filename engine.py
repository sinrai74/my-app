"""競艇予想エンジン（競艇最新.txtから抽出）"""

import re, os, pickle, warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    USE_LGB = True
    print('[LightGBM] 利用可能')
except ImportError:
    USE_LGB = False
    print('[LightGBM] 未インストール → HistGradientBoosting で代替')

# ════════════════════════════════════════════════════════════
# ━━ ① 全場設定：BASE_CONFIG + COURSE_DIFF ━━
# ════════════════════════════════════════════════════════════

BASE_CONFIG = {
    "1コース補正":  1.30,
    "2コース補正":  1.05,
    "3コース補正":  1.00,
    "4コース補正":  0.90,
    "5コース補正":  0.85,
    "6コース補正":  0.75,
    "P_MIN":        0.24,
    "EV_MIN":       1.50,
    "PAY_CAP":      50000,
    "当地重み":     1.20,
    "モーター重み": 1.50,
    "Kelly_kf":     0.50,
    "Kelly_max":    0.10,
    "荒れ閾値":     0.50,    # 荒れ指数がこれ以下なら見送り
    "ST_DIFF_MIN":  0.20,
    "ナイター補正": 1.00,    # ナイターでのコース補正倍率（1.0=変化なし）
    "風補正感度":   0.05,    # 追い風1m/sあたりの外コースブースト
    "干潮補正":     0.00,    # 干潮時のイン有利ブースト
}

# 場ごとの差分設定（BASEから変えたい項目のみ記述）
COURSE_DIFF = {
    # ── 場コード: 差分dict ──────────────────────────────
    "01": {"name":"桐生",   "1コース補正":1.40, "当地重み":1.40},
    "02": {"name":"戸田",   "1コース補正":1.42, "4コース補正":0.82, "5コース補正":0.74,
                            "6コース補正":0.68, "当地重み":1.45, "荒れ閾値":0.20},
    "03": {"name":"江戸川", "1コース補正":1.05, "2コース補正":1.05, "3コース補正":1.02,
                            "4コース補正":1.00, "5コース補正":0.96, "6コース補正":0.90,
                            "EV_MIN":1.60, "PAY_CAP":100000, "当地重み":1.80,
                            "Kelly_kf":0.45, "Kelly_max":0.08, "荒れ閾値":0.80},
    "04": {"name":"平和島", "1コース補正":1.15, "4コース補正":1.05, "EV_MIN":1.60},
    "05": {"name":"多摩川", "1コース補正":1.35, "当地重み":1.30},
    "06": {"name":"浜名湖", "1コース補正":1.18, "2コース補正":1.15, "3コース補正":1.08,
                            "EV_MIN":1.55, "モーター重み":1.60, "荒れ閾値":0.55},
    "07": {"name":"蒲郡",   "1コース補正":1.38},
    "08": {"name":"常滑",   "1コース補正":1.37},
    "09": {"name":"津",     "1コース補正":1.36},
    "10": {"name":"三国",   "1コース補正":1.35},
    "11": {"name":"びわこ", "1コース補正":1.34},
    "12": {"name":"住之江", "1コース補正":1.33},
    "13": {"name":"尼崎",   "1コース補正":1.34},
    "14": {"name":"鳴門",   "1コース補正":1.36},
    "15": {"name":"丸亀",   "1コース補正":1.35},
    "16": {"name":"児島",   "1コース補正":1.33},
    "17": {"name":"宮島",   "1コース補正":1.32},
    "18": {"name":"徳山",   "1コース補正":1.34},
    "19": {"name":"下関",   "1コース補正":1.36},
    "20": {"name":"若松",   "1コース補正":1.37},
    "21": {"name":"芦屋",   "1コース補正":1.25, "EV_MIN":1.55},
    "22": {"name":"福岡",   "1コース補正":1.28, "EV_MIN":1.55},
    "23": {"name":"唐津",   "1コース補正":1.30, "EV_MIN":1.55},
    "24": {"name":"大村",   "1コース補正":1.45, "EV_MIN":1.40,
                            "荒れ閾値":0.35},
}

# 場名→場コードの逆引きマップ
VENUE_NAME_MAP = {v["name"]: k for k, v in COURSE_DIFF.items()}
VENUE_CODE_MAP = {k: v["name"] for k, v in COURSE_DIFF.items()}

def get_venue_config(venue_code: str) -> dict:
    """場コードから完全設定を取得（BASE + 差分をマージ）"""
    config = BASE_CONFIG.copy()
    diff   = COURSE_DIFF.get(str(venue_code).zfill(2), {})
    config.update({k: v for k, v in diff.items() if k != "name"})
    config["name"] = diff.get("name", f"場{venue_code}")
    return config

def print_config_comparison(venue_codes=None):
    """全場設定の比較表を表示"""
    if venue_codes is None:
        venue_codes = sorted(COURSE_DIFF.keys())
    keys = [k for k in BASE_CONFIG.keys() if k not in ("ナイター補正","風補正感度","干潮補正")]
    header = "項目\tBASE\t" + "\t".join(COURSE_DIFF[c]["name"] for c in venue_codes)
    print(header)
    for k in keys:
        row = [k, str(BASE_CONFIG[k])]
        for c in venue_codes:
            val = get_venue_config(c)[k]
            base_val = BASE_CONFIG[k]
            mark = "★" if val != base_val else ""
            row.append(f"{val}{mark}")
        print("\t".join(row))

# ════════════════════════════════════════════════════════════
# ━━ ② 動的EV_MIN：EV_MIN = f(荒れ率) ━━
# ════════════════════════════════════════════════════════════

def compute_dynamic_ev_min(
    k_df: pd.DataFrame,
    venue_code: str,
    base_ev_min: float,
    arashi_weight: float = 0.8,
    min_ev: float = 1.2,
    max_ev: float = 2.5,
) -> float:
    """
    過去結果から場ごとの「荒れ率」を計算し、EV_MINを動的に調整する。

    荒れ率 = 払戻 >= 5000円のレース割合
    荒れやすい場 → EV_MINを上げる（低オッズの本命軸を避ける）
    荒れにくい場 → EV_MINを下げる（安定的なイン軸を積極的に狙う）

    arashi_weight: 荒れ率の影響度（0=無視, 1=完全追従）
    """
    vc_df = k_df[k_df['場コード'] == str(venue_code).zfill(2)]
    if len(vc_df) < 10:
        return base_ev_min   # データ不足 → BASE値を使用

    # 荒れ率（払戻5000円以上 = 荒れレース）
    arashi_rate = (vc_df['払戻'].fillna(0) >= 5000).mean()

    # 平均払戻（外れ値を除く）
    avg_payout = vc_df['払戻'].clip(upper=50000).mean()

    # 動的EV_MIN計算
    # 荒れ率が高い場ほどEV_MINを引き上げる
    # 荒れ率0.3 → EV_MIN変化なし / 0.5 → +0.15 / 0.7 → +0.3
    delta = (arashi_rate - 0.3) * arashi_weight
    dynamic_ev = base_ev_min + delta

    dynamic_ev = round(np.clip(dynamic_ev, min_ev, max_ev), 3)
    print(f'  [{VENUE_CODE_MAP.get(venue_code, venue_code)}] '
          f'荒れ率={arashi_rate:.3f} 平均払戻={avg_payout:.0f} '
          f'→ 動的EV_MIN: {base_ev_min} → {dynamic_ev}')
    return dynamic_ev


def compute_all_dynamic_ev(k_df: pd.DataFrame) -> dict:
    """全場の動的EV_MINを一括計算して辞書で返す"""
    results = {}
    for vc in k_df['場コード'].unique():
        cfg = get_venue_config(vc)
        results[vc] = compute_dynamic_ev_min(k_df, vc, cfg['EV_MIN'])
    return results

# ════════════════════════════════════════════════════════════
# ━━ ③ バリューベット判定：model_prob > implied_prob × α ━━
# ════════════════════════════════════════════════════════════

VALUE_THRESHOLD = 1.20   # model_prob > implied_prob × 1.20 でバリュー認定

def compute_implied_prob(odds: float) -> float:
    """オッズから控除率を考慮した実効的インプライドプロバビリティを計算"""
    if odds <= 1.0: return 1.0
    raw_prob = 1.0 / odds
    # 競艇の控除率は約25%（払戻率75%）
    # インプライドプロブは控除率で割り引く
    return raw_prob * 0.75   # 控除率補正済み implied_prob

def is_value_bet(model_prob: float, odds: float,
                 threshold: float = VALUE_THRESHOLD) -> bool:
    """
    バリューベット判定。
    model_prob > implied_prob × threshold の場合TRUE。
    例: odds=5.0 → implied_prob=0.15 → threshold=1.2 → model_prob>0.18 でバリュー
    """
    if odds <= 1.0 or model_prob <= 0: return False
    implied = compute_implied_prob(odds)
    return model_prob > implied * threshold

def compute_value_score(model_prob: float, odds: float) -> float:
    """
    バリュースコア = model_prob / implied_prob
    1.0 = オッズ通り / >1.0 = 過小評価（バリュー） / <1.0 = 過大評価（バリュー外）
    """
    if odds <= 1.0 or model_prob <= 0: return 0.0
    implied = compute_implied_prob(odds)
    return round(model_prob / implied, 3) if implied > 0 else 0.0

# ════════════════════════════════════════════════════════════
# ━━ ④ 時間帯・気象補正 ━━
# ════════════════════════════════════════════════════════════

# ナイター開催場の場コード（日没後にレースが行われる）
NIGHT_RACE_VENUES = {'04','06','12','17','20','21','22','23','24'}

def is_night_race(venue_code: str, race_no: int) -> bool:
    """ナイターレース判定（場コード + レースNo12R以降）"""
    return (str(venue_code).zfill(2) in NIGHT_RACE_VENUES) and (race_no >= 7)

def compute_time_correction(
    venue_code: str,
    race_no: int,
    node_day: int = 1,
    wind_speed: float = 0.0,      # 風速(m/s)
    wind_direction: str = '',      # 追い風='追', 向かい風='向', 横風='横'
    tide_level: str = '',          # 干潮='干', 満潮='満'
    cfg: dict = None,
) -> dict:
    """
    会場×時間帯補正を計算して辞書で返す。

    返り値:
        night_flag        : ナイターフラグ
        night_course_adj  : ナイター時のコース補正係数（外コース不利）
        wind_outer_boost  : 追い風時の外コースブースト
        tide_in_boost     : 干潮時のイン有利ブースト
        combined_in_adj   : 1コースへの総合補正
        combined_outer_adj: 4〜6コースへの総合補正
    """
    if cfg is None:
        cfg = get_venue_config(venue_code)

    night = is_night_race(venue_code, race_no)

    # ナイター補正：視界が落ちる→外コースはリスク増・インがやや有利
    night_in_adj    = 1.04 if night else 1.00
    night_outer_adj = 0.95 if night else 1.00

    # 風補正：追い風→外コースのまくりが届きやすい / 向かい風→イン有利
    wind_outer = 1.0
    wind_in    = 1.0
    if wind_speed > 0 and wind_direction:
        sensitivity = cfg.get('風補正感度', 0.05)
        if '追' in wind_direction:
            wind_outer = 1.0 + wind_speed * sensitivity       # 追い風は外コースブースト
            wind_in    = 1.0 - wind_speed * sensitivity * 0.5
        elif '向' in wind_direction:
            wind_in    = 1.0 + wind_speed * sensitivity * 0.7 # 向かい風はイン有利
            wind_outer = 1.0 - wind_speed * sensitivity * 0.3
        elif '横' in wind_direction:
            wind_outer = 1.0 + wind_speed * sensitivity * 0.5 # 横風は外コース微ブースト

    # 干満潮補正：干潮→水面穏やかでイン安定 / 満潮→うねりで外台頭
    tide_in    = 1.0
    tide_outer = 1.0
    if tide_level:
        tide_boost = cfg.get('干潮補正', 0.0)
        if '干' in tide_level:
            tide_in    = 1.0 + tide_boost
            tide_outer = 1.0 - tide_boost * 0.5
        elif '満' in tide_level:
            tide_in    = 1.0 - tide_boost * 0.5
            tide_outer = 1.0 + tide_boost

    combined_in_adj    = night_in_adj    * wind_in    * tide_in
    combined_outer_adj = night_outer_adj * wind_outer * tide_outer

    return {
        'night_flag':         int(night),
        'night_in_adj':       round(night_in_adj,    3),
        'night_outer_adj':    round(night_outer_adj, 3),
        'wind_outer_boost':   round(wind_outer,      3),
        'wind_in_adj':        round(wind_in,         3),
        'tide_in_boost':      round(tide_in,         3),
        'tide_outer_adj':     round(tide_outer,      3),
        'combined_in_adj':    round(combined_in_adj,    3),
        'combined_outer_adj': round(combined_outer_adj, 3),
    }

# ════════════════════════════════════════════════════════════
# ── パーサー群（全場対応）
# ════════════════════════════════════════════════════════════

def parse_k_file(filepath: str) -> pd.DataFrame:
    with open(filepath, 'rb') as f:
        text = f.read().decode('shift_jis', errors='replace')
    ZH = str.maketrans('０１２３４５６７８９－', '0123456789-')
    records = []; vc = None
    for line in text.split('\r\n'):
        m = re.match(r'^(\d{2})KBGN$', line.strip())
        if m: vc = m.group(1); continue
        if re.match(r'^\d{2}KEND$', line.strip()): vc = None; continue
        if vc is None: continue
        m2 = re.match(r'\s+(\d{1,2})R\s+(\d-\d-\d)\s+(\d+)', line.translate(ZH))
        if m2:
            p = m2.group(2).split('-')
            records.append({
                '場コード': vc,
                '場名':     VENUE_CODE_MAP.get(vc, f'場{vc}'),
                'レースNo': int(m2.group(1)),
                '組番':     m2.group(2),
                '1着':      int(p[0]),
                '2着':      int(p[1]),
                '3着':      int(p[2]),
                '払戻':     int(m2.group(3)),
                '返還フラグ': 0,
            })
    df = pd.DataFrame(records)
    vc_counts = df.groupby('場名').size().to_dict() if len(df) > 0 else {}
    print(f'[K] {filepath}: {len(df)}R / {df["場コード"].nunique()}場 {vc_counts}')
    return df


FAN_FIELDS = [
    ('登番',4),('名前漢字',16),('名前カナ',15),('支部',4),('級',2),
    ('年号',1),('生年月日',6),('性別',1),('年齢',2),('身長',3),
    ('体重',2),('血液型',2),('勝率',4),('複勝率',4),
    ('1着回数',3),('2着回数',3),('出走回数',3),('優出回数',2),('優勝回数',2),('平均ST',3),
]
for _c in range(1,7):
    FAN_FIELDS += [(f'{_c}コース進入回数',3),(f'{_c}コース複勝率',4),
                   (f'{_c}コース平均ST',3),(f'{_c}コース平均ST順位',3)]
FAN_FIELDS += [('前期級',2),('前々期級',2),('前々々期級',2),('前期能力指数',4),('今期能力指数',4),
               ('年',4),('期',1),('算出期間自',8),('算出期間至',8),('養成期',3)]
for _c in range(1,7):
    for _r in range(1,7): FAN_FIELDS.append((f'{_c}コース{_r}着回数',3))
    for _s in ['F','L0','L1','K0','K1','S0','S1','S2']: FAN_FIELDS.append((f'{_c}コース{_s}回数',2))
for _s in ['L0','L1','K0','K1']: FAN_FIELDS.append((f'コースなし{_s}回数',2))
FAN_FIELDS.append(('出身地',6))
FAN_RECORD_BYTES = 416

def _d(s, dec):
    try: return int(s)/(10**dec)
    except: return None

def parse_fan_files(filepaths):
    recs = []
    for fp in filepaths:
        if not os.path.exists(fp): continue
        raw = open(fp,'rb').read().decode('shift_jis')
        for line in [l for l in raw.split('\r\n') if l.strip()]:
            b = line.encode('shift_jis')
            if len(b) != FAN_RECORD_BYTES: continue
            rec, pos = {}, 0
            for name, size in FAN_FIELDS:
                rec[name] = b[pos:pos+size].decode('shift_jis', errors='replace').strip()
                pos += size
            recs.append(rec)
    df = pd.DataFrame(recs)
    for col, dec in [('勝率',2),('複勝率',1),('平均ST',2),('今期能力指数',2),('前期能力指数',2)]:
        df[col] = df[col].apply(lambda x: _d(x, dec))
    for col in ['出走回数','優出回数','優勝回数']:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    for c in range(1,7):
        df[f'{c}コース進入回数'] = pd.to_numeric(df[f'{c}コース進入回数'], errors='coerce').fillna(0)
        df[f'{c}コース複勝率']   = df[f'{c}コース複勝率'].apply(lambda x: _d(x,1) or 0.0)
        df[f'{c}コース平均ST']   = df[f'{c}コース平均ST'].apply(lambda x: _d(x,2) or 0.18)
    df['算出期間至'] = pd.to_numeric(df['算出期間至'], errors='coerce')
    df = df.sort_values('算出期間至', ascending=False).drop_duplicates('登番', keep='first').reset_index(drop=True)
    print(f'[fan] {len(df)} 選手')
    return df


RE_VS  = re.compile(r'^(\d{2})BBGN$')
RE_VN  = re.compile(r'(ボートレース\S+)')
RE_RC  = re.compile(r'[\s　]*([０-９\d]+)Ｒ')
RE_PL  = re.compile(r'^([1-6])\s+(\d{4})')
RE_PF  = re.compile(r'^([1-6])\s+(\d{4})(.{4})(\d{2})(.{2})(\d{2})([AB][12])\s+'
                    r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+'
                    r'(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\dSFLK ]+?)\s*$')
RE_DAY  = re.compile(r'第\s*([０-９\d]+)\s*日')
RE_DATE = re.compile(r'(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日')

def parse_bangumi(filepath):
    """全場対応（フィルタなし）"""
    raw   = open(filepath, 'rb').read().decode('shift_jis', errors='replace')
    lines = raw.split('\r\n')
    recs  = []; vc = vn = ''; rno = None; nday = rdate = ''
    for line in lines:
        if not line.strip() or line.startswith('STARTB') or line.startswith('-'): continue
        if line.lstrip().startswith('艇') or line.lstrip().startswith('番'): continue
        m = RE_VS.match(line.strip())
        if m: vc, vn, rno, nday, rdate = m.group(1), '', None, '', ''; continue
        if re.match(r'^\d{2}BEND$', line.strip()): continue
        if not vn and RE_VN.search(line):
            vn = RE_VN.search(line).group(1).replace('\u3000', '')
            md = RE_DAY.search(line); dt = RE_DATE.search(line)
            if md: nday  = md.group(1).translate(ZEN2HAN)
            if dt: rdate = f'{dt.group(1)}-{int(dt.group(2)):02d}-{int(dt.group(3)):02d}'
            continue
        if vn and not nday:
            md = RE_DAY.search(line)
            if md: nday = md.group(1).translate(ZEN2HAN)
        if vn and not rdate:
            dt = RE_DATE.search(line)
            if dt: rdate = f'{dt.group(1)}-{int(dt.group(2)):02d}-{int(dt.group(3)):02d}'
        m = RE_RC.match(line)
        if m:
            rs = re.search(r'[０-９\d]+', line)
            if rs: rno = int(rs.group().translate(ZEN2HAN))
            continue
        if RE_PL.match(line) and rno is not None:
            pm = RE_PF.match(line)
            if pm:
                recs.append({
                    '場コード': vc, '場名': vn,
                    '節日': int(nday) if nday else None,
                    '開催日': rdate, 'レースNo': rno,
                    '艇番': int(pm.group(1)),
                    '登番': pm.group(2).strip(), '選手名': pm.group(3).strip(),
                    '年齢': int(pm.group(4)), '支部': pm.group(5).strip(),
                    '体重': int(pm.group(6)), '級': pm.group(7).strip(),
                    '全国勝率': float(pm.group(8)),  '全国2率': float(pm.group(9)),
                    '当地勝率': float(pm.group(10)), '当地2率': float(pm.group(11)),
                    'モーターNO': int(pm.group(12)), 'モーター2率': float(pm.group(13)),
                    'ボートNO': int(pm.group(14)),   'ボート2率': float(pm.group(15)),
                    '今節成績': pm.group(16).strip(),
                })
    df = pd.DataFrame(recs)
    print(f'[番組表] {filepath}: {len(df)}行 / {df["場名"].nunique()}場 / '
          f'{df.groupby(["場コード","レースNo"]).ngroups}レース')
    return df

# ════════════════════════════════════════════════════════════
# ── 結合・結果付与
# ════════════════════════════════════════════════════════════

FAN_MERGE_COLS = (
    ['登番','勝率','複勝率','平均ST','今期能力指数','前期能力指数',
     '名前漢字','支部','級','出走回数','優出回数','優勝回数']
    + [f'{c}コース進入回数' for c in range(1,7)]
    + [f'{c}コース複勝率'   for c in range(1,7)]
    + [f'{c}コース平均ST'   for c in range(1,7)]
)

def merge_bangumi_fan(bangumi_df, fan_df):
    fan_sub = fan_df[[c for c in FAN_MERGE_COLS if c in fan_df.columns]].copy()
    fan_sub['登番'] = fan_sub['登番'].astype(str).str.strip()
    df = bangumi_df.copy()
    df['登番'] = df['登番'].astype(str).str.strip()
    merged = df.merge(fan_sub, on='登番', how='left', suffixes=('_番組','_fan'))
    if '級_番組' in merged.columns:
        merged['級'] = merged['級_番組'].fillna('')
    merged.drop(columns=[c for c in merged.columns if c in ('級_番組','級_fan')],
                inplace=True, errors='ignore')
    merged['選手名'] = merged['選手名'].replace('', None).fillna(merged.get('名前漢字', ''))
    return merged

def attach_k_results(merged_df, k_df):
    rows = []
    for _, r in k_df.iterrows():
        for rank, col in [(1,'1着'),(2,'2着'),(3,'3着')]:
            rows.append({'場コード': str(r['場コード']), 'レースNo': r['レースNo'],
                         '艇番': r[col], '着順_k': rank, '払戻': r['払戻'],
                         '人気': np.nan, '返還フラグ': r['返還フラグ']})
    result_long = pd.DataFrame(rows)
    merged = merged_df.copy()
    merged['場コード'] = merged['場コード'].astype(str)
    merged = merged.merge(result_long, on=['場コード','レースNo','艇番'], how='left')
    for (vc, rno), idx in merged.groupby(['場コード','レースNo']).groups.items():
        sub = merged.loc[idx]
        for seq, i in enumerate(sub[sub['着順_k'].isna()].index, start=4):
            merged.loc[i, '着順_k'] = seq
        for col in ['払戻','返還フラグ']:
            if col not in merged.columns: merged[col] = np.nan
            val = sub[col].dropna().values if col in sub.columns else []
            if len(val) > 0: merged.loc[idx, col] = merged.loc[idx, col].fillna(val[0])
    merged['着順']       = pd.to_numeric(merged['着順_k'], errors='coerce').astype('Int64')
    merged.drop(columns=['着順_k'], inplace=True, errors='ignore')
    merged['払戻']       = pd.to_numeric(merged['払戻'], errors='coerce')
    merged['人気']       = merged.get('人気', np.nan)
    merged['返還フラグ'] = merged['返還フラグ'].fillna(0).astype(int)
    print(f'[結果付与] {merged["着順"].notna().sum()} / {len(merged)} 件')
    return merged

# ════════════════════════════════════════════════════════════
# ── 今節成績パーサー
# ════════════════════════════════════════════════════════════

def parse_today_results_safe(s, node_day, race_no):
    empty = {'今節出走数':0,'今節F':0,'今節L':0,'今節K':0,'今節連対率':0.0,
             '今節平均着順':3.5,'今節1着回数':0,'今節直近3連対率':0.0}
    if not isinstance(s, str) or not s.strip(): return empty
    text = s.strip()
    all_places = [int(d) for d in re.findall(r'[1-6]', text)]
    max_usable = max(0, (int(node_day)-1)*12) if node_day and not (
        isinstance(node_day, float) and np.isnan(node_day)) else 0
    places = all_places[-max_usable:] if max_usable > 0 else []
    n = len(places); top2 = sum(1 for p in places if p <= 2)
    r3 = places[-3:] if len(places) >= 3 else places
    return {
        '今節出走数': n, '今節F': text.count('F'), '今節L': text.count('L'), '今節K': text.count('K'),
        '今節連対率': round(top2/n, 3) if n > 0 else 0.0,
        '今節平均着順': round(np.mean(places), 2) if places else 3.5,
        '今節1着回数': sum(1 for p in places if p == 1),
        '今節直近3連対率': round(sum(1 for p in r3 if p <= 2)/len(r3), 3) if r3 else 0.0,
    }

def estimate_odds(ninki_series):
    filled = ninki_series.fillna(10)
    return (np.exp(1.182 * np.log1p(filled) + 5.214) / 100).clip(lower=1.0)

# ════════════════════════════════════════════════════════════
# ━━ 統合特徴量エンジニアリング（全場設定+時間帯補正対応）━━
# ════════════════════════════════════════════════════════════

def engineer_features(
    df,
    wind_speed: float = 0.0,
    wind_direction: str = '',
    tide_level: str = '',
    dynamic_ev_map: dict = None,   # {場コード: 動的EV_MIN}
):
    """
    全場統合版特徴量エンジニアリング。
    各レースの場コードから get_venue_config() で設定を取得し、
    コース補正・当地重み・時間帯補正を動的に適用する。
    """
    df = df.copy()
    df['級_数値'] = df['級'].map({'A1':4,'A2':3,'B1':2,'B2':1}).fillna(0)

    # 予想進入コース
    entry_cols = [f'{c}コース進入回数' for c in range(1,7)]
    existing   = [c for c in entry_cols if c in df.columns]
    if existing:
        ec    = df[existing].fillna(0)
        total = ec.sum(axis=1)
        best  = ec.idxmax(axis=1).str.extract(r'(\d)コース').astype(float).squeeze()
        df['予想進入'] = np.where(total >= 5, best, df['艇番']).astype(int)
    else:
        df['予想進入'] = df['艇番'].astype(int)
    df['進入強度'] = (df['艇番'] - df['予想進入']).astype(int)

    # 進入コース別特徴量
    c_idx   = df['予想進入'].astype(int).clip(1, 6)
    st_vals = np.array([df.get(f'{c}コース平均ST', pd.Series(0, index=df.index)).fillna(0).values
                        for c in range(1,7)])
    chosen_st = st_vals[c_idx.values - 1, np.arange(len(df))]
    fallback  = df['平均ST'].fillna(0.18).values
    df['進入コースST'] = np.where(chosen_st > 0.05, chosen_st, fallback)

    w2_vals = np.array([df.get(f'{c}コース複勝率', pd.Series(0, index=df.index)).fillna(0).values
                        for c in range(1,7)])
    df['進入コース複勝率'] = w2_vals[c_idx.values - 1, np.arange(len(df))]

    in_vals   = np.array([df.get(f'{c}コース進入回数', pd.Series(0, index=df.index)).fillna(0).values
                          for c in range(1,7)])
    chosen_in = in_vals[c_idx.values - 1, np.arange(len(df))]
    total_in  = in_vals.sum(axis=0)
    df['進入コース進入割合'] = np.where(total_in > 0, np.round(chosen_in / total_in.clip(1), 3), 0.0)

    # ── 【統合】場コードごとに設定を取得してコース補正を適用 ──────
    course_keys = [f'{c}コース補正' for c in range(1,7)]

    def _get_course_correction(row):
        cfg = get_venue_config(row['場コード'])
        # 時間帯補正を取得
        tc = compute_time_correction(
            row['場コード'], row.get('レースNo', 6),
            node_day=row.get('節日', 1),
            wind_speed=wind_speed, wind_direction=wind_direction,
            tide_level=tide_level, cfg=cfg)
        c = int(row.get('予想進入', row.get('艇番', 1)))
        base = cfg.get(f'{c}コース補正', 1.0)
        # 時間帯補正を適用
        if c == 1:
            return base * tc['combined_in_adj']
        elif c >= 4:
            return base * tc['combined_outer_adj']
        return base

    df['コース補正'] = df.apply(_get_course_correction, axis=1)

    # ── 【統合】ナイター・時間帯フラグをDFに追加 ────────────────
    df['ナイターフラグ'] = df.apply(
        lambda r: int(is_night_race(r['場コード'], r.get('レースNo', 1))), axis=1)

    # ── 今節成績 ───────────────────────────────────────────────
    parsed = df.apply(lambda r: parse_today_results_safe(
        r['今節成績'],
        None if (isinstance(r.get('節日'), float) and np.isnan(r.get('節日', np.nan))) else r.get('節日'),
        r.get('レースNo', 1)), axis=1)
    for key in ['今節出走数','今節F','今節L','今節K','今節連対率','今節平均着順','今節1着回数','今節直近3連対率']:
        df[key] = parsed.apply(lambda x: x.get(key, 0))

    grp_key = ['場コード', 'レースNo']
    G = df.groupby(grp_key)

    df['人気_filled'] = df['人気'].fillna(G['人気'].transform('median')).fillna(10)
    df['推定オッズ']  = estimate_odds(df['人気_filled'])
    df['人気_IN順位'] = G['人気_filled'].transform(lambda x: x.rank(pct=True))
    df['人気差']      = df['人気_filled'] - G['人気_filled'].transform('min')

    出走 = df['出走回数'].fillna(0)
    df['優出率'] = np.where(出走 > 0, df['優出回数'].fillna(0) / 出走, 0)
    df['優勝率'] = np.where(出走 > 0, df['優勝回数'].fillna(0) / 出走, 0)
    df['人気_逆数']      = 1.0 / df['人気_filled'].clip(lower=1)
    df['人気×勝率']     = df['人気_filled'] * df['全国勝率'].fillna(0)
    df['人気×モーター'] = df['人気_filled'] * df['モーター2率'].fillna(0)
    df['人気×能力指数'] = df['人気_filled'] * df['今期能力指数'].fillna(0)
    df['内有利度']   = df['コース補正'] * (1 - df['進入コースST'].clip(0, 0.99))
    df['外まくり力'] = df['進入コース複勝率'] * (df['予想進入'] >= 3).astype(float)
    df['イン信頼度'] = (df['予想進入'] == 1).astype(float) * df['全国勝率'].fillna(0)

    # ── 【統合】当地重み付きスコア（場ごとに重みが変わる）──────
    def _local_weight(vc):
        return get_venue_config(vc).get('当地重み', BASE_CONFIG['当地重み'])
    def _motor_weight(vc):
        return get_venue_config(vc).get('モーター重み', BASE_CONFIG['モーター重み'])

    lw = df['場コード'].map(_local_weight).fillna(BASE_CONFIG['当地重み'])
    mw = df['場コード'].map(_motor_weight).fillna(BASE_CONFIG['モーター重み'])

    df['当地スコア']   = df['当地勝率'].fillna(0) * lw
    df['モータースコア'] = df['モーター2率'].fillna(0) * mw + df['ボート2率'].fillna(0) * 0.5

    # ── 【統合】バリュースコア（③の実装）──────────────────────
    df['インプライドプロブ'] = df['推定オッズ'].apply(compute_implied_prob)
    # 予測確率はこの時点ではまだないので、全国勝率_IN順位の代理値を使う
    # （predict_all内で正式計算）
    df['バリュースコア_仮'] = 0.0   # predict_all で上書き

    # ── IN順位 ────────────────────────────────────────────────
    rank_specs = [
        ('全国勝率',False),('全国2率',False),('当地勝率',False),('当地2率',False),
        ('今期能力指数',False),('平均ST',True),('モーター2率',False),('ボート2率',False),
        ('今節連対率',False),('進入コース複勝率',False),('当地スコア',False),
    ]
    for col, asc in rank_specs:
        if col in df.columns:
            df[f'{col}_IN順位'] = G[col].transform(lambda x: x.rank(pct=True, ascending=asc))

    # ── 差分系 ────────────────────────────────────────────────
    diff_cols = ['全国勝率','モーター2率','今期能力指数','当地勝率','当地2率','今節連対率','進入コース複勝率']
    for col in diff_cols:
        if col in df.columns:
            gm = G[col].transform('mean')
            gx = G[col].transform('max')
            gn = G[col].transform('min')
            df[f'{col}_IN偏差'] = df[col] - gm
            df[f'{col}_MAX差']  = gx - df[col]
            df[f'{col}_MIN差']  = df[col] - gn

    df['モーター偏差'] = df['モーター2率'] - G['モーター2率'].transform('mean')
    df['荒れ指数']     = G['全国勝率'].transform('std').fillna(0)
    df['ST拮抗度']     = G['進入コースST'].transform('std').fillna(0)
    df['ST優位度']     = G['進入コースST'].transform('min') - df['進入コースST']
    df['ST不利度']     = df['進入コースST'] - G['進入コースST'].transform('min')

    if '人気_IN順位' in df.columns:
        df['勝率順位差']     = df['全国勝率_IN順位']  - df['人気_IN順位']
        df['モーター順位差'] = df['モーター2率_IN順位'] - df['人気_IN順位']
        df['能力順位差']     = df['今期能力指数_IN順位'] - df['人気_IN順位']
    if '平均ST_IN順位' in df.columns:
        df['ST順位差'] = (1 - df['平均ST_IN順位']) - df['人気_IN順位']

    # フラグ
    df['イン崩壊指数'] = (
        (df['予想進入']==1).astype(int)
        * (1 - df['進入コース複勝率'].clip(0,1))
        * df['ST不利度'].clip(0, 0.1) * 10
    )
    df['イン崩壊フラグ'] = (
        (df['予想進入'] == 1)
        & ((1 - df['進入コース複勝率'].fillna(0)) > 0.55)
        & (df['ST不利度'] > 0.010)
    ).astype(int)
    df['過小評価フラグ'] = (
        (df.get('勝率順位差', pd.Series(0, index=df.index)) > 0.2)
        & (df['人気_IN順位'] > 0.6)
    ).astype(int)
    df['クラスタ別勝率'] = (
        df.groupby('コース補正')['1着フラグ'].transform('mean').fillna(0.167)
        if '1着フラグ' in df.columns else 0.167
    )

    # 展開クラスタ（コース補正で動的判定）
    df['展開クラスタ'] = np.where(
        (df['コース補正'] >= 1.35) & (df['進入コースST'] <= 0.16), 1,
        np.where((df['予想進入'].isin([2,3])) & (df['進入コース複勝率'] >= 0.35), 2,
        np.where((df['予想進入'] >= 4) & (df['進入コース複勝率'] >= 0.35), 3, 0)))

    in1_wr  = df[df['予想進入']==1].groupby(grp_key)['全国勝率'].first().rename('_in1')
    in2_wr  = df[df['予想進入']==2].groupby(grp_key)['全国勝率'].first().rename('_in2')
    wr_diff = (in1_wr - in2_wr).rename('_diff').reset_index()
    df = df.merge(wr_diff, on=grp_key, how='left')
    df['イン対抗指数'] = df['_diff'].fillna(0.0)
    df.drop(columns=['_diff'], inplace=True, errors='ignore')

    in_st = df[df['予想進入']==1].groupby(grp_key)['進入コースST'].first().rename('_inST').reset_index()
    df = df.merge(in_st, on=grp_key, how='left')
    df['インST'] = df['_inST'].fillna(0.18)
    df.drop(columns=['_inST'], inplace=True, errors='ignore')

    df['節日']      = pd.to_numeric(df['節日'], errors='coerce').fillna(1).astype(int)
    df['初日フラグ'] = (df['節日'] == 1).astype(int)

    # ── 【統合】動的EV_MINを列として付与 ─────────────────────
    if dynamic_ev_map:
        df['動的EV_MIN'] = df['場コード'].map(dynamic_ev_map).fillna(
            df['場コード'].map(lambda vc: get_venue_config(vc).get('EV_MIN', BASE_CONFIG['EV_MIN'])))
    else:
        df['動的EV_MIN'] = df['場コード'].map(
            lambda vc: get_venue_config(vc).get('EV_MIN', BASE_CONFIG['EV_MIN']))

    return df


def build_targets(df):
    df = df.copy()
    # PAY_CAPは場ごとに異なる → 場コードから取得
    df['1着フラグ']   = (df['着順'] == 1).astype('Int64')
    df['3着内フラグ'] = (df['着順'] <= 3).astype('Int64')
    df.loc[df['返還フラグ'] == 1, '払戻'] = None

    def _capped_payout(r):
        cap = get_venue_config(r['場コード']).get('PAY_CAP', BASE_CONFIG['PAY_CAP'])
        if r['1着フラグ'] == 1 and pd.notna(r['払戻']):
            return min(float(r['払戻']), cap)
        return 0.0 if r['返還フラグ'] == 0 else None

    df['期待値'] = df.apply(_capped_payout, axis=1)
    df['log払戻'] = df.apply(
        lambda r: np.log1p(min(r['払戻'],
                              get_venue_config(r['場コード']).get('PAY_CAP', BASE_CONFIG['PAY_CAP'])))
        if pd.notna(r.get('払戻')) and r.get('払戻', 0) > 0 else None, axis=1)
    return df

# ════════════════════════════════════════════════════════════
# ── 【統合】特徴量リスト
# ════════════════════════════════════════════════════════════

FEATURE_COLS = [
    '年齢','体重','級_数値','全国勝率','全国2率','当地勝率','当地2率',
    '今期能力指数','前期能力指数','平均ST','出走回数','優出率','優勝率',
    'モーター2率','ボート2率','艇番','予想進入','進入強度',
    '進入コースST','進入コース複勝率','進入コース進入割合','コース補正',
    '1コース複勝率','2コース複勝率','3コース複勝率','4コース複勝率','5コース複勝率','6コース複勝率',
    '今節出走数','今節F','今節L','今節K','今節連対率','今節平均着順','今節1着回数','今節直近3連対率',
    '人気_filled','推定オッズ','人気_IN順位','人気_逆数','人気差',
    '人気×勝率','人気×モーター','人気×能力指数',
    '勝率順位差','モーター順位差','能力順位差','ST順位差',
    'イン対抗指数','イン信頼度',
    '全国勝率_IN順位','全国2率_IN順位','当地勝率_IN順位','当地2率_IN順位',
    '今期能力指数_IN順位','平均ST_IN順位','モーター2率_IN順位','ボート2率_IN順位',
    '今節連対率_IN順位','進入コース複勝率_IN順位','当地スコア_IN順位',
    '全国勝率_IN偏差','全国勝率_MAX差','全国勝率_MIN差',
    'モーター2率_IN偏差','モーター2率_MAX差','モーター2率_MIN差',
    '今期能力指数_IN偏差','今期能力指数_MAX差','今期能力指数_MIN差',
    '当地勝率_IN偏差','当地勝率_MAX差','当地勝率_MIN差',
    '当地2率_IN偏差','当地2率_MAX差','当地2率_MIN差',
    '今節連対率_IN偏差','今節連対率_MAX差','今節連対率_MIN差',
    '進入コース複勝率_IN偏差','進入コース複勝率_MAX差','進入コース複勝率_MIN差',
    'モーター偏差','展開クラスタ','クラスタ別勝率',
    'ST優位度','ST不利度','イン崩壊指数','イン崩壊フラグ','過小評価フラグ',
    '荒れ指数','ST拮抗度','内有利度','外まくり力','インST','節日','初日フラグ',
    # ── 統合新規特徴量 ──
    '当地スコア','モータースコア','インプライドプロブ',
    'ナイターフラグ','動的EV_MIN',
]

# ════════════════════════════════════════════════════════════
# ── モデル
# ════════════════════════════════════════════════════════════

def make_clf(spw=1.0):
    if USE_LGB:
        return lgb.LGBMClassifier(
            n_estimators=400, learning_rate=0.04, max_depth=-1, num_leaves=31,
            subsample=0.78, colsample_bytree=0.78, min_child_samples=15,
            scale_pos_weight=spw, random_state=42, verbose=-1)
    cw = {0:1.0, 1:spw} if spw != 1.0 else None
    return HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.04, max_leaf_nodes=31,
        min_samples_leaf=15, early_stopping=True, n_iter_no_change=30,
        validation_fraction=0.1, class_weight=cw, random_state=42)

def make_clf_cal(spw=1.0):
    return CalibratedClassifierCV(make_clf(spw), method='isotonic', cv=2)

def make_reg():
    if USE_LGB:
        return lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.04, max_depth=-1, num_leaves=31,
            subsample=0.78, colsample_bytree=0.78, min_child_samples=15,
            random_state=42, verbose=-1)
    return HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.04, max_leaf_nodes=31,
        min_samples_leaf=15, early_stopping=True, n_iter_no_change=30,
        validation_fraction=0.1, random_state=42)

def get_X(df):
    feats = [c for c in FEATURE_COLS if c in df.columns]
    return df[feats].copy().fillna(df[feats].median()), feats

def fit_models(train, add_noise=True):
    tr = train.copy()
    if add_noise and '人気_filled' in tr.columns:
        ns = tr['人気_filled'].std() * 0.1; np.random.seed(42)
        tr['人気_filled'] = (tr['人気_filled'] + np.random.normal(0, ns, len(tr))).clip(lower=1)
        tr['推定オッズ']  = estimate_odds(tr['人気_filled'])
        tr['人気_逆数']   = tr['人気_filled'].apply(lambda x: 1.0/x if x > 0 else 0.0)
        tr['人気差']      = tr.groupby(['場コード','レースNo'])['人気_filled'].transform(lambda x: x - x.min())
        tr['人気×勝率']     = tr['人気_filled'] * tr['全国勝率'].fillna(0)
        tr['人気×モーター'] = tr['人気_filled'] * tr['モーター2率'].fillna(0)
        tr['人気×能力指数'] = tr['人気_filled'] * tr['今期能力指数'].fillna(0)

    X_tr, feats = get_X(tr)
    valid  = (tr['返還フラグ'] == 0)
    models = {'feature_names': feats}
    y_win  = tr.loc[valid, '1着フラグ'].astype(int)
    y_top3 = tr.loc[valid, '3着内フラグ'].astype(int)
    spw_win  = (y_win==0).sum() / max((y_win==1).sum(), 1)
    spw_top3 = (y_top3==0).sum() / max((y_top3==1).sum(), 1)
    print(f'  学習: n={valid.sum()} / spw_win={spw_win:.1f}')

    clf_win  = make_clf_cal(spw_win);  clf_win.fit(X_tr[valid],  y_win);  models['win']  = clf_win
    clf_top3 = make_clf_cal(spw_top3); clf_top3.fit(X_tr[valid], y_top3); models['top3'] = clf_top3
    reg_rank = make_reg(); reg_rank.fit(X_tr[valid], tr.loc[valid, '着順'].astype(float)); models['rank'] = reg_rank

    win_mask = valid & (tr['1着フラグ'] == 1) & tr['log払戻'].notna()
    if win_mask.sum() >= 10:
        rp = make_reg(); rp.fit(X_tr[win_mask], tr.loc[win_mask, 'log払戻'].astype(float))
        models['payout'] = rp; print(f'  払戻モデル: {win_mask.sum()}件')
    else:
        models['payout'] = None

    if '推定オッズ' in tr.columns and '全国勝率_IN順位' in tr.columns:
        ev_mask  = valid & tr['推定オッズ'].notna()
        ev_proxy = tr['全国勝率_IN順位'].fillna(0.5) * tr['推定オッズ'].fillna(10) / 10
        y_ev     = (ev_proxy[ev_mask] > 1.0).astype(int)
        n1, n0   = y_ev.sum(), (y_ev==0).sum()
        if n1 > 5 and n0 > 5:
            ce = make_clf_cal(n0/max(n1,1)); ce.fit(X_tr[ev_mask], y_ev); models['ev_clf'] = ce
        else:
            models['ev_clf'] = None
    else:
        models['ev_clf'] = None
    return models


def predict_all(models, df):
    X, _ = get_X(df); df = df.copy()
    df['予測_1着確率']      = models['win'].predict_proba(X)[:,1]
    df['予測_3着内確率']    = models['top3'].predict_proba(X)[:,1]
    df['予測_着順']         = models['rank'].predict(X).clip(1, 6)
    df['予測_EV_2段階']     = (
        df['予測_1着確率'] * np.expm1(models['payout'].predict(X).clip(0)) / 100
        if models.get('payout') else df['予測_1着確率'] * df['推定オッズ'])
    df['予測_EVフラグ確率'] = (
        models['ev_clf'].predict_proba(X)[:,1]
        if models.get('ev_clf') else df['予測_1着確率'])

    df['人気歪み補正'] = 1.0 + df['人気_IN順位'].fillna(0.5) * 0.15
    df['真期待値']     = df['予測_1着確率'] * df['推定オッズ'] * df['人気歪み補正']
    df['log真期待値']  = df['予測_1着確率'] * np.log1p(df['推定オッズ'])
    df['予測不確実性'] = (
        df['予測_1着確率']
        - df.groupby(['場コード','レースNo'])['予測_1着確率'].transform(
            lambda x: x.rank(method='min', ascending=True, pct=True))
    ).abs()

    # ── ③ バリュースコアの正式計算 ────────────────────────────
    df['バリュースコア'] = df.apply(
        lambda r: compute_value_score(r['予測_1着確率'], r['推定オッズ']), axis=1)
    df['バリューフラグ'] = df.apply(
        lambda r: int(is_value_bet(r['予測_1着確率'], r['推定オッズ'])), axis=1)

    rank_s  = (7 - df['予測_着順']) / 6
    ev_max  = df.groupby(['場コード','レースNo'])['予測_EV_2段階'].transform('max').clip(lower=0.001)

    # ── アンサンブルスコア（バリュースコアを追加） ──────────────
    val_max = df.groupby(['場コード','レースNo'])['バリュースコア'].transform('max').clip(0.001)

    df['アンサンブルスコア'] = (
        df['予測_1着確率']             * 0.28
        + df['予測_3着内確率']         * 0.12
        + rank_s                       * 0.10
        + (df['予測_EV_2段階']/ev_max) * 0.22
        + df['予測_EVフラグ確率']      * 0.12
        + (df['バリュースコア']/val_max) * 0.10   # バリュースコアをスコアに組み込む
        + df.get('当地スコア', pd.Series(0, index=df.index)).fillna(0) /
          df.groupby(['場コード','レースNo'])['当地スコア'].transform('max').clip(0.001) * 0.06
    )

    for col in ['予測_1着確率','真期待値','アンサンブルスコア']:
        df[f'{col}_IN順位'] = df.groupby(['場コード','レースNo'])[col].transform(
            lambda x: x.rank(method='min', ascending=False).astype(int))
    return df


def timeseries_cv(df, fast_mode=True):
    node_days  = sorted(df['節日'].unique())
    all_pred   = []; fold_metrics = []
    eval_days  = node_days[-2:] if fast_mode and len(node_days) > 2 else node_days
    print(f'[TimeSeries CV] 節日: {node_days} / fast_mode={fast_mode}')
    for i, test_day in enumerate(node_days):
        train_days = [d for d in node_days if d < test_day]
        if not train_days: continue
        if fast_mode and test_day not in eval_days: continue
        train = df[df['節日'].isin(train_days)]
        test  = df[df['節日'] == test_day]
        if len(train[train['返還フラグ']==0]) < 30: continue
        print(f'  fold{i}: train=節日{train_days}({len(train)}行) → test=節日{test_day}({len(test)}行)')
        models_cv = fit_models(train); pred = predict_all(models_cv, test); all_pred.append(pred)
        valid = pred[pred['返還フラグ']==0]
        try:
            auc = roc_auc_score(valid['1着フラグ'].astype(int), valid['予測_1着確率'])
            fold_metrics.append({'fold':i,'test_day':test_day,'n_test':len(valid),'AUC_win':round(auc,4)})
        except Exception as e:
            fold_metrics.append({'fold':i,'test_day':test_day,'error':str(e)})
    pred_df    = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    metrics_df = pd.DataFrame(fold_metrics)
    print('\n[CV結果]'); print(metrics_df.to_string(index=False))
    if 'AUC_win' in metrics_df.columns:
        print(f'  平均AUC: {metrics_df["AUC_win"].mean():.4f}')
    print('\n[最終モデル] 全データで学習中...')
    final_models = fit_models(df)
    return {'pred_df':pred_df, 'final_models':final_models, 'metrics_df':metrics_df}

# ════════════════════════════════════════════════════════════
# ── 【統合】Kelly・購入ロジック（場ごとに設定を動的適用）
# ════════════════════════════════════════════════════════════

def kelly_bet(p_win, odds, bankroll, venue_code='01', true_ev=1.0, unc=0.0):
    """場コードから Kelly_kf / Kelly_max を動的取得"""
    cfg     = get_venue_config(venue_code)
    kf      = cfg.get('Kelly_kf', BASE_CONFIG['Kelly_kf'])
    max_frac = cfg.get('Kelly_max', BASE_CONFIG['Kelly_max'])
    p_min   = cfg.get('P_MIN', BASE_CONFIG['P_MIN'])

    if p_win < p_min: return 0.0
    b = odds - 1
    if b <= 0: return 0.0
    kf_full = (b * p_win - (1 - p_win)) / b
    if kf_full <= 0: return 0.0
    boost    = min(1.5, true_ev)
    unc_adj  = max(0.5, 1.0 - unc)
    raw_bet  = bankroll * kf_full * kf * boost * unc_adj
    capped   = min(raw_bet, bankroll * max_frac)
    return max(0.0, (capped // 100) * 100)

def _build_sanrentan_by_prob(grp, axis_no, sc, tc):
    prob_map = grp.set_index('艇番')['予測_3着内確率'].to_dict()
    scored = []
    for s in sc:
        p_s = prob_map.get(s, 0.1) * 0.6
        for t in tc:
            if t == s: continue
            p_t = prob_map.get(t, 0.1) * 0.4
            scored.append((p_s * p_t, f'{axis_no}-{s}-{t}'))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:6]]

def generate_bets(race_df, bankroll=10000):
    bets = []; cur_bk = bankroll

    for (vc, rno), grp in race_df.sort_values(['場コード','レースNo']).groupby(
            ['場コード','レースNo'], sort=False):
        if grp['返還フラグ'].max() == 1: continue
        grp = grp.sort_values('アンサンブルスコア_IN順位').reset_index(drop=True)

        cfg        = get_venue_config(vc)
        venue_name = cfg['name']
        arashi_thr = cfg.get('荒れ閾値', BASE_CONFIG['荒れ閾値'])

        if grp['荒れ指数'].mean() < arashi_thr:
            bets.append({'場名':venue_name,'場コード':vc,'レースNo':rno,'購入':False,
                         '理由':'荒れ指数低すぎ（見送り）','軸艇番':None,'ベット額':0,
                         '残高':cur_bk,'3連単':[],'2連単':[],'ワイド':[],'3連単点数':0})
            continue
        if '予測不確実性' in grp.columns and grp['予測不確実性'].mean() > 0.45:
            bets.append({'場名':venue_name,'場コード':vc,'レースNo':rno,'購入':False,
                         '理由':'予測不確実性高すぎ（見送り）','軸艇番':None,'ベット額':0,
                         '残高':cur_bk,'3連単':[],'2連単':[],'ワイド':[],'3連単点数':0})
            continue

        # ── 動的EV_MIN + P_MIN で軸候補を選定 ──────────────────
        ev_min  = grp['動的EV_MIN'].iloc[0] if '動的EV_MIN' in grp.columns else cfg.get('EV_MIN', BASE_CONFIG['EV_MIN'])
        p_min   = cfg.get('P_MIN', BASE_CONFIG['P_MIN'])

        # ③ バリューフラグも追加条件として考慮
        cands = grp[
            (grp['真期待値'] > ev_min) &
            (grp['予測_1着確率'] > p_min) &
            (grp.get('バリューフラグ', pd.Series(1, index=grp.index)) == 1)
        ]
        # バリューフラグ条件で全滅した場合はバリューフラグなしで再試行
        if len(cands) == 0:
            cands = grp[(grp['真期待値'] > ev_min) & (grp['予測_1着確率'] > p_min)]

        if len(cands) == 0:
            bets.append({'場名':venue_name,'場コード':vc,'レースNo':rno,'購入':False,
                         '理由':'EV/P/バリュー基準未達','軸艇番':None,'ベット額':0,
                         '残高':cur_bk,'3連単':[],'2連単':[],'ワイド':[],'3連単点数':0})
            continue

        top_axes = cands.sort_values('真期待値', ascending=False).head(2)
        n_axes   = len(top_axes)

        for _, axis in top_axes.iterrows():
            axis_no = int(axis['艇番'])
            p    = float(axis['予測_1着確率'])
            odds = float(axis['推定オッズ'])
            ev   = float(axis['真期待値'])
            unc  = float(axis.get('予測不確実性', 0.0))
            bk_per_axis = cur_bk / n_axes
            bet  = max(kelly_bet(p, odds, bk_per_axis, vc, true_ev=ev, unc=unc), 100)
            top3 = grp.head(3)['艇番'].tolist()
            top5 = grp.head(5)['艇番'].tolist()
            sc   = [b for b in top3 if b != axis_no]
            tc   = [b for b in top5 if b != axis_no]
            st   = _build_sanrentan_by_prob(grp, axis_no, sc, tc)
            bets.append({
                '場名':venue_name, '場コード':vc, 'レースNo':rno, '購入':True,
                '理由':f'EV={ev:.2f} P={p:.3f} 軸{axis_no} バリュー={axis.get("バリュースコア",0):.2f}',
                '軸艇番':axis_no, '真期待値':ev, '予測_1着確率':p,
                'バリュースコア':float(axis.get('バリュースコア', 0)),
                'ベット額':int(bet), '残高（購入前）':cur_bk,
                '3連単点数':len(st), '3連単':st,
                '2連単':[f'{axis_no}-{b}' for b in sc],
                'ワイド':[f'{min(axis_no,b)}-{max(axis_no,b)}' for b in sc],
            })
        cur_bk = max(cur_bk - sum(
            b['ベット額'] for b in bets[-n_axes:] if b.get('購入')), 0)

    df_bets = pd.DataFrame(bets)
    if '軸艇番' in df_bets.columns:
        df_bets['軸艇番'] = pd.array(
            [int(x) if pd.notna(x) and x is not None else pd.NA
             for x in df_bets['軸艇番']], dtype='Int64')
    return df_bets


def simulate_returns(pred_df, bets_df, bankroll=10000):
    total_bet = total_ret = 0.0; results = []; cur_bk = bankroll
    for _, bet in bets_df[bets_df['購入'] == True].iterrows():
        grp = pred_df[(pred_df['場名'] == bet['場名']) & (pred_df['レースNo'] == bet['レースNo'])]
        if len(grp) == 0 or grp['返還フラグ'].max() == 1: continue
        wr = grp[grp['着順'] == 1]
        if len(wr) == 0: continue
        winner   = int(wr['艇番'].values[0])
        haraisho = wr['払戻'].values[0]
        ba = float(bet['ベット額']); total_bet += ba
        st_list   = bet['3連単'] if isinstance(bet['3連単'], list) else []
        n_tickets = max(len(st_list), 1); bet_per = ba / n_tickets
        axis_no   = bet['軸艇番']
        hit = (winner == axis_no) and pd.notna(haraisho)
        ret = float(haraisho) * (bet_per / 100) if hit else 0.0
        total_ret += ret; cur_bk += (ret - ba)
        results.append({
            '場名':bet['場名'], '場コード':bet['場コード'], 'レースNo':bet['レースNo'],
            '軸':axis_no, '実際1着':winner, '的中':hit,
            '払戻':haraisho, 'ベット':ba, '点数':n_tickets,
            '回収':ret, '残高':cur_bk, 'EV':bet.get('真期待値',0),
            'バリュースコア':bet.get('バリュースコア',0),
        })
    n_r = len(results); n_h = sum(1 for r in results if r['的中'])
    roi = total_ret / total_bet * 100 if total_bet > 0 else 0
    return {
        'レース数':n_r, '投資':total_bet, '回収':total_ret, '回収率':round(roi,1),
        '的中数':n_h, '的中率':round(n_h/n_r*100,1) if n_r>0 else 0,
        '最終残高':cur_bk, '詳細':pd.DataFrame(results),
    }

# ════════════════════════════════════════════════════════════
# ── フラグ・買い目生成
# ════════════════════════════════════════════════════════════

def recalc_flags(df):
    df = df.copy()
    in_boat  = df['予想進入'] == 1
    low_win2 = df['進入コース複勝率'].fillna(0) < 0.40
    high_std = df.get('ST不利度', pd.Series(0, index=df.index)).fillna(0) > 0.005
    df['イン崩壊フラグ']  = (in_boat & (low_win2 | high_std)).astype(int)
    df['過小評価フラグ']  = (
        (df.get('勝率順位差', pd.Series(0, index=df.index)).fillna(0) > 0.2)
        & (df['人気_IN順位'].fillna(0) > 0.6)
    ).astype(int)
    df['ST狙い目フラグ']  = (
        (df.get('ST順位差', pd.Series(0, index=df.index)).fillna(0) >= 0.20)
        & (df['予測_1着確率'].fillna(0) >= 0.15)
    ).astype(int)
    return df

def build_race_picks(df):
    picks = []
    for (vc, rno), grp in df.groupby(['場コード','レースNo']):
        grp = grp.sort_values('アンサンブルスコア_IN順位').reset_index(drop=True)
        if grp['返還フラグ'].max() == 1: continue

        cfg        = get_venue_config(vc)
        venue_name = cfg['name']
        arashi_thr = cfg.get('荒れ閾値', BASE_CONFIG['荒れ閾値'])
        ev_min     = float(grp['動的EV_MIN'].iloc[0]) if '動的EV_MIN' in grp.columns else cfg['EV_MIN']
        p_min      = cfg.get('P_MIN', BASE_CONFIG['P_MIN'])
        night      = bool(grp['ナイターフラグ'].iloc[0]) if 'ナイターフラグ' in grp.columns else False

        if grp['荒れ指数'].mean() < arashi_thr:
            picks.append({
                '場名':venue_name,'場コード':vc,'レースNo':rno,
                '判定':'❌荒れ指数低（見送り）','軸艇番':'—','軸選手':'—',
                '軸確率':0,'軸真期待値':0,'軸推定オッズ':0,'バリュースコア':0,
                '動的EV_MIN':ev_min,'ナイター':'🌙' if night else '',
                '2着候補':'','3着候補':'','3連単点数':0,
                '3連単①':'','3連単②':'','3連単③':'','3連単④':'','3連単⑤':'','3連単⑥':'',
                '注意フラグ':'荒れ指数低すぎ','イン崩壊':'','過小評価艇':'','ST狙い目艇':'',
            })
            continue

        # バリューフラグ込みの候補選定
        cands = grp[
            (grp['予測_1着確率'] >= p_min) & (grp['真期待値'] >= ev_min)
            & (grp.get('バリューフラグ', pd.Series(1, index=grp.index)) == 1)
        ]
        if len(cands) == 0:
            cands = grp[(grp['予測_1着確率'] >= p_min) & (grp['真期待値'] >= ev_min)]
        if len(cands) == 0:
            ac = grp[grp['予測_1着確率'] >= p_min]
            axis_rows  = [ac.iloc[0]] if len(ac) > 0 else [grp.iloc[0]]
            judge_base = '⚠️EV不足' if len(ac) > 0 else '❌見送り'
        else:
            axis_rows  = [row for _, row in cands.sort_values('真期待値', ascending=False).head(2).iterrows()]
            judge_base = '✅買い'

        ic    = int(grp[grp['予想進入']==1]['イン崩壊フラグ'].max()) if len(grp[grp['予想進入']==1]) > 0 else 0
        uw    = grp[grp['過小評価フラグ']==1]['艇番'].tolist()
        sb    = grp[grp['ST狙い目フラグ']==1]['艇番'].tolist()
        vb    = grp[grp.get('バリューフラグ', pd.Series(0, index=grp.index))==1]['艇番'].tolist() \
                if 'バリューフラグ' in grp.columns else []

        for i, axis_row in enumerate(axis_rows):
            axis_no = int(axis_row['艇番'])
            judge   = judge_base if len(axis_rows)==1 else f'{judge_base}[{"主" if i==0 else "副"}軸]'

            notes = []
            if night:  notes.append('🌙ナイター')
            if ic:     notes.append('⚠️イン崩壊')
            if uw:     notes.append(f'💎過小評価:{uw}番')
            if sb:     notes.append(f'🚀ST狙い目:{sb}番')
            if vb:     notes.append(f'💰バリュー:{vb}番')
            if i == 1: notes.append('（副軸）')

            top3 = [int(b) for b in grp.head(3)['艇番'].tolist()]
            top5 = [int(b) for b in grp.head(5)['艇番'].tolist()]
            sc   = [b for b in top3 if b != axis_no][:2]
            tc   = [b for b in top5 if b != axis_no][:4]
            st   = _build_sanrentan_by_prob(grp, axis_no, sc, tc)

            picks.append({
                '場名':venue_name, '場コード':vc, 'レースNo':rno,
                '判定':judge, '軸艇番':axis_no, '軸選手':str(axis_row['選手名']),
                '軸確率':round(float(axis_row['予測_1着確率']),3),
                '軸真期待値':round(float(axis_row['真期待値']),2),
                '軸推定オッズ':round(float(axis_row['推定オッズ']),1),
                'バリュースコア':round(float(axis_row.get('バリュースコア',0)),3),
                '動的EV_MIN':round(ev_min, 3),
                'ナイター':'🌙' if night else '',
                '2着候補':str(sc), '3着候補':str(tc), '3連単点数':len(st),
                '3連単①':st[0] if len(st)>0 else '', '3連単②':st[1] if len(st)>1 else '',
                '3連単③':st[2] if len(st)>2 else '', '3連単④':st[3] if len(st)>3 else '',
                '3連単⑤':st[4] if len(st)>4 else '', '3連単⑥':st[5] if len(st)>5 else '',
                '注意フラグ':' / '.join(notes) if notes else '—',
                'イン崩壊':'⚠️' if ic else '',
                '過小評価艇':str(uw) if uw else '',
                'ST狙い目艇':str(sb) if sb else '',
            })
    return picks

def build_detail_df(df):
    cols = [
        '場名','レースNo','艇番','選手名','予測_1着確率','真期待値',
        'アンサンブルスコア_IN順位','予想進入','進入コース複勝率','進入コースST',
        'ST順位差','イン崩壊フラグ','過小評価フラグ','ST狙い目フラグ',
        '全国勝率','当地勝率','モーター2率','今期能力指数','勝率順位差','荒れ指数',
        'バリュースコア','バリューフラグ','インプライドプロブ',
        '当地スコア','モータースコア','動的EV_MIN','ナイターフラグ',
        'クラスタ別勝率','着順','払戻','返還フラグ',
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values(['場名','レースNo','アンサンブルスコア_IN順位']).reset_index(drop=True)

# ════════════════════════════════════════════════════════════
# ── Excel出力
# ════════════════════════════════════════════════════════════

def _style_ws(ws, col_widths=None):
    H_FILL = PatternFill('solid', fgColor='1F3864')
    H_FONT = Font(color='FFFFFF', bold=True)
    thin   = Side(style='thin', color='CCCCCC')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for cell in ws[1]:
        cell.fill = H_FILL; cell.font = H_FONT
        cell.alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)
        cell.border = border
    ws.row_dimensions[1].height = 28
    if col_widths:
        for i, w in enumerate(col_widths, 1):
            ws.column_dimensions[get_column_letter(i)].width = w
    else:
        for col in ws.columns:
            ws.column_dimensions[col[0].column_letter].width = 13

def save_picks_excel(picks_df, detail_df, output_file):
    BUY   = PatternFill('solid', fgColor='FFD700')
    WARN  = PatternFill('solid', fgColor='FFB347')
    SKIP  = PatternFill('solid', fgColor='D3D3D3')
    HIT   = PatternFill('solid', fgColor='90EE90')
    FLAG  = PatternFill('solid', fgColor='FF6B6B')
    ST_C  = PatternFill('solid', fgColor='87CEEB')
    VAL_C = PatternFill('solid', fgColor='FFFACD')  # バリュー色（薄黄）
    NIGHT_C = PatternFill('solid', fgColor='E8E8FF') # ナイター色（薄紫）
    thin  = Side(style='thin', color='CCCCCC')
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def _picks_rows(ws, df):
        jc = list(df.columns).index('判定')+1
        fc = list(df.columns).index('イン崩壊')+1 if 'イン崩壊' in df.columns else None
        nc = list(df.columns).index('ナイター')+1 if 'ナイター' in df.columns else None
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            j = str(row[jc-1].value or '')
            for c in row:
                c.border = border
                c.alignment = Alignment(horizontal='center', vertical='center')
            fill = BUY if '✅' in j else WARN if '⚠️' in j else SKIP
            for c in row: c.fill = fill
            if fc and str(row[fc-1].value or '') == '⚠️':
                row[fc-1].fill = FLAG; row[fc-1].font = Font(bold=True)
            if nc and '🌙' in str(row[nc-1].value or ''):
                row[nc-1].fill = NIGHT_C

    def _detail_rows(ws, df):
        cols = df.columns.tolist()
        uf = cols.index('イン崩壊フラグ')+1 if 'イン崩壊フラグ' in cols else None
        sc = cols.index('ST狙い目フラグ')+1 if 'ST狙い目フラグ' in cols else None
        ov = cols.index('過小評価フラグ')+1 if '過小評価フラグ' in cols else None
        hc = cols.index('着順')+1          if '着順'          in cols else None
        vf = cols.index('バリューフラグ')+1 if 'バリューフラグ' in cols else None
        for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
            for c in row:
                c.border = border
                c.alignment = Alignment(horizontal='center', vertical='center')
            if vf and row[vf-1].value == 1:
                for c in row: c.fill = VAL_C
            if uf and row[uf-1].value == 1:
                for c in row: c.fill = FLAG
            elif sc and row[sc-1].value == 1:
                for c in row: c.fill = ST_C
            elif ov and row[ov-1].value == 1:
                for c in row: c.fill = BUY
            if hc and row[hc-1].value == 1:
                row[hc-1].fill = HIT; row[hc-1].font = Font(bold=True)

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        picks_df.to_excel(writer, sheet_name='📋全場買い目', index=False)
        ws = writer.sheets['📋全場買い目']
        _style_ws(ws)
        _picks_rows(ws, picks_df); ws.freeze_panes = 'A2'

        detail_df.to_excel(writer, sheet_name='🔍選手詳細', index=False)
        ws2 = writer.sheets['🔍選手詳細']
        _style_ws(ws2); _detail_rows(ws2, detail_df); ws2.freeze_panes = 'A2'

        # 場ごとにシートを分ける
        for venue_code in picks_df['場コード'].unique():
            vn = VENUE_CODE_MAP.get(str(venue_code).zfill(2), f'場{venue_code}')
            sn = vn[:8]
            vp = picks_df[picks_df['場コード'] == venue_code]
            vp.to_excel(writer, sheet_name=f'{sn}_買', index=False)
            _style_ws(writer.sheets[f'{sn}_買'])
            _picks_rows(writer.sheets[f'{sn}_買'], vp)
            vd = detail_df[detail_df['場名'] == vn]
            if len(vd) > 0:
                vd.to_excel(writer, sheet_name=f'{sn}_詳', index=False)
                _style_ws(writer.sheets[f'{sn}_詳'])
                _detail_rows(writer.sheets[f'{sn}_詳'], vd)
    print(f'[完了] {output_file}')