"""
config.py  ── 全場設定・定数
"""

# ── ベース設定（全場共通デフォルト値）────────────────────────
BASE_CONFIG: dict = {
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
    "荒れ閾値":     0.50,
    "ST_DIFF_MIN":  0.20,
    "ナイター補正": 1.00,
    "風補正感度":   0.05,
    "干潮補正":     0.00,
}

# ── 場ごとの差分（BASEから変えたい項目のみ記述）──────────────
COURSE_DIFF: dict = {
    "01": {"name":"桐生",   "1コース補正":1.40, "当地重み":1.40, "荒れ閾値":0.15},
    "02": {"name":"江戸川", "1コース補正":1.05, "2コース補正":1.05, "3コース補正":1.02,
                            "4コース補正":1.00, "5コース補正":0.96, "6コース補正":0.90,
                            "EV_MIN":1.60, "PAY_CAP":100000, "当地重み":1.80,
                            "Kelly_kf":0.45, "Kelly_max":0.08, "荒れ閾値":0.80},
    "03": {"name":"戸田",   "1コース補正":1.42, "4コース補正":0.82, "5コース補正":0.74,
                            "6コース補正":0.68, "当地重み":1.45, "荒れ閾値":0.20},
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
    "24": {"name":"大村",   "1コース補正":1.45, "EV_MIN":1.40, "荒れ閾値":0.35},
}

# ── 逆引きマップ ──────────────────────────────────────────────
VENUE_CODE_MAP: dict = {k: v["name"] for k, v in COURSE_DIFF.items()}
VENUE_NAME_MAP: dict = {v["name"]: k for k, v in COURSE_DIFF.items()}

# ── ナイター開催場 ────────────────────────────────────────────
NIGHT_RACE_VENUES: set = {"04","06","12","17","20","21","22","23","24"}

# ── バリューベット閾値 ────────────────────────────────────────
VALUE_THRESHOLD: float = 1.20   # model_prob > implied_prob × 1.20


def get_venue_config(venue_code: str) -> dict:
    """場コードから完全設定を返す（BASE + 差分マージ）"""
    config = BASE_CONFIG.copy()
    diff   = COURSE_DIFF.get(str(venue_code).zfill(2), {})
    config.update({k: v for k, v in diff.items() if k != "name"})
    config["name"] = diff.get("name", f"場{venue_code}")
    return config


def print_config_comparison(venue_codes: list | None = None) -> None:
    """全場設定の比較表を表示"""
    if venue_codes is None:
        venue_codes = sorted(COURSE_DIFF.keys())
    skip = {"ナイター補正","風補正感度","干潮補正"}
    keys = [k for k in BASE_CONFIG if k not in skip]
    print("項目\tBASE\t" + "\t".join(COURSE_DIFF[c]["name"] for c in venue_codes))
    for k in keys:
        row = [k, str(BASE_CONFIG[k])]
        for c in venue_codes:
            val = get_venue_config(c)[k]
            row.append(f"{val}{'★' if val != BASE_CONFIG[k] else ''}")
        print("\t".join(row))
