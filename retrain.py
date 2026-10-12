"""
retrain.py  ── 最新データで model_all.pkl を自動再学習

GitHub Actions から毎週実行される。
公式サイトから直近4週間のB/Kファイルをダウンロードして再学習する。
"""

from __future__ import annotations
import os
import sys
import gzip
import pickle
import logging
import requests
import warnings
from datetime import date, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

DATA_DIR = Path("retrain_data")
MODEL_PATH = Path("model_all.pkl")
WEEKS_BACK = 4  # 直近4週間分を使用

BASE_URL = "https://www.boatrace.jp/owpc/pc/extra/data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.boatrace.jp/",
}


def download_file(url: str, dest: Path) -> bool:
    """ファイルをダウンロードして保存"""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code == 200 and len(r.content) > 1000:
            dest.write_bytes(r.content)
            return True
    except Exception as e:
        log.debug("ダウンロード失敗: %s %s", url, e)
    return False


def download_race_data(target_date: date) -> tuple[Path | None, Path | None]:
    """指定日のB/KファイルをLZH形式でダウンロードして展開"""
    import subprocess
    import shutil
    date_str = target_date.strftime("%y%m%d")
    year = target_date.strftime("%Y")
    month = target_date.strftime("%m")

    b_lzh = DATA_DIR / f"b{date_str}.lzh"
    k_lzh = DATA_DIR / f"k{date_str}.lzh"
    b_txt = DATA_DIR / f"B{date_str.upper()}.TXT"
    k_txt = DATA_DIR / f"K{date_str.upper()}.TXT"

    # すでに展開済みならスキップ
    if b_txt.exists() and k_txt.exists():
        return b_txt, k_txt

    # ダウンロード
    b_url = f"http://www1.mbrace.or.jp/od2/B/{year}{month}/b{date_str}.lzh"
    k_url = f"http://www1.mbrace.or.jp/od2/K/{year}{month}/k{date_str}.lzh"

    b_ok = download_file(b_url, b_lzh)
    k_ok = download_file(k_url, k_lzh)

    if not b_ok or not k_ok:
        return None, None

    # 【バグ修正】7zの実行パスがWindows専用の固定パス
    # ("C:/Program Files/7-Zip/7z.exe")でハードコードされており、
    # GitHub Actions(Linux)上では常に失敗していた
    # （FileNotFoundError→"LZH展開失敗"となり、再学習データが0件になっていた）。
    # PATH上の7z(Linuxではp7zip-fullが入れる実行ファイル名)をまず探し、
    # 見つからなければ従来のWindows固定パスにフォールバックする。
    seven_zip = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
    if not seven_zip:
        # Windowsでshutil.whichがPATH未登録の7zを見つけられない場合の保険
        win_path = "C:/Program Files/7-Zip/7z.exe"
        if os.path.exists(win_path):
            seven_zip = win_path
    if not seven_zip:
        log.warning("LZH展開失敗: 7zの実行ファイルが見つかりません（PATH・Windows既定パスとも不在）")
        return None, None

    # LZH展開（7zが必要）
    try:
        for lzh, txt in [(b_lzh, b_txt), (k_lzh, k_txt)]:
            if lzh.exists():
                result = subprocess.run(
                    [seven_zip, "x", str(lzh), f"-o{DATA_DIR}", "-y"],
                    capture_output=True, timeout=30
                )
                lzh.unlink(missing_ok=True)
    except Exception as e:
        log.warning("LZH展開失敗: %s", e)
        return None, None

    if b_txt.exists() and k_txt.exists():
        return b_txt, k_txt
    return None, None


def check_label_integrity(merged, n_entries: int) -> None:
    """結果付与後の学習データを検証する。異常があれば例外で止める。"""
    if len(merged) != n_entries:
        raise ValueError(f"結果付与で行数が変わった: {n_entries} → {len(merged)}")
    keys = ["_file_date", "場コード", "レースNo"]
    dup = merged.duplicated(subset=keys + ["艇番"], keep=False)
    if dup.any():
        raise ValueError(f"同じ日・場・レース・艇番の行が重複: {int(dup.sum())}行")
    winners = (merged["着順"] == 1).groupby([merged[k] for k in keys]).sum()
    if (winners > 1).any():
        raise ValueError(f"1着が2艇以上のレース: {int((winners > 1).sum())}件")
    log.info("ラベル検証: %d日 / %dレース / %d行 / 結果なし %dレース",
             merged["_file_date"].nunique(), len(winners), len(merged),
             int((winners == 0).sum()))


def retrain():
    DATA_DIR.mkdir(exist_ok=True)

    # 直近4週間の日付リスト
    today = date.today()
    dates = [today - timedelta(days=i) for i in range(1, WEEKS_BACK * 7 + 1)]

    log.info("データ収集開始: %d日分", len(dates))

    # ファイルダウンロード
    b_files, k_files, file_dates = [], [], []
    for d in dates:
        b, k = download_race_data(d)
        if b and k:
            b_files.append(b)
            k_files.append(k)
            file_dates.append(d.strftime("%Y-%m-%d"))

    log.info("取得成功: B=%d K=%d", len(b_files), len(k_files))

    if len(b_files) < 7:
        log.error("データ不足（%d日分）→ 再学習スキップ", len(b_files))
        sys.exit(1)

    # fanファイルを探す（リポジトリ内）
    fan_files = list(Path(".").glob("fan*.txt")) + list(DATA_DIR.glob("fan*.txt"))

    # データ結合
    sys.path.insert(0, ".")
    from parsers import parse_bangumi, parse_k_file, attach_k_results, parse_fan_files, merge_bangumi_fan
    from features import FEATURE_COLS, is_night_race
    import pandas as pd
    import numpy as np

    # 複数日分を連結するため、取得した日付を両方のDFに付けて結合のキーにする
    # （docs/incidents/2026-10-12_retrain-label-cross-date-mix.md）
    all_b, all_k = [], []
    for bf, kf, fd in zip(b_files, k_files, file_dates):
        try:
            bdf = parse_bangumi(bf)
            kdf = parse_k_file(kf)
            bdf["_file_date"] = fd
            kdf["_file_date"] = fd
            all_b.append(bdf)
            all_k.append(kdf)
        except Exception as e:
            log.warning("パースエラー: %s", e)

    if not all_b:
        log.error("番組表データなし")
        sys.exit(1)

    bangumi_df = pd.concat(all_b, ignore_index=True)
    k_df = pd.concat(all_k, ignore_index=True)

    if fan_files:
        fan_df = parse_fan_files(fan_files)
        merged = merge_bangumi_fan(bangumi_df, fan_df)
    else:
        merged = bangumi_df

    n_entries = len(merged)
    merged = attach_k_results(merged, k_df, date_col="_file_date")
    check_label_integrity(merged, n_entries)
    merged["1着フラグ"]  = (merged["着順"] == 1).astype(int)
    merged["3着内フラグ"] = (merged["着順"] <= 3).astype(int)
    merged["log払戻"]    = np.log1p(merged["払戻"].fillna(0) / 100)
    merged["返還フラグ"] = merged["返還フラグ"].fillna(0).astype(int)

    # 特徴量追加
    for col in ["全国勝率", "全国2率", "当地勝率", "当地2率", "モーター2率"]:
        if col in merged.columns:
            g = merged.groupby(["場コード", "レースNo", "開催日"])[col]
            merged[f"{col}_IN順位"] = g.rank(ascending=False, method="min").astype(float)
            merged[f"{col}_IN偏差"] = (merged[col] - g.transform("mean")).astype("float32")
            merged[f"{col}_MAX差"]  = (g.transform("max") - merged[col]).astype("float32")
            merged[f"{col}_MIN差"]  = (merged[col] - g.transform("min")).astype("float32")
    if "平均ST" in merged.columns:
        g = merged.groupby(["場コード", "レースNo", "開催日"])["平均ST"]
        merged["平均ST_IN順位"] = g.rank(ascending=True, method="min").astype(float)
    if "級" in merged.columns:
        merged["級_数値"] = merged["級"].map({"A1":4,"A2":3,"B1":2,"B2":1}).fillna(2)
    if "出走回数" in merged.columns:
        merged["優出率"] = (merged["優出回数"].fillna(0) / merged["出走回数"].replace(0, np.nan).fillna(1)).astype("float32")
        merged["優勝率"] = (merged["優勝回数"].fillna(0) / merged["出走回数"].replace(0, np.nan).fillna(1)).astype("float32")
    merged["ナイターフラグ"] = merged.apply(
        lambda r: int(is_night_race(str(r["場コード"]), r["レースNo"])), axis=1)

    feats = [c for c in FEATURE_COLS if c in merged.columns]
    log.info("学習データ: %d行 / 特徴量: %d個", len(merged), len(feats))

    # サンプリング（メモリ節約）
    if len(merged) > 100000:
        merged = merged.sample(n=100000, random_state=42)

    # 学習
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import roc_auc_score

    X = merged[feats].fillna(0).values.astype("float32")
    valid = (merged["返還フラグ"] == 0).values
    y_win = merged["1着フラグ"].astype(int).values

    clf_win = CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=20,
            early_stopping=True, n_iter_no_change=20,
            validation_fraction=0.1, random_state=42),
        method="isotonic", cv=2)
    clf_win.fit(X[valid], y_win[valid])
    auc = roc_auc_score(y_win[valid], clf_win.predict_proba(X[valid])[:,1])
    log.info("1着AUC: %.4f", auc)

    y_top3 = merged["3着内フラグ"].astype(int).values
    clf_top3 = CalibratedClassifierCV(
        HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
            max_leaf_nodes=31, min_samples_leaf=20,
            early_stopping=True, random_state=42),
        method="isotonic", cv=2)
    clf_top3.fit(X[valid], y_top3[valid])

    reg_rank = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
        max_leaf_nodes=31, min_samples_leaf=20,
        early_stopping=True, random_state=42)
    reg_rank.fit(X[valid], merged.loc[merged.index[valid], "着順"].astype(float).values)

    win_mask = valid & (merged["1着フラグ"].values == 1) & merged["log払戻"].notna().values
    reg_pay = None
    if win_mask.sum() >= 10:
        reg_pay = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
        reg_pay.fit(X[win_mask], merged.loc[merged.index[win_mask], "log払戻"].astype(float).values)

    models = {
        "feature_names": feats,
        "win":    clf_win,
        "top3":   clf_top3,
        "rank":   reg_rank,
        "payout": reg_pay,
        "ev_clf": None,
    }

    with gzip.open(MODEL_PATH, "wb") as f:
        pickle.dump({"models": models, "feature_cols": feats}, f)

    log.info("✅ 再学習完了 AUC=%.4f → %s", auc, MODEL_PATH)

    # 一時データ削除
    import shutil
    shutil.rmtree(DATA_DIR, ignore_errors=True)


if __name__ == "__main__":
    retrain()
