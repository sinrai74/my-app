"""
parsers.py  ── fan / 番組表(B) / 結果(K) ファイルのパーサー
"""

from __future__ import annotations
import re
from pathlib import Path
import pandas as pd
import numpy as np

from config import VENUE_CODE_MAP

ZEN2HAN = str.maketrans("０１２３４５６７８９", "0123456789")

# ════════════════════════════════════════════════════════════
# K 結果ファイル
# ════════════════════════════════════════════════════════════

def parse_k_file(filepath: str | Path) -> pd.DataFrame:
    """K??????.TXT → 1レース1行のDF"""
    with open(filepath, "rb") as f:
        text = f.read().decode("shift_jis", errors="replace")
    ZH = str.maketrans("０１２３４５６７８９－", "0123456789-")
    records: list[dict] = []
    vc: str | None = None
    for line in text.split("\r\n"):
        m = re.match(r"^(\d{2})KBGN$", line.strip())
        if m:
            vc = m.group(1)
            continue
        if re.match(r"^\d{2}KEND$", line.strip()):
            vc = None
            continue
        if vc is None:
            continue
        m2 = re.match(r"\s+(\d{1,2})R\s+(\d-\d-\d)\s+(\d+)", line.translate(ZH))
        if m2:
            p = m2.group(2).split("-")
            records.append({
                "場コード":   vc,
                "場名":       VENUE_CODE_MAP.get(vc, f"場{vc}"),
                "レースNo":   int(m2.group(1)),
                "組番":       m2.group(2),
                "1着":        int(p[0]),
                "2着":        int(p[1]),
                "3着":        int(p[2]),
                "払戻":       int(m2.group(3)),
                "返還フラグ": 0,
            })
    df = pd.DataFrame(records)
    vc_counts = df.groupby("場名").size().to_dict() if len(df) > 0 else {}
    print(f"[K] {Path(filepath).name}: {len(df)}R / {df['場コード'].nunique()}場 {vc_counts}")
    return df


# ════════════════════════════════════════════════════════════
# fan 選手データ
# ════════════════════════════════════════════════════════════

_FAN_FIELDS: list[tuple[str, int]] = [
    ("登番",4),("名前漢字",16),("名前カナ",15),("支部",4),("級",2),
    ("年号",1),("生年月日",6),("性別",1),("年齢",2),("身長",3),
    ("体重",2),("血液型",2),("勝率",4),("複勝率",4),
    ("1着回数",3),("2着回数",3),("出走回数",3),("優出回数",2),("優勝回数",2),("平均ST",3),
]
for _c in range(1, 7):
    _FAN_FIELDS += [
        (f"{_c}コース進入回数", 3), (f"{_c}コース複勝率", 4),
        (f"{_c}コース平均ST", 3),   (f"{_c}コース平均ST順位", 3),
    ]
_FAN_FIELDS += [
    ("前期級",2),("前々期級",2),("前々々期級",2),
    ("前期能力指数",4),("今期能力指数",4),
    ("年",4),("期",1),("算出期間自",8),("算出期間至",8),("養成期",3),
]
for _c in range(1, 7):
    for _r in range(1, 7):
        _FAN_FIELDS.append((f"{_c}コース{_r}着回数", 3))
    for _s in ["F","L0","L1","K0","K1","S0","S1","S2"]:
        _FAN_FIELDS.append((f"{_c}コース{_s}回数", 2))
for _s in ["L0","L1","K0","K1"]:
    _FAN_FIELDS.append((f"コースなし{_s}回数", 2))
_FAN_FIELDS.append(("出身地", 6))
_FAN_RECORD_BYTES = 416


def _to_float(s: str, dec: int) -> float | None:
    try:
        return int(s) / (10 ** dec)
    except (ValueError, TypeError):
        return None


def parse_fan_files(filepaths: list[str | Path]) -> pd.DataFrame:
    """fan*.txt（複数可）→ 選手マスタDF"""
    recs: list[dict] = []
    for fp in filepaths:
        if not Path(fp).exists():
            continue
        raw = Path(fp).read_bytes().decode("shift_jis")
        for line in [l for l in raw.split("\r\n") if l.strip()]:
            b = line.encode("shift_jis")
            if len(b) != _FAN_RECORD_BYTES:
                continue
            rec: dict = {}
            pos = 0
            for name, size in _FAN_FIELDS:
                rec[name] = b[pos:pos+size].decode("shift_jis", errors="replace").strip()
                pos += size
            recs.append(rec)

    df = pd.DataFrame(recs)
    for col, dec in [("勝率",2),("複勝率",1),("平均ST",2),("今期能力指数",2),("前期能力指数",2)]:
        df[col] = df[col].apply(lambda x: _to_float(x, dec))
    for col in ["出走回数","優出回数","優勝回数"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    for c in range(1, 7):
        df[f"{c}コース進入回数"] = pd.to_numeric(df[f"{c}コース進入回数"], errors="coerce").fillna(0)
        df[f"{c}コース複勝率"]   = df[f"{c}コース複勝率"].apply(lambda x: _to_float(x, 1) or 0.0)
        df[f"{c}コース平均ST"]   = df[f"{c}コース平均ST"].apply(lambda x: _to_float(x, 2) or 0.18)
    df["算出期間至"] = pd.to_numeric(df["算出期間至"], errors="coerce")
    df = (df.sort_values("算出期間至", ascending=False)
            .drop_duplicates("登番", keep="first")
            .reset_index(drop=True))
    print(f"[fan] {len(df)} 選手")
    return df


# ════════════════════════════════════════════════════════════
# 番組表（B ファイル）
# ════════════════════════════════════════════════════════════

_RE_VS  = re.compile(r"^(\d{2})BBGN$")
_RE_VN  = re.compile(r"(ボートレース\S+)")
_RE_RC  = re.compile(r"[\s　]*([０-９\d]+)Ｒ")
_RE_PL  = re.compile(r"^([1-6])\s+(\d{4})")
_RE_PF  = re.compile(
    r"^([1-6])\s+(\d{4})(.{4})(\d{2})(.{2})(\d{2})([AB][12])\s+"
    r"([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+"
    r"(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\dSFLK ]+?)\s*$"
)
_RE_DAY  = re.compile(r"第\s*([０-９\d]+)\s*日")
_RE_DATE = re.compile(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日")


def parse_bangumi(filepath: str | Path) -> pd.DataFrame:
    """B??????.TXT → 番組表DF（全場）"""
    raw   = Path(filepath).read_bytes().decode("shift_jis", errors="replace")
    lines = raw.split("\r\n")
    recs: list[dict] = []
    vc = vn = ""
    rno: int | None = None
    nday = rdate = ""

    for line in lines:
        if not line.strip() or line.startswith("STARTB") or line.startswith("-"):
            continue
        if line.lstrip().startswith("艇") or line.lstrip().startswith("番"):
            continue

        m = _RE_VS.match(line.strip())
        if m:
            vc, vn, rno, nday, rdate = m.group(1), "", None, "", ""
            continue
        if re.match(r"^\d{2}BEND$", line.strip()):
            continue

        if not vn and _RE_VN.search(line):
            vn = _RE_VN.search(line).group(1).replace("\u3000", "")
            md = _RE_DAY.search(line)
            dt = _RE_DATE.search(line)
            if md:
                nday = md.group(1).translate(ZEN2HAN)
            if dt:
                rdate = f"{dt.group(1)}-{int(dt.group(2)):02d}-{int(dt.group(3)):02d}"
            continue

        if vn and not nday:
            md = _RE_DAY.search(line)
            if md:
                nday = md.group(1).translate(ZEN2HAN)
        if vn and not rdate:
            dt = _RE_DATE.search(line)
            if dt:
                rdate = f"{dt.group(1)}-{int(dt.group(2)):02d}-{int(dt.group(3)):02d}"

        m = _RE_RC.match(line)
        if m:
            rs = re.search(r"[０-９\d]+", line)
            if rs:
                rno = int(rs.group().translate(ZEN2HAN))
            continue

        if _RE_PL.match(line) and rno is not None:
            pm = _RE_PF.match(line)
            if pm:
                recs.append({
                    "場コード":   vc,
                    "場名":       vn,
                    "節日":       int(nday) if nday else None,
                    "開催日":     rdate,
                    "レースNo":   rno,
                    "艇番":       int(pm.group(1)),
                    "登番":       pm.group(2).strip(),
                    "選手名":     pm.group(3).strip(),
                    "年齢":       int(pm.group(4)),
                    "支部":       pm.group(5).strip(),
                    "体重":       int(pm.group(6)),
                    "級":         pm.group(7).strip(),
                    "全国勝率":   float(pm.group(8)),
                    "全国2率":    float(pm.group(9)),
                    "当地勝率":   float(pm.group(10)),
                    "当地2率":    float(pm.group(11)),
                    "モーターNO": int(pm.group(12)),
                    "モーター2率":float(pm.group(13)),
                    "ボートNO":   int(pm.group(14)),
                    "ボート2率":  float(pm.group(15)),
                    "今節成績":   pm.group(16).strip(),
                })

    df = pd.DataFrame(recs)
    print(
        f"[番組表] {Path(filepath).name}: {len(df)}行 / "
        f"{df['場名'].nunique()}場 / "
        f"{df.groupby(['場コード','レースNo']).ngroups}レース"
    )
    return df


# ════════════════════════════════════════════════════════════
# 結合・結果付与
# ════════════════════════════════════════════════════════════

_FAN_MERGE_COLS: list[str] = (
    ["登番","勝率","複勝率","平均ST","今期能力指数","前期能力指数",
     "名前漢字","支部","級","出走回数","優出回数","優勝回数"]
    + [f"{c}コース進入回数" for c in range(1, 7)]
    + [f"{c}コース複勝率"   for c in range(1, 7)]
    + [f"{c}コース平均ST"   for c in range(1, 7)]
)


def merge_bangumi_fan(bangumi_df: pd.DataFrame, fan_df: pd.DataFrame) -> pd.DataFrame:
    """番組表 + fan を登番で結合"""
    fan_sub = fan_df[[c for c in _FAN_MERGE_COLS if c in fan_df.columns]].copy()
    fan_sub["登番"] = fan_sub["登番"].astype(str).str.strip()
    df = bangumi_df.copy()
    df["登番"] = df["登番"].astype(str).str.strip()
    merged = df.merge(fan_sub, on="登番", how="left", suffixes=("_番組","_fan"))
    if "級_番組" in merged.columns:
        merged["級"] = merged["級_番組"].fillna("")
    merged.drop(
        columns=[c for c in merged.columns if c in ("級_番組","級_fan")],
        inplace=True, errors="ignore"
    )
    merged["選手名"] = merged["選手名"].replace("", None).fillna(merged.get("名前漢字",""))
    return merged


def attach_k_results(merged_df: pd.DataFrame, k_df: pd.DataFrame) -> pd.DataFrame:
    """番組表DFにK結果を付与"""
    rows: list[dict] = []
    for _, r in k_df.iterrows():
        for rank, col in [(1,"1着"),(2,"2着"),(3,"3着")]:
            rows.append({
                "場コード":   str(r["場コード"]),
                "レースNo":   r["レースNo"],
                "艇番":       r[col],
                "着順_k":     rank,
                "払戻":       r["払戻"],
                "人気":       np.nan,
                "返還フラグ": r["返還フラグ"],
            })
    result_long = pd.DataFrame(rows)
    merged = merged_df.copy()
    merged["場コード"] = merged["場コード"].astype(str)
    merged = merged.merge(result_long, on=["場コード","レースNo","艇番"], how="left")

    for (vc, rno), idx in merged.groupby(["場コード","レースNo"]).groups.items():
        sub = merged.loc[idx]
        for seq, i in enumerate(sub[sub["着順_k"].isna()].index, start=4):
            merged.loc[i, "着順_k"] = seq
        for col in ["払戻","返還フラグ"]:
            if col not in merged.columns:
                merged[col] = np.nan
            val = sub[col].dropna().values if col in sub.columns else []
            if len(val) > 0:
                merged.loc[idx, col] = merged.loc[idx, col].fillna(val[0])

    merged["着順"]       = pd.to_numeric(merged["着順_k"], errors="coerce").astype("Int64")
    merged.drop(columns=["着順_k"], inplace=True, errors="ignore")
    merged["払戻"]       = pd.to_numeric(merged["払戻"], errors="coerce")
    merged["人気"]       = merged.get("人気", np.nan)
    merged["返還フラグ"] = merged["返還フラグ"].fillna(0).astype(int)
    print(f"[結果付与] {merged['着順'].notna().sum()} / {len(merged)} 件")
    return merged


# ════════════════════════════════════════════════════════════
# ファイル自動検索ユーティリティ
# ════════════════════════════════════════════════════════════

def find_files(directory: str | Path, pattern: str) -> list[Path]:
    """
    ディレクトリからパターンに合うファイルを列挙。
    Colabが付ける "(2)" などの重複サフィックスにも対応。
    """
    d = Path(directory)
    seen: dict[str, Path] = {}
    for f in sorted(d.iterdir()):
        clean = re.sub(r"\s*\(\d+\)", "", f.name).strip()
        if re.match(pattern, clean, re.I):
            if clean not in seen or f.name == clean:
                seen[clean] = f
    return sorted(seen.values())


def discover_files(directory: str | Path = ".") -> dict[str, list[Path]]:
    """
    指定ディレクトリから fan / 番組表 / K結果 ファイルを自動検索して返す。

    Returns:
        {
          "fan":           [Path, ...],
          "bangumi_all":   [Path, ...],   # 日付昇順
          "bangumi_today": Path | None,   # 最新=今日
          "bangumi_past":  [Path, ...],   # それ以外
          "k_files":       [Path, ...],
        }
    """
    fan_files   = find_files(directory, r"^fan\d{4}\.txt$")
    bangumi_all = find_files(directory, r"^B\d{6}\.TXT$")
    k_files     = find_files(directory, r"^K\d{6}\.TXT$")

    bangumi_today = bangumi_all[-1] if bangumi_all else None
    bangumi_past  = bangumi_all[:-1]

    return {
        "fan":           fan_files,
        "bangumi_all":   bangumi_all,
        "bangumi_today": bangumi_today,
        "bangumi_past":  bangumi_past,
        "k_files":       k_files,
    }
