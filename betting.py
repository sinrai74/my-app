"""
betting.py  ── Kelly基準 / 買い目生成 / シミュレーション
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import BASE_CONFIG, get_venue_config, VALUE_THRESHOLD
from features import is_value_bet


# ════════════════════════════════════════════════════════════
# Kelly 基準ベット額計算
# ════════════════════════════════════════════════════════════

def kelly_bet(
    p_win: float,
    odds: float,
    bankroll: float,
    venue_code: str = "01",
    true_ev: float = 1.0,
    unc: float = 0.0,
) -> float:
    """
    場コードから Kelly_kf / Kelly_max / P_MIN を動的取得してベット額を計算。

    Returns:
        ベット額（100円単位切り捨て）
    """
    cfg      = get_venue_config(venue_code)
    kf       = cfg.get("Kelly_kf",  BASE_CONFIG["Kelly_kf"])
    max_frac = cfg.get("Kelly_max", BASE_CONFIG["Kelly_max"])
    p_min    = cfg.get("P_MIN",     BASE_CONFIG["P_MIN"])

    if p_win < p_min:
        return 0.0
    b = odds - 1
    if b <= 0:
        return 0.0
    kf_full = (b * p_win - (1 - p_win)) / b
    if kf_full <= 0:
        return 0.0

    boost    = min(1.5, true_ev)
    unc_adj  = max(0.5, 1.0 - unc)
    raw_bet  = bankroll * kf_full * kf * boost * unc_adj
    capped   = min(raw_bet, bankroll * max_frac)
    return max(0.0, (capped // 100) * 100)


# ════════════════════════════════════════════════════════════
# 3連単生成（確率順）
# ════════════════════════════════════════════════════════════

def build_sanrentan(grp: pd.DataFrame, axis_no: int, sc: list[int], tc: list[int]) -> list[str]:
    """確率順で3連単を最大6点生成"""
    prob_map = grp.set_index("艇番")["予測_3着内確率"].to_dict()
    scored: list[tuple[float, str]] = []
    for s in sc:
        p_s = prob_map.get(s, 0.1) * 0.6
        for t in tc:
            if t == s:
                continue
            p_t = prob_map.get(t, 0.1) * 0.4
            scored.append((p_s * p_t, f"{axis_no}-{s}-{t}"))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:6]]


# ════════════════════════════════════════════════════════════
# 買い目生成
# ════════════════════════════════════════════════════════════

def generate_bets(race_df: pd.DataFrame, bankroll: float = 10000) -> pd.DataFrame:
    """
    全レースに対して買い目とベット額を生成する。

    判定ロジック:
      1. 荒れ指数が閾値以下 → 見送り
      2. 予測不確実性が高すぎる → 見送り
      3. 動的EV_MIN + P_MIN + バリューフラグで軸候補を選定
      4. 上位2軸まで採用し Kelly 基準でベット額を計算
    """
    bets: list[dict] = []
    cur_bk = bankroll

    for (vc, rno), grp in (
        race_df.sort_values(["場コード","レースNo"])
               .groupby(["場コード","レースNo"], sort=False)
    ):
        if grp["返還フラグ"].max() == 1:
            continue
        grp = grp.sort_values("アンサンブルスコア_IN順位").reset_index(drop=True)

        cfg        = get_venue_config(vc)
        venue_name = cfg["name"]
        arashi_thr = cfg.get("荒れ閾値", BASE_CONFIG["荒れ閾値"])
        p_min      = cfg.get("P_MIN",    BASE_CONFIG["P_MIN"])
        ev_min     = float(grp["動的EV_MIN"].iloc[0]) if "動的EV_MIN" in grp.columns \
                     else cfg.get("EV_MIN", BASE_CONFIG["EV_MIN"])

        def _skip(reason: str) -> dict:
            return {
                "場名":venue_name,"場コード":vc,"レースNo":rno,
                "購入":False,"理由":reason,"軸艇番":None,
                "ベット額":0,"残高":cur_bk,
                "3連単":[],"2連単":[],"ワイド":[],"3連単点数":0,
            }

        if grp["荒れ指数"].mean() < arashi_thr:
            bets.append(_skip("荒れ指数低すぎ（見送り）")); continue
        if "予測不確実性" in grp.columns and grp["予測不確実性"].mean() > 0.45:
            bets.append(_skip("予測不確実性高すぎ（見送り）")); continue

        # バリューフラグ込みで軸候補を絞る
        cands = grp[
            (grp["真期待値"] > ev_min)
            & (grp["予測_1着確率"] > p_min)
            & (grp.get("バリューフラグ", pd.Series(1, index=grp.index)) == 1)
        ]
        if len(cands) == 0:
            cands = grp[(grp["真期待値"] > ev_min) & (grp["予測_1着確率"] > p_min)]
        if len(cands) == 0:
            bets.append(_skip("EV/P/バリュー基準未達")); continue

        top_axes = cands.sort_values("真期待値", ascending=False).head(2)
        n_axes   = len(top_axes)

        for _, axis in top_axes.iterrows():
            axis_no = int(axis["艇番"])
            p       = float(axis["予測_1着確率"])
            odds    = float(axis["推定オッズ"])
            ev      = float(axis["真期待値"])
            unc     = float(axis.get("予測不確実性", 0.0))

            bet = max(kelly_bet(p, odds, cur_bk / n_axes, vc, true_ev=ev, unc=unc), 100)

            top3 = grp.head(3)["艇番"].tolist()
            top5 = grp.head(5)["艇番"].tolist()
            sc   = [b for b in top3 if b != axis_no]
            tc   = [b for b in top5 if b != axis_no]
            st   = build_sanrentan(grp, axis_no, sc, tc)

            bets.append({
                "場名":        venue_name,
                "場コード":    vc,
                "レースNo":    rno,
                "購入":        True,
                "理由":        f"EV={ev:.2f} P={p:.3f} 軸{axis_no} "
                               f"バリュー={axis.get('バリュースコア',0):.2f}",
                "軸艇番":      axis_no,
                "真期待値":    ev,
                "予測_1着確率": p,
                "バリュースコア": float(axis.get("バリュースコア", 0)),
                "ベット額":    int(bet),
                "残高（購入前）": cur_bk,
                "3連単点数":   len(st),
                "3連単":       st,
                "2連単":       [f"{axis_no}-{b}" for b in sc],
                "ワイド":      [f"{min(axis_no,b)}-{max(axis_no,b)}" for b in sc],
            })

        cur_bk = max(
            cur_bk - sum(b["ベット額"] for b in bets[-n_axes:] if b.get("購入")), 0)

    df_bets = pd.DataFrame(bets)
    if "軸艇番" in df_bets.columns:
        df_bets["軸艇番"] = pd.array(
            [int(x) if pd.notna(x) and x is not None else pd.NA
             for x in df_bets["軸艇番"]], dtype="Int64")
    return df_bets


# ════════════════════════════════════════════════════════════
# 買い目ピックス（Excel出力用の整形）
# ════════════════════════════════════════════════════════════

def build_race_picks(df: pd.DataFrame) -> list[dict]:
    """予測DFから買い目ピックスのリストを作成"""
    picks: list[dict] = []

    for (vc, rno), grp in df.groupby(["場コード","レースNo"]):
        grp = grp.sort_values("アンサンブルスコア_IN順位").reset_index(drop=True)
        if grp["返還フラグ"].max() == 1:
            continue

        cfg        = get_venue_config(vc)
        venue_name = cfg["name"]
        arashi_thr = cfg.get("荒れ閾値", BASE_CONFIG["荒れ閾値"])
        ev_min     = float(grp["動的EV_MIN"].iloc[0]) if "動的EV_MIN" in grp.columns \
                     else cfg.get("EV_MIN", BASE_CONFIG["EV_MIN"])
        p_min      = cfg.get("P_MIN", BASE_CONFIG["P_MIN"])
        night      = bool(grp["ナイターフラグ"].iloc[0]) if "ナイターフラグ" in grp.columns else False

        def _skip_row(reason: str) -> dict:
            return {
                "場名":venue_name,"場コード":vc,"レースNo":rno,
                "判定":f"❌{reason}","軸艇番":"—","軸選手":"—",
                "軸確率":0,"軸真期待値":0,"軸推定オッズ":0,"バリュースコア":0,
                "動的EV_MIN":ev_min,"ナイター":"🌙" if night else "",
                "2着候補":"","3着候補":"","3連単点数":0,
                **{f"3連単{i}":"" for i in ["①","②","③","④","⑤","⑥"]},
                "注意フラグ":reason,"イン崩壊":"","過小評価艇":"","ST狙い目艇":"",
            }

        if grp["荒れ指数"].mean() < arashi_thr:
            picks.append(_skip_row("荒れ指数低（見送り）")); continue

        cands = grp[
            (grp["予測_1着確率"] >= p_min) & (grp["真期待値"] >= ev_min)
            & (grp.get("バリューフラグ", pd.Series(1, index=grp.index)) == 1)
        ]
        if len(cands) == 0:
            cands = grp[(grp["予測_1着確率"] >= p_min) & (grp["真期待値"] >= ev_min)]
        if len(cands) == 0:
            ac = grp[grp["予測_1着確率"] >= p_min]
            axis_rows  = [ac.iloc[0]] if len(ac) > 0 else [grp.iloc[0]]
            judge_base = "⚠️EV不足" if len(ac) > 0 else "❌見送り"
        else:
            axis_rows  = [r for _, r in cands.sort_values("真期待値", ascending=False).head(2).iterrows()]
            judge_base = "✅買い"

        ic  = int(grp[grp["予想進入"]==1]["イン崩壊フラグ"].max()) if len(grp[grp["予想進入"]==1]) > 0 else 0
        uw  = grp[grp["過小評価フラグ"]==1]["艇番"].tolist()
        sb  = grp[grp.get("ST狙い目フラグ", pd.Series(0, index=grp.index))==1]["艇番"].tolist() \
              if "ST狙い目フラグ" in grp.columns else []
        vb  = grp[grp.get("バリューフラグ",  pd.Series(0, index=grp.index))==1]["艇番"].tolist() \
              if "バリューフラグ"  in grp.columns else []
        # 高配当フラグ・スコア
        hpf = grp[grp.get("高配当フラグ", pd.Series(0, index=grp.index))==1]["艇番"].tolist() \
              if "高配当フラグ" in grp.columns else []
        race_hp_score = float(grp.get("高配当スコア", pd.Series(0, index=grp.index)).fillna(0).max()) \
                        if "高配当スコア" in grp.columns else 0.0

        for i, axis_row in enumerate(axis_rows):
            axis_no = int(axis_row["艇番"])
            # 高配当期待が高い場合は判定を強化
            jb = ("✅🔥高配当期待" if race_hp_score >= 1.5 and "✅" in judge_base else judge_base)
            judge = jb if len(axis_rows) == 1 \
                      else f"{jb}[{'主' if i==0 else '副'}軸]"

            notes: list[str] = []
            if night:              notes.append("🌙ナイター")
            if ic:                 notes.append("⚠️イン崩壊")
            if uw:                 notes.append(f"💎過小評価:{uw}番")
            if sb:                 notes.append(f"🚀ST狙い目:{sb}番")
            if vb:                 notes.append(f"💰バリュー:{vb}番")
            if hpf:                notes.append(f"🔥高配当期待(≥¥8,000):{hpf}番")
            if race_hp_score>=1.0: notes.append(f"💹高配当スコア:{race_hp_score:.2f}")
            if i == 1:             notes.append("（副軸）")

            top3 = [int(b) for b in grp.head(3)["艇番"].tolist()]
            top5 = [int(b) for b in grp.head(5)["艇番"].tolist()]
            sc   = [b for b in top3 if b != axis_no][:2]
            tc   = [b for b in top5 if b != axis_no][:4]
            st   = build_sanrentan(grp, axis_no, sc, tc)

            picks.append({
                "場名":        venue_name,
                "場コード":    vc,
                "レースNo":    rno,
                "判定":        judge,
                "軸艇番":      axis_no,
                "軸選手":      str(axis_row["選手名"]),
                "軸確率":      round(float(axis_row["予測_1着確率"]),  3),
                "軸真期待値":  round(float(axis_row["真期待値"]),       2),
                "軸推定オッズ":round(float(axis_row["推定オッズ"]),     1),
                "バリュースコア": round(float(axis_row.get("バリュースコア",0)), 3),
                "高配当スコア":   round(race_hp_score, 3),
                "予測払戻":       round(float(axis_row.get("予測払戻", axis_row.get("推定オッズ",10)*100)), 0),
                "高配当フラグ":   int(axis_row.get("高配当フラグ", 0)),
                "動的EV_MIN":  round(ev_min, 3),
                "ナイター":    "🌙" if night else "",
                "2着候補":     str(sc),
                "3着候補":     str(tc),
                "3連単点数":   len(st),
                "3連単①":     st[0] if len(st) > 0 else "",
                "3連単②":     st[1] if len(st) > 1 else "",
                "3連単③":     st[2] if len(st) > 2 else "",
                "3連単④":     st[3] if len(st) > 3 else "",
                "3連単⑤":     st[4] if len(st) > 4 else "",
                "3連単⑥":     st[5] if len(st) > 5 else "",
                "注意フラグ":  " / ".join(notes) if notes else "—",
                "イン崩壊":    "⚠️" if ic else "",
                "過小評価艇":  str(uw) if uw else "",
                "ST狙い目艇":  str(sb) if sb else "",
            })

    return picks


def build_detail_df(df: pd.DataFrame) -> pd.DataFrame:
    """詳細分析用DFを整形"""
    cols = [
        "場名","レースNo","艇番","選手名","予測_1着確率","真期待値",
        "アンサンブルスコア_IN順位","予想進入","進入コース複勝率","進入コースST",
        "ST順位差","イン崩壊フラグ","過小評価フラグ","ST狙い目フラグ",
        "全国勝率","当地勝率","モーター2率","今期能力指数","勝率順位差","荒れ指数",
        "バリュースコア","バリューフラグ","インプライドプロブ",
        "当地スコア","モータースコア","動的EV_MIN","ナイターフラグ",
        "クラスタ別勝率","着順","払戻","返還フラグ",
        # 高配当関連（★追加）
        "高配当スコア","予測払戻","高配当フラグ",
    ]
    cols = [c for c in cols if c in df.columns]
    return (df[cols]
            .sort_values(["場名","レースNo","アンサンブルスコア_IN順位"])
            .reset_index(drop=True))


# ════════════════════════════════════════════════════════════
# シミュレーション
# ════════════════════════════════════════════════════════════

def simulate_returns(
    pred_df: pd.DataFrame,
    bets_df: pd.DataFrame,
    bankroll: float = 10000,
) -> dict:
    total_bet = total_ret = 0.0
    results: list[dict] = []
    cur_bk = bankroll

    for _, bet in bets_df[bets_df["購入"] == True].iterrows():
        grp = pred_df[
            (pred_df["場名"]    == bet["場名"])
            & (pred_df["レースNo"] == bet["レースNo"])
        ]
        if len(grp) == 0 or grp["返還フラグ"].max() == 1:
            continue
        wr = grp[grp["着順"] == 1]
        if len(wr) == 0:
            continue

        winner   = int(wr["艇番"].values[0])
        haraisho = wr["払戻"].values[0]
        ba       = float(bet["ベット額"])
        total_bet += ba

        st_list   = bet["3連単"] if isinstance(bet["3連単"], list) else []
        n_tickets = max(len(st_list), 1)
        bet_per   = ba / n_tickets
        axis_no   = bet["軸艇番"]
        hit       = (winner == axis_no) and pd.notna(haraisho)
        ret       = float(haraisho) * (bet_per / 100) if hit else 0.0
        total_ret += ret
        cur_bk    += (ret - ba)

        results.append({
            "場名":        bet["場名"],
            "場コード":    bet["場コード"],
            "レースNo":    bet["レースNo"],
            "軸":          axis_no,
            "実際1着":     winner,
            "的中":        hit,
            "払戻":        haraisho,
            "ベット":      ba,
            "点数":        n_tickets,
            "回収":        ret,
            "残高":        cur_bk,
            "EV":          bet.get("真期待値", 0),
            "バリュースコア": bet.get("バリュースコア", 0),
        })

    n_r = len(results)
    n_h = sum(1 for r in results if r["的中"])
    roi = total_ret / total_bet * 100 if total_bet > 0 else 0.0

    return {
        "レース数":  n_r,
        "投資":      total_bet,
        "回収":      total_ret,
        "回収率":    round(roi, 1),
        "的中数":    n_h,
        "的中率":    round(n_h / n_r * 100, 1) if n_r > 0 else 0,
        "最終残高":  cur_bk,
        "詳細":      pd.DataFrame(results),
    }
