"""今の朝刊（ルール方式）との比較（Project 12）。

今の朝刊は、毎朝の出力（ranking_cache.json）を my-app の git の履歴に残している。各レース日について
最初の commit（多くは朝 7〜10 時 JST）の出力を、その日の朝の予想として使う（本当の事前予想）。

比べる2つの課題（同じ日・同じレースの集合で、上位 10 レースの的中の割合を比べる）：
- 1号艇の危険（all_danger）：1号艇が1着にならない。新モデルの点数は 1 − p_win_1。
- 万舟（all_manshuu）：3連単の確定配当が 10,000 円以上。新モデルの点数は、実際の結果が新モデルの
  3連単の上位20通りに入らない確率（1 − 上位20通りの確率の合計）。オッズは使わない。

  python -m asahi_ai.compare_legacy extract --repo MY_APP_GIT_DIR --out signals.json
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess

import numpy as np
import pandas as pd


def extract(repo: str) -> list[dict]:
    log = subprocess.run(["git", "-C", repo, "log", "--reverse", "--format=%H %cI", "--", "ranking_cache.json"],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    seen = set()
    rows = []
    for line in filter(None, log):
        h, t = line.split()
        out = subprocess.run(["git", "-C", repo, "show", f"{h}:ranking_cache.json"], capture_output=True, text=True)
        try:
            d = json.loads(out.stdout)
        except json.JSONDecodeError:
            continue
        date = d.get("date")
        if not date or date in seen:
            continue
        seen.add(date)
        for kind in ("all_danger", "all_manshuu"):
            for x in d.get(kind) or []:
                rows.append({"date": date, "kind": kind, "venue": f"{int(x['venue_num']):02d}",
                             "race_no": int(x["race"]), "score": x.get("score"), "commit": h, "committed_at": t})
    return rows


def compare(signals: pd.DataFrame, pred: pd.DataFrame, races: pd.DataFrame, results: pd.DataFrame,
            payouts: pd.DataFrame, top_n: int = 10) -> dict:
    """pred：新モデルの予測（race_id・p_win_1〜6・tri）。同じ日の、両方が予想したレースで比べる。"""
    r = races[["race_id", "date", "venue", "race_no"]].copy()
    r["d8"] = pd.to_datetime(r["date"]).dt.strftime("%Y%m%d")
    s = signals.merge(r, left_on=["date", "venue", "race_no"], right_on=["d8", "venue", "race_no"],
                      suffixes=("", "_r"))
    win1 = results[(results["lane"] == 1)].set_index("race_id")["finish_position"]
    tp = payouts[(payouts["bet_type"] == "trifecta") & (payouts["payout_kind"] == "hit")]
    manshuu = tp.groupby("race_id")["payout_yen"].max() >= 10000
    p = pred.set_index("race_id")
    from . import model
    act = model.actual_trifecta(results.merge(r[["race_id"]], on="race_id"))
    out = {}
    for kind, label in (("all_danger", "boat1_lost"), ("all_manshuu", "manshuu")):
        rows = collections.defaultdict(lambda: {"legacy_hits": 0, "new_hits": 0, "n": 0, "days": 0})
        res = {"legacy_hits": 0, "new_hits": 0, "picked": 0, "days": 0}
        for d8, g in s[s["kind"] == kind].groupby("date"):
            g = g[g["race_id"].isin(p.index)]
            if label == "boat1_lost":
                g = g[g["race_id"].isin(win1.index)]
                y = (win1.reindex(g["race_id"]) != 1).to_numpy()
                new_score = 1 - p.loc[g["race_id"], "p_win_1"].to_numpy()
            else:
                g = g[g["race_id"].isin(manshuu.index)]
                y = manshuu.reindex(g["race_id"]).fillna(False).to_numpy()
                tri = np.stack(p.loc[g["race_id"], "tri"].to_numpy())
                new_score = 1 - np.sort(tri, axis=1)[:, -20:].sum(1)
            if len(g) < top_n:
                continue
            k = top_n
            leg = np.argsort(-g["score"].to_numpy(), kind="stable")[:k]
            new = np.argsort(-new_score, kind="stable")[:k]
            res["legacy_hits"] += int(y[leg].sum())
            res["new_hits"] += int(y[new].sum())
            res["picked"] += k
            res["days"] += 1
            res.setdefault("base_rate_sum", 0)
            res["base_rate_sum"] += float(y.mean())
        if res["picked"]:
            res["legacy_precision"] = res["legacy_hits"] / res["picked"]
            res["new_precision"] = res["new_hits"] / res["picked"]
            res["base_rate"] = res.pop("base_rate_sum") / res["days"]
        out[label] = res
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--repo", required=True)
    e.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = extract(a.repo)
    json.dump(rows, open(a.out, "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps({"rows": len(rows), "days": len({r["date"] for r in rows})}))


if __name__ == "__main__":
    main()
