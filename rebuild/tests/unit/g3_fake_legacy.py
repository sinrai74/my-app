"""正式G3 テスト用の偽の Legacy（写し取りと組み立てのテスト用）。

apply_buyscore は Legacy と同じ形（引数名 candidates・context、局所変数 cfg、
その場での並べ替え、戻り値の辞書）を持つ。計算は x_buyscore の純関数を使う。
"""

from __future__ import annotations

import x_buyscore as L

FAKE_CFG: dict = {}
MODE = {"raise": False, "zero_amount_top": False, "call_twice": False, "ctx_upset_offset": 0.0,
        "force_exhibition_true": False, "alter_return_unc": False, "raise_outside": False,
        "no_cfg": False, "dup_combo": False, "ctx_match_offset": 0.0, "approx_false": False,
        "alter_return_prob": False, "extra_ctx_key": False, "legacy_kelly_offset": 0.0}


def apply_buyscore(candidates, context, venue="", race="", date=""):
    cfg = FAKE_CFG
    if MODE["raise"]:
        raise RuntimeError("fake failure")
    for c in candidates:
        c["buyscore"] = L.calc_buyscore(c, context, cfg)
        c["kelly"] = L.kelly_fraction(c.get("prob", 0), c.get("odds", 0), c["buyscore"], cfg) \
            + MODE["legacy_kelly_offset"]
    candidates.sort(key=lambda x: x["buyscore"], reverse=True)
    reason = L.check_passthrough(candidates, context, cfg)
    if reason:
        return {"passthrough": True, "passthrough_reason": reason, "buy": [], "capital_reason": None}
    buy = [dict(c) for c in candidates]
    if MODE["zero_amount_top"]:
        buy = buy[1:]
    return {"passthrough": False, "passthrough_reason": None, "buy": buy, "capital_reason": None}


def apply_buyscore_nocfg(candidates, context, venue="", race="", date=""):
    """局所変数 cfg を持たない偽物（設定を写し取れない場合の試験用）。"""
    settings = FAKE_CFG
    for c in candidates:
        c["buyscore"] = L.calc_buyscore(c, context, settings)
        c["kelly"] = L.kelly_fraction(c.get("prob", 0), c.get("odds", 0), c["buyscore"], settings)
    candidates.sort(key=lambda x: x["buyscore"], reverse=True)
    reason = L.check_passthrough(candidates, context, settings)
    if reason:
        return {"passthrough": True, "passthrough_reason": reason, "buy": [], "capital_reason": None}
    return {"passthrough": False, "passthrough_reason": None, "buy": [dict(c) for c in candidates],
            "capital_reason": None}


def evaluate_bets(patterns, ml_probs, odds_map, bankroll=10000, target_lanes=None,
                  has_exhibition=True, boats=None, upset_score=0.0, odds_dropped=None,
                  weather=None, venue_num=0, race_number=0, race_date="", venue_name=""):
    """Legacy `_evaluate_bets` の最小の偽物（候補は patterns["combos"] から作る）。"""
    if MODE["raise_outside"]:
        raise KeyError("fake evaluate_bets failure")
    top = []
    combos = list(patterns.get("combos", []))
    if MODE["dup_combo"] and combos:
        combos.append(combos[0])
    for combo in combos:
        o = odds_map.get(combo)
        if not o:
            continue
        p = patterns["probs"][combo]
        top.append({"combo": combo, "prob": p, "ev": p * o, "odds": o, "confidence": 0.5,
                    "uncertainty": patterns["unc"][combo], "disagreement": 0.1, "composite": 0.6})
    if not top:
        return top
    context = {
        "match_index": min(100, upset_score * 10.5), "match_index_approx": True,
        "race_type": patterns.get("race_type", "イン逃げ型"), "has_exhibition": has_exhibition,
        "market_gap": 0.2, "upset_score": upset_score + MODE["ctx_upset_offset"], "ex_rank_1st": 0,
    }
    if MODE["force_exhibition_true"]:
        context["has_exhibition"] = True
    if MODE["ctx_match_offset"]:
        context["match_index"] = context["match_index"] + MODE["ctx_match_offset"]
    if MODE["approx_false"]:
        context["match_index_approx"] = False
    if MODE["extra_ctx_key"]:
        context["extra"] = 1
    apply = apply_buyscore_nocfg if MODE["no_cfg"] else apply_buyscore
    try:
        result = apply(top, context, venue=venue_name, race=str(race_number), date=race_date)
        if MODE["call_twice"]:
            apply([dict(c) for c in top], dict(context))
    except Exception:
        return top
    if result["passthrough"]:
        t = dict(top[0])
        if MODE["alter_return_prob"]:
            t["prob"] = t["prob"] + 0.001
        t.update(purchased=False, skip_reason=result["passthrough_reason"], amount=0)
        return [t]
    out = []
    for c in result["buy"]:
        c = dict(c)
        c.update(purchased=True, skip_reason=None)
        if MODE["alter_return_prob"]:
            c["prob"] = c["prob"] + 0.001
        if MODE["alter_return_unc"]:
            c["uncertainty"] = round(c.get("uncertainty", 0.0) + 0.01, 3)
        out.append(c)
    return out
