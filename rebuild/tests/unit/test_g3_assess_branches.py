"""正式G3（B案）：監査 F-04・F-05 への対応。

F-04：Legacy の context・candidate を、Rebuild の build_input_* を使わず、
      notify_arashi.py 2695〜2713行目の式どおりにテスト側で独立に組み立てて比べる。
F-05：境界値が目的の分岐に届くことを、見送り理由が実際に出たことで確かめる。
"""

from __future__ import annotations

import random
import unittest

import x_buyscore as L

from core import buyscore as R
from core.g3_assess import assess_equivalent, assess_with_input
from models.g3_input import BuyScoreInput, MlBonusResult
from tests.unit import g3_support as S


def _legacy_context_independent(score_post, race_type, has_exhibition, market_gap):
    """notify_arashi.py 2695〜2713行目の buyscore_context と同じ式（独立に記述）。"""
    approx_match_index = min(100, score_post * 10.5)
    return {
        "match_index": approx_match_index,
        "match_index_approx": True,
        "race_type": race_type,
        "has_exhibition": has_exhibition,
        "market_gap": market_gap,
        "upset_score": score_post,
        "ex_rank_1st": 0,
    }


def _legacy_four(cand, ctx, cfg):
    score = L.calc_buyscore(dict(cand), dict(ctx), cfg)
    kelly = L.kelly_fraction(cand["prob"], cand["odds"], score, cfg)
    scored = dict(cand, buyscore=score)
    skip = L.check_passthrough([scored], dict(ctx), cfg)
    inv = L.investment_type(score, cand["ev"], cand["odds"], ctx["race_type"])
    return (score, inv, kelly, skip)


def _reason_kind(reason):
    if reason is None:
        return "buy"
    return reason.split("(")[0]


class IndependentLegacyContextTests(unittest.TestCase):
    """F-04：入力の組み立てを Rebuild 側の関数に頼らない比較（approx=True は Legacy の本番の挙動）。"""

    @classmethod
    def setUpClass(cls):
        cls.cfg = S.load_freeze_config()

    def _case(self, rng):
        return dict(
            pred_prob=rng.uniform(0.0, 0.5), pred_ev=rng.uniform(0.5, 6.0),
            pred_odds=rng.uniform(1.0, 120.0), uncertainty=rng.uniform(0.0, 1.0),
            disagreement=rng.uniform(0.0, 1.0), composite=rng.uniform(0.0, 1.0),
            has_exhibition=rng.random() < 0.5, market_gap=rng.uniform(-1.0, 1.0),
            score_post=rng.uniform(0.0, 13.5),
            race_type=rng.choice(S.RACE_TYPES),
        )

    def test_random_10000_independent(self):
        rng = random.Random(20261115)
        failures, kinds = [], set()
        for i in range(S.N_RANDOM):
            k = self._case(rng)
            ev = S.evaluation(race_type=k["race_type"])
            pr = S.prediction(pred_prob=k["pred_prob"], pred_ev=k["pred_ev"], pred_odds=k["pred_odds"])
            bonus = MlBonusResult(score_post=k["score_post"], target_post=(2, 3), applied=False, bonus=0.0)
            bi = BuyScoreInput(eval_id=ev.eval_id, pred_combo=pr.pred_combo, uncertainty=k["uncertainty"],
                               disagreement=k["disagreement"], composite=k["composite"],
                               has_exhibition=k["has_exhibition"], market_gap=k["market_gap"],
                               match_index_approx=True)
            r = assess_with_input(ev, pr, self.cfg, bonus, bi)
            got = (r.buyscore, r.investment_type, r.kelly_fraction, r.skip_reason)
            cand = {"combo": pr.pred_combo, "prob": k["pred_prob"], "ev": k["pred_ev"], "odds": k["pred_odds"],
                    "uncertainty": k["uncertainty"], "disagreement": k["disagreement"],
                    "composite": k["composite"]}
            ctx = _legacy_context_independent(k["score_post"], k["race_type"], k["has_exhibition"],
                                              k["market_gap"])
            want = _legacy_four(cand, ctx, self.cfg)
            kinds.add(_reason_kind(want[3]))
            if got != want:
                failures.append({"i": i, "input": k, "rebuild": got, "legacy": want})
        if failures:
            path = S.save_failures("independent_context", failures)
            self.fail(f"{len(failures)} mismatches; saved {path}")
        # この分布で実際に通った分岐（記録）。買い・BuyScore不足・不確実性高は必ず出る
        self.assertTrue({"buy", "BuyScore不足", "不確実性高"} <= kinds, kinds)


class BranchReachTests(unittest.TestCase):
    """F-05：各見送り理由に実際に届く入力で、Rebuild と Legacy が一致すること。"""

    @classmethod
    def setUpClass(cls):
        cls.cfg = S.load_freeze_config()
        cls.thr = cls.cfg["thresholds"]

    def test_check_passthrough_thresholds_direct(self):
        # 純関数の閾値の前後（buyscore 59.9/60/60.1、EV 1.27/1.28/1.29、match_index 39.9/40/40.1、
        # uncertainty 0.75 前後、composite 0.40 前後、disagreement 0.80 前後）を直接与えて比べる
        base_c = {"combo": "1-2-3", "prob": 0.1, "ev": 2.0, "odds": 20.0, "uncertainty": 0.3,
                  "disagreement": 0.2, "composite": 0.6, "buyscore": 70.0}
        base_x = {"match_index": 60.0, "match_index_approx": False, "has_exhibition": True}
        variants = []
        for v in (59.9, 60.0, 60.1):
            variants.append(({"buyscore": v}, {}))
        for v in (1.27, 1.28, 1.29):
            variants.append(({"ev": v}, {}))
        for v in (39.9, 40.0, 40.1):
            for a in (True, False):
                variants.append(({}, {"match_index": v, "match_index_approx": a}))
        for v in (0.749, 0.75, 0.751):
            variants.append(({"uncertainty": v}, {}))
        for v in (0.39, 0.40, 0.41):
            variants.append(({"composite": v}, {"has_exhibition": False}))
        for v in (0.799, 0.80, 0.801):
            variants.append(({"disagreement": v}, {}))
        seen = set()
        for dc, dx in variants:
            c = dict(base_c, **dc)
            x = dict(base_x, **dx)
            a = R.check_passthrough([dict(c)], dict(x), self.cfg)
            b = L.check_passthrough([dict(c)], dict(x), self.cfg)
            self.assertEqual(a, b, (dc, dx))
            seen.add(_reason_kind(b))
        self.assertEqual(seen, {"buy", "BuyScore不足", "期待値不足", "AI一致指数不足", "不確実性高",
                                "展示なし+信頼度不足", "モデル不一致"})

    def test_assess_with_input_reaches_each_reason(self):
        # assess_with_input の経路（buyscore の計算を含む）で、各見送り理由に届く入力を探して比べる
        rng = random.Random(20261116)
        found = {}
        for _ in range(60000):
            k = dict(pred_prob=rng.uniform(0.05, 0.5), pred_ev=rng.uniform(1.0, 4.0),
                     pred_odds=rng.uniform(3.0, 40.0), uncertainty=rng.uniform(0.0, 0.95),
                     disagreement=rng.uniform(0.0, 0.95), composite=rng.uniform(0.2, 1.0),
                     has_exhibition=rng.random() < 0.5, market_gap=rng.uniform(-0.5, 1.0),
                     score_post=rng.uniform(0.5, 9.0), approx=rng.random() < 0.3)
            ev = S.evaluation()
            pr = S.prediction(pred_prob=k["pred_prob"], pred_ev=k["pred_ev"], pred_odds=k["pred_odds"])
            bonus = MlBonusResult(score_post=k["score_post"], target_post=(2, 3), applied=False, bonus=0.0)
            bi = BuyScoreInput(eval_id=ev.eval_id, pred_combo=pr.pred_combo, uncertainty=k["uncertainty"],
                               disagreement=k["disagreement"], composite=k["composite"],
                               has_exhibition=k["has_exhibition"], market_gap=k["market_gap"],
                               match_index_approx=k["approx"])
            r = assess_with_input(ev, pr, self.cfg, bonus, bi)
            cand = {"combo": pr.pred_combo, "prob": k["pred_prob"], "ev": k["pred_ev"], "odds": k["pred_odds"],
                    "uncertainty": k["uncertainty"], "disagreement": k["disagreement"], "composite": k["composite"]}
            ctx = _legacy_context_independent(k["score_post"], "", k["has_exhibition"], k["market_gap"])
            ctx["match_index_approx"] = k["approx"]
            want = _legacy_four(cand, ctx, self.cfg)
            self.assertEqual((r.buyscore, r.investment_type, r.kelly_fraction, r.skip_reason), want, k)
            found.setdefault(_reason_kind(want[3]), k)
        required = {"buy", "BuyScore不足", "期待値不足", "AI一致指数不足", "不確実性高",
                    "展示なし+信頼度不足", "モデル不一致"}
        self.assertTrue(required <= set(found), sorted(found))

    def test_equivalent_path_reaches_beyond_buyscore(self):
        # 入力の追加なしの経路でも、BuyScore不足より先の分岐に届く入力で既存 assess と一致すること
        rng = random.Random(20261117)
        kinds = set()
        for _ in range(20000):
            ev = S.evaluation(match_index=rng.uniform(0.0, 100.0), upset_score=rng.uniform(0.0, 9.5))
            pr = S.prediction(pred_prob=rng.uniform(0.05, 0.6), pred_ev=rng.uniform(1.0, 5.0),
                              pred_odds=rng.uniform(2.0, 40.0))
            a = R.DefaultBuyEngine().assess(ev, pr, self.cfg)
            b = assess_equivalent(ev, pr, self.cfg)
            self.assertEqual(a, b)
            kinds.add(_reason_kind(a.skip_reason))
        self.assertTrue({"BuyScore不足", "不確実性高"} <= kinds, kinds)
        # 入力の追加なしでは uncertainty が既定値 1.0 になるため、BuyScore を超えた入力は
        # AI一致指数不足か不確実性高で止まる（買いには届かない）。届いた分岐を記録として残す
        self.assertNotIn("buy", kinds)


if __name__ == "__main__":
    unittest.main()


class ContextShapeTests(unittest.TestCase):
    def test_input_context_has_legacy_seven_keys(self):
        # ex_rank_1st は buyscore の計算には使われないが、Legacy と同じ7キーの形であることを確かめる
        from actions.g3_entrypoint import LEGACY_CONTEXT_KEYS
        from core.g3_assess import build_input_context
        ev = S.evaluation()
        bonus = MlBonusResult(score_post=5.0, target_post=(2, 3), applied=False, bonus=0.0)
        bi = BuyScoreInput(eval_id=ev.eval_id, pred_combo="1-2-3", uncertainty=0.5, disagreement=0.5,
                           composite=0.5, has_exhibition=False, market_gap=0.0, match_index_approx=True)
        self.assertEqual(sorted(build_input_context(ev, bonus, bi)), sorted(LEGACY_CONTEXT_KEYS))
