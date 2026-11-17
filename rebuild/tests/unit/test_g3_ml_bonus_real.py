"""正式G3（B案）N-10：実際の出走データ（golden4 の100件）と実際の ML モデルで、加点の経路を通す。

- 入力：golden4 の実際の boats（Ver4 の固定入力）。Rebuild の upset は core.upset.calculate_upset_score、
  ML の確率は Legacy の `_predict_win_prob`（model_all.pkl。境界の中で、作業フォルダに置いて読む）。
- 事実（このテストで固定する）：100件のうち加点が起きるのは 20260704_12_08 の1件だけ。
  そのレースは、実際の `_evaluate_bets` が「軸候補なし（実力差不足）」で候補を出さない（no_candidate）。
  したがって、このデータでは「加点ありの購入」は確認できない（未確認として残す）。
- オッズの実データは無い。このレースは候補の生成の前段で見送られるので、オッズは結果に影響しない。
- 実行には requirements-ml.txt の版（scikit-learn 1.5.2・lightgbm 4.5.0）が要る。読めなければ skip する
  （理由を表示する。合格の扱いにしない）。
- joblib が物理コア数を数えるための lscpu を、境界の外で先に済ませる（計算結果は変えない）。
  境界の中で外部コマンドを起動すると違反になるため。
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import unittest

from actions import g3_entrypoint as G
from core.upset import calculate_upset_score
from shadow.g3_isolation import G3Isolation
from tests.unit import g3_support as S

GOLDEN4 = S.REBUILD_DIR / "tests/regression/golden4"
BONUS_RACE = "20260704_12_08"


def _model_loadable() -> str:
    try:
        import sklearn  # noqa: F401
        import lightgbm  # noqa: F401
        from joblib.externals.loky import cpu_count
    except ImportError as exc:
        return f"ML packages missing: {exc}"
    import sklearn as _sk
    if _sk.__version__ != "1.5.2":
        return f"scikit-learn {_sk.__version__} != 1.5.2 (requirements-ml.txt)"
    cpu_count(only_physical_cores=True)
    return ""


_SKIP = _model_loadable()


class _Factors:
    def __init__(self, f):
        self.f = f

    def get_venue_course_factor(self, venue_name, course):
        return {"factor": self.f[str(course)]}


@unittest.skipIf(_SKIP, _SKIP)
class RealMlBonusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import notify_arashi as N

        cls.N = N
        cls.cfg = json.loads((GOLDEN4 / "asahi_config_freeze.json").read_text(encoding="utf-8"))
        manifest = json.loads((GOLDEN4 / "manifest.json").read_text(encoding="utf-8"))
        fields = {f.name for f in dataclasses.fields(N.BoatInfo)}
        cls.races = {}
        with S.fake_token_env(), G3Isolation({"model_all.pkl": S.REPO_DIR / "model_all.pkl"}) as iso:
            assert iso.established, iso.errors
            for e in manifest["entries"]:
                inp = json.loads((GOLDEN4 / "inputs" / f"{e['eval_id']}.json").read_text(encoding="utf-8"))
                up = calculate_upset_score(
                    boats=inp["boats"], config=cls.cfg, race_grade=inp["race_grade"],
                    venue_num=inp["venue_num"], is_night=inp["is_night"],
                    venue_stats=_Factors(inp["danger_venue_factors"]["course_rentai"]))
                boats = [N.BoatInfo(**{k: v for k, v in b.items() if k in fields}) for b in inp["boats"]]
                cls.races[e["eval_id"]] = (up, boats, N._predict_win_prob(boats))
        cls.prep = iso.summary()

    def test_prep_isolation_and_model(self):
        self.assertTrue(self.prep["verified"], self.prep)
        self.assertEqual(self.prep["network_attempts_observed"], 0)
        self.assertEqual(len(self.races), 100)
        self.assertTrue(all(ml for _u, _b, ml in self.races.values()), "ML prediction failed")

    def test_bonus_occurs_only_in_one_race(self):
        bonus = sorted(eid for eid, (_u, _b, ml) in self.races.items()
                       if max(v for k, v in ml.items() if k != 1) > ml[1])
        self.assertEqual(bonus, [BONUS_RACE])

    def test_bonus_race_through_real_evaluate_bets(self):
        up, boats, ml = self.races[BONUS_RACE]
        odds = {f"{a}-{b}-{c}": 20.0 + i * 0.7
                for i, (a, b, c) in enumerate(itertools.permutations(range(1, 7), 3))}
        ev = S.evaluation(upset_score=up.upset_score, win_probs=ml)
        with S.fake_token_env():
            rec = G.evaluate_race_g3(
                evaluation=ev, upset=up, legacy_boats=boats, odds_map=odds,
                weather=self.N.WeatherInfo(), buy_config={"_version": "unused"},
                evaluate_bets=self.N._evaluate_bets, generate_patterns=self.N._generate_patterns,
                isolation_factory=S.iso_factory())
        b = rec["ml_bonus"]
        best = max(v for k, v in ml.items() if k != 1)
        self.assertTrue(b["applied"])
        self.assertEqual(b["score_pre"], up.upset_score)
        self.assertEqual(b["bonus"], min(best * 8.0, 4.0))
        self.assertEqual(b["score_post"], up.upset_score + b["bonus"])
        self.assertEqual(rec["sequence"], ["ml_bonus", "generate_patterns", "evaluate_bets"])
        self.assertEqual(rec["patterns"]["input_score"], b["score_post"])
        self.assertEqual(rec["patterns"]["input_target"], list(b["target_post"]))
        # 実際の Legacy は候補を出さない（軸候補なし）。購入のケースはこのデータには無い
        self.assertEqual(rec["outcome"], G.OUTCOME_NO_CANDIDATE)
        self.assertIsNone(rec["match"])
        self.assertTrue(rec["isolation"]["verified"], rec["isolation"])
        self.assertEqual(rec["isolation"]["network_attempts_observed"], 0)


if __name__ == "__main__":
    unittest.main()
