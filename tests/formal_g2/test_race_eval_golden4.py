"""正式G2：golden4 の固定入力による評価段階の回帰テスト（ローカル専用）。

用途の制限：golden4 は評価段階（L → Ver4Engine.evaluate／凍結版 v4 → compare）の
回帰テストにだけ使う。正式G2 の入力の証拠・2026-10-01 の代替・同日の再現データではない。
golden4 の expected との突き合わせは、生成時のデータに依存するため合格条件にしない。

実行（リポジトリ直下で）：
  PYTHONPATH="rebuild:." python -P -B -m unittest discover -s tests/formal_g2 -t tests/formal_g2
"""

import dataclasses
import glob
import json
import os
import unittest

ROOT = os.path.realpath(os.getcwd())
INPUTS = os.path.join(ROOT, "rebuild", "tests", "regression", "golden4", "inputs")


@dataclasses.dataclass
class _Boat:
    data: dict


class _FakeFetcher:
    def __init__(self):
        self.calls = []


class _FakeProvider:
    """golden4 の入力から Race と L を返す（出走表の取得経路は通らない）。"""

    def __init__(self, race, boats, race_grade, fetcher):
        self._race = race
        self._boats = boats
        self._fetcher = fetcher
        self._cache = {}
        key = (race.race_date, race.venue_num, race.race_number)
        self._cache[key] = ({"race_grade_number": race_grade}, [_Boat(b) for b in boats])

    def resolve_race(self, race_date, venue_num, race_number):
        self._fetcher.calls.append(race_date)
        return self._race

    def resolve_boats(self, race_date, venue_num, race_number):
        return [dict(b) for b in self._boats]


class Golden4EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from formal_g2 import race_eval
        from formal_g2.instrument import Instrument
        from x_asahi_scoring import load_asahi_config

        cls.race_eval = race_eval
        cls.comparator = race_eval.load_comparator(ROOT)
        cls.cfg = load_asahi_config()
        cls.instrument = Instrument(ROOT)
        cls.files = sorted(glob.glob(os.path.join(INPUTS, "*.json")))
        cls.preload = race_eval.preload_venue_caches(ROOT, "golden4")

    def test_all_100_inputs_diff_zero_and_featureset_ok(self):
        from models.race import Race
        from adapters.providers import _grade_number_to_str

        self.assertEqual(len(self.files), 100)
        failures = []
        for path in self.files:
            with open(path, encoding="utf-8") as f:
                inp = json.load(f)
            date, venue_s, race_s = inp["eval_id"].split("_")
            race = Race(
                race_date=date, venue_num=inp["venue_num"], venue_name=inp["venue"] or "",
                race_number=int(race_s), close_time="", is_night=inp["is_night"], entries=(),
                grade=_grade_number_to_str(int(inp["race_grade"])),
            )
            fetcher = _FakeFetcher()
            provider = _FakeProvider(race, inp["boats"], int(inp["race_grade"]), fetcher)
            ctx = self.race_eval.RunContext(ROOT, date, [], self.cfg, self.comparator,
                                            self.instrument, provider, fetcher, self.preload)
            rec = self.race_eval.evaluate_race(
                ctx, {"eval_id": inp["eval_id"], "venue_num": inp["venue_num"], "race_number": int(race_s)}
            )
            if rec.get("result_category") != "diff_zero":
                failures.append((inp["eval_id"], rec.get("result_category"), rec.get("exception"),
                                 (rec.get("diffs") or [])[:3]))
                continue
            if rec["values"]["rebuild"] is None:
                failures.append((inp["eval_id"], "rebuild values not serializable"))
            if not rec["d13_2"]["venue_caches_match_preload"]:
                failures.append((inp["eval_id"], "venue caches differ from preload", rec["preload_match"]))
        self.assertTrue(self.preload["ok"], self.preload.get("problems"))
        self.assertEqual(failures, [])


if __name__ == "__main__":
    unittest.main()
