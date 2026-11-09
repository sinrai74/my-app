"""§6.3 の5・6（GPT判断：確定）：100件目で到達を確定する。101件目以降は数えず、取り消す材料にもしない。

到達と正式な達成（d5：正本の保存の後）は別のもの。到達の後は新しい区間を作らない（GPT 指示、2026-11-08）。
"""
import json
import os
import tempfile
import unittest

from formal_g2.aggregate import main as agg
from test_aggregate import BASELINE, BASELINE_SHA, FORMAL_OK, PERMS, _instrument_ok, _race, _write_run


def E(n):
    """test_aggregate._programs_for がラベルの n 番目に付ける eval_id。"""
    return f"20990101_01_{n:02d}"


def _labels(n, prefix="r"):
    return [f"{prefix}{i:03d}" for i in range(1, n + 1)]


def _inst(n_races):
    inst = _instrument_ok()
    inst["monitor_state"]["thread_checks"] = 2 * n_races + 2  # W7：thread_checks ≥ 2N+2
    return inst


def _aggregate(races, targets):
    with tempfile.TemporaryDirectory() as base:
        _write_run(base, 1, 1, "20990101", races, targets, run_overrides={"instrument": _inst(len(races))})
        v = [agg.verify_run(os.path.join(base, x)) for x in sorted(os.listdir(base))]
        self_check = [x for x in v if not x["verified"]]
        assert not self_check, self_check[0]["problems"]
        return agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)


def _clean(n):
    labels = _labels(n)
    return _aggregate([_race(x) for x in labels], labels), labels


class StreakReachTests(unittest.TestCase):
    def test_99_no_reach(self):
        res, _ = _clean(99)
        self.assertIsNone(res["reach"])
        self.assertIsNone(res["after_reach"])
        self.assertEqual(res["best_segment"], 99)
        self.assertEqual(res["g2_status"], "undetermined")
        self.assertEqual(res["segments"], [{"start": E(1), "count": 99, "ended_by": None}])

    def test_100_one_reach_record_identifying_the_100th(self):
        res, labels = _clean(100)
        r = res["reach"]
        self.assertEqual(r["event_type"], "g2_streak_100_reached")
        self.assertEqual(r["status"], "reached_not_yet_formally_achieved")
        self.assertEqual(r["streak_count"], 100)
        self.assertEqual(r["first_race_key"], E(1))
        self.assertEqual(r["hundredth_race_key"], E(100))
        self.assertEqual(r["race_date"], "20990101")
        self.assertEqual(str(r["evaluation_run_id"]), "1")
        self.assertEqual(r["hundredth_race_closed_at"], "2099-01-01 11:39:00")  # 対象一覧の値をそのまま使う
        self.assertIn("hundredth_race_closed_at", r)
        self.assertEqual(res["segments"], [{"start": E(1), "count": 100, "ended_by": "reached_100",
                                            "hundredth": E(100)}])
        self.assertEqual(res["best_segment"], 100)
        self.assertEqual(res["g2_status"], "reached_100_unconfirmed")
        self.assertEqual(res["after_reach"], {"items": 0, "added_not_counted": 0})

    def test_101_and_150_stay_at_100(self):
        for n in (101, 150):
            with self.subTest(n=n):
                res, _ = _clean(n)
                self.assertEqual(res["best_segment"], 100)
                self.assertEqual(res["reach"]["hundredth_race_key"], E(100))
                self.assertEqual([s["count"] for s in res["segments"]], [100])
                self.assertEqual(res["after_reach"], {"items": n - 100, "added_not_counted": n - 100})
                # 101件目以降は証拠として sequence に残る
                self.assertEqual(len([x for x in res["sequence"] if x.get("state") == "added"]), n)

    def test_trouble_after_100th_does_not_change_or_cancel_reach(self):
        labels = _labels(103)
        races = [_race(x) for x in labels[:100]] + [_race(labels[100], "mismatch"),
                                                     _race(labels[101], "not_established"), _race(labels[102])]
        res = _aggregate(races, labels)
        self.assertEqual(res["reach"]["hundredth_race_key"], E(100))
        self.assertEqual(res["best_segment"], 100)
        self.assertEqual([s["count"] for s in res["segments"]], [100])  # 到達の後に新しい区間を作らない
        self.assertEqual(res["after_reach"], {"items": 3, "added_not_counted": 1})
        # 未処理（記録の無い対象）が後に続いても同じ
        res2 = _aggregate([_race(x) for x in labels[:100]], labels)
        self.assertEqual(res2["reach"]["hundredth_race_key"], E(100))
        self.assertEqual(res2["after_reach"]["items"], 3)

    def test_excluded_100th_gives_no_reach(self):
        labels = _labels(100)
        races = [_race(x) for x in labels[:99]] + [_race(labels[99], "mismatch")]
        res = _aggregate(races, labels)
        self.assertIsNone(res["reach"])
        self.assertEqual(res["best_segment"], 99)
        self.assertEqual(res["g2_status"], "undetermined")

    def test_reach_in_later_segment_and_no_second_segment_after_reach(self):
        labels = _labels(206)
        races = ([_race(x) for x in labels[:5]] + [_race(labels[5], "mismatch")]
                 + [_race(x) for x in labels[6:]])
        res = _aggregate(races, labels)
        self.assertEqual(res["segments"][0], {"start": E(1), "count": 5, "ended_by": "mismatch"})
        self.assertEqual(res["segments"][1]["count"], 100)
        self.assertEqual(res["reach"]["first_race_key"], E(7))
        self.assertEqual(res["reach"]["hundredth_race_key"], E(106))
        self.assertEqual(res["reach"]["segment_index"], 1)
        self.assertEqual(len(res["segments"]), 2)  # r107〜r206 の100件で2回目の到達を作らない
        self.assertEqual(res["after_reach"], {"items": 100, "added_not_counted": 100})

    def test_rerun_is_identical(self):
        labels = _labels(120)
        races = [_race(x) for x in labels]
        a = _aggregate(races, labels)
        b = _aggregate(races, labels)
        self.assertEqual(json.dumps(a["reach"], sort_keys=True), json.dumps(b["reach"], sort_keys=True))
        self.assertEqual(a["segments"], b["segments"])

    def test_reach_is_not_formal_achievement(self):
        res, _ = _clean(100)
        self.assertNotIn("achieved", json.dumps(res["g2_status"]))
        self.assertNotEqual(res["reach"]["status"], "achieved")
        self.assertNotIn("achievement", res)

    def test_unregistered_formal_run_never_reaches(self):
        labels = _labels(100)
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race(x) for x in labels], labels,
                       run_overrides={"instrument": _inst(100)})
            v = [agg.verify_run(os.path.join(base, x)) for x in os.listdir(base)]
            ctx = dict(FORMAL_OK, registration_verified=False)
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, ctx)
        self.assertIsNone(res["reach"])
        self.assertEqual(res["best_segment"], 0)


if __name__ == "__main__":
    unittest.main()
