"""判断C 7(2-b)・7(2-c)（07f §6.8 A。GPT判断 11/04、判断C 全体は 07g で GPT が最終承認）の集約側の判定。

- (2-b) ①〜③（unstable_hash_keys が空でない／venue_caches_match_preload が False／b_c_match が False）：
  そのレースと、その実行で後に続く対象レースを正式カウントから外す。評価が続いた記録は残す。
- (2-b) ④・(2-c)（fan ファイル名の変化／f_fan_file == ""／1ファイルだけの空のキャッシュ）：その実行全体を外す。
- 7(5)：(3)(4) にも当たれば、(2-a)〜(2-c) を優先する。
- 判定には記録された f_fan_file と f_fan_cache のハッシュだけを使う。当たらないことを fan 情報の完全さの証明にしない。
"""
import os
import tempfile
import unittest

from formal_g2.aggregate import main as agg
from test_aggregate import FORMAL_OK, PERMS, _race, _write_run

FAN = "fan2607.txt"
FULL = "f" * 64  # 空でないキャッシュのハッシュ（任意の値）


def E(n):
    return f"20990101_01_{n:02d}"


def snap(ff=FAN, fc=FULL):
    return {"f_fan_file": ff, "f_fan_cache": fc, "b_L": "x"}


def fan_race(label, cat="diff_zero", points=None, **fields):
    r = _race(label, cat)
    r["hashes"] = points or {"before": snap(), "after_rebuild": snap(), "after_v4": snap()}
    r["unstable_hash_keys"] = []
    r["b_c_match"] = True
    r["d13_2"] = {"venue_caches_match_preload": True}
    for k, v in fields.items():
        if k == "venue_caches_match_preload":
            r["d13_2"][k] = v
        else:
            r[k] = v
    return r


def closed_rec(label, kind="fail", stage="legacy_v4", step="after_rebuild_evaluate"):
    """評価側の close() と同じ形の記録（race_eval.evaluate_race。比較の前に閉じ、hashes を記録しない）。"""
    r = _race(label, "not_established", ok=False)
    for k in ("hashes", "values", "comparable_hash", "diffs"):
        r.pop(k, None)
    r["input_identity_ok"] = False
    if kind == "fail":
        r["stage"] = stage
        r["exception"] = {"class": "ValueError", "message": "x", "stage": stage}
        r["config_detected"] = None
        r["result_category"] = "compare_exception" if stage == "compare" else "not_established"
    else:
        r["stage"] = None
        r["exception"] = None
        r["config_detected"] = {"at_step": step, "phase": "x", "detections": []}
    return r


def run_agg(races, targets):
    with tempfile.TemporaryDirectory() as base:
        _write_run(base, 1, 1, "20990101", races, targets)
        v = [agg.verify_run(os.path.join(base, x)) for x in sorted(os.listdir(base))]
        bad = [x for x in v if not x["verified"]]
        assert not bad, bad[0]["problems"]
        return agg.aggregate(v, PERMS, "20990101", "20990101", agg_baseline(), agg_baseline_sha(), FORMAL_OK)


def agg_baseline():
    from test_aggregate import BASELINE
    return BASELINE


def agg_baseline_sha():
    from test_aggregate import BASELINE_SHA
    return BASELINE_SHA


def states(res):
    return [x.get("state") for x in res["sequence"] if "eval_id" in x]


L5 = ["a", "b", "c", "d", "e"]


class Judgment2bRaceTests(unittest.TestCase):
    def test_clean_run_counts(self):
        res = run_agg([fan_race(x) for x in L5], L5)
        self.assertEqual(states(res), ["added"] * 5)
        self.assertEqual(res["days"][0]["fan_check"]["reasons"], [])
        self.assertEqual(res["days"][0]["fan_check"]["snapshots"], 15)
        self.assertNotIn("count_stopped_2b", res["days"][0])

    def test_each_of_1_to_3_stops_that_race_and_later_ones(self):
        for name, kw in (("1", {"unstable_hash_keys": ["e_venue_stats"]}),
                         ("2", {"venue_caches_match_preload": False}),
                         ("3", {"b_c_match": False})):
            with self.subTest(cond=name):
                races = [fan_race(x) for x in L5[:2]] + [fan_race("c", **kw)] + [fan_race(x) for x in L5[3:]]
                res = run_agg(races, L5)
                self.assertEqual(states(res), ["added", "added"] + [agg.STOP_2B_STATE] * 3)
                stopped = [x for x in res["sequence"] if x.get("state") == agg.STOP_2B_STATE]
                self.assertEqual({x["stopped_by"] for x in stopped}, {E(3)})
                # 評価が続いた記録（元の状態）は残す
                self.assertEqual([x["original_state"] for x in stopped], ["added"] * 3)
                self.assertEqual(res["days"][0]["count_stopped_2b"]["eval_id"], E(3))
                self.assertEqual(len(res["days"][0]["count_stopped_2b"]["reasons"]), 1)
                self.assertEqual(res["segments"][0], {"start": E(1), "count": 2, "ended_by": agg.STOP_2B_STATE})

    def test_priority_over_7_3(self):
        """7(5)：比較不成立（7(3)）にも当たるレースは、(2-b) の状態にする。"""
        r = fan_race("b", unstable_hash_keys=["b_L"])
        r["d9_states"] = ["compare_not_established"]
        res = run_agg([fan_race("a"), r, fan_race("c")], ["a", "b", "c"])
        self.assertEqual(states(res), ["added", agg.STOP_2B_STATE, agg.STOP_2B_STATE])
        self.assertEqual(res["sequence"][1]["original_state"], "compare_not_established")

    def test_unprocessed_later_targets_are_also_stopped(self):
        res = run_agg([fan_race("a", b_c_match=False)], ["a", "b"])
        self.assertEqual(states(res), [agg.STOP_2B_STATE, agg.STOP_2B_STATE])
        self.assertEqual(res["sequence"][1]["original_state"], "unprocessed")

    def test_only_the_three_fields_and_only_explicit_false(self):
        """d13_2 のほかの項目（7(3)）は (2-b) に使わない。値が False でなければ当たらない（推測で広げない）。"""
        r = fan_race("b")
        r["d13_2"].update({"race_grade_match": False, "provider_cache_unchanged": False})
        r2 = fan_race("c", b_c_match=None, venue_caches_match_preload=None)
        res = run_agg([fan_race("a"), r, r2], ["a", "b", "c"])
        self.assertNotIn(agg.STOP_2B_STATE, states(res))
        self.assertEqual(agg.race_2b_reasons({"unstable_hash_keys": [], "b_c_match": True, "d13_2": {}}), [])
        self.assertEqual(agg.race_2b_reasons({"unstable_hash_keys": "x"}), [])


    def test_duplicate_eval_id_record_still_stops(self):
        """同じ eval_id の記録が2つあり、片方が ① に当たる：そのレースと後のレースを止める（監査の指摘1）。"""
        races = [fan_race("a"), fan_race("b"), fan_race("b", unstable_hash_keys=["b_L"]), fan_race("c")]
        res = run_agg(races, ["a", "b", "c"])
        self.assertEqual(states(res), ["added", agg.STOP_2B_STATE, agg.STOP_2B_STATE])
        self.assertEqual(res["sequence"][1]["original_state"], "duplicate_eval_id")

    def test_first_trigger_wins(self):
        races = [fan_race("a"), fan_race("b", b_c_match=False), fan_race("c", unstable_hash_keys=["b_L"])]
        res = run_agg(races, ["a", "b", "c"])
        self.assertEqual(res["days"][0]["count_stopped_2b"], {"eval_id": E(2), "reasons": ["3_b_c_match_false"]})
        self.assertEqual({x["stopped_by"] for x in res["sequence"] if "stopped_by" in x}, {E(2)})

    def test_malformed_records_do_not_crash(self):
        self.assertEqual(agg.race_2b_reasons({"d13_2": ["x"]}), [])
        self.assertEqual(agg.race_2b_reasons("x"), [])
        self.assertEqual(agg.fan_run_check(["x", {"hashes": ["y"]}])["snapshots"], 0)


class JudgmentFanRunTests(unittest.TestCase):
    def _excluded(self, races, targets):
        res = run_agg(races, targets)
        self.assertEqual(states(res), [])
        self.assertTrue(res["days"][0]["excluded"].startswith("judgment_c_7_fan:"))
        self.assertEqual(res["segments"], [])
        self.assertEqual(res["best_segment"], 0)
        return res

    def test_2b4_fan_file_changed_between_races(self):
        later = {"before": snap("fan2606.txt"), "after_rebuild": snap("fan2606.txt"), "after_v4": snap("fan2606.txt")}
        res = self._excluded([fan_race("a"), fan_race("b", points=later)], ["a", "b"])
        self.assertEqual(res["days"][0]["fan_check"]["reasons"], ["2b4_fan_file_changed"])
        self.assertEqual(res["days"][0]["fan_check"]["fan_files"], ["fan2606.txt", FAN])

    def test_2c1_fan_file_empty(self):
        pts = {"before": snap(""), "after_rebuild": snap(""), "after_v4": snap("")}
        res = self._excluded([fan_race("a", points=pts), fan_race("b", points=pts)], ["a", "b"])
        self.assertEqual(res["days"][0]["fan_check"]["reasons"], ["2c1_fan_file_empty"])

    def test_2c2_single_file_empty_cache(self):
        empty = agg._stable_hash({FAN: {}})
        pts = {"before": snap(fc=empty), "after_rebuild": snap(fc=empty), "after_v4": snap(fc=empty)}
        res = self._excluded([fan_race("a", points=pts), fan_race("b", points=pts)], ["a", "b"])
        self.assertEqual(res["days"][0]["fan_check"]["reasons"], ["2c2_single_file_empty_cache"])
        self.assertEqual(res["days"][0]["fan_check"]["empty_cache_at"][0], {"eval_id": E(1), "point": "before"})

    def test_2c2_one_snapshot_is_enough(self):
        empty = agg._stable_hash({FAN: {}})
        pts = {"before": snap(), "after_rebuild": snap(), "after_v4": snap(fc=empty)}
        self._excluded([fan_race("a"), fan_race("b", points=pts)], ["a", "b"])

    def test_not_excluded_cases(self):
        """原文の3条件に当たらないものは外さない（キャッシュが0ファイルの {}、別名のファイルの空など）。"""
        for name, fc in (("other_name_empty", agg._stable_hash({"fan2606.txt": {}})),
                         ("one_racer", agg._stable_hash({FAN: {"4444": {}}}))):
            with self.subTest(name=name):
                pts = {"before": snap(fc=fc), "after_rebuild": snap(fc=fc), "after_v4": snap(fc=fc)}
                res = run_agg([fan_race("a", points=pts)], ["a"])
                self.assertEqual(states(res), ["added"])

    def test_fan_exclusion_has_priority_over_2b_race(self):
        later = {"before": snap("fan2606.txt"), "after_rebuild": snap("fan2606.txt"), "after_v4": snap("fan2606.txt")}
        res = self._excluded([fan_race("a", b_c_match=False), fan_race("b", points=later)], ["a", "b"])
        self.assertNotIn("count_stopped_2b", res["days"][0])

    def test_hash_definition_matches_evaluate_side(self):
        """(2-c) ②は、評価側と同じ stable_hash の定義で計算し直す。"""
        from formal_g2 import race_eval
        for obj in ({FAN: {}}, {}, {"fanあ.txt": {}}, {FAN: {"1": {"x": [1.5, None]}}}):
            self.assertEqual(agg._stable_hash(obj), race_eval.stable_hash(obj))

    def test_u8_change_within_a_race_only_is_not_2b4(self):
        """U8：原文どおりレースの間の変化に限る。1つのレースの中だけの変化は ④ に当てはめない（観測として記録）。"""
        pts = {"before": snap(), "after_rebuild": snap("fan2606.txt"), "after_v4": snap()}
        res = run_agg([fan_race("a"), fan_race("b", points=pts), fan_race("c")], ["a", "b", "c"])
        self.assertEqual(states(res), ["added"] * 3)
        fc = res["days"][0]["fan_check"]
        self.assertEqual(fc["reasons"], [])
        self.assertEqual(fc["within_race_changes"], [E(2)])

    def test_u8_change_at_race_boundary(self):
        """前のレースの最後の時点と次のレースの最初の時点の間で変われば ④。"""
        pts_a = {"before": snap(), "after_rebuild": snap(), "after_v4": snap("fan2606.txt")}
        pts_b = {"before": snap("fan2606.txt"), "after_rebuild": snap("fan2606.txt"), "after_v4": snap("fan2606.txt")}
        res = run_agg([fan_race("a", points=pts_a), fan_race("b", points=pts_b)], ["a", "b"])
        self.assertEqual(res["days"][0]["fan_check"]["reasons"], [], "a の中の変化だけ（境目では同じ）")
        pts_b2 = {"before": snap(), "after_rebuild": snap(), "after_v4": snap()}
        res = self._excluded([fan_race("a", points=pts_a), fan_race("b", points=pts_b2)], ["a", "b"])
        self.assertEqual(res["days"][0]["fan_check"]["changes_between_races"],
                         [{"from": E(1), "to": E(2), "files": ["fan2606.txt", FAN]}])

    def test_u8_boundary_skips_records_closed_before_compare(self):
        later = {"before": snap("fan2606.txt"), "after_rebuild": snap("fan2606.txt"), "after_v4": snap("fan2606.txt")}
        mid = closed_rec("b")
        res = self._excluded([fan_race("a"), mid, fan_race("c", points=later)], ["a", "b", "c"])
        self.assertEqual(res["days"][0]["excluded"], "judgment_c_7_fan:2b4_fan_file_changed")
        self.assertEqual(res["days"][0]["fan_check"]["changes_between_races"][0]["from"], E(1))
        self.assertEqual(res["days"][0]["fan_check"]["changes_between_races"][0]["to"], E(3))

    def test_u9_zero_file_cache(self):
        """U9：0ファイルだけでは (2-c) に当てはめない。名前があるのに0ファイルは読み込み失敗として証拠不完全。"""
        z = agg._stable_hash({})
        pts = {"before": snap(fc=z), "after_rebuild": snap(fc=z), "after_v4": snap(fc=z)}
        res = run_agg([fan_race("a", points=pts)], ["a"])
        self.assertEqual(states(res), [])
        self.assertEqual(res["days"][0]["excluded"], "evidence_incomplete:judgment_c_7_fan")
        self.assertEqual(res["days"][0]["fan_check"]["reasons"], [])
        self.assertIn("read failure", res["days"][0]["fan_check"]["incomplete"][0]["problem"])
        # 名前が空で0ファイル：(2-c) ① だけ（読み込みは行われない）
        pts = {"before": snap("", z), "after_rebuild": snap("", z), "after_v4": snap("", z)}
        res = run_agg([fan_race("a", points=pts)], ["a"])
        self.assertEqual(res["days"][0]["excluded"], "judgment_c_7_fan:2c1_fan_file_empty")


class FanEvidenceIncompleteTests(unittest.TestCase):
    """U5（GPT 決定 2026-11-09）：判定に必要な証拠が不完全なら、条件該当とは別の理由で実行全体を外す。"""

    def _incomplete(self, rec, extra=()):
        res = run_agg([fan_race("a"), rec, *extra], ["a", "b", *[f"x{i}" for i in range(len(extra))]])
        self.assertEqual(states(res), [])
        self.assertEqual(res["days"][0]["excluded"], "evidence_incomplete:judgment_c_7_fan")
        self.assertEqual(res["days"][0]["fan_check"]["reasons"], [])
        self.assertGreaterEqual(res["days"][0]["fan_check"]["incomplete_count"], 1)
        return res

    def test_each_kind(self):
        cases = {
            "file_missing": {"before": {"f_fan_cache": FULL}, "after_rebuild": snap(), "after_v4": snap()},
            "file_not_str": {"before": snap(ff=None), "after_rebuild": snap(), "after_v4": snap()},
            "file_list": {"before": snap(ff=["fan"]), "after_rebuild": snap(), "after_v4": snap()},
            "cache_unserializable": {"before": snap(fc="UNSERIALIZABLE:TypeError"), "after_rebuild": snap(),
                                     "after_v4": snap()},
            "cache_missing": {"before": {"f_fan_file": FAN}, "after_rebuild": snap(), "after_v4": snap()},
            "cache_not_hex": {"before": snap(fc="F" * 64), "after_rebuild": snap(), "after_v4": snap()},
            "point_missing": {"before": snap(), "after_v4": snap()},
            "point_not_dict": {"before": snap(), "after_rebuild": "x", "after_v4": snap()},
        }
        for name, pts in cases.items():
            with self.subTest(name=name):
                self._incomplete(fan_race("b", points=pts))

    def test_compared_record_without_hashes(self):
        r = fan_race("b")
        r.pop("hashes")
        self._incomplete(r)
        r2 = fan_race("b")
        r2["hashes"] = ["x"]
        self._incomplete(r2)

    def test_condition_and_incomplete_recorded_separately(self):
        pts = {"before": snap(""), "after_rebuild": snap(""), "after_v4": {"f_fan_file": ""}}
        res = run_agg([fan_race("a", points=pts)], ["a"])
        self.assertEqual(res["days"][0]["excluded"],
                         "judgment_c_7_fan:2c1_fan_file_empty;evidence_incomplete:judgment_c_7_fan")

    def test_v1_records_closed_before_compare_are_race_level(self):
        """V1：評価側が比較の前に正当に閉じた記録は、そのレースだけ 0（07f §6.4・§6.5 の表）。"""
        for name, r in (("fail_resolve", closed_rec("b", stage="resolve")),
                        ("fail_rebuild", closed_rec("b", stage="rebuild_evaluate")),
                        ("fail_v4", closed_rec("b", stage="legacy_v4")),
                        ("fail_compare", closed_rec("b", stage="compare")),
                        ("config_after_resolve", closed_rec("b", "config", step="after_resolve")),
                        ("config_after_compare", closed_rec("b", "config", step="after_compare"))):
            with self.subTest(name=name):
                self.assertTrue(agg.closed_before_compare(r))
                res = run_agg([fan_race("a"), r, fan_race("c")], ["a", "b", "c"])
                self.assertEqual(states(res), ["added", "compare_not_established", "added"])
                self.assertEqual(res["days"][0]["fan_check"]["incomplete_count"], 0)

    def test_v1_no_reading_as_closed(self):
        """閉じたことを記録の形で確かめられないもの（読み替え）は、U5 の証拠不完全として実行全体を外す。"""
        def mod(**kw):
            r = closed_rec("b")
            for k, v in kw.items():
                if v is KeyError:
                    r.pop(k, None)
                else:
                    r[k] = v
            return r
        cases = {
            "plain_not_established": dict(stage=None, exception=None, config_detected=None),
            "unknown_stage": dict(stage="later", exception={"class": "E", "message": "x", "stage": "later"}),
            "stage_mismatch": dict(exception={"class": "E", "message": "x", "stage": "resolve"}),
            "wrong_category": dict(result_category="compare_exception"),
            "states": dict(d9_states=[]),
            "identity_true": dict(input_identity_ok=True),
            "identity_missing": dict(input_identity_ok=KeyError),
            "has_values": dict(values={"legacy": {}, "rebuild": {}}),
            "has_diffs": dict(diffs=[]),
            "exception_str": dict(exception="ValueError: x"),
            "exception_class_not_str": dict(exception={"class": 1, "message": "x", "stage": "legacy_v4"}),
            "exception_message_missing": dict(exception={"class": "E", "stage": "legacy_v4"}),
            "config_too": dict(config_detected={"at_step": "after_resolve"}),
            "diff_zero_no_hashes": dict(result_category="diff_zero"),
        }
        cfg_cases = {
            "config_unknown_step": dict(config_detected={"at_step": "before"}),
            "config_with_stage": dict(stage="resolve"),
            "config_compare_exception": dict(result_category="compare_exception"),
        }
        for name, kw in cases.items():
            with self.subTest(name=name):
                r = mod(**kw)
                self.assertFalse(agg.closed_before_compare(r))
                self._incomplete(r)
        for name, kw in cfg_cases.items():
            with self.subTest(name=name):
                r = closed_rec("b", "config")
                r.update(kw)
                self.assertFalse(agg.closed_before_compare(r))
                self._incomplete(r)


class EvaluateSideBasisTests(unittest.TestCase):
    """V2（GPT 決定 2026-11-09）：「fan ファイル名があるのにキャッシュが0ファイル＝読み込みの失敗」の根拠を、
    評価側の既存コード（変えない）で確かめる。評価側のキャッシュの仕様が変わればここが失敗し、再確認を促す。
    根拠：notify_arashi._load_fan_file（成功すれば0件でも _FAN_CACHE[filepath] に入れる。OSError なら {} を返し
    キャッシュに入れない）、notify_arashi._extract_boats_from_program（ファイル名が空でなければ読み込む）、
    race_eval.evaluate_race（入力の解決＝艇の作成の後に、評価前の時点のハッシュを取る）。"""

    def setUp(self):
        import notify_arashi
        self.na = notify_arashi
        self.saved = dict(notify_arashi._FAN_CACHE)
        notify_arashi._FAN_CACHE.clear()

    def tearDown(self):
        self.na._FAN_CACHE.clear()
        self.na._FAN_CACHE.update(self.saved)

    def test_read_failure_leaves_cache_empty(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "fan9999.txt")
            self.assertEqual(self.na._load_fan_file(missing), {})
        self.assertEqual(self.na._FAN_CACHE, {})
        self.assertEqual(agg._stable_hash(self.na._FAN_CACHE), agg._stable_hash({}))

    def test_successful_read_with_zero_records_is_cached_under_name(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "fan9999.txt")
            open(p, "wb").close()
            self.assertEqual(self.na._load_fan_file(p), {})
        self.assertEqual(self.na._FAN_CACHE, {p: {}})
        self.assertEqual(agg._stable_hash(self.na._FAN_CACHE), agg._stable_hash({p: {}}))

    def test_boats_are_built_with_fan_load_before_first_snapshot(self):
        import inspect
        from formal_g2 import race_eval
        src = inspect.getsource(self.na._extract_boats_from_program)
        self.assertIn("fan_data = _load_fan_file(fan_file) if fan_file else {}", src)
        ev = inspect.getsource(race_eval.evaluate_race)
        self.assertLess(ev.index("ctx.provider.resolve_boats("), ev.index("h_before = _cache_hashes("))


if __name__ == "__main__":
    unittest.main()
