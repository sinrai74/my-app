"""正式G2：集約の追加照合（2026-11-05、07g・GPT 判断 2-1〜2-3・1-2）。

- 実行メタデータの照合（run.json・manifest・meta.json の API の値）
- 条件5（programs_raw.json から対象一覧・targets_sha256・program_sha256 を計算し直す）
- comparable_hash の対象（既決の5項目。構造だけ）
- 条件7 の表記（利用可能性・通信の監視結果・監視の限界を分ける）

評価側と集約側の定義の一致は、評価側の関数（formal_g2.programs・race_eval・凍結版 _extract_programs）と
合成データで比べる。実データでの一致（R1 の156件）は、このテストの範囲ではない。

実行（リポジトリ直下で）：
  PYTHONPATH="rebuild:." python -P -B -m unittest discover -s tests/formal_g2 -t tests/formal_g2
"""

import hashlib
import json
import os
import tempfile
import unittest
import zipfile

from formal_g2.aggregate import main as agg
from test_aggregate import BASELINE, BASELINE_RAW, BASELINE_SHA, FORMAL_OK, PERMS, _race, _write_run, permitted_by


def _verified(base):
    return [agg.verify_run(os.path.join(base, x)) for x in sorted(os.listdir(base))]


def _edit_meta(d, **kw):
    p = os.path.join(d, "meta.json")
    with open(p, encoding="utf-8") as f:
        m = json.load(f)
    for k, v in kw.items():
        if v is None:
            m.pop(k, None)
        else:
            m[k] = v
    with open(p, "w", encoding="utf-8") as f:
        json.dump(m, f)


def _rewrite_artifact(d, edit):
    """artifact.zip の中身を edit(files) で書き換え、manifest・digest を合わせ直す（改ざんの検出ではなく、
    照合の内容そのものを試すため）。"""
    p = os.path.join(d, "artifact.zip")
    with zipfile.ZipFile(p) as z:
        files = {n: z.read(n) for n in z.namelist()}
    manifest = json.loads(files.pop("manifest.json"))
    edit(files, manifest)
    manifest["files"] = {n: {"sha256": hashlib.sha256(b).hexdigest()} for n, b in files.items()}
    import io
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, b in files.items():
            z.writestr(n, b)
        z.writestr("manifest.json", json.dumps(manifest))
    blob = buf.getvalue()
    with open(p, "wb") as f:
        f.write(blob)
    _edit_meta(d, expected_digest="sha256:" + hashlib.sha256(blob).hexdigest())


def _edit_run_json(d, fn, also_manifest=True):
    def edit(files, manifest):
        run = json.loads(files["run.json"])
        fn(run)
        files["run.json"] = json.dumps(run).encode()
        if also_manifest:
            manifest["run"] = dict(run["info"])
    _rewrite_artifact(d, edit)


def _aggregate_one(setup=None):
    with tempfile.TemporaryDirectory() as base:
        d = _write_run(base, 1, 1, "20990101", [_race("a"), _race("b")], ["a", "b"])
        if setup:
            setup(d)
        v = _verified(base)
        return v[0], agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)


class RunMetadataTests(unittest.TestCase):
    def test_consistent_metadata_passes_and_counts(self):
        vr, res = _aggregate_one()
        self.assertTrue(vr["run_metadata"]["ok"], vr["run_metadata"]["problems"])
        self.assertEqual(res["days"][0]["run_metadata_check"], "ok")
        self.assertEqual(res["best_segment"], 2)
        rec = vr["run_metadata"]["recorded"]
        self.assertEqual(rec["api"]["job_id"], "125133673866")  # API 由来の値を別に記録する
        self.assertEqual(rec["run_json"]["github_run_id"], "1")

    def _assert_excluded(self, setup, expected_fragment):
        vr, res = _aggregate_one(setup)
        self.assertFalse(vr["run_metadata"]["ok"])
        self.assertTrue(any(expected_fragment in p for p in vr["run_metadata"]["problems"]),
                        vr["run_metadata"]["problems"])
        self.assertEqual(res["days"][0]["excluded"], "run_metadata")
        self.assertEqual(res["best_segment"], 0)
        self.assertEqual(res["baseline_failures"][0]["check"], "run_metadata")  # 終了コード 3 の対象
        # 比較値の差分とは混同しない（レースの状態は記録されない）
        self.assertEqual([s for s in res["sequence"] if "state" in s], [])

    def test_api_fields_missing_is_evidence_defect(self):
        # 今の集約 workflow は api_* を取得しない。その場合は不合格になる（fail-closed）
        self._assert_excluded(lambda d: _edit_meta(d, api_head_sha=None), "API run fields unavailable")

    def test_run_id_and_attempt_mismatch(self):
        # 2026-11-06：run_id は API の値（api_run_id）と照合する。入力の run_id は API の値との一致だけを見る
        self._assert_excluded(lambda d: _edit_meta(d, run_id="2"), "requested run_id does not match API run id")
        self._assert_excluded(lambda d: _edit_meta(d, api_run_id="2"), "run_id mismatch")
        self._assert_excluded(lambda d: _edit_meta(d, api_run_attempt="2", latest_attempt="2"), "run_attempt mismatch")
        self._assert_excluded(lambda d: _edit_meta(d, api_event="push"), "API event is not workflow_dispatch")
        for k in ("api_run_id", "api_run_attempt", "api_event"):
            self._assert_excluded(lambda d, k=k: _edit_meta(d, **{k: None}), "API run fields unavailable")
        # meta の attempt を変えると attempt 1 として選ばれなくなる（既存の規則）ので、run.json 側を変える
        self._assert_excluded(lambda d: _edit_run_json(d, lambda r: r["info"].update(github_run_attempt="2")),
                              "run_attempt mismatch")
        self._assert_excluded(lambda d: _edit_meta(d, latest_attempt="0"), "greater than latest attempt")

    def test_head_sha_ref_workflow_ref_against_api(self):
        self._assert_excluded(lambda d: _edit_meta(d, api_head_sha="1" * 40), "github_sha does not match")
        self._assert_excluded(lambda d: _edit_meta(d, api_workflow_path=".github/workflows/other.yml"),
                              "github_workflow_ref does not match")
        # head_branch が違う場合（ref は main のまま）
        self._assert_excluded(lambda d: _edit_meta(d, api_head_branch="feature/x"), "github_ref does not match")

    def test_started_and_finished_required(self):
        self._assert_excluded(lambda d: _edit_run_json(d, lambda r: r["info"].pop("started_at")), "started_at missing")
        self._assert_excluded(lambda d: _edit_run_json(d, lambda r: r.pop("finished_at")), "finished_at missing")
        self._assert_excluded(lambda d: _edit_run_json(d, lambda r: r.update(finished_at="2099/01/01 10:00")),
                              "finished_at missing or not ISO8601")
        self._assert_excluded(lambda d: _edit_run_json(d, lambda r: r.update(finished_at="2099-01-01T09:05:00")),
                              "finished_at missing or not ISO8601 with offset")  # 時差が無い
        self._assert_excluded(lambda d: _edit_run_json(d, lambda r: r.update(finished_at="2099-01-01T08:00:00+09:00")),
                              "finished_at earlier than started_at")

    def test_manifest_and_run_json_must_match(self):
        self._assert_excluded(
            lambda d: _edit_run_json(d, lambda r: r["info"].update(github_sha="2" * 40), also_manifest=False),
            "manifest run info differs")

    def test_investigation_run_is_not_judged_but_metadata_is_recorded(self):
        with tempfile.TemporaryDirectory() as base:
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], mode="investigation")
            _edit_meta(d, api_head_sha=None)
            v = _verified(base)
            self.assertFalse(v[0]["run_metadata"]["ok"])  # 記録はする
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)
            self.assertEqual(res["baseline_failures"], [])  # 正式な実行でなければ判定の対象にしない


class Condition5Tests(unittest.TestCase):
    def test_consistent_records_pass(self):
        vr, res = _aggregate_one()
        c5 = vr["condition5"]
        self.assertEqual(c5["status"], "ok", c5["problems"])
        self.assertEqual(c5["checks"]["program_sha256"]["matched"], 2)
        self.assertEqual(res["days"][0]["condition5_check"], "ok")

    def _assert_c5_fails(self, setup, fragment):
        vr, res = _aggregate_one(setup)
        self.assertEqual(vr["condition5"]["status"], "failed")
        self.assertTrue(any(fragment in p for p in vr["condition5"]["problems"]), vr["condition5"]["problems"])
        self.assertEqual(res["days"][0]["excluded"], "condition5")
        self.assertEqual(res["best_segment"], 0)

    def test_raw_bytes_changed(self):
        def edit(files, manifest):
            files["programs_raw.json"] = files["programs_raw.json"] + b" "
        self._assert_c5_fails(lambda d: _rewrite_artifact(d, edit), "raw_sha256 mismatch")

    def test_targets_and_sha_mismatch(self):
        self._assert_c5_fails(
            lambda d: _edit_run_json(d, lambda r: r["programs"]["targets"].pop()), "targets mismatch")
        self._assert_c5_fails(
            lambda d: _edit_run_json(d, lambda r: r["programs"].update(targets_sha256="0" * 64)),
            "targets_sha256 mismatch")

    def test_program_sha256_mismatch(self):
        def edit(files, manifest):
            lines = [json.loads(x) for x in files["races.jsonl"].decode().splitlines()]
            lines[0]["program_sha256"] = "f" * 64
            files["races.jsonl"] = "\n".join(json.dumps(x) for x in lines).encode()
        self._assert_c5_fails(lambda d: _rewrite_artifact(d, edit), "program_sha256 mismatch: 1")

    def test_programs_key_none_or_empty_does_not_fall_back_to_today(self):
        for value in (None, []):
            with self.subTest(value=value):
                def edit(files, manifest, value=value):
                    data = json.loads(files["programs_raw.json"])
                    files["programs_raw.json"] = json.dumps({"programs": value, "today": data}).encode()
                self._assert_c5_fails(lambda d: _rewrite_artifact(d, edit), "programs list invalid")

    def test_today_used_only_when_programs_key_absent(self):
        raw = json.dumps({"programs": [{"race_stadium_number": 1, "race_number": 1}]}).encode()
        self.assertEqual(agg._extract_programs_by_key(json.loads(raw)), [{"race_stadium_number": 1, "race_number": 1}])
        self.assertEqual(agg._extract_programs_by_key({"today": {"programs": [1]}}), [1])
        self.assertIsNone(agg._extract_programs_by_key({"programs": None, "today": {"programs": [1]}}))

    def test_eval_id_uses_run_race_date(self):
        # program の中の race_date が違っても、eval_id の日付は実行の race_date（引数）を使う
        def edit(files, manifest):
            data = json.loads(files["programs_raw.json"])
            for x in data["programs"]:
                x["race_date"] = "2000-01-01"
            files["programs_raw.json"] = json.dumps(data).encode()
            run = json.loads(files["run.json"])
            for t in run["programs"]["targets"]:
                t["race_date_in_program"] = "2000-01-01"
            run["programs"]["raw_sha256"] = hashlib.sha256(files["programs_raw.json"]).hexdigest()
            run["programs"]["targets_sha256"] = hashlib.sha256(json.dumps(
                run["programs"]["targets"], ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode()).hexdigest()
            files["run.json"] = json.dumps(run).encode()
            lines = [json.loads(x) for x in files["races.jsonl"].decode().splitlines()]
            for x, p in zip(lines, data["programs"]):
                x["program_sha256"] = agg._stable_hash(p)
            files["races.jsonl"] = "\n".join(json.dumps(x) for x in lines).encode()
        vr, _res = _aggregate_one(lambda d: _rewrite_artifact(d, edit))
        self.assertEqual(vr["condition5"]["status"], "ok", vr["condition5"]["problems"])

    def test_missing_programs_raw(self):
        def edit(files, manifest):
            files.pop("programs_raw.json")
        self._assert_c5_fails(lambda d: _rewrite_artifact(d, edit), "programs_raw.json missing")

    def test_stopped_run_without_programs_is_not_applicable(self):
        res = agg.condition5_check(None, {"status": "stopped", "info": {"race_date": "20990101"}}, [])
        self.assertEqual(res["status"], "not_applicable_stopped_run")
        res = agg.condition5_check(None, {"status": "completed", "info": {"race_date": "20990101"}}, [])
        self.assertEqual(res["status"], "failed")


class SameDefinitionAsEvaluationSideTests(unittest.TestCase):
    """評価側の関数と、集約側で書き直した関数が、同じ入力に同じ結果を返すこと（合成データ）。"""

    def test_extract_programs_matches_frozen_function(self):
        import notify_arashi  # 凍結版（読み込むだけ）
        cases = [None, {}, {"programs": [{"a": 1}]}, {"programs": []}, {"programs": None},
                 {"today": {"programs": [{"b": 2}]}}, {"today": None}, {"programs": [], "today": {"programs": [1]}}]
        for c in cases:
            with self.subTest(case=c):
                self.assertEqual(agg._extract_programs_by_key(c), notify_arashi._extract_programs(c))

    def test_sort_key_and_hashes_match_evaluation_side(self):
        from formal_g2 import programs as ev_programs
        from formal_g2 import race_eval
        progs = [
            {"race_date": "2099-01-01", "race_stadium_number": 3, "race_number": 2, "race_closed_at": "2099-01-01 10:00:00",
             "x": float("nan"), "名": "全角"},
            {"race_date": "2099-01-01", "race_stadium_number": 1, "race_number": 12, "race_closed_at": "2099-01-01 10:00:00"},
            {"race_date": "2099-01-01", "race_stadium_number": 1, "race_number": 3, "race_closed_at": "2099-01-01 09:30:00"},
        ]
        for p in progs:
            self.assertEqual(agg._race_sort_key(p), ev_programs.race_sort_key(p))
            self.assertEqual(agg._stable_hash(p), race_eval.stable_hash(p))
        self.assertEqual(agg.FIVE_ITEMS, race_eval.FIVE_ITEMS)

    def test_condition5_accepts_what_fix_programs_records(self):
        """評価側の fix_programs が記録した値を、集約側の照合がそのまま通すこと（programs・today の両形）。"""
        from formal_g2 import programs as ev_programs
        from formal_g2 import race_eval
        progs = [
            {"race_date": "2099-01-01", "race_stadium_number": 2, "race_number": 1, "race_closed_at": "2099-01-01 11:00:00"},
            {"race_date": "2099-01-01", "race_stadium_number": 1, "race_number": 1, "race_closed_at": "2099-01-01 11:00:00"},
        ]

        class _Resp:
            def __init__(self, body):
                self.content = body
                self.status_code = 200

            def raise_for_status(self):
                return None

            def json(self):
                return json.loads(self.content)

        for shape in ({"programs": progs}, {"today": {"programs": progs}}):
            with self.subTest(shape=list(shape)):
                body = json.dumps(shape, ensure_ascii=False).encode("utf-8")
                fixed = ev_programs.fix_programs("20990101", http_get=lambda url, body=body: _Resp(body))
                run = {"status": "completed", "info": {"race_date": "20990101"},
                       "programs": {k: fixed[k] for k in ("raw_sha256", "targets", "targets_sha256")}}
                by_id = {t["eval_id"]: p for t, p in zip(fixed["targets"], sorted(fixed["programs"],
                                                                                  key=ev_programs.race_sort_key))}
                races = [{"eval_id": e, "program_sha256": race_eval.stable_hash(p)} for e, p in by_id.items()]
                res = agg.condition5_check(fixed["raw_bytes"], run, races)
                self.assertEqual(res["status"], "ok", res["problems"])
                self.assertEqual(res["checks"]["program_sha256"]["matched"], 2)


class ComparableScopeTests(unittest.TestCase):
    def test_five_items_only(self):
        self.assertTrue(agg.comparable_scope_ok(_race("a")))
        rec = _race("a")
        rec["values"]["legacy"]["run_id"] = "1"  # 実行メタデータが混ざっている
        self.assertFalse(agg.comparable_scope_ok(rec))
        self.assertEqual(agg.race_state(rec, True), "evidence_incomplete")
        rec = _race("a")
        rec["values"]["rebuild"].pop("rank_index")
        self.assertEqual(agg.race_state(rec, True), "evidence_incomplete")
        rec = _race("a")
        rec["values"]["rebuild"] = None
        rec["comparable_hash"]["rebuild"] = None
        self.assertEqual(agg.race_state(rec, True), "evidence_incomplete")
        rec = _race("a")
        rec["comparable_hash"]["legacy"] = "UNSERIALIZABLE:TypeError"
        self.assertEqual(agg.race_state(rec, True), "evidence_incomplete")

    def test_time_dependent_value_inside_items_is_not_removed(self):
        rec = _race("a")
        rec["values"]["legacy"]["danger_breakdown"] = {"evaluated_at": "2099-01-01T00:00:00+09:00"}
        self.assertTrue(agg.comparable_scope_ok(rec))
        self.assertEqual(rec["values"]["legacy"]["danger_breakdown"], {"evaluated_at": "2099-01-01T00:00:00+09:00"})


class Condition7RecordTests(unittest.TestCase):
    def test_availability_and_network_are_recorded_separately(self):
        run = {"checks": {"release": {"release_storage_available": False}},
               "instrument": {"side_effect_events": [{"kind": "network", "detail": "('h', 443)"},
                                                     {"kind": "file_write", "detail": "x"}],
                              "limitations": {"c_extension_direct_side_effects": "unverified"}}}
        r = agg.release_and_network_record(run)
        self.assertIs(r["release_storage_available"], False)
        self.assertEqual(r["network_events_observed"]["count"], 1)
        self.assertEqual(r["monitoring_limitations"], {"c_extension_direct_side_effects": "unverified"})
        self.assertIn("証明ではない", r["note"])

    def test_aggregate_output_has_note(self):
        from contextlib import redirect_stdout
        from io import StringIO
        with tempfile.TemporaryDirectory() as tmp:
            base = os.path.join(tmp, "a")
            os.makedirs(base)
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            bpath, perms = os.path.join(tmp, "b.json"), os.path.join(tmp, "p.json")
            with open(bpath, "wb") as f:
                f.write(BASELINE_RAW)
            with open(perms, "w", encoding="utf-8") as f:
                json.dump(PERMS, f)
            out = os.path.join(tmp, "out")
            with permitted_by(), redirect_stdout(StringIO()):
                rc = agg.main(["--artifacts-dir", base, "--first-date", "20990101", "--last-date", "20990101",
                               "--out", out, "--baseline", bpath])
            with open(os.path.join(out, "aggregate_result.json"), encoding="utf-8") as f:
                res = json.load(f)
        # 2026-11-06：main() は登録の確認の手段を持たないので、正式な実行は常に合格しない（終了コード 3）。
        # この run には集約側の基準も渡していないので、最初に当たる理由は集約側の基準が無いこと。
        self.assertEqual(rc, agg.EXIT_VERIFICATION_FAILED)
        self.assertEqual(res["days"][0]["excluded"], "aggregate_baseline_no_baseline")
        self.assertEqual(res["best_segment"], 0)
        self.assertIs(res["formal_judgment"]["registration_verified"], False)
        self.assertIn("証明ではない", res["release_availability_note"])
        self.assertIn("release_and_network", res["runs"][0])
        self.assertTrue(res["runs"][0]["run_metadata"]["ok"])
        self.assertEqual(res["runs"][0]["condition5"]["status"], "ok")


if __name__ == "__main__":
    unittest.main()
