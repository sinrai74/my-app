"""正式G2：集約・照合（C-6・§6.3〜§6.6）。標準ライブラリだけで動くこと。

実行（リポジトリ直下で）：
  PYTHONPATH="rebuild:." python -P -B -m unittest discover -s tests/formal_g2 -t tests/formal_g2
"""

import hashlib
import io
import json
import os
import tempfile
import unittest
import zipfile

from formal_g2.aggregate import main as agg


# Q8（run 41401359888）のジョブログ冒頭と、artifact に記録された環境変数の値
Q8_LOG_HEAD = (
    "\ufeff2026-11-02T06:40:09.9012067Z Current runner version: '2.338.2'\n"
    "2026-11-02T06:40:09.9047428Z ##[endgroup]\n"
    "2026-11-02T06:40:09.9048018Z ##[group]Runner Image\n"
    "2026-11-02T06:40:09.9048627Z Image: ubuntu-24.04\n"
    "2026-11-02T06:40:09.9049179Z Version: 20261025.353.1\n"
    "2026-11-02T06:40:09.9050812Z Included Software: https://github.com/actions/runner-images/blob/"
    "ubuntu24/20261025.353/images/ubuntu/Ubuntu2404-Readme.md\n"
    "2026-11-02T06:40:09.9052829Z Image Release: https://github.com/actions/runner-images/releases/tag/"
    "ubuntu24%2F20261025.353\n"
    "2026-11-02T06:40:09.9053848Z ##[endgroup]\n"
)
Q8_ENV_IMAGE = {"ImageOS": "ubuntu24", "ImageVersion": "20261026.354.1", "identified": True}
# 2026-11-06：評価ジョブのログに、GITHUB_TOKEN Permissions の表示と、取得のステップの証跡がある形（合成）。
# 「Metadata: read」は合成の値で、permissions: {} のときの実際の表示は Actions で確かめるまで未決定。
EVAL_LOG_TAIL = (
    "2026-11-02T06:40:09.9060000Z ##[group]GITHUB_TOKEN Permissions\n"
    "2026-11-02T06:40:09.9060001Z Metadata: read\n"
    "2026-11-02T06:40:09.9060002Z ##[endgroup]\n"
    "2026-11-02T06:40:10.0000000Z ##[group]Run set -euo pipefail\n"
    "2026-11-02T06:40:10.0000001Z \x1b[36;1m  echo \"g2_fetch_head=${head}\"\x1b[0m\n"
    "2026-11-02T06:40:10.0000002Z ##[endgroup]\n"
    "2026-11-02T06:40:11.0000000Z g2_credentials_check=none\n"
    "2026-11-02T06:40:11.0000001Z g2_fetch_head=" + "0" * 40 + "\n"
)
EXPECTED_TOKEN_PERMISSIONS = ["Metadata: read"]
# 2026-11-06（GPT 判断・M7-3 は本人承認待ち）：評価側の基準の runner_image は Image だけ
BASELINE = {"runner_image": {"image": "ubuntu-24.04"}}
BASELINE_RAW = json.dumps(BASELINE, sort_keys=True).encode("utf-8")
BASELINE_SHA = hashlib.sha256(BASELINE_RAW).hexdigest()
# 正式な判定に必要な条件をすべて満たした場合（登録の確認はテストでだけ True にできる。main() は常に False）
FORMAL_OK = {"aggregate_baseline_check": "ok", "expected_eval_token_permissions": EXPECTED_TOKEN_PERMISSIONS,
             "registration_verified": True}


FORBIDDEN_ZERO = {lab: 0 for lab in agg.EXPECTED_FORBIDDEN_LABELS}


def _instrument_ok():
    return {"target_resolution": {lab: {"resolved": True} for lab in agg.EXPECTED_FORBIDDEN_LABELS},
            "call_counts": dict(FORBIDDEN_ZERO), "unresolved_targets": [], "prior_profiler_present": False,
            "threads_at_start": 1, "side_effect_events": [],
            "monitor_state": {"started": True, "stopped": True, "hook_violations": [], "monitor_errors": [],
                              "thread_starts": [], "thread_checks": 100, "unobserved_thread_violations": []}}


FIVE = ("danger_score", "danger_breakdown", "upset_score", "rank_index", "featured_boats")


def _race(eid, cat="diff_zero", ok=True, counts=None):
    # values・comparable_hash は、評価側と同じ形（両側とも既決の5項目、SHA-256 の16進64桁）にする
    return {"eval_id": eid, "result_category": cat, "d9_states": [] if ok else ["compare_not_established"],
            "counts": dict(FORBIDDEN_ZERO, **(counts or {})), "d5": {},
            "values": {"legacy": {k: 0 for k in FIVE}, "rebuild": {k: 0 for k in FIVE}},
            "comparable_hash": {"legacy": "a" * 64, "rebuild": "a" * 64},
            "hashes": {}, "d13_2": {}, "diffs": [], "input_identity_ok": ok}


def _programs_for(date, labels):
    """ラベル（"a" など）ごとに出走表の要素を作り、評価側と同じ規則で eval_id・一覧・ハッシュを作る。"""
    progs = [{"race_stadium_number": 1, "race_number": i + 1, "race_closed_at": f"2099-01-01T{i:05d}",
              "race_date": date} for i in range(len(labels))]
    eids = {lab: f"{date}_01_{i + 1:02d}" for i, lab in enumerate(labels)}
    targets = [{"eval_id": eids[lab], "venue_num": 1, "race_number": i + 1,
                "race_closed_at": progs[i]["race_closed_at"], "race_date_in_program": date}
               for i, lab in enumerate(labels)]
    raw = json.dumps({"programs": progs}, ensure_ascii=False).encode("utf-8")
    tsha = hashlib.sha256(json.dumps(targets, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()
    psha = {eids[lab]: hashlib.sha256(json.dumps(progs[i], sort_keys=True, ensure_ascii=False, allow_nan=True,
                                                 separators=(",", ":")).encode("utf-8")).hexdigest()
            for i, lab in enumerate(labels)}
    return raw, {"raw_sha256": hashlib.sha256(raw).hexdigest(), "targets": targets, "targets_sha256": tsha,
                 "http_get_calls": 1}, eids, psha


def _run_json(date, targets, config_detected=False, preload_ok=True):
    return {"status": "completed", "programs": {"targets": [{"eval_id": t} for t in targets]},
            "config_detection": {"detected": config_detected}, "preload": {"ok": preload_ok},
            "checks": {"env_self_check": [], "baseline_diffs": [], "modules_pre_eval": {"ok": True},
                       "modules_post_eval": {"ok": True}, "tests_not_loaded": {"T3_ok": True},
                       "shadow_package_loaded": [],
                       "token": {"github_token_present": False, "token_like_env_names": []},
                       "baseline_identity": {"path": "baseline.json", "sha256": BASELINE_SHA,
                                             "sha256_at_end": BASELINE_SHA, "unchanged_during_run": True}},
            "info": {"race_date": date, "github_ref": "refs/heads/main",
                     "github_workflow_ref": "o/r/.github/workflows/formal_g2_evaluate.yml@refs/heads/main"},
            "instrument": _instrument_ok()}


def _make_zip(files, manifest_extra=None, complete=True):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        listed = {}
        for name, data in files.items():
            z.writestr(name, data)
            listed[name] = {"sha256": hashlib.sha256(data).hexdigest()}
        manifest = {"files": listed, "complete": complete}
        manifest.update(manifest_extra or {})
        z.writestr("manifest.json", json.dumps(manifest))
    return buf.getvalue()


def _write_run(base, rid, att, date, races, targets, digest_ok=True, complete=True, started="2099-01-02T00:00:00Z",
               mode="formal", permission="P1", config_detected=False, preload_ok=True,
               log_text=None, log_fetch_ok=True, env_image=None, run_overrides=None, check_overrides=None):
    d = os.path.join(base, f"{rid}-{att}")
    os.makedirs(d)
    run = _run_json(date, targets, config_detected, preload_ok)
    # 条件5：評価側と同じ規則の programs_raw.json・一覧・program_sha256 を付ける。ラベルは eval_id に置き換える
    raw, programs, eids, psha = _programs_for(date, list(targets))  # レースの記録は対象一覧の中にあるものとする
    run["programs"] = programs
    races = [dict(r, eval_id=eids[r["eval_id"]], program_sha256=psha[eids[r["eval_id"]]]) for r in races]
    # 実行メタデータ：run.json・manifest・meta.json（API の値）を一致させる
    run["info"].update({"mode": mode, "permission_id": permission, "github_run_id": str(rid),
                        "github_run_attempt": str(att), "github_sha": "0" * 40,
                        "started_at": "2099-01-01T09:00:00+09:00"})
    run["finished_at"] = "2099-01-01T09:05:00+09:00"
    run["env_snapshot"] = {"image": env_image or dict(Q8_ENV_IMAGE)}
    base_info = dict(run["info"])
    run.update(run_overrides or {})
    for k, v in base_info.items():  # info を置き換えたテストでも、識別情報は残す（ref などの上書きは保つ）
        run["info"].setdefault(k, v)
    for k, v in (check_overrides or {}).items():
        if v is None:
            run["checks"].pop(k, None)
        else:
            run["checks"][k] = v
    files = {
        "run.json": json.dumps(run).encode(),
        "races.jsonl": "\n".join(json.dumps(r) for r in races).encode(),
        "programs_raw.json": raw,
    }
    blob = _make_zip(files, {"run": dict(run["info"])}, complete)
    with open(os.path.join(d, "artifact.zip"), "wb") as f:
        f.write(blob)
    digest = hashlib.sha256(blob).hexdigest() if digest_ok else "0" * 64
    log = (log_text if log_text is not None else Q8_LOG_HEAD + EVAL_LOG_TAIL).encode("utf-8")
    meta = {"run_id": str(rid), "run_attempt": str(att), "expected_digest": "sha256:" + digest,
            "run_started_at": started, "job_id": "125133673866", "job_count": 1,
            "log_fetch_ok": log_fetch_ok, "log_fetch_error": None if log_fetch_ok else "HTTP 404",
            "log_file": "job.log" if log_fetch_ok else None,
            "log_sha256": hashlib.sha256(log).hexdigest() if log_fetch_ok else None,
            "log_bytes": len(log) if log_fetch_ok else None, "log_fetched_at": "2099-01-02T00:00:00Z",
            "latest_attempt": str(att), "api_run_id": str(rid), "api_run_attempt": str(att),
            "api_head_sha": "0" * 40, "api_head_branch": "main",
            "api_workflow_path": ".github/workflows/formal_g2_evaluate.yml", "api_repository": "o/r",
            "api_event": "workflow_dispatch"}
    if log_fetch_ok:
        with open(os.path.join(d, "job.log"), "wb") as f:
            f.write(log)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f)
    return d


PERMS = [{"permission_id": "P1", "race_date": "20990101",
          "window_start": "2099-01-01T00:00:00Z", "window_end": "2099-01-03T00:00:00Z"}]


class VerifyTests(unittest.TestCase):
    def test_digest_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as base:
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], digest_ok=False)
            self.assertFalse(agg.verify_run(d)["verified"])

    def test_incomplete_manifest_fails(self):
        with tempfile.TemporaryDirectory() as base:
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], complete=False)
            self.assertFalse(agg.verify_run(d)["verified"])

    def test_missing_digest_is_unverified(self):
        with tempfile.TemporaryDirectory() as base:
            d = _write_run(base, 1, 1, "20990101", [_race("a")], ["a"])
            meta = os.path.join(d, "meta.json")
            with open(meta, encoding="utf-8") as f:
                m = json.load(f)
            m["expected_digest"] = ""
            with open(meta, "w", encoding="utf-8") as f:
                json.dump(m, f)
            self.assertFalse(agg.verify_run(d)["verified"])


class CountingTests(unittest.TestCase):
    def test_mismatch_ends_segment_and_unprocessed_ends_segment(self):
        with tempfile.TemporaryDirectory() as base:
            races = [_race("a"), _race("b"), _race("c", "mismatch"), _race("d")]
            _write_run(base, 1, 1, "20990101", races, ["a", "b", "c", "d", "e"])
            v = [agg.verify_run(os.path.join(base, x)) for x in os.listdir(base)]
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)
            self.assertEqual([s["count"] for s in res["segments"]], [2, 1])
            self.assertEqual(res["segments"][1]["ended_by"], "unprocessed")
            self.assertEqual(res["g2_status"], "undetermined")

    def test_later_run_does_not_replace_first(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a", "mismatch")], ["a"], started="2099-01-01T01:00:00Z")
            _write_run(base, 2, 1, "20990101", [_race("a")], ["a"], started="2099-01-01T02:00:00Z")
            v = [agg.verify_run(os.path.join(base, x)) for x in sorted(os.listdir(base))]
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)
            self.assertEqual(res["days"][0]["selected_run"]["run_id"], "1")
            self.assertEqual(res["best_segment"], 0)

    def test_run_without_permission_is_not_formal(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], permission="OTHER")
            v = [agg.verify_run(os.path.join(base, x)) for x in os.listdir(base)]
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)
            self.assertEqual(res["days"][0]["class"], "pending_confirmation")

    def test_duplicate_eval_id(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a"), _race("a")], ["a"])
            v = [agg.verify_run(os.path.join(base, x)) for x in os.listdir(base)]
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)
            self.assertEqual(res["sequence"][0]["state"], "duplicate_eval_id")

    def test_config_detected_run_excludes_all_races(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a"), _race("b")], ["a", "b"], config_detected=True)
            v = [agg.verify_run(os.path.join(base, x)) for x in os.listdir(base)]
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)
            self.assertEqual(res["best_segment"], 0)
            self.assertEqual(res["days"][0]["excluded"], "config_detected")
            self.assertEqual([s for s in res["sequence"] if "state" in s], [])

    def test_failed_preload_is_not_added(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], preload_ok=False)
            v = [agg.verify_run(os.path.join(base, x)) for x in os.listdir(base)]
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)
            self.assertEqual(res["best_segment"], 0)

    def test_aggregator_does_not_import_evaluation_code(self):
        import subprocess
        import sys

        root = os.path.realpath(os.getcwd())
        code = ("import sys; import formal_g2.aggregate.main;"
                "bad=[n for n in sys.modules if n.split('.')[0] in "
                "('notify_arashi','x_asahi_scoring','x_venue_stats','core','adapters','models')];"
                "print(bad)")
        p = subprocess.run([sys.executable, "-B", "-P", "-c", code], cwd=root,
                           env=dict(os.environ, PYTHONPATH=os.path.join(root, "rebuild")),
                           capture_output=True, text=True, timeout=120)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(p.stdout.strip(), "[]")


class ThreadObservationCheckTests(unittest.TestCase):
    """集約側の独立検査：monitor_state.unobserved_thread_violations と thread_checks（2*N+2 以上）。"""

    def _run(self, n_targets=3, status="completed", **ms_overrides):
        inst = _instrument_ok()
        inst["monitor_state"]["thread_checks"] = 2 * n_targets + 2
        for k, v in ms_overrides.items():
            if v is _DROP:
                inst["monitor_state"].pop(k, None)
            else:
                inst["monitor_state"][k] = v
        return {"status": status, "instrument": inst,
                "programs": {"targets": [{"eval_id": f"e{i}"} for i in range(n_targets)]}}

    def _problems(self, run, n=None):
        return agg.w7_run_problems(run, n)

    def test_clean_completed_run(self):
        self.assertEqual(self._problems(self._run(), 3), [])
        self.assertEqual(self._problems(self._run()), [])  # N を省略すると対象一覧の件数を使う

    def test_unobserved_violation(self):
        run = self._run(unobserved_thread_violations=[{"where": "after e1", "count": 1, "alive": 2}])
        self.assertIn("unobserved live threads: 1", self._problems(run, 3))

    def test_monitor_state_missing(self):
        run = self._run()
        run["instrument"].pop("monitor_state")
        self.assertIn("monitor state missing", self._problems(run, 3))
        run["instrument"]["monitor_state"] = "broken"
        self.assertIn("monitor state missing", self._problems(run, 3))

    def test_fields_missing_or_wrong_type(self):
        cases = {
            "violations missing": (dict(unobserved_thread_violations=_DROP), "unobserved thread record missing"),
            "violations not list": (dict(unobserved_thread_violations={"x": 1}), "unobserved thread record has wrong type"),
            "checks missing": (dict(thread_checks=_DROP), "thread_checks missing"),
        }
        for name, (ov, expected) in cases.items():
            with self.subTest(name=name):
                self.assertIn(expected, self._problems(self._run(**ov), 3))
        for bad in (True, False, "8", -1, 8.0, None):
            with self.subTest(thread_checks=bad):
                probs = self._problems(self._run(thread_checks=bad), 3)
                self.assertTrue(any(p.startswith("thread_checks has wrong type or value") for p in probs), probs)

    def test_threshold(self):
        n = 3
        self.assertTrue(any("thread_checks 7 < 2*3+2" in p for p in self._problems(self._run(thread_checks=7), n)))
        self.assertEqual(self._problems(self._run(thread_checks=8), n), [])
        self.assertEqual(self._problems(self._run(thread_checks=50), n), [])
        # races.jsonl の実際の件数が多ければ、それに合わせて求める
        self.assertTrue(self._problems(self._run(thread_checks=8), 4))

    def test_incomplete_or_stopped_runs_are_not_eligible(self):
        for status in ("stopped", "running", None):
            with self.subTest(status=status):
                run = self._run(status=status)
                probs = self._problems(run, 1)
                self.assertIn(f"run not completed (status={status!r})", probs)
                self.assertFalse(agg.run_level_ok(run, 1))
        # 途中停止で N が分からない場合も、適格扱いにしない
        run = self._run(status="stopped")
        self.assertIn("processed race count unknown (thread_checks cannot be verified)", self._problems(run))

    def test_aggregate_rejects_completed_formal_run_with_unobserved_thread(self):
        with tempfile.TemporaryDirectory() as base:
            inst = _instrument_ok()
            inst["monitor_state"]["unobserved_thread_violations"] = [{"where": "after a", "count": 1, "alive": 2}]
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], run_overrides={"instrument": inst})
            v = [agg.verify_run(os.path.join(base, x)) for x in os.listdir(base)]
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)
            self.assertEqual(res["days"][0]["excluded"], "w7_monitoring")
            self.assertEqual(res["best_segment"], 0)

    def test_aggregate_rejects_completed_formal_run_with_too_few_checks(self):
        with tempfile.TemporaryDirectory() as base:
            inst = _instrument_ok()
            inst["monitor_state"]["thread_checks"] = 3  # 1レースなら 4 以上が必要
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], run_overrides={"instrument": inst})
            v = [agg.verify_run(os.path.join(base, x)) for x in os.listdir(base)]
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA, FORMAL_OK)
            self.assertEqual(res["days"][0]["excluded"], "w7_monitoring")


_DROP = object()


if __name__ == "__main__":
    unittest.main()
