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
BASELINE = {"runner_image": {"image": "ubuntu-24.04", "version": "20261025.353.1"}}
BASELINE_RAW = json.dumps(BASELINE, sort_keys=True).encode("utf-8")
BASELINE_SHA = hashlib.sha256(BASELINE_RAW).hexdigest()


def _race(eid, cat="diff_zero", ok=True):
    return {"eval_id": eid, "result_category": cat, "d9_states": [] if ok else ["compare_not_established"],
            "counts": {}, "d5": {}, "values": {}, "comparable_hash": {}, "hashes": {}, "d13_2": {},
            "diffs": [], "input_identity_ok": ok}


def _run_json(date, targets, config_detected=False, preload_ok=True):
    return {"status": "completed", "programs": {"targets": [{"eval_id": t} for t in targets]},
            "config_detection": {"detected": config_detected}, "preload": {"ok": preload_ok},
            "checks": {"env_self_check": [], "baseline_diffs": [], "modules_pre_eval": {"ok": True},
                       "modules_post_eval": {"ok": True}, "tests_not_loaded": {"T3_ok": True},
                       "shadow_package_loaded": [],
                       "baseline_identity": {"path": "baseline.json", "sha256": BASELINE_SHA,
                                             "sha256_at_end": BASELINE_SHA, "unchanged_during_run": True}},
            "info": {"race_date": date}}


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
    run["env_snapshot"] = {"image": env_image or dict(Q8_ENV_IMAGE)}
    run.update(run_overrides or {})
    for k, v in (check_overrides or {}).items():
        if v is None:
            run["checks"].pop(k, None)
        else:
            run["checks"][k] = v
    files = {
        "run.json": json.dumps(run).encode(),
        "races.jsonl": "\n".join(json.dumps(r) for r in races).encode(),
    }
    blob = _make_zip(files, {"run": {"mode": mode, "permission_id": permission, "race_date": date}}, complete)
    with open(os.path.join(d, "artifact.zip"), "wb") as f:
        f.write(blob)
    digest = hashlib.sha256(blob).hexdigest() if digest_ok else "0" * 64
    log = (log_text if log_text is not None else Q8_LOG_HEAD).encode("utf-8")
    meta = {"run_id": str(rid), "run_attempt": str(att), "expected_digest": "sha256:" + digest,
            "run_started_at": started, "job_id": "125133673866", "job_count": 1,
            "log_fetch_ok": log_fetch_ok, "log_fetch_error": None if log_fetch_ok else "HTTP 404",
            "log_file": "job.log" if log_fetch_ok else None,
            "log_sha256": hashlib.sha256(log).hexdigest() if log_fetch_ok else None,
            "log_bytes": len(log) if log_fetch_ok else None, "log_fetched_at": "2099-01-02T00:00:00Z"}
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
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA)
            self.assertEqual([s["count"] for s in res["segments"]], [2, 1])
            self.assertEqual(res["segments"][1]["ended_by"], "unprocessed")
            self.assertEqual(res["g2_status"], "undetermined")

    def test_later_run_does_not_replace_first(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a", "mismatch")], ["a"], started="2099-01-01T01:00:00Z")
            _write_run(base, 2, 1, "20990101", [_race("a")], ["a"], started="2099-01-01T02:00:00Z")
            v = [agg.verify_run(os.path.join(base, x)) for x in sorted(os.listdir(base))]
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA)
            self.assertEqual(res["days"][0]["selected_run"]["run_id"], "1")
            self.assertEqual(res["best_segment"], 0)

    def test_run_without_permission_is_not_formal(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], permission="OTHER")
            v = [agg.verify_run(os.path.join(base, x)) for x in os.listdir(base)]
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA)
            self.assertEqual(res["days"][0]["class"], "pending_confirmation")

    def test_duplicate_eval_id(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a"), _race("a")], ["a"])
            v = [agg.verify_run(os.path.join(base, x)) for x in os.listdir(base)]
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA)
            self.assertEqual(res["sequence"][0]["state"], "duplicate_eval_id")

    def test_config_detected_run_excludes_all_races(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a"), _race("b")], ["a", "b"], config_detected=True)
            v = [agg.verify_run(os.path.join(base, x)) for x in os.listdir(base)]
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA)
            self.assertEqual(res["best_segment"], 0)
            self.assertEqual(res["days"][0]["excluded"], "config_detected")
            self.assertEqual([s for s in res["sequence"] if "state" in s], [])

    def test_failed_preload_is_not_added(self):
        with tempfile.TemporaryDirectory() as base:
            _write_run(base, 1, 1, "20990101", [_race("a")], ["a"], preload_ok=False)
            v = [agg.verify_run(os.path.join(base, x)) for x in os.listdir(base)]
            res = agg.aggregate(v, PERMS, "20990101", "20990101", BASELINE, BASELINE_SHA)
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


if __name__ == "__main__":
    unittest.main()
