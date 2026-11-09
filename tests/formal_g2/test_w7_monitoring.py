"""正式G2：禁止関数の実行時監視（W7・C-5・通知・Release 書き込み）と fail-closed。

- 実際の禁止対象（A〜D・ML）は、関数を実行せずに、そのコードオブジェクトで照合する
  （通知・Release 書き込みを実際に動かさないため）。
- 呼び出しの捕捉の仕組み（別名・メソッド・デコレータ・import 時・スレッド）は、
  一時ディレクトリの合成モジュールで実際に呼び出して確かめる。

実行（リポジトリ直下で）：
  PYTHONPATH="rebuild:." python -P -B -m unittest discover -s tests/formal_g2 -t tests/formal_g2
"""

import _thread
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest

from formal_g2 import instrument as ins
from formal_g2.aggregate import main as agg

ROOT = os.path.realpath(os.getcwd())


class FakeFrame:
    def __init__(self, code):
        self.f_code = code


class RealTargetTests(unittest.TestCase):
    def test_all_targets_resolve_and_lists_match(self):
        inst = ins.Instrument(ROOT)
        inst._resolve_targets()
        unresolved = [k for k, v in inst.resolution.items() if not v["resolved"]]
        self.assertEqual(unresolved, [])
        self.assertEqual(len(ins.FORBIDDEN_LABELS), 47)  # 新規 42 件 + 既存の ML・Legacy 取得 5 件
        self.assertEqual({c: tuple(v) for c, v in ins.FORBIDDEN_CATEGORIES.items()}, agg.EXPECTED_FORBIDDEN)
        self.assertNotIn("x_release_storage.download_file", ins.FORBIDDEN_LABELS)

    def test_code_objects_match_and_protocols_do_not(self):
        code = textwrap.dedent("""
            import json, os
            from formal_g2 import instrument as ins
            import notify_arashi, x_buyscore, x_release_storage
            import core.buyscore as cb
            mods = {"notify_arashi.py": notify_arashi, "x_buyscore.py": x_buyscore,
                    "x_release_storage.py": x_release_storage, "rebuild/core/buyscore.py": cb}
            inst = ins.Instrument(os.getcwd()); inst._resolve_targets()
            out = {}
            for cat, items in ins._FORBIDDEN_SPEC.items():
                for path, qual, label in items:
                    obj = mods[path]
                    for part in qual.split("."):
                        obj = getattr(obj, part)
                    c = obj.__code__
                    out[label] = inst._targets.get((c.co_filename, c.co_name, c.co_firstlineno))
            proto = [cb.KellyStrategy.__call__.__code__, cb.BuyEngine.assess.__code__]
            out["_protocol_hits"] = [inst._targets.get((c.co_filename, c.co_name, c.co_firstlineno)) for c in proto]
            alias = notify_arashi._apply_buyscore.__code__
            out["_alias"] = inst._targets.get((alias.co_filename, alias.co_name, alias.co_firstlineno))
            print(json.dumps(out))
        """)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "rebuild"), ROOT]))
        env.pop("GITHUB_TOKEN", None)
        p = subprocess.run([sys.executable, "-B", "-P", "-c", code], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=600)
        self.assertEqual(p.returncode, 0, p.stderr[-2000:])
        out = json.loads(p.stdout.strip().splitlines()[-1])
        for label in ins.FORBIDDEN_LABELS:
            self.assertEqual(out[label], label, label)
        self.assertEqual(out["_protocol_hits"], [None, None])
        self.assertEqual(out["_alias"], "x_buyscore.apply_buyscore")

    def test_each_category_call_is_counted_and_fails_closed(self):
        import notify_arashi
        import x_buyscore
        import x_release_storage
        import core.buyscore as cb
        samples = {
            "A_buyscore": [x_buyscore.calc_buyscore, cb.DefaultBuyEngine.assess, notify_arashi._evaluate_bets],
            "B_ml_and_legacy_fetch": [notify_arashi._predict_win_prob, notify_arashi._safe_get],
            "C_notify": [notify_arashi.send_email, notify_arashi._run_main],
            "D_release_write": [x_release_storage.upload_file],
        }
        for cat, funcs in samples.items():
            for fn in funcs:
                with self.subTest(cat=cat, fn=fn.__qualname__):
                    inst = ins.Instrument(ROOT)
                    inst._resolve_targets()
                    inst.started, inst.threads_at_start, inst.prior_profiler_present = True, 1, False
                    inst._profile(FakeFrame(fn.__code__), "call", None)
                    self.assertEqual(sum(inst.forbidden_counts()[cat].values()), 1)
                    self.assertIn("forbidden calls: 1", inst.monitoring_problems())

    def test_allowed_download_file_is_not_flagged(self):
        import x_release_storage
        inst = ins.Instrument(ROOT)
        inst._resolve_targets()
        inst.started, inst.threads_at_start, inst.prior_profiler_present = True, 1, False
        inst._profile(FakeFrame(x_release_storage.download_file.__code__), "call", None)
        self.assertEqual(inst.counts["x_release_storage.download_file"], 1)
        self.assertEqual(inst.monitoring_problems(), [])


SYNTH = '''
import functools

CALLS = []

def target(x=0):
    CALLS.append(x)
    return x

def deco(fn):
    @functools.wraps(fn)
    def wrapper(*a, **k):
        return fn(*a, **k)
    return wrapper

@deco
def decorated():
    return 1

class Engine:
    def assess(self):
        return 2

class Proto:
    def assess(self):
        return 3

target(-1)  # import 時の呼び出し
'''


class MechanismTests(unittest.TestCase):
    """合成モジュールで、実際の呼び出しの捕捉を確かめる。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self.tmp.name)
        with open(os.path.join(self.root, "synth_mod.py"), "w", encoding="utf-8") as f:
            f.write(SYNTH)
        self.watched = {("synth_mod.py", "target"): "target", ("synth_mod.py", "decorated"): "decorated",
                        ("synth_mod.py", "Engine.assess"): "Engine.assess"}

    def tearDown(self):
        sys.setprofile(None)
        threading.setprofile(None)
        sys.modules.pop("synth_mod_test", None)
        self.tmp.cleanup()

    def _load(self):
        spec = importlib.util.spec_from_file_location("synth_mod_test", os.path.join(self.root, "synth_mod.py"))
        mod = importlib.util.module_from_spec(spec)
        sys.modules["synth_mod_test"] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_alias_method_decorator_and_import_time(self):
        inst = ins.Instrument(self.root, watched=self.watched, watched_modules={})
        inst.start()
        try:
            inst.set_phase("import")
            mod = self._load()
            inst.set_phase("evaluate")
            alias = mod.target
            alias(1)
            mod.Engine().assess()
            mod.Proto().assess()   # 同じ名前の別クラスは数えない
            mod.decorated()        # wraps したラッパー経由でも、中の関数を数える
        finally:
            inst.stop()
        self.assertEqual(inst.counts, {"target": 2, "decorated": 1, "Engine.assess": 1})
        self.assertEqual(inst.phase_counts["import"], {"target": 1})
        self.assertTrue(inst.check_hooks.__self__ is inst)

    def test_thread_via_threading_is_monitored(self):
        inst = ins.Instrument(self.root, watched=self.watched, watched_modules={})
        inst.start()
        try:
            mod = self._load()
            t = threading.Thread(target=mod.target, args=(5,))
            t.start()
            t.join()
        finally:
            inst.stop()
        self.assertEqual(inst.counts["target"], 2)  # import 時 + スレッド内
        self.assertEqual([s["via_threading"] for s in inst.thread_starts], [True])
        self.assertFalse([p for p in inst.monitoring_problems() if "thread" in p])

    def test_thread_with_exception_still_counted(self):
        inst = ins.Instrument(self.root, watched=self.watched, watched_modules={})
        inst.start()
        try:
            mod = self._load()
            old = threading.excepthook
            threading.excepthook = lambda args: None
            try:
                t = threading.Thread(target=lambda: (mod.target(9), 1 / 0))
                t.start()
                t.join()
            finally:
                threading.excepthook = old
        finally:
            inst.stop()
        self.assertEqual(inst.counts["target"], 2)

    def test_raw_thread_is_not_guaranteed_and_fails_closed(self):
        inst = ins.Instrument(self.root, watched=self.watched, watched_modules={})
        inst.start()
        done = threading.Event()
        try:
            _thread.start_new_thread(done.set, ())
            done.wait(5)
            time.sleep(0.05)
        finally:
            inst.stop()
        self.assertEqual([s["via_threading"] for s in inst.thread_starts], [False])
        self.assertTrue(any("outside threading" in p for p in inst.monitoring_problems()))

    def test_hook_removed_or_replaced_fails_closed(self):
        for action in ("remove_sys", "replace_sys", "replace_threading"):
            with self.subTest(action=action):
                inst = ins.Instrument(self.root, watched=self.watched, watched_modules={})
                inst.start()
                try:
                    if action == "remove_sys":
                        sys.setprofile(None)
                    elif action == "replace_sys":
                        sys.setprofile(lambda *a: None)
                    else:
                        threading.setprofile(lambda *a: None)
                    self.assertFalse(inst.check_hooks("test"))
                finally:
                    inst.stop()
                self.assertTrue(any("hook removed or replaced" in p for p in inst.monitoring_problems()))

    def test_prior_profiler_fails_closed(self):
        sys.setprofile(lambda *a: None)
        inst = ins.Instrument(self.root, watched=self.watched, watched_modules={})
        try:
            inst.start()
        finally:
            inst.stop()
        self.assertTrue(inst.prior_profiler_present)
        self.assertIn("profiler already present before monitoring", inst.monitoring_problems())

    def test_unresolved_fails_closed(self):
        inst = ins.Instrument(self.root, watched={**self.watched, ("synth_mod.py", "missing"): "missing"},
                              watched_modules={})
        inst.start()
        inst.stop()
        self.assertTrue(any(p.startswith("unresolved targets") for p in inst.monitoring_problems()))

    def test_monitor_error_fails_closed(self):
        inst = ins.Instrument(self.root, watched=self.watched, watched_modules={})
        inst.start()
        try:
            inst._targets = None  # 監視処理の中で例外を起こす
            self._load()
        finally:
            inst.stop()
        self.assertTrue(inst.monitor_errors)
        self.assertTrue(any(p.startswith("monitor errors") for p in inst.monitoring_problems()))

    def test_process_start_fails_closed(self):
        inst = ins.Instrument(self.root, watched=self.watched, watched_modules={})
        inst.start()
        try:
            subprocess.run([sys.executable, "-c", "pass"], check=True)
        finally:
            inst.stop()
        self.assertIn("process started (not monitored)", inst.monitoring_problems())


class AggregateW7Tests(unittest.TestCase):
    """集約は評価側の印を信用せず、記録を独立に検査する。"""

    def setUp(self):
        from test_aggregate import _instrument_ok
        self.ok = _instrument_ok()

    def _run(self, **inst_overrides):
        inst = json.loads(json.dumps(self.ok))
        inst.update(inst_overrides)
        return {"instrument": inst, "status": "completed", "programs": {"targets": [{"eval_id": "a"}]}}

    def test_good_record_passes(self):
        self.assertEqual(agg.w7_run_problems(self._run()), [])

    def test_rejections(self):
        counts = dict(self.ok["call_counts"])
        counts["x_buyscore.apply_buyscore"] = 1
        missing_label = dict(self.ok["call_counts"])
        missing_label.pop("notify_arashi.send_email")
        res_unresolved = dict(self.ok["target_resolution"], **{"x_buyscore.stars": {"resolved": False}})
        ms_bad_thread = dict(self.ok["monitor_state"], thread_starts=[{"via_threading": False}])
        ms_hook = dict(self.ok["monitor_state"], hook_violations=[{"where": "x"}])
        ms_err = dict(self.ok["monitor_state"], monitor_errors=["profile: TypeError"])
        cases = {
            "missing record": {"instrument": None},
            "forbidden call": self._run(call_counts=counts),
            "label not monitored": self._run(call_counts=missing_label),
            "unresolved": self._run(target_resolution=res_unresolved),
            "prior profiler": self._run(prior_profiler_present=True),
            "prior profiler unrecorded": self._run(prior_profiler_present=None),
            "threads at start": self._run(threads_at_start=2),
            "raw thread": self._run(monitor_state=ms_bad_thread),
            "hook": self._run(monitor_state=ms_hook),
            "monitor error": self._run(monitor_state=ms_err),
            "monitor state missing": self._run(monitor_state=None),
            "process": self._run(side_effect_events=[{"kind": "process"}]),
        }
        for name, run in cases.items():
            with self.subTest(name=name):
                if run.get("instrument") is None:
                    run = {}
                self.assertTrue(agg.w7_run_problems(run), name)

    def test_race_with_forbidden_call_is_not_added(self):
        from test_aggregate import _race
        self.assertEqual(agg.race_state(_race("a", counts={"notify_arashi.send_line": 1}), True), "forbidden_call")
        rec = _race("a")
        rec["counts"].pop("score_by_model")
        self.assertEqual(agg.race_state(rec, True), "evidence_incomplete")
        self.assertEqual(agg.race_state(_race("a", counts={"x_release_storage.download_file": 1}), True), "added")


class AggregateFormalRefAndW7ExitTests(unittest.TestCase):
    def _main(self, tmp, mode="formal", run_overrides=None, races=None):
        from test_aggregate import BASELINE_RAW, PERMS, _race, _write_run, permitted_by
        base = os.path.join(tmp, "a")
        os.makedirs(base)
        _write_run(base, 1, 1, "20990101", races or [_race("a")], ["a"], mode=mode, run_overrides=run_overrides)
        bpath, perms = os.path.join(tmp, "b.json"), os.path.join(tmp, "p.json")
        with open(bpath, "wb") as f:
            f.write(BASELINE_RAW)
        with open(perms, "w", encoding="utf-8") as f:
            json.dump(PERMS, f)
        out = os.path.join(tmp, "out")
        from contextlib import redirect_stdout
        from io import StringIO
        with permitted_by(), redirect_stdout(StringIO()):
            rc = agg.main(["--artifacts-dir", base, "--first-date", "20990101", "--last-date", "20990101",
                           "--out", out, "--baseline", bpath])
        with open(os.path.join(out, "aggregate_result.json"), encoding="utf-8") as f:
            return rc, json.load(f)

    def test_formal_on_non_main_ref_is_rejected(self):
        bad = {"info": {"race_date": "20990101", "github_ref": "refs/heads/feature/formal-g2-independent",
                        "github_workflow_ref": "o/r/.github/workflows/x.yml@refs/heads/feature/formal-g2-independent"}}
        with tempfile.TemporaryDirectory() as tmp:
            rc, res = self._main(tmp, run_overrides=bad)
        self.assertEqual(rc, agg.EXIT_VERIFICATION_FAILED)
        self.assertEqual(res["days"][0]["excluded"], "ref_not_main")
        self.assertEqual(res["best_segment"], 0)

    def test_investigation_on_feature_ref_is_not_ref_checked(self):
        bad = {"info": {"race_date": "20990101", "github_ref": "refs/heads/feature/formal-g2-independent",
                        "github_workflow_ref": "o/r/x.yml@refs/heads/feature/formal-g2-independent"}}
        with tempfile.TemporaryDirectory() as tmp:
            rc, res = self._main(tmp, mode="investigation", run_overrides=bad)
        self.assertEqual(rc, 0)
        self.assertNotIn("excluded", res["days"][0])
        self.assertEqual(res["runs"][0]["ref"]["github_ref"], "refs/heads/feature/formal-g2-independent")

    def test_completed_formal_run_with_forbidden_call_is_rejected(self):
        from test_aggregate import _instrument_ok
        inst = _instrument_ok()
        inst["call_counts"]["notify_arashi._evaluate_bets"] = 1
        with tempfile.TemporaryDirectory() as tmp:
            rc, res = self._main(tmp, run_overrides={"instrument": inst})
        self.assertEqual(rc, agg.EXIT_VERIFICATION_FAILED)
        self.assertEqual(res["days"][0]["excluded"], "w7_monitoring")
        self.assertTrue(res["runs"][0]["w7_check"])

    def test_missing_monitoring_record_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rc, res = self._main(tmp, run_overrides={"instrument": None})
        self.assertEqual(rc, agg.EXIT_VERIFICATION_FAILED)
        self.assertEqual(res["days"][0]["excluded"], "w7_monitoring")


_EVAL_CODE = r"""
import json, os, sys
import requests
sys.path.append(os.path.join(os.getcwd(), "tests", "formal_g2"))
from test_programs_synthetic import RAW
import formal_g2.programs as P
mode, out = sys.argv[1], sys.argv[2]
def fake(url):
    if mode == "forbidden":
        import x_buyscore
        x_buyscore.stars(50.0, {})  # 禁止対象（純粋な関数。副作用なし）
    r = requests.models.Response(); r.status_code = 200; r._content = RAW; r.url = url
    return r
P.default_http_get = fake
from formal_g2 import evaluate_main
rc = evaluate_main.main(["--race-date", "20990101", "--mode", "investigation", "--out", out])
print("RC", rc)
"""


class EvaluateSideW7Tests(unittest.TestCase):
    def _run(self, mode):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "out")
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "rebuild"), ROOT]))
            env.pop("GITHUB_TOKEN", None)
            p = subprocess.run([sys.executable, "-B", "-P", "-c", _EVAL_CODE, mode, out], cwd=ROOT, env=env,
                               capture_output=True, text=True, timeout=600)
            self.assertEqual(p.returncode, 0, p.stderr[-2000:])
            rc = int([ln for ln in p.stdout.splitlines() if ln.startswith("RC ")][-1].split()[1])
            with open(os.path.join(out, "run.json"), encoding="utf-8") as f:
                run = json.load(f)
            stop = None
            if os.path.exists(os.path.join(out, "stop.json")):
                with open(os.path.join(out, "stop.json"), encoding="utf-8") as f:
                    stop = json.load(f)
            return rc, run, stop, p.stdout

    def test_clean_run_records_and_summarizes(self):
        rc, run, stop, log = self._run("clean")
        self.assertEqual(rc, 0, stop)
        self.assertTrue(run["w7"]["ok"])
        self.assertEqual(run["w7"]["forbidden_total"], 0)
        self.assertEqual(run["instrument"]["unresolved_targets"], [])
        self.assertIn("G2 W7: ok=True forbidden_total=0", log)
        self.assertIn("G2 13-2: {'races': 3, 'd13_2_all_true': 3, 'diff_zero': 3, 'w7_ok': 3}", log)

    def test_forbidden_call_stops_at_next_boundary(self):
        rc, run, stop, log = self._run("forbidden")
        self.assertEqual(rc, 2)
        self.assertEqual(stop["code"], "W7")
        self.assertFalse(run["w7"]["ok"])
        self.assertEqual(run["w7"]["forbidden_by_category"]["A_buyscore"], 1)
        self.assertEqual(run["instrument"]["counts_by_phase"]["programs"]["x_buyscore.stars"], 1)
        self.assertEqual(stop["processed_eval_ids_count"], 0)  # 最初のレースを始める前に止まる
        self.assertTrue(agg.w7_run_problems(run))


_EVAL_IN_RACE = r"""
import json, os, sys
import requests
sys.path.append(os.path.join(os.getcwd(), "tests", "formal_g2"))
from test_programs_synthetic import RAW
import formal_g2.programs as P
out = sys.argv[1]
def fake(url):
    r = requests.models.Response(); r.status_code = 200; r._content = RAW; r.url = url
    return r
P.default_http_get = fake
_orig = P.FixedFetcher.__call__
_n = {"calls": 0}
def patched(self, race_date):
    _n["calls"] += 1
    if _n["calls"] == 2:          # 2つ目のレースの評価の途中で、禁止対象を1回呼ぶ
        import x_buyscore
        x_buyscore.stars(50.0, {})
    return _orig(self, race_date)
P.FixedFetcher.__call__ = patched
from formal_g2 import evaluate_main
print("RC", evaluate_main.main(["--race-date", "20990101", "--mode", "investigation", "--out", out]))
"""


class InRaceForbiddenCallTests(unittest.TestCase):
    """レースの評価の途中で禁止対象が呼ばれた場合（合成の出走表。3レース）。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        out = os.path.join(cls.tmp.name, "out")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "rebuild"), ROOT]))
        env.pop("GITHUB_TOKEN", None)
        p = subprocess.run([sys.executable, "-B", "-P", "-c", _EVAL_IN_RACE, out], cwd=ROOT, env=env,
                           capture_output=True, text=True, timeout=600)
        assert p.returncode == 0, p.stderr[-2000:]
        cls.rc = int([ln for ln in p.stdout.splitlines() if ln.startswith("RC ")][-1].split()[1])
        with open(os.path.join(out, "run.json"), encoding="utf-8") as f:
            cls.run_json = json.load(f)
        with open(os.path.join(out, "stop.json"), encoding="utf-8") as f:
            cls.stop = json.load(f)
        with open(os.path.join(out, "races.jsonl"), encoding="utf-8") as f:
            cls.races = [json.loads(x) for x in f if x.strip()]
        cls.targets = [t["eval_id"] for t in cls.run_json["programs"]["targets"]]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_stops_after_the_race_and_does_not_start_next(self):
        self.assertEqual(self.rc, 2)
        self.assertEqual(self.stop["code"], "W7")
        self.assertEqual(self.stop["reason"], f"monitoring problem at after {self.targets[1]}")
        self.assertEqual([r["eval_id"] for r in self.races], self.targets[:2])  # 3つ目は始めない
        self.assertEqual(self.stop["last_started_eval_id"], self.targets[1])

    def test_race_records_forbidden_call_without_overwriting_13_2(self):
        first, second = self.races
        self.assertTrue(first["w7"]["ok"])
        self.assertEqual(second["w7"], {"forbidden_calls": {"x_buyscore.stars": 1}, "forbidden_total": 1,
                                        "labels_checked": len(ins.FORBIDDEN_LABELS), "ok": False})
        # 13-2 と比較の結果は W7 で上書きしない（別の項目として残る）
        self.assertTrue(all(v is True for v in second["d13_2"].values()))
        self.assertEqual(second["result_category"], "diff_zero")
        self.assertTrue(second["input_identity_ok"])

    def test_aggregate_rejects_the_race_and_the_run(self):
        first, second = self.races
        self.assertFalse(agg.run_level_ok(self.run_json))
        self.assertTrue(any("forbidden calls" in p for p in agg.w7_run_problems(self.run_json)))
        self.assertEqual(agg.race_state(second, agg.run_level_ok(self.run_json)), "forbidden_call")
        self.assertEqual(agg.race_state(second, True), "forbidden_call")  # run 単位の判定に関係なく
        self.assertNotEqual(agg.race_state(first, agg.run_level_ok(self.run_json)), "added")
        self.assertFalse(self.run_json["w7"]["ok"])


class ThreadClassificationTests(unittest.TestCase):
    """c_call による threading 経由／_thread 直接の区別。"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self.tmp.name)
        with open(os.path.join(self.root, "tmod.py"), "w", encoding="utf-8") as f:
            f.write("def target():\n    return 1\n")
        spec = importlib.util.spec_from_file_location("tmod_test", os.path.join(self.root, "tmod.py"))
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.inst = ins.Instrument(self.root, watched={("tmod.py", "target"): "target"}, watched_modules={})

    def tearDown(self):
        sys.setprofile(None)
        threading.setprofile(None)
        self.tmp.cleanup()

    def _run(self, starter):
        done = threading.Event()

        def body():
            self.seen_before_run = len(self.inst.thread_starts)
            self.mod.target()
            done.set()
        self.inst.start()
        try:
            starter(body)
            done.wait(5)
            time.sleep(0.05)
        finally:
            self.inst.stop()
        return [s["via_threading"] for s in self.inst.thread_starts]

    def test_threading_paths_are_covered_and_recorded_before_run(self):
        from concurrent.futures import ThreadPoolExecutor

        class Sub(threading.Thread):
            def start(self):
                super().start()

        def executor(body):
            with ThreadPoolExecutor(1) as ex:
                ex.submit(body).result()
        starters = {
            "Thread": lambda body: threading.Thread(target=body).start(),
            "Timer": lambda body: threading.Timer(0.01, body).start(),
            "ThreadPoolExecutor": executor,
            "subclass_super_start": lambda body: Sub(target=body).start(),
        }
        for name, starter in starters.items():
            with self.subTest(path=name):
                self.tearDown()
                self.setUp()
                self.assertEqual(self._run(starter), [True])
                self.assertEqual(self.inst.counts["target"], 1)
                self.assertEqual(self.seen_before_run, 1)  # 開始の記録は、新しいスレッドが動く前
                self.assertFalse([p for p in self.inst.monitoring_problems() if "thread" in p])

    def test_direct_thread_paths_are_not_covered(self):
        class RawStart(threading.Thread):
            def start(self):  # threading.Thread.start を通らずに直接起動する
                _thread.start_new_thread(self.run, ())

        starters = {
            "_thread.start_new_thread": lambda body: _thread.start_new_thread(body, ()),
            "via_python_lambda": lambda body: list(map(lambda f: f(body, ()), [_thread.start_new_thread])),
            "subclass_reimplemented_start": lambda body: RawStart(target=body).start(),
        }
        for name, starter in starters.items():
            with self.subTest(path=name):
                self.tearDown()
                self.setUp()
                self.assertEqual(self._run(starter), [False])
                self.assertEqual(self.inst.counts["target"], 0)  # 監視されていない（だから不適格にする）
                self.assertTrue(any("outside threading" in p for p in self.inst.monitoring_problems()))

    # ---- 生きているスレッドの観測の検査（check_threads） ----
    def _partial_start(self, release, started):
        import functools

        def body():
            started.set()
            release.wait(5)
        functools.partial(_thread.start_new_thread, body, ())()

    def test_c_to_c_start_alive_at_boundary_is_ineligible(self):
        """functools.partial 経由（c_call が出ない）でも、区切りの時点で動いていれば不適格。"""
        release, started = threading.Event(), threading.Event()
        self.inst.start()
        try:
            self._partial_start(release, started)
            started.wait(5)
            self.assertEqual(self.inst.thread_starts, [])  # 開始そのものは検知できない
            self.inst.check_hooks("after race-1")  # 戻り値はフックの状態。スレッドの検査は記録に残る
        finally:
            release.set()
            time.sleep(0.05)
            self.inst.stop()
        v = self.inst.unobserved_thread_violations
        self.assertEqual((v[0]["where"], v[0]["count"]), ("after race-1", 1))
        self.assertTrue(any(p.startswith("unobserved live threads at after race-1: 1")
                            for p in self.inst.monitoring_problems()))

    def test_multiple_unobserved_threads(self):
        release = threading.Event()
        starts = [threading.Event(), threading.Event()]
        self.inst.start()
        try:
            for st in starts:
                self._partial_start(release, st)
                st.wait(5)
            self.inst.check_hooks("before race-2")
        finally:
            release.set()
            time.sleep(0.05)
            self.inst.stop()
        self.assertEqual(self.inst.unobserved_thread_violations[0]["count"], 2)

    def test_checks_at_start_and_stop(self):
        release, started = threading.Event(), threading.Event()
        self._partial_start(release, started)  # 監視の開始より前から動いているスレッド
        started.wait(5)
        try:
            self.inst.start()
            self.inst.stop()
        finally:
            release.set()
            time.sleep(0.05)
        wheres = [v["where"] for v in self.inst.unobserved_thread_violations]
        self.assertEqual(wheres, ["start", "stop"])
        self.assertEqual(self.inst.snapshot()["monitor_state"]["thread_checks"], 2)

    def test_monitored_threads_are_not_flagged(self):
        from concurrent.futures import ThreadPoolExecutor
        release = threading.Event()
        self.inst.start()
        try:
            t = threading.Thread(target=release.wait, args=(5,))
            t.start()
            ex = ThreadPoolExecutor(2)
            ex.submit(lambda: None).result()  # 待機中の作業スレッドが生きたまま残る
            tm = threading.Timer(5, lambda: None)
            tm.start()
            time.sleep(0.05)
            self.inst.check_hooks("after race-1")
            ok = not self.inst.unobserved_thread_violations
            tm.cancel()
            tm.join()
            release.set()
            t.join()
            ex.shutdown(wait=True)
        finally:
            self.inst.stop()
        self.assertTrue(ok)
        self.assertEqual(self.inst.unobserved_thread_violations, [])

    def test_generator_in_live_thread_is_not_flagged(self):
        release, ready = threading.Event(), threading.Event()

        def gen():
            for i in range(3):
                yield i

        def body():
            list(gen())
            ready.set()
            release.wait(5)
        self.inst.start()
        try:
            t = threading.Thread(target=body)
            t.start()
            ready.wait(5)
            self.inst.check_hooks("after race-1")
            ok = not self.inst.unobserved_thread_violations
            release.set()
            t.join()
        finally:
            self.inst.stop()
        self.assertTrue(ok)

    def test_ended_thread_is_removed_and_not_flagged_later(self):
        """終わったスレッドの ID は観測済みから外す（ID が再利用されても取り違えない）。"""
        idents = []
        self.inst.start()
        try:
            t = threading.Thread(target=lambda: idents.append(_thread.get_ident()))
            t.start()
            t.join()
            time.sleep(0.05)
            self.assertNotIn(idents[0], self.inst._observed_threads)
            self.inst.check_hooks("after race-1")
            ok = not self.inst.unobserved_thread_violations
            # 同じ ID が別のスレッドに再利用された状態：観測済みの集合に無いので、
            # そのスレッドが監視されていなければ不適格になる（プロファイラを通らない限り追加されない）
            fake_alive = {idents[0]}
            self.assertEqual(sorted(fake_alive - self.inst._observed_threads), [idents[0]])
        finally:
            self.inst.stop()
        self.assertTrue(ok)
        self.assertEqual(self.inst.unobserved_thread_violations, [])

    @unittest.expectedFailure
    def test_known_gap_short_lived_c_to_c_thread(self):
        """既知の限界：C の中から起動され（c_call が出ない）、区切りより前に終わった短命のスレッドは
        検知できない。区切りの時点で動いていないため、check_threads にも現れない。"""
        import functools
        done = threading.Event()
        self.inst.start()
        try:
            functools.partial(_thread.start_new_thread, done.set, ())()
            done.wait(5)
            time.sleep(0.05)
            self.inst.check_hooks("after race-1")
        finally:
            self.inst.stop()
        self.assertTrue(self.inst.monitoring_problems())

if __name__ == "__main__":
    unittest.main()
