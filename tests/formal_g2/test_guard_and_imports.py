"""正式G2：探索順・名前の重なり・tests の扱い・comparator の直接読み込み（W3）。

別プロセスで起動して、プロセス状態の影響を受けないようにする。
実行（リポジトリ直下で）：
  PYTHONPATH="rebuild:." python -P -B -m unittest discover -s tests/formal_g2 -t tests/formal_g2
"""

import json
import os
import subprocess
import sys
import unittest

ROOT = os.path.realpath(os.getcwd())
ENV = dict(os.environ)
ENV.pop("GITHUB_TOKEN", None)
ENV["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "rebuild"), ROOT])


def run_py(code, safe_path=True, env=None):
    args = [sys.executable, "-B"] + (["-P"] if safe_path else []) + ["-c", code]
    return subprocess.run(args, cwd=ROOT, env=env or ENV, capture_output=True, text=True, timeout=600)


class SearchPathTests(unittest.TestCase):
    def test_guard_passes_with_P(self):
        p = run_py("from formal_g2 import guard; import json, os;"
                   "r = guard.check_search_path(os.getcwd());"
                   "c = guard.check_colliding_modules(os.getcwd());"
                   "print(json.dumps({k: v.get('under_rebuild') for k, v in c.items()}))")
        self.assertEqual(p.returncode, 0, p.stderr)
        res = json.loads(p.stdout.strip().splitlines()[-1])
        for name in ("models", "features", "output"):
            self.assertTrue(res[name])

    def test_without_P_root_comes_first_and_is_detected(self):
        p = run_py("from formal_g2 import guard; import os\n"
                   "try:\n guard.check_search_path(os.getcwd()); print('NOT_DETECTED')\n"
                   "except guard.GuardStop as g: print('DETECTED', g.code)", safe_path=False)
        self.assertIn("DETECTED A-1", p.stdout, p.stderr)

    def test_token_present_stops(self):
        env = dict(ENV, GITHUB_TOKEN="dummy")
        p = run_py("from formal_g2 import guard\n"
                   "try:\n guard.check_token_absent(); print('NOT_STOPPED')\n"
                   "except guard.GuardStop as g: print('STOPPED', g.code)", env=env)
        self.assertIn("STOPPED A-2", p.stdout, p.stderr)


class ConfigTests(unittest.TestCase):
    """ルートの config.py は実行しない。一時ディレクトリの config.py を代わりに使う。"""

    def test_find_spec_recorded_without_import(self):
        p = run_py("import sys, json; from formal_g2 import guard;"
                   "r = guard.config_find_spec_record();"
                   "print(json.dumps({'rec': r, 'loaded': 'config' in sys.modules}))")
        self.assertEqual(p.returncode, 0, p.stderr)
        res = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertIn("origin", res["rec"])
        self.assertFalse(res["loaded"])

    def test_loaded_config_stops_A1(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.py"), "w", encoding="utf-8") as f:
                f.write("X = 1\n")
            p = run_py(f"import sys; sys.path.insert(0, {d!r}); import config\n"
                       "from formal_g2 import guard\n"
                       f"try:\n guard.check_config_absent({d!r}, 'before_import'); print('NOT_STOPPED')\n"
                       "except guard.GuardStop as g: print('STOPPED', g.code)")
            self.assertIn("STOPPED A-1", p.stdout, p.stderr)


class RaceDateTests(unittest.TestCase):
    def test_formal_requires_before_run_date(self):
        from datetime import datetime, timedelta, timezone
        from formal_g2 import guard
        now = datetime(2099, 1, 2, 0, 30, tzinfo=timezone(timedelta(hours=9)))
        self.assertTrue(guard.check_race_date_formal("20990101", now)["before_run_date"])
        with self.assertRaises(guard.GuardStop) as cm:
            guard.check_race_date_formal("20990102", now)
        self.assertEqual(cm.exception.code, "DATE")
        # UTC では前日でも、JST の当日なら停止する
        now_utc = datetime(2099, 1, 1, 16, 0, tzinfo=timezone.utc)  # = 2099-01-02 01:00 JST
        with self.assertRaises(guard.GuardStop):
            guard.check_race_date_formal("20990102", now_utc)


class TestsPackageTests(unittest.TestCase):
    def test_T1_T2_recorded_and_T3_not_loaded(self):
        p = run_py("from formal_g2 import guard; import json, os;"
                   "t = guard.tests_probe(os.getcwd());"
                   "from formal_g2 import race_eval; race_eval.load_comparator(os.getcwd());"
                   "t.update(guard.tests_not_loaded(os.getcwd())); print(json.dumps(t))")
        self.assertEqual(p.returncode, 0, p.stderr)
        res = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertIn("T1_rebuild_tests_init_exists", res)
        self.assertIn("T2_find_spec_origin", res)
        self.assertTrue(res["T3_ok"], res)


class ComparatorDirectLoadTests(unittest.TestCase):
    def test_shadow_init_not_executed(self):
        p = run_py("import os, sys, json; from formal_g2 import race_eval;"
                   "m = race_eval.load_comparator(os.getcwd());"
                   "files = sorted(os.path.relpath(getattr(v,'__file__','') or '', os.getcwd())"
                   " for v in list(sys.modules.values()) if (getattr(v,'__file__','') or '')"
                   ".startswith(os.path.join(os.getcwd(),'rebuild','shadow')));"
                   "print(json.dumps({'shadow': race_eval.shadow_package_loaded(), 'files': files,"
                   " 'has_compare': hasattr(m, 'compare')}))")
        self.assertEqual(p.returncode, 0, p.stderr)
        res = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual(res["shadow"], [])
        self.assertEqual(res["files"], [os.path.join("rebuild", "shadow", "comparator.py")])
        self.assertTrue(res["has_compare"])


if __name__ == "__main__":
    unittest.main()
