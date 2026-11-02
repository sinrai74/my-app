"""正式G2：呼び出し・モジュール実行の観測（W7-a〜c の仕組み）。

関数を差し替えずに、別名 import・関数内 import の呼び出しを数えられること、
同じ名前のメソッド（Protocol など）を数えないこと、<module> の実行を検出できることを確かめる。
実行（リポジトリ直下で）：
  PYTHONPATH="rebuild:." python -P -B -m unittest discover -s tests/formal_g2 -t tests/formal_g2
"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

ROOT = os.path.realpath(os.getcwd())
ENV = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(ROOT, "rebuild"), ROOT]))
ENV.pop("GITHUB_TOKEN", None)

TARGET_MOD = textwrap.dedent("""
    class Proto:
        def run(self):
            return 0

    class Real:
        def run(self):
            return 1

    def watched():
        return 1
""")


def _run(script):
    p = subprocess.run([sys.executable, "-B", "-P", "-c", script], cwd=ROOT, env=ENV,
                       capture_output=True, text=True, timeout=120)
    return p


class ProfileCountTests(unittest.TestCase):
    def test_alias_inner_import_and_same_name_methods(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "target_mod.py"), "w", encoding="utf-8") as f:
                f.write(TARGET_MOD)
            script = textwrap.dedent(f"""
                import sys, json
                sys.path.insert(0, {d!r})
                from formal_g2.instrument import Instrument
                inst = Instrument({d!r}, watched={{("target_mod.py", "watched"): "watched",
                                                  ("target_mod.py", "Real.run"): "Real.run"}},
                                  watched_modules={{}})
                inst.start()
                import target_mod
                original = target_mod.watched
                from target_mod import watched as alias
                alias()
                def inner():
                    from target_mod import watched
                    return watched()
                inner()
                target_mod.watched()
                target_mod.Proto().run()
                target_mod.Real().run()
                inst.stop()
                print(json.dumps({{"watched": inst.counts["watched"], "real": inst.counts["Real.run"],
                                  "same": target_mod.watched is original,
                                  "unresolved": inst.snapshot()["unresolved_targets"]}}))
            """)
            p = _run(script)
            self.assertEqual(p.returncode, 0, p.stderr)
            res = json.loads(p.stdout.strip().splitlines()[-1])
            self.assertEqual(res["watched"], 3)
            self.assertEqual(res["real"], 1)  # Proto.run は数えない
            self.assertTrue(res["same"])
            self.assertEqual(res["unresolved"], [])

    def test_module_code_execution_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "config.py"), "w", encoding="utf-8") as f:
                f.write("X = 1\n")
            script = textwrap.dedent(f"""
                import sys, json
                sys.path.insert(0, {d!r})
                from formal_g2.instrument import Instrument
                inst = Instrument({d!r}, watched={{}}, watched_modules={{"config.py": "config"}})
                inst.start()
                before = inst.config_executed
                import config
                inst.stop()
                print(json.dumps({{"before": before, "after": inst.config_executed,
                                  "n": len(inst.detections)}}))
            """)
            p = _run(script)
            self.assertEqual(p.returncode, 0, p.stderr)
            res = json.loads(p.stdout.strip().splitlines()[-1])
            self.assertFalse(res["before"])
            self.assertTrue(res["after"])
            self.assertEqual(res["n"], 1)

    def test_default_targets_resolve_with_firstlineno(self):
        from formal_g2.instrument import Instrument
        inst = Instrument(ROOT)
        inst._resolve_targets()
        self.assertEqual(sorted(k for k, v in inst.resolution.items() if not v["resolved"]), [])
        # engine.py の Protocol（EvaluationEngine.evaluate）ではなく Ver4Engine.evaluate を指すこと
        self.assertEqual(inst.resolution["Ver4Engine.evaluate"]["qualname"], "Ver4Engine.evaluate")

    def test_snapshot_marks_limitations_unverified(self):
        from formal_g2.instrument import Instrument
        snap = Instrument(ROOT).snapshot()
        self.assertEqual(snap["limitations"]["c_extension_direct_calls"], "unverified")


if __name__ == "__main__":
    unittest.main()
