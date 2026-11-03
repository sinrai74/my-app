"""正式G2：M7-2・M7-3 の照合（W3）。

実行（リポジトリ直下で）：
  PYTHONPATH="rebuild:." python -P -B -m unittest discover -s tests/formal_g2 -t tests/formal_g2
"""

import os
import tempfile
import unittest

from formal_g2 import env_baseline as eb

LOCK = """# [locked]
requests==2.32.3
Charset_Normalizer==3.4.0
# [allowed-extra]
pip==24.2
"""


class LockTests(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".lock")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(LOCK)
        self.lock = eb.parse_lock(self.path)

    def tearDown(self):
        os.remove(self.path)

    def test_names_are_normalized(self):
        self.assertIn("charset-normalizer", self.lock["locked"])

    def test_exact_match_ok(self):
        inst = {"requests": "2.32.3", "charset-normalizer": "3.4.0", "pip": "24.2"}
        self.assertTrue(eb.check_lock(self.lock, inst)["ok"])

    def test_missing(self):
        r = eb.check_lock(self.lock, {"requests": "2.32.3", "pip": "24.2"})
        self.assertFalse(r["ok"])
        self.assertEqual(r["missing"], ["charset-normalizer"])

    def test_version_mismatch(self):
        r = eb.check_lock(self.lock, {"requests": "2.32.4", "charset-normalizer": "3.4.0", "pip": "24.2"})
        self.assertFalse(r["ok"])
        self.assertEqual(r["mismatched"][0]["name"], "requests")

    def test_extra_package(self):
        inst = {"requests": "2.32.3", "charset-normalizer": "3.4.0", "pip": "24.2", "scikit-learn": "1.5.2"}
        r = eb.check_lock(self.lock, inst)
        self.assertFalse(r["ok"])
        self.assertEqual(r["extra"][0]["name"], "scikit-learn")

    def test_list_unavailable(self):
        self.assertFalse(eb.check_lock(self.lock, None)["ok"])

    def test_unpinned_line_rejected(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("# [locked]\nrequests>=2\n")
        with self.assertRaises(ValueError):
            eb.parse_lock(self.path)


class SnapshotTests(unittest.TestCase):
    def _snap(self, image_os="ubuntu24", image_version="20261101.1"):
        return {
            "image": {"ImageOS": image_os, "ImageVersion": image_version,
                      "identified": bool(image_os) and bool(image_version)},
            "lock": {"sha256": "h", "check": {"ok": True}},
            "settings": {"cfg_has_venue_cfg_key": False},
            "code": {"a.py": "x"},
            "motor_history_header_sha256": "h",
            "python": {"version_info": [3, 11, 9, "final", 0]},
            "installed": {"pip": "24.2"},
        }

    def test_env_image_missing_alone_is_not_a_problem(self):
        # M7-3（2026-11-03）：環境変数は観測値。Runner Image（ジョブログ由来）は集約で必須
        snap = self._snap(image_version=None)
        snap["os"] = {"available": True}
        self.assertEqual(eb.self_check(snap), [])

    def test_baseline_diff_detected(self):
        snap = self._snap()
        base = eb.baseline_view(snap)
        # M7-3：runner_image（ジョブログ由来）は登録時に加える
        base["runner_image"] = {"image": "ubuntu-24.04", "version": "20261025.353.1"}
        self.assertEqual(eb.compare_to_baseline(snap, base), [])
        base["code"] = {"a.py": "y"}
        self.assertEqual(eb.compare_to_baseline(snap, base), ["code:a.py"])

    def test_module_allowlist(self):
        loaded = [{"module": "m", "path": "a.py"}, {"module": "odds_fetch", "path": "odds_fetch.py"}]
        r = eb.check_modules(loaded, ["a.py"])
        self.assertFalse(r["ok"])
        self.assertEqual(r["not_allowed"][0]["path"], "odds_fetch.py")


class OsInfoTests(unittest.TestCase):
    """Q5.2：OS 情報は os.uname() から取り、subprocess を起動しない。"""

    def test_os_info_matches_uname(self):
        info = eb.os_info()
        u = os.uname()
        self.assertTrue(info["available"])
        self.assertEqual((info["system"], info["release"], info["version"], info["machine"]),
                         (u.sysname, u.release, u.version, u.machine))
        self.assertEqual(eb.platform_string(info), f"{u.sysname}-{u.release}-{u.machine}")

    def test_os_info_unavailable_is_not_eligible(self):
        from unittest import mock
        with mock.patch.object(eb.os, "uname", side_effect=OSError("no uname")):
            info = eb.os_info()
        self.assertFalse(info["available"])
        self.assertIsNone(eb.platform_string(info))
        snap = {"image": {"identified": True}, "lock": {"sha256": "h", "check": {"ok": True}},
                "settings": {"cfg_has_venue_cfg_key": False}, "code": {"a.py": "x"},
                "motor_history_header_sha256": "h", "os": info}
        self.assertIn("OS information unavailable (os.uname)", eb.self_check(snap))

    def test_collect_snapshot_starts_no_subprocess_and_no_extra_file_ops(self):
        import json
        import subprocess
        import sys
        root = os.path.realpath(os.getcwd())
        code = (
            "import sys, os, json\n"
            "ev = []\n"
            "def h(e, a):\n"
            "    if e in ('subprocess.Popen', 'os.system', 'os.posix_spawn') or e.startswith(('os.exec', 'os.spawn')):\n"
            "        ev.append(['process', e])\n"
            "    elif e == 'open':\n"
            "        p, mode, flags = str(a[0]), a[1], a[2] if len(a) > 2 else 0\n"
            "        w = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC)))\n"
            "        if w or p == os.devnull:\n"
            "            ev.append(['open', p])\n"
            "    elif e in ('os.remove', 'os.rename', 'os.replace', 'os.mkdir', 'os.rmdir'):\n"
            "        ev.append(['fs', e])\n"
            "from formal_g2 import env_baseline as eb\n"
            "sys.addaudithook(h)\n"
            "snap = eb.collect_snapshot(os.getcwd(), {})\n"
            "print(json.dumps({'events': ev, 'os': snap['os'], 'platform': snap['platform']}))\n"
        )
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([os.path.join(root, "rebuild"), root]))
        env.pop("GITHUB_TOKEN", None)
        p = subprocess.run([sys.executable, "-B", "-P", "-c", code], cwd=root, env=env,
                           capture_output=True, text=True, timeout=300)
        self.assertEqual(p.returncode, 0, p.stderr)
        res = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual(res["events"], [])
        self.assertTrue(res["os"]["available"])
        self.assertTrue(res["platform"])


if __name__ == "__main__":
    unittest.main()
