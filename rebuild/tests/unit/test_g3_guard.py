"""段階4-2：g3_guard（排他・プロファイラ入口アサート）の回帰テスト。"""
import sys
import unittest

from shadow.g3_guard import G3ProfilerPresent, assert_no_profiler, profiler_state


class GuardTests(unittest.TestCase):
    def test_no_profiler_ok(self):
        # 既存プロファイラが無い前提（テスト実行中に設定していない）
        if sys.getprofile() is not None:
            self.skipTest("a profiler is already installed in this environment")
        st = assert_no_profiler()
        self.assertFalse(st["sys"])

    def test_profiler_present_raises(self):
        prev = sys.getprofile()
        try:
            sys.setprofile(lambda *a: None)
            self.assertTrue(profiler_state()["sys"])
            with self.assertRaises(G3ProfilerPresent):
                assert_no_profiler()
        finally:
            sys.setprofile(prev)


if __name__ == "__main__":
    unittest.main()
