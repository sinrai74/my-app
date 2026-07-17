"""
SubprocessGitClient の単体テスト（FakeProcess）＋結合テスト（実git・tempdir）。

Step3計画書 §5・§6、Step3-3指示に対応。
- 単体: FakeRunnerでコマンド列・分岐・失敗系を検証（git実行なし）
- 結合: tempdir内にbare remote + cloneを作り、実gitバイナリで検証（ネットワーク不要）
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from storage.clients.git_client import SubprocessGitClient
from storage.exceptions import StorageError


class _FakeResult:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeRunner:
    """gitコマンドの呼び出しを記録し、スクリプト化した結果を返すrunner。"""

    def __init__(self, results: dict[str, _FakeResult] | None = None) -> None:
        self.calls: list[list[str]] = []
        self._results = results or {}
        self._sequence: dict[str, list[_FakeResult]] = {}

    def set_sequence(self, key: str, results: list[_FakeResult]) -> None:
        self._sequence[key] = list(results)

    def __call__(self, cmd, cwd=None, **kwargs):  # noqa: ANN001  # encoding/errors/timeout等は検証対象外のため吸収
        self.calls.append(cmd)
        key = " ".join(cmd[1:3]) if len(cmd) >= 3 else cmd[1] if len(cmd) > 1 else ""
        full = " ".join(cmd[1:])
        # 完全一致 > 前方2語 の順で解決
        for k in (full, key):
            if k in self._sequence and self._sequence[k]:
                return self._sequence[k].pop(0)
            if k in self._results:
                return self._results[k]
        return _FakeResult(0)


def _cmd_list(runner: FakeRunner) -> list[str]:
    return [" ".join(c[1:]) for c in runner.calls]


class TestSubprocessGitClientUnit(unittest.TestCase):
    def test_add_commit_push_success(self) -> None:
        """変更あり・push成功の基本フロー。"""
        runner = FakeRunner(
            {"diff --cached": _FakeResult(1)}  # 差分あり
        )
        client = SubprocessGitClient(Path("/repo"), runner=runner)
        client.commit_and_push([Path("hit_record.csv")], "feat: 追記")
        cmds = _cmd_list(runner)
        self.assertEqual(cmds[0], "add -- hit_record.csv")
        self.assertIn("commit -m feat: 追記", cmds)
        self.assertIn("push origin main", cmds)

    def test_no_changes_skips_commit_and_push(self) -> None:
        """変更なし（diff --cached が0）ならcommit/pushしない（冪等）。"""
        runner = FakeRunner({"diff --cached": _FakeResult(0)})
        client = SubprocessGitClient(Path("/repo"), runner=runner)
        client.commit_and_push([Path("x.csv")], "msg")
        cmds = _cmd_list(runner)
        self.assertTrue(any(c.startswith("add") for c in cmds))
        self.assertFalse(any(c.startswith("commit") for c in cmds))
        self.assertFalse(any(c.startswith("push") for c in cmds))

    def test_push_conflict_triggers_rebase_then_retry_push(self) -> None:
        """push失敗 -> pull --rebase -> 再push成功。"""
        runner = FakeRunner({"diff --cached": _FakeResult(1)})
        runner.set_sequence(
            "push origin main", [_FakeResult(1, stderr="rejected"), _FakeResult(0)]
        )
        client = SubprocessGitClient(Path("/repo"), runner=runner)
        client.commit_and_push([Path("x.csv")], "msg")
        cmds = _cmd_list(runner)
        self.assertEqual(cmds.count("push origin main"), 2)
        self.assertIn("pull --rebase origin main", cmds)
        # 順序: 最初のpush -> rebase -> 2回目push
        i_push1 = cmds.index("push origin main")
        i_rebase = cmds.index("pull --rebase origin main")
        self.assertLess(i_push1, i_rebase)

    def test_push_retry_only_once(self) -> None:
        """再push も失敗したらStorageError（無限リトライしない）。"""
        runner = FakeRunner({"diff --cached": _FakeResult(1)})
        runner.set_sequence(
            "push origin main",
            [_FakeResult(1, stderr="rejected"), _FakeResult(1, stderr="rejected again")],
        )
        client = SubprocessGitClient(Path("/repo"), runner=runner)
        with self.assertRaises(StorageError) as ctx:
            client.commit_and_push([Path("x.csv")], "msg")
        self.assertIn("after rebase retry", str(ctx.exception))
        self.assertEqual(_cmd_list(runner).count("push origin main"), 2)

    def test_rebase_failure_raises(self) -> None:
        """rebase自体が失敗したらStorageError。"""
        runner = FakeRunner({"diff --cached": _FakeResult(1)})
        runner.set_sequence("push origin main", [_FakeResult(1, stderr="rejected")])
        runner.set_sequence(
            "pull --rebase", [_FakeResult(1, stderr="conflict")]
        )
        client = SubprocessGitClient(Path("/repo"), runner=runner)
        with self.assertRaises(StorageError) as ctx:
            client.commit_and_push([Path("x.csv")], "msg")
        self.assertIn("pull --rebase failed", str(ctx.exception))

    def test_git_not_found_raises_storage_error(self) -> None:
        """Should2: gitバイナリ不在（FileNotFoundError）はStorageErrorへ。"""
        def _raise(*a, **k):
            raise FileNotFoundError("git")
        client = SubprocessGitClient(Path("/repo"), runner=_raise)
        with self.assertRaises(StorageError) as ctx:
            client.commit_and_push([Path("x.csv")], "msg")
        self.assertIn("git executable not found", str(ctx.exception))

    def test_timeout_raises_storage_error(self) -> None:
        """Should1: タイムアウトはStorageErrorへ正規化される。"""
        import subprocess as _sp
        def _timeout(*a, **k):
            raise _sp.TimeoutExpired(cmd="git", timeout=60.0)
        client = SubprocessGitClient(Path("/repo"), runner=_timeout)
        with self.assertRaises(StorageError) as ctx:
            client.commit_and_push([Path("x.csv")], "msg")
        self.assertIn("timed out", str(ctx.exception))

    def test_push_conflict_error_has_diagnostics(self) -> None:
        """Should3: 再push失敗のStorageErrorに診断情報が含まれること。"""
        runner = FakeRunner({"diff --cached": _FakeResult(1)})
        runner.set_sequence(
            "push origin main",
            [_FakeResult(1, stderr="rejected"), _FakeResult(1, stderr="again")],
        )
        client = SubprocessGitClient(Path("/repo"), runner=runner)
        with self.assertRaises(StorageError) as ctx:
            client.commit_and_push([Path("x.csv")], "msg")
        msg = str(ctx.exception)
        self.assertIn("rebase_performed=True", msg)
        self.assertIn("attempts=2", msg)

    def test_add_failure_raises(self) -> None:
        runner = FakeRunner()
        runner.set_sequence("add --", [_FakeResult(128, stderr="pathspec error")])
        client = SubprocessGitClient(Path("/repo"), runner=runner)
        with self.assertRaises(StorageError) as ctx:
            client.commit_and_push([Path("x.csv")], "msg")
        self.assertIn("git add", str(ctx.exception))


@unittest.skipIf(shutil.which("git") is None, "git binary not available")
class TestSubprocessGitClientIntegration(unittest.TestCase):
    """実gitバイナリ・tempdir・ローカルbare remoteでの結合テスト（ネットワーク不要）。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)
        self.remote = root / "remote.git"
        self.work = root / "work"
        subprocess.run(
            ["git", "init", "--bare", "-b", "main", str(self.remote)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "clone", str(self.remote), str(self.work)],
            check=True, capture_output=True,
        )
        self._config(self.work)
        # 初期コミットを作ってmainを確立
        (self.work / "seed.txt").write_text("seed", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=self.work, check=True, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "seed"], cwd=self.work, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "push", "origin", "main"], cwd=self.work, check=True, capture_output=True
        )

    @staticmethod
    def _config(repo: Path) -> None:
        subprocess.run(["git", "config", "user.email", "t@example.com"], cwd=repo, check=True)
        subprocess.run(["git", "config", "user.name", "tester"], cwd=repo, check=True)

    def test_commit_and_push_reflects_on_remote(self) -> None:
        client = SubprocessGitClient(self.work, remote="origin", branch="main")
        (self.work / "hit_record.csv").write_text("date\n20260714\n", encoding="utf-8")
        client.commit_and_push([Path("hit_record.csv")], "feat: 実績追記")
        # 別cloneでremoteに反映されていることを確認
        verify = Path(self._tmp.name) / "verify"
        subprocess.run(
            ["git", "clone", str(self.remote), str(verify)], check=True, capture_output=True
        )
        self.assertTrue((verify / "hit_record.csv").exists())

    def test_no_changes_is_idempotent(self) -> None:
        client = SubprocessGitClient(self.work, remote="origin", branch="main")
        # ファイルを作らずcommit_and_push -> 例外なく何もしない
        (self.work / "unchanged.csv").write_text("x", encoding="utf-8")
        client.commit_and_push([Path("unchanged.csv")], "first")
        # 2回目は変更なし -> 冪等（例外が出ない）
        client.commit_and_push([Path("unchanged.csv")], "second-noop")

    def test_push_conflict_recovered_by_rebase(self) -> None:
        """remoteが先行した状態からpush競合→rebase→再pushで回復すること。"""
        # 別cloneからremoteを先行させる
        other = Path(self._tmp.name) / "other"
        subprocess.run(
            ["git", "clone", str(self.remote), str(other)], check=True, capture_output=True
        )
        self._config(other)
        (other / "other.csv").write_text("o", encoding="utf-8")
        subprocess.run(["git", "add", "."], cwd=other, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "other"], cwd=other, check=True, capture_output=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=other, check=True, capture_output=True)

        # workは古いまま新規コミットをpushしようとする -> 競合 -> rebaseで回復
        client = SubprocessGitClient(self.work, remote="origin", branch="main")
        (self.work / "mine.csv").write_text("m", encoding="utf-8")
        client.commit_and_push([Path("mine.csv")], "feat: mine")

        verify = Path(self._tmp.name) / "verify2"
        subprocess.run(
            ["git", "clone", str(self.remote), str(verify)], check=True, capture_output=True
        )
        self.assertTrue((verify / "mine.csv").exists())
        self.assertTrue((verify / "other.csv").exists())  # 双方が残る


if __name__ == "__main__":
    unittest.main()
