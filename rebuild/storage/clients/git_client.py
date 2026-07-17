"""
SubprocessGitClient: subprocess による git add / commit / push（GitClient実装）。

設計書 v1.1.8 ⑥（commit不可分化）、Step3計画書 §5（S2: git操作をPython側へ集約）・
リスク3（push競合時は pull --rebase -> 再push を1回だけ）に基づく。

責務: gitコマンドの実行のみ。runner（subprocess.run互換）をコンストラクタ注入し、
単体テストはFakeProcessで完結する（実git・ネットワーク不要）。結合テストは
tempdir内の実gitリポジトリ（bare remote + clone）で実施する（ネットワーク不要）。

失敗はStorageErrorへ正規化する。コミット対象の変更が無い場合は何もしない（冪等）。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from storage.exceptions import StorageError

# gitコマンドのタイムアウト（秒）。ネットワークを伴うpush/pullも含め十分な既定値。
_GIT_TIMEOUT_SECONDS: float = 60.0

# subprocess.run 互換の呼び出しシグネチャ（テストで差し替え可能にする）
Runner = Callable[..., "subprocess.CompletedProcess[str]"]


class SubprocessGitClient:
    """指定リポジトリでgit操作を行うGitClient実装。"""

    def __init__(
        self,
        repo_dir: Path,
        runner: Optional[Runner] = None,
        remote: str = "origin",
        branch: str = "main",
    ) -> None:
        self._repo_dir = Path(repo_dir)
        self._run = runner if runner is not None else self._default_runner
        self._remote = remote
        self._branch = branch

    @staticmethod
    def _default_runner(*args: Any, **kwargs: Any) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(*args, **kwargs)

    def _git(self, *cmd: str) -> "subprocess.CompletedProcess[str]":
        """gitサブコマンドを実行し、CompletedProcessを返す（チェックは呼び出し側）。"""
        try:
            return self._run(
                ["git", *cmd],
                cwd=str(self._repo_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",  # Windowsの既定cp932でUTF-8出力を復号すると
                errors="replace",  # 失敗するため明示（Actions/Linuxでは無影響）
                timeout=_GIT_TIMEOUT_SECONDS,
            )
        except FileNotFoundError as exc:
            # Should2: gitバイナリが無い場合はStorageErrorへ正規化
            raise StorageError(
                "git executable not found (install git or check PATH)"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise StorageError(
                f"git {' '.join(cmd)} timed out after {_GIT_TIMEOUT_SECONDS}s"
            ) from exc

    def _git_checked(self, *cmd: str) -> str:
        """gitサブコマンドを実行し、非0終了はStorageErrorへ正規化する。"""
        result = self._git(*cmd)
        if result.returncode != 0:
            raise StorageError(
                f"git {' '.join(cmd)} failed (exit {result.returncode}): "
                f"{(result.stderr or '').strip()}"
            )
        return result.stdout or ""

    def commit_and_push(self, paths: list[Path], message: str) -> None:
        """指定パスを add・commit・push する。

        - 変更が無い場合は commit せず push もしない（冪等）
        - push が競合（remote先行）で失敗した場合、pull --rebase -> push を
          1回だけ再試行する。それでも失敗すれば StorageError（リスク3）
        """
        for path in paths:
            self._git_checked("add", "--", str(path))

        if not self._has_staged_changes():
            return  # コミット対象なし（冪等）

        self._git_checked("commit", "-m", message)

        push = self._git("push", self._remote, self._branch)
        if push.returncode == 0:
            return

        # 競合の可能性 -> pull --rebase して1回だけ再push
        rebase = self._git("pull", "--rebase", self._remote, self._branch)
        if rebase.returncode != 0:
            raise StorageError(
                f"git pull --rebase failed after push conflict (exit "
                f"{rebase.returncode}): {(rebase.stderr or '').strip()}"
            )
        push_retry = self._git("push", self._remote, self._branch)
        if push_retry.returncode != 0:
            raise StorageError(
                f"git push failed after rebase retry "
                f"[attempts=2, rebase_performed=True, remote={self._remote}, "
                f"branch={self._branch}] (exit {push_retry.returncode}): "
                f"{(push_retry.stderr or '').strip()}"
            )

    def _has_staged_changes(self) -> bool:
        """ステージ済みの変更があるか（git diff --cached --quiet の終了コードで判定）。

        --quiet は差分ありで終了コード1、差分なしで0を返す。
        """
        result = self._git("diff", "--cached", "--quiet")
        return result.returncode == 1
