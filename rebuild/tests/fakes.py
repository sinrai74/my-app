"""
テスト用Fake実装（ReleaseClient / GitClient）。

Step3計画書 §5（モック戦略）に基づく。本番コードからはimportしない。
呼び出し順序の検証と失敗注入（トランザクションの失敗系テスト）を提供する。
"""

from __future__ import annotations

from pathlib import Path

from storage.exceptions import StorageError


class FakeReleaseClient:
    """メモリ上のdictでReleasesアセットを模倣する。呼び出し履歴を記録する。"""

    def __init__(self, fail_on_upload: bool = False) -> None:
        self.assets: dict[str, bytes] = {}
        self.calls: list[tuple[str, str]] = []  # (操作, アセット名)
        self._fail_on_upload = fail_on_upload

    def upload_asset(self, file_path: Path, asset_name: str) -> None:
        self.calls.append(("upload", asset_name))
        if self._fail_on_upload:
            raise StorageError(f"injected upload failure: {asset_name}")
        self.assets[asset_name] = Path(file_path).read_bytes()

    def download_asset(self, asset_name: str, dest_path: Path) -> None:
        self.calls.append(("download", asset_name))
        if asset_name not in self.assets:
            raise StorageError(f"asset not found: {asset_name}")
        Path(dest_path).write_bytes(self.assets[asset_name])

    def list_assets(self) -> list[str]:
        self.calls.append(("list", ""))
        return sorted(self.assets.keys())

    def delete_asset(self, asset_name: str) -> None:
        self.calls.append(("delete", asset_name))
        if asset_name not in self.assets:
            raise StorageError(f"asset not found: {asset_name}")
        del self.assets[asset_name]


class FakeGitClient:
    """commit_and_pushの呼び出しを記録するだけのFake。失敗注入可能。"""

    def __init__(self, fail_on_commit: bool = False) -> None:
        self.commits: list[tuple[tuple[str, ...], str]] = []  # (paths, message)
        self.call_attempts: int = 0  # 失敗時も含む呼び出し試行回数（Should-A: 未実行の明示検証用）
        self._fail_on_commit = fail_on_commit

    def commit_and_push(self, paths: list[Path], message: str) -> None:
        self.call_attempts += 1
        if self._fail_on_commit:
            raise StorageError("injected commit failure")
        self.commits.append((tuple(str(p) for p in paths), message))
