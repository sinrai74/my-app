"""
外部クライアントのProtocol定義（設計書 v1.1.7 ⑩ clients/）。

Step3計画書 §5（GitHub API依存部分とモック戦略）に基づく。
実装（GithubReleaseClient / SubprocessGitClient）とは分離し、
DurableStore等の利用側はこのProtocolのみに依存する（DI）。
テストではFake実装（tests/fakes.py）を注入する。
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class ReleaseClient(Protocol):
    """GitHub Releases（data-store-v2タグ）へのアセット入出力。

    命名規約（Step3計画書S3確定）:
    - 最新版: {name}.{ext}（上書き）
    - 日次スナップショット: {name}_{YYYYMMDD}.{ext}
    現行 data-store タグへの書込は禁止（S6・Freeze比較基準の汚染防止）。
    """

    def upload_asset(self, file_path: Path, asset_name: str) -> None:
        """ファイルを指定アセット名でアップロードする（同名は置換）。失敗はStorageError。"""
        ...

    def download_asset(self, asset_name: str, dest_path: Path) -> None:
        """アセットをローカルへ取得する。存在しない・失敗はStorageError。"""
        ...

    def list_assets(self) -> list[str]:
        """アセット名の一覧を返す。失敗はStorageError。"""
        ...

    def delete_asset(self, asset_name: str) -> None:
        """アセットを削除する（保持世代の整理用）。失敗はStorageError。"""
        ...


class GitClient(Protocol):
    """git add / commit / push の実行。"""

    def commit_and_push(self, paths: list[Path], message: str) -> None:
        """指定パスをadd・commit・pushする。

        push競合時は pull --rebase -> 再push を1回だけ試行し、
        それでも失敗すればStorageError（Step3計画書リスク3）。
        コミット対象の変更が無い場合は何もしない（冪等）。
        """
        ...
