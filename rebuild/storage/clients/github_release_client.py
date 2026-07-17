"""
GithubReleaseClient: GitHub Releases への実クライアント（ReleaseClient実装）。

設計書 v1.1.8 ⑥・⑫、Step3計画書 §5（S1: urllib標準ライブラリのみ／
S3: 命名規約／S6: 新タグdata-store-v2）に基づく。

責務: HTTP通信のみ。リトライ判断はRetryPolicy（retry.py）へ分離（Should-B）。
- opener（urllib.request.OpenerDirector互換）とsleeper（time.sleep互換）は
  コンストラクタ注入。単体テストはFakeOpener/FakeSleeperのみで完結し、
  実GitHub通信は行わない
- トークンは引数注入（環境変数参照はpipelines層の責務）
- 失敗はすべてStorageErrorへ正規化する（リトライ枯渇・非リトライ対象・
  ネットワーク異常・入力不正）
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from storage.clients.retry import DEFAULT_RETRY_POLICY, RetryPolicy
from storage.exceptions import StorageError

_API_BASE = "https://api.github.com"
_UPLOADS_BASE = "https://uploads.github.com"


class GithubReleaseClient:
    """GitHub Releases（指定タグ）のアセットを操作するReleaseClient実装。"""

    def __init__(
        self,
        owner: str,
        repo: str,
        tag: str,
        token: str,
        opener: Optional[Any] = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._owner = owner
        self._repo = repo
        self._tag = tag
        self._token = token
        self._opener = opener if opener is not None else urllib.request.build_opener()
        self._retry = retry_policy
        self._sleep = sleeper

    # ==================== ReleaseClient実装 ====================

    def upload_asset(self, file_path: Path, asset_name: str) -> None:
        """ファイルを指定アセット名でアップロードする。同名の既存アセットは置換。"""
        path = Path(file_path)
        try:
            body = path.read_bytes()
        except OSError as exc:
            raise StorageError(f"cannot read upload source for asset {asset_name!r}: {path}: {exc}") from exc
        release = self._get_release()
        existing_id = _find_asset_id(release, asset_name)
        if existing_id is not None:
            self._delete_asset_by_id(existing_id)
        url = (
            f"{_UPLOADS_BASE}/repos/{self._owner}/{self._repo}/releases/"
            f"{release['id']}/assets?name={asset_name}"
        )
        self._request(
            "POST",
            url,
            body=body,
            extra_headers={"Content-Type": "application/octet-stream"},
            context=f" [asset={asset_name}, release_id={release['id']}]",
        )

    def download_asset(self, asset_name: str, dest_path: Path) -> None:
        """アセットをローカルへ取得する。存在しなければStorageError。"""
        release = self._get_release()
        asset_id = _find_asset_id(release, asset_name)
        if asset_id is None:
            raise StorageError(f"asset not found on release (tag={self._tag}): {asset_name}")
        url = f"{_API_BASE}/repos/{self._owner}/{self._repo}/releases/assets/{asset_id}"
        data = self._request(
            "GET", url, extra_headers={"Accept": "application/octet-stream"}
        )
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def list_assets(self) -> list[str]:
        release = self._get_release()
        return sorted(asset["name"] for asset in release.get("assets", []))

    def delete_asset(self, asset_name: str) -> None:
        release = self._get_release()
        asset_id = _find_asset_id(release, asset_name)
        if asset_id is None:
            raise StorageError(f"asset not found on release (tag={self._tag}): {asset_name}")
        self._delete_asset_by_id(asset_id)

    # ==================== 内部 ====================

    def _get_release(self) -> dict[str, Any]:
        url = f"{_API_BASE}/repos/{self._owner}/{self._repo}/releases/tags/{self._tag}"
        data = self._request("GET", url, context=f" [tag={self._tag}]")
        try:
            release = json.loads(data)
        except json.JSONDecodeError as exc:
            raise StorageError(f"broken release info JSON: {exc}") from exc
        if "id" not in release:
            raise StorageError("release info has no 'id'")
        return release

    def _delete_asset_by_id(self, asset_id: int) -> None:
        url = f"{_API_BASE}/repos/{self._owner}/{self._repo}/releases/assets/{asset_id}"
        self._request("DELETE", url)

    def _request(
        self,
        method: str,
        url: str,
        body: Optional[bytes] = None,
        extra_headers: Optional[dict[str, str]] = None,
        context: str = "",
    ) -> bytes:
        """1リクエストをRetryPolicyに従って実行する。

        リトライ判断（対象ステータス・待機秒・上限）はRetryPolicyへ完全委譲し、
        本メソッドはHTTP実行と例外の正規化のみを行う（Should-B責務分離）。
        """
        headers = {
            "Authorization": f"Bearer {self._token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "asakan-ai-storage",
        }
        if extra_headers:
            headers.update(extra_headers)
        last_error = ""
        for attempt in range(1, self._retry.max_attempts + 1):
            if attempt >= 2:
                self._sleep(self._retry.delay_before_attempt(attempt))
            request = urllib.request.Request(
                url, data=body, headers=headers, method=method
            )
            try:
                with self._opener.open(
                    request, timeout=self._retry.timeout_seconds
                ) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                status = exc.code
                last_error = f"HTTP {status} for {method} {url}{context}"
                if not RetryPolicy.is_retryable_status(status):
                    raise StorageError(f"non-retryable: {last_error}") from exc
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = f"network error for {method} {url}{context}: {exc}"
                if not RetryPolicy.is_retryable_network_error():
                    raise StorageError(last_error) from exc
        raise StorageError(
            f"retry exhausted after {self._retry.max_attempts} attempts: {last_error}"
        )


def _find_asset_id(release: dict[str, Any], asset_name: str) -> Optional[int]:
    for asset in release.get("assets", []):
        if asset.get("name") == asset_name:
            return asset.get("id")
    return None
