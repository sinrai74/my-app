"""
GithubReleaseClient / RetryPolicy の単体テスト。

Step3-2指示のテスト行列①〜⑥に対応。FakeOpener/FakeSleeperのみで完結し、
実GitHubへの通信は一切行わない（⑥の要件）。
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from storage.clients.github_release_client import GithubReleaseClient
from storage.clients.retry import RetryPolicy
from storage.exceptions import StorageError


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.github.com/x", code=code, msg="err", hdrs=None, fp=io.BytesIO(b"")
    )


class FakeOpener:
    """スクリプト（応答 or 例外の列）どおりに応答するopener。リクエストを記録する。"""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.requests: list = []

    def open(self, request, timeout=None):  # noqa: ANN001
        self.requests.append(request)
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


class FakeSleeper:
    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def _release_json(assets: list[dict] | None = None) -> _FakeResponse:
    return _FakeResponse(
        json.dumps({"id": 777, "assets": assets or []}).encode("utf-8")
    )


def _client(script: list, sleeper: FakeSleeper | None = None) -> tuple[GithubReleaseClient, FakeOpener, FakeSleeper]:
    opener = FakeOpener(script)
    slp = sleeper or FakeSleeper()
    client = GithubReleaseClient(
        owner="sinrai74",
        repo="my-app",
        tag="data-store-v2",
        token="test-token",
        opener=opener,
        retry_policy=RetryPolicy(max_attempts=3, base_delay_seconds=2.0),
        sleeper=slp,
    )
    return client, opener, slp


class TestRetryPolicy(unittest.TestCase):
    def test_defaults_match_design_spec(self) -> None:
        """既定値が設計書⑫の確定値（30秒・3回・指数）であること。"""
        policy = RetryPolicy()
        self.assertEqual(policy.max_attempts, 3)
        self.assertEqual(policy.timeout_seconds, 30.0)
        self.assertEqual(policy.delay_before_attempt(2), 2.0)
        self.assertEqual(policy.delay_before_attempt(3), 4.0)

    def test_retryable_statuses(self) -> None:
        self.assertTrue(RetryPolicy.is_retryable_status(429))
        self.assertTrue(RetryPolicy.is_retryable_status(500))
        self.assertTrue(RetryPolicy.is_retryable_status(503))
        for code in (400, 401, 403, 404, 422):
            self.assertFalse(RetryPolicy.is_retryable_status(code), code)


class TestUploadNormal(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.src = Path(self._tmp.name) / "hit_record.csv"
        self.src.write_bytes(b"date,venue\n20260714,suminoe\n")

    def test_upload_latest_asset(self) -> None:
        """テスト①: 最新版アップロード（release取得 -> POST）。"""
        client, opener, _ = _client([_release_json(), _FakeResponse(b"{}")])
        client.upload_asset(self.src, "hit_record.csv")
        self.assertEqual(len(opener.requests), 2)
        post = opener.requests[1]
        self.assertEqual(post.get_method(), "POST")
        self.assertIn("releases/777/assets?name=hit_record.csv", post.full_url)
        self.assertEqual(post.data, self.src.read_bytes())
        self.assertEqual(post.get_header("Content-type"), "application/octet-stream")

    def test_upload_snapshot_asset_name(self) -> None:
        """テスト①: 日次スナップショット名（S3命名規約）でのアップロード。"""
        client, opener, _ = _client([_release_json(), _FakeResponse(b"{}")])
        client.upload_asset(self.src, "hit_record_20260714.csv")
        self.assertIn("name=hit_record_20260714.csv", opener.requests[1].full_url)

    def test_upload_replaces_existing_asset(self) -> None:
        """同名アセットが既存の場合、DELETE -> POST の順で置換されること。"""
        existing = [{"id": 555, "name": "hit_record.csv"}]
        client, opener, _ = _client(
            [_release_json(existing), _FakeResponse(b""), _FakeResponse(b"{}")]
        )
        client.upload_asset(self.src, "hit_record.csv")
        methods = [r.get_method() for r in opener.requests]
        self.assertEqual(methods, ["GET", "DELETE", "POST"])
        self.assertIn("assets/555", opener.requests[1].full_url)

    def test_auth_header_present(self) -> None:
        client, opener, _ = _client([_release_json()])
        client.list_assets()
        self.assertEqual(
            opener.requests[0].get_header("Authorization"), "Bearer test-token"
        )


class TestRetryBehavior(unittest.TestCase):
    def test_temporary_error_then_success(self) -> None:
        """テスト②: 一時エラー（500）でリトライし、2回目で成功すること。"""
        client, opener, sleeper = _client([_http_error(500), _release_json()])
        assets = client.list_assets()
        self.assertEqual(assets, [])
        self.assertEqual(len(opener.requests), 2)
        self.assertEqual(sleeper.delays, [2.0])  # 指数バックオフ初回

    def test_rate_limit_429_is_retried(self) -> None:
        client, opener, sleeper = _client([_http_error(429), _release_json()])
        client.list_assets()
        self.assertEqual(sleeper.delays, [2.0])

    def test_retry_exhausted_raises_storage_error(self) -> None:
        """テスト②: リトライ上限（3回）到達でStorageError。待機は2回（2秒・4秒）。"""
        client, opener, sleeper = _client(
            [_http_error(500), _http_error(500), _http_error(500)]
        )
        with self.assertRaises(StorageError) as ctx:
            client.list_assets()
        self.assertIn("retry exhausted", str(ctx.exception))
        self.assertEqual(len(opener.requests), 3)
        self.assertEqual(sleeper.delays, [2.0, 4.0])


class TestNonRetryable(unittest.TestCase):
    def test_auth_error_401_fails_immediately(self) -> None:
        """テスト③: 認証エラーはリトライせず即StorageError。"""
        client, opener, sleeper = _client([_http_error(401)])
        with self.assertRaises(StorageError) as ctx:
            client.list_assets()
        self.assertIn("non-retryable", str(ctx.exception))
        self.assertEqual(len(opener.requests), 1)
        self.assertEqual(sleeper.delays, [])

    def test_not_found_404_fails_immediately(self) -> None:
        client, opener, sleeper = _client([_http_error(404)])
        with self.assertRaises(StorageError):
            client.list_assets()
        self.assertEqual(len(opener.requests), 1)
        self.assertEqual(sleeper.delays, [])

    def test_invalid_input_missing_source_file(self) -> None:
        """テスト③: 入力不正（アップロード元ファイル不存在）はHTTPに至らず失敗。"""
        client, opener, _ = _client([])
        with self.assertRaises(StorageError):
            client.upload_asset(Path("/nonexistent/nothing.csv"), "x.csv")
        self.assertEqual(opener.requests, [])

    def test_download_unknown_asset_raises(self) -> None:
        client, _, _ = _client([_release_json()])
        with self.assertRaises(StorageError) as ctx:
            client.download_asset("missing.csv", Path("/tmp/x.csv"))
        self.assertIn("asset not found", str(ctx.exception))


class TestNetworkErrors(unittest.TestCase):
    def test_timeout_is_retried_then_exhausted(self) -> None:
        """テスト④: Timeoutはリトライ対象。枯渇でStorageError。"""
        client, opener, sleeper = _client(
            [TimeoutError("t"), TimeoutError("t"), TimeoutError("t")]
        )
        with self.assertRaises(StorageError):
            client.list_assets()
        self.assertEqual(len(opener.requests), 3)

    def test_url_error_then_success(self) -> None:
        """テスト④: URLErrorはリトライ対象で、回復すれば成功する。"""
        client, opener, _ = _client(
            [urllib.error.URLError("dns"), _release_json()]
        )
        self.assertEqual(client.list_assets(), [])


class TestDownloadAndDelete(unittest.TestCase):
    def test_download_writes_bytes(self) -> None:
        assets = [{"id": 900, "name": "hit_record.csv"}]
        client, opener, _ = _client(
            [_release_json(assets), _FakeResponse(b"CSVDATA")]
        )
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "sub" / "hit_record.csv"
            client.download_asset("hit_record.csv", dest)
            self.assertEqual(dest.read_bytes(), b"CSVDATA")
        self.assertEqual(
            opener.requests[1].get_header("Accept"), "application/octet-stream"
        )

    def test_delete_asset(self) -> None:
        assets = [{"id": 901, "name": "old.csv"}]
        client, opener, _ = _client([_release_json(assets), _FakeResponse(b"")])
        client.delete_asset("old.csv")
        self.assertEqual(opener.requests[1].get_method(), "DELETE")


if __name__ == "__main__":
    unittest.main()
