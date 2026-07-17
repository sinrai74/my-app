"""
RetryPolicy: リトライ方針（回数・待機時間・対象判定）の独立定義。

設計書 v1.1.8 ⑫（リトライ・タイムアウト確定値: GitHub API = タイムアウト30秒・
3回・指数バックオフ）、Step3-1レビューShould-B（HTTP処理との責務分離）に基づく。

GithubReleaseClient本体はHTTP通信のみを担当し、
「何回・何秒待って・どの失敗なら再試行するか」の判断はすべて本クラスが持つ。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """リトライ方針。既定値は設計書⑫のGitHub API確定値。"""

    max_attempts: int = 3  # 総試行回数（初回を含む）
    base_delay_seconds: float = 2.0  # 指数バックオフの初期待機（2, 4, 8...秒）
    timeout_seconds: float = 30.0  # 1リクエストのタイムアウト

    def delay_before_attempt(self, next_attempt: int) -> float:
        """次の試行（2回目以降、1始まり）の前に待機する秒数。

        next_attempt=2 -> base, 3 -> base*2, 4 -> base*4 ...（指数バックオフ）
        """
        if next_attempt < 2:
            raise ValueError("delay is defined only before retry attempts (>=2)")
        return self.base_delay_seconds * (2 ** (next_attempt - 2))

    @staticmethod
    def is_retryable_status(status_code: int) -> bool:
        """HTTPステータスがリトライ対象か。

        - 429（レート制限）・5xx（サーバ側の一時異常）: リトライする
        - 4xx（400/401/403/404/422等）: 呼び出し側の問題であり
          リトライしても解決しないため即失敗させる
        """
        return status_code == 429 or 500 <= status_code <= 599

    @staticmethod
    def is_retryable_network_error() -> bool:
        """ネットワーク層の異常（URLError・タイムアウト）はリトライ対象。"""
        return True


# GitHub API向けの既定リトライ方針（設計書⑫確定値）。
# GithubReleaseClient等が既定値として参照する単一の集約点（Should-2: 一元管理）。
DEFAULT_RETRY_POLICY: "RetryPolicy" = RetryPolicy()
