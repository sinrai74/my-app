"""
外部クライアント（GitHub Releases / git）のProtocolと実装。

設計書 v1.1.7 ⑩、Step3計画書 §5 に基づく。
Step3-1: protocols.py（Protocol定義）
Step3-2: retry.py（RetryPolicy）, github_release_client.py（実装）
Step3-3: git_client.py（SubprocessGitClient）
Step3-3: git_client.py（実装・予定）
"""

from storage.clients.git_client import SubprocessGitClient
from storage.clients.github_release_client import GithubReleaseClient
from storage.clients.protocols import GitClient, ReleaseClient
from storage.clients.retry import RetryPolicy

__all__ = [
    "GitClient",
    "GithubReleaseClient",
    "ReleaseClient",
    "RetryPolicy",
    "SubprocessGitClient",
]
