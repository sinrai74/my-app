"""日次通知件数カウンタ（S4.3・C-2方式）。

「通知対象として生成された通知リクエスト1件」を1件として、race_date 単位
（JST暦日）で件数を保持する。実行をまたいで保持するため、評価JSONLと同じ
方式（ローカル追記＋Releases退避＋git commit）で永続化する。

重要（実装報告事項）:
  - これは **Phase0.5 ⑥ の保存対象表に存在しない新規永続データ**であり、
    L439「表にない新しいファイルの作成はレビュー必須」の対象になる。
  - EvaluationRepository / DurableEvaluationStore の責務は拡張しない
    （評価データとは別ファイル・別クラス）。
  - IdempotencyStore / CacheStore / MetricsStore は代用しない。
  - 並列実行の排他制御は行わない（今回の範囲外）。

保存先: notification_counts/{race_date}.json
  {"race_date": "YYYYMMDD", "count": <int>}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Protocol


class _ReleaseClient(Protocol):
    def upload_asset(self, tag: str, asset_name: str, file_path: Path) -> None: ...


class _GitClient(Protocol):
    def commit_and_push(self, paths: list[str], message: str) -> None: ...


class DailyNotificationCounter:
    """race_date 単位の通知リクエスト件数を読む・加算する。

    release/git を渡した場合のみ、加算のたびに退避・commitまで行う
    （評価JSONLの DurableEvaluationStore と同じ順序: 書込→退避→commit）。
    """

    def __init__(
        self,
        base_dir: str = "notification_counts",
        *,
        release: Optional[_ReleaseClient] = None,
        git: Optional[_GitClient] = None,
        tag: str = "data-store-v2",
    ) -> None:
        self._base_dir = Path(base_dir)
        self._release = release
        self._git = git
        self._tag = tag

    def path_for(self, race_date: str) -> Path:
        return self._base_dir / f"{race_date}.json"

    def count(self, race_date: str) -> int:
        """保存済み件数を返す（ファイルが無ければ0）。"""
        path = self.path_for(race_date)
        if not path.exists():
            return 0
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        value = data.get("count")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(
                f"invalid notification count in {path}: {value!r} "
                "(no default value is supplied)"
            )
        return value

    def add(self, race_date: str, delta: int) -> int:
        """件数を加算して保存し、加算後の件数を返す（delta=0は何もしない）。"""
        if delta < 0:
            raise ValueError("delta must be >= 0")
        if delta == 0:
            return self.count(race_date)
        new_count = self.count(race_date) + delta
        path = self.path_for(race_date)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as f:
            json.dump(
                {"race_date": race_date, "count": new_count}, f, ensure_ascii=False
            )
        if self._release is not None:
            self._release.upload_asset(
                self._tag, f"notification_counts_{race_date}.json", path
            )
        if self._git is not None:
            self._git.commit_and_push(
                [str(path)], f"notification count {race_date}={new_count}"
            )
        return new_count
