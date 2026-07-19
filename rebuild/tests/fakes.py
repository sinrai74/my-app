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


class FakeVenueStatsProvider:
    """VenueStatsProviderのFake。固定の分類・補正値を返す（Golden回帰・単体テスト用）。"""

    def __init__(
        self,
        water_types: dict[str, dict] | None = None,
        course_factors: dict[str, dict] | None = None,
    ) -> None:
        self.water_types = water_types or {}
        self.course_factors = course_factors or {}

    def classify_water_type(self, venue_name: str) -> dict:
        return self.water_types.get(venue_name, {"type": "standard", "label": "標準"})

    def get_venue_course_factor(self, venue_name: str, course: int) -> dict:
        return self.course_factors.get(venue_name, {"factor": 1.0, "samples": 0})


class FakePredictor:
    """ML予測器のFake（Step4計画書C5: ゴールデン期の予測を固定注入する）。

    eval_id -> win_probs（艇番->勝率）の固定辞書を返すCallable。
    実predictorの最終シグネチャはStep4-5（Ver4Engine.predict統合）で確定するが、
    「外部から注入された予測値をそのまま返す」という契約はここで固定する。
    """

    def __init__(self, fixed: dict[str, dict[int, float]]) -> None:
        self.fixed = dict(fixed)
        self.calls: list[str] = []

    def __call__(self, eval_id: str) -> dict[int, float]:
        self.calls.append(eval_id)
        if eval_id not in self.fixed:
            raise KeyError(f"no fixed prediction for {eval_id}")
        return dict(self.fixed[eval_id])


class FakePredictionStrategy:
    """PredictionStrategyのFake（Step4-5: predict DI境界の検証用）。

    受け取ったevaluation/odds/configを記録し、eval_idを引き継いだ
    最小のPredictionを返す。買い目生成の実体はStep4-6で結線する。
    """

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def __call__(self, evaluation, odds, config):
        from models.evaluation import Prediction

        self.calls.append((evaluation, odds, config))
        return Prediction(
            eval_id=evaluation.eval_id,
            pred_combo="1-2-3",
            pred_prob=0.1,
            pred_ev=1.0,
            pred_odds=10.0,
            confidence=0.5,
            why_bet="fake",
            patterns=(),
        )
