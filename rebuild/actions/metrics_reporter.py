"""S6: SystemMetrics の生成・保存・月次Release退避（run-level）。

設計根拠（Phase0.5）:
  - ③3.16 / ⑥ L434: ジョブ実行ごとに1レコード。当日jsonは全置換、月次jsonlは
    追記、metrics_id一意、月次でReleasesへ、欠損許容だが欠損自体をWARNING
  - L298: metrics_id = {date}_{job}_{run_id}（run_id は Actions の run_id）
  - ⑲19.2: counters/rates/timings/errors の項目名

S6で追加する設計判断（既存資料に明示根拠がないもの）:
  - job_name = "production_evaluation"
  - Metrics の Release tag は data-store-v2、asset は metrics_{YYYYMM}.jsonl
    （latest asset は作らない。当月は退避対象外。未退避の全月を対象）
  - Metrics は git管理外・Releasesのみ（本モジュールは commit/push をしない）
  - 「Production処理開始」= single_race_date() 完了後（呼び出し側の責務）

方針:
  - 未計測の counter / rate はキー自体を作らない（0で捏造しない）
  - errors は実在する例外クラス（StorageError / ParseError / ConfigError）のみ
    分類し、存在しない6分類を新設しない
  - Metrics の保存・退避の失敗は WARNING のみ。ジョブの exit code を変えない
  - Freeze対象（models / metrics_store / durability / repositories）は無変更で、
    既存API（write_snapshot / append_monthly / read_monthly / list_assets /
    upload_asset）だけを使う
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol

from models.output import SystemMetrics

log = logging.getLogger(__name__)

JOB_NAME = "production_evaluation"
RELEASE_TAG = "data-store-v2"  # S6の設計判断
SCHEMA_VERSION = 1
_MONTHLY_FILE = re.compile(r"^(\d{6})\.jsonl$")

# errors に用いる分類は実在する例外クラスのみ（6分類を新設しない）
KNOWN_ERROR_TYPES = ("StorageError", "ParseError", "ConfigError")


class _MetricsStore(Protocol):
    def write_snapshot(self, metrics_list: list[SystemMetrics]) -> None: ...
    def read_snapshot(self) -> list[SystemMetrics]: ...
    def append_monthly(self, metrics: SystemMetrics) -> None: ...
    def read_monthly(self, yyyymm: str) -> list[SystemMetrics]: ...


class _ReleaseClient(Protocol):
    def list_assets(self) -> list[str]: ...
    def upload_asset(self, file_path: Path, asset_name: str) -> None: ...


def metrics_id_for(race_date: str, run_id: str, job_name: str = JOB_NAME) -> str:
    """Phase0.5 L298: {date}_{job}_{run_id}。実行日時で race_date を代替しない。"""
    return f"{race_date}_{job_name}_{run_id}"


def monthly_asset_name(yyyymm: str) -> str:
    return f"metrics_{yyyymm}.jsonl"


def build_errors(
    race_errors: "list[Mapping[str, str]] | None" = None,
    exception: Optional[BaseException] = None,
) -> dict[str, Any]:
    """errors dict を組み立てる（実在分類のみ・未知の分類名を作らない）。

    race_errors は run_production_day の errors（"TypeName: message" 形式）。
    """
    entries: list[str] = [str(e.get("error", "")) for e in (race_errors or [])]
    if exception is not None:
        entries.append(f"{type(exception).__name__}: {exception}")
    if not entries:
        return {}
    errors: dict[str, Any] = {"total": len(entries)}
    for name in KNOWN_ERROR_TYPES:
        count = sum(1 for e in entries if e.startswith(f"{name}:"))
        if count:
            errors[name] = count
    errors["last_message"] = entries[-1][:300]
    return errors


class MetricsReporter:
    """SystemMetrics を1件生成し、保存・月次退避まで行う（失敗は WARNING）。"""

    def __init__(
        self,
        metrics_store: _MetricsStore,
        *,
        monthly_dir: str = "metrics",
        release: Optional[_ReleaseClient] = None,
        job_name: str = JOB_NAME,
    ) -> None:
        self._store = metrics_store
        self._monthly_dir = Path(monthly_dir)
        self._release = release
        self._job_name = job_name

    def build(
        self,
        *,
        race_date: str,
        run_id: str,
        started_at: str,
        finished_at: str,
        duration_seconds: float,
        status: str,
        counters: Mapping[str, Any],
        errors: Mapping[str, Any],
    ) -> SystemMetrics:
        """SystemMetrics を組み立てる（値の加工・補完はしない）。"""
        return SystemMetrics(
            metrics_id=metrics_id_for(race_date, run_id, self._job_name),
            race_date=race_date,
            job_name=self._job_name,
            run_id=run_id,
            started_at=started_at,
            finished_at=finished_at,
            duration_seconds=duration_seconds,
            status=status,
            counters=dict(counters),
            rates={},  # 未実測のため空（0.0で捏造しない）
            errors=dict(errors),
            schema_version=SCHEMA_VERSION,
        )

    def report(self, metrics: SystemMetrics) -> None:
        """保存（snapshot / monthly）と月次Release退避。失敗はWARNINGのみ。"""
        self._save_snapshot(metrics)
        self._append_monthly(metrics)
        self.sync_monthly_releases(metrics.race_date)

    # ---- 保存 ---------------------------------------------------------

    def _save_snapshot(self, metrics: SystemMetrics) -> None:
        """当日 system_metrics.json を全置換（当日分の配列として保持）。"""
        try:
            existing = self._store.read_snapshot()
        except Exception as exc:  # noqa: BLE001 計測データのため継続（⑥欠損許容）
            log.warning("Metrics snapshot read failed: %s", exc)
            existing = []
        same_day = [
            m for m in existing
            if m.race_date == metrics.race_date
            and m.metrics_id != metrics.metrics_id
        ]
        try:
            self._store.write_snapshot(same_day + [metrics])
        except Exception as exc:  # noqa: BLE001
            log.warning("Metrics snapshot write failed: %s", exc)

    def _append_monthly(self, metrics: SystemMetrics) -> None:
        """月次JSONLへ追記。同一 metrics_id が既にあれば追記せずWARNING。

        内容が同一であるとは仮定しない（既存行は上書きしない）。
        """
        yyyymm = metrics.race_date[:6]
        try:
            existing = self._store.read_monthly(yyyymm)
        except Exception as exc:  # noqa: BLE001
            log.warning("Metrics monthly read failed (%s): %s", yyyymm, exc)
            return
        if any(m.metrics_id == metrics.metrics_id for m in existing):
            log.warning(
                "Metrics monthly append skipped: metrics_id already exists "
                "(%s); existing line is not overwritten", metrics.metrics_id,
            )
            return
        try:
            self._store.append_monthly(metrics)
        except Exception as exc:  # noqa: BLE001
            log.warning("Metrics monthly append failed: %s", exc)

    # ---- 月次Release --------------------------------------------------

    def unreleased_months(self, race_date: str) -> list[str]:
        """未退避月（ローカルにあり・当月以外・Release assetが無い月）を古い順に返す。"""
        if self._release is None:
            return []
        try:
            assets = set(self._release.list_assets())
        except Exception as exc:  # noqa: BLE001
            log.warning("Metrics release asset listing failed: %s", exc)
            return []
        current = race_date[:6]
        months: list[str] = []
        if not self._monthly_dir.exists():
            return []
        for path in sorted(self._monthly_dir.iterdir()):
            matched = _MONTHLY_FILE.match(path.name)
            if matched is None:
                continue
            yyyymm = matched.group(1)
            if yyyymm == current:
                continue  # 当月は退避対象外
            if monthly_asset_name(yyyymm) in assets:
                continue  # 退避済みはskip
            months.append(yyyymm)
        return months

    def sync_monthly_releases(self, race_date: str) -> list[str]:
        """未退避月をすべてReleaseへupload（古い月から）。失敗はWARNINGで継続。"""
        uploaded: list[str] = []
        for yyyymm in self.unreleased_months(race_date):
            path = self._monthly_dir / f"{yyyymm}.jsonl"
            try:
                self._release.upload_asset(path, monthly_asset_name(yyyymm))
            except Exception as exc:  # noqa: BLE001 次回実行で再試行する
                log.warning(
                    "Metrics monthly release upload failed (%s): %s", yyyymm, exc
                )
                continue
            log.info("Metrics monthly released: %s", monthly_asset_name(yyyymm))
            uploaded.append(yyyymm)
        return uploaded
