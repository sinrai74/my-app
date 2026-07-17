"""
MetricsStore: システムKPI（SystemMetrics）の保存・読込・追記。

設計書 v1.1.8 ⑥・⑲（システムKPI: system_metrics.json は当日全置換、
metrics/{YYYYMM}.jsonl は追記）、Step3計画書 §2 に基づく。

責務の厳守（Step3-5指示）:
- 本ストアは 保存・読込・append のみを担当する
- 集計・分析・平均計算・WARNING/ERROR判定などのビジネスロジックは実装しない
  （判定は services 層の責務、設計書⑲19.3）
- モデル⇔dict変換は SystemMetricsSerializer のみを利用する
  （json.dumps/loads・ファイルI/Oは本ストアが担当）

依存: 標準ライブラリ ＋ storage内部（Serializer/例外）のみ。
"""

from __future__ import annotations

import json
from pathlib import Path

from models.output import SystemMetrics
from storage.exceptions import ParseError
from storage.serializers.metrics_serializer import SystemMetricsSerializer


class MetricsStore:
    """system_metrics.json（当日全置換）と月次JSONL（追記）を管理する。

    - snapshot_path: 最新スナップショット（当日全置換）。system_metrics.json
    - monthly_dir: metrics/ ディレクトリ。{YYYYMM}.jsonl を追記する
    """

    def __init__(self, snapshot_path: Path, monthly_dir: Path) -> None:
        self._snapshot_path = Path(snapshot_path)
        self._monthly_dir = Path(monthly_dir)

    def write_snapshot(self, metrics_list: list[SystemMetrics]) -> None:
        """当日分のスナップショットを全置換で書き出す（⑲: 当日全置換）。

        集計はせず、渡されたリストをそのまま直列化して保存する。
        """
        self._snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        payload = [SystemMetricsSerializer.to_dict(m) for m in metrics_list]
        with open(self._snapshot_path, "w", encoding="utf-8", newline="") as f:
            json.dump(payload, f, ensure_ascii=False)

    def read_snapshot(self) -> list[SystemMetrics]:
        """スナップショットを読み込む。ファイル無しは空リスト。"""
        if not self._snapshot_path.exists():
            return []
        with open(self._snapshot_path, encoding="utf-8") as f:
            try:
                payload = json.load(f)
            except json.JSONDecodeError as exc:
                raise ParseError(
                    f"{self._snapshot_path.name}: broken JSON: {exc}"
                ) from exc
        if not isinstance(payload, list):
            raise ParseError(f"{self._snapshot_path.name}: expected JSON array")
        return [SystemMetricsSerializer.from_dict(item) for item in payload]

    def append_monthly(self, metrics: SystemMetrics) -> None:
        """月次JSONL（metrics/{YYYYMM}.jsonl）へ1件追記する（⑲: 月次追記）。

        月は metrics.race_date（YYYYMMDD）の先頭6桁から決定する。集計はしない。
        """
        if len(metrics.race_date) < 6:
            raise ParseError(
                f"race_date must be YYYYMMDD to derive month: {metrics.race_date!r}"
            )
        yyyymm = metrics.race_date[:6]
        self._monthly_dir.mkdir(parents=True, exist_ok=True)
        path = self._monthly_dir / f"{yyyymm}.jsonl"
        line = json.dumps(
            SystemMetricsSerializer.to_dict(metrics), ensure_ascii=False
        )
        with open(path, "a", encoding="utf-8", newline="") as f:
            f.write(line + "\n")

    def read_monthly(self, yyyymm: str) -> list[SystemMetrics]:
        """指定月のJSONLを読み込む。ファイル無しは空リスト。"""
        path = self._monthly_dir / f"{yyyymm}.jsonl"
        if not path.exists():
            return []
        metrics: list[SystemMetrics] = []
        with open(path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if text == "":
                    continue
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ParseError(
                        f"{path.name} line {line_no}: broken JSON: {exc}"
                    ) from exc
                metrics.append(SystemMetricsSerializer.from_dict(data))
        return metrics
