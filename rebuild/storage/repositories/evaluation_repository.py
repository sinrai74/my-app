"""
EvaluationRepository: evaluations/{date}.jsonl の読み書き。

設計書 v1.1.6 ⑥（保存層責務: 追記のみ・上書き禁止・eval_id一意性）、
Step2-6承認スコープに基づく。

責務:
- ファイルI/O・json.dumps/loads はここに集約する（Serializerはdictまで）
- RankingEntry.source_eval_id -> RaceEvaluation の解決（find_by_eval_id、案P）
- Releases退避・git commitとの不可分化はStep3以降（本Repositoryの範囲外）

読み込み挙動の定義（Step2-6レビュー観点）:
- ファイルが存在しない -> 空リスト（その日付の評価がまだ無い正常状態）
- 空ファイル -> 空リスト
- 壊れたJSON行 -> ParseError（行番号付き。黙殺しない）
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from models.evaluation import RaceEvaluation
from storage.exceptions import ParseError, StorageError
from storage.serializers.evaluation_serializer import RaceEvaluationSerializer


class EvaluationRepository:
    """1つのJSONLファイルを担当するRepository（パスはDIで注入）。"""

    def __init__(self, jsonl_path: Path) -> None:
        self._path = Path(jsonl_path)

    @property
    def path(self) -> Path:
        """担当ファイルのパス（読み取り専用）。DurableStoreの退避・commit対象特定用（v1.1.7）。"""
        return self._path

    def load_all(self) -> list[RaceEvaluation]:
        """全評価を読み込む。ファイル無し・空ファイルは空リスト。"""
        if not self._path.exists():
            return []
        evaluations: list[RaceEvaluation] = []
        with open(self._path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if text == "":
                    continue
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ParseError(
                        f"{self._path.name} line {line_no}: broken JSON: {exc}"
                    ) from exc
                evaluations.append(RaceEvaluationSerializer.from_dict(data))
        return evaluations

    def find_by_eval_id(self, eval_id: str) -> Optional[RaceEvaluation]:
        """eval_idでRaceEvaluationを解決する（RankingEntry参照の解決地点・案P）。"""
        for evaluation in self.load_all():
            if evaluation.eval_id == eval_id:
                return evaluation
        return None

    def append(self, evaluation: RaceEvaluation) -> None:
        """評価を1件追記する（追記のみ・上書き禁止 = 設計書⑥）。

        同一eval_idが既に存在する場合はStorageError（一意性チェック）。
        """
        existing_ids = {e.eval_id for e in self.load_all()}
        if evaluation.eval_id in existing_ids:
            raise StorageError(
                f"duplicate eval_id (append-only contract): {evaluation.eval_id}"
            )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            RaceEvaluationSerializer.to_dict(evaluation), ensure_ascii=False
        )
        with open(self._path, "a", encoding="utf-8", newline="") as f:
            f.write(line + "\n")
