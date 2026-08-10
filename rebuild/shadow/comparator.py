"""
Shadow比較エンジン（Step5-6）: JSON正規化 → 再帰比較 → 差分出力。

Step5-0〜5-5で各Pipelineテストが個別に使っていた比較方式（eval_id/
field_path/legacy/rebuild形式）を、Shadow運用の本番コードとして集約する。
比較方式そのものは変更しない（新しい比較方式の追加は禁止）。

モデル→dictの変換は、既存Serializer（storage.serializers、Step2完成資産）
がある型はそれを再利用し、無い型（Race/FeatureSet/BuyAssessment/
RenderResult/NotificationRequest）のみ属性の単純コピー（判定・加工なし）
で辞書化する。
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from models.evaluation import Prediction, RaceEvaluation
from storage.serializers.evaluation_serializer import (
    PredictionSerializer,
    RaceEvaluationSerializer,
)

# 型 → 既存Serializer.to_dict の対応表（唯一の分岐点。優先順位1）。
# 新しい型別変換ロジックを増やす場合はここへ既存Serializerを追加するのみ
# （Serializerを新規作成することはしない＝指示の禁止事項）。
_SERIALIZERS: dict[type, Any] = {
    RaceEvaluation: RaceEvaluationSerializer.to_dict,
    Prediction: PredictionSerializer.to_dict,
}


def to_comparable(obj: Any) -> Any:
    """比較用dictへ変換する共通ヘルパー（固定優先順位のみ・判定/計算なし）。

    優先順位:
      1. 既存Serializerがある型 → Serializer.to_dict を利用
      2. dataclass → dataclasses.asdict()
      3. __dict__ を持つ通常オブジェクト → vars()
      4. __dict__ が無く __slots__ を持つオブジェクト → __slots__を汎用列挙
    型ごとの個別変換（RenderResultだけ/NotificationRequestだけ等のif）は
    追加しない。段階4は「__slots__を持つ」という構造的性質のみで判定する
    汎用処理であり、特定型を対象としない。
    """
    if obj is None:
        return None
    serializer = _SERIALIZERS.get(type(obj))
    if serializer is not None:
        return serializer(obj)
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: to_comparable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_comparable(v) for v in obj]
    if hasattr(obj, "__dict__"):
        return {k: to_comparable(v) for k, v in vars(obj).items()}
    if hasattr(obj, "__slots__"):
        return {
            name: to_comparable(getattr(obj, name))
            for name in _iter_slots(obj)
            if hasattr(obj, name)
        }
    return obj


def _iter_slots(obj: Any) -> list[str]:
    """継承階層も含めて__slots__を汎用的に列挙する（特定型に依存しない）。"""
    names: list[str] = []
    for klass in type(obj).__mro__:
        slots = getattr(klass, "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        names.extend(slots)
    return names


def compare(eval_id: str, legacy: Any, rebuild: Any, path: str = "$") -> list[dict]:
    """legacy/rebuildを再帰比較し、差分を eval_id/field_path/legacy/rebuild
    形式のリストで返す（Step5-0で確定した方式。新方式は追加しない）。
    """
    legacy_c = json.loads(
        json.dumps(to_comparable(legacy), ensure_ascii=False, sort_keys=True)
    )
    rebuild_c = json.loads(
        json.dumps(to_comparable(rebuild), ensure_ascii=False, sort_keys=True)
    )
    diffs: list[dict] = []

    def rec(le: Any, re: Any, p: str) -> None:
        if isinstance(le, dict) and isinstance(re, dict):
            for key in sorted(set(le)):
                rec(le.get(key), re.get(key), f"{p}.{key}")
        elif isinstance(le, list) and isinstance(re, list):
            if len(le) != len(re):
                diffs.append({
                    "eval_id": eval_id, "field_path": f"{p}.__len__",
                    "legacy": len(le), "rebuild": len(re),
                })
            for i, (a, b) in enumerate(zip(le, re)):
                rec(a, b, f"{p}[{i}]")
        elif le != re:
            diffs.append({
                "eval_id": eval_id, "field_path": p, "legacy": le, "rebuild": re,
            })

    rec(legacy_c, rebuild_c, path)
    return diffs
