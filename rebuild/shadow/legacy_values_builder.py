"""
shadow/legacy_values_builder.py（Step6-3-21）: legacy_values の組み立て。

責務（Step6-3-5-3 §1.1 で確定）:
  - 呼び出し側が用意した LegacySentRecord の辞書から、eval_id に対応する
    レコードを取り出す。
  - LegacySentRecord.race_view() を呼ぶ。
  - run_and_compare の legacy_values 引数へ渡す辞書を組み立てる。

担当しない処理（Step6-3-5-3 §1.2 で確定）:
  - sent_*.txt のパス組み立て（呼び出し側）
  - load_sent_records() の呼び出し（呼び出し側。Step6-3-14 O-2 で日付単位
    生成を前提としたため、読み込みは呼び出し側が行う）
  - Legacy値の補正・整形・デフォルト補完（race_view() の戻り値をそのまま使う）
  - 比較処理（shadow/comparator.py の compare が行う）
  - Pipeline の実行

比較対象（Step6-3-5-5 §4 で確定）:
  - race のみ。evaluation / feature_set / prediction / buy_assessment /
    render_results / notification_requests は生成しない。

eval_id 不在時（Step6-3-18 §1.3 で確定）:
  - None を返す。legacy_source.get_legacy_record の戻り値型
    （Optional[LegacySentRecord]）と同じ形式に合わせる。
  - 呼び出し側が None を検出して skip 判断を行う（Step6-3-18 §2.3）。
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

log = logging.getLogger(__name__)


def build_legacy_values(
    records: Mapping[str, Any],
    eval_id: str,
) -> Optional[dict[str, dict]]:
    """legacy_values を組み立てる。eval_id が records に無ければ None を返す。

    Args:
        records: load_sent_records() の戻り値（eval_id -> LegacySentRecord）。
        eval_id: 対象レースの識別子（runner.py と同一形式）。

    Returns:
        {"race": <record.race_view() の戻り値>}。
        eval_id が records に存在しない場合は None。
    """
    record = records.get(eval_id)
    if record is None:
        log.info(
            "legacy_values not built: eval_id=%s is not in sent records",
            eval_id,
        )
        return None
    return {"race": record.race_view()}
