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

比較対象（Step6-3-5-5 §4 → Step6-3 Stage 2 で拡張）:
  - Stage 2以前: race のみ。
  - Stage 2以降: race に加えて evaluation（evaluation_view）と
    buy_decision（purchase_view）を追加する。
    * evaluation: sent_*.txt に存在する upset_score / race_type /
      danger_score のみ（evaluation_view の戻り値をそのまま使う）。
    * buy_decision: sent_*.txt の buy / buy_amounts から導く
      purchased_combos / purchased_amounts / n_bets / total_cost
      （purchase_view の戻り値をそのまま使う。順序保持・再計算なし）。
    * feature_set / prediction / buy_assessment / render_results /
      notification_requests は依然として生成しない。
      - prediction: LegacyラップのためShadow独立検証にならず対象外。
      - buy_assessment(G3-A): sent_*.txt に buyscore 等が0件のため
        Legacy比較材料が存在せず対象外（Rebuild単体テストで担保）。
  - evaluation_view / purchase_view が空（該当キーなし）の場合は、
    そのstageを legacy_values へ入れない（比較側で「取得元なし」扱い）。

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
        {"race": <record.race_view()>,
         ["evaluation": <record.evaluation_view()>,]  # 非空のときのみ
         ["buy_decision": <record.purchase_view()>]}  # 取得できたときのみ
        eval_id が records に存在しない場合は None。

    設計（Step6-3 Stage 2）:
      - race は従来通り常に含める。
      - evaluation は evaluation_view() が非空の場合のみ含める
        （sent に upset_score 等が無い古い行では含めない）。
      - buy_decision は purchase_view() が None でない場合のみ含める
        （buy / buy_amounts を持つ行のみ。無い行では含めない）。
      - いずれも record のビューをそのまま使う。値の加工・補完はしない。
    """
    record = records.get(eval_id)
    if record is None:
        log.info(
            "legacy_values not built: eval_id=%s is not in sent records",
            eval_id,
        )
        return None
    values: dict[str, dict] = {"race": record.race_view()}

    evaluation_view = record.evaluation_view()
    if evaluation_view:
        values["evaluation"] = evaluation_view

    purchase_view = record.purchase_view()
    if purchase_view is not None:
        values["buy_decision"] = purchase_view

    return values
