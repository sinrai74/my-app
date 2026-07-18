"""
tools/verify_expected_unchanged.py — Golden再生成の妥当性検証（Step4-2是正用）。

用途: Wrapper v1.1.0での再生成は「入力の拡張（danger_venue_factors追記）」であり、
期待出力（expected/*.json）は不変でなければならない。本スクリプトは
再生成前に退避した旧manifestと新manifestを比較し、全100件の
expected_sha256 が一致することを機械的に証明する。

1件でも不一致があれば、生成環境の条件（motor_history.csv等）が
Golden生成時から変化していることを意味する。その場合は続行せず報告すること
（実装ルール10）。

使い方:
  1. 再生成前:  copy tests\\regression\\golden4\\manifest.json manifest_old.json
  2. 再生成:    python rebuild\\tools\\golden_wrapper.py ...
  3. 検証:      python rebuild\\tools\\verify_expected_unchanged.py manifest_old.json rebuild\\tests\\regression\\golden4\\manifest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_expected_unchanged.py <old_manifest> <new_manifest>")
        return 2
    old = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    new = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))

    old_map = {e["eval_id"]: e["expected_sha256"] for e in old["entries"]}
    new_map = {e["eval_id"]: e["expected_sha256"] for e in new["entries"]}

    if set(old_map) != set(new_map):
        print("NG: eval_id集合が一致しません")
        print("  old_only:", sorted(set(old_map) - set(new_map)))
        print("  new_only:", sorted(set(new_map) - set(old_map)))
        return 1

    mismatches = [
        eval_id for eval_id in sorted(old_map)
        if old_map[eval_id] != new_map[eval_id]
    ]
    if mismatches:
        print(f"NG: 期待出力が変化しています（{len(mismatches)}件）。続行せず報告してください。")
        for eval_id in mismatches[:20]:
            print(
                f"  eval_id={eval_id} field=expected_sha256 "
                f"old={old_map[eval_id]} new={new_map[eval_id]}"
            )
        return 1

    # 補助フィールドの差分もeval_id/field/old/new形式で表示（情報提供のみ・NGにはしない）
    for field in ("count", "baseline_tag", "source_commit"):
        if old.get(field) != new.get(field):
            print(
                f"  info: eval_id=- field={field} "
                f"old={old.get(field)!r} new={new.get(field)!r}"
            )

    print(f"OK: 期待出力は全{len(old_map)}件で不変（入力のみ拡張されたことを確認）")
    print(f"  old generator: {old.get('generator_version')} -> new: {new.get('generator_version')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
