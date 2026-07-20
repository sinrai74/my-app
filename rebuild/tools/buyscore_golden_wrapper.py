"""
tools/buyscore_golden_wrapper.py — BuyEngine Golden Input/Output生成（Step4-6）。

目的:
  実運用ログ buyscore_log.jsonl（70件）を入力に、移植元 x_buyscore.py の
  純関数（calc_buyscore/investment_type/kelly_fraction/check_passthrough）を
  「外側から」呼び、buy_score/investment_type/kelly_fraction/skip_reason の
  期待値を生成する。移植元は import して呼ぶだけで一切変更しない。

厳守事項（Step4-6指示）:
  - 入力は実運用ログ70件のみ（人工データ追加禁止）。
  - 不足入力はレガシー実装と同じ .get(key, default) の既定値で補完
    （＝Wrapperでは candidate/context に無いキーを足さず、移植元関数の
      デフォルトに委ねる）。
  - 期待値のSHA-256をmanifestへ記録し変更禁止化。

生成物:
  tests/regression/golden_buyscore/inputs/{idx}.json
  tests/regression/golden_buyscore/expected/{idx}.json
  tests/regression/golden_buyscore/manifest.json
  tests/regression/golden_buyscore/buyscore_config_freeze.json

実行（運用者Windows・freeze-v1-baseline状態）:
  set PYTHONPATH=C:\\Users\\81809\\Desktop\\my-app
  python rebuild\\tools\\buyscore_golden_wrapper.py \\
      --log buyscore_log.jsonl \\
      --out rebuild\\tests\\regression\\golden_buyscore
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

# 移植元は「呼ぶだけ」。本体は変更しない。
from x_buyscore import (
    calc_buyscore,
    check_passthrough,
    investment_type,
    kelly_fraction,
    load_config,
)

GOLDEN_SCHEMA_VERSION: int = 1
GENERATOR_VERSION: str = "1.0.0"

# freeze config を固定配置するための既知パス（存在すればコピーする）
_CONFIG_CANDIDATES = ("buyscore_config.json",)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _current_commit() -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def generate(log_path: Path, out_dir: Path, source_commit: Optional[str]) -> None:
    cfg = load_config()

    inputs_dir = out_dir / "inputs"
    expected_dir = out_dir / "expected"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    expected_dir.mkdir(parents=True, exist_ok=True)

    # freeze configを固定配置（存在する最初の候補）
    for name in _CONFIG_CANDIDATES:
        src = Path(name)
        if src.exists():
            shutil.copy2(src, out_dir / "buyscore_config_freeze.json")
            break

    entries: list[dict[str, Any]] = []
    with open(log_path, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]

    for idx, line in enumerate(lines):
        record = json.loads(line)
        # ログの candidates[0] を候補、レコード全体を context素材とする。
        # 不足キーは足さない（移植元の .get デフォルトに委ねる＝指示準拠）。
        candidates = record.get("candidates", [])
        candidate = candidates[0] if candidates else {}
        context = {
            "match_index": record.get("match_index", 0.0),
            "race_type": record.get("race_type", "混戦"),
        }

        # 入力を保存（移植元へ渡す素材そのもの）
        race_input = {"candidate": candidate, "context": context}
        in_path = inputs_dir / f"{idx:03d}.json"
        _write_json(in_path, race_input)

        # 移植元純関数を外側から呼ぶ
        buy_score = calc_buyscore(candidate, context, cfg)
        scored = dict(candidate)
        scored["buyscore"] = buy_score
        skip_reason = check_passthrough([scored], context, cfg)
        inv_type = investment_type(
            buy_score,
            candidate.get("ev", 0.0),
            candidate.get("odds", 0.0),
            context.get("race_type", "混戦"),
        )
        kelly = kelly_fraction(
            candidate.get("prob", 0), candidate.get("odds", 0), buy_score, cfg
        )

        expected = {
            "buy_score": buy_score,
            "investment_type": inv_type,
            "kelly_fraction": kelly,
            "skip_reason": skip_reason,
        }
        out_path = expected_dir / f"{idx:03d}.json"
        _write_json(out_path, expected)

        entries.append(
            {
                "idx": f"{idx:03d}",
                "input_sha256": _sha256_path(in_path),
                "expected_sha256": _sha256_path(out_path),
            }
        )

    manifest = {
        "golden_schema_version": GOLDEN_SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "created_from": "x_buyscore pure functions (unmodified)",
        "baseline_tag": "freeze-v1-baseline",
        "source_commit": source_commit or _current_commit(),
        "log_source": str(log_path),
        "count": len(entries),
        "entries": entries,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"generated {len(entries)} buyscore golden pairs into {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BuyEngine Golden.")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args()
    generate(args.log, args.out, args.source_commit)


if __name__ == "__main__":
    main()
