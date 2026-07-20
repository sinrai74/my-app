"""
BuyEngine Golden回帰: 実運用ログ由来の70件でBuyScore4項目を再現（Step4-6）。

Step4-6指示に対応。Golden一式（golden_buyscore/）とfreeze configフィクスチャが
存在する環境で実行され、未生成環境ではskipする。

Goldenが70件である理由:
  入力は実運用ログ buyscore_log.jsonl 由来であり、buyscoreの実データは
  70件しか存在しないため（人工データ追加は指示により禁止）。

検証対象（4項目のみ。decision/confidence/recommendationは存在しないため対象外）:
  buy_score / investment_type / kelly_fraction / skip_reason
  - 数値: 1e-6一致
  - 文字列: 完全一致
  - None: 完全一致

移植元純関数（calc_buyscore/investment_type/kelly_fraction/check_passthrough）を
core.buyscore経由で呼び、Wrapperが生成した期待値と比較する。
config改変検出のためfreeze SHA-256を検証する。
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from core.buyscore import (
    calc_buyscore,
    check_passthrough,
    investment_type,
    kelly_fraction,
)

GOLDEN_DIR = Path("tests/regression/golden_buyscore")
MANIFEST = GOLDEN_DIR / "manifest.json"
CONFIG_FIXTURE = GOLDEN_DIR / "buyscore_config_freeze.json"
CONFIG_SHA256 = "da6a4edaf8220aa52ed3e4577c23ff443509512a971d0ba322079eb4c4cd6f1d"

FLOAT_TOL = 1e-6


@unittest.skipUnless(
    MANIFEST.exists() and CONFIG_FIXTURE.exists(),
    "BuyEngine Golden or freeze config fixture not present "
    "(run tools/buyscore_golden_wrapper.py first).",
)
class TestBuyscoreGolden(unittest.TestCase):
    """実データ70件でのBuyScore4項目再現（100%一致が合否基準）。"""

    @classmethod
    def setUpClass(cls) -> None:
        digest = hashlib.sha256(CONFIG_FIXTURE.read_bytes()).hexdigest()
        if digest != CONFIG_SHA256:
            raise AssertionError(
                f"buyscore_config_freeze.json modified: sha256={digest} "
                f"(expected {CONFIG_SHA256})"
            )
        cls.config = json.loads(CONFIG_FIXTURE.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_count_is_70(self) -> None:
        """実運用ログ由来のため70件で固定。"""
        self.assertEqual(self.manifest["count"], 70)
        self.assertEqual(len(self.manifest["entries"]), 70)

    def test_hashes_match(self) -> None:
        for entry in self.manifest["entries"]:
            idx = entry["idx"]
            for kind, sha_key in (("inputs", "input_sha256"),
                                  ("expected", "expected_sha256")):
                path = GOLDEN_DIR / kind / f"{idx}.json"
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(
                    digest, entry[sha_key],
                    msg=f"idx={idx} kind={kind} golden modified",
                )

    def test_all_70_reproduce_buyscore_fields(self) -> None:
        for entry in self.manifest["entries"]:
            idx = entry["idx"]
            with self.subTest(idx=idx):
                self._verify_one(idx)

    def _verify_one(self, idx: str) -> None:
        race_input = json.loads(
            (GOLDEN_DIR / "inputs" / f"{idx}.json").read_text(encoding="utf-8")
        )
        expected = json.loads(
            (GOLDEN_DIR / "expected" / f"{idx}.json").read_text(encoding="utf-8")
        )
        candidate = race_input["candidate"]
        context = race_input["context"]

        buy_score = calc_buyscore(candidate, context, self.config)
        scored = dict(candidate)
        scored["buyscore"] = buy_score
        skip_reason = check_passthrough([scored], context, self.config)
        inv_type = investment_type(
            buy_score, candidate.get("ev", 0.0), candidate.get("odds", 0.0),
            context.get("race_type", "混戦"),
        )
        kelly = kelly_fraction(
            candidate.get("prob", 0), candidate.get("odds", 0), buy_score, self.config
        )

        self.assertAlmostEqual(
            buy_score, float(expected["buy_score"]), delta=FLOAT_TOL,
            msg=f"idx={idx} field=buy_score "
                f"expected={expected['buy_score']!r} actual={buy_score!r}",
        )
        self.assertEqual(
            inv_type, expected["investment_type"],
            msg=f"idx={idx} field=investment_type "
                f"expected={expected['investment_type']!r} actual={inv_type!r}",
        )
        self.assertAlmostEqual(
            kelly, float(expected["kelly_fraction"]), delta=FLOAT_TOL,
            msg=f"idx={idx} field=kelly_fraction "
                f"expected={expected['kelly_fraction']!r} actual={kelly!r}",
        )
        self.assertEqual(
            skip_reason, expected["skip_reason"],
            msg=f"idx={idx} field=skip_reason "
                f"expected={expected['skip_reason']!r} actual={skip_reason!r}",
        )


if __name__ == "__main__":
    unittest.main()
