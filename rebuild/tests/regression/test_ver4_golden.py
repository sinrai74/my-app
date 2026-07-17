"""
Ver4 Golden Input 回帰テスト基盤（Step4-0）。

設計書 v1.1.8 ⑭、Step4計画書 §5・C6/C7 に対応。

本ファイルはStep4-0時点では「基盤」のみを提供する:
  - Golden一式（inputs/expected/manifest）が存在すれば、manifestのSHA-256を検証する
  - Golden一式が未生成なら skip する（実データ環境でtools/golden_wrapper.pyを実行後に有効化）
  - Ver4Engineとの一致比較（入力→出力）はStep4-5で本テストに追加する
    （このコメント位置に assertion を差し込む。基盤の骨格は完成済み）

C7（変更禁止）: manifestに記録されたSHA-256と、実ファイルのSHA-256の一致を検証する。
不一致（＝Goldenが改変された）ならFail。正当な更新時のみ承認記録とともに再生成する。
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

GOLDEN4_DIR = Path("tests/regression/golden4")
MANIFEST = GOLDEN4_DIR / "manifest.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _golden_available() -> bool:
    return MANIFEST.exists()


@unittest.skipUnless(
    _golden_available(),
    "Ver4 Golden not generated yet. Run tools/golden_wrapper.py in a freeze-time "
    "data environment first (see tests/regression/golden4/README.md).",
)
class TestVer4GoldenIntegrity(unittest.TestCase):
    """Golden一式が存在する場合のみ走る整合性テスト（C6件数・C7ハッシュ）。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_count_is_100(self) -> None:
        """C6: Golden Input件数が100件で固定されていること。"""
        self.assertEqual(self.manifest["count"], 100)
        self.assertEqual(len(self.manifest["entries"]), 100)

    def test_input_and_expected_hashes_match(self) -> None:
        """C7: 各Golden Input/期待出力JSONのSHA-256がmanifestと一致すること（改変検出）。"""
        for entry in self.manifest["entries"]:
            eval_id = entry["eval_id"]
            in_path = GOLDEN4_DIR / "inputs" / f"{eval_id}.json"
            out_path = GOLDEN4_DIR / "expected" / f"{eval_id}.json"
            self.assertTrue(in_path.exists(), f"missing input: {eval_id}")
            self.assertTrue(out_path.exists(), f"missing expected: {eval_id}")
            self.assertEqual(
                _sha256(in_path), entry["input_sha256"],
                msg=f"input JSON modified: {eval_id}",
            )
            self.assertEqual(
                _sha256(out_path), entry["expected_sha256"],
                msg=f"expected JSON modified: {eval_id}",
            )

    def test_source_commit_recorded(self) -> None:
        """Should: 取得元コミットIDがmanifestに保持されていること（null許容）。"""
        self.assertIn("source_commit", self.manifest)

    def test_generator_version_recorded(self) -> None:
        """Should: Wrapper版数がmanifestに記録されていること。"""
        self.assertIn("generator_version", self.manifest)

    # --- Step4-5でここにVer4Engine一致テストを追加する ---
    # def test_ver4_engine_matches_expected(self) -> None:
    #     from core.engines.ver4 import Ver4Engine
    #     for entry in self.manifest["entries"]:
    #         race_input = load_input(entry["eval_id"])
    #         got = Ver4Engine(...).evaluate(...)  # 入力→出力
    #         assert_matches(got, load_expected(entry["eval_id"]), tol=1e-6)


if __name__ == "__main__":
    unittest.main()
