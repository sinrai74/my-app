"""正式G3（B案）：G2 の42ファイル・本番入口・build_bundle・Legacy を変えていないこと（07 §10・§11）。"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BASELINE = REPO / "g2_baseline/registered/evaluate/baseline_evaluate_9558249_20261108.json"
NEW_MODULES = ("models.g3_input", "core.ml_bonus", "core.g3_assess", "shadow.g3_capture",
               "actions.g3_entrypoint")
UNCHANGED_OUTSIDE_42 = ("rebuild/actions/shadow_entrypoint.py", "rebuild/actions/production_entrypoint.py",
                        "rebuild/shadow/prediction_provider.py", "rebuild/pipelines/buy_pipeline.py",
                        "rebuild/actions/flags.py")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


class G2BaselineTests(unittest.TestCase):
    def test_g1_42_files_match_registered_baseline(self):
        code = json.loads(BASELINE.read_text(encoding="utf-8"))["code"]
        self.assertEqual(len(code), 42)
        diffs = [p for p, h in code.items() if _sha(REPO / p) != h]
        self.assertEqual(diffs, [])

    def test_g2_no_42_file_imports_new_modules(self):
        code = json.loads(BASELINE.read_text(encoding="utf-8"))["code"]
        pat = re.compile(r"\b(g3_input|ml_bonus|g3_assess|g3_capture|g3_entrypoint)\b")
        hits = [p for p in code if p.endswith(".py") and pat.search((REPO / p).read_text(encoding="utf-8"))]
        self.assertEqual(hits, [])

    def test_g3_new_modules_not_in_allowlist(self):
        allow = set(json.loads(BASELINE.read_text(encoding="utf-8"))["module_allowlist"])
        for m in NEW_MODULES:
            self.assertNotIn("rebuild/" + m.replace(".", "/") + ".py", allow)


class UnchangedOutsideTests(unittest.TestCase):
    def test_p1_p3_p4_files_unchanged_from_head(self):
        try:
            for rel in UNCHANGED_OUTSIDE_42:
                head = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
                                      capture_output=True, check=True).stdout
                self.assertEqual(hashlib.sha256(head).hexdigest(), _sha(REPO / rel), rel)
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            self.fail(f"cannot verify against git HEAD: {exc}")  # 監査 F-16：skip にしない


if __name__ == "__main__":
    unittest.main()
