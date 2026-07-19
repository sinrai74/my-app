"""
Ver4Engine Golden統合回帰: Golden入力100件からのv4全項目再現（Step4-5）。

Step4計画書 §5・C2/C6/C7/C8、Step4-5テスト要件④に対応。

検証内容:
  A. build_legacy_evaluation の出力dict全体を、期待出力JSON（expected/*.json）
     と再帰比較（数値1e-6・文字列/None/bool完全一致・キー集合一致）。
     danger_score / upset_score / rank_index / featured_boats / upset_detail
     （_features含む）/ water_type / venue_factor_1c / boat1 / boat1_features /
     model_version / venue / venue_num のすべてを含む。
  B. evaluate() が生成する RaceEvaluation の全フィールド検証
     （danger/upset/match_index/rank_index正準形/featured正準形/
      win_probs=FakePredictor固定値/engine系メタ/evaluated_at）。
  C. predict() が PredictionStrategy 経由で eval_id を引き継ぐこと。

失敗時は eval_id と JSONパス、expected / actual を表示する。
"""

from __future__ import annotations

import hashlib
import json
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any

from core.engine import Ver4Engine
from core.rank import compute_match_index
from models.evaluation import FeatureSet
from models.race import Race
from tests.fakes import FakePredictionStrategy, FakePredictor

GOLDEN4_DIR = Path("tests/regression/golden4")
MANIFEST = GOLDEN4_DIR / "manifest.json"
CONFIG_FIXTURE = GOLDEN4_DIR / "asahi_config_freeze.json"
CONFIG_SHA256 = "6a7862b8dc36006854d557fa8e2bfd12823433a0c51c5f092bb7f357b57520b3"

FLOAT_TOL = 1e-6

# race_grade_number（BoatraceOpenAPI生値）は0-4に限らない（レガシーも
# 未知値を検証せず.get(n, default)でフォールバックするのみ）。
# Race.grade復元も同じ規約に合わせ、未知はf"grade{n}"とする
# （Ver4Engine._grade_to_numberの逆写像として1対1になるよう定義）。
_GRADE_NUM_TO_STR = {0: "一般", 1: "G3", 2: "G2", 3: "G1", 4: "SG"}


def _grade_num_to_race_grade_str(n: int) -> str:
    return _GRADE_NUM_TO_STR.get(n, f"grade{n}")
_FIXED_NOW = datetime(2026, 7, 18, 7, 30, 0)


class _GoldenVenueProvider:
    """Golden生成時のvenue統計を再現するProvider。

    - classify_water_type: 期待出力に記録された water_type dict をそのまま返す
    - get_venue_course_factor:
        lane1 -> 期待出力の venue_factor_1c dict（全フィールド。
                 v4トップレベル・_featuresのfactor/samples参照を満たす）
        lane2-6 -> Golden入力に記録されたレーン別factor（danger/rank用）
    """

    def __init__(self, water_type: dict, venue_factor_1c: dict,
                 course_rentai_factors: dict[str, float]) -> None:
        self._water_type = water_type
        self._venue_factor_1c = venue_factor_1c
        self._factors = course_rentai_factors

    def classify_water_type(self, venue_name: str) -> dict:
        return self._water_type

    def get_venue_course_factor(self, venue_name: str, course: int) -> dict:
        if course == 1:
            return self._venue_factor_1c
        return {"factor": self._factors[str(course)]}


@unittest.skipUnless(
    MANIFEST.exists() and CONFIG_FIXTURE.exists(),
    "Ver4 Golden or freeze config fixture not present.",
)
class TestEngineGolden(unittest.TestCase):
    """Golden 100件でのv4全項目再現（100%一致が合否基準）。"""

    @classmethod
    def setUpClass(cls) -> None:
        digest = hashlib.sha256(CONFIG_FIXTURE.read_bytes()).hexdigest()
        if digest != CONFIG_SHA256:
            raise AssertionError(f"asahi_config_freeze.json modified: {digest}")
        cls.config = json.loads(CONFIG_FIXTURE.read_text(encoding="utf-8"))
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_all_100_inputs_reproduce_full_evaluation(self) -> None:
        for entry in self.manifest["entries"]:
            eval_id = entry["eval_id"]
            with self.subTest(eval_id=eval_id):
                self._verify_one(eval_id)

    # ---------------- 1件分の検証 ----------------

    def _verify_one(self, eval_id: str) -> None:
        race_input = json.loads(
            (GOLDEN4_DIR / "inputs" / f"{eval_id}.json").read_text(encoding="utf-8")
        )
        expected = json.loads(
            (GOLDEN4_DIR / "expected" / f"{eval_id}.json").read_text(encoding="utf-8")
        )
        recorded = race_input.get("danger_venue_factors")
        if recorded is None:
            self.skipTest("golden inputs are v1.0.0; regenerate with wrapper v1.1.0")

        # 前提の自己検証: 記録lane1 factorと期待venue_factor_1cの一致
        self.assertAlmostEqual(
            float(recorded["course_rentai"]["1"]),
            float(expected["venue_factor_1c"]["factor"]),
            delta=FLOAT_TOL,
            msg=f"eval_id={eval_id} path=venue_factor_lane1_selfcheck",
        )

        boats = race_input["boats"]
        provider = _GoldenVenueProvider(
            water_type=expected["water_type"],
            venue_factor_1c=expected["venue_factor_1c"],
            course_rentai_factors=recorded["course_rentai"],
        )
        predictor = FakePredictor({eval_id: {1: 0.40, 2: 0.20, 3: 0.15}})
        strategy = FakePredictionStrategy()
        engine = Ver4Engine(
            boats_resolver=lambda race: boats,
            venue_stats=provider,
            predictor=predictor,
            prediction_strategy=strategy,
        )

        # ---- A. v4 dict全体の再帰一致 ----
        legacy = engine.build_legacy_evaluation(
            boats,
            self.config,
            race_grade=race_input["race_grade"],
            venue_num=race_input["venue_num"],
            is_night=race_input["is_night"],
        )
        canon = json.loads(json.dumps(legacy, ensure_ascii=False))
        self._assert_deep_equal(eval_id, "$", expected, canon)

        # ---- B. RaceEvaluation全フィールド ----
        race_grade = race_input["race_grade"]
        date, venue_s, race_s = eval_id.split("_")
        race = Race(
            race_date=date,
            venue_num=race_input["venue_num"],
            venue_name=race_input["venue"] or "",
            race_number=int(race_s),
            close_time="",
            is_night=race_input["is_night"],
            entries=(),
            grade=_grade_num_to_race_grade_str(race_grade),
        )
        feature_set = FeatureSet(
            eval_id=eval_id, feature_schema_version=1, built_at="golden",
            boat_features={1: {}}, race_features={}, local_features=None,
            missing_keys=(),
        )
        evaluation = engine.evaluate(race, feature_set, None, self.config, _FIXED_NOW)

        self.assertEqual(evaluation.eval_id, eval_id)
        self.assertAlmostEqual(
            evaluation.danger_score, float(expected["danger_score"]),
            delta=FLOAT_TOL, msg=f"eval_id={eval_id} path=RaceEvaluation.danger_score",
        )
        self.assertAlmostEqual(
            evaluation.upset_score, float(expected["upset_score"]),
            delta=FLOAT_TOL, msg=f"eval_id={eval_id} path=RaceEvaluation.upset_score",
        )
        self.assertAlmostEqual(
            evaluation.match_index,
            compute_match_index(float(expected["upset_score"])),
            delta=FLOAT_TOL, msg=f"eval_id={eval_id} path=RaceEvaluation.match_index",
        )
        self._assert_deep_equal(
            eval_id, "$.RaceEvaluation.rank_index",
            expected["rank_index"],
            json.loads(json.dumps(evaluation.rank_index, ensure_ascii=False)),
        )
        self._assert_deep_equal(
            eval_id, "$.RaceEvaluation.featured_boats",
            {"featured": expected["featured_boats"]},
            json.loads(json.dumps(evaluation.featured_boats, ensure_ascii=False)),
        )
        self.assertEqual(
            evaluation.danger_breakdown and set(evaluation.danger_breakdown.keys()),
            set(expected["danger_breakdown"].keys()),
            msg=f"eval_id={eval_id} path=RaceEvaluation.danger_breakdown.keys",
        )
        self.assertEqual(evaluation.model_version, expected["model_version"])
        self.assertEqual(evaluation.win_probs, {1: 0.40, 2: 0.20, 3: 0.15})
        self.assertEqual(evaluation.engine_name, "ver4")
        self.assertEqual(evaluation.engine_version, "4.0.0")
        self.assertEqual(evaluation.feature_schema_version, 1)
        self.assertEqual(evaluation.evaluated_at, _FIXED_NOW.isoformat())
        self.assertEqual(evaluation.upset_reasons, ())
        self.assertEqual(evaluation.race_type, "")
        self.assertEqual(evaluation.venue_name, race_input["venue"] or "")
        self.assertTrue(evaluation.is_night == race_input["is_night"])

        # ---- C. Prediction（Strategy委譲・eval_id引き継ぎ） ----
        prediction = engine.predict(evaluation, None, self.config)
        self.assertEqual(
            prediction.eval_id, eval_id,
            msg=f"eval_id={eval_id} path=Prediction.eval_id",
        )

    # ---------------- 再帰比較 ----------------

    def _assert_deep_equal(
        self, eval_id: str, path: str, exp: Any, act: Any
    ) -> None:
        if isinstance(exp, dict):
            self.assertIsInstance(
                act, dict, msg=f"eval_id={eval_id} path={path} "
                               f"expected=dict actual={type(act).__name__}"
            )
            self.assertEqual(
                set(act.keys()), set(exp.keys()),
                msg=f"eval_id={eval_id} path={path}.keys "
                    f"expected={sorted(exp)} actual={sorted(act)}",
            )
            for key in exp:
                self._assert_deep_equal(eval_id, f"{path}.{key}", exp[key], act[key])
        elif isinstance(exp, list):
            self.assertIsInstance(
                act, list, msg=f"eval_id={eval_id} path={path} "
                               f"expected=list actual={type(act).__name__}"
            )
            self.assertEqual(
                len(act), len(exp),
                msg=f"eval_id={eval_id} path={path}.length "
                    f"expected={len(exp)} actual={len(act)}",
            )
            for i, (e, a) in enumerate(zip(exp, act)):
                self._assert_deep_equal(eval_id, f"{path}[{i}]", e, a)
        elif isinstance(exp, bool) or exp is None:
            self.assertEqual(
                act, exp,
                msg=f"eval_id={eval_id} path={path} "
                    f"expected={exp!r} actual={act!r}",
            )
        elif isinstance(exp, (int, float)):
            self.assertIsInstance(
                act, (int, float),
                msg=f"eval_id={eval_id} path={path} "
                    f"expected={exp!r} actual={act!r}",
            )
            self.assertNotIsInstance(
                act, bool,
                msg=f"eval_id={eval_id} path={path} "
                    f"expected={exp!r} actual={act!r}",
            )
            self.assertAlmostEqual(
                float(act), float(exp), delta=FLOAT_TOL,
                msg=f"eval_id={eval_id} path={path} "
                    f"expected={exp!r} actual={act!r}",
            )
        else:
            self.assertEqual(
                act, exp,
                msg=f"eval_id={eval_id} path={path} "
                    f"expected={exp!r} actual={act!r}",
            )


if __name__ == "__main__":
    unittest.main()
