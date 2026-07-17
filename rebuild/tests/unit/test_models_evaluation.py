"""
Feature/Evaluation系データモデル（models/evaluation.py）の単体テスト。

設計書 Phase0.5 v1.1 ③3.3〜3.6、⑤5.4（純粋性・再現性を支えるデータ不変性）に対応。
標準ライブラリ unittest のみを使用する。
"""

from __future__ import annotations

import unittest

from models.evaluation import BuyDecision, FeatureSet, Prediction, RaceEvaluation


def _build_sample_feature_set() -> FeatureSet:
    """テスト用の最小構成 FeatureSet を組み立てるヘルパー。"""
    return FeatureSet(
        eval_id="20260713_12_05",
        feature_schema_version=1,
        built_at="2026-07-13T06:00:00+09:00",
        boat_features={
            1: {"win_rate": 6.50, "motor_rate2": 38.5, "avg_st": 0.16},
            2: {"win_rate": 5.20, "motor_rate2": 35.0, "avg_st": 0.18},
        },
        race_features={"venue_factor": 1.02, "field_strength": 0.75},
    )


def _build_sample_evaluation() -> RaceEvaluation:
    """テスト用の最小構成 RaceEvaluation を組み立てるヘルパー。"""
    return RaceEvaluation(
        eval_id="20260713_12_05",
        race_date="20260713",
        venue_num=12,
        venue_name="住之江",
        race_number=5,
        is_night=False,
        engine_name="ver4",
        engine_version="4.2.0",
        feature_schema_version=1,
        model_version="v20260701",
        evaluated_at="2026-07-13T06:05:00+09:00",
        danger_score=78.5,
        danger_breakdown={"win_rate_low": 28, "motor_bad": 0.0},
        upset_score=42.0,
        upset_reasons=("1号艇平均ST遅め",),
        rank_index={"1": 0.80, "2": 0.65},
        featured_boats={"featured": [1, 4]},
        win_probs={1: 0.42, 2: 0.18, 3: 0.15, 4: 0.12, 5: 0.08, 6: 0.05},
        race_type="本命",
        match_index=44.1,
        features=_build_sample_feature_set(),
    )


class TestFeatureSet(unittest.TestCase):
    def test_construct_minimum_fields(self) -> None:
        """必須項目のみでFeatureSetが構築できること。local_featuresは既定でNone。"""
        fs = _build_sample_feature_set()
        self.assertEqual(fs.feature_schema_version, 1)
        self.assertIsNone(fs.local_features)
        self.assertEqual(fs.missing_keys, ())

    def test_immutable(self) -> None:
        """frozen dataclassのため属性再代入がFrozenInstanceErrorになること。"""
        fs = _build_sample_feature_set()
        with self.assertRaises(Exception):
            fs.feature_schema_version = 2  # type: ignore[misc]

    def test_missing_keys_records_unmeasurable_features(self) -> None:
        """missing_keysで『計算不能』と『未計算』を区別できること（設計書③3.3）。"""
        fs = FeatureSet(
            eval_id="20260713_12_05",
            feature_schema_version=1,
            built_at="2026-07-13T06:00:00+09:00",
            boat_features={1: {"win_rate": 6.50}},
            race_features={},
            missing_keys=("course_st_1c", "ability_trend"),
        )
        self.assertIn("course_st_1c", fs.missing_keys)
        self.assertEqual(len(fs.missing_keys), 2)

    def test_to_dict_and_from_dict_roundtrip(self) -> None:
        """to_dict -> from_dict の往復で同値のFeatureSetが復元できること。

        JSON化そのもの（json.dumps等）はstorage層の責務のため、ここではdict構造への
        変換・復元のみを検証する（設計書⑤責務分離: models層はJSONを意識しない）。
        """
        fs = _build_sample_feature_set()
        restored = FeatureSet.from_dict(fs.to_dict())
        self.assertEqual(fs, restored)

    def test_to_dict_converts_boat_features_keys_to_str(self) -> None:
        """to_dict()はboat_featuresの艇番キー(int)をJSON互換のstr型に変換すること。"""
        fs = _build_sample_feature_set()
        d = fs.to_dict()
        self.assertIn("1", d["boat_features"])
        self.assertNotIn(1, d["boat_features"])

    def test_from_dict_converts_boat_features_keys_back_to_int(self) -> None:
        """from_dict()はboat_featuresのキーをstrからintへ復元すること。"""
        fs = _build_sample_feature_set()
        d = fs.to_dict()
        restored = FeatureSet.from_dict(d)
        self.assertIn(1, restored.boat_features)

    def test_to_dict_with_local_features_none(self) -> None:
        """local_featuresがNoneの場合、to_dict()結果でもNoneのまま保持されること。"""
        fs = _build_sample_feature_set()
        d = fs.to_dict()
        self.assertIsNone(d["local_features"])
        restored = FeatureSet.from_dict(d)
        self.assertIsNone(restored.local_features)

    def test_to_dict_with_local_features_present(self) -> None:
        """local_featuresが設定されている場合、往復変換でも値が保持されること。"""
        fs = FeatureSet(
            eval_id="20260713_12_05",
            feature_schema_version=1,
            built_at="2026-07-13T06:00:00+09:00",
            boat_features={1: {"win_rate": 6.50}},
            race_features={},
            local_features={1: {"local_1st_rate_diff": 0.05}},
        )
        restored = FeatureSet.from_dict(fs.to_dict())
        self.assertEqual(restored.local_features[1]["local_1st_rate_diff"], 0.05)


class TestRaceEvaluation(unittest.TestCase):
    def test_construct_with_feature_set_type(self) -> None:
        """featuresがFeatureSet型そのものとして保持されること（案A採用の確認）。"""
        evaluation = _build_sample_evaluation()
        self.assertIsInstance(evaluation.features, FeatureSet)
        self.assertEqual(evaluation.features.feature_schema_version, 1)

    def test_immutable(self) -> None:
        """frozen dataclassのため属性再代入がFrozenInstanceErrorになること。"""
        evaluation = _build_sample_evaluation()
        with self.assertRaises(Exception):
            evaluation.danger_score = 0.0  # type: ignore[misc]

    def test_optional_fields_default_to_none(self) -> None:
        """hot_motor_score・awakening_score・local_advantageは任意項目であり、
        未指定時はNoneであること。
        """
        evaluation = _build_sample_evaluation()
        self.assertIsNone(evaluation.hot_motor_score)
        self.assertIsNone(evaluation.awakening_score)
        self.assertIsNone(evaluation.local_advantage)

    def test_engine_name_and_version_are_recorded(self) -> None:
        """engine_name・engine_versionが記録されること
        （設計書⑤5.2 EvaluationEngine: エンジン世代差替え時も再現性を保証するため）。
        """
        evaluation = _build_sample_evaluation()
        self.assertEqual(evaluation.engine_name, "ver4")
        self.assertEqual(evaluation.engine_version, "4.2.0")

    def test_match_index_is_required_and_held_as_evaluation_result(self) -> None:
        """match_indexが必須フィールドとして保持されること（設計書③3.4 v1.1.1）。

        match_indexは評価時に導出される評価結果であり、Mapper/Serializer/
        Repository/表示層で再計算されない契約を持つ。models層では
        「値を保持できること」と「必須であること」のみを検証する。
        """
        evaluation = _build_sample_evaluation()
        self.assertEqual(evaluation.match_index, 44.1)
        field_names = {f for f in evaluation.__dataclass_fields__}
        self.assertIn("match_index", field_names)

    def test_optional_evaluation_fields_accept_explicit_none(self) -> None:
        """v1.1.4のOptional契約: 7フィールドに明示的なNone（当時未記録）を
        渡して構築できること。既定値は持たないため省略は不可（TypeError）。
        """
        evaluation = RaceEvaluation(
            eval_id="20260704_01_01",
            race_date="20260704",
            venue_num=1,
            venue_name="桐生",
            race_number=1,
            is_night=False,
            engine_name="legacy",
            engine_version="0.0.0",
            feature_schema_version=0,
            model_version=None,
            evaluated_at="",
            danger_score=None,
            danger_breakdown=None,
            upset_score=9.5,
            upset_reasons=(),
            rank_index=None,
            featured_boats=None,
            win_probs=None,
            race_type="1殴り荒れ型",
            match_index=None,
            features=_build_sample_feature_set(),
        )
        self.assertIsNone(evaluation.danger_score)
        self.assertIsNone(evaluation.model_version)
        self.assertIsNone(evaluation.match_index)
        # 省略（引数不足）はTypeErrorになること
        with self.assertRaises(TypeError):
            RaceEvaluation(  # type: ignore[call-arg]
                eval_id="20260704_01_01",
                race_date="20260704",
                venue_num=1,
                venue_name="桐生",
                race_number=1,
                is_night=False,
                engine_name="legacy",
                engine_version="0.0.0",
                feature_schema_version=0,
                evaluated_at="",
                upset_score=9.5,
                upset_reasons=(),
                race_type="1殴り荒れ型",
                features=_build_sample_feature_set(),
            )


class TestPrediction(unittest.TestCase):
    def test_construct(self) -> None:
        """必須項目でPredictionが構築できること。"""
        prediction = Prediction(
            eval_id="20260713_12_05",
            pred_combo="1-2-3",
            pred_prob=0.185,
            pred_ev=1.42,
            pred_odds=7.7,
            confidence=0.72,
            why_bet="1号艇信頼度高",
            patterns=({"combo": "1-2-3", "prob": 0.185},),
        )
        self.assertEqual(prediction.pred_combo, "1-2-3")
        self.assertEqual(len(prediction.patterns), 1)

    def test_immutable(self) -> None:
        """frozen dataclassのため属性再代入がFrozenInstanceErrorになること。"""
        prediction = Prediction(
            eval_id="20260713_12_05",
            pred_combo="1-2-3",
            pred_prob=0.185,
            pred_ev=1.42,
            pred_odds=7.7,
            confidence=0.72,
            why_bet="1号艇信頼度高",
            patterns=(),
        )
        with self.assertRaises(Exception):
            prediction.pred_combo = "1-3-2"  # type: ignore[misc]


class TestBuyDecision(unittest.TestCase):
    def test_construct_purchased(self) -> None:
        """購入判定=Trueの場合、skip_reasonは省略できること（既定None）。"""
        decision = BuyDecision(
            eval_id="20260713_12_05",
            purchased=True,
            buyscore=68.5,
            investment_type="通常",
            n_bets=3,
            cost=900,
            kelly_fraction=0.05,
            config_version="12",
        )
        self.assertTrue(decision.purchased)
        self.assertIsNone(decision.skip_reason)

    def test_construct_skipped_with_reason(self) -> None:
        """見送り時にskip_reasonを設定できること（設計書③3.6: 見送り時必須）。

        必須であることの強制（バリデーション）自体はStep1のmodels層の責務外とし、
        ここでは値を保持できることのみを確認する。
        """
        decision = BuyDecision(
            eval_id="20260713_12_05",
            purchased=False,
            buyscore=12.0,
            investment_type="見送り",
            n_bets=0,
            cost=0,
            kelly_fraction=0.0,
            config_version="12",
            skip_reason="danger_score閾値未達",
        )
        self.assertFalse(decision.purchased)
        self.assertEqual(decision.skip_reason, "danger_score閾値未達")

    def test_immutable(self) -> None:
        """frozen dataclassのため属性再代入がFrozenInstanceErrorになること。"""
        decision = BuyDecision(
            eval_id="20260713_12_05",
            purchased=True,
            buyscore=68.5,
            investment_type="通常",
            n_bets=3,
            cost=900,
            kelly_fraction=0.05,
            config_version="12",
        )
        with self.assertRaises(Exception):
            decision.purchased = False  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
