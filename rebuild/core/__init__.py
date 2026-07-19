"""
core層（L2）: AI評価エンジンの純粋計算（設計書 v1.1.8 ⑤・⑩）。

依存はmodels＋注入されたconfig/provider/predictorのみ。
storage / output / services / pipelines への依存は禁止。

Step4-2: danger.py（calc_danger_score: 1号艇危険度の純関数移植）
Step4-3: upset.py（calculate_upset_score / calc_boat_score: 荒れ確率の純関数移植）
Step4-4: rank.py（calc_rank_index / select_featured_boats / compute_match_index）
Step4-5: engine.py（Ver4Engine: v4統合・RaceEvaluation/Prediction生成）
"""

from core.danger import CourseFactorProvider, calc_danger_score
from core.engine import (
    ENGINE_NAME,
    ENGINE_VERSION,
    EvaluationEngine,
    Ver4Engine,
)
from core.rank import (
    calc_lane_rank_scores,
    calc_rank_index,
    calc_rank_probabilities,
    compute_match_index,
    select_featured_boats,
)
from core.upset import UpsetResult, calc_boat_score, calculate_upset_score

__all__ = [
    "CourseFactorProvider",
    "ENGINE_NAME",
    "ENGINE_VERSION",
    "EvaluationEngine",
    "Ver4Engine",
    "UpsetResult",
    "calc_boat_score",
    "calc_danger_score",
    "calc_lane_rank_scores",
    "calc_rank_index",
    "calc_rank_probabilities",
    "calculate_upset_score",
    "compute_match_index",
    "select_featured_boats",
]
