"""
Ver4評価エンジン統合（L2 Core）: build_race_evaluation_v4 相当の統合と
RaceEvaluation / Prediction の生成。

移植元（x_asahi_scoring.py、freeze-v1-baseline）:
  - build_race_evaluation_v4          L940-L1028 → Ver4Engine.build_legacy_evaluation
  - upset内の統合ブロック（rank_index_ctx・_features組み立て） L536-L590
    → Ver4Engine内で再現（Danger/Upset/Rank/Featuredの各移植関数は無変更で呼ぶだけ）
基準コミット: 3a7f9c3dd7c628255285aefbe5b3e03978ec93b3

Feature Freeze厳守（C2）:
  venue未指定時のwater_type/venue_factor_1cの既定dict、boat1欠損時の""、
  _featuresの空文字センチネル（""＝未算出）、Noneゲート、丸め、
  rank_index_ctx（match_index式）、detailへの_features注入まで
  v4のdict出力をバイト等価（JSON正規化後の完全一致）で再現する。

設計注記（⑤5.3との関係）:
  ③3.3は「coreはRaceEntry生値を直接読まない」とするが、Ver4のFreeze再現には
  レガシーBoatInfoの17属性（FeatureSet未収載のlocal_win/course_place_counts等）が
  不可欠である。そのためVer4Engineは、Step4-1のFeatureInputsと同形式の
  「Ver4互換boats素材（属性Mapping列）」をBoatsResolverとしてDI注入で受け取る。
  ⑤5.3の禁止はVer5/MLHybrid以降の新エンジンに適用し、Ver4はGolden 100%一致を
  正当性根拠とする例外とする（設計書への追記はStep5で提案する）。

責務・依存（⑩）:
  - 依存はmodels＋標準ライブラリ＋core内部のみ。storage層依存禁止
  - ファイル・API・時刻へ直接アクセスしない（config/now/providerは注入）
  - Danger / Upset / Rank / Featured / FeatureBuilder のコードは変更しない
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from core.danger import Boat
from core.rank import calc_rank_index, compute_match_index, select_featured_boats
from core.upset import calculate_upset_score
from models.evaluation import FeatureSet, Prediction, RaceEvaluation
from models.race import OddsSnapshot, Race, Weather

ENGINE_NAME: str = "ver4"
ENGINE_VERSION: str = "4.0.0"  # rebuild側エンジンのsemver（設計書⑤5.2）


class VenueStatsProvider(Protocol):
    """venue統計の供給者（features.VenueStatsProviderと同形・構造的Protocol）。"""

    def classify_water_type(self, venue_name: str) -> Mapping[str, Any]: ...

    def get_venue_course_factor(
        self, venue_name: str, course: int
    ) -> Mapping[str, Any]: ...


# eval_id -> {艇番: 勝率}。C5: Step4ではFakePredictorで固定注入する
Predictor = Callable[[str], dict[int, float]]

# boats素材の解決（Race -> Ver4互換のBoatInfo属性Mapping列）。
# 実データ結線（BoatInfo/fanファイル由来）はStep5のpipelines責務。
BoatsResolver = Callable[[Race], Sequence[Boat]]

# Prediction構築戦略。買い目生成（pred_combo/EV算出）のレガシーは
# notify_arashi側にあり本Stepの移植対象外のため、DI境界として定義する。
PredictionStrategy = Callable[
    [RaceEvaluation, Optional[OddsSnapshot], dict], Prediction
]

# ── v4のvenue未指定時デフォルト（L997-L1005 機械転記） ──
_NO_VENUE_WATER_TYPE: dict[str, Any] = {
    "type": "unknown", "label": "不明", "course1_win_rate": 0.0,
    "source": "no_venue", "samples": 0,
}
_NO_VENUE_FACTOR_1C: dict[str, Any] = {
    "factor": 1.0, "venue_win_rate": 0.0, "national_win_rate": 0.0,
    "samples": 0, "water_type": "unknown",
}


class EvaluationEngine(Protocol):
    """AI評価エンジンの抽象（設計書⑤5.2）。"""

    engine_name: str
    engine_version: str

    def evaluate(
        self,
        race: Race,
        feature_set: FeatureSet,
        weather: Optional[Weather],
        config: dict,
        now: datetime,
    ) -> RaceEvaluation: ...

    def predict(
        self,
        evaluation: RaceEvaluation,
        odds: Optional[OddsSnapshot],
        config: dict,
    ) -> Prediction: ...


class Ver4Engine:
    """Ver4統合エンジン（build_race_evaluation_v4 の忠実再現＋モデル化）。"""

    engine_name: str = ENGINE_NAME
    engine_version: str = ENGINE_VERSION

    def __init__(
        self,
        boats_resolver: BoatsResolver,
        venue_stats: Optional[VenueStatsProvider] = None,
        predictor: Optional[Predictor] = None,
        prediction_strategy: Optional[PredictionStrategy] = None,
    ) -> None:
        self._boats_resolver = boats_resolver
        self._venue_stats = venue_stats
        self._predictor = predictor
        self._prediction_strategy = prediction_strategy

    # ================= v4 dict（レガシー完全再現） =================

    def build_legacy_evaluation(
        self,
        boats: Sequence[Boat],
        config: dict,
        race_grade: int = 0,
        venue_num: int = 0,
        is_night: bool = False,
    ) -> dict[str, Any]:
        """build_race_evaluation_v4（L940-L1028）と同一のdictを返す。"""
        boat1 = next((b for b in boats if b.get("lane") == 1), None)

        # upsetコア（Step4-3移植）＋統合ブロック（L536-L538）の再現
        upset = calculate_upset_score(
            boats,
            config,
            race_grade=race_grade,
            venue_num=venue_num,
            is_night=is_night,
            venue_stats=self._venue_stats,
        )
        venue_name = upset.venue_name
        rank_index_ctx = {
            "match_index": compute_match_index(upset.upset_score),
            "upset_score": upset.upset_score,
        }
        rank_index = calc_rank_index(
            boats, rank_index_ctx, config,
            venue_name=venue_name, venue_stats=self._venue_stats,
        )
        featured_boats = select_featured_boats(
            boats, rank_index, upset.danger_score, config
        )

        # _features組み立て（L541-L590の再現。空文字""＝未算出センチネルを保存）
        venue_water_type: Any = ""
        venue_factor_val: Any = ""
        course_sample_confidence: Any = ""
        if venue_name and self._venue_stats:
            wt = self._venue_stats.classify_water_type(venue_name)
            venue_water_type = wt["label"]
            vf = self._venue_stats.get_venue_course_factor(venue_name, 1)
            venue_factor_val = vf["factor"]
            course_sample_confidence = vf["samples"]

        ability_trend_1c: Any = ""
        if boat1 and boat1.get("ability_curr") and boat1.get("ability_prev"):
            ability_trend_1c = round(
                boat1["ability_curr"] - boat1["ability_prev"], 2
            )

        course_f_rate_1c: Any = ""
        course_l_rate_1c: Any = ""
        course_rentai2_1c: Any = ""
        nyuko = boat1.get("course_nyuko") if boat1 else None
        if boat1 and nyuko and nyuko[0] > 0:
            nyuko0 = nyuko[0]
            if boat1.get("course_f_count"):
                course_f_rate_1c = round(
                    boat1["course_f_count"][0] / nyuko0 * 100, 1
                )
            if boat1.get("course_l_count"):
                course_l_rate_1c = round(
                    boat1["course_l_count"][0] / nyuko0 * 100, 1
                )
            place_counts = boat1.get("course_place_counts")
            if place_counts and sum(place_counts[0]) > 0:
                counts0 = place_counts[0]
                course_rentai2_1c = round(
                    (counts0[0] + counts0[1]) / nyuko0 * 100, 1
                )

        in_lane1 = bool(boat1) and bool(nyuko) and nyuko[0] > 0
        features: dict[str, Any] = {
            "win_rate": boat1.get("win_rate") if boat1 else None,
            "motor": boat1.get("motor") if boat1 else None,
            "avg_st": boat1.get("avg_st") if boat1 else None,
            "racer_class": boat1.get("racer_class") if boat1 else None,
            "course_st_1c": boat1["course_st"][0] if in_lane1 else None,
            "course_rank_1c": boat1["course_rank"][0] if in_lane1 else None,
            "danger_breakdown": upset.danger_breakdown,
            "danger_score_v3": upset.danger_score,
            "rank_index": rank_index,
            "featured_boats": featured_boats,
            "model_version": config.get("model_version", "unknown"),
            "venue_water_type": venue_water_type,
            "venue_factor": venue_factor_val,
            "ability_trend": ability_trend_1c,
            "course_f_rate_1c": course_f_rate_1c,
            "course_l_rate_1c": course_l_rate_1c,
            "course_rentai2_1c": course_rentai2_1c,
            "course_sample_confidence": course_sample_confidence,
        }
        upset_detail = dict(upset.detail)
        upset_detail["_features"] = features

        # v4トップレベル（L989-L1027の再現）
        if venue_name and self._venue_stats:
            water_type = self._venue_stats.classify_water_type(venue_name)
            venue_factor_1c = self._venue_stats.get_venue_course_factor(venue_name, 1)
        else:
            water_type = dict(_NO_VENUE_WATER_TYPE)
            venue_factor_1c = dict(_NO_VENUE_FACTOR_1C)

        return {
            "model_version": config.get("model_version", "unknown"),
            "venue": venue_name or "",
            "venue_num": venue_num,
            "boat1": {
                "lane": 1,
                "name": boat1.get("name") if boat1 else "",
                "racer_class": boat1.get("racer_class") if boat1 else "",
            },
            "danger_score": features.get("danger_score_v3", 0.0),
            "danger_breakdown": features.get("danger_breakdown", {}),
            "water_type": water_type,
            "venue_factor_1c": venue_factor_1c,
            "rank_index": features.get("rank_index", {}),
            "featured_boats": features.get("featured_boats", []),
            "upset_score": upset.upset_score,
            "upset_detail": upset_detail,
            "boat1_features": {
                "ability_trend": features.get("ability_trend"),
                "course_f_rate_1c": features.get("course_f_rate_1c"),
                "course_l_rate_1c": features.get("course_l_rate_1c"),
                "course_rentai2_1c": features.get("course_rentai2_1c"),
                "course_sample_confidence": features.get("course_sample_confidence"),
            },
        }

    # ================= RaceEvaluation（モデル化） =================

    def evaluate(
        self,
        race: Race,
        feature_set: FeatureSet,
        weather: Optional[Weather],
        config: dict,
        now: datetime,
    ) -> RaceEvaluation:
        """v4評価をRaceEvaluationモデルへ写像する（設計書③3.4）。

        weatherは「朝刊のみ」ポリシー（⑨資産17）により評価へ使用しない
        （シグネチャ互換のため受け取るのみ）。
        """
        boats = self._boats_resolver(race)
        legacy = self.build_legacy_evaluation(
            boats,
            config,
            race_grade=_grade_to_number(race.grade),
            venue_num=race.venue_num,
            is_night=race.is_night,
        )
        win_probs = self._predictor(race.eval_id) if self._predictor else None
        return RaceEvaluation(
            eval_id=race.eval_id,
            race_date=race.race_date,
            venue_num=race.venue_num,
            venue_name=race.venue_name,
            race_number=race.race_number,
            is_night=race.is_night,
            engine_name=self.engine_name,
            engine_version=self.engine_version,
            feature_schema_version=feature_set.feature_schema_version,
            features=feature_set,
            model_version=legacy["model_version"],
            evaluated_at=now.isoformat(),
            danger_score=legacy["danger_score"],
            danger_breakdown=legacy["danger_breakdown"],
            upset_score=legacy["upset_score"],
            # v4評価基盤に理由文の生成は存在しない（notify系のreasonsはL5/
            # 買い目フロー由来でStep5移植対象）ため空タプル
            upset_reasons=(),
            # 正準形: rank_indexはstr laneキー（Serializerフィクスチャ準拠）
            rank_index={str(lane): v for lane, v in legacy["rank_index"].items()},
            # 正準形: featured_boatsは{"featured": [...]}ラップ（同上）
            featured_boats={"featured": legacy["featured_boats"]},
            win_probs=win_probs,
            # race_type分類（notify_arashi L1742 classify系）はGolden対象外の
            # ため本Stepでは未分類""とする（接続はStep4-6/5で承認のうえ実施）
            race_type="",
            match_index=compute_match_index(legacy["upset_score"]),
        )

    # ================= Prediction =================

    def predict(
        self,
        evaluation: RaceEvaluation,
        odds: Optional[OddsSnapshot],
        config: dict,
    ) -> Prediction:
        """Prediction構築（設計書③3.5・⑤5.2）。

        買い目生成の実体（pred_combo候補列挙・確率/EV算出）のレガシーは
        notify_arashi側の買い目フローにあり、x_asahi_scoring（Freeze対象の
        評価基盤）には存在しない。推測実装の禁止（実装ルール2）に従い、
        本Stepでは実体を発明せずPredictionStrategyのDI境界のみ確定する。
        戦略未注入で呼ばれた場合は明示的に失敗する（サイレント失敗禁止⑫）。
        """
        if self._prediction_strategy is None:
            raise NotImplementedError(
                "PredictionStrategy is not wired yet (planned: Step4-6). "
                "Inject prediction_strategy to Ver4Engine."
            )
        return self._prediction_strategy(evaluation, odds, config)


def _grade_to_number(grade: Optional[str]) -> int:
    """Race.grade（SG/G1/G2/G3/一般/None/"gradeN"）→ v4のrace_grade番号。

    対応は移植元GRADE_EFFECTS/_GRADE_NAMES（0:一般,1:G3,2:G2,3:G1,4:SG）の逆引き。
    未知・Noneは0（一般）＝レガシーの既定値と同一。
    ただし race_grade_number は BoatraceOpenAPI 生値であり0-4に限らない
    （レガシーも未知値を検証せず.get(n, default)でフォールバックするのみ）。
    そのため "gradeN" 形式（Race.grade表現側の未知値エンコード）は
    Nへ復元し、5番以降のグレードでも upset/rank 計算へ正しく渡す。
    """
    mapping = {"一般": 0, "G3": 1, "G2": 2, "G1": 3, "SG": 4}
    if grade in mapping:
        return mapping[grade]
    if grade and grade.startswith("grade"):
        suffix = grade[len("grade"):]
        if suffix.isdigit():
            return int(suffix)
    return 0
