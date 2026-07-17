"""
Feature / Evaluation系データモデル。

設計書 Phase0.5 v1.1 ③3.3（FeatureSet）・3.4（RaceEvaluation）・
3.5（Prediction）・3.6（BuyDecision）に対応する。

依存: 標準ライブラリのみ（dataclasses, typing）。
JSONファイルの読み書き・永続化・Repository処理はStep1の対象外とし、
本モジュールはメモリ上のドメインモデルとしての責務のみを持つ
（設計書⑤5.1 coreの担当範囲、⑩ 依存ルール）。
to_dict/from_dictはJSON互換のdict構造への変換補助のみを提供し、
実際のファイルI/Oは後続レイヤー（storage/）の責務とする。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class FeatureSet:
    """
    Design Spec: §3.3 FeatureSet

    AIエンジン（core/）への唯一の入力。
    coreはRaceEntryの生値を直接読んで独自加工してはならず、
    特徴量はすべてこのモデルを経由する（設計書⑤5.3の禁止事項）。

    boat_features のキー集合は RaceEvaluation.features（§3.4）の
    保存キーと1:1対応する。

    保存方法について: FeatureSet自体は保存しない
    （設計書③3.3の理由: 全値がRaceEvaluation.featuresとして
    評価結果側に記録され再現可能なため、二重保存による乖離事故を防ぐ）。
    feature_schema_versionのみRaceEvaluationへ転記する。
    """

    eval_id: str
    feature_schema_version: int
    built_at: str  # ISO8601 JST
    boat_features: dict[int, dict[str, float]]  # 艇番(1-6) -> 特徴量名 -> 値
    race_features: dict[str, float]  # レース単位の特徴量
    local_features: Optional[dict[int, dict[str, float]]] = None  # 当地コース特徴量（Phase3接続まではNone）
    missing_keys: tuple[str, ...] = field(default_factory=tuple)  # 計算不能だった特徴量名

    def to_dict(self) -> dict[str, Any]:
        """JSON互換のdict構造へ変換する（シリアライズ補助のみ。ファイルI/Oは行わない）。"""
        return {
            "eval_id": self.eval_id,
            "feature_schema_version": self.feature_schema_version,
            "built_at": self.built_at,
            "boat_features": {str(k): v for k, v in self.boat_features.items()},
            "race_features": dict(self.race_features),
            "local_features": (
                {str(k): v for k, v in self.local_features.items()}
                if self.local_features is not None
                else None
            ),
            "missing_keys": list(self.missing_keys),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureSet":
        """dict構造からFeatureSetを復元する（デシリアライズ補助のみ）。"""
        local_features = data.get("local_features")
        return cls(
            eval_id=data["eval_id"],
            feature_schema_version=data["feature_schema_version"],
            built_at=data["built_at"],
            boat_features={int(k): v for k, v in data["boat_features"].items()},
            race_features=dict(data["race_features"]),
            local_features=(
                {int(k): v for k, v in local_features.items()}
                if local_features is not None
                else None
            ),
            missing_keys=tuple(data.get("missing_keys", [])),
        )


@dataclass(frozen=True)
class RaceEvaluation:
    """
    Design Spec: §3.4 RaceEvaluation

    評価結果。システムの心臓（設計書①1.2「1回の評価、無限の成果物」）。
    評価したら必ず1レース=1レコードで保存する契約を持つ（保存自体はstorage層の責務）。
    項目の欠損はnullで表現し「計算不能」と「未計算」を区別する
    （本dataclassではOptional[...]=Noneとして表現する）。

    features は FeatureSet 型そのものを保持する（メモリ上のドメインモデルとしての型）。
    JSON保存時のシリアライズ（features -> dict）はstorage層（後続Step）の責務とし、
    models層はJSON形式を意識しない。

    Optional契約（v1.1.4）: danger_score / danger_breakdown / rank_index /
    featured_boats / win_probs / match_index / model_version のOptionalは、
    レガシーデータとの互換性および未計算状態（None=当時未記録）を表現するためである。
    Ver4以降の新規評価では、EvaluationEngineが必要な項目を設定する責務を持つ
    （新規評価での省略を許可するものではない）。Optional型でも既定値は持たせず、
    呼び出し側は明示的にNoneを渡す。
    """

    eval_id: str
    race_date: str
    venue_num: int
    venue_name: str
    race_number: int
    is_night: bool
    engine_name: str  # "ver4" / "ver5" / "ml_hybrid"
    engine_version: str  # semver 例 "4.2.0"
    feature_schema_version: int
    model_version: Optional[str]  # None=当時未記録（レガシー互換）
    evaluated_at: str  # ISO8601 JST
    danger_score: Optional[float]  # None=当時未記録（レガシー互換・③3.4のnull表現）
    danger_breakdown: Optional[dict[str, Any]]
    upset_score: float
    upset_reasons: tuple[str, ...]
    rank_index: Optional[dict[str, Any]]
    featured_boats: Optional[dict[str, Any]]
    win_probs: Optional[dict[int, float]]  # 艇番 -> 勝率
    race_type: str  # 本命/中穴/大穴
    match_index: Optional[float]  # 評価時に導出されるAI一致指数（設計書③3.4 v1.1.1追加）。
    # match_index is a derived evaluation metric. It is produced during evaluation
    # and stored as part of RaceEvaluation. It is not recalculated by Mapper,
    # Serializer, Repository, or presentation layers.
    features: FeatureSet
    hot_motor_score: Optional[float] = None
    awakening_score: Optional[float] = None
    local_advantage: Optional[dict[str, Any]] = None  # 当地コース特徴量（Phase3で接続）


@dataclass(frozen=True)
class Prediction:
    """
    Design Spec: §3.5 Prediction

    買い目予想。RaceEvaluationに対して1件対応する。
    """

    eval_id: str
    pred_combo: str  # 例 "1-2-3"
    pred_prob: float
    pred_ev: float
    pred_odds: float
    confidence: float
    why_bet: str
    patterns: tuple[dict[str, Any], ...]  # 全候補（コンボ生成器出力）。
    # 前提条件: dict内の値はJSONプリミティブ（str/int/float/bool/None/list/dict）
    # のみとする（CSV/JSON往復同一性の成立条件）


@dataclass(frozen=True)
class BuyDecision:
    """
    Design Spec: §3.6 BuyDecision

    購入判定。core/buyscore（BuyEngine実装）が出力する。
    """

    eval_id: str
    purchased: bool
    buyscore: Optional[float]  # None=当時未記録（BuyScoreエンジン導入前のレガシー行）。
    # v1.1.5 Optional契約: 新規判定ではBuyEngineが必ず設定する（省略許可ではない）
    investment_type: str  # 転がし/一発/通常/見送り
    n_bets: int
    cost: int
    kelly_fraction: Optional[float]  # None=当時未記録（Kelly概念導入前）。
    # v1.1.6 Optional契約: 「未計算」と「計算結果0.0」を区別するため。
    # 新規判定ではBuyEngineが必ず設定する（省略許可ではない）
    config_version: str  # buyscore_config の _version
    skip_reason: Optional[str] = None  # 見送り時必須（呼び出し側で保証する）
