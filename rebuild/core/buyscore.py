"""
BuyEngine（L2 Core・純関数）: 買い目候補のBuyScore・投資タイプ・Kelly・見送り判定。

移植元（x_buyscore.py、freeze-v1-baseline）:
  - _odds_band_bonus     L125-L133 → _odds_band_bonus
  - _race_type_bonus     L135-L140 → _race_type_bonus
  - calc_buyscore        L142-L208 → calc_buyscore
  - check_passthrough    L214-L268 → check_passthrough（skip_reason生成）
  - investment_type      L281-L288 → investment_type
  - kelly_fraction       L303-L340 → kelly_fraction
基準コミット: 3a7f9c3dd7c628255285aefbe5b3e03978ec93b3
buyscore_config freeze SHA-256:
  da6a4edaf8220aa52ed3e4577c23ff443509512a971d0ba322079eb4c4cd6f1d

Feature Freeze厳守（Step4-6指示）:
  if条件・閾値・round桁数・max/min・clamp・.getデフォルト（0.0がFalseになる
  挙動を含む）・None判定・各算出順をすべて完全再現する。
  改善・最適化・リファクタリングは禁止。

スコープ（Step4-6指示）:
  資金管理（get_bet_multiplier_extended等の外部状態依存）・purchased・
  cost・n_bets・ドローダウン管理はStep5以降の責務であり実装しない。
  stars/risk_level/korogashi/ippatsu/rank_labels/format_*も本Stepの
  実装対象外（BuyScore・投資タイプ・Kelly・見送り判定のみ）。

責務・依存（⑤5.3・⑩）:
  - 純関数のみ。config読込・ファイルI/O・時刻取得・APIアクセスなし
    （configは呼び出し側がfreeze configをdictで渡す）
  - 依存はmodels＋標準ライブラリ＋core内部のみ。storage層依存禁止
  - Kelly計算はKellyStrategy Protocolで注入可能（既定はDefaultKellyStrategy）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence

from core.exceptions import ValidationError
from models.evaluation import Prediction, RaceEvaluation


# ==================== 純関数（移植元の忠実再現） ====================


def _odds_band_bonus(odds: float, band_cfg: Mapping[str, float]) -> float:
    """オッズ帯補正値（移植元L125-L133）。境界は<=判定。"""
    if odds <= 5:
        return band_cfg.get("1_5", 0.0)
    if odds <= 12:
        return band_cfg.get("6_12", 0.0)
    if odds <= 25:
        return band_cfg.get("13_25", 0.0)
    if odds <= 40:
        return band_cfg.get("26_40", 0.0)
    if odds <= 80:
        return band_cfg.get("41_80", 0.0)
    return band_cfg.get("80_over", 0.0)


def _race_type_bonus(race_type: str) -> float:
    """レースタイプ補正の正規化値（移植元L135-L140）。未知タイプは0.0。"""
    bonus_map = {"本命戦": 0.05, "混戦": 0.0, "超混戦": 0.03, "荒れ戦": 0.05}
    return bonus_map.get(race_type, 0.0)


def calc_buyscore(
    candidate: Mapping[str, Any],
    context: Mapping[str, Any],
    cfg: dict,
) -> float:
    """BuyScore 0〜100（移植元L142-L208）。

    正規化分母（EV2.0満点・prob0.08満点）、.getデフォルト
    （uncertainty=0.5等）、負の重み、展示未取得-0.08、clamp、round(,1)を
    すべて移植元と同一順序で再現する。
    """
    if "weights" not in cfg or "thresholds" not in cfg:
        raise ValidationError(
            "buyscore config missing required section: "
            f"weights={'weights' in cfg} thresholds={'thresholds' in cfg}"
        )
    w = cfg["weights"]
    thr = cfg["thresholds"]

    ev = candidate.get("ev", 0.0)
    prob = candidate.get("prob", 0.0)
    odds = candidate.get("odds", 0.0)
    uncertainty = candidate.get("uncertainty", 0.5)
    disagreement = candidate.get("disagreement", 0.5)
    composite = candidate.get("composite", 0.0)

    match_index = context.get("match_index", 0.0)
    race_type = context.get("race_type", "混戦")
    market_gap = context.get("market_gap", 0.0)

    # ── 各要素を 0.0〜1.0 に正規化（分母は移植元コメントの根拠のまま） ──
    ev_n = min(ev / 2.0, 1.0)
    prob_n = min(prob / 0.08, 1.0)
    match_n = min(match_index / 100.0, 1.0)
    market_gap_n = min(abs(market_gap) / thr.get("market_gap_max", 3.0), 1.0)
    uncertainty_n = min(uncertainty, 1.0)
    disagree_n = min(disagreement, 1.0)
    calib_n = composite  # すでに 0〜1

    # ── 重み付き合算（順序・既定値とも移植元と同一） ──
    raw = (
        ev_n * w.get("ev", 0.35)
        + prob_n * w.get("prob", 0.20)
        + match_n * w.get("match_index", 0.15)
        + market_gap_n * w.get("market_gap", 0.10)
        + uncertainty_n * w.get("uncertainty", -0.10)
        + disagree_n * w.get("disagreement", -0.05)
        + calib_n * w.get("calibration", 0.10)
    )

    # ── レースタイプ補正（重みrace_typeを通して1回だけ加算） ──
    raw += w.get("race_type", 0.05) * _race_type_bonus(race_type)

    # ── オッズ帯補正 ──
    raw += _odds_band_bonus(odds, cfg.get("odds_band_bonus", {}))

    # ── 展示未取得ペナルティ ──
    if not context.get("has_exhibition", True):
        raw -= 0.08

    # 0〜100 にスケール・クランプ
    score = max(0.0, min(raw * 100.0, 100.0))
    return round(score, 1)


def check_passthrough(
    candidates: Sequence[Mapping[str, Any]],
    context: Mapping[str, Any],
    cfg: dict,
) -> Optional[str]:
    """見送りなら理由文字列、買うならNone（移植元L214-L268）。

    判定順（BuyScore不足→期待値不足→AI一致指数不足→不確実性高→
    展示なし+信頼度不足→モデル不一致）と各f-string書式を完全再現する。
    """
    if "thresholds" not in cfg:
        raise ValidationError("buyscore config missing required section: thresholds")
    thr = cfg["thresholds"]

    if not candidates:
        return "候補なし"

    best = candidates[0]
    best_score = best.get("buyscore", 0.0)
    ev = best.get("ev", 0.0)
    uncertainty = best.get("uncertainty", 1.0)
    disagreement = best.get("disagreement", 1.0)
    match_index = context.get("match_index", 0.0)
    has_exhibition = context.get("has_exhibition", True)

    if best_score < thr.get("buyscore_min", 60):
        return f"BuyScore不足({best_score:.0f})"

    if ev < thr.get("ev_min", 1.28):
        return f"期待値不足(EV{ev:.2f})"

    # match_indexが近似値の場合は単独では見送りにしない（移植元と同一）
    if match_index < thr.get("match_index_min", 40):
        if not context.get("match_index_approx", False):
            return f"AI一致指数不足({match_index:.0f})"

    if uncertainty > thr.get("uncertainty_max", 0.75):
        return f"不確実性高({uncertainty:.2f})"

    composite = best.get("composite", 0.0)
    if not has_exhibition and composite < 0.40:
        return f"展示なし+信頼度不足({composite:.2f})"

    if disagreement > thr.get("disagreement_max", 0.80):
        return f"モデル不一致({disagreement:.2f})"

    return None


def investment_type(buyscore: float, ev: float, odds: float, race_type: str) -> str:
    """投資タイプ分類（移植元L281-L288）。判定順を完全再現。"""
    if buyscore < 60:
        return "見送り"
    if race_type == "本命戦":
        return "堅実"
    if ev >= 2.0 and odds >= 30:
        return "穴狙い"
    if ev >= 1.5:
        return "期待値重視"
    return "堅実"


def kelly_fraction(
    prob: float,
    odds: float,
    buyscore: float,
    cfg: dict,
) -> float:
    """BuyScore反映Kelly係数（移植元L303-L340）。

    zero_below/half_below閾値・b<=0で0.0・raw_kelly<=0で0.0・
    fraction縮小・round(,4)を完全再現する。
    """
    kelly_cfg = cfg.get("kelly", {})
    zero_below = kelly_cfg.get("zero_below", 50)
    half_below = kelly_cfg.get("half_below", 60)
    fraction = kelly_cfg.get("fraction", 0.25)

    if buyscore < zero_below:
        return 0.0

    b = odds - 1.0
    q = 1.0 - prob
    raw_kelly = (b * prob - q) / b if b > 0 else 0.0
    if raw_kelly <= 0:
        return 0.0

    k = raw_kelly * fraction

    if buyscore < half_below:
        k *= 0.5

    return round(max(k, 0.0), 4)


# ==================== KellyStrategy（DI） ====================


class KellyStrategy(Protocol):
    """Kelly係数算出の抽象（Step4-6指示のDI境界）。"""

    def __call__(
        self, prob: float, odds: float, buyscore: float, cfg: dict
    ) -> float: ...


class DefaultKellyStrategy:
    """既定戦略: 移植元kelly_fraction（L303-L340）をそのまま用いる。"""

    def __call__(
        self, prob: float, odds: float, buyscore: float, cfg: dict
    ) -> float:
        return kelly_fraction(prob, odds, buyscore, cfg)


# ==================== BuyEngine ====================


@dataclass(frozen=True)
class BuyAssessment:
    """BuyEngineの純粋評価結果（Step4-6のスコープ4項目＋config版数）。

    Legacy: apply_buyscore（x_buyscore.py L645-）の候補付与部から
    資金管理（multiplier/purchased/cost/n_bets）を除いた純粋計算部分。
    Freeze Commit: 3a7f9c3

    Consumed by: Step5のBuyDecision組み立て（purchased/n_bets/cost/
    skip_reasonの最終確定は資金管理・点数制限と併せて上位層で行う）。
    """

    eval_id: str
    buyscore: float
    investment_type: str
    kelly_fraction: float
    skip_reason: Optional[str]
    config_version: str


class BuyEngine(Protocol):
    """買い判定エンジンの抽象（設計書⑤5.2系・Step4-6指示）。"""

    def assess(
        self,
        evaluation: RaceEvaluation,
        prediction: Prediction,
        config: dict,
    ) -> BuyAssessment: ...


class DefaultBuyEngine:
    """Ver4互換のBuyEngine実装。

    入力はRaceEvaluation/Prediction/Configのみ（外部依存禁止）。
    candidate/contextへの写像は移植元のキー名に合わせる:
      candidate: combo/prob/ev/odds ← Prediction.pred_*
      context:   match_index/race_type/upset_score ← RaceEvaluation
    レガシーの.get既定値（uncertainty=0.5等、モデル未収載キー）は
    移植元と同一の既定値で自然に補完される（指示準拠）。
    """

    def __init__(self, kelly_strategy: Optional[KellyStrategy] = None) -> None:
        self._kelly: KellyStrategy = kelly_strategy or DefaultKellyStrategy()

    def assess(
        self,
        evaluation: RaceEvaluation,
        prediction: Prediction,
        config: dict,
    ) -> BuyAssessment:
        if evaluation is None or prediction is None:
            raise ValidationError("evaluation and prediction are required")
        if evaluation.eval_id != prediction.eval_id:
            raise ValidationError(
                f"eval_id mismatch: evaluation={evaluation.eval_id!r} "
                f"prediction={prediction.eval_id!r}"
            )
        for field_name, value in (
            ("pred_prob", prediction.pred_prob),
            ("pred_ev", prediction.pred_ev),
            ("pred_odds", prediction.pred_odds),
        ):
            if value is None:
                raise ValidationError(f"prediction.{field_name} is required (None)")

        candidate: dict[str, Any] = {
            "combo": prediction.pred_combo,
            "prob": prediction.pred_prob,
            "ev": prediction.pred_ev,
            "odds": prediction.pred_odds,
        }
        context: dict[str, Any] = {
            "match_index": evaluation.match_index,
            "race_type": evaluation.race_type,
            "upset_score": evaluation.upset_score,
        }

        # 移植元apply_buyscoreと同一順: buyscore付与 → kelly → 見送り判定
        score = calc_buyscore(candidate, context, config)
        kelly = self._kelly(
            candidate.get("prob", 0), candidate.get("odds", 0), score, config
        )
        scored_candidate = dict(candidate)
        scored_candidate["buyscore"] = score
        skip = check_passthrough([scored_candidate], context, config)
        inv_type = investment_type(
            score, candidate.get("ev", 0.0), candidate.get("odds", 0.0),
            context.get("race_type", "混戦"),
        )
        return BuyAssessment(
            eval_id=evaluation.eval_id,
            buyscore=score,
            investment_type=inv_type,
            kelly_fraction=kelly,
            skip_reason=skip,
            config_version=str(config.get("_version", "")),
        )
