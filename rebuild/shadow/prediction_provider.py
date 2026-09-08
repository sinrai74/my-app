"""
PredictionProvider実体（Step6-1・W2結線）: Legacy _evaluate_bets をラップ。

方針（Step5-0(a) / Step6-1レビュー確定）:
  Prediction生成の実体はLegacy notify_arashi._evaluate_bets（L1834）。
  これを分解・再実装・コピーせず、importして「呼ぶだけ」。
  _evaluate_betsはオッズ・展示・パターン等の中間データを引数に取るため、
  それらは呼び出し側（Shadow計測配線）がPredictionContextとして供給する。

責務: RaceEvaluation + PredictionContext → Legacy _evaluate_bets呼び出し →
  戻り値をPredictionモデルへ写像するだけ。EV/prob/buyscore/kelly算出は
  一切しない（すべて_evaluate_bets内部の責務）。

engine.predictは結線しない（Step6-1レビュー(c)）。Prediction生成は
本Providerが担い、core.engine.predictはNotImplementedErrorのまま据え置く。

依存: shadow → models / Legacy(_evaluate_bets, import利用)。
  Legacyコードは一切変更しない（呼ぶだけ）。
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, Optional, Sequence

from models.evaluation import LegacyPurchaseResult, Prediction, RaceEvaluation

log = logging.getLogger(__name__)

# Legacy _evaluate_bets のシグネチャ（呼ぶだけ・改変しない）
LegacyEvaluateBets = Callable[..., Any]


class PredictionContext:
    """_evaluate_betsが必要とする中間データ（呼び出し側が供給）。

    RaceEvaluationからは得られない実行時データ（オッズ・展示・パターン・
    boats等）を保持する。本Providerはこれを_evaluate_betsへ渡すだけで、
    値の生成・加工はしない。
    """

    __slots__ = (
        "patterns", "ml_probs", "odds_map", "boats", "upset_score",
        "weather", "has_exhibition", "target_lanes", "odds_dropped",
        "bankroll",
    )

    def __init__(
        self,
        patterns: Mapping[str, list],
        ml_probs: Mapping[int, float],
        odds_map: Mapping[str, float],
        boats: Optional[Sequence[Any]] = None,
        upset_score: float = 0.0,
        weather: Any = None,
        has_exhibition: bool = True,
        target_lanes: Optional[list[int]] = None,
        odds_dropped: Optional[list[str]] = None,
        bankroll: int = 10000,
    ) -> None:
        self.patterns = patterns
        self.ml_probs = ml_probs
        self.odds_map = odds_map
        self.boats = boats
        self.upset_score = upset_score
        self.weather = weather
        self.has_exhibition = has_exhibition
        self.target_lanes = target_lanes
        self.odds_dropped = odds_dropped
        self.bankroll = bankroll


class LegacyPredictionProvider:
    """_evaluate_bets をラップするPredictionProvider実体。

    provide(evaluation, config) は PredictionProvider Protocol に適合するが、
    _evaluate_betsは追加の中間データ（PredictionContext）を要するため、
    contextはコンストラクタ注入 or provide時のcontext_resolverで供給する。
    """

    def __init__(
        self,
        context_resolver: Callable[[RaceEvaluation], PredictionContext],
        evaluate_bets: Optional[LegacyEvaluateBets] = None,
        result_mapper: Optional[Callable[[Any, RaceEvaluation], Prediction]] = None,
    ) -> None:
        # context_resolver: eval_id等から中間データを用意する（呼び出し側の責務）
        self._context_resolver = context_resolver
        self._evaluate_bets = evaluate_bets
        self._result_mapper = result_mapper or _default_result_mapper

    def provide(self, evaluation: RaceEvaluation, config: dict) -> Prediction:
        log.info("PredictionProvider provide start eval_id=%s", evaluation.eval_id)
        evaluate_bets = self._evaluate_bets
        if evaluate_bets is None:
            from notify_arashi import _evaluate_bets as evaluate_bets  # 遅延import

        ctx = self._context_resolver(evaluation)
        # Legacy _evaluate_bets を「呼ぶだけ」。引数はcontextからそのまま渡す
        legacy_result = evaluate_bets(
            patterns=ctx.patterns,
            ml_probs=ctx.ml_probs,
            odds_map=ctx.odds_map,
            bankroll=ctx.bankroll,
            target_lanes=ctx.target_lanes,
            has_exhibition=ctx.has_exhibition,
            boats=ctx.boats,
            upset_score=ctx.upset_score,
            odds_dropped=ctx.odds_dropped,
            weather=ctx.weather,
            venue_num=evaluation.venue_num,
            race_number=evaluation.race_number,
            race_date=evaluation.race_date,
            venue_name=evaluation.venue_name,
        )
        prediction = self._result_mapper(legacy_result, evaluation)
        log.info("PredictionProvider provide end eval_id=%s", evaluation.eval_id)
        return prediction


def _default_result_mapper(
    legacy_result: Any, evaluation: RaceEvaluation
) -> Prediction:
    """_evaluate_betsの戻り値 → Prediction写像（読み取りのみ・算出しない）。

    Step6-1レビュー反映: デフォルト補完（0.0等の仮値設定）は禁止。
    必須キーが欠損している場合は ValueError を送出して停止する
    （比較不能として扱う。自己判断での穴埋めはしない）。
    値の加工・EV再計算・確率/オッズ補正は一切行わない。
    """
    if not isinstance(legacy_result, dict):
        # None・タプル等は写像方法が未確定。サイレント推測をしない。
        raise ValueError(
            "legacy _evaluate_bets returned non-dict "
            f"(type={type(legacy_result).__name__}); "
            "inject a custom result_mapper for this shape"
        )

    required = ("combo", "prob", "ev", "odds", "confidence")
    missing = [key for key in required if key not in legacy_result]
    if missing:
        raise ValueError(
            f"legacy _evaluate_bets result missing required keys: {missing} "
            f"(eval_id={evaluation.eval_id}); no default value is supplied"
        )

    why = legacy_result.get("why_bet", ())
    if isinstance(why, list):
        why = tuple(why)
    return Prediction(
        eval_id=evaluation.eval_id,
        pred_combo=legacy_result["combo"],
        pred_prob=float(legacy_result["prob"]),
        pred_ev=float(legacy_result["ev"]),
        pred_odds=float(legacy_result["odds"]),
        confidence=float(legacy_result["confidence"]),
        why_bet=str(why) if not isinstance(why, str) else why,
        patterns=tuple(legacy_result.get("buy", ())),
    )


class _EvaluateBetsCapture:
    """`_evaluate_bets` を1回だけ呼び、Prediction用（result[0]、K1・既存契約）
    に加えて、直近の生list全体を eval_id 単位で保持するCallable。

    背景（Legacy購入結果をBuyDecisionへ渡すための配線・W案）:
      Legacyの実際の購入対象は `recommended[0]`（Prediction用・K1）ではなく、
      `assign_rank_labels` 後の全購入確定list（最大4点）である。この全list
      をBuyDecisionBuilderへ渡す必要があるが、`_evaluate_bets` は内部で
      資金管理ロジック（hit_record.csv等の外部状態を読む）を呼ぶため、
      2回呼び出すと異なる結果を返すリスクがある（冪等性が保証されない）。
      したがって1回の呼び出し結果を、Prediction用途と購入結果用途の両方で
      共有する。

    既存契約との関係:
      __call__ の外部挙動（引数・戻り値・空list時のValueError）は、
      `actions/shadow_entrypoint.py::build_bundle` 内の既存ローカル関数
      `_evaluate_bets_first`（Step6-2c-9・K1）と完全に同一。K1・
      LegacyPredictionProviderの既存外部契約は変更しない。
      追加されるのは「呼び出し後に last_purchase_result(eval_id) で
      全listを読める」という副次的なアクセサのみ。

    スレッド安全性: 本クラスは単一プロセス・単一レース逐次処理を前提とする
    （既存のShadow/本番運用と同じ前提）。並行実行はサポートしない。
    """

    def __init__(
        self, evaluate_bets: Optional[LegacyEvaluateBets] = None
    ) -> None:
        # DIパターンはLegacyPredictionProviderと同一（未指定時のみ遅延import）。
        self._evaluate_bets = evaluate_bets
        self._last_eval_id: Optional[str] = None
        self._last_raw_result: Optional[list] = None

    def __call__(self, **kwargs: Any) -> Any:
        evaluate_bets = self._evaluate_bets
        if evaluate_bets is None:
            from notify_arashi import _evaluate_bets as evaluate_bets

        result = evaluate_bets(**kwargs)

        # eval_id は既存の組み立て規約（race_date_venue02d_race02d）に従う。
        # models/race.py・evaluation_pipeline.py等と同一形式。
        self._last_eval_id = "{}_{:02d}_{:02d}".format(
            kwargs["race_date"], kwargs["venue_num"], kwargs["race_number"],
        )
        self._last_raw_result = result if isinstance(result, list) else None

        if isinstance(result, list) and not result:
            # 既存 _evaluate_bets_first と同一のエラー契約（変更しない）。
            raise ValueError(
                "_evaluate_bets returned an empty list; no bet candidate is "
                "available for this race (no default value is supplied)"
            )
        if isinstance(result, list):
            return result[0]
        return result

    def last_purchase_result(self, eval_id: str) -> LegacyPurchaseResult:
        """直近の `__call__` が保持した全listを LegacyPurchaseResult として返す。

        eval_id不一致（別レースの呼び出し後に取得された等）はサイレントに
        許容せず例外とする。__call__ が一度も成功していない場合も例外とする。
        """
        if self._last_eval_id != eval_id or self._last_raw_result is None:
            raise ValueError(
                "no captured _evaluate_bets result for eval_id="
                f"{eval_id!r} (last captured eval_id={self._last_eval_id!r}); "
                "last_purchase_result() must be called immediately after "
                "the corresponding provide() call for the same race"
            )
        return LegacyPurchaseResult(
            eval_id=eval_id, purchases=tuple(self._last_raw_result)
        )
