"""
EvaluationPipeline（Step5-2）: L0→L1→L2→L3 の結線のみ。

役割: Provider（取得）→ FeatureBuilder（L1）→ Ver4Engine（L2 evaluate）→
      DurableStore（L3 保存）を「順に呼ぶだけ」。

厳守（Step5-2指示・結線のみ）:
  本Pipelineは一切の計算・判定・補正・分類変換・investment_type生成・
  race_type変換をしない。各部品の戻り値を次の部品へ渡すだけである。
  スコアや閾値に触れたくなったらそれはPipelineの責務外（実装を止めて提案）。

依存（Protocolのみ参照・具象非依存）:
  - RaceProvider / BoatsResolver（adapters）: race・boatsの取得
  - FeatureBuilder（features）: FeatureSet構築
  - EvaluationEngine（core.engine）: evaluate
  - now_provider: 現在時刻の注入（core純粋性維持・Pipelineは時刻を生成しない）
  - config: freeze config（読むだけ・変更しない）
  - DurableEvaluationStore（任意）: 保存。None時は保存せず評価結果のみ返す
    （Shadow影稼働では保存を注入せず、比較用に結果だけ取得する）

FeatureInputs について:
  FeatureBuilderはFeatureInputs(boats=...)を要求する。boatsはBoatsResolverが
  返すVer4互換Mapping列であり、Pipelineはそれをそのまま包んで渡すだけ
  （属性の加工・選別はしない）。
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from features.feature_builder import FeatureBuilder, FeatureInputs
from models.evaluation import RaceEvaluation
from models.race import Race

log = logging.getLogger(__name__)

NowProvider = Callable[[], datetime]


class _RaceSource(Protocol):
    """Pipelineが必要とする取得口（RaceProvider＋BoatsResolverの合成）。"""

    def resolve_race(
        self, race_date: str, venue_num: int, race_number: int
    ) -> Race: ...

    def resolve_boats(
        self, race_date: str, venue_num: int, race_number: int
    ) -> Sequence[Mapping[str, Any]]: ...


class _EvaluationEngine(Protocol):
    """core.engine.EvaluationEngine のうち本Pipelineが使う部分。"""

    def evaluate(
        self,
        race: Race,
        feature_set: Any,
        weather: Any,
        config: dict,
        now: datetime,
    ) -> RaceEvaluation: ...


class _DurableStore(Protocol):
    def append_durably(
        self, evaluation: RaceEvaluation, commit_message: str
    ) -> None: ...


class EvaluationPipeline:
    """評価の結線パイプライン（計算しない）。"""

    def __init__(
        self,
        race_source: _RaceSource,
        feature_builder: FeatureBuilder,
        engine: _EvaluationEngine,
        now_provider: NowProvider,
        config: dict,
        durable_store: Optional[_DurableStore] = None,
    ) -> None:
        self._race_source = race_source
        self._feature_builder = feature_builder
        self._engine = engine
        self._now_provider = now_provider
        self._config = config
        self._durable_store = durable_store

    def evaluate_race(
        self,
        race_date: str,
        venue_num: int,
        race_number: int,
        *,
        persist: bool = False,
        commit_message: Optional[str] = None,
    ) -> RaceEvaluation:
        """1レースを評価して返す。persist=Trueかつdurable_store注入時のみ保存する。

        結線の順序:
          1. race  = race_source.resolve_race(...)      （L0取得）
          2. boats = race_source.resolve_boats(...)      （L0取得・Ver4互換）
          3. fs    = feature_builder.build(race, FeatureInputs(boats), built_at)
                                                          （L1）
          4. ev    = engine.evaluate(race, fs, race.weather, config, now)
                                                          （L2・engine内部で
                                                            boats_resolverを再利用）
          5. persist時: durable_store.append_durably(ev, msg)   （L3）
        いずれのステップも戻り値を次へ渡すだけ。加工・判定はしない。
        """
        start = time.monotonic()
        eval_id = f"{race_date}_{venue_num:02d}_{race_number:02d}"
        log.info("EvaluationPipeline start eval_id=%s", eval_id)

        now = self._now_provider()
        race = self._race_source.resolve_race(race_date, venue_num, race_number)
        boats = self._race_source.resolve_boats(race_date, venue_num, race_number)

        feature_set = self._feature_builder.build(
            race, FeatureInputs(boats=tuple(boats)), built_at=now.isoformat()
        )
        evaluation = self._engine.evaluate(
            race, feature_set, race.weather, self._config, now
        )

        if persist:
            if self._durable_store is None:
                # サイレントに保存を握りつぶさない（⑫）。保存要求と注入の不整合を明示。
                raise ValueError(
                    "persist=True requires durable_store to be injected"
                )
            self._durable_store.append_durably(
                evaluation, commit_message or f"eval {eval_id}"
            )

        log.info(
            "EvaluationPipeline end eval_id=%s persisted=%s elapsed=%.3fs",
            eval_id, persist, time.monotonic() - start,
        )
        return evaluation
