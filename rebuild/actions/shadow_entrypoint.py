"""
Shadow運用エントリポイント（Step5-6実装・Step6-2b起動配線追加）。

責務: Provider等の具象をDIで組み立て、ShadowRunnerへ渡して実行するだけ。
禁止: 判定・計算・補正・HTML生成・通知生成。

Step6-2b（起動配線）のスコープ:
  「Shadowが起動し、最初のレースまで到達すること」のみを目的とする。
  - 対象レースは環境変数 TARGET_RACE で明示指定（Legacy fetch_programsに
    依存せず、API・ネットワーク・開催状況の影響を切り離すため）
  - 1レース限定（Step6-2c-10でTARGET_RACESによる複数レース対応を追加）
  - PredictionContextのcontext_resolverは結線しない。実際の例外を観測して
    から最小限の配線を次Stepで行う（推測実装の回避）

注記: 既存の GitHub Actions（.yml）ファイルは変更していない。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

from actions.flags import use_rebuild_pipeline
from actions.wiring import PipelineBundle, assemble_pipelines
from shadow.aggregator import ShadowAggregate, ShadowAggregator
from shadow.go_no_go import GoNoGoCriteria, evaluate_go_no_go
from shadow.runner import ShadowRunner

log = logging.getLogger(__name__)


def run_shadow_for_race(
    bundle: PipelineBundle,
    race_date: str,
    venue_num: int,
    race_number: int,
    output_paths: dict,
    legacy_values: dict | None = None,
) -> dict:
    """1レース分のShadow実行（呼ぶだけ）。判定はGoNoGoCriteria側で行う。"""
    log.info("Pipeline start date=%s venue=%s race=%s",
              race_date, venue_num, race_number)
    runner = ShadowRunner(bundle)
    result = runner.run_and_compare(
        race_date, venue_num, race_number, output_paths, legacy_values
    )
    log.info("Evaluation complete eval_id=%s", result.eval_id)
    log.info("Buy complete eval_id=%s", result.eval_id)
    log.info("Output complete eval_id=%s", result.eval_id)
    log.info("Notification complete eval_id=%s (0 real sends)", result.eval_id)
    return {
        "eval_id": result.eval_id,
        "diff_count": len(result.diffs),
        "diffs": result.diffs,
    }


def decide(criteria: GoNoGoCriteria) -> str:
    """Go/No-Go判定を呼ぶだけ（ロジックはshadow.go_no_go側）。"""
    log.info("Go/No-Go judge start")
    result = evaluate_go_no_go(criteria)
    log.info("Go/No-Go judge end decision=%s", result.decision)
    return result.decision


def should_use_rebuild() -> bool:
    """切替フラグを読むだけ（DI切替の判断材料。Legacyコードは書き換えない）。"""
    return use_rebuild_pipeline()


# ==================== Step6-2b: 起動配線 ====================


def parse_target_race(value: str | None = None) -> tuple[str, int, int]:
    """対象レースを環境変数 TARGET_RACE から解析する（明示指定方式）。

    形式: "YYYYMMDD_venue_race"（例 "20260704_12_5"）
    Legacy fetch_programs に依存せず、API・ネットワーク・開催状況の影響を
    切り離して起動確認するための方式（Step6-2b方針）。

    未指定・不正形式は補完せず ValueError（デフォルト補完禁止）。
    """
    raw = value if value is not None else os.environ.get("TARGET_RACE", "")
    raw = raw.strip()
    if not raw:
        raise ValueError(
            "TARGET_RACE is required (format: YYYYMMDD_venue_race, "
            'e.g. "20260704_12_5"); no default value is supplied'
        )
    parts = raw.split("_")
    if len(parts) != 3:
        raise ValueError(
            f"TARGET_RACE has invalid format: {raw!r} "
            "(expected YYYYMMDD_venue_race)"
        )
    race_date, venue_raw, race_raw = parts
    if not (race_date.isdigit() and len(race_date) == 8):
        raise ValueError(f"TARGET_RACE date is invalid: {race_date!r}")
    return race_date, int(venue_raw), int(race_raw)


def _load_freeze_configs() -> tuple[dict, dict]:
    """Freeze対象のconfigを読むだけ（変更・補完はしない）。

    asahi_config.json（SHA 6a7862b8…）と buyscore_config.json（SHA da6a4eda…）。
    Legacyの読み込み関数をimportして呼ぶ（パス解決をLegacyに委ねる）。
    """
    from x_asahi_scoring import load_asahi_config
    from x_buyscore import load_config as load_buyscore_config

    return load_asahi_config(), load_buyscore_config()


def build_bundle(
    eval_config: dict | None = None, buy_config: dict | None = None
) -> PipelineBundle:
    """具象部品を組み立ててPipelineBundleを返すだけ（計算・判定なし）。

    Legacy部品は import して渡すのみ（無改変）:
      - notify_arashi.fetch_programs / _extract_boats_from_program（Provider経由）
      - x_venue_stats（VenueStatsProvider互換のモジュール関数）
      - x_asahi_scoring.load_config / x_buyscore.load_config（Freeze config読取）
    通知は NullNotifier のみ登録し、実送信を物理的に禁止する。

    context_resolver は本Stepでは結線しない（Step6-2b方針）。
    PredictionProviderへ到達した時点の実際の例外を観測するため、
    未結線であることを明示する resolver を渡す。
    """
    from adapters.providers import BoatsProvider
    from core.buyscore import DefaultBuyEngine
    from core.engine import Ver4Engine
    from features.feature_builder import DefaultFeatureBuilder
    from notification.service import NotificationService
    from pipelines.wiring import RaceArgBoatsResolver
    from shadow.notifier import NullNotifier
    from shadow.prediction_provider import (
        LegacyPredictionProvider,
        PredictionContext,
    )

    import x_venue_stats  # Legacy: VenueStatsProvider互換（import利用のみ）

    if eval_config is None or buy_config is None:
        loaded_eval, loaded_buy = _load_freeze_configs()
        eval_config = eval_config if eval_config is not None else loaded_eval
        buy_config = buy_config if buy_config is not None else loaded_buy

    boats_provider = BoatsProvider()
    feature_builder = DefaultFeatureBuilder(venue_stats=x_venue_stats)
    engine = Ver4Engine(
        boats_resolver=RaceArgBoatsResolver(boats_provider),
        venue_stats=x_venue_stats,
    )

    def _context_resolver_required(evaluation):
        # Step6-2c-2: PredictionContextの必須入力（patterns/ml_probs/odds_map）を
        # 結線する。boats（Step6-2c-1）に加え、同一の計算チェーンで得られる
        # upset_score/target_lanes も分割せず一緒に渡す（同じ計算を二度求めない）。
        #
        # Legツール関数はすべて import して呼ぶだけ（改変・コピー・再実装なし）。
        from notify_arashi import (
            _generate_patterns,
            _predict_win_prob,
        )
        from odds_fetch import fetch_odds
        from x_asahi_scoring import calculate_upset_score_v2

        program, boat_objs = boats_provider._get_source(
            evaluation.race_date, evaluation.venue_num, evaluation.race_number
        )

        # race_grade は Legツール同様 program の race_grade_number から取得
        # （notify_arashi L1182: prog.get('race_grade_number', 0) or 0）。
        race_grade = int(program.get("race_grade_number", 0) or 0)

        # ml_probs: boatsから（MLモデル model_all.pkl。未ロード時はLegツールが空dict）
        ml_probs = _predict_win_prob(boat_objs)

        # upset_score / target_lanes: 純粋関数。patternsの前提。
        upset_score, _detail, target_lanes = calculate_upset_score_v2(
            boat_objs,
            race_grade,
            venue_num=evaluation.venue_num,
            is_night=evaluation.is_night,
            config=eval_config,
        )

        # patterns: 上記の成果物から（純粋関数）
        patterns = _generate_patterns(target_lanes, upset_score)

        # odds_map: レース識別子から（過去レースは空dictになりやすい＝Legツール挙動）
        venue_code = str(evaluation.venue_num).zfill(2)
        odds_map = fetch_odds(
            evaluation.race_number, venue_code, evaluation.race_date
        ) or {}

        # 必須3項目＋同一チェーンの成果物（boats/upset_score/target_lanes）を渡す。
        # weather/has_exhibition/odds_dropped/bankroll は未結線（次Step以降）。
        return PredictionContext(
            patterns=patterns,
            ml_probs=ml_probs,
            odds_map=odds_map,
            boats=boat_objs,
            upset_score=upset_score,
            target_lanes=target_lanes,
        )

    def _evaluate_bets_first(**kwargs):
        # Step6-2c-9: Legツール _evaluate_bets のreturn値listから1件を取り出す。
        #
        # 設計根拠（Step6-2c-8 §18）:
        #   - 設置箇所 S2: LegacyPredictionProviderの evaluate_bets DI注入
        #     （_evaluate_bets / _default_result_mapper / provide 本体は無改変）
        #   - 選定基準 K1: index 0
        #     （Legツール呼び出し元 notify_arashi L3546 が recommended[0] を参照。
        #       見送り経路は top[:1] の1件、購入経路は buyscore降順sort後の先頭）
        #   - 空list: ValueError送出
        #     （_default_result_mapper と同方針。仮値補完はしない）
        #   - dict以外の型検証は _default_result_mapper（dict要求）へ委譲する
        from notify_arashi import _evaluate_bets

        result = _evaluate_bets(**kwargs)
        if isinstance(result, list) and not result:
            raise ValueError(
                "_evaluate_bets returned an empty list; no bet candidate is "
                "available for this race (no default value is supplied)"
            )
        if isinstance(result, list):
            return result[0]
        return result

    notification_service = NotificationService({
        "mail": NullNotifier("mail"),
        "line": NullNotifier("line"),
        "discord": NullNotifier("discord"),
        "x": NullNotifier("x"),
    })

    return assemble_pipelines(
        race_source=boats_provider,
        feature_builder=feature_builder,
        engine=engine,
        now_provider=lambda: datetime.now(timezone.utc),
        eval_config=eval_config,
        durable_store=None,
        prediction_provider=LegacyPredictionProvider(
            context_resolver=_context_resolver_required,
            evaluate_bets=_evaluate_bets_first,
        ),
        buy_engine=DefaultBuyEngine(),
        buy_config=buy_config,
        output_renderers={},
        notification_service=notification_service,
    )


def run_shadow(
    race_date: str,
    venue_num: int,
    race_number: int,
    bundle: PipelineBundle | None = None,
) -> dict:
    """1レース分のShadowを起動する（Step6-2b: 到達確認が目的）。

    比較用のlegacy_valuesは渡さない（起動確認が目的のため）。
    Outputも空（output_renderers={}）で、HTML生成は行わない。
    """
    log.info("Shadow launch start date=%s venue=%s race=%s",
             race_date, venue_num, race_number)
    log.info("USE_REBUILD_PIPELINE=%s (Shadow requires False)",
             use_rebuild_pipeline())

    active_bundle = bundle if bundle is not None else build_bundle()
    log.info("PipelineBundle created")

    runner = ShadowRunner(active_bundle)
    log.info("ShadowRunner created")

    result = runner.run_and_compare(
        race_date, venue_num, race_number, output_paths={}
    )
    log.info("Shadow launch reached race eval_id=%s", result.eval_id)
    return {
        "eval_id": result.eval_id,
        "diff_count": len(result.diffs),
        "diffs": result.diffs,
    }


# ==================== Step6-2c-10: 複数レースShadow実行 ====================


def parse_target_races(value: str | None = None) -> list[tuple[str, int, int]]:
    """複数レースを環境変数 TARGET_RACES から解析する（D-4）。

    形式: "YYYYMMDD_venue_race" をカンマ区切りで列挙
      例 "20260704_12_5,20260704_12_6,20260704_01_1"

    既存の parse_target_race()（単一・TARGET_RACE）は変更しない。
    未指定・不正形式は補完せず ValueError（デフォルト補完禁止）。
    """
    raw = value if value is not None else os.environ.get("TARGET_RACES", "")
    raw = raw.strip()
    if not raw:
        raise ValueError(
            "TARGET_RACES is required (comma-separated "
            'YYYYMMDD_venue_race, e.g. "20260704_12_5,20260704_12_6"); '
            "no default value is supplied"
        )
    races: list[tuple[str, int, int]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        races.append(parse_target_race(token))
    if not races:
        raise ValueError(
            f"TARGET_RACES contains no valid race: {raw!r}"
        )
    return races


def run_shadow_multiple(
    races: list[tuple[str, int, int]],
    bundle: PipelineBundle | None = None,
    required: int = 100,
) -> dict:
    """複数レース分のShadowを1実行で処理し、連続一致数を集約する（D-2/D-4）。

    集約は shadow.aggregator（既存 ConsecutiveMatchCounter を利用する薄い層）
    に委譲する。判定式・比較ロジックは変更しない。
    D-5（実行間の永続化）は本Stepの対象外＝1実行内で完結する。
    """
    log.info("Shadow multi-race start count=%d required=%d",
             len(races), required)
    log.info("USE_REBUILD_PIPELINE=%s (Shadow requires False)",
             use_rebuild_pipeline())

    active_bundle = bundle if bundle is not None else build_bundle()
    log.info("PipelineBundle created")

    runner = ShadowRunner(active_bundle)
    log.info("ShadowRunner created")

    aggregator = ShadowAggregator(required=required)
    races_report: list[dict] = []
    for race_date, venue_num, race_number in races:
        result = runner.run_and_compare(
            race_date, venue_num, race_number, output_paths={}
        )
        aggregator.record(result)
        races_report.append({
            "eval_id": result.eval_id,
            "diff_count": len(result.diffs),
            "diffs": result.diffs,
        })
        log.info("Shadow multi-race progress eval_id=%s streak=%d",
                 result.eval_id, aggregator.consecutive_matches)

    aggregate = aggregator.aggregate()
    for line in aggregate.summary_lines():
        log.info("Shadow aggregate %s", line)
    log.info(
        "Shadow multi-race end total=%d streak=%d satisfied=%s",
        aggregate.total_races, aggregate.current_streak, aggregator.satisfied,
    )
    return {
        "races": races_report,
        "total_races": aggregate.total_races,
        "matched_races": aggregate.matched_races,
        "diff_races": aggregate.diff_races,
        "shadow_consecutive_matches": aggregate.current_streak,
        "max_streak": aggregate.max_streak,
        "broken_at": aggregate.broken_at,
        "required": required,
        "satisfied": aggregator.satisfied,
    }


def main(argv: list[str] | None = None) -> int:
    """CLIエントリ（引数解析と呼び出しのみ・判定/計算なし）。

    対象レースは TARGET_RACE 環境変数で明示指定する。
    例外はそのまま報告して終了コード1（補正・握りつぶしはしない）。
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # TARGET_RACES（複数）が指定されていれば複数レース経路、
    # なければ既存の TARGET_RACE（単一）経路を使う（後方互換）。
    multi_raw = os.environ.get("TARGET_RACES", "").strip()
    if multi_raw:
        try:
            races = parse_target_races()
        except ValueError as exc:
            log.error("TARGET_RACES error: %s", exc)
            return 1
        required = int(os.environ.get("SHADOW_REQUIRED_MATCHES", "100"))
        try:
            report = run_shadow_multiple(races, required=required)
        except Exception as exc:
            log.error("Shadow multi-race failed: %s: %s",
                      type(exc).__name__, exc)
            raise
    else:
        try:
            race_date, venue_num, race_number = parse_target_race()
        except ValueError as exc:
            log.error("TARGET_RACE error: %s", exc)
            return 1

        try:
            report = run_shadow(race_date, venue_num, race_number)
        except Exception as exc:  # 起動確認のため例外内容を明示して終了
            log.error("Shadow launch failed: %s: %s", type(exc).__name__, exc)
            raise

    out_path = os.environ.get("SHADOW_REPORT_PATH", "shadow_diff_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log.info("Shadow report written path=%s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
