"""
正式G3（B案）：1レースの G3 評価の組み立て（(b-2) ハイブリッド方式）。

流れ（判断記録 第3版 §6・追補1）:
  1. Rebuild の評価：Ver4Engine に predictor（win_probs）を注入して RaceEvaluation を作り、
     同じ入力で core.upset.calculate_upset_score を呼び直して UpsetResult を得る。
  2. Rebuild の ML 加点：core.ml_bonus.apply_ml_bonus。
  3. 予想：加点後の upset・target と、それから作った patterns を PredictionContext に入れ、
     Legacy の `_evaluate_bets` を1回だけ呼ぶ（LegacyPredictionProvider 経由）。
     その間、apply_buyscore の呼び出しを写し取る（shadow.g3_capture）。
  4. 入力つきの評価：core.g3_assess.assess_with_input。
  5. 同一性の条件（追補1 §1）の検査と、Legacy の比較基準値の取り出し、記録。

既存の build_bundle（actions/shadow_entrypoint.py）と本番入口は変更しない。
正式G2 の評価側 baseline の42ファイルからは import しない新規ファイル。

この関数は G3 の連続カウントを行わない（100連続の定義・候補0件の扱いなどは未決定。
第3版 §12）。記録の区分だけを付ける。
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime
from typing import Any, Callable, Mapping, Optional, Sequence

from core.engine import Ver4Engine, _grade_to_number
from core.exceptions import ValidationError
from core.g3_assess import assess_with_input, build_input_context
from core.ml_bonus import apply_ml_bonus
from core.upset import UpsetResult, calculate_upset_score
from models.evaluation import RaceEvaluation
from models.g3_input import BuyScoreInput, G3InputValidationError
from shadow.g3_capture import STATUS_CAPTURED, ApplyBuyscoreCapture
from shadow.prediction_provider import (
    LegacyPredictionProvider,
    PredictionContext,
    _EvaluateBetsCapture,
)

# ---- 記録の区分（G3 の連続カウントの扱いは未決定。区分だけを付ける） ----
OUTCOME_COMPARED = "compared"                       # 比較が成立した（一致・不一致は match で示す）
OUTCOME_NO_PREDICTION = "no_prediction"             # 条件8：win_probs が空・None
OUTCOME_NO_ODDS = "no_odds"                         # odds_map が空（Legacy は _evaluate_bets を呼ばない）
OUTCOME_NO_CANDIDATE = "no_candidate"               # _evaluate_bets が空のリストを返した
OUTCOME_LEGACY_EXCEPTION = "legacy_exception"       # apply_buyscore が例外
OUTCOME_CAPTURE_FAILED = "capture_failed"           # 写し取りの失敗
OUTCOME_INPUT_DEFECT = "input_defect"               # 条件1・3・5・6・7 の食い違い
OUTCOME_PREDICTION_DRIFT = "not_established_prediction_drift"  # 追補1 §3 の1
OUTCOME_LEGACY_EVALUATE_BETS_EXCEPTION = "legacy_evaluate_bets_exception"  # _evaluate_bets 自体の例外（F-03）
OUTCOME_PREDICTION_MAPPING_ERROR = "prediction_mapping_error"  # 戻り値から Prediction を作れない（F-03）

# 区分の優先順位（F-13）：条件1の不成立（入力不備）を最優先し、その後に
# no_prediction → no_odds → _evaluate_bets の結果 → 写し取り → 同一性 → 比較 の順で判定する。

# repo 直下の Git 追跡の設定（freeze 版）。F-01：Legacy が使った設定と一致するかを記録する。
FREEZE_CONFIG_SHA256 = "da6a4edaf8220aa52ed3e4577c23ff443509512a971d0ba322079eb4c4cd6f1d"

LEGACY_CONTEXT_KEYS = (
    "match_index", "match_index_approx", "race_type", "has_exhibition",
    "market_gap", "upset_score", "ex_rank_1st",
)


def _sha256_json(obj: Any) -> str:
    data = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def build_rebuild_evaluation(
    *,
    race: Any,
    feature_set: Any,
    engine_boats: Sequence[Any],
    config: dict,
    venue_stats: Any,
    win_probs: Optional[Mapping[int, float]],
    now: datetime,
) -> tuple[RaceEvaluation, UpsetResult]:
    """Rebuild の評価。predictor は eval_id を受け取り、事前に1回だけ求めた win_probs を返す。"""
    engine = Ver4Engine(
        boats_resolver=lambda _race: engine_boats,
        venue_stats=venue_stats,
        predictor=(lambda _eval_id: win_probs),
    )
    evaluation = engine.evaluate(race, feature_set, None, config, now)
    upset = calculate_upset_score(
        engine_boats,
        config,
        race_grade=_grade_to_number(race.grade),
        venue_num=race.venue_num,
        is_night=race.is_night,
        venue_stats=venue_stats,
    )
    return evaluation, upset


def _check(cond: dict, name: str, ok: bool, detail: str = "") -> None:
    cond[name] = {"ok": bool(ok), "detail": detail}


def _freeze_config() -> Optional[dict]:
    """repo 直下の buyscore_config.json（freeze 版。SHA-256 を確かめたうえで読む）。読めなければ None。"""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                        "buyscore_config.json")
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if hashlib.sha256(data).hexdigest() != FREEZE_CONFIG_SHA256:
        return None
    return json.loads(data.decode("utf-8"))


def evaluate_race_g3(
    *,
    evaluation: RaceEvaluation,
    upset: UpsetResult,
    legacy_boats: Sequence[Any],
    odds_map: Mapping[str, float],
    weather: Any,
    buy_config: dict,
    evaluate_bets: Callable[..., Any],
    generate_patterns: Callable[[list, float], Any],
    legacy_target: Optional[Sequence[int]] = None,
    capture_factory: Callable[[], ApplyBuyscoreCapture] = ApplyBuyscoreCapture,
    run_id: Optional[str] = None,
    executed_at: Optional[str] = None,
) -> dict[str, Any]:
    """1レースの G3 評価。戻り値は JSON にできる記録（辞書）。連続カウントは行わない。"""
    rec: dict[str, Any] = {
        "eval_id": evaluation.eval_id,
        "run_id": run_id,
        "executed_at": executed_at,
        "outcome": None,
        "conditions": {},
        "ml_bonus": None,
        "target": {
            "rebuild_upset_target": list(upset.target_lanes),
            "rebuild_target_post": None,
            "legacy_target": list(legacy_target) if legacy_target is not None else None,
            "legacy_target_source": "caller" if legacy_target is not None else None,
            "rebuild_post_equals_legacy": None,
            "legacy_target_call_count": "not_implemented",
        },
        "evaluate_bets_call_count": 0,
        "capture": None,
        "config": None,
        "rebuild": None,
        "legacy": None,
        "match": None,
        # 第3版 §12 の8・9：比較方法は未決定。match は == による仮の表示で、判定の定義ではない
        "match_note": "provisional_equality_not_a_judgment_rule",
    }
    cond = rec["conditions"]

    # 条件1：加点前の upset の一致（区分の優先順位の最上位。F-13）
    _check(cond, "c1_upset_pre", upset.upset_score == evaluation.upset_score,
           f"upset={upset.upset_score!r} evaluation={evaluation.upset_score!r}")
    if not cond["c1_upset_pre"]["ok"]:
        rec["outcome"] = OUTCOME_INPUT_DEFECT
        return rec

    # 条件8：win_probs
    win_probs = evaluation.win_probs
    if not win_probs:
        rec["outcome"] = OUTCOME_NO_PREDICTION
        return rec

    lane_order = [b.lane for b in legacy_boats]
    bonus = apply_ml_bonus(upset.upset_score, upset.target_lanes, win_probs, lane_order)
    rec["ml_bonus"] = asdict(bonus)
    rec["target"]["rebuild_target_post"] = list(bonus.target_post)
    if legacy_target is not None:
        rec["target"]["rebuild_post_equals_legacy"] = list(legacy_target) == list(bonus.target_post)

    if not odds_map:
        rec["outcome"] = OUTCOME_NO_ODDS
        return rec

    patterns = generate_patterns(list(bonus.target_post), bonus.score_post)
    ctx = PredictionContext(
        patterns=patterns,
        ml_probs=win_probs,
        odds_map=odds_map,
        boats=list(legacy_boats),
        upset_score=bonus.score_post,
        weather=weather,
        has_exhibition=False,
        target_lanes=list(bonus.target_post),
        odds_dropped=[],
    )
    # 条件3・5：組み立ての検査（同じ値を組み立てているため検証力は無い。F-07）
    _check(cond, "c3_post_values",
           ctx.upset_score == bonus.score_post and list(ctx.target_lanes) == list(bonus.target_post),
           "assembly_check_only")
    _check(cond, "c5_win_probs",
           ctx.ml_probs is win_probs and dict(ctx.ml_probs) == dict(evaluation.win_probs),
           "assembly_check_only")

    capture = capture_factory()
    state: dict[str, Any] = {"raw": None, "exc": None}

    def _evaluate_bets_observed(**kwargs: Any) -> Any:
        rec["evaluate_bets_call_count"] += 1
        with capture:
            try:
                result = evaluate_bets(**kwargs)
            except Exception as exc:  # noqa: BLE001 — 区分に記録して外へ伝える
                state["exc"] = type(exc).__name__
                raise
        state["raw"] = result
        return result

    ev_capture = _EvaluateBetsCapture(evaluate_bets=_evaluate_bets_observed)
    provider = LegacyPredictionProvider(
        context_resolver=lambda _ev: ctx, evaluate_bets=ev_capture,
    )

    def _capture_summary() -> dict[str, Any]:
        return {
            "status": capture.status,
            "call_count": capture.call_count,
            "return_count": capture.return_count,
            "errors": list(capture.errors),
            "call_context": capture.call_context,
            "call_candidates": capture.call_candidates,
            "return_candidates": capture.return_candidates,
            "return_value": capture.return_value,
            "call_candidates_sha256": _sha256_json(capture.call_candidates),
            "call_context_sha256": _sha256_json(capture.call_context),
            "return_candidates_sha256": _sha256_json(capture.return_candidates),
            "return_value_sha256": _sha256_json(capture.return_value),
            "cfg_sha256": _sha256_json(capture.return_cfg) if capture.return_cfg is not None else None,
        }

    try:
        prediction = provider.provide(evaluation, buy_config)
    except Exception as exc:  # noqa: BLE001 — 区分を分ける（F-03）
        rec["capture"] = _capture_summary()
        if state["exc"] is not None:
            rec["outcome"] = OUTCOME_LEGACY_EVALUATE_BETS_EXCEPTION
            rec["exception"] = state["exc"]
        elif state["raw"] is not None and len(state["raw"]) == 0:
            rec["outcome"] = OUTCOME_NO_CANDIDATE
        else:
            rec["outcome"] = OUTCOME_PREDICTION_MAPPING_ERROR
            rec["exception"] = type(exc).__name__
        return rec
    raw = list(ev_capture.last_purchase_result(evaluation.eval_id).purchases)

    rec["capture"] = _capture_summary()
    if capture.status == "legacy_exception":
        rec["outcome"] = OUTCOME_LEGACY_EXCEPTION
        return rec
    if capture.status != STATUS_CAPTURED:
        rec["outcome"] = OUTCOME_CAPTURE_FAILED
        return rec
    # F-02：Legacy の設定を写し取れなければ、既定値・引数で埋めずに捕捉失敗とする
    if capture.return_cfg is None:
        rec["capture"]["errors"].append("cfg_not_captured")
        rec["outcome"] = OUTCOME_CAPTURE_FAILED
        return rec

    # 条件7：pred_combo がちょうど1件
    matches = [c for c in capture.call_candidates if c.get("combo") == prediction.pred_combo]
    _check(cond, "c7_pred_combo_once", len(matches) == 1, f"count={len(matches)}")
    if len(matches) != 1:
        rec["outcome"] = OUTCOME_INPUT_DEFECT
        return rec
    cap_cand = matches[0]
    cap_ctx = capture.call_context

    # F-19：Prediction の prob・ev・odds と、写し取った呼び出し時の候補の値
    _check(cond, "c7b_prediction_values",
           all(getattr(prediction, a) == cap_cand.get(k)
               for a, k in (("pred_prob", "prob"), ("pred_ev", "ev"), ("pred_odds", "odds"))),
           f"prediction=({prediction.pred_prob!r},{prediction.pred_ev!r},{prediction.pred_odds!r})")

    try:
        buy_input = BuyScoreInput(
            eval_id=evaluation.eval_id,
            pred_combo=prediction.pred_combo,
            uncertainty=cap_cand.get("uncertainty"),
            disagreement=cap_cand.get("disagreement"),
            composite=cap_cand.get("composite"),
            has_exhibition=cap_ctx.get("has_exhibition"),
            market_gap=cap_ctx.get("market_gap"),
            match_index_approx=cap_ctx.get("match_index_approx"),
        )
    except G3InputValidationError as exc:
        _check(cond, "buy_input_valid", False, str(exc))
        rec["outcome"] = OUTCOME_INPUT_DEFECT
        return rec

    # 写し取りの値と戻り値の値の一致（04 T13）
    raw_same = [c for c in raw if c.get("combo") == prediction.pred_combo]
    if raw_same:
        r0 = raw_same[0]
        _check(cond, "t13_capture_vs_return",
               all(r0.get(k) == cap_cand.get(k) for k in ("uncertainty", "disagreement", "composite")))

    # 条件6：Legacy が受け取った context と、Rebuild の入力・期待値
    rebuild_ctx = build_input_context(evaluation, bonus, buy_input)
    diffs6 = [k for k in ("upset_score", "match_index") if cap_ctx.get(k) != rebuild_ctx.get(k)]
    _check(cond, "c6_context", not diffs6, f"diff_keys={diffs6}")
    # F-07：has_exhibition は PredictionContext に渡した値と、approx は Legacy の既定の挙動（True）と比べる
    _check(cond, "c6_flags",
           cap_ctx.get("has_exhibition") is ctx.has_exhibition and cap_ctx.get("match_index_approx") is True,
           f"has_exhibition={cap_ctx.get('has_exhibition')!r} approx={cap_ctx.get('match_index_approx')!r}")
    _check(cond, "context_keys", tuple(sorted(cap_ctx)) == tuple(sorted(LEGACY_CONTEXT_KEYS)),
           f"keys={sorted(cap_ctx)}")

    if not all(v["ok"] for v in cond.values()):
        rec["outcome"] = OUTCOME_INPUT_DEFECT
        return rec

    # 設定（本人の判断 2026-11-12：(a) Legacy が実際に読み込んだ設定で評価し、出どころと freeze 版との一致を記録）
    legacy_cfg = capture.return_cfg
    freeze = _freeze_config()
    rec["config"] = {
        "source": "captured_legacy_cfg",
        "sha256": _sha256_json(legacy_cfg),
        "version": str(legacy_cfg.get("_version", "")) if isinstance(legacy_cfg, dict) else None,
        "matches_freeze": (legacy_cfg == freeze) if freeze is not None else None,
        "freeze_file_sha256": FREEZE_CONFIG_SHA256 if freeze is not None else None,
        "buy_config_equal": legacy_cfg == buy_config,
    }
    try:
        rebuild = assess_with_input(evaluation, prediction, legacy_cfg, bonus, buy_input)
    except ValidationError as exc:
        _check(cond, "assess_valid", False, str(exc))
        rec["outcome"] = OUTCOME_INPUT_DEFECT
        return rec
    rec["rebuild"] = {
        "buyscore": rebuild.buyscore,
        "investment_type": rebuild.investment_type,
        "kelly_fraction": rebuild.kelly_fraction,
        "skip_reason": rebuild.skip_reason,
        "config_version": rebuild.config_version,
    }

    # Legacy の比較基準値（追補1 §3 の1：pred_combo の候補から取る）
    import x_buyscore  # Legacy（読み込むだけ）
    ret = capture.return_value or {}
    ret_cands = capture.return_candidates or []
    top = max(ret_cands, key=lambda c: c.get("buyscore", float("-inf"))) if ret_cands else None
    same = [c for c in ret_cands if c.get("combo") == prediction.pred_combo]
    legacy_c = same[0] if same else None
    rec["legacy"] = {
        "pred_combo_candidate": (
            {"buyscore": legacy_c.get("buyscore"), "kelly": legacy_c.get("kelly")} if legacy_c else None
        ),
        "buyscore_top_combo": top.get("combo") if top else None,
        "passthrough": ret.get("passthrough"),
        "passthrough_reason": ret.get("passthrough_reason"),
        "capital_reason": ret.get("capital_reason"),
        # Legacy は investment_type を保存しない。写し取った入力と Legacy の buyscore から再計算した値
        "investment_type_recomputed": (
            x_buyscore.investment_type(
                legacy_c.get("buyscore"), cap_cand.get("ev", 0.0), cap_cand.get("odds", 0.0),
                cap_ctx.get("race_type", "混戦"),
            ) if legacy_c and legacy_c.get("buyscore") is not None else None
        ),
    }
    if legacy_c is None or top is None or top.get("combo") != prediction.pred_combo:
        rec["outcome"] = OUTCOME_PREDICTION_DRIFT
        return rec

    # 4項目の一致（仮の表示。連続カウントは行わない）
    legacy_skip = ret.get("passthrough_reason") if ret.get("passthrough") else None
    rec["match"] = {
        "buyscore": rebuild.buyscore == legacy_c.get("buyscore"),
        "kelly_fraction": rebuild.kelly_fraction == legacy_c.get("kelly"),
        "skip_reason": rebuild.skip_reason == legacy_skip,
        "investment_type": rebuild.investment_type == rec["legacy"]["investment_type_recomputed"],
    }
    rec["match"]["all"] = all(rec["match"].values())
    rec["outcome"] = OUTCOME_COMPARED
    return rec
