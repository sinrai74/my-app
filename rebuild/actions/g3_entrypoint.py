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

Legacy を呼ぶ間は shadow.g3_isolation の境界（N-01）を張る。境界を張れなければ Legacy を呼ばず、
実行中に境界が想定外の通信・外部コマンド・外への書き込みを止めた場合は比較を成立扱いにしない。

既存の build_bundle（actions/shadow_entrypoint.py）と本番入口は変更しない。
正式G2 の評価側 baseline の42ファイルからは import しない新規ファイル。

この関数は G3 の連続カウントを行わない（100連続の定義・候補0件の扱いなどは未決定。
第3版 §12）。記録の区分だけを付ける。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from core.engine import Ver4Engine, _grade_to_number
from core.exceptions import ValidationError
from core.g3_assess import assess_with_input, build_input_context
from core.ml_bonus import apply_ml_bonus
from core.upset import UpsetResult, calculate_upset_score
from models.evaluation import RaceEvaluation
from models.g3_input import BuyScoreInput, G3InputValidationError
from shadow.g3_capture import STATUS_CAPTURED, ApplyBuyscoreCapture
from shadow.g3_isolation import G3Isolation
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
OUTCOME_ISOLATION_NOT_ESTABLISHED = "isolation_not_established"  # N-01：境界を張れない（Legacy は呼ばない）
OUTCOME_ISOLATION_VIOLATION = "isolation_violation"  # N-01：実行中に境界が想定外の操作を止めた
OUTCOME_CONFIG_SOURCE_UNKNOWN = "config_source_unknown"  # N-06：設定の出どころを識別できない

# 区分の優先順位（F-13）：条件1の不成立（入力不備）を最優先し、その後に
# no_prediction → no_odds → 境界の不成立 → 境界の違反 → _evaluate_bets の結果 → 写し取り
# → 同一性 → 設定の出どころ → 比較 の順で判定する。

LEGACY_CONFIG_FILE = "buyscore_config.json"           # x_buyscore.CONFIG_FILE（作業ディレクトリからの相対パス）
LEGACY_CONFIG_DEFAULT_FILE = "buyscore_config.default.json"  # x_buyscore.CONFIG_DEFAULT_FILE

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


def _boat_dict(b: Any) -> Any:
    if dataclasses.is_dataclass(b) and not isinstance(b, type):
        return dataclasses.asdict(b)
    if isinstance(b, Mapping):
        return dict(b)
    return {"lane": getattr(b, "lane", None), "repr_type": type(b).__name__}


def _candidate_ids(cands: Optional[Sequence[Mapping[str, Any]]]) -> Optional[list[dict[str, Any]]]:
    if cands is None:
        return None
    return [{"index": i, "combo": c.get("combo"), "sha256": _sha256_json(c)} for i, c in enumerate(cands)]


def read_workdir_config_files() -> dict[str, Any]:
    """作業ディレクトリ（境界の中）の設定のファイルを、Legacy とは独立に読む（A-02）。

    戻り値：名前 → {"sha256", "parsed"（JSON の辞書）, "parse_error"}。無ければ None。
    """
    out: dict[str, Any] = {}
    for name in (LEGACY_CONFIG_FILE, LEGACY_CONFIG_DEFAULT_FILE):
        p = Path(name)
        if not p.is_file():
            out[name] = None
            continue
        data = p.read_bytes()
        item: dict[str, Any] = {"sha256": hashlib.sha256(data).hexdigest(), "parsed": None,
                                "parse_error": None}
        try:
            parsed = json.loads(data.decode("utf-8"))
            if isinstance(parsed, dict):
                item["parsed"] = parsed
            else:
                item["parse_error"] = "not_a_dict"  # G-26：型名を格納しない
        except (UnicodeDecodeError, ValueError) as exc:
            item["parse_error"] = "unicode_decode_error" if isinstance(exc, UnicodeDecodeError) else "value_error"  # G-26
        out[name] = item
    return out


def determine_config_source(files_start: Mapping[str, Any], files_end: Mapping[str, Any],
                            captured_cfg: Any, reload_cfg: Any,
                            release_calls: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """N-06：Legacy が使った設定の出どころ。完全に一致する出どころを識別できなければ "unknown"。

    境界の中では Release からの取得は層1 で必ず失敗する。x_buyscore.load_config の分岐
    （_ensure_config_file → ローカルのファイル → 既定値）のどれを通ったかを、
    実行の前後に独立に読んだファイルの中身と、写し取った設定・境界の中での読み直しとの
    **完全な一致**で判定する（A-02：壊れたファイル・一部の項目だけのファイルで既定値が補われた
    場合は、どのファイルとも一致しないので unknown）。
      workdir_file          ：buyscore_config.json が開始時からあり、前後で変わらず、中身＝写し取った設定
      default_template_copy ：buyscore_config.json が無く、ひな形（default.json）の中身＝写し取った設定
      builtin_defaults      ：どちらのファイルも前後で無く、読み直し＝写し取った設定
    """
    cs, ce = files_start.get(LEGACY_CONFIG_FILE), files_end.get(LEGACY_CONFIG_FILE)
    ds = files_start.get(LEGACY_CONFIG_DEFAULT_FILE)
    calls = [c for c in release_calls
             if c.get("function") == "download_file" and list(c.get("args") or [])[:1] == [LEGACY_CONFIG_FILE]]
    detail: dict[str, Any] = {
        # 層1 の差し替えは常に止めるので、回数は参考の記録（判定には使わない）
        "release_download_attempts": len(calls),
        "workdir_config_sha256_start": cs and cs["sha256"],
        "workdir_config_sha256_end": ce and ce["sha256"],
        "workdir_config_parse_error_start": cs and cs["parse_error"],
        "workdir_default_sha256_start": ds and ds["sha256"],
        "workdir_default_parse_error_start": ds and ds["parse_error"],
        "reload_equal": captured_cfg is not None and reload_cfg == captured_cfg,
        "file_equals_captured": None,
        "reason": None,
    }
    source = "unknown"
    if captured_cfg is None:
        detail["reason"] = "no_captured_cfg"
    elif not detail["reload_equal"]:
        detail["reason"] = "reload_mismatch"
    elif cs is not None:
        if ce is None or ce["sha256"] != cs["sha256"]:
            detail["reason"] = "config_file_changed"
        elif cs["parsed"] is None:
            detail["reason"] = "config_file_invalid"
        elif cs["parsed"] != captured_cfg:
            detail["file_equals_captured"] = False
            detail["reason"] = "config_file_not_equal_captured"
        else:
            detail["file_equals_captured"] = True
            source = "workdir_file"
    elif ds is not None:
        if ds["parsed"] is None:
            detail["reason"] = "default_file_invalid"
        elif ds["parsed"] != captured_cfg:
            detail["file_equals_captured"] = False
            detail["reason"] = "default_file_not_equal_captured"
        elif ce is None or ce["sha256"] != ds["sha256"]:
            detail["reason"] = "default_copy_not_observed"
        else:
            detail["file_equals_captured"] = True
            source = "default_template_copy"
    elif ce is None and files_end.get(LEGACY_CONFIG_DEFAULT_FILE) is None:
        source = "builtin_defaults"
    else:
        detail["reason"] = "config_file_appeared"
    detail["source_file_sha256"] = {"workdir_file": cs and cs["sha256"],
                                    "default_template_copy": ds and ds["sha256"]}.get(source)
    return {"source": source, "source_detail": detail}


def _check(cond: dict, name: str, ok: bool, check_detail: Any = None) -> None:  # G-27：detail→check_detail
    cond[name] = {"ok": bool(ok), "check_detail": check_detail}


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
    isolation_factory: Callable[[], G3Isolation],
) -> dict[str, Any]:
    """1レースの G3 評価。戻り値は JSON にできる記録（辞書）。連続カウントは行わない。

    isolation_factory は必須（N-01）。作業フォルダに何を置くか（設定・成績・モデル）は呼び出し側が決める。
    """
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
            "status": None,
        },
        "inputs": None,
        "evaluate_bets_call_count": 0,
        "evaluate_bets_args": None,
        "sequence": [],
        "win_probs": None,
        "patterns": None,
        "prediction": None,
        "race_type": None,
        "candidates": None,
        "isolation": None,
        "capture": None,
        "config": None,
        "rebuild": None,
        "legacy": None,
        "comparison": None,
        "match": None,
        # 第3版 §12 の8・9：比較方法は未決定。match は == による仮の表示で、判定の定義ではない
        "match_note": "provisional_equality_not_a_judgment_rule",
    }
    cond = rec["conditions"]

    # 条件1：加点前の upset の一致（区分の優先順位の最上位。F-13）
    _check(cond, "c1_upset_pre", upset.upset_score == evaluation.upset_score,
           {"upset": upset.upset_score, "evaluation": evaluation.upset_score})
    if not cond["c1_upset_pre"]["ok"]:
        rec["outcome"] = OUTCOME_INPUT_DEFECT
        return rec

    # 条件8：win_probs
    win_probs = evaluation.win_probs
    if not win_probs:
        rec["outcome"] = OUTCOME_NO_PREDICTION
        return rec
    rec["win_probs"] = {str(k): v for k, v in dict(win_probs).items()}

    lane_order = [b.lane for b in legacy_boats]
    rec["inputs"] = {
        "lane_order": list(lane_order),
        "legacy_boats_sha256": _sha256_json([_boat_dict(b) for b in legacy_boats]),
        "legacy_boats": [_boat_dict(b) for b in legacy_boats],
        "odds_map_count": len(odds_map or {}),
        "odds_map_sha256": _sha256_json(dict(odds_map or {})),
        "odds_map": dict(odds_map or {}),
    }
    bonus = apply_ml_bonus(upset.upset_score, upset.target_lanes, win_probs, lane_order)
    rec["sequence"].append("ml_bonus")
    rec["ml_bonus"] = {"score_pre": upset.upset_score, "target_pre": list(upset.target_lanes),
                       **asdict(bonus)}
    rec["target"]["rebuild_target_post"] = list(bonus.target_post)
    if legacy_target is not None:
        rec["target"]["rebuild_post_equals_legacy"] = list(legacy_target) == list(bonus.target_post)
        rec["target"]["status"] = "compared"
    else:
        rec["target"]["status"] = "legacy_target_not_provided"  # 呼び出し側が渡さない限り照合しない

    if not odds_map:
        rec["outcome"] = OUTCOME_NO_ODDS
        return rec

    patterns = generate_patterns(list(bonus.target_post), bonus.score_post)
    rec["sequence"].append("generate_patterns")
    rec["patterns"] = {"input_target": list(bonus.target_post), "input_score": bonus.score_post,
                       "sha256": _sha256_json(patterns),
                       "body": json.loads(json.dumps(patterns, ensure_ascii=False, default=str))}
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
        rec["sequence"].append("evaluate_bets")
        # B-05：Legacy に実際に渡した入力（大きなものは上の inputs・patterns・win_probs に記録済み）
        rec["evaluate_bets_args"] = {
            **{k: v for k, v in kwargs.items()
               if isinstance(v, (str, int, float, bool)) or v is None},
            "weather": _boat_dict(kwargs.get("weather")) if kwargs.get("weather") is not None else None,
            "weather_type": type(kwargs.get("weather")).__name__,
            "target_lanes": list(kwargs.get("target_lanes") or []),
            "odds_dropped": list(kwargs.get("odds_dropped") or []),
            # C-12：実際に渡した大きな入力の SHA-256（記録の inputs・patterns・win_probs との同一性の照合用）
            "patterns_sha256": _sha256_json(kwargs.get("patterns")),
            "ml_probs_sha256": _sha256_json({str(k): v for k, v in dict(kwargs.get("ml_probs") or {}).items()}),
            "odds_map_sha256": _sha256_json(dict(kwargs.get("odds_map") or {})),
            "boats_sha256": _sha256_json([_boat_dict(b) for b in (kwargs.get("boats") or [])]),
        }
        ea = rec["evaluate_bets_args"]
        ea["matches_record"] = {
            "patterns": ea["patterns_sha256"] == (rec.get("patterns") or {}).get("sha256"),
            "ml_probs": ea["ml_probs_sha256"] == _sha256_json(rec.get("win_probs") or {}),
            "odds_map": ea["odds_map_sha256"] == (rec.get("inputs") or {}).get("odds_map_sha256"),
            "boats": ea["boats_sha256"] == (rec.get("inputs") or {}).get("legacy_boats_sha256"),
        }
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
        # G-23：capture.errors のクラス名を unexpected_error に正規化（固定ラベルは維持。凍結 g3_capture.py は不変）
        _errors = [e if e in ("existing_profiler", "cfg_not_captured", "last_purchase_result_error")
                   else "unexpected_error" for e in capture.errors]
        return {
            "status": capture.status,
            "call_count": capture.call_count,
            "return_count": capture.return_count,
            "errors": _errors,
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

    # N-01：Legacy を呼ぶ間だけ境界を張る。境界の中で、Legacy が読み込む設定を読み直す（N-06）
    iso = isolation_factory()
    out: dict[str, Any] = {"prediction": None, "provide_exc": None, "raw": None, "raw_error": None,
                           "reload_cfg": None, "reload_error": None, "files_start": {}, "files_end": {}}
    with iso:
        if iso.established:
            out["files_start"] = read_workdir_config_files()
            try:
                out["prediction"] = provider.provide(evaluation, buy_config)
            except Exception as exc:  # noqa: BLE001 — 区分を分ける（F-03）
                out["provide_exc"] = exc
            if out["provide_exc"] is None:
                try:
                    out["raw"] = list(ev_capture.last_purchase_result(evaluation.eval_id).purchases)
                except Exception as exc:  # noqa: BLE001 — N-09：区分に記録する
                    out["raw_error"] = type(exc).__name__
            iso.phase = "config_reload"
            try:
                import x_buyscore as _xb  # Legacy（境界の中で読み込むだけ）

                out["reload_cfg"] = _xb.load_config()
            except Exception as exc:  # noqa: BLE001
                out["reload_error"] = "unexpected_error"  # G-25：例外クラス名を格納しない
            out["files_end"] = read_workdir_config_files()
            iso.phase = "run_end"
    rec["isolation"] = iso.summary()
    if not iso.established:
        rec["outcome"] = OUTCOME_ISOLATION_NOT_ESTABLISHED
        return rec
    if not iso.verified:
        rec["capture"] = _capture_summary()
        rec["outcome"] = OUTCOME_ISOLATION_VIOLATION
        return rec

    prediction = out["prediction"]
    if out["provide_exc"] is not None:
        exc = out["provide_exc"]
        rec["capture"] = _capture_summary()
        if state["exc"] is not None:
            rec["outcome"] = OUTCOME_LEGACY_EVALUATE_BETS_EXCEPTION
            rec["exception"] = "unexpected_error"  # G-22：区分は outcome が保持
        elif state["raw"] is not None and len(state["raw"]) == 0:
            rec["outcome"] = OUTCOME_NO_CANDIDATE
        else:
            rec["outcome"] = OUTCOME_PREDICTION_MAPPING_ERROR
            rec["exception"] = "unexpected_error"  # G-22：区分は outcome が保持
        return rec
    rec["capture"] = _capture_summary()
    if out["raw_error"] is not None:
        rec["capture"]["errors"].append("last_purchase_result_error")  # G-24：クラス名を格納しない
        rec["outcome"] = OUTCOME_CAPTURE_FAILED
        return rec
    raw = out["raw"]
    rec["prediction"] = {
        "pred_combo": prediction.pred_combo, "pred_prob": prediction.pred_prob,
        "pred_ev": prediction.pred_ev, "pred_odds": prediction.pred_odds,
        "confidence": prediction.confidence, "patterns_count": len(prediction.patterns),
    }
    rec["candidates"] = {"call": _candidate_ids(capture.call_candidates),
                         "return": _candidate_ids(capture.return_candidates),
                         "evaluate_bets_return": _candidate_ids(raw)}
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
    _check(cond, "c7_pred_combo_once", len(matches) == 1, {"count": len(matches)})
    if len(matches) != 1:
        rec["outcome"] = OUTCOME_INPUT_DEFECT
        return rec
    cap_cand = matches[0]
    cap_ctx = capture.call_context

    # F-19：Prediction の prob・ev・odds と、写し取った呼び出し時の候補の値
    _check(cond, "c7b_prediction_values",
           all(getattr(prediction, a) == cap_cand.get(k)
               for a, k in (("pred_prob", "prob"), ("pred_ev", "ev"), ("pred_odds", "odds"))),
           {"pred_prob": prediction.pred_prob, "pred_ev": prediction.pred_ev, "pred_odds": prediction.pred_odds})

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
        _check(cond, "buy_input_valid", False, "input_validation_failed")  # G-28：例外本文を格納しない
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
    rec["race_type"] = {"legacy_context": cap_ctx.get("race_type"),
                        "rebuild_input_context": rebuild_ctx.get("race_type"),
                        # 印だけ（判定の定義ではない。第3版 §8 の「影響しないことの確認」は未実装）
                        "equal": cap_ctx.get("race_type") == rebuild_ctx.get("race_type")}
    diffs6 = [k for k in ("upset_score", "match_index") if cap_ctx.get(k) != rebuild_ctx.get(k)]
    _check(cond, "c6_context", not diffs6, {"diff_keys": diffs6})
    # F-07：has_exhibition は PredictionContext に渡した値と、approx は Legacy の既定の挙動（True）と比べる
    _check(cond, "c6_flags",
           cap_ctx.get("has_exhibition") is ctx.has_exhibition and cap_ctx.get("match_index_approx") is True,
           {"has_exhibition": cap_ctx.get("has_exhibition"), "match_index_approx": cap_ctx.get("match_index_approx")})
    _check(cond, "context_keys", tuple(sorted(cap_ctx)) == tuple(sorted(LEGACY_CONTEXT_KEYS)),
           {"keys": sorted(cap_ctx)})

    if not all(v["ok"] for v in cond.values()):
        rec["outcome"] = OUTCOME_INPUT_DEFECT
        return rec

    # 設定（本人の判断 2026-11-12：(a) Legacy が実際に読み込んだ設定で評価し、出どころと freeze 版との一致を記録）
    legacy_cfg = capture.return_cfg
    freeze = _freeze_config()
    src = determine_config_source(out["files_start"], out["files_end"], legacy_cfg, out["reload_cfg"],
                                  rec["isolation"]["release_calls"])
    rec["config"] = {
        "source": src["source"],
        "source_detail": {**src["source_detail"], "reload_error": out["reload_error"]},
        "captured_from": "apply_buyscore_local_cfg",
        "body": legacy_cfg,
        "sha256": _sha256_json(legacy_cfg),
        "version": str(legacy_cfg.get("_version", "")) if isinstance(legacy_cfg, dict) else None,
        "matches_freeze": (legacy_cfg == freeze) if freeze is not None else None,
        "freeze_file_sha256": FREEZE_CONFIG_SHA256 if freeze is not None else None,
        "buy_config_equal": legacy_cfg == buy_config,
    }
    if src["source"] == "unknown":
        rec["outcome"] = OUTCOME_CONFIG_SOURCE_UNKNOWN
        return rec
    try:
        rebuild = assess_with_input(evaluation, prediction, legacy_cfg, bonus, buy_input)
    except ValidationError as exc:
        _check(cond, "assess_valid", False, "assess_validation_failed")  # G-28：例外本文を格納しない
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
    rec["comparison"] = [
        {"item": "buyscore", "rebuild": rebuild.buyscore, "legacy": legacy_c.get("buyscore")},
        {"item": "kelly_fraction", "rebuild": rebuild.kelly_fraction, "legacy": legacy_c.get("kelly")},
        {"item": "skip_reason", "rebuild": rebuild.skip_reason, "legacy": legacy_skip},
        {"item": "investment_type", "rebuild": rebuild.investment_type,
         "legacy": rec["legacy"]["investment_type_recomputed"]},
    ]
    rec["outcome"] = OUTCOME_COMPARED
    return rec
