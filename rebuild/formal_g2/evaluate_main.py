"""正式G2：評価の入口。

起動（リポジトリ直下で）：
  PYTHONPATH="rebuild:<repository-root>" python -P -B -m formal_g2.evaluate_main \
      --race-date YYYYMMDD --mode investigation|formal [--permission-id ID] [--baseline PATH]

処理の順序：
  1 観測の開始 → 2 実行前の検査（許可ID・日付・トークン・探索順・解決先・tests・config の解決先）
  → 3 評価コードの読み込み（comparator は直接読み込む）と shadow・config の検査
  → 4 Release が使えないこと → 5 cfg の読み込み → 6 M7 → 7 会場統計・会場設定の事前読み込み
  → 8 一覧の固定（GET 1回）→ 9 各レースの評価（区切りごとに config を検査）→ 10 終了時の検査
  → 11 run.json・stop.json（停止時）・完了目録

- 加算・連続数は判定しない（記録だけ。§6.3〜§6.5 の 8）。
- 停止した場合は stop.json を書き、終了コード 2 で終わる。停止記録はレースごとの記録と分ける。
- 書き込むのは --out のディレクトリだけ（artifact にするもの）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from formal_g2.instrument import Instrument, repo_modules  # 標準ライブラリだけに依存

JST = timezone(timedelta(hours=9))
EXIT_STOPPED = 2


def _now() -> str:
    return datetime.now(JST).isoformat()


def _run_info(args: argparse.Namespace) -> dict[str, Any]:
    env = os.environ
    return {
        "race_date": args.race_date,
        "mode": args.mode,
        "permission_id": args.permission_id,
        "github_run_id": env.get("GITHUB_RUN_ID"),
        "github_run_attempt": env.get("GITHUB_RUN_ATTEMPT"),
        "github_ref": env.get("GITHUB_REF"),
        "github_sha": env.get("GITHUB_SHA"),
        "github_workflow_ref": env.get("GITHUB_WORKFLOW_REF"),
        "started_at": _now(),
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="formal_g2.evaluate_main")
    p.add_argument("--race-date", required=True)
    p.add_argument("--mode", required=True, choices=("investigation", "formal"))
    p.add_argument("--permission-id", default="")
    p.add_argument("--baseline", default="", help="登録済みの基準 JSON（正式な実行では必須）")
    p.add_argument("--out", default="g2_out")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = os.path.realpath(os.getcwd())

    # 1. 観測の開始（評価対象のモジュールを読み込む前）
    instrument = Instrument(repo_root)
    instrument.start()
    instrument.set_phase("guard")

    from formal_g2 import guard
    from formal_g2.manifest import write_manifest
    from formal_g2.records import RecordWriter

    writer = RecordWriter(args.out)
    run: dict[str, Any] = {"info": _run_info(args), "status": "running", "stop_ref": None, "checks": {}}
    state: dict[str, Any] = {"targets": [], "last_started": None, "list_fixed": False,
                             "summary": {"races": 0, "d13_2_all_true": 0, "diff_zero": 0, "w7_ok": 0}}

    def finish(status: str, stop: Optional[dict] = None) -> int:
        instrument.stop()
        run["status"] = status
        run["finished_at"] = _now()
        run["checks"]["config_at_end"] = guard.config_loaded(repo_root)
        run["checks"]["tests_not_loaded"] = guard.tests_not_loaded(repo_root)
        run["repo_modules_loaded"] = repo_modules(repo_root)
        run["instrument"] = instrument.snapshot()
        run["config_detection"] = {
            "detected": instrument.config_executed or run["checks"]["config_at_end"]["loaded"],
            "executed_module_code": instrument.config_executed,
            "loaded_at_end": run["checks"]["config_at_end"]["loaded"],
        }
        ident = run["checks"].get("baseline_identity")
        if ident is not None and ident.get("sha256") is not None:
            end = _baseline_identity(args.baseline)
            ident["sha256_at_end"] = end["sha256"]
            ident["unchanged_during_run"] = end["sha256"] == ident["sha256"]
            if stop is None and not ident["unchanged_during_run"]:
                stop = _stop_record("A-1", "baseline changed during run", "end", state, writer,
                                    {"baseline_identity": ident})
                status = run["status"] = "stopped"
        if stop is None and run["config_detection"]["detected"]:
            stop = _stop_record("CONFIG", "config detected at end of run", "end", state, writer,
                                {"mechanism": "end_check", **run["checks"]["config_at_end"],
                                 "detections": list(instrument.detections)})
            status = run["status"] = "stopped"
        writer.close()
        if stop is not None:
            writer.write_json("stop.json", stop)
            run["stop_ref"] = "stop.json"
        run["w7"] = _w7_summary(run["instrument"])
        writer.write_json("run.json", run)
        write_manifest(args.out, run["info"], writer.written_eval_ids, state["targets"], complete=True)
        _print_summary(run, writer, state["summary"])
        return 0 if status == "completed" else EXIT_STOPPED

    def stop(code: str, reason: str, detail: Optional[dict] = None) -> int:
        return finish("stopped", _stop_record(code, reason, instrument.phase, state, writer, detail))

    def w7_boundary(where: str) -> Optional[int]:
        """レースの区切りでの監視の確認。禁止対象の呼び出し・監視の異常があれば止める（両モード）。"""
        instrument.check_hooks(where)
        dynamic = [p for p in instrument.monitoring_problems()
                   if not p.startswith(("unresolved targets", "profiler already present", "threads at start"))]
        if dynamic:
            return stop("W7", f"monitoring problem at {where}", {"problems": dynamic})
        return None

    # 1b. 監視の前提（fail-closed）。正式な実行では、監視を保証できなければ始めない。
    static = [p for p in instrument.monitoring_problems()
              if p.startswith(("unresolved targets", "profiler already present", "threads at start"))]
    run["checks"]["monitoring_at_start"] = {"problems": static}
    if args.mode == "formal" and static:
        return stop("W7", "monitoring cannot be guaranteed", {"problems": static})

    # 2. 実行前の検査
    try:
        if args.mode == "formal":
            if not args.permission_id:
                raise guard.GuardStop("A-1", "formal run requires --permission-id")
            run["checks"]["race_date"] = guard.check_race_date_formal(args.race_date)
        else:
            run["checks"]["race_date"] = {"race_date": args.race_date, "restricted": False,
                                          "note": "investigation run: no date restriction"}
        run["checks"]["token"] = guard.check_token_absent()
        run["checks"]["search_path"] = guard.check_search_path(repo_root)
        run["checks"]["colliding_modules"] = guard.check_colliding_modules(repo_root)
        run["checks"]["tests_probe"] = guard.tests_probe(repo_root)
        run["checks"]["config_find_spec"] = guard.config_find_spec_record()
        run["checks"]["config_before_import"] = guard.check_config_absent(repo_root, "before_import")
    except guard.GuardStop as g:
        return stop(g.code, g.reason, g.detail)

    # 3. 評価コードの読み込み（ここでの副作用は phase=import として記録される）
    instrument.set_phase("import")
    try:
        import notify_arashi  # noqa: F401  凍結版（変更しない）
        from x_asahi_scoring import load_asahi_config
        from formal_g2 import race_eval
        from formal_g2.env_baseline import (
            CANDIDATE_A, CANDIDATE_B, baseline_view, check_modules, collect_snapshot,
            compare_to_baseline, formal_g2_eval_files, self_check,
        )
        from formal_g2.programs import ProgramsUnavailable, fix_programs

        comparator = race_eval.load_comparator(repo_root)
    except Exception as exc:  # noqa: BLE001  依存を解決できない場合は止める
        return stop("A-1", f"import failed: {type(exc).__name__}: {exc}")
    instrument.set_phase("guard")
    run["checks"]["shadow_package_loaded"] = race_eval.shadow_package_loaded()
    run["checks"]["comparator_load"] = {
        "method": "importlib.util.spec_from_file_location",
        "alias": race_eval.COMPARATOR_ALIAS,
        "path": "rebuild/shadow/comparator.py",
    }
    try:
        if run["checks"]["shadow_package_loaded"]:
            raise guard.GuardStop("A-1", "shadow package was loaded")
        run["checks"]["config_after_import"] = guard.check_config_absent(repo_root, "after_import")
        # 4. Release が使えないこと
        run["checks"]["release"] = guard.check_release_unavailable()
    except guard.GuardStop as g:
        return stop(g.code, g.reason, g.detail)

    # 5. 1回の実行で1つの設定オブジェクトを両側に渡す
    cfg = load_asahi_config()

    # 6. M7：基準（§1〜§3）
    snapshot = collect_snapshot(repo_root, cfg)
    run["env_snapshot"] = snapshot
    problems = self_check(snapshot)
    run["checks"]["env_self_check"] = problems
    # 基準の同一性：内容の SHA-256 を記録し、集約で同じ内容の基準を使ったことを照合する。
    # 実行の終わりにも計り直し、途中で変わっていないことを記録する（finish）。
    baseline = None
    run["checks"]["baseline_identity"] = _baseline_identity(args.baseline)
    if args.baseline and run["checks"]["baseline_identity"]["read_error"] is None:
        try:
            baseline = json.loads(run["checks"]["baseline_identity"].pop("_raw"))
            if not isinstance(baseline, dict):
                raise ValueError("baseline is not a JSON object")
        except ValueError as exc:
            baseline = None
            run["checks"]["baseline_identity"]["read_error"] = f"{type(exc).__name__}: {exc}"
    run["checks"]["baseline_identity"].pop("_raw", None)
    if args.mode == "formal" and run["checks"]["baseline_identity"]["read_error"] is not None:
        return stop("A-1", "baseline unreadable", {"baseline_identity": run["checks"]["baseline_identity"]})
    run["checks"]["baseline_diffs"] = None if baseline is None else compare_to_baseline(snapshot, baseline)
    allowlist = (baseline or {}).get("module_allowlist")
    if allowlist is None:
        allowlist = sorted(set(CANDIDATE_A) | set(CANDIDATE_B) | set(formal_g2_eval_files(repo_root)))
        run["checks"]["module_allowlist_source"] = "candidate (not approved)"
    else:
        run["checks"]["module_allowlist_source"] = "registered baseline"
    run["checks"]["modules_pre_eval"] = check_modules(repo_modules(repo_root), allowlist)
    run["baseline_candidate"] = baseline_view(snapshot)  # 登録の判断材料（登録はしない）
    if args.mode == "formal":
        if problems:
            return stop("A-1", f"environment not eligible: {problems}")
        if baseline is None:
            return stop("A-1", "no registered baseline (--baseline)")
        if run["checks"]["baseline_diffs"]:
            return stop("A-1", f"baseline mismatch: {run['checks']['baseline_diffs']}")
        if not run["checks"]["modules_pre_eval"]["ok"]:
            return stop("A-1", "repository module outside allowlist was loaded")

    # 7. 会場統計・会場設定の事前読み込み（必ず行う）
    instrument.set_phase("preload")
    try:
        run["preload"] = race_eval.preload_venue_caches(repo_root, args.race_date)
    except race_eval.PreloadStop as p:
        run["preload"] = p.record
        return stop("PRELOAD", p.reason, p.record)
    instrument.set_phase("guard")
    try:
        run["checks"]["config_after_preload"] = guard.check_config_absent(repo_root, "after_preload")
    except guard.GuardStop as g:
        return stop(g.code, g.reason, g.detail)

    # 8. 一覧の固定（C-1：GET 1回）
    instrument.set_phase("programs")
    try:
        fixed = fix_programs(args.race_date)
    except ProgramsUnavailable as exc:
        return stop("LIST", f"programs list could not be fixed: {exc}")
    writer.write_bytes("programs_raw.json", fixed["raw_bytes"])
    run["programs"] = {k: fixed[k] for k in
                       ("url", "status_code", "raw_sha256", "targets", "targets_sha256", "http_get_calls")}
    state["targets"] = [t["eval_id"] for t in fixed["targets"]]
    state["list_fixed"] = True

    # 9. 各レースの評価（C-0・C-2）。config を検出したら新しいレースを始めない。
    instrument.set_phase("evaluate")
    ctx = race_eval.build_context(repo_root, args.race_date, fixed["programs"], cfg, comparator,
                                  instrument, run["preload"])
    for target in fixed["targets"]:
        if instrument.config_executed or "config" in sys.modules:
            return stop("CONFIG", "config detected before starting a race",
                        {"mechanism": "between_races", "next_eval_id": target["eval_id"],
                         "detections": list(instrument.detections), **guard.config_loaded(repo_root)})
        halted = w7_boundary(f"before {target['eval_id']}")
        if halted is not None:
            return halted
        state["last_started"] = target["eval_id"]
        try:
            rec = race_eval.evaluate_race(ctx, target)
        except Exception as exc:  # noqa: BLE001  レースの外の異常は実行全体の停止
            return stop("RUN", f"{type(exc).__name__}: {exc}")
        writer.append_race(rec)
        sm = state["summary"]
        sm["races"] += 1
        sm["d13_2_all_true"] += bool(rec.get("d13_2")) and all(v is True for v in rec["d13_2"].values())
        sm["diff_zero"] += rec.get("result_category") == "diff_zero"
        sm["w7_ok"] += (rec.get("w7") or {}).get("ok") is True
        halted = w7_boundary(f"after {target['eval_id']}")
        if halted is not None:
            return halted
        if rec.get("config_detected"):
            return stop("CONFIG", "config detected during a race",
                        {"mechanism": "race_boundary", "eval_id": target["eval_id"],
                         **rec["config_detected"], **guard.config_loaded(repo_root)})

    # 10. 終了時の検査（config の再検査は finish の中で行う）
    run["checks"]["modules_post_eval"] = check_modules(repo_modules(repo_root), allowlist)
    return finish("completed")


def _w7_summary(inst: dict[str, Any]) -> dict[str, Any]:
    """禁止対象の呼び出しと監視の状態の要約（集約は、この要約ではなく元の記録を独立に検査する）。"""
    forb = inst.get("forbidden") or {}
    return {
        "ok": not inst.get("monitoring_problems"),
        "problems": list(inst.get("monitoring_problems") or []),
        "forbidden_total": forb.get("total"),
        "forbidden_by_category": {c: sum(v.values()) for c, v in (forb.get("counts_by_category") or {}).items()},
        "forbidden_import_phase": forb.get("import_phase_counts"),
        "unresolved_count": len(inst.get("unresolved_targets") or []),
        "monitor_state": {k: v for k, v in (inst.get("monitor_state") or {}).items()
                          if k in ("started", "stopped", "hook_checks")},
        "hook_violations": len((inst.get("monitor_state") or {}).get("hook_violations") or []),
        "thread_starts": len((inst.get("monitor_state") or {}).get("thread_starts") or []),
    }


def _print_summary(run: dict[str, Any], writer: Any, d13: dict[str, int]) -> None:
    """ジョブログ用の短い要約（補助の証拠）。トークン・URL・認証情報は出さない。"""
    w7 = run.get("w7") or {}
    races = getattr(writer, "written_eval_ids", [])
    ident = (run.get("checks") or {}).get("baseline_identity") or {}
    image = (run.get("env_snapshot") or {}).get("image") or {}
    lines = [
        f"G2 summary: status={run.get('status')} stop_ref={run.get('stop_ref')} races_written={len(races)}",
        f"G2 W7: ok={w7.get('ok')} forbidden_total={w7.get('forbidden_total')} "
        f"by_category={w7.get('forbidden_by_category')} import_phase={w7.get('forbidden_import_phase')} "
        f"unresolved={w7.get('unresolved_count')} hook_violations={w7.get('hook_violations')} "
        f"thread_starts={w7.get('thread_starts')} problems={w7.get('problems')}",
        f"G2 13-2: {d13}",
        f"G2 baseline_identity: sha256={ident.get('sha256')} at_end={ident.get('sha256_at_end')} "
        f"unchanged={ident.get('unchanged_during_run')} read_error={'yes' if ident.get('read_error') else 'no'} "
        f"baseline_diffs={(run.get('checks') or {}).get('baseline_diffs')}",
        f"G2 env image (observation only): ImageOS={image.get('ImageOS')} ImageVersion={image.get('ImageVersion')}",
    ]
    for ln in lines:
        print(ln, flush=True)


def _baseline_identity(path: str) -> dict[str, Any]:
    """基準 JSON の内容の SHA-256（パスではなく内容で同一性を見る）。読めなければ read_error。"""
    out: dict[str, Any] = {"path": path or None, "sha256": None, "bytes": None, "read_error": None}
    if not path:
        return out
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as exc:
        out["read_error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["sha256"] = hashlib.sha256(raw).hexdigest()
    out["bytes"] = len(raw)
    out["_raw"] = raw
    return out


def _stop_record(code: str, reason: str, phase: str, state: dict, writer: Any,
                 detail: Optional[dict]) -> dict[str, Any]:
    return {
        "code": code,
        "reason": reason,
        "phase": phase,
        "at": _now(),
        "list_fixed": bool(state.get("list_fixed")),
        "last_started_eval_id": state.get("last_started"),
        "processed_eval_ids_count": len(writer.written_eval_ids),
        "detail": detail or {},
    }


if __name__ == "__main__":
    sys.exit(main())
