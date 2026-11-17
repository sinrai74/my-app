"""
正式G3（B案）段階3：調査用の診断の入口（ネットワークの無い段で動かす）。

- 正式G3 ではない。連続数・合否は判定しない。記録を書くだけ。
- 親のプロセスは、1レースごとに子のプロセスを起動する（B-06・B-07：境界の状態をレースの間で持ち越さない。
  子の SystemExit・KeyboardInterrupt・時間切れ・異常終了は、親が記録する）。
- 子のプロセスの手順（1レース）：
    1. joblib の物理コア数を、境界の外で先に数える（境界の中で lscpu を起動すると違反になるため）。
    2. 準備の境界：model_all.pkl・fan*.txt・motor_history.csv だけを置き、出走表から boats と
       Rebuild の RaceEvaluation を作り、Legacy の `_predict_win_prob` で ML の確率を出す（F-10）。
    3. 評価の境界：freeze 版の設定（da6a4eda…）だけを置き、evaluate_race_g3 を呼ぶ。
       hit_record.csv は置かない（本人の判断 2026-11-13）。
    4. B-03：比較の成立・数える候補の個別の条件の結果を記録する（数え方は未決定。合計の判定はしない）。
- 入力（取得の段で作ったもの）：出走表（programs_raw.json）と、オッズ（odds.json）。
  オッズの時点の扱いは未決定なので、取得の時刻と種類を記録するだけにする。
- weather は取得しない（Legacy の既定の WeatherInfo()）。記録に weather_source を残す。
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

JST = timezone(timedelta(hours=9))
REBUILD_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = REBUILD_DIR.parent
FREEZE_CONFIG_SHA256 = "da6a4edaf8220aa52ed3e4577c23ff443509512a971d0ba322079eb4c4cd6f1d"
PREP_SEED_NAMES = ("model_all.pkl", "motor_history.csv")  # fan*.txt はすべて置く（Legacy は最新の名前を選ぶ）
CHILD_TIMEOUT_SEC = 600
WEATHER_SOURCE = "not_fetched_legacy_default_WeatherInfo"
RECORD_KIND = "g3_stage3_diagnostic_race"

CODE_FILES = (
    "rebuild/actions/g3_diagnostic_entry.py", "rebuild/actions/g3_entrypoint.py",
    "rebuild/shadow/g3_isolation.py", "rebuild/shadow/g3_capture.py", "rebuild/core/g3_assess.py",
    "rebuild/core/ml_bonus.py", "rebuild/models/g3_input.py", "notify_arashi.py", "x_buyscore.py",
    "x_release_storage.py", "x_asahi_scoring.py", "x_venue_stats.py", "rebuild/core/engine.py",
    "buyscore_config.json", "model_all.pkl", "motor_history.csv",
)
PACKAGES = ("numpy", "pandas", "scikit-learn", "lightgbm", "joblib", "requests")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> Optional[str]:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError:
        return None


def environment_manifest() -> dict[str, Any]:
    pk = {}
    for name in PACKAGES:
        try:
            pk[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pk[name] = None
    return {
        "python": sys.version.split()[0],
        "packages": pk,
        "code_sha256": {f: _sha256_file(REPO_DIR / f) for f in CODE_FILES},
        "fan_files": {p.name: _sha256_file(p) for p in sorted(REPO_DIR.glob("fan*.txt"))},
    }


def eval_id_of(race_date: str, venue_num: int, race_number: int) -> str:
    return f"{race_date}_{int(venue_num):02d}_{int(race_number):02d}"


def list_races(race_date: str, programs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in programs:
        vn, rn = int(p["race_stadium_number"]), int(p["race_number"])
        out.append({"eval_id": eval_id_of(race_date, vn, rn), "venue_num": vn, "race_number": rn,
                    "race_closed_at": p.get("race_closed_at")})
    return sorted(out, key=lambda r: (str(r["race_closed_at"] or ""), r["venue_num"], r["race_number"]))


# ---------------------------------------------------------------- 子のプロセス（1レース）


class _FixedFetcher:
    """BoatsProvider に渡す fetcher。固定した一覧だけを返す。"""

    def __init__(self, race_date: str, programs: list) -> None:
        self.race_date, self.programs, self.calls = race_date, programs, []

    def __call__(self, race_date: str) -> list:
        self.calls.append(race_date)
        return self.programs if race_date == self.race_date else []


def _asdict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    return obj


TIME_FIELDS = frozenset({"evaluated_at", "built_at"})  # 実行ごとに変わる時刻（C-13：SHA-256 から除く）


def _evaluation_record(ev: Any, upset: Any) -> dict[str, Any]:
    """F-10：RaceEvaluation の照合・記録（第3版 §2）。sha256 は時刻の欄を除いた内容の SHA-256。"""
    def _drop_times(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _drop_times(v) for k, v in x.items() if k not in TIME_FIELDS}
        if isinstance(x, list):
            return [_drop_times(v) for v in x]
        return x
    body = _drop_times(json.loads(json.dumps(_asdict(ev), ensure_ascii=False, default=str)))
    return {
        "eval_id": ev.eval_id, "race_date": ev.race_date, "venue_num": ev.venue_num,
        "race_number": ev.race_number, "venue_name": getattr(ev, "venue_name", None),
        "upset_score": ev.upset_score, "race_type": getattr(ev, "race_type", None),
        "win_probs": {str(k): v for k, v in dict(ev.win_probs or {}).items()},
        "upset_result": json.loads(json.dumps(_asdict(upset), ensure_ascii=False, default=str)),
        "sha256": _sha256_bytes(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":"), default=str).encode("utf-8")),
    }


def count_candidate_checks(rec: Mapping[str, Any], prep_summary: Optional[Mapping[str, Any]],
                           diag: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """B-03：比較の成立と数える候補の、個別の条件の結果だけ（数え方は未決定。合計の判定はしない）。

    C-10：評価に至らなかった項目は False ではなく None（「不一致」と「評価に至らず」を区別する。
    候補0件の扱いは未決定なので、ここで False にして暗黙に除外しない）。
    """
    cfg = rec.get("config")
    iso = rec.get("isolation")
    match = rec.get("match")
    diag = diag or {}
    odds_meta = diag.get("odds_meta") or {}

    def _v(present: bool, value: Any) -> Any:
        return value if present else None
    return {
        "outcome_compared": _v(bool(rec), rec.get("outcome") == "compared"),
        "comparison_4_items_equal": _v(isinstance(match, dict), (match or {}).get("all") is True),
        "config_source_is_workdir_file": _v(isinstance(cfg, dict), (cfg or {}).get("source") == "workdir_file"),
        "matches_freeze_is_true": _v(isinstance(cfg, dict), (cfg or {}).get("matches_freeze") is True),
        "seeded_config_sha256_is_freeze": _v(isinstance(iso, dict),
                                             ((iso or {}).get("seeded") or {}).get("buyscore_config.json")
                                             == FREEZE_CONFIG_SHA256),
        "eval_isolation_verified": _v(isinstance(iso, dict), (iso or {}).get("verified") is True),
        "prep_isolation_verified": _v(prep_summary is not None, (prep_summary or {}).get("verified") is True),
        "eval_id_consistent": _v("eval_id_consistency" in diag,
                                 (diag.get("eval_id_consistency") or {}).get("evaluation_eval_id_matches") is True),
        "evaluate_bets_call_count_is_1": _v(bool(rec), rec.get("evaluate_bets_call_count") == 1),
        "odds_error": odds_meta.get("error") if odds_meta else None,
        "odds_kind": odds_meta.get("kind") if odds_meta else None,
        "target_status": ((rec.get("target") or {}).get("status")) if rec else None,
        "child_returncode_zero": "pending_parent",   # 親が埋める
        "strace_evidence": "pending_external",       # 別の検査（g3_diagnostic_verify）が埋める
        "note": "individual_results_only_counting_rule_undecided",
    }


def run_child(race_date: str, eval_id: str, programs_raw_path: Path, odds_path: Path) -> dict[str, Any]:
    """1レースの診断。例外は記録に入れる（秘密は記録しない）。"""
    started = datetime.now(JST).isoformat()
    rec: dict[str, Any] = {"kind": RECORD_KIND, "eval_id": eval_id, "race_date": race_date,
                           "started_at": started, "stage": "start", "error": None,
                           "weather_source": WEATHER_SOURCE}
    try:
        from joblib.externals.loky import cpu_count

        cpu_count(only_physical_cores=True)
    except Exception:  # noqa: BLE001 — G-20：例外クラス名を公開レコードに残さない
        rec["joblib_prewarm_error"] = "unexpected_error"

    from shadow.g3_isolation import G3Isolation

    raw = json.loads(programs_raw_path.read_bytes())
    programs = raw["programs"] if isinstance(raw, dict) else raw
    odds_all = json.loads(odds_path.read_text(encoding="utf-8"))
    rec["inputs_sha256"] = {"programs_raw": _sha256_file(programs_raw_path), "odds": _sha256_file(odds_path)}
    race = next((r for r in list_races(race_date, programs) if r["eval_id"] == eval_id), None)
    if race is None:
        rec.update(stage="lookup", error="race_not_in_programs")
        return rec
    odds_entry = (odds_all.get("races") or {}).get(eval_id) or {}
    rec["odds_meta"] = {k: odds_entry.get(k) for k in ("fetched_at", "kind", "count", "error")}
    odds_map = odds_entry.get("odds") or {}

    # 2. 準備の境界（F-10：boats・RaceEvaluation・ML）
    prep_seeds: dict[str, Any] = {n: REPO_DIR / n for n in PREP_SEED_NAMES}
    for p in sorted(REPO_DIR.glob("fan*.txt")):
        prep_seeds[p.name] = p
    prep: dict[str, Any] = {}
    rec["stage"] = "prepare"
    with G3Isolation(prep_seeds) as piso:
        if piso.established:
            try:
                import notify_arashi as N
                import x_venue_stats
                from actions.g3_entrypoint import build_rebuild_evaluation
                from adapters.providers import BoatsProvider
                from models.evaluation import FeatureSet
                from x_asahi_scoring import load_asahi_config

                cfg = load_asahi_config()
                x_venue_stats.compute_venue_course_stats()
                provider = BoatsProvider(programs_fetcher=_FixedFetcher(race_date, programs))
                vn, rn = race["venue_num"], race["race_number"]
                race_obj = provider.resolve_race(race_date, vn, rn)
                engine_boats = list(provider.resolve_boats(race_date, vn, rn))
                _program, boat_objs = provider._get_source(race_date, vn, rn)
                win_probs = N._predict_win_prob(boat_objs)  # F-10：Legacy の predictor を境界の中で呼ぶ
                now = datetime.now(JST)
                fs = FeatureSet(eval_id=eval_id, feature_schema_version=1, built_at=now.isoformat(),
                                boat_features={1: {}}, race_features={}, local_features=None, missing_keys=())
                ev, up = build_rebuild_evaluation(race=race_obj, feature_set=fs, engine_boats=engine_boats,
                                                  config=cfg, venue_stats=x_venue_stats,
                                                  win_probs=win_probs, now=now)
                prep.update(evaluation=ev, upset=up, boats=boat_objs, weather=N.WeatherInfo())
                # C-11：asahi_config.json は作業フォルダに置かない（組み込みの既定値）。その事実と中身の SHA-256
                prep["inputs"] = {
                    "asahi_config_json_in_workdir": Path("asahi_config.json").exists(),
                    "asahi_config_sha256": _sha256_bytes(json.dumps(cfg, ensure_ascii=False, sort_keys=True,
                                                                    default=str).encode("utf-8")),
                    "fan_file_used": N._get_fan_file(),
                }
            except Exception as exc:  # noqa: BLE001
                from shadow.g3_isolation import _free_text_digest

                prep["error"] = "unexpected_error"  # G-19：例外クラス名を格納しない
                # G-18：原本全文（例外の型＋文）から digest。切り詰めない・本文を残さない
                prep["trace_tail"] = _free_text_digest(
                    "".join(traceback.format_exception_only(type(exc), exc)))
    rec["prep_isolation"] = piso.summary()
    rec["prep_inputs"] = prep.get("inputs")
    if not piso.established or not piso.verified or "error" in prep:
        rec.update(stage="prepare", error=prep.get("error") or "prep_isolation_not_verified",
                   prep_trace_tail=prep.get("trace_tail"))
        rec["count_candidate_checks"] = count_candidate_checks({}, rec["prep_isolation"], rec)
        return rec

    ev, up, boats = prep["evaluation"], prep["upset"], prep["boats"]
    # eval_id から boats を引いた結果の照合（F-10）
    rec["evaluation"] = _evaluation_record(ev, up)
    rec["eval_id_consistency"] = {
        "evaluation_eval_id_matches": ev.eval_id == eval_id,
        "boats_lanes": [getattr(b, "lane", None) for b in boats],
        "boats_count": len(boats),
    }
    if ev.eval_id != eval_id:
        rec.update(stage="prepare", error="eval_id_mismatch")
        rec["count_candidate_checks"] = count_candidate_checks({}, rec["prep_isolation"], rec)
        return rec

    # 3. 評価の境界（freeze 版の設定だけ）
    freeze_path = REPO_DIR / "buyscore_config.json"
    if _sha256_file(freeze_path) != FREEZE_CONFIG_SHA256:
        rec.update(stage="evaluate", error="freeze_config_sha_mismatch")
        rec["count_candidate_checks"] = count_candidate_checks({}, rec["prep_isolation"], rec)
        return rec
    import notify_arashi as N  # 準備の境界の中で読み込み済み
    from actions import g3_entrypoint as G

    rec["stage"] = "evaluate"
    g3 = G.evaluate_race_g3(
        evaluation=ev, upset=up, legacy_boats=boats, odds_map=odds_map, weather=prep["weather"],
        buy_config={"_version": "not_used_by_g3_rebuild_side"},
        evaluate_bets=N._evaluate_bets, generate_patterns=N._generate_patterns,
        run_id=os.environ.get("G3_RUN_ID") or None, executed_at=datetime.now(JST).isoformat(),
        isolation_factory=lambda: G3Isolation({"buyscore_config.json": freeze_path}),
    )
    rec["g3"] = g3
    rec["stage"] = "done"
    rec["count_candidate_checks"] = count_candidate_checks(g3, rec["prep_isolation"], rec)
    return rec


# ---------------------------------------------------------------- 親のプロセス


def _child_env() -> dict[str, str]:
    from shadow.g3_isolation import env_name_is_sensitive

    env = {k: v for k, v in os.environ.items() if not env_name_is_sensitive(k)}
    env["PYTHONPATH"] = os.pathsep.join([str(REBUILD_DIR), str(REPO_DIR)])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def default_child_cmd(race_date: str, eval_id: str, programs: Path, odds: Path, out: Path,
                      nonce: str = "") -> list[str]:
    return [sys.executable, "-B", "-m", "actions.g3_diagnostic_entry", "child", "--race-date", race_date,
            "--eval-id", eval_id, "--programs", str(programs), "--odds", str(odds), "--out", str(out),
            "--nonce", nonce]


def run_parent(race_date: str, programs_raw: Path, odds: Path, out_dir: Path,
               only: Optional[Sequence[str]] = None,
               child_cmd: Callable[..., list[str]] = default_child_cmd,
               timeout: int = CHILD_TIMEOUT_SEC) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # 本人判断2：専用 out_dir の再利用（非空）を拒否。空は許可（C-13 非削除と両立＝拒否≠削除）。
    if any(out_dir.iterdir()):
        raise SystemExit(f"run_parent: out_dir is not empty, refusing to reuse: {out_dir}")
    raw = json.loads(programs_raw.read_bytes())
    programs = raw["programs"] if isinstance(raw, dict) else raw
    races = list_races(race_date, programs)
    if only:
        races = [r for r in races if r["eval_id"] in set(only)]
    records_path = out_dir / "records.jsonl"
    summary: dict[str, Any] = {"race_date": race_date, "races": len(races), "outcomes": {},
                               "child_failures": 0, "started_at": datetime.now(JST).isoformat()}
    with open(records_path, "w", encoding="utf-8") as fh:
        for r in races:
            # C-13：実行ごとの一意の名前と nonce（以前の実行の出力を取り違えない。削除もしない）
            nonce = uuid.uuid4().hex
            child_out = out_dir / "races" / f"{r['eval_id']}.{nonce}.json"
            child_out.parent.mkdir(parents=True, exist_ok=True)
            entry: dict[str, Any]
            try:
                p = subprocess.run(child_cmd(race_date, r["eval_id"], programs_raw, odds, child_out, nonce),
                                   cwd=str(REBUILD_DIR), env=_child_env(), capture_output=True,
                                   text=True, timeout=timeout)
                rc, err_full = p.returncode, p.stderr or ""
            except subprocess.TimeoutExpired:
                rc, err_full = "timeout", ""
            from shadow.g3_isolation import _free_text_digest

            loaded: Optional[dict] = None
            problem = None
            if rc != 0:
                problem = "child_failed"
            elif not child_out.is_file():
                problem = "child_output_missing"
            else:
                try:
                    loaded = json.loads(child_out.read_text(encoding="utf-8"))
                except ValueError:
                    problem = "child_output_invalid"
                if loaded is not None and (loaded.get("eval_id") != r["eval_id"] or loaded.get("nonce") != nonce):
                    problem, loaded = "child_output_mismatch", None
            if loaded is not None:
                entry = loaded
            else:
                entry = {"kind": RECORD_KIND, "eval_id": r["eval_id"], "race_date": race_date,
                         "stage": "child", "error": problem, "child_returncode": rc,
                         # C-06 廃止：原本全文から digest（本文を残さない）。G-17
                         "child_stderr_tail": _free_text_digest(err_full) if err_full else None,
                         "count_candidate_checks": count_candidate_checks({}, None)}
                summary["child_failures"] += 1
            entry["child_returncode"] = rc
            entry["child_output_file"] = child_out.name
            entry.setdefault("count_candidate_checks", count_candidate_checks({}, None))
            entry["count_candidate_checks"]["child_returncode_zero"] = rc == 0
            # F-10：記録の中の自由な文は、書く前にもう一度伏せ字を通す（子の出力をそのまま信じない）
            from shadow.g3_isolation import redact_free_text

            entry = redact_free_text(entry)
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            key = (entry.get("g3") or {}).get("outcome") or f"error:{entry.get('error')}"
            summary["outcomes"][key] = summary["outcomes"].get(key, 0) + 1
    summary["finished_at"] = datetime.now(JST).isoformat()
    # C-05：ジョブが成功でも中身が空・全滅のことを、記録の上で明示する（成否の基準は決めない）
    summary["records_count"] = sum(1 for _ in open(records_path, encoding="utf-8"))
    summary["eval_ids"] = [r["eval_id"] for r in races]
    summary["no_target_races"] = len(races) == 0
    summary["all_children_failed"] = len(races) > 0 and summary["child_failures"] == len(races)
    summary["no_odds_count"] = summary["outcomes"].get("no_odds", 0)
    summary["records_sha256"] = _sha256_file(records_path)
    summary["environment"] = environment_manifest()
    summary["inputs_sha256"] = {"programs_raw": _sha256_file(programs_raw), "odds": _sha256_file(odds)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str),
                                          encoding="utf-8")
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="G3 stage-3 diagnostic (investigation only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("parent")
    pp.add_argument("--race-date", required=True)
    pp.add_argument("--programs", required=True, type=Path)
    pp.add_argument("--odds", required=True, type=Path)
    pp.add_argument("--out-dir", required=True, type=Path)
    pp.add_argument("--only", default="")
    pc = sub.add_parser("child")
    pc.add_argument("--race-date", required=True)
    pc.add_argument("--eval-id", required=True)
    pc.add_argument("--programs", required=True, type=Path)
    pc.add_argument("--odds", required=True, type=Path)
    pc.add_argument("--out", required=True, type=Path)
    pc.add_argument("--nonce", default="")
    a = ap.parse_args(argv)
    from shadow.g3_isolation import redact_free_text

    if a.cmd == "parent":
        s = run_parent(a.race_date, a.programs.resolve(), a.odds.resolve(), a.out_dir.resolve(),
                       only=[x for x in a.only.split(",") if x])
        print(json.dumps({k: s[k] for k in ("race_date", "races", "outcomes", "child_failures",
                                            "records_count", "no_target_races", "all_children_failed")},
                         ensure_ascii=False))
        return 0
    rec = run_child(a.race_date, a.eval_id, a.programs.resolve(), a.odds.resolve())
    rec["nonce"] = a.nonce
    rec = redact_free_text(rec)
    a.out.write_text(json.dumps(rec, ensure_ascii=False, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    # F-10：ログ・print・例外の表示を含め、標準出力・標準エラーに出る文字を伏せ字にする（プロセスとして動くときだけ）
    from shadow.g3_isolation import install_redacting_streams

    install_redacting_streams()
    sys.exit(main())
