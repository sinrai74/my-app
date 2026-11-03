"""正式G2：集約・証拠照合（C-6・§6.3〜§6.6・判断B B3）。

標準ライブラリだけを使い、評価コード・Legacy のコードは読み込まない。

入力（--artifacts-dir の下に、実行ごとのディレクトリ）：
  <run_id>-<run_attempt>/artifact.zip   … 評価の artifact（zip のまま）
  <run_id>-<run_attempt>/job.log        … 評価ジョブのログ原本（GitHub Actions API から取得）
  <run_id>-<run_attempt>/meta.json      … {"run_id", "run_attempt", "artifact_name",
                                            "expected_digest", "run_started_at",
                                            "job_id", "job_count", "log_fetch_ok", "log_fetch_error",
                                            "log_file", "log_sha256", "log_bytes", "log_fetched_at"}
--permissions：個別の開始許可の一覧（形式は未決。下の load_permissions を参照）
--baseline：登録済みの基準 JSON（M7-3：runner_image をジョブログの値と照合する）

出力：
  aggregate_result.json（判定に使った集約コード・workflow の SHA-256 を含む）
  logs/<run_id>-<run_attempt>.log（ジョブログ原本の写し）
  manifest.json（上の各ファイルの存在・サイズ・SHA-256。"complete": true）
終了コード：照合の失敗（artifact・ジョブログの検証、基準の読み込み、正式な実行の ref・基準の
同一性・Runner Image・W7 の監視の照合）が1件でもあれば 3、無ければ 0。評価側の設計どおりの安全停止
（CONFIG・PRELOAD など）は、記録と除外が正しければ 0。

M7-3（2026-11-03 の判断）：Runner Image の識別値は、ジョブログの「Set up job →
Runner Image」グループの Image・Version を基準とする。artifact 内の環境変数
ImageOS・ImageVersion は観測値として別に記録し、ログの値の代わりには使わない。
- 状態は「未判定」「100件到達・達成未確定」まで。「正式G2の判定で100件達成」は、
  証拠一式の commit が済んだあとに人が記録する（判断B B3）。
- 開催実績を確かめる方法が未決のため、失敗日は判定しない（確認待ちとして扱う）。
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime, timedelta
from typing import Any, Optional

REQUIRED_RACE_KEYS = ("eval_id", "result_category", "d9_states", "counts", "d5")
REQUIRED_COMPARED_KEYS = ("values", "comparable_hash", "hashes", "d13_2", "diffs")
TARGET = 100
EXIT_VERIFICATION_FAILED = 3

# ジョブログの各行は "<ISO8601 の時刻>Z <本文>" の形（先頭に BOM が付くことがある）。
_LOG_TS_RE = re.compile(r"^﻿?\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ?")
_RUNNER_IMAGE_VERSION_RE = re.compile(r"\d{8}\.\d+\.\d+")
_RUNNER_IMAGE_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
RUNNER_IMAGE_GROUP = "##[group]Runner Image"

# 禁止対象の監視ラベル（評価側の instrument.FORBIDDEN_CATEGORIES と同じ内容を、ここで独立に持つ。
# 評価コードを読み込まないため。両者の一致はテストで確かめる）。
EXPECTED_FORBIDDEN: dict[str, tuple[str, ...]] = {
    "A_buyscore": tuple(
        [f"x_buyscore.{q}" for q in (
            "_ensure_config_file", "load_config", "_odds_band_bonus", "_race_type_bonus", "calc_buyscore",
            "check_passthrough", "stars", "investment_type", "risk_level", "kelly_fraction",
            "get_bet_multiplier_extended", "check_korogashi", "check_ippatsu", "assign_rank_labels",
            "format_buy_reason", "format_exclude_reason", "format_passthrough_message", "format_buy_message",
            "save_buyscore_log", "apply_buyscore")]
        + [f"core.buyscore.{q}" for q in (
            "_odds_band_bonus", "_race_type_bonus", "calc_buyscore", "check_passthrough", "investment_type",
            "kelly_fraction", "DefaultKellyStrategy.__call__", "DefaultBuyEngine.__init__",
            "DefaultBuyEngine.assess")]
        + ["notify_arashi._evaluate_bets"]),
    "B_ml_and_legacy_fetch": ("score_by_model", "_load_ml_model", "_predict_win_prob", "fetch_programs",
                              "_safe_get"),
    "C_notify": tuple(f"notify_arashi.{q}" for q in (
        "_run_main", "send_line", "send_notification", "send_email", "send_summary_if_none",
        "send_preview_notification", "_check_odds_drop")),
    "D_release_write": tuple(f"x_release_storage.{q}" for q in (
        "upload_file", "_upload_asset", "_delete_asset", "_get_or_create_release", "restore_from_backup")),
}
EXPECTED_FORBIDDEN_LABELS = frozenset(lab for v in EXPECTED_FORBIDDEN.values() for lab in v)
FORMAL_REF = "refs/heads/main"


def _is_count(v: Any) -> bool:
    """0 以上の整数（bool は整数として受け入れない）。"""
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def w7_run_problems(run: dict[str, Any], races_count: Optional[int] = None) -> list[str]:
    """評価側の合格の印を信用せず、記録された回数・対象一覧・解決状態を独立に検査する。

    races_count：races.jsonl に書かれたレースの数（N）。生きているスレッドの観測の検査は、
    監視の開始時・各レースの前後・停止時に行われるため、thread_checks >= 2*N+2 を求める。
    指定が無い場合、完了した実行では対象一覧の件数を N とする。途中で止まった実行は、
    検査の回数にかかわらず、ここで不適格の理由を加える（適格扱いにしない）。
    """
    inst = run.get("instrument")
    if not isinstance(inst, dict):
        return ["monitoring record missing"]
    problems = []
    res = inst.get("target_resolution") or {}
    counts = inst.get("call_counts") or {}
    missing = sorted(lab for lab in EXPECTED_FORBIDDEN_LABELS if lab not in res or lab not in counts)
    if missing:
        problems.append(f"forbidden labels not monitored: {missing[:5]} ({len(missing)})")
    unresolved = sorted(lab for lab, v in res.items() if not (isinstance(v, dict) and v.get("resolved") is True))
    if unresolved or inst.get("unresolved_targets"):
        problems.append(f"unresolved targets: {(unresolved or inst.get('unresolved_targets'))[:5]}")
    called = {lab: counts.get(lab) for lab in EXPECTED_FORBIDDEN_LABELS
              if lab in counts and counts.get(lab) != 0}
    if called:
        problems.append(f"forbidden calls: {called}")
    if inst.get("prior_profiler_present") is not False:
        problems.append("profiler present before monitoring (or unrecorded)")
    if inst.get("threads_at_start") != 1:
        problems.append(f"threads at start = {inst.get('threads_at_start')}")
    ms = inst.get("monitor_state")
    if not isinstance(ms, dict):
        problems.append("monitor state missing")
    else:
        if ms.get("started") is not True or ms.get("stopped") is not True:
            problems.append("monitor not started/stopped properly")
        if ms.get("hook_violations"):
            problems.append("monitor hook removed or replaced")
        if ms.get("monitor_errors"):
            problems.append("monitor errors")
        if not isinstance(ms.get("thread_starts"), list):
            problems.append("thread start record missing")
        elif any(t.get("via_threading") is not True for t in ms["thread_starts"]):
            problems.append("thread started outside threading (not monitored)")
        # 生きているスレッドが、すべてプロファイラで観測済みだったか（instrument.check_threads）
        if "unobserved_thread_violations" not in ms:
            problems.append("unobserved thread record missing")
        elif not isinstance(ms["unobserved_thread_violations"], list):
            problems.append("unobserved thread record has wrong type")
        elif ms["unobserved_thread_violations"]:
            problems.append(f"unobserved live threads: {len(ms['unobserved_thread_violations'])}")
        checks = ms.get("thread_checks")
        if "thread_checks" not in ms:
            problems.append("thread_checks missing")
        elif not _is_count(checks):
            problems.append(f"thread_checks has wrong type or value: {checks!r}")
        else:
            n = races_count
            if n is None and run.get("status") == "completed":
                n = len((run.get("programs") or {}).get("targets") or [])
            if not _is_count(n):
                problems.append("processed race count unknown (thread_checks cannot be verified)")
            elif checks < 2 * n + 2:
                problems.append(f"thread_checks {checks} < 2*{n}+2 (thread observation incomplete)")
    if run.get("status") != "completed":
        problems.append(f"run not completed (status={run.get('status')!r})")
    if any((e or {}).get("kind") == "process" for e in inst.get("side_effect_events") or []):
        problems.append("process started (not monitored)")
    return problems


def ref_problems(run: dict[str, Any]) -> list[str]:
    info = run.get("info") or {}
    problems = []
    if info.get("github_ref") != FORMAL_REF:
        problems.append(f"github_ref is not {FORMAL_REF}: {info.get('github_ref')}")
    if not str(info.get("github_workflow_ref") or "").endswith("@" + FORMAL_REF):
        problems.append("github_workflow_ref does not end with @" + FORMAL_REF)
    return problems
GROUP_END = "##[endgroup]"


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def self_version() -> dict[str, Optional[str]]:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    out: dict[str, Optional[str]] = {}
    for rel in ("rebuild/formal_g2/aggregate/main.py", "rebuild/formal_g2/aggregate/__init__.py",
                "rebuild/formal_g2/aggregate/fetch_log.py", ".github/workflows/formal_g2_aggregate.yml"):
        p = os.path.join(repo_root, rel)
        try:
            with open(p, "rb") as f:
                out[rel] = _sha256(f.read())
        except OSError:
            out[rel] = None
    return out


# ---- 照合（C-6 の 3） ----
def verify_run(run_dir: str) -> dict[str, Any]:
    res: dict[str, Any] = {"dir": os.path.basename(run_dir), "verified": False, "problems": []}
    try:
        with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
        with open(os.path.join(run_dir, "artifact.zip"), "rb") as f:
            blob = f.read()
    except OSError as exc:
        res["problems"].append(f"missing input: {exc}")
        return res
    res["meta"] = meta
    expected = (meta.get("expected_digest") or "").removeprefix("sha256:")
    actual = _sha256(blob)
    res["artifact_sha256"] = actual
    if not expected:
        res["problems"].append("expected digest unavailable (unverified)")
    elif expected != actual:
        res["problems"].append("artifact digest mismatch")
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
        names = set(zf.namelist())
        manifest = json.loads(zf.read("manifest.json"))
    except Exception as exc:  # noqa: BLE001
        res["problems"].append(f"artifact unreadable: {type(exc).__name__}")
        return res
    if manifest.get("complete") is not True:
        res["problems"].append("manifest not complete")
    listed = manifest.get("files") or {}
    for name, info in listed.items():
        if name not in names:
            res["problems"].append(f"listed file missing: {name}")
        elif _sha256(zf.read(name)) != info.get("sha256"):
            res["problems"].append(f"file sha256 mismatch: {name}")
    unlisted = sorted(names - set(listed) - {"manifest.json"})
    if unlisted:
        res["problems"].append(f"unlisted files: {unlisted}")
    res["manifest"] = manifest
    artifact_ok = not res["problems"]
    run_json = None
    if artifact_ok:
        run_json = json.loads(zf.read("run.json"))
    # M7-3：ジョブログ（Runner Image）の照合。環境変数の値は観測値として別に記録する。
    log_check = verify_job_log(run_dir, meta)
    res["job_log"] = log_check
    res["runner_image"] = {
        "log": log_check.get("runner_image"),
        "env_observed": ((run_json or {}).get("env_snapshot") or {}).get("image"),
    }
    res["runner_image"]["env_version_differs_from_log"] = _env_differs(res["runner_image"])
    res["problems"].extend(f"job log: {p}" for p in log_check["problems"])
    if not res["problems"]:
        res["verified"] = True
        res["run"] = run_json
        races = []
        if "races.jsonl" in names:
            for line in zf.read("races.jsonl").decode("utf-8").splitlines():
                if line.strip():
                    races.append(json.loads(line))
        res["races"] = races
    return res


def _env_differs(ri: dict[str, Any]) -> Optional[bool]:
    log, env = ri.get("log"), ri.get("env_observed")
    if not log or not isinstance(env, dict) or env.get("ImageVersion") is None:
        return None  # 比べられない（どちらかが無い）
    return env.get("ImageVersion") != log.get("version")


# ---- M7-3：ジョブログの Runner Image ----
def parse_runner_image(text: str) -> dict[str, Any]:
    """ジョブログから「Runner Image」グループの Image・Version を取り出す。

    グループがちょうど1つあり、その中に Image と Version がそれぞれちょうど1行ずつ、
    決まった形の値で入っている場合だけ ok。それ以外は ok=False（推測で補わない）。
    """
    out: dict[str, Any] = {"ok": False, "image": None, "version": None, "lines": {}, "problems": []}
    lines = [_LOG_TS_RE.sub("", ln, count=1) for ln in text.splitlines()]
    starts = [i for i, ln in enumerate(lines) if ln.strip() == RUNNER_IMAGE_GROUP]
    if len(starts) != 1:
        out["problems"].append(f"Runner Image group count = {len(starts)} (expected 1)")
        return out
    start = starts[0]
    end = next((i for i in range(start + 1, len(lines)) if lines[i].strip() == GROUP_END), None)
    if end is None:
        out["problems"].append("Runner Image group is not closed")
        return out
    found: dict[str, list[tuple[int, str]]] = {"Image": [], "Version": []}
    for i in range(start + 1, end):
        m = re.fullmatch(r"(Image|Version):\s*(.*?)\s*", lines[i])
        if m:
            found[m.group(1)].append((i + 1, m.group(2)))
    for key, pattern in (("Image", _RUNNER_IMAGE_NAME_RE), ("Version", _RUNNER_IMAGE_VERSION_RE)):
        vals = found[key]
        if len(vals) != 1:
            out["problems"].append(f"Runner Image '{key}' line count = {len(vals)} (expected 1)")
            continue
        line_no, value = vals[0]
        if not value or pattern.fullmatch(value) is None:
            out["problems"].append(f"Runner Image '{key}' value not parseable")
            continue
        out[key.lower()] = value
        out["lines"][key] = line_no
    out["group_lines"] = [start + 1, end + 1]
    out["ok"] = not out["problems"]
    return out


def verify_job_log(run_dir: str, meta: dict[str, Any]) -> dict[str, Any]:
    """ジョブログ原本の取得状況・SHA-256・サイズ・Runner Image の値を確かめる。

    取得失敗・HTTP エラー・ファイル欠落・ハッシュ不一致・グループ欠落・解析失敗は、
    すべて problems に入る（＝不合格）。環境変数の値で代わりにすることはしない。
    """
    res: dict[str, Any] = {"ok": False, "problems": [], "runner_image": None,
                           "job_id": meta.get("job_id"), "fetched_at": meta.get("log_fetched_at")}
    if meta.get("log_fetch_ok") is not True:
        res["problems"].append(f"log fetch failed: {meta.get('log_fetch_error') or 'no fetch record'}")
        return res
    name = meta.get("log_file") or ""
    if not name or os.path.basename(name) != name:
        res["problems"].append("log file name invalid")
        return res
    path = os.path.join(run_dir, name)
    try:
        with open(path, "rb") as f:
            blob = f.read()
    except OSError as exc:
        res["problems"].append(f"log file missing: {type(exc).__name__}")
        return res
    res["file"] = name
    res["sha256"] = _sha256(blob)
    res["bytes"] = len(blob)
    if not meta.get("log_sha256"):
        res["problems"].append("expected log sha256 unavailable")
    elif meta["log_sha256"] != res["sha256"]:
        res["problems"].append("log sha256 mismatch")
    if meta.get("log_bytes") is not None and str(meta["log_bytes"]) != str(len(blob)):
        res["problems"].append("log size mismatch")
    if not blob:
        res["problems"].append("log is empty")
    try:
        text = blob.decode("utf-8")
    except UnicodeDecodeError:
        res["problems"].append("log not decodable as UTF-8")
        return res
    parsed = parse_runner_image(text)
    res["parse"] = {k: parsed[k] for k in ("lines", "problems") if k in parsed}
    res["parse"]["group_lines"] = parsed.get("group_lines")
    res["problems"].extend(parsed["problems"])
    if parsed["ok"]:
        res["runner_image"] = {"image": parsed["image"], "version": parsed["version"]}
    res["ok"] = not res["problems"]
    return res


def baseline_identity_check(run: dict[str, Any], baseline_sha256: Optional[str]) -> str:
    """評価側と集約側で同じ内容の基準を使ったか（内容の SHA-256 で照合。パスは問わない）。

    "ok" / "no_baseline" / "identity_unrecorded" / "identity_changed_during_run" / "identity_mismatch"
    """
    if baseline_sha256 is None:
        return "no_baseline"
    ident = (run.get("checks") or {}).get("baseline_identity")
    if not isinstance(ident, dict) or not ident.get("sha256"):
        return "identity_unrecorded"
    if ident.get("unchanged_during_run") is not True or ident.get("sha256_at_end") != ident["sha256"]:
        return "identity_changed_during_run"
    return "ok" if ident["sha256"] == baseline_sha256 else "identity_mismatch"


def runner_image_check(vr: dict[str, Any], baseline: Optional[dict[str, Any]]) -> str:
    """M7-3 の照合結果："ok" / "no_baseline" / "baseline_invalid" / "log_unverified" / "mismatch"。"""
    log = (vr.get("runner_image") or {}).get("log")
    if not log:
        return "log_unverified"
    if baseline is None:
        return "no_baseline"
    ri = baseline.get("runner_image")
    if not isinstance(ri, dict) or set(ri) != {"image", "version"}:
        return "baseline_invalid"
    return "ok" if (ri.get("image"), ri.get("version")) == (log["image"], log["version"]) else "mismatch"


# ---- 正式な実行かどうか（§6.3〜§6.5 の 6） ----
def load_permissions(path: Optional[str]) -> list[dict[str, Any]]:
    """[{"permission_id", "race_date", "window_start", "window_end"}]（ISO8601）。形式は未決。"""
    if not path:
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_formal(vr: dict[str, Any], permissions: list[dict[str, Any]]) -> bool:
    info = (vr.get("manifest") or {}).get("run") or {}
    if info.get("mode") != "formal":
        return False
    started = (vr.get("meta") or {}).get("run_started_at")
    for p in permissions:
        if p.get("permission_id") != info.get("permission_id") or p.get("race_date") != info.get("race_date"):
            continue
        if started and p.get("window_start") and p.get("window_end"):
            if p["window_start"] <= started <= p["window_end"]:
                return True
    return False


def select_formal(runs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """run_attempt 1 の実行開始日時が最も早いもの。同時刻なら run_id の小さいもの。置き換えない。"""
    firsts = [r for r in runs if str((r.get("meta") or {}).get("run_attempt")) == "1"]
    if not firsts:
        return None
    return sorted(
        firsts, key=lambda r: (r["meta"].get("run_started_at") or "", int(r["meta"].get("run_id") or 0))
    )[0]


# ---- レースの状態（§6.3〜§6.5 の 3） ----
def race_state(rec: dict[str, Any], run_ok: bool) -> str:
    if any(k not in rec for k in REQUIRED_RACE_KEYS):
        return "evidence_incomplete"
    counts = rec.get("counts") or {}
    if any(lab not in counts for lab in EXPECTED_FORBIDDEN_LABELS):
        return "evidence_incomplete"  # 禁止対象の回数が記録されていない
    if any(counts.get(lab) != 0 for lab in EXPECTED_FORBIDDEN_LABELS):
        return "forbidden_call"  # このレースは加算しない
    cat = rec.get("result_category")
    if cat in ("not_established", "compare_exception"):
        return "compare_not_established"
    if any(k not in rec for k in REQUIRED_COMPARED_KEYS):
        return "evidence_incomplete"
    if cat == "mismatch":
        return "mismatch"
    states = rec.get("d9_states") or []
    if "compare_not_established" in states:
        return "compare_not_established"
    if "input_identity_unverified" in states:
        return "input_identity_unverified"
    if not run_ok:
        return "unclassified_event"  # M7 等の実行単位の条件を満たさない
    if cat == "diff_zero" and rec.get("input_identity_ok") is True:
        return "added"
    return "unclassified_event"


def run_level_ok(run: dict[str, Any], races_count: Optional[int] = None) -> bool:
    checks = run.get("checks") or {}
    return (
        run.get("status") == "completed"
        and not (run.get("config_detection") or {}).get("detected", True)
        and (run.get("preload") or {}).get("ok") is True
        and not checks.get("env_self_check")
        and checks.get("baseline_diffs") == []
        and (checks.get("modules_pre_eval") or {}).get("ok") is True
        and (checks.get("modules_post_eval") or {}).get("ok") is True
        and (checks.get("tests_not_loaded") or {}).get("T3_ok") is True
        and not checks.get("shadow_package_loaded")
        and not w7_run_problems(run, races_count)
    )


def aggregate(verified: list[dict[str, Any]], permissions: list[dict[str, Any]],
              first_date: str, last_date: str,
              baseline: Optional[dict[str, Any]] = None,
              baseline_sha256: Optional[str] = None) -> dict[str, Any]:
    """基準が無い・評価側と同じ基準だと確かめられない・形が正しくない・ジョブログの
    Runner Image と一致しない実行は、その日の全レースを正式カウントから除外し、区間を
    終える（M7-3・基準の同一性。評価側の A-1 に相当）。これらは baseline_failures に入る。"""
    baseline_failures: list[dict[str, Any]] = []
    by_date: dict[str, list[dict[str, Any]]] = {}
    for vr in verified:
        if is_formal(vr, permissions):
            d = ((vr.get("manifest") or {}).get("run") or {}).get("race_date")
            by_date.setdefault(d, []).append(vr)

    d0 = datetime.strptime(first_date, "%Y%m%d").date()
    d1 = datetime.strptime(last_date, "%Y%m%d").date()
    days = []
    seq: list[dict[str, Any]] = []
    cur = d0
    while cur <= d1:
        ds = cur.strftime("%Y%m%d")
        sel = select_formal(by_date.get(ds, []))
        day: dict[str, Any] = {"race_date": ds, "selected_run": None, "class": "pending_confirmation"}
        if sel is not None:
            day["selected_run"] = sel["meta"]
            day["ignored_runs"] = [r["meta"] for r in by_date[ds] if r is not sel]
            if not sel["verified"]:
                day["verification_failed"] = sel["problems"]
                seq.append({"break": "verification_failed", "race_date": ds})
            else:
                run = sel["run"]
                targets = [t["eval_id"] for t in (run.get("programs") or {}).get("targets", [])]
                if run.get("status") == "completed":
                    id_check = baseline_identity_check(run, baseline_sha256 if baseline is not None else None)
                    ri_check = runner_image_check(sel, baseline)
                else:
                    # 設計どおりの安全停止：実行単位で不適格（run_level_ok）になり、加算されない。
                    # 基準の照合は行わない（停止が照合の失敗として数えられないようにする）。
                    id_check = ri_check = "not_applicable_stopped_run"
                day["baseline_identity_check"] = id_check
                day["runner_image_check"] = ri_check
                day["ref_check"] = ref_problems(run) or "ok"
                w7p = w7_run_problems(run, len(sel.get("races") or []))
                day["w7_check"] = (w7p or "ok") if run.get("status") == "completed" else "not_applicable_stopped_run"
                passing = ("ok", "not_applicable_stopped_run")
                if day["ref_check"] != "ok":
                    failed_check = "ref_not_main"  # 正式な実行は main で行う（調査用の実行には適用しない）
                elif id_check not in passing:
                    failed_check = f"baseline_{id_check}"
                elif ri_check not in passing:
                    failed_check = f"runner_image_{ri_check}"
                elif day["w7_check"] not in passing:
                    failed_check = "w7_monitoring"
                else:
                    failed_check = None
                if failed_check is not None:
                    # ref・基準の同一性・M7-3 の Runner Image・W7 の監視のいずれかを満たさない
                    day["class"] = "normal" if targets else day["class"]
                    day["excluded"] = failed_check
                    seq.append({"break": failed_check, "race_date": ds})
                    baseline_failures.append({"race_date": ds, "run": sel["meta"], "check": failed_check})
                elif (run.get("config_detection") or {}).get("detected", True):
                    # config を検出した実行は、全レースを正式カウントから除外し、区間を終える
                    day["class"] = "normal" if targets else day["class"]
                    day["excluded"] = "config_detected"
                    seq.append({"break": "config_detected", "race_date": ds})
                elif targets:
                    day["class"] = "normal"
                    ok = run_level_ok(run, len(sel.get("races") or []))
                    recs: dict[str, list] = {}
                    for r in sel.get("races", []):
                        recs.setdefault(r.get("eval_id"), []).append(r)
                    for eid in targets:
                        rl = recs.get(eid, [])
                        if len(rl) == 1:
                            st = race_state(rl[0], ok)
                        elif len(rl) > 1:
                            st = "duplicate_eval_id"
                        else:
                            st = "unprocessed"
                        seq.append({"eval_id": eid, "state": st})
        if day["class"] != "normal" and not day.get("verification_failed") and not day.get("excluded"):
            seq.append({"break": "pending_confirmation_day", "race_date": ds})
        days.append(day)
        cur += timedelta(days=1)

    segments = []
    count = 0
    start = None
    for item in seq:
        if item.get("state") == "added":
            if count == 0:
                start = item["eval_id"]
            count += 1
            continue
        if count:
            segments.append({"start": start, "count": count, "ended_by": item.get("state") or item.get("break")})
        count, start = 0, None
    if count:
        segments.append({"start": start, "count": count, "ended_by": None})

    best = max((s["count"] for s in segments), default=0)
    status = "reached_100_unconfirmed" if best >= TARGET else "undetermined"
    return {"days": days, "sequence": seq, "segments": segments, "best_segment": best, "g2_status": status,
            "baseline_failures": baseline_failures}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="formal_g2.aggregate.main")
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--permissions", default="")
    p.add_argument("--first-date", required=True)
    p.add_argument("--last-date", required=True)
    p.add_argument("--out", default="g2_aggregate_out")
    p.add_argument("--baseline", default="", help="登録済みの基準 JSON（M7-3 の runner_image を照合する）")
    args = p.parse_args(argv)

    loaded_repo_modules = sorted(
        n for n, m in sys.modules.items()
        if n.split(".")[0] in ("notify_arashi", "x_asahi_scoring", "x_venue_stats", "core", "adapters", "models")
    )
    baseline = None
    baseline_sha256 = None
    baseline_error = None
    if args.baseline:
        try:
            with open(args.baseline, "rb") as f:
                raw = f.read()
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                raise ValueError("baseline is not a JSON object")
            baseline, baseline_sha256 = parsed, _sha256(raw)
        except (OSError, ValueError) as exc:
            baseline_error = f"{type(exc).__name__}: {exc}"  # 取得不能：照合できない（終了コード 3）
    verified = [
        verify_run(os.path.join(args.artifacts_dir, d))
        for d in sorted(os.listdir(args.artifacts_dir))
        if os.path.isdir(os.path.join(args.artifacts_dir, d))
    ]
    result = aggregate(verified, load_permissions(args.permissions), args.first_date, args.last_date,
                       baseline, baseline_sha256)
    result["runs"] = [
        {k: v for k, v in vr.items()
         if k in ("dir", "verified", "problems", "meta", "artifact_sha256", "job_log", "runner_image")}
        for vr in verified
    ]
    for out_run, vr in zip(result["runs"], verified):
        run_json = vr.get("run") or {}
        out_run["w7_check"] = (w7_run_problems(run_json, len(vr.get("races") or []))
                               if vr.get("verified") else None)
        out_run["ref"] = {k: (run_json.get("info") or {}).get(k) for k in ("github_ref", "github_workflow_ref", "mode")}
    result["baseline"] = {"path": args.baseline or None, "sha256": baseline_sha256,
                          "read_error": baseline_error,
                          "runner_image": (baseline or {}).get("runner_image")}
    result["aggregator_version"] = self_version()
    result["evaluation_code_loaded"] = loaded_repo_modules
    result["generated_at"] = datetime.now().astimezone().isoformat()
    failed = [vr["dir"] for vr in verified if not vr["verified"]]
    result["verification_failed_runs"] = failed

    os.makedirs(args.out, exist_ok=True)
    # ジョブログ原本の写し（C-6：証跡として残す）
    log_dir = os.path.join(args.out, "logs")
    for vr in verified:
        name = (vr.get("job_log") or {}).get("file")
        if name:
            os.makedirs(log_dir, exist_ok=True)
            shutil.copyfile(os.path.join(args.artifacts_dir, vr["dir"], name),
                            os.path.join(log_dir, f"{vr['dir']}.log"))
    with open(os.path.join(args.out, "aggregate_result.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=1))
    write_output_manifest(args.out)

    # 照合の結果をログに出す（値はハッシュと識別値だけ。トークン等は扱わない）
    print(f"aggregate: runs={len(verified)} verified={len(verified) - len(failed)} failed={len(failed)} "
          f"baseline_sha256={baseline_sha256} baseline_read_error={baseline_error} "
          f"baseline_failures={len(result['baseline_failures'])} "
          f"best_segment={result['best_segment']} g2_status={result['g2_status']}")
    for vr in verified:
        ri = vr.get("runner_image") or {}
        print(f"run {vr['dir']}: verified={vr['verified']} runner_image_log={ri.get('log')} "
              f"env_observed_version={(ri.get('env_observed') or {}).get('ImageVersion')} "
              f"problems={vr['problems']}")
    for rr in result["runs"]:
        print(f"run {rr['dir']}: mode={rr['ref'].get('mode')} ref={rr['ref'].get('github_ref')} "
              f"w7_problems={rr['w7_check']}")
    for bf in result["baseline_failures"]:
        print(f"baseline check failed: race_date={bf['race_date']} run={bf['run'].get('run_id')}-"
              f"{bf['run'].get('run_attempt')} check={bf['check']}")
    # 終了コード 3：集約の照合の失敗（artifact・ジョブログ・基準の読み込み・基準の照合）。
    # 評価側の設計どおりの安全停止（CONFIG・PRELOAD など）は、記録と除外が正しければ 0。
    if failed or baseline_error is not None or result["baseline_failures"]:
        return EXIT_VERIFICATION_FAILED
    return 0


def write_output_manifest(out_dir: str) -> str:
    """集約の出力の目録（存在・サイズ・SHA-256）。最後に書き、"complete": true を含める。"""
    files: dict[str, Any] = {}
    for dirpath, _dirs, filenames in os.walk(out_dir):
        for fn in filenames:
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, out_dir).replace(os.sep, "/")
            if rel == "manifest.json":
                continue
            with open(path, "rb") as f:
                blob = f.read()
            files[rel] = {"sha256": _sha256(blob), "bytes": len(blob)}
    path = os.path.join(out_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"files": dict(sorted(files.items())), "complete": True},
                           ensure_ascii=False, sort_keys=True))
        f.write("\n")
    return path


if __name__ == "__main__":
    sys.exit(main())
