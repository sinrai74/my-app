"""正式G2：集約・証拠照合（C-6・§6.3〜§6.6・判断B B3）。

標準ライブラリだけを使い、評価コード・Legacy のコードは読み込まない。

入力（--artifacts-dir の下に、実行ごとのディレクトリ）：
  <run_id>-<run_attempt>/artifact.zip   … 評価の artifact（zip のまま）
  <run_id>-<run_attempt>/job.log        … 評価ジョブのログ原本（GitHub Actions API から取得）
  <run_id>-<run_attempt>/meta.json      … {"run_id", "run_attempt", "artifact_name",
                                            "expected_digest", "run_started_at",
                                            "job_id", "job_count", "log_fetch_ok", "log_fetch_error",
                                            "log_file", "log_sha256", "log_bytes", "log_fetched_at",
                                            "api_run_id", "api_run_attempt", "api_head_sha", "api_head_branch",
                                            "api_workflow_path", "api_repository", "api_event"}
                                          （api_* は attempts の API の応答。2026-11-06 に追加）
--permissions：個別の開始許可の一覧（形式は未決。下の load_permissions を参照）
--baseline：評価側の基準 JSON（M7-3：runner_image の Image をジョブログの値と照合する。Version は観測値）
--aggregate-baseline：集約側の基準 JSON（集約の4ファイルの SHA-256 と Python の版。2026-11-06 に追加）
--tools-json：gh・jq の版（観測値。照合しない）

出力：
  aggregate_result.json（判定に使った集約コード・workflow の SHA-256 を含む）
  logs/<run_id>-<run_attempt>.log（ジョブログ原本の写し）
  manifest.json（上の各ファイルの存在・サイズ・SHA-256。"complete": true）
終了コード：照合の失敗（artifact・ジョブログの検証、基準の読み込み、正式な実行の ref・基準の
同一性・Runner Image・W7 の監視・実行メタデータ・条件5 の照合）が1件でもあれば 3、無ければ 0。
評価側の設計どおりの安全停止（CONFIG・PRELOAD など）は、記録と除外が正しければ 0。

2026-11-05 の追加（07g、GPT 判断 2-1〜2-3・1-2）：
- 実行メタデータの照合（run_metadata_check）：run.json の run_id・run_attempt と meta.json（API）、
  head_sha・ref・workflow_ref と API の値（meta.json の api_* が必要。今の集約 workflow は取得しないので
  欠落として不合格）、started_at・finished_at（必須）、manifest と run.json の実行識別情報。
- 条件5（condition5_check）：programs_raw.json から対象一覧・targets_sha256・program_sha256 を計算し直す。
- comparable_hash の対象（comparable_scope_ok）：values の両側がちょうど5項目であること（構造だけ）。
- 条件7（release_and_network_record）：利用可能性・通信の監視結果・監視の限界を別々に記録する。

2026-11-06 の追加（§6.8 B の条件1・3・5・6、GPT 判断。M7-3・baseline の登録は本人の判断待ち）：
- run_id・run_attempt は API の値（api_run_id・api_run_attempt）と照合し、event が workflow_dispatch であること。
- 評価ジョブのログの検査：E2（アクションの入力のトークン）、E3（GITHUB_TOKEN Permissions の表示。main() は
  EXPECTED_EVAL_TOKEN_PERMISSIONS＝「Metadata: read」の1行だけを合格とする）、E5（g2_fetch_head と API の head_sha、g2_credentials_check）。E4（評価プロセスの環境）は run.json。
- 集約側の基準の照合（aggregate_baseline_check）。
- Runner Image は Image だけを照合し、Version は観測値として記録する。
- baseline の登録（2026-11-08、本人が案A を採用。07h §2.8）：main() は、リポジトリの
  g2_baseline/registered/approvals.json（承認記録）と、--baseline・--aggregate-baseline に渡された基準を照合する
  （registration_check）。承認記録が無い・形が正しくない・一致しない場合は登録を確認できないとし、正式な実行に
  合格を出さない。候補のファイルが置かれているだけでは登録とみなさない。

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


# ---- 実行メタデータの照合（07g §3.5、2026-11-05 の GPT 判断 2-1） ----
# 比較値の差分とは別の「証跡の不備」として扱う。欠落・形式不正・不一致は、その実行を正式判定に通さない。
# API 由来の値（api_*）は、集約の workflow が attempts の応答から meta.json に入れる（2026-11-06 の変更）。
# 入力の run_id・artifact 名の attempt（meta.run_id・meta.run_attempt）は API の値の代わりにしない。
# api_* が1つでも欠けていれば不合格（fail-closed）。
META_API_FIELDS = ("api_run_id", "api_run_attempt", "api_head_sha", "api_head_branch", "api_workflow_path",
                   "api_repository", "api_event")
EXPECTED_API_EVENT = "workflow_dispatch"


def _parse_ts(value: Any) -> Optional[datetime]:
    """ISO8601 の日時（時差つき）を読む。読めない・時差が無い場合は None。"""
    if not isinstance(value, str) or not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo is not None else None


def run_metadata_check(vr: dict[str, Any]) -> dict[str, Any]:
    """run.json・manifest・meta.json（集約の workflow が API から取った値）の実行識別情報を照合する。

    返り値 {"ok": bool, "problems": [...], "api_fields_present": bool, "recorded": {...}}。
    """
    out: dict[str, Any] = {"ok": False, "problems": [], "api_fields_present": False}
    run = vr.get("run") or {}
    meta = vr.get("meta") or {}
    info = run.get("info")
    if not isinstance(info, dict):
        out["problems"].append("run.json info missing")
        return out
    man_run = (vr.get("manifest") or {}).get("run")
    p = out["problems"]
    # 1. API の値がそろっているか
    present = all(str(meta.get(k) or "") for k in META_API_FIELDS)
    out["api_fields_present"] = present
    if not present:
        missing = [k for k in META_API_FIELDS if not str(meta.get(k) or "")]
        p.append(f"API run fields unavailable: {missing}")
    else:
        # 2. run_id・run_attempt は API の値と照合する（入力の値・artifact 名の値は、API の値との一致だけを見る）
        if str(info.get("github_run_id") or "") != str(meta["api_run_id"]):
            p.append(f"run_id mismatch: run.json={info.get('github_run_id')!r} api={meta['api_run_id']!r}")
        if str(meta.get("run_id") or "") != str(meta["api_run_id"]):
            p.append("requested run_id does not match API run id")
        if str(info.get("github_run_attempt") or "") != str(meta["api_run_attempt"]):
            p.append(f"run_attempt mismatch: run.json={info.get('github_run_attempt')!r} api={meta['api_run_attempt']!r}")
        if str(meta.get("run_attempt") or "") != str(meta["api_run_attempt"]):
            p.append("artifact attempt does not match API run attempt")
        latest = meta.get("latest_attempt")
        try:
            if latest is None or int(meta["api_run_attempt"]) > int(latest):
                p.append("run_attempt greater than latest attempt reported by API (or latest unknown)")
        except (TypeError, ValueError):
            p.append("run_attempt or latest_attempt not an integer")
        if meta["api_event"] != EXPECTED_API_EVENT:
            p.append(f"API event is not {EXPECTED_API_EVENT}: {meta['api_event']!r}")
        # head_sha・ref・workflow_ref を API の値と照合する
        if info.get("github_sha") != meta["api_head_sha"]:
            p.append("github_sha does not match API head_sha")
        if info.get("github_ref") != "refs/heads/" + str(meta["api_head_branch"]):
            p.append("github_ref does not match API head_branch")
        expected_wf = f"{meta['api_repository']}/{meta['api_workflow_path']}@{info.get('github_ref')}"
        if info.get("github_workflow_ref") != expected_wf:
            p.append("github_workflow_ref does not match API repository/path/ref")
    # 3. started_at・finished_at は必須（形式・順序）
    started = _parse_ts(info.get("started_at"))
    finished = _parse_ts(run.get("finished_at"))
    if started is None:
        p.append("started_at missing or not ISO8601 with offset")
    if finished is None:
        p.append("finished_at missing or not ISO8601 with offset")
    if started is not None and finished is not None and finished < started:
        p.append("finished_at earlier than started_at")
    # 4. manifest と run.json の実行識別情報
    if not isinstance(man_run, dict):
        p.append("manifest run info missing")
    elif man_run != info:
        diff = sorted(k for k in set(man_run) | set(info) if man_run.get(k) != info.get(k))
        p.append(f"manifest run info differs from run.json info: {diff}")
    # 5. API 由来の値の記録（照合とは別に残す）
    out["recorded"] = {
        "run_json": {k: info.get(k) for k in ("github_run_id", "github_run_attempt", "github_sha", "github_ref",
                                             "github_workflow_ref", "started_at")} | {"finished_at": run.get("finished_at")},
        "api": {k: meta.get(k) for k in ("run_id", "run_attempt", "latest_attempt", "run_started_at", "job_id",
                                         "job_count", "log_sha256", "log_bytes", "log_fetched_at", *META_API_FIELDS)},
    }
    out["ok"] = not p
    return out


# ---- 条件5（07f §6.4・§6.5 の4、07g・2026-11-05 の GPT 判断 1-3） ----
def _extract_programs_by_key(data: Any) -> Any:
    """凍結版 _extract_programs と同じ規則を、集約の中で書き直したもの。

    `programs` キーがあればその値をそのまま使う（None・空でも today に切り替えない）。
    キーが無い場合に限り today.programs を使う。値の妥当性は呼び出し側で確かめる。
    """
    if not data:
        return []
    if not isinstance(data, dict):
        return None
    if "programs" in data:
        return data.get("programs", [])
    return (data.get("today") or {}).get("programs", [])


def _race_sort_key(p: dict) -> tuple:
    return (str(p.get("race_closed_at", "") or ""), int(p.get("race_stadium_number", 0) or 0),
            int(p.get("race_number", 0) or 0))


def _stable_hash(obj: Any) -> str:
    """判断D 2 の方法（評価側 race_eval.stable_hash と同じ文字列化）。"""
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=True, separators=(",", ":"))
    return _sha256(s.encode("utf-8"))


def condition5_check(raw: Optional[bytes], run: dict[str, Any], races: list[dict[str, Any]]) -> dict[str, Any]:
    """programs_raw.json から対象一覧と program_sha256 を計算し直し、評価側の記録と照合する。

    返り値 {"status": "ok"|"failed"|"not_applicable_stopped_run", "problems": [...], "checks": {...}}。
    """
    out: dict[str, Any] = {"status": "failed", "problems": [], "checks": {}}
    rec = run.get("programs")
    if not isinstance(rec, dict) or "raw_sha256" not in rec:
        if run.get("status") != "completed":
            out["status"] = "not_applicable_stopped_run"
        else:
            out["problems"].append("programs record missing in run.json")
        return out
    p, c = out["problems"], out["checks"]
    race_date = (run.get("info") or {}).get("race_date")
    if not isinstance(race_date, str) or not re.fullmatch(r"\d{8}", race_date):
        p.append("race_date (run argument) missing or invalid")
        return out
    if raw is None:
        p.append("programs_raw.json missing in artifact")
        return out
    c["raw_sha256"] = _sha256(raw) == rec.get("raw_sha256")                      # (i)
    try:
        data = json.loads(raw)                                                    # (ii)
    except ValueError:
        p.append("programs_raw.json is not JSON")
        return out
    programs = _extract_programs_by_key(data)
    if not isinstance(programs, list) or not programs or not all(isinstance(x, dict) for x in programs):
        p.append("programs list invalid (None, empty, or not a list of objects)")
        return out
    try:
        ordered = sorted(programs, key=_race_sort_key)                           # (iii)
        targets = [{"eval_id": f"{race_date}_{int(x['race_stadium_number']):02d}_{int(x['race_number']):02d}",
                    "venue_num": int(x["race_stadium_number"]), "race_number": int(x["race_number"]),
                    "race_closed_at": x.get("race_closed_at"), "race_date_in_program": x.get("race_date")}
                   for x in ordered]                                              # (iv)
    except (KeyError, TypeError, ValueError) as exc:
        p.append(f"programs element invalid: {type(exc).__name__}")
        return out
    targets_sha = _sha256(json.dumps(targets, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8"))      # (v)
    c["targets"] = targets == rec.get("targets")
    c["targets_sha256"] = targets_sha == rec.get("targets_sha256")
    by_id: dict[str, dict] = {}
    dup = set()
    for t, x in zip(targets, ordered):
        if t["eval_id"] in by_id:
            dup.add(t["eval_id"])
        by_id[t["eval_id"]] = x
    if dup:
        p.append(f"duplicate eval_id in programs: {sorted(dup)[:5]}")
    matched, mismatched, absent, unknown = 0, [], [], []                         # (vi)
    for r in races:
        eid = r.get("eval_id")
        if "program_sha256" not in r:
            absent.append(eid)
            continue
        x = by_id.get(eid)
        if x is None:
            unknown.append(eid)
        elif _stable_hash(x) == r["program_sha256"]:
            matched += 1
        else:
            mismatched.append(eid)
    c["program_sha256"] = {"matched": matched, "mismatched": mismatched[:20], "mismatched_count": len(mismatched),
                           "absent": absent[:20], "absent_count": len(absent), "not_in_programs": unknown[:20]}
    for k in ("raw_sha256", "targets", "targets_sha256"):
        if c[k] is not True:
            p.append(f"{k} mismatch")
    if mismatched:
        p.append(f"program_sha256 mismatch: {len(mismatched)}")
    if unknown:
        p.append(f"race records not in programs: {len(unknown)}")
    out["status"] = "ok" if not p else "failed"
    return out


# ---- comparable_hash の対象（07g §3.5） ----
FIVE_ITEMS = ("danger_score", "danger_breakdown", "upset_score", "rank_index", "featured_boats")


def comparable_scope_ok(rec: dict[str, Any]) -> bool:
    """comparable_hash が既決の5項目の値から作られた記録の形になっているか（構造だけを見る）。

    values の Legacy 側・Rebuild 側のキーがちょうど5項目で、comparable_hash が両側の SHA-256 を持つこと。
    run_id・時刻などのメタデータのキーが values に入っていれば不合格。値の中身（時刻に左右される値を
    含むか）は見ない（除外しない）。ハッシュの値そのものは、比較器の正規化を使わないので計算し直さない。
    """
    values, ch = rec.get("values"), rec.get("comparable_hash")
    if not isinstance(values, dict) or not isinstance(ch, dict):
        return False
    for side in ("legacy", "rebuild"):
        v = values.get(side)
        h = ch.get(side)
        if not isinstance(v, dict) or set(v) != set(FIVE_ITEMS):
            return False
        if not (isinstance(h, str) and re.fullmatch(r"[0-9a-f]{64}", h)):
            return False
    return True


# ---- 条件7 の表記（07g §3.8 の7、2026-11-05 の GPT 判断 1-2） ----
RELEASE_AVAILABILITY_NOTE = (
    "release_storage_available=false は x_release_storage.is_available() が False だった"
    "（GITHUB_TOKEN・GITHUB_REPOSITORY が無い）ことを示すだけで、外部との通信が無かったことの証明ではない。"
    "通信は評価側の監査フック（Python の範囲）で別に記録したもので、C 拡張から直接の通信は観測していない。"
    "評価側の artifact にはこの注記が無い（評価側は変更していない）。"
)


def release_and_network_record(run: dict[str, Any]) -> dict[str, Any]:
    """利用可能性と、通信の監視結果と、監視の限界を、別々の項目として記録する。"""
    inst = run.get("instrument") or {}
    events = inst.get("side_effect_events") or []
    net = [e.get("detail") for e in events if isinstance(e, dict) and e.get("kind") == "network"]
    return {
        "release_storage_available": ((run.get("checks") or {}).get("release") or {}).get("release_storage_available"),
        "network_events_observed": {"count": len(net), "details": net[:20]},
        "monitoring_limitations": inst.get("limitations"),
        "note": RELEASE_AVAILABILITY_NOTE,
    }


# ---- 評価ジョブのログの検査（§6.8 B の条件1・3・5・6。2026-11-06 の GPT 判断） ----
# E2：アクションの入力にトークンの値（GitHub が *** で伏せたもの）が渡っていないこと。
# E3：ジョブの「GITHUB_TOKEN Permissions」の表示。想定の表示と完全に一致する場合だけ合格にする。
#     想定が与えられなければ不合格（fail-closed）。想定の表示は EXPECTED_EVAL_TOKEN_PERMISSIONS。
# E5：取得のステップが出す g2_fetch_head=<40桁> と g2_credentials_check=none が、それぞれちょうど1行あること。
#     g2_fetch_head は API の head_sha と一致すること。
# いずれも、ログに *** が無いことだけを合格の根拠にはしない（E1 の静的な検査・E4 のプロセスの環境と組み合わせる）。
_TOKEN_INPUT_RE = re.compile(r"^\s+[A-Za-z0-9_-]*token[A-Za-z0-9_-]*:\s*\*\*\*\s*$", re.IGNORECASE)
_FETCH_HEAD_RE = re.compile(r"g2_fetch_head=([0-9a-f]{40})")
CREDENTIALS_CHECK_LINE = "g2_credentials_check=none"
TOKEN_PERMISSIONS_GROUP = "##[group]GITHUB_TOKEN Permissions"
# E3 の想定の表示（GPT 決定、2026-11-06）。根拠は、評価 workflow（permissions: {}）の調査 run 42190083860
# （attempt 1）のジョブログで、グループの中が「Metadata: read」の1行だけだったこと。
# グループの中の空でない行が、この並びと完全に一致する場合だけ合格にする（大文字・小文字も区別する）。
# Contents・Actions などの権限の追加、行の欠け・増え、表示の形の違い、グループの欠落・重複・未終了は不合格。
EXPECTED_EVAL_TOKEN_PERMISSIONS: tuple[str, ...] = ("Metadata: read",)


def eval_log_check(text: str) -> dict[str, Any]:
    """評価ジョブのログ原本から E2・E3・E5 の材料を取り出す（判定は aggregate で行う）。"""
    lines = [_LOG_TS_RE.sub("", ln, count=1) for ln in text.splitlines()]
    token_lines = [i + 1 for i, ln in enumerate(lines) if _TOKEN_INPUT_RE.match(ln)]
    perm: dict[str, Any] = {"display": None, "problem": None}
    starts = [i for i, ln in enumerate(lines) if ln.strip() == TOKEN_PERMISSIONS_GROUP]
    if len(starts) != 1:
        perm["problem"] = f"GITHUB_TOKEN Permissions group count = {len(starts)} (expected 1)"
    else:
        end = next((j for j in range(starts[0] + 1, len(lines)) if lines[j].strip() == GROUP_END), None)
        if end is None:
            perm["problem"] = "GITHUB_TOKEN Permissions group is not closed"
        else:
            perm["display"] = [lines[j].strip() for j in range(starts[0] + 1, end) if lines[j].strip()]
    heads = [m.group(1) for ln in lines if (m := _FETCH_HEAD_RE.fullmatch(ln.strip()))]
    creds = [ln for ln in lines if ln.strip() == CREDENTIALS_CHECK_LINE]
    return {
        "token_inputs": {"ok": not token_lines, "lines": token_lines[:20]},
        "token_permissions": perm,
        "fetch": {"heads": heads[:5], "head_count": len(heads), "credentials_check_count": len(creds)},
    }


def fetch_check(eval_log: Optional[dict[str, Any]], meta: dict[str, Any]) -> list[str]:
    """E5：取得のステップの証跡を、API の head_sha と照合する。"""
    if not isinstance(eval_log, dict):
        return ["evaluate job log not checked"]
    f = eval_log.get("fetch") or {}
    problems = []
    if f.get("head_count") != 1:
        problems.append(f"g2_fetch_head line count = {f.get('head_count')} (expected 1)")
    elif not meta.get("api_head_sha") or f["heads"][0] != meta.get("api_head_sha"):
        problems.append("g2_fetch_head does not match API head_sha")
    if f.get("credentials_check_count") != 1:
        problems.append(f"{CREDENTIALS_CHECK_LINE} line count = {f.get('credentials_check_count')} (expected 1)")
    return problems


def token_permissions_check(eval_log: Optional[dict[str, Any]], expected: Optional[list[str]]) -> list[str]:
    """E3：GITHUB_TOKEN Permissions の表示を、想定の表示と照合する。想定が無ければ不合格。"""
    if not isinstance(eval_log, dict):
        return ["evaluate job log not checked"]
    perm = eval_log.get("token_permissions") or {}
    if perm.get("problem"):
        return [perm["problem"]]
    if expected is None:
        return ["expected GITHUB_TOKEN Permissions display not determined (to be confirmed on Actions)"]
    if perm.get("display") != list(expected):
        return [f"GITHUB_TOKEN Permissions display differs: {perm.get('display')!r}"]
    return []


def process_env_check(run: dict[str, Any]) -> list[str]:
    """E4：評価プロセスに GITHUB_TOKEN が無く、名前に TOKEN を含む環境変数も無いこと（ACTIONS_RUNTIME_TOKEN を含む）。"""
    tok = (run.get("checks") or {}).get("token")
    if not isinstance(tok, dict):
        return ["token check record missing in run.json"]
    problems = []
    if tok.get("github_token_present") is not False:
        problems.append("GITHUB_TOKEN present (or unrecorded) in evaluation process")
    if tok.get("token_like_env_names") != []:
        problems.append(f"token-like environment variables in evaluation process: {tok.get('token_like_env_names')!r}")
    return problems


# ---- 集約側の基準（2026-11-06 の GPT 判断：評価側と分けて固定する） ----
# 形：{"kind": "aggregate", "files": {<self_version() の4ファイル>: <SHA-256>}, "python_version": [3, 11, 9]}
# gh・jq の版は観測値として記録するだけで、照合しない。
def aggregate_baseline_check(path: Optional[str]) -> dict[str, Any]:
    """集約側の基準と、この集約のコード（self_version）・Python の版を照合する。"""
    out: dict[str, Any] = {"status": "no_baseline", "path": path or None, "sha256": None, "diffs": []}
    if not path:
        return out
    try:
        with open(path, "rb") as f:
            raw = f.read()
        base = json.loads(raw)
        if not isinstance(base, dict):
            raise ValueError("aggregate baseline is not a JSON object")
    except (OSError, ValueError) as exc:
        out["status"] = "read_error"
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out
    out["sha256"] = _sha256(raw)
    mine = self_version()
    files = base.get("files")
    if base.get("kind") != "aggregate" or not isinstance(files, dict):
        out["status"] = "invalid"
        return out
    for rel in sorted(set(mine) | set(files)):
        if mine.get(rel) is None or files.get(rel) != mine.get(rel):
            out["diffs"].append(f"file:{rel}")
    if base.get("python_version") != list(sys.version_info[:3]):
        out["diffs"].append("python_version")
    out["status"] = "ok" if not out["diffs"] else "mismatch"
    return out


# aggregate() の formal_context の既定：どの条件も満たさない（正式な合格を出さない）
DEFAULT_FORMAL_CONTEXT: dict[str, Any] = {
    "aggregate_baseline_check": "not_checked",
    "expected_eval_token_permissions": None,  # aggregate() を直接呼ぶときの既定（不合格）。main() は EXPECTED_EVAL_TOKEN_PERMISSIONS を渡す
    "registration_verified": False,  # aggregate() を直接呼ぶときの既定（不合格）。main() は registration_check の結果を渡す
}
REGISTRATION_NOTE = (
    "baseline の登録は案A（07h §2.8、2026-11-08 に本人が採用）：g2_baseline/registered/approvals.json の承認記録"
    "（種類・パス・SHA-256・承認日）と、使った基準のパス・内容の SHA-256 を照合する。登録を確認できなければ、"
    "正式な実行に合格を出さない。候補のファイルが置かれているだけでは登録とみなさない。"
)

# ---- baseline の登録の確認（案A。07h §2.8、2026-11-08 に本人が採用） ----
# 承認記録：<リポジトリ>/g2_baseline/registered/approvals.json
#   {"kind": "g2_baseline_registration", "format_version": 1,
#    "entries": [{"type": "evaluate"|"aggregate", "path": "g2_baseline/registered/<type>/<name>.json",
#                 "sha256": "<64桁の小文字の16進>", "approved_on": "YYYY-MM-DD", "note": "<任意>"}]}
# 種類ごとにちょうど1件。登録の対象は、そのパスに置かれ、内容の SHA-256 が一致するファイルだけ。
# 照合に使った基準（--baseline・--aggregate-baseline）は、パス（リポジトリからの相対、実体のパス）と内容の
# SHA-256 の両方が、その種類の記録と一致しなければならない。どれか1つでも満たさなければ確認できないとする。
REGISTRY_REL = "g2_baseline/registered/approvals.json"
REGISTRY_KIND = "g2_baseline_registration"
REGISTRY_FORMAT_VERSION = 1
REGISTRY_TYPES = ("evaluate", "aggregate")
_REGISTRY_REQUIRED = ("type", "path", "sha256", "approved_on")
_REGISTRY_OPTIONAL = ("note",)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_REG_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+\.json$")


def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", ".."))


def _repo_relpath(repo_root: str, path: str) -> Optional[str]:
    """path の実体の、リポジトリからの相対パス（posix）。リポジトリの外なら None。"""
    try:
        root = os.path.realpath(repo_root)
        real = os.path.realpath(path)
        rel = os.path.relpath(real, root)
    except (OSError, ValueError):
        return None
    if rel == os.curdir or rel.startswith(os.pardir + os.sep) or rel == os.pardir or os.path.isabs(rel):
        return None
    return rel.replace(os.sep, "/")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """承認記録の JSON で、同じキーが2回以上ある object を拒否する（後の値で黙って上書きしない）。"""
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}")
        out[k] = v
    return out


def _registry_entry_problems(e: Any) -> list[str]:
    if not isinstance(e, dict):
        return ["entry is not an object"]
    probs = []
    missing = [k for k in _REGISTRY_REQUIRED if k not in e]
    extra = [k for k in e if k not in _REGISTRY_REQUIRED + _REGISTRY_OPTIONAL]
    if missing:
        probs.append(f"missing keys {missing}")
    if extra:
        probs.append(f"unexpected keys {sorted(extra)}")
    t = e.get("type")
    if t not in REGISTRY_TYPES:
        probs.append(f"type {t!r} not in {list(REGISTRY_TYPES)}")
    path = e.get("path")
    if not isinstance(path, str):
        probs.append("path is not a string")
    else:
        parts = path.split("/")
        if (len(parts) != 4 or parts[:3] != ["g2_baseline", "registered", str(t)]
                or not _REG_NAME_RE.match(parts[3]) or parts[3] in (".json", "..json") or "\\" in path):
            probs.append(f"path {path!r} is not g2_baseline/registered/{t}/<name>.json")
    if not (isinstance(e.get("sha256"), str) and _SHA256_RE.match(e["sha256"])):
        probs.append("sha256 is not 64 lowercase hex digits")
    d = e.get("approved_on")
    if not (isinstance(d, str) and _DATE_RE.match(d)):
        probs.append("approved_on is not YYYY-MM-DD")
    else:
        try:
            datetime.strptime(d, "%Y-%m-%d")
        except ValueError:
            probs.append("approved_on is not a valid date")
    if "note" in e and not isinstance(e["note"], str):
        probs.append("note is not a string")
    return probs


def registration_check(repo_root: str,
                       eval_baseline_path: Optional[str], eval_baseline_sha256: Optional[str],
                       agg_baseline_path: Optional[str], agg_baseline_sha256: Optional[str]) -> dict[str, Any]:
    """案A の登録の確認。status が "ok" のときだけ registration_verified=True にする。

    status："ok" / "registry_missing" / "registry_invalid" / "registered_file_invalid" /
            "evaluate_baseline_not_registered" / "aggregate_baseline_not_registered"
            （main() は、この関数の想定外の例外を "check_error" として扱う）
    """
    out: dict[str, Any] = {"status": None, "registry": REGISTRY_REL, "registry_sha256": None,
                           "problems": [], "entries": {}}

    def fail(status: str, problem: str) -> dict[str, Any]:
        out["status"] = status
        out["problems"].append(problem)
        return out

    reg_path = os.path.join(repo_root, *REGISTRY_REL.split("/"))
    if not os.path.isfile(reg_path) or os.path.islink(reg_path):
        return fail("registry_missing", f"{REGISTRY_REL} not found (or is a symlink)")
    if _repo_relpath(repo_root, reg_path) != REGISTRY_REL:
        # 途中のディレクトリがシンボリックリンクで、実体が別の場所にある
        return fail("registry_invalid", f"{REGISTRY_REL} does not resolve to itself inside the repository")
    try:
        with open(reg_path, "rb") as f:
            raw = f.read()
        reg = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except (OSError, ValueError) as exc:
        return fail("registry_invalid", f"{type(exc).__name__}: {exc}")
    out["registry_sha256"] = _sha256(raw)
    if not isinstance(reg, dict) or set(reg) != {"kind", "format_version", "entries"}:
        return fail("registry_invalid", "registry must be an object with exactly kind, format_version, entries")
    if reg.get("kind") != REGISTRY_KIND or reg.get("format_version") != REGISTRY_FORMAT_VERSION \
            or isinstance(reg.get("format_version"), bool):
        return fail("registry_invalid", "kind or format_version differs")
    entries = reg.get("entries")
    if not isinstance(entries, list):
        return fail("registry_invalid", "entries is not a list")
    by_type: dict[str, list[dict[str, Any]]] = {t: [] for t in REGISTRY_TYPES}
    for i, e in enumerate(entries):
        probs = _registry_entry_problems(e)
        if probs:
            return fail("registry_invalid", f"entry {i}: {'; '.join(probs)}")
        by_type[e["type"]].append(e)
    for t in REGISTRY_TYPES:
        if len(by_type[t]) != 1:
            return fail("registry_invalid", f"{t}: {len(by_type[t])} entries (expected exactly 1)")
    for t in REGISTRY_TYPES:
        e = by_type[t][0]
        p = os.path.join(repo_root, *e["path"].split("/"))
        if not os.path.isfile(p) or os.path.islink(p) or _repo_relpath(repo_root, p) != e["path"]:
            return fail("registered_file_invalid", f"{t}: {e['path']} not found as a regular file")
        try:
            with open(p, "rb") as f:
                got = _sha256(f.read())
        except OSError as exc:
            return fail("registered_file_invalid", f"{t}: {e['path']} unreadable ({type(exc).__name__})")
        if got != e["sha256"]:
            return fail("registered_file_invalid", f"{t}: SHA-256 of {e['path']} differs from the record")
        out["entries"][t] = {"path": e["path"], "sha256": e["sha256"], "approved_on": e["approved_on"]}
    for t, path, sha, status in (("evaluate", eval_baseline_path, eval_baseline_sha256,
                                  "evaluate_baseline_not_registered"),
                                 ("aggregate", agg_baseline_path, agg_baseline_sha256,
                                  "aggregate_baseline_not_registered")):
        e = by_type[t][0]
        if not path or not sha:
            return fail(status, f"{t}: no baseline was given or it could not be read")
        rel = _repo_relpath(repo_root, path)
        if rel != e["path"]:
            return fail(status, f"{t}: baseline path {rel!r} is not the registered path {e['path']!r}")
        if sha != e["sha256"]:
            return fail(status, f"{t}: baseline SHA-256 differs from the registered one")
    out["status"] = "ok"
    return out


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
    res["eval_log"] = log_check.get("eval_log")
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
        raw = zf.read("programs_raw.json") if "programs_raw.json" in names else None
        res["condition5"] = condition5_check(raw, run_json, races)
        res["run_metadata"] = run_metadata_check(res)
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
    res["eval_log"] = eval_log_check(text)  # E2・E3・E5 の材料（判定は aggregate）
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
    """M7-3 の照合結果："ok" / "no_baseline" / "baseline_invalid" / "log_unverified" / "mismatch"。

    2026-11-06 の GPT 判断（M7-3 の本人承認は未了）：基準の固定条件は Image だけ。基準の runner_image は
    {"image": str} の形とし、Version はジョブログの観測値として記録するだけで照合しない。
    """
    log = (vr.get("runner_image") or {}).get("log")
    if not log:
        return "log_unverified"
    if baseline is None:
        return "no_baseline"
    ri = baseline.get("runner_image")
    if not isinstance(ri, dict) or set(ri) != {"image"} or not isinstance(ri.get("image"), str) or not ri["image"]:
        return "baseline_invalid"
    return "ok" if ri["image"] == log["image"] else "mismatch"


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
    if not comparable_scope_ok(rec):
        return "evidence_incomplete"  # comparable_hash が5項目の値から作られた形になっていない
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
              baseline_sha256: Optional[str] = None,
              formal_context: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """基準が無い・評価側と同じ基準だと確かめられない・形が正しくない・ジョブログの
    Runner Image と一致しない実行は、その日の全レースを正式カウントから除外し、区間を
    終える（M7-3・基準の同一性。評価側の A-1 に相当）。これらは baseline_failures に入る。

    formal_context（2026-11-06）：集約側の基準の照合結果、E3 の想定の表示、登録の確認の結果。
    省略すると DEFAULT_FORMAL_CONTEXT（どれも満たさない）になり、正式な実行には合格を出さない。
    main() は registration_check（案A、2026-11-08）の結果が "ok" のときだけ registration_verified=True を渡す。
    """
    ctx = dict(DEFAULT_FORMAL_CONTEXT, **(formal_context or {}))
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
                md = sel.get("run_metadata") or {"ok": False, "problems": ["run metadata not checked"]}
                day["run_metadata_check"] = "ok" if md.get("ok") else md.get("problems")
                c5 = sel.get("condition5") or {"status": "failed", "problems": ["condition5 not checked"]}
                day["condition5_check"] = c5["status"] if c5["status"] != "failed" else c5.get("problems")
                ev = sel.get("eval_log")
                # E2（アクションの入力のトークン）・E4（評価プロセスの環境）は、止まった実行にも当てはめる
                day["eval_token_input_check"] = ("ok" if isinstance(ev, dict) and (ev.get("token_inputs") or {}).get("ok")
                                                 else ["token value passed to an action input (or log not checked)"])
                day["eval_process_env_check"] = process_env_check(run) or "ok"
                passing = ("ok", "not_applicable_stopped_run")
                if day["ref_check"] != "ok":
                    failed_check = "ref_not_main"  # 正式な実行は main で行う（調査用の実行には適用しない）
                elif id_check not in passing:
                    failed_check = f"baseline_{id_check}"
                elif ri_check not in passing:
                    failed_check = f"runner_image_{ri_check}"
                elif day["w7_check"] not in passing:
                    failed_check = "w7_monitoring"
                elif day["run_metadata_check"] != "ok":
                    failed_check = "run_metadata"  # 証跡の不備（比較値の差分ではない）
                elif day["condition5_check"] not in passing:
                    failed_check = "condition5"  # 対象一覧・program_sha256 の照合の不合格
                elif day["eval_token_input_check"] != "ok":
                    failed_check = "eval_token_input"  # E2
                elif day["eval_process_env_check"] != "ok":
                    failed_check = "eval_process_token_env"  # E4
                else:
                    failed_check = None
                if failed_check is None and run.get("status") == "completed" \
                        and not (run.get("config_detection") or {}).get("detected", True) and targets:
                    # 加算されうる実行にだけ当てはめる条件（止まった実行は、もともと加算されない）
                    day["aggregate_baseline_check"] = ctx["aggregate_baseline_check"]
                    day["eval_fetch_check"] = fetch_check(ev, sel.get("meta") or {}) or "ok"
                    day["eval_token_permissions_check"] = (
                        token_permissions_check(ev, ctx["expected_eval_token_permissions"]) or "ok")
                    day["registration_check"] = "ok" if ctx["registration_verified"] is True else "unverified"
                    if day["aggregate_baseline_check"] != "ok":
                        failed_check = f"aggregate_baseline_{day['aggregate_baseline_check']}"
                    elif day["eval_fetch_check"] != "ok":
                        failed_check = "eval_fetch"  # E5
                    elif day["eval_token_permissions_check"] != "ok":
                        failed_check = "eval_token_permissions"  # E3
                    elif day["registration_check"] != "ok":
                        failed_check = "baseline_registration_unverified"  # 候補の基準では正式な合格を出さない
                if failed_check is not None:
                    # ref・基準の同一性・M7-3 の Runner Image・W7 の監視・実行メタデータ・条件5・E2〜E5・
                    # 集約側の基準・登録の確認のいずれかを満たさない
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
    p.add_argument("--baseline", default="", help="評価側の基準 JSON（M7-3 の runner_image を照合する）")
    p.add_argument("--aggregate-baseline", default="", help="集約側の基準 JSON（集約のコード・Python の版）")
    p.add_argument("--tools-json", default="", help="gh・jq の版（観測値。照合しない）")
    args = p.parse_args(argv)
    try:
        tools = json.loads(args.tools_json) if args.tools_json else None
    except ValueError:
        tools = {"unparsed": args.tools_json[:200]}
    agg_base = aggregate_baseline_check(args.aggregate_baseline or None)

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
    # 登録の確認（案A、2026-11-08）。基準が読めない場合は SHA-256 が無いので確認できない。
    # E3 の想定の表示は EXPECTED_EVAL_TOKEN_PERMISSIONS（GPT 決定、2026-11-06）。
    try:
        registration = registration_check(_repo_root(), args.baseline or None, baseline_sha256,
                                          args.aggregate_baseline or None, agg_base.get("sha256"))
    except Exception as exc:  # 想定外の例外でも登録を確認できたとはしない（fail-closed）
        registration = {"status": "check_error", "registry": REGISTRY_REL, "registry_sha256": None,
                        "problems": [f"{type(exc).__name__}: {exc}"], "entries": {}}
    registration_verified = registration["status"] == "ok"
    formal_context = {"aggregate_baseline_check": agg_base["status"],
                      "expected_eval_token_permissions": list(EXPECTED_EVAL_TOKEN_PERMISSIONS),
                      "registration_verified": registration_verified}
    result = aggregate(verified, load_permissions(args.permissions), args.first_date, args.last_date,
                       baseline, baseline_sha256, formal_context)
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
        out_run["run_metadata"] = vr.get("run_metadata")
        out_run["condition5"] = vr.get("condition5")
        out_run["release_and_network"] = release_and_network_record(run_json) if vr.get("verified") else None
        out_run["eval_log"] = vr.get("eval_log")
        out_run["eval_process_env"] = process_env_check(run_json) if vr.get("verified") else None
        out_run["eval_fetch"] = fetch_check(vr.get("eval_log"), vr.get("meta") or {}) if vr.get("verified") else None
        # E3 の照合結果を実行ごとに記録する（調査の実行を含む。正式カウントの判定は aggregate() のゲートで行う）
        out_run["eval_token_permissions"] = (
            token_permissions_check(vr.get("eval_log"), formal_context["expected_eval_token_permissions"])
            if vr.get("verified") else None)
    result["release_availability_note"] = RELEASE_AVAILABILITY_NOTE
    result["aggregate_baseline"] = agg_base
    result["aggregate_environment"] = {  # 観測値（照合しない）
        "github_sha": os.environ.get("GITHUB_SHA"), "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "python_version": list(sys.version_info[:3]), "tools": tools,
    }
    result["registration"] = registration
    result["formal_judgment"] = {"registration_verified": registration_verified, "note": REGISTRATION_NOTE,
                                 "expected_eval_token_permissions": formal_context["expected_eval_token_permissions"]}
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
          f"best_segment={result['best_segment']} g2_status={result['g2_status']} "
          f"aggregate_baseline={agg_base['status']} registration_verified={registration_verified} "
          f"registration={registration['status']}")
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
