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
<run_id>-<run_attempt>/permission_evidence.json … 正式な実行の許可ファイルの履歴（集約 workflow が API で取る。
                                          項目4、2026-11-08。下の permission_check を参照）
--permissions：廃止（2026-11-08）。正式な実行の許可は、リポジトリの g2_permissions/<permission_id>.json と
  その履歴（permission_evidence.json）だけで確かめる。渡すとエラー（終了コード 2）。
--baseline：評価側の基準 JSON（M7-3：runner_image の Image をジョブログの値と照合する。Version は観測値）
--aggregate-baseline：集約側の基準 JSON（集約の4ファイルの SHA-256 と Python の版。2026-11-06 に追加）
--tools-json：gh・jq の版（観測値。照合しない）

出力：
  aggregate_result.json（判定に使った集約コード・workflow の SHA-256 を含む）
  logs/<run_id>-<run_attempt>.log（ジョブログ原本の写し）
  manifest.json（上の各ファイルの存在・サイズ・SHA-256。"complete": true）
終了コード：照合の失敗（artifact・ジョブログの検証、基準の読み込み、正式な実行の ref・基準の
同一性・Runner Image・W7 の監視・実行メタデータ・条件5 の照合）が1件でもあれば 3。
それが無く、race_closed_at の検証の失敗（input_order_failures）があれば 4（V4、2026-11-09）。どちらも無ければ 0。
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
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

REQUIRED_RACE_KEYS = ("eval_id", "result_category", "d9_states", "counts", "d5")
REQUIRED_COMPARED_KEYS = ("values", "comparable_hash", "hashes", "d13_2", "diffs")
TARGET = 100
REACH_EVENT_TYPE = "g2_streak_100_reached"
REACH_STATUS = "reached_not_yet_formally_achieved"  # 100件到達。正式な達成（d5 の正本の保存の後）とは別
EXIT_VERIFICATION_FAILED = 3
# V4（GPT 決定 2026-11-09）：実行の入力の整合・順序の検証（race_closed_at）の失敗。baseline の登録・照合の失敗
# （3）とは分ける。両方があるときは 3 を返す（どちらも結果の JSON に別々に記録する）。
EXIT_INPUT_ORDER_FAILED = 4

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
                "rebuild/formal_g2/aggregate/fetch_log.py", "rebuild/formal_g2/aggregate/canonical.py",
                ".github/workflows/formal_g2_aggregate.yml"):
        p = os.path.join(repo_root, rel)
        try:
            with open(p, "rb") as f:
                out[rel] = _sha256(f.read())
        except OSError:
            out[rel] = None
    return out


# ---- P4 4・5（07f §6.4・§6.5 の 6。GPT判断 11/04・11/05）----
# 4：次の4つを別々の事実として記録する。artifact の保存の失敗、正本の保存の失敗、証拠の欠落、保持期限切れ。
#    正本（g2_evidence/<permission_id>/）の保存と確認は d5 で、集約はまだ確かめない（"not_checked"）。
ARTIFACT_STATUS_FACTS = {
    "missing": "artifact_save_failure",        # 実行に artifact が無い（保存されなかった）
    "expired": "retention_expired",            # artifact はあったが保持期限切れ（中身を確かめられない）
    "download_failed": "artifact_download_failed",  # 4つの事実のどれでもない、集約側の取得の失敗
}
CANONICAL_SAVE_STATUS = "not_checked (d5 not implemented in the aggregate)"


_ARTIFACT_NAME_RE = re.compile(r"^g2-([0-9]{8})-([0-9]+)-([0-9]+)$")


def unavailable_run_date(vr: dict[str, Any]) -> Optional[str]:
    """中身（manifest）を確かめられない実行の race_date を、artifact の名前（評価 workflow の
    g2-<race_date>-<run_id>-<run_attempt>）から取る。名前が無い・形が違う・run_id/attempt が合わないなら None。"""
    if vr.get("manifest") is not None:
        return None
    m = vr.get("meta") or {}
    mm = _ARTIFACT_NAME_RE.match(str(m.get("artifact_name") or ""))
    if not mm or mm.group(2) != str(m.get("run_id")) or mm.group(3) != str(m.get("run_attempt")):
        return None
    try:
        datetime.strptime(mm.group(1), "%Y%m%d")
    except ValueError:
        return None
    return mm.group(1)


def record_facts(verified: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for vr in verified:
        fact = vr.get("record_fact")
        if fact:
            m = vr.get("meta") or {}
            out.append({"run": vr.get("dir"), "run_id": m.get("run_id"), "run_attempt": m.get("run_attempt"),
                        "fact": fact, "artifact_expires_at": m.get("artifact_expires_at")})
    return out


def input_identity(verified: list[dict[str, Any]], first_date: str, last_date: str) -> dict[str, Any]:
    """P4 5：「同じ run_id・同じ成果物の SHA-256」かどうかを比べるための、集約の入力の識別。"""
    runs = sorted(({"run_id": str((vr.get("meta") or {}).get("run_id")),
                    "run_attempt": str((vr.get("meta") or {}).get("run_attempt")),
                    "artifact_sha256": vr.get("artifact_sha256")} for vr in verified),
                  key=lambda x: (x["run_id"], x["run_attempt"], str(x["artifact_sha256"])))
    return {"first_date": first_date, "last_date": last_date, "runs": runs}


def _race_states(res: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for x in res.get("sequence") or []:
        if isinstance(x, dict) and "eval_id" in x:
            out[str(x["eval_id"])] = str(x.get("state"))
    return out


REACH_IDENTITY_KEYS = ("first_race_key", "hundredth_race_key", "race_date", "evaluation_run_id",
                       "evaluation_run_attempt")


def _race_run_identity(res: dict[str, Any]) -> dict[str, tuple[str, str, Optional[str]]]:
    """eval_id → その日に選ばれた実行の (run_id, run_attempt, 成果物の SHA-256)。"""
    sha = {(r.get("run_id"), r.get("run_attempt")): r.get("artifact_sha256")
           for r in ((res.get("input_identity") or {}).get("runs") or []) if isinstance(r, dict)}
    by_date = {}
    for d in res.get("days") or []:
        sr = d.get("selected_run") if isinstance(d, dict) else None
        if isinstance(sr, dict):
            k = (str(sr.get("run_id")), str(sr.get("run_attempt")))
            by_date[d.get("race_date")] = (k[0], k[1], sha.get(k))
    out = {}
    for e in _race_states(res):
        ident = by_date.get(e.split("_", 1)[0])
        if ident is not None:
            out[e] = ident
    return out


def _segment_ids(res: dict[str, Any]) -> list[str]:
    """到達した区間（1件目〜100件目）の eval_id。到達が無ければ空。"""
    reach = res.get("reach") if isinstance(res, dict) else None
    if not isinstance(reach, dict):
        return []
    ids = [str(x["eval_id"]) for x in res.get("sequence") or [] if isinstance(x, dict) and "eval_id" in x]
    try:
        a, b = ids.index(str(reach.get("first_race_key"))), ids.index(str(reach.get("hundredth_race_key")))
    except ValueError:
        return []
    return ids[a:b + 1]


def compare_with_previous(prev: Any, cur: dict[str, Any]) -> dict[str, Any]:
    """P4 5：集約のやり直しの結果を、前回の結果と比べて記録する。前回の結果は変えない・置き換えない。

    同じ入力かどうかは、レースごとに、その日に選ばれた実行の (run_id, run_attempt, 成果物の SHA-256) で決める
    （原文「同じ run_id、同じ成果物の SHA-256」。監査 2026-11-09 の指摘3）。集約の期間や、ほかの実行が増えたか
    どうかには左右されない。
    - 同じ実行・同じ成果物のレースで結果が違う：P4-D の疑義として H1（前回 +1＝added だったレースに限る。
      前回 0 だったレースの違いは事象として記録するだけ。7 の 10）。H2 は付けない（P4 11）。
    - 選ばれた実行・成果物が違うレース、どちらかにしか無いレース：事象として記録するだけにする。
    - 前回の到達の記録は変えずに写す。今回の到達がそれと違えば、達成の判定を保留する（main()）。
    前回の正式実行記録を自動で 0 にしない（訂正は P4 12）。
    """
    out: dict[str, Any] = {"previous_result_sha256": None, "comparison": None,
                           "note": "the previous result is not overwritten, cancelled or replaced; "
                                   "no automatic 0 (corrections only by P4 12)"}
    if not isinstance(prev, dict) or not isinstance(prev.get("input_identity"), dict):
        out["comparison"] = "previous_unusable"
        return out
    pi, ci = prev["input_identity"], cur["input_identity"]
    same_runs = pi.get("runs") == ci.get("runs")
    same_period = (pi.get("first_date"), pi.get("last_date")) == (ci.get("first_date"), ci.get("last_date"))
    out["comparison"] = ("same_input" if same_runs and same_period
                         else "period_changed" if same_runs else "different_input")
    out["aggregator_version"] = {"previous": prev.get("aggregator_version"), "current": cur.get("aggregator_version"),
                                 "same": prev.get("aggregator_version") == cur.get("aggregator_version")}
    ps, cs = _race_states(prev), _race_states(cur)
    pid_, cid_ = _race_run_identity(prev), _race_run_identity(cur)
    changed = sorted(e for e in set(ps) | set(cs) if ps.get(e) != cs.get(e))
    pr, cr = prev.get("reach"), cur.get("reach")
    out["previous_reach"] = pr  # 前回の到達の記録（不変。ここでは写すだけ）
    out["previous_p4_holds"] = prev.get("p4_holds") if isinstance(prev.get("p4_holds"), dict) else None
    out["reach_same"] = ((pr is None and cr is None) or (isinstance(pr, dict) and isinstance(cr, dict)
                         and all(pr.get(k) == cr.get(k) for k in REACH_IDENTITY_KEYS)))
    out["previous_reached_segment"] = _segment_ids(prev)
    out["race_differences"] = [{"eval_id": e, "previous_state": ps.get(e), "current_state": cs.get(e),
                                "same_run_and_artifact": e in pid_ and pid_.get(e) == cid_.get(e)
                                and pid_[e][2] is not None}
                               for e in changed][:200]
    out["race_difference_count"] = len(changed)
    same_art = {e for e in changed if e in pid_ and pid_.get(e) == cid_.get(e) and pid_[e][2] is not None}
    out["doubts"] = [{"eval_id": e, "previous_state": ps.get(e), "current_state": cs.get(e),
                      "run": list(pid_[e]), "case": "P4-D", "hold": "H1"}
                     for e in sorted(same_art) if ps.get(e) == "added"]
    out["events_only"] = [e for e in changed if e not in {d["eval_id"] for d in out["doubts"]}][:200]
    return out


def subsequent_run_differences(verified: list[dict[str, Any]], result: dict[str, Any]) -> list[dict[str, Any]]:
    """後続の実行（同じ race_date の、選ばれなかった実行：2回目の正式実行・調査用の実行）との結果の違いを
    事象として記録する。疑義（P4 10）・正本の候補（P4 14・15）の扱いは段階5 で行い、ここでは判定に使わない。"""
    out = []
    for day in result.get("days") or []:
        sel_meta = day.get("selected_run")
        if not sel_meta:
            continue
        sel = next((vr for vr in verified if vr.get("meta") is sel_meta or vr.get("meta") == sel_meta), None)
        if sel is None or not sel.get("verified"):
            continue
        sel_cat = {r.get("eval_id"): r.get("result_category") for r in sel.get("races") or [] if isinstance(r, dict)}
        for vr in verified:
            if vr is sel or not vr.get("verified"):
                continue
            info = (vr.get("manifest") or {}).get("run") or {}
            if info.get("race_date") != day["race_date"]:
                continue
            for r in vr.get("races") or []:
                if not isinstance(r, dict):
                    continue
                e = r.get("eval_id")
                if sel_cat.get(e) != r.get("result_category"):
                    out.append({"race_date": day["race_date"], "eval_id": e, "selected_run": sel.get("dir"),
                                "other_run": vr.get("dir"), "other_mode": info.get("mode"),
                                "order_vs_selected": _order_vs_reach(vr.get("meta") or {}, sel.get("meta")),
                                "selected_category": sel_cat.get(e), "other_category": r.get("result_category")})
    return out[:500]


# ---- 段階5：P4 10・14・15・15-2（07f §6.4・§6.5 の 7。GPT判断：確定 11/04）----
# 15-2 の「記録している入力の識別子とハッシュ」として比べる項目（評価前の時点。*_id はメモリの番地なので除く）
# Y2（GPT 決定 2026-11-09）：15-2 で比べる「記録している入力の識別子とハッシュ」を固定する。
# どれも評価前の時点（hashes.before）の値で、評価側（race_eval._cache_hashes）が書く。*_id（メモリの番地）は除く。
#   program_sha256   出走表の要素（provider から読んだ program）の stable_hash（race_eval.py）
#   b_L              Rebuild 側の艇ごとの入力の列 L の stable_hash
#   c_legacy_boats   Legacy 側の艇ごとの入力の列（c_vars）の stable_hash
#   d_cfg            評価の設定 ctx.cfg の stable_hash
#   e_venue_stats    x_venue_stats._VENUE_STATS_CACHE の stable_hash
#   e_venue_config   x_venue_stats._VENUE_CONFIG_CACHE の stable_hash
#   f_fan_cache      notify_arashi._FAN_CACHE の stable_hash
#   f_fan_file       notify_arashi._get_fan_file() のファイル名（文字列。中身は f_fan_cache で比べる）
#   a_boatinfo       provider の艇の情報（boat_objs）の stable_hash
# 正規化：f_fan_file 以外は小文字16進64桁の SHA-256 だけを受け付ける（大文字・UNSERIALIZABLE・欠落は不可）。
# 欠損時：どれか1つでも ① か比べる実行で欠ける・形が違えば P4-C（確定できない）とし、欠損値から同一性を推測しない。
INPUT_IDENTITY_KEYS = ("b_L", "c_legacy_boats", "d_cfg", "e_venue_stats", "e_venue_config", "f_fan_cache",
                       "f_fan_file", "a_boatinfo")
COMPARABLE_STATES = ("added", "mismatch")  # 正常に比べられたレース（差分ゼロ・不一致が確定した）


def _rec_by_eid(vr: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, list] = {}
    for r in vr.get("races") or []:
        if isinstance(r, dict):
            out.setdefault(str(r.get("eval_id")), []).append(r)
    return {k: v[0] for k, v in out.items() if len(v) == 1}


def input_case(r1: dict[str, Any], r2: dict[str, Any]) -> tuple[str, list[str]]:
    """15-2・P4-A〜C：① と比べる実行の入力が同じか。("A", []) すべて一致／("B", 違う項目)／("C", 確定できない項目)。"""
    undetermined, differ = [], []
    pairs = [("program_sha256", r1.get("program_sha256"), r2.get("program_sha256"))]
    h1 = ((r1.get("hashes") or {}).get("before") or {}) if isinstance(r1.get("hashes"), dict) else {}
    h2 = ((r2.get("hashes") or {}).get("before") or {}) if isinstance(r2.get("hashes"), dict) else {}
    pairs += [(k, h1.get(k) if isinstance(h1, dict) else None, h2.get(k) if isinstance(h2, dict) else None)
              for k in INPUT_IDENTITY_KEYS]
    for k, a, b in pairs:
        ok_a = isinstance(a, str) and (k == "f_fan_file" or _HEX64_RE.match(a) is not None)
        ok_b = isinstance(b, str) and (k == "f_fan_file" or _HEX64_RE.match(b) is not None)
        if not (ok_a and ok_b):
            undetermined.append(k)
        elif a != b:
            differ.append(k)
    if undetermined:
        return "C", undetermined
    return ("B", differ) if differ else ("A", [])


def _states_of(sub: dict[str, Any], vr: dict[str, Any]) -> tuple[Optional[str], dict[str, str]]:
    """内部の集約（force_formal）の結果から、実行全体の不適格の理由と、レースの状態を取る。
    15(a)：途中で止まった実行・実行全体の条件（run_level_ok）を満たさない実行も、全体として不適格とする。"""
    day = (sub.get("days") or [{}])[0]
    run = vr.get("run") or {}
    unfit = (day.get("excluded") or ("verification_failed" if day.get("verification_failed") else None)
             or (None if day.get("class") == "normal" else "not_normal_day")
             or (f"stopped (status={run.get('status')!r})" if run.get("status") != "completed" else None)
             or (None if run_level_ok(run, len(vr.get("races") or [])) else "run_level_checks_failed"))
    return unfit, _race_states(sub)


def _investigation_states(vr: dict[str, Any]) -> tuple[Optional[str], dict[str, str], dict[str, Any]]:
    """③（調査用の実行）：比較の規則（15）は当てはめない。artifact の照合（verify_run）に合格していれば、
    正式の関門（ref・基準・E3・登録・W7 など）に関わらず、レースの状態を ① と同じ race_state の規則で求める。
    関門の結果は記録だけにする（監査 2026-11-09 の指摘4）。"""
    if not vr.get("verified"):
        return "verification_failed", {}, {}
    run = vr.get("run") or {}
    by: dict[Any, list] = {}
    for r in vr.get("races") or []:
        if isinstance(r, dict):
            by.setdefault(r.get("eval_id"), []).append(r)
    states = {str(k): (race_state(v[0], True) if len(v) == 1 else "duplicate_eval_id") for k, v in by.items()}
    gates = {"status": run.get("status"), "run_level_ok": run_level_ok(run, len(vr.get("races") or []))}
    return None, states, gates


def stage5(verified: list[dict[str, Any]], result: dict[str, Any], baseline: Optional[dict[str, Any]],
           baseline_sha256: Optional[str], formal_context: dict[str, Any]) -> dict[str, Any]:
    """P4 14 の区分、15 の ② との比較、15-2・P4 8・P4-A〜C による H1・H2。どの記録も消さない・変えない。
    ① の記録を自動で 0 にしない。② を正本に繰り上げない（その日の扱いは aggregate() のまま）。"""
    states1 = _race_states(result)
    classification, comparisons, doubts, events = [], [], [], []
    insufficient: list[dict[str, Any]] = []
    by_dir = {vr.get("dir"): vr for vr in verified}
    for day in result.get("days") or []:
        ds = day["race_date"]
        sel_meta = day.get("selected_run") or day.get("first_formal_candidate")
        sel = next((vr for vr in verified if sel_meta is not None and vr.get("meta") is sel_meta), None)
        c: dict[str, Any] = {"race_date": ds, "first_formal": sel.get("dir") if sel else None,
                             "second_formal": [], "investigation": [], "undetermined": []}
        same_day = [vr for vr in verified if vr is not sel
                    and ((vr.get("manifest") or {}).get("run") or {}).get("race_date") == ds]
        for vr in same_day:
            info = (vr.get("manifest") or {}).get("run") or {}
            mode = info.get("mode")
            order = _order_vs_reach(vr.get("meta") or {}, sel.get("meta")) if sel else "no_first_formal"
            if mode == "formal" and sel is not None and order == "after":
                kind = ("technical_retry" if str((vr.get("meta") or {}).get("run_id")) ==
                        str((sel.get("meta") or {}).get("run_id")) else "new_dispatch")
                c["second_formal"].append({"run": vr.get("dir"), "kind": kind})
                events.append({"event": "formal_processing_overlap", "race_date": ds, "run": vr.get("dir"),
                               "first_formal": sel.get("dir"), "kind": kind})
            elif mode == "investigation":
                c["investigation"].append({"run": vr.get("dir"), "order_vs_first_formal": order})
            else:
                c["undetermined"].append({"run": vr.get("dir"), "mode": mode, "order_vs_first_formal": order})
                events.append({"event": "run_class_undetermined", "race_date": ds, "run": vr.get("dir"),
                               "mode": mode, "order_vs_first_formal": order})
        perms = sorted({str(((vr.get("permission_check") or {}).get("permission") or {}).get("permission_id"))
                        for vr in [sel] + same_day if vr is not None
                        and ((vr.get("permission_check") or {}).get("permission") or {}).get("permission_id")})
        if len(perms) > 1:  # P3：1つの race_date の正式な実行の許可は1回まで
            events.append({"event": "multiple_permissions_for_race_date", "race_date": ds, "permissions": perms})
        classification.append(c)
        if sel is None:
            continue
        # 15(a)：中身を確かめられない（artifact の無い・期限切れなど）実行で、① より後に始まったもの
        for b in verified:
            if (b.get("manifest") is None and unavailable_run_date(b) == ds
                    and _order_vs_reach(b.get("meta") or {}, sel.get("meta")) == "after"):
                events.append({"event": "compared_run_unfit", "race_date": ds, "run": b.get("dir"),
                               "reason": f"unavailable ({b.get('record_fact')})", "source": "P4-15"})
        recs1 = _rec_by_eid(sel)
        targets1 = [t.get("eval_id") for t in ((sel.get("run") or {}).get("programs") or {}).get("targets") or []
                    if isinstance(t, dict)]
        for kind_key, source in (("second_formal", "P4-15"), ("investigation", "P4-8")):
            for item in c[kind_key]:
                vr2 = by_dir[item["run"]]
                gates: dict[str, Any] = {}
                try:
                    if source == "P4-15":
                        sub = aggregate([vr2], [], ds, ds, baseline, baseline_sha256, formal_context,
                                        force_formal=True)
                        unfit, states2 = _states_of(sub, vr2)
                    else:
                        unfit, states2, gates = _investigation_states(vr2)
                except Exception as exc:  # noqa: BLE001  比べる実行の形の異常で集約全体を止めない（記録する）
                    unfit, states2 = f"check_error: {type(exc).__name__}", {}
                cmp: dict[str, Any] = {"run": item["run"], "first_formal": sel.get("dir"), "source": source,
                                       "run_unfit": unfit, "formal_gates_observed": gates, "races": []}
                comparisons.append(cmp)
                if unfit is not None:
                    # 15(a)：② を比較の対象から外し、事象を記録する。① はそのまま。H1・H2 は付けない
                    events.append({"event": "compared_run_unfit", "race_date": ds, "run": item["run"],
                                   "reason": unfit, "source": source})
                    continue
                recs2 = _rec_by_eid(vr2)
                for eid in sorted({str(x) for x in targets1} | set(states2)):
                    s1, s2 = states1.get(eid), states2.get(eid)
                    entry = {"eval_id": eid, "first_state": s1, "other_state": s2}
                    if s1 != "added":
                        entry["case"] = "event_only_first_not_added"  # 10：① で 0 だったレースは事象だけ
                        if s1 != s2:
                            cmp["races"].append(entry)
                        continue
                    if s2 not in COMPARABLE_STATES:
                        if source == "P4-8":
                            # Y5：③ のレースが比べられない。入力の同一性を確定できないなら P4-C（H1・H2）。
                            # レース単位の比較不能だけでは P4-C にしない。記録が無い・重複していて区別できないなら
                            # 保留して証拠不足を記録する。
                            r2x = recs2.get(eid)
                            if not isinstance(r2x, dict):
                                entry.update(case="investigation_evidence_insufficient", hold=True)
                                cmp["races"].append(entry)
                                insufficient.append({"eval_id": eid, "race_date": ds, "run": item["run"]})
                                continue
                            icase, ikeys = input_case(recs1.get(eid) or {}, r2x)
                            if icase == "C":
                                entry.update(case="investigation_input_undetermined", input_case="P4-C",
                                             input_keys=ikeys, H1=True, H2=True)
                                cmp["races"].append(entry)
                                doubts.append({"eval_id": eid, "race_date": ds, "source": source,
                                               "run": item["run"], "first_formal": sel.get("dir"),
                                               "case": "P4-C", "H1": True, "H2": True})
                                continue
                            entry["case"] = "investigation_race_not_comparable"
                            cmp["races"].append(entry)
                            continue
                        entry["case"] = "15b_not_comparable"  # 比べられない：外して事象。H1・H2 なし
                        cmp["races"].append(entry)
                        continue
                    r1, r2 = recs1.get(eid) or {}, recs2.get(eid) or {}
                    if not isinstance(r1, dict) or not isinstance(r2, dict):
                        r1, r2 = {}, {}
                    same = (s1 == s2 and isinstance(r1.get("comparable_hash"), dict)
                            and r1.get("comparable_hash") == r2.get("comparable_hash"))
                    if same:
                        continue
                    case, keys = input_case(r1, r2)
                    if source == "P4-15":
                        h2 = case == "C" or (case == "B" and s2 == "mismatch")  # 15-2
                    else:
                        h2 = case in ("B", "C")  # P4 8・P4-A〜C（調査用の再実行）。A の H2 は非決定性の確認後だけ
                    entry.update(case="15c_mismatch" if source == "P4-15" else "investigation_difference",
                                 input_case=f"P4-{case}", input_keys=keys, H1=True, H2=h2)
                    cmp["races"].append(entry)
                    doubts.append({"eval_id": eid, "race_date": ds, "source": source, "run": item["run"],
                                   "first_formal": sel.get("dir"), "case": f"P4-{case}", "H1": True, "H2": h2})
    return {"run_classification": classification, "comparisons": comparisons, "doubts": doubts,
            "events": events, "evidence_insufficient": insufficient,
            "not_implemented": ["P4 10: defects found later in a formal record (needs a record format)",
                                "P4 10: judgment C 7(6) condition (needs a decision on how to detect it)",
                                "H1/H2 release by an approved correction / closing record (P4 11/12; format undecided)"]}


# ---- 照合（C-6 の 3） ----
def verify_run(run_dir: str) -> dict[str, Any]:
    res: dict[str, Any] = {"dir": os.path.basename(run_dir), "verified": False, "problems": []}
    try:
        with open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, ValueError) as exc:
        res["problems"].append(f"missing input: {type(exc).__name__}: {exc}")
        res["record_fact"] = "evidence_missing"
        return res
    res["meta"] = meta
    # P4 4（07f §6.4・§6.5 の 6）：artifact の保存の失敗・保持期限切れ・取得の失敗を、別々の事実として記録する
    status = meta.get("artifact_status", "downloaded") if isinstance(meta, dict) else None
    if status in ARTIFACT_STATUS_FACTS:
        res["record_fact"] = ARTIFACT_STATUS_FACTS[status]
        res["problems"].append(f"artifact not available: {status}")
        return res
    if status != "downloaded":
        res["record_fact"] = "evidence_missing"
        res["problems"].append(f"unknown artifact_status: {status!r}")
        return res
    try:
        with open(os.path.join(run_dir, "artifact.zip"), "rb") as f:
            blob = f.read()
    except OSError as exc:
        res["problems"].append(f"missing input: {exc}")
        res["record_fact"] = "evidence_missing"
        return res
    ev_path = os.path.join(run_dir, "permission_evidence.json")
    if os.path.isfile(ev_path):
        try:
            with open(ev_path, "rb") as f:
                res["permission_evidence"] = json.loads(f.read(), object_pairs_hook=_no_duplicate_keys)
        except (OSError, ValueError) as exc:
            res["permission_evidence"] = {"fetch_ok": False, "fetch_errors": [f"unreadable: {type(exc).__name__}"]}
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
        res["record_fact"] = "evidence_missing"
        return res
    if manifest.get("complete") is not True:
        res["problems"].append("manifest not complete")
        res["record_fact"] = "evidence_missing"
    listed = manifest.get("files") or {}
    for name, info in listed.items():
        if name not in names:
            res["problems"].append(f"listed file missing: {name}")
            res["record_fact"] = "evidence_missing"
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
# ---- 正式な実行ごとの開始許可（項目4。2026-11-08 に本人が方式を選び、GPT が形式を決めた） ----
# 本人が GitHub の画面（ブラウザ）で g2_permissions/<permission_id>.json を commit する。Claude は作らない。
# GitHub の web-flow の署名は、GitHub が commit を作ったことを示すだけで、本人のブラウザ操作の暗号学的な
# 証明ではない（API で作った commit も同じ属性になりうる）。この残余リスクは、運用の取り決め（Claude は
# 許可ファイルを作成・変更・削除しない）を含む方式として扱う。U1U2 §1-3 を満たすとの判断は本人の選択による。
PERMISSION_DIR = "g2_permissions"
PERMISSION_SCHEMA_VERSION = 1
PERMISSION_APPROVER = "sinrai74"
PERMISSION_COMMITTER = "web-flow"
PERMISSION_METHOD = "github_web_ui"
PERMISSION_WORKFLOW = ".github/workflows/formal_g2_evaluate.yml"
PERMISSION_KEYS = ("schema_version", "permission_id", "race_date", "window_start", "window_end", "workflow",
                   "ref", "mode", "approved_on", "approved_by", "approval_method")
PERMISSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_UTC_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_RACE_DATE_RE = re.compile(r"^\d{8}$")
PERMISSION_HISTORY_LIMIT = 100  # API の per_page。これと同じ件数なら、履歴が切れている可能性があるので不合格
# U4（GPT 決定 2026-11-09）：commit の取得 API が返す変更ファイルは最大 300。300 件ちょうどは「300 を超えて切れた」場合と
# 区別できないので、300 未満であることを確かめる（fail-closed）。ブラウザで許可ファイルを作る commit は1ファイル。
PERMISSION_COMMIT_FILES_LIMIT = 300


def _parse_utc_z(v: Any) -> Optional[datetime]:
    """UTC の ISO 8601 Z 表記（秒まで）だけを受け付ける。"""
    if not isinstance(v, str) or not _UTC_Z_RE.match(v):
        return None
    try:
        return datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_api_time(v: Any) -> Optional[datetime]:
    """API の時刻（タイムゾーン付きの ISO 8601）を、タイムゾーンを解釈して読む。タイムゾーンが無ければ None。"""
    if not isinstance(v, str) or not v:
        return None
    try:
        t = datetime.fromisoformat(v[:-1] + "+00:00" if v.endswith("Z") else v)
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def permission_file_problems(doc: Any, permission_id: str) -> list[str]:
    """許可ファイルの形式（schema_version 1）。"""
    if not isinstance(doc, dict):
        return ["permission file is not a JSON object"]
    probs = []
    missing = [k for k in PERMISSION_KEYS if k not in doc]
    extra = sorted(k for k in doc if k not in PERMISSION_KEYS)
    if missing:
        probs.append(f"missing keys {missing}")
    if extra:
        probs.append(f"unexpected keys {extra}")
    sv = doc.get("schema_version")
    if isinstance(sv, bool) or sv != PERMISSION_SCHEMA_VERSION:
        probs.append(f"schema_version must be {PERMISSION_SCHEMA_VERSION}")
    if doc.get("permission_id") != permission_id:
        probs.append("permission_id differs from the file name")
    rd = doc.get("race_date")
    if not (isinstance(rd, str) and _RACE_DATE_RE.match(rd)):
        probs.append("race_date is not YYYYMMDD")
    else:
        try:
            datetime.strptime(rd, "%Y%m%d")
        except ValueError:
            probs.append("race_date is not a valid date")
    ws, we = _parse_utc_z(doc.get("window_start")), _parse_utc_z(doc.get("window_end"))
    if ws is None or we is None:
        probs.append("window_start/window_end must be UTC ISO 8601 with Z (YYYY-MM-DDTHH:MM:SSZ)")
    elif not ws < we:
        probs.append("window_start must be before window_end")
    if doc.get("workflow") != PERMISSION_WORKFLOW:
        probs.append(f"workflow must be {PERMISSION_WORKFLOW}")
    if doc.get("ref") != FORMAL_REF:
        probs.append(f"ref must be {FORMAL_REF}")
    if doc.get("mode") != "formal":
        probs.append("mode must be formal")
    ao = doc.get("approved_on")
    if not (isinstance(ao, str) and _DATE_RE.match(ao)):
        probs.append("approved_on is not YYYY-MM-DD")
    else:
        try:
            datetime.strptime(ao, "%Y-%m-%d")
        except ValueError:
            probs.append("approved_on is not a valid date")
    if doc.get("approved_by") != PERMISSION_APPROVER:
        probs.append(f"approved_by must be {PERMISSION_APPROVER}")
    if doc.get("approval_method") != PERMISSION_METHOD:
        probs.append(f"approval_method must be {PERMISSION_METHOD}")
    return probs


def permission_check(repo_root: str, vr: dict[str, Any]) -> dict[str, Any]:
    """正式な実行（mode formal）の許可の照合。status が "ok" のときだけ、その実行を正式な実行として扱う。

    照合の条件（GPT 決定、2026-11-08 の 1〜12）：
    1 許可ファイルが g2_permissions/<permission_id>.json に通常のファイルとしてある
    2 ファイル名と permission_id が一致する　3 JSON・必須項目・型・値（重複キーは拒否）
    4 実行の HEAD・集約の HEAD のどちらまでの履歴でも、そのファイルに触れた commit がちょうど1件で、追加（added）
    5 追加の後に変更・削除・再追加が無い（4 の両方の履歴が同じ1件）
    6 author が sinrai74　7 committer が web-flow　8 署名の verification.verified が true
    9 commit の時刻が実行の開始より前　10 許可の commit が実行の HEAD に含まれる（compare が ahead か identical）
    11 今のファイルの中身（git の blob）が、許可の commit のときの blob と同じ
    12 race_date・時間の窓・workflow・ref・mode・permission_id が実行の情報と一致する
    API の失敗・履歴の不足・不一致・許可の不在は、どれも許可されていないとする（fail-closed）。
    """
    info = (vr.get("manifest") or {}).get("run") or {}
    meta = vr.get("meta") or {}
    pid = info.get("permission_id")
    out: dict[str, Any] = {"status": "not_permitted", "permission_id": pid, "path": None, "blob_sha": None,
                           "commit_sha": None, "author": None, "committer": None, "verified": None,
                           "verification_reason": None, "commit_time": None,
                           "run_started_at": meta.get("run_started_at"), "conditions": {}, "problems": []}
    cond = out["conditions"]

    def done() -> dict[str, Any]:
        failed = [k for k, v in cond.items() if v is not True]
        out["status"] = "ok" if len(cond) == 12 and not failed and not out["problems"] else "not_permitted"
        out["permission"] = ({k: doc[k] for k in ("permission_id", "race_date", "window_start", "window_end")}
                             if out["status"] == "ok" else None)
        return out

    doc: Any = None
    if not (isinstance(pid, str) and PERMISSION_ID_RE.match(pid) and pid not in (".", "..")):
        out["problems"].append("permission_id is missing or has invalid characters")
        return done()
    rel = f"{PERMISSION_DIR}/{pid}.json"
    out["path"] = rel
    p = os.path.join(repo_root, PERMISSION_DIR, f"{pid}.json")
    cond["1_file_exists"] = (os.path.isfile(p) and not os.path.islink(p) and _repo_relpath(repo_root, p) == rel)
    data = b""
    if cond["1_file_exists"]:
        try:
            with open(p, "rb") as f:
                data = f.read()
            doc = json.loads(data, object_pairs_hook=_no_duplicate_keys)
        except (OSError, ValueError) as exc:
            out["problems"].append(f"permission file unreadable: {type(exc).__name__}: {exc}")
        out["blob_sha"] = _git_blob_sha(data) if data else None
    cond["2_name_matches_id"] = isinstance(doc, dict) and doc.get("permission_id") == pid
    fprobs = permission_file_problems(doc, pid) if doc is not None else ["permission file not read"]
    cond["3_format_ok"] = not fprobs
    out["problems"] += fprobs if doc is not None else []

    ev = vr.get("permission_evidence")
    if not isinstance(ev, dict) or ev.get("fetch_ok") is not True or ev.get("permission_id") != pid \
            or ev.get("path") != rel:
        out["problems"].append("permission history not fetched (or fetch failed / for another file)")
        for k in ("4_added_once", "5_not_changed_after", "6_author", "7_committer", "8_signature_verified",
                  "9_committed_before_run", "10_in_run_head", "11_blob_matches", "12_matches_run"):
            cond[k] = False
        return done()
    hr, hm = ev.get("history_run_head"), ev.get("history_aggregate_head")
    commits = {c.get("sha"): c for c in (ev.get("commits") or []) if isinstance(c, dict)}
    ok_lists = isinstance(hr, list) and isinstance(hm, list)
    truncated = ok_lists and (len(hr) >= PERMISSION_HISTORY_LIMIT or len(hm) >= PERMISSION_HISTORY_LIMIT)
    if truncated:
        out["problems"].append("permission history may be truncated")
    sha = hr[0] if ok_lists and len(hr) == 1 else None
    c = commits.get(sha) if sha else None
    files = [f for f in ((c or {}).get("files") or [])
             if isinstance(f, dict) and (f.get("filename") == rel or f.get("previous_filename") == rel)]
    fc = (c or {}).get("files_count")
    files_ok = isinstance(fc, int) and not isinstance(fc, bool) and 1 <= fc < PERMISSION_COMMIT_FILES_LIMIT
    if c is not None and not files_ok:
        out["problems"].append(f"commit files_count {fc!r} is not below {PERMISSION_COMMIT_FILES_LIMIT} "
                               "(the file list may be truncated)")
    # U4：改名・削除・再追加が疑われるもの（previous_filename がある、追加でない）は不合格
    cond["4_added_once"] = (sha is not None and not truncated and c is not None and files_ok and len(files) == 1
                            and files[0].get("status") == "added" and files[0].get("filename") == rel
                            and not files[0].get("previous_filename"))
    # 集約の HEAD が main であること（別の枝から集約すると、main での変更・削除が見えない。監査の指摘2）
    agg_ref_ok = ev.get("aggregate_ref") == FORMAL_REF
    if not agg_ref_ok:
        out["problems"].append(f"aggregate ref is not {FORMAL_REF}: {ev.get('aggregate_ref')!r}")
    cond["5_not_changed_after"] = cond["4_added_once"] and hm == [sha] and agg_ref_ok
    out["commit_sha"] = sha
    out["author"] = (c or {}).get("author_login")
    out["committer"] = (c or {}).get("committer_login")
    out["verified"] = (c or {}).get("verified")
    out["verification_reason"] = (c or {}).get("verification_reason")
    out["commit_time"] = (c or {}).get("committer_date")
    cond["6_author"] = out["author"] == PERMISSION_APPROVER
    cond["7_committer"] = out["committer"] == PERMISSION_COMMITTER
    cond["8_signature_verified"] = out["verified"] is True
    ct, started = _parse_api_time(out["commit_time"]), _parse_api_time(meta.get("run_started_at"))
    cond["9_committed_before_run"] = ct is not None and started is not None and ct < started
    out["history_limits"] = ev.get("history_limits")  # 取得範囲の限界（証拠に記録する。U4）
    cmp = ev.get("compare") or {}
    cond["10_in_run_head"] = (sha is not None and cmp.get("base") == sha
                              and cmp.get("head") == meta.get("api_head_sha") and bool(cmp.get("head"))
                              and cmp.get("status") in ("ahead", "identical"))
    cond["11_blob_matches"] = (bool(out["blob_sha"]) and len(files) == 1
                               and files[0].get("sha") == out["blob_sha"])
    m12 = isinstance(doc, dict) and cond["3_format_ok"]
    if m12:
        ws, we = _parse_utc_z(doc["window_start"]), _parse_utc_z(doc["window_end"])
        m12 = (doc["race_date"] == info.get("race_date") and started is not None and ws <= started <= we
               and doc["workflow"] == meta.get("api_workflow_path") and doc["ref"] == info.get("github_ref")
               and doc["mode"] == info.get("mode") and doc["permission_id"] == pid)
    cond["12_matches_run"] = bool(m12)
    return done()


def is_formal(vr: dict[str, Any], permissions: list[dict[str, Any]]) -> bool:
    """許可（main() では permission_check に合格したものだけ）と、permission_id・race_date・時間の窓で対応する
    mode formal の実行。時刻はタイムゾーンを解釈して比べる（文字列では比べない）。"""
    info = (vr.get("manifest") or {}).get("run") or {}
    if info.get("mode") != "formal":
        return False
    started = _parse_api_time((vr.get("meta") or {}).get("run_started_at"))
    for p in permissions:
        if p.get("permission_id") != info.get("permission_id") or p.get("race_date") != info.get("race_date"):
            continue
        if "run" in p and p["run"] != vr.get("dir"):
            continue  # 別の実行が照合に合格した許可（同じ permission_id でも、この実行は許可されていない）
        ws, we = _parse_api_time(p.get("window_start")), _parse_api_time(p.get("window_end"))
        if started is not None and ws is not None and we is not None and ws <= started <= we:
            return True
    return False


# ---- race_closed_at の検証（07f §6.4・§6.5 の 8。GPT判断 11/04）----
# 評価側の処理の順序は変えない。集約側で、固定した対象一覧（targets）の順について次を確かめる。
#   race_closed_at が日時として読めること／日時の昇順／同じ時刻なら会場番号、R 番号の順
# 合わない・確かめられない場合は、そのレースだけでなく実行全体を正式カウントから外す。
# 条件5 (iii)（文字列として並べ直した一覧の照合）とは別の確認として、両方を行う。
# 形式：番組表 API の実データ（2026-11-01 の programs_raw.json、168件）は "YYYY-MM-DD HH:MM:SS" だけで、
# タイムゾーンを含まない。この形だけを日時として読む（ほかの形は「確かめられない」として外す）。
RACE_CLOSED_AT_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$")  # \\d は全角数字も通すので使わない


def race_closed_at_check(targets: Any) -> list[str]:
    """問題の一覧を返す（空なら合格）。"""
    if not isinstance(targets, list) or not targets:
        return ["targets missing or empty"]
    probs: list[str] = []
    keys = []
    for i, t in enumerate(targets):
        if not isinstance(t, dict):
            probs.append(f"target[{i}] is not an object")
            continue
        ca, vn, rn = t.get("race_closed_at"), t.get("venue_num"), t.get("race_number")
        dt = None
        if isinstance(ca, str) and RACE_CLOSED_AT_RE.match(ca):
            try:
                dt = datetime.strptime(ca, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                dt = None
        if dt is None:
            # V3：推測で変換・補正しない。形が違えば API の形の変更の疑いとして原因を記録する
            probs.append(f"target[{i}] {t.get('eval_id')}: race_closed_at not readable as a datetime "
                         f"(expected 'YYYY-MM-DD HH:MM:SS'; possible API format change): {ca!r}")
        if not (isinstance(vn, int) and not isinstance(vn, bool) and isinstance(rn, int) and not isinstance(rn, bool)):
            probs.append(f"target[{i}] {t.get('eval_id')}: venue_num/race_number not integers: {vn!r}/{rn!r}")
            continue
        if dt is not None:
            keys.append((i, t.get("eval_id"), (dt, vn, rn)))
    if probs:
        return probs[:20]
    for (i0, e0, k0), (i1, e1, k1) in zip(keys, keys[1:]):
        if not k0 < k1:  # 同じ (時刻, 会場, R) の重複も順序を確かめられないとする
            probs.append(f"target[{i1}] {e1} is not after target[{i0}] {e0} "
                         "(race_closed_at, then venue_num, then race_number)")
    return probs[:20]


def _ordered_records(sel: dict[str, Any], targets: list[Any]) -> list[dict[str, Any]]:
    """対象一覧の順のレースの記録（同じ eval_id が複数あれば、記録の順にすべて）。"""
    by: dict[Any, list] = {}
    for r in sel.get("races") or []:
        if isinstance(r, dict):
            by.setdefault(r.get("eval_id"), []).append(r)
    return [r for eid in targets for r in by.get(eid, [])]


def select_formal(runs: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """P4 14 の ①：許可に対応する正式な実行のうち、開始がいちばん早いもの（run_attempt=1 であることだけでは
    決めない。開始時刻は日時として読んで比べる）。置き換えない。開始の順が決まらない場合は、aggregate() が
    その日を「実行の順が不明」として区切る（ここでの並びは、記録に残す実行を決めるためだけに使う）。"""
    if not runs:
        return None

    def key(r: dict[str, Any]) -> tuple[Any, ...]:
        m = r.get("meta") or {}
        t = _parse_api_time(m.get("run_started_at"))
        rid, att = str(m.get("run_id") or ""), str(m.get("run_attempt") or "")
        return (t is None, t or datetime.min.replace(tzinfo=timezone.utc),
                int(rid) if rid.isdigit() else 0, int(att) if att.isdigit() else 0)
    return sorted(runs, key=key)[0]


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


# ---- 判断C 7(2-b)・7(2-c)（07f §6.8 A。GPT判断 11/04、判断C は 07g で GPT が最終承認） ----
HASH_POINTS = ("before", "after_rebuild", "after_v4")
STOP_2B_STATE = "count_stopped_2b"  # 7(2-b) ①〜③：そのレースと、その実行で後に続く対象レースを正式カウントから外す


def race_2b_reasons(rec: dict[str, Any]) -> list[str]:
    """7(2-b) ①〜③（レース単位）。対象はこの3つに限る（d13_2 全体は使わない。7(3) の項目は混ぜない）。
    記録に値があり、条件に当たるときだけ当たるとする（値が無いものは、ほかの規則で 0 になる）。"""
    out: list[str] = []
    if not isinstance(rec, dict):
        return out
    u = rec.get("unstable_hash_keys")
    if isinstance(u, list) and u:
        out.append("1_unstable_hash_keys")
    d13 = rec.get("d13_2")
    if isinstance(d13, dict) and d13.get("venue_caches_match_preload") is False:
        out.append("2_venue_caches_match_preload_false")
    if rec.get("b_c_match") is False:
        out.append("3_b_c_match_false")
    return out


_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
# V1（GPT 決定 2026-11-09）：評価側が比較の前に正当に閉じた記録（race_eval.evaluate_race の fail() と
# config_hit() の後の close()）だけを、3時点の fan の証拠が無くてもレース単位の 0 とする。閉じたことを
# 記録の形で確かめられないもの（例：not_established とだけ書いて hashes が無い）は U5 の証拠不完全とする。
CLOSE_FAIL_STAGES = ("resolve", "rebuild_evaluate", "legacy_v4", "compare")
CLOSE_CONFIG_STEPS = ("after_resolve", "after_rebuild_evaluate", "after_legacy_v4", "after_compare")
_COMPARE_ONLY_KEYS = ("hashes", "values", "comparable_hash", "diffs", "meta_check")


def closed_before_compare(rec: Any) -> bool:
    """評価側の close() の形（race_eval.py）と一致するときだけ True。読み替えはしない。"""
    if not isinstance(rec, dict) or any(k in rec for k in _COMPARE_ONLY_KEYS):
        return False
    if rec.get("d9_states") != ["compare_not_established"] or rec.get("input_identity_ok") is not False:
        return False
    cat, stage, exc, cfg = rec.get("result_category"), rec.get("stage"), rec.get("exception"), rec.get("config_detected")
    if isinstance(exc, dict):  # fail(stage, exc)
        return (stage in CLOSE_FAIL_STAGES and exc.get("stage") == stage
                and isinstance(exc.get("class"), str) and isinstance(exc.get("message"), str)
                and cat == ("compare_exception" if stage == "compare" else "not_established") and cfg is None)
    if exc is None and stage is None and isinstance(cfg, dict):  # config_hit(step) の後の close()
        return cfg.get("at_step") in CLOSE_CONFIG_STEPS and cat == "not_established"
    return False


def fan_run_check(ordered_recs: list[dict[str, Any]]) -> dict[str, Any]:
    """7(2-b) ④ と 7(2-c)（実行単位）。判定には記録された f_fan_file と f_fan_cache のハッシュだけを使う。

    条件に当たったもの（reasons）と、判定に必要な証拠が不完全なもの（incomplete。U5）は別の理由として返す。
    - ④・(2-c) ③（U8：原文どおりレースの間の変化に限る）：前のレースの最後の時点（after_v4）と次のレースの
      最初の時点（before）で f_fan_file が違う。1つのレースの中だけの変化は当てはめない（観測として記録する）
    - (2-c) ①：記録された f_fan_file に "" がある
    - (2-c) ②：f_fan_cache が、{f_fan_file: {}}（1ファイルだけで空のキャッシュ）を同じ stable_hash の定義で
      計算し直した値と一致する
    - U9：キャッシュが0ファイル（{}）であることだけでは (2-c) に当てはめない。f_fan_file が空なら (2-c) ①、
      空でなければ読み込みの失敗の結果として証拠不完全（U5 と同じ扱い）にする
    - U5：比較まで進んだ記録（hashes がある、または result_category が diff_zero・mismatch）で、3時点の
      どれかが無い、f_fan_file が無い・文字列でない、f_fan_cache が SHA-256 の形でない（UNSERIALIZABLE を含む）
    V1：比較の前に正当に閉じた記録（closed_before_compare）は、評価側が3時点を記録しない仕様なので対象に
    しない（レース単位の比較不能として 0）。それ以外の記録は、どれも3時点がそろっていることを求める。
    保証しないこと（07f）：抽出に実際に使ったファイル、全艇分の fan 情報、欠けが無いこと。当たらないことを、
    fan の選択・読み込みの一貫や fan 情報の完全さの証明として扱わない。
    """
    races: list[tuple[Any, list[dict[str, Any]]]] = []
    incomplete: list[dict[str, Any]] = []
    for rec in ordered_recs:
        if not isinstance(rec, dict):
            continue
        if closed_before_compare(rec):
            continue  # V1：比較の前に正当に閉じた記録（そのレースだけ 0）
        eid = rec.get("eval_id")
        hashes = rec.get("hashes")
        if not isinstance(hashes, dict):
            incomplete.append({"eval_id": eid, "point": None, "problem": "hashes missing or not an object"})
            continue
        snaps = []
        for pt in HASH_POINTS:
            h = hashes.get(pt)
            if not isinstance(h, dict):
                incomplete.append({"eval_id": eid, "point": pt, "problem": "snapshot missing"})
                continue
            ff, fc = h.get("f_fan_file"), h.get("f_fan_cache")
            if not isinstance(ff, str):
                incomplete.append({"eval_id": eid, "point": pt, "problem": "f_fan_file missing or not a string"})
                continue
            if not (isinstance(fc, str) and _HEX64_RE.match(fc)):
                incomplete.append({"eval_id": eid, "point": pt,
                                   "problem": "f_fan_cache is not a sha256 (unserializable or missing)"})
                continue
            if ff and fc == _stable_hash({}):
                # U9：fan ファイル名があるのにキャッシュが0ファイル。読み込みに成功すれば {ファイル名: …} が
                # キャッシュに入り（0件でも）、OSError のときだけ {} のまま（notify_arashi._load_fan_file）。
                # 評価前の時点の前に、出走表から艇を作る処理（_extract_boats_from_program）で読み込みが行われる
                # （監査 2026-11-09 の確認）。そのためこの形は読み込みの失敗の結果とみて、証拠不完全とする。
                incomplete.append({"eval_id": eid, "point": pt,
                                   "problem": "zero-file fan cache with a non-empty f_fan_file (read failure)"})
                continue
            snaps.append({"eval_id": eid, "point": pt, "f_fan_file": ff, "f_fan_cache": fc})
        races.append((eid, snaps))
    all_snaps = [x for _, sn in races for x in sn]
    reasons = []
    changes = []
    within = []
    prev_last = None
    for eid, sn in races:
        if sn and prev_last is not None and sn[0]["f_fan_file"] != prev_last["f_fan_file"]:
            changes.append({"from": prev_last["eval_id"], "to": eid,
                            "files": [prev_last["f_fan_file"], sn[0]["f_fan_file"]]})
        if len({x["f_fan_file"] for x in sn}) > 1:
            within.append(eid)
        if sn:
            prev_last = sn[-1]
    if changes:
        reasons.append("2b4_fan_file_changed")
    if any(x["f_fan_file"] == "" for x in all_snaps):
        reasons.append("2c1_fan_file_empty")
    empty_one = [x for x in all_snaps if x["f_fan_file"] and x["f_fan_cache"] == _stable_hash({x["f_fan_file"]: {}})]
    if empty_one:
        reasons.append("2c2_single_file_empty_cache")
    zero_files = [x for x in all_snaps if x["f_fan_cache"] == _stable_hash({})]
    return {"reasons": reasons, "incomplete": incomplete[:20], "incomplete_count": len(incomplete),
            "snapshots": len(all_snaps), "fan_files": sorted({x["f_fan_file"] for x in all_snaps}),
            "changes_between_races": changes[:10],
            "empty_cache_at": [{"eval_id": x["eval_id"], "point": x["point"]} for x in empty_one][:10],
            # 観測だけ（除外の条件にしない）：1つのレースの中だけの変化（U8）、0ファイルのキャッシュ（U9）
            "within_race_changes": within[:10],
            "zero_file_cache_at": [{"eval_id": x["eval_id"], "point": x["point"]} for x in zero_files][:10]}


def _order_vs_reach(meta: dict[str, Any], reach_meta: Optional[dict[str, Any]]) -> str:
    """違反した実行の開始が、到達した実行の開始の後か前か（U6）。決められなければ "unknown"。"""
    if reach_meta is None:
        return "unknown"
    t, rt = _parse_api_time(meta.get("run_started_at")), _parse_api_time(reach_meta.get("run_started_at"))
    if t is not None and rt is not None and t != rt:
        return "after" if t > rt else "before"
    rid, rrid = str(meta.get("run_id") or ""), str(reach_meta.get("run_id") or "")
    att, ratt = str(meta.get("run_attempt") or ""), str(reach_meta.get("run_attempt") or "")
    if rid and rid == rrid and att.isdigit() and ratt.isdigit() and att != ratt:
        return "after" if int(att) > int(ratt) else "before"  # 同じ run の再実行だけ
    return "unknown"


def fan_exclusion(fc: dict[str, Any]) -> Optional[str]:
    """条件に当たったもの（判断C 7）と証拠の不足（U5）を別の理由として返す。どちらも無ければ None。"""
    parts = []
    if fc["reasons"]:
        parts.append("judgment_c_7_fan:" + ",".join(fc["reasons"]))
    if fc["incomplete_count"]:
        parts.append("evidence_incomplete:judgment_c_7_fan")
    return ";".join(parts) or None


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
              formal_context: Optional[dict[str, Any]] = None,
              force_formal: bool = False,
              corrections: Optional[dict[str, str]] = None) -> dict[str, Any]:
    """force_formal（段階5）：与えた実行を、許可の照合なしに正式な実行として扱う。P4 15 の ②・③ の
    レースの状態を ① と同じ規則で求めるための内部の呼び出しだけに使う（加算・到達の判定には使わない）。

    基準が無い・評価側と同じ基準だと確かめられない・形が正しくない・ジョブログの
    Runner Image と一致しない実行は、その日の全レースを正式カウントから除外し、区間を
    終える（M7-3・基準の同一性。評価側の A-1 に相当）。これらは baseline_failures に入る。

    formal_context（2026-11-06）：集約側の基準の照合結果、E3 の想定の表示、登録の確認の結果。
    省略すると DEFAULT_FORMAL_CONTEXT（どれも満たさない）になり、正式な実行には合格を出さない。
    main() は registration_check（案A、2026-11-08）の結果が "ok" のときだけ registration_verified=True を渡す。
    """
    ctx = dict(DEFAULT_FORMAL_CONTEXT, **(formal_context or {}))
    baseline_failures: list[dict[str, Any]] = []
    input_order_failures: list[dict[str, Any]] = []  # V4：race_closed_at の検証の失敗
    race_info: dict[Any, dict[str, Any]] = {}  # eval_id → 到達記録に使う既存の値（race_date・race_closed_at・評価の実行）
    by_date: dict[str, list[dict[str, Any]]] = {}
    for vr in verified:
        if force_formal or is_formal(vr, permissions):
            d = ((vr.get("manifest") or {}).get("run") or {}).get("race_date")
            by_date.setdefault(d, []).append(vr)

    # P4 4：artifact を確かめられない実行（保存の失敗・保持期限切れ・取得の失敗・証拠の欠落）。その日の対象レース
    # は 0 とし、あとから別の実行で評価し直して連続数に入れない。日付が分かるもの（artifact の名前）は、その日を
    # 区切る（選ばれた実行より後に始まったと確かめられた場合を除く）。日付が分からないものは main() で保留にする。
    unavailable_by_date: dict[str, list[dict[str, Any]]] = {}
    unattributed: list[dict[str, Any]] = []
    for vr in verified:
        if vr.get("manifest") is None:
            dd = unavailable_run_date(vr)
            if dd is None:
                unattributed.append(vr)
            else:
                unavailable_by_date.setdefault(dd, []).append(vr)

    d0 = datetime.strptime(first_date, "%Y%m%d").date()
    d1 = datetime.strptime(last_date, "%Y%m%d").date()
    days = []
    seq: list[dict[str, Any]] = []
    cur = d0
    while cur <= d1:
        ds = cur.strftime("%Y%m%d")
        sel = select_formal(by_date.get(ds, []))
        day: dict[str, Any] = {"race_date": ds, "selected_run": None, "class": "pending_confirmation"}
        blockers = [b for b in unavailable_by_date.get(ds, [])
                    if sel is None or _order_vs_reach(b.get("meta") or {}, sel.get("meta")) != "after"]
        # 段階5（P4 14）・Y1：① が決まらない（開始の順が分からない正式な実行がある）日は、保留する
        order_unknown = ([r for r in by_date.get(ds, []) if r is not sel
                          and _order_vs_reach(r.get("meta") or {}, sel.get("meta")) != "after"]
                         if sel is not None and not force_formal else [])
        # Y3（GPT 決定 2026-11-09）：同じ race_date に許可が2つ以上ある日は、正常な正式集約の対象にせず保留する
        day_perms = sorted({str(((r.get("manifest") or {}).get("run") or {}).get("permission_id"))
                            for r in by_date.get(ds, [])}) if not force_formal else []
        # Y4（GPT 決定 2026-11-09）：未判定の実行（① より後でない正式の表示の実行、formal・investigation 以外の
        # モード、モードの無い実行）がある日は、① を数えず保留する。調査用と推測して外さない。
        undetermined = []
        if sel is not None and not force_formal:
            for r in verified:
                info_r = (r.get("manifest") or {}).get("run") or {}
                if r is sel or r.get("manifest") is None or info_r.get("race_date") != ds:
                    continue
                mode_r = info_r.get("mode")
                if mode_r == "investigation":
                    continue
                if mode_r == "formal" and _order_vs_reach(r.get("meta") or {}, sel.get("meta")) == "after":
                    continue  # ②
                undetermined.append(r.get("dir"))
        if blockers:
            day["selected_run"] = sel["meta"] if sel is not None else None
            day["excluded"] = "artifact_unavailable"
            day["unavailable_runs"] = [{"run": b.get("dir"), "fact": b.get("record_fact")} for b in blockers]
            seq.append({"break": "artifact_unavailable", "race_date": ds})
        elif order_unknown or len(day_perms) > 1 or undetermined:
            day["selected_run"] = None
            day["first_formal_candidate"] = sel["meta"]  # 記録のため（加算・比較の正本には使わない）
            day["pending_reason"] = ("formal_order_unknown" if order_unknown
                                     else "multiple_permissions_for_race_date" if len(day_perms) > 1
                                     else "undetermined_run_on_day")
            if order_unknown:
                day["order_unknown_runs"] = [r.get("dir") for r in [sel] + order_unknown]
            if len(day_perms) > 1:
                day["permissions"] = day_perms
            if undetermined:
                day["undetermined_runs"] = undetermined
            seq.append({"break": "pending_confirmation_day", "reason": day["pending_reason"], "race_date": ds})
        elif sel is not None:
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
                if run.get("status") == "completed":
                    day["race_closed_at_check"] = (race_closed_at_check((run.get("programs") or {}).get("targets"))
                                                   or "ok")
                else:
                    day["race_closed_at_check"] = "not_applicable_stopped_run"
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
                elif day["race_closed_at_check"] not in passing:
                    failed_check = "race_closed_at"  # 07f §6.4・§6.5 の 8：実行全体を外す
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
                if failed_check == "race_closed_at":
                    # V4：実行全体を外す。baseline の失敗（baseline_failures）とは分けて記録する
                    day["class"] = "normal" if targets else day["class"]
                    day["excluded"] = failed_check
                    seq.append({"break": failed_check, "race_date": ds})
                    input_order_failures.append({"race_date": ds, "run": sel["meta"], "check": failed_check,
                                                 "problems": day["race_closed_at_check"]})
                elif failed_check is not None:
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
                elif targets and fan_exclusion(fan_run_check(_ordered_records(sel, targets))):
                    # 判断C 7(2-b) ④・7(2-c)、または判定に必要な証拠の不足（U5）：その実行全体を正式カウントから外す
                    # （評価処理は止まったと扱わない）。7(5)：(3)(4) にも当たっても、こちらを優先する。
                    fc = fan_run_check(_ordered_records(sel, targets))
                    day["class"] = "normal"
                    day["fan_check"] = fc
                    day["excluded"] = fan_exclusion(fc)
                    seq.append({"break": day["excluded"], "race_date": ds})
                elif targets:
                    day["class"] = "normal"
                    day["fan_check"] = fan_run_check(_ordered_records(sel, targets))
                    ok = run_level_ok(run, len(sel.get("races") or []))
                    recs: dict[str, list] = {}
                    for r in sel.get("races", []):
                        recs.setdefault(r.get("eval_id"), []).append(r)
                    stopped_2b: Optional[dict[str, Any]] = None
                    closed_at = {t.get("eval_id"): t.get("race_closed_at")
                                 for t in (run.get("programs") or {}).get("targets", [])}
                    for eid in targets:
                        race_info.setdefault(eid, {
                            "race_date": ds, "race_closed_at": closed_at.get(eid),
                            "evaluation_run_id": (sel.get("meta") or {}).get("run_id"),
                            "evaluation_run_attempt": (sel.get("meta") or {}).get("run_attempt")})
                        rl = recs.get(eid, [])
                        if len(rl) == 1:
                            st = race_state(rl[0], ok)
                        elif len(rl) > 1:
                            st = "duplicate_eval_id"
                        else:
                            st = "unprocessed"
                        # 同じ eval_id の記録が複数あっても、どれかが ①〜③ に当たれば止める（監査 2026-11-09 の指摘1）
                        r2b = sorted({x for r in rl for x in race_2b_reasons(r)})
                        if stopped_2b is None and r2b:
                            stopped_2b = {"eval_id": eid, "reasons": r2b}
                            day["count_stopped_2b"] = stopped_2b
                        if stopped_2b is not None:
                            # 7(2-b) ①〜③：このレースと後に続く対象レースは 0。7(5)：(3)(4) より優先する。
                            # 評価が続いた記録はそのまま残す（元の状態を original_state に残す）
                            seq.append({"eval_id": eid, "state": STOP_2B_STATE, "original_state": st,
                                        "stopped_by": stopped_2b["eval_id"]})
                            continue
                        if st == "added" and (corrections or {}).get(str(eid)) == "zero":
                            # P4 12・13：承認された訂正記録（0）を当てて計算し直す。元の記録は書き換えない
                            seq.append({"eval_id": eid, "state": "corrected_zero", "original_state": st})
                            continue
                        seq.append({"eval_id": eid, "state": st})
        if day["class"] != "normal" and not day.get("verification_failed") and not day.get("excluded"):
            seq.append({"break": "pending_confirmation_day", "race_date": ds})
        days.append(day)
        cur += timedelta(days=1)

    # §6.3 の5・6（GPT判断：確定）：ある連続区間で100件目の加算が成立した時点を「100件到達」とし、
    # その時点で到達の判定を確定する。101件目以降は数えず、到達を取り消す材料にも使わない。
    # 到達のあとは新しい区間を作らない（GPT 指示、2026-11-08）。到達と正式な達成（d5）は別。
    segments = []
    count = 0
    start = None
    reach: Optional[dict[str, Any]] = None
    reach_index: Optional[int] = None
    for i, item in enumerate(seq):
        if item.get("state") == "added":
            if count == 0:
                start = item["eval_id"]
            count += 1
            if count == TARGET:
                hundredth = item["eval_id"]
                segments.append({"start": start, "count": TARGET, "ended_by": "reached_100",
                                 "hundredth": hundredth})
                info = race_info.get(hundredth) or {}
                reach = {"event_type": REACH_EVENT_TYPE, "race_date": info.get("race_date"),
                         "segment_index": len(segments) - 1, "streak_count": TARGET,
                         "first_race_key": start, "hundredth_race_key": hundredth,
                         "hundredth_race_closed_at": info.get("race_closed_at"),
                         "evaluation_run_id": info.get("evaluation_run_id"),
                         "evaluation_run_attempt": info.get("evaluation_run_attempt"),
                         "aggregate_run_id": None, "status": REACH_STATUS}
                reach_index = i
                count, start = 0, None
                break
            continue
        if count:
            segments.append({"start": start, "count": count, "ended_by": item.get("state") or item.get("break")})
        count, start = 0, None
    if count:
        segments.append({"start": start, "count": count, "ended_by": None})

    after = seq[reach_index + 1:] if reach_index is not None else []
    best = TARGET if reach is not None else max((s["count"] for s in segments), default=0)
    status = "reached_100_unconfirmed" if reach is not None else "undetermined"
    return {"days": days, "sequence": seq, "segments": segments, "best_segment": best, "g2_status": status,
            "baseline_failures": baseline_failures, "input_order_failures": input_order_failures, "reach": reach,
            "unattributed_unavailable_runs": [vr.get("dir") for vr in unattributed],
            # 101件目以降（到達の後の項目）は証拠として sequence に残し、数には加えない
            "after_reach": None if reach is None else {
                "items": len(after), "added_not_counted": sum(1 for x in after if x.get("state") == "added")}}


def write_canonical_bundle(args: Any, verified: list[dict[str, Any]], result: dict[str, Any], result_bytes: bytes,
                           prev_state: dict[str, Any], all_h1: list[dict[str, Any]], all_h2: list[dict[str, Any]],
                           active1: list[str], active2: list[str], baseline_sha256: Optional[str],
                           agg_base: dict[str, Any], agg_run: str) -> None:
    """2.1・2.2：評価の実行ごとの正本、集約の正本、状態スナップショット、H1/H2 のイベント（追記後の全体）。"""
    from formal_g2.aggregate import canonical
    root = os.path.join(args.out, "canonical_bundle")
    files: dict[str, bytes] = {}
    cls = {}
    for c in (result.get("stage5") or {}).get("run_classification") or []:
        if c.get("first_formal"):
            cls[c["first_formal"]] = "first_formal"
        for x in c.get("second_formal") or []:
            cls[x["run"]] = "second_formal"
        for x in c.get("investigation") or []:
            cls[x["run"]] = "investigation"
        for x in c.get("undetermined") or []:
            cls[x["run"]] = "undetermined"
    run_dirs, run_records = [], []
    for vr in verified:
        if vr.get("manifest") is None:
            continue
        rd = canonical.run_dir_for((vr.get("manifest") or {}).get("run") or {}, vr.get("meta") or {})
        # 再監査 M-2：照合に合格した実行だけ、まだ正本の無いときだけ、実行の正本を束に入れる。
        # 既にある正本は書き直さない（集約ごとの違いは input_manifest に記録し、到達の確認で正本と照合する）
        exists = os.path.lexists(os.path.join(_repo_root(), *rd.split("/")))
        status = ("included" if vr.get("verified") and not exists
                  else "already_saved" if exists else "not_included_unverified")
        if status == "included":
            files.update(canonical.build_run_files(vr, rd, os.path.join(args.artifacts_dir, vr["dir"])))
        run_dirs.append(rd)
        m = vr.get("meta") or {}
        # 集約ごとに変わりうる値は、集約の正本に記録する（2.1 の実行区分・照合の結果・基準・取得の結果と時刻）
        run_records.append({"run": vr.get("dir"), "dir": rd, "canonical_in_bundle": status,
                            "classification": cls.get(vr.get("dir"), "undetermined"),
                            "verified": bool(vr.get("verified")), "problems": (vr.get("problems") or [])[:50],
                            "artifact_sha256": vr.get("artifact_sha256"),
                            "baseline": {"path": args.baseline or None, "sha256": baseline_sha256},
                            "fetch": {"log_fetch_ok": m.get("log_fetch_ok"), "log_fetch_error": m.get("log_fetch_error"),
                                      "log_fetched_at": m.get("log_fetched_at"), "latest_attempt": m.get("latest_attempt"),
                                      "artifact_expires_at": m.get("artifact_expires_at")}})
    adir = f"{canonical.AGGREGATIONS_DIR}/{agg_run}"
    snap_rel = f"{canonical.SNAPSHOT_DIR}/{agg_run}.json"
    snap = {"schema_version": canonical.SCHEMA_VERSION, "kind": "g2_state_snapshot", "state_version_id": agg_run,
            "aggregate_run": agg_run, "previous_state_version_id": prev_state.get("state_version_id"),
            "previous_state": {"path": prev_state.get("path"), "sha256": prev_state.get("expected_sha256"),
                               "status": prev_state.get("status")},
            "h1_active": active1, "h2_active": active2, "reach": result.get("reach"),
            "best_segment": result.get("best_segment"), "aggregate_result_sha256": canonical.sha256(result_bytes)}
    inputs = {"schema_version": canonical.SCHEMA_VERSION, "kind": "g2_aggregation_inputs",
              "runs": (result.get("input_identity") or {}).get("runs"),
              "first_date": args.first_date, "last_date": args.last_date,
              "evaluate_baseline": {"path": args.baseline or None, "sha256": baseline_sha256},
              "aggregate_baseline": {"path": args.aggregate_baseline or None, "sha256": agg_base.get("sha256")},
              "previous_state": snap["previous_state"], "evaluation_runs": run_records}
    files[f"{adir}/aggregate_result.json"] = result_bytes
    files[f"{adir}/input_manifest.json"] = canonical._dumps(inputs)
    files[snap_rel] = canonical._dumps(snap)
    ev1 = (prev_state.get("h1_events_bytes") or b"") + b"".join(canonical._line(e) for e in all_h1[len(prev_state.get("h1_events") or []):])
    ev2 = (prev_state.get("h2_events_bytes") or b"") + b"".join(canonical._line(e) for e in all_h2[len(prev_state.get("h2_events") or []):])
    prefixes = []
    log0 = prev_state.get("aggregation_log_bytes") or b""
    for rel, data in ((canonical.H1_EVENTS, ev1), (canonical.H2_EVENTS, ev2), (canonical.AGGREGATION_LOG, log0)):
        files[rel] = data
        prefixes.append({"path": rel, "lines": len(data.splitlines()), "sha256": canonical.sha256(data)})
    listed = [f"{adir}/aggregate_result.json", f"{adir}/input_manifest.json", snap_rel]
    files[f"{adir}/manifest.json"] = canonical._dumps({
        "schema_version": canonical.SCHEMA_VERSION, "kind": "g2_aggregation_manifest", "aggregate_run": agg_run,
        "commit_sha": os.environ.get("GITHUB_SHA", "local"), "generated_at": result.get("generated_at"),
        "aggregate_result": f"{adir}/aggregate_result.json", "snapshot": snap_rel,
        "evaluation_run_dirs": run_dirs, "inputs": f"{adir}/input_manifest.json",
        "result_sha256": canonical.sha256(result_bytes), "reach": result.get("reach"),
        "best_segment": result.get("best_segment"), "h1_active": active1, "h2_active": active2,
        "files": [canonical.entry(r, files[r]) for r in listed], "append_only_prefixes": prefixes})
    # 集約の順番の記録に、この集約の manifest を1行追記する（次の集約は、これを最新として求める）
    files[canonical.AGGREGATION_LOG] = log0 + canonical._line({
        "aggregate_run": agg_run, "manifest": f"{adir}/manifest.json",
        "manifest_sha256": canonical.sha256(files[f"{adir}/manifest.json"]),
        "previous": {"path": prev_state.get("path"), "sha256": prev_state.get("expected_sha256")}})
    for rel, data in files.items():
        dst = os.path.join(root, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            f.write(data)


# 段階6（d5）：解除の承認を確かめる方法（既定は「未決定」：どの解除も有効にしない）。テストでだけ差し替える。
def _approval_verifier(repo_root: str, res: dict[str, Any]) -> tuple[bool, str]:
    from formal_g2.aggregate import canonical
    return canonical.approval_method_undecided(repo_root, res)


def main(argv: Optional[list[str]] = None) -> int:
    from formal_g2.aggregate import canonical
    p = argparse.ArgumentParser(prog="formal_g2.aggregate.main")
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--permissions", default=None, help=argparse.SUPPRESS)  # 廃止（2026-11-08）。渡すとエラー
    p.add_argument("--first-date", required=True)
    p.add_argument("--last-date", required=True)
    p.add_argument("--out", default="g2_aggregate_out")
    p.add_argument("--baseline", default="", help="評価側の基準 JSON（M7-3 の runner_image を照合する）")
    p.add_argument("--aggregate-baseline", default="", help="集約側の基準 JSON（集約のコード・Python の版）")
    p.add_argument("--tools-json", default="", help="gh・jq の版（観測値。照合しない）")
    p.add_argument("--previous-result", default=None, help=argparse.SUPPRESS)  # 廃止（段階6）。渡すとエラー
    p.add_argument("--previous-state", default="",
                   help="前回の正本：g2_evidence/_aggregate/aggregations/<id>/manifest.json か genesis.json（Y6：必須）")
    p.add_argument("--previous-state-sha256", default="", help="--previous-state のファイルの SHA-256（Y6：必須）")
    args = p.parse_args(argv)
    if args.previous_result is not None:
        p.error("--previous-result is no longer accepted: use --previous-state and --previous-state-sha256 "
                "(the previous canonical aggregation manifest, stage 6 / Y6)")
    if args.permissions is not None:
        p.error("--permissions is no longer accepted: formal permissions are read from g2_permissions/ "
                "and verified against their commit history (permission_evidence.json)")
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
    # 項目4：mode formal の実行ごとに許可を照合する。合格した許可だけを正式な実行の許可として使う。
    # 許可の無い（照合に合格しない）formal の実行は、黙って外さず違反として記録する。
    # U1（GPT 決定 2026-11-09）：許可1件で許す formal の実行は1回だけ。照合に合格した実行のうち最初のもの
    # （実行の開始時刻・run_id・run_attempt の順）だけを許可し、同じ permission_id の2回目以降は違反とする。
    permitted: list[dict[str, Any]] = []
    permission_violations: list[dict[str, Any]] = []
    checked: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for vr in verified:
        info = (vr.get("manifest") or {}).get("run") or {}
        if info.get("mode") != "formal":
            continue
        try:
            pc = permission_check(_repo_root(), vr)
        except Exception as exc:  # 想定外の例外でも許可があったとはしない（fail-closed）
            pc = {"status": "check_error", "permission_id": info.get("permission_id"), "conditions": {},
                  "problems": [f"{type(exc).__name__}: {exc}"]}
        vr["permission_check"] = pc
        checked.append((vr, info, pc))

    def _order(item: tuple[dict[str, Any], dict[str, Any], dict[str, Any]]) -> tuple[Any, ...]:
        meta = item[0].get("meta") or {}
        t = _parse_api_time(meta.get("run_started_at"))
        rid, att = str(meta.get("run_id") or ""), str(meta.get("run_attempt") or "")
        return (t is None, t or datetime.min.replace(tzinfo=timezone.utc),
                int(rid) if rid.isdigit() else 0, int(att) if att.isdigit() else 0, item[0].get("dir") or "")

    used: dict[str, str] = {}
    for vr, info, pc in sorted(checked, key=_order):
        ok = pc.get("status") == "ok" and isinstance(pc.get("permission"), dict)
        if ok and pc["permission"]["permission_id"] in used:
            pc["status"] = "permission_already_used"
            pc["used_by"] = used[pc["permission"]["permission_id"]]
            pc.setdefault("problems", []).append(
                f"permission {pc['permission']['permission_id']} was already used by run {used[pc['permission']['permission_id']]}")
            pc["permission"] = None
            ok = False
        if ok:
            used[pc["permission"]["permission_id"]] = vr.get("dir")
            # 許可は、照合に合格したその実行（dir）にだけ結び付ける（監査 2026-11-09 の指摘1）
            permitted.append(dict(pc["permission"], run=vr.get("dir")))
        else:
            permission_violations.append({"event_type": "g2_permission_violation", "run": vr.get("dir"),
                                          "race_date": info.get("race_date"),
                                          "permission_id": info.get("permission_id"),
                                          "status": pc.get("status"),
                                          "used_by": pc.get("used_by"),
                                          "failed_conditions": sorted(k for k, v in (pc.get("conditions") or {}).items()
                                                                      if v is not True),
                                          "problems": pc.get("problems")})
    # 段階6・Y6：前回の正本（状態スナップショット・H1/H2 のイベント・解除イベント）を必ず読む
    try:
        prev_state = canonical.load_previous_state(_repo_root(), args.previous_state or None,
                                                   args.previous_state_sha256 or None)
    except Exception as exc:  # noqa: BLE001  想定外の例外でも、前回の状態があったとはしない
        prev_state = {"status": "previous_state_unavailable", "problems": [f"{type(exc).__name__}: {exc}"],
                      "h1_events": [], "h2_events": [], "resolution_events": [], "snapshot": None,
                      "aggregate_result": None, "kind": None, "path": args.previous_state,
                      "expected_sha256": args.previous_state_sha256, "state_version_id": None}
    # 解除イベント（4）：検証して、有効なものだけを当てはめる。承認の確認の方法が決まるまで、どれも有効にならない
    resolutions = canonical.validate_resolutions(_repo_root(), prev_state.get("resolution_events") or [],
                                                 prev_state.get("h1_events") or [], prev_state.get("h2_events") or [],
                                                 verifier=_approval_verifier)
    h1_by_id = {e["event_id"]: e for e in prev_state.get("h1_events") or [] if isinstance(e.get("event_id"), str)}
    corrections: dict[str, str] = {}
    conflicting_corrections: set = set()
    for r in resolutions["valid"]:
        if r["event_type"] == "h1_resolution" and r["target_h1_event_id"] in h1_by_id:
            e_ = str(h1_by_id[r["target_h1_event_id"]]["eval_id"])
            if e_ in corrections and corrections[e_] != r["decision"]:
                conflicting_corrections.add(e_)  # 同じレースに違う訂正：どちらも当てない（監査 LOW）
            corrections[e_] = r["decision"]
    for e_ in conflicting_corrections:
        corrections.pop(e_, None)
    if conflicting_corrections:
        resolutions["valid"] = [r for r in resolutions["valid"] if not (
            r["event_type"] == "h1_resolution" and r["target_h1_event_id"] in h1_by_id
            and str(h1_by_id[r["target_h1_event_id"]]["eval_id"]) in conflicting_corrections)]
    result = aggregate(verified, permitted, args.first_date, args.last_date,
                       baseline, baseline_sha256, formal_context, corrections=corrections)
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
        out_run["permission_check"] = vr.get("permission_check")  # formal の実行だけ（調査の実行は None）
    result["release_availability_note"] = RELEASE_AVAILABILITY_NOTE
    result["aggregate_baseline"] = agg_base
    result["aggregate_environment"] = {  # 観測値（照合しない）
        "github_sha": os.environ.get("GITHUB_SHA"), "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "python_version": list(sys.version_info[:3]), "tools": tools,
    }
    result["registration"] = registration
    if result.get("reach") is not None:
        result["reach"]["aggregate_run_id"] = os.environ.get("GITHUB_RUN_ID")
    # U2・U3（GPT 決定 2026-11-09）：100件目の到達（reach）は変えない。違反は reach とは別のイベントとして記録する。
    # U6（GPT 決定 2026-11-09）：到達の前後は、日付ではなく実行の開始時刻で決める（violation_order_vs_reach）。
    #   違反した実行の開始が到達した実行の開始より後 → "after"（達成の判定を止めない。§6.3(6)）
    #   前 → "before"、同じ時刻・比べられない → "unknown"（どちらも達成の判定を止める。fail-closed）
    #   run_id・attempt の大小だけで順序を決めない。同じ run_id の再実行（attempt）だけは、attempt の番号が
    #   実行の順を表す既存の証拠なので、開始時刻で決まらないときに使う。
    reach = result.get("reach")
    reach_meta: Optional[dict[str, Any]] = None
    if reach is not None:
        for vr in verified:
            m = vr.get("meta") or {}
            if (str(m.get("run_id")) == str(reach.get("evaluation_run_id"))
                    and str(m.get("run_attempt")) == str(reach.get("evaluation_run_attempt"))):
                reach_meta = m
                break
    by_dir = {vr.get("dir"): vr for vr in verified}
    for v in permission_violations:
        v["violation_order_vs_reach"] = (_order_vs_reach((by_dir.get(v["run"]) or {}).get("meta") or {}, reach_meta)
                                         if reach is not None else "no_reach")
    result["permission_violations"] = permission_violations
    # Y1（GPT 決定 2026-11-09）：同じ許可の2回目以降（permission_already_used）は手続き違反として記録する。
    # 許可を使った実行より後に始まったと確かめられれば、それだけを理由に達成の判定を止めない。
    # 順序が分からなければ止めずに保留する（achievement_holds）。② を ① に繰り上げず、① の記録も変えない。
    for v in permission_violations:
        if v["status"] == "permission_already_used":
            v["order_vs_permitted_run"] = _order_vs_reach((by_dir.get(v["run"]) or {}).get("meta") or {},
                                                          (by_dir.get(v.get("used_by")) or {}).get("meta"))
            v["classification"] = ("procedure_violation" if v["order_vs_permitted_run"] == "after"
                                   else "procedure_violation_order_unknown")
    halting = [v for v in permission_violations if v["violation_order_vs_reach"] != "after"
               and v["status"] != "permission_already_used"]
    if halting:
        # 許可の無い正式な実行がある：達成の判定を止める（到達の記録は消さず、reach.status も変えない。U3・U10）
        result["achievement_halted"] = {"reason": "formal run without a verified permission",
                                        "runs": [v["run"] for v in halting]}
        result["g2_status"] = "halted_permission_violation"
    # U7（GPT 決定 2026-11-09）：許可1件で1回の照合（U1）は、この集約に入力した run_ids の範囲だけで保証する
    result["permission_single_use_scope"] = {
        "checked_within": "input run_ids only",
        "run_ids_checked": sorted({str((vr.get("meta") or {}).get("run_id")) for vr in verified}),
        "not_guaranteed": "formal evaluation runs that were not given to this aggregate (they are not listed or checked)"}
    result["formal_judgment"] = {"registration_verified": registration_verified, "note": REGISTRATION_NOTE,
                                 "expected_eval_token_permissions": formal_context["expected_eval_token_permissions"]}
    result["baseline"] = {"path": args.baseline or None, "sha256": baseline_sha256,
                          "read_error": baseline_error,
                          "runner_image": (baseline or {}).get("runner_image")}
    result["aggregator_version"] = self_version()
    # P4 4・5（段階4、GPT 指示 2026-11-09）
    result["record_facts"] = record_facts(verified)
    result["canonical_save"] = CANONICAL_SAVE_STATUS
    result["input_identity"] = input_identity(verified, args.first_date, args.last_date)
    result["subsequent_run_differences"] = subsequent_run_differences(verified, result)
    result["previous_state"] = {k: prev_state.get(k) for k in ("status", "problems", "path", "expected_sha256",
                                                                "kind", "state_version_id")}
    result["resolutions"] = resolutions["results"]
    result["corrections_applied"] = corrections
    result["reaggregation"] = None
    if prev_state.get("status") == "ok" and prev_state.get("kind") == "aggregation":
        result["reaggregation"] = compare_with_previous(prev_state["aggregate_result"], result)
        result["reaggregation"]["previous_result_sha256"] = prev_state["expected_sha256"]
        result["reaggregation"]["previous_result_source"] = {"path": prev_state["path"],
                                                             "kind": "canonical aggregation manifest (d5)"}
        # 承認された訂正（P4 12・13）による違いは、集約のやり直しの疑義（P4-D）にしない（事象として記録する）
        rg_ = result["reaggregation"]
        explained_ = [d for d in rg_.get("doubts") or [] if str(d["eval_id"]) in corrections]
        rg_["doubts"] = [d for d in rg_.get("doubts") or [] if str(d["eval_id"]) not in corrections]
        rg_["explained_by_correction"] = [d["eval_id"] for d in explained_]
    # 段階5：P4 14 の区分、15・15-2、P4 8（調査用の実行）
    s5 = stage5(verified, result, baseline, baseline_sha256, formal_context)
    result["stage5"] = s5
    # 疑義（P4 10）：集約のやり直し（5、P4-D）と、② との比較（15(c)）・調査用の実行（8）。H1 と H2 は別々に持つ
    all_doubts = ([dict(d, source="P4-5", H1=True, H2=False) for d in (result["reaggregation"] or {}).get("doubts") or []]
                  + s5["doubts"])
    result["doubts"] = all_doubts
    # H1・H2 のイベント（追記だけ。過去のイベントは消さない・変えない）。同じ疑義は同じ ID で、2回は加えない
    old_h1 = list(prev_state.get("h1_events") or [])
    old_h2 = list(prev_state.get("h2_events") or [])
    known1 = {e.get("event_id") for e in old_h1}
    known2 = {e.get("event_id") for e in old_h2}
    new_h1, new_h2 = [], []
    agg_run = f"{os.environ.get('GITHUB_RUN_ID', 'local')}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
    for d in all_doubts:
        base_ev = {"eval_id": str(d["eval_id"]), "source": d.get("source"), "case": d.get("case"),
                   "run": d.get("run"), "first_formal": d.get("first_formal"),
                   "previous_result_sha256": (result["reaggregation"] or {}).get("previous_result_sha256")
                   if d.get("source") == "P4-5" else None}
        for kind, flag, known, new in (("h1", "H1", known1, new_h1), ("h2", "H2", known2, new_h2)):
            if not d.get(flag):
                continue
            eid_ = canonical.event_id(kind, base_ev)
            d[f"{kind}_event_id"] = eid_
            if eid_ not in known and all(e["event_id"] != eid_ for e in new):
                new.append(dict(base_ev, schema_version=canonical.SCHEMA_VERSION, event_type=kind, event_id=eid_,
                                created_by_aggregate=agg_run))
    all_h1, all_h2 = old_h1 + new_h1, old_h2 + new_h2
    resolved1 = {r["target_h1_event_id"] for r in resolutions["valid"] if r["event_type"] == "h1_resolution"}
    resolved2 = {r["target_h2_event_id"] for r in resolutions["valid"] if r["event_type"] == "h2_resolution"}
    # 有効な H1・H2 は、追記だけのイベントの記録の全体から、有効に解除されたものを除いたもの
    # （与えられたスナップショットの一覧には頼らない。監査 2026-11-09 H-1）
    prev_active1 = {e["event_id"] for e in old_h1}
    prev_active2 = {e["event_id"] for e in old_h2}
    active1 = sorted({e["event_id"] for e in all_h1} - resolved1)
    active2 = sorted({e["event_id"] for e in all_h2} - resolved2)
    by1 = {e["event_id"]: e for e in all_h1}
    by2 = {e["event_id"]: e for e in all_h2}
    carried = {"H1": sorted(prev_active1 - resolved1), "H2": sorted(prev_active2 - resolved2),
               "problems": [] if prev_state.get("status") == "ok" else
               [f"{prev_state.get('status')}: {prev_state.get('problems')}"]}
    result["h1_events_new"], result["h2_events_new"] = new_h1, new_h2
    result["p4_holds"] = {"H1": sorted({by1[x]["eval_id"] for x in active1}), "H1_event_ids": active1,
                          "H2": [{"event_id": x, "eval_id": by2[x]["eval_id"], "source": by2[x].get("source"),
                                  "case": by2[x].get("case")} for x in active2],
                          "resolved": {"H1": sorted(resolved1), "H2": sorted(resolved2)},
                          "carried_from_previous": carried,
                          "note": "H1: streaks containing these races are not used for the achievement judgment "
                                  "until an approved correction record (P4 11/12). H2: the whole G2 achievement "
                                  "judgment is held until an approved closing record. Neither is released "
                                  "automatically, and neither sets a formal record to 0."}
    # 達成の判定の保留（P4 4・5、P4 11 の H1）。許可の違反による停止（achievement_halted）とは別に記録する（U10）。
    # reach は変えない。
    achievement_holds = []
    reached = set(_segment_ids(result)) | set((result["reaggregation"] or {}).get("previous_reached_segment") or [])
    h1_in = sorted(set(result["p4_holds"]["H1"]) & reached)
    if h1_in:
        achievement_holds.append({"reason": "H1 doubt inside the reached segment (P4 11)", "eval_ids": h1_in})
    rg = result["reaggregation"] or {}
    corrected = set(corrections)
    explained = bool(corrected & set(rg.get("previous_reached_segment") or []))
    if rg.get("previous_reach") is not None and rg.get("reach_same") is False and not explained:
        achievement_holds.append({"reason": "reach differs from the previous aggregate result (P4 5)",
                                  "previous_hundredth": rg["previous_reach"].get("hundredth_race_key"),
                                  "current_hundredth": (result.get("reach") or {}).get("hundredth_race_key")})
    if result["p4_holds"]["H2"]:
        achievement_holds.append({"reason": "H2 (P4 11, 15-2): whole G2 achievement judgment held",
                                  "items": result["p4_holds"]["H2"]})
    if s5["evidence_insufficient"]:
        achievement_holds.append({"reason": "investigation comparison: evidence insufficient (Y5)",
                                  "items": s5["evidence_insufficient"]})
    order_unknown_reuse = [v["run"] for v in permission_violations
                           if v.get("classification") == "procedure_violation_order_unknown"]
    if order_unknown_reuse:
        achievement_holds.append({"reason": "second use of a permission whose order is unknown (Y1)",
                                  "runs": order_unknown_reuse})
    try:
        gh_check = canonical.git_history_check(_repo_root())
    except Exception as exc:  # noqa: BLE001  確かめられなければ合格にしない
        gh_check = {"status": "history_unavailable", "problems": [f"{type(exc).__name__}: {exc}"]}
    result["git_history_check"] = gh_check
    if gh_check.get("status") != "ok":
        achievement_holds.append({"reason": "git history check of g2_evidence failed (Z3)",
                                  "status": gh_check.get("status"), "problems": gh_check.get("problems")[:10]})
    if prev_state.get("status") != "ok":
        # Y6：前回の正本・状態を確かめられない。H1・H2 なしとして扱わず、達成の判定全体を保留する
        achievement_holds.append({"reason": "previous_state_unavailable (Y6)", "status": prev_state.get("status"),
                                  "problems": prev_state.get("problems")})
    if result.get("unattributed_unavailable_runs"):
        achievement_holds.append({"reason": "run without verifiable evidence that cannot be attributed to a "
                                            "race_date (P4 4)", "runs": result["unattributed_unavailable_runs"]})
    result["achievement_holds"] = achievement_holds
    if achievement_holds and result["g2_status"] != "halted_permission_violation":
        result["g2_status"] = "held_p4"
    # d5：正本の保存の確認（P4 1 の正本保存完了）。到達した区間の実行の正本が、checkout の g2_evidence/ にあり、
    # manifest どおりであること。正式な達成記録は、人が正本の commit のあとに記録する（07f d5）。集約は
    # 「記録してよい状態か」を示すだけで、達成とは書かない。
    seg = set(_segment_ids(result))
    seg_runs = []
    for day in result.get("days") or []:
        sr = day.get("selected_run")
        if sr and any(str(x.get("eval_id", "")).startswith(day["race_date"] + "_") and x.get("eval_id") in seg
                      for x in result.get("sequence") or [] if isinstance(x, dict)):
            vr = next((v for v in verified if v.get("meta") is sr), None)
            if vr is not None:
                seg_runs.append(vr)
    canon_checks = []
    for vr in seg_runs:
        rd = canonical.run_dir_for((vr.get("manifest") or {}).get("run") or {}, vr.get("meta") or {})
        vm = canonical.verify_manifest(_repo_root(), f"{rd}/manifest.json")
        ok = vm["status"] == "ok" and (vm.get("document") or {}).get("files") is not None and any(
            e.get("path") == f"{rd}/artifact.zip" and e.get("sha256") == vr.get("artifact_sha256")
            for e in vm["document"]["files"])
        canon_checks.append({"run": vr.get("dir"), "dir": rd, "status": "saved_verified" if ok
                             else ("not_saved" if vm["status"] == "manifest_unavailable" else "mismatch"),
                             "problems": vm["problems"][:10]})
    result["canonical_save"] = {"reached_segment_runs": canon_checks,
                                "note": "save is done by a separate, individually permitted commit (d5); "
                                        "a failed or missing save is recorded separately from aggregation failures"}
    blocking = []
    if result.get("reach") is None:
        blocking.append("no reach")
    if result.get("achievement_halted"):
        blocking.append("achievement_halted")
    if achievement_holds:
        blocking.append("achievement_holds")
    if not formal_context.get("registration_verified"):
        blocking.append("registration not verified")
    if any(c["status"] != "saved_verified" for c in canon_checks) or not canon_checks:
        blocking.append("canonical save not verified for the reached segment")
    result["achievement_readiness"] = {"ready_for_human_record": not blocking, "blocking": blocking}
    result["evaluation_code_loaded"] = loaded_repo_modules
    result["generated_at"] = datetime.now().astimezone().isoformat()
    failed = [vr["dir"] for vr in verified if not vr["verified"]]
    result["verification_failed_runs"] = failed

    # W5・不変条件8：終了コードと、その原因をすべて記録する（3 が優先しても 4 の原因を残す）
    # Y1：同じ許可の2回目（手続き違反）は記録するが、照合の失敗（3）にはしない
    real_violations = [v for v in permission_violations if v["status"] != "permission_already_used"]
    verification_causes = {"verification_failed_runs": failed, "baseline_read_error": baseline_error,
                           "baseline_failures": len(result["baseline_failures"]),
                           "permission_violations": len(real_violations),
                           "procedure_violations_recorded": len(permission_violations) - len(real_violations)}
    v_fail = bool(failed or baseline_error is not None or result["baseline_failures"] or real_violations)
    o_fail = bool(result["input_order_failures"])
    result["exit_status"] = {
        "code": EXIT_VERIFICATION_FAILED if v_fail else EXIT_INPUT_ORDER_FAILED if o_fail else 0,
        "verification_failed": v_fail, "verification_causes": verification_causes,
        "input_order_failed": o_fail,
        "input_order_causes": [{"race_date": x["race_date"], "run": (x.get("run") or {}).get("run_id"),
                                "run_attempt": (x.get("run") or {}).get("run_attempt"), "check": x["check"],
                                "problems": x.get("problems")} for x in result["input_order_failures"]]}
    # 記録してよい状態の条件に、集約の終了コードが 0 であることを加える（監査 M-1）
    if result["exit_status"]["code"] != 0:
        result["achievement_readiness"]["blocking"].append(f"aggregate exit code {result['exit_status']['code']}")
        result["achievement_readiness"]["ready_for_human_record"] = False
    result["achievement_readiness"]["note"] = ("even when ready, the formal achievement record is written by a "
                                               "person after this aggregation's canonical bundle is committed (d5)")
    os.makedirs(args.out, exist_ok=True)
    # ジョブログ原本の写し（C-6：証跡として残す）
    log_dir = os.path.join(args.out, "logs")
    for vr in verified:
        name = (vr.get("job_log") or {}).get("file")
        if name:
            os.makedirs(log_dir, exist_ok=True)
            shutil.copyfile(os.path.join(args.artifacts_dir, vr["dir"], name),
                            os.path.join(log_dir, f"{vr['dir']}.log"))
    result_bytes = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=1).encode("utf-8")
    # d5：正本の束（作業ツリーに置く前のもの）。保存は canonical.stage と、個別に許可された commit で行う
    try:
        if prev_state.get("status") != "ok":
            # 再監査 L-1：前回の状態を確かめられない集約からは束を作らない（鎖を壊さない）
            raise RuntimeError(f"previous state not verified ({prev_state.get('status')}); bundle not built")
        write_canonical_bundle(args, verified, result, result_bytes, prev_state, all_h1, all_h2, active1, active2,
                               baseline_sha256, agg_base, agg_run)
    except Exception as exc:  # noqa: BLE001  束を作れなくても集約の結果は残す（保存の失敗として別に記録する）
        err = f"{type(exc).__name__}: {exc}"
        shutil.rmtree(os.path.join(args.out, "canonical_bundle"), ignore_errors=True)
        result["canonical_bundle_error"] = err
        result["achievement_readiness"]["ready_for_human_record"] = False
        result["achievement_readiness"]["blocking"].append("canonical bundle could not be built")
        result_bytes = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=1).encode("utf-8")
    with open(os.path.join(args.out, "aggregate_result.json"), "wb") as f:
        f.write(result_bytes)
    write_output_manifest(args.out)

    # 照合の結果をログに出す（値はハッシュと識別値だけ。トークン等は扱わない）
    print(f"aggregate: runs={len(verified)} verified={len(verified) - len(failed)} failed={len(failed)} "
          f"baseline_sha256={baseline_sha256} baseline_read_error={baseline_error} "
          f"baseline_failures={len(result['baseline_failures'])} "
          f"best_segment={result['best_segment']} g2_status={result['g2_status']} "
          f"aggregate_baseline={agg_base['status']} registration_verified={registration_verified} "
          f"registration={registration['status']} permission_violations={len(permission_violations)}")
    for vr in verified:
        ri = vr.get("runner_image") or {}
        print(f"run {vr['dir']}: verified={vr['verified']} runner_image_log={ri.get('log')} "
              f"env_observed_version={(ri.get('env_observed') or {}).get('ImageVersion')} "
              f"problems={vr['problems']}")
    for rr in result["runs"]:
        print(f"run {rr['dir']}: mode={rr['ref'].get('mode')} ref={rr['ref'].get('github_ref')} "
              f"w7_problems={rr['w7_check']}")
    for pv in permission_violations:
        print(f"formal run without verified permission: run={pv['run']} permission_id={pv['permission_id']} "
              f"failed={pv['failed_conditions']}")
    for bf in result["baseline_failures"]:
        print(f"baseline check failed: race_date={bf['race_date']} run={bf['run'].get('run_id')}-"
              f"{bf['run'].get('run_attempt')} check={bf['check']}")
    # 終了コード 3：集約の照合の失敗（artifact・ジョブログ・基準の読み込み・基準の照合）。
    # 評価側の設計どおりの安全停止（CONFIG・PRELOAD など）は、記録と除外が正しければ 0。
    for of in result["input_order_failures"]:
        print(f"input order check failed (race_closed_at): race_date={of['race_date']} "
              f"run={of['run'].get('run_id')}-{of['run'].get('run_attempt')} problems={of['problems'][:3]}")
    return result["exit_status"]["code"]


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
