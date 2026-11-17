"""
正式G3（段階4-2）：G3 専用 permission の照合（GPT最終判断④・doc 17 §2-4, §3）。

G2 方式（本人が GitHub 画面で g2_permissions/<id>.json を commit、集約が SHA・窓・mode を照合）を
基礎に、G3 専用へ分離する：
  - 保存先は g3_permissions/（G2 permission は流用しない。別 ID 空間）。
  - investigation と formal を mode で区別。**investigation は permission 不要**（G2 と同様に
    formal のみ必須）。formal の再試行には新しい permission を要する（permission_id は再利用しない前提）。
  - UTC 期間（window_start/end の Z 表記）・JST 日付ガードで比較（文字列比較ではない）。
  - fail-closed：形式不正・窓外・日付不正・mode 不一致・approved_by 不一致は不許可。

本モジュールは **許可ファイルの形式・窓・日付・mode の純粋な検証**を担う。
git 履歴の12条件（commit がちょうど1件 added／author=sinrai74／committer=web-flow／署名 verified／
blob SHA 一致／実行 HEAD に含まれる等）は、現行 G2 の aggregate.permission_check と同じ方式で
**actions 層（workflow 実行時）に配線**する（Actions 起動は段階4-2 の許可外のため、本モジュールでは
インターフェースの想定だけを記し実装しない）。承認者・証拠の最終形式と本人承認手順は本人確認事項。

正式G2 の評価側 baseline 42ファイル・formal_g2/・Legacy・instrument は変更しない新規ファイル。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

JST = timezone(timedelta(hours=9))

G3_PERMISSION_DIR = "g3_permissions"
G3_PERMISSION_SCHEMA_VERSION = 1
G3_PERMISSION_APPROVER = "sinrai74"
G3_PERMISSION_METHOD = "github_web_ui"
G3_FORMAL_REF = "refs/heads/main"
# GPT確定（2026-11-18）：G3 formal の workflow パスと ref を確定。
G3_PERMISSION_WORKFLOW: Optional[str] = ".github/workflows/formal_g3_evaluate.yml"
G3_PERMISSION_COMMITTER = "web-flow"
# 履歴・ファイル数の限界（G2 と同じ。API per_page=100 と同数なら切れている可能性で不合格）
G3_PERMISSION_HISTORY_LIMIT = 100
G3_PERMISSION_COMMIT_FILES_LIMIT = 300
_PERMISSION_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")

MODE_FORMAL = "formal"
MODE_INVESTIGATION = "investigation"
_VALID_MODES = (MODE_FORMAL, MODE_INVESTIGATION)

PERMISSION_KEYS = ("schema_version", "permission_id", "race_date", "window_start", "window_end",
                   "workflow", "ref", "mode", "approved_on", "approved_by", "approval_method")

_RACE_DATE_RE = re.compile(r"^\d{8}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_UTC_Z_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _parse_utc_z(v: Any) -> Optional[datetime]:
    if not isinstance(v, str) or not _UTC_Z_RE.match(v):
        return None
    try:
        return datetime.strptime(v, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def permission_file_problems(doc: Any, permission_id: str, *,
                             expected_workflow: Optional[str] = G3_PERMISSION_WORKFLOW,
                             require_formal: bool = True) -> list[str]:
    """許可ファイルの形式（schema_version 1）。problems が空なら形式 OK。

    require_formal=True（formal 実行の照合）では mode は formal のみを許す。
    investigation の記録ファイル（任意）を検証する場合は require_formal=False。
    expected_workflow=None（G3 workflow 未確定）のときは workflow の一致検査をしない
    （未確定であることを problems に記録する）。
    """
    if not isinstance(doc, dict):
        return ["permission file is not a JSON object"]
    probs: list[str] = []
    missing = [k for k in PERMISSION_KEYS if k not in doc]
    extra = sorted(k for k in doc if k not in PERMISSION_KEYS)
    if missing:
        probs.append(f"missing keys {missing}")
    if extra:
        probs.append(f"unexpected keys {extra}")
    sv = doc.get("schema_version")
    if isinstance(sv, bool) or sv != G3_PERMISSION_SCHEMA_VERSION:
        probs.append(f"schema_version must be {G3_PERMISSION_SCHEMA_VERSION}")
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
    mode = doc.get("mode")
    if mode not in _VALID_MODES:
        probs.append(f"mode must be one of {_VALID_MODES}")
    elif require_formal and mode != MODE_FORMAL:
        probs.append("mode must be formal for a formal run")
    if expected_workflow is None:
        probs.append("G3 workflow is undetermined (doc 17 §5-3): set G3_PERMISSION_WORKFLOW to enable workflow check")
    elif doc.get("workflow") != expected_workflow:
        probs.append(f"workflow must be {expected_workflow}")
    if doc.get("ref") != G3_FORMAL_REF:
        probs.append(f"ref must be {G3_FORMAL_REF}")
    ao = doc.get("approved_on")
    if not (isinstance(ao, str) and _DATE_RE.match(ao)):
        probs.append("approved_on is not YYYY-MM-DD")
    else:
        try:
            datetime.strptime(ao, "%Y-%m-%d")
        except ValueError:
            probs.append("approved_on is not a valid date")
    if doc.get("approved_by") != G3_PERMISSION_APPROVER:
        probs.append(f"approved_by must be {G3_PERMISSION_APPROVER}")
    if doc.get("approval_method") != G3_PERMISSION_METHOD:
        probs.append(f"approval_method must be {G3_PERMISSION_METHOD}")
    return probs


def check_race_date_formal(race_date: str, now: Optional[datetime] = None) -> dict[str, Any]:
    """formal 実行：race_date が JST の実行日より前であること（G2 guard.check_race_date_formal と同じ規則）。"""
    today = (now or datetime.now(JST)).astimezone(JST).strftime("%Y%m%d")
    before = isinstance(race_date, str) and _RACE_DATE_RE.match(race_date) is not None and race_date < today
    return {"race_date": race_date, "run_date_jst": today, "before_run_date": bool(before), "ok": bool(before)}


def within_window(doc: Any, run_started_at: Any) -> dict[str, Any]:
    """実行開始時刻（UTC Z）が許可の窓 [window_start, window_end) に入るか。タイムゾーンで比較。"""
    ws, we = _parse_utc_z(doc.get("window_start")), _parse_utc_z(doc.get("window_end"))
    t = _parse_utc_z(run_started_at)
    # 監査 B-4：成功・失敗で同じキー集合を返す（呼び出し側のキー前提での KeyError を防ぐ）
    base = {"window_start": doc.get("window_start") if isinstance(doc, dict) else None,
            "window_end": doc.get("window_end") if isinstance(doc, dict) else None,
            "run_started_at": run_started_at, "reason": None}
    if ws is None or we is None or t is None:
        return {**base, "ok": False, "reason": "unparseable_time"}
    return {**base, "ok": bool(ws <= t < we)}


def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_sha256(s: Any) -> bool:
    return isinstance(s, str) and _SHA256_RE.match(s) is not None


def investigation_needs_permission() -> bool:
    """investigation は permission 不要（GPT④・G2 と同様に formal のみ必須）。"""
    return False


# ==================== git 履歴12条件（GPT確定 2026-11-18・G2 と同方式） ====================
# G2 の aggregate.permission_check と同じ判定を G3 定数で行う。GitHub API の取得（履歴・commit・
# compare・署名）は actions 層（workflow 実行時）が行い、その結果を vr["permission_evidence"] に
# 入れて渡す。本関数は取得済み evidence に対する純粋判定（Actions を起動しない）。
# G2 のコード・基準は変更しない（同型のロジックを G3 側に配線）。

import os as _os  # noqa: E402


def _repo_relpath(repo_root: str, path: str) -> Optional[str]:
    try:
        root = _os.path.realpath(repo_root)
        real = _os.path.realpath(path)
        rel = _os.path.relpath(real, root)
    except (OSError, ValueError):
        return None
    if rel == _os.curdir or rel.startswith(_os.pardir + _os.sep) or rel == _os.pardir or _os.path.isabs(rel):
        return None
    return rel.replace(_os.sep, "/")


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key {k!r}")
        out[k] = v
    return out


def _git_blob_sha(data: bytes) -> str:
    import hashlib as _h
    return _h.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _parse_api_time(v: Any) -> Optional[datetime]:
    if not isinstance(v, str) or not v:
        return None
    try:
        t = datetime.fromisoformat(v[:-1] + "+00:00" if v.endswith("Z") else v)
    except ValueError:
        return None
    return t if t.tzinfo is not None else None


def workflow_consistency(repo_root: str, doc: Any, meta: Any, info: Any) -> dict[str, Any]:
    """①：実際の workflow ファイルの実在・permission の workflow 指定・実行時 workflow/ref の一致。"""
    wf = G3_PERMISSION_WORKFLOW
    meta = meta if isinstance(meta, dict) else {}
    info = info if isinstance(info, dict) else {}
    wf_path = _os.path.join(repo_root, *wf.split("/")) if isinstance(wf, str) else None
    file_exists = bool(wf_path and _os.path.isfile(wf_path) and not _os.path.islink(wf_path))
    doc_wf = doc.get("workflow") if isinstance(doc, dict) else None
    doc_ref = doc.get("ref") if isinstance(doc, dict) else None
    conds = {
        "workflow_file_exists": file_exists,
        "permission_workflow_matches": doc_wf == wf,
        "run_workflow_matches": meta.get("api_workflow_path") == wf,
        "permission_ref_matches": doc_ref == G3_FORMAL_REF,
        "run_ref_matches": info.get("github_ref") == G3_FORMAL_REF,
    }
    return {"ok": all(conds.values()), "conditions": conds, "expected_workflow": wf,
            "expected_ref": G3_FORMAL_REF}


def permission_check(repo_root: str, vr: dict[str, Any]) -> dict[str, Any]:
    """G3 formal 実行の許可照合（G2 permission_check と同じ12条件、G3 定数）。

    status が "ok" のときだけ正式な実行として扱う。API 失敗・履歴不足・不一致・不在は fail-closed。
    vr["permission_evidence"]（actions 層が取得）: fetch_ok・permission_id・path・history_run_head・
    history_aggregate_head・commits[{sha,files,author_login,committer_login,verified,
    verification_reason,committer_date,files_count}]・aggregate_ref・compare{base,head,status}。
    """
    info = (vr.get("manifest") or {}).get("run") or {}
    meta = vr.get("meta") or {}
    pid = info.get("permission_id")
    out: dict[str, Any] = {"status": "not_permitted", "permission_id": pid, "path": None, "blob_sha": None,
                           "commit_sha": None, "author": None, "committer": None, "verified": None,
                           "commit_time": None, "run_started_at": meta.get("run_started_at"),
                           "conditions": {}, "problems": []}
    cond = out["conditions"]

    def done() -> dict[str, Any]:
        failed = [k for k, v in cond.items() if v is not True]
        out["status"] = "ok" if len(cond) == 12 and not failed and not out["problems"] else "not_permitted"
        out["permission"] = ({k: doc[k] for k in ("permission_id", "race_date", "window_start", "window_end")}
                             if out["status"] == "ok" else None)
        return out

    doc: Any = None
    if not (isinstance(pid, str) and _PERMISSION_ID_RE.match(pid) and pid not in (".", "..")):
        out["problems"].append("permission_id is missing or has invalid characters")
        return done()
    rel = f"{G3_PERMISSION_DIR}/{pid}.json"
    out["path"] = rel
    p = _os.path.join(repo_root, G3_PERMISSION_DIR, f"{pid}.json")
    cond["1_file_exists"] = (_os.path.isfile(p) and not _os.path.islink(p) and _repo_relpath(repo_root, p) == rel)
    data = b""
    if cond["1_file_exists"]:
        try:
            with open(p, "rb") as f:
                data = f.read()
            doc = json.loads(data, object_pairs_hook=_no_duplicate_keys)
        except (OSError, ValueError) as exc:
            out["problems"].append(f"permission file unreadable: {type(exc).__name__}")
        out["blob_sha"] = _git_blob_sha(data) if data else None
    cond["2_name_matches_id"] = isinstance(doc, dict) and doc.get("permission_id") == pid
    fprobs = permission_file_problems(doc, pid, require_formal=True) if doc is not None else ["permission file not read"]
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
    truncated = ok_lists and (len(hr) >= G3_PERMISSION_HISTORY_LIMIT or len(hm) >= G3_PERMISSION_HISTORY_LIMIT)
    if truncated:
        out["problems"].append("permission history may be truncated")
    sha = hr[0] if ok_lists and len(hr) == 1 else None
    c = commits.get(sha) if sha else None
    files = [f for f in ((c or {}).get("files") or [])
             if isinstance(f, dict) and (f.get("filename") == rel or f.get("previous_filename") == rel)]
    fc = (c or {}).get("files_count")
    files_ok = isinstance(fc, int) and not isinstance(fc, bool) and 1 <= fc < G3_PERMISSION_COMMIT_FILES_LIMIT
    if c is not None and not files_ok:
        out["problems"].append(f"commit files_count {fc!r} is not below {G3_PERMISSION_COMMIT_FILES_LIMIT}")
    cond["4_added_once"] = (sha is not None and not truncated and c is not None and files_ok and len(files) == 1
                            and files[0].get("status") == "added" and files[0].get("filename") == rel
                            and not files[0].get("previous_filename"))
    agg_ref_ok = ev.get("aggregate_ref") == G3_FORMAL_REF
    if not agg_ref_ok:
        out["problems"].append(f"aggregate ref is not {G3_FORMAL_REF}: {ev.get('aggregate_ref')!r}")
    cond["5_not_changed_after"] = cond["4_added_once"] and hm == [sha] and agg_ref_ok
    out["commit_sha"] = sha
    out["author"] = (c or {}).get("author_login")
    out["committer"] = (c or {}).get("committer_login")
    out["verified"] = (c or {}).get("verified")
    out["commit_time"] = (c or {}).get("committer_date")
    cond["6_author"] = out["author"] == G3_PERMISSION_APPROVER
    cond["7_committer"] = out["committer"] == G3_PERMISSION_COMMITTER
    cond["8_signature_verified"] = out["verified"] is True
    ct, started = _parse_api_time(out["commit_time"]), _parse_api_time(meta.get("run_started_at"))
    cond["9_committed_before_run"] = ct is not None and started is not None and ct < started
    cmp = ev.get("compare") or {}
    cond["10_in_run_head"] = (sha is not None and cmp.get("base") == sha
                              and cmp.get("head") == meta.get("api_head_sha") and bool(cmp.get("head"))
                              and cmp.get("status") in ("ahead", "identical"))
    cond["11_blob_matches"] = (bool(out["blob_sha"]) and len(files) == 1
                               and files[0].get("sha") == out["blob_sha"])
    m12 = isinstance(doc, dict) and cond["3_format_ok"]
    if m12:
        ws, we = _parse_utc_z(doc["window_start"]), _parse_utc_z(doc["window_end"])
        wc = workflow_consistency(repo_root, doc, meta, info)
        m12 = (doc["race_date"] == info.get("race_date") and started is not None
               and ws is not None and we is not None and ws <= started <= we
               and doc["mode"] == info.get("mode") and doc["permission_id"] == pid and wc["ok"])
        out["workflow_consistency"] = wc
    cond["12_matches_run"] = bool(m12)
    return done()
