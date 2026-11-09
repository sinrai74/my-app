"""正式G2：正本（d5）の束の作成・保存・検証、状態スナップショット、H1・H2 のイベントと解除（段階6）。

標準ライブラリだけを使う。通信しない。commit・push はしない（正本は、本人の個別の許可を得た commit で保存する。
07f §6.6 d5）。このモジュールの保存（stage）は、作業ツリーへのファイルの書き込みと読み戻しの確認だけを行う。

置き場所（07f d5「証拠一式を g2_evidence/<permission_id>/ に保存する」を守り、集約の記録は別に置く）：
  g2_evidence/<permission_id>/<run_id>-<run_attempt>/        評価の実行ごとの正本（①・②・③・未判定）
      manifest.json  evaluation_evidence.json  artifact_manifest.json  artifact.zip  meta.json  [job.log]
  g2_evidence/_aggregate/unattached/<race_date>/<run_id>-<run_attempt>/   許可 ID の無い・使えない実行
  g2_evidence/_aggregate/aggregations/<aggregate_run_id>-<attempt>/
      manifest.json  aggregate_result.json  input_manifest.json
  g2_evidence/_aggregate/state/snapshots/<state_version_id>.json
  g2_evidence/_aggregate/events/h1_events.jsonl  h2_events.jsonl  resolution_events.jsonl（追記だけ）
  g2_evidence/_aggregate/state/snapshots/genesis.json         最初の状態（本人が作る。H1・H2 なし）
"_" で始まる名前は集約の記録のために取っておく。"_" で始まる permission_id の正本は作らない（衝突として扱う）。
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import zipfile
from typing import Any, Callable, Optional

ROOT = "g2_evidence"
AGG = f"{ROOT}/_aggregate"
EVENTS_DIR = f"{AGG}/events"
SNAPSHOT_DIR = f"{AGG}/state/snapshots"
AGGREGATIONS_DIR = f"{AGG}/aggregations"
H1_EVENTS = f"{EVENTS_DIR}/h1_events.jsonl"
H2_EVENTS = f"{EVENTS_DIR}/h2_events.jsonl"
RESOLUTION_EVENTS = f"{EVENTS_DIR}/resolution_events.jsonl"
# 集約の正本の順番の記録（追記だけ）。前回の状態は、この記録の最後の行でなければならない（古い状態の再利用を防ぐ）
AGGREGATION_LOG = f"{AGG}/aggregation_log.jsonl"
APPEND_ONLY = (H1_EVENTS, H2_EVENTS, RESOLUTION_EVENTS, AGGREGATION_LOG)
AGG_RUN_RE = re.compile(r"^[0-9]+-[0-9]+$")
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
# 実行の正本に残す meta の項目（取得の時刻など、集約のたびに変わる値は入れない。監査 2026-11-09 H-2）
SOURCE_META_KEYS = ("run_id", "run_attempt", "artifact_name", "expected_digest", "run_started_at", "job_id",
                    "job_count", "log_fetch_ok", "log_sha256", "log_bytes", "api_run_id", "api_run_attempt",
                    "api_head_sha", "api_head_branch", "api_workflow_path", "api_repository", "api_event",
                    "artifact_status")
GENESIS = f"{SNAPSHOT_DIR}/genesis.json"
SCHEMA_VERSION = 1
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
# H1 の訂正の内容（07f P4 12：「0」か「維持」。根拠が無いときは「疑義のまま」で、解除の記録は作らない）
H1_DECISIONS = ("zero", "maintain")
H2_DECISIONS = ("closed",)  # P4 11：原因と対処を書いた事象の終了記録
RESOLUTION_KEYS_COMMON = ("schema_version", "event_type", "resolution_id", "decision", "reason", "evidence",
                          "approval_reference", "approved_by", "approved_at", "created_at")


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _dumps(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _line(obj: Any) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _no_dup(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"duplicate key: {k}")
        out[k] = v
    return out


def loads(b: bytes) -> Any:
    return json.loads(b, object_pairs_hook=_no_dup)


def _safe_rel(rel: str) -> bool:
    if not isinstance(rel, str) or not rel.startswith(ROOT + "/") or "\\" in rel or "\x00" in rel:
        return False
    parts = rel.split("/")
    return all(p not in ("", ".", "..") for p in parts)


def _abs(repo_root: str, rel: str) -> str:
    return os.path.join(repo_root, *rel.split("/"))


def read_repo_file(repo_root: str, rel: str) -> bytes:
    """リポジトリの中の通常のファイルを読む（シンボリックリンク・外に出るパスは拒否する）。"""
    if not _safe_rel(rel):
        raise ValueError(f"unsafe path: {rel!r}")
    p = _abs(repo_root, rel)
    real_root = os.path.realpath(repo_root)
    if os.path.islink(p) or not os.path.isfile(p):
        raise FileNotFoundError(rel)
    if os.path.relpath(os.path.realpath(p), real_root).replace(os.sep, "/") != rel:
        raise ValueError(f"path resolves outside itself: {rel!r}")
    with open(p, "rb") as f:
        return f.read()


def entry(rel: str, data: bytes) -> dict[str, Any]:
    return {"path": rel, "size": len(data), "sha256": sha256(data)}


# ---- 束（bundle）の作成 ----
def run_dir_for(info: dict[str, Any], meta: dict[str, Any]) -> str:
    pid, rid, att = info.get("permission_id"), str(meta.get("run_id")), str(meta.get("run_attempt"))
    tail = f"{rid}-{att}"
    if isinstance(pid, str) and SAFE_NAME_RE.match(pid):
        return f"{ROOT}/{pid}/{tail}"
    rd = info.get("race_date") if isinstance(info.get("race_date"), str) else "unknown_date"
    return f"{AGG}/unattached/{rd if re.match(r'^[0-9]{8}$', rd) else 'unknown_date'}/{tail}"


def artifact_manifest(blob: bytes) -> dict[str, Any]:
    out = {"artifact_sha256": sha256(blob), "artifact_size": len(blob), "files": [], "readable": True}
    try:
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for n in sorted(z.namelist()):
                d = z.read(n)
                out["files"].append({"path": n, "size": len(d), "sha256": sha256(d)})
    except Exception as exc:  # noqa: BLE001
        out["readable"] = False
        out["error"] = type(exc).__name__
    return out


def build_run_files(vr: dict[str, Any], run_dir: str, raw_dir: str) -> dict[str, bytes]:
    """評価の実行1件の正本（2.1）。どの集約から作っても同じバイト列になるものだけを入れる（監査 H-2）。
    元の artifact とジョブログはそのまま写す。実行区分・照合の結果・基準・取得の時刻は、集約の正本
    （input_manifest.json の runs）に記録する（集約のたびに変わりうるため）。"""
    info = (vr.get("manifest") or {}).get("run") or {}
    meta = vr.get("meta") or {}
    files: dict[str, bytes] = {}
    for name in ("artifact.zip", "job.log"):
        p = os.path.join(raw_dir, name)
        if os.path.isfile(p) and not os.path.islink(p):
            with open(p, "rb") as f:
                files[f"{run_dir}/{name}"] = f.read()
    files[f"{run_dir}/source_meta.json"] = _dumps({k: meta.get(k) for k in SOURCE_META_KEYS})
    blob = files.get(f"{run_dir}/artifact.zip", b"")
    am = artifact_manifest(blob) if blob else {"artifact_sha256": None, "files": [], "readable": False}
    files[f"{run_dir}/artifact_manifest.json"] = _dumps(dict(am, schema_version=SCHEMA_VERSION))
    ev = {"schema_version": SCHEMA_VERSION, "kind": "g2_evaluation_evidence",
          "race_date": info.get("race_date"), "permission_id": info.get("permission_id"),
          "evaluation_run_id": meta.get("run_id"), "run_attempt": meta.get("run_attempt"),
          "run_started_at": meta.get("run_started_at"),
          "run_finished_at": (vr.get("run") or {}).get("finished_at"), "mode": info.get("mode"),
          "workflow": meta.get("api_workflow_path"), "ref": info.get("github_ref"),
          "head_sha": meta.get("api_head_sha"),
          "artifact": {"name": meta.get("artifact_name"), "digest": meta.get("expected_digest"),
                       "sha256": sha256(blob) if blob else None,
                       "status": meta.get("artifact_status", "downloaded")},
          "provenance": {"source": "GitHub Actions artifact and job log fetched by the aggregate workflow",
                         "classification_verification_baseline_and_fetch_time":
                             "recorded per aggregation in aggregations/<id>/input_manifest.json"}}
    files[f"{run_dir}/evaluation_evidence.json"] = _dumps(ev)
    files[f"{run_dir}/manifest.json"] = _dumps({
        "schema_version": SCHEMA_VERSION, "kind": "g2_evaluation_manifest", "dir": run_dir,
        "evaluation_run_id": meta.get("run_id"), "run_attempt": meta.get("run_attempt"),
        "files": sorted((entry(k, v) for k, v in files.items()), key=lambda e: e["path"])})
    return files


def event_id(kind: str, d: dict[str, Any]) -> str:
    # 同じ疑義は同じ ID（集約のたびに新しいイベントを作らない。前回の結果の SHA-256 は ID に入れない）
    key = {k: d.get(k) for k in ("eval_id", "source", "case", "run", "first_formal")}
    return f"{kind}-" + sha256(_line(key))[:24]


def parse_jsonl(data: bytes) -> tuple[list[dict[str, Any]], list[str]]:
    rows, probs = [], []
    if data and not data.endswith(b"\n"):
        probs.append("file does not end with a newline (possibly truncated)")
    for i, ln in enumerate(data.splitlines()):
        try:
            obj = loads(ln)
        except ValueError as exc:
            probs.append(f"line {i + 1}: {exc}")
            continue
        if not isinstance(obj, dict):
            probs.append(f"line {i + 1}: not an object")
            continue
        rows.append(obj)
    return rows, probs


# ---- 検証 ----
def verify_manifest(repo_root: str, manifest_rel: str, expected_sha256: Optional[str] = None) -> dict[str, Any]:
    """manifest に書いた全ファイルの存在・サイズ・SHA-256 を確かめる（名前やサイズだけで一致としない）。
    events の追記ファイルは、manifest に書いた行数・先頭部分の SHA-256 で確かめる（あとの追記を許す）。"""
    out: dict[str, Any] = {"manifest": manifest_rel, "status": "ok", "problems": [], "manifest_sha256": None}
    try:
        raw = read_repo_file(repo_root, manifest_rel)
        man = loads(raw)
    except (OSError, ValueError) as exc:
        out.update(status="manifest_unavailable", problems=[f"{type(exc).__name__}: {exc}"])
        return out
    out["manifest_sha256"] = sha256(raw)
    if expected_sha256 is not None and out["manifest_sha256"] != expected_sha256:
        out["status"] = "manifest_sha_mismatch"
        out["problems"].append("manifest sha256 differs from the recorded value")
        return out
    if not isinstance(man, dict) or man.get("schema_version") != SCHEMA_VERSION or not isinstance(man.get("files"), list):
        out.update(status="manifest_invalid", problems=["manifest shape invalid"])
        return out
    out["document"] = man
    for e in man["files"]:
        if not (isinstance(e, dict) and isinstance(e.get("path"), str) and isinstance(e.get("size"), int)
                and isinstance(e.get("sha256"), str)):
            out["problems"].append(f"bad entry: {e!r}"[:200])
            continue
        try:
            d = read_repo_file(repo_root, e["path"])
        except (OSError, ValueError) as exc:
            out["problems"].append(f"missing: {e['path']} ({type(exc).__name__})")
            continue
        if len(d) != e["size"]:
            out["problems"].append(f"size mismatch: {e['path']}")
        elif sha256(d) != e["sha256"]:
            out["problems"].append(f"sha256 mismatch: {e['path']}")
    for e in man.get("append_only_prefixes") or []:
        try:
            d = read_repo_file(repo_root, e["path"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            out["problems"].append(f"append-only file missing: {e!r} ({type(exc).__name__})"[:200])
            continue
        lines = d.splitlines(keepends=True)
        n = e.get("lines")
        if not isinstance(n, int) or len(lines) < n or sha256(b"".join(lines[:n])) != e.get("sha256"):
            out["problems"].append(f"append-only prefix changed: {e.get('path')}")
    if out["problems"]:
        out["status"] = "mismatch"
    return out


# ---- 保存（作業ツリーへの書き込みだけ。commit はしない） ----
def stage(bundle_root: str, repo_root: str) -> dict[str, Any]:
    """束を作業ツリーに保存する。既存のファイルは上書きしない。
    - 同じパス・同じ内容：同一の保存（identical）
    - 追記だけのファイル（events/*.jsonl）：既存の内容が新しい内容の先頭と一致するときだけ追記
    - それ以外の違い：衝突（conflict）。1つでも衝突があれば、何も書かない
    書いたあとに読み戻し、全ファイルの SHA-256 と、束の中の全 manifest を確かめる。途中の失敗は成功にしない。"""
    out: dict[str, Any] = {"status": None, "written": [], "identical": [], "appended": [], "conflicts": [],
                           "problems": [], "manifests": []}
    plan: list[tuple[str, bytes, str]] = []
    for dp, dns, fns in os.walk(bundle_root):
        dns.sort()
        for fn in sorted(fns):
            src = os.path.join(dp, fn)
            rel = os.path.relpath(src, bundle_root).replace(os.sep, "/")
            if os.path.islink(src) or not _safe_rel(rel):
                out["conflicts"].append(f"unsafe bundle path: {rel}")
                continue
            if rel.split("/")[1].startswith("_") and not rel.startswith(AGG + "/"):
                out["conflicts"].append(f"reserved name: {rel}")
                continue
            with open(src, "rb") as f:
                new = f.read()
            dst = _abs(repo_root, rel)
            if os.path.lexists(dst):
                try:
                    old = read_repo_file(repo_root, rel)
                except (OSError, ValueError) as exc:
                    out["conflicts"].append(f"existing path unusable: {rel} ({type(exc).__name__})")
                    continue
                if old == new:
                    out["identical"].append(rel)
                elif rel in APPEND_ONLY and new.startswith(old) and (not old or old.endswith(b"\n")):
                    plan.append((rel, new, "append"))
                else:
                    out["conflicts"].append(f"different content at existing path: {rel}")
            else:
                plan.append((rel, new, "write"))
    if out["conflicts"]:
        out["status"] = "conflict"
        return out
    real_root = os.path.realpath(repo_root)
    for rel, _, _ in plan:
        # 途中のディレクトリがシンボリックリンクで、リポジトリの外に書くことを防ぐ（書く前に確かめる）
        d = os.path.dirname(_abs(repo_root, rel))
        while not os.path.exists(d) and d != os.path.dirname(d):
            d = os.path.dirname(d)
        if os.path.relpath(os.path.realpath(d), real_root).startswith(".."):
            out["conflicts"].append(f"parent directory resolves outside the repository: {rel}")
    if out["conflicts"]:
        out["status"] = "conflict"
        return out
    tmp = None
    try:
        for rel, data, kind in plan:
            dst = _abs(repo_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dst), prefix=".g2tmp-")
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, dst)
            tmp = None
            (out["written"] if kind == "write" else out["appended"]).append(rel)
    except OSError as exc:
        if tmp and os.path.exists(tmp):
            os.remove(tmp)
        out["status"] = "write_failed"
        out["problems"].append(f"{type(exc).__name__}: {exc} (partially written: {out['written'] + out['appended']})")
        return out
    for rel, data, _ in plan:
        try:
            if read_repo_file(repo_root, rel) != data:
                out["problems"].append(f"read-back differs: {rel}")
        except (OSError, ValueError) as exc:
            out["problems"].append(f"read-back failed: {rel} ({type(exc).__name__})")
    for dp, _, fns in os.walk(bundle_root):
        if "manifest.json" in fns:
            rel = os.path.relpath(os.path.join(dp, "manifest.json"), bundle_root).replace(os.sep, "/")
            v = verify_manifest(repo_root, rel)
            out["manifests"].append({"manifest": rel, "status": v["status"]})
            out["problems"] += v["problems"]
    out["status"] = "verify_failed" if out["problems"] else "ok"
    return out


# ---- 状態（前回の正本）の読み込み（Y6） ----
def load_previous_state(repo_root: str, manifest_rel: Optional[str], expected_sha256: Optional[str]) -> dict[str, Any]:
    """前回の集約の正本（aggregations/<id>/manifest.json）か、最初の状態（genesis.json）を読む。
    1 指定 2 存在 3 形式 4 必須項目・型 5 SHA-256 6 由来 7 H1・H2 の復元 8 解除イベントの検証（main 側）
    どれかに失敗したら status != "ok"（main() は達成の判定全体を保留し、H1・H2 を消さない）。"""
    out: dict[str, Any] = {"status": "ok", "problems": [], "path": manifest_rel, "expected_sha256": expected_sha256,
                           "kind": None, "snapshot": None, "aggregate_result": None, "state_version_id": None,
                           "h1_events": [], "h2_events": [], "resolution_events": []}

    def fail(status: str, msg: str) -> dict[str, Any]:
        out["status"] = status
        out["problems"].append(msg)
        return out

    if not manifest_rel or not expected_sha256:
        return fail("previous_state_not_given", "previous state path and sha256 are required (Y6)")
    if not (isinstance(expected_sha256, str) and HEX64_RE.match(expected_sha256)):
        return fail("previous_state_invalid", "expected sha256 is not a lowercase sha256")
    try:
        log_bytes = read_repo_file(repo_root, AGGREGATION_LOG)
    except FileNotFoundError:
        log_bytes = b""
    except (OSError, ValueError) as exc:
        return fail("previous_state_unavailable", f"{AGGREGATION_LOG}: {type(exc).__name__}")
    log_rows, log_probs = parse_jsonl(log_bytes)
    if log_probs:
        return fail("previous_state_invalid", f"{AGGREGATION_LOG}: {log_probs[:3]}")
    out["aggregation_log"] = log_rows
    out["aggregation_log_bytes"] = log_bytes
    if manifest_rel == GENESIS:
        try:
            raw = read_repo_file(repo_root, GENESIS)
            g = loads(raw)
        except (OSError, ValueError) as exc:
            return fail("previous_state_unavailable", f"genesis: {type(exc).__name__}: {exc}")
        if sha256(raw) != expected_sha256:
            return fail("previous_state_sha_mismatch", "genesis sha256 differs from the given value")
        if g != {"schema_version": SCHEMA_VERSION, "kind": "g2_state_genesis", "state_version_id": "genesis",
                 "h1_active": [], "h2_active": [], "reach": None}:
            return fail("previous_state_invalid", "genesis snapshot shape invalid")
        # 最初の状態は、まだ集約の正本も H1/H2 のイベントも無いときだけ使える（古い状態の再利用を防ぐ。監査 H-1）
        if log_rows:
            return fail("previous_state_not_latest", "genesis given but aggregation records already exist")
        # ログを消して genesis に戻すことを防ぐ：集約の正本・スナップショットが1つでもあれば使えない（再監査 M-1）
        for rel_dir, allowed in ((AGGREGATIONS_DIR, set()), (SNAPSHOT_DIR, {"genesis.json"})):
            d = _abs(repo_root, rel_dir)
            if os.path.isdir(d) and set(os.listdir(d)) - allowed:
                return fail("previous_state_not_latest", f"genesis given but {rel_dir} is not empty")
        for rel in (H1_EVENTS, H2_EVENTS):
            try:
                if read_repo_file(repo_root, rel).strip():
                    return fail("previous_state_not_latest", f"genesis given but {rel} is not empty")
            except FileNotFoundError:
                pass
            except (OSError, ValueError) as exc:
                return fail("previous_state_unavailable", f"{rel}: {type(exc).__name__}")
        out.update(kind="genesis", snapshot=g, state_version_id="genesis")
    else:
        if not (manifest_rel.startswith(AGGREGATIONS_DIR + "/") and manifest_rel.endswith("/manifest.json")
                and manifest_rel.count("/") == AGGREGATIONS_DIR.count("/") + 2):
            return fail("previous_state_invalid", f"not an aggregation manifest path: {manifest_rel!r}")
        v = verify_manifest(repo_root, manifest_rel, expected_sha256)
        if v["status"] != "ok":
            return fail("previous_state_unavailable" if v["status"] == "manifest_unavailable"
                        else "previous_state_sha_mismatch" if v["status"] == "manifest_sha_mismatch"
                        else "previous_state_invalid", f"{v['status']}: {v['problems'][:5]}")
        man = v["document"]
        agg_id = man.get("aggregate_run")
        # 由来：manifest の場所と、中の集約の識別が一致する
        if (man.get("kind") != "g2_aggregation_manifest" or not isinstance(agg_id, str)
                or not AGG_RUN_RE.match(agg_id)
                or manifest_rel != f"{AGGREGATIONS_DIR}/{agg_id}/manifest.json"
                or not (isinstance(man.get("commit_sha"), str) and HEX40_RE.match(man["commit_sha"]))):
            return fail("previous_state_provenance_unknown", "aggregation manifest provenance fields inconsistent")
        # 集約の順番の記録の全体を確かめる（再監査 L-2）：各行の manifest が実在して SHA-256 が合い、
        # 各行の previous が1つ前の行（最初の行は genesis）を指している
        try:
            prev_ref = {"path": GENESIS, "sha256": sha256(read_repo_file(repo_root, GENESIS))}
        except (OSError, ValueError):
            prev_ref = {"path": GENESIS, "sha256": None}
        for k, row in enumerate(log_rows):
            if not (isinstance(row, dict) and isinstance(row.get("manifest"), str)
                    and isinstance(row.get("manifest_sha256"), str)):
                return fail("previous_state_invalid", f"aggregation log line {k + 1} malformed")
            pv = row.get("previous") or {}
            if pv.get("path") != prev_ref["path"] or ("sha256" in prev_ref and pv.get("sha256") != prev_ref["sha256"]):
                return fail("previous_state_invalid", f"aggregation log line {k + 1} does not chain to the line before")
            try:
                if sha256(read_repo_file(repo_root, row["manifest"])) != row["manifest_sha256"]:
                    return fail("previous_state_invalid", f"aggregation log line {k + 1}: manifest sha256 mismatch")
            except (OSError, ValueError) as exc:
                return fail("previous_state_invalid", f"aggregation log line {k + 1}: {type(exc).__name__}")
            prev_ref = {"path": row["manifest"], "sha256": row["manifest_sha256"]}
        # 最新であること：集約の順番の記録の最後の行が、この manifest（パスと SHA-256）である
        last = log_rows[-1] if log_rows else None
        if not (isinstance(last, dict) and last.get("manifest") == manifest_rel
                and last.get("manifest_sha256") == expected_sha256 and last.get("aggregate_run") == agg_id):
            return fail("previous_state_not_latest", "the given manifest is not the last entry of the aggregation log")
        try:
            snap = loads(read_repo_file(repo_root, man["snapshot"]))
            res = loads(read_repo_file(repo_root, man["aggregate_result"]))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return fail("previous_state_unavailable", f"{type(exc).__name__}: {exc}")
        listed = {e.get("path") for e in man["files"] if isinstance(e, dict)}
        if man.get("snapshot") not in listed or man.get("aggregate_result") not in listed:
            return fail("previous_state_invalid", "snapshot/result not covered by the manifest")
        if not (isinstance(snap, dict) and snap.get("kind") == "g2_state_snapshot"
                and snap.get("schema_version") == SCHEMA_VERSION and snap.get("aggregate_run") == agg_id
                and isinstance(snap.get("h1_active"), list) and isinstance(snap.get("h2_active"), list)
                and all(isinstance(x, str) for x in snap["h1_active"] + snap["h2_active"])):
            return fail("previous_state_invalid", "state snapshot shape invalid")
        if (snap.get("previous_state") or {}).get("status") != "ok":
            # その集約自身が前回の状態を確かめられなかった：引き継ぐべき保留が分からない（監査 LOW）
            return fail("previous_state_invalid", "the previous aggregation itself had no verified previous state")
        out.update(kind="aggregation", snapshot=snap, aggregate_result=res,
                   state_version_id=snap.get("state_version_id"))
    # 7：H1・H2 の復元（スナップショットの有効なイベントが、イベントの記録に全部ある）
    for rel, key in ((H1_EVENTS, "h1_events"), (H2_EVENTS, "h2_events"), (RESOLUTION_EVENTS, "resolution_events")):
        try:
            data = read_repo_file(repo_root, rel)
        except FileNotFoundError:
            data = b""
        except (OSError, ValueError) as exc:
            return fail("previous_state_unavailable", f"{rel}: {type(exc).__name__}")
        rows, probs = parse_jsonl(data)
        if probs:
            return fail("previous_state_invalid", f"{rel}: {probs[:3]}")
        out[key] = rows
        out[key + "_bytes"] = data
    if out["kind"] == "aggregation":
        pref = {e.get("path"): e.get("lines") for e in (man.get("append_only_prefixes") or []) if isinstance(e, dict)}
        for rel, key in ((H1_EVENTS, "h1_events"), (H2_EVENTS, "h2_events")):
            if pref.get(rel) != len(out[key]):
                return fail("previous_state_not_latest",
                            f"{rel} has events not covered by the given aggregation manifest")
    ids1 = {r.get("event_id") for r in out["h1_events"]}
    ids2 = {r.get("event_id") for r in out["h2_events"]}
    snap = out["snapshot"]
    if not set(snap["h1_active"]) <= ids1 or not set(snap["h2_active"]) <= ids2:
        return fail("previous_state_unrestorable", "active H1/H2 in the snapshot are not all in the event records")
    for rows, kind in ((out["h1_events"], "h1"), (out["h2_events"], "h2")):
        seen = set()
        for r in rows:
            if (r.get("event_type") != kind or not isinstance(r.get("event_id"), str) or r["event_id"] in seen
                    or not isinstance(r.get("eval_id"), str)):  # 再監査 L-3：必須の項目
                return fail("previous_state_invalid", f"{kind} events: bad or duplicate event")
            seen.add(r["event_id"])
    return out


# ---- Git の履歴による確かめ（Z3、GPT 決定 2026-11-09） ----
# 集約の workflow は checkout を fetch-depth: 0 で行い、手元の git の履歴だけを読む（通信しない・トークンを使わない）。
# アプリの追記の制御・集約時の検証・この履歴の確かめ・本人による commit の確認を重ねる。履歴だけで改ざんを完全に
# 防げるわけではない（履歴そのものを書き換えた force push などは、本人の確認と GitHub 側の設定による）。
def _git(repo_root: str, *args: str) -> bytes:
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "GIT_CONFIG_NOSYSTEM": "1", "HOME": "/nonexistent",
           "LC_ALL": "C", "GIT_NO_REPLACE_OBJECTS": "1"}
    r = subprocess.run(["git", "-c", "log.showSignature=false", "-C", repo_root, *args], capture_output=True, env=env, timeout=120, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"git {args[0]} failed: {r.stderr.decode('utf-8', 'replace')[:200]}")
    return r.stdout


def git_history_check(repo_root: str) -> dict[str, Any]:
    """g2_evidence/ の commit の履歴を確かめる。
    - 追記だけのファイル（events・集約の順番の記録）：各 commit の版が、1つ前の版の後ろに足しただけである
    - それ以外の正本のファイル：一度追加されたあと、変更・削除・改名されていない
    - 作業ツリーの中身が、最後に commit された版と同じ（commit されていない変更が無い）"""
    out: dict[str, Any] = {"status": "ok", "problems": [], "commits": 0}

    def fail(status: str, msg: str) -> dict[str, Any]:
        out["status"] = status
        out["problems"].append(msg)
        return out
    try:
        if _git(repo_root, "rev-parse", "--is-shallow-repository").strip() != b"false":
            return fail("history_unavailable", "repository is shallow (checkout with fetch-depth: 0 is required)")
        try:
            _git(repo_root, "rev-parse", "--verify", "-q", "HEAD")
        except RuntimeError:
            # まだ commit が無い：g2_evidence/ も無ければ確かめるものが無い
            if os.path.lexists(_abs(repo_root, ROOT)):
                return fail("history_unavailable", "g2_evidence exists but the repository has no commits")
            return out
        # 最終監査（2026-11-09）：merge commit の中の変更も見る（first-parent の差分）。順は commit の並び（topo）
        log = _git(repo_root, "log", "--first-parent", "--diff-merges=first-parent", "--topo-order",
                   "--format=%x00%H", "--name-status", "--no-renames", "--", ROOT)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return fail("history_unavailable", f"{type(exc).__name__}: {exc}")
    commits = [c for c in log.decode("utf-8", "replace").split("\x00") if c.strip()]
    out["commits"] = len(commits)
    added: dict[str, int] = {}
    for c in commits:
        lines = c.strip().splitlines()
        for ln in lines[1:]:
            parts = ln.split("\t")
            if len(parts) != 2:
                continue
            st, path = parts
            if path in APPEND_ONLY:
                continue
            if st != "A":
                out["problems"].append(f"canonical file changed after it was added ({st}): {path}")
            else:
                added[path] = added.get(path, 0) + 1
    for path, n in added.items():
        if n > 1:
            out["problems"].append(f"canonical file added more than once: {path}")
    for rel in APPEND_ONLY:
        try:
            hs = [h for h in _git(repo_root, "log", "--first-parent", "--diff-merges=first-parent", "--topo-order",
                                  "--format=%H", "--reverse", "--no-renames", "--", rel).decode().split() if h]
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            return fail("history_unavailable", f"{rel}: {type(exc).__name__}")
        prev = b""
        for h in hs:
            try:
                cur = _git(repo_root, "show", f"{h}:{rel}")
            except RuntimeError:
                out["problems"].append(f"append-only file deleted at {h[:12]}: {rel}")
                prev = None
                break
            if not (cur.startswith(prev) and (not prev or prev.endswith(b"\n"))):
                out["problems"].append(f"append-only file rewritten at {h[:12]}: {rel}")
            prev = cur
        if prev is not None:
            try:
                now = read_repo_file(repo_root, rel)
            except FileNotFoundError:
                now = b""
            except (OSError, ValueError) as exc:
                out["problems"].append(f"{rel}: {type(exc).__name__}")
                continue
            if now != prev:
                out["problems"].append(f"working tree differs from the last committed version: {rel}")
    # 作業ツリーの g2_evidence/ に、commit されていない変更・追跡されていないファイルが無い
    try:
        st = _git(repo_root, "status", "--porcelain", "-z", "--untracked-files=all", "--", ROOT)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        return fail("history_unavailable", f"git status: {type(exc).__name__}")
    if st.strip(b"\x00"):
        out["problems"].append("uncommitted or untracked changes under g2_evidence/")
    if out["problems"]:
        out["status"] = "history_mismatch"
    return out


# ---- 解除イベント（4） ----
ApprovalVerifier = Callable[[str, dict[str, Any]], tuple[bool, str]]


def approval_method_undecided(repo_root: str, res: dict[str, Any]) -> tuple[bool, str]:
    """解除の本人承認を確かめる方法は、正式な判断でまだ決まっていない（P4 11・12 は「ユーザーが承認した」とだけ
    定める）。新しい承認方式を作らないため、どの解除も有効にしない（fail-closed）。"""
    return False, "approval verification method not decided (P4 11/12)"


def validate_resolutions(repo_root: str, rows: list[dict[str, Any]], h1_events: list[dict[str, Any]],
                         h2_events: list[dict[str, Any]],
                         verifier: ApprovalVerifier = approval_method_undecided) -> dict[str, Any]:
    """解除イベントを検証する。有効なものだけを返す。解除は自動では作らない・当てはめない（承認の確認が要る）。"""
    by1 = {e["event_id"]: e for e in h1_events if isinstance(e.get("event_id"), str)}
    by2 = {e["event_id"]: e for e in h2_events if isinstance(e.get("event_id"), str)}
    results, valid = [], []
    seen_ids: dict[str, int] = {}
    targets: dict[str, set] = {}
    for r in rows:
        rid = r.get("resolution_id")
        seen_ids[str(rid)] = seen_ids.get(str(rid), 0) + 1
        t = r.get("target_h1_event_id") if r.get("event_type") == "h1_resolution" else r.get("target_h2_event_id")
        targets.setdefault(str(t), set()).add(json.dumps({"d": r.get("decision"), "r": rid}, sort_keys=True))
    for r in rows:
        probs = []
        et = r.get("event_type")
        if et == "h1_resolution":
            keys = set(RESOLUTION_KEYS_COMMON) | {"target_h1_event_id", "target_race_key"}
            target = by1.get(r.get("target_h1_event_id"))
            if target is None:
                probs.append("target H1 event not found")
            elif r.get("target_race_key") != target.get("eval_id"):
                probs.append("target race does not match the H1 event")
            if r.get("decision") not in H1_DECISIONS:
                probs.append(f"decision must be one of {H1_DECISIONS}")
            elif r.get("decision") == "maintain" and target is not None and target.get("case") == "P4-C":
                probs.append("maintain is not allowed for P4-C (07f)")
            tid = r.get("target_h1_event_id")
        elif et == "h2_resolution":
            keys = set(RESOLUTION_KEYS_COMMON) | {"target_h2_event_id"}
            if by2.get(r.get("target_h2_event_id")) is None:
                probs.append("target H2 event not found")
            if r.get("decision") not in H2_DECISIONS:
                probs.append(f"decision must be one of {H2_DECISIONS}")
            tid = r.get("target_h2_event_id")
        else:
            results.append({"resolution_id": r.get("resolution_id"), "valid": False,
                            "problems": ["unknown event_type"]})
            continue
        if set(r) != keys:
            probs.append(f"keys mismatch: missing {sorted(keys - set(r))} extra {sorted(set(r) - keys)}")
        if r.get("schema_version") != SCHEMA_VERSION or isinstance(r.get("schema_version"), bool):
            probs.append("schema_version")
        for k in ("resolution_id", "reason", "approval_reference", "approved_by", "approved_at", "created_at"):
            if not (isinstance(r.get(k), str) and r.get(k)):
                probs.append(f"{k} missing or not a string")
        if seen_ids.get(str(r.get("resolution_id")), 0) > 1:
            probs.append("duplicate resolution_id")
        if len(targets.get(str(tid), ())) > 1:
            probs.append("more than one resolution for the same target (conflicting or duplicate)")
        ev = r.get("evidence")
        if not (isinstance(ev, list) and ev):
            probs.append("evidence missing")
        else:
            for e in ev:
                if not (isinstance(e, dict) and set(e) == {"path", "sha256"} and isinstance(e.get("path"), str)
                        and isinstance(e.get("sha256"), str) and HEX64_RE.match(e["sha256"])):
                    probs.append(f"bad evidence entry: {e!r}"[:150])
                    continue
                try:
                    if sha256(read_repo_file(repo_root, e["path"])) != e["sha256"]:
                        probs.append(f"evidence sha256 mismatch: {e['path']}")
                except (OSError, ValueError) as exc:
                    probs.append(f"evidence missing: {e['path']} ({type(exc).__name__})")
        if not probs:
            ok, why = verifier(repo_root, r)
            if not ok:
                probs.append(f"approval not verified: {why}")
        results.append({"resolution_id": r.get("resolution_id"), "event_type": et, "target": tid,
                        "decision": r.get("decision"), "valid": not probs, "problems": probs})
        if not probs:
            valid.append(r)
    return {"results": results, "valid": valid}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="正式G2 正本の束の保存（作業ツリーへの書き込みだけ）・検証")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage")
    s.add_argument("--bundle", required=True)
    s.add_argument("--repo", required=True)
    v = sub.add_parser("verify")
    v.add_argument("--repo", required=True)
    v.add_argument("--manifest", required=True)
    v.add_argument("--sha256", default=None)
    a = p.parse_args(argv)
    if a.cmd == "stage":
        r = stage(a.bundle, a.repo)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0 if r["status"] == "ok" else 5 if r["status"] == "conflict" else 6
    r = verify_manifest(a.repo, a.manifest, a.sha256)
    r.pop("document", None)
    print(json.dumps(r, ensure_ascii=False, indent=2))
    return 0 if r["status"] == "ok" else 6


if __name__ == "__main__":
    sys.exit(main())
