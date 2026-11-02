"""正式G2：集約・証拠照合（C-6・§6.3〜§6.6・判断B B3）。

標準ライブラリだけを使い、評価コード・Legacy のコードは読み込まない。

入力（--artifacts-dir の下に、実行ごとのディレクトリ）：
  <run_id>-<run_attempt>/artifact.zip   … 評価の artifact（zip のまま）
  <run_id>-<run_attempt>/meta.json      … {"run_id", "run_attempt", "artifact_name",
                                            "expected_digest", "run_started_at"}
--permissions：個別の開始許可の一覧（形式は未決。下の load_permissions を参照）

出力：aggregate_result.json（判定に使った集約コード・workflow の SHA-256 を含む）
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
import sys
import zipfile
from datetime import datetime, timedelta
from typing import Any, Optional

REQUIRED_RACE_KEYS = ("eval_id", "result_category", "d9_states", "counts", "d5")
REQUIRED_COMPARED_KEYS = ("values", "comparable_hash", "hashes", "d13_2", "diffs")
TARGET = 100


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def self_version() -> dict[str, Optional[str]]:
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    out: dict[str, Optional[str]] = {}
    for rel in ("rebuild/formal_g2/aggregate/main.py", "rebuild/formal_g2/aggregate/__init__.py",
                ".github/workflows/formal_g2_aggregate.yml"):
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
    if not res["problems"]:
        res["verified"] = True
        res["run"] = json.loads(zf.read("run.json"))
        races = []
        if "races.jsonl" in names:
            for line in zf.read("races.jsonl").decode("utf-8").splitlines():
                if line.strip():
                    races.append(json.loads(line))
        res["races"] = races
    return res


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


def run_level_ok(run: dict[str, Any]) -> bool:
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
    )


def aggregate(verified: list[dict[str, Any]], permissions: list[dict[str, Any]],
              first_date: str, last_date: str) -> dict[str, Any]:
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
                if (run.get("config_detection") or {}).get("detected", True):
                    # config を検出した実行は、全レースを正式カウントから除外し、区間を終える
                    day["class"] = "normal" if targets else day["class"]
                    day["excluded"] = "config_detected"
                    seq.append({"break": "config_detected", "race_date": ds})
                elif targets:
                    day["class"] = "normal"
                    ok = run_level_ok(run)
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
    return {"days": days, "sequence": seq, "segments": segments, "best_segment": best, "g2_status": status}


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="formal_g2.aggregate.main")
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--permissions", default="")
    p.add_argument("--first-date", required=True)
    p.add_argument("--last-date", required=True)
    p.add_argument("--out", default="g2_aggregate_out")
    args = p.parse_args(argv)

    loaded_repo_modules = sorted(
        n for n, m in sys.modules.items()
        if n.split(".")[0] in ("notify_arashi", "x_asahi_scoring", "x_venue_stats", "core", "adapters", "models")
    )
    verified = [
        verify_run(os.path.join(args.artifacts_dir, d))
        for d in sorted(os.listdir(args.artifacts_dir))
        if os.path.isdir(os.path.join(args.artifacts_dir, d))
    ]
    result = aggregate(verified, load_permissions(args.permissions), args.first_date, args.last_date)
    result["runs"] = [
        {k: v for k, v in vr.items() if k in ("dir", "verified", "problems", "meta", "artifact_sha256")}
        for vr in verified
    ]
    result["aggregator_version"] = self_version()
    result["evaluation_code_loaded"] = loaded_repo_modules
    result["generated_at"] = datetime.now().astimezone().isoformat()
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "aggregate_result.json"), "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
