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
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

JST = timezone(timedelta(hours=9))

G3_PERMISSION_DIR = "g3_permissions"
G3_PERMISSION_SCHEMA_VERSION = 1
G3_PERMISSION_APPROVER = "sinrai74"
G3_PERMISSION_METHOD = "github_web_ui"
G3_FORMAL_REF = "refs/heads/main"
# G3 の workflow パスは未確定（本人確認事項 doc 17 §5-3）。確定前は None で「workflow 未確定」を明示。
G3_PERMISSION_WORKFLOW: Optional[str] = None

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
