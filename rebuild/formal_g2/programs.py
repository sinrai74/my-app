"""正式G2：出走表一覧の取得と固定（判断C C-1・C-2）。

- 1回の実行につき、対象日の一覧を1回だけ取得する（再試行しない。today.json に切り替えない）。
- 取得先 URL・ヘッダ・タイムアウトは、凍結版 notify_arashi の定数をそのまま使う。
- 1回の GET で得た同じ Response から、生のバイト列（r.content）と r.json() の両方を取る。
- 例外の扱いは notify_arashi._safe_get に合わせる：GET・raise_for_status()・r.json() の
  requests.RequestException（JSON を読めない場合を含む）は「取得できなかった」とする。
  それ以外の例外は外へ伝える。
- programs の取り出しは、凍結版の _extract_programs をそのまま使う。
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Callable, Optional

RACE_DATE_RE = re.compile(r"^\d{8}$")


class ProgramsUnavailable(Exception):
    """一覧を固定できなかった（その日は通常の日にならない）。"""


def program_url(race_date: str) -> str:
    import notify_arashi  # 定数を読むだけ

    return f"{notify_arashi.PROGRAMS_URL}/{race_date[:4]}/{race_date}.json"


def default_http_get(url: str):  # noqa: ANN201  requests.Response を返す
    """凍結版 _safe_get と同じヘッダ・タイムアウトで GET を1回だけ行い、Response を返す。"""
    import requests

    import notify_arashi

    return requests.get(url, headers=notify_arashi.HTTP_HEADERS, timeout=notify_arashi.HTTP_TIMEOUT)


def race_sort_key(p: dict) -> tuple:
    return (
        str(p.get("race_closed_at", "") or ""),
        int(p.get("race_stadium_number", 0) or 0),
        int(p.get("race_number", 0) or 0),
    )


def fix_programs(race_date: str, http_get: Optional[Callable[[str], Any]] = None) -> dict[str, Any]:
    """一覧を1回取得して固定する。返り値は記録用の情報と、固定した programs。"""
    if not RACE_DATE_RE.match(race_date):
        raise ValueError(f"race_date must be YYYYMMDD: {race_date!r}")
    import requests

    import notify_arashi

    getter = http_get or default_http_get
    url = program_url(race_date)
    calls = {"http_get": 0}

    try:
        calls["http_get"] += 1
        r = getter(url)
        r.raise_for_status()
        raw = r.content
        data = r.json()
    except requests.RequestException as exc:  # _safe_get と同じ範囲
        raise ProgramsUnavailable(f"fetch failed: {type(exc).__name__}: {exc}") from exc

    raw_sha = hashlib.sha256(raw).hexdigest()
    programs = notify_arashi._extract_programs(data)  # 凍結版の関数をそのまま使う
    if not programs:
        raise ProgramsUnavailable("programs list is empty")

    ordered = sorted(programs, key=race_sort_key)
    targets = [
        {
            "eval_id": f"{race_date}_{int(p['race_stadium_number']):02d}_{int(p['race_number']):02d}",
            "venue_num": int(p["race_stadium_number"]),
            "race_number": int(p["race_number"]),
            "race_closed_at": p.get("race_closed_at"),
            "race_date_in_program": p.get("race_date"),
        }
        for p in ordered
    ]
    targets_sha = hashlib.sha256(
        json.dumps(targets, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "url": url,
        "status_code": getattr(r, "status_code", None),
        "raw_bytes": raw,
        "raw_sha256": raw_sha,
        "programs": programs,
        "targets": targets,
        "targets_sha256": targets_sha,
        "http_get_calls": calls["http_get"],
    }


class FixedFetcher:
    """BoatsProvider に渡す fetcher。固定した一覧だけを返し、呼び出し回数を数える。"""

    def __init__(self, race_date: str, programs: list[dict]) -> None:
        self._race_date = race_date
        self._programs = programs
        self.calls: list[str] = []

    def __call__(self, race_date: str) -> list[dict]:
        self.calls.append(race_date)
        if race_date != self._race_date:
            return []
        return self._programs
