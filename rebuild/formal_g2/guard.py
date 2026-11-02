"""正式G2：実行前・実行中・終了時の検査（判断C C-0 3・C-4、W3）。

検査に失敗した場合は GuardStop を送出する。code は停止の区分：
  A-1：探索順・解決先・M7・許可リスト・shadow・config（評価の前）
  A-2：GITHUB_TOKEN・Release が使える状態
  DATE：正式な実行で race_date が JST の実行日以降
  CONFIG：評価中・終了時にルートの config.py が実行・読み込みされた
"""

from __future__ import annotations

import importlib.util
import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

JST = timezone(timedelta(hours=9))

# ルートと rebuild/ で名前が重なるパッケージ（rebuild 側が先に見つかること）。
COLLIDING_NAMES = ("models", "features", "output")
# rebuild/config/ は __init__.py を持たないため、import config はルートの config.py を指す。
# G2 経路は config を必要としないので、「読み込まれていないこと」を検査する。
CONFIG_NAME = "config"
CONFIG_FILE = "config.py"


class GuardStop(Exception):
    def __init__(self, code: str, reason: str, detail: Optional[dict] = None) -> None:
        super().__init__(f"[{code}] {reason}")
        self.code = code
        self.reason = reason
        self.detail = detail or {}


def check_token_absent() -> dict[str, Any]:
    """C-4：GITHUB_TOKEN が環境に無いこと。値は記録しない。"""
    present = "GITHUB_TOKEN" in os.environ
    record = {
        "github_token_present": present,
        "token_like_env_names": sorted(k for k in os.environ if "TOKEN" in k.upper()),
    }
    if present:
        raise GuardStop("A-2", "GITHUB_TOKEN is present in the environment")
    return record


def check_search_path(repo_root: str) -> dict[str, Any]:
    """探索順：先頭2つが <root>/rebuild, <root> の順で、'' を含まないこと。cwd はリポジトリ直下。"""
    root = os.path.realpath(repo_root)
    expected = [os.path.join(root, "rebuild"), root]
    head = [os.path.realpath(p) if p else p for p in sys.path[:2]]
    record = {
        "sys_path": list(sys.path),
        "safe_path_flag": bool(getattr(sys.flags, "safe_path", False)),
        "dont_write_bytecode": bool(sys.dont_write_bytecode),
        "cwd": os.getcwd(),
    }
    if "" in sys.path:
        raise GuardStop("A-1", "sys.path contains '' (run with python -P)")
    if head != expected:
        raise GuardStop("A-1", f"sys.path head {head!r} != expected {expected!r}")
    if os.path.realpath(os.getcwd()) != root:
        raise GuardStop("A-1", "cwd is not the repository root")
    return record


def check_colliding_modules(repo_root: str) -> dict[str, Any]:
    """models / features / output が rebuild/ 側に解決されること。"""
    rebuild_dir = os.path.join(os.path.realpath(repo_root), "rebuild") + os.sep
    result: dict[str, Any] = {}
    for name in COLLIDING_NAMES:
        spec = importlib.util.find_spec(name)
        origin = None if spec is None else spec.origin
        locations = None if spec is None else list(spec.submodule_search_locations or [])
        where = origin or (locations[0] if locations else None)
        ok = bool(where) and os.path.realpath(where).startswith(rebuild_dir)
        result[name] = {"origin": origin, "search_locations": locations, "under_rebuild": ok}
        if not ok:
            raise GuardStop("A-1", f"module {name!r} does not resolve under rebuild/: {where!r}")
    return result


# ---- config ----
def config_find_spec_record() -> dict[str, Any]:
    """find_spec('config') の解決先を記録する（import はしない）。"""
    spec = importlib.util.find_spec(CONFIG_NAME)
    return {
        "origin": None if spec is None else spec.origin,
        "search_locations": None if spec is None else list(spec.submodule_search_locations or []),
    }


def config_loaded(repo_root: str) -> dict[str, Any]:
    """sys.modules に 'config'、またはルートの config.py から読み込まれたモジュールがあるか。"""
    target = os.path.realpath(os.path.join(repo_root, CONFIG_FILE))
    by_name = CONFIG_NAME in sys.modules
    by_file = sorted(
        n for n, m in list(sys.modules.items())
        if getattr(m, "__file__", None) and os.path.realpath(m.__file__) == target
    )
    return {"in_sys_modules_by_name": by_name, "modules_from_root_config": by_file,
            "loaded": by_name or bool(by_file)}


def check_config_absent(repo_root: str, phase: str, code: str = "A-1") -> dict[str, Any]:
    rec = config_loaded(repo_root)
    rec["phase"] = phase
    if rec["loaded"]:
        raise GuardStop(code, f"config is loaded ({phase})", rec)
    return rec


# ---- 日付 ----
def check_race_date_formal(race_date: str, now: Optional[datetime] = None) -> dict[str, Any]:
    """正式な実行：race_date が JST の実行日より前であること（C-3 の具体化）。"""
    today = (now or datetime.now(JST)).astimezone(JST).strftime("%Y%m%d")
    rec = {"race_date": race_date, "run_date_jst": today, "before_run_date": race_date < today}
    if not rec["before_run_date"]:
        raise GuardStop("DATE", f"formal run requires race_date < {today} (JST)", rec)
    return rec


# ---- tests（W3 の T1〜T3） ----
def tests_probe(repo_root: str) -> dict[str, Any]:
    """T1・T2：tests は import せず、実在と探索結果を記録するだけ。"""
    root = os.path.realpath(repo_root)
    spec = importlib.util.find_spec("tests")
    return {
        "T1_rebuild_tests_init_exists": os.path.isfile(os.path.join(root, "rebuild", "tests", "__init__.py")),
        "T1_root_tests_dir_exists": os.path.isdir(os.path.join(root, "tests")),
        "T1_root_tests_init_exists": os.path.isfile(os.path.join(root, "tests", "__init__.py")),
        "T2_find_spec_origin": None if spec is None else spec.origin,
        "T2_find_spec_locations": None if spec is None else list(spec.submodule_search_locations or []),
    }


def tests_not_loaded(repo_root: str) -> dict[str, Any]:
    """T3：tests パッケージ・tests 配下のファイルが読み込まれていないこと。"""
    root = os.path.realpath(repo_root)
    dirs = (os.path.join(root, "rebuild", "tests") + os.sep, os.path.join(root, "tests") + os.sep)
    bad = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None) or ""
        real = os.path.realpath(f) if f else ""
        if name == "tests" or name.startswith("tests.") or (real and real.startswith(dirs)):
            bad.append(name)
    return {"T3_loaded_tests_modules": sorted(bad), "T3_ok": not bad}


def check_release_unavailable() -> dict[str, Any]:
    """C-4：x_release_storage.is_available() が False であること（判断D 13-2）。"""
    import x_release_storage  # 読み込みのみ（Legacy は変更しない）

    available = bool(x_release_storage.is_available())
    if available:
        raise GuardStop("A-2", "x_release_storage.is_available() is True")
    return {"release_storage_available": available}
