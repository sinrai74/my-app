"""正式G2：実行環境の基準（M7・M7-2・M7-3・§6.7）。

- 基準の値を集める（collect_snapshot）
- 登録済みの基準と照合する（compare_to_baseline）
- 読み込まれたリポジトリ内モジュールを許可リストと照合する（check_modules）

基準の登録そのもの（最初の登録・登録し直し）はユーザー本人の判断であり、
本モジュールは値の収集と照合だけを行う。
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import sys
from typing import Any, Iterable, Optional

LOCK_FILE = "requirements-g2.lock"
EVALUATE_WORKFLOW = ".github/workflows/formal_g2_evaluate.yml"

# M7 許可リスト・コード基準の候補（GPT の承認前。登録前に確定する）。
# A：評価に必要なもの
CANDIDATE_A: tuple[str, ...] = (
    EVALUATE_WORKFLOW,
    LOCK_FILE,
    "rebuild/shadow/comparator.py",
    "rebuild/core/engine.py",
    "rebuild/core/danger.py",
    "rebuild/core/rank.py",
    "rebuild/core/upset.py",
    "rebuild/models/evaluation.py",
    "rebuild/models/race.py",
    "rebuild/adapters/providers.py",
    "rebuild/adapters/exceptions.py",
    "rebuild/storage/serializers/evaluation_serializer.py",
    "rebuild/storage/serializers/common.py",
    "rebuild/storage/exceptions.py",
    "rebuild/tools/golden_wrapper.py",
    "notify_arashi.py",
    "x_asahi_scoring.py",
    "x_venue_stats.py",
    "x_release_storage.py",
)
# B：パッケージ初期化などで読み込まれ、最上位のコードは実行されるが評価には不要なもの
#   （GPT の個別判断待ち）
CANDIDATE_B: tuple[str, ...] = (
    "rebuild/core/__init__.py",
    "rebuild/core/buyscore.py",
    "rebuild/core/exceptions.py",
    "rebuild/models/__init__.py",
    "rebuild/models/output.py",
    "rebuild/models/record.py",
    "rebuild/adapters/__init__.py",
    "rebuild/storage/__init__.py",
    "rebuild/storage/serializers/__init__.py",
    "rebuild/storage/serializers/metrics_serializer.py",
    "rebuild/storage/serializers/news_serializer.py",
    "rebuild/storage/serializers/ranking_serializer.py",
    "rebuild/storage/serializers/verification_serializer.py",
    "x_buyscore.py",
)
# C：読み込まれない見込みのもの（許可リストに入れない。読み込まれたら不一致）
EXPECTED_NOT_LOADED: tuple[str, ...] = (
    "features.py",
    "migration.py",
    "odds_fetch.py",
    "config.py",
    "models.py",
    "output.py",
    "rebuild/shadow/__init__.py",
)


def sha256_file(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_lock(path: str) -> dict[str, dict[str, str]]:
    """ロックファイルを読む。'# [locked]' と '# [allowed-extra]' の2区分。"""
    sections: dict[str, dict[str, str]] = {"locked": {}, "allowed-extra": {}}
    current: Optional[str] = None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            m = re.fullmatch(r"#\s*\[(locked|allowed-extra)\]", line)
            if m:
                current = m.group(1)
                continue
            if not line or line.startswith("#"):
                continue
            if current is None:
                raise ValueError(f"package line outside a section: {line!r}")
            if "==" not in line:
                raise ValueError(f"not an exact pin: {line!r}")
            name, version = line.split("==", 1)
            sections[current][normalize_name(name.strip())] = version.strip()
    return sections


def installed_packages() -> Optional[dict[str, str]]:
    try:
        pkgs: dict[str, str] = {}
        for dist in importlib.metadata.distributions():
            name = dist.metadata.get("Name")
            if name:
                pkgs[normalize_name(name)] = dist.version
        return pkgs
    except Exception:  # 一覧を取得できない場合は不適格（§1 3(c)）
        return None


def check_lock(lock: dict[str, dict[str, str]], installed: Optional[dict[str, str]]) -> dict[str, Any]:
    """§1 3(b)(c)：(名前, バージョン) の集合が、ロック対象＋許可追加と完全一致すること。"""
    if installed is None:
        return {"ok": False, "reason": "installed package list unavailable"}
    expected = dict(lock["locked"])
    expected.update(lock["allowed-extra"])
    missing = sorted(n for n in expected if n not in installed)
    mismatched = [
        {"name": n, "expected": expected[n], "installed": installed[n]}
        for n in sorted(expected)
        if n in installed and installed[n] != expected[n]
    ]
    extra = sorted(n for n in installed if n not in expected)
    return {
        "ok": not (missing or mismatched or extra),
        "missing": missing,
        "mismatched": mismatched,
        "extra": [{"name": n, "installed": installed[n]} for n in extra],
    }


def formal_g2_eval_files(repo_root: str) -> list[str]:
    base = os.path.join(repo_root, "rebuild", "formal_g2")
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in ("aggregate", "__pycache__")]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(os.path.relpath(os.path.join(dirpath, fn), repo_root).replace(os.sep, "/"))
    return sorted(out)


def motor_history_header_sha(repo_root: str) -> Optional[str]:
    """motor_history.csv の先頭行（行末の \\r\\n / \\n を除く）のバイト列の SHA-256。"""
    try:
        with open(os.path.join(repo_root, "motor_history.csv"), "rb") as f:
            first = f.readline()
    except OSError:
        return None
    return hashlib.sha256(first.rstrip(b"\r\n")).hexdigest()


OS_FIELDS = ("system", "release", "version", "machine")


def os_info() -> dict[str, Any]:
    """OS 情報を os.uname() から取る（subprocess を起動しない。platform.platform() は
    processor の取得で `uname -p` を起動するため使わない）。取れない値は推測で埋めない。"""
    try:
        u = os.uname()
        info: dict[str, Any] = {"system": u.sysname, "release": u.release,
                                "version": u.version, "machine": u.machine}
        error = None
    except Exception as exc:  # noqa: BLE001
        info = {k: None for k in OS_FIELDS}
        error = f"{type(exc).__name__}: {exc}"
    info["source"] = "os.uname"
    info["error"] = error
    info["available"] = error is None and all(isinstance(info[k], str) and info[k] for k in OS_FIELDS)
    return info


def platform_string(info: dict[str, Any]) -> Optional[str]:
    """従来の記録項目 "platform"（文字列）の互換表現：<system>-<release>-<machine>。
    platform.platform() の末尾にあった libc の部分（例 -with-glibc2.39）は含まない。"""
    if not info.get("available"):
        return None
    return f"{info['system']}-{info['release']}-{info['machine']}"


def collect_snapshot(repo_root: str, cfg: Optional[dict]) -> dict[str, Any]:
    """M7 の基準項目の値（§3 の 1）を集める。"""
    lock_path = os.path.join(repo_root, LOCK_FILE)
    try:
        lock = parse_lock(lock_path)
        lock_error = None
    except Exception as exc:  # noqa: BLE001
        lock = {"locked": {}, "allowed-extra": {}}
        lock_error = f"{type(exc).__name__}: {exc}"
    installed = installed_packages()
    code_paths = sorted(set(CANDIDATE_A) | set(CANDIDATE_B) | set(formal_g2_eval_files(repo_root)))
    venue_cfg_path = os.path.join(repo_root, "venue_config.json")
    image_os = os.environ.get("ImageOS")
    image_version = os.environ.get("ImageVersion")
    osinfo = os_info()
    return {
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info),
            "implementation": platform.python_implementation(),
        },
        "platform": platform_string(osinfo),
        "os": osinfo,
        "image": {
            "ImageOS": image_os,
            "ImageVersion": image_version,
            "identified": bool(image_os) and bool(image_version),
        },
        "lock": {
            "path": LOCK_FILE,
            "sha256": sha256_file(lock_path),
            "parse_error": lock_error,
            "check": check_lock(lock, installed) if lock_error is None else {"ok": False, "reason": lock_error},
        },
        "installed": installed,
        "code": {p: sha256_file(os.path.join(repo_root, p)) for p in code_paths},
        "settings": {
            "asahi_config_sha256": sha256_file(os.path.join(repo_root, "asahi_config.json")),
            "cfg_has_venue_cfg_key": None if cfg is None else ("_venue_cfg" in cfg),
            "venue_config_exists": os.path.isfile(venue_cfg_path),
            "venue_config_sha256": sha256_file(venue_cfg_path),
        },
        "motor_history_header_sha256": motor_history_header_sha(repo_root),
    }


def baseline_view(snapshot: dict[str, Any]) -> dict[str, Any]:
    """基準として照合する項目だけを取り出す（登録時もこの形で保存する）。"""
    return {
        "python_version_info": snapshot["python"]["version_info"],
        "image": {"ImageOS": snapshot["image"]["ImageOS"], "ImageVersion": snapshot["image"]["ImageVersion"]},
        "lock_sha256": snapshot["lock"]["sha256"],
        "installed": snapshot["installed"],
        "code": snapshot["code"],
        "settings": snapshot["settings"],
        "motor_history_header_sha256": snapshot["motor_history_header_sha256"],
    }


def self_check(snapshot: dict[str, Any]) -> list[str]:
    """基準と比べる前に、それ自体で不適格になる条件（§1 3(c)、§2 4）。"""
    problems = []
    if not snapshot["image"]["identified"]:
        problems.append("OS image could not be identified (ImageOS/ImageVersion)")
    if not (snapshot.get("os") or {}).get("available"):
        problems.append("OS information unavailable (os.uname)")
    if not snapshot["lock"]["check"].get("ok"):
        problems.append("installed packages do not match requirements-g2.lock")
    if snapshot["settings"]["cfg_has_venue_cfg_key"]:
        problems.append("cfg has _venue_cfg key")
    missing_code = [p for p, h in snapshot["code"].items() if h is None]
    if missing_code:
        problems.append(f"code files missing: {missing_code}")
    if snapshot["motor_history_header_sha256"] is None:
        problems.append("motor_history.csv header unreadable")
    return problems


def compare_to_baseline(snapshot: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """登録済みの基準との差（空なら一致）。"""
    view = baseline_view(snapshot)
    diffs = []
    for key in ("python_version_info", "image", "lock_sha256", "installed", "settings",
                "motor_history_header_sha256"):
        if view[key] != baseline.get(key):
            diffs.append(key)
    base_code = baseline.get("code") or {}
    for path in sorted(set(view["code"]) | set(base_code)):
        if view["code"].get(path) != base_code.get(path):
            diffs.append(f"code:{path}")
    return diffs


def check_modules(loaded: Iterable[dict[str, str]], allowlist: Iterable[str]) -> dict[str, Any]:
    """読み込まれたリポジトリ内モジュールのうち、許可リストに無いもの。"""
    allowed = set(allowlist)
    extras = [m for m in loaded if m["path"] not in allowed]
    return {"ok": not extras, "not_allowed": extras}
