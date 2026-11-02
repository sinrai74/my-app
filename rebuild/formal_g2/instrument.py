"""正式G2：呼び出し・副作用の観測（W7・判断D 8）。

評価対象のふるまいを変えないため、関数の差し替え（ラッパー）は使わない。
- 呼び出しの観測：sys.setprofile / threading.setprofile で、コードオブジェクトの
  (co_filename, co_name, co_firstlineno) の組を数える。co_firstlineno は、観測の開始時に
  ast で対象ファイルから求める（デコレータがあれば最初のデコレータの行）。
  これにより、Protocol などにある同じ名前のメソッドを数えない。
  別名 import・関数内 import でも、同じコードオブジェクトが呼ばれれば捕捉できる。
- モジュールの実行の観測：ルートの config.py・features.py の <module> のコードが
  実行されたら印を立てる（例外は投げない。停止は呼び出し側が区切りで行う）。
- 副作用の観測：sys.addaudithook で、通信・書き込みモードの open・プロセス起動などを
  記録する（記録するだけで、止めない）。
- 環境変数の変化：観測区間の前後で os.environ のハッシュを比べる。

捕捉できないもの（記録上は 'unverified' として扱う）：
- 観測開始より前に作られたスレッド
- C 拡張から直接行われる呼び出し・副作用
- 既存のプロファイラを上書きした場合の、そのプロファイラ側の観測
"""

from __future__ import annotations

import ast
import hashlib
import os
import sys
import threading
import time
from typing import Any, Optional

# 観測する関数：(リポジトリ相対パス, 修飾名) -> ラベル
WATCHED_FUNCTIONS: dict[tuple[str, str], str] = {
    ("notify_arashi.py", "score_by_model"): "score_by_model",
    ("notify_arashi.py", "_load_ml_model"): "_load_ml_model",
    ("notify_arashi.py", "_predict_win_prob"): "_predict_win_prob",
    ("notify_arashi.py", "fetch_programs"): "fetch_programs",
    ("notify_arashi.py", "_safe_get"): "_safe_get",
    ("rebuild/core/engine.py", "Ver4Engine.evaluate"): "Ver4Engine.evaluate",
    ("rebuild/core/engine.py", "Ver4Engine.build_legacy_evaluation"): "Ver4Engine.build_legacy_evaluation",
    ("x_asahi_scoring.py", "build_race_evaluation_v4"): "build_race_evaluation_v4",
    ("rebuild/shadow/comparator.py", "compare"): "compare",
    ("rebuild/adapters/providers.py", "BoatsProvider.resolve_boats"): "BoatsProvider.resolve_boats",
    ("rebuild/adapters/providers.py", "BoatsProvider.resolve_race"): "BoatsProvider.resolve_race",
    ("x_release_storage.py", "download_file"): "x_release_storage.download_file",
}

# <module> の実行を観測するルートのファイル：(リポジトリ相対パス) -> 印の名前
WATCHED_MODULES: dict[str, str] = {
    "config.py": "config",
    "features.py": "root_features",
}

# 監査イベントのうち記録するもの
_AUDIT_NETWORK = {"socket.connect", "socket.getaddrinfo", "urllib.Request"}
_AUDIT_PROCESS = {"subprocess.Popen", "os.system", "os.posix_spawn"}
_AUDIT_FS_CHANGE = {"os.remove", "os.rename", "os.replace", "os.mkdir", "os.rmdir",
                    "shutil.copyfile", "shutil.move", "shutil.rmtree"}


def _env_digest() -> str:
    h = hashlib.sha256()
    for k in sorted(os.environ.keys()):
        h.update(k.encode("utf-8", "replace") + b"\0")
        h.update(os.environ[k].encode("utf-8", "replace") + b"\0")
    return h.hexdigest()


def first_lines(path: str) -> dict[str, int]:
    """ファイル内の関数・メソッドの修飾名 -> co_firstlineno に当たる行番号。"""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    out: dict[str, int] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for ch in ast.iter_child_nodes(node):
            if isinstance(ch, ast.ClassDef):
                walk(ch, prefix + ch.name + ".")
            elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                line = min([ch.lineno] + [d.lineno for d in ch.decorator_list])
                out.setdefault(prefix + ch.name, line)

    walk(tree, "")
    return out


class Instrument:
    """観測器。start() は評価対象のモジュールを読み込む前に呼ぶ。"""

    def __init__(self, repo_root: str, watched: Optional[dict[tuple[str, str], str]] = None,
                 watched_modules: Optional[dict[str, str]] = None) -> None:
        self.repo_root = os.path.realpath(repo_root)
        self._watched = dict(WATCHED_FUNCTIONS if watched is None else watched)
        self._watched_modules = dict(WATCHED_MODULES if watched_modules is None else watched_modules)
        self._targets: dict[tuple[str, str, int], str] = {}
        self.resolution: dict[str, Any] = {}
        self.counts: dict[str, int] = {label: 0 for label in self._watched.values()}
        self._module_targets = {
            os.path.join(self.repo_root, rel): flag for rel, flag in self._watched_modules.items()
        }
        self.module_flags: dict[str, bool] = {flag: False for flag in self._watched_modules.values()}
        self.detections: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.threads_at_start: Optional[int] = None
        self.prior_profiler_present: Optional[bool] = None
        self._env_before: Optional[str] = None
        self._phase = "init"
        self._audit_installed = False
        self._lock = threading.Lock()

    def _resolve_targets(self) -> None:
        cache: dict[str, Optional[dict[str, int]]] = {}
        for (rel, qualname), label in self._watched.items():
            path = os.path.join(self.repo_root, rel)
            if path not in cache:
                try:
                    cache[path] = first_lines(path)
                except (OSError, SyntaxError):
                    cache[path] = None
            lines = cache[path]
            line = None if lines is None else lines.get(qualname)
            name = qualname.split(".")[-1]
            self.resolution[label] = {"path": rel, "qualname": qualname, "co_firstlineno": line,
                                      "resolved": line is not None}
            if line is not None:
                self._targets[(path, name, line)] = label

    # ---- 呼び出し・モジュール実行の観測 ----
    def _profile(self, frame, event, arg):  # noqa: ANN001
        if event != "call":
            return
        code = frame.f_code
        label = self._targets.get((code.co_filename, code.co_name, code.co_firstlineno))
        if label is not None:
            with self._lock:
                self.counts[label] += 1
            return
        if code.co_name == "<module>":
            flag = self._module_targets.get(os.path.realpath(code.co_filename))
            if flag is not None:
                with self._lock:
                    self.module_flags[flag] = True
                    self.detections.append({"flag": flag, "phase": self._phase, "at": time.time(),
                                            "file": os.path.relpath(code.co_filename, self.repo_root)})

    # ---- 副作用の観測 ----
    def _audit(self, event: str, args: tuple) -> None:
        kind = None
        detail: Any = None
        if event == "open":
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            )
            if writing:
                kind, detail = "file_write", str(args[0])
        elif event in _AUDIT_NETWORK:
            kind, detail = "network", repr(args[:2])
        elif event in _AUDIT_PROCESS or event.startswith("os.exec") or event.startswith("os.spawn"):
            kind, detail = "process", event
        elif event in _AUDIT_FS_CHANGE:
            kind, detail = "fs_change", f"{event} {args[:1]!r}"
        elif event in ("os.putenv", "os.unsetenv"):
            kind, detail = "env_change", event
        if kind is None:
            return
        with self._lock:
            self.events.append({"phase": self._phase, "kind": kind, "detail": detail})

    def start(self) -> None:
        self._resolve_targets()
        self.threads_at_start = threading.active_count()
        self.prior_profiler_present = sys.getprofile() is not None
        self._env_before = _env_digest()
        sys.setprofile(self._profile)
        threading.setprofile(self._profile)
        if not self._audit_installed:
            sys.addaudithook(self._audit)
            self._audit_installed = True

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def config_executed(self) -> bool:
        return bool(self.module_flags.get("config"))

    def stop(self) -> None:
        sys.setprofile(None)
        threading.setprofile(None)  # type: ignore[arg-type]
        self._phase = "stopped"

    def snapshot(self) -> dict[str, Any]:
        """観測結果。確かめられない項目は 'unverified' を明示する。"""
        return {
            "call_counts": dict(self.counts),
            "target_resolution": dict(self.resolution),
            "unresolved_targets": sorted(k for k, v in self.resolution.items() if not v["resolved"]),
            "module_flags": dict(self.module_flags),
            "module_detections": list(self.detections),
            "side_effect_events": list(self.events),
            "env_changed": None if self._env_before is None else self._env_before != _env_digest(),
            "threads_at_start": self.threads_at_start,
            "threads_now": threading.active_count(),
            "prior_profiler_present": self.prior_profiler_present,
            "limitations": {
                "threads_started_before_observation": "unverified",
                "c_extension_direct_calls": "unverified",
                "c_extension_direct_side_effects": "unverified",
            },
        }


def repo_modules(repo_root: str) -> list[dict[str, str]]:
    """sys.modules のうち、リポジトリ内のファイルから読み込まれたもの。"""
    root = os.path.realpath(repo_root) + os.sep
    out = []
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f:
            continue
        real = os.path.realpath(f)
        if real.startswith(root):
            out.append({"module": name, "path": os.path.relpath(real, repo_root).replace(os.sep, "/")})
    return sorted(out, key=lambda d: (d["path"], d["module"]))


def modules_from_file(repo_root: str, rel: str) -> list[str]:
    """指定したリポジトリ内ファイルから読み込まれたモジュール名。"""
    target = os.path.realpath(os.path.join(repo_root, rel))
    return sorted(
        n for n, m in list(sys.modules.items())
        if (getattr(m, "__file__", None) and os.path.realpath(m.__file__) == target)
    )
