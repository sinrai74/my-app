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

- 禁止対象（FORBIDDEN_CATEGORIES）の呼び出し、監視対象の未解決、既存のプロファイラ、監視フックの
  解除・置換、監視の異常、threading を通らないスレッドの開始、プロセスの起動は、monitoring_problems
  に入り、実行全体を不適格にする（fail-closed）。
- スレッドの開始は、プロファイラの c_call で _thread.start_new_thread を捕まえて記録する。
- あわせて、監視の開始時・各レースの前後・停止時に、生きているスレッド（sys._current_frames）が
  すべてプロファイラで観測済みかを確かめる。C の中からの起動（functools.partial など）で c_call が
  出ない場合も、区切りの時点で動いていれば捕まえる。区切りより前に終わった短命のスレッドは捕まえ
  られない（既知の限界）。

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
import _thread
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

# 禁止対象（呼ばれてはならない関数）：区分 -> [(リポジトリ相対パス, 修飾名)]
# ラベルは "<ファイルの短い名前>.<修飾名>"。既存の ML・Legacy 取得のラベルは従来の名前のまま。
# 監視のために追加の import はしない（ファイルを ast で読んで行番号を求めるだけ）。
_FORBIDDEN_SPEC: dict[str, list[tuple[str, str, str]]] = {
    "A_buyscore": [
        *[("x_buyscore.py", q, f"x_buyscore.{q}") for q in (
            "_ensure_config_file", "load_config", "_odds_band_bonus", "_race_type_bonus", "calc_buyscore",
            "check_passthrough", "stars", "investment_type", "risk_level", "kelly_fraction",
            "get_bet_multiplier_extended", "check_korogashi", "check_ippatsu", "assign_rank_labels",
            "format_buy_reason", "format_exclude_reason", "format_passthrough_message", "format_buy_message",
            "save_buyscore_log", "apply_buyscore")],
        *[("rebuild/core/buyscore.py", q, f"core.buyscore.{q}") for q in (
            "_odds_band_bonus", "_race_type_bonus", "calc_buyscore", "check_passthrough", "investment_type",
            "kelly_fraction", "DefaultKellyStrategy.__call__", "DefaultBuyEngine.__init__",
            "DefaultBuyEngine.assess")],
        ("notify_arashi.py", "_evaluate_bets", "notify_arashi._evaluate_bets"),
    ],
    "B_ml_and_legacy_fetch": [
        ("notify_arashi.py", "score_by_model", "score_by_model"),
        ("notify_arashi.py", "_load_ml_model", "_load_ml_model"),
        ("notify_arashi.py", "_predict_win_prob", "_predict_win_prob"),
        ("notify_arashi.py", "fetch_programs", "fetch_programs"),
        ("notify_arashi.py", "_safe_get", "_safe_get"),
    ],
    "C_notify": [
        ("notify_arashi.py", q, f"notify_arashi.{q}") for q in (
            "_run_main", "send_line", "send_notification", "send_email", "send_summary_if_none",
            "send_preview_notification", "_check_odds_drop")
    ],
    "D_release_write": [
        ("x_release_storage.py", q, f"x_release_storage.{q}") for q in (
            "upload_file", "_upload_asset", "_delete_asset", "_get_or_create_release", "restore_from_backup")
    ],
}
FORBIDDEN_CATEGORIES: dict[str, list[str]] = {c: [lab for _p, _q, lab in v] for c, v in _FORBIDDEN_SPEC.items()}
FORBIDDEN_LABELS: frozenset[str] = frozenset(lab for v in FORBIDDEN_CATEGORIES.values() for lab in v)
for _cat, _items in _FORBIDDEN_SPEC.items():
    for _p, _q, _lab in _items:
        WATCHED_FUNCTIONS.setdefault((_p, _q), _lab)

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
# 新しいスレッドの開始：Python 3.11 には監査イベントがないため、プロファイラの c_call で
# _thread.start_new_thread の呼び出しを捕まえる（threading.Thread.start からなら監視が引き継がれる）
_START_NEW_THREAD = _thread.start_new_thread
_THREAD_START_CODE = threading.Thread.start.__code__


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
        # 監視そのものの状態（fail-closed の判定に使う）
        self.started = False
        self.stopped = False
        self.phase_counts: dict[str, dict[str, int]] = {}
        self.hook_checks: list[dict[str, Any]] = []
        self.hook_violations: list[dict[str, Any]] = []
        self.monitor_errors: list[str] = []
        self.thread_starts: list[dict[str, Any]] = []
        # プロファイラが実際に観測したスレッド（ID）。スレッドの最も外側のフレームが戻ったら外す
        # （ID の再利用で、別のスレッドを観測済みと取り違えないため）。
        self._observed_threads: set[int] = set()
        self.thread_checks = 0
        self.unobserved_thread_violations: list[dict[str, Any]] = []

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
        if event == "return" and frame.f_back is None:
            self._observed_threads.discard(_thread.get_ident())  # このスレッドの Python の実行が終わった
        else:
            self._observed_threads.add(_thread.get_ident())
        if event == "c_call":
            if arg is _START_NEW_THREAD:
                covered = frame.f_code is _THREAD_START_CODE
                with self._lock:
                    self.thread_starts.append({"phase": self._phase, "via_threading": covered,
                                               "caller": frame.f_code.co_name})
            return
        if event != "call":
            return
        try:
            code = frame.f_code
            label = self._targets.get((code.co_filename, code.co_name, code.co_firstlineno))
            if label is not None:
                with self._lock:
                    self.counts[label] += 1
                    per = self.phase_counts.setdefault(self._phase, {})
                    per[label] = per.get(label, 0) + 1
                return
            if code.co_name == "<module>":
                flag = self._module_targets.get(os.path.realpath(code.co_filename))
                if flag is not None:
                    with self._lock:
                        self.module_flags[flag] = True
                        self.detections.append({"flag": flag, "phase": self._phase, "at": time.time(),
                                                "file": os.path.relpath(code.co_filename, self.repo_root)})
        except Exception as exc:  # noqa: BLE001  監視の異常は記録し、評価のふるまいは変えない
            self.monitor_errors.append(f"profile: {type(exc).__name__}")

    # ---- 副作用の観測 ----
    def _audit(self, event: str, args: tuple) -> None:
        try:
            self._audit_inner(event, args)
        except Exception as exc:  # noqa: BLE001
            self.monitor_errors.append(f"audit: {type(exc).__name__}")

    def _audit_inner(self, event: str, args: tuple) -> None:
        if self.stopped or not self.started:
            return
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
        self.prior_profiler_present = (sys.getprofile() is not None
                                       or threading.getprofile() is not None)
        self._env_before = _env_digest()
        sys.setprofile(self._profile)
        threading.setprofile(self._profile)
        if not self._audit_installed:
            sys.addaudithook(self._audit)
            self._audit_installed = True
        self.started = True
        self.check_threads("start")

    def check_threads(self, where: str) -> bool:
        """生きているスレッド（sys._current_frames）が、すべてプロファイラで観測済みか。

        観測していないスレッドが動いていれば、監視を保証できないので不適格にする。
        区切りの前に終わった短命のスレッドは、ここでは捕まえられない（既知の限界）。
        """
        self.thread_checks += 1
        alive = set(sys._current_frames())
        unobserved = sorted(alive - self._observed_threads)
        if unobserved:
            self.unobserved_thread_violations.append({"where": where, "count": len(unobserved),
                                                      "alive": len(alive)})
        return not unobserved

    def check_hooks(self, where: str) -> bool:
        """監視フックが外されていない・置き換えられていないか（レースの区切りと終了時に呼ぶ）。
        あわせて、観測していないスレッドが動いていないかも確かめる（check_threads）。"""
        self.check_threads(where)
        main_ok = sys.getprofile() == self._profile
        thread_ok = threading.getprofile() == self._profile
        rec = {"where": where, "sys_profile_ok": main_ok, "threading_profile_ok": thread_ok}
        self.hook_checks.append(rec)
        if not (main_ok and thread_ok):
            self.hook_violations.append(rec)
        return main_ok and thread_ok

    def forbidden_counts(self) -> dict[str, dict[str, int]]:
        return {c: {lab: self.counts.get(lab, 0) for lab in labs} for c, labs in FORBIDDEN_CATEGORIES.items()}

    def monitoring_problems(self) -> list[str]:
        """run 全体を不適格にする監視の問題（fail-closed）。空なら問題なし。"""
        problems = []
        if not self.started:
            problems.append("monitor not started")
        unresolved = sorted(k for k, v in self.resolution.items() if not v["resolved"])
        if unresolved:
            problems.append(f"unresolved targets: {unresolved}")
        if self.prior_profiler_present:
            problems.append("profiler already present before monitoring")
        if self.hook_violations:
            problems.append(f"monitor hook removed or replaced: {self.hook_violations[0]['where']}")
        if self.monitor_errors:
            problems.append(f"monitor errors: {self.monitor_errors[:3]}")
        if self.threads_at_start != 1:
            problems.append(f"threads at start = {self.threads_at_start} (expected 1)")
        uncovered = [t for t in self.thread_starts if not t["via_threading"]]
        if uncovered:
            problems.append(f"thread started outside threading (not monitored): {len(uncovered)}")
        if self.unobserved_thread_violations:
            v = self.unobserved_thread_violations[0]
            problems.append(f"unobserved live threads at {v['where']}: {v['count']}")
        if any(e["kind"] == "process" for e in self.events):
            problems.append("process started (not monitored)")
        total = sum(n for per in self.forbidden_counts().values() for n in per.values())
        if total:
            problems.append(f"forbidden calls: {total}")
        return problems

    def set_phase(self, phase: str) -> None:
        self._phase = phase

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def config_executed(self) -> bool:
        return bool(self.module_flags.get("config"))

    def stop(self) -> None:
        self.check_hooks("stop")
        sys.setprofile(None)
        threading.setprofile(None)  # type: ignore[arg-type]
        self._phase = "stopped"
        self.stopped = True

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
            "forbidden": {
                "categories": {c: list(labs) for c, labs in FORBIDDEN_CATEGORIES.items()},
                "counts_by_category": self.forbidden_counts(),
                "total": sum(n for per in self.forbidden_counts().values() for n in per.values()),
                "import_phase_counts": {lab: n for lab, n in self.phase_counts.get("import", {}).items()
                                        if lab in FORBIDDEN_LABELS},
            },
            "counts_by_phase": {ph: dict(c) for ph, c in self.phase_counts.items()},
            "monitor_state": {
                "started": self.started,
                "stopped": self.stopped,
                "hook_checks": len(self.hook_checks),
                "hook_violations": list(self.hook_violations),
                "monitor_errors": list(self.monitor_errors),
                "thread_starts": list(self.thread_starts),
                "thread_checks": self.thread_checks,
                "unobserved_thread_violations": list(self.unobserved_thread_violations),
            },
            "monitoring_problems": self.monitoring_problems(),
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
