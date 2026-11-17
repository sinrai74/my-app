"""
正式G3（B案）：Legacy の apply_buyscore 呼び出し時・return 時の値の写し取り。

- sys.setprofile による観測だけ（差し替え・追加の呼び出しはしない）。
- 対象は「x_buyscore.py の apply_buyscore」の code object（ファイル・関数名・
  最初の行番号の組）で識別する。別名 `_apply_buyscore` で import されていても同じ。
- call 時：candidates・context を深い複製で写し取る（Legacy の buyscore・kelly は未付与）。
- return 時：戻り値の辞書と、その場で書き換え・並べ替えられた candidates を写し取る。
  例外で抜けたとき（profile の return 事象の arg が None）は legacy_exception。
- 観測関数の中の例外はすべて受け止め、Legacy の処理に伝えない。
- 既存のプロファイラがある場合は、写し取りを行わず capture_failed とする。

正式G2 の instrument.py とは別のモジュール（G2 の評価側 baseline の42ファイルを
変更しないため）。G3 の入口でだけ使う。
"""

from __future__ import annotations

import copy
import os
import sys
from typing import Any, Callable, Optional

STATUS_CAPTURED = "captured"
STATUS_NOT_CALLED = "not_called"
STATUS_LEGACY_EXCEPTION = "legacy_exception"
STATUS_CAPTURE_FAILED = "capture_failed"


def _norm(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def target_code_key(func: Callable[..., Any]) -> tuple[str, str, int]:
    code = func.__code__
    return (_norm(code.co_filename), code.co_name, code.co_firstlineno)


class ApplyBuyscoreCapture:
    """1回の `_evaluate_bets` の実行の間だけ有効にする写し取り。

    使い方:
        cap = ApplyBuyscoreCapture()
        with cap:
            result = _evaluate_bets(...)
        cap.status, cap.call_candidates, cap.call_context, ...
    """

    def __init__(self, target: Optional[Callable[..., Any]] = None) -> None:
        if target is None:
            import x_buyscore  # Legacy（読み込むだけ・変更しない）
            target = x_buyscore.apply_buyscore
        self._key = target_code_key(target)
        self._prev_profile: Any = None
        self._installed = False
        self._depth_frames: set[int] = set()
        self.call_count = 0
        self.return_count = 0
        self.call_candidates: Optional[list] = None
        self.call_context: Optional[dict] = None
        self.return_value: Optional[dict] = None
        self.return_candidates: Optional[list] = None
        self.return_cfg: Optional[dict] = None
        self.exited_by_exception = False
        self.errors: list[str] = []

    # ---- 観測関数 ----
    def _is_target(self, frame: Any) -> bool:
        code = frame.f_code
        if code.co_name != self._key[1] or code.co_firstlineno != self._key[2]:
            return False
        return _norm(code.co_filename) == self._key[0]

    def _profile(self, frame: Any, event: str, arg: Any) -> None:
        try:
            if event == "call" and self._is_target(frame):
                self.call_count += 1
                self._depth_frames.add(id(frame))
                if self.call_count == 1:
                    loc = frame.f_locals
                    self.call_candidates = copy.deepcopy(loc.get("candidates"))
                    self.call_context = copy.deepcopy(loc.get("context"))
            elif event == "return" and id(frame) in self._depth_frames and self._is_target(frame):
                self._depth_frames.discard(id(frame))
                self.return_count += 1
                if self.return_count == 1:
                    if arg is None:
                        self.exited_by_exception = True
                    else:
                        self.return_value = copy.deepcopy(arg)
                        loc = frame.f_locals
                        self.return_candidates = copy.deepcopy(loc.get("candidates"))
                        # Legacy が実際に使った設定（apply_buyscore 内の load_config() の結果）
                        self.return_cfg = copy.deepcopy(loc.get("cfg"))
        except Exception as exc:  # noqa: BLE001 — Legacy に伝えない
            self.errors.append(type(exc).__name__)

    # ---- 有効化 ----
    def __enter__(self) -> "ApplyBuyscoreCapture":
        current = sys.getprofile()
        if current is not None:
            self.errors.append("existing_profiler")
            return self
        sys.setprofile(self._profile)
        self._installed = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self._installed:
            sys.setprofile(None)
            self._installed = False
        return False

    # ---- 結果 ----
    @property
    def status(self) -> str:
        if self.errors:
            return STATUS_CAPTURE_FAILED
        if self.call_count == 0:
            return STATUS_NOT_CALLED
        if self.call_count > 1 or self.return_count > 1:
            return STATUS_CAPTURE_FAILED
        if self.exited_by_exception:
            return STATUS_LEGACY_EXCEPTION
        if self.return_count != 1 or self.call_candidates is None or self.call_context is None:
            return STATUS_CAPTURE_FAILED
        return STATUS_CAPTURED
