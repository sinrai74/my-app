"""
正式G3（B案）用の入力型。

- MlBonusResult: Rebuild が算出した ML 加点の結果（Rebuild 由来）。
- BuyScoreInput: Legacy `_evaluate_bets` の中で apply_buyscore 呼び出し時に
  写し取った値（Legacy 由来）。Legacy が計算した buyscore・kelly・skip_reason 等は
  持たない（計算前の時点の値だけ）。

正式G2 の評価側 baseline の42ファイルに含まれない新規ファイル。
models/__init__.py からは import しない（42ファイルを変更しないため）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


class G3InputValidationError(ValueError):
    """G3 入力型の検証エラー（範囲外・None・型違い・欠け）。"""


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise G3InputValidationError(f"{name} must be bool (got {type(value).__name__})")


def _require_real(name: str, value: Any) -> float:
    if value is None:
        raise G3InputValidationError(f"{name} is required (None)")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise G3InputValidationError(f"{name} must be a real number (got {type(value).__name__})")
    f = float(value)
    if math.isnan(f) or math.isinf(f):
        raise G3InputValidationError(f"{name} must be finite (got {value!r})")
    return f


def _require_unit(name: str, value: Any) -> None:
    f = _require_real(name, value)
    if not 0.0 <= f <= 1.0:
        raise G3InputValidationError(f"{name} must be within [0, 1] (got {value!r})")


def _require_str(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise G3InputValidationError(f"{name} must be a non-empty str")


@dataclass(frozen=True)
class MlBonusResult:
    score_post: float
    target_post: tuple[int, ...]
    applied: bool
    bonus: float

    def __post_init__(self) -> None:
        f = _require_real("score_post", self.score_post)
        if f < 0.0:
            raise G3InputValidationError(f"score_post must be >= 0 (got {self.score_post!r})")
        if not isinstance(self.target_post, tuple) or not all(
            isinstance(x, int) and not isinstance(x, bool) for x in self.target_post
        ):
            raise G3InputValidationError("target_post must be tuple[int, ...]")
        _require_bool("applied", self.applied)
        b = _require_real("bonus", self.bonus)
        if not 0.0 <= b <= 4.0:
            raise G3InputValidationError(f"bonus must be within [0, 4] (got {self.bonus!r})")
        if not self.applied and b != 0.0:
            raise G3InputValidationError("bonus must be 0 when applied is False")


@dataclass(frozen=True)
class BuyScoreInput:
    eval_id: str
    pred_combo: str
    uncertainty: float
    disagreement: float
    composite: float
    has_exhibition: bool
    market_gap: float
    match_index_approx: bool

    def __post_init__(self) -> None:
        _require_str("eval_id", self.eval_id)
        _require_str("pred_combo", self.pred_combo)
        _require_unit("uncertainty", self.uncertainty)
        _require_unit("disagreement", self.disagreement)
        _require_unit("composite", self.composite)
        _require_bool("has_exhibition", self.has_exhibition)
        _require_real("market_gap", self.market_gap)
        _require_bool("match_index_approx", self.match_index_approx)
