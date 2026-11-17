"""
正式G3（段階4-2）：BuyAssessment 4項目の比較判定（純関数）。

GPT最終判断（2026-11-18）・本人承認（doc 17 §1③, §2-1, §2-3）を反映:
- 比較対象＝buyscore・investment_type・kelly_fraction・skip_reason。
- skip_reason は発生層（capital／zero_amount／eval_logic／none）を区別し、
  **同層のみ比較**。層が異なれば比較不能（incomparable、一致扱いにしない）。
  Rebuild の純粋 assess は eval_logic 層（または none＝買い）しか出さない。
  Legacy の資金管理（capital_reason）・金額0円（zero_amount）は Rebuild に無い層。
- 数値比較（buyscore・kelly_fraction）：本人承認の既定＝「計算時に丸め済みの値
  （buyscore 1桁・kelly 4桁）に厳密 ==（別途の許容差なし）」。compare_policy=None は
  「未設定＝比較不能」（GPT 指示）。
- None 規則：None 対 None＝一致／None 対 文字列＝不一致／欠損・不正値＝比較不能。

結果の区分：MATCH（一致）／MISMATCH（不一致）／INCOMPARABLE（比較不能）。
レース単位の判定 judge_race：いずれかが INCOMPARABLE なら比較不能、
すべて MATCH なら一致、それ以外（MATCH と MISMATCH のみ）は不一致。

正式G2 の評価側 baseline 42ファイル・Legacy・instrument は変更しない新規ファイル。
core/__init__.py からは import しない。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

# ---- 比較結果 ----
MATCH = "match"
MISMATCH = "mismatch"
INCOMPARABLE = "incomparable"

# ---- skip_reason の発生層 ----
LAYER_NONE = "none"             # 見送りでない（買い。skip_reason=None）
LAYER_EVAL_LOGIC = "eval_logic"  # check_passthrough 由来（Rebuild と同層）
LAYER_CAPITAL = "capital"       # 資金管理（get_bet_multiplier_extended）。Legacy のみ
LAYER_ZERO_AMOUNT = "zero_amount"  # 金額0円の多重判定。Legacy のみ

# x_buyscore.py 735-740 の zero_amount_reason の目印（文字列の先頭）
_ZERO_AMOUNT_MARKER = "ベット金額が0円"


@dataclass(frozen=True)
class ComparePolicy:
    """数値比較の方針。

    本人承認の既定（DEFAULT_COMPARE_POLICY）＝追加の丸めも許容差もしない厳密 ==。
    buyscore・kelly は計算側（core.buyscore）で round(…,1)／round(…,4) 済みのため、
    ここで再度丸めない。許容差を使う場合のみ *_tol を設定する（正式規則の確定が要る）。
    """
    buyscore_round: Optional[int] = None
    kelly_round: Optional[int] = None
    buyscore_tol: Optional[float] = None
    kelly_tol: Optional[float] = None


# 本人承認（doc 17 §1③, item 1）：丸め済み値に厳密 ==（別途の許容差なし）
DEFAULT_COMPARE_POLICY = ComparePolicy()


def _is_number(v: Any) -> bool:
    return (not isinstance(v, bool)) and isinstance(v, (int, float)) and math.isfinite(float(v))


def compare_number(rebuild_v: Any, legacy_v: Any, round_: Optional[int], tol: Optional[float]) -> str:
    """数値1項目の比較。欠損・非数・非有限は比較不能。"""
    if not _is_number(rebuild_v) or not _is_number(legacy_v):
        return INCOMPARABLE
    a, b = float(rebuild_v), float(legacy_v)
    if round_ is not None:
        a, b = round(a, round_), round(b, round_)
    if tol is not None:
        return MATCH if abs(a - b) <= tol else MISMATCH
    return MATCH if a == b else MISMATCH


def compare_string(rebuild_v: Any, legacy_v: Any) -> str:
    """文字列1項目（investment_type）の比較。どちらかが None・非文字列は比較不能。

    注（監査 B-1）：仕様④の「None対None=一致」は skip_reason 専用（None=買いの意味）。
    investment_type は実コード上つねに str（x_buyscore.investment_type／core はつねに str を返す。
    entrypoint の recomputed が None になるのは legacy buyscore=None のときだけで、そこは既に
    OUTCOME_PREDICTION_DRIFT で除外済み）。欠損は保守側で比較不能とする。
    """
    if not isinstance(rebuild_v, str) or not isinstance(legacy_v, str):
        return INCOMPARABLE
    return MATCH if rebuild_v == legacy_v else MISMATCH


def classify_legacy_skip_layer(passthrough: Any, passthrough_reason: Any, capital_reason: Any) -> str:
    """Legacy `_evaluate_bets`（apply_buyscore）の戻りから skip_reason の発生層を判定する。

    x_buyscore.apply_buyscore（645-740行目）:
      - 資金管理で multiplier==0 → passthrough_reason = capital_reason（688-697行目）
      - 見送り判定 check_passthrough → passthrough_reason = pass_reason（700-710行目）
      - 金額0円の全滅 → passthrough_reason = zero_amount_reason（735-740行目）
    """
    if not passthrough:
        return LAYER_NONE
    r = passthrough_reason
    if r is None:
        # passthrough=True なのに理由が無い（想定外）。純粋層とは断定できない
        return LAYER_CAPITAL  # 保守側：Rebuild の eval_logic/none とは別層＝比較不能になる
    if capital_reason is not None and r == capital_reason:
        return LAYER_CAPITAL
    if isinstance(r, str) and r.startswith(_ZERO_AMOUNT_MARKER):
        return LAYER_ZERO_AMOUNT
    return LAYER_EVAL_LOGIC


def _rebuild_skip_layer(rebuild_skip: Any) -> str:
    # Rebuild の純粋 assess は eval_logic 層（見送り）か none（買い）しか出さない
    return LAYER_NONE if rebuild_skip is None else LAYER_EVAL_LOGIC


def compare_skip_reason(rebuild_skip: Any, legacy_layer: str, legacy_reason: Any,
                        normalize: Any = None) -> str:
    """skip_reason の比較。同層（none／eval_logic）のみ比較、資金管理・金額0円層は比較不能。

    normalize: 正規化関数（既定＝恒等。本人承認 doc 17 §1③ item4＝同層内で文字列厳密一致、
    表記ゆれ吸収はしない）。
    """
    norm = normalize or (lambda s: s)
    rebuild_layer = _rebuild_skip_layer(rebuild_skip)
    # Legacy が資金管理・金額0円層なら、Rebuild の層（none/eval_logic）と異なる → 比較不能
    if legacy_layer in (LAYER_CAPITAL, LAYER_ZERO_AMOUNT):
        return INCOMPARABLE
    # ここから Legacy は none か eval_logic。層一致のときだけ値を比べる
    rebuild_none = rebuild_layer == LAYER_NONE
    legacy_none = legacy_layer == LAYER_NONE
    if rebuild_none and legacy_none:
        return MATCH                      # None 対 None
    if rebuild_none != legacy_none:
        return MISMATCH                   # None 対 文字列
    # 両方 eval_logic（文字列）。同層・正規化後の厳密一致
    if not isinstance(rebuild_skip, str):
        return INCOMPARABLE
    lr = legacy_reason
    if not isinstance(lr, str):
        return INCOMPARABLE
    return MATCH if norm(rebuild_skip) == norm(lr) else MISMATCH


def compare_four_items(
    *,
    rebuild_buyscore: Any, legacy_buyscore: Any,
    rebuild_kelly: Any, legacy_kelly: Any,
    rebuild_investment_type: Any, legacy_investment_type: Any,
    rebuild_skip_reason: Any,
    legacy_passthrough: Any, legacy_passthrough_reason: Any, legacy_capital_reason: Any,
    policy: Optional[ComparePolicy] = DEFAULT_COMPARE_POLICY,
    normalize: Any = None,
) -> dict[str, Any]:
    """4項目の項目別比較＋レース単位の判定を返す。

    policy=None は「compare_policy 未設定」＝数値は比較不能（GPT 指示）。
    """
    if policy is None:
        bs = ks = INCOMPARABLE
        policy_set = False
    else:
        bs = compare_number(rebuild_buyscore, legacy_buyscore, policy.buyscore_round, policy.buyscore_tol)
        ks = compare_number(rebuild_kelly, legacy_kelly, policy.kelly_round, policy.kelly_tol)
        policy_set = True
    it = compare_string(rebuild_investment_type, legacy_investment_type)
    legacy_layer = classify_legacy_skip_layer(legacy_passthrough, legacy_passthrough_reason,
                                              legacy_capital_reason)
    sr = compare_skip_reason(rebuild_skip_reason, legacy_layer, legacy_passthrough_reason, normalize)
    items = {"buyscore": bs, "kelly_fraction": ks, "investment_type": it, "skip_reason": sr}
    return {
        "items": items,
        "race_judge": judge_race(items),
        "skip_layer": {"legacy": legacy_layer, "rebuild": _rebuild_skip_layer(rebuild_skip_reason)},
        # ③ 記録（GPT確定）：元の文字列・発生層・比較結果を残す
        "skip_detail": {
            "legacy_layer": legacy_layer,
            "rebuild_layer": _rebuild_skip_layer(rebuild_skip_reason),
            "legacy_reason": legacy_passthrough_reason,
            # 監査 B-b：capital/zero 層のとき元の資金管理理由も残す（③「元文字列記録」の厳格化）
            "legacy_capital_reason": legacy_capital_reason,
            "rebuild_reason": rebuild_skip_reason,
            "result": sr,
        },
        "compare_policy_set": policy_set,
    }


def judge_race(items: dict[str, str]) -> str:
    """レース単位の判定：INCOMPARABLE が1つでもあれば比較不能、全 MATCH なら一致、他は不一致。"""
    vals = list(items.values())
    if any(v == INCOMPARABLE for v in vals):
        return INCOMPARABLE
    if all(v == MATCH for v in vals):
        return MATCH
    return MISMATCH
