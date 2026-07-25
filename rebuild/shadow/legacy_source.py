"""
Legacy比較データ抽出（Step6-1）: 既存出力物を読み取るだけ。

方針（Step6-1レビュー確定）:
  Legacyへフック・print・log・callback・Serializerを一切追加しない。
  既存の出力物のみを読み取る。優先順位:
    1. sent_*.txt   （予測配信ログ。1行=1レースのJSON）
    2. hit_record.csv（結果照合後の記録）
    3. HTML / CSV   （Output比較用・byte比較はOutputPipeline側で実施）

本モジュールは「読むだけ」。Legacyの中間値を計算・再現しない。
sent_*.txtに存在するフィールドのみを比較材料として提供する。

提供できる比較ビュー（Step6-2 取得元確定後）:
  race_view / feature_view / evaluation_view / prediction_view / buy_view。
  本モジュールがsent_*.txtから提供する段階:
    Race / FeatureSet(feat_*のみ) / RaceEvaluation / Prediction /
    BuyAssessment(sentに存在する項目のみ)
  RenderResult は Legacy生成の実HTMLをbyte読み取り（shadow/legacy_output.py）、
  NotificationRequest は build_message() の戻り値(subject, body)をラップ取得
  （shadow/legacy_notification.py）する。いずれもLegacyは読む/呼ぶだけ。

sent_*.txt に実在するfeat_*（実データ255行の集計で確認済み・7種）:
  feat_win_rate / feat_motor / feat_avg_st / feat_racer_class /
  feat_course_st_1c / feat_course_rank_1c / feat_danger_breakdown
  ※ 90件(35%)の行にのみ存在（記録開始時期による）。存在する行のみ比較する。
  ※ 上記以外のFeatureはLegacy取得元なし（比較対象外として記録）。
  ※ buyscoreはsent_*.txtに0件（記録なし）＝buy_viewは空になり比較対象外。

依存: shadow → 標準ライブラリのみ（models/coreへも依存しない読み取り専用）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


class LegacySentRecord:
    """sent_*.txt の1行（1レース）から読み取ったLegacy値。

    生JSONのうち、Shadow比較に使うフィールドだけを保持する。
    値の加工・補正はしない（読み取ったそのまま）。
    """

    __slots__ = ("eval_id", "raw")

    def __init__(self, eval_id: str, raw: dict) -> None:
        self.eval_id = eval_id
        self.raw = raw

    # --- 比較対象ごとのビュー（読み取ったフィールドを素直に返すだけ） ---

    def prediction_view(self) -> dict:
        """Prediction相当（combo/prob/ev/odds）。sentにある値のみ。"""
        return {
            "pred_combo": self.raw.get("combo"),
            "pred_prob": self.raw.get("prob"),
            "pred_ev": self.raw.get("ev"),
            "pred_odds": self.raw.get("odds"),
        }

    def buy_view(self) -> dict:
        """BuyAssessment相当のうちsentに存在する項目のみ。

        sent_*.txtにはbuyscore/investment_type/kelly_fraction/skip_reasonが
        常には無い（記録時期により異なる）。存在するキーだけ返す
        （欠損キーは比較側で扱う。ここで補完・仮値はしない）。
        """
        view: dict[str, Any] = {}
        for key in ("buyscore", "investment_type", "kelly_fraction", "skip_reason"):
            if key in self.raw:
                view[key] = self.raw[key]
        return view

    # sent_*.txt に実在するfeat_*キー（実データ集計で確認済み・7種）。
    # ここに無いFeatureは「Legacy取得元なし」として比較対象外になる。
    FEATURE_KEYS: tuple[str, ...] = (
        "feat_win_rate", "feat_motor", "feat_avg_st", "feat_racer_class",
        "feat_course_st_1c", "feat_course_rank_1c", "feat_danger_breakdown",
    )

    def feature_view(self) -> dict:
        """FeatureSet相当のうちsent_*.txtに存在するfeat_*のみ。

        FeatureBuilderの再実行はしない（二重計算・責務逸脱の回避）。
        存在するキーだけを返し、欠損キーは補完しない。
        """
        return {
            key: self.raw[key] for key in self.FEATURE_KEYS if key in self.raw
        }

    def missing_feature_keys(self) -> tuple[str, ...]:
        """Legacy取得元が無かったfeat_*キー（記録用）。"""
        return tuple(key for key in self.FEATURE_KEYS if key not in self.raw)

    def race_view(self) -> dict:
        """Race相当（venue_num/race/night等）。sentにある値のみ。"""
        return {
            "venue_num": self.raw.get("venue_num"),
            "race_number": self.raw.get("race"),
            "is_night": bool(self.raw.get("night")) if "night" in self.raw else None,
            "venue_name": self.raw.get("venue"),
        }

    def evaluation_view(self) -> dict:
        """RaceEvaluation相当のうちsentに存在する項目のみ。"""
        view: dict[str, Any] = {}
        if "upset_score" in self.raw:
            view["upset_score"] = self.raw["upset_score"]
        if "race_type" in self.raw:
            view["race_type"] = self.raw["race_type"]
        if "danger_score_v3" in self.raw:
            view["danger_score"] = self.raw["danger_score_v3"]
        return view


def load_sent_records(sent_path: str) -> dict[str, LegacySentRecord]:
    """sent_*.txt を読み、eval_id -> LegacySentRecord の辞書を返す。

    __no_bets_evaluated__ 等のメタ行（keyがレースIDでない）はスキップする。
    ファイルが無い/空なら空辞書（Shadow側で「Legacy値なし」として扱う）。
    """
    path = Path(sent_path)
    if not path.exists():
        return {}
    records: dict[str, LegacySentRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = raw.get("key", "")
        # レースID形式（YYYYMMDD_venue_race[...]）のみ対象。メタ行は除外
        parts = key.split("_")
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        eval_id = f"{parts[0]}_{int(parts[1]):02d}_{int(parts[2]):02d}"
        records[eval_id] = LegacySentRecord(eval_id=eval_id, raw=raw)
    return records


def get_legacy_record(
    sent_path: str, eval_id: str
) -> Optional[LegacySentRecord]:
    """特定eval_idのLegacy値を取得（無ければNone）。"""
    return load_sent_records(sent_path).get(eval_id)
