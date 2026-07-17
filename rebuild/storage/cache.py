"""
CacheStore: 当日限りの一時データキャッシュ（schema_version・date検証まで）。

設計書 v1.1.8 ⑥（同日内の一時データはActionsキャッシュ。当日以外は破棄）、
Step3計画書 §2 に基づく。

責務の厳守（Step3-4指示）:
- 本ストアは「保存」と「schema_version・dateの検証」までを担当する
- 「キャッシュを使うか、作り直すか」の利用判断は上位層（services/pipelines）へ委譲する。
  本ストアは検証に通らないキャッシュをNoneとして返すだけで、再生成はしない

保存形式: 単一JSONファイル。ラッパに schema_version と cached_date（YYYYMMDD, JST）を
持ち、payloadを内包する。読み出し時に現在の対象日付・スキーマ版と照合する。

メタデータ拡張余地（Should-2）: ラッパは dict 構造であり、将来 generator_version 等の
メタデータを追加できる。save() に任意メタデータを受け取る余地を残すため、
本実装では payload とラッパ層を分離している（payload内に混ぜない）。

依存: 標準ライブラリのみ。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from storage.exceptions import ParseError

CACHE_SCHEMA_VERSION: int = 1


class CacheStore:
    """当日限りのキャッシュを1ファイルで管理する。"""

    def __init__(self, json_path: Path, schema_version: int = CACHE_SCHEMA_VERSION) -> None:
        self._path = Path(json_path)
        self._schema_version = schema_version

    def save(self, payload: dict[str, Any], cached_date: str) -> None:
        """payloadを当日(cached_date)のキャッシュとして保存する（全置換）。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        wrapper = {
            "schema_version": self._schema_version,
            "cached_date": cached_date,
            "payload": payload,
        }
        with open(self._path, "w", encoding="utf-8", newline="") as f:
            json.dump(wrapper, f, ensure_ascii=False)

    def load_if_valid(self, current_date: str) -> Optional[dict[str, Any]]:
        """検証に通ればpayloadを返し、通らなければNoneを返す。

        検証項目（本ストアの責務範囲）:
        - ファイルが存在すること
        - schema_versionが一致すること（版差 -> None）
        - cached_dateがcurrent_dateと一致すること（当日以外 -> None）

        Noneを返した場合の再生成・利用可否の判断は上位層に委ねる（本ストアはしない）。
        破損JSON・必須キー欠落はParseError（サイレント失敗の禁止）。
        """
        if not self._path.exists():
            return None
        with open(self._path, encoding="utf-8") as f:
            try:
                wrapper = json.load(f)
            except json.JSONDecodeError as exc:
                raise ParseError(f"{self._path.name}: broken cache JSON: {exc}") from exc
        for key in ("schema_version", "cached_date", "payload"):
            if key not in wrapper:
                raise ParseError(f"{self._path.name}: cache missing key {key!r}")
        if wrapper["schema_version"] != self._schema_version:
            return None
        if wrapper["cached_date"] != current_date:
            return None
        return wrapper["payload"]
