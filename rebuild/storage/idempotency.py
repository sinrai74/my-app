"""
IdempotencyStore: 冪等性記録の保存・照会。

設計書 v1.1.8 ⑥（冪等性の一元化: 現行の sent_*.txt 等3方式を
(channel, message_key) の1方式へ統一）、Step3計画書 §2 に基づく。

責務の厳守（Step3-4指示）:
- 本ストアは「記録済みか否かの保存・照会」のみを担当する
- 「送信してよいか」「復元不能時は送信済み扱いに倒す」等のビジネス判断は
  実装しない（それはL5＝Step4のnotifier/servicesの責務）

キー設計（設計書⑥）:
- (channel, message_key) の組で一意。channelは通知種別（例: "arashi_mail"）、
  message_keyは対象の一意識別（例: "{race_date}:{eval_id}"）
- append-only。一度記録したキーは削除・変更しない（ファイル追記のみ）

将来拡張の余地（Should-1・今回は実装しない）:
- (channel, message_key) は将来 MessageIdentity 値オブジェクトへまとめられる想定。
  本ストアの公開シグネチャは channel/message_key の2引数を保つが、内部表現を
  値オブジェクト化してもAPIを壊さないよう、2値の組を単位として扱っている。

依存: 標準ライブラリのみ。ファイルI/Oは本ストアが担当（Repository同様のI/O層）。
"""

from __future__ import annotations

import json
from pathlib import Path

from storage.exceptions import ParseError


class IdempotencyStore:
    """(channel, message_key) の記録を1つのJSONLファイルで管理する。"""

    def __init__(self, jsonl_path: Path) -> None:
        self._path = Path(jsonl_path)

    def is_recorded(self, channel: str, message_key: str) -> bool:
        """指定キーが記録済みかを返す（照会のみ。可否判断はしない）。"""
        target = (channel, message_key)
        for entry in self._load():
            if (entry["channel"], entry["message_key"]) == target:
                return True
        return False

    def record(self, channel: str, message_key: str) -> None:
        """キーを1件追記する（append-only）。

        既に記録済みの場合は何もしない（冪等。重複追記を避ける）。
        「記録済みなら送信しない」等の判断は行わず、記録の有無だけを管理する。
        """
        if self.is_recorded(channel, message_key):
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"channel": channel, "message_key": message_key}, ensure_ascii=False
        )
        with open(self._path, "a", encoding="utf-8", newline="") as f:
            f.write(line + "\n")

    def _load(self) -> list[dict[str, str]]:
        if not self._path.exists():
            return []
        entries: list[dict[str, str]] = []
        with open(self._path, encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                text = line.strip()
                if text == "":
                    continue
                try:
                    data = json.loads(text)
                except json.JSONDecodeError as exc:
                    raise ParseError(
                        f"{self._path.name} line {line_no}: broken JSON: {exc}"
                    ) from exc
                if "channel" not in data or "message_key" not in data:
                    raise ParseError(
                        f"{self._path.name} line {line_no}: "
                        "missing 'channel' or 'message_key'"
                    )
                entries.append(data)
        return entries
