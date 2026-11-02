"""正式G2：記録の書き出し（判断D 11・D5・§6.6）。

書き込むのは出力ディレクトリ（artifact になるもの）だけ。
レースごとの記録は1行1件の JSON（races.jsonl）で、書くたびに flush する。
"""

from __future__ import annotations

import json
import os
from typing import Any


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=True, default=repr)


class RecordWriter:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self._races_path = os.path.join(out_dir, "races.jsonl")
        self._races = open(self._races_path, "w", encoding="utf-8")
        self.written_eval_ids: list[str] = []

    def write_bytes(self, name: str, data: bytes) -> str:
        path = os.path.join(self.out_dir, name)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def write_json(self, name: str, obj: Any) -> str:
        path = os.path.join(self.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write(_dump(obj))
            f.write("\n")
        return path

    def append_race(self, record: dict[str, Any]) -> None:
        self._races.write(_dump(record))
        self._races.write("\n")
        self._races.flush()
        self.written_eval_ids.append(record["eval_id"])

    def close(self) -> None:
        if not self._races.closed:
            self._races.close()
