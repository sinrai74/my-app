"""正式G2：完了目録（C-6・§6.6 の 3）。

完了目録は実行の最後に書き、"complete": true を含める。
印の無い目録、または目録の無い実行は、記録処理が途中で止まったものとして扱う。
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

MANIFEST_NAME = "manifest.json"


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: str, run_info: dict[str, Any], processed_eval_ids: list[str],
                   target_eval_ids: list[str], complete: bool) -> str:
    files = {}
    for name in sorted(os.listdir(out_dir)):
        if name == MANIFEST_NAME:
            continue
        path = os.path.join(out_dir, name)
        if os.path.isfile(path):
            files[name] = {"sha256": file_sha256(path), "bytes": os.path.getsize(path)}
    manifest = {
        "run": run_info,
        "files": files,
        "target_eval_ids": target_eval_ids,
        "processed_eval_ids": processed_eval_ids,
        "complete": bool(complete),
    }
    path = os.path.join(out_dir, MANIFEST_NAME)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        f.write("\n")
    return path
