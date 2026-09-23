"""S4（本番運用ポリシー）が使う設定の読み込み（Phase0.5 ⑮ の最小実装）。

設計書の該当箇所:
  - ⑮ config/pipeline.json: 「締切前分数」/ config/delivery.json: 「1日投稿数上限」
  - 「起動時にスキーマ検証し、不正ならConfigErrorで即終了（黙って既定値にしない）」
  - 設定JSONは `_version` 単調増加

本モジュールは S4 が必要とする2項目だけを読む。⑮の未決定項目には触れず、
既定値による補完もしない（欠落・型不正は ConfigError）。

置き場所: storage/exceptions.py は Freeze 対象（Step5-0 ⑧）のため
ConfigError はここで定義する。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

CONFIG_DIR = "config"
PIPELINE_CONFIG = "pipeline.json"
DELIVERY_CONFIG = "delivery.json"

# 設計書⑮の項目名をそのままキーとして用いる（独自名称を作らない）
KEY_VERSION = "_version"
KEY_DEADLINE_MINUTES = "締切前分数"
KEY_DAILY_NOTIFICATION_LIMIT = "1日投稿数上限"


class ConfigError(Exception):
    """設定の欠落・型不正・スキーマ不一致（黙って既定値にしない）。"""


def _load_json(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file is not valid JSON: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config root must be an object: {path}")
    return data


def _require_version(data: Mapping[str, Any], path: Path) -> int:
    value = data.get(KEY_VERSION)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ConfigError(f"{KEY_VERSION} must be a positive int: {path}")
    return value


def _require_positive_int(data: Mapping[str, Any], key: str, path: Path) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{key} must be a positive int: {path}")
    return value


def load_deadline_minutes(config_dir: str = CONFIG_DIR) -> int:
    """config/pipeline.json の「締切前分数」を読む（検証付き）。"""
    path = Path(config_dir) / PIPELINE_CONFIG
    data = _load_json(path)
    _require_version(data, path)
    return _require_positive_int(data, KEY_DEADLINE_MINUTES, path)


def load_daily_notification_limit(config_dir: str = CONFIG_DIR) -> int:
    """config/delivery.json の「1日投稿数上限」を読む（検証付き）。"""
    path = Path(config_dir) / DELIVERY_CONFIG
    data = _load_json(path)
    _require_version(data, path)
    return _require_positive_int(data, KEY_DAILY_NOTIFICATION_LIMIT, path)
