#!/usr/bin/env python3
"""
ops_check.py  ── 運用チェックレポート

Git管理・.gitignore・CSVスキーマ・マイグレーション・バックアップ・
整合性チェックが、本プロジェクトの運用ルール通りに機能しているかを
まとめて確認し、レポートとして表示する。

CI や定期実行、あるいは人間が手動で「今の状態は健全か」を確認する
ためのエントリポイント。個々のチェックの詳細ロジックは
csv_schema.py / schema_version.py / migration.py / data_integrity.py
に委譲し、本ファイルは「まとめて呼んで表示する」役割に徹する。
"""

from __future__ import annotations

import os
import subprocess
import sys

from csv_schema import CURRENT_SCHEMA_VERSION
from schema_version import check_schema, HIT_RECORD_CSV
from data_integrity import validate_csv_file

# Git管理対象外にすべきファイル（.gitignoreに書かれているべきパターン）
_EXPECTED_IGNORED = [
    "hit_record.csv",
    "motor_history.csv",
    "hit_record_backup_*.csv",
    "logs/",
    "*.log",
    "__pycache__/",
    "*.pyc",
    "venv/",
    ".venv/",
    "*.tmp",
    "*.bak",
    ".DS_Store",
    "Thumbs.db",
]


def check_gitignore(path: str = ".gitignore") -> tuple[bool, str]:
    """.gitignore が存在し、必須パターンを含んでいるかを確認する。"""
    if not os.path.exists(path):
        return False, ".gitignore が存在しません"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    missing = [p for p in _EXPECTED_IGNORED if p not in content]
    if missing:
        return False, f"必須パターン不足: {missing}"
    return True, "必須パターンすべて含まれています"


def check_git_tracked_runtime_data() -> tuple[bool, str]:
    """
    運用データ（hit_record.csv 等）が誤ってGit管理下に入っていないかを確認する。
    Gitが使えない環境（.git がない等）ではスキップ扱いにする。
    """
    if not os.path.isdir(".git"):
        return True, "Gitリポジトリではないためスキップ"

    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except Exception as e:
        return True, f"git ls-files 実行不可のためスキップ ({e})"

    tracked = set(result.stdout.splitlines())
    leaked = [f for f in ("hit_record.csv", "motor_history.csv") if f in tracked]
    if leaked:
        return False, f"運用データがGit管理下にあります: {leaked}（git rm --cached が必要）"
    return True, "運用データはGit管理対象外です"


def check_migration_backup() -> tuple[bool, str]:
    """
    バックアップの仕組み自体が機能する状態か（migration.py が存在し、
    make_backup が呼び出し可能か）を確認する。実際にバックアップを
    作成することはしない（副作用を避けるため）。
    """
    try:
        from migration import make_backup  # noqa: F401
        return True, "バックアップ機構は利用可能です"
    except Exception as e:
        return False, f"migration.py の読み込みに失敗: {e}"


def run_all_checks() -> dict:
    """全チェックを実行し、結果をまとめて返す。"""
    results = {}

    # ── Git管理対象 ──────────────────────────────────────
    ok, msg = check_git_tracked_runtime_data()
    results["Git管理対象"] = (ok, msg)

    # ── .gitignore ───────────────────────────────────────
    ok, msg = check_gitignore()
    results[".gitignore"] = (ok, msg)

    # ── CSV Version ──────────────────────────────────────
    results["CSV Version"] = (True, f"v{CURRENT_SCHEMA_VERSION}（期待バージョン）")

    # ── Schema Check（実ファイルと期待バージョンの一致） ──
    schema_status = check_schema(HIT_RECORD_CSV)
    schema_ok = not schema_status["needs_migration"]
    results["Schema Check"] = (schema_ok, schema_status["reason"])

    # ── Migration（機構自体が使えるか） ────────────────────
    ok, msg = check_migration_backup()
    results["Migration"] = (ok, msg)

    # ── Backup（機構自体が使えるか。上のMigrationと同じ判定元）──
    results["Backup"] = (ok, "make_backup() 呼び出し可能" if ok else msg)

    # ── Data Integrity ───────────────────────────────────
    integrity = validate_csv_file(HIT_RECORD_CSV)
    results["Data Integrity"] = (
        integrity["ok"],
        "異常なし" if integrity["ok"] else "異常を検出（ログを確認してください）",
    )

    return results


def print_ops_report(results: dict) -> bool:
    """レポートを表示し、全項目OKかどうかを返す。"""
    print("=" * 64)
    print(" 運用チェックレポート".center(60))
    print("=" * 64)
    all_ok = True
    for label, (ok, msg) in results.items():
        status = "OK" if ok else "NG"
        all_ok = all_ok and ok
        print(f" {label:<16} {status:<4} {msg}")
    print("=" * 64)
    print(f" 総合判定: {'OK（すべて正常）' if all_ok else 'NG（要対応項目あり）'}")
    print("=" * 64)
    return all_ok


if __name__ == "__main__":
    _results = run_all_checks()
    _all_ok = print_ops_report(_results)
    sys.exit(0 if _all_ok else 1)
