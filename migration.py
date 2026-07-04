#!/usr/bin/env python3
"""
migration.py  ── hit_record.csv スキーママイグレーション基盤

旧スキーマの hit_record.csv を、csv_schema.CURRENT_SCHEMA_VERSION が示す
最新スキーマへ安全に変換する。既存データは失わない。

────────────────────────────────────────────────────────────
特徴
────────────────────────────────────────────────────────────
・変換前に自動バックアップ（hit_record_backup_YYYYMMDD_HHMMSS.csv）
・段階マイグレーション: v1→v2→v3… と1段ずつ確実に適用（チェーン）
・原子的置換: 一時ファイルに書いてから rename するので途中失敗で壊れない
・完了レポート: 総件数・旧/新スキーマ・追加列・補完件数を表示

────────────────────────────────────────────────────────────
将来スキーマを増やすときは（例: v2 → v3）
────────────────────────────────────────────────────────────
1. csv_schema.py に v3 の列定義と新規列の初期値を追加。
2. このファイルに migrate_v2_to_v3(rows) を実装。
3. MIGRATIONS に (2, 3): migrate_v2_to_v3 を1行追加。
   → これだけで v1 のファイルも v1→v2→v3 と自動で最新まで上がる。

コマンドラインから:
    python migration.py            # 必要なら自動マイグレーション
    python migration.py --check    # 判定のみ（変換しない）
    python migration.py --dry-run  # 変換内容を表示するが書き込まない
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone

from csv_schema import (
    CURRENT_SCHEMA_VERSION,
    SCHEMA_COLUMNS,
    NEW_COLUMN_DEFAULTS,
    get_columns,
)
from schema_version import (
    HIT_RECORD_CSV,
    detect_current_file_version,
    write_recorded_version,
    check_schema,
)
from data_integrity import validate_csv_file

JST = timezone(timedelta(hours=9))


# ════════════════════════════════════════════════════════════
# 個別マイグレーション関数（1段ぶんの変換）
# ════════════════════════════════════════════════════════════
# 各関数は「旧バージョンの行リスト(list[dict])」を受け取り、
# 「新バージョンの行リスト(list[dict])」を返す。
# 行の追加・削除はせず、列の追加/初期値補完のみを行うのが基本。

def migrate_v1_to_v2(rows: list[dict]) -> list[dict]:
    """
    Version1(22列) → Version2(34列)。
    購入判定分離(BuyScore)列 + Phase2学習用特徴量列を追加し、
    旧データには csv_schema.NEW_COLUMN_DEFAULTS の初期値を補完する。
    """
    v2_cols = get_columns(2)
    new_cols = [c for c in v2_cols if c not in get_columns(1)]

    migrated = []
    for row in rows:
        new_row = dict(row)  # 既存の値はすべて保持
        for col in new_cols:
            # 既に値が入っていればそれを尊重し、なければ初期値で補完
            if not new_row.get(col):
                new_row[col] = NEW_COLUMN_DEFAULTS.get(col, "")
        migrated.append(new_row)
    return migrated


# ════════════════════════════════════════════════════════════
# マイグレーション登録表（チェーン）
# ════════════════════════════════════════════════════════════
# (from_version, to_version): 変換関数
# 連続する 1段ずつ を登録する。飛び級は書かない（v1→v3 は v1→v2→v3 で表現）。
MIGRATIONS = {
    (1, 2): migrate_v1_to_v2,
}


# ════════════════════════════════════════════════════════════
# バックアップ
# ════════════════════════════════════════════════════════════

def make_backup(csv_file: str = HIT_RECORD_CSV) -> str:
    """
    hit_record_backup_YYYYMMDD_HHMMSS.csv を作成し、パスを返す。
    変換失敗時でも元データへ戻せるようにするための保険。
    """
    ts = datetime.now(JST).strftime("%Y%m%d_%H%M%S")
    base = os.path.splitext(os.path.basename(csv_file))[0]
    backup_name = f"{base}_backup_{ts}.csv"
    backup_path = os.path.join(os.path.dirname(os.path.abspath(csv_file)), backup_name)
    shutil.copy2(csv_file, backup_path)
    return backup_path


# ════════════════════════════════════════════════════════════
# 読み込み / 書き込み
# ════════════════════════════════════════════════════════════

def _read_rows(csv_file: str) -> list[dict]:
    """CSVを list[dict] で読み込む。"""
    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows_atomic(csv_file: str, columns: list[str], rows: list[dict]) -> None:
    """
    指定列順でCSVを原子的に書き出す。
    同ディレクトリの一時ファイルに書いてから rename することで、
    書き込み途中でプロセスが落ちても元ファイルは壊れない。
    未知の余分な列があっても落ちないよう extrasaction='ignore' を使う。
    """
    dir_ = os.path.dirname(os.path.abspath(csv_file)) or "."
    fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=dir_)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                # 欠けている列は空文字で埋める（DictWriterは欠損キーで例外を出すため）
                writer.writerow({c: row.get(c, "") for c in columns})
        os.replace(tmp_path, csv_file)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# ════════════════════════════════════════════════════════════
# マイグレーション実行
# ════════════════════════════════════════════════════════════

def _build_migration_path(from_v: int, to_v: int) -> list[tuple[int, int]]:
    """
    from_v から to_v まで、1段ずつのマイグレーション経路を組み立てる。
    途中の変換が未登録なら例外を投げる。
    """
    path = []
    v = from_v
    while v < to_v:
        step = (v, v + 1)
        if step not in MIGRATIONS:
            raise RuntimeError(f"マイグレーション {step[0]}→{step[1]} が未登録です")
        path.append(step)
        v += 1
    return path


def run_migration(
    csv_file: str = HIT_RECORD_CSV,
    target_version: int = CURRENT_SCHEMA_VERSION,
    dry_run: bool = False,
) -> dict:
    """
    csv_file を target_version まで段階マイグレーションする。

    戻り値: 完了レポート(dict)
    """
    status = check_schema(csv_file)

    if not status["exists"]:
        return {"performed": False, "reason": status["reason"]}

    from_v = status["current_version"]
    if from_v is None:
        return {
            "performed": False,
            "reason": "現在のスキーマバージョンを判定できないため、自動変換を中止しました。",
        }

    if from_v >= target_version:
        return {
            "performed": False,
            "reason": f"変換不要（現在 v{from_v} ≧ 目標 v{target_version}）",
        }

    # ── 変換経路を確定 ───────────────────────────────────────
    path = _build_migration_path(from_v, target_version)

    # ── 元データ読み込み ─────────────────────────────────────
    original_rows = _read_rows(csv_file)
    total_records = len(original_rows)
    old_columns = get_columns(from_v)
    new_columns = get_columns(target_version)
    added_columns = [c for c in new_columns if c not in old_columns]

    # ── 補完件数カウント用（変換前に「その列が空だった行数」を数える）──
    fill_counts = {c: 0 for c in added_columns}
    for row in original_rows:
        for c in added_columns:
            if not row.get(c):
                fill_counts[c] += 1

    # ── チェーン適用 ─────────────────────────────────────────
    rows = original_rows
    for (a, b) in path:
        rows = MIGRATIONS[(a, b)](rows)

    report = {
        "performed": True,
        "dry_run": dry_run,
        "total_records": total_records,
        "old_version": from_v,
        "new_version": target_version,
        "old_columns": old_columns,
        "new_columns": new_columns,
        "added_columns": added_columns,
        "fill_counts": fill_counts,
        "backup_path": None,
    }

    if dry_run:
        report["reason"] = "dry-run のため書き込みはしていません。"
        return report

    # ── バックアップ → 書き込み → バージョン記録 ─────────────
    backup_path = make_backup(csv_file)
    report["backup_path"] = backup_path
    try:
        _write_rows_atomic(csv_file, new_columns, rows)
        write_recorded_version(target_version)
    except Exception as e:
        # 書き込み失敗時はバックアップから復元
        shutil.copy2(backup_path, csv_file)
        report["performed"] = False
        report["reason"] = f"変換失敗: {e}（バックアップから復元しました）"
        return report

    # ── 変換後の整合性確認 ───────────────────────────────────
    # 書き込んだ直後のファイルを読み直し、ヘッダー・列数・型に異常がないか
    # 検査する。異常があれば、たとえ書き込み自体は成功していてもバック
    # アップから復元し、失敗として報告する（サイレント破損を防ぐ）。
    integrity = validate_csv_file(csv_file, expected_version=target_version)
    report["integrity_check"] = integrity
    if not integrity["ok"]:
        shutil.copy2(backup_path, csv_file)
        report["performed"] = False
        report["reason"] = "変換後の整合性チェックで異常を検出したため、バックアップから復元しました。"
        return report

    report["reason"] = "マイグレーション成功。"
    return report


# ════════════════════════════════════════════════════════════
# 起動時フック（他モジュールから呼ぶ用）
# ════════════════════════════════════════════════════════════

def ensure_schema(csv_file: str = HIT_RECORD_CSV, auto: bool = True) -> dict:
    """
    プログラム起動時に呼び出す想定。以下を1回のチェックでまとめて行う:
      1. CSV存在確認
      2. ヘッダー確認（既知のスキーマと一致するか）
      3. schema_version確認（期待バージョンと一致するか）
      4. （一致 or 変換成功時）変換後の整合性確認

    動作:
      - 最新スキーマなら何もしない（整合性チェックのみ実施）。
      - 古いスキーマで auto=True なら自動マイグレーションを実行する。
      - バージョンを判定できない「未知のヘッダー」の場合は、
        自動変換の対象外とし fatal=True を返す。
        → 呼び出し側はこの場合サイレントに動作を続けず、
          実行を停止してエラーメッセージを表示すること。

    戻り値は check_schema() の結果に "migration"（実行時）と
    "fatal"（起動停止すべきか）を足したもの。
    """
    status = check_schema(csv_file)
    status["fatal"] = False

    # ヘッダーが最新なのにバージョン記録ファイルがない場合、記録だけ補う
    if not status["needs_migration"] and status["exists"] \
            and status["current_version"] == CURRENT_SCHEMA_VERSION:
        write_recorded_version(CURRENT_SCHEMA_VERSION)
        status["integrity_check"] = validate_csv_file(csv_file)
        if not status["integrity_check"]["ok"]:
            status["fatal"] = True
            status["reason"] = "スキーマは最新ですが、データ整合性チェックで異常を検出しました。"
        return status

    if status["needs_migration"]:
        if status["current_version"] is None:
            # ヘッダーが既知のどのバージョンとも一致せず、記録ファイルもない
            # ＝ どう変換すべきか機械的に判断できない状態。
            # サイレントに動作を続けるのは禁止なので、自動変換せず fatal を立てる。
            status["fatal"] = True
            return status
        if auto:
            status["migration"] = run_migration(csv_file)
            if not status["migration"]["performed"] and status["current_version"] != CURRENT_SCHEMA_VERSION:
                # マイグレーションを試みたが失敗した（整合性チェック含む）
                status["fatal"] = True

    return status


# ════════════════════════════════════════════════════════════
# レポート表示
# ════════════════════════════════════════════════════════════

def print_report(report: dict) -> None:
    """完了確認レポートを人間が読みやすい形で表示する。"""
    print("=" * 60)
    print(" hit_record.csv マイグレーション結果")
    print("=" * 60)

    if not report.get("performed"):
        print(f" 実行: なし")
        print(f" 理由: {report.get('reason', '')}")
        print("=" * 60)
        return

    print(f" 実行           : {'DRY-RUN（書き込みなし）' if report.get('dry_run') else '完了'}")
    print(f" 総レコード数   : {report['total_records']} 件")
    print(f" 旧スキーマ     : v{report['old_version']}（{len(report['old_columns'])}列）")
    print(f" 新スキーマ     : v{report['new_version']}（{len(report['new_columns'])}列）")
    print(f" 追加した列     : {len(report['added_columns'])} 列")
    for c in report["added_columns"]:
        default = NEW_COLUMN_DEFAULTS.get(c, "")
        default_disp = f'"{default}"' if default != "" else "（空欄）"
        filled = report["fill_counts"].get(c, 0)
        print(f"   - {c:<22} 初期値={default_disp:<8} 補完 {filled} 件")
    if report.get("backup_path"):
        print(f" バックアップ   : {os.path.basename(report['backup_path'])}")
    integrity = report.get("integrity_check")
    if integrity is not None:
        print(f" 整合性確認     : {'OK' if integrity['ok'] else '異常あり'}")
    print(f" 結果           : {report.get('reason', '')}")
    print("=" * 60)


# ════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════

def _main(argv: list[str]) -> int:
    check_only = "--check" in argv
    dry_run = "--dry-run" in argv

    if check_only:
        status = check_schema()
        print("=" * 60)
        print(" スキーマ判定")
        print("=" * 60)
        print(f" ファイル存在   : {status['exists']}")
        print(f" 現在バージョン : {status['current_version']}")
        print(f" 期待バージョン : {status['expected_version']}")
        print(f" 要変換         : {status['needs_migration']}")
        print(f" 詳細           : {status['reason']}")
        print("=" * 60)
        return 0

    report = run_migration(dry_run=dry_run)
    print_report(report)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
