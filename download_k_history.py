#!/usr/bin/env python3
"""
download_k_history.py  ── Kファイル自動ダウンロード 独立モジュール

【独立モジュール】既存Bot（notify_arashi.py・x_ranking.py等）には一切
組み込まれていない。x_local_course_stats.py・x_kfile_race_parser.py・
k_race_history_schema.py・k_race_history_integrity.py とも独立しており、
本モジュールが生成する data/k_files/ フォルダを、それらのモジュールの
--k-dir 引数にそのまま渡すことで連携する（ファイル経由の疎結合）。

【既存コードの再利用】ダウンロード・ZIP展開ロジックは、既存の
（ユーザー提供の）予想エンジンスクリプト内 _download_and_extract() と
同一のURL・ヘッダー・展開方式を踏襲している。重複実装を避けるため、
新しい方式を独自に考案せず、動作実績のあるロジックをそのまま移植した。

【重要な注意】本モジュールの実際のダウンロード動作は、開発環境の
ネットワーク制限（boatrace.jp への外部通信が許可されていない）により
実機検証できていない。ロジック自体は動作実績のある既存コードの移植で
あるため信頼度は高いが、初回実行時は必ず少量の日付範囲
（例: 直近1週間）で試し、正常にファイルが取得できることを確認してから
2023年1月からの大量取得に進むこと。

【使い方】
  # 2023-01-01 から 2026-07-08 まで一括ダウンロード
  python download_k_history.py --from 2023-01-01 --to 2026-07-08

  # --to を省略すると今日の日付まで取得する
  python download_k_history.py --from 2026-07-01

  # 保存先を変更したい場合
  python download_k_history.py --from 2023-01-01 --to 2026-07-08 --out-dir data/k_files

取得したファイルは data/k_files/K{YYYYMMDD}.TXT として保存される。
そのまま以下で履歴DBを初回構築できる:
  python x_local_course_stats.py --init --k-dir data/k_files
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import time
import zipfile
from datetime import date, datetime, timedelta

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("download_k_history")

# ════════════════════════════════════════════════════════════
# ダウンロード設定（既存スクリプトの _download_and_extract() と同一）
# ════════════════════════════════════════════════════════════
_BASE_URL = "https://www.boatrace.jp/owpc/pc/extra/data"
_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":          "application/zip,*/*",
    "Accept-Language": "ja,en;q=0.9",
    "Referer":         "https://www.boatrace.jp/owpc/pc/extra/data/download.html",
}
DEFAULT_OUT_DIR = os.path.join("data", "k_files")
DEFAULT_INTERVAL_SEC = 1.5  # リクエスト間隔（既存スクリプト推奨値）
REQUEST_TIMEOUT = 15


class DownloadResult:
    """1日分のダウンロード結果を表す列挙的な文字列定数。"""
    OK = "ok"                # 新規ダウンロード成功
    SKIPPED = "skipped"      # 既にTXTが存在したためスキップ
    NO_EVENT = "no_event"    # 404（その日は開催なし。異常ではない）
    ERROR = "error"          # ダウンロード・展開に失敗（ログに記録すべき異常）


def _download_and_extract_k_file(target_date: date, out_dir: str) -> tuple[str, str]:
    """
    指定日のKファイルをダウンロードし、ZIPからTXTを展開して保存する。

    【既存コード再利用】ロジックは既存の _download_and_extract() と同一
    （URL・ヘッダー・404判定・ZIP展開）。ここでは日付単位のラッパーと
    してまとめ、戻り値を (DownloadResult定数, メッセージ) に統一している。
    """
    ds = target_date.strftime("%Y%m%d")
    filename = os.path.join(out_dir, f"K{ds}.TXT")

    if os.path.exists(filename):
        return DownloadResult.SKIPPED, f"{filename} は既に存在します"

    url = f"{_BASE_URL}/K{ds}.zip"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        return DownloadResult.ERROR, f"通信エラー: {e}"

    if resp.status_code == 404:
        return DownloadResult.NO_EVENT, "404（開催なし）"

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        return DownloadResult.ERROR, f"HTTPエラー: {e}"

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            names = z.namelist()
            for name in names:
                if name.upper().endswith(".TXT"):
                    data = z.read(name)
                    os.makedirs(out_dir, exist_ok=True)
                    with open(filename, "wb") as f:
                        f.write(data)
                    return DownloadResult.OK, f"{filename} ({len(data):,} bytes)"
        return DownloadResult.ERROR, f"ZIP内にTXTが見つかりません: {names}"
    except zipfile.BadZipFile as e:
        return DownloadResult.ERROR, f"ZIP展開失敗: {e}"


def download_range(
    start: date,
    end: date,
    out_dir: str = DEFAULT_OUT_DIR,
    interval_sec: float = DEFAULT_INTERVAL_SEC,
) -> dict:
    """
    start〜end（両端含む）の期間について、1日ずつKファイルをダウンロードする。

    【中断・再開対応】各日付は「ファイルが既に存在するかどうか」だけで
    スキップ判定するため、専用の進捗ファイルは持たない。中断後に同じ
    コマンドを再実行すれば、既にダウンロード済みの日付は自動的に
    スキップされ、未取得の日付だけが処理される。

    戻り値: {
      "total_days": int, "ok": int, "skipped": int,
      "no_event": int, "error": int,
      "errors": [{"date": "YYYYMMDD", "message": str}, ...],
      "elapsed_sec": float,
    }
    """
    if start > end:
        raise ValueError(f"--from({start}) が --to({end}) より後になっています")

    total_days = (end - start).days + 1
    report = {
        "total_days": total_days, "ok": 0, "skipped": 0,
        "no_event": 0, "error": 0, "errors": [],
        "elapsed_sec": 0.0,
    }

    log.info("━━ Kファイル自動ダウンロード ━━")
    log.info("期間: %s 〜 %s（%d日分）保存先: %s", start, end, total_days, out_dir)

    start_time = time.time()
    d = start
    day_index = 0
    while d <= end:
        day_index += 1
        ds = d.strftime("%Y%m%d")

        result, message = _download_and_extract_k_file(d, out_dir)

        if result == DownloadResult.OK:
            report["ok"] += 1
            log.info("[%d/%d] %s: 取得成功 - %s", day_index, total_days, ds, message)
        elif result == DownloadResult.SKIPPED:
            report["skipped"] += 1
            log.info("[%d/%d] %s: スキップ（既存） - %s", day_index, total_days, ds, message)
        elif result == DownloadResult.NO_EVENT:
            report["no_event"] += 1
            log.info("[%d/%d] %s: 開催なし - %s", day_index, total_days, ds, message)
        else:  # ERROR
            report["error"] += 1
            report["errors"].append({"date": ds, "message": message})
            log.error("[%d/%d] %s: 失敗 - %s", day_index, total_days, ds, message)

        # 新規ダウンロード・エラー時のみ待機（スキップ時はアクセスしていないため待たない）
        if result in (DownloadResult.OK, DownloadResult.NO_EVENT, DownloadResult.ERROR):
            time.sleep(interval_sec)

        d += timedelta(days=1)

    report["elapsed_sec"] = round(time.time() - start_time, 1)
    return report


def print_summary(report: dict) -> None:
    """処理終了後の取得件数サマリーを表示する。"""
    print("=" * 60)
    print(" Kファイル ダウンロード結果サマリー")
    print("=" * 60)
    print(f" 対象日数       : {report['total_days']}日")
    print(f" 新規取得成功   : {report['ok']}件")
    print(f" スキップ（既存）: {report['skipped']}件")
    print(f" 開催なし（404）: {report['no_event']}件")
    print(f" 失敗           : {report['error']}件")
    print(f" 処理時間       : {report['elapsed_sec']}秒")
    if report["errors"]:
        print(" 失敗した日付一覧:")
        for e in report["errors"]:
            print(f"   {e['date']}: {e['message']}")
    print("=" * 60)
    print(f" 結果: {'✅ 正常終了' if report['error'] == 0 else '⚠️ 一部失敗あり（上記一覧を確認してください）'}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kファイル自動ダウンロード 独立モジュール")
    parser.add_argument("--from", dest="date_from", required=True, help="開始日 YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", default=None, help="終了日 YYYY-MM-DD（省略時は今日）")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help=f"保存先フォルダ（デフォルト: {DEFAULT_OUT_DIR}）")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_SEC, help="リクエスト間隔・秒（デフォルト1.5）")
    args = parser.parse_args()

    try:
        start = datetime.strptime(args.date_from, "%Y-%m-%d").date()
    except ValueError:
        parser.error(f"--from の形式が不正です: {args.date_from}（YYYY-MM-DD で指定してください）")
        return

    if args.date_to:
        try:
            end = datetime.strptime(args.date_to, "%Y-%m-%d").date()
        except ValueError:
            parser.error(f"--to の形式が不正です: {args.date_to}（YYYY-MM-DD で指定してください）")
            return
    else:
        end = date.today()

    report = download_range(start, end, out_dir=args.out_dir, interval_sec=args.interval)
    print_summary(report)


if __name__ == "__main__":
    main()
