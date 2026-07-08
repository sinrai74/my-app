#!/usr/bin/env python3
"""
download_k_history.py  ── Kファイル自動ダウンロード 独立モジュール

【独立モジュール】既存Bot（notify_arashi.py・x_ranking.py等）には一切
組み込まれていない。x_local_course_stats.py・x_kfile_race_parser.py・
k_race_history_schema.py・k_race_history_integrity.py とも独立しており、
本モジュールが生成する data/k_files/ フォルダを、それらのモジュールの
--k-dir 引数にそのまま渡すことで連携する（ファイル経由の疎結合）。

【既存コードの再利用】ダウンロード部分（URL構造・ヘッダー）は、既存の
（ユーザー提供の）予想エンジンスクリプト内 _download_and_extract() の
設計を踏襲している。展開形式は実データ検証の結果 ZIP ではなく
LZH（LHa 2.x, lh5圧縮）であることが判明した。

【展開処理について】当初PyPIの`lhafile`パッケージ利用を検討したが、
C拡張（lzhlib）を含むため、Windows環境でインストールする際に
Microsoft Visual C++ Build Tools が必要になり、環境構築の妨げになる
ことが判明した。この依存を避けるため、lh5展開アルゴリズムを純粋
Pythonで実装した独自モジュール lzh_extract.py を新規作成し、それを
使用する（外部Cライブラリ・追加のビルド環境は一切不要）。

【依存パッケージ】requests のみ（標準的なpip installで導入可能）。
lzh_extract.py は本プロジェクト内の独立ファイルであり、追加インストール
は不要。

【重要な注意】本モジュールの実際のダウンロード動作は、開発環境の
ネットワーク制限（boatrace.jp への外部通信が許可されていない）により
実機検証できていない。LZH展開処理自体は、実際に取得されたLZHファイル
（k260707.lzh）で展開結果が元のK260707.TXTとMD5完全一致することを
確認済みで信頼度が高い。初回実行時は必ず少量の日付範囲
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
import logging
import os
import time
from datetime import date, datetime, timedelta

import requests

import lzh_extract

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("download_k_history")

# ════════════════════════════════════════════════════════════
# ダウンロード設定
# ════════════════════════════════════════════════════════════
# 【実データ検証済み】ダウンロードされるアーカイブは ZIP ではなく
# LZH（LHa 2.x, lh5圧縮）形式（k260707.lzh で確認）。
# 展開には lzh_extract.py（純粋Python実装、外部Cライブラリ不要）を使用する。
#
# 【重要・修正】過去の確定Kファイルは www.boatrace.jp ではなく
# www1.mbrace.or.jp から配布されている。www.boatrace.jp/owpc/pc/extra/data/
# 以下は当日速報用のURLで、過去日付を指定するとステータス200のまま
# HTMLの案内/エラーページが返ってくるため「ZIP展開失敗」になっていた。
# 正しいURL構造は年月サブディレクトリ＋小文字ファイル名:
#   http://www1.mbrace.or.jp/od2/K/{yyyymm}/k{yymmdd}.lzh
# 例: 2026-07-01 → http://www1.mbrace.or.jp/od2/K/202607/k260701.lzh
_BASE_URL = "http://www1.mbrace.or.jp/od2/K"
_HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept":          "application/zip,*/*",
    "Accept-Language": "ja,en;q=0.9",
    "Referer":         "https://www1.mbrace.or.jp/od2/K/dindex.html",
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
    指定日のKファイルをダウンロードし、LZHからTXTを展開して保存する。

    【実データで確認済みの事実】boatrace.jp が配布するアーカイブは
    ZIP ではなく LZH（LHa 2.x, lh5圧縮）形式（k260707.lzh で確認、
    展開結果は元の K260707.TXT と MD5完全一致）。
    """
    ds = target_date.strftime("%Y%m%d")
    filename = os.path.join(out_dir, f"K{ds}.TXT")

    if os.path.exists(filename):
        return DownloadResult.SKIPPED, f"{filename} は既に存在します"

    # www1.mbrace.or.jp の実際の配置形式: /od2/K/{yyyymm}/k{yymmdd}.lzh
    # （年月ディレクトリ・ファイル名は小文字k・年は2桁）
    yyyymm = target_date.strftime("%Y%m")
    yymmdd = target_date.strftime("%y%m%d")
    url = f"{_BASE_URL}/{yyyymm}/k{yymmdd}.lzh"
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
        entries = lzh_extract.extract_all(resp.content)
        for name, data in entries.items():
            if name.upper().endswith(".TXT"):
                os.makedirs(out_dir, exist_ok=True)
                with open(filename, "wb") as f:
                    f.write(data)
                return DownloadResult.OK, f"{filename} ({len(data):,} bytes)"
        return DownloadResult.ERROR, f"LZH内にTXTが見つかりません: {list(entries.keys())}"
    except lzh_extract.BadLzhFile as e:
        # 【デバッグ情報】原因特定用。ロジックには一切影響しない。
        # サーバーがステータス200でLZH以外の何か（HTMLエラーページ等）を
        # 返している可能性が高いため、実際に届いた内容を確認できるよう
        # ステータスコード・Content-Type・先頭バイトをログに出す。
        content_type = resp.headers.get("Content-Type", "不明")
        content_len = len(resp.content)
        preview = resp.content[:300]
        try:
            preview_text = preview.decode("utf-8", errors="replace")
        except Exception:
            preview_text = repr(preview)
        log.error(
            "[DEBUG download] url=%s status=%s content-type=%s content-length=%d\n"
            "先頭300バイト: %s",
            url, resp.status_code, content_type, content_len, preview_text,
        )
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
