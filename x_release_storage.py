#!/usr/bin/env python3
"""
x_release_storage.py  ── GitHub Releases 共通永続化ストレージ

【設計】
GitHub Actions のジョブ実行環境は毎回使い捨て（ephemeral）のため、
実行間でデータを引き継ぐには GitHub 外（またはリポジトリのメタデータ
領域）への永続化が必要になる。本モジュールは、長期保存が必須の
運用データ（hit_record.csv 等）を GitHub Releases の1つの固定タグ
（DATA_STORE_TAG = "data-store"）のアセットとして保存する。

  ・新しい Release や tag は作成しない（常に同じ1つの Release を使い回す）。
  ・各ファイルは「本体」＋「直前バックアップ(.bak)」の2世代のみ保持する。
  ・GitHub Actions Cache / Artifacts は本モジュールの対象外
    （それらは immutable かつ有効期限があり、長期保存に向かないため）。
  ・既存コードの open()/csv.DictReader 等は変更しない。
    「読み込み前に download_file() でローカルに取得」
    「書き込み後に upload_file() で Release へ反映」を
    呼び出し側に追加するだけで済むように設計する。

【失敗時の方針】
API呼び出しが失敗しても例外は投げない（Botの処理自体を止めないため）。
戻り値の bool / None で成否を呼び出し側に伝え、呼び出し側が
「今回はスキップして継続する」を選べるようにする。
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import re
import shutil
import time
from typing import Optional

import requests

log = logging.getLogger("x_release_storage")

DATA_STORE_TAG = "data-store"
API_BASE = "https://api.github.com"
UPLOAD_BASE = "https://uploads.github.com"
REQUEST_TIMEOUT = 30  # 秒


# ════════════════════════════════════════════════════════════
# 認証・リポジトリ情報
# ════════════════════════════════════════════════════════════

def _get_token() -> Optional[str]:
    return os.environ.get("GITHUB_TOKEN")


def _get_repo() -> Optional[str]:
    """"owner/repo" 形式。GitHub Actions では GITHUB_REPOSITORY が自動セットされる。"""
    return os.environ.get("GITHUB_REPOSITORY")


def _headers() -> Optional[dict]:
    token = _get_token()
    if not token:
        log.warning("[release_storage] GITHUB_TOKEN が設定されていません")
        return None
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def is_available() -> bool:
    """本モジュールが動作可能な環境か（トークン・リポジトリ情報が揃っているか）。"""
    _token = _get_token()
    _repo = _get_repo()
    _result = bool(_token and _repo)
    # 【デバッグログ】原因特定用。ロジックには一切影響しない。値そのものは伏せる。
    log.info(
        "[DEBUG release_storage] is_available: GITHUB_TOKEN存在=%s GITHUB_REPOSITORY存在=%s → 結果=%s",
        bool(_token), bool(_repo), _result,
    )
    return _result


# ════════════════════════════════════════════════════════════
# Release 取得・作成（data-store タグを常に1つだけ使う）
# ════════════════════════════════════════════════════════════

def _get_or_create_release() -> Optional[dict]:
    """
    data-store タグの Release を取得する。存在しなければ作成する。
    新しい tag・Release は data-store の1つ以外は絶対に作らない。
    """
    headers = _headers()
    repo = _get_repo()
    # 【デバッグログ】原因特定用。ロジックには一切影響しない。
    log.info(
        "[DEBUG release_storage] _get_or_create_release: 処理開始 headers取得=%s repo=%s",
        bool(headers), repo,
    )
    if not headers or not repo:
        log.info(
            "[DEBUG release_storage] _get_or_create_release: headers/repo不足のため終了 "
            "(headers取得=%s repo取得=%s)",
            bool(headers), bool(repo),
        )
        return None

    url = f"{API_BASE}/repos/{repo}/releases/tags/{DATA_STORE_TAG}"
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        log.warning("[release_storage] Release取得失敗（通信エラー）: %s", e)
        return None

    # 【デバッグログ】原因特定用。ロジックには一切影響しない。
    log.info("[DEBUG release_storage] GET %s → status=%s", url, resp.status_code)

    if resp.status_code == 200:
        return resp.json()

    if resp.status_code == 404:
        # 初回のみ作成する。以後は同じ Release を使い回す。
        create_url = f"{API_BASE}/repos/{repo}/releases"
        payload = {
            "tag_name": DATA_STORE_TAG,
            "name": "Data Store (internal, do not delete)",
            "body": "Botの運用データ永続化専用。x_release_storage.py が自動管理する。",
            "draft": False,
            "prerelease": True,
        }
        try:
            resp2 = requests.post(create_url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        except requests.RequestException as e:
            log.warning("[release_storage] Release作成失敗（通信エラー）: %s", e)
            return None
        if resp2.status_code in (200, 201):
            log.info("[release_storage] data-store Releaseを新規作成しました")
            return resp2.json()
        log.warning("[release_storage] Release作成失敗: status=%s body=%s", resp2.status_code, resp2.text[:200])
        return None

    log.warning("[release_storage] Release取得失敗: status=%s", resp.status_code)
    return None


def _find_asset(release: dict, name: str) -> Optional[dict]:
    for asset in release.get("assets", []):
        if asset.get("name") == name:
            return asset
    return None


# ════════════════════════════════════════════════════════════
# アセットのダウンロード・アップロード・削除
# ════════════════════════════════════════════════════════════

def _download_asset(asset: dict, dest_path: str) -> bool:
    headers = _headers()
    if not headers:
        return False
    dl_headers = dict(headers)
    dl_headers["Accept"] = "application/octet-stream"
    try:
        resp = requests.get(asset["url"], headers=dl_headers, timeout=REQUEST_TIMEOUT, stream=True)
    except requests.RequestException as e:
        log.warning("[release_storage] アセットダウンロード失敗（通信エラー）: %s", e)
        return False
    if resp.status_code != 200:
        log.warning("[release_storage] アセットダウンロード失敗: status=%s", resp.status_code)
        return False

    tmp_path = dest_path + ".downloading"
    try:
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        os.replace(tmp_path, dest_path)
    except OSError as e:
        log.warning("[release_storage] ダウンロード内容の保存失敗: %s", e)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False
    return True


def _delete_asset(asset_id: int) -> bool:
    headers = _headers()
    repo = _get_repo()
    if not headers or not repo:
        return False
    url = f"{API_BASE}/repos/{repo}/releases/assets/{asset_id}"
    try:
        resp = requests.delete(url, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        log.warning("[release_storage] アセット削除失敗（通信エラー）: %s", e)
        return False
    # 204: 削除成功, 404: 既に存在しない(実質成功扱い)
    return resp.status_code in (204, 404)


def _upload_asset(release: dict, name: str, local_path: str) -> Optional[dict]:
    """
    release に name というアセットをアップロードする。
    同名アセットが既に存在する場合は Releases API の仕様上アップロードできない
    （422エラーになる）ため、呼び出し側で事前に削除してから呼ぶこと。
    """
    headers = _headers()
    if not headers:
        return None
    upload_headers = dict(headers)
    upload_headers["Content-Type"] = "application/octet-stream"

    release_id = release["id"]
    url = f"{UPLOAD_BASE}/repos/{_get_repo()}/releases/{release_id}/assets?name={name}"
    try:
        with open(local_path, "rb") as f:
            data = f.read()
        resp = requests.post(url, headers=upload_headers, data=data, timeout=REQUEST_TIMEOUT)
    except (requests.RequestException, OSError) as e:
        log.warning("[release_storage] アセットアップロード失敗: %s", e)
        return None

    if resp.status_code in (200, 201):
        return resp.json()
    log.warning("[release_storage] アセットアップロード失敗: status=%s body=%s", resp.status_code, resp.text[:200])
    return None



def _rename_asset(asset_id: int, new_name: str) -> Optional[dict]:
    """アセット名を変更する（Releases API: PATCH /releases/assets/{id}）。"""
    headers = _headers()
    repo = _get_repo()
    if not headers or not repo:
        return None
    url = f"{API_BASE}/repos/{repo}/releases/assets/{asset_id}"
    try:
        resp = requests.patch(url, headers=headers, json={"name": new_name}, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        log.warning("[release_storage] アセット名変更失敗（通信エラー）: %s", e)
        return None
    if resp.status_code == 200:
        return resp.json()
    log.warning("[release_storage] アセット名変更失敗: status=%s body=%s", resp.status_code, resp.text[:200])
    return None


# ════════════════════════════════════════════════════════════
# 追記型ファイルの統合（保存時に Release 上の最新版と突き合わせる）
# ════════════════════════════════════════════════════════════
#
# 追記型ファイルは「Release 上の最新版 ∪ ローカル」を保存する。
# ローカルが古い・ダウンロードに失敗した・別の実行が先に保存した、
# のいずれの場合も、Release 上の既存の行を失わないようにするため。
#
# 統合は「同一の行（または同一の key）を二重に持たない」ことだけを行い、
# 内容の異なる行のどちらを正とするかは決めない（仕様が未確定のため）。
#   lines          : 行の文字列が完全に同じものだけを1つにする（buyscore_log.jsonl）
#   csv_rows       : ヘッダが同じ場合に限り、全列が同じ行だけを1つにする（hit_record.csv）
#   sent_key_first : JSON の "key" が同じ行は、先に保存された行を残す（sent_YYYYMMDD.txt）
#                    notify_arashi.py の sent ファイル書き込み（同じ key は追加しない）と同じ規則

_SENT_RE = re.compile(r"^sent_\d{8}\.txt$")


class MergeConflict(Exception):
    """機械的に統合できない（ヘッダ不一致など）。保存を中止する。"""


def merge_kind(filename: str) -> Optional[str]:
    base = os.path.basename(filename)
    if base == "hit_record.csv":
        return "csv_rows"
    if base == "buyscore_log.jsonl":
        return "lines"
    if _SENT_RE.match(base):
        return "sent_key_first"
    return None


def _read_text(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def _nonempty_lines(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.strip()]


def _sent_key(line: str) -> str:
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return str(obj.get("key", line))
    except (ValueError, TypeError):
        pass
    return line.strip()


def _csv_header(text: str) -> list[str]:
    for row in csv.reader(io.StringIO(text)):
        if row:
            return row
    return []


def _check_csv_widths(text: str, header: Optional[list[str]]) -> None:
    """header（None なら text の先頭行）と列数の違う行があれば MergeConflict。"""
    rows = [row for row in csv.reader(io.StringIO(text)) if row]
    if header is None:
        if not rows:
            return
        header, rows = rows[0], rows[1:]
    bad = [i for i, row in enumerate(rows) if len(row) != len(header)]
    if bad:
        raise MergeConflict(f"列数がヘッダ（{len(header)}列）と一致しない行が {len(bad)} 行ある")


def merge_text(kind: str, remote: str, local: str) -> str:
    """remote（Release 上の最新版）に local を統合した内容を返す。remote の行は必ず残す。"""
    if not remote:
        if kind == "csv_rows":
            _check_csv_widths(local, None)
        return local
    if not local or local == remote:
        return remote
    if local.startswith(remote):
        # ローカルが Release 版の末尾に追記しただけの通常ケース。バイト列をそのまま使う
        if kind == "csv_rows":
            _check_csv_widths(local[len(remote):], _csv_header(remote))
        return local

    if kind == "lines":
        r = _nonempty_lines(remote)
        seen = set(r)
        add = [ln for ln in _nonempty_lines(local) if ln not in seen and not seen.add(ln)]
        if not add:
            return remote
        return "\n".join(r + add) + "\n"

    if kind == "sent_key_first":
        r = _nonempty_lines(remote)
        keys = {_sent_key(ln) for ln in r}
        add = []
        for ln in _nonempty_lines(local):
            k = _sent_key(ln)
            if k not in keys:
                keys.add(k)
                add.append(ln)
        if not add:
            return remote
        return "\n".join(r + add)

    if kind == "csv_rows":
        r_rows = list(csv.reader(io.StringIO(remote)))
        l_rows = list(csv.reader(io.StringIO(local)))
        r_rows = [row for row in r_rows if row]
        l_rows = [row for row in l_rows if row]
        if not r_rows:
            return local
        if not l_rows:
            return remote
        if r_rows[0] != l_rows[0]:
            raise MergeConflict(
                f"ヘッダ不一致（Release {len(r_rows[0])}列 / ローカル {len(l_rows[0])}列）"
            )
        seen = {tuple(row) for row in r_rows[1:]}
        add = []
        for row in l_rows[1:]:
            t = tuple(row)
            if t not in seen:
                seen.add(t)
                add.append(row)
        if not add:
            return remote
        width = len(r_rows[0])
        if any(len(row) != width for row in add):
            raise MergeConflict(f"追加行の列数がヘッダ（{width}列）と一致しない")
        buf = io.StringIO()
        w = csv.writer(buf)  # csv.DictWriter の既定（QUOTE_MINIMAL, "\r\n"）と同じ
        sep = "" if remote.endswith(("\n", "\r")) else "\r\n"
        w.writerows(add)
        return remote + sep + buf.getvalue()

    raise ValueError(f"unknown merge kind: {kind}")


def _write_atomic(path: str, text: str) -> None:
    tmp = path + ".merging"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ════════════════════════════════════════════════════════════
# 公開インターフェース（呼び出し側はこの3関数だけ使えばよい）
# ════════════════════════════════════════════════════════════

def download_file(filename: str, local_path: str) -> bool:
    """
    Release から filename をダウンロードして local_path に保存する。
    アセットが存在しない場合（初回起動等）は False を返す
    （呼び出し側は「ファイルなし＝空から開始」として扱うこと）。
    is_available() が False の場合も False を返す（処理は継続させる）。

    本体が無い場合の扱い：
      ・filename.new がある → 前回の保存が「本体の切替」の直前で止まった状態。
        .new が最新の内容なので、それを読む。
      ・追記型ファイルで .bak だけがある → 旧方式の保存の途中（本体削除〜再アップロード
        の間）か、その失敗後の状態。空から始めると履歴を失うため .bak を読む。
    """
    if not is_available():
        log.info("[release_storage] GITHUB_TOKEN/GITHUB_REPOSITORY未設定のためスキップ: %s", filename)
        return False

    release = _get_or_create_release()
    if not release:
        return False

    asset = _find_asset(release, filename)
    if not asset:
        staged = _find_asset(release, filename + ".new")
        backup = _find_asset(release, filename + ".bak")
        if staged:
            log.warning("[release_storage] %s の本体が無く .new があるため .new を読みます", filename)
            asset = staged
        elif backup and merge_kind(filename):
            log.warning("[release_storage] %s の本体が無いため .bak を読みます（追記型）", filename)
            asset = backup
        else:
            log.info("[release_storage] %s のアセットが存在しません（初回起動）", filename)
            return False

    ok = _download_asset(asset, local_path)
    if ok:
        log.info("[release_storage] %s をダウンロードしました", filename)
    return ok


def upload_file(local_path: str, filename: str, merge: Optional[bool] = None) -> bool:
    """
    local_path の内容を Release へ filename として保存する。

    保存の手順（Release 上に「本体が無い時間」と「部分的な内容」を作らない）：
      1. 追記型ファイル（merge_kind）は、Release 上の最新版をダウンロードしてローカルと統合する。
         Release 上に版があるのにダウンロードできない場合は保存しない（False）。
      2. 新しい内容を filename.new としてアップロードし、サイズ（と取得できれば sha256）を検証する。
      3. 旧 .bak を削除し、現在の本体の名前を filename.bak に変える。
      4. filename.new の名前を filename に変える。
    1〜3 のどこで失敗しても、Release 上の本体は元のまま残る（.new は片付ける）。
    4 で失敗した場合は、本体の代わりに .new が最新として残り、download_file と次回の
    upload_file がそれを本体として扱う。

    merge: None（既定）＝ファイル名から自動判定、False＝統合しない（全置換。手動の整理用）。
    """
    if not is_available():
        log.info("[release_storage] GITHUB_TOKEN/GITHUB_REPOSITORY未設定のためスキップ: %s", filename)
        return False
    if not os.path.exists(local_path):
        log.warning("[release_storage] アップロード対象が存在しません: %s", local_path)
        return False

    release = _get_or_create_release()
    if not release:
        return False

    stage_name = filename + ".new"
    backup_name = filename + ".bak"
    kind = merge_kind(filename) if merge is None or merge else None

    main = _find_asset(release, filename)
    staged = _find_asset(release, stage_name)

    # ── 0. 前回の切替が途中で止まっていたら、先に .new を本体へ切り替える ──
    if not main and staged:
        log.warning("[release_storage] %s: 前回の切替の途中状態を回復します（.new → 本体）", filename)
        if not _rename_asset(staged["id"], filename):
            log.error("[release_storage] %s: 途中状態の回復に失敗。保存を中止します", filename)
            return False
        release = _get_or_create_release()
        if not release:
            return False
        main = _find_asset(release, filename)
        staged = None

    # ── 1. 追記型ファイルは Release 上の最新版と統合する ──────────
    if kind:
        base = main or _find_asset(release, backup_name)
        if base:
            tmp_remote = local_path + ".remote.tmp"
            try:
                if not _download_asset(base, tmp_remote):
                    log.error("[release_storage] %s: Release 上の版を取得できないため保存しません", filename)
                    return False
                remote_text = _read_text(tmp_remote)
            finally:
                if os.path.exists(tmp_remote):
                    os.remove(tmp_remote)
            local_text = _read_text(local_path)
            try:
                merged = merge_text(kind, remote_text, local_text)
            except MergeConflict as e:
                log.error("[release_storage] %s: 統合できないため保存しません: %s", filename, e)
                return False
            if merged != local_text:
                _write_atomic(local_path, merged)
            if main and merged == remote_text:
                log.info("[release_storage] %s: 追加分なし（保存不要）", filename)
                return True

    # ── 2. 新しい内容を .new としてアップロードし、検証する ──────────
    if staged and not _delete_asset(staged["id"]):
        log.error("[release_storage] %s: 古い .new を削除できないため保存しません", filename)
        return False
    if staged:
        release = _get_or_create_release()
        if not release:
            return False

    up = _upload_asset(release, stage_name, local_path)
    if not up:
        log.error("[release_storage] 本体アップロード失敗: %s", filename)
        return False

    local_size = os.path.getsize(local_path)
    remote_size = up.get("size")
    digest = up.get("digest") or ""
    bad = remote_size != local_size
    if not bad and digest.startswith("sha256:"):
        bad = digest.split(":", 1)[1] != _sha256(local_path)
    if bad:
        log.error(
            "[release_storage] アップロード後の検証失敗: %s (local=%d, remote=%s)",
            filename, local_size, remote_size,
        )
        _delete_asset(up["id"])
        return False

    # ── 3. 現在の本体を .bak にする ─────────────────────────────
    if main:
        old_bak = _find_asset(release, backup_name)
        if old_bak and not _delete_asset(old_bak["id"]):
            log.error("[release_storage] %s: 旧 .bak を削除できないため保存を中止します", filename)
            _delete_asset(up["id"])
            return False
        if not _rename_asset(main["id"], backup_name):
            log.error("[release_storage] %s: 本体を .bak にできないため保存を中止します", filename)
            _delete_asset(up["id"])
            return False

    # ── 4. .new を本体にする ──────────────────────────────────
    if not _rename_asset(up["id"], filename):
        time.sleep(1)
        if not _rename_asset(up["id"], filename):
            log.error(
                "[release_storage] %s: 本体への切替に失敗。最新の内容は %s に残っています"
                "（次回の読込・保存で回復）", filename, stage_name,
            )
            return False

    log.info("[release_storage] %s をアップロードしました（%dバイト）", filename, local_size)
    return True


def restore_from_backup(filename: str, local_path: str) -> bool:
    """
    filename.bak を Release からダウンロードして local_path に復元する。
    本体アセットが壊れている/検証に失敗した場合の手動復旧に使う。
    """
    if not is_available():
        return False
    release = _get_or_create_release()
    if not release:
        return False
    backup_name = filename + ".bak"
    asset = _find_asset(release, backup_name)
    if not asset:
        log.warning("[release_storage] バックアップが存在しません: %s", backup_name)
        return False
    ok = _download_asset(asset, local_path)
    if ok:
        log.info("[release_storage] %s から復元しました", backup_name)
    return ok
