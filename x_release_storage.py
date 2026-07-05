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

import logging
import os
import shutil
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
    return bool(_get_token() and _get_repo())


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
    if not headers or not repo:
        return None

    url = f"{API_BASE}/repos/{repo}/releases/tags/{DATA_STORE_TAG}"
    try:
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        log.warning("[release_storage] Release取得失敗（通信エラー）: %s", e)
        return None

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


# ════════════════════════════════════════════════════════════
# 公開インターフェース（呼び出し側はこの3関数だけ使えばよい）
# ════════════════════════════════════════════════════════════

def download_file(filename: str, local_path: str) -> bool:
    """
    Release から filename をダウンロードして local_path に保存する。
    アセットが存在しない場合（初回起動等）は False を返す
    （呼び出し側は「ファイルなし＝空から開始」として扱うこと）。
    is_available() が False の場合も False を返す（処理は継続させる）。
    """
    if not is_available():
        log.info("[release_storage] GITHUB_TOKEN/GITHUB_REPOSITORY未設定のためスキップ: %s", filename)
        return False

    release = _get_or_create_release()
    if not release:
        return False

    asset = _find_asset(release, filename)
    if not asset:
        log.info("[release_storage] %s のアセットが存在しません（初回起動）", filename)
        return False

    ok = _download_asset(asset, local_path)
    if ok:
        log.info("[release_storage] %s をダウンロードしました", filename)
    return ok


def upload_file(local_path: str, filename: str) -> bool:
    """
    local_path の内容を Release へ filename としてアップロードする。
    アップロード前に、現在の filename アセットがあれば filename.bak へ
    世代交代してから、新しい内容を filename としてアップロードする
    （本体＋直前バックアップの2世代のみ保持）。

    アップロード後は再ダウンロードしてサイズを検証する。
    検証に失敗しても .bak への自動巻き戻しは行わない
    （restore_from_backup() を呼び出し側が明示的に使うこと）。
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

    backup_name = filename + ".bak"

    # ── ①現在の本体があれば .bak へ世代交代 ──────────────────
    current_asset = _find_asset(release, filename)
    if current_asset:
        tmp_bak = local_path + ".bak.tmp"
        if _download_asset(current_asset, tmp_bak):
            old_bak = _find_asset(release, backup_name)
            if old_bak and not _delete_asset(old_bak["id"]):
                log.warning("[release_storage] 旧.bak削除失敗: %s（続行）", backup_name)
            if not _upload_asset(release, backup_name, tmp_bak):
                log.warning("[release_storage] .bakアップロード失敗: %s（続行）", backup_name)
            if os.path.exists(tmp_bak):
                os.remove(tmp_bak)
        else:
            log.warning("[release_storage] 世代交代用ダウンロード失敗（.bak更新スキップ、続行）: %s", filename)

    # ── ②新しい内容を本体としてアップロード ──────────────────
    if current_asset and not _delete_asset(current_asset["id"]):
        log.warning("[release_storage] 旧本体削除失敗: %s（続行を試みる）", filename)

    # release情報にキャッシュされたassets一覧は古い可能性があるため作り直す
    release = _get_or_create_release()
    if not release:
        return False

    result = _upload_asset(release, filename, local_path)
    if not result:
        log.error("[release_storage] 本体アップロード失敗: %s", filename)
        return False

    # ── ③アップロード後の検証（サイズ比較） ────────────────
    local_size = os.path.getsize(local_path)
    remote_size = result.get("size")
    if remote_size != local_size:
        log.error(
            "[release_storage] アップロード後のサイズ不一致: %s (local=%d, remote=%s)",
            filename, local_size, remote_size,
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
