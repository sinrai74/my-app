"""
正式G3（B案）：G3 の診断の経路で、Legacy が Release・ネットワーク・外部コマンドに触れないための境界（N-01）。

Legacy（notify_arashi.py・x_buyscore.py・x_release_storage.py）は変更しない。この境界の中でだけ、
次の4層を同時に張る。どれか1層が漏れても、他の層で止まり、記録に残る。

  層1 Release の関数の差し替え：x_release_storage の関数（download_file・upload_file・
      _get_or_create_release など、モジュールで定義された関数すべて）と、他のモジュールが
      取り込んだその写しを、呼び出しを記録して失敗の値を返す関数に置き換える。
      Legacy は「Release が使えないとき」の既存の分岐（ローカルのファイル）で動く。
  層2 監査フック（sys.addaudithook）：ソケットの接続・名前解決・送信、外部コマンド
      （subprocess・os.system・exec・spawn・fork）、作業フォルダの外への書き込み・削除・移動を、
      実行の前に例外（G3IsolationBlocked）で止め、記録する。
  層3 環境変数：トークン・認証情報らしい名前の変数と、GitHub・git・メールなどの変数を外す（値は記録しない）。
      HOME・TMPDIR を作業フォルダの中に、HTTP(S)_PROXY を接続できない 127.0.0.1:9 にする。
  層4 作業フォルダ：新しい一時フォルダを作業ディレクトリにする。Legacy が相対パスで読む
      設定・成績・モデル（buyscore_config.json・hit_record.csv・model_all.pkl など）は、
      呼び出し側が seed_files で明示したものだけを置く（何を置くかはこのモジュールでは決めない）。

境界を張った直後に自己試験を行い、各層が実際に止めることを確かめる。自己試験が1つでも
失敗したら established=False とし、呼び出し側は Legacy を呼ばない。
実行中に層2 が何か1つでも止めた場合（Legacy の側の想定外の経路）は verified=False とする。
層1 で止めた Release の呼び出しは「想定どおり遮断した記録」で、verified を下げない。

制約：プロセス全体の状態（作業ディレクトリ・環境変数・監査フック）を使うので、同時に1つだけ。
入れ子・並行の利用は established=False になる。監査フックは外せないので、境界の外では何もしない。
"""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Mapping, Optional, Union

RELEASE_MODULE = "x_release_storage"

# 層1：差し替えた関数が返す値（Release が使えないときの、元の関数の失敗の値と同じ型）
_STUB_RETURNS: dict[str, Any] = {
    "download_file": False,
    "upload_file": False,
    "restore_from_backup": False,
    "is_available": False,
    "_download_asset": False,
    "_delete_asset": False,
}
_RELEASE_URL_ATTRS = ("API_BASE", "UPLOAD_BASE")
_DEAD_URL = "http://127.0.0.1:9"

# 層2：止めるイベント
_NETWORK_EVENTS = frozenset({
    "socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
    "socket.gethostbyname_ex", "socket.sendto", "socket.sendmsg", "socket.bind",
    "urllib.Request", "http.client.connect", "smtplib.connect", "ftplib.connect",
    "poplib.connect", "imaplib.open", "nntplib.connect", "telnetlib.Telnet.open",
    "webbrowser.open",
})
_PROCESS_EVENTS = frozenset({
    "subprocess.Popen", "os.system", "os.exec", "os.posix_spawn", "os.spawn",
    "os.fork", "os.forkpty", "os.startfile", "pty.spawn",
})
_PATH_WRITE_EVENTS = {
    # イベント名: 書き込み先のパスが入っている引数の位置
    "os.remove": (0,), "os.rmdir": (0,), "os.mkdir": (0,), "os.rename": (0, 1),
    "os.symlink": (1,), "os.link": (1,), "os.truncate": (0,), "os.chmod": (0,),
    "os.chown": (0,), "os.utime": (0,), "os.chflags": (0,), "os.lchflags": (0,),
    "shutil.copyfile": (1,), "shutil.copymode": (1,), "shutil.copystat": (1,),
    "shutil.copytree": (1,), "shutil.rmtree": (0,), "shutil.move": (0, 1),
    "shutil.chown": (0,),
}
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

# 層3：外す環境変数（名前で判断。値は読まない・記録しない）
_ENV_NAME_PATTERN = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|_PASS$|_PASS_|CREDENTIAL|API_?KEY|ACCESS_?KEY|PRIVATE_?KEY|_KEY$"
    r"|_KEY_|_PAT$|_PAT_|^PAT_|WEBHOOK|AUTH|COOKIE|SESSION|_PWD$|_PWD_|CREDS)",
    re.IGNORECASE,
)
# 取得の段（ネットワークあり）で「あれば止まる」名前（C-07：境界と同じ名前の型＋既知の名前。
# GitHub が既定で置く GITHUB_SHA・RUNNER_OS などは対象にしない）
_REFUSE_EXACT = frozenset({"NETRC", "SSH_AUTH_SOCK", "GIT_ASKPASS", "SSH_ASKPASS", "GH_PAT",
                           "AWS_ACCESS_KEY_ID", "GMAIL_APP_PASS"})


# 再監査：Linux のデスクトップ・ログインの名前（XDG_SESSION_ID・DBUS_SESSION_BUS_ADDRESS など）は秘密ではないので、
# 取得の段の「あれば止まる」の対象にしない（境界の中では外す）
_REFUSE_EXEMPT_PREFIXES = ("XDG_SESSION_", "XDG_SEAT", "XDG_VTNR")
_REFUSE_EXEMPT_EXACT = frozenset({"DBUS_SESSION_BUS_ADDRESS"})


def env_name_must_refuse(name: str) -> bool:
    if name in _REFUSE_EXEMPT_EXACT:
        return False
    for prefix in _REFUSE_EXEMPT_PREFIXES:
        if name.startswith(prefix):
            # 再監査 E-07（R-n05）：外すのは接頭辞の部分だけ。残りに秘密の名前の型があれば止める（XDG_SESSION_TOKEN など）
            return bool(_ENV_NAME_PATTERN.search("X_" + name[len(prefix):]))
    return bool(_ENV_NAME_PATTERN.search(name)) or name in _REFUSE_EXACT
_ENV_PREFIXES = ("GITHUB_", "GH_", "GIT_", "ACTIONS_", "LINE_", "SMTP", "MAIL", "GMAIL",
                 "SSH_", "AWS_", "AZURE_", "GOOGLE_", "GCLOUD", "CLOUDSDK_")
_ENV_EXACT = frozenset({"NETRC", "CURL_HOME", "WGETRC", "REQUESTS_CA_BUNDLE_AUTH"})
_ENV_SET_KEYS = ("HOME", "TMPDIR", "TEMP", "TMP", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                 "http_proxy", "https_proxy", "all_proxy", "NO_PROXY", "no_proxy", "NETRC",
                 "XDG_CONFIG_HOME", "XDG_CACHE_HOME")

SeedSource = Union[str, os.PathLike, bytes]


class G3IsolationBlocked(PermissionError):
    """境界が止めた操作。OSError の子なので、Legacy の既存の失敗の分岐に入る（記録は残る）。"""


# ---- 監査フック（プロセスに1つだけ入れ、境界の中でだけ働く） ----
_ACTIVE: Optional["G3Isolation"] = None
_HOOK_INSTALLED = False
_HOOK_LOCK = threading.Lock()
_TLS = threading.local()


def _hook(event: str, args: tuple) -> None:
    iso = _ACTIVE
    if iso is None or getattr(_TLS, "busy", False):
        return
    _TLS.busy = True
    try:
        iso._on_event(event, args)
    finally:
        _TLS.busy = False


def _ensure_hook() -> None:
    global _HOOK_INSTALLED
    with _HOOK_LOCK:
        if not _HOOK_INSTALLED:
            sys.addaudithook(_hook)
            _HOOK_INSTALLED = True


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://\S*")
_SECRET_WORDS = r"(?:token|secret|password|passwd|pwd|pass|api[-_]?key|access[-_]?key|secret[-_]?key|private[-_]?key|auth|authorization|cookie|session|credential)"
_SECRET_PATTERNS = (
    # E-04：PEM の秘密鍵（終わりの行が無い・切れていても、その先を伏せる）
    # F-10：PGP の "PRIVATE KEY BLOCK" も同じに扱う
    re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----"
               r"(?:.*?-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----|.*)", re.DOTALL),
    # E-04：JWT（header.payload.signature）
    re.compile(r"\beyJ[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]{4,}\.[A-Za-z0-9_\-]*"),
    # E-04：引用符で始まる値（password="x"・api_key='x'・{'access_token': b'x'}・client_secret="x"）
    re.compile(r"""(?i)["']?[A-Za-z0-9_\-]*""" + r"(?:token|secret|password|passwd|pwd|pass|api[-_]?key|access[-_]?key"
               r"|private[-_]?key|auth|authorization|cookie|session|credential)" + r"""[A-Za-z0-9_\-]*["']?\s*[=:]\s*"""
               r"""[bBrRuU]?(["'])(?:[^"'\\]|\\.)*\1"""),
    # E-04：パスワード・秘密の名前の値は、行の終わりまで伏せる（GMAIL_APP_PASS=abcd efgh … の2語目以降を残さない）
    # F-10：token・credential・api_key などの値も、行の終わりまで伏せる（token = abc def ghi・client_secret=abc def）
    re.compile(r"(?i)[A-Za-z0-9_\-]*(?:password|passwd|_pass\b|_pass_|app_pass|secret|token|credential|creds"
               r"|api[-_]?key|access[-_]?key|private[-_]?key|\bpwd)[A-Za-z0-9_\-]*\s*[=:][^\n]*"),
    # F-10：空白で区切った形（line_channel_access_token VALUE）
    re.compile(r"(?i)\b[A-Za-z0-9_\-]*(?:token|secret|password|passwd|credential|creds|api[-_]?key)\b\s+[^\s,;&\"']+"),
    # 既知のトークンの形
    re.compile(r"(gh[pousr]_[A-Za-z0-9]{8,}|github_pat_[A-Za-z0-9_]{8,}|xox[abprs]-[A-Za-z0-9-]{8,}"
               r"|glpat-[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{16,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16})"),
    # "token": "X"・'token': 'X'
    re.compile(r"""(?i)(["']?""" + _SECRET_WORDS + r"""["']?\s*:\s*)(["'])(?:[^"'\\]|\\.)*\2"""),
    # D-05：Basic・Bearer の認証（値の前の語だけを伏せて値が残ることを防ぐため、先に当てる）
    re.compile(r"(?i)\b(?:basic|bearer)\s+[A-Za-z0-9+/=._~\-]+"),
    # token=X・token: X・password X・Authorization: token X・Bearer X・Basic X
    re.compile(r"(?i)\b(" + _SECRET_WORDS + r"|bearer|basic)\b(\s*[=:]\s*|\s+)(?:token\s+|bearer\s+|basic\s+)?[^\s,;&\"']+"),
    # D-05：名前の前後に別の語がついた形（X_TOKEN=…・aws_secret_access_key = …・GH_PAT=…）
    re.compile(r"(?i)[A-Za-z0-9_\-]*(?:" + _SECRET_WORDS + r"|_pat\b|^pat_|\bpat)[A-Za-z0-9_\-]*\s*[=:]\s*[^\s,;&\"']+"),
    # 利用者情報つきの宛先（scheme なし）user:X@host
    re.compile(r"\b[^\s:@/]+:[^\s@]+@[A-Za-z0-9.\-]+"),
)


def _redact_url(url: str) -> str:
    """URL は scheme://host[:port]/*** だけを残す（利用者情報・パス・パラメータ・クエリ・断片を伏せる）。"""
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        port = f":{parts.port}" if parts.port else ""
    except ValueError:
        return "url:***"
    return f"{parts.scheme}://{host}{port}/***"


def redact_text(text: str, limit: int = 200, keep: str = "head") -> str:
    """A-06・B-04・C-07：URL を伏せ、トークン・パスワードらしい部分を消す。伏せたあとで長さを切る
    （C-06：keep="tail" なら末尾を残す。例外の最後の行を失わないため）。"""
    text = _URL_RE.sub(lambda m: _redact_url(m.group(0)), text)
    for pat in _SECRET_PATTERNS:
        text = pat.sub("***", text)
    if len(text) <= limit:
        return text
    return text[-limit:] if keep == "tail" else text[:limit]


_PEM_BEGIN_RE = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----")
_PEM_END_RE = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY(?: BLOCK)?-----")
FREE_TEXT_KEYS = frozenset({"error", "message", "detail", "reason", "traceback", "trace", "stderr", "stdout"})
FREE_TEXT_SUFFIXES = ("_error", "_tail", "_message", "_traceback", "_stderr", "_stdout")


# ===== F-10／C-06：公開レコードの許可値と再帰 fail-closed 検出器（採用済み設計 12_F-10_C-06） =====
# G-01：定数
RAW_OUTPUT_KEYS = frozenset({"child_stderr_tail", "prep_trace_tail", "trace_tail"})
LABEL_KEYS = frozenset({"error", "message", "detail", "reason"})
_STREAM_WORDS = frozenset({"stderr", "stdout", "traceback", "trace"})
_DIGEST_KEYS = frozenset({"redacted", "len", "lines"})
_TYPE_NAME_KEYS = frozenset({"repr_type", "weather_type"})
STRUCTURED_OUTPUT_EXEMPT = frozenset({"stderr_count", "stdout_bytes", "trace_count", "trace_id"})
_CLASS_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_TRACE_ID_RE = re.compile(r"[0-9a-fA-F-]{8,64}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX32 = re.compile(r"[0-9a-f]{32}\Z")
ALLOWED_ERROR_LABELS = frozenset({
    "race_not_in_programs", "prep_isolation_not_verified", "eval_id_mismatch",
    "freeze_config_sha_mismatch", "child_failed", "child_output_missing",
    "child_output_invalid", "child_output_mismatch", "unexpected_error"})
ALLOWED_ODDS_ERROR = frozenset({"empty", "unexpected_error"})
ALLOWED_REASON_LABELS = frozenset({
    "no_captured_cfg", "reload_mismatch", "config_file_changed", "config_file_invalid",
    "config_file_not_equal_captured", "default_file_invalid", "default_file_not_equal_captured",
    "default_copy_not_observed", "config_file_appeared"})
ALLOWED_PARSE_ERROR = frozenset({"not_a_dict", "unicode_decode_error", "value_error"})
CAPTURE_ERRORS_LABELS = frozenset({"unexpected_error", "existing_profiler",
                                   "last_purchase_result_error", "cfg_not_captured"})
ISOLATION_ERRORS_LABELS = frozenset({"another_isolation_active", "setup_interrupted",
                                     "setup_error", "teardown_error", "self_test_failed"})
CHECK_DETAIL_FIXED_LABELS = frozenset({"assembly_check_only",
                                       "input_validation_failed", "assess_validation_failed"})
# 本人判断1：context_keys は LEGACY_CONTEXT_KEYS の部分集合。
# actions.g3_entrypoint.LEGACY_CONTEXT_KEYS（L75-78）の写し（循環 import を避けるため複製。両者は同期必須）。
_LEGACY_CONTEXT_KEYS = frozenset({
    "match_index", "match_index_approx", "race_type", "has_exhibition",
    "market_gap", "upset_score", "ex_rank_1st"})

# パス別ラベル規則（error 集合を使い回さない。パスは _norm_path 正規化後の完全パス）
_LABEL_PATH_RULES = {
    "error": ALLOWED_ERROR_LABELS,
    "odds_meta.error": ALLOWED_ODDS_ERROR,
    "count_candidate_checks.odds_error": ALLOWED_ODDS_ERROR,
    "joblib_prewarm_error": frozenset({"unexpected_error"}),
    "g3.exception": frozenset({"unexpected_error"}),
    "g3.config.source_detail.reason": ALLOWED_REASON_LABELS,
    "g3.config.source_detail.reload_error": frozenset({"unexpected_error"}),
    "g3.config.source_detail.workdir_config_parse_error_start": ALLOWED_PARSE_ERROR,
    "g3.config.source_detail.workdir_default_parse_error_start": ALLOWED_PARSE_ERROR,
}
_ERRORS_PATH_RULES = {
    "g3.capture.errors": CAPTURE_ERRORS_LABELS,
    "g3.isolation.errors": ISOLATION_ERRORS_LABELS,
    "prep_isolation.errors": ISOLATION_ERRORS_LABELS,
}
_TYPE_NAME_PATHS = frozenset({
    "g3.inputs.legacy_boats[].repr_type",
    "g3.evaluate_bets_args.weather.repr_type",
    "g3.evaluate_bets_args.weather_type",
})
_RAW_PATHS = frozenset({"child_stderr_tail", "prep_trace_tail"})

# B（F-10 受入ブロッカー最小修正）：Legacy 自身の整形済み診断文字列を、当該パス限定・型/構造/長さ上限付きで許可。
# 対象は外部プロセス生出力でも例外情報でもなく、公開診断 record の本来の比較対象。
# 生出力3フィールドの digest・例外クラス名/本文の正規化は一切変えない。許可は下記3パスのみ。
_FREETEXT_MAX_DETAIL_ITEMS = 16
_FREETEXT_MAX_DETAIL_KEY = 64
_FREETEXT_MAX_DETAIL_VALUE = 256
_FREETEXT_MAX_MESSAGE = 1024   # 実測ワースト ~512。偽ブロック(fail-closed)回避の余白を確保
_FREETEXT_MAX_REASON = 256
_FREETEXT_MAX_COMBO = 32       # combo は "1-2-3" 形の短い買い目ID
_FREETEXT_MAX_EXCLUDED = 256   # excluded リスト長の上限（無制限を防ぐ）


def _is_str_map_bounded(v: Any) -> bool:
    """str→str の dict（キー数・キー長・値長を上限で制限）。ネスト・非 str 値・自由文スカラーは不許可。"""
    if not isinstance(v, dict) or len(v) > _FREETEXT_MAX_DETAIL_ITEMS:
        return False
    for k, val in v.items():
        if not (isinstance(k, str) and len(k) <= _FREETEXT_MAX_DETAIL_KEY
                and isinstance(val, str) and len(val) <= _FREETEXT_MAX_DETAIL_VALUE):
            return False
    return True


def _is_excluded_item(e: Any) -> bool:
    """excluded の要素は {combo, reason} のみ。combo・reason とも None か上限内 str に限定
    （combo も型・長さを強制し、dict 等での生データ密輸を fail-closed で遮断。非再帰ガードの前提）。"""
    if not isinstance(e, dict) or set(e) != {"combo", "reason"}:
        return False
    combo, reason = e.get("combo"), e.get("reason")
    if not (combo is None or (isinstance(combo, str) and len(combo) <= _FREETEXT_MAX_COMBO)):
        return False
    return reason is None or (isinstance(reason, str) and len(reason) <= _FREETEXT_MAX_REASON)


def _is_excluded_list(v: Any) -> bool:
    return isinstance(v, list) and len(v) <= _FREETEXT_MAX_EXCLUDED and all(_is_excluded_item(e) for e in v)


_FREETEXT_PATH_RULES = {
    "evaluation.upset_result.detail": _is_str_map_bounded,
    "g3.capture.return_value.message":
        lambda v: v is None or (isinstance(v, str) and len(v) <= _FREETEXT_MAX_MESSAGE),
    "g3.capture.return_value.excluded": _is_excluded_list,
}

_SUMMARY_SCHEMA = {
    "race_date": str, "races": int, "outcomes": dict, "child_failures": int,
    "started_at": str, "finished_at": str, "records_count": int, "eval_ids": list,
    "no_target_races": bool, "all_children_failed": bool, "no_odds_count": int,
    "records_sha256": (str, type(None)), "environment": dict, "inputs_sha256": dict,
}


# G-02：digest（原本全体から。本文・ハッシュは残さない）
def _free_text_digest(text: str) -> dict:
    return {"redacted": True, "len": len(text),
            "lines": (text.count("\n") + 1 if text else 0)}


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_finite_number(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return True
    return isinstance(v, float) and math.isfinite(v)


def _is_hash_or_none(v: Any) -> bool:
    return v is None or (isinstance(v, str) and bool(_HEX64.match(v)))


# G-03：digest の厳密検証（3キー・True のみ・len/lines は bool を除く非負整数）
def _is_valid_digest(obj: Any) -> bool:
    if not isinstance(obj, dict) or set(obj) != set(_DIGEST_KEYS):
        return False
    if obj["redacted"] is not True:
        return False
    for k in ("len", "lines"):
        v = obj[k]
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            return False
    return True


def _is_type_name(v: Any) -> bool:
    return isinstance(v, str) and bool(_CLASS_NAME_RE.match(v)) and len(v) <= 64


# G-05：構造化例外の補助値
def _is_valid_exempt(key: str, v: Any) -> bool:
    if key == "trace_id":
        return isinstance(v, str) and bool(_TRACE_ID_RE.match(v))
    return _is_int(v) and v >= 0


# G-04：条件別 check_detail スキーマ（producer 実測の値・型に一致）
_CHECK_DETAIL_SCHEMA = {
    "c1_upset_pre": lambda v: isinstance(v, dict) and set(v) == {"upset", "evaluation"}
    and _is_finite_number(v["upset"]) and _is_finite_number(v["evaluation"]),
    "c3_post_values": lambda v: v == "assembly_check_only",
    "c5_win_probs": lambda v: v == "assembly_check_only",
    "c7_pred_combo_once": lambda v: isinstance(v, dict) and set(v) == {"count"}
    and _is_int(v["count"]) and v["count"] >= 0,
    "c7b_prediction_values": lambda v: isinstance(v, dict)
    and set(v) == {"pred_prob", "pred_ev", "pred_odds"}
    and all(_is_finite_number(v[k]) for k in ("pred_prob", "pred_ev", "pred_odds")),
    "buy_input_valid": lambda v: v in (None, "input_validation_failed"),
    "t13_capture_vs_return": lambda v: v is None,
    "c6_context": lambda v: isinstance(v, dict) and set(v) == {"diff_keys"}
    and isinstance(v["diff_keys"], list)
    and all(x in ("upset_score", "match_index") for x in v["diff_keys"])
    and len(v["diff_keys"]) == len(set(v["diff_keys"])),
    "c6_flags": lambda v: isinstance(v, dict) and set(v) == {"has_exhibition", "match_index_approx"}
    and all(v[k] is None or isinstance(v[k], bool) for k in ("has_exhibition", "match_index_approx")),
    "context_keys": lambda v: isinstance(v, dict) and set(v) == {"keys"}
    and isinstance(v["keys"], list) and all(isinstance(x, str) for x in v["keys"])
    and set(v["keys"]) <= _LEGACY_CONTEXT_KEYS
    and len(v["keys"]) == len(set(v["keys"])),
    "assess_valid": lambda v: v in (None, "assess_validation_failed"),
}


def is_allowed_check_detail(cond_name: str, v: Any) -> bool:
    fn = _CHECK_DETAIL_SCHEMA.get(cond_name)
    return bool(fn) and bool(fn(v))


def _norm_path(p: str) -> str:
    return re.sub(r"\[\d+\]", "[]", p)


# G-06／06a：再帰 fail-closed 検出器（不適合キーの dotted path を昇順・重複除去で返す）
def unclassified_output_keys(obj: Any, _path: str = "", _parent: str = "") -> list:
    bad: list = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            ks = str(k)
            kp = f"{_path}.{ks}" if _path else ks
            np = _norm_path(kp)
            if np in _FREETEXT_PATH_RULES:
                # B：当該パス限定で Legacy 整形文字列を許可。検証済みなので内容へは再帰しない。
                if not _FREETEXT_PATH_RULES[np](v):
                    bad.append(kp)
                continue
            if ks in RAW_OUTPUT_KEYS:
                if np in _RAW_PATHS:
                    if not (v is None or _is_valid_digest(v)):
                        bad.append(kp)
                else:
                    bad.append(kp)
            elif np in _LABEL_PATH_RULES:
                if not (v is None or (isinstance(v, str) and v in _LABEL_PATH_RULES[np])):
                    bad.append(kp)
            elif ks == "check_detail":
                if not is_allowed_check_detail(_parent, v):
                    bad.append(kp)
            elif ks == "errors":
                rule = _ERRORS_PATH_RULES.get(np)
                if rule is None or not (isinstance(v, list)
                                        and all(isinstance(e, str) and e in rule for e in v)):
                    bad.append(kp)
                continue
            elif ks in _TYPE_NAME_KEYS:
                if np not in _TYPE_NAME_PATHS or not _is_type_name(v):
                    bad.append(kp)
            elif ks in ("message", "detail"):
                if v is not None:
                    bad.append(kp)
            elif ks in STRUCTURED_OUTPUT_EXEMPT:
                if not _is_valid_exempt(ks, v):
                    bad.append(kp)
            elif ks in ("error", "reason", "exception", "reload_error") or ks.endswith("_error") \
                    or (_STREAM_WORDS & set(ks.split("_"))):
                bad.append(kp)
            bad.extend(unclassified_output_keys(v, kp, ks))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            bad.extend(unclassified_output_keys(v, f"{_path}[{i}]", _parent))
    return sorted(set(bad))


# G-07：collect 前ゲート（records.jsonl＋summary.json＋races の未分類/欠落/不整合で ok=False）
def scan_output_tree(out_dir: "Path") -> dict:
    problems: list = []
    warnings: list = []
    eids = None
    eid_ok = False
    rec_path, sum_path, races_dir = out_dir / "records.jsonl", out_dir / "summary.json", out_dir / "races"
    if not rec_path.is_file():
        problems.append("records.jsonl:missing")
    if not sum_path.is_file():
        problems.append("summary.json:missing")
    summary = None
    if sum_path.is_file():
        try:
            summary = json.loads(sum_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            problems.append("summary.json:parse_or_read")
    if not isinstance(summary, dict):
        summary = None

    rec_eids: list = []
    raw_lines = 0
    rec_io_ok = False
    rec_clean = True
    if rec_path.is_file():
        try:
            with open(rec_path, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    raw_lines += 1
                    if not line.strip():
                        problems.append(f"records.jsonl[{i}]:blank")
                        rec_clean = False
                        continue
                    try:
                        r = json.loads(line)
                    except ValueError:
                        problems.append(f"records.jsonl[{i}]:parse")
                        rec_clean = False
                        continue
                    if not isinstance(r, dict):
                        problems.append(f"records.jsonl[{i}]:not_object")
                        rec_clean = False
                        continue
                    eid = r.get("eval_id")
                    if not (isinstance(eid, str) and eid):
                        problems.append(f"records.jsonl[{i}]:eval_id")
                        rec_clean = False
                    else:
                        rec_eids.append(eid)
                    u = unclassified_output_keys(r)
                    if u:
                        rec_clean = False
                    problems += [f"records.jsonl[{i}].{p}" for p in u]
            rec_io_ok = True
        except (OSError, UnicodeDecodeError):
            problems.append("records.jsonl:read")
    rec_content_ok = rec_io_ok and rec_clean
    if rec_content_ok and len(rec_eids) != len(set(rec_eids)):
        problems.append("records:duplicate_eval_id")

    if summary is not None:
        for k, t in _SUMMARY_SCHEMA.items():
            if k not in summary:
                problems.append(f"summary:missing:{k}")
            elif t is int:
                if not _is_int(summary[k]):
                    problems.append(f"summary:type:{k}")
            elif t is bool:
                if not isinstance(summary[k], bool):
                    problems.append(f"summary:type:{k}")
            elif not isinstance(summary[k], t):
                problems.append(f"summary:type:{k}")
        if any(k not in _SUMMARY_SCHEMA for k in summary):
            warnings.append("summary:unknown_keys_present")
        if rec_io_ok:
            rs = summary.get("records_sha256")
            if not (isinstance(rs, str) and _HEX64.fullmatch(rs)):
                problems.append("summary.records_sha256:invalid")
            else:
                try:
                    if hashlib.sha256(rec_path.read_bytes()).hexdigest() != rs:
                        problems.append("summary.records_sha256:mismatch")
                except OSError:
                    problems.append("records.jsonl:read")
        eids = summary.get("eval_ids")
        eid_ok = isinstance(eids, list) and all(isinstance(e, str) and e for e in eids)
        if eids is not None and not eid_ok:
            problems.append("summary.eval_ids:invalid")
        if eid_ok and len(eids) != len(set(eids)):
            problems.append("summary.eval_ids:duplicate")
        # no_target_races の意味的整合（producer: no_target_races == (len(races)==0)==(len(eval_ids)==0)）
        if eid_ok and isinstance(summary.get("no_target_races"), bool):
            if summary["no_target_races"] != (len(eids) == 0):
                problems.append("summary.no_target_races:inconsistent")
        if rec_content_ok:
            rc, rz = summary.get("records_count"), summary.get("races")
            if _is_int(rc) and rc != raw_lines:
                problems.append("summary.records_count:mismatch")
            if eid_ok and _is_int(rc) and rc != len(eids):
                problems.append("records_count_vs_eval_ids:mismatch")
            if eid_ok and _is_int(rz) and rz != len(eids):
                problems.append("races_count_vs_eval_ids:mismatch")
            if eid_ok and sorted(rec_eids) != sorted(eids):
                problems.append("records_vs_eval_ids:mismatch")
        env = summary.get("environment")
        if isinstance(env, dict):
            for req4 in ("python", "packages", "code_sha256", "fan_files"):
                if req4 not in env:
                    problems.append(f"environment:missing:{req4}")
            if "python" in env and not isinstance(env["python"], str):
                problems.append("environment.python:type")
            for sub in ("packages", "code_sha256", "fan_files"):
                d = env.get(sub)
                if sub in env and not isinstance(d, dict):
                    problems.append(f"environment.{sub}:type")
                elif isinstance(d, dict):
                    for v in d.values():
                        if sub == "packages":
                            if not (v is None or isinstance(v, str)):
                                problems.append(f"environment.{sub}:value")
                                break
                        elif not _is_hash_or_none(v):
                            problems.append(f"environment.{sub}:hash")
                            break
        ins = summary.get("inputs_sha256")
        if isinstance(ins, dict):
            for req in ("programs_raw", "odds"):
                if req not in ins:
                    problems.append(f"inputs_sha256:missing:{req}")
            for v in ins.values():
                if not _is_hash_or_none(v):
                    problems.append("inputs_sha256:hash")
                    break
        oc = summary.get("outcomes")
        if isinstance(oc, dict):
            for k, v in oc.items():
                if not (isinstance(k, str) and _is_int(v) and v >= 0):
                    problems.append("outcomes:shape")
                    break

    expected = set(eids) if eid_ok else set()
    if races_dir.is_symlink() or (races_dir.exists() and not races_dir.is_dir()):
        problems.append("races:not_directory")
    elif not races_dir.exists():
        if eid_ok and eids:
            problems.append("races:dir_missing")
    else:
        seen: dict = {}
        for p in sorted(races_dir.iterdir()):
            if p.is_symlink() or not p.is_file() or p.suffix != ".json":
                problems.append(f"races:unexpected_entry:{p.name}")
                continue
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                problems.append(f"races/{p.name}:parse_or_read")
                continue
            if not isinstance(obj, dict):
                problems.append(f"races/{p.name}:not_object")
                continue
            reid, rnonce = obj.get("eval_id"), obj.get("nonce")
            valid = True
            if not (isinstance(reid, str) and reid):
                problems.append(f"races/{p.name}:eval_id")
                valid = False
            if not (isinstance(rnonce, str) and _HEX32.fullmatch(rnonce)):
                problems.append(f"races/{p.name}:nonce")
                valid = False
            problems += [f"races/{p.name}.{q}" for q in unclassified_output_keys(obj)]
            if valid and p.name != f"{reid}.{rnonce}.json":
                problems.append(f"races/{p.name}:name_mismatch")
                valid = False
            if valid and eid_ok and reid not in expected:
                problems.append(f"races:orphan:{reid}")
                valid = False
            if valid:
                seen[reid] = seen.get(reid, 0) + 1
        for reid, n in seen.items():
            if n > 1:
                problems.append(f"races:duplicate_eval_id:{reid}")

    return {"ok": not problems, "problems": sorted(set(problems)), "warnings": sorted(set(warnings))}


def redact_lines(lines: list[str]) -> list[str]:
    """F-10：行ごとに伏せ字をかける。秘密鍵の始まりの行から終わりの行までは、行をまたいで丸ごと伏せる。"""
    out: list[str] = []
    in_key = False
    for line in lines:
        if in_key:
            out.append("***")
            if _PEM_END_RE.search(line):
                in_key = False
            continue
        if _PEM_BEGIN_RE.search(line) and not _PEM_END_RE.search(line):
            in_key = True
        out.append(redact_text(line, 1 << 20))
    return out


def redact_free_text(obj: Any, _key: str = "") -> Any:
    """F-10：記録の中の自由な文（誤り・末尾・文など）に伏せ字をかける。値の欄（eval_id・SHA-256 など）は変えない。"""
    if isinstance(obj, dict):
        return {k: redact_free_text(v, str(k)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_free_text(v, _key) for v in obj]
    if isinstance(obj, str) and _key in RAW_OUTPUT_KEYS:  # G-08：生出力は本文を残さず digest に
        return _free_text_digest(obj)
    if isinstance(obj, str) and (_key in FREE_TEXT_KEYS or _key.endswith(FREE_TEXT_SUFFIXES)):
        return "\n".join(redact_lines(obj.split("\n")))
    return obj


class RedactingStream:
    """F-10：標準出力・標準エラーに書く文字を、行ごとに伏せ字にしてから渡す（ログ・print・例外の表示を含む）。"""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._buf = ""
        self._in_key = False

    def _emit(self, line: str, newline: bool) -> None:
        if self._in_key:
            text = "***"
            if _PEM_END_RE.search(line):
                self._in_key = False
        else:
            if _PEM_BEGIN_RE.search(line) and not _PEM_END_RE.search(line):
                self._in_key = True
            text = redact_text(line, 1 << 20)
        self._inner.write(text + ("\n" if newline else ""))

    def write(self, s: str) -> int:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._emit(line, True)
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self._emit(self._buf, False)
            self._buf = ""
        self._inner.flush()

    def isatty(self) -> bool:
        return False

    def fileno(self) -> int:
        return self._inner.fileno()

    @property
    def encoding(self) -> str:
        return getattr(self._inner, "encoding", "utf-8")


def install_redacting_streams() -> None:
    """プロセスの標準出力・標準エラーを RedactingStream にする（二重にはしない）。"""
    import atexit

    for name in ("stdout", "stderr"):
        cur = getattr(sys, name)
        if not isinstance(cur, RedactingStream):
            wrapped = RedactingStream(cur)
            setattr(sys, name, wrapped)
            atexit.register(wrapped.flush)


def _redact(text: str) -> str:
    return redact_text(text, 200, "head")


def _summarize_arg(a: Any, _depth: int = 0) -> Any:
    if isinstance(a, bool) or a is None or isinstance(a, (int, float)):
        return a
    if isinstance(a, str):
        return _redact(a)
    if isinstance(a, bytes):
        return f"bytes[{len(a)}]"
    if isinstance(a, os.PathLike):
        return _redact(os.fspath(a) if isinstance(os.fspath(a), str) else "bytes-path")
    if isinstance(a, tuple) and _depth == 0 and len(a) <= 4:
        return [_summarize_arg(x, 1) for x in a]  # 例：socket の宛先 (host, port)
    if hasattr(a, "full_url"):  # urllib.request.Request
        return _redact(str(getattr(a, "full_url", "")))
    return type(a).__name__


def env_name_is_sensitive(name: str) -> bool:
    """境界の中で外す名前。F-10：取得の段で「止める」名前（env_name_must_refuse）は、必ず外す側にも入る。"""
    return (bool(_ENV_NAME_PATTERN.search(name)) or name.startswith(_ENV_PREFIXES) or name in _ENV_EXACT
            or env_name_must_refuse(name))


def _leftover_sensitive_env() -> list[str]:
    """境界が自分で設定した変数（HOME・NETRC など。値は境界の中のパス・接続できない宛先）を除く残り。"""
    return sorted(k for k in os.environ if env_name_is_sensitive(k) and k not in _ENV_SET_KEYS)


class G3Isolation:
    """G3 の診断で Legacy を呼ぶ間だけ張る境界。`with` で使う。

    seed_files: 作業フォルダに置くファイル（名前 → 元のファイルのパス、または中身の bytes）。
                既定は空（何も置かない）。何を置くかは呼び出し側が決め、記録に残る。
    keep_workdir: True なら終了後も作業フォルダを消さない（調べるため）。既定は消す。
    """

    def __init__(self, seed_files: Optional[Mapping[str, SeedSource]] = None, *,
                 keep_workdir: bool = False) -> None:
        self.seed_files = dict(seed_files or {})
        self.keep_workdir = keep_workdir
        self.phase = "setup"
        self.established = False
        self.verified = False
        self.errors: list[str] = []
        self.self_test: dict[str, bool] = {}
        self.release_calls: list[dict[str, Any]] = []
        self.blocked_events: list[dict[str, Any]] = []
        self.env_removed_names: list[str] = []
        self.patched: list[str] = []
        self.workdir: Optional[Path] = None
        self.workdir_files_start: dict[str, str] = {}
        self.workdir_files_end: dict[str, str] = {}
        self.seeded: dict[str, str] = {}
        self.legacy_modules_loaded_before: list[str] = []
        self.post_check: dict[str, bool] = {}
        self._outside_probe_dir: Optional[Path] = None
        self._saved_cwd: Optional[str] = None
        self._saved_env: dict[str, Optional[str]] = {}
        self._saved_attrs: list[tuple[Any, str, Any]] = []
        self._stubs: dict[str, Any] = {}
        self._saved_tempdir: Any = None
        self._tempdir_saved = False
        self._torn_down = False
        self._saved_dont_write_bytecode: Optional[bool] = None
        self._entered = False

    # ---- 層2 の判定 ----
    def _record_block(self, kind: str, event: str, detail: Any) -> None:
        self.blocked_events.append({"phase": self.phase, "kind": kind, "event": event,
                                    "target": detail})  # G-09：detail→target（A-06 伏せ字は _summarize_arg 済）

    def _inside_workdir(self, p: Any) -> bool:
        if isinstance(p, int) or p is None or self.workdir is None:
            return True  # ファイル記述子（すでに開いたもの）は open の時点で判定済み
        try:
            s = os.fsdecode(p) if isinstance(p, (bytes, os.PathLike)) else str(p)
            full = os.path.realpath(os.path.join(os.getcwd(), s))
            root = os.path.realpath(str(self.workdir))
            return full == root or full.startswith(root + os.sep)
        except Exception:  # noqa: BLE001 — 判定できないパスは外とみなす
            return False

    def _on_event(self, event: str, args: tuple) -> None:
        if event in _NETWORK_EVENTS:
            # socket.* の最初の引数はソケット自身なので宛先から記録する。urllib.Request は URL を伏せ字にする
            first = 1 if event.startswith("socket.") and event not in (
                "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
                "socket.gethostbyname_ex") else 0
            self._record_block("network", event, [_summarize_arg(a) for a in args[first:first + 2]])
            raise G3IsolationBlocked(f"G3 isolation: {event} blocked")
        if event in _PROCESS_EVENTS:
            self._record_block("process", event, [_summarize_arg(a) for a in args[:1]])
            raise G3IsolationBlocked(f"G3 isolation: {event} blocked")
        if event == "open":
            path, mode, flags = (list(args) + [None, None, None])[:3]
            writing = (isinstance(mode, str) and any(ch in mode for ch in "wax+")) or \
                      (isinstance(flags, int) and bool(flags & _WRITE_FLAGS))
            if writing and not self._inside_workdir(path):
                self._record_block("write_outside", event, _summarize_arg(path))
                raise G3IsolationBlocked(f"G3 isolation: write outside workdir blocked")
            return
        idx = _PATH_WRITE_EVENTS.get(event)
        if idx is not None:
            for i in idx:
                if i < len(args) and not self._inside_workdir(args[i]):
                    self._record_block("write_outside", event, _summarize_arg(args[i]))
                    raise G3IsolationBlocked(f"G3 isolation: {event} outside workdir blocked")

    # ---- 層1 ----
    def _make_stub(self, name: str):
        ret = _STUB_RETURNS.get(name)

        def stub(*args: Any, **kwargs: Any) -> Any:
            self.release_calls.append({
                "phase": self.phase, "function": name,
                "args": [_summarize_arg(a) for a in args],
                "kwargs": {k: _summarize_arg(v) for k, v in kwargs.items()},
                "blocked": True,
            })
            return ret
        stub.__name__ = f"g3_blocked_{name}"
        stub._g3_isolation_stub = True  # type: ignore[attr-defined]
        return stub

    def _patch_release(self) -> None:
        mod = importlib.import_module(RELEASE_MODULE)
        originals = {}
        for name, obj in list(vars(mod).items()):
            if inspect.isfunction(obj) and getattr(obj, "__module__", None) == RELEASE_MODULE:
                originals[id(obj)] = name
                stub = self._make_stub(name)
                self._stubs[name] = stub
                self._saved_attrs.append((mod, name, obj))
                setattr(mod, name, stub)
                self.patched.append(f"{RELEASE_MODULE}.{name}")
        for attr in _RELEASE_URL_ATTRS:
            if hasattr(mod, attr):
                self._saved_attrs.append((mod, attr, getattr(mod, attr)))
                setattr(mod, attr, _DEAD_URL)
        # 他のモジュールが `from x_release_storage import download_file` のように取り込んだ写し
        for mname, m in list(sys.modules.items()):
            if m is None or m is mod:
                continue
            try:
                items = list(vars(m).items())
            except Exception:  # noqa: BLE001 — 属性を列挙できないモジュールは対象外
                continue
            for aname, aval in items:
                name = originals.get(id(aval)) if inspect.isfunction(aval) else None
                if name is not None:
                    self._saved_attrs.append((m, aname, aval))
                    setattr(m, aname, self._stubs[name])
                    self.patched.append(f"{mname}.{aname}")
        self._release_module = mod

    # ---- 層3 ----
    def _scrub_env(self) -> None:
        for k in list(os.environ):
            if env_name_is_sensitive(k):
                self._saved_env.setdefault(k, os.environ.get(k))
                del os.environ[k]
                self.env_removed_names.append(k)
        assert self.workdir is not None
        home = self.workdir / ".g3_home"
        tmp = self.workdir / ".g3_tmp"
        home.mkdir()
        tmp.mkdir()
        values = {
            "HOME": str(home), "TMPDIR": str(tmp), "TEMP": str(tmp), "TMP": str(tmp),
            "HTTP_PROXY": _DEAD_URL, "HTTPS_PROXY": _DEAD_URL, "ALL_PROXY": _DEAD_URL,
            "http_proxy": _DEAD_URL, "https_proxy": _DEAD_URL, "all_proxy": _DEAD_URL,
            "NO_PROXY": "", "no_proxy": "", "NETRC": str(home / ".netrc_absent"),
            "XDG_CONFIG_HOME": str(home), "XDG_CACHE_HOME": str(home),
        }
        for k in _ENV_SET_KEYS:
            self._saved_env.setdefault(k, os.environ.get(k))
            os.environ[k] = values[k]
        self.env_removed_names.sort()
        self._saved_tempdir = tempfile.tempdir
        self._tempdir_saved = True
        tempfile.tempdir = str(tmp)

    def _restore_env(self) -> None:
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self._saved_env.clear()
        if self._tempdir_saved:  # A-08：自分が変えたときだけ戻す（入れ子で不成立の側が None にしない）
            tempfile.tempdir = self._saved_tempdir
            self._tempdir_saved = False

    # ---- 層4 ----
    def _snapshot(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.workdir is None:
            return out
        for p in sorted(self.workdir.rglob("*")):
            if p.is_file():
                out[p.relative_to(self.workdir).as_posix()] = _sha256_file(p)
        return out

    def _seed(self) -> None:
        assert self.workdir is not None
        for name, src in self.seed_files.items():
            if os.path.isabs(name) or ".." in Path(name).parts:
                raise ValueError(f"seed name must be relative: {name!r}")
            dest = self.workdir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            if isinstance(src, bytes):
                dest.write_bytes(src)
            else:
                shutil.copyfile(os.fspath(src), dest)
            self.seeded[name] = _sha256_file(dest)

    # ---- 自己試験 ----
    def _expect_blocked(self, name: str, fn) -> None:
        before = len(self.blocked_events)
        try:
            fn()
            ok = False
        except G3IsolationBlocked:
            ok = len(self.blocked_events) > before
        except Exception:  # noqa: BLE001 — 境界以外の理由で失敗したら試験不合格
            ok = False
        self.self_test[name] = ok

    def _run_self_test(self) -> None:
        self.phase = "self_test"

        def _connect():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.settimeout(0.2)
                s.connect(("127.0.0.1", 9))
            finally:
                s.close()
        self._expect_blocked("socket_connect", _connect)
        self._expect_blocked("getaddrinfo", lambda: socket.getaddrinfo("localhost", 443))
        self._expect_blocked("subprocess", lambda: subprocess.run(["true"], check=False))
        self._expect_blocked("os_system", lambda: os.system("true"))
        assert self._outside_probe_dir is not None
        outside = self._outside_probe_dir / "probe.txt"
        self._expect_blocked("write_outside", lambda: open(outside, "w").close())
        self._expect_blocked("remove_outside", lambda: os.remove(self._outside_probe_dir / "keep.txt"))
        try:
            import requests  # noqa: PLC0415 — Legacy と同じ経路を試す

            # requests は止めた例外を自分の例外に包むので、「例外になり、層2 が記録した」ことで判定する
            before = len(self.blocked_events)
            try:
                requests.get("http://127.0.0.1:9/g3-isolation-self-test", timeout=1)
                self.self_test["requests"] = False
            except Exception:  # noqa: BLE001
                self.self_test["requests"] = any(
                    e["kind"] == "network" for e in self.blocked_events[before:])
        except ImportError:
            self.self_test["requests"] = False
        # 中は書ける
        try:
            p = Path("g3_probe_inside.txt")
            p.write_text("x")
            p.unlink()
            self.self_test["write_inside"] = True
        except Exception:  # noqa: BLE001
            self.self_test["write_inside"] = False
        # 層1：Release の関数は差し替えた関数になっている
        n = len(self.release_calls)
        mod = self._release_module
        r1 = mod.download_file("g3_probe", "g3_probe")
        r2 = mod.upload_file("g3_probe", "g3_probe")
        self.self_test["release_stub"] = (r1 is False and r2 is False and
                                          len(self.release_calls) == n + 2 and
                                          not Path("g3_probe").exists())
        # 層3：トークンらしい名前の変数が残っていない
        self.self_test["env_scrubbed"] = not _leftover_sensitive_env()
        # 自己試験の記録は実行の記録と分ける
        self._self_test_blocked = [e for e in self.blocked_events if e["phase"] == "self_test"]
        self._self_test_release = [c for c in self.release_calls if c["phase"] == "self_test"]
        self.blocked_events = [e for e in self.blocked_events if e["phase"] != "self_test"]
        self.release_calls = [c for c in self.release_calls if c["phase"] != "self_test"]

    # ---- 開始・終了 ----
    def __enter__(self) -> "G3Isolation":
        global _ACTIVE
        if self._entered:
            raise RuntimeError("G3Isolation is single-use")
        self._entered = True
        try:
            _ensure_hook()
            if _ACTIVE is not None:
                self.errors.append("another_isolation_active")
                return self
            self.legacy_modules_loaded_before = sorted(
                m for m in ("notify_arashi", "x_buyscore", RELEASE_MODULE, "x_asahi_scoring",
                            "x_venue_stats", "odds_fetch") if m in sys.modules)
            self.workdir = Path(tempfile.mkdtemp(prefix="g3iso_"))
            self._outside_probe_dir = Path(tempfile.mkdtemp(prefix="g3iso_outside_"))
            (self._outside_probe_dir / "keep.txt").write_text("keep")
            self._seed()
            self.workdir_files_start = self._snapshot()
            self._saved_cwd = os.getcwd()
            self._patch_release()
            self._scrub_env()
            self._saved_dont_write_bytecode = sys.dont_write_bytecode
            sys.dont_write_bytecode = True
            os.chdir(self.workdir)
            _ACTIVE = self
            self._run_self_test()
            self.established = bool(self.self_test) and all(self.self_test.values())
            if not self.established:
                self.errors.append("self_test_failed")  # G-10：失敗名は公開レコードに残さない
        except Exception as exc:  # noqa: BLE001 — 境界を張れなければ不成立として記録
            self.errors.append("setup_error")  # G-10：例外クラス名を格納しない
            self.established = False
        except BaseException:
            # A-08：KeyboardInterrupt など。with の本体に入らないので __exit__ は呼ばれない。ここで戻す
            self.established = False
            self.errors.append("setup_interrupted")
            self._teardown()
            raise
        self.phase = "run"
        return self

    def _post_checks(self) -> None:
        mod = getattr(self, "_release_module", None)
        stubs_intact = mod is not None and all(
            getattr(mod, n, None) is s for n, s in self._stubs.items())
        same_module = mod is not None and sys.modules.get(RELEASE_MODULE) is mod
        xb = sys.modules.get("x_buyscore")
        na = sys.modules.get("notify_arashi")
        legacy_refs = all(getattr(m, RELEASE_MODULE, mod) is mod for m in (xb, na) if m is not None)
        self.post_check = {
            "release_stubs_intact": bool(stubs_intact),
            "release_module_identity": bool(same_module),
            "legacy_uses_patched_module": bool(legacy_refs),
            "hook_active": _ACTIVE is self,
            "env_still_scrubbed": not _leftover_sensitive_env(),
            "cwd_is_workdir": self.workdir is not None and
                              os.path.realpath(os.getcwd()) == os.path.realpath(str(self.workdir)),
        }

    def _teardown(self) -> None:
        """A-08：何度呼んでもよく、途中の手順が例外になっても残りの復元を続ける。"""
        global _ACTIVE
        if self._torn_down:
            return
        self._torn_down = True
        errors: list[BaseException] = []

        def step(fn) -> None:
            try:
                fn()
            except BaseException as exc:  # noqa: BLE001 — 復元は最後まで続ける
                errors.append(exc)
                self.errors.append("teardown_error")  # G-10：例外クラス名を格納しない

        if _ACTIVE is self:
            self.phase = "teardown"
            step(self._post_checks)
            step(lambda: setattr(self, "workdir_files_end", self._snapshot()))
            _ACTIVE = None

        def restore_attrs() -> None:
            while self._saved_attrs:
                obj, name, val = self._saved_attrs.pop()
                setattr(obj, name, val)
        step(restore_attrs)
        step(self._restore_env)

        def restore_bytecode() -> None:
            if self._saved_dont_write_bytecode is not None:
                sys.dont_write_bytecode = self._saved_dont_write_bytecode
        step(restore_bytecode)

        def restore_cwd() -> None:
            if self._saved_cwd is not None:
                os.chdir(self._saved_cwd)
        step(restore_cwd)

        def remove_dirs() -> None:
            if not self.keep_workdir and self.workdir is not None:
                shutil.rmtree(self.workdir, ignore_errors=True)
            if self._outside_probe_dir is not None:
                shutil.rmtree(self._outside_probe_dir, ignore_errors=True)
        step(remove_dirs)
        self.verified = (self.established and not self.blocked_events and not errors and
                         bool(self.post_check) and all(self.post_check.values()))
        for exc in errors:  # KeyboardInterrupt などは、復元を終えてから伝える
            if not isinstance(exc, Exception):
                raise exc

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._teardown()
        return False

    # ---- 記録 ----
    def release_calls_of(self, function: str, filename: Optional[str] = None,
                         phase: Optional[str] = None) -> list[dict[str, Any]]:
        out = []
        for c in self.release_calls:
            if c["function"] != function or (phase is not None and c["phase"] != phase):
                continue
            if filename is not None and (not c["args"] or c["args"][0] != filename):
                continue
            out.append(c)
        return out

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = {}
        for c in self.release_calls:
            kinds[c["function"]] = kinds.get(c["function"], 0) + 1
        return {
            "established": self.established,
            "verified": self.verified,
            "errors": list(self.errors),
            "self_test": dict(self.self_test),
            "post_check": dict(self.post_check),
            "release_calls": list(self.release_calls),
            "release_calls_by_function": kinds,
            # 観測した事実だけを書く（層2 が止めた件数。0件は「この境界の中で観測されなかった」の意味）
            "network_attempts_observed": sum(1 for e in self.blocked_events if e["kind"] == "network"),
            "process_attempts_observed": sum(1 for e in self.blocked_events if e["kind"] == "process"),
            "write_outside_attempts_observed": sum(
                1 for e in self.blocked_events if e["kind"] == "write_outside"),
            "blocked_events": list(self.blocked_events),
            "blocked_event_count": len(self.blocked_events),
            "env_removed_names": list(self.env_removed_names),
            "patched": list(self.patched),
            "seeded": dict(self.seeded),
            "workdir_files_start": dict(self.workdir_files_start),
            "workdir_files_end": dict(self.workdir_files_end),
            "legacy_modules_loaded_before": list(self.legacy_modules_loaded_before),
        }
