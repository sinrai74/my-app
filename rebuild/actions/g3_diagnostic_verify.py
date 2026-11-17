"""
正式G3（B案）段階3：診断の段の外からの証拠（strace）を検査し、各レースの記録に結果を入れる（B-02）。

root が `python -I -P -B <BASE/src の写し>/rebuild/actions/g3_diagnostic_verify.py` で動かす（再監査 F-05・F-10）。
標準ライブラリだけを使う。

- Python の監査フックは完全な境界ではない（再監査 B-02）。診断の段は OS の分離（unshare の net・pid・mount）で
  動かし、strace の記録を独立の証拠として、ここで検査する。
- strace の前提：`strace -f -qq -yy -s 256 -e trace=%network,execve,execveat,<名前空間などの呼び出し> -o <file>`。
- 合格の条件（ok）：
  1. 通信の呼び出しが、AF_UNIX・AF_NETLINK のソケット以外で0件。種類を判定できない呼び出しも不合格。
     ただし非ローカルの「失敗した bind」（送信元アドレス割当の失敗。例：netns 下で EADDRNOTAVAIL）は、
     データ授受も listen も成立せず通信にならないため、network_bind_failed に記録するが合否には含めない
     （connect・send 系・listen・accept・成功 bind・結果不明は従来どおり不合格）。
  2. AF_UNIX の connect の成功が0件（パス・抽象を問わない）。失敗した宛先は一覧に記録する。
  3. AF_UNIX の bind・listen・accept・accept4 の成功が0件（再監査 F-06。パス・抽象を問わない。socketpair は認める）。
  4. AF_UNIX の宛先へ送る sendto・sendmsg・sendmmsg が0件（E-05）。
  5. 起動（execve・execveat）の成功は、許可した絶対パス（--allow-exec）だけ。結果を判定できない起動は不合格。
  6. 名前空間・マウントなどの呼び出し（unshare・setns・mount 系・ptrace・bpf）の成功が0件。
     CLONE_NEW* か CLONE_UNTRACED の旗を持つ clone・clone3、io_uring_setup は、試しただけで不合格（再監査 F-03）。
  7. 許可した python の起動が1件以上ある（空の記録を合格にしない）。
- 証拠の照合（E-02・F-08）：
  - 状態（public/status.txt）と sysctl の記録（public/sysctl.txt）は root の持ち物の通常のファイルで、
    state=ok・sysctl_state=restored であること。ok でなければ、out の中のファイルは1つも開かずに不合格（F-02・F-03）。
  - strace の記録の SHA-256 は、root が書いた strace.sha256 の、絶対パスが完全に一致するちょうど1行と照合する。
  - records.jsonl・summary.json は、root が作った一覧（out.list）の "./records.jsonl"・"./summary.json" の
    ちょうど1件ずつと、種類（通常のファイル）・大きさ・SHA-256 を照合する。開くときはリンクをたどらない。
- 伏せ字（F-10）：生の strace の記録は root だけが読める場所に残し、外に出すのは伏せ字にした写しだけ。
  伏せ字にした写しにも同じ判定をかけ、生の記録の判定と一致しなければ不合格（判定に要る情報を失わないこと）。
- 限界（limits に書く）：strace は追跡の外の動きを写さない。外への壁は unshare --net で、strace は意図しない
  通信の検出のための独立の証拠。SHA-256 が示すのは「凍結の後に変わっていない」ことだけ（F-09）。
- 終了コード：0 合格、2 strace・証拠・状態の不合格、3 記録の不整合・欠落・壊れた行。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

NET_CALLS = ("connect", "bind", "listen", "accept", "accept4", "sendto", "sendmsg", "sendmmsg")
EXEC_CALLS = ("execve", "execveat")
NS_CALLS = ("unshare", "setns", "mount", "umount2", "pivot_root", "open_tree", "move_mount", "fsopen",
            "fsmount", "fsconfig", "mount_setattr", "ptrace", "bpf")
CLONE_CALLS = ("clone", "clone3")
URING_CALLS = ("io_uring_setup",)
CALLS = NET_CALLS + EXEC_CALLS + NS_CALLS + CLONE_CALLS + URING_CALLS
# strace に記録させる呼び出し（台本と同じ。静的なテストで照合する）
STRACE_TRACE = "%network,execve,execveat," + ",".join(NS_CALLS + CLONE_CALLS + URING_CALLS)
LINE_RE = re.compile(r"^(?:\[pid\s+)?(\d+)\]?\s+(.*)$")
START_RE = re.compile(r"^(" + "|".join(sorted(CALLS, key=len, reverse=True)) + r")\((.*)$")
RESUMED_RE = re.compile(r"^<\.\.\.\s+(" + "|".join(sorted(CALLS, key=len, reverse=True))
                        + r")\s+resumed>\s*(?:<pid changed to (\d+) \.\.\.>)?(.*)$")
PID_CHANGED_RE = re.compile(r"<pid changed to (\d+) \.\.\.>\s*$")
# D-01：戻り値は、引用符の中身を消した本文の「最後の」") = N" から読む
RESULT_RE = re.compile(r"\)\s*=\s*(-?\d+|\?)")
QUOTED_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
FAMILY_RE = re.compile(r"sa_family=(AF_[A-Z0-9]+)")
FD_KIND_RE = re.compile(r"^\s*\d+<([A-Za-z0-9]+)(?:-[A-Za-z0-9]+)?(?::|>)")
CLONE_BAD_RE = re.compile(r"\bCLONE_(?:NEW[A-Z]+|UNTRACED)\b")
LOCAL_FAMILIES = frozenset({"AF_UNIX", "AF_LOCAL", "AF_NETLINK"})
UNIX_FAMILIES = frozenset({"AF_UNIX", "AF_LOCAL"})
LOCAL_FD_KINDS = frozenset({"UNIX", "NETLINK"})
LIMITS = ["processes outside the tracer (if any) are not recorded by strace",
          "the network wall is unshare --net; strace is independent evidence of unintended access",
          "sha256 shows only that files did not change after the freeze, not that they were correct before it",
          "the raw strace log stays in a root-only folder on the runner; only the redacted copy and both "
          "sha256 values leave the runner"]
JUDGEMENT_KEYS = ("ok", "lines", "calls", "network_local_count", "python_seen")
JUDGEMENT_LISTS = ("network_non_local", "network_unknown_family", "network_bind_failed",
                   "unix_connect_succeeded", "unix_server_succeeded",
                   "unix_send_to_path", "unix_connect_failed", "exec_allowed", "exec_not_allowed", "exec_unresolved",
                   "namespace_calls", "clone_flags_rejected", "io_uring_calls")


def _strip_quoted(args: str) -> str:
    """送信データなどの文字列の中身を消す（C-08）。"""
    return QUOTED_RE.sub('""', args)


def _join_lines(lines: Iterable[str]) -> tuple[list[tuple[str, str, str]], list[str]]:
    """(pid, 呼び出し名, 引数〜結果) の一覧。unfinished と resumed を pid ごとにつなぐ（C-01）。"""
    pending: dict[tuple[str, str], str] = {}
    calls: list[tuple[str, str, str]] = []
    for raw in lines:
        m = LINE_RE.match(raw.rstrip("\n"))
        if not m:
            continue
        pid, rest = m.group(1), m.group(2)
        r = RESUMED_RE.match(rest)
        if r:
            name = r.group(1)
            head = pending.pop((pid, name), None)
            if head is None and r.group(2):
                # strace 6.8：スレッドが execve すると、resumed の行は新しい pid（"pid changed to"）になる。
                # 同じ呼び出しの unfinished がちょうど1件のときだけ、それとつなぐ
                same = [k for k in pending if k[1] == name]
                if len(same) == 1:
                    head = pending.pop(same[0])
            calls.append((pid, name, (head if head is not None else "\x00missing-start\x00") + r.group(3)))
            continue
        s = START_RE.match(rest)
        if not s:
            continue
        name, body = s.group(1), s.group(2)
        if body.rstrip().endswith("<unfinished ...>"):
            pending[(pid, name)] = body.rstrip()[: -len("<unfinished ...>")]
            continue
        changed = PID_CHANGED_RE.search(body)
        if changed:
            pending[(changed.group(1), name)] = body[: changed.start()]
            continue
        calls.append((pid, name, body))
    unresolved = [f"{pid} {name}({body[:120]}" for (pid, name), body in pending.items()]
    return calls, unresolved


def _succeeded(ret: Optional[str]) -> Optional[bool]:
    """戻り値が成功か。判定できなければ None。"""
    if ret is None or ret == "?":
        return None
    return not ret.startswith("-")


def analyze(lines: Iterable[str], allow_exec: Sequence[str]) -> dict[str, Any]:
    allow = {str(Path(p)) for p in allow_exec}
    lines = list(lines)
    calls, unresolved = _join_lines(lines)
    net_bad: list[str] = []
    net_unknown: list[str] = []
    net_bind_failed: list[str] = []
    unix_connect_ok: list[str] = []
    unix_server_ok: list[str] = []
    unix_send_to_path: list[str] = []
    unix_connect_failed: list[str] = []
    ns_calls: list[str] = []
    clone_bad: list[str] = []
    uring: list[str] = []
    net_local = 0
    exec_ok: list[str] = []
    exec_bad: list[str] = []
    exec_unknown: list[str] = []
    for pid, name, body in calls:
        plain = _strip_quoted(body)
        results = RESULT_RE.findall(plain)
        ret = results[-1] if results else None
        line = f"{pid} {name}({body.strip()[:200]}"
        if name in NET_CALLS:
            fam = FAMILY_RE.search(plain)
            fdk = FD_KIND_RE.match(plain)
            kind_local = None
            if fdk:
                kind_local = fdk.group(1).upper() in LOCAL_FD_KINDS
            if fam:
                fam_local = fam.group(1) in LOCAL_FAMILIES
                local = fam_local and kind_local is not False
            else:
                local = kind_local
            if local is None:
                net_unknown.append(line)
            elif not local:
                # 非ローカルの「失敗した bind」（送信元アドレス割当の失敗。例：netns 下で EADDRNOTAVAIL）は
                # データ授受も listen も成立せず、通信にはならない。観測として記録するが合否には含めない。
                # connect・send*・listen・accept・成功 bind・結果不明は従来どおり違反（net_bad）として扱う。
                if name == "bind" and _succeeded(ret) is False:
                    net_bind_failed.append(line)
                else:
                    net_bad.append(line)
            else:
                net_local += 1
                is_unix = (fam is not None and fam.group(1) in UNIX_FAMILIES) or \
                          (fdk is not None and fdk.group(1).upper() == "UNIX")
                if name in ("sendto", "sendmsg", "sendmmsg") and fam is not None and fam.group(1) in UNIX_FAMILIES:
                    path = QUOTED_RE.search(body.split(fam.group(0), 1)[1])
                    unix_send_to_path.append(path.group(1) if path else "?")
                if name == "connect":
                    path = QUOTED_RE.search(body)
                    target = path.group(1) if path else "?"
                    (unix_connect_ok if _succeeded(ret) is not False else unix_connect_failed).append(target)
                if name in ("bind", "listen", "accept", "accept4") and is_unix and _succeeded(ret) is not False:
                    unix_server_ok.append(line)
        elif name in EXEC_CALLS:
            q = QUOTED_RE.search(body)
            path = q.group(1) if q else None
            if "\x00missing-start\x00" in body:
                path = None
            if ret is None or ret == "?" or path is None:
                exec_unknown.append(line)
            elif ret == "0":
                (exec_ok if path in allow else exec_bad).append(path)
        elif name in NS_CALLS:
            if _succeeded(ret) is not False or "\x00missing-start\x00" in body:
                ns_calls.append(line)
        elif name in CLONE_CALLS:
            if CLONE_BAD_RE.search(plain) or "\x00missing-start\x00" in body:
                clone_bad.append(line)
        elif name in URING_CALLS:
            uring.append(line)
    exec_unknown += [u for u in unresolved if u.split(" ", 1)[1].startswith(EXEC_CALLS)]
    net_unknown += [u for u in unresolved if u.split(" ", 1)[1].startswith(NET_CALLS)]
    ns_calls += [u for u in unresolved if u.split(" ", 1)[1].startswith(NS_CALLS)]
    clone_bad += [u for u in unresolved if u.split(" ", 1)[1].startswith(CLONE_CALLS)
                  and CLONE_BAD_RE.search(_strip_quoted(u))]
    uring += [u for u in unresolved if u.split(" ", 1)[1].startswith(URING_CALLS)]
    python_seen = any("python" in Path(p).name for p in exec_ok)
    ok = (bool(lines) and python_seen and not net_bad and not net_unknown and not unix_connect_ok
          and not unix_server_ok and not unix_send_to_path and not exec_bad and not exec_unknown
          and not ns_calls and not clone_bad and not uring)
    return {"lines": len(lines), "calls": len(calls),
            "network_non_local": net_bad, "network_unknown_family": net_unknown,
            "network_bind_failed": net_bind_failed,
            "network_local_count": net_local, "unix_connect_succeeded": unix_connect_ok,
            "unix_server_succeeded": unix_server_ok, "unix_send_to_path": unix_send_to_path,
            "unix_connect_failed": sorted(set(unix_connect_failed)),
            "exec_allowed": sorted(set(exec_ok)), "exec_not_allowed": exec_bad,
            "exec_unresolved": exec_unknown, "namespace_calls": ns_calls, "clone_flags_rejected": clone_bad,
            "io_uring_calls": uring, "allow_exec": sorted(allow), "python_seen": python_seen,
            "ok": ok, "limits": list(LIMITS)}


# ---------------------------------------------------------------- 伏せ字（F-10）


def _placeholder(content: str) -> str:
    return f"<redacted len={len(content)} sha256={hashlib.sha256(content.encode('utf-8', 'surrogateescape')).hexdigest()[:16]}>"


def _requote(s: str) -> str:
    """伏せ字の後の文字列を、strace の引用符の中身として正しい形に戻す（裸の " を作らない・末尾の \\ を閉じる）。"""
    out = []
    i = 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            if i + 1 < len(s):
                out.append(s[i:i + 2])
                i += 2
                continue
            out.append("\\\\")
            i += 1
            continue
        out.append('\\"' if c == '"' else c)
        i += 1
    return "".join(out)


def _load_redactor():
    """本体と同じ伏せ字（shadow.g3_isolation.redact_text）。-I で動くので、この写しの rebuild を探す場所に足す
    （BASE/src は root の持ち物で、実行の前と後で SHA-256 を照合する）。読めなければ、より強い代わりを使う。"""
    here = str(Path(__file__).resolve().parents[1])
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        from shadow.g3_isolation import redact_text
        return "g3_isolation", lambda t: redact_text(t, 1 << 20)
    except ImportError:
        return "fallback", _fallback_redact


def _load_unclassified():
    """F-10 検出器（shadow.g3_isolation.unclassified_output_keys）を本体と同じ場所から読む。
    -I で動くので redactor と同じく rebuild を探す場所に足す。読めなければ None（呼び出し側が fail-closed）。"""
    here = str(Path(__file__).resolve().parents[1])
    if here not in sys.path:
        sys.path.insert(0, here)
    try:
        from shadow.g3_isolation import unclassified_output_keys
        return unclassified_output_keys
    except ImportError:
        return None


def _redact_text(text: str) -> str:
    global _REDACTOR
    if _REDACTOR[1] is None:
        _REDACTOR = _load_redactor()
    return _REDACTOR[1](text)


_FALLBACK_RE = re.compile(r"(?i)(token|secret|passw|pwd|credential|creds|api[-_]?key|auth|cookie|session|bearer|basic)")


def _fallback_redact(text: str) -> str:
    """g3_isolation を読めないとき（-I）の代わり：秘密の語を含む文字列は丸ごと伏せる（安全側）。"""
    return "***" if _FALLBACK_RE.search(text) else text


_REDACTOR: tuple = ("unset", None)


def redact_strace_line(raw: str) -> str:
    """形を保つ伏せ字。呼び出し名・pid・fd の注記・旗・戻り値・誤りの名前は残す。
    - 起動の最初の文字列（パス）は、伏せ字で変わらなければそのまま残す（起動の判定に要る）。変わるなら置き換え、
      判定が生の記録と食い違って不合格になる（安全側）。
    - 起動の他の文字列（argv）と sun_path は、伏せ字を通す。
    - それ以外の文字列（送ったデータなど）は、長さと SHA-256 の先頭だけを残す。
    """
    m = LINE_RE.match(raw.rstrip("\n"))
    rest = m.group(2) if m else raw
    s = START_RE.match(rest)
    name = s.group(1) if s else None
    pieces: list[str] = []
    pos = 0
    first = True
    for q in QUOTED_RE.finditer(raw):
        pieces.append(raw[pos:q.start()])
        content = q.group(1)
        before = raw[max(0, q.start() - 12):q.start()]
        if name in EXEC_CALLS and first:
            red = _redact_text(content)
            new = content if red == content else _placeholder(content)
        elif name in EXEC_CALLS or "sun_path=" in before:
            new = _requote(_redact_text(content))
        else:
            new = _placeholder(content)
        pieces.append('"' + new + '"')
        pos = q.end()
        first = False
    pieces.append(raw[pos:])
    return "".join(pieces)


def judgement(summary: dict[str, Any]) -> dict[str, Any]:
    j: dict[str, Any] = {k: summary.get(k) for k in JUDGEMENT_KEYS}
    for k in JUDGEMENT_LISTS:
        v = summary.get(k) or []
        j[k + "_count"] = len(v)
    for k in ("exec_allowed", "exec_not_allowed"):
        j[k] = sorted(summary.get(k) or [])
    return j


# ---------------------------------------------------------------- 証拠の読み方（リンクをたどらない）


EVIDENCE_UID = 0


def _open_regular(path: Path, require_root: bool = False) -> tuple[Optional[bytes], Optional[str]]:
    """(中身, 問題)。リンク・通常のファイル以外・持ち主の違いは問題として返す。"""
    try:
        fd = os.open(str(path), os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    except OSError as exc:
        return None, f"cannot_open:{type(exc).__name__}"
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None, "not_regular"
        if require_root and (st.st_uid != EVIDENCE_UID or st.st_mode & 0o022):
            return None, "not_root_owned_or_writable"
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b:
                break
            chunks.append(b)
        return b"".join(chunks), None
    finally:
        os.close(fd)


def read_kv(path: Path) -> tuple[dict[str, str], Optional[str]]:
    data, problem = _open_regular(path, require_root=True)
    if data is None:
        return {}, problem
    out: dict[str, str] = {}
    for line in data.decode("utf-8", "replace").splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() in out:
                return {}, f"duplicate_key:{k.strip()}"
            out[k.strip()] = v.strip()
    return out, None


def sha_line_matches(sha_file: Optional[Path], target: Path, data: Optional[bytes]) -> Optional[bool]:
    """sha256sum の形（"<hash>  <path>"）のファイルの、パスが完全に一致するちょうど1行と照合する（F-08）。"""
    if sha_file is None:
        return None
    content, problem = _open_regular(sha_file, require_root=True)
    if content is None or data is None:
        return False
    rows = [l.split(None, 1) for l in content.decode("utf-8", "replace").splitlines() if l.strip()]
    hits = [r for r in rows if len(r) == 2 and r[1].strip() == str(target)]
    if len(hits) != 1:
        return False
    return hits[0][0] == hashlib.sha256(data).hexdigest()


def list_entry_matches(listing: Optional[list], rel: str, data: Optional[bytes]) -> Optional[bool]:
    """root の一覧（out.list）の rel のちょうど1件と、種類・大きさ・SHA-256 を照合する（F-08）。"""
    if listing is None:
        return None
    if data is None or not isinstance(listing, list):
        return False
    hits = [e for e in listing if isinstance(e, dict) and e.get("path") == rel]
    if len(hits) != 1:
        return False
    e = hits[0]
    return (e.get("type") == "f" and e.get("size") == len(data)
            and e.get("sha256") == hashlib.sha256(data).hexdigest())


# ---------------------------------------------------------------- 記録


def apply_to_records(data: bytes, summary: dict[str, Any], out: Path) -> tuple[int, int, list]:
    """(件数, 壊れた行の数, 未分類キー)。壊れた行は書き出さない（E-06）。
    G-13：各レコードを F-10 検出器にかける。G-13a：未分類キーがあれば records_final.jsonl を
    新規作成・上書きしない（既存ファイルは truncate も削除もしない）。"""
    status = "ok" if summary["ok"] else "failed"
    detector = _load_unclassified()
    count = bad = 0
    rows: list = []
    unclassified: list = []
    if detector is None:
        unclassified.append("<detector_unavailable>")  # fail-closed：検出器を読めないときは成果物を作らない
    for line in data.decode("utf-8", "replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            if not isinstance(r, dict):
                raise ValueError("not an object")
        except ValueError:
            bad += 1
            continue
        if detector is not None:
            unclassified.extend(detector(r))
        r.setdefault("count_candidate_checks", {})["strace_evidence"] = status
        rows.append(r)
        count += 1
    unclassified = sorted(set(unclassified))
    if not unclassified:  # G-13a：未分類が無いときだけ out を開く（未分類時は既存を触らない）
        with open(out, "w", encoding="utf-8") as dst:
            for r in rows:
                dst.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    return count, bad, unclassified


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="verify strace evidence of the G3 diagnostic step")
    ap.add_argument("--strace", required=True, type=Path, help="生の strace の記録（root だけが読める場所）")
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--summary", type=Path, default=None, help="summary.json of the parent (C-05)")
    ap.add_argument("--allow-exec", action="append", default=[], help="absolute path allowed to exec")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--status-file", type=Path, default=None, help="root が書いた状態（public/status.txt）")
    ap.add_argument("--sysctl-file", type=Path, default=None, help="root が書いた sysctl の記録（public/sysctl.txt）")
    ap.add_argument("--expect-strace-sha256-file", type=Path, default=None,
                    help="root が書いた strace の記録の SHA-256（E-02・F-08）")
    ap.add_argument("--expect-out-list", type=Path, default=None,
                    help="root が作った出力の一覧（out.list。F-01・F-08）")
    ap.add_argument("--evidence-owner-uid", type=int, default=0,
                    help="証拠のファイル（状態・SHA-256・一覧）の持ち主の uid（既定 0 = root。テストでは自分の uid）")
    ap.add_argument("--apparmor-file", type=Path, default=None,
                    help="root が書いた AppArmor の記録（public/apparmor.txt。R-1。既定は status-file と同じフォルダ）")
    ap.add_argument("--writable-after-file", type=Path, default=None,
                    help="root が書いた診断後の書ける場所の走査（public/writable_after.meta。R-2。既定は同上）")
    a = ap.parse_args(argv)
    global EVIDENCE_UID, _REDACTOR
    EVIDENCE_UID = a.evidence_owner_uid
    _REDACTOR = _load_redactor()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 状態と sysctl（F-02・F-03）：ok でなければ、out の中は開かない
    gate: dict[str, Any] = {"status": None, "sysctl": None, "problems": []}
    if a.status_file is not None:
        st, problem = read_kv(a.status_file)
        gate["status"] = st or problem
        if problem or st.get("state") != "ok":
            gate["problems"].append(f"state:{st.get('state') if st else problem}")
    if a.sysctl_file is not None:
        sy, problem = read_kv(a.sysctl_file)
        gate["sysctl"] = sy or problem
        if problem or sy.get("sysctl_state") != "restored":
            gate["problems"].append(f"sysctl:{sy.get('sysctl_state') if sy else problem}")
    # 証拠フォルダは、既定では status-file と同じ場所（public/）
    pubdir = a.status_file.parent if a.status_file is not None else None
    # R-1：AppArmor の記録（記録のみ。門にはしない。absent・read_failed・値 を区別して載せる）
    aa_file = a.apparmor_file or (pubdir / "apparmor.txt" if pubdir else None)
    if aa_file is not None and aa_file.exists():
        aa, aa_problem = read_kv(aa_file)
        gate["apparmor"] = aa or aa_problem
    elif aa_file is not None:
        gate["apparmor"] = "file_absent"
    # R-2：診断後の書ける場所の走査の結果を門にする（書ける場所>0・硬い誤り>0・ファイル無し は不合格）
    wa_file = a.writable_after_file or (pubdir / "writable_after.meta" if pubdir else None)
    if wa_file is not None:
        if wa_file.exists():
            wa, wa_problem = read_kv(wa_file)
            gate["writable_after"] = wa or wa_problem
            try:
                if wa_problem or int(wa.get("writable_hits", "1")) != 0 or int(wa.get("hard_errors", "1")) != 0:
                    gate["problems"].append(f"writable_after:{wa.get('writable_hits')}/{wa.get('hard_errors')}")
            except (TypeError, ValueError):
                gate["problems"].append("writable_after:unreadable")
        else:
            gate["problems"].append("writable_after:file_absent")
    if gate["problems"]:
        summary = {"ok": False, "gate": gate, "opened_outputs": False, "limits": list(LIMITS)}
        _write(a.out_dir / "strace_summary.json", json.dumps(summary, ensure_ascii=False, indent=1))
        print(json.dumps({"ok": False, "gate_problems": gate["problems"]}))
        return 2

    # 2. strace（生）と、伏せ字にした写し（F-10）
    data, sproblem = _open_regular(a.strace, require_root=True)
    raw_lines = (data or b"").decode("utf-8", "surrogateescape").splitlines()
    summary = analyze(raw_lines, a.allow_exec)
    summary["gate"] = gate
    summary["strace_present"] = data is not None
    summary["strace_problem"] = sproblem
    summary["strace_sha256_raw"] = hashlib.sha256(data or b"").hexdigest()
    red_lines = [redact_strace_line(l) for l in raw_lines]
    red_text = "\n".join(red_lines) + ("\n" if red_lines else "")
    red_bytes = red_text.encode("utf-8", "surrogateescape")
    (a.out_dir / "strace_redacted.txt").write_bytes(red_bytes)
    summary["strace_sha256_redacted"] = hashlib.sha256(red_bytes).hexdigest()
    summary["redaction_impl"] = _REDACTOR[0]
    red_summary = analyze(red_lines, a.allow_exec)
    jr, jd = judgement(summary), judgement(red_summary)
    summary["redaction_preserves_judgement"] = jr == jd
    if jr != jd:
        summary["redaction_judgement_diff"] = sorted(k for k in jr if jr[k] != jd.get(k))
        summary["ok"] = False
    # 要約に載せる抜き出しは、伏せ字にした写しのほうの一覧にする（生の文字列を外に出さない）
    for k in JUDGEMENT_LISTS:
        if k in ("exec_allowed", "exec_not_allowed"):
            continue
        summary[k] = red_summary.get(k)
    (a.out_dir / "strace_redacted.sha256").write_text(f"{summary['strace_sha256_redacted']}  strace_redacted.txt\n",
                                                      encoding="utf-8")

    # 3. 証拠の照合（E-02・F-08）
    listing = None
    list_problem = None
    if a.expect_out_list is not None:
        lb, list_problem = _open_regular(a.expect_out_list, require_root=True)
        try:
            listing = json.loads(lb) if lb is not None else None
        except ValueError:
            list_problem = "out_list_corrupt"
        if listing is None:
            listing = []
    rec_bytes, rec_problem = _open_regular(a.records)
    sum_bytes, sum_problem = (_open_regular(a.summary) if a.summary is not None else (None, None))
    evidence = {"strace_sha256_matches_root": sha_line_matches(a.expect_strace_sha256_file, a.strace, data),
                "records_matches_out_list": list_entry_matches(listing if a.expect_out_list else None,
                                                               "./records.jsonl", rec_bytes),
                "summary_matches_out_list": (list_entry_matches(listing if a.expect_out_list else None,
                                                                "./summary.json", sum_bytes)
                                             if a.summary is not None else None),
                "out_list_problem": list_problem}
    summary["evidence_integrity"] = evidence
    if any(evidence[k] is False for k in ("strace_sha256_matches_root", "records_matches_out_list",
                                          "summary_matches_out_list")) or list_problem:
        summary["ok"] = False

    # 4. 記録との整合（C-05・D-07・E-06）
    integrity: dict[str, Any] = {"records_present": rec_bytes is not None, "records_problem": rec_problem,
                                 "summary_present": None, "records": None, "summary_races": None,
                                 "records_match_summary": None,
                                 "unclassified_output_keys": [], "records_final_written": False}
    if rec_bytes is not None:
        integrity["records"], integrity["records_corrupt_lines"], _unclassified = apply_to_records(
            rec_bytes, summary, a.out_dir / "records_final.jsonl")
        integrity["unclassified_output_keys"] = _unclassified
        integrity["records_final_written"] = not _unclassified  # 書けたとき（例外は伝播）だけ True
    if a.summary is not None:
        integrity["summary_present"] = sum_bytes is not None
        integrity["summary_problem"] = sum_problem
        if sum_bytes is not None:
            try:
                ps = json.loads(sum_bytes.decode("utf-8", "replace"))
                if not isinstance(ps, dict):
                    raise ValueError("not an object")
            except ValueError:
                ps = {}
                integrity["summary_corrupt"] = True
            for k in ("child_failures", "all_children_failed", "no_target_races", "no_odds_count"):
                integrity[k] = ps.get(k)
            integrity["summary_races"] = ps.get("races")
            integrity["records_match_summary"] = integrity["records"] == ps.get("races")
            integrity["records_sha256_match"] = hashlib.sha256(rec_bytes or b"").hexdigest() == ps.get("records_sha256")
            ids = []
            for l in (rec_bytes or b"").decode("utf-8", "replace").splitlines():
                if l.strip():
                    try:
                        ids.append(json.loads(l).get("eval_id"))
                    except (ValueError, AttributeError):
                        ids.append(None)
            expected_ids = ps.get("eval_ids")
            integrity["eval_ids_match"] = (isinstance(expected_ids, list) and len(ids) == len(set(ids))
                                           and sorted(map(str, ids)) == sorted(map(str, expected_ids)))
    integrity["ok"] = bool(integrity["records_present"]) and not integrity.get("records_corrupt_lines") \
        and not integrity.get("unclassified_output_keys") and (
        a.summary is None or (integrity["records_match_summary"] is True and
                              integrity.get("records_sha256_match") is True and
                              integrity.get("eval_ids_match") is True))
    summary["integrity"] = integrity
    summary["opened_outputs"] = True
    _write(a.out_dir / "strace_summary.json", json.dumps(summary, ensure_ascii=False, indent=1, default=str))
    print(json.dumps({"ok": summary["ok"], "integrity_ok": integrity["ok"], "lines": summary["lines"],
                      "network_non_local": len(summary["network_non_local"]),
                      "network_bind_failed": len(summary.get("network_bind_failed") or []),
                      "network_unknown_family": len(summary["network_unknown_family"]),
                      "unix_connect_succeeded": len(summary["unix_connect_succeeded"]),
                      "unix_server_succeeded": len(summary["unix_server_succeeded"]),
                      "unix_send_to_path": len(summary["unix_send_to_path"]),
                      "namespace_calls": len(summary["namespace_calls"]),
                      "clone_flags_rejected": len(summary["clone_flags_rejected"]),
                      "io_uring_calls": len(summary["io_uring_calls"]),
                      "redaction_preserves_judgement": summary["redaction_preserves_judgement"],
                      "evidence_integrity": evidence,
                      "exec_not_allowed": len(summary["exec_not_allowed"]),
                      "exec_unresolved": len(summary["exec_unresolved"]),
                      "records": integrity["records"], "summary_races": integrity["summary_races"]}))
    if not summary["ok"] or integrity.get("unclassified_output_keys"):
        return 2
    return 0 if integrity["ok"] else 3


if __name__ == "__main__":
    sys.exit(main())
