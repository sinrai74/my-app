"""
正式G3（B案）段階3：出力の一覧・写し・照合（root が `python -I` で動かす。標準ライブラリだけを使う）。

再監査 F-01・第3版の実装条件：
- フォルダは親から順に、リンクをたどらない開き方（openat 相当：dir_fd と O_NOFOLLOW・O_DIRECTORY）でたどる。
- 通常のファイル・フォルダ以外（シンボリックリンク・FIFO・ソケット・デバイスなど）があれば不合格。
- 通常のファイルのリンクの数が 1 でなければ不合格（外への hardlink を拒む）。フォルダのリンクの数は見ない。
- 一覧を作った後、写し、写し元と写し先をもう一度歩いて、相対パス・種類・大きさ・SHA-256 が一覧と完全に一致すること。
  写す間は、開いた FD の inode・大きさ・更新時刻も一覧と照合する。一致しなければ不合格（何も消さない）。

使い方：
  g3_diagnostic_collect.py list  <src_dir> <list_out>
  g3_diagnostic_collect.py copy  <src_dir> <dst_dir> <list_in> <collect_list_out>
  g3_diagnostic_collect.py check <dir> <list_in>
終了コード：0 一致・2 引数の誤り・94 不合格。
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from typing import Any, Optional

DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOCTTY", 0)
CHUNK = 1 << 20


class CollectError(Exception):
    """一覧・写し・照合の不合格。"""


def _open_root(path: str) -> int:
    """根のフォルダを、絶対パスの各部分からリンクをたどらずに開く。相対パスは cwd を前に付ける。"""
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    parts = [p for p in os.path.normpath(path).split("/") if p]
    fd = os.open("/", DIR_FLAGS)
    try:
        for part in parts:
            if part in (".", ".."):
                raise CollectError(f"path must be normalized: {path}")
            nfd = os.open(part, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = nfd
    except OSError as exc:
        os.close(fd)
        raise CollectError(f"cannot open directory without following links: {path}: {type(exc).__name__}") from exc
    return fd


def _hash_fd(fd: int) -> str:
    h = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        b = os.read(fd, CHUNK)
        if not b:
            break
        h.update(b)
    return h.hexdigest()


def _entry_of(st: os.stat_result, rel: str, kind: str, sha: Optional[str]) -> dict[str, Any]:
    return {"path": rel, "type": kind, "mode": stat.S_IMODE(st.st_mode), "uid": st.st_uid, "gid": st.st_gid,
            "size": st.st_size if kind == "f" else 0, "nlink": st.st_nlink if kind == "f" else None,
            "ino": st.st_ino, "dev": st.st_dev, "mtime_ns": st.st_mtime_ns if kind == "f" else None,
            "sha256": sha}


def walk(root: str) -> list[dict[str, Any]]:
    """root の下を、リンクをたどらずに歩いた一覧（相対パスの順）。不合格の項目があれば CollectError。"""
    out: list[dict[str, Any]] = []
    rfd = _open_root(root)
    try:
        _walk_dir(rfd, ".", out)
    finally:
        os.close(rfd)
    out.sort(key=lambda e: e["path"])
    return out


def _walk_dir(dfd: int, rel: str, out: list[dict[str, Any]]) -> None:
    for name in sorted(os.listdir(dfd)):
        child = f"{rel}/{name}"
        st = os.stat(name, dir_fd=dfd, follow_symlinks=False)
        if stat.S_ISDIR(st.st_mode):
            cfd = os.open(name, DIR_FLAGS, dir_fd=dfd)
            try:
                cst = os.fstat(cfd)
                if (cst.st_ino, cst.st_dev) != (st.st_ino, st.st_dev):
                    raise CollectError(f"directory changed while opening: {child}")
                out.append(_entry_of(cst, child, "d", None))
                _walk_dir(cfd, child, out)
            finally:
                os.close(cfd)
        elif stat.S_ISREG(st.st_mode):
            if st.st_nlink != 1:
                raise CollectError(f"regular file with link count {st.st_nlink}: {child}")
            ffd = os.open(name, FILE_FLAGS, dir_fd=dfd)
            try:
                fst = os.fstat(ffd)
                if (fst.st_ino, fst.st_dev, fst.st_size) != (st.st_ino, st.st_dev, st.st_size) \
                        or not stat.S_ISREG(fst.st_mode):
                    raise CollectError(f"file changed while opening: {child}")
                out.append(_entry_of(fst, child, "f", _hash_fd(ffd)))
            finally:
                os.close(ffd)
        else:
            raise CollectError(f"not a regular file or directory: {child} (mode {oct(st.st_mode)})")


def _comparable(e: dict[str, Any]) -> tuple:
    return (e["path"], e["type"], e["size"], e["sha256"])


def compare(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> list[str]:
    """相対パス・種類・大きさ・SHA-256 の違い（重複したパスも違いとする）。"""
    problems = []
    for name, lst in (("expected", expected), ("actual", actual)):
        paths = [e["path"] for e in lst]
        if len(paths) != len(set(paths)):
            problems.append(f"duplicate paths in {name}")
    exp = {e["path"]: _comparable(e) for e in expected}
    act = {e["path"]: _comparable(e) for e in actual}
    for p in sorted(set(exp) | set(act)):
        if exp.get(p) != act.get(p):
            problems.append(f"mismatch: {p}")
    return problems


def copy_tree(src: str, dst: str, listing: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """listing のとおりに写す。dst は空のフォルダ。写す間も写し元の FD を一覧と照合する。"""
    sfd, dfd = _open_root(src), _open_root(dst)
    try:
        if os.listdir(dfd):
            raise CollectError("destination is not empty")
        for e in listing:
            parts = e["path"].split("/")[1:]
            if any(p in ("", ".", "..") for p in parts):
                raise CollectError(f"bad path in listing: {e['path']}")
            ps, pd = _descend(sfd, parts[:-1]), _descend(dfd, parts[:-1])
            try:
                if e["type"] == "d":
                    os.mkdir(parts[-1], 0o755, dir_fd=pd)
                elif e["type"] == "f":
                    _copy_file(ps, pd, parts[-1], e)
                else:
                    raise CollectError(f"unexpected type in listing: {e['path']}")
            finally:
                if ps != sfd:
                    os.close(ps)
                if pd != dfd:
                    os.close(pd)
    finally:
        os.close(sfd)
        os.close(dfd)
    return walk(dst)


def _descend(rootfd: int, parts: list[str]) -> int:
    fd = rootfd
    for p in parts:
        nfd = os.open(p, DIR_FLAGS, dir_fd=fd)
        if fd != rootfd:
            os.close(fd)
        fd = nfd
    return fd


def _copy_file(ps: int, pd: int, name: str, e: dict[str, Any]) -> None:
    ifd = os.open(name, FILE_FLAGS, dir_fd=ps)
    try:
        st = os.fstat(ifd)
        if not stat.S_ISREG(st.st_mode) or st.st_nlink != 1 or \
                (st.st_ino, st.st_dev, st.st_size, st.st_mtime_ns) != (e["ino"], e["dev"], e["size"], e["mtime_ns"]):
            raise CollectError(f"source changed after listing: {e['path']}")
        ofd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                      0o644, dir_fd=pd)
        h = hashlib.sha256()
        try:
            while True:
                b = os.read(ifd, CHUNK)
                if not b:
                    break
                h.update(b)
                os.write(ofd, b)
        finally:
            os.close(ofd)
        if h.hexdigest() != e["sha256"]:
            raise CollectError(f"source content changed after listing: {e['path']}")
        st2 = os.fstat(ifd)
        if (st2.st_size, st2.st_mtime_ns) != (e["size"], e["mtime_ns"]):
            raise CollectError(f"source changed while copying: {e['path']}")
    finally:
        os.close(ifd)


def _write_json(path: str, obj: Any) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, sort_keys=True)


def _read_json(path: str) -> Any:
    fd = os.open(path, FILE_FLAGS)
    with os.fdopen(fd, encoding="utf-8") as fh:
        return json.load(fh)


def main(argv: Optional[list[str]] = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    try:
        if len(a) == 3 and a[0] == "list":
            listing = walk(a[1])
            _write_json(a[2], listing)
            print(json.dumps({"collect_list": "ok", "entries": len(listing)}))
            return 0
        if len(a) == 5 and a[0] == "copy":
            listing = _read_json(a[3])
            before = walk(a[1])
            problems = compare(listing, before)
            if problems:
                raise CollectError("source differs from listing before copy: " + "; ".join(problems[:5]))
            copied = copy_tree(a[1], a[2], listing)
            after = walk(a[1])
            problems = compare(listing, copied) + compare(listing, after)
            if problems:
                raise CollectError("; ".join(problems[:5]))
            _write_json(a[4], copied)
            print(json.dumps({"collect_copy": "ok", "entries": len(copied)}))
            return 0
        if len(a) == 3 and a[0] == "check":
            problems = compare(_read_json(a[2]), walk(a[1]))
            if problems:
                raise CollectError("; ".join(problems[:5]))
            print(json.dumps({"collect_check": "ok"}))
            return 0
    except (CollectError, OSError, ValueError) as exc:
        print(json.dumps({"collect": "failed", "reason": f"{type(exc).__name__}: {exc}"[:300]}))
        return 94
    print("usage: list <src> <out> | copy <src> <dst> <list> <out> | check <dir> <list>")
    return 2


if __name__ == "__main__":
    sys.exit(main())
