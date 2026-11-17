#!/usr/bin/env bash
# 正式G3（B案）段階3：診断の段（ネットワークなし）を root で組み立てる台本。workflow から sudo で呼ぶ。
# 第3版の修正設計（再監査 F-01〜F-10）。実装条件に従い、満たせない前提があれば必ず止める（fail closed）。
#
# 使い方（root で）：
#   g3_diagnostic_run.sh <workspace> <base> <python> <race_date> <only> <run_id> <runner_kind> [<ld_library_path>]
#   runner_kind：github-hosted（本番）か local-test（手元の試験。/opt/g3test/LOCAL_TEST_HOST が要る）。
#   workspace に g3_in/programs_raw.json・g3_in/odds.json があること。base は存在しないこと（新しく作る）。
#
# 大枠：
#   1. 事前の確認（C1〜C6）：cgroup v2・実行ごとの cgroup・cgroup.kill・実際に止まる・抜け出せない・排他（flock）。
#      どれかを満たせなければ、診断を始めずに止める（F-02）。cgroup が使えないとき uid へ切り替えない。
#   2. 専用の利用者（g3diag。補助グループなし）・既存プロセスの確認（F-06）。
#   3. root の持つ読むだけの写し（BASE/src）と、root の持つ証拠のフォルダ（BASE/evidence、0700）。出力は BASE/out。
#   4. sysctl（user.max_user_namespaces=0）を、trap を張ってから退避・設定し、終わりに必ず戻す（F-03）。
#   5. 実行ごとの cgroup に入れ、unshare の net・pid・mount の中で strace の下に動かす（F-02・F-03・F-04・F-07）。
#   6. cgroup.kill で止め、名前空間の中（PID 1 の終了）とホスト側（cgroup が空）と開いた FD を分けて確かめる。
#      止まるまで凍結しない。止まらなければ kill_failed（検査を必ず不合格にする）。
#   7. 出力を root の持ち物にして書き込みを外し、リンクなどを拒み、一覧（out.list）と SHA-256 を root が書く（F-01・F-08）。
#      通常のファイルだけを BASE/collect に、リンクをたどらずに写す。生の strace は evidence（root 専用）に残す（F-10）。
set -euo pipefail

WS="$1"; BASE="$2"; PY="$3"; RACE_DATE="$4"; ONLY="$5"; RUN_ID="$6"; RUNNER_KIND="$7"; LDLP="${8:-}"
USER_NAME="${G3_DIAG_USER:-g3diag}"
PUB=""; STATE="unknown"; SYSCTL_STATE="not_changed"; SYSCTL_PREV=""; LOCKFD=""; CG=""; FROZEN=0

log() { echo "g3_run: $*"; }
die() { log "$*"; exit "${2:-90}"; }

# ---- 事前の素の確認
[ "$(id -u)" = "0" ] || die "must run as root"
[ -e "$BASE" ] && die "base already exists: $BASE"
case "$PY" in /*) ;; *) die "python path must be absolute";; esac
case "$RUNNER_KIND" in
  github-hosted) ;;
  local-test) [ -f /opt/g3test/LOCAL_TEST_HOST ] || die "local-test requires /opt/g3test/LOCAL_TEST_HOST";;
  *) die "runner_kind must be github-hosted or local-test";;
esac
SETPRIV="$(command -v setpriv)"; ENVBIN="$(command -v env)"; STRACE="$(command -v strace)"
UNSHARE="$(command -v unshare)"; FLOCK="$(command -v flock)"
SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
INNER="$SELF_DIR/g3_diagnostic_inner.sh"; SOCK_FINDER="$SELF_DIR/g3_find_writable_sockets.sh"
COLLECT="$SELF_DIR/g3_diagnostic_collect.py"; WRITABLE_FINDER="$SELF_DIR/g3_find_writable_paths.sh"
for f in "$INNER" "$SOCK_FINDER" "$COLLECT" "$WRITABLE_FINDER"; do [ -f "$f" ] || die "missing helper: $f"; done
APPARMOR_SYSCTL="/proc/sys/kernel/apparmor_restrict_unprivileged_userns"
# strace に記録させる呼び出しは、検査の側（verify）と同じものを使う（静的テストで照合する）
G3_STRACE_TRACE="$("$PY" -I -P -B -c 'import sys; sys.path.insert(0, "'"$SELF_DIR"'/.."); from actions.g3_diagnostic_verify import STRACE_TRACE; print(STRACE_TRACE)')"
[ -n "$G3_STRACE_TRACE" ] || die "cannot read STRACE_TRACE from verifier"

# ---- 排他（C6）：診断の最初から最後まで flock を持ち続ける
exec {LOCKFD}>/run/g3diag.lock
"$FLOCK" -n "$LOCKFD" || die "another g3 diagnostic run holds the lock" 97

# ---- 1. cgroup v2 の事前の確認（C1〜C5）。満たせなければ診断を始めない（uid へ切り替えない）
CG2ROOT="$(awk '{for(i=7;i<=NF;i++) if($i=="-"){if($(i+1)=="cgroup2"){print $5; exit} else break}}' /proc/self/mountinfo || true)"
[ -n "$CG2ROOT" ] && [ -d "$CG2ROOT" ] || die "cgroup v2 is not available (C1)" 97
MYCG="$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup || true)"
[ -n "$MYCG" ] || die "cannot read own cgroup (C1)" 97
PARENT_CG="$CG2ROOT$MYCG"
# この cgroup で子を作れるよう、必要なら +pids などを subtree に降ろす前に、まず自分の cgroup を使えることだけ確かめる
CG="$PARENT_CG/g3diag-$RUN_ID"
[ -e "$CG" ] && die "run cgroup already exists: $CG" 97
mkdir "$CG" 2>/dev/null || die "cannot create run cgroup (C2)" 97
[ -e "$CG/cgroup.kill" ] && [ -e "$CG/cgroup.events" ] || die "cgroup.kill is not available (C3)" 97
# C4：試しに sleep を入れて、cgroup.kill で5秒以内に空になること
CGTEST="$PARENT_CG/g3diagtest-$RUN_ID"
mkdir "$CGTEST" 2>/dev/null || die "cannot create test cgroup (C4)" 97
( echo $BASHPID > "$CGTEST/cgroup.procs"; exec sleep 30 ) &
tpid=$!
sleep 0.3
echo 1 > "$CGTEST/cgroup.kill" || { kill -KILL $tpid 2>/dev/null || true; die "cgroup.kill write failed (C4)" 97; }
ok=0; for _ in $(seq 1 50); do
  grep -q '^populated 0$' "$CGTEST/cgroup.events" 2>/dev/null && { ok=1; break; }
  sleep 0.1
done
wait $tpid 2>/dev/null || true
rmdir "$CGTEST" 2>/dev/null || true
[ "$ok" = 1 ] || die "cgroup.kill did not stop a test process (C4)" 97
# C5：追跡される側（g3diag）が、cgroupfs の中に書ける cgroup.procs を持たないこと（抜け出せない）。
#     探索の権限不足・エラー・未探索があれば合格にしない（実装条件）。
DUID0="$(id -u "$USER_NAME" 2>/dev/null || echo "")"
DGID0="$(id -g "$USER_NAME" 2>/dev/null || echo "")"
if [ -n "$DUID0" ]; then
  # root の作業用の一時ファイルは mktemp で（共有 /tmp の予測可能な名前へのシンボリックリンク攻撃を避ける）
  c5out="$(mktemp)"
  c5err="$($SETPRIV --reuid="$DUID0" --regid="$DGID0" --clear-groups --no-new-privs \
          /usr/bin/env -i PATH="$PATH" find "$CG2ROOT" -name cgroup.procs -writable -print 2>&1 1>"$c5out" || true)"
  if [ -s "$c5out" ]; then rm -f "$c5out"; die "dedicated user can write cgroup.procs (C5)" 97; fi
  if [ -n "$c5err" ]; then rm -f "$c5out"; die "C5 cgroup scan had errors: ${c5err%%$'\n'*}" 97; fi
  rm -f "$c5out"
fi

# ---- 失敗・終了のときに必ず通る後始末（sysctl を戻す・cgroup.kill・cgroup を外す）
cleanup() {
  local rc=$?
  set +e
  if [ -n "$CG" ] && [ -d "$CG" ]; then echo 1 > "$CG/cgroup.kill" 2>/dev/null; fi
  _restore_sysctl
  if [ -n "$CG" ] && [ -d "$CG" ]; then
    for _ in $(seq 1 20); do grep -q '^populated 0$' "$CG/cgroup.events" 2>/dev/null && break; sleep 0.2; done
    rmdir "$CG" 2>/dev/null
  fi
  return $rc
}
_restore_sysctl() {
  [ "$SYSCTL_STATE" = "changed" ] || return 0
  local now; now="$(cat /proc/sys/user/max_user_namespaces 2>/dev/null || echo "")"
  if [ "$now" = "$SYSCTL_PREV" ]; then SYSCTL_STATE="restored"; _write_sysctl; return 0; fi
  if echo "$SYSCTL_PREV" > /proc/sys/user/max_user_namespaces 2>/dev/null; then
    now="$(cat /proc/sys/user/max_user_namespaces 2>/dev/null || echo "")"
    if [ "$now" = "$SYSCTL_PREV" ]; then SYSCTL_STATE="restored"; else SYSCTL_STATE="restore_unconfirmed"; fi
  else SYSCTL_STATE="restore_failed"; fi
  _write_sysctl
}
_write_sysctl() {
  [ -n "$PUB" ] || return 0
  { echo "max_user_namespaces_prev=$SYSCTL_PREV"; echo "sysctl_state=$SYSCTL_STATE";
    echo "runner_kind=$RUNNER_KIND"; } > "$PUB/sysctl.txt" 2>/dev/null || true
  chmod 0444 "$PUB/sysctl.txt" 2>/dev/null || true
}
# R-1：AppArmor の設定（kernel.apparmor_restrict_unprivileged_userns）の現在値を記録する（変更はしない）。
#   不存在・取得失敗・値を、別々に記録する。値を取れなくても、ここでは止めない（設計上の扱い）：
#   user namespace を実際に塞ぐのは max_user_namespaces=0 と、追跡される側で unshare -U/-m が失敗することの
#   事前確認であり、そちらが権威。AppArmor は証拠の記録のみ。
_write_apparmor() {
  [ -n "$PUB" ] || return 0
  local state val
  if [ ! -e "$APPARMOR_SYSCTL" ]; then
    state="absent"; val=""
  elif val="$(cat "$APPARMOR_SYSCTL" 2>/dev/null)"; then
    state="present"
  else
    state="read_failed"; val=""
  fi
  { echo "apparmor_restrict_unprivileged_userns_state=$state";
    echo "apparmor_restrict_unprivileged_userns_value=$val"; } > "$PUB/apparmor.txt" 2>/dev/null || true
  chmod 0444 "$PUB/apparmor.txt" 2>/dev/null || true
  echo "g3_apparmor=$state value=$val"
}

# R-2：追跡される側（g3diag）から書ける場所を、与えた木（workspace・BASE/src・Python の検索パス）で探す。
#   診断の前と後の両方で呼ぶ。書ける場所が1件でも、または権限拒否以外の硬い誤りがあれば、合格にしない。
#   権限拒否（たどれない）は「未探索」として数えて記録する：g3diag が入れないフォルダは、その中に書くこともできない
#   （フォルダに作るには x+w が要る）。検査の主体は本番と同じ RUN_AS（setpriv で g3diag・補助グループなし）。
#   $1=ラベル（before/after）。合格なら 0、不合格なら 1 を返す。
_writable_scan() {
  local label="$1" out="$PUB/writable_$1.list0" err="$PUB/writable_$1.err" meta="$PUB/writable_$1.meta"
  local -a roots=("$WS" "$BASE/src")
  # Python の検索パス（sys.path）を、本番と同じ -I で取り、NUL 区切りで足す。
  # NUL はシェル変数に入らないので、プロセス置換から直接 mapfile で読む。
  local -a sproots=()
  mapfile -t -d '' sproots < <("$PY" -I -P -B -c 'import sys, os
sys.stdout.write("\0".join(p for p in sys.path if p and os.path.isdir(p)))' 2>/dev/null || true)
  roots+=("${sproots[@]}")
  "${RUN_AS[@]}" "${ENVS_CHK[@]}" bash "$WRITABLE_FINDER" "${roots[@]}" >"$out" 2>"$err" || true
  local hits denied hard
  hits="$(tr -cd '\0' <"$out" | wc -c | tr -d ' ')"
  denied="$(grep -cE 'Permission denied|Operation not permitted|MISSING ' "$err" 2>/dev/null || true)"
  hard="$(grep -vE 'Permission denied|Operation not permitted|MISSING ' "$err" 2>/dev/null | grep -c . || true)"
  { echo "# R-2 writable scan ($label). 主体=g3diag(uid=$DUID gid=$DGID, 補助グループなし)";
    echo "# 除外=proc,sysfs,cgroup,cgroup2（擬似ファイル系。書けても意味がなく探索が重い）。-xdev は使わない";
    echo "# 根（検査した木）:"; printf '#   %s\n' "${roots[@]}";
    echo "writable_hits=$hits"; echo "unexplored_permission_denied=$denied"; echo "hard_errors=$hard";
    echo "# find -writable の結果だけで全領域の安全を断定はしない。未探索（権限拒否）は g3diag が入れない＝書けない範囲";
  } > "$meta" 2>/dev/null || true
  chmod 0444 "$out" "$err" "$meta" 2>/dev/null || true
  echo "g3_writable_$label=hits:$hits unexplored:$denied hard_errors:$hard"
  [ "$hits" = 0 ] && [ "$hard" = 0 ]
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

# ---- 2. 専用の利用者（自分のグループだけ）と、既存プロセスの確認（F-06）
USER_PREEXISTED=1
if ! id -u "$USER_NAME" >/dev/null 2>&1; then
  USER_PREEXISTED=0
  useradd --system --no-create-home --home-dir "$BASE/home" --shell /usr/sbin/nologin --user-group "$USER_NAME"
fi
DUID="$(id -u "$USER_NAME")"; DGID="$(id -g "$USER_NAME")"
{ [ "$DUID" != "0" ] && [ "$DGID" != "0" ]; } || die "dedicated user must not be root" 91
[ "$(id -G "$USER_NAME")" = "$DGID" ] || die "dedicated user has extra groups" 91
# 同じ uid を使う利用者が他にいないこと（uid で止める予備の前提）
[ "$(getent passwd | awk -F: -v u="$DUID" '$3==u' | wc -l)" = "1" ] || die "uid $DUID is shared by several users" 91
# 実行の前に、この利用者のプロセスが残っていないこと（F-06）
if pgrep -u "$DUID" >/dev/null 2>&1; then die "processes of the dedicated user already exist" 91; fi

# ---- 3. フォルダ：src・evidence は root の持ち物。evidence は 0700（生の strace を外に出さない：F-10）
mkdir -m 0755 "$BASE"
mkdir -m 0755 "$BASE/src" "$BASE/out"
mkdir -m 0700 "$BASE/evidence"
install -d -m 0755 "$BASE/evidence/public"   # 証拠のうち、外に出してよいもの（状態・SHA-256・一覧・伏せ字の写し）
PUB="$BASE/evidence/public"
tar -C "$WS" --exclude=./.git -cf - . | tar -C "$BASE/src" -xf -
chown -R root:root "$BASE/src" "$BASE/evidence"
chmod -R u+rwX,go+rX,go-w "$BASE/src"
# 出力のフォルダだけを専用の利用者に渡す（src・evidence は root のまま。追跡される側はここにだけ書ける）
chown "$DUID:$DGID" "$BASE/out"
RUNHOME="/tmp/g3diag/home"   # 名前空間の中の tmpfs の下（F-07）
# BASE/src 全体の SHA-256（実行の後で変わっていないことを確かめる：F-05）
( cd "$BASE/src" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum ) > "$BASE/evidence/src.sha256"
SRC_SHA_BEFORE="$(sha256sum "$BASE/evidence/src.sha256" | cut -d' ' -f1)"

# 専用の利用者が workspace・写しの検査のコードに書けないこと（E-02・F-05）
RUN_AS=("$SETPRIV" --reuid="$DUID" --regid="$DGID" --clear-groups --no-new-privs)
ENVS_CHK=("$ENVBIN" -i PATH="$PATH")
for target in "$WS/rebuild/actions/g3_diagnostic_verify.py" "$BASE/src/rebuild/actions/g3_diagnostic_verify.py" \
              "$BASE/src/rebuild/actions/g3_diagnostic_collect.py"; do
  if "${RUN_AS[@]}" "${ENVS_CHK[@]}" test -w "$target"; then die "dedicated user can write $target" 92; fi
done
PGROUPS="$("${RUN_AS[@]}" "${ENVS_CHK[@]}" id -G)"
[ "$PGROUPS" = "$DGID" ] || die "process groups are not only the dedicated group: $PGROUPS" 92
for s in /var/run/docker.sock /run/docker.sock /run/containerd/containerd.sock /var/run/containerd/containerd.sock; do
  if [ -S "$s" ]; then
    sg="$(stat -c %g "$s")"; so="$(stat -c %u "$s")"
    if [ "$sg" = "$DGID" ] || [ "$so" = "$DUID" ]; then die "dedicated user can reach $s" 92; fi
  fi
done
# R-2：診断の前の書ける場所の走査（workspace・BASE/src・Python の検索パス）。書ける場所があれば止める
if ! _writable_scan before; then die "dedicated user can write inside the scanned trees (before)" 92; fi

# ---- 4. sysctl（F-03）・AppArmor の記録（R-1）：trap は上で張り済み。終わりに cleanup が必ず戻す
_write_apparmor
SYSCTL_PREV="$(cat /proc/sys/user/max_user_namespaces 2>/dev/null || echo "")"
[ -n "$SYSCTL_PREV" ] || die "cannot read user.max_user_namespaces" 95
if echo 0 > /proc/sys/user/max_user_namespaces 2>/dev/null; then
  now="$(cat /proc/sys/user/max_user_namespaces 2>/dev/null || echo "")"
  [ "$now" = "0" ] || die "sysctl did not take effect" 95
  SYSCTL_STATE="changed"; _write_sysctl
else
  die "cannot set user.max_user_namespaces=0" 95
fi
# 事前の確認：追跡される側が user・mount の名前空間を作れないこと
if "${RUN_AS[@]}" "${ENVS_CHK[@]}" "$UNSHARE" -U -r true 2>/dev/null; then die "dedicated user can create a user namespace" 92; fi
if "${RUN_AS[@]}" "${ENVS_CHK[@]}" "$UNSHARE" -m true 2>/dev/null; then die "dedicated user can create a mount namespace" 92; fi

# ---- 5. 本体：実行ごとの cgroup に入れ、unshare の net・pid・mount の中で strace の下に動かす
echo "g3_run_user=${USER_NAME} uid=${DUID} gid=${DGID} groups=${PGROUPS} user_preexisted=${USER_PREEXISTED}"
# 分離の確認（本番と同じ組み立て）：つながったら止める
"$UNSHARE" --net -- "${RUN_AS[@]}" "${ENVS_CHK[@]}" "$PY" -B -c "import socket,sys
try:
    socket.create_connection(('1.1.1.1', 443), 3); print('network reachable'); sys.exit(1)
except OSError as e:
    print('g3_netns_check=isolated', type(e).__name__)"

rc=0
# cgroup.procs に自分を入れてから unshare を exec する小さな包み。中の子孫はすべてこの cgroup に入る。
WRAP="$BASE/evidence/wrap.sh"
cat > "$WRAP" <<WRAPEOF
#!/usr/bin/env bash
set -euo pipefail
echo \$\$ > "$CG/cgroup.procs"
exec "$UNSHARE" --net --pid --fork --kill-child --mount --propagation private --mount-proc -- \\
  /usr/bin/env -i PATH="$PATH" PY="$PY" STRACE="$STRACE" SETPRIV="$SETPRIV" DUID="$DUID" DGID="$DGID" \\
    LDLP="$LDLP" RUN_ID="$RUN_ID" RACE_DATE="$RACE_DATE" ONLY="$ONLY" BASE="$BASE" SRC="$BASE/src" \\
    OUT="$BASE/out" EVIDENCE="$BASE/evidence" RUNHOME="$RUNHOME" SOCK_FINDER="$SOCK_FINDER" \\
    G3_STRACE_TRACE="$G3_STRACE_TRACE" \\
  bash "$INNER"
WRAPEOF
chmod 0700 "$WRAP"
bash "$WRAP" || rc=$?
INNER_RC=$rc

# ---- 6. 止める・確かめる（F-02）。名前空間の中（PID 1 の終了）とホスト側（cgroup が空）と開いた FD を分ける
echo 1 > "$CG/cgroup.kill" 2>/dev/null || true
CG_EMPTY=0
for _ in $(seq 1 50); do grep -q '^populated 0$' "$CG/cgroup.events" 2>/dev/null && { CG_EMPTY=1; break; }; sleep 0.2; done
# 予備：ホスト全体で g3diag のプロセスを止める（uid が専用で、実行前に0を確認済みなので安全）
UID_AFTER=0
for _ in $(seq 1 10); do
  pkill -KILL -u "$DUID" 2>/dev/null || true
  if ! pgrep -u "$DUID" >/dev/null 2>&1; then UID_AFTER=0; break; fi
  UID_AFTER=1; sleep 1
done
# 抜け出し：この実行の cgroup の外にいる g3diag のプロセスがないか
STRAY=0
if pgrep -u "$DUID" >/dev/null 2>&1; then STRAY=1; fi
# 開いた FD：BASE/out を指す開いた FD がないか
OPEN_FDS=0
for l in /proc/[0-9]*/fd/*; do
  tgt="$(readlink "$l" 2>/dev/null || true)"
  case "$tgt" in "$BASE/out"/*|"$BASE/out") OPEN_FDS=$((OPEN_FDS+1));; esac
done
KILL_OK=0
if [ "$CG_EMPTY" = 1 ] && [ "$UID_AFTER" = 0 ] && [ "$STRAY" = 0 ] && [ "$OPEN_FDS" = 0 ]; then KILL_OK=1; fi

# ---- 7. 凍結・一覧・SHA-256・写し（F-01・F-08・F-10）
# どの場合でも、out は root の持ち物・書き込み不可にする（新しく開くことを防ぐ）
chown -R root:root "$BASE/out" || true
chmod -R a-w,a+rX "$BASE/out" || true
FROZEN=1
# BASE/src が実行の前後で変わっていないこと（F-05）
( cd "$BASE/src" && find . -type f -print0 | sort -z | xargs -0 -r sha256sum ) > "$BASE/evidence/src_after.sha256" || true
SRC_SHA_AFTER="$(sha256sum "$BASE/evidence/src_after.sha256" | cut -d' ' -f1)"
SRC_UNCHANGED=0; [ "$SRC_SHA_AFTER" = "$SRC_SHA_BEFORE" ] && SRC_UNCHANGED=1
# R-2：診断の後も、書ける場所の走査（前後の両方で検査する）
WRITABLE_AFTER_OK=0; if _writable_scan after; then WRITABLE_AFTER_OK=1; fi

# strace の記録（生）の SHA-256 は、止まったかどうかに関わらず root が書く（evidence は 0700）
if [ -f "$BASE/evidence/strace.txt" ]; then
  chmod 0444 "$BASE/evidence/strace.txt" || true
  sha256sum "$BASE/evidence/strace.txt" > "$PUB/strace.sha256" || true
  chmod 0444 "$PUB/strace.sha256" || true
fi

if [ "$KILL_OK" = 1 ] && [ "$SRC_UNCHANGED" = 1 ] && [ "$WRITABLE_AFTER_OK" = 1 ]; then
  STATE="ok"
  # F-10 G-11：collect の前に出力木を検査（records.jsonl＋summary.json＋races の未分類/欠落/不整合）。
  #   コードは信頼される WS 側（SELF_DIR/.. ＝ rebuild）、データは BASE/out（子が書いた未信頼入力）。
  SCAN_JSON="$("$PY" -I -P -B -c 'import sys, json, pathlib
sys.path.insert(0, "'"$SELF_DIR"'/..")
from shadow.g3_isolation import scan_output_tree
print(json.dumps(scan_output_tree(pathlib.Path(sys.argv[1]))))' "$BASE/out" 2>/dev/null)"
  # 合否：JSON 解析失敗・crash・ok 偽・空 はすべて不合格側（成功扱いで後続へ進まない）
  if ! printf '%s' "$SCAN_JSON" | "$PY" -I -P -B -c 'import sys, json
try: sys.exit(0 if json.load(sys.stdin).get("ok") is True else 1)
except Exception: sys.exit(1)'; then
    STATE="unclassified_output"
    log "unclassified output detected before collect (no collect; verify must fail)"
  fi
  # 警告：summary:unknown_keys_present が実在するときだけ固定文言（キー名・値・problems は出さない）
  if printf '%s' "$SCAN_JSON" | "$PY" -I -P -B -c 'import sys, json
try: w = json.load(sys.stdin).get("warnings") or []
except Exception: w = []
sys.exit(0 if "summary:unknown_keys_present" in w else 1)'; then
    log "scan warning: summary contains unrecognized keys (ignored; not recorded)"
  fi
  if [ "$STATE" = "ok" ]; then
    # 一覧（out.list。リンク・通常のファイル以外・外への hardlink があれば collect は失敗する）
    if "$PY" -I -P -B "$COLLECT" list "$BASE/out" "$PUB/out.list"; then
      sha256sum "$PUB/out.list" > "$PUB/out.list.sha256" || true
      # 通常のファイルだけを、リンクをたどらずに BASE/collect に写す
      mkdir -m 0755 "$BASE/collect"
      if "$PY" -I -P -B "$COLLECT" copy "$BASE/out" "$BASE/collect" "$PUB/out.list" "$PUB/collect.list"; then
        chown -R root:root "$BASE/collect"; chmod -R a-w,a+rX "$BASE/collect"
        sha256sum "$PUB/collect.list" > "$PUB/collect.list.sha256" || true
      else
        STATE="collect_invalid"
      fi
    else
      STATE="collect_invalid"
    fi
  fi
else
  if [ "$KILL_OK" != 1 ]; then STATE="kill_failed"
  elif [ "$SRC_UNCHANGED" != 1 ]; then STATE="src_changed"
  else STATE="writable_dirty"; fi
  log "diagnostic not frozen cleanly: kill_ok=$KILL_OK src_unchanged=$SRC_UNCHANGED writable_after_ok=$WRITABLE_AFTER_OK (no collect; verify must fail)"
fi

# 状態を記録（検査が最初に読む。ok 以外なら検査は out を開かない）
{ echo "state=$STATE"; echo "inner_rc=$INNER_RC"; echo "cg_empty=$CG_EMPTY"; echo "uid_after=$UID_AFTER";
  echo "stray=$STRAY"; echo "open_fds_into_out=$OPEN_FDS"; echo "src_unchanged=$SRC_UNCHANGED";
  echo "writable_after_ok=$WRITABLE_AFTER_OK";
  echo "user_preexisted=$USER_PREEXISTED"; echo "kill_ok=$KILL_OK"; } > "$PUB/status.txt"
chmod 0444 "$PUB/status.txt"
for f in "$PUB/out.list" "$PUB/out.list.sha256" "$PUB/collect.list" "$PUB/collect.list.sha256"; do
  [ -f "$f" ] && chmod 0444 "$f" || true
done
echo "g3_run_state=${STATE} inner_rc=${INNER_RC} kill_ok=${KILL_OK}"

# 診断が清く止まり凍結できたときだけ rc 0。それ以外は失敗（検査も必ず不合格）
if [ "$STATE" = "ok" ]; then exit "$INNER_RC"; fi
case "$STATE" in kill_failed) exit 93;; src_changed) exit 92;; writable_dirty) exit 98;; collect_invalid) exit 94;; unclassified_output) exit 99;; *) exit 90;; esac
