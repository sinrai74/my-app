#!/usr/bin/env bash
# 正式G3（B案）段階3：名前空間の中で PID 1 として動く台本（root のまま。外側の台本から unshare で呼ばれる）。
#
# 役割（第3版 §0・F-04・F-07）：
#   - マウントを private にして、/run・/tmp・/var/tmp・/dev/shm に空の tmpfs をかぶせる
#     （外のソケット・/tmp を見えなくする）。/var/run は /run への link なら /run だけにかぶせる。
#   - 書ける AF_UNIX のソケットが残っていないかを、追跡される側（g3diag）として調べる。あれば止める（F-04）。
#   - strace を exec する（PID 1 になる）。strace が終わると、カーネルがこの名前空間の中を残らず止める（F-02）。
#
# 環境変数で受け取る（外側の台本が env -i で渡す）：
#   PY STRACE SETPRIV DUID DGID LDLP RUN_ID RACE_DATE ONLY BASE SRC OUT EVIDENCE PRIV_RUN RUNHOME SOCK_FINDER
set -euo pipefail

log() { echo "g3_inner: $*"; }

# 1. マウントを private にして、外へ伝わらないようにする
mount --make-rprivate / 2>/dev/null || mount --make-rprivate / -o remount 2>/dev/null || {
  log "cannot make mounts private"; exit 80; }

# 2. /run（と、link でなければ /var/run）・/tmp・/var/tmp・/dev/shm に空の tmpfs
varrun_real="$(readlink -f /var/run || echo /var/run)"
targets=(/run /tmp /var/tmp /dev/shm)
if [ "$varrun_real" != "/run" ]; then targets+=(/var/run); fi
for d in "${targets[@]}"; do
  [ -d "$d" ] || continue
  mount -t tmpfs -o "nosuid,nodev,noexec,mode=1777" g3tmp "$d" || { log "cannot tmpfs $d"; exit 81; }
done
# 追跡される側の HOME と TMPDIR は、この名前空間の中の tmpfs の下に作る
install -d -m 0700 -o "$DUID" -g "$DGID" "$RUNHOME" "/tmp/g3diag" || { log "cannot make runhome"; exit 81; }
cat /proc/self/mountinfo > "$EVIDENCE/mountinfo.txt" 2>/dev/null || true

# 3. 書ける AF_UNIX のソケットが、追跡される側から見えないこと（F-04）
#    追跡される側（g3diag。補助グループなし）として、proc・sys・cgroup を除いて、書けるソケットを探す。
#    SOCK_FINDER は find の台本のパス（root の持ち物）。見つかれば止める。
sock_out="$EVIDENCE/writable_sockets.txt"
set +e
$SETPRIV --reuid="$DUID" --regid="$DGID" --clear-groups --no-new-privs \
  /usr/bin/env -i PATH="$PATH" bash "$SOCK_FINDER" > "$sock_out" 2>"$EVIDENCE/sock_finder.err"
sf_rc=$?
set -e
if [ "$sf_rc" != 0 ]; then log "socket finder failed rc=$sf_rc"; exit 82; fi
if [ -s "$sock_out" ]; then log "writable sockets are reachable by the dedicated user"; sed 's/^/  /' "$sock_out"; exit 82; fi
log "writable_sockets=none"

# 4. 本体：strace を exec（PID 1 になる）。追跡される側だけ g3diag。
cd "$SRC/rebuild"
exec "$STRACE" --kill-on-exit -f -qq -yy -s 256 -e "trace=${G3_STRACE_TRACE}" -o "$EVIDENCE/strace.txt" \
  "$SETPRIV" --reuid="$DUID" --regid="$DGID" --clear-groups --no-new-privs \
  /usr/bin/env -i PATH="$PATH" HOME="$RUNHOME" LANG=C.UTF-8 LD_LIBRARY_PATH="$LDLP" \
    PYTHONPATH="$SRC/rebuild:$SRC" G3_RUN_ID="$RUN_ID" USE_REBUILD_PIPELINE=false TMPDIR=/tmp/g3diag \
    PYTHONDONTWRITEBYTECODE=1 \
  "$PY" -P -B -m actions.g3_diagnostic_entry parent --race-date "$RACE_DATE" \
    --programs "$SRC/g3_in/programs_raw.json" --odds "$SRC/g3_in/odds.json" \
    --out-dir "$OUT" --only "$ONLY"
