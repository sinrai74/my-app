#!/usr/bin/env bash
# 正式G3（B案）段階3：追跡される側（g3diag）として走らせ、書ける AF_UNIX のソケットを探す（再監査 F-04）。
# proc・sys・cgroup は除く。-writable は access(2) なので、権限・ACL・読み込み専用のマウントが反映される。
# たどれないフォルダは「つなげない」と同じ扱い（g3diag の権限で見える範囲だけを見る）。
# 標準出力に、書けるソケットのパスを1行ずつ。エラーでも 0 で終わる（本体の判定は、出力が空かどうかで行う）。
set -uo pipefail
find / -xdev \
  \( -path /proc -o -path /sys -o -fstype proc -o -fstype sysfs -o -fstype cgroup -o -fstype cgroup2 \) -prune -o \
  -type s -writable -print 2>/dev/null || true
# 他のマウント（/run などの tmpfs は空のはずだが、別のマウントがあれば）も見る
for mp in $(awk '$2 !~ /^\/(proc|sys)/ {print $2}' /proc/self/mounts 2>/dev/null | sort -u); do
  [ "$mp" = "/" ] && continue
  find "$mp" -xdev -type s -writable -print 2>/dev/null || true
done
exit 0
