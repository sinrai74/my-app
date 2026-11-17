#!/usr/bin/env bash
# 正式G3（B案）段階3：追跡される側（g3diag）として走らせ、与えた木の中で書ける場所を探す（再監査 F-05・第3版 R-2）。
# 引数：検査する根のフォルダ（workspace・BASE/src・Python の検索パス・必要な実行時パス）。
# proc・sysfs・cgroup・cgroup2 は除く（擬似ファイル系。書けても意味がなく、探索が重い）。-xdev は使わない
#   （木の下の別マウントの書ける場所も見るため）。
# 出力：
#   標準出力（fd1）… 書ける場所を NUL 区切りで（ファイル名に改行が含まれても壊れない）。
#   標準エラー（fd2）… find の誤り。呼ぶ側が「権限拒否（たどれない＝g3diag には届かない。安全）」と
#                       「それ以外の硬い誤り（成功扱いにしない）」に分ける。
# access(2)（-writable）で判定するので、権限・ACL・読み込み専用のマウントが反映される。
# 戻り値は常に 0（合否は呼ぶ側が、書ける場所の件数と硬い誤りの有無で決める）。
set -uo pipefail
PRUNE=( '(' -path /proc -o -path /sys -o -fstype proc -o -fstype sysfs -o -fstype cgroup -o -fstype cgroup2 ')' -prune )
for root in "$@"; do
  [ -n "$root" ] || continue
  if [ ! -e "$root" ]; then echo "MISSING $root" >&2; continue; fi
  find "$root" "${PRUNE[@]}" -o -writable -print0 || true
done
exit 0
