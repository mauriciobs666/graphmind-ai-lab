#!/usr/bin/env bash
# Compare the shipped assert idiom with two candidate fixes, positive and negative controls.
set -euo pipefail
big=$(printf 'needle\n'; head -c 200000 /dev/zero | tr '\0' 'x')
small=$'m.msgId\nmn3\nCached execution: 0'
run() { # $1 label, $2 idiom, $3 haystack var, $4 needle ; prints matches/2000
  local x=${!3} hit=0
  for i in $(seq 1 2000); do
    case $2 in
      pipe_q)  if echo "$x" | grep -qF -- "$4"; then hit=$((hit+1)); fi ;;
      herestr) if grep -qF -- "$4" <<<"$x"; then hit=$((hit+1)); fi ;;
      pipe_noq) if echo "$x" | grep -F -- "$4" >/dev/null; then hit=$((hit+1)); fi ;;
    esac
  done
  echo "$1 $2 hay=$3 needle=$4 -> matched $hit/2000"
}
for idiom in pipe_q herestr pipe_noq; do
  run present $idiom small mn3
  run present $idiom big needle
  run absent  $idiom small zzz
  run absent  $idiom big zzz
done
