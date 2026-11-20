#!/usr/bin/env bash
# Probe: does `echo "$x" | grep -qF needle` under pipefail report a miss although the needle is present?
set -euo pipefail
small=$'m.msgId\nmn3\nCached execution: 0\nQuery internal execution time: 0.412 milliseconds'
big=$(printf 'needle\n'; head -c 200000 /dev/zero | tr '\0' 'x')
for label in small big; do
  x=${!label}; miss=0
  for i in $(seq 1 2000); do
    if echo "$x" | grep -qF "${1:-mn3}"; then :; else miss=$((miss+1)); fi
  done
  echo "$label: false misses in 2000 = $miss"
done
