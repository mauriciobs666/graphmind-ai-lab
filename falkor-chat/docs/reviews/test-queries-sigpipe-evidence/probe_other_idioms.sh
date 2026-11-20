#!/usr/bin/env bash
# Probe the non-helper idioms under set -euo pipefail: does the command substitution return a
# non-zero status (which, in the suite, aborts the script via set -e) although the answer is
# well-defined? `|| exit 9` makes the substitution status explicit, since set -e is suspended
# on the left of `||`.
set -euo pipefail
N=${1:-200}
big=$'h.col\n\n'"$(head -c 300000 /dev/zero | tr '\0' 'x')"       # header, blank line, then bulk
many=$(for i in $(seq 1 60000); do echo "row m1 $i"; done)        # many matching lines
count() { local name=$1 code=$2 bad=0 i; for ((i=0;i<N;i++)); do ( eval "$code" ) >/dev/null 2>&1 || bad=$((bad+1)); done; echo "$name -> spurious failure $bad/$N"; }
count "OLD awk  (printf | awk exit)"    'actual=$big; t=$(printf "%s\n" "$actual" | awk "/^\$/{exit} {print}") || exit 9; [ "$t" = "h.col" ]'
count "NEW awk  (awk <<<)"              'actual=$big; t=$(awk "/^\$/{exit} {print}" <<<"$actual") || exit 9; [ "$t" = "h.col" ]'
count "OLD line (echo|grep -n|head -1)" 'out=$many; l=$(echo "$out" | grep -n "m1" | head -1 | cut -d: -f1) || exit 9; [ "$l" = 1 ]'
count "NEW line (grep -n -m1 <<<|cut)"  'out=$many; l=$(grep -n -m1 "m1" <<<"$out" | cut -d: -f1) || exit 9; [ "$l" = 1 ]'
# semantics kept: an absent needle still makes the substitution non-zero, exactly as before
count "OLD line, absent needle (expect N/N)" 'out=$many; l=$(echo "$out" | grep -n "zz" | head -1 | cut -d: -f1) || exit 9; true'
count "NEW line, absent needle (expect N/N)" 'out=$many; l=$(grep -n -m1 "zz" <<<"$out" | cut -d: -f1) || exit 9; true'
