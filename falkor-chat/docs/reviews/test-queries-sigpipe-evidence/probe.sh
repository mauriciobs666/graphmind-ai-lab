#!/usr/bin/env bash
# Probe: do test_queries.sh's assert_contains / assert_not_contains give false results?
# Usage: probe.sh <path-to-test_queries.sh> [iterations]
# Extracts the two helpers verbatim from the given script and runs them under the
# script's own `set -euo pipefail`. Reports, per control, how many of N calls gave the
# WRONG verdict (a false red or a false green). Correct helpers: every count is 0.
set -euo pipefail
SRC="$1"; N="${2:-200}"

# Pull the two helper definitions out of the script, verbatim.
eval "$(sed -n '/^assert_contains() {/,/^}/p; /^assert_not_contains() {/,/^}/p' "$SRC")"
declare -F assert_contains >/dev/null && declare -F assert_not_contains >/dev/null \
  || { echo "could not extract helpers from $SRC"; exit 2; }

# Large output (~300 KB) with the needle in the first line; small output modelled on a redis-cli reply.
big=$(printf 'needle\n'; head -c 300000 /dev/zero | tr '\0' 'x')
small=$'m.msgId\nmn3\nCached execution: 0\nQuery internal execution time: 0.412 milliseconds'
dash=$'-1\n--flag value\nCached execution: 0'   # leading-dash needles

# run_control <helper> <haystack-var> <needle> <expected verdict: pass|fail>
run_control() {
  local helper=$1 hayvar=$2 needle=$3 want=$4 wrong=0 i
  local hay=${!hayvar}
  for ((i = 0; i < N; i++)); do
    PASS=0 FAIL=0
    # Helpers print; discard. Run in the current shell so `set -e`/pipefail apply as in the suite.
    "$helper" "probe" "$needle" "$hay" >/dev/null 2>&1 || true
    if [ "$want" = pass ] && [ "$PASS" -ne 1 ]; then wrong=$((wrong + 1)); fi
    if [ "$want" = fail ] && [ "$FAIL" -ne 1 ]; then wrong=$((wrong + 1)); fi
  done
  printf '%-20s %-6s needle=%-14q want=%-4s -> wrong verdicts %d/%d\n' \
    "$helper" "$hayvar" "$needle" "$want" "$wrong" "$N"
  TOTAL_WRONG=$((TOTAL_WRONG + wrong))
}

TOTAL_WRONG=0
echo "probe of: $SRC  (N=$N)"
# assert_contains: present -> pass, absent -> fail
run_control assert_contains     big   needle  pass
run_control assert_contains     big   zzz     fail
run_control assert_contains     small mn3     pass
run_control assert_contains     small zzz     fail
# assert_not_contains: present -> fail, absent -> pass
run_control assert_not_contains big   needle  fail
run_control assert_not_contains big   zzz     pass
run_control assert_not_contains small mn3     fail
run_control assert_not_contains small zzz     pass
# leading '-' in the expected string must be a pattern, not a grep option
run_control assert_contains     dash  -1      pass
run_control assert_contains     dash  --flag  pass
run_control assert_contains     dash  -zzz    fail
run_control assert_not_contains dash  -1      fail
run_control assert_not_contains dash  --flag  fail
run_control assert_not_contains dash  -zzz    pass
echo "TOTAL wrong verdicts: $TOTAL_WRONG"
