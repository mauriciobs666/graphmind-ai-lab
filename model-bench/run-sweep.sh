#!/usr/bin/env bash
# Resilient runner for the catalog sweep (Unit U5). Reads a plan file of
# "model|pack" lines, invokes ./run.sh run for each, catches failures without
# stopping (FR-5), and appends one line per attempt to the log incrementally
# so progress survives a kill. Usage:
#   ./run-sweep.sh <plan-file> <session-id> <log-file>
set -uo pipefail   # NOT -e: a failing run must not stop the loop (FR-5)

PLAN="$1"
SESSION="$2"
LOG="$3"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

touch "$LOG"

total=$(grep -c . "$PLAN")
n=0
ok=0
fail=0
consecutive_unreachable=0

while IFS='|' read -r model pack; do
  [ -z "$model" ] && continue
  n=$((n+1))
  ts_start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[$ts_start] ($n/$total) START model=$model pack=$pack" | tee -a "$LOG"
  out="$("$HERE/run.sh" run --pack "$pack" --model "$model" --session "$SESSION" 2>&1)"
  rc=$?
  ts_end="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ $rc -eq 0 ]; then
    ok=$((ok+1))
    consecutive_unreachable=0
    echo "[$ts_end] ($n/$total) OK   model=$model pack=$pack" | tee -a "$LOG"
  else
    fail=$((fail+1))
    echo "[$ts_end] ($n/$total) FAIL model=$model pack=$pack rc=$rc" | tee -a "$LOG"
    # Indent the captured output so it's visibly attached to this FAIL line
    # but never mistaken for the next START line.
    echo "$out" | sed 's/^/    | /' | tee -a "$LOG"

    if [ "$rc" -eq 3 ]; then
      # exit 3 = LM Studio unreachable (per README's closed exit-code set).
      # A single model failing to load is a different failure shape and does
      # NOT hit this branch. Confirm independently rather than trusting rc
      # alone, then require two in a row before treating this as the whole
      # API being down (a single flaky probe should not abort 70 remaining
      # combinations).
      if curl -s -o /dev/null -m 5 "http://localhost:1234/api/v0/models"; then
        consecutive_unreachable=0
      else
        consecutive_unreachable=$((consecutive_unreachable+1))
        echo "[$ts_end] LM Studio API probe failed independently (consecutive=$consecutive_unreachable)" | tee -a "$LOG"
      fi
      if [ "$consecutive_unreachable" -ge 2 ]; then
        echo "==== SWEEP HALTED: LM Studio API unreachable for 2 consecutive combinations (whole-API outage suspected, not a single-model load failure) at ($n/$total). $ok ok, $fail fail so far. Remaining combinations NOT attempted. ====" | tee -a "$LOG"
        exit 42
      fi
    else
      consecutive_unreachable=0
    fi
  fi
done < "$PLAN"

echo "==== SWEEP DONE: $ok ok, $fail fail, $total total ====" | tee -a "$LOG"
