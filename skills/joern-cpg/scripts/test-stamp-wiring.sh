#!/usr/bin/env bash
# test-stamp-wiring.sh — regression test for the CpgBuildInfo stamp block in
# pipeline.sh. Run it: skills/joern-cpg/scripts/test-stamp-wiring.sh
#
# WHY THIS EXISTS. The stamp's correctness was verified three times at the wrong
# altitude. The Cypher construct was executed against a live FalkorDB; the bash
# helpers were exercised in isolation; both passed — and the shipped pipeline was
# broken anyway, because `pipeline.sh` called cpg_provenance_stamp inside `$(…)`
# and the subshell discarded the allow-list the next assertion depends on. Every
# `--load` build would have failed after a multi-hour parse, with the marker
# already replaced, under a message saying the condition was impossible.
#
# So this test drives THE ACTUAL LINES OF pipeline.sh — extracted between two
# anchors, never retyped — against a fake `redis-cli` that models FalkorDB's
# reply shapes. It needs no FalkorDB, writes to no graph, and takes under a
# second. What it protects is the CALL PATH, which is the level every previous
# check skipped.
#
# AND THEN THE FIRST VERSION OF THIS FILE REPEATED THE MISTAKE ONE LEVEL UP.
# Its oracle was `rc == 0 -> PASS`, else scrape stray key names out of the
# output. It never asked WHICH non-zero, and never asked whether the failure
# branch had finished. So deleting `replay_stamp`'s definition — the exact
# defect the file was written to close — left all six cases green: the case was
# aborting at rc 127 on `replay_stamp: command not found`, and the scrape still
# found its expected key among the lines printed before the abort. The harness
# was sound and the oracle was soft, which is the same failure as before wearing
# a different hat. Hence, now, per case: an EXACT expected exit code, and — for
# every case that is supposed to fail — a POSITIVE assertion that the branch
# actually reached its end and printed the rendered stamp.
#
# It deliberately does NOT test the Cypher semantics (whether `SET b = {…}`
# really replaces): the fake models that, it does not prove it. That half is
# graph-dba's executed probes, recorded in git-provenance.sh's docstring. This
# half is the wiring.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FAIL=0

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

# ---- fake redis-cli: models the reply shapes the stamp block provokes --------
# State lives in $WORK/marker.keys (one key per line) and $WORK/parsed_at.
# MODE=correct         -> `SET b = {…}` replaces the property set (real semantics)
# MODE=merge           -> the SERVER merges instead of replacing (models a
#                         FalkorDB semantics change; the CLIENT emitting `+=` is
#                         modelled separately — see "derived from the query")
# MODE=stamp_rejected  -> the stamp write comes back as a Cypher error
# MODE=stamp_lost      -> the stamp write reports success but the marker keeps
#                         an OLDER build's PARSED_AT (the write did not land)
# MODE=stray_error     -> the stray read comes back as a BARE runtime error,
#                         which is what FalkorDB really returns and what
#                         redis-cli really exits 0 on
#
# REPLY SHAPES ARE MEASURED, NOT IMAGINED. Probed read-only against the live
# instance 2026-09-08 (`GRAPH.RO_QUERY cpg_falkorchat`):
#   success        -> column header, rows, `Cached execution: 0`,
#                     `Query internal execution time: 0.525177 milliseconds`
#   runtime error  -> ONE bare line (`Unknown function 'x'`, `Type mismatch: …`),
#                     no header, no trailer, redis-cli exit 0
#   syntax error   -> one bare `errMsg: …` line, redis-cli exit 0
# The trailer is what pipeline.sh's stray read now requires, so the fake has to
# emit it or the whole suite would pass for the wrong reason.
cat > "$WORK/redis-cli" <<'FAKE'
#!/usr/bin/env bash
set -uo pipefail
q="${!#}"   # last arg is the query
ok_trailer() { echo "Cached execution: 0"; echo "Query internal execution time: 0.1 milliseconds"; }
case "$q" in
  *"MERGE (b:CpgBuildInfo)"*)
    if [ "${MODE:-correct}" = stamp_rejected ]; then
      # a bare `errMsg:` line, exit 0 — the shape a Cypher error really has
      echo "errMsg: Invalid input 'X': expected a clause line: 2, column: 1"; exit 0
    fi
    # keys of the map = lines of the form `NAME: value` whose value is not NULL
    new="$(printf '%s\n' "$q" | sed -nE 's/^[[:space:]]*([A-Z][A-Z0-9_]*): (.*)$/\1 \2/p' \
           | grep -v ' NULL,\?$' | sed 's/ .*//')"
    # MERGE-VS-REPLACE IS DERIVED FROM THE QUERY TEXT, not only from MODE. `SET
    # b = {` is the replacing form; anything else the client might emit (`+=`, a
    # reversion to `b.X = …`) merges. Without this derivation the harness
    # dictated the semantics it was supposed to be observing, and changing
    # git-provenance.sh's emitted Cypher to `SET b += {` passed the whole suite
    # green — while pipeline.sh's own failure message tells the operator that a
    # `+=` reversion is one of the two things to go and check.
    qmode=merge
    case "$q" in *"SET b = {"*) qmode=replace ;; esac
    if [ "${MODE:-correct}" = merge ] || [ "$qmode" = merge ]; then
      cat "$WORK/marker.keys" > "$WORK/.tmp"; printf '%s\n' $new >> "$WORK/.tmp"
      sort -u "$WORK/.tmp" > "$WORK/marker.keys"
    else
      printf '%s\n' $new | sort -u > "$WORK/marker.keys"
    fi
    if [ "${MODE:-correct}" = stamp_lost ]; then
      echo "2026-01-01T00:00:00Z" > "$WORK/parsed_at"
    else
      printf '%s\n' "$q" | sed -nE 's/^[[:space:]]*PARSED_AT: "([^"]*)".*/\1/p' > "$WORK/parsed_at"
    fi
    echo "Properties set: 8"; ok_trailer; exit 0 ;;
  *"RETURN b.PARSED_AT"*)
    echo "b.PARSED_AT"; cat "$WORK/parsed_at"; ok_trailer; exit 0 ;;
  *"UNWIND keys(b)"*)
    if [ "${MODE:-correct}" = stray_error ]; then
      # BARE, no `errMsg:` prefix, no trailer, exit 0 — measured, not invented
      echo "Type mismatch: expected Map, Node, Edge, or Null but was String"; exit 0
    fi
    allow="$(printf '%s\n' "$q" | sed -nE "s/.*NOT k IN \[(.*)\].*/\1/p" | tr -d "'" | tr ',' ' ')"
    echo "stray"
    while read -r k; do
      [ -n "$k" ] || continue
      hit=0; for a in $allow; do [ "$a" = "$k" ] && hit=1; done
      [ "$hit" = 0 ] && echo "STRAY_KEY=$k"
    done < "$WORK/marker.keys"
    ok_trailer
    exit 0 ;;
esac
echo "errMsg: fake redis-cli got an unmodelled query"; exit 0
FAKE
chmod +x "$WORK/redis-cli"
export WORK PATH="$WORK:$PATH"

# ---- the block under test, lifted verbatim from pipeline.sh ------------------
START='  BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"'
END='  echo "pipeline: stamp verified by read-back." >&2'
BLOCK="$(awk -v s="$START" -v e="$END" '$0==s{f=1} f{print} $0==e{exit}' "$HERE/pipeline.sh")"
if [ -z "$BLOCK" ]; then
  echo "FAIL: could not extract the stamp block from pipeline.sh (START anchor moved)"; exit 1
fi
# Losing the END anchor is otherwise diagnosed as `syntax error near unexpected
# token 'fi'`, because extraction runs on to the file-closing `fi`. Correct
# outcome, misleading message; say what actually happened.
case "$BLOCK" in
  *"$END"*) ;;
  *) echo "FAIL: the extracted block never reaches the END anchor (it moved or was reworded)"; exit 1 ;;
esac
case "$BLOCK" in
  *'cpg_provenance_stray_query'*) ;;
  *) echo "FAIL: extracted block does not contain the stray assertion"; exit 1 ;;
esac

# run_block <mode> <provenance> <extra bash appended to the block>
#   Prints the run's merged stdout+stderr and EXITS WITH THE BLOCK'S OWN rc.
#   It has to be the exit status, not a global: the caller reads this through
#   `$(…)`, and a subshell discards an assignment — which is the very defect
#   this whole test file exists to catch, and which it hit here on first run.
run_block() {
  local mode="$1" prov="$2" extra="$3" out rc
  out="$(
    MODE="$mode" bash -c '
      set -euo pipefail
      . "'"$HERE"'/git-provenance.sh"
      HOST=localhost; PORT=6379; GRAPH=cpg_fake; SRC=/src
      PARSED_AT=2026-09-08T07:00:00Z; PROVENANCE="'"$prov"'"
      # cpg_provenance_capture ALWAYS initialises these four, to "" when it
      # fails — pipeline.sh log lines index them without a default, so an
      # unset one is a `set -u` abort. Mirror that contract exactly.
      CPG_SOURCE_ORIGIN=""; CPG_SOURCE_COMMIT=""; CPG_SOURCE_TREE=""; CPG_SOURCE_DIRTY=""
      if [ "$PROVENANCE" != none ]; then
        CPG_SOURCE_ORIGIN=falkor-chat/server; CPG_SOURCE_COMMIT=aaaa; CPG_SOURCE_TREE=bbbb; CPG_SOURCE_DIRTY=false
      fi
      '"$BLOCK"'
      '"$extra"'
    ' 2>&1
  )"; rc=$?
  printf '%s' "$out"
  return "$rc"
}

# run_case <name> <mode> <initial marker keys> <provenance> <expect_rc> <expect> [must-contain]…
#   <expect>  PASS, or the stray key names the run is supposed to report
#   THE EXIT CODE IS ASSERTED EXACTLY. `non-zero` is not an assertion: rc 127
#   from an undefined function is non-zero and means the branch never ran.
run_case() {
  local name="$1" mode="$2" initial="$3" prov="$4" expect_rc="$5" expect="$6"; shift 6
  local out rc got problems="" want
  printf '%s\n' $initial | sort -u > "$WORK/marker.keys"; : > "$WORK/parsed_at"
  # the wiring assertion this whole file exists for, appended after the block
  out="$(run_block "$mode" "$prov" 'echo "ALLOWLIST=[${CPG_STAMPED_KEYS:-<UNSET>}]"')"; rc=$?

  [ "$rc" = "$expect_rc" ] || problems="${problems}rc: expected $expect_rc, got $rc"$'\n'
  if [ "$rc" -eq 0 ]; then got=PASS; else
    got="$(printf '%s\n' "$out" | sed -n 's/^pipeline:   \([A-Z][A-Z0-9_]*\)$/\1/p' | tr '\n' ' ' | sed 's/ $//')"
    [ -n "$got" ] || got="FAILED(rc=$rc)"
  fi
  [ "$got" = "$expect" ] || problems="${problems}outcome: expected [$expect], got [$got]"$'\n'

  # POSITIVE assertion on every modelled failure: the branch has to have REACHED
  # ITS END and printed the rendered stamp. Without this a case can "fail
  # correctly" by aborting early — which is exactly how deleting replay_stamp
  # stayed green.
  if [ "$expect_rc" != 0 ]; then
    case "$out" in
      *"--- begin stamp ---"*) ;;
      *) problems="${problems}the failure branch printed no '--- begin stamp ---' block"$'\n' ;;
    esac
    printf '%s\n' "$out" | grep -q '^MERGE (b:CpgBuildInfo)$' \
      || problems="${problems}the printed block does not carry the stamp Cypher"$'\n'
  fi
  for want in "$@"; do
    case "$out" in
      *"$want"*) ;;
      *) problems="${problems}output lacks: $want"$'\n' ;;
    esac
  done

  if [ -z "$problems" ]; then
    echo "  PASS  $name  -> rc=$rc $got"
  else
    echo "  FAIL  $name"
    printf '%s' "$problems" | sed 's/^/          /'
    printf '%s\n' "$out" | sed 's/^/        | /'; FAIL=1
  fi
  printf '%s\n' "$out" | grep -o 'ALLOWLIST=\[[^]]*\]' | sed 's/^/        /'
}

P8="BUILT_AT PARSED_AT SOURCE_PATH PROVENANCE SOURCE_ORIGIN SOURCE_COMMIT SOURCE_TREE SOURCE_DIRTY"

echo "stamp wiring:"
run_case "correct replace over a hand-authored marker" correct "$P8 MARKER_ORIGIN NOTE MARKER_EVIDENCE" parse-root 0 PASS
run_case "correct replace, provenance=none"            correct "$P8 MARKER_ORIGIN NOTE"                 none      0 PASS
run_case "regression: merge semantics, foreign key"    merge   "$P8 MARKER_EVIDENCE"                    parse-root 1 MARKER_EVIDENCE
run_case "regression: merge, pipeline-clean marker"    merge   "$P8"                                    parse-root 0 PASS
run_case "subsumption: stale SOURCE_* under none"      merge   "$P8"                                    none      1 "SOURCE_COMMIT SOURCE_DIRTY SOURCE_ORIGIN SOURCE_TREE"

# The two branches where the stamp did NOT land. These are the only branches in
# which re-sending the Cypher is the fix, and until 2026-09-08 they were the two
# that did not print it — the replay was wired into the three branches that had
# already read the stamp back. So assert the WORDING, not just the exit code:
# a branch that tells the operator to re-send must be a branch where re-sending
# helps.
run_case "stamp rejected by FalkorDB"                  stamp_rejected "$P8" parse-root 1 "FAILED(rc=1)" \
  "only the provenance marker is missing" "does NOT need repeating — only the stamp does"
run_case "stamp write did not land"                    stamp_lost     "$P8" parse-root 1 "FAILED(rc=1)" \
  "the freshness stamp did not land" "does NOT need repeating — only the stamp does"

# The stray read comes back as a BARE runtime error. redis-cli exits 0, the
# reply carries no STRAY_KEY= and matches none of rq's error prefixes, so before
# the positive trailer requirement this reported "stamp verified by read-back"
# over a marker that was never checked.
run_case "stray read returns a bare runtime error"     stray_error    "$P8" parse-root 1 "FAILED(rc=1)" \
  "could not verify the marker's property list" "the stamp is NOT what needs"

# ---- P6-5(a): the stray query's own empty-allow-list refusal -----------------
# git-provenance.sh calls this one of "two mechanisms" protecting the
# allow-list. It is unreachable through the block — the call-site guard fires
# first — so deleting it was invisible to every case above. Call it directly.
#
# BOTH SHAPES, and the second is the one that matters. `unset` is killed by
# `set -u` whether the guard exists or not, so a test that only unsets proves
# nothing about the guard; SET-BUT-EMPTY is what a caller actually produces
# (cpg_provenance_stamp assigns CPG_STAMPED_KEYS="" before its first _cpg_prop),
# and with the guard gone it renders `NOT k IN []` and returns 0 — every
# property a stray, reported as a finding about the graph.
echo "stray-query guard (direct call, allow-list unusable):"
for shape in 'unset CPG_STAMPED_KEYS' 'CPG_STAMPED_KEYS=""'; do
  sq_out="$(bash -c '
      set -uo pipefail
      . "'"$HERE"'/git-provenance.sh"
      '"$shape"'
      cpg_provenance_stray_query
    ' 2>&1)"; sq_rc=$?
  sq_bad=""
  [ "$sq_rc" = 1 ] || sq_bad="expected rc 1, got $sq_rc"
  case "$sq_out" in *"CPG_STAMPED_KEYS is empty"*) ;; *) sq_bad="${sq_bad:+$sq_bad; }no refusal naming CPG_STAMPED_KEYS" ;; esac
  case "$sq_out" in *"NOT k IN []"*) sq_bad="${sq_bad:+$sq_bad; }it EMITTED a query with an empty allow-list" ;; esac
  if [ -z "$sq_bad" ]; then
    echo "  PASS  [$shape] refuses to emit a query that would call every property a stray"
  else
    echo "  FAIL  [$shape] $sq_bad"
    printf '%s\n' "$sq_out" | sed 's/^/        | /'; FAIL=1
  fi
done

# ---- mutation: would this test have caught P5-1? ----------------------------
# Revert the call site to the subshell form that shipped in 0da3eb9/5417f0e and
# assert the block now refuses to run. If this case ever reports PASS, the guard
# at the call site has been lost and the allow-list can silently go empty again.
MUT="$(printf '%s\n' "$BLOCK" | sed \
  -e 's/^  cpg_provenance_stamp "\$BUILT_AT" "\$PARSED_AT" "\$SRC" "\$PROVENANCE"$/  STAMP="$(cpg_provenance_stamp "$BUILT_AT" "$PARSED_AT" "$SRC" "$PROVENANCE")"/' \
  -e 's/^  STAMP="\$CPG_STAMP_CYPHER"$//')"
printf '%s\n' $P8 > "$WORK/marker.keys"; : > "$WORK/parsed_at"
mut_out="$(MODE=correct bash -c '
    set -euo pipefail
    . "'"$HERE"'/git-provenance.sh"
    HOST=localhost; PORT=6379; GRAPH=cpg_fake; SRC=/src
    PARSED_AT=2026-09-08T07:00:00Z; PROVENANCE=parse-root
    CPG_SOURCE_ORIGIN=falkor-chat/server; CPG_SOURCE_COMMIT=aaaa; CPG_SOURCE_TREE=bbbb; CPG_SOURCE_DIRTY=false
    '"$MUT"'
  ' 2>&1)"; mut_rc=$?
echo "P5-1 mutation (subshell call site):"
case "$mut_rc:$mut_out" in
  0:*)   echo "  FAIL  the subshell call site was accepted — the guard is gone"; FAIL=1 ;;
  *internal*) echo "  PASS  refused, naming the wiring: $(printf '%s\n' "$mut_out" | grep -m1 internal)" ;;
  *)     echo "  FAIL  refused, but not with the wiring message (rc=$mut_rc)"; printf '%s\n' "$mut_out" | sed 's/^/        | /'; FAIL=1 ;;
esac

# ---- mutation: the call-site guard's own diagnostic (P6-6) ------------------
# The guard's message has two paths and only one had ever run. Drop the
# allow-list AFTER a successful stamp so CPG_STAMP_CYPHER is set and
# CPG_STAMPED_KEYS is not — the one case this branch uniquely handles. The
# diagnostic must describe the Cypher, not paste it: the previous
# `${CPG_STAMP_CYPHER:+<set>}${CPG_STAMP_CYPHER:-<empty>}` printed `<set>`
# followed by the whole multi-line map literal.
MUT2="$(printf '%s\n' "$BLOCK" | sed \
  -e 's/^  STAMP="\$CPG_STAMP_CYPHER"$/  STAMP="$CPG_STAMP_CYPHER"; CPG_STAMPED_KEYS=""/')"
printf '%s\n' $P8 > "$WORK/marker.keys"; : > "$WORK/parsed_at"
mut2_out="$(MODE=correct bash -c '
    set -euo pipefail
    . "'"$HERE"'/git-provenance.sh"
    HOST=localhost; PORT=6379; GRAPH=cpg_fake; SRC=/src
    PARSED_AT=2026-09-08T07:00:00Z; PROVENANCE=parse-root
    CPG_SOURCE_ORIGIN=falkor-chat/server; CPG_SOURCE_COMMIT=aaaa; CPG_SOURCE_TREE=bbbb; CPG_SOURCE_DIRTY=false
    '"$MUT2"'
  ' 2>&1)"; mut2_rc=$?
echo "P6-6 mutation (allow-list lost after a populated stamp):"
mut2_problems=""
[ "$mut2_rc" = 1 ] || mut2_problems="${mut2_problems}expected rc 1, got $mut2_rc"$'\n'
case "$mut2_out" in
  *"did not populate this shell"*) ;;
  *) mut2_problems="${mut2_problems}no internal-wiring message"$'\n' ;;
esac
case "$mut2_out" in
  *"CPG_STAMP_CYPHER=<set, "*) ;;
  *) mut2_problems="${mut2_problems}the diagnostic does not report the Cypher's shape"$'\n' ;;
esac
if printf '%s\n' "$mut2_out" | grep -q '^MERGE (b:CpgBuildInfo)$'; then
  mut2_problems="${mut2_problems}the diagnostic pasted the whole map literal into the message"$'\n'
fi
if [ -z "$mut2_problems" ]; then
  echo "  PASS  refused, reporting the Cypher's shape rather than its text"
else
  echo "  FAIL"; printf '%s' "$mut2_problems" | sed 's/^/          /'
  printf '%s\n' "$mut2_out" | sed 's/^/        | /'; FAIL=1
fi

[ "$FAIL" = 0 ] && echo "all stamp-wiring cases passed" || echo "STAMP WIRING TEST FAILED"
exit "$FAIL"
