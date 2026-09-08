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
# It deliberately does NOT test the Cypher semantics (whether `SET b = {…}`
# really replaces): the fake models that, it does not prove it. That half is
# graph-dba's executed probes, recorded in git-provenance.sh's docstring. This
# half is the wiring.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
FAIL=0

WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT

# ---- fake redis-cli: models the three replies the stamp block provokes -------
# State lives in $WORK/marker.keys (one key per line) and $WORK/parsed_at.
# MODE=correct      -> `SET b = {…}` replaces the property set (real semantics)
# MODE=merge        -> `SET b = {…}` merges instead (models a regression: `+=`,
#                      a reversion to `b.X = …`, or a FalkorDB semantics change)
cat > "$WORK/redis-cli" <<'FAKE'
#!/usr/bin/env bash
set -uo pipefail
q="${!#}"   # last arg is the query
case "$q" in
  *"MERGE (b:CpgBuildInfo)"*)
    # keys of the map = lines of the form `NAME: value` whose value is not NULL
    new="$(printf '%s\n' "$q" | sed -nE 's/^[[:space:]]*([A-Z][A-Z0-9_]*): (.*)$/\1 \2/p' \
           | grep -v ' NULL,\?$' | sed 's/ .*//')"
    if [ "${MODE:-correct}" = merge ]; then
      cat "$WORK/marker.keys" > "$WORK/.tmp"; printf '%s\n' $new >> "$WORK/.tmp"
      sort -u "$WORK/.tmp" > "$WORK/marker.keys"
    else
      printf '%s\n' $new | sort -u > "$WORK/marker.keys"
    fi
    printf '%s\n' "$q" | sed -nE 's/^[[:space:]]*PARSED_AT: "([^"]*)".*/\1/p' > "$WORK/parsed_at"
    echo "Properties set: 8"; exit 0 ;;
  *"RETURN b.PARSED_AT"*)
    echo "b.PARSED_AT"; cat "$WORK/parsed_at"; exit 0 ;;
  *"UNWIND keys(b)"*)
    allow="$(printf '%s\n' "$q" | sed -nE "s/.*NOT k IN \[(.*)\].*/\1/p" | tr -d "'" | tr ',' ' ')"
    echo "stray"
    while read -r k; do
      [ -n "$k" ] || continue
      hit=0; for a in $allow; do [ "$a" = "$k" ] && hit=1; done
      [ "$hit" = 0 ] && echo "STRAY_KEY=$k"
    done < "$WORK/marker.keys"
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
  echo "FAIL: could not extract the stamp block from pipeline.sh (anchors moved)"; exit 1
fi
case "$BLOCK" in
  *'cpg_provenance_stray_query'*) ;;
  *) echo "FAIL: extracted block does not contain the stray assertion"; exit 1 ;;
esac

run_case() {   # run_case <name> <mode> <initial marker keys> <provenance> <expect: PASS|stray names>
  local name="$1" mode="$2" initial="$3" prov="$4" expect="$5" out rc
  printf '%s\n' $initial | sort -u > "$WORK/marker.keys"; : > "$WORK/parsed_at"
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
      # the wiring assertion this whole file exists for:
      echo "ALLOWLIST=[${CPG_STAMPED_KEYS:-<UNSET>}]"
    ' 2>&1
  )"; rc=$?
  local got
  if [ "$rc" -eq 0 ]; then got=PASS; else
    got="$(printf '%s\n' "$out" | sed -n 's/^pipeline:   \([A-Z][A-Z0-9_]*\)$/\1/p' | tr '\n' ' ' | sed 's/ $//')"
    [ -n "$got" ] || got="FAILED(rc=$rc)"
  fi
  if [ "$got" = "$expect" ]; then
    echo "  PASS  $name  -> $got"
  else
    echo "  FAIL  $name  -> expected [$expect], got [$got]"; printf '%s\n' "$out" | sed 's/^/        | /'; FAIL=1
  fi
  printf '%s\n' "$out" | grep -o 'ALLOWLIST=\[[^]]*\]' | sed 's/^/        /'
}

P8="BUILT_AT PARSED_AT SOURCE_PATH PROVENANCE SOURCE_ORIGIN SOURCE_COMMIT SOURCE_TREE SOURCE_DIRTY"

echo "stamp wiring:"
run_case "correct replace over a hand-authored marker" correct "$P8 MARKER_ORIGIN NOTE MARKER_EVIDENCE" parse-root PASS
run_case "correct replace, provenance=none"            correct "$P8 MARKER_ORIGIN NOTE"                 none      PASS
run_case "regression: merge semantics, foreign key"    merge   "$P8 MARKER_EVIDENCE"                    parse-root MARKER_EVIDENCE
run_case "regression: merge, pipeline-clean marker"    merge   "$P8"                                    parse-root PASS
run_case "subsumption: stale SOURCE_* under none"      merge   "$P8"                                    none      "SOURCE_COMMIT SOURCE_DIRTY SOURCE_ORIGIN SOURCE_TREE"

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

[ "$FAIL" = 0 ] && echo "all stamp-wiring cases passed" || echo "STAMP WIRING TEST FAILED"
exit "$FAIL"
