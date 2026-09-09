#!/usr/bin/env bash
# pipeline.sh — run the full Joern -> FalkorDB pipeline end to end, for ANY source.
#   build-cpg (parse) -> export-cpg (neo4jcsv) -> cpg-to-falkordb (transform [+ load])
#
# Generic: the caller says WHAT to build; there are no baked-in project/app names.
#
# Usage: pipeline.sh <source> [--graph NAME] [--workdir DIR] [--language LANG]
#                    [--repr R] [--reset] [--load] [--host H] [--port P]
#                    [--verify-prefix PREFIX ...] [--source-origin PATH]
#   <source>     source dir/file to analyze (required)
#   --graph      FalkorDB graph key             (default cpg_<basename>)
#   --workdir    scratch dir for cpg.bin/export (default ./joern-work)
#   --language   joern frontend token           (else joern-parse auto-detects;
#                                                 for Python use `pythonsrc`, NOT
#                                                 `python` — see SKILL.md gotchas)
#   --repr       joern-export repr              (default cpg)
#   --reset      GRAPH.DELETE the target graph before loading (destructive,
#                guard-gated) so the load is clean; no-op if it doesn't exist
#   --load       ingest into FalkorDB (else stops at the .cypher artifact)
#   --host/--port  FalkorDB endpoint            (default localhost:6379)
#   --verify-prefix PREFIX  after --load, assert count(METHOD nodes whose
#                FILENAME STARTS WITH PREFIX) > 0; repeatable. FILENAME is
#                relative to <source> (the parse root), NOT the repo root — a
#                wrong root produces a healthy-looking graph that answers
#                prefix-filtered queries (e.g. cpg-analysis's test-gap recipe,
#                which filters on a `tests/`-style prefix) with silent zero
#                rows. Pass e.g. `--verify-prefix tests/` whenever a downstream
#                query will filter FILENAME by prefix; no prefix is assumed by
#                default since the pipeline is generic. A failing prefix exits
#                the pipeline non-zero — see SKILL.md Gotchas for the fix
#                (rebuild from a parse root that includes the expected prefix).
#   --source-origin PATH  the real, git-TRACKED directory <source> was staged
#                from, when <source> is a pruned scratch copy (the documented
#                way to scope a parse — see SKILL.md Gotchas). Provenance is
#                then derived from PATH instead of from the parse root, which
#                has no git identity of its own. Without it, a staged copy
#                yields PROVENANCE=none and no SOURCE_COMMIT — deliberately,
#                since a scratch copy sitting inside a repo would otherwise
#                inherit that repo's HEAD, which describes a different tree.
#
# Provenance: SOURCE_COMMIT/SOURCE_TREE/SOURCE_DIRTY are captured BEFORE the
# parse and scoped to the source (see scripts/git-provenance.sh for why: a
# stamp-time `git -C "$SRC"` races HEAD across a multi-hour build and reports
# repo-wide dirt). Capture happens when the pipeline starts, so stage the copy
# immediately before invoking it.
#
# Robustness: after transform the pipeline ASSERTS the CPG produced nodes and
# fails loudly otherwise — joern-parse exits 0 even when a frontend fails, so a
# silent empty build would otherwise pass. Loading uses cpg-to-falkordb's
# single-socket loader (no per-statement redis-cli). After --load it verifies
# node/edge counts in the graph, and — if --verify-prefix was given — that the
# expected FILENAME prefix(es) actually resolve to a nonzero METHOD count.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC="${1:?usage: pipeline.sh <source> [--graph NAME] [--workdir DIR] [--language LANG] [--repr R] [--reset] [--load] [--host H] [--port P] [--verify-prefix PREFIX ...] [--source-origin PATH]}"
shift
GRAPH=""; WORKDIR="./joern-work"; LANGUAGE=""; REPR="cpg"; RESET=""; LOAD=""
HOST="${FALKORDB_HOST:-localhost}"; PORT="${FALKORDB_PORT:-6379}"
SOURCE_ORIGIN_ARG=""; SOURCE_ORIGIN_SET=""
VERIFY_PREFIXES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --graph) GRAPH="$2"; shift 2 ;;
    # SET is tracked separately so `--source-origin ""` fails fast rather than
    # silently falling through to the parse-root branch.
    --source-origin) SOURCE_ORIGIN_ARG="$2"; SOURCE_ORIGIN_SET=1; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --language) LANGUAGE="$2"; shift 2 ;;
    --repr) REPR="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --reset) RESET=1; shift ;;
    --load) LOAD="--load"; shift ;;
    --verify-prefix) VERIFY_PREFIXES+=("$2"); shift 2 ;;
    *) echo "pipeline: unknown arg '$1'" >&2; exit 2 ;;
  esac
done
[ -n "$GRAPH" ] || GRAPH="cpg_$(basename "$SRC" | tr -cs 'A-Za-z0-9_' '_')"

CPG="$WORKDIR/cpg.bin"; EXPORT="$WORKDIR/export"; CYPHER="$WORKDIR/load.cypher"

# NOTE: `mkdir -p "$WORKDIR"` deliberately happens AFTER the capture block below,
# not here. SOURCE_DIRTY counts untracked files under the source, so creating the
# pipeline's own scratch directory first makes the pipeline dirty the very tree it
# is about to measure — `pipeline.sh . --load` with the default `./joern-work`
# could then never report a clean source, permanently disabling the consumer's
# tree-identity check for that build shape.

# ---- provenance: captured HERE, before the parse, scoped to the source ----
# Not at stamp time. A real build runs for hours; deriving the commit after the
# load records whatever HEAD happens to be then — a tree that was never parsed
# — and an unscoped `git status` reports dirt from anywhere in the repo. Both
# were observed on the 2026-09-07 cpg_falkorchat build; see git-provenance.sh.
# shellcheck source=./git-provenance.sh
. "$HERE/git-provenance.sh"
PARSED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
PROVENANCE=none
if [ -n "$SOURCE_ORIGIN_SET" ]; then
  if cpg_provenance_capture "$SOURCE_ORIGIN_ARG"; then
    PROVENANCE=source-origin
  else
    echo "pipeline: FAILED — --source-origin '$SOURCE_ORIGIN_ARG' has no usable git provenance" >&2
    echo "pipeline: (missing path, not inside a git work tree, or inside one but untracked)." >&2
    echo "pipeline: Failing now rather than after a multi-hour build that would stamp nothing." >&2
    exit 2
  fi
elif cpg_provenance_capture "$SRC"; then
  PROVENANCE=parse-root
fi
if [ "$PROVENANCE" = none ]; then
  echo "pipeline: WARNING — no git provenance for the parse root '$SRC'." >&2
  echo "pipeline: SOURCE_COMMIT/SOURCE_TREE/SOURCE_DIRTY will NOT be stamped (PROVENANCE=none)," >&2
  echo "pipeline: so consumers get raw build age as their only freshness signal." >&2
  echo "pipeline: If '$SRC' is a staged copy of a tracked directory, stop now and re-run with" >&2
  echo "pipeline:   --source-origin <the tracked directory it was staged from>" >&2
else
  # Full OIDs are stamped (see git-provenance.sh); these log lines abbreviate for
  # readability only.
  if [ -n "$CPG_SOURCE_TREE" ]; then TREE_SHOWN="${CPG_SOURCE_TREE:0:12}"; else TREE_SHOWN="(none — source not committed)"; fi
  echo "pipeline: provenance ($PROVENANCE) — origin=$CPG_SOURCE_ORIGIN commit=${CPG_SOURCE_COMMIT:0:12}" >&2
  echo "pipeline: tree=$TREE_SHOWN dirty=$CPG_SOURCE_DIRTY parsedAt=$PARSED_AT" >&2
  if [ "$CPG_SOURCE_DIRTY" = true ]; then
    echo "pipeline: NOTE — '$CPG_SOURCE_ORIGIN' has uncommitted or untracked changes; this graph" >&2
    echo "pipeline: will be stamped SOURCE_DIRTY=true, i.e. it matches no commit exactly." >&2
  fi
fi

# Only now is it safe to create the scratch dir (see the NOTE above). A workdir
# left inside the source by an EARLIER run is still counted, and reordering
# cannot undo that — so say so, since such a workdir also feeds itself to the
# parse on the next run.
mkdir -p "$WORKDIR"
if [ "$CPG_SOURCE_DIRTY" = true ] && command -v realpath >/dev/null 2>&1; then
  _wd="$(realpath -m "$WORKDIR" 2>/dev/null || true)"
  _sr="$(realpath -m "$SRC" 2>/dev/null || true)"
  case "$_wd/" in
    "$_sr"/*) echo "pipeline: NOTE — the workdir '$WORKDIR' sits INSIDE the parse root. If it survived" >&2
              echo "pipeline: an earlier run, it is what made SOURCE_DIRTY true (and it will be parsed" >&2
              echo "pipeline: too). Point --workdir outside the source." >&2 ;;
  esac
  unset _wd _sr
fi

echo "== [1/3] build CPG ==" >&2
JOERN_LANGUAGE="$LANGUAGE" "$HERE/build-cpg.sh" "$SRC" "$CPG"

echo "== [2/3] export neo4jcsv ==" >&2
"$HERE/export-cpg.sh" "$CPG" "$EXPORT" "$REPR" neo4jcsv

# joern-parse exits 0 even when the frontend fails, yielding an empty CPG. A
# successful export of a real build has node CSVs — assert that before loading.
if ! find "$EXPORT" -name 'nodes_*_data.csv' -size +0c -print -quit | grep -q .; then
  echo "pipeline: FAILED — export produced no node data under $EXPORT." >&2
  echo "pipeline: the CPG is empty; the parse frontend likely failed (check the build log;" >&2
  echo "pipeline: for Python pass --language pythonsrc, not python)." >&2
  exit 1
fi

# Optional destructive reset so --load lands in a clean graph.
if [ -n "$RESET" ] && [ -n "$LOAD" ]; then
  if redis-cli -h "$HOST" -p "$PORT" GRAPH.LIST | grep -qx "$GRAPH"; then
    echo "== reset graph '$GRAPH' (GRAPH.DELETE — destructive, guard-gated) ==" >&2
    redis-cli -h "$HOST" -p "$PORT" GRAPH.DELETE "$GRAPH"
  fi
fi

echo "== [3/3] transform -> FalkorDB Cypher ($GRAPH) ==" >&2
python3 "$HERE/cpg-to-falkordb.py" "$EXPORT" -o "$CYPHER" --graph "$GRAPH" --host "$HOST" --port "$PORT" $LOAD

echo "pipeline: done. Cypher artifact: $CYPHER" >&2
if [ -n "$LOAD" ]; then
  # Take the standalone integer result row — NOT a tail-grep of all digits, which
  # would grab digits from the "Query internal execution time: 0.08 ms" stat line.
  count() { redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$GRAPH" "$1" --no-raw \
              | awk '/^[0-9]+$/{last=$0} END{print last}'; }
  N="$(count 'MATCH (n) RETURN count(n)')"
  E="$(count 'MATCH ()-[r]->() RETURN count(r)')"
  echo "pipeline: loaded '$GRAPH' on $HOST:$PORT — nodes=$N edges=$E" >&2

  # FILENAME is relative to the parse root ($SRC), not the repo root (see
  # SKILL.md Gotchas). A wrong root still yields healthy node/edge counts
  # above, so that check alone cannot catch it — verify the prefix(es) the
  # caller expects downstream queries to filter on actually resolve.
  if [ "${#VERIFY_PREFIXES[@]}" -gt 0 ]; then
    VERIFY_FAILED=0
    for PREFIX in "${VERIFY_PREFIXES[@]}"; do
      PCOUNT="$(count "MATCH (m:METHOD) WHERE m.FILENAME STARTS WITH \"$PREFIX\" RETURN count(m)")"
      if [ -z "$PCOUNT" ] || [ "$PCOUNT" = "0" ]; then
        echo "pipeline: VERIFY FAILED — 0 METHOD nodes with FILENAME STARTS WITH '$PREFIX' in '$GRAPH'." >&2
        VERIFY_FAILED=1
      else
        echo "pipeline: verify OK — $PCOUNT METHOD nodes with FILENAME STARTS WITH '$PREFIX'." >&2
      fi
    done
    if [ "$VERIFY_FAILED" -eq 1 ]; then
      echo "pipeline: the graph looks healthy by node/edge count but the expected FILENAME" >&2
      echo "pipeline: prefix(es) above resolve to nothing — FILENAME is relative to the parse" >&2
      echo "pipeline: root ('$SRC'), not the repo root. Rebuild from a parse root that includes" >&2
      echo "pipeline: the expected prefix (see SKILL.md Gotchas: 'FILENAME is relative to the" >&2
      echo "pipeline: parse root')." >&2
      exit 1
    fi
  fi

  # Freshness marker (cpg-agent-adoption M4, FR-5/FR-6) — written only after the
  # load and any --verify-prefix checks have fully succeeded, so a stamped graph
  # means "built successfully at this time," never "an attempt was made." One
  # singleton node per graph; MERGE (no property in the pattern) keeps it that
  # way across both --reset (fresh graph) and --append (existing graph) loads —
  # freshness tracks "when was this graph's content last touched," not "when was
  # it first created."
  #
  # BUILT_AT stays "when this graph's content was last touched"; PARSED_AT is
  # the separate, earlier fact a consumer actually needs to ask "has the source
  # moved since?" — on a 3h build the two differ by 3h, and anchoring a
  # `git log --since` on BUILT_AT silently excludes every commit made *during*
  # the build, which is exactly when a concurrent session commits.
  #
  # The provenance fields were captured before the parse (see above) and are
  # written here verbatim — never re-derived. The stamp is a MAP ASSIGNMENT
  # (`SET b = {…}`), which REPLACES the marker's whole property set, so an
  # --append re-stamp cannot leave a previous build's SOURCE_COMMIT — or a
  # hand-authored NOTE — standing over new content. Nothing survives a stamp
  # except what that stamp wrote. See git-provenance.sh for the executed
  # evidence.
  BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  # CALLED AS A STATEMENT, NOT IN `$(…)`. cpg_provenance_stamp renders the
  # Cypher into CPG_STAMP_CYPHER *and* records the properties it wrote into
  # CPG_STAMPED_KEYS, which the stray assertion below needs; a command
  # substitution runs it in a subshell and throws the second one away. This line
  # was `STAMP="$(cpg_provenance_stamp …)"` for two commits and every --load
  # build would have failed on an empty allow-list, after the parse, with the
  # marker already replaced.
  cpg_provenance_stamp "$BUILT_AT" "$PARSED_AT" "$SRC" "$PROVENANCE"
  STAMP="$CPG_STAMP_CYPHER"
  if [ -z "${STAMP:-}" ] || [ -z "${CPG_STAMPED_KEYS:-}" ]; then
    # Report the Cypher's SHAPE, never its text. This line used to read
    # `${CPG_STAMP_CYPHER:+<set>}${CPG_STAMP_CYPHER:-<empty>}`, which for a set
    # variable prints `<set>` and then the entire multi-line map literal —
    # burying the two words that matter under the thing the operator already
    # has. The branch had never been executed when it was written; it is now
    # covered by test-stamp-wiring.sh's P6-6 mutation case.
    if [ -n "${CPG_STAMP_CYPHER:-}" ]; then _cy="<set, ${#CPG_STAMP_CYPHER} chars>"; else _cy="<empty>"; fi
    echo "pipeline: FAILED — internal: cpg_provenance_stamp did not populate this shell" >&2
    echo "pipeline:   CPG_STAMP_CYPHER=$_cy CPG_STAMPED_KEYS=${CPG_STAMPED_KEYS:-<empty>}" >&2
    echo "pipeline: this is a bug in the pipeline, not a finding about '$GRAPH' — the graph has" >&2
    echo "pipeline: NOT been stamped and is otherwise untouched. Check that the call above is a" >&2
    echo "pipeline: statement and not a \$(…) substitution (skills/joern-cpg/scripts/git-provenance.sh)." >&2
    exit 1
  fi

  # `redis-cli` EXITS 0 ON AN ERROR REPLY and prints the error to stdout, so the
  # old `… >/dev/null` discarded every failure and `set -e` saw success. A
  # NON-ZERO exit from it means one thing only — the server was unreachable, and
  # the message went to stderr instead (measured 2026-09-09 against port 6399:
  # `Could not connect to Redis at localhost:6399: Connection refused`, exit 1).
  # So `$?` is reliable for "unreachable" and blind to "rejected".
  #
  # SUCCESS IS RECOGNISED POSITIVELY, NOT BY ENUMERATING ERROR SHAPES. From
  # 9124a1f until 2026-09-09 this helper classified failure with a prefix list
  # (`errMsg:*|ERR *|WRONGTYPE*|*read-only*`). That is a blacklist of error
  # shapes, and FalkorDB emits several with no prefix at all, so `rq` returned 0
  # on them: a failed query read as SUCCESS at every call site passing no
  # [must-contain]. Re-measured 2026-09-09 by extracting this function verbatim
  # and pointing it at a live graph (throwaway key, since deleted):
  #
  #   RETURN (((                       errMsg: Invalid input …              caught
  #   CREATE (…) on GRAPH.RO_QUERY     graph.RO_QUERY is to be executed …   caught
  #   RETURN nosuchfunc(1)             Unknown function 'nosuchfunc'        MISSED
  #   RETURN keys(n.k), k an integer   Type mismatch: expected Map, …       MISSED
  #   UNWIND [1,0] AS x RETURN 1/x     Division by zero                     MISSED
  #   a long scan with TIMEOUT 1       Query timed out                      MISSED
  #
  # Every one of those is a single bare line at redis-cli exit 0. What they have
  # in common is not a prefix — it is the ABSENCE of the statistics trailer that
  # a query the server ran to completion always ends with:
  #
  #   count(n)                             <- the projection's column header
  #   1                                    <- rows (there may be none)
  #   Cached execution: 0
  #   Query internal execution time: 0.14 milliseconds   <- ALWAYS THE LAST LINE
  #
  # Hence the gate: THE REPLY'S LAST LINE MUST BEGIN `Query internal execution
  # time:`. Positive, so an error shape nobody has met yet fails CLOSED instead
  # of passing. Anchoring on the LAST line rather than searching the whole reply
  # is worth it for two measured reasons: `Query timed out` shares the trailer's
  # first word, and a row of returned data could carry the literal.
  #
  # WHAT THE GATE COVERS, EXACTLY: any GRAPH.QUERY / GRAPH.RO_QUERY reply the
  # server did not run to completion — parse error, runtime error, read-only
  # refusal, timeout, empty reply — plus, via the exit status above, an
  # unreachable server. A query ABORTED MID-STREAM by a runtime error is covered
  # for a specific reason, not by luck: FalkorDB discards the rows already
  # produced and answers with one bare line, so there is no partial reply for the
  # gate to mistake for a whole one (`UNWIND [1,0] AS x RETURN 1/x` returns
  # `Division by zero` and nothing else — no header, no rows, no trailer).
  #
  # WHAT IT DOES NOT COVER — three things, and the third is the one that will
  # bite a future call site:
  #   1. A query that ran to completion and answered WRONG.
  #   2. A query that answered about the wrong thing.
  #   3. A COMPLETE-LOOKING REPLY THAT IS MISSING ROWS. `RESULTSET_SIZE` (10000
  #      on this instance, `GRAPH.CONFIG GET RESULTSET_SIZE`) caps a result set
  #      SILENTLY and still emits a normal trailer. Measured 2026-09-09:
  #      `UNWIND range(1,200000) AS x RETURN x` comes back rc 0, 10003 lines,
  #      last data row `10000`, last line the trailer — the gate PASSES a reply
  #      that lost 190,000 rows, and nothing in the reply says so.
  # So the trailer proves the server RAN THE QUERY TO COMPLETION. It does not
  # prove the reply carries every row the query matched, and it proves nothing
  # about correctness. A caller that needs COMPLETENESS must check the row count
  # against an expectation of its own; the three call sites below return 1, 1 and
  # <= 8 rows, so none of them is near the cap. Asserting anything about a
  # reply's contents remains the caller's job — that is what [must-contain] and
  # the PARSED_AT read-back below are for.
  #
  # WRITES CARRY THE TRAILER TOO, which the previous version of this comment
  # declined to assume and therefore left the stamp write ungated. Measured
  # 2026-09-09 on this exact shape — `MERGE (b:CpgBuildInfo) SET b = {…}
  # RETURN 1` — on the run that created the node and on the re-run where the
  # MERGE matched: both end with the trailer. That is why the gate is
  # unconditional instead of per call site.
  #
  # <cypher> [redis-command] [must-contain] — the command defaults to
  # GRAPH.QUERY (needed for the stamp, which writes). The two READS below pass
  # GRAPH.RO_QUERY instead: a GRAPH.QUERY against a graph that does not exist
  # MATERIALIZES it, and while this graph certainly exists by now, the failure
  # messages tell the operator to use GRAPH.RO_QUERY for the identical read — so
  # the pipeline should not be doing the thing it warns against.
  #
  # THE TRAILER IS SPECIFIC TO THOSE TWO COMMANDS, and that is a PRECONDITION on
  # calling rq, enforced OUTSIDE it. GRAPH.DELETE, for one, answers the bare
  # status `OK` (observed 2026-09-09) and nothing else, so the gate would read a
  # successful delete as a failure. rq does NOT check its own second argument,
  # and the reason is worth the four lines, because the obvious guard was written
  # here and then removed:
  #   * A runtime `return 2` is UNREADABLE. Every call site is
  #     `if ! VAR="$(rq …)"`, which collapses every non-zero to "false" — 1 and 2
  #     are the same branch (verified 2026-09-09) — so the distinction reached
  #     nobody, and at the stamp site the caller went on to print "FalkorDB
  #     rejected the freshness stamp … <no reply>" about a call that never left
  #     this shell. That is the same overclaim the read-back branch below was
  #     just fixed for.
  #   * A runtime `exit 2` CANNOT ESCAPE: rq runs inside `$(…)`, so the exit
  #     kills the subshell and the script continues (verified 2026-09-09; the
  #     same command-substitution trap that ate cpg_provenance_stamp's
  #     allow-list two commits ago).
  #   * Deleting the guard is SAFE, because the untrapped behaviour already fails
  #     CLOSED, twice over. A non-query reply carries no trailer, so rq returns 1
  #     — and through rq a GRAPH.DELETE cannot even reach its bare `OK`, because
  #     rq always appends the cypher as a third argument: the reply is
  #     `ERR wrong number of arguments for 'graph.DELETE' command`, rc 1 (both
  #     observed 2026-09-09). The cost of a misuse is a false failure, never a
  #     false pass — a different risk class from the silent pass K-009 was about.
  # What enforces the precondition instead is a STATIC check over this file, in
  # test-stamp-wiring.sh: every `rq` call site's command argument must be absent,
  # GRAPH.QUERY, or GRAPH.RO_QUERY, in any case, whether the site is a `$(…)`
  # substitution or a bare statement. It reddens on all four of those shapes —
  # which the runtime guard could not do, since removing THAT left the suite
  # byte-identical. Its one stated blind spot is a command reaching rq through a
  # VARIABLE; see the block's own header in test-stamp-wiring.sh.
  #
  # [must-contain] stays per call site, because it is a semantic assertion about
  # one reply's contents and not a liveness test. The stray check below still
  # passes the trailer explicitly: redundant with the gate by design, kept so
  # that the one assertion whose PASS is "no rows" carries its own guarantee at
  # the point where a future edit would read it.
  rq() {
    local out
    out="$(redis-cli -h "$HOST" -p "$PORT" "${2:-GRAPH.QUERY}" "$GRAPH" "$1" 2>&1)" || { printf '%s' "$out"; return 1; }
    printf '%s' "$out"
    # The LAST line, not a substring anywhere in the reply. See above.
    case "${out##*$'\n'}" in
      "Query internal execution time:"*) ;;
      *) return 1 ;;
    esac
    if [ -n "${3:-}" ]; then
      case "$out" in
        *"$3"*) ;;
        *) return 1 ;;
      esac
    fi
    return 0
  }

  # Every failure below this point leaves a graph that is fully loaded and needs
  # NO re-parse. The five branches split in two, and the split is the whole
  # point of printing anything:
  #   * THE STAMP DID NOT LAND (rejected, or read back absent) — re-sending the
  #     Cypher IS the fix.                                     -> replay_stamp
  #   * THE STAMP DID LAND and a LATER assertion failed — re-sending it changes
  #     nothing the message is complaining about, and saying otherwise
  #     contradicts the branch's own first line.               -> show_stamp
  # Both print the same delimited block, because the reason for printing at all
  # is shared: the stamp is a multi-line map literal with escaped quotes,
  # strictly harder to reconstruct than the flat SET clause the old "re-stamp by
  # hand" advice was written for. Only the advice around it differs.
  #
  # Defined here, before every caller — under `set -e` an undefined function
  # aborts with 127 and swallows the message it was supposed to print. That is
  # not hypothetical: deleting this definition was invisible to the wiring test
  # until the oracle started asserting the exit code and the block's presence
  # (test-stamp-wiring.sh, expect_rc + the `--- begin stamp ---` assertion).
  print_stamp() {   # print_stamp <lead-in line>… — renders $STAMP verbatim
    local line
    for line in "$@"; do echo "pipeline: $line" >&2; done
    echo "pipeline: --- begin stamp ---" >&2
    printf '%s\n' "$STAMP" >&2
    echo "pipeline: --- end stamp ---" >&2
  }
  replay_stamp() {  # the stamp did NOT land: this is the fix
    print_stamp \
      "the load succeeded and does NOT need repeating — only the stamp does." \
      "copy the Cypher below verbatim; do not retype it."
    echo "pipeline: save it to a file and send it with:" >&2
    echo "pipeline:   redis-cli -h $HOST -p $PORT GRAPH.QUERY $GRAPH \"\$(cat <file>)\"" >&2
  }
  show_stamp() {    # the stamp DID land: this is evidence, not a fix
    print_stamp \
      "the load AND the stamp both succeeded — the stamp is NOT what needs" \
      "repeating. This build's rendered Cypher is below so you can compare it" \
      "against the marker as it now stands; re-sending it verbatim would" \
      "reproduce exactly the state being complained about."
  }

  if ! STAMP_OUT="$(rq "$STAMP")"; then
    echo "pipeline: FAILED — FalkorDB rejected the freshness stamp for '$GRAPH':" >&2
    echo "pipeline:   ${STAMP_OUT:-<no reply>}" >&2
    echo "pipeline: the load itself succeeded; only the provenance marker is missing. Do NOT" >&2
    echo "pipeline: treat this graph as stamped — on an --append build the PREVIOUS build's" >&2
    echo "pipeline: marker is still standing over the new content." >&2
    replay_stamp
    exit 1
  fi

  # Read-back assertion. The stamp is the one write whose silent failure is
  # invisible, and the guarantee two other documents now rest on — "every field
  # rewritten on every stamp, an absent one removed rather than left stale" —
  # holds only if the write actually landed. PARSED_AT is the discriminator: it
  # is unique to this run, so a surviving older marker cannot match it.
  #
  # THE TWO FAILURES ARE SEPARATED, and until 2026-09-09 they were not: this
  # line read `… GRAPH.RO_QUERY || true)` and judged only the reply's TEXT, so a
  # read-back that never ran — the query errored, the server went away — fell
  # into the branch below and told the operator "the freshness stamp did not
  # land", a claim about the GRAPH that the run had no evidence for, followed by
  # advice to re-send a stamp that may well be sitting there correctly. `|| true`
  # was defensible while rq's status could not see a bare runtime error; now that
  # it can, discarding it is throwing away the one signal that distinguishes
  # "checked, and it is absent" from "could not check".
  if ! STAMP_BACK="$(rq 'MATCH (b:CpgBuildInfo) RETURN b.PARSED_AT' GRAPH.RO_QUERY)"; then
    echo "pipeline: FAILED — could not read the freshness stamp back from '$GRAPH':" >&2
    echo "pipeline:   ${STAMP_BACK:-<no reply>}" >&2
    echo "pipeline: the read-back query did not run to completion, so this says NOTHING about" >&2
    echo "pipeline: whether the stamp landed — do not infer either way. Check by hand:" >&2
    echo "pipeline:   redis-cli -h $HOST -p $PORT GRAPH.RO_QUERY $GRAPH \"MATCH (b:CpgBuildInfo) RETURN b.PARSED_AT\"" >&2
    print_stamp \
      "whether the stamp landed is UNKNOWN — the CHECK failed, not necessarily the" \
      "stamp. This build's rendered Cypher is below so you can compare it against the" \
      "marker as it actually stands; re-send it only once you have established that" \
      "the marker is missing or describes a different build."
    exit 1
  fi
  case "$STAMP_BACK" in
    *"$PARSED_AT"*) ;;
    *) echo "pipeline: FAILED — the freshness stamp did not land in '$GRAPH'." >&2
       echo "pipeline: read back: ${STAMP_BACK:-<no reply>}" >&2
       echo "pipeline: expected a CpgBuildInfo marker with PARSED_AT=$PARSED_AT. The marker now" >&2
       echo "pipeline: in the graph, if any, describes a DIFFERENT build — do not trust it." >&2
       replay_stamp
       exit 1 ;;
  esac

  # Stray-property assertion. The read-back above proves THIS build's marker
  # landed; it cannot see a property the stamp failed to CLEAR. ORDERING IS
  # LOAD-BEARING, in that one direction only: a graph with NO marker returns
  # zero rows here too, i.e. a pass, so this check has to follow the read-back
  # that already excludes that case (verified 2026-09-08 by running this block
  # standalone against a graph with no CpgBuildInfo node: "PASSED").
  #
  # WHAT THIS CATCHES IS NARROW: THE STAMP FAILED TO ERASE A FOREIGN KEY THAT
  # WAS ALREADY ON THE MARKER. It does not detect "the replace semantics
  # changed" — that framing was here and it overclaimed. A `+=`, a reversion to
  # `b.X = …`, or a FalkorDB treating `=` as a merge all leave exactly the eight
  # stamped keys on a graph whose previous marker was pipeline-clean, so none of
  # them fires there; and an edit that drops a property out of the map cannot
  # fire this at all, since _cpg_prop drops it from the allow-list in the same
  # call. Two graphs carry hand-authored markers today (cpg_falkorchat,
  # cpg_deprecated_salesperson) and on each one's next rebuild this fires on
  # exactly the defect the arc was about; after that it is a cheap standing
  # guard against a foreign key reintroduced by any writer other than the stamp.
  # DO NOT REMOVE IT — on that reason, which is checkable, rather than the
  # bigger one, which was not.
  #
  # The allow-list is generated from the stamp's own map (CPG_STAMPED_KEYS — see
  # _cpg_prop), so it cannot drift from what was written. It is also asserted
  # non-empty at the call site above, because an empty one makes every property
  # a stray — that was the shipped state for two commits, not a hypothetical.
  #
  # This check is NEGATIVE — "no rows" is the pass — so unlike the read-back
  # above it does not fail closed for free: an error reply contains no
  # STRAY_KEY= and would sail through. The status check alone did NOT close
  # that, because rq's blacklist cannot see a bare runtime error (see rq).
  # What closes it is the third argument below: the reply must carry the
  # statistics trailer a completed query always ends with, so "no rows" can
  # only be reached from a query the server actually ran.
  if ! STRAY_Q="$(cpg_provenance_stray_query)"; then
    echo "pipeline: FAILED — internal: could not build the marker property check for '$GRAPH'." >&2
    echo "pipeline: this is a bug in the pipeline, not a finding about the graph. The stamp DID" >&2
    echo "pipeline: land (it was read back above); only this last assertion could not run." >&2
    show_stamp
    exit 1
  fi
  if ! STRAY_BACK="$(rq "$STRAY_Q" GRAPH.RO_QUERY 'Query internal execution time:')"; then
    echo "pipeline: FAILED — could not verify the marker's property list in '$GRAPH':" >&2
    echo "pipeline:   ${STRAY_BACK:-<no reply>}" >&2
    echo "pipeline: the load and the stamp both succeeded, but the marker is unverified — it may" >&2
    echo "pipeline: carry properties from an earlier build. Do not trust it until checked by hand:" >&2
    echo "pipeline:   redis-cli -h $HOST -p $PORT GRAPH.RO_QUERY $GRAPH \"MATCH (b:CpgBuildInfo) RETURN keys(b)\"" >&2
    show_stamp
    exit 1
  fi
  case "$STRAY_BACK" in
    *STRAY_KEY=*)
      echo "pipeline: FAILED — the marker in '$GRAPH' carries properties this build did not write:" >&2
      printf '%s\n' "$STRAY_BACK" | sed -n 's/^STRAY_KEY=/pipeline:   /p' >&2
      echo "pipeline: The stamp is a map assignment (SET b = {…}), which replaces the marker's" >&2
      echo "pipeline: ENTIRE property set — so a key above that the stamp did not write means the" >&2
      echo "pipeline: replace did not erase it. Do not just clear those keys and re-run: check" >&2
      echo "pipeline: first that cpg_provenance_stamp still emits 'SET b = {' and not 'b.X =' or" >&2
      echo "pipeline: '+=' (skills/joern-cpg/scripts/git-provenance.sh), and that this FalkorDB" >&2
      echo "pipeline: still treats '=' as a replace rather than a merge. If either has changed," >&2
      echo "pipeline: every guarantee resting on it is suspect too" >&2
      echo "pipeline: (skills/cpg-analysis/references/freshness.md)." >&2
      echo "pipeline: To inspect the marker as it stands:" >&2
      echo "pipeline:   redis-cli -h $HOST -p $PORT GRAPH.RO_QUERY $GRAPH \"MATCH (b:CpgBuildInfo) RETURN keys(b)\"" >&2
      show_stamp
      exit 1 ;;
  esac

  echo "pipeline: stamped '$GRAPH' — BUILT_AT=$BUILT_AT PARSED_AT=$PARSED_AT SOURCE_PATH=$SRC" >&2
  echo "pipeline: provenance=$PROVENANCE origin=${CPG_SOURCE_ORIGIN:-—} commit=${CPG_SOURCE_COMMIT:0:12}${CPG_SOURCE_COMMIT:+…} tree=${CPG_SOURCE_TREE:0:12}${CPG_SOURCE_TREE:+…} dirty=${CPG_SOURCE_DIRTY:-—} (full OIDs are in the marker)" >&2
  echo "pipeline: stamp verified by read-back." >&2
fi