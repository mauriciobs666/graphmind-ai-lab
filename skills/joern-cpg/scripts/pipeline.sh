#!/usr/bin/env bash
# pipeline.sh — run the full Joern -> FalkorDB pipeline end to end, for ANY source.
#   build-cpg (parse) -> export-cpg (neo4jcsv) -> cpg-to-falkordb (transform [+ load])
#
# Generic: the caller says WHAT to build; there are no baked-in project/app names.
#
# Usage: pipeline.sh <source> [--graph NAME] [--workdir DIR] [--language LANG]
#                    [--repr R] [--reset] [--load] [--host H] [--port P]
#                    [--verify-prefix PREFIX ...]
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
#
# Robustness: after transform the pipeline ASSERTS the CPG produced nodes and
# fails loudly otherwise — joern-parse exits 0 even when a frontend fails, so a
# silent empty build would otherwise pass. Loading uses cpg-to-falkordb's
# single-socket loader (no per-statement redis-cli). After --load it verifies
# node/edge counts in the graph, and — if --verify-prefix was given — that the
# expected FILENAME prefix(es) actually resolve to a nonzero METHOD count.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC="${1:?usage: pipeline.sh <source> [--graph NAME] [--workdir DIR] [--language LANG] [--repr R] [--reset] [--load] [--host H] [--port P] [--verify-prefix PREFIX ...]}"
shift
GRAPH=""; WORKDIR="./joern-work"; LANGUAGE=""; REPR="cpg"; RESET=""; LOAD=""
HOST="${FALKORDB_HOST:-localhost}"; PORT="${FALKORDB_PORT:-6379}"
VERIFY_PREFIXES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --graph) GRAPH="$2"; shift 2 ;;
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

mkdir -p "$WORKDIR"
CPG="$WORKDIR/cpg.bin"; EXPORT="$WORKDIR/export"; CYPHER="$WORKDIR/load.cypher"

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
  BUILT_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  STAMP="MERGE (b:CpgBuildInfo) SET b.BUILT_AT = \"$BUILT_AT\", b.SOURCE_PATH = \"$SRC\""
  if command -v git >/dev/null 2>&1 && git -C "$SRC" rev-parse --short HEAD >/dev/null 2>&1; then
    SHA="$(git -C "$SRC" rev-parse --short HEAD)"
    DIRTY=false
    [ -n "$(git -C "$SRC" status --porcelain 2>/dev/null)" ] && DIRTY=true
    STAMP="$STAMP, b.SOURCE_COMMIT = \"$SHA\", b.SOURCE_DIRTY = $DIRTY"
  fi
  redis-cli -h "$HOST" -p "$PORT" GRAPH.QUERY "$GRAPH" "$STAMP" >/dev/null
  echo "pipeline: stamped '$GRAPH' — BUILT_AT=$BUILT_AT SOURCE_PATH=$SRC" >&2
fi