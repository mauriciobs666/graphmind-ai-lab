#!/usr/bin/env bash
# pipeline.sh — run the full Joern -> FalkorDB pipeline end to end, for ANY source.
#   build-cpg (parse) -> export-cpg (neo4jcsv) -> cpg-to-falkordb (transform [+ load])
#
# Generic: the caller says WHAT to build; there are no baked-in project/app names.
#
# Usage: pipeline.sh <source> [--graph NAME] [--workdir DIR] [--language LANG]
#                    [--repr R] [--reset] [--load] [--host H] [--port P]
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
#
# Robustness: after transform the pipeline ASSERTS the CPG produced nodes and
# fails loudly otherwise — joern-parse exits 0 even when a frontend fails, so a
# silent empty build would otherwise pass. Loading uses cpg-to-falkordb's
# single-socket loader (no per-statement redis-cli). After --load it verifies
# node/edge counts in the graph.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC="${1:?usage: pipeline.sh <source> [--graph NAME] [--workdir DIR] [--language LANG] [--repr R] [--reset] [--load] [--host H] [--port P]}"
shift
GRAPH=""; WORKDIR="./joern-work"; LANGUAGE=""; REPR="cpg"; RESET=""; LOAD=""
HOST="${FALKORDB_HOST:-localhost}"; PORT="${FALKORDB_PORT:-6379}"
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
fi