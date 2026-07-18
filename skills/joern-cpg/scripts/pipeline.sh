#!/usr/bin/env bash
# pipeline.sh — run the full Joern -> FalkorDB pipeline end to end.
#   build-cpg (parse) -> export-cpg (neo4jcsv) -> cpg-to-falkordb (transform [+ load])
#
# Usage: pipeline.sh <source> [--graph NAME] [--workdir DIR] [--load]
#                    [--repr cpg] [--host H] [--port P]
#   <source>   source dir/file to analyze (required)
#   --graph    FalkorDB graph key             (default cpg_<basename>)
#   --workdir  scratch dir for cpg.bin/export (default ./joern-work)
#   --load     ingest into FalkorDB (else stops at the .cypher artifact)
#
# Loading REFUSES a non-empty graph; reset with `redis-cli GRAPH.DELETE <graph>`
# (destructive — escalates via joern's guard) before a clean reload.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SRC="${1:?usage: pipeline.sh <source> [--graph NAME] [--workdir DIR] [--load] [--repr R] [--host H] [--port P]}"
shift
GRAPH=""; WORKDIR="./joern-work"; REPR="cpg"; LOAD=""; HOST="${FALKORDB_HOST:-localhost}"; PORT="${FALKORDB_PORT:-6379}"
while [ $# -gt 0 ]; do
  case "$1" in
    --graph) GRAPH="$2"; shift 2 ;;
    --workdir) WORKDIR="$2"; shift 2 ;;
    --repr) REPR="$2"; shift 2 ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --load) LOAD="--load"; shift ;;
    *) echo "pipeline: unknown arg '$1'" >&2; exit 2 ;;
  esac
done
[ -n "$GRAPH" ] || GRAPH="cpg_$(basename "$SRC" | tr -cs 'A-Za-z0-9_' '_')"

mkdir -p "$WORKDIR"
CPG="$WORKDIR/cpg.bin"; EXPORT="$WORKDIR/export"; CYPHER="$WORKDIR/load.cypher"

echo "== [1/3] build CPG ==" >&2
"$HERE/build-cpg.sh" "$SRC" "$CPG"
echo "== [2/3] export neo4jcsv ==" >&2
"$HERE/export-cpg.sh" "$CPG" "$EXPORT" "$REPR" neo4jcsv
echo "== [3/3] transform -> FalkorDB Cypher ($GRAPH) ==" >&2
python3 "$HERE/cpg-to-falkordb.py" "$EXPORT" -o "$CYPHER" --graph "$GRAPH" --host "$HOST" --port "$PORT" $LOAD

echo "pipeline: done. Cypher artifact: $CYPHER" >&2
[ -n "$LOAD" ] && echo "pipeline: loaded graph '$GRAPH' on $HOST:$PORT" >&2 || true
