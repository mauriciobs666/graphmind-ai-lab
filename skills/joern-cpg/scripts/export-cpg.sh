#!/usr/bin/env bash
# export-cpg.sh — Stage 3: export a CPG binary to an on-disk graph format.
#
# Usage: export-cpg.sh [cpg.bin] [outdir] [repr] [format]
#   cpg.bin  input CPG               (default ./cpg.bin)
#   outdir   output directory        (default ./cpg-export) — RECREATED each run
#   repr     all|ast|cdg|cfg|cpg|cpg14|ddg|pdg   (default cpg)
#   format   dot|graphml|graphson|neo4jcsv       (default neo4jcsv)
#
# joern-export requires --out to NOT pre-exist, so this removes outdir first.
# neo4jcsv is the format the FalkorDB transformer (cpg-to-falkordb.py) consumes.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./joern-env.sh
. "$HERE/joern-env.sh"

CPG="${1:-cpg.bin}"
OUTDIR="${2:-cpg-export}"
REPR="${3:-cpg}"
FORMAT="${4:-neo4jcsv}"

[ -f "$CPG" ] || { echo "export-cpg: CPG not found: $CPG (run build-cpg.sh first)" >&2; exit 1; }

# --out must not exist; recreate it. Guard against obviously unsafe targets.
case "$OUTDIR" in
  ""|"/"|"$HOME"|".") echo "export-cpg: refusing unsafe outdir '$OUTDIR'" >&2; exit 1 ;;
esac
[ -e "$OUTDIR" ] && { echo "export-cpg: clearing existing $OUTDIR" >&2; rm -rf "${OUTDIR:?}"; }

echo "export-cpg: joern-export $CPG --repr $REPR --format $FORMAT -o $OUTDIR" >&2
joern-export "$CPG" --repr "$REPR" --format "$FORMAT" -o "$OUTDIR"
echo "export-cpg: exported to $OUTDIR ($(find "$OUTDIR" -type f | wc -l) files)" >&2
