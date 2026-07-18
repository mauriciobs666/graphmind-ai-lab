#!/usr/bin/env bash
# build-cpg.sh — Stage 1: parse a source tree into a CPG binary (overlays applied).
#
# Usage: build-cpg.sh <source-dir-or-file> [output.bin] [-- <extra joern-parse args>]
#   output.bin defaults to ./cpg.bin
#   JOERN_LANGUAGE=<lang>  forces a frontend (else joern-parse auto-detects).
#   Anything after `--` is passed to joern-parse verbatim (e.g. --nooverlays).
#
# Example: build-cpg.sh ../falkor-chat/app cpg.bin
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./joern-env.sh
. "$HERE/joern-env.sh"

SRC="${1:?usage: build-cpg.sh <source-dir-or-file> [output.bin] [-- <extra joern-parse args>]}"
shift
OUT="cpg.bin"
if [ "${1:-}" != "" ] && [ "${1:-}" != "--" ]; then OUT="$1"; shift; fi
[ "${1:-}" = "--" ] && shift

[ -e "$SRC" ] || { echo "build-cpg: source not found: $SRC" >&2; exit 1; }

LANG_ARGS=()
[ -n "${JOERN_LANGUAGE:-}" ] && LANG_ARGS=(--language "$JOERN_LANGUAGE")

echo "build-cpg: joern-parse $SRC -o $OUT ${LANG_ARGS[*]} $*" >&2
joern-parse "$SRC" -o "$OUT" "${LANG_ARGS[@]}" "$@"
echo "build-cpg: wrote $OUT ($(du -h "$OUT" | cut -f1))" >&2
