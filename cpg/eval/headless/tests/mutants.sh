#!/bin/bash
# mutants.sh — mutation check for test_prepare_bin.sh. Each mutant is a sed-edited COPY of
# run-writer.sh in a temp dir (the tracked file is never touched); the suite must go red on each.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG="$HERE/../run-writer.sh"
WORK="$(mktemp -d)"
trap 'rm -rf -- "$WORK"' EXIT

# name | sed expression
MUTANTS=(
	'sha-check-disabled|s/if \[\[ "\$got" != "\$want" \]\]; then/if false; then/'
	'mismatch-overwrites|s/die 3 "existing pin in .*/rm -f -- "$BIN_DIR"\/claude-* "$PIN_FILE"/'
	'orphan-overwrites|s/\[\[ -z "\$existing" \]\] || die 3/[[ -z "$existing" ]] || rm -f -- "$existing" #/'
	'fallback-silent|s/^\t\tfallback_banner "\$ver"$/\t\t:/'
	'fallback-misnamed|s/ver="\$(basename -- "\$cur")"; fallback=1/fallback=1/'
)

survived=0
for m in "${MUTANTS[@]}"; do
	name="${m%%|*}" expr="${m#*|}"
	mut="$WORK/$name.sh"
	sed -e "$expr" "$ORIG" >"$mut"
	if cmp -s "$ORIG" "$mut"; then
		printf 'BROKEN  %-22s sed did not apply\n' "$name"; survived=$((survived + 1)); continue
	fi
	if ! bash -n "$mut"; then
		printf 'BROKEN  %-22s mutant does not parse\n' "$name"; survived=$((survived + 1)); continue
	fi
	if RUN_WRITER="$mut" bash "$HERE/test_prepare_bin.sh" >"$WORK/$name.log" 2>&1; then
		printf 'SURVIVED %-22s\n' "$name"; survived=$((survived + 1))
	else
		printf 'killed  %-22s %s\n' "$name" "$(grep -m3 '^FAIL' "$WORK/$name.log" | tr '\n' ';')"
	fi
done
printf '\n%d of %d mutants survived\n' "$survived" "${#MUTANTS[@]}"
[[ "$survived" -eq 0 ]]
