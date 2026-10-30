#!/bin/bash
# test_prepare_bin.sh — S53 prepare-bin tests (docs/plans/connected-code-graph-p0-2.md §3 S53 (1)).
# Each case runs against a fake Claude install under its own temp HOME; the real ~ is never read.
#   RUN_WRITER=<path> overrides the script under test (used by mutants.sh).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_WRITER="${RUN_WRITER:-$HERE/../run-writer.sh}"
TMP_ROOT="$(mktemp -d)"
trap 'chmod -R u+w "$TMP_ROOT" 2>/dev/null; rm -rf -- "$TMP_ROOT"' EXIT

pass=0 fail=0
ok()   { pass=$((pass + 1)); printf 'ok   %s\n' "$1"; }
bad()  { fail=$((fail + 1)); printf 'FAIL %s\n' "$1"; }
check() { local name="$1"; shift; if "$@"; then ok "$name"; else bad "$name"; fi; }

# new_home <name> <versions...> <current> — fake HOME with fake builds; launcher -> <current>.
new_home() {
	local h="$TMP_ROOT/$1"; shift
	local cur="${!#}" v
	mkdir -p "$h/.local/share/claude/versions" "$h/.local/bin"
	for v in "${@:1:$#-1}"; do
		printf '#!/bin/sh\necho "%s (Claude Code)"\n' "$v" >"$h/.local/share/claude/versions/$v"
		chmod 0755 "$h/.local/share/claude/versions/$v"
	done
	ln -s "$h/.local/share/claude/versions/$cur" "$h/.local/bin/claude"
	printf '%s\n' "$h"
}

# run <home> — run prepare-bin under a scrubbed env; sets RC, OUT (stderr+stdout).
run() { OUT="$(env -i HOME="$1" PATH=/usr/bin:/bin LANG=C.UTF-8 bash "$RUN_WRITER" prepare-bin 2>&1)"; RC=$?; }
pin()  { grep -m1 "^$2=" "$1/.local/share/ccg-eval/bin/PIN" | cut -d= -f2-; }

# --- 1. copy -------------------------------------------------------------------------------
H="$(new_home copy 2.1.312 2.1.314 2.1.314)"
B="$H/.local/share/ccg-eval/bin"
SRC_SHA="$(sha256sum "$H/.local/share/claude/versions/2.1.312" | cut -d' ' -f1)"
LAUNCH_BEFORE="$(readlink "$H/.local/bin/claude")"
run "$H"
check "copy: exit 0"                  test "$RC" -eq 0
check "copy: binary is byte-identical" cmp -s "$H/.local/share/claude/versions/2.1.312" "$B/claude-2.1.312"
check "copy: PIN sha256 = source"     test "$(pin "$H" sha256)" = "$SRC_SHA"
check "copy: PIN version 2.1.312"     test "$(pin "$H" version)" = "2.1.312"
check "copy: PIN fallback=0"          test "$(pin "$H" fallback)" = "0"
check "copy: sidecar verifies"        sha256sum -c --quiet "$B/claude-2.1.312.sha256"
check "copy: no fallback marker"      test ! -e "$B/FALLBACK-REREAD-6.3"
check "copy: copy runs standalone"    test "$(env -i PATH=/usr/bin:/bin "$B/claude-2.1.312")" = "2.1.312 (Claude Code)"
check "copy: launcher untouched"      test "$(readlink "$H/.local/bin/claude")" = "$LAUNCH_BEFORE"
check "copy: no temp files left"      test -z "$(find "$B" -name '.*' -type f)"

# --- 2. idempotent re-run ------------------------------------------------------------------
INODE="$(stat -c %i:%Y "$B/claude-2.1.312")"; PIN_BEFORE="$(cat "$B/PIN")"
run "$H"
check "rerun: exit 0"                 test "$RC" -eq 0
check "rerun: says no-op"             grep -q 'no-op' <<<"$OUT"
check "rerun: binary not rewritten"   test "$(stat -c %i:%Y "$B/claude-2.1.312")" = "$INODE"
check "rerun: PIN not rewritten"      test "$(cat "$B/PIN")" = "$PIN_BEFORE"
# Re-run still a no-op after the source build is pruned (the copy is the authority).
rm -f "$H/.local/share/claude/versions/2.1.312"
run "$H"
check "rerun after prune: exit 0, no fallback" test "$RC" -eq 0 -a "$(pin "$H" fallback)" = "0"

# --- 3. mismatch refusal -------------------------------------------------------------------
H="$(new_home mismatch 2.1.312 2.1.314 2.1.314)"; B="$H/.local/share/ccg-eval/bin"
run "$H"
chmod u+w "$B/claude-2.1.312"; printf 'tampered\n' >>"$B/claude-2.1.312"
TAMPERED="$(sha256sum "$B/claude-2.1.312" | cut -d' ' -f1)"
run "$H"
check "mismatch: exit 3"              test "$RC" -eq 3
check "mismatch: copy not overwritten" test "$(sha256sum "$B/claude-2.1.312" | cut -d' ' -f1)" = "$TAMPERED"

# PIN records a different hash than the (intact) binary.
H="$(new_home badpin 2.1.312 2.1.314 2.1.314)"; B="$H/.local/share/ccg-eval/bin"
run "$H"
chmod u+w "$B/PIN"; sed -i 's/^sha256=.*/sha256=0000000000000000000000000000000000000000000000000000000000000000/' "$B/PIN"
run "$H"
check "bad PIN hash: exit 3"          test "$RC" -eq 3

# A binary already present without a PIN record.
H="$(new_home orphan 2.1.312 2.1.314 2.1.314)"; B="$H/.local/share/ccg-eval/bin"
mkdir -p "$B"; printf 'something else\n' >"$B/claude-2.1.312"
run "$H"
check "orphan binary: exit 3"         test "$RC" -eq 3
check "orphan binary: not overwritten" test "$(cat "$B/claude-2.1.312")" = "something else"

# --- 4. missing-2.1.312 fallback -----------------------------------------------------------
H="$(new_home fallback 2.1.310 2.1.314 2.1.314)"; B="$H/.local/share/ccg-eval/bin"
run "$H"
check "fallback: exit 0"              test "$RC" -eq 0
check "fallback: took current build"  cmp -s "$H/.local/share/claude/versions/2.1.314" "$B/claude-2.1.314"
check "fallback: no 2.1.312 name"     test ! -e "$B/claude-2.1.312"
check "fallback: PIN version 2.1.314" test "$(pin "$H" version)" = "2.1.314"
check "fallback: PIN fallback=1"      test "$(pin "$H" fallback)" = "1"
check "fallback: PIN sha256 = source" test "$(pin "$H" sha256)" = "$(sha256sum "$H/.local/share/claude/versions/2.1.314" | cut -d' ' -f1)"
check "fallback: marker written"      grep -q 'K-P0-12' "$B/FALLBACK-REREAD-6.3"
check "fallback: loud banner"         grep -q 'WARNING: pinned build 2.1.312 is NOT on disk' <<<"$OUT"
run "$H"
check "fallback rerun: no-op, still loud" test "$RC" -eq 0 -a -n "$(grep 'WARNING' <<<"$OUT")"

# --- 5. nothing to copy --------------------------------------------------------------------
H="$TMP_ROOT/empty"; mkdir -p "$H"
run "$H"
check "no source: exit 4"             test "$RC" -eq 4

# --- 6. usage ------------------------------------------------------------------------------
OUT="$(bash "$RUN_WRITER" start 2>&1)"; RC=$?
check "unimplemented verb: exit 2"    test "$RC" -eq 2

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[[ "$fail" -eq 0 ]]
