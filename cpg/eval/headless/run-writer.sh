#!/bin/bash
# run-writer.sh — the C-1001 P0 headless-writer launcher (S53, unit P0-44).
#
# Contract: docs/plans/connected-code-graph-p0-2.md v1.2 §2 D20, §3 row S53, §6.2–§6.3, §13 K-P0-12.
#
#   run-writer.sh prepare-bin        copy the pinned Claude CLI build into a launcher-owned location
#   run-writer.sh --help
#
# Verbs still to come (after S52): prepare, start, resume, status. They read the pin record that
# prepare-bin writes (pin_read) and never the Claude versions directory.
#
# prepare-bin
#   Copies $CLAUDE_VERSIONS_DIR/$CCG_PIN_VERSION (default ~/.local/share/claude/versions/2.1.312)
#   to $CCG_EVAL_HOME/bin/claude-<version> and records its SHA-256 in
#     $CCG_EVAL_HOME/bin/claude-<version>.sha256   (sha256sum -c format)
#     $CCG_EVAL_HOME/bin/PIN                       (key=value: version, sha256, binary, source, ...)
#   If the pinned version file is gone, it takes the current build (the target of
#   ~/.local/bin/claude), sets fallback=1 in PIN and writes bin/FALLBACK-REREAD-6.3 — §6.3's flags
#   must then be re-read against that build's --help before S55 (K-P0-12).
#   Idempotent: when PIN exists and the recorded binary still hashes to the recorded SHA-256 it is
#   a no-op. Any other existing state (hash mismatch, a binary without a PIN, a PIN without its
#   binary) is refused, never overwritten.
#   Read-only towards ~/.local/bin/claude and ~/.local/share/claude/; writes only under
#   $CCG_EVAL_HOME/bin/.
#
# Exit codes: 0 = done or already prepared (fallback included: read the banner); 2 = usage;
# 3 = existing state refused; 4 = no source build found; 5 = copy failed verification.
#
# Environment overrides (tests use a temp HOME instead; these exist for the same purpose):
#   CCG_EVAL_HOME        default $HOME/.local/share/ccg-eval
#   CLAUDE_VERSIONS_DIR  default $HOME/.local/share/claude/versions
#   CLAUDE_LAUNCHER      default $HOME/.local/bin/claude
#   CCG_PIN_VERSION      default 2.1.312
set -euo pipefail
umask 022

: "${HOME:?HOME must be set}"
CCG_EVAL_HOME="${CCG_EVAL_HOME:-$HOME/.local/share/ccg-eval}"
CLAUDE_VERSIONS_DIR="${CLAUDE_VERSIONS_DIR:-$HOME/.local/share/claude/versions}"
CLAUDE_LAUNCHER="${CLAUDE_LAUNCHER:-$HOME/.local/bin/claude}"
CCG_PIN_VERSION="${CCG_PIN_VERSION:-2.1.312}"
BIN_DIR="$CCG_EVAL_HOME/bin"
PIN_FILE="$BIN_DIR/PIN"
FALLBACK_MARK="$BIN_DIR/FALLBACK-REREAD-6.3"

log() { printf 'run-writer: %s\n' "$*" >&2; }
die() { local code="$1"; shift; log "ERROR: $*"; exit "$code"; }

usage() {
	sed -n '2,/^set -euo pipefail$/{/^#/s/^# \{0,1\}//p}' "$0"
}

sha256_of() { sha256sum -- "$1" | cut -d' ' -f1; }

# pin_get KEY — print the value of KEY from PIN (exact key match, first occurrence).
pin_get() {
	local key="$1" line
	while IFS= read -r line; do
		if [[ "$line" == "$key="* ]]; then
			printf '%s\n' "${line#"$key="}"
			return 0
		fi
	done <"$PIN_FILE"
	return 1
}

# pin_verify — 0 when PIN names a binary that exists and hashes to the recorded SHA-256.
# Shared by prepare-bin's idempotency check and (later) start/resume's binary check.
pin_verify() {
	local want bin got
	want="$(pin_get sha256)" || { log "PIN has no sha256"; return 1; }
	bin="$(pin_get binary)" || { log "PIN has no binary"; return 1; }
	[[ "$want" =~ ^[0-9a-f]{64}$ ]] || { log "PIN sha256 is malformed: $want"; return 1; }
	[[ -f "$bin" && ! -L "$bin" ]] || { log "pinned binary missing or a symlink: $bin"; return 1; }
	got="$(sha256_of "$bin")"
	if [[ "$got" != "$want" ]]; then
		log "pinned binary $bin hashes to $got, PIN records $want"
		return 1
	fi
	return 0
}

fallback_banner() {
	local ver="$1"
	cat >&2 <<EOF
run-writer: ********************************************************************************
run-writer: WARNING: pinned build $CCG_PIN_VERSION is NOT on disk; took the CURRENT build $ver.
run-writer: §6.3's argv flag facts were verified against $CCG_PIN_VERSION's --help only.
run-writer: Before S55, re-read §6.3's flags against: $BIN_DIR/claude-$ver --help  (K-P0-12)
run-writer: Marker: $FALLBACK_MARK
run-writer: ********************************************************************************
EOF
}

cmd_prepare_bin() {
	[[ $# -eq 0 ]] || die 2 "prepare-bin takes no arguments"

	# 1. Existing state: a consistent PIN is a no-op; anything else is refused.
	if [[ -e "$PIN_FILE" ]]; then
		if pin_verify; then
			local v s
			v="$(pin_get version)"; s="$(pin_get sha256)"
			log "already prepared: claude $v sha256=$s ($(pin_get binary)) — no-op"
			[[ "$(pin_get fallback || true)" == "1" ]] && fallback_banner "$v"
			return 0
		fi
		die 3 "existing pin in $BIN_DIR does not verify; refusing to overwrite (inspect by hand)"
	fi
	local existing
	existing="$(find "$BIN_DIR" -maxdepth 1 -name 'claude-*' -print -quit 2>/dev/null || true)"
	[[ -z "$existing" ]] || die 3 "found $existing without a PIN record; refusing to overwrite"

	# 2. Source: the pinned version file, else the current build (fallback, K-P0-12).
	local src="$CLAUDE_VERSIONS_DIR/$CCG_PIN_VERSION" ver="$CCG_PIN_VERSION" fallback=0
	if [[ ! -f "$src" ]]; then
		local cur
		cur="$(readlink -f -- "$CLAUDE_LAUNCHER" 2>/dev/null || true)"
		[[ -n "$cur" && -f "$cur" ]] \
			|| die 4 "neither $src nor a current build behind $CLAUDE_LAUNCHER exists"
		src="$cur"; ver="$(basename -- "$cur")"; fallback=1
		[[ "$ver" =~ ^[0-9A-Za-z._+-]+$ ]] || die 4 "current build has an unusable name: $ver"
	fi
	local dest="$BIN_DIR/claude-$ver"

	# 3. Copy to a temp name in the same directory, verify, then rename into place.
	mkdir -p -- "$BIN_DIR"
	local src_sha tmp got
	src_sha="$(sha256_of "$src")"
	tmp="$(mktemp -- "$BIN_DIR/.claude-$ver.XXXXXX")"
	trap 'rm -f -- "$tmp"' EXIT
	cp -- "$src" "$tmp"
	got="$(sha256_of "$tmp")"
	[[ "$got" == "$src_sha" ]] || die 5 "copy hashes to $got, source $src to $src_sha"
	chmod 0555 -- "$tmp"
	mv -n -- "$tmp" "$dest"
	[[ ! -e "$tmp" ]] || die 3 "$dest appeared concurrently; refusing to overwrite"
	trap - EXIT

	# 4. Record. sidecar first, PIN last (PIN's presence means "complete").
	printf '%s  %s\n' "$got" "$dest" >"$dest.sha256"
	local pin_tmp
	pin_tmp="$(mktemp -- "$BIN_DIR/.PIN.XXXXXX")"
	{
		printf 'version=%s\n' "$ver"
		printf 'sha256=%s\n' "$got"
		printf 'binary=%s\n' "$dest"
		printf 'source=%s\n' "$src"
		printf 'requested=%s\n' "$CCG_PIN_VERSION"
		printf 'fallback=%s\n' "$fallback"
		printf 'prepared_at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	} >"$pin_tmp"
	chmod 0444 -- "$pin_tmp"
	mv -- "$pin_tmp" "$PIN_FILE"

	if [[ "$fallback" == "1" ]]; then
		printf '%s\n' \
			"Pinned build $CCG_PIN_VERSION was gone at prepare-bin time; took current build $ver." \
			"Re-read docs/plans/connected-code-graph-p0-2.md §6.3's flags against" \
			"  $dest --help" \
			"before S55, then S55 and S54 re-run (K-P0-12)." >"$FALLBACK_MARK"
		fallback_banner "$ver"
	fi
	log "prepared: claude $ver sha256=$got -> $dest"
}

main() {
	local verb="${1:-}"
	[[ $# -gt 0 ]] && shift
	case "$verb" in
		prepare-bin) cmd_prepare_bin "$@" ;;
		-h|--help|help) usage ;;
		prepare|start|resume|status) die 2 "verb '$verb' is not implemented yet (S53, after S52)" ;;
		'') usage >&2; exit 2 ;;
		*) die 2 "unknown verb: $verb (try --help)" ;;
	esac
}

main "$@"
