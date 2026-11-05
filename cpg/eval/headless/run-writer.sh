#!/bin/bash
# run-writer.sh — the C-1001 P0 headless-writer launcher (S53, unit P0-44).
#
# Contract: docs/plans/connected-code-graph-p0-2.md v1.2 §2 D20, §3 row S53, §6.2–§6.3, §13 K-P0-12.
#
#   run-writer.sh prepare-bin                       copy the pinned Claude CLI build into a launcher-owned location
#   run-writer.sh prepare <E> <role> <stratum> --brief FILE [--kit-src DIR]
#                                                   lay out one role root (idempotent)
#   run-writer.sh start   [--dry-run] [--restart] [--model M] [--timeout S] [--session-id U]
#                         <E> <role> <stratum>      first attempt of a new session (n=1)
#   run-writer.sh resume  [--dry-run] [--assume-ended] [--timeout S] [--session-id U]
#                         <E> <role> <stratum>      next attempt of the recorded session (n+1)
#   run-writer.sh status  [<E> [<role> <stratum>]]  read-only: sessions, attempts, drift check
#   run-writer.sh --help
#
# <E> is the eval commit (first 12 chars are used). <role> is KA KB QW AN1 AN2 rehearsal capture.
# Everything lives under $CCG_EVAL_HOME/<E>/ (docs/plans/connected-code-graph-p0-2.md §6.2).
# The later verbs read the binary path and hash from bin/PIN (written by prepare-bin), never from
# the Claude versions directory.
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
# prepare   Creates $CCG_EVAL_HOME/<E>/{runs,work} and roles/<role>-<stratum>/{kit-src,BRIEF.md,out}.
#   kit-src is a read-only copy of --kit-src (default <E>/kit-src); BRIEF.md a read-only copy of
#   --brief. Re-running with identical inputs is a no-op; different inputs are refused (exit 3),
#   never overwritten.
# start     Builds argv with build_argv (the §6.3 launch command), records the attempt in
#   runs/index.jsonl, runs it in the foreground, writes runs/<uuid>.1.stream.jsonl, records the end.
#   Refuses if the role already has a session (--restart: allowed only when out/ is empty), if
#   the pinned binary no longer hashes to PIN, or if the guard/python3 are not usable.
#   --dry-run prints the command and writes nothing. Defaults per role: model, agent and timeout
#   (KA opus 4h, KB sonnet 4h, QW opus 2h, AN1 sonnet 1h, AN2 opus 1h, rehearsal/capture sonnet 30m).
# resume    The same build_argv with --resume <uuid>, the fixed continuation prompt and the next
#   attempt number. Reuses the recorded model/agent. Refuses an unknown session, a changed
#   CLI version / binary / settings.json / agents.json hash against ANY earlier attempt, and a
#   previous attempt with no end record (unless --assume-ended).
#
# Exit codes: 0 = done / no-op (prepare-bin fallback included: read the banner); 2 = usage;
# 3 = existing state or drift refused; 4 = a required input is missing; 5 = copy failed verification.
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
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETTINGS_FILE="$HERE/settings.json"
AGENTS_FILE="$HERE/agents.json"
GUARD_FILE="$HERE/guard-paths.sh"
CONTINUE_PROMPT="continue; your output file lists what is already done"
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

# ---------------------------------------------------------------------------------------------
# S53 verbs: prepare, start, resume, status
# ---------------------------------------------------------------------------------------------

ROLES="KA KB QW AN1 AN2 rehearsal capture"

# role_defaults ROLE — sets D_AGENT D_MODEL D_TIMEOUT (plan §6.1 / §6.3).
role_defaults() {
	case "$1" in
		KA)        D_AGENT=ccg-key-writer-provider; D_MODEL=opus;   D_TIMEOUT=14400 ;;
		KB)        D_AGENT=ccg-key-writer-consumer; D_MODEL=sonnet; D_TIMEOUT=14400 ;;
		QW)        D_AGENT=ccg-question-writer;     D_MODEL=opus;   D_TIMEOUT=7200 ;;
		AN1)       D_AGENT=ccg-annotator;           D_MODEL=sonnet; D_TIMEOUT=3600 ;;
		AN2)       D_AGENT=ccg-annotator;           D_MODEL=opus;   D_TIMEOUT=3600 ;;
		rehearsal|capture)
		           D_AGENT=ccg-annotator;           D_MODEL=sonnet; D_TIMEOUT=1800 ;;
		*) die 2 "unknown role '$1' (one of: $ROLES)" ;;
	esac
}

# set_target E ROLE STRATUM — validates and sets E12 EDIR RUNS ROLE_ROOT INDEX.
set_target() {
	local e="$1" role="$2" stratum="$3"
	[[ "$e" =~ ^[0-9A-Za-z][0-9A-Za-z._-]*$ ]] || die 2 "bad eval commit '$e'"
	[[ " $ROLES " == *" $role "* ]] || die 2 "unknown role '$role' (one of: $ROLES)"
	[[ "$stratum" =~ ^[a-z][a-z0-9-]*$ ]] || die 2 "bad stratum '$stratum' (lowercase, digits, '-')"
	E12="${e:0:12}"
	T_ROLE="$role"; T_STRATUM="$stratum"
	EDIR="$CCG_EVAL_HOME/$E12"
	RUNS="$EDIR/runs"
	ROLE_ROOT="$EDIR/roles/$role-$stratum"
	INDEX="$RUNS/index.jsonl"
}

need_uint() { [[ "$2" =~ ^[1-9][0-9]*$ ]] || die 2 "$1 must be a positive integer, got '$2'"; }
need_uuid() { [[ "$1" =~ ^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$ ]] || die 2 "not a lowercase UUID: $1"; }
new_uuid() { tr 'A-F' 'a-f' </proc/sys/kernel/random/uuid; }

# idx_get LINE KEY — value of KEY in one index line (values never contain , " or }).
idx_get() { sed -n "s/.*\"$2\":\"\{0,1\}\([^\",}]*\).*/\1/p" <<<"$1"; }

# idx_events SESSION EVENT — all matching index lines, oldest first.
idx_events() {
	[[ -f "$INDEX" ]] || return 0
	grep -F -- "\"event\":\"$2\"" "$INDEX" | grep -F -- "\"session\":\"$1\"" || true
}

# next_attempt SESSION — one past the highest n seen in the index or among existing stream files.
next_attempt() {
	local u="$1" max=0 line n f
	while IFS= read -r line; do
		[[ -n "$line" ]] || continue
		n="$(idx_get "$line" n)"; [[ "$n" =~ ^[0-9]+$ ]] && (( n > max )) && max="$n"
	done < <(idx_events "$u" start)
	for f in "$RUNS/$u".*.stream.jsonl; do
		[[ -e "$f" ]] || continue
		n="${f#"$RUNS/$u."}"; n="${n%.stream.jsonl}"
		[[ "$n" =~ ^[0-9]+$ ]] && (( n > max )) && max="$n"
	done
	printf '%s\n' $((max + 1))
}

MAX_PROMPT_BYTES=100000   # one exec argument must stay well under the 128 KiB kernel limit

# check_prompt_size FILE — refuse an oversized brief before anything is written.
check_prompt_size() {
	local sz; sz="$(wc -c <"$1")"
	(( sz <= MAX_PROMPT_BYTES )) || die 2 "$1 is $sz bytes; a prompt over $MAX_PROMPT_BYTES bytes cannot be passed as one argument"
}

# idx_append LINE — append one record; if a killed launcher left a partial last line, break it first.
idx_append() {
	if [[ -s "$INDEX" && -n "$(tail -c1 "$INDEX")" ]]; then printf '\n' >>"$INDEX"; fi
	printf '%s\n' "$1" >>"$INDEX"
}

# acquire_lock — exclusive per role+stratum, held until this launcher exits (mkdir is atomic).
# A lock whose recorded pid is dead is stale and is taken over once.
LOCK_DIR=""
acquire_lock() {
	local d="$RUNS/.lock-$T_ROLE-$T_STRATUM" pid
	if ! mkdir -- "$d" 2>/dev/null; then
		pid="$(cat -- "$d/pid" 2>/dev/null || true)"
		if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
			log "taking over a stale lock (pid $pid is gone): $d"
			rm -rf -- "${d:?}"
			mkdir -- "$d" 2>/dev/null || die 3 "$T_ROLE-$T_STRATUM is locked by another launcher: $d"
		else
			die 3 "$T_ROLE-$T_STRATUM is locked by another launcher (pid ${pid:-unknown}): $d"
		fi
	fi
	printf '%s\n' "$$" >"$d/pid"
	LOCK_DIR="$d"
	trap 'rm -rf -- "${LOCK_DIR:?}"' EXIT
}

json_str() { local s="${1//\\/\\\\}"; printf '"%s"' "${s//\"/\\\"}"; }

# check_launch_inputs — refuse unless the pin verifies and the repo-side files are usable.
# Sets BIN CLI_VERSION BIN_SHA SETTINGS_SHA AGENTS_SHA.
check_launch_inputs() {
	[[ -f "$PIN_FILE" ]] || die 4 "no PIN in $BIN_DIR: run 'run-writer.sh prepare-bin' first"
	pin_verify || die 3 "the pinned binary does not verify against PIN; refusing to launch"
	BIN="$(pin_get binary)"; CLI_VERSION="$(pin_get version)"; BIN_SHA="$(pin_get sha256)"
	[[ -x "$BIN" ]] || die 3 "pinned binary is not executable: $BIN"
	[[ -f "$SETTINGS_FILE" ]] || die 4 "missing $SETTINGS_FILE"
	[[ -f "$AGENTS_FILE" ]] || die 4 "missing $AGENTS_FILE"
	[[ -x "$GUARD_FILE" ]] || die 4 "path guard missing or not executable: $GUARD_FILE"
	[[ -x /usr/bin/python3 ]] || die 4 "the path guard needs /usr/bin/python3"
	SETTINGS_SHA="$(sha256_of "$SETTINGS_FILE")"
	AGENTS_SHA="$(sha256_of "$AGENTS_FILE")"
}

# build_argv MODE AGENT MODEL TIMEOUT SESSION PROMPT — the ONE argv builder (§6.3), used by start
# and resume. MODE is start|resume; it alone picks --session-id vs --resume. Fills ARGV.
build_argv() {
	local mode="$1" agent="$2" model="$3" tmo="$4" session="$5" prompt="$6" sflag
	case "$mode" in
		start)  sflag=--session-id ;;
		resume) sflag=--resume ;;
		*) die 2 "build_argv: bad mode $mode" ;;
	esac
	ARGV=(
		env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 CCG_GUARD="$GUARD_FILE"
		timeout --kill-after=60 "$tmo"
		"$BIN" -p --restricted --strict-mcp-config --disable-slash-commands
		--tools Read,Grep,Glob,Write,Edit
		--allowedTools Read Grep Glob Write Edit
		--disallowedTools Bash WebFetch WebSearch Agent
		--settings "$SETTINGS_FILE"
		--agents "$AGENTS_FILE" --agent "$agent"
		--model "$model" --permission-mode dontAsk --permission-prompts none
		--output-format stream-json --verbose
		"$sflag" "$session" "$prompt"
	)
}

# shq TOKEN — shell-quote only when needed (printf %q would escape the commas of --tools).
shq() { if [[ "$1" =~ ^[A-Za-z0-9_@%+=:,./-]+$ ]]; then printf '%s' "$1"; else printf '%q' "$1"; fi; }

print_command() { # print_command STREAM — one argv token per line, so a diff is line-exact
	local t
	printf 'cd %s\n' "$(shq "$ROLE_ROOT")"
	for t in "${ARGV[@]}"; do printf '  %s\n' "$(shq "$t")"; done
	printf '  > %s\n' "$(shq "$1")"
}

# launch MODE AGENT MODEL TIMEOUT SESSION PROMPT N DRY
launch() {
	local mode="$1" agent="$2" model="$3" tmo="$4" session="$5" prompt="$6" n="$7" dry="$8"
	local stream="$RUNS/$session.$n.stream.jsonl"
	build_argv "$mode" "$agent" "$model" "$tmo" "$session" "$prompt"
	if [[ "$dry" == 1 ]]; then
		printf '# DRY RUN (%s, attempt %s): nothing written, nothing started\n' "$mode" "$n"
		print_command "$stream"
		return 0
	fi
	# New file only: noclobber makes this fail if the log exists, never truncating an earlier one.
	( set -o noclobber; : >"$stream" ) 2>/dev/null || die 3 "stream log already exists: $stream"
	local start; start="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
	idx_append "$(printf '{"event":"start","role":%s,"stratum":%s,"eval":%s,"session":"%s","n":%s,"mode":"%s","agent":%s,"model":%s,"timeout_s":%s,"cli_version":%s,"bin_sha256":"%s","settings_sha256":"%s","agents_sha256":"%s","stream":%s,"start":"%s"}\n' \
		"$(json_str "$T_ROLE")" "$(json_str "$T_STRATUM")" "$(json_str "$E12")" "$session" "$n" "$mode" \
		"$(json_str "$agent")" "$(json_str "$model")" "$tmo" "$(json_str "$CLI_VERSION")" \
		"$BIN_SHA" "$SETTINGS_SHA" "$AGENTS_SHA" "$(json_str "$(basename -- "$stream")")" "$start")"
	log "attempt $n of session $session ($T_ROLE-$T_STRATUM, $model): $stream"
	# Background child + wait, so TERM/INT to the launcher is forwarded to `timeout` (which stops
	# claude) and the end record is still written. stdin is /dev/null.
	local rc=0 child
	( cd "$ROLE_ROOT" && exec "${ARGV[@]}" ) </dev/null >>"$stream" &
	child=$!
	trap 'kill -TERM "$child" 2>/dev/null || true' TERM INT
	wait "$child" || rc=$?
	while kill -0 "$child" 2>/dev/null; do wait "$child" || rc=$?; done
	trap - TERM INT
	idx_append "$(printf '{"event":"end","session":"%s","n":%s,"end":"%s","exit":%s}' \
		"$session" "$n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc")"
	log "attempt $n finished: exit $rc"
	return "$rc"
}

require_role_root() {
	[[ -d "$ROLE_ROOT" && -f "$ROLE_ROOT/BRIEF.md" && -d "$ROLE_ROOT/out" && -d "$ROLE_ROOT/kit-src" ]] \
		|| die 4 "role root not prepared: $ROLE_ROOT (run 'prepare' first)"
	[[ -d "$RUNS" ]] || die 4 "no runs dir: $RUNS"
}

parse_target() { # parse_target ARGS... — sets POS (E role stratum) after option parsing by caller
	[[ ${#POS[@]} -eq 3 ]] || die 2 "expected <E> <role> <stratum>, got ${#POS[@]} argument(s)"
	set_target "${POS[0]}" "${POS[1]}" "${POS[2]}"
}

cmd_prepare() {
	local brief="" kit="" a
	POS=()
	while [[ $# -gt 0 ]]; do
		a="$1"; shift
		case "$a" in
			--brief) [[ $# -ge 1 ]] || die 2 "--brief needs a file"; brief="$1"; shift ;;
			--kit-src) [[ $# -ge 1 ]] || die 2 "--kit-src needs a directory"; kit="$1"; shift ;;
			-*) die 2 "prepare: unknown option $a" ;;
			*) POS+=("$a") ;;
		esac
	done
	parse_target
	[[ -n "$brief" ]] || die 2 "prepare needs --brief FILE"
	[[ -f "$brief" ]] || die 4 "brief not found: $brief"
	kit="${kit:-$EDIR/kit-src}"
	[[ -d "$kit" ]] || die 4 "kit-src not found: $kit"
	check_prompt_size "$brief"
	local link; link="$(find "$kit" -type l -print -quit)"
	[[ -z "$link" ]] || die 3 "kit-src contains a symlink ($link); flatten the kit first (a link could point outside the role root)"

	if [[ -e "$ROLE_ROOT" ]]; then
		if [[ -f "$ROLE_ROOT/BRIEF.md" ]] && cmp -s "$brief" "$ROLE_ROOT/BRIEF.md" \
			&& [[ -d "$ROLE_ROOT/kit-src" && -d "$ROLE_ROOT/out" ]] \
			&& diff -r -q -- "$kit" "$ROLE_ROOT/kit-src" >/dev/null 2>&1; then
			log "role root already prepared with identical inputs: $ROLE_ROOT — no-op"
			return 0
		fi
		die 3 "role root exists with different content; refusing to overwrite: $ROLE_ROOT"
	fi

	mkdir -p -- "$RUNS" "$EDIR/work" "$EDIR/roles"
	local tmp
	tmp="$(mktemp -d -- "$EDIR/roles/.prepare.XXXXXX")"
	trap 'chmod -R u+w -- "$tmp" 2>/dev/null; rm -rf -- "$tmp"' EXIT
	cp -a -- "$kit" "$tmp/kit-src"
	cp -- "$brief" "$tmp/BRIEF.md"
	mkdir -- "$tmp/out"
	chmod -R a-w -- "$tmp/kit-src"
	chmod 0444 -- "$tmp/BRIEF.md"
	chmod 0755 -- "$tmp"
	mv -T -- "$tmp" "$ROLE_ROOT" 2>/dev/null || die 3 "$ROLE_ROOT appeared concurrently; refusing to overwrite"
	trap - EXIT
	log "prepared role root: $ROLE_ROOT"
}

cmd_start() {
	local dry=0 restart=0 model="" tmo="" session="" a
	POS=()
	while [[ $# -gt 0 ]]; do
		a="$1"; shift
		case "$a" in
			--dry-run) dry=1 ;;
			--restart) restart=1 ;;
			--model) [[ $# -ge 1 ]] || die 2 "--model needs a value"; model="$1"; shift ;;
			--timeout) [[ $# -ge 1 ]] || die 2 "--timeout needs seconds"; tmo="$1"; shift ;;
			--session-id) [[ $# -ge 1 ]] || die 2 "--session-id needs a UUID"; session="$1"; shift ;;
			-*) die 2 "start: unknown option $a" ;;
			*) POS+=("$a") ;;
		esac
	done
	parse_target
	role_defaults "$T_ROLE"
	model="${model:-$D_MODEL}"; tmo="${tmo:-$D_TIMEOUT}"
	[[ "$model" =~ ^[0-9A-Za-z][0-9A-Za-z._-]*$ ]] || die 2 "bad model '$model'"
	need_uint "--timeout" "$tmo"
	if [[ -n "$session" ]]; then need_uuid "$session"; else session="$(new_uuid)"; fi
	require_role_root
	check_launch_inputs
	check_prompt_size "$ROLE_ROOT/BRIEF.md"
	[[ "$dry" == 1 ]] || acquire_lock

	local prior=""
	if [[ -f "$INDEX" ]]; then
		prior="$(grep -F -- "\"role\":$(json_str "$T_ROLE"),\"stratum\":$(json_str "$T_STRATUM")," "$INDEX" | grep -F '"event":"start"' | tail -n 1 || true)"
	fi
	if [[ -n "$prior" ]]; then
		[[ "$restart" == 1 ]] || die 3 "$T_ROLE-$T_STRATUM already has session $(idx_get "$prior" session); use resume, or --restart for a fresh session on an empty out/"
	fi
	if [[ -n "$(find "$ROLE_ROOT/out" -mindepth 1 -print -quit)" ]]; then
		die 3 "out/ is not empty ($ROLE_ROOT/out); a fresh session starts from an empty output (the launcher never deletes it)"
	fi
	[[ -z "$(idx_events "$session" start)" ]] || die 3 "session id $session is already in the index"
	compgen -G "$RUNS/$session.*.stream.jsonl" >/dev/null && die 3 "stream logs for session $session already exist in $RUNS"
	local n; n="$(next_attempt "$session")"
	local prompt; prompt="$(<"$ROLE_ROOT/BRIEF.md")"
	[[ "$prompt" != -* ]] || die 4 "BRIEF.md must not start with '-' (it would parse as an option)"
	[[ -n "$prompt" ]] || die 4 "BRIEF.md is empty"
	[[ "$dry" == 1 ]] || mkdir -p -- "$RUNS"
	launch start "$D_AGENT" "$model" "$tmo" "$session" "$prompt" "$n" "$dry"
}

cmd_resume() {
	local dry=0 assume=0 tmo="" session="" a
	POS=()
	while [[ $# -gt 0 ]]; do
		a="$1"; shift
		case "$a" in
			--dry-run) dry=1 ;;
			--assume-ended) assume=1 ;;
			--timeout) [[ $# -ge 1 ]] || die 2 "--timeout needs seconds"; tmo="$1"; shift ;;
			--session-id) [[ $# -ge 1 ]] || die 2 "--session-id needs a UUID"; session="$1"; shift ;;
			-*) die 2 "resume: unknown option $a" ;;
			*) POS+=("$a") ;;
		esac
	done
	parse_target
	require_role_root
	check_launch_inputs
	[[ "$dry" == 1 ]] || acquire_lock

	# The session is the latest recorded one for this role+stratum; --session-id must match it.
	local last=""
	[[ -f "$INDEX" ]] && last="$(grep -F -- "\"role\":$(json_str "$T_ROLE"),\"stratum\":$(json_str "$T_STRATUM")," "$INDEX" | grep -F '"event":"start"' | tail -n 1 || true)"
	[[ -n "$last" ]] || die 3 "no recorded session for $T_ROLE-$T_STRATUM; nothing to resume"
	local known; known="$(idx_get "$last" session)"
	if [[ -n "$session" ]]; then
		need_uuid "$session"
		[[ -n "$(idx_events "$session" start)" ]] || die 3 "unknown session id $session"
		[[ "$session" == "$known" ]] || die 3 "session $session is not the current session of $T_ROLE-$T_STRATUM ($known)"
	fi
	session="$known"

	# Drift: every earlier attempt of the session must match the current CLI, binary and configs.
	local line key want got
	while IFS= read -r line; do
		[[ -n "$line" ]] || continue
		for key in cli_version:"$CLI_VERSION" bin_sha256:"$BIN_SHA" settings_sha256:"$SETTINGS_SHA" agents_sha256:"$AGENTS_SHA"; do
			want="${key#*:}"; got="$(idx_get "$line" "${key%%:*}")"
			[[ "$got" == "$want" ]] || die 3 "${key%%:*} changed since attempt $(idx_get "$line" n) (recorded $got, now $want); refusing to resume"
		done
	done < <(idx_events "$session" start)

	local n; n="$(next_attempt "$session")"
	local prev ends
	prev="$(idx_get "$(idx_events "$session" start | tail -n 1)" n)"   # last INDEXED attempt
	ends="$(idx_events "$session" end | grep -F -- "\"n\":$prev," || true)"
	if [[ -z "$ends" && "$assume" != 1 ]]; then
		die 3 "attempt $prev has no end record (still running, or the launcher died); use --assume-ended if you are sure it is not running"
	fi
	local agent model; agent="$(idx_get "$last" agent)"; model="$(idx_get "$last" model)"
	tmo="${tmo:-$(idx_get "$last" timeout_s)}"; need_uint "--timeout" "$tmo"
	launch resume "$agent" "$model" "$tmo" "$session" "$CONTINUE_PROMPT" "$n" "$dry"
}

cmd_status() {
	local d
	if [[ $# -eq 0 ]]; then
		for d in "$CCG_EVAL_HOME"/*/; do
			[[ -d "$d/runs" ]] && printf '%s\n' "$(basename -- "$d")"
		done
		return 0
	fi
	POS=("$@")
	if [[ $# -eq 1 ]]; then
		[[ "$1" =~ ^[0-9A-Za-z][0-9A-Za-z._-]*$ ]] || die 2 "bad eval commit '$1'"
		E12="${1:0:12}"; EDIR="$CCG_EVAL_HOME/$E12"; RUNS="$EDIR/runs"; INDEX="$RUNS/index.jsonl"
		T_ROLE=""; T_STRATUM=""
	else
		parse_target
	fi
	[[ -d "$EDIR" ]] || die 4 "no such eval dir: $EDIR"
	if [[ -f "$PIN_FILE" ]]; then
		if pin_verify 2>/dev/null; then printf 'binary: claude %s sha256=%s verified\n' "$(pin_get version)" "$(pin_get sha256)"
		else printf 'binary: PIN does NOT verify\n'; fi
	else
		printf 'binary: no PIN (run prepare-bin)\n'
	fi
	printf 'settings.json sha256=%s\nagents.json   sha256=%s\n' "$(sha256_of "$SETTINGS_FILE")" "$(sha256_of "$AGENTS_FILE")"
	[[ -f "$INDEX" ]] || { printf 'no attempts recorded\n'; return 0; }
	local line role stratum session n end rc drift
	while IFS= read -r line; do
		[[ -n "$line" ]] || continue
		role="$(idx_get "$line" role)"; stratum="$(idx_get "$line" stratum)"
		[[ -z "$T_ROLE" || ( "$role" == "$T_ROLE" && "$stratum" == "$T_STRATUM" ) ]] || continue
		session="$(idx_get "$line" session)"; n="$(idx_get "$line" n)"
		end="$(idx_events "$session" end | grep -F -- "\"n\":$n," | tail -n 1 || true)"
		if [[ -n "$end" ]]; then rc="exit $(idx_get "$end" exit)"; else rc="no end record"; fi
		drift=""
		[[ "$(idx_get "$line" settings_sha256)" == "$(sha256_of "$SETTINGS_FILE")" ]] || drift="$drift settings.json-changed"
		[[ "$(idx_get "$line" agents_sha256)" == "$(sha256_of "$AGENTS_FILE")" ]] || drift="$drift agents.json-changed"
		[[ "$(idx_get "$line" bin_sha256)" == "$(pin_get sha256 2>/dev/null || true)" ]] || drift="$drift binary-changed"
		printf '%s-%s session=%s n=%s %s model=%s cli=%s start=%s%s\n' "$role" "$stratum" "$session" "$n" "$rc" \
			"$(idx_get "$line" model)" "$(idx_get "$line" cli_version)" "$(idx_get "$line" start)" \
			"${drift:+ DRIFT:$drift}"
	done < <(grep -F '"event":"start"' "$INDEX" || true)
}

main() {
	local verb="${1:-}"
	[[ $# -gt 0 ]] && shift
	case "$verb" in
		prepare-bin) cmd_prepare_bin "$@" ;;
		-h|--help|help) usage ;;
		prepare) cmd_prepare "$@" ;;
		start) cmd_start "$@" ;;
		resume) cmd_resume "$@" ;;
		status) cmd_status "$@" ;;
		'') usage >&2; exit 2 ;;
		*) die 2 "unknown verb: $verb (try --help)" ;;
	esac
}

main "$@"
