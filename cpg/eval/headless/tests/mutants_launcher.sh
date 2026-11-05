#!/bin/bash
# mutants_launcher.sh — mutation check for test_launcher.sh. Each mutant is a COPY of run-writer.sh
# with one exact-string edit (the tracked file is never touched); the suite must go red on each.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIG="$HERE/../run-writer.sh"
WORK="$(mktemp -d)"
trap 'rm -rf -- "${WORK:?}"' EXIT

# name<TAB>old<TAB>new   (old must occur in run-writer.sh; "\n" in a field is a newline)
MUTANTS="$WORK/mutants.tsv"
cat >"$MUTANTS" <<'TSV'
argv-no-tools	--tools Read,Grep,Glob,Write,Edit\n		--allowedTools	--allowedTools
argv-no-disable-slash	--strict-mcp-config --disable-slash-commands	--strict-mcp-config
argv-no-restricted	-p --restricted --strict	-p --strict
argv-wrong-tools	--tools Read,Grep,Glob,Write,Edit	--tools Read,Grep,Glob,Write,Edit,Bash
guard-not-passed	 CCG_GUARD="$GUARD_FILE"	
env-not-scrubbed	env -i HOME=	env HOME=
resume-uses-session-id	resume) sflag=--resume ;;	resume) sflag=--session-id ;;
start-uses-resume	start)  sflag=--session-id ;;	start)  sflag=--resume ;;
prompt-continue-changed	continue; your output file lists what is already done	continue
resume-model-not-reused	agent="$(idx_get "$last" agent)"; model="$(idx_get "$last" model)"	agent="$(idx_get "$last" agent)"; model=sonnet
no-orphan-scan-and-no-noclobber	( set -o noclobber; : >"$stream" )	( : >"$stream" )
no-orphan-scan-and-no-noclobber	[[ -e "$f" ]] \|\| continue	continue
no-orphan-scan	[[ -e "$f" ]] \|\| continue	continue
drift-settings-unchecked	settings_sha256:"$SETTINGS_SHA" 	
drift-agents-unchecked	 agents_sha256:"$AGENTS_SHA"; do	; do
drift-binary-unchecked	bin_sha256:"$BIN_SHA" 	
drift-cli-unchecked	cli_version:"$CLI_VERSION" 	
pin-unverified	pin_verify \|\| die 3 "the pinned binary does not verify against PIN; refusing to launch"	true
unknown-session-allowed	[[ -n "$(idx_events "$session" start)" ]] \|\| die 3 "unknown session id $session"	true
no-end-check-disabled	if [[ -z "$ends" && "$assume" != 1 ]]; then	if false; then
second-start-allowed	[[ "$restart" == 1 ]] \|\| die 3 "$T_ROLE	true || die 3 "$T_ROLE
nonempty-out-allowed	if [[ -n "$(find "$ROLE_ROOT/out" -mindepth 1 -print -quit)" ]]; then	if false; then
existing-stream-allowed-on-start	compgen -G "$RUNS/$session.*.stream.jsonl" >/dev/null \&\& die 3	false \&\& die 3
end-record-missing	"$session" "$n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >>"$INDEX"	"$session" "$n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" >/dev/null
index-no-bin-hash	"bin_sha256":"%s"	"bin_x":"%s"
index-no-cli-version	"cli_version":%s	"cli_x":%s
dry-run-launches	if [[ "$dry" == 1 ]]; then	if false; then
timeout-default-changed	D_MODEL=opus;   D_TIMEOUT=14400 ;;	D_MODEL=opus;   D_TIMEOUT=100 ;;
ka-model-changed	D_AGENT=ccg-key-writer-provider; D_MODEL=opus;	D_AGENT=ccg-key-writer-provider; D_MODEL=sonnet;
prepare-overwrites-different	die 3 "role root exists with different content; refusing to overwrite: $ROLE_ROOT"	return 0
prepare-not-idempotent	log "role root already prepared with identical inputs: $ROLE_ROOT — no-op"\n			return 0	die 3 "x"
kit-writable	chmod -R a-w -- "$tmp/kit-src"	true
stream-name-reused	"$RUNS/$session.$n.stream.jsonl"	"$RUNS/$session.1.stream.jsonl"
cwd-not-role-root	( cd "$ROLE_ROOT" \&\& "${ARGV[@]}" )	( "${ARGV[@]}" )
TSV

python3 - "$ORIG" "$MUTANTS" "$WORK" <<'PY'
import sys, re
orig = open(sys.argv[1]).read()
for line in open(sys.argv[2]).read().split("\n"):
    if not line.strip():
        continue
    name, old, new = line.split("\t", 2) if line.count("\t") >= 2 else (line.split("\t") + ["", ""])[:3]
    un = lambda t: t.replace("\\n", "\n").replace("\\|", "|").replace("\\&", "&")
    old, new = un(old), un(new)
    import os
    path = f"{sys.argv[3]}/{name}.sh"
    cur = open(path).read() if os.path.exists(path) else orig
    if cur.count(old) != 1:
        print(f"BROKEN {name}: old string occurs {cur.count(old)} times", file=sys.stderr)
        open(f"{sys.argv[3]}/{name}.broken", "w").close()
        continue
    open(path, "w").write(cur.replace(old, new))
PY

total=0 survived=0
declare -A seen=()
while IFS=$'\t' read -r name _; do
	[[ -n "$name" && -z "${seen[$name]:-}" ]] || continue
	seen[$name]=1
	total=$((total + 1))
	mut="$WORK/$name.sh"
	if [[ -e "$WORK/$name.broken" ]]; then printf 'BROKEN  %-34s edit did not apply\n' "$name"; survived=$((survived + 1)); continue; fi
	if ! bash -n "$mut"; then printf 'BROKEN  %-34s mutant does not parse\n' "$name"; survived=$((survived + 1)); continue; fi
	if RUN_WRITER="$mut" bash "$HERE/test_launcher.sh" >"$WORK/$name.log" 2>&1; then
		printf 'SURVIVED %-34s\n' "$name"; survived=$((survived + 1))
	else
		printf 'killed  %-34s %s\n' "$name" "$(grep -m2 '^FAIL' "$WORK/$name.log" | tr '\n' ';' | cut -c1-110)"
	fi
done <"$MUTANTS"
printf '\n%d of %d mutants survived\n' "$survived" "$total"
[[ "$survived" -eq 0 ]]
