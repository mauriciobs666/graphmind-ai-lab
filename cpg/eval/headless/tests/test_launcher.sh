#!/bin/bash
# test_launcher.sh — S53 prepare/start/resume/status tests
# (docs/plans/connected-code-graph-p0-2.md §3 S53 (2)-(5), §9.1). No claude process that calls the
# model is ever started: the pinned "binary" is a stub shell script under a temp HOME, and the
# repo-side files (settings.json, agents.json, guard-paths.sh) are copied into a temp dir so drift
# tests can edit them.   RUN_WRITER=<path> overrides the script under test (used by mutants_launcher.sh).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$HERE/.." && pwd)"
RUN_WRITER="${RUN_WRITER:-$SRC_DIR/run-writer.sh}"
TMP_ROOT="$(mktemp -d)"
trap 'chmod -R u+w "${TMP_ROOT:?}" 2>/dev/null; rm -rf -- "${TMP_ROOT:?}"' EXIT

pass=0 fail=0
ok()   { pass=$((pass + 1)); printf 'ok   %s\n' "$1"; }
bad()  { fail=$((fail + 1)); printf 'FAIL %s\n' "$1"; }
check() { local name="$1"; shift; if "$@"; then ok "$name"; else bad "$name"; fi; }
has()  { grep -qF -- "$1" <<<"$2"; }
hasnt() { ! grep -qF -- "$1" <<<"$2"; }

# --- fixture: HOME with a stub claude pinned by prepare-bin, and a private copy of headless/ ---
H="$TMP_ROOT/home"; HL="$TMP_ROOT/hl"; EV=0123456789abcdef0123
mkdir -p "$H/.local/share/claude/versions" "$H/.local/bin" "$HL"
cat >"$H/.local/share/claude/versions/2.1.312" <<'STUB'
#!/bin/sh
{ echo "PWD=$(pwd)"; echo "STDIN=$(readlink /proc/$$/fd/0)"; env | sort; for a in "$@"; do echo "ARG:$a"; done; } >"$HOME/stub.log"
echo '{"type":"result","stub":true}'
[ -f "$HOME/stub.sleep" ] && exec sleep "$(cat "$HOME/stub.sleep")"
exit "$(cat "$HOME/stub.rc" 2>/dev/null || echo 0)"
STUB
chmod 0755 "$H/.local/share/claude/versions/2.1.312"
ln -s "$H/.local/share/claude/versions/2.1.312" "$H/.local/bin/claude"
cp "$RUN_WRITER" "$HL/run-writer.sh"
cp "$SRC_DIR/settings.json" "$SRC_DIR/agents.json" "$SRC_DIR/guard-paths.sh" "$HL/"
chmod +x "$HL/guard-paths.sh"
EVH="$H/.local/share/ccg-eval"; E12="${EV:0:12}"; ED="$EVH/$E12"; RUNS="$ED/runs"; IDX="$RUNS/index.jsonl"

rw() { OUT="$(env -i HOME="$H" PATH=/usr/bin:/bin LANG=C.UTF-8 CANARY=leak bash "$HL/run-writer.sh" "$@" 2>&1 <<<"pending stdin")"; RC=$?; }
rwo() { OUT="$(env -i HOME="$H" PATH=/usr/bin:/bin LANG=C.UTF-8 bash "$HL/run-writer.sh" "$@" 2>/dev/null)"; RC=$?; }

rw prepare-bin; check "fixture: prepare-bin ok" test "$RC" -eq 0
BIN="$EVH/bin/claude-2.1.312"; BIN_SHA="$(sha256sum "$BIN" | cut -d' ' -f1)"
SET_SHA="$(sha256sum "$HL/settings.json" | cut -d' ' -f1)"; AG_SHA="$(sha256sum "$HL/agents.json" | cut -d' ' -f1)"

mkdir -p "$ED/kit-src/sub"; echo "hello" >"$ED/kit-src/README.md"; echo "x" >"$ED/kit-src/sub/f.txt"
printf '# Brief\nread kit-src/README.md, write out/done.txt\nline "three" $x\n' >"$TMP_ROOT/brief.md"
U1=11111111-1111-4111-8111-111111111111
CPROMPT='continue; your output file lists what is already done'

# --- 1. prepare -----------------------------------------------------------------------------
rw prepare "$EV" KA flow --brief "$TMP_ROOT/brief.md"
RR="$ED/roles/KA-flow"
check "prepare: exit 0"                    test "$RC" -eq 0
check "prepare: BRIEF.md identical"        cmp -s "$TMP_ROOT/brief.md" "$RR/BRIEF.md"
check "prepare: BRIEF.md read-only"        test ! -w "$RR/BRIEF.md"
check "prepare: kit-src copied"            diff -rq "$ED/kit-src" "$RR/kit-src"
check "prepare: kit-src read-only"         test -z "$(find "$RR/kit-src" -perm /222 -print -quit)"
check "prepare: out/ empty and writable"   test -d "$RR/out" -a -w "$RR/out" -a -z "$(ls -A "$RR/out")"
check "prepare: runs/ and work/ exist"     test -d "$RUNS" -a -d "$ED/work"
check "prepare: no temp dir left"          test -z "$(find "$ED/roles" -maxdepth 1 -name '.prepare.*')"
STAMP="$(stat -c %i:%Y "$RR/BRIEF.md")"
rw prepare "$EV" KA flow --brief "$TMP_ROOT/brief.md"
check "prepare rerun: exit 0"              test "$RC" -eq 0
check "prepare rerun: says no-op"          has "no-op" "$OUT"
check "prepare rerun: nothing rewritten"   test "$(stat -c %i:%Y "$RR/BRIEF.md")" = "$STAMP"
echo other >"$TMP_ROOT/brief2.md"
rw prepare "$EV" KA flow --brief "$TMP_ROOT/brief2.md"
check "prepare: different brief refused (3)" test "$RC" -eq 3
check "prepare: refused brief untouched"   cmp -s "$TMP_ROOT/brief.md" "$RR/BRIEF.md"
echo more >"$ED/kit-src/new.txt"
rw prepare "$EV" KA flow --brief "$TMP_ROOT/brief.md"
check "prepare: different kit refused (3)" test "$RC" -eq 3
rm -f -- "${ED:?}/kit-src/new.txt"
rw prepare "$EV" KA flow;                                    check "prepare: no --brief is usage (2)" test "$RC" -eq 2
rw prepare "$EV" KA flow --brief "$TMP_ROOT/none.md";        check "prepare: missing brief (4)" test "$RC" -eq 4
rw prepare "$EV" ZZ flow --brief "$TMP_ROOT/brief.md";       check "prepare: bad role is usage (2)" test "$RC" -eq 2
rw prepare "$EV" KA 'Bad/..' --brief "$TMP_ROOT/brief.md";   check "prepare: bad stratum is usage (2)" test "$RC" -eq 2
rw prepare "$EV" KB flow --brief "$TMP_ROOT/brief.md" --kit-src "$TMP_ROOT/nokit"
check "prepare: missing kit-src (4)"       test "$RC" -eq 4
rw prepare "$EV" KB flow --brief "$TMP_ROOT/brief.md"
rw prepare "$EV" QW flow --brief "$TMP_ROOT/brief.md"
rw prepare "$EV" AN2 flow --brief "$TMP_ROOT/brief.md"

# --- 2. dry runs: one builder -----------------------------------------------------------------
rwo start --dry-run --session-id $U1 "$EV" KA flow; DS="$OUT"; DS_RC=$RC
check "dry start: exit 0"                  test "$DS_RC" -eq 0
check "dry start: nothing written"         test ! -e "$IDX" -a -z "$(ls -A "$RUNS")"
check "dry start: --session-id U"          grep -qx -- "  --session-id" <<<"$DS"
check "dry start: session id printed"      has "$U1" "$DS"
check "dry start: no --resume"             hasnt "--resume" "$DS"
check "dry: --tools line"                  grep -qx -- "  --tools" <<<"$DS"
check "dry: tools value"                   grep -qx -- "  Read,Grep,Glob,Write,Edit" <<<"$DS"
check "dry: --disable-slash-commands"      grep -qx -- "  --disable-slash-commands" <<<"$DS"
for f in --restricted --strict-mcp-config --verbose -p --allowedTools --disallowedTools stream-json dontAsk --permission-prompts --settings --agents; do
	check "dry: has $f"                    grep -qx -- "  $f" <<<"$DS"
done
check "dry: env -i first"                  test "$(grep -m3 . <<<"$DS" | tail -1)" = "  env"
check "dry: -i flag"                       grep -qx -- "  -i" <<<"$DS"
check "dry: CCG_GUARD passed"              grep -qxF "  CCG_GUARD=$HL/guard-paths.sh" <<<"$DS"
check "dry: PATH minimal"                  grep -qx "  PATH=/usr/bin:/bin" <<<"$DS"
check "dry: binary from PIN"               grep -qxF "  $BIN" <<<"$DS"
check "dry: settings path"                 grep -qxF "  $HL/settings.json" <<<"$DS"
check "dry: agents path"                   grep -qxF "  $HL/agents.json" <<<"$DS"
check "dry: KA agent"                      grep -qx "  ccg-key-writer-provider" <<<"$DS"
check "dry: KA model opus"                 test "$(grep -A1 -x -- '  --model' <<<"$DS" | tail -1)" = "  opus"
check "dry: KA timeout 14400"              grep -qx "  14400" <<<"$DS"
check "dry: stream name n=1"               grep -qxF "  > $RUNS/$U1.1.stream.jsonl" <<<"$DS"
check "dry: prompt is BRIEF.md"            grep -qxF "  $(printf '%q' "$(<"$TMP_ROOT/brief.md")")" <<<"$DS"
rwo start --dry-run "$EV" KB flow
check "dry: KB agent"                      grep -qx "  ccg-key-writer-consumer" <<<"$OUT"
check "dry: KB model sonnet"               test "$(grep -A1 -x -- '  --model' <<<"$OUT" | tail -1)" = "  sonnet"
rwo start --dry-run "$EV" AN2 flow
check "dry: AN2 agent"                     grep -qx "  ccg-annotator" <<<"$OUT"
check "dry: AN2 opus"                      test "$(grep -A1 -x -- '  --model' <<<"$OUT" | tail -1)" = "  opus"
check "dry: AN2 timeout 3600"              grep -qx "  3600" <<<"$OUT"
rwo start --dry-run "$EV" QW flow;  check "dry: QW timeout 7200" grep -qx "  7200" <<<"$OUT"
rwo start --dry-run --model haiku --timeout 99 "$EV" KB flow
check "dry: model override"                grep -qx "  haiku" <<<"$OUT"
check "dry: timeout override"              grep -qx "  99" <<<"$OUT"

# --- 3. real start via the stub ---------------------------------------------------------------
rw start --session-id $U1 "$EV" KA flow
S1="$RUNS/$U1.1.stream.jsonl"
check "start: exit 0"                      test "$RC" -eq 0
check "start: stream log written"          grep -q '"stub":true' "$S1"
check "start: stub cwd is the role root"   grep -qx "PWD=$RR" "$H/stub.log"
check "start: stub got --session-id U"     test "$(grep -A1 -x 'ARG:--session-id' "$H/stub.log" | tail -1)" = "ARG:$U1"
check "start: stub got the brief"          grep -qxF 'ARG:# Brief' "$H/stub.log"
check "start: env is scrubbed (canary)"    bash -c "! grep -q CANARY '$H/stub.log'"
check "start: env has CCG_GUARD"           grep -qx "CCG_GUARD=$HL/guard-paths.sh" "$H/stub.log"
check "start: env HOME kept"               grep -qx "HOME=$H" "$H/stub.log"
check "start: index has start record"      grep -qF "\"event\":\"start\"" "$IDX"
check "start: index cli_version"           grep -qF '"cli_version":"2.1.312"' "$IDX"
check "start: index binary sha"            grep -qF "\"bin_sha256\":\"$BIN_SHA\"" "$IDX"
check "start: index settings sha"          grep -qF "\"settings_sha256\":\"$SET_SHA\"" "$IDX"
check "start: index agents sha"            grep -qF "\"agents_sha256\":\"$AG_SHA\"" "$IDX"
check "start: index n=1 + role + stratum"  grep -qF "\"role\":\"KA\",\"stratum\":\"flow\"" "$IDX"
check "start: index end record exit 0"     grep -qE '"event":"end","session":"'$U1'","n":1,.*"exit":0' "$IDX"
rw start --session-id $U1 "$EV" KA flow
check "start: re-start refused (3)"        test "$RC" -eq 3
rw start "$EV" KA flow;  check "start: second start refused (3)" test "$RC" -eq 3

# --- 4. dry resume differs only in session flag, prompt, stream name ---------------------------
rwo resume --dry-run "$EV" KA flow; DR="$OUT"
check "dry resume: exit 0"                 test "$RC" -eq 0
check "dry resume: --resume U"             test "$(grep -A1 -x -- '  --resume' <<<"$DR" | tail -1)" = "  $U1"
check "dry resume: no --session-id"        hasnt "--session-id" "$DR"
check "dry resume: stream n=2"             grep -qxF "  > $RUNS/$U1.2.stream.jsonl" <<<"$DR"
check "dry resume: fixed prompt"           grep -qxF "  $(printf '%q' "$CPROMPT")" <<<"$DR"
check "dry resume: agent reused"           grep -qx "  ccg-key-writer-provider" <<<"$DR"
check "dry resume: model reused"           test "$(grep -A1 -x -- '  --model' <<<"$DR" | tail -1)" = "  opus"
# the dry output of `start` for the same session, compared line by line with resume's
diff <(grep -v '^#' <<<"$DS") <(grep -v '^#' <<<"$DR") >"$TMP_ROOT/dd.txt"
check "dry diff: exactly 3 lines differ each side" test "$(grep -c '^<' "$TMP_ROOT/dd.txt")" -eq 3 -a "$(grep -c '^>' "$TMP_ROOT/dd.txt")" -eq 3
want_start="$(printf '%s\n' '--session-id' "> $RUNS/$U1.1.stream.jsonl" "$(printf '%q' "$(<"$TMP_ROOT/brief.md")")" | sort)"
want_res="$(printf '%s\n' '--resume' "> $RUNS/$U1.2.stream.jsonl" "$(printf '%q' "$CPROMPT")" | sort)"
check "dry diff: start side = flag, prompt, stream"  test "$(grep '^<' "$TMP_ROOT/dd.txt" | sed 's/^< *//' | sort)" = "$want_start"
check "dry diff: resume side = flag, prompt, stream" test "$(grep '^>' "$TMP_ROOT/dd.txt" | sed 's/^> *//' | sort)" = "$want_res"
check "dry resume: nothing written"        test ! -e "$RUNS/$U1.2.stream.jsonl"

# --- 5. real resumes: new file per attempt, earlier logs untouched ----------------------------
S1_SUM="$(sha256sum "$S1" | cut -d' ' -f1)"
echo 7 >"$H/stub.rc"
rw resume "$EV" KA flow
S2="$RUNS/$U1.2.stream.jsonl"
check "resume: exit code is the stub's (7)" test "$RC" -eq 7
check "resume: attempt 2 log exists"       test -s "$S2"
check "resume: attempt 1 log untouched"    test "$(sha256sum "$S1" | cut -d' ' -f1)" = "$S1_SUM"
check "resume: stub got --resume U"        test "$(grep -A1 -x 'ARG:--resume' "$H/stub.log" | tail -1)" = "ARG:$U1"
check "resume: stub got no --session-id"   bash -c "! grep -qx 'ARG:--session-id' '$H/stub.log'"
check "resume: stub got --restricted"      grep -qx 'ARG:--restricted' "$H/stub.log"
check "resume: stub got --disable-slash-commands" grep -qx 'ARG:--disable-slash-commands' "$H/stub.log"
check "resume: stub got --tools value"     grep -qx 'ARG:Read,Grep,Glob,Write,Edit' "$H/stub.log"
check "resume: stub got continuation prompt" grep -qxF "ARG:$CPROMPT" "$H/stub.log"
check "resume: index n=2 start recorded"   test "$(grep '"event":"start"' "$IDX" | grep -cF '"session":"'$U1'","n":2,')" -eq 1
check "resume: index n=2 end exit 7"       grep -qE '"event":"end","session":"'$U1'","n":2,.*"exit":7' "$IDX"
check "resume: attempt 2 records the bin hash" bash -c "grep '\"event\":\"start\"' '$IDX' | grep -F '\"n\":2,' | grep -qF '\"bin_sha256\":\"$BIN_SHA\"'"
# an orphan log for n=3 (launcher died before indexing it) must be skipped, never written to
echo orphan >"$RUNS/$U1.3.stream.jsonl"
rw resume "$EV" KA flow
check "resume: skips an orphan log (n=4)"  test -s "$RUNS/$U1.4.stream.jsonl"
check "resume: orphan log untouched"       test "$(cat "$RUNS/$U1.3.stream.jsonl")" = orphan
rm -f -- "${H:?}/stub.rc"
# timeout: the stub sleeps 5 s, --timeout 1
echo 5 >"$H/stub.sleep"
rw resume --timeout 1 "$EV" KA flow
check "timeout: exit 124 recorded"         grep -qE '"event":"end","session":"'$U1'","n":5,.*"exit":124' "$IDX"
rm -f -- "${H:?}/stub.sleep"

# --- 6. resume refusals ------------------------------------------------------------------------
N_BEFORE="$(ls "$RUNS" | wc -l)"
no_new() { test "$(ls "$RUNS" | wc -l)" -eq "$N_BEFORE"; }
cp "$HL/settings.json" "$TMP_ROOT/settings.bak"; echo '{"x":1}' >>"$HL/settings.json"
rw resume "$EV" KA flow
check "refuse: settings.json changed (3)"  test "$RC" -eq 3
check "refuse: settings message"           has "settings_sha256 changed" "$OUT"
check "refuse: no new log (settings)"      no_new
rwo resume --dry-run "$EV" KA flow;        check "refuse: dry-run refuses too" test "$RC" -eq 3
rw status "$EV" KA flow;                   check "status: shows DRIFT"  has "settings.json-changed" "$OUT"
cp "$TMP_ROOT/settings.bak" "$HL/settings.json"
cp "$HL/agents.json" "$TMP_ROOT/agents.bak"; echo ' ' >>"$HL/agents.json"
rw resume "$EV" KA flow
check "refuse: agents.json changed (3)"    test "$RC" -eq 3
check "refuse: agents message"             has "agents_sha256 changed" "$OUT"
check "refuse: no new log (agents)"        no_new
cp "$TMP_ROOT/agents.bak" "$HL/agents.json"
chmod u+w "$BIN"; cp "$BIN" "$TMP_ROOT/bin.bak"; echo '# tamper' >>"$BIN"; chmod 0555 "$BIN"
rw resume "$EV" KA flow
check "refuse: binary changed (3)"         test "$RC" -eq 3
check "refuse: no new log (binary)"        no_new
rw start --dry-run --session-id 22222222-2222-4222-8222-222222222222 "$EV" KB flow
check "refuse: start refuses a changed binary (3)" test "$RC" -eq 3
chmod u+w "$BIN"; cp "$TMP_ROOT/bin.bak" "$BIN"; chmod 0555 "$BIN"
# a deliberate re-pin (binary AND PIN replaced consistently): PIN verifies, but the attempts' hash differs
chmod u+w "$BIN" "$EVH/bin/PIN"; cp "$EVH/bin/PIN" "$TMP_ROOT/PIN.bak0"; echo '# other build' >>"$BIN"; chmod 0555 "$BIN"
sed -i "s/^sha256=.*/sha256=$(sha256sum "$BIN" | cut -d' ' -f1)/" "$EVH/bin/PIN"
rw resume "$EV" KA flow
check "refuse: re-pinned binary (3)"       test "$RC" -eq 3
check "refuse: re-pin message"             has "bin_sha256 changed" "$OUT"
check "refuse: no new log (re-pin)"        no_new
chmod u+w "$BIN"; cp "$TMP_ROOT/bin.bak" "$BIN"; chmod 0555 "$BIN"; cp "$TMP_ROOT/PIN.bak0" "$EVH/bin/PIN"; chmod 0444 "$EVH/bin/PIN"
# a changed CLI version (PIN version differs from the attempts' recorded one)
cp "$EVH/bin/PIN" "$TMP_ROOT/PIN.bak"; chmod u+w "$EVH/bin/PIN"; sed -i 's/^version=.*/version=9.9.9/' "$EVH/bin/PIN"
rw resume "$EV" KA flow
check "refuse: cli version changed (3)"    test "$RC" -eq 3
check "refuse: cli version message"        has "cli_version changed" "$OUT"
cp "$TMP_ROOT/PIN.bak" "$EVH/bin/PIN"; chmod 0444 "$EVH/bin/PIN"
rw resume --session-id 33333333-3333-4333-8333-333333333333 "$EV" KA flow
check "refuse: unknown session id (3)"     test "$RC" -eq 3
check "refuse: unknown session message"    has "unknown session" "$OUT"
rw resume "$EV" QW flow
check "refuse: role with no session (3)"   test "$RC" -eq 3
check "refuse: no new log (unknown)"       no_new
rw resume --dry-run "$EV" KA flow
check "restored: resume works again"       test "$RC" -eq 0
# an attempt with no end record
printf '{"event":"start","role":"QW","stratum":"flow","eval":"%s","session":"44444444-4444-4444-8444-444444444444","n":1,"mode":"start","agent":"ccg-question-writer","model":"opus","timeout_s":7200,"cli_version":"2.1.312","bin_sha256":"%s","settings_sha256":"%s","agents_sha256":"%s","stream":"x","start":"t"}\n' "$E12" "$BIN_SHA" "$SET_SHA" "$AG_SHA" >>"$IDX"
rw resume --dry-run "$EV" QW flow
check "refuse: previous attempt has no end (3)" test "$RC" -eq 3
check "refuse: no-end message"             has "no end record" "$OUT"
rw resume --dry-run --assume-ended "$EV" QW flow
check "assume-ended: allowed"              test "$RC" -eq 0

# --- 7. start refusals and --restart -----------------------------------------------------------
U2=55555555-5555-4555-8555-555555555555
echo keep >"$ED/roles/KB-flow/out/KB.jsonl"
rw start --session-id $U2 "$EV" KB flow
check "start: non-empty out/ refused (3)"  test "$RC" -eq 3
check "start: out/ left alone"             test "$(cat "$ED/roles/KB-flow/out/KB.jsonl")" = keep
rm -f -- "${ED:?}/roles/KB-flow/out/KB.jsonl"
echo pre >"$RUNS/$U2.1.stream.jsonl"
rw start --session-id $U2 "$EV" KB flow
check "start: existing stream log refused (3)" test "$RC" -eq 3
check "start: existing log untouched"      test "$(cat "$RUNS/$U2.1.stream.jsonl")" = pre
rm -f -- "${RUNS:?}/$U2.1.stream.jsonl"
rw start --session-id $U2 "$EV" KB flow; check "start: KB ok" test "$RC" -eq 0
rw start --restart --session-id 66666666-6666-4666-8666-666666666666 "$EV" KB flow
check "restart: new session on empty out/" test "$RC" -eq 0
check "restart: new session is current"    test "$(grep '"role":"KB"' "$IDX" | grep '"event":"start"' | tail -1 | grep -c 66666666-6666)" -eq 1
rw start --session-id not-a-uuid "$EV" AN2 flow; check "start: bad uuid is usage (2)" test "$RC" -eq 2
rw start "$EV" KA nostratum;               check "start: unprepared role root (4)" test "$RC" -eq 4
rw bogus;                                  check "unknown verb (2)" test "$RC" -eq 2

# --- 7b. lock, signals, size, stdin, symlinks, torn index line ------------------------------------
check "stdin: launch gets /dev/null"       grep -qx 'STDIN=/dev/null' "$H/stub.log"
rw prepare "$EV" AN1 flow --brief "$TMP_ROOT/brief.md"
echo 2 >"$H/stub.sleep"
( env -i HOME="$H" PATH=/usr/bin:/bin LANG=C.UTF-8 bash "$HL/run-writer.sh" start "$EV" AN1 flow >"$TMP_ROOT/p1.out" 2>&1; echo $? >"$TMP_ROOT/p1.rc" ) &
( env -i HOME="$H" PATH=/usr/bin:/bin LANG=C.UTF-8 bash "$HL/run-writer.sh" start "$EV" AN1 flow >"$TMP_ROOT/p2.out" 2>&1; echo $? >"$TMP_ROOT/p2.rc" ) &
wait
rm -f -- "${H:?}/stub.sleep"
check "lock: parallel starts, exactly one start line" test "$(grep '"event":"start"' "$IDX" | grep -c '"role":"AN1"')" -eq 1
check "lock: parallel starts, exit codes are {0,3}"   test "$(sort "$TMP_ROOT/p1.rc" "$TMP_ROOT/p2.rc" | tr '\n' ' ')" = "0 3 "
check "lock: released after exit"          test -z "$(find "$RUNS" -maxdepth 1 -name '.lock-*' -print -quit)"
# a live lock refuses, a stale lock (dead pid) is taken over
mkdir "$RUNS/.lock-AN1-flow"; echo $$ >"$RUNS/.lock-AN1-flow/pid"
rw resume --assume-ended "$EV" AN1 flow
check "lock: live lock refuses resume (3)" test "$RC" -eq 3
check "lock: live lock message"            has "locked by another launcher" "$OUT"
echo 999999 >"$RUNS/.lock-AN1-flow/pid"
rw resume "$EV" AN1 flow
check "lock: stale lock taken over"        test "$RC" -eq 0
check "lock: stale lock released after"    test ! -e "$RUNS/.lock-AN1-flow"
# SIGTERM to the launcher stops the child and still writes an end record
echo 37 >"$H/stub.sleep"
env -i HOME="$H" PATH=/usr/bin:/bin LANG=C.UTF-8 bash "$HL/run-writer.sh" resume "$EV" AN1 flow >"$TMP_ROOT/sig.out" 2>&1 &
LP=$!
for _ in $(seq 1 50); do grep -q 'n":3,.*"mode":"resume"' "$IDX" && break; sleep 0.1; done
sleep 0.5; kill -TERM "$LP"; wait "$LP"; SIG_RC=$?
rm -f -- "${H:?}/stub.sleep"
check "signal: launcher exits non-zero"    test "$SIG_RC" -ne 0
check "signal: end record written"         grep -qE '"event":"end","session":"[^"]*","n":3,.*"exit":(143|124|[0-9]+)' "$IDX"
for _ in $(seq 1 50); do pgrep -f '[s]leep 37' >/dev/null || break; sleep 0.1; done
check "signal: child gone"                 bash -c "! pgrep -f '[s]leep 37' >/dev/null"
check "signal: lock released"              test ! -e "$RUNS/.lock-AN1-flow"
# oversized prompt: refused before any session id / stream / index line is written
head -c 100001 /dev/zero | tr '\0' 'a' >"$TMP_ROOT/big.md"
rw prepare "$EV" QW big --brief "$TMP_ROOT/big.md"
check "size: prepare rejects a big brief (2)" test "$RC" -eq 2
rw prepare "$EV" QW small --brief "$TMP_ROOT/brief.md"
chmod u+w "$ED/roles/QW-small/BRIEF.md"; cp "$TMP_ROOT/big.md" "$ED/roles/QW-small/BRIEF.md"
N_IDX="$(wc -l <"$IDX")"; N_RUNS="$(ls "$RUNS" | wc -l)"
rw start "$EV" QW small
check "size: start rejects a big brief (2)" test "$RC" -eq 2
check "size: nothing written"              test "$(wc -l <"$IDX")" -eq "$N_IDX" -a "$(ls "$RUNS" | wc -l)" -eq "$N_RUNS"
# a symlink in the kit is refused
mkdir -p "$TMP_ROOT/kitln"; ln -s /etc/passwd "$TMP_ROOT/kitln/pw"
rw prepare "$EV" QW link --brief "$TMP_ROOT/brief.md" --kit-src "$TMP_ROOT/kitln"
check "prepare: kit symlink refused (3)"   test "$RC" -eq 3
check "prepare: no role root for it"       test ! -e "$ED/roles/QW-link"
# a torn last index line must not swallow the next record
printf '{"event":"sta' >>"$IDX"
rw resume "$EV" KA flow
check "torn index: resume ok"              test "$RC" -eq 0
check "torn index: every record is whole"  test "$(grep -vc '^{"event":".*}$' "$IDX")" -eq 1
check "torn index: new start line intact"  grep -qE '^\{"event":"start".*"session":"'$U1'","n":6,' "$IDX"

# --- 8. status ----------------------------------------------------------------------------------
rw status; check "status: no args lists eval dirs" has "$E12" "$OUT"
rw status "$EV"
check "status E: lists sessions"           has "KA-flow session=$U1 n=5 exit 124" "$OUT"
check "status E: binary verified"          has "binary: claude 2.1.312" "$OUT"
check "status: no drift when unchanged"    hasnt DRIFT "$OUT"
rw status "$EV" KB flow
check "status E role: filters"             bash -c "! grep -q 'KA-flow' <<<'$OUT' && grep -q KB-flow <<<'$OUT'"
check "bash -n run-writer.sh"              bash -n "$HL/run-writer.sh"

printf '\n%d passed, %d failed\n' "$pass" "$fail"
[[ "$fail" -eq 0 ]]
