#!/usr/bin/env bash
# model-preflight.sh (test) — offline tests for scripts/model-preflight.sh and
# its wiring into health-check.sh / bring-up.sh / tear-down.sh
# (docs/plans/standard-local-model.md §3.3, §6.1). Runs copies of the real
# scripts in a fixture tree; stub `curl` prints fixture JSON (or exits 7) and
# logs argv, stub `opencode` logs argv. No LM Studio, no docker.
#
# Usage: ./model-preflight.sh
set -uo pipefail

REAL_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts" && pwd)"

PASS=0
FAIL=0
FAILED_NAMES=()
if [ -t 1 ]; then R=$'\033[31m'; G=$'\033[32m'; Z=$'\033[0m'; else R=; G=; Z=; fi

FIXTURE_ROOT="$(mktemp -d)"
FIX_TANK="$FIXTURE_ROOT/opencode/agents/tank"
FIX_SCRIPTS="$FIX_TANK/scripts"
mkdir -p "$FIX_SCRIPTS" "$FIX_TANK/state"
cp "$REAL_SCRIPTS_DIR"/*.sh "$FIX_SCRIPTS/"

FAKE_BIN="$(mktemp -d)"
CURL_LOG="$FIXTURE_ROOT/curl.log"
CURL_BODY="$FIXTURE_ROOT/curl.body"
OC_LOG="$FIXTURE_ROOT/opencode.log"
cat >"$FAKE_BIN/curl" <<EOF2
#!/usr/bin/env bash
printf '%s\n' "\$*" >> "$CURL_LOG"
if [ -n "\${CURL_EXIT:-}" ]; then exit "\$CURL_EXIT"; fi
cat "$CURL_BODY"
EOF2
cat >"$FAKE_BIN/opencode" <<EOF2
#!/usr/bin/env bash
printf '%s\n' "\$@" >> "$OC_LOG"
exit 0
EOF2
chmod +x "$FAKE_BIN/curl" "$FAKE_BIN/opencode"
trap 'rm -rf "$FIXTURE_ROOT" "$FAKE_BIN"' EXIT

# Distinct values everywhere: id, port and contexts differ from each other.
ID="acme/model-x"
OTHER="zeta/other-y"
BASE_URL="http://127.0.0.1:4321"

write_config() {  # write_config MODEL BASEURL
  cat >"$FIX_TANK/opencode.json" <<JSON
{
  "provider": {"lmstudio": {"options": {"baseURL": "$2"}}},
  "agent": {"tank": {"model": "$1"}}
}
JSON
}

# inst ID CTX PAR  -> JSON object ("-" for PAR omits config.parallel)
inst() {
  if [ "$3" = "-" ]; then
    printf '{"id":"%s","config":{"context_length":%s}}' "$1" "$2"
  else
    printf '{"id":"%s","config":{"context_length":%s,"parallel":%s}}' "$1" "$2" "$3"
  fi
}
# entry KEY TYPE INSTANCES...
entry() {
  local key="$1" type="$2" first=1 out=""
  shift 2
  for i in "$@"; do
    if [ $first -eq 1 ]; then out="$i"; first=0; else out="$out,$i"; fi
  done
  printf '{"key":"%s","type":"%s","loaded_instances":[%s]}' "$key" "$type" "$out"
}
# set_models ENTRIES...
set_models() {
  local first=1 out=""
  for e in "$@"; do
    if [ $first -eq 1 ]; then out="$e"; first=0; else out="$out,$e"; fi
  done
  printf '{"models":[%s]}' "$out" >"$CURL_BODY"
}

reset() {
  : >"$CURL_LOG"; : >"$OC_LOG"
  unset CURL_EXIT
  write_config "lmstudio/$ID" "$BASE_URL/v1"
  set_models "$(entry "$ID" llm)"
}

run() {  # run SCRIPT ARG... -> STDOUT STDERR EXIT_CODE (env passes through)
  local script="$1"; shift
  local o e
  o="$(mktemp)"; e="$(mktemp)"
  PATH="$FAKE_BIN:$PATH" "$FIX_SCRIPTS/$script" "$@" >"$o" 2>"$e"
  EXIT_CODE=$?
  STDOUT="$(cat "$o")"; STDERR="$(cat "$e")"
  rm -f "$o" "$e"
}

pass() { PASS=$((PASS + 1)); printf '%sPASS%s %s\n' "$G" "$Z" "$1"; }
fail() { FAIL=$((FAIL + 1)); FAILED_NAMES+=("$1"); printf '%sFAIL%s %s\n  %s\n' "$R" "$Z" "$1" "$2"; }

# check NAME EXPECTED_EXIT SUBSTR...   (substrings must all appear in stdout+stderr)
check() {
  local name="$1" want="$2"; shift 2
  if [ "$EXIT_CODE" -ne "$want" ]; then
    fail "$name" "expected exit $want, got $EXIT_CODE (stderr: $STDERR / stdout: $STDOUT)"; return
  fi
  local s all="$STDOUT$STDERR"
  for s in "$@"; do
    case "$all" in *"$s"*) ;; *) fail "$name" "missing '$s' in: $all"; return ;; esac
  done
  pass "$name"
}
check_absent() {  # check_absent NAME SUBSTR  (must not appear in output)
  case "$STDOUT$STDERR" in *"$2"*) fail "$1" "unexpected '$2' in: $STDOUT$STDERR" ;; *) pass "$1" ;; esac
}

# --- 1 unreachable ----------------------------------------------------------
reset; CURL_EXIT=7 run model-preflight.sh
check "1 unreachable: exit 1, names the URL root" 1 "$BASE_URL" "not reachable"

# --- 2 not the models JSON ---------------------------------------------------
reset; echo '<html>nope</html>' >"$CURL_BODY"; run model-preflight.sh
check "2a non-JSON answer: exit 1" 1 "not reachable"
reset; echo '{"data":[]}' >"$CURL_BODY"; run model-preflight.sh
check "2b JSON without models[]: exit 1" 1 "not reachable"

# --- 3 unknown key -----------------------------------------------------------
reset; set_models "$(entry "$OTHER" llm "$(inst "$OTHER" 32768 1)")"; run model-preflight.sh
check "3 no entry with the key: exit 1, names id" 1 "$ID" "not in LM Studio"

# --- 4 not loaded ------------------------------------------------------------
reset; run model-preflight.sh
check "4 not loaded: exit 0 NOTE, minimum named" 0 "NOTE:" "$ID not loaded" "20480" "Prerequisites"
check_absent "4b NOTE does not claim a per-model default" "per-model default"

# --- 5 context ---------------------------------------------------------------
reset; set_models "$(entry "$ID" llm "$(inst "$ID" 20480 1)")"; run model-preflight.sh
check "5a context exactly the minimum: exit 0 OK" 0 "OK: $ID loaded, context 20480, parallel 1"
check_absent "5a2 no NOTE when only the own copy is loaded" "NOTE:"
reset; set_models "$(entry "$ID" llm "$(inst "$ID" 20479 1)")"; run model-preflight.sh
check "5b context minimum-1: exit 1" 1 "REFUSED" "20479"
reset; set_models "$(entry "$ID" llm "$(inst "$ID" 8192 1)")"; run model-preflight.sh
check "5c context 8192: exit 1, both numbers and lms unload" 1 "REFUSED: $ID loaded at 8192, tank needs" "20480" "lms unload $ID"
reset; set_models "$(entry "$ID" llm "$(inst "$ID" 32768 1)")"; run model-preflight.sh
check "5d context 32768: exit 0 OK" 0 "OK: $ID loaded, context 32768, parallel 1"

# --- 6 copies ----------------------------------------------------------------
reset; set_models "$(entry "$ID" llm "$(inst "$ID" 32768 1)" "$(inst "$ID:2" 24000 1)")"; run model-preflight.sh
check "6a two instances: exit 1, both ids named" 1 "REFUSED: 2 copies of $ID" "$ID:2" "lms unload"
reset; set_models "$(entry "$ID" llm "$(inst "$ID:2" 32768 1)")"; run model-preflight.sh
check "6b single instance with id <id>:2: exit 1" 1 "REFUSED" "$ID:2"

# --- 7 id and URL root from opencode.json ------------------------------------
reset
set_models "$(entry "$ID" llm "$(inst "$ID" 24000 1)")" "$(entry "$OTHER" llm "$(inst "$OTHER" 8192 1)")"
run model-preflight.sh
check "7a no arg: id from agent.tank.model, lmstudio/ stripped, inner slash kept" 0 "OK: $ID loaded, context 24000"
case "$(cat "$CURL_LOG")" in
  *"$BASE_URL/api/v1/models") pass "7b URL root: trailing /v1 stripped" ;;
  *) fail "7b URL root: trailing /v1 stripped" "curl got: $(cat "$CURL_LOG")" ;;
esac
reset; write_config "lmstudio/$ID" "$BASE_URL/v1/"; run model-preflight.sh
case "$(cat "$CURL_LOG")" in
  *"$BASE_URL/api/v1/models") pass "7c URL root: trailing /v1/ stripped" ;;
  *) fail "7c URL root: trailing /v1/ stripped" "curl got: $(cat "$CURL_LOG")" ;;
esac
reset
set_models "$(entry "$ID" llm "$(inst "$ID" 24000 1)")" "$(entry "$OTHER" llm "$(inst "$OTHER" 8192 1)")"
run model-preflight.sh "$OTHER"
check "7d explicit argument overrides the config id" 1 "REFUSED: $OTHER loaded at 8192"

# --- 8 cwd independence ------------------------------------------------------
reset
set_models "$(entry "$ID" llm "$(inst "$ID" 24000 1)")"
( cd / && PATH="$FAKE_BIN:$PATH" "$FIX_SCRIPTS/model-preflight.sh" >"$FIXTURE_ROOT/o" 2>"$FIXTURE_ROOT/e"; echo $? >"$FIXTURE_ROOT/c" )
if [ "$(cat "$FIXTURE_ROOT/c")" = 0 ] && grep -q "OK: $ID loaded, context 24000" "$FIXTURE_ROOT/o" "$FIXTURE_ROOT/e"; then
  pass "8 run from / still reads the agent's opencode.json"
else
  fail "8 run from /" "exit $(cat "$FIXTURE_ROOT/c"): $(cat "$FIXTURE_ROOT/o" "$FIXTURE_ROOT/e")"
fi

# --- 9 parallel --------------------------------------------------------------
reset; set_models "$(entry "$ID" llm "$(inst "$ID" 24000 4)")"; run model-preflight.sh
check "9a parallel 4: exit 1, both numbers" 1 "REFUSED: $ID loaded with parallel 4, tank needs 1"
reset; set_models "$(entry "$ID" llm "$(inst "$ID" 24000 1)")"; run model-preflight.sh
check "9b parallel 1: exit 0 OK" 0 "OK: $ID loaded, context 24000, parallel 1"
check_absent "9b2 no NOTE when only the own copy is loaded" "NOTE:"
reset; set_models "$(entry "$ID" llm "$(inst "$ID" 24000 -)")"; run model-preflight.sh
check "9c parallel absent: exit 1 (fail-safe)" 1 "REFUSED" "parallel"

# --- 10 other LLM instance ---------------------------------------------------
reset
set_models "$(entry "$ID" llm "$(inst "$ID" 24000 1)")" \
  "$(entry "$OTHER" llm "$(inst "$OTHER:3" 4096 2)")" \
  "$(entry "embed/emb-z" embedding "$(inst "embed/emb-z" 2048 1)")"
run model-preflight.sh
check "10a other LLM loaded: exit 0, NOTE names it" 0 "OK: $ID" "NOTE:" "$OTHER:3"
check_absent "10b embedding instance not named" "embed/emb-z"
reset
set_models "$(entry "$ID" llm)" "$(entry "$OTHER" llm "$(inst "$OTHER" 4096 1)")"
run model-preflight.sh
check "10c other LLM NOTE also on the not-loaded path" 0 "$ID not loaded" "$OTHER"

# --- 11 bypass ---------------------------------------------------------------
reset; TANK_MODEL_PREFLIGHT=0 run model-preflight.sh
check "11a =0: exit 0 with WARNING" 0 "WARNING:"
[ ! -s "$CURL_LOG" ] && pass "11b =0: curl never called" || fail "11b =0: curl never called" "$(cat "$CURL_LOG")"
for v in false 00 1 ""; do
  reset; CURL_EXIT=7 TANK_MODEL_PREFLIGHT="$v" run model-preflight.sh
  check "11c TANK_MODEL_PREFLIGHT='$v' does not skip" 1 "not reachable"
done

# --- 12 wiring ---------------------------------------------------------------
HC_EXPECT="run
--agent
tank
--title
tank health-check
Run the default health/hygiene check: container/service status, disk usage, dangling images/volumes, and Docker build-cache usage. Report what you find."
BU_EXPECT="run
--agent
tank
--title
tank bring-up: slug-q
Bring up the 'slug-q' environment."
TD_EXPECT="run
--agent
tank
--title
tank tear-down: slug-q
Tear down the 'slug-q' environment."

wire() {  # wire NAME SCRIPT EXPECTED_ARGV ARG...
  local name="$1" script="$2" expect="$3"; shift 3
  reset; set_models "$(entry "$ID" llm "$(inst "$ID" 8192 1)")"; run "$script" "$@"
  if [ "$EXIT_CODE" -ne 1 ] || [ -s "$OC_LOG" ]; then
    fail "$name refuses" "exit $EXIT_CODE, opencode argv: $(cat "$OC_LOG")"
  else pass "$name: preflight refuses -> exit 1, opencode never called"; fi
  reset; set_models "$(entry "$ID" llm "$(inst "$ID" 24000 1)")"; run "$script" "$@"
  if [ "$EXIT_CODE" -ne 0 ] || [ "$(cat "$OC_LOG")" != "$expect" ]; then
    fail "$name passes" "exit $EXIT_CODE, opencode argv: $(cat "$OC_LOG")"
  else pass "$name: preflight passes -> opencode called with unchanged argv"; fi
}
wire "12a health-check.sh" health-check.sh "$HC_EXPECT"
wire "12b bring-up.sh" bring-up.sh "$BU_EXPECT" slug-q
wire "12c tear-down.sh" tear-down.sh "$TD_EXPECT" slug-q

for s in bring-up.sh tear-down.sh; do
  reset; run "$s"
  if [ "$EXIT_CODE" -eq 1 ] && [ ! -s "$CURL_LOG" ] && [ ! -s "$OC_LOG" ]; then
    pass "12d $s: 0 args -> usage, exit 1, curl never called"
  else fail "12d $s: 0 args" "exit $EXIT_CODE curl: $(cat "$CURL_LOG") oc: $(cat "$OC_LOG")"; fi
done

echo
echo "----------------------------------------"
printf 'PASS: %d  FAIL: %d\n' "$PASS" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  echo "Failed:"; for n in "${FAILED_NAMES[@]}"; do echo "  - $n"; done
  exit 1
fi
exit 0
