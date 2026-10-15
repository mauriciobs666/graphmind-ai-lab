#!/usr/bin/env bash
# test-ensure-environment.sh — regression test for teco's environment hook,
# claude/teco/hooks/ensure-environment.sh (cobb K-054/K-055).
#
# Run it, from anywhere:
#   claude/teco/hooks/test-ensure-environment.sh             # test the shipped hook
#   claude/teco/hooks/test-ensure-environment.sh --mutants   # also prove it kills 3 mutants
#   HOOK=/path/to/copy.sh claude/teco/hooks/test-ensure-environment.sh   # test a copy
#   TEST_VERBOSE=1 ...                                           # also print the CPG report
# Exit: 0 all green (and, with --mutants, every mutant red) · 1 a failure · 2 setup error.
# Takes ~15s (~1 min with --mutants), mostly four 2s probe timeouts per suite run.
#
# Needs: bash, docker, the falkordb image (FALKORDB_IMAGE, default below), redis-cli,
# git, flock, setsid, python3 (to parse the hook's JSON; the hook itself needs none).
#
# NEVER TOUCHES THE LIVE SERVICES. The hook runs from a copy inside a throwaway
# sandbox that is its own git repo, so its REPO_DIR, its launch scripts (recording
# stubs) and its git reads are all the sandbox's. Its probes point at:
#   - a throwaway FalkorDB this script starts (`docker run --rm`, 127.0.0.1:
#     TEST_FALKORDB_PORT, default 6396) and stops on exit, for the CPG cells;
#   - stub `redis-cli` / `curl` on a curated PATH, for the probe-matrix cells.
# State, logs and locks go under the sandbox (XDG_STATE_HOME), never ~/.local/state.
#
# WHAT IT PINS, and why each part exists (claude/docs/reviews/teco-env-autostart.md, Pass 2):
#   1. Probe matrix: FalkorDB {PONG, refused, -LOADING, hang, redis-cli missing}
#      x /health {refused, 200, 503, curl timeout}: which script is launched, and
#      the message. A refused /health launches the server whatever FalkorDB's state.
#   2. Launched environment: with FALKORCHAT_ZZ_SENTINEL and other knobs exported,
#      the only FALKORCHAT_* the server script sees is FALKORCHAT_WS_ID=agent-team.
#   3. Lock held by a dummy flock: "already being started", never "has been started",
#      and nothing launched.
#   4. CPG shapes on a real FalkorDB, each classified per
#      skills/cpg-analysis/references/freshness.md, incl. the two shapes that alone
#      catch a mistyped MARKER_ORIGIN and a loosened SOURCE_DIRTY test, and an empty
#      BUILT_AT on a pipeline marker.
#   5. A GRAPH.LIST error reply is reported "could not list graphs", never silence.
#   6. stdin left open costs at most ~2s (the drain is a bash builtin, no `timeout`).
# The oracle is per-cell and positive: an expected launch must actually reach the
# recording stub, an expected silence must be empty stdout, and every non-silent
# output must parse as the hook's JSON shape.
#
# --mutants applies three mutants to COPIES of the hook (the shipped file is never
# edited), checks each sed really changed the copy, and requires each to turn the
# suite red: MARKER_ORIGIN->MARKER_ORIGN · `!= "false"`->`= "true"` on SOURCE_DIRTY ·
# a GRAPH.LIST error reply treated as "no graphs".
set -uo pipefail

HERE="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
HOOK="${HOOK:-$HERE/ensure-environment.sh}"
TEST_FALKORDB_PORT="${TEST_FALKORDB_PORT:-6396}"
FALKORDB_IMAGE="${FALKORDB_IMAGE:-falkordb/falkordb:v4.18.11}"
AGENT_TEAM_TEST_PORT=18299   # only ever reaches the curl stub
MUTANTS=0; [ "${1:-}" = "--mutants" ] && MUTANTS=1

die() { echo "SETUP ERROR: $*" >&2; exit 2; }
[ -f "$HOOK" ] || die "hook not found: $HOOK"
for t in docker redis-cli git flock setsid python3; do
  command -v "$t" >/dev/null 2>&1 || die "$t not installed"
done

WORK="$(mktemp -d)"
CONTAINER="ensure-env-test-$$"
HOLDER_PID=""
cleanup() {
  [ -n "$HOLDER_PID" ] && kill "$HOLDER_PID" 2>/dev/null
  docker kill "$CONTAINER" >/dev/null 2>&1   # --rm: killing it also removes it
  rm -rf "$WORK"
}
trap cleanup EXIT

# ── curated PATH: real tools minus redis-cli and curl, which cells choose ────
SYSBIN="$WORK/sysbin"; mkdir -p "$SYSBIN"
for t in bash sh env cat date grep head cut tr wc readlink dirname mkdir sleep git \
         flock setsid nohup timeout rm ls sed true mv; do
  p="$(command -v "$t")" || die "$t not installed"; ln -s "$p" "$SYSBIN/$t"
done
REALREDIS="$WORK/realredis"; mkdir -p "$REALREDIS"; ln -s "$(command -v redis-cli)" "$REALREDIS/redis-cli"
STUBCURL="$WORK/stubcurl"; STUBREDIS="$WORK/stubredis"; mkdir -p "$STUBCURL" "$STUBREDIS"
cat >"$STUBCURL/curl" <<'EOF'
#!/bin/bash
# STUB_CURL_RC: the exit code to fake (7 refused, 0 answered, 28 timeout); STUB_HTTP: the code.
rc="${STUB_CURL_RC:-0}"
if [ "$rc" -eq 0 ]; then printf '%s' "${STUB_HTTP:-200}"; else printf '000'; fi
exit "$rc"
EOF
cat >"$STUBREDIS/redis-cli" <<'EOF'
#!/bin/bash
# STUB_PING: pong | refused | loading | hang. STUB_LIST: empty | error.
for a in "$@"; do last="$a"; done
case "$*" in
  *GRAPH.LIST*)
    [ "${STUB_LIST:-empty}" = error ] && { echo "ERR unknown command 'GRAPH.LIST', with args beginning with: "; exit 0; }
    exit 0 ;;
esac
case "${STUB_PING:-pong}" in
  pong)    echo PONG ;;
  refused) echo "Could not connect to Redis at 127.0.0.1:6379: Connection refused" >&2; exit 1 ;;
  loading) echo "LOADING Redis is loading the dataset in memory" ;;
  hang)    sleep 5 ;;
esac
EOF
chmod +x "$STUBCURL/curl" "$STUBREDIS/redis-cli"

# ── sandbox repo: two commits to src/, so a marker can be fresh or 1 commit stale ─
SB="$WORK/repo"
mkdir -p "$SB/src" "$SB/claude/teco/hooks" "$SB/falkor-chat/scripts"
g() { git -C "$SB" -c user.name=test -c user.email=test@example.invalid -c commit.gpgsign=false "$@"; }
g init -q
echo one >"$SB/src/a.txt"; g add src; g commit -qm one
C_A="$(g rev-parse HEAD)"; T_A="$(g rev-parse HEAD:./src)"
echo two >"$SB/src/a.txt"; g add src; g commit -qm two
C_B="$(g rev-parse HEAD)"; T_B="$(g rev-parse HEAD:./src)"

REC="$WORK/rec"
# The server stub runs under the hook's `env -i`, so its record path is baked in.
cat >"$SB/falkor-chat/scripts/start_agent_team.sh" <<EOF
#!/bin/bash
env >"$REC/server.env.tmp" && mv "$REC/server.env.tmp" "$REC/server.env"
EOF
cat >"$SB/falkor-chat/scripts/start_falkordb.sh" <<EOF
#!/bin/bash
echo "\$*" >"$REC/falkordb.args"
EOF
chmod +x "$SB/falkor-chat/scripts/"*.sh
HOOK_SB="$SB/claude/teco/hooks/ensure-environment.sh"

# ── throwaway FalkorDB + CPG fixtures ────────────────────────────────────────
if (exec 3<>"/dev/tcp/127.0.0.1/$TEST_FALKORDB_PORT") 2>/dev/null; then
  die "port $TEST_FALKORDB_PORT is in use; set TEST_FALKORDB_PORT to a free one"
fi
docker run --rm -d --name "$CONTAINER" -p "127.0.0.1:$TEST_FALKORDB_PORT:6379" "$FALKORDB_IMAGE" >/dev/null \
  || die "could not start $FALKORDB_IMAGE"
for _ in $(seq 1 60); do
  [ "$(redis-cli -p "$TEST_FALKORDB_PORT" ping 2>/dev/null)" = PONG ] && break; sleep 0.25
done
[ "$(redis-cli -p "$TEST_FALKORDB_PORT" ping 2>/dev/null)" = PONG ] || die "test FalkorDB never answered"

seed() {  # <graph> <cypher>
  local r; r="$(redis-cli -p "$TEST_FALKORDB_PORT" GRAPH.QUERY "$1" "$2" 2>&1)"
  case "$r" in *rror*) die "seeding $1: $r" ;; esac
}
TS="'2026-10-14T10:00:00Z'"
PIPE="PROVENANCE:'source-origin', SOURCE_ORIGIN:'src', SOURCE_COMMIT:'$C_B'"
# expected class per graph: fresh (not mentioned) · stale · unassessed · ignored (not mentioned)
declare -A EXPECT=()
fixture() { EXPECT[$1]="$2"; seed "$1" "$3"; }
fixture cpg_fresh_bool      fresh      "CREATE (:CpgBuildInfo {BUILT_AT:$TS, $PIPE, SOURCE_TREE:'$T_B', SOURCE_DIRTY:false})"
fixture cpg_fresh_str       fresh      "CREATE (:CpgBuildInfo {BUILT_AT:$TS, PROVENANCE:'parse-root', SOURCE_ORIGIN:'src', SOURCE_COMMIT:'$C_B', SOURCE_TREE:'$T_B', SOURCE_DIRTY:'false'})"
fixture cpg_stale           stale      "CREATE (:CpgBuildInfo {BUILT_AT:$TS, PROVENANCE:'source-origin', SOURCE_ORIGIN:'src', SOURCE_COMMIT:'$C_A', SOURCE_TREE:'$T_A', SOURCE_DIRTY:false})"
fixture cpg_no_marker       unassessed "CREATE (:METHOD {NAME:'x'})"
fixture cpg_hand_pipeprov   unassessed "CREATE (:CpgBuildInfo {BUILT_AT:$TS, $PIPE, SOURCE_TREE:'$T_B', SOURCE_DIRTY:false, MARKER_ORIGIN:'graph-dba by hand', NOTE:'n'})"
fixture cpg_no_dirty        unassessed "CREATE (:CpgBuildInfo {BUILT_AT:$TS, $PIPE, SOURCE_TREE:'$T_B'})"
fixture cpg_empty_builtat   unassessed "CREATE (:CpgBuildInfo {BUILT_AT:'', $PIPE, SOURCE_TREE:'$T_B', SOURCE_DIRTY:false})"
fixture cpg_now_builtat     unassessed "CREATE (:CpgBuildInfo {BUILT_AT:'now', $PIPE, SOURCE_TREE:'$T_B', SOURCE_DIRTY:false})"
fixture cpg_hand_unknown    unassessed "CREATE (:CpgBuildInfo {BUILT_AT:'unknown', MARKER_ORIGIN:'graph-dba by hand', STATUS:'frozen'})"
fixture cpg_dirty           unassessed "CREATE (:CpgBuildInfo {BUILT_AT:$TS, $PIPE, SOURCE_TREE:'$T_B', SOURCE_DIRTY:true})"
fixture cpg_prov_none       unassessed "CREATE (:CpgBuildInfo {BUILT_AT:$TS, PROVENANCE:'none'})"
fixture cpg_legacy          unassessed "CREATE (:CpgBuildInfo {BUILT_AT:$TS, SOURCE_COMMIT:'$C_B', SOURCE_DIRTY:false})"
fixture cpg_backfilled      unassessed "CREATE (:CpgBuildInfo {BUILT_AT:$TS, PROVENANCE:'hand-backfilled', SOURCE_ORIGIN:'src', SOURCE_COMMIT:'$C_B', SOURCE_TREE:'$T_B', SOURCE_DIRTY:false})"
fixture cpg_two_markers     unassessed "CREATE (:CpgBuildInfo {BUILT_AT:$TS}), (:CpgBuildInfo {BUILT_AT:$TS})"
fixture cpg_bad_origin      unassessed "CREATE (:CpgBuildInfo {BUILT_AT:$TS, PROVENANCE:'source-origin', SOURCE_ORIGIN:'nope', SOURCE_COMMIT:'$C_B', SOURCE_TREE:'$T_B', SOURCE_DIRTY:false})"
fixture other_graph         ignored    "CREATE (:CpgBuildInfo {BUILT_AT:'', SOURCE_DIRTY:true})"

# ── running one cell ─────────────────────────────────────────────────────────
FAILS=0; OUT=""; SM=""; AC=""; CELL=""; STATE=""
fail() { echo "  FAIL [$CELL] $*"; FAILS=$((FAILS + 1)); }

run_hook() {  # <path-prefix> [VAR=value ...]; sets OUT, SM, AC; fresh state per cell
  local pathpre="$1"; shift
  STATE="$WORK/state-$RANDOM$RANDOM"; mkdir -p "$STATE"
  rm -rf "$REC"; mkdir -p "$REC"
  OUT="$(env -i PATH="$pathpre:$SYSBIN" HOME="$WORK/home" XDG_STATE_HOME="$STATE" \
         CLAUDE_PROJECT_DIR="$SB" AGENT_TEAM_PORT="$AGENT_TEAM_TEST_PORT" \
         FALKORCHAT_ZZ_SENTINEL=leak FALKORCHAT_EMBEDDING_MODEL=leak FALKORCHAT_STOREFRONT_ENABLED=1 \
         UVICORN_ARGS=--reload EMBEDDING_DIM=7 "$@" "$HOOK_SB" </dev/null)"
  SM=""; AC=""
  [ -z "$OUT" ] && return 0
  local parsed
  if ! parsed="$(printf '%s' "$OUT" | python3 -c '
import json, sys
d = json.load(sys.stdin)
assert d["hookSpecificOutput"]["hookEventName"] == "SessionStart"
print(d["systemMessage"]); print("@@CTX@@"); print(d["hookSpecificOutput"]["additionalContext"])')"; then
    fail "output is not the hook JSON shape: $(printf '%s' "$OUT" | head -c 200)"; return 0
  fi
  SM="${parsed%%@@CTX@@*}"; AC="${parsed#*@@CTX@@}"
}
LOGDIR() { echo "$STATE/graphmind-ai-lab"; }
launched() {  # <server|falkordb>: the hook decided to launch it (synchronous log line)
  local f; [ "$1" = server ] && f="agent-team-$AGENT_TEAM_TEST_PORT.log" || f="falkordb-$2.log"
  grep -q 'launching:' "$(LOGDIR)/$f" 2>/dev/null
}
wait_rec() {  # <file>: the detached launch really ran the script
  local i; for i in $(seq 1 50); do [ -s "$REC/$1" ] && return 0; sleep 0.1; done; return 1
}
has()   { printf '%s' "$SM$AC" | grep -qF -- "$1" || fail "expected '$1' in output"; }
hasnt() { printf '%s' "$SM$AC" | grep -qF -- "$1" && fail "unexpected '$1' in output"; }

check_server_env() {
  wait_rec server.env || { fail "server launch never reached start_agent_team.sh"; return; }
  local fc; fc="$(grep '^FALKORCHAT_' "$REC/server.env" | sort | tr '\n' ' ')"
  [ "$fc" = "FALKORCHAT_WS_ID=agent-team " ] || fail "server saw FALKORCHAT_* = '$fc' (want only FALKORCHAT_WS_ID=agent-team)"
  grep -qE '^(UVICORN_ARGS|EMBEDDING_DIM)=' "$REC/server.env" && fail "a per-run knob leaked into the server env"
  grep -q "^FALKORDB_PORT=$CELL_FPORT\$" "$REC/server.env" || fail "FALKORDB_PORT not pinned to $CELL_FPORT"
  grep -q '^FALKORDB_HOST=127.0.0.1$' "$REC/server.env" || fail "FALKORDB_HOST not pinned"
  grep -q '^PATH=' "$REC/server.env" && grep -q '^HOME=' "$REC/server.env" || fail "PATH/HOME not passed through"
}

# ── suite ────────────────────────────────────────────────────────────────────
run_suite() {
  local hook="$1" ping curl k want
  cp "$hook" "$HOOK_SB"; chmod +x "$HOOK_SB"
  FAILS=0
  CELL_FPORT=16390   # the stub redis-cli ignores the port; it only has to be pinned through

  # 1-2. probe matrix (stub redis-cli and curl)
  for curl in 7 200 503 28; do
    for ping in pong refused loading hang missing; do
      CELL="health=$curl falkordb=$ping"
      local pre="$STUBCURL:$STUBREDIS"; [ "$ping" = missing ] && pre="$STUBCURL"
      local crc=0 http="$curl"; case "$curl" in 7|28) crc="$curl"; http=000 ;; esac
      run_hook "$pre" FALKORDB_PORT="$CELL_FPORT" STUB_PING="$ping" STUB_CURL_RC="$crc" STUB_HTTP="$http"
      # the server is launched exactly when /health refuses, whatever FalkorDB's state
      if [ "$curl" = 7 ]; then
        launched server || fail "server not launched on a refused /health"
        has "starting it in the background"; has "has been started detached"
        check_server_env
      else
        launched server && fail "server launched although /health did not refuse"
      fi
      # FalkorDB alone: only when it refused and the server launch is not handling it
      if [ "$ping" = refused ] && [ "$curl" != 7 ]; then
        launched falkordb "$CELL_FPORT" || fail "FalkorDB not launched although refused"
        wait_rec falkordb.args || fail "FalkorDB launch never reached start_falkordb.sh"
        has "FalkorDB was down on :$CELL_FPORT"
      else
        launched falkordb "$CELL_FPORT" && fail "FalkorDB launched separately"
      fi
      case "$ping" in
        loading) has "not ready (answered: LOADING" ;;
        hang)    has "no reply within 2s" ;;
        missing) has "redis-cli not installed" ;;
      esac
      [ "$curl" = 7 ] && [ "$ping" = refused ] && has "FalkorDB was down too"
      case "$curl" in
        503) if [ "$ping" = pong ]; then has "HTTP 503 although FalkorDB answers"; else has "expected while FalkorDB is not up"; fi ;;
        28)  has "curl exit 28" ;;
      esac
      [ "$curl" = 200 ] && [ "$ping" = pong ] && [ -n "$OUT" ] && fail "expected silence, got: $SM"
    done
  done

  # 3. lock already held: no launch, no claim of a start
  CELL="lock held"
  STATE_PRE="$WORK/state-held"; mkdir -p "$STATE_PRE/graphmind-ai-lab"
  flock "$STATE_PRE/graphmind-ai-lab/agent-team-$AGENT_TEAM_TEST_PORT.lock" sleep 30 & HOLDER_PID=$!
  sleep 0.3
  rm -rf "$REC"; mkdir -p "$REC"
  OUT="$(env -i PATH="$STUBCURL:$STUBREDIS:$SYSBIN" HOME="$WORK/home" XDG_STATE_HOME="$STATE_PRE" \
         CLAUDE_PROJECT_DIR="$SB" AGENT_TEAM_PORT="$AGENT_TEAM_TEST_PORT" FALKORDB_PORT="$CELL_FPORT" \
         STUB_PING=pong STUB_CURL_RC=7 "$HOOK_SB" </dev/null)"
  kill "$HOLDER_PID" 2>/dev/null; wait "$HOLDER_PID" 2>/dev/null; HOLDER_PID=""
  SM="$OUT"; AC=""
  has "already being started"; hasnt "has been started"; hasnt "starting it in the background"
  STATE="$STATE_PRE"; launched server && fail "launched although the lock was held"
  sleep 0.3; [ -e "$REC/server.env" ] && fail "start_agent_team.sh ran although the lock was held"

  # 5. GRAPH.LIST error reply
  CELL="GRAPH.LIST error"
  run_hook "$STUBCURL:$STUBREDIS" FALKORDB_PORT="$CELL_FPORT" STUB_PING=pong STUB_LIST=error STUB_CURL_RC=0 STUB_HTTP=200
  has "could not list graphs"

  # 6. stdin left open by the caller
  CELL="stdin open"
  rm -f "$WORK/stdin.done"
  ( { sleep 6; } | { env -i PATH="$STUBCURL:$STUBREDIS:$SYSBIN" HOME="$WORK/home" XDG_STATE_HOME="$WORK/state-stdin" \
      CLAUDE_PROJECT_DIR="$SB" AGENT_TEAM_PORT="$AGENT_TEAM_TEST_PORT" FALKORDB_PORT="$CELL_FPORT" \
      STUB_PING=pong STUB_CURL_RC=0 STUB_HTTP=200 "$HOOK_SB" >/dev/null; : >"$WORK/stdin.done"; } ) &
  local hp=$! i
  for i in $(seq 1 50); do [ -e "$WORK/stdin.done" ] && break; sleep 0.1; done
  [ -e "$WORK/stdin.done" ] || fail "hook still running 5s after start with stdin left open"
  wait "$hp" 2>/dev/null

  # 4. CPG shapes against the real throwaway FalkorDB
  CELL="CPG shapes"; CELL_FPORT="$TEST_FALKORDB_PORT"
  run_hook "$STUBCURL:$REALREDIS" FALKORDB_PORT="$TEST_FALKORDB_PORT" STUB_CURL_RC=0 STUB_HTTP=200
  [ -n "${TEST_VERBOSE:-}" ] && printf '   --- CPG report (additionalContext) ---\n%s\n' "$AC" | sed 's/^/   /'
  local got
  got="$(printf '%s\n' "$AC" | awk '/STALE:/{s="stale"} /NOT ASSESSED/{s="unassessed"} /^- /{split(substr($0,3),a,/[: ]/); print a[1]"="s}')"
  for k in "${!EXPECT[@]}"; do
    want="${EXPECT[$k]}"
    local line; line="$(printf '%s\n' "$got" | grep "^$k=" || true)"
    case "$want" in
      fresh|ignored) [ -n "$line" ] && fail "$k ($want) was reported: $line" ;;
      *) [ "$line" = "$k=$want" ] || fail "$k: want $want, got '${line:-not mentioned}'" ;;
    esac
  done
  printf '%s' "$AC" | grep -q "cpg_stale: src changed since the build (1 commits" || fail "stale commit count is not 1"

  CELL="CPG budget spent"
  run_hook "$STUBCURL:$REALREDIS" FALKORDB_PORT="$TEST_FALKORDB_PORT" STUB_CURL_RC=0 STUB_HTTP=200 CPG_BUDGET_SECS=0
  has "time budget spent"

  return "$FAILS"
}

echo "== shipped hook: $HOOK"
t0=$SECONDS
run_suite "$HOOK"; shipped=$?
echo "   $shipped failure(s), $((SECONDS - t0))s"
[ "$MUTANTS" -eq 0 ] && { [ "$shipped" -eq 0 ] && echo "PASS" || echo "FAIL"; [ "$shipped" -eq 0 ]; exit; }

mutant() {  # <name> <sed expression>
  local copy="$WORK/mutant-$1.sh"
  sed -e "$2" "$HOOK" >"$copy"
  if cmp -s "$HOOK" "$copy"; then echo "== mutant $1: SED DID NOT APPLY"; return 1; fi
  echo "== mutant $1"
  run_suite "$copy" >"$WORK/mutant-$1.out"; local f=$?
  if [ "$f" -gt 0 ]; then
    echo "   RED, killed ($f failure(s)):"; sed 's/^/   /' "$WORK/mutant-$1.out"; return 0
  fi
  echo "   GREEN, SURVIVED"; return 1
}
ok=0
[ "$shipped" -eq 0 ] || ok=1
mutant marker-origin 's/b\.MARKER_ORIGIN/b.MARKER_ORIGN/g' || ok=1
mutant dirty-true    's/\[ "\$dirty" != "false" \]/[ "$dirty" = "true" ]/' || ok=1
mutant list-error    's/if ! graph_list_ok "\$lrc" "\$out"; then/if false \&\& ! graph_list_ok "$lrc" "$out"; then/' || ok=1
[ "$ok" -eq 0 ] && echo "PASS (shipped green, all mutants red)" || echo "FAIL"
exit "$ok"
