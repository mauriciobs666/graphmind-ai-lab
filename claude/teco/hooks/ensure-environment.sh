#!/usr/bin/env bash
# Environment readiness for `teco` (cobb K-054). Two ways in, one script:
#   - SessionStart hook in teco's frontmatter (matcher startup|resume|clear|fork):
#     runs at agent start when teco is the main session (`claude --agent teco`).
#   - Run by hand, from the repo root, as teco's step-1 probe at decomposition
#     (both modes; the only path for a delegated teco, which gets no
#     SessionStart):   claude/teco/hooks/ensure-environment.sh </dev/null
#
# What it does:
#   1. FalkorDB (127.0.0.1:${FALKORDB_PORT:-6379}). Probed with `redis-cli ping`.
#   2. The `ws:agent-team` falkor-chat server (MCP `falkor-chat-agent-team`,
#      http://127.0.0.1:${AGENT_TEAM_PORT:-8200}/health). Started detached via
#      falkor-chat/scripts/start_agent_team.sh whenever /health refuses the
#      connection, whatever FalkorDB's state: that script starts FalkorDB when it
#      is down, waits on a falkordb-dev that is still loading, and fails under
#      the lock, harmlessly, on anything else. FalkorDB alone is started via
#      start_falkordb.sh -d when it is down and the server is not being launched.
#   3. CPG freshness, CHECK AND REPORT ONLY (never rebuilds). For every loaded
#      `cpg_*` graph, reads the :CpgBuildInfo marker read-only (GRAPH.RO_QUERY)
#      and classifies it by skills/cpg-analysis/references/freshness.md (check 0,
#      plus check 2's commit count when check 0 says the source moved). That
#      recipe is the authority; this script implements its rules, it does not
#      restate them. Anything the recipe says needs a human read (hand-authored
#      marker, legacy stamp, provenance outside the closed set, no marker,
#      dirty capture), and a graph list that could not be read, is reported as
#      "not assessed", never as fresh.
#   The `cypher` MCP needs nothing here: it is a stdio container Claude Code
#   spawns itself from .mcp.json at session start, and it reaches FalkorDB only
#   at query time, recovering once FalkorDB is up (docs/plans/cpg-query-access.md
#   §7.3). Docker itself being down breaks both it and FalkorDB; that surfaces
#   as a failed bring-up in the log below.
#
# Contract:
#   - ADDITIVE AND IDEMPOTENT ONLY. Starts what is down; never stops, restarts,
#     or resets anything. "Down" means connection refused and nothing else:
#     curl exit 7 for the server, redis-cli's "Connection refused" for FalkorDB.
#     Anything that answers (non-200, -LOADING, an error reply) or times out is
#     reported and left alone.
#   - BOUNDED. Each probe is capped at 2s, and the CPG pass starts a graph only
#     while 2s of its time budget remain: about 12s worst case, inside the 30s
#     hook timeout with room for one ~13s process-spawn stall (a WSL2 host quirk
#     that hits any spawned process, `timeout` included: claude/devops/ops-quirks.md).
#     The slow bring-up runs detached in its own
#     session (setsid + nohup, all fds redirected), so it outlives this script
#     and the Claude Code session. The server is meant to be always on.
#   - SINGLE-FLIGHT. `flock -n` on a per-port lock file, held by the detached
#     process for its whole life, so two concurrent launches start one server.
#     The lock is tested first: when it is held, the report says "already being
#     started" and nothing is launched.
#   - ALLOWLISTED ENVIRONMENT. The shared server must not inherit a session's
#     falkor-chat overrides, so it is launched under `env -i` with only the
#     host plumbing in KEEP_ENV below, plus FALKORCHAT_WS_ID=agent-team and the
#     probed FalkorDB host/port. No FALKORCHAT_* or per-run knob gets through.
#   - SCOPED TO THIS REPO. teco is deployed user-wide via
#     ~/.claude/agents/teco -> claude/teco; in any other project this is a no-op.
#   - FAIL-OPEN. Exit status is always 0; failures degrade to a message.
#   - DEPENDENCIES: bash, redis-cli, curl, git, flock, setsid, timeout, date.
#     No python3.
#
# Test: claude/teco/hooks/test-ensure-environment.sh (needs Docker; spins up its
# own FalkorDB on a spare port, never touches the live services).
#
# Matcher: startup|resume|clear|fork, i.e. every fresh MCP connection set or
# context reset. Not `compact`: teco's step 1 re-runs this script per goal, which
# covers a message lost to compaction and a service that died mid-session.
#
# Logs live outside the repo: ${XDG_STATE_HOME:-~/.local/state}/graphmind-ai-lab/.
#
# Output: silent when every service is up and every CPG is fresh. Otherwise one
# JSON object on stdout: `systemMessage` (shown to the user) plus
# `additionalContext` (read by teco). The MCP client connects in parallel with
# SessionStart (it does not wait for it) and retries a refused connect only ~3
# times over ~7s, then gives up for the session. So after a server start the
# message tells the user to `/mcp` -> Reconnect if it shows failed (K-054 Q2,
# verified on Claude Code 2.1.300; claude/cobb/kaizen/history.md 2026-10-14).

set -u

REPO_DIR="$(cd "$(dirname "$(readlink -f "$0")")/../../.." && pwd)"
FC_SCRIPTS="$REPO_DIR/falkor-chat/scripts"

# Drain stdin (hook input JSON; not needed) with bash builtins only, so no
# process is spawned for it. Stops at EOF, after 1s of silence, or after 2s in
# all, so a caller that leaves stdin open cannot hang the script.
if [ ! -t 0 ]; then
  while [ "$SECONDS" -lt 2 ] && IFS= read -r -t 1 _; do :; done
fi

# Scope: only act when the session's project is this repo.
proj="$(readlink -f "${CLAUDE_PROJECT_DIR:-$PWD}" 2>/dev/null || echo "${CLAUDE_PROJECT_DIR:-$PWD}")"
case "$proj/" in
  "$REPO_DIR"/*) ;;
  *) exit 0 ;;
esac

AGENT_TEAM_PORT="${AGENT_TEAM_PORT:-8200}"
FALKORDB_PORT="${FALKORDB_PORT:-6379}"
CPG_BUDGET_SECS="${CPG_BUDGET_SECS:-6}"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/graphmind-ai-lab"
mkdir -p "$STATE_DIR" 2>/dev/null || STATE_DIR="${TMPDIR:-/tmp}"
SERVER_LOG="$STATE_DIR/agent-team-$AGENT_TEAM_PORT.log"
SERVER_LOCK="$STATE_DIR/agent-team-$AGENT_TEAM_PORT.lock"
FALKOR_LOG="$STATE_DIR/falkordb-$FALKORDB_PORT.log"
FALKOR_LOCK="$STATE_DIR/falkordb-$FALKORDB_PORT.lock"

# What the launched scripts genuinely need from the caller's environment, and
# nothing else: PATH (redis-cli, docker, python3), HOME/USER/LOGNAME (the
# opencode config lookup, pip, docker's config), locale, temp dir, the XDG dirs
# pip and docker read, Docker endpoint selection, and proxy/CA settings for
# pip's network access. Only those that are set are passed.
KEEP_ENV="PATH HOME USER LOGNAME LANG LC_ALL TMPDIR XDG_RUNTIME_DIR XDG_CONFIG_HOME XDG_CACHE_HOME
  DOCKER_HOST DOCKER_CONTEXT DOCKER_CONFIG
  http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY SSL_CERT_FILE SSL_CERT_DIR"
kept_env=()
for v in $KEEP_ENV; do
  [ -n "${!v+x}" ] && kept_env+=("$v=${!v}")
done

msgs=()
ctx=()
say()  { msgs+=("$1"); }   # shown to the user
tell() { ctx+=("$1"); }    # read by teco

# Held only on flock's own conflict code; any other failure (flock missing, an
# unwritable lock file) is not evidence of a launch in flight.
lock_held() { flock -n -E 75 "$1" true 2>/dev/null; [ "$?" -eq 75 ]; }

launch_detached() {  # <lock> <log> <cmd...>; returns 1 when the lock is already held
  local lock="$1" log="$2"; shift 2
  lock_held "$lock" && return 1
  printf '\n=== %s  ensure-environment.sh (teco) launching: %s\n' "$(date -Is)" "$*" >>"$log" 2>/dev/null
  # flock -n: a launch that wins the race after the test above exits quietly.
  # setsid: own session/process group. All fds redirected -> returns at once.
  ( cd "$REPO_DIR/falkor-chat" && \
    setsid nohup flock -n "$lock" "$@" >>"$log" 2>&1 </dev/null & ) >/dev/null 2>&1
  return 0
}

# ── probe FalkorDB: up | down (refused) | notready (answers/timeout) | noprobe ─
falkor=noprobe; falkor_detail="redis-cli not installed"
if command -v redis-cli >/dev/null 2>&1; then
  out="$(timeout 2 redis-cli -h 127.0.0.1 -p "$FALKORDB_PORT" ping 2>&1)"; prc=$?
  if [ "$out" = "PONG" ]; then
    falkor=up
  elif printf '%s' "$out" | grep -q "Connection refused"; then
    falkor=down
  elif [ "$prc" -eq 124 ]; then
    falkor=notready; falkor_detail="no reply within 2s"
  else
    falkor=notready; falkor_detail="answered: $(printf '%s' "$out" | head -n1 | cut -c1-120)"
  fi
fi

# ── probe the ws:agent-team server ───────────────────────────────────────────
code="$(curl -s -o /dev/null -w '%{http_code}' --max-time 2 "http://127.0.0.1:$AGENT_TEAM_PORT/health" 2>/dev/null)"
rc=$?

server_launch=0   # 0 none · 1 launched · 2 a launch already holds the lock
if [ "$rc" -eq 7 ]; then
  if launch_detached "$SERVER_LOCK" "$SERVER_LOG" \
       env -i "${kept_env[@]}" FALKORCHAT_WS_ID=agent-team FALKORDB_HOST=127.0.0.1 \
           FALKORDB_PORT="$FALKORDB_PORT" AGENT_TEAM_PORT="$AGENT_TEAM_PORT" \
           "$FC_SCRIPTS/start_agent_team.sh"; then
    server_launch=1
    extra=""
    case "$falkor" in
      down) extra=" FalkorDB was down too; the same script starts it first." ;;
      notready|noprobe) extra=" FalkorDB is not confirmed up ($falkor_detail); the script waits up to 30s for it and gives up harmlessly if it never answers (see the log)." ;;
    esac
    say "ws:agent-team server was down on :$AGENT_TEAM_PORT; starting it in the background (log: $SERVER_LOG).$extra If /mcp then shows falkor-chat-agent-team as failed (startup took longer than the MCP client's ~7s retry window), select it and choose Reconnect once it is up."
    tell "Environment: the ws:agent-team server on :$AGENT_TEAM_PORT was down and has been started detached (log: $SERVER_LOG).$extra The falkor-chat-agent-team MCP connects by itself only if the server came up within the client's ~7s retry window. If the mcp__falkor-chat-agent-team__* tools are missing or fail when you need them, poll 'curl -sf http://127.0.0.1:$AGENT_TEAM_PORT/health' until 200, then ask the user to run /mcp -> falkor-chat-agent-team -> Reconnect (you cannot reconnect it yourself). Do not start a second copy."
  else
    server_launch=2
    say "ws:agent-team server on :$AGENT_TEAM_PORT is not answering yet, but it is already being started by an earlier launch (log: $SERVER_LOG); nothing new was launched."
    tell "Environment: the ws:agent-team server on :$AGENT_TEAM_PORT refused /health while another launch holds its start lock, so it is already being started (log: $SERVER_LOG). Nothing new was launched. Poll 'curl -sf http://127.0.0.1:$AGENT_TEAM_PORT/health' until 200; if the log shows the start failed, route to devops. If the mcp__falkor-chat-agent-team__* tools are missing once it is up, ask the user to run /mcp -> falkor-chat-agent-team -> Reconnect."
  fi
elif [ "$rc" -eq 0 ] && [ "$code" != "200" ]; then
  if [ "$falkor" = up ]; then
    say "ws:agent-team server on :$AGENT_TEAM_PORT answers /health with HTTP $code although FalkorDB answers; left untouched (log: $SERVER_LOG)."
    tell "Environment: the ws:agent-team server on :$AGENT_TEAM_PORT returns HTTP $code from /health while FalkorDB answers PONG. It was NOT restarted (never restart a shared service). If it persists, route to devops."
  else
    tell "Environment: the ws:agent-team server on :$AGENT_TEAM_PORT returns HTTP $code from /health. That is expected while FalkorDB is not up (/health is 503 exactly when FalkorDB is unreachable), so it is not a separate problem: re-probe /health once FalkorDB answers PONG, and route to devops only if a non-200 persists after that. The server was NOT restarted."
  fi
elif [ "$rc" -ne 0 ]; then
  say "ws:agent-team server on :$AGENT_TEAM_PORT did not answer /health (curl exit $rc); left untouched."
  tell "Environment: probing http://127.0.0.1:$AGENT_TEAM_PORT/health failed with curl exit $rc (not 'connection refused'), so nothing was started. If it persists, route to devops."
fi

# ── FalkorDB on its own (when the server launch is not already handling it) ──
if [ "$falkor" = down ] && [ "$server_launch" -eq 0 ]; then
  if launch_detached "$FALKOR_LOCK" "$FALKOR_LOG" "$FC_SCRIPTS/start_falkordb.sh" -d; then
    say "FalkorDB was down on :$FALKORDB_PORT; starting it in the background (log: $FALKOR_LOG)."
    tell "Environment: FalkorDB on :$FALKORDB_PORT was down and is being started detached (log: $FALKOR_LOG). Re-probe with 'redis-cli -p $FALKORDB_PORT ping' before a unit needs it; a failed bring-up is a devops blocker."
  else
    say "FalkorDB is down on :$FALKORDB_PORT and already being started by an earlier launch (log: $FALKOR_LOG)."
    tell "Environment: FalkorDB on :$FALKORDB_PORT refused the connection while another launch holds its start lock, so it is already being started (log: $FALKOR_LOG). Re-probe with 'redis-cli -p $FALKORDB_PORT ping' before a unit needs it; a failed bring-up is a devops blocker."
  fi
elif [ "$falkor" = notready ] || [ "$falkor" = noprobe ]; then
  say "FalkorDB on :$FALKORDB_PORT is not ready ($falkor_detail); left untouched."
  tell "Environment: FalkorDB on :$FALKORDB_PORT could not be confirmed up ($falkor_detail). It was not started or restarted. A -LOADING reply clears by itself; re-probe before a unit needs it, and route anything that persists to devops."
fi

# ── CPG freshness: check and report only (freshness.md is the authority) ────
# A pipeline BUILT_AT is `date -u +%Y-%m-%dT%H:%M:%SZ` (skills/joern-cpg). `date -d`
# alone also accepts '', 'now' and bare digits, so the shape is matched first.
is_iso_ts() {
  [[ "$1" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2} ]] && date -d "$1" >/dev/null 2>&1
}
# redis-cli exits 0 on an error reply and prints it bare (ERR …, LOADING …), so
# a failed GRAPH.LIST is recognised by its text, not only by the exit status.
graph_list_ok() {  # <exit status> <output>
  [ "$1" -eq 0 ] || return 1
  ! printf '%s\n' "$2" | head -n1 | grep -qE '^([A-Z][A-Z_]+ |Could not connect|Error)'
}
cpg_check() {
  local graphs g out lrc first rows n bt prov mo origin commit tree dirty cur cnt deadline
  local -a stale=() unassessed=()
  out="$(timeout 2 redis-cli -h 127.0.0.1 -p "$FALKORDB_PORT" GRAPH.LIST 2>&1)"; lrc=$?
  if ! graph_list_ok "$lrc" "$out"; then
    [ "$lrc" -eq 124 ] && out="no reply within 2s"
    unassessed+=("could not list graphs ($(printf '%s' "$out" | head -n1 | cut -c1-100)), so no CPG was checked")
    out=""
  fi
  graphs="$(printf '%s\n' "$out" | grep '^cpg_' || true)"
  deadline=$(( SECONDS + CPG_BUDGET_SECS ))
  local q="MATCH (b:CpgBuildInfo) RETURN '#M#' + coalesce(toString(b.BUILT_AT),'<null>') + '|' + coalesce(toString(b.PROVENANCE),'<null>') + '|' + coalesce(toString(b.MARKER_ORIGIN),'<null>') + '|' + coalesce(toString(b.SOURCE_ORIGIN),'<null>') + '|' + coalesce(toString(b.SOURCE_COMMIT),'<null>') + '|' + coalesce(toString(b.SOURCE_TREE),'<null>') + '|' + coalesce(toString(b.SOURCE_DIRTY),'<null>') AS m"
  while IFS= read -r g; do
    [ -z "$g" ] && continue
    if [ $(( SECONDS + 2 )) -gt "$deadline" ]; then   # a query may take its full 2s
      unassessed+=("$g and any later graphs (time budget spent; not checked)"); break
    fi
    out="$(timeout 2 redis-cli --raw -h 127.0.0.1 -p "$FALKORDB_PORT" GRAPH.RO_QUERY "$g" "$q" 2>&1)"
    first="$(printf '%s\n' "$out" | head -n1)"
    if [ "$first" != "m" ]; then
      unassessed+=("$g: marker unreadable ($(printf '%s' "$first" | cut -c1-80)); it may be mid-load")
      continue
    fi
    rows="$(printf '%s\n' "$out" | grep '^#M#' || true)"
    n="$(printf '%s' "$rows" | grep -c '^#M#' || true)"
    if [ "$n" -eq 0 ]; then
      unassessed+=("$g: no freshness marker, so no signal (freshness.md: treat as stale, and a graph mid-rebuild looks the same)")
      continue
    elif [ "$n" -gt 1 ]; then
      unassessed+=("$g: $n markers where freshness.md expects one, read them by hand")
      continue
    fi
    IFS='|' read -r bt prov mo origin commit tree dirty <<<"${rows#\#M\#}"
    if [ "$mo" != "<null>" ]; then
      if ! is_iso_ts "$bt"; then
        unassessed+=("$g: hand-written marker (BUILT_AT '$bt'), a frozen snapshot that is not rebuildable on demand, read its NOTE/STATUS")
      else
        unassessed+=("$g: hand-authored marker (provenance '$prov'): freshness.md requires reading its NOTE before check 0")
      fi
      continue
    fi
    if ! is_iso_ts "$bt"; then
      unassessed+=("$g: BUILT_AT '$bt' is not an ISO-8601 timestamp on a pipeline marker, read the whole node")
      continue
    fi
    case "$prov" in
      parse-root|source-origin) ;;
      none)
        unassessed+=("$g: provenance 'none' (no git identity), only parsedAt age applies, judge it by hand"); continue ;;
      "<null>")
        unassessed+=("$g: pre-2026-09-07 stamp (no provenance), check 0 unavailable, judge it by hand"); continue ;;
      *)
        unassessed+=("$g: provenance '$prov' is outside the closed set, so fail closed and read the whole node"); continue ;;
    esac
    if [ "$dirty" != "false" ]; then
      unassessed+=("$g: SOURCE_DIRTY is '$dirty', so check 0 is skipped (the graph matches no commit exactly)")
      continue
    fi
    if [ "$tree" = "<null>" ] || [ "$origin" = "<null>" ]; then
      unassessed+=("$g: no SOURCE_TREE/SOURCE_ORIGIN, so check 0 is unavailable")
      continue
    fi
    cur="$(cd "$REPO_DIR" && git rev-parse --verify -q "HEAD:./$origin" 2>/dev/null)"
    if [ -z "$cur" ]; then
      unassessed+=("$g: SOURCE_ORIGIN '$origin' does not resolve at HEAD")
      continue
    fi
    [ "$cur" = "$tree" ] && continue   # check 0: unchanged since capture, fresh
    cnt="?"
    if [ "$commit" != "<null>" ]; then
      cnt="$(cd "$REPO_DIR" && git log --oneline "$commit..HEAD" -- "$origin" 2>/dev/null | wc -l | tr -d ' ')"
    fi
    stale+=("$g: $origin changed since the build ($cnt commits since ${commit:0:8}; built $bt)")
  done <<<"$graphs"

  if [ "${#stale[@]}" -gt 0 ]; then
    local list; list="$(printf '\n- %s' "${stale[@]}")"
    say "CPG freshness, stale (rebuilds belong to graph-dba; none was started):$list"
    tell "CPG freshness at start (report only, nothing rebuilt). STALE:$list
Before a unit leans on one of these, ask graph-dba to rebuild it (teco step 3)."
  fi
  if [ "${#unassessed[@]}" -gt 0 ]; then
    local list; list="$(printf '\n- %s' "${unassessed[@]}")"
    say "CPG freshness, not assessed automatically:$list"
    tell "CPG freshness at start. NOT ASSESSED (so not fresh):$list
Apply skills/cpg-analysis/references/freshness.md by hand before a unit relies on one of these."
  fi
}
if [ "$falkor" = up ]; then
  cpg_check
else
  tell "CPG freshness was not checked because FalkorDB is not up. Re-run claude/teco/hooks/ensure-environment.sh </dev/null once it is."
fi

[ "${#msgs[@]}" -eq 0 ] && [ "${#ctx[@]}" -eq 0 ] && exit 0
if [ "${#msgs[@]}" -eq 0 ]; then
  msgs+=("teco environment check: see context.")
fi

json_escape() {
  local s; s="$(printf '%s' "$1" | tr -d '\000-\010\013\014\016-\037')"
  s=${s//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\n'/\\n}; s=${s//$'\t'/\\t}; s=${s//$'\r'/}
  printf '%s' "$s"
}
sm="$(printf '%s\n' "${msgs[@]}")"; ac="$(printf '%s\n' "${ctx[@]}")"
printf '{"systemMessage": "%s", "hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "%s"}}\n' \
  "$(json_escape "${sm%$'\n'}")" "$(json_escape "${ac%$'\n'}")"
exit 0
