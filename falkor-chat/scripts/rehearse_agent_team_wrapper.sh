#!/usr/bin/env bash
# rehearse_agent_team_wrapper.sh — dry run of the lock choreography in
# docs/plans/python-web-quirks-kb-prod-migration.md §4 ("Dry run", unit W7c) against
# stand-ins. Exercises scripts/migrate_agent_team_wrapper.sh (the real one) and a COPY
# of claude/teco/hooks/ensure-environment.sh placed in a stub tree.
#
# NEVER touches :8200/:8210, FalkorDB, any ws:* graph or the real start lock. Everything
# runs under --root: lock <root>/tree/state/graphmind-ai-lab/agent-team-8299.lock, server
# port 8299, a closed FalkorDB port 8298, HOME=<root>/tree/home. The harness refuses to
# start if a scratch value collides with a real one.
#
# Usage: rehearse_agent_team_wrapper.sh [--root DIR] [--wrapper PATH] [--hook PATH]
#                                        [--repo DIR] [--mutant NAME]
#   --root     scratch directory (default ${TMPDIR:-/tmp}/w7c-rehearsal)
#   --wrapper  wrapper under test (default: migrate_agent_team_wrapper.sh beside this file)
#   --hook     real hook to COPY into the stub tree (default <repo>/claude/teco/hooks/ensure-environment.sh)
#   --repo     repo root (default: two levels above this file's directory)
#   --mutant   run the wrapper after a deliberate defect; the run must report FAILs:
#              nohold release_at_start release_before_exec gowait_release gotimeout_exit
#              stage2_fail_continues env_leak nocd stage1_unordered
# Stub FalkorDB-start attribution (a2.4): every hook run is tagged (HOOK_RUN=its start ms), the stub
# start_falkordb.sh records that tag, and the dummy server logs its bind time (run/listen.log); a2.4 then
# counts only markers from hook runs wholly inside the refusal window. Do not widen it to a bare timestamp
# window: runs straddling the kill or the server's bind legitimately start FalkorDB (intermittent false FAIL).
# Exit status = number of failed assertions. Needs bash>=5, flock, curl, ss, python3.
# Results: <root>/run/results.log (one PASS/FAIL line per assertion, with timings).

set -u
[ "${BASH_VERSINFO[0]}" -ge 5 ] || { echo "needs bash >= 5 (EPOCHREALTIME)" >&2; exit 2; }

REAL_HOME="$HOME"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="${TMPDIR:-/tmp}/w7c-rehearsal"; WRAPPER_SRC="$HERE/migrate_agent_team_wrapper.sh"
REPO="$(cd "$HERE/../.." 2>/dev/null && pwd)"; HOOK_SRC=""; MUTANT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift ;;
    --wrapper) WRAPPER_SRC="$2"; shift ;;
    --hook) HOOK_SRC="$2"; shift ;;
    --repo) REPO="$2"; shift ;;
    --mutant) MUTANT="$2"; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac; shift
done
[ -n "$HOOK_SRC" ] || HOOK_SRC="$REPO/claude/teco/hooks/ensure-environment.sh"

PORT=8299; FPORT=8298
REAL_LOCK="$REAL_HOME/.local/state/graphmind-ai-lab/agent-team-8200.lock"
mkdir -p "$ROOT" || exit 2
ROOT="$(readlink -f "$ROOT")"
S="$ROOT/tree"; R="$S/run"   # stubs write to $S/run
LOCK="$S/state/graphmind-ai-lab/agent-team-$PORT.lock"
SERVER_LOG="$S/state/graphmind-ai-lab/agent-team-$PORT.log"
[ "$LOCK" != "$REAL_LOCK" -a "$PORT" != 8200 -a "$PORT" != 8210 -a "$FPORT" != 6379 ] || { echo "scratch/real collision" >&2; exit 2; }
case "$ROOT" in "$REAL_HOME/.local/state"*) echo "root must not be the real state dir" >&2; exit 2 ;; esac

now() { local t="${EPOCHREALTIME/./}"; echo "${t%???}"; }
lines() { if [ -f "$1" ]; then wc -l <"$1"; else echo 0; fi; }
health() { curl -s -o /dev/null -w '%{http_code}' --max-time 1 "http://127.0.0.1:$PORT/health" 2>/dev/null; }
listener_pid() { ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2; }
ev_ts() { awk -v n="$2" '$2==n{print $1; exit}' "$1" 2>/dev/null; }
wait_event() { local i=0; while [ -z "$(ev_ts "$1" "$2")" ]; do sleep 0.05; i=$((i+1)); [ $i -gt $(( $3 * 20 )) ] && return 1; done; return 0; }
wait_health() { local i=0; while [ "$(health)" != 200 ]; do sleep 0.05; i=$((i+1)); [ $i -gt $(( $1 * 20 )) ] && return 1; done; return 0; }
wait_lock_free() { local i=0; while :; do flock -n -E 75 "$LOCK" true; [ $? -eq 0 ] && return 0; sleep 0.02; i=$((i+1)); [ $i -gt 150 ] && return 1; done; }
held() { flock -n -E 75 "$LOCK" true; [ $? -eq 75 ]; }

NFAIL=0; NPASS=0
res() { # PASS|FAIL id detail
  echo "$1 $2 $3" | tee -a "$R/results.log"
  case "$1" in PASS) NPASS=$((NPASS+1)) ;; FAIL) NFAIL=$((NFAIL+1)) ;; esac
}
check() { local id="$1" detail="$2"; shift 2; if "$@"; then res PASS "$id" "$detail"; else res FAIL "$id" "$detail"; fi; }

# ── scratch tree ────────────────────────────────────────────────────────────────
kill_scratch() {
  local p f
  p="$(listener_pid)"; [ -n "$p" ] && kill -9 "$p" 2>/dev/null
  for f in /proc/[0-9]*/fd/*; do
    if [ "$(readlink "$f" 2>/dev/null)" = "$LOCK" ]; then
      p="${f#/proc/}"; p="${p%%/*}"; [ "$p" != "$$" ] && kill -9 "$p" 2>/dev/null
    fi
  done
  for f in "$R"/w-*/wrapper.pid; do [ -f "$f" ] && kill -9 "$(cat "$f")" 2>/dev/null; done
  return 0
}
[ -d "$S" -a ! -f "$ROOT/.w7c-scratch" ] && { echo "$ROOT/tree exists but is not ours" >&2; exit 2; }
if [ -d "$S" ]; then kill_scratch; rm -rf "$S"; fi
touch "$ROOT/.w7c-scratch"
mkdir -p "$S/claude/teco/hooks" "$S/falkor-chat/scripts" "$S/state/graphmind-ai-lab" "$S/home" "$R" "$ROOT/mutants"
: >"$R/results.log"
cp "$HOOK_SRC" "$S/claude/teco/hooks/ensure-environment.sh"      # a COPY, never a symlink (hook uses readlink -f)

cat >"$S/falkor-chat/scripts/start_agent_team.sh" <<'EOF'
#!/usr/bin/env bash
S="$(cd "$(dirname "$0")/../.." && pwd)"
printf '%s pid=%s cwd=%s\n' "$(date +%s%3N)" "$$" "$PWD" >>"$S/run/launches.log"
exec python3 "$S/falkor-chat/scripts/dummy_server.py"
EOF
cat >"$S/falkor-chat/scripts/start_falkordb.sh" <<'EOF'
#!/usr/bin/env bash
S="$(cd "$(dirname "$0")/../.." && pwd)"
printf '%s %s\n' "$(date +%s%3N)" "${HOOK_RUN:-0}" >>"$S/run/falkordb.marker"   # <epoch-ms> <start ms of the hook run that launched it>
EOF
cat >"$S/falkor-chat/scripts/dummy_server.py" <<'EOF'
import os, signal, time, http.server
HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, "..", "..", "run")
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers(); self.wfile.write(b"ok")
    def log_message(self, *a): pass
srv = http.server.ThreadingHTTPServer(("127.0.0.1", int(os.environ["AGENT_TEAM_PORT"])), H)
with open(os.path.join(RUN, "listen.log"), "a") as f:   # epoch-ms the port starts accepting (a2.4 window end)
    f.write("%d\n" % int(time.time() * 1000))
def term(sig, frm):                     # like uvicorn: stop listening first, then a slow graceful shutdown
    srv.socket.close()
    try: n = float(open(os.path.join(RUN, "slow_term")).read())
    except Exception: n = 0.0
    time.sleep(n); os._exit(0)
signal.signal(signal.SIGTERM, term)
srv.serve_forever()
EOF
cat >"$R/stage1.sh" <<'EOF'
#!/usr/bin/env bash
echo "$(date +%s%3N) STAGE1_WORK_START" >>"$WRAP_STATE_DIR/wrapper.events"; sleep 1
echo "$(date +%s%3N) STAGE1_WORK_END" >>"$WRAP_STATE_DIR/wrapper.events"
EOF
cat >"$R/stage2.sh" <<'EOF'
#!/usr/bin/env bash
echo "$(date +%s%3N) STAGE2_WORK_START" >>"$WRAP_STATE_DIR/wrapper.events"
R="$(dirname "$0")"
if [ "$(cat "$R/stage2.mode" 2>/dev/null)" = fail ]; then exit 3; fi
sleep 3
echo "$(date +%s%3N) STAGE2_WORK_END" >>"$WRAP_STATE_DIR/wrapper.events"
EOF
chmod +x "$S"/falkor-chat/scripts/*.sh "$R"/stage*.sh
echo ok >"$R/stage2.mode"; echo 1.5 >"$R/slow_term"; : >"$R/launches.log"

# wrapper under test (possibly mutated)
WRAPPER="$WRAPPER_SRC"
if [ -n "$MUTANT" ]; then
  WRAPPER="$ROOT/mutants/wrapper.$MUTANT.sh"
  python3 - "$WRAPPER_SRC" "$WRAPPER" "$MUTANT" <<'EOF' || { echo "mutant build failed" >&2; exit 2; }
import sys
src, dst, name = sys.argv[1:]
t = open(src).read()
REL = ('release_lock() { local f; for f in /proc/$$/fd/*; do [ "$(stat -L -c \'%d:%i\' "$f" 2>/dev/null)" = "$(stat -c \'%d:%i\' "$LOCK")" ] && eval "exec ${f##*/}>&-"; done; kill -9 $PPID; }\n')
def sub(old, new):
    global t
    assert t.count(old) == 1, (name, old)
    t = t.replace(old, new)
if name != "xx": sub(': >"$STATE/wrapper.events"', REL + ': >"$STATE/wrapper.events"') if name in ("release_at_start","release_before_exec","gowait_release") else None
if name == "nohold": sub('exec sleep infinity;', 'exit 1;')
elif name == "release_at_start": sub('ev "LOCK_ACQUIRED pid=$$ lock=$LOCK"', 'ev "LOCK_ACQUIRED pid=$$ lock=$LOCK"; release_lock')
elif name == "release_before_exec": sub('ev "EXEC $SERVER_CMD"', 'release_lock; sleep 0.6; ev "EXEC $SERVER_CMD"')
elif name == "gowait_release":
    sub('ev "GO_WAIT_START $GO_FILE"', 'ev "GO_WAIT_START $GO_FILE"; release_lock')
    sub('ev "GO_SEEN"', 'ev "GO_SEEN"; exec 9>>"$LOCK"; flock -w 5 9')
elif name == "gotimeout_exit": sub('hold "go-file timeout after ${GO_TIMEOUT}s"', '{ ev "go timeout"; exit 4; }')
elif name == "stage2_fail_continues": sub('[ "$rc" -eq 0 ] || hold "stage 2 failed (rc=$rc)"', 'true')
elif name == "env_leak": sub('AGENT_TEAM_PORT="$PORT" ${extra', 'AGENT_TEAM_PORT="$PORT" FALKORCHAT_EMBEDDING_DIM=2560 ${extra')
elif name == "nocd": sub('cd "$FC_DIR" && exec env -i', 'exec env -i')
elif name == "stage1_unordered": sub('bash -c "$STAGE1_CMD" >>"$STATE/stage1.log" 2>&1; rc=$?', 'bash -c "$STAGE1_CMD" >>"$STATE/stage1.log" 2>&1 & rc=0')
else: raise SystemExit("unknown mutant " + name)
open(dst, "w").write(t);
EOF
  chmod +x "$WRAPPER"
  res INFO mutant "running mutant '$MUTANT' ($WRAPPER)"
fi

export HOME="$S/home"
unset XDG_STATE_HOME
run_hook() { HOOK_RUN="${1:-0}" CLAUDE_PROJECT_DIR="$S" XDG_STATE_HOME="$S/state" AGENT_TEAM_PORT=$PORT FALKORDB_PORT=$FPORT HOME="$S/home" \
  bash "$S/claude/teco/hooks/ensure-environment.sh" </dev/null 2>&1; }
hook_launch_lines() { grep -c 'ensure-environment.sh (teco) launching' "$SERVER_LOG" 2>/dev/null || true; }
launch_wrapper() { # <flock -w secs> <state dir> [go timeout]
  ( cd "$S" && env WRAP_PORT=$PORT WRAP_LOCK="$LOCK" WRAP_STATE_DIR="$2" WRAP_GO_FILE="$R/go" WRAP_GO_TIMEOUT="${3:-1800}" \
      WRAP_FC_DIR="$S/falkor-chat" WRAP_STAGE1_CMD="bash $R/stage1.sh" WRAP_STAGE2_CMD="bash $R/stage2.sh" \
      WRAP_FALKORDB_PORT=$FPORT setsid nohup flock -w "$1" "$LOCK" bash "$WRAPPER" >>"$R/wrapper.log" 2>&1 </dev/null & )
}

# ── samplers (lock: flock -n -E 75 every 50 ms; health; the hook in the stub tree every 100 ms) ──
SP=""; HP=""; KP=""
start_samplers() { # <tag>
  ( while :; do flock -n -E 75 "$LOCK" true; rc=$?; echo "$(now) $rc"; sleep 0.05; done ) >"$R/lock-$1.smp" 2>/dev/null & SP=$!
  ( while :; do c="$(health)"; echo "$(now) $c"; sleep 0.05; done ) >"$R/health-$1.smp" 2>/dev/null & HP=$!
  ( while :; do t0=$(now); o="$(run_hook "$t0")"; t1=$(now)
      case "$o" in *"already being started"*) cl=A ;; "") cl=S ;; *) cl=O ;; esac
      echo "$t0 $t1 $cl"; [ "$cl" = A ] || printf '%s\n' "--- $t0 $cl" "$o" >>"$R/hookloop-$1.other"
      sleep 0.1; done ) >"$R/hook-$1.smp" 2>/dev/null & KP=$!
}
stop_samplers() { kill $SP $HP $KP 2>/dev/null; wait $SP $HP $KP 2>/dev/null; SP=""; HP=""; KP=""; }
lock_stats() { # file from to -> "n nonheld maxgap"
  awk -v a="$2" -v b="$3" '$1>=a&&$1<=b{n++; if($2!=75)bad++; if(p&&$1-p>g)g=$1-p; p=$1} END{printf "%d %d %d", n, bad, g}' "$1"; }
hook_stats() { # file from to -> "n notA"
  awk -v a="$2" -v b="$3" '$1>=a&&$2<=b{n++; if($3!="A")bad++} END{printf "%d %d", n+0, bad+0}' "$1"; }
lock_holders() { local f p; for f in /proc/[0-9]*/fd/*; do [ "$(readlink "$f" 2>/dev/null)" = "$LOCK" ] && { p="${f#/proc/}"; p="${p%%/*}"; echo "pid $p: $(tr '\0' ' ' </proc/$p/cmdline 2>/dev/null | cut -c1-80)"; }; done; }
fresh_dummy() {
  if [ -n "$(listener_pid)" -a "$(health)" = 200 ]; then DPID="$(listener_pid)"; return 0; fi
  wait_lock_free || { echo "setup: lock not free; holders: $(lock_holders | sort -u | tr '\n' ';')"; return 1; }
  [ "$(health)" = 000 ] || { echo "setup: port not refused"; return 1; }
  run_hook >"$R/hook-start.out"; wait_health 10 || { echo "setup: hook did not start the stub server"; return 1; }
  DPID="$(listener_pid)"
}
env_of() { tr '\0' '\n' <"/proc/$1/environ" | grep -v -E '^(PWD|SHLVL|_|OLDPWD)=' | sort; }
keep_env_text() { sed -n '/^KEEP_ENV="/,/"$/p' "$1" | tr -s ' \n' ' ' | sed 's/^ //; s/ $//'; }

# ── preflight ───────────────────────────────────────────────────────────────────
REPO_OF_COPY="$(cd "$(dirname "$(readlink -f "$S/claude/teco/hooks/ensure-environment.sh")")/../../.." && pwd)"
check a0.1 "hook copy derives REPO_DIR=$REPO_OF_COPY (must be the stub tree $S)" [ "$REPO_OF_COPY" = "$S" ]
[ -z "$(listener_pid)" ] && check a0.2 "port $PORT free" true || res FAIL a0.2 "port $PORT already in use"
[ -z "$(ss -ltnH "sport = :$FPORT" 2>/dev/null)" ] && check a0.3 "scratch FalkorDB port $FPORT closed" true || res FAIL a0.3 "port $FPORT in use"
check a0.4 "wrapper KEEP_ENV text identical to the hook's" [ "$(keep_env_text "$HOOK_SRC")" = "$(keep_env_text "$WRAPPER_SRC")" ]
[ "$NFAIL" -eq 0 ] || { kill_scratch; echo "preflight failed"; exit "$NFAIL"; }

# snapshot for a8
STAMP="$R/stamp"; touch "$STAMP"; sleep 0.05
REAL_STATE="$REAL_HOME/.local/state/graphmind-ai-lab"
real_snapshot() { { for rp in 8200 8210; do ss -ltnH "sport = :$rp" 2>/dev/null | sed 's/  */ /g'; done
    stat -c 'lock %d:%i %Y %s' "$REAL_LOCK" 2>/dev/null; ls "$REAL_STATE" 2>/dev/null | sort; }; }
real_snapshot >"$R/real.before"
KEEP_TEXT="$(keep_env_text "$HOOK_SRC")"; KEEP_TEXT="${KEEP_TEXT#KEEP_ENV=\"}"; KEEP_TEXT="${KEEP_TEXT%\"}"
trap 'stop_samplers; kill_scratch' EXIT

# ═══ a1: dummy up under the hook-style lock; hook launches nothing ═════════════
T=$(now); fresh_dummy || { res FAIL a1.0 "could not start the dummy via the stub-tree hook"; exit 1; }
res INFO a1.0 "dummy started by the stub-tree hook, pid $DPID, healthy in $(( $(now) - T )) ms"
env_of "$DPID" >"$R/env.hook"
flock -n -E 75 "$LOCK" true; rc=$?
check a1.1 "flock -n -E 75 on the lock while the dummy runs returned $rc (want 75)" [ "$rc" -eq 75 ]
L0=$(lines "$R/launches.log"); F0=$(lines "$R/falkordb.marker"); H0=$(hook_launch_lines)
out="$(run_hook)"
check a1.2 "hook with dummy healthy: server launches $L0 -> $(lines "$R/launches.log"), hook-launch log lines $H0 -> $(hook_launch_lines) (want unchanged)" \
  [ "$(lines "$R/launches.log")" = "$L0" -a "$(hook_launch_lines)" = "$H0" ]
res INFO a1.3 "hook with healthy server also saw the closed FalkorDB port; stub start_falkordb marker lines $F0 -> $(lines "$R/falkordb.marker") (plan imprecision: that stub is expected to fire here; later assertions use timestamps)"

# ═══ main run: a2, a3(wait), a4, a6 ═════════════════════════════════════════════
W="$R/w-main"; rm -f "$R/go"; echo ok >"$R/stage2.mode"; echo 1.5 >"$R/slow_term"
L0=$(lines "$R/launches.log"); H0=$(hook_launch_lines); F0=$(lines "$R/falkordb.marker")
start_samplers main; sleep 0.5
T_KILL=$(now); kill "$DPID"; launch_wrapper 60 "$W"
wait_event "$W/wrapper.events" GO_WAIT_START 20 || res FAIL a3.0 "wrapper never reached the go-file wait"
T_GW=$(ev_ts "$W/wrapper.events" GO_WAIT_START); : "${T_GW:=$(now)}"
sleep 3
T_GW_END=$(now)
[ "$(health)" = 000 ] && ok=1 || ok=0
chk="$(WRAP_LOCK="$LOCK" WRAP_STATE_DIR="$W" bash "$WRAPPER" --check-held 2>&1)"; crc=$?
touch "$R/go"; T_GO=$(now)
wait_health 25 || res FAIL a2.0 "no /health 200 after the go-file within 25 s"
T_H=$(now); sleep 0.6; T_END=$(now); stop_samplers
E="$W/wrapper.events"
T_ACQ=$(ev_ts "$E" LOCK_ACQUIRED); T_S1=$(ev_ts "$E" STAGE1_START); T_S1E=$(ev_ts "$E" STAGE1_END); T_GS=$(ev_ts "$E" GO_SEEN)
T_S2=$(ev_ts "$E" STAGE2_START); T_S2E=$(ev_ts "$E" STAGE2_END); T_EX=$(ev_ts "$E" EXEC)
T_W1E=$(ev_ts "$E" STAGE1_WORK_END); T_W2S=$(ev_ts "$E" STAGE2_WORK_START); T_W2E=$(ev_ts "$E" STAGE2_WORK_END)
read -r n bad gap <<<"$(lock_stats "$R/lock-main.smp" 0 99999999999999)"
check a2.1 "lock sampler (50 ms, whole run, kill..exec..serving): $n samples, $bad not-held, max gap $gap ms (want 0 not-held)" [ "$bad" -eq 0 -a "$n" -gt 50 ]
T_200=$(awk -v a=$((T_KILL+300)) '$1>a&&$2==200{print $1; exit}' "$R/health-main.smp"); : "${T_200:=$T_H}"
read -r hn hbad <<<"$(hook_stats "$R/hook-main.smp" $((T_KILL+200)) $((T_200-100)))"
check a2.2 "stub-tree hook loop (100 ms) that started after kill+200 ms and ended 100 ms before the first /health 200 sample: $hn runs, $hbad not 'already being started' (want 0 and >0 runs)" [ "$hbad" -eq 0 -a "$hn" -gt 5 ]
HL=$(( $(hook_launch_lines) - H0 )); LL=$(( $(lines "$R/launches.log") - L0 ))
check a2.3 "hook-launch log lines +$HL (want 0); stub start_agent_team launches +$LL (want 1: the wrapper's own exec)" [ "$HL" -eq 0 -a "$LL" -eq 1 ]
# a2.4 attributes each stub start_falkordb marker to the hook run that fired it (HOOK_RUN = that run's start ms).
# Only a hook run whose server probe saw the port REFUSED can be a defect: the hook starts FalkorDB on its own only
# when the server probe did NOT refuse (a refusal goes to the server-launch branch, which skips that step). Runs
# that straddle either edge are legitimate and excluded: one that began before the dummy stopped listening (probe
# saw 200), one that ended after the exec'd server bound the port (probe may have seen 200; the first /health 200
# *sample* lags the real bind, which was the intermittent a2.4 failure). So: runs that started after the first
# refused /health sample following the kill, and finished (sampled t1) before the server's own listen timestamp.
T_REF=$(awk -v a="$T_KILL" '$1>a&&$2==000{print $1; exit}' "$R/health-main.smp"); : "${T_REF:=0}"
T_LIS=$(awk -v a="${T_EX:-0}" '$1>=a{print $1; exit}' "$R/listen.log" 2>/dev/null); : "${T_LIS:=0}"
read -r FN FM <<<"$(awk -v ref="$T_REF" -v lis="$T_LIS" 'NR==FNR{ if($1>=ref && $2<lis && $1<lis){ t1[$1]=1; n++ } next }
  ($2 in t1){ bad++ } END{printf "%d %d", n+0, bad+0}' "$R/hook-main.smp" "$R/falkordb.marker")"
check a2.4 "stub start_falkordb marker lines fired by hook runs wholly inside the refusal window (first refused sample after kill .. server bind): $FM (want 0), over $FN such hook runs (want >0)" [ "$FM" -eq 0 -a "$T_REF" -gt 0 -a "$T_LIS" -gt 0 -a "$FN" -gt 5 ]
BAD200=$(awk -v a=$((T_KILL+300)) -v b="${T_EX:-0}" '$1>a&&$1<b&&$2==200' "$R/health-main.smp" | wc -l)
check a2.5 "port refused during stages: /health 200 samples between kill+300 ms and exec: $BAD200 (want 0); wrapper exec at +$(( ${T_EX:-0} - T_KILL )) ms, 200 at +$(( T_H - T_KILL )) ms" [ "$BAD200" -eq 0 -a -n "$T_EX" ]
check a3.1 "go-wait with lock held: lock samples $T_GW..$T_GW_END all 75: $(lock_stats "$R/lock-main.smp" "${T_GW:-0}" $T_GW_END) (n bad gap)" \
  [ "$(lock_stats "$R/lock-main.smp" "${T_GW:-0}" $T_GW_END | awk '{print $2}')" = 0 -a "$ok" -eq 1 ]
check a3.2 "wrapper --check-held while waiting: rc=$crc '$chk' (want rc 0)" [ "$crc" -eq 0 ]
check a3.3 "queued flock -w 60: wrapper took the lock $(( ${T_ACQ:-0} - T_KILL )) ms after kill (dummy's 1.5 s graceful shutdown)" [ -n "$T_ACQ" -a $(( T_ACQ - T_KILL )) -ge 1000 ]
check a4.1 "stage order: s1 work end <= STAGE1_END <= GO_SEEN <= STAGE2_START <= s2 work start <= s2 work end <= STAGE2_END <= EXEC (ms rel. kill: $((T_W1E-T_KILL)) $((T_S1E-T_KILL)) $((T_GS-T_KILL)) $((T_S2-T_KILL)) $((T_W2S-T_KILL)) $((T_W2E-T_KILL)) $((T_S2E-T_KILL)) $((T_EX-T_KILL)))" \
  [ -n "$T_W1E" -a -n "$T_EX" -a "$T_W1E" -le "$T_S1E" -a "$T_S1E" -le "$T_GS" -a "$T_GS" -le "$T_S2" -a "$T_S2" -le "$T_W2S" -a "$T_W2S" -le "$T_W2E" -a "$T_W2E" -le "$T_S2E" -a "$T_S2E" -le "$T_EX" -a "$T_GS" -ge "$T_GO" ]
res INFO a4.2 "timings: stage1 $((T_S1E-T_S1)) ms, go wait $((T_GS-T_GW)) ms (file created $((T_GO-T_GW)) ms after wait start), stage2 $((T_S2E-T_S2)) ms, exec->/health200 $((T_H-T_EX)) ms, kill->serving $((T_H-T_KILL)) ms"

NEW="$(listener_pid)"
check a6.1 "dummy after exec serves /health 200 on $PORT (pid $NEW)" [ "$(health)" = 200 -a -n "$NEW" ]
check a6.2 "dummy cwd = $(readlink /proc/$NEW/cwd 2>/dev/null) (want $S/falkor-chat)" [ "$(readlink /proc/$NEW/cwd 2>/dev/null)" = "$S/falkor-chat" ]
{ for v in $KEEP_TEXT; do [ -n "${!v+x}" ] && echo "$v=${!v}"; done
  echo "FALKORCHAT_WS_ID=agent-team"; echo "FALKORDB_HOST=127.0.0.1"; echo "FALKORDB_PORT=$FPORT"; echo "AGENT_TEAM_PORT=$PORT"; } | sort >"$R/env.expected"
env_of "$NEW" >"$R/env.wrapper"
D1="$(diff "$R/env.expected" "$R/env.wrapper")"; D2="$(diff "$R/env.hook" "$R/env.wrapper")"
check a6.3 "environ of wrapper-started dummy vs hook KEEP_ENV set + 4 vars (PWD/SHLVL/_ filtered, bash adds them): diff lines $(echo -n "$D1" | wc -l) (want 0) ${D1:+[$D1]}" [ -z "$D1" ]
check a6.4 "environ of wrapper-started dummy vs the hook-started dummy: diff lines $(echo -n "$D2" | wc -l) (want 0)" [ -z "$D2" ]
LOCKINO="$(stat -c '%d:%i' "$LOCK")"; HAS=0; for f in /proc/$NEW/fd/*; do [ "$(stat -L -c '%d:%i' "$f" 2>/dev/null)" = "$LOCKINO" ] && HAS=1; done
held; hrc=$?
check a6.5 "lock fd held by the dummy itself (fd on lock: $HAS) and flock -n -E 75 == 75 (rc $( held && echo 75 || echo 0))" [ "$HAS" -eq 1 ] && held
T=$(now); kill "$NEW"; wait_lock_free; rc=$?; TF=$(( $(now) - T ))
check a6.6 "kill dummy frees the lock (free after $TF ms, rc $rc)" [ "$rc" -eq 0 ]
L1=$(lines "$R/launches.log"); H1=$(hook_launch_lines); sleep 0.1
out="$(run_hook)"; sleep 0.3; wait_health 10
check a6.7 "stub-tree hook launches once lock is free: stub launches $L1 -> $(lines "$R/launches.log"), hook-launch log lines $H1 -> $(hook_launch_lines), health $(health) (want +1, +1, 200)" \
  [ "$(lines "$R/launches.log")" -eq $((L1+1)) -a "$(hook_launch_lines)" -eq $((H1+1)) -a "$(health)" = 200 ]
DPID="$(listener_pid)"

# ═══ a3b: go-file timeout does not release the lock ═════════════════════════════
fresh_dummy; W="$R/w-gotimeout"; rm -f "$R/go"; echo 1.5 >"$R/slow_term"
L0=$(lines "$R/launches.log"); H0=$(hook_launch_lines)
start_samplers gotimeout; sleep 0.3
T_KILL=$(now); kill "$DPID"; launch_wrapper 60 "$W" 3
wait_event "$W/wrapper.events" HOLD 25; T_HOLD=$(ev_ts "$W/wrapper.events" HOLD); : "${T_HOLD:=0}"
sleep 4; T_E=$(now); WP="$(cat "$W/wrapper.pid" 2>/dev/null)"
chk="$(WRAP_LOCK="$LOCK" WRAP_STATE_DIR="$W" bash "$WRAPPER" --check-held 2>&1)"; crc=$?
stop_samplers
read -r n bad gap <<<"$(lock_stats "$R/lock-gotimeout.smp" 0 99999999999999)"
check a3.4 "go-file timeout (3 s) -> wrapper HOLDs ($(( T_HOLD - $(ev_ts "$W/wrapper.events" GO_WAIT_START) )) ms after wait start), pid $WP is '$(cat /proc/$WP/cmdline 2>/dev/null | tr '\0' ' ')', stage 2 never started: '$(ev_ts "$W/wrapper.events" STAGE2_START)'" \
  [ -n "$(ev_ts "$W/wrapper.events" HOLD)" -a "$(tr '\0' ' ' </proc/$WP/cmdline 2>/dev/null)" = "sleep infinity " -a -z "$(ev_ts "$W/wrapper.events" STAGE2_START)" ]
check a3.5 "lock held through the timeout and 4 s after: $n samples, $bad not-held, max gap $gap ms; --check-held rc=$crc" [ "$bad" -eq 0 -a "$crc" -eq 0 ]
read -r hn hbad <<<"$(hook_stats "$R/hook-gotimeout.smp" $((T_KILL+200)) $T_E)"
check a3.6 "hook loop over the same window: $hn runs, $hbad not 'already being started'; stub launches +$(( $(lines "$R/launches.log") - L0 )), hook-launch lines +$(( $(hook_launch_lines) - H0 )) (want 0)" \
  [ "$hbad" -eq 0 -a "$(lines "$R/launches.log")" = "$L0" -a "$(hook_launch_lines)" = "$H0" ]
kill "$WP" 2>/dev/null; wait_lock_free

# ═══ a5: stage 2 failure holds; sleeper kill frees; recovery handoff ═════════════
fresh_dummy; W="$R/w-fail"; touch "$R/go"; echo fail >"$R/stage2.mode"
L0=$(lines "$R/launches.log"); H0=$(hook_launch_lines)
start_samplers fail; sleep 0.3
T_KILL=$(now); kill "$DPID"; launch_wrapper 60 "$W"
wait_event "$W/wrapper.events" HOLD 25; T_HOLD=$(ev_ts "$W/wrapper.events" HOLD); : "${T_HOLD:=0}"
sleep 10; T_E=$(now); WP="$(cat "$W/wrapper.pid" 2>/dev/null)"
read -r n bad gap <<<"$(lock_stats "$R/lock-fail.smp" $T_HOLD $T_E)"
check a5.1 "stage 2 rc!=0 -> HOLD at +$(( T_HOLD - T_KILL )) ms; sampled 10 s from HOLD: $n samples, $bad not-held, max gap $gap ms; sleeper '$(tr '\0' ' ' </proc/$WP/cmdline 2>/dev/null)'" \
  [ -n "$(ev_ts "$W/wrapper.events" STAGE2_END)" -a "$bad" -eq 0 -a "$(tr '\0' ' ' </proc/$WP/cmdline 2>/dev/null)" = "sleep infinity " ]
read -r hn hbad <<<"$(hook_stats "$R/hook-fail.smp" $((T_KILL+200)) $T_E)"
BAD200=$(awk -v a=$((T_KILL+300)) '$1>a&&$2==200' "$R/health-fail.smp" | wc -l)
check a5.2 "hook loop kill..HOLD+10 s: $hn runs, $hbad not 'already being started'; stub launches +$(( $(lines "$R/launches.log") - L0 )), hook-launch lines +$(( $(hook_launch_lines) - H0 )), /health 200 samples $BAD200 (want 0)" \
  [ "$hbad" -eq 0 -a "$(lines "$R/launches.log")" = "$L0" -a "$(hook_launch_lines)" = "$H0" -a "$BAD200" -eq 0 ]
# recovery, "queue first, kill second": a second wrapper queues behind the sleeper, then the sleeper is killed
W2="$R/w-recover"; echo ok >"$R/stage2.mode"
launch_wrapper 60 "$W2"; sleep 0.4
check a5.3 "second wrapper queued behind the sleeper has not started (no LOCK_ACQUIRED yet)" [ -z "$(ev_ts "$W2/wrapper.events" LOCK_ACQUIRED)" ]
T=$(now); kill "$WP"; wait_event "$W2/wrapper.events" LOCK_ACQUIRED 10; TQ=$(ev_ts "$W2/wrapper.events" LOCK_ACQUIRED)
wait_health 25; T_H=$(now); sleep 0.5; stop_samplers
read -r n bad gap <<<"$(lock_stats "$R/lock-fail.smp" 0 99999999999999)"
check a5.4 "kill of the sleeper hands the lock to the queued wrapper in $(( TQ - T )) ms; whole run (hold + handoff + serving): $n samples, $bad not-held, max gap $gap ms; hook-launch lines +$(( $(hook_launch_lines) - H0 )) (want 0 not-held, 0 launches)" \
  [ -n "$TQ" -a "$bad" -eq 0 -a "$(hook_launch_lines)" = "$H0" ]
DPID="$(listener_pid)"

# ═══ a7: -w 2 timeout while the old holder is still shutting down ═══════════════
fresh_dummy; W="$R/w-timeout"; rm -rf "$W"; echo 4 >"$R/slow_term"; L0=$(lines "$R/launches.log"); H0=$(hook_launch_lines)
T_KILL=$(now); kill "$DPID"
( env WRAP_PORT=$PORT WRAP_LOCK="$LOCK" WRAP_STATE_DIR="$W" WRAP_FC_DIR="$S/falkor-chat" WRAP_STAGE1_CMD="bash $R/stage1.sh" WRAP_STAGE2_CMD="bash $R/stage2.sh" \
    flock -w 2 "$LOCK" bash "$WRAPPER" >>"$R/wrapper.log" 2>&1 ); frc=$?; T_TO=$(now)
DIRMADE=$([ -d "$W" ] && echo yes || echo no)
flock -n -E 75 "$LOCK" true; bare=$?
chk="$(WRAP_LOCK="$LOCK" WRAP_STATE_DIR="$W" bash "$WRAPPER" --check-held 2>&1)"; crc=$?
check a7.1 "flock -w 2 vs a holder shutting down for 4 s: exited rc=$frc after $(( T_TO - T_KILL )) ms; wrapper state dir created: $DIRMADE (want rc!=0, no)" [ "$frc" -ne 0 -a "$DIRMADE" = no ]
check a7.2 "wrapper --check-held aborts the plan: rc=$crc '$chk' (want rc 1)" [ "$crc" -eq 1 ]
res INFO a7.3 "bare 'flock -n -E 75' right after the timeout returned $bare (75 = old holder still holds, 0 = free): the plan's step-3 assertion alone is $( [ "$bare" -eq 75 ] && echo 'PASSING SPURIOUSLY' || echo 'failing' ) here"
sleep 3; wait_lock_free; rc=$?; H1=$(hook_launch_lines); out="$(run_hook)"; sleep 0.3; wait_health 10
check a7.4 "once the slow holder exits the lock is free (rc $rc) and the hook launches (+$(( $(hook_launch_lines) - H1 )) lines): the hole the timeout leaves, which --check-held catches" [ "$rc" -eq 0 -a "$(hook_launch_lines)" -eq $((H1+1)) ]
DPID="$(listener_pid)"

# ═══ a8: cleanup and containment ════════════════════════════════════════════════
kill_scratch; sleep 0.5
check a8.1 "no listener on $PORT or $FPORT" [ -z "$(ss -ltnH "sport = :$PORT" 2>/dev/null)" -a -z "$(ss -ltnH "sport = :$FPORT" 2>/dev/null)" ]
LEFT=0; for f in /proc/[0-9]*/fd/*; do [ "$(readlink "$f" 2>/dev/null)" = "$LOCK" ] && LEFT=$((LEFT+1)); done
check a8.2 "no process holds the scratch lock ($LEFT)" [ "$LEFT" -eq 0 ]
real_snapshot >"$R/real.after"
check a8.3 "real :8200/:8210 listeners, real lock (inode/mtime/size) and real state-dir listing unchanged" [ "$(cat "$R/real.before")" = "$(cat "$R/real.after")" ]
OUT="$(find "$REPO/falkor-chat" "$REPO/claude/teco" \( -name .git -o -name node_modules -o -name .venv -o -name __pycache__ \) -prune -o -type f -newer "$STAMP" -print 2>/dev/null | grep -v -E 'migrate_agent_team_wrapper.sh$|rehearse_agent_team_wrapper.sh$' | head -5)"
check a8.4 "no file under falkor-chat/ or claude/teco/ changed since the run started (ignoring the wrapper/harness; the rest of the repo is other sessions' work): ${OUT:-none}" [ -z "$OUT" ]
echo "SUMMARY pass=$NPASS fail=$NFAIL mutant=${MUTANT:-none}" | tee -a "$R/results.log"
exit "$NFAIL"
