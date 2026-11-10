#!/usr/bin/env bash
set -euo pipefail
# test-ccg-falkordb.sh — stub-harness regression test for cpg/infra/ccg-falkordb.sh.
#
# NEVER touches real Docker or Redis. `docker`, `redis-cli`, `ss` and `openssl` are stubs on PATH
# (written below into a temp dir), backed by a fake "world" of containers in that temp dir. The
# harness refuses to run if `docker` does not resolve to its own stub.
#
# Main property (review M1, docs/reviews/connected-code-graph-p0-impl.md § S01): across
#   verb      x {up, down, down --force, reset-p0 --yes}
#   override  x {none, NAME=falkordb-dev, NAME=scratch-x, PORT=6391, CONFIG_DIR=alt, combos}
#   state     x {running, exited, absent}
#   label     x {present, absent}
# no `docker stop`/`docker rm` reaches an unlabelled container, and no Redis write reaches an
# instance whose container is unlabelled (so never falkordb-dev). The stubs record violations
# themselves, at the moment they happen. The probe also runs `--throwaway up` / `--throwaway down`.
# Plus positive controls (the guards must not refuse everything), the down-guard-probes-the-
# right-instance case, status drift checks (m1/m2), reset-p0's post-flush check (m3), and P1-48:
# `up` sets and reports TIMEOUT_MAX 10000 / MAX_INFO_QUERIES 0, `status` accepts the P1 steady
# state (nonzero TIMEOUT_MAX, the four ccg_* ACL users) and flags anything else, and
# `--throwaway` refuses 6379/6390 before any docker call, refuses published/listening ports,
# picks a free one, survives a lost port race, and never acts on a container without its label.
#
# Usage: cpg/infra/tests/test-ccg-falkordb.sh [path/to/ccg-falkordb.sh]

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${1:-$HERE/../ccg-falkordb.sh}"
SCRIPT="$(cd "$(dirname "$SCRIPT")" && pwd)/$(basename "$SCRIPT")"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
STUB="$TMP/bin"
mkdir -p "$STUB"

# ------------------------------------------------------------------ stubs
cat >"$STUB/docker" <<'EOF'
#!/usr/bin/env bash
W="$STUB_WORLD"; echo "docker $*" >>"$W/log"
viol() { echo "$*" >>"$W/violations"; }
lbl() { cat "$W/c/$1/label" 2>/dev/null; }
twl() { cat "$W/c/$1/twlabel" 2>/dev/null; }
cmd="$1"; shift
case "$cmd" in
  info) exit 0 ;;
  inspect)
    name=""; fmt=""
    while [ $# -gt 0 ]; do
      case "$1" in --format|-f) fmt="$2"; shift ;; *) name="$1" ;; esac; shift
    done
    d="$W/c/$name"
    if [ ! -d "$d" ]; then echo; echo "Error: No such object: $name" >&2; exit 1; fi
    case "$fmt" in
      *State.Status*)        cat "$d/state" ;;
      *Labels*throwaway*)    echo "$(twl "$name")" ;;
      *Labels*)              echo "$(lbl "$name")" ;;
      *Config.Image*)        cat "$d/image" ;;
      *HostConfig.MemorySwap*) cat "$d/memswap" ;;
      *HostConfig.Memory*)   cat "$d/mem" ;;
      *) echo "stub docker inspect: unsupported format $fmt" >&2; exit 3 ;;
    esac ;;
  port)
    name="$1"; d="$W/c/$name"
    [ -d "$d" ] || { echo "Error: No such container: $name" >&2; exit 1; }
    [ "$(cat "$d/state")" = running ] || exit 0
    if [ -n "${2:-}" ]; then sed 's/^6379\/tcp -> //' "$d/ports"; else cat "$d/ports"; fi ;;
  ps)
    all=0; filt=""
    while [ $# -gt 0 ]; do case "$1" in -a) all=1 ;; --filter) filt="$2"; shift ;; esac; shift; done
    for d in "$W"/c/*; do
      [ -d "$d" ] || continue; n="${d##*/}"
      [ "$all" = 1 ] || [ "$(cat "$d/state")" = running ] || continue
      case "$filt" in *throwaway*) [ "$(twl "$n")" = 1 ] || continue; echo "$n	$(cat "$d/state")	" ; continue ;; esac
      sed 's/^6379\/tcp -> \(.*\)$/\1->6379\/tcp/' "$d/ports" | paste -sd, -
    done ;;
  stop)
    name="$1"; d="$W/c/$name"
    [ -d "$d" ] || { echo "Error: No such container: $name" >&2; exit 1; }
    [ "$(lbl "$name")" = 1 ] || viol "docker stop $name (unlabelled)"
    echo exited >"$d/state" ;;
  rm)
    name="$1"; d="$W/c/$name"
    [ -d "$d" ] || { echo "Error: No such container: $name" >&2; exit 1; }
    [ "$(lbl "$name")" = 1 ] || viol "docker rm $name (unlabelled)"
    [ "$(cat "$d/state")" = running ] && { echo "Error: container is running" >&2; exit 1; }
    for p in "$W"/p/*; do [ -e "$p" ] && [ "$(cat "$p")" = "$name" ] && rm -f "$p"; done
    rm -rf "$d" ;;
  run)
    name=""; label=""; twlabel=""; hp=""; image=""; mem=0; memswap=0
    tobytes() { case "$1" in *g) echo $(( ${1%g} * 1073741824 ));; *m) echo $(( ${1%m} * 1048576 ));; *) echo "$1";; esac; }
    while [ $# -gt 0 ]; do
      case "$1" in
        --name) name="$2"; shift ;;
        --name=*) name="${1#--name=}" ;;
        --label) case "$2" in ccg-falkordb.throwaway=1) twlabel=1 ;; *) label="$2" ;; esac; shift ;;
        -p) hp="$2"; shift ;;
        -e|-v) shift ;;
        --memory) mem="$(tobytes "$2")"; shift ;;
        --memory=*) mem="$(tobytes "${1#--memory=}")" ;;
        --memory-swap) memswap="$(tobytes "$2")"; shift ;;
        --memory-swap=*) memswap="$(tobytes "${1#--memory-swap=}")" ;;
        -d|--pull=*|--init|--rm) ;;
        -*) ;;
        *) image="$1" ;;
      esac; shift
    done
    [ -d "$W/c/$name" ] && { echo "Conflict: name $name in use" >&2; exit 125; }
    d="$W/c/$name"; mkdir -p "$d"
    echo running >"$d/state"
    [ "$label" = "ccg-falkordb=1" ] && echo 1 >"$d/label" || : >"$d/label"
    echo "$twlabel" >"$d/twlabel"
    ip="${hp%%:*}"; rest="${hp#*:}"; port="${rest%%:*}"
    # A port taken between the script's check and the run (W/race/<port>): like real docker,
    # the container is created but cannot start.
    if [ -e "$W/race/$port" ]; then echo created >"$d/state"; echo "Error: bind: address already in use" >&2; exit 125; fi
    # A container that starts but never answers PING (W/noping) or refuses GRAPH.CONFIG SET (W/nocfg).
    [ -e "$W/noping" ] && touch "$d/noping"; [ -e "$W/nocfg" ] && touch "$d/nocfg"
    echo "6379/tcp -> $ip:$port" >"$d/ports"
    echo "$port" >"$d/hostport"; echo "$name" >"$W/p/$port"
    echo 0 >"$d/keys"; echo 1 >"$d/auth"; echo "$image" >"$d/image"
    echo 41811 >"$d/modver"; echo 0 >"$d/tmax"; echo 1000 >"$d/miq"; echo default >"$d/acl"
    echo "$mem" >"$d/mem"; echo "$memswap" >"$d/memswap"
    echo deadbeef ;;
  *) echo "stub docker: unsupported command $cmd" >&2; exit 3 ;;
esac
EOF

cat >"$STUB/redis-cli" <<'EOF'
#!/usr/bin/env bash
W="$STUB_WORLD"
host=127.0.0.1; port=6379
while [ $# -gt 0 ]; do
  case "$1" in -h) host="$2"; shift ;; -p) port="$2"; shift ;; --no-auth-warning) ;; *) break ;; esac
  shift
done
echo "redis-cli -p $port $*" >>"$W/log"
name="$(cat "$W/p/$port" 2>/dev/null || true)"; d="$W/c/$name"
if [ -z "$name" ] || [ "$(cat "$d/state" 2>/dev/null)" != running ]; then
  echo "Could not connect to Redis at $host:$port: Connection refused" >&2; exit 1
fi
cmd="$(echo "$1" | tr a-z A-Z)"
case "$cmd" in
  FLUSHALL|FLUSHDB|SET|DEL|GRAPH.DELETE|GRAPH.QUERY)
    [ "$(cat "$d/label")" = 1 ] || echo "redis write $cmd to $name:$port (unlabelled)" >>"$W/violations" ;;
  CONFIG|GRAPH.CONFIG|ACL)
    case "$(echo "${2:-}" | tr a-z A-Z)" in SET|SETUSER|DELUSER|RESETSTAT|REWRITE)
      [ "$(cat "$d/label")" = 1 ] || echo "redis write $cmd $2 to $name:$port (unlabelled)" >>"$W/violations" ;;
    esac ;;
esac
if [ "$(cat "$d/auth")" = 1 ] && [ "${REDISCLI_AUTH:-}" != "$(cat "$W/pw")" ]; then
  echo "NOAUTH Authentication required."; exit 0
fi
k="$(cat "$d/keys")"
[ -e "$d/noping" ] && { echo "LOADING Redis is loading the dataset in memory"; exit 0; }
case "$cmd" in
  PING) echo PONG ;;
  DBSIZE) echo "$k" ;;
  FLUSHALL) [ -e "$d/flushfail" ] || echo 0 >"$d/keys"; echo OK ;;
  SET) echo $((k+1)) >"$d/keys"; echo OK ;;
  INFO)
    case "$(echo "${2:-}" | tr A-Z a-z)" in
      keyspace) printf '# Keyspace\r\n'; [ "$k" -gt 0 ] && printf 'db0:keys=%s,expires=0,avg_ttl=0\r\n' "$k" ;;
      memory) printf '# Memory\r\nused_memory_human:1.00M\r\n' ;;
    esac ;;
  CONFIG)
    case "$3" in
      maxmemory) printf 'maxmemory\n1610612736\n' ;;
      maxmemory-policy) printf 'maxmemory-policy\nnoeviction\n' ;;
      save) printf 'save\n\n' ;;
      appendonly) printf 'appendonly\nno\n' ;;
    esac ;;
  MODULE) printf 'name\ngraph\nver\n%s\npath\n/var/lib/falkordb/bin/falkordb.so\nargs\nMAX_QUEUED_QUERIES\n25\n' "$(cat "$d/modver")" ;;
  GRAPH.CONFIG)
    f=""; case "${3:-}" in TIMEOUT_MAX) f=tmax ;; MAX_INFO_QUERIES) f=miq ;; esac
    if [ -z "$f" ]; then echo "ERR unknown config ${3:-}"; exit 0; fi
    case "$(echo "$2" | tr a-z A-Z)" in
      GET) printf '%s\n%s\n' "$3" "$(cat "$d/$f")" ;;
      SET) if [ -e "$d/nocfg" ]; then echo "ERR config refused"; else echo "$4" >"$d/$f"; echo OK; fi ;;
    esac ;;
  ACL) cat "$d/acl" ;;
  *) echo "ERR unknown command '$1'" ;;
esac
exit 0
EOF

cat >"$STUB/ss" <<'EOF'
#!/usr/bin/env bash
# ss -ltnH "sport = :PORT" -> a line if some world container publishes PORT
p="$(echo "$*" | sed -n 's/.*sport = :\([0-9]*\).*/\1/p')"
[ -n "$p" ] && { [ -e "$STUB_WORLD/p/$p" ] || [ -e "$STUB_WORLD/listen/$p" ]; } && echo "LISTEN 0 4096 127.0.0.1:$p 0.0.0.0:*"
exit 0
EOF

cat >"$STUB/openssl" <<'EOF'
#!/usr/bin/env bash
cat "$STUB_WORLD/pw"
EOF
# wait_ready polls with sleep; nothing here needs real time to pass
printf '#!/usr/bin/env bash\nexit 0\n' >"$STUB/sleep"
chmod +x "$STUB"/*

export PATH="$STUB:$PATH"
[ "$(command -v docker)" = "$STUB/docker" ] || { echo "FATAL: docker is not the stub" >&2; exit 2; }
[ "$(command -v redis-cli)" = "$STUB/redis-cli" ] || { echo "FATAL: redis-cli is not the stub" >&2; exit 2; }

# ------------------------------------------------------------------ world
W=""
mkc() { # name state label hostport keys [ports-file-content]
  local d="$W/c/$1"; mkdir -p "$d"
  echo "$2" >"$d/state"; echo "$3" >"$d/label"; echo "$5" >"$d/keys"; echo "$4" >"$d/hostport"
  echo "$4" >"$d/hostport"; echo "$1" >"$W/p/$4"
  if [ -n "${6:-}" ]; then printf '%s\n' "$6" >"$d/ports"; else echo "6379/tcp -> 127.0.0.1:$4" >"$d/ports"; fi
  echo 1 >"$d/auth"; echo falkordb/falkordb:v4.18.11 >"$d/image"; echo 41811 >"$d/modver"
  echo 10000 >"$d/tmax"; echo 0 >"$d/miq"; echo default >"$d/acl"; : >"$d/twlabel"
  echo 2147483648 >"$d/mem"; echo 2147483648 >"$d/memswap"
}
new_world() {
  W="$(mktemp -d "$TMP/w.XXXX")"; mkdir -p "$W/c" "$W/p" "$W/listen" "$W/race" "$W/home/.config/ccg" "$W/alt"
  echo testpw >"$W/pw"
  echo testpw >"$W/home/.config/ccg/falkordb-admin.pass"; chmod 600 "$W/home/.config/ccg/falkordb-admin.pass"
  echo otherpw >"$W/alt/falkordb-admin.pass"
  : >"$W/log"; : >"$W/violations"
  # the shared instance: no label, no auth, 0.0.0.0 + [::]
  mkc falkordb-dev running "" 6379 7 $'6379/tcp -> 0.0.0.0:6379\n6379/tcp -> [::]:6379'
  echo 0 >"$W/c/falkordb-dev/auth"
  # some other unlabelled instance on 6391 with an empty keyspace
  mkc other-probe running "" 6391 0
  echo 0 >"$W/c/other-probe/auth"
}
run_s() { # env-assignments... -- args...
  local envs=()
  while [ "$1" != -- ]; do envs+=("$1"); shift; done; shift
  set +e
  OUT="$(env HOME="$W/home" STUB_WORLD="$W" "${envs[@]+"${envs[@]}"}" timeout 20 bash "$SCRIPT" "$@" 2>&1)"
  RC=$?
  set -e
}

PASS=0; FAIL=0; FAILS=()
ok()   { PASS=$((PASS+1)); }
bad()  { FAIL=$((FAIL+1)); FAILS+=("$1"); }
check() { if eval "$2"; then ok; else bad "$1 :: rc=$RC :: $(echo "$OUT" | tail -2 | tr '\n' ' ')"; fi; }

# ------------------------------------------------------------------ M1 coverage probe
VERBS=("up" "down" "down --force" "reset-p0 --yes" "--throwaway up" "--throwaway down")
OVERRIDES=(
  "none"
  "CCG_FALKORDB_NAME=falkordb-dev"
  "CCG_FALKORDB_NAME=scratch-x"
  "CCG_FALKORDB_PORT=6391"
  "CCG_CONFIG_DIR=@alt"
  "CCG_FALKORDB_NAME=scratch-x CCG_FALKORDB_PORT=6391"
  "CCG_FALKORDB_NAME=falkordb-dev CCG_FALKORDB_PORT=6390"
  "CCG_FALKORDB_NAME=scratch-x CCG_CONFIG_DIR=@alt"
)
N_PROBE=0
for verb in "${VERBS[@]}"; do
  for ov in "${OVERRIDES[@]}"; do
    target=ccg-falkordb
    case "$ov" in *NAME=falkordb-dev*) target=falkordb-dev ;; *NAME=scratch-x*) target=scratch-x ;; esac
    for st in running exited absent; do
      for lab in 1 ""; do
        # falkordb-dev is the real shared instance: never labelled, never absent
        if [ "$target" = falkordb-dev ] && { [ "$lab" = 1 ] || [ "$st" = absent ]; }; then continue; fi
        new_world
        if [ "$target" = falkordb-dev ]; then
          echo "$st" >"$W/c/falkordb-dev/state"
        elif [ "$st" != absent ]; then
          hp=6390; [ "$target" = scratch-x ] && hp=6395
          mkc "$target" "$st" "$lab" "$hp" 0
        fi
        envs=()
        if [ "$ov" != none ]; then
          for a in $ov; do envs+=("${a/@alt/$W/alt}"); done
        fi
        # shellcheck disable=SC2086
        run_s "${envs[@]+"${envs[@]}"}" -- $verb
        N_PROBE=$((N_PROBE+1))
        check "probe verb='$verb' ov='$ov' state=$st label='${lab:-none}'" '[ ! -s "$W/violations" ]'
        [ -s "$W/violations" ] && FAILS[-1]+=" :: $(tr '\n' ';' <"$W/violations")"
        # falkordb-dev must be exactly as it was
        if [ "$target" != falkordb-dev ]; then
          check "probe falkordb-dev intact verb='$verb' ov='$ov' state=$st label='${lab:-none}'" \
            '[ "$(cat "$W/c/falkordb-dev/state" 2>/dev/null)" = running ] && [ "$(cat "$W/c/falkordb-dev/keys")" = 7 ]'
        fi
      done
    done
  done
done

# ------------------------------------------------------------------ positive controls
new_world; mkc ccg-falkordb running 1 6390 0; run_s -- down
check "P1 down removes own empty instance" '[ $RC = 0 ] && [ ! -d "$W/c/ccg-falkordb" ]'
run_s -- down
check "P1b down again is a no-op" '[ $RC = 0 ]'

new_world; mkc ccg-falkordb running 1 6390 3; run_s -- down
check "P2 down refuses own instance with keys" '[ $RC != 0 ] && [ -d "$W/c/ccg-falkordb" ]'
run_s -- down --force
check "P2b down --force removes it" '[ $RC = 0 ] && [ ! -d "$W/c/ccg-falkordb" ]'

new_world; mkc ccg-falkordb exited 1 6390 0; run_s -- up
check "P3 up recreates own stopped instance" '[ $RC = 0 ] && [ "$(cat "$W/c/ccg-falkordb/state")" = running ] && [ "$(cat "$W/c/ccg-falkordb/label")" = 1 ]'

new_world; run_s -- up
check "P4 up from absent" '[ $RC = 0 ] && [ "$(cat "$W/c/ccg-falkordb/state")" = running ]'
run_s -- up
check "P4b up again is a no-op" '[ $RC = 0 ] && [ "$(grep -c "^docker run" "$W/log")" = 1 ]'
run_s -- status
check "P4c status of a fresh instance passes" '[ $RC = 0 ]'

new_world; mkc ccg-falkordb running 1 6390 3; run_s -- reset-p0 --yes
check "P5 reset-p0 empties own instance" '[ $RC = 0 ] && [ "$(cat "$W/c/ccg-falkordb/keys")" = 0 ]'

new_world; mkc ccg-falkordb running 1 6390 3; run_s CCG_FALKORDB_PORT=6391 -- down
check "P6 down guard counts the keys of the container it removes, not \$PORT's" '[ $RC != 0 ] && [ -d "$W/c/ccg-falkordb" ]'

new_world; mkc ccg-falkordb running "" 6390 0; run_s -- up
check "P7 up refuses a running unlabelled ccg-falkordb (no silent no-op)" '[ $RC != 0 ]'

# ------------------------------------------------------------------ status drift (m1, m2)
st_case() { # description setup-command
  new_world; mkc ccg-falkordb running 1 6390 0; eval "$2"; run_s -- status
  check "S $1 -> status fails" '[ $RC != 0 ]'
}
new_world; mkc ccg-falkordb running 1 6390 0; run_s -- status
check "S0 clean instance -> status passes" '[ $RC = 0 ]'
st_case "no password"        'echo 0 >"$W/c/ccg-falkordb/auth"'
st_case "label missing"      ': >"$W/c/ccg-falkordb/label"'
st_case "wrong image"        'echo falkordb/falkordb:latest >"$W/c/ccg-falkordb/image"'
st_case "wrong module ver"   'echo 41900 >"$W/c/ccg-falkordb/modver"'
st_case "two binding lines"  'printf "6379/tcp -> 127.0.0.1:6390\n6379/tcp -> [::1]:6390\n" >"$W/c/ccg-falkordb/ports"'
st_case "no binding"         ': >"$W/c/ccg-falkordb/ports"'
st_case "TIMEOUT_MAX 0 (restarted behind up's back)" 'echo 0 >"$W/c/ccg-falkordb/tmax"'
st_case "MAX_INFO_QUERIES default 1000" 'echo 1000 >"$W/c/ccg-falkordb/miq"'
st_case "ACL probe user left" 'printf "default\nprobe\n" >"$W/c/ccg-falkordb/acl"'
st_case "ACL user beside the P1 set" 'printf "default\nccg_build\nccg_buildx\n" >"$W/c/ccg-falkordb/acl"'
st_case "ACL ccg_raw_ with no system" 'printf "default\nccg_raw_\n" >"$W/c/ccg-falkordb/acl"'
st_case "ACL bare ccg_build (pre-§7 shared name)" 'printf "default\nccg_build\n" >"$W/c/ccg-falkordb/acl"'
st_case "ACL system with a trailing separator (shop_)" 'printf "default\nccg_build_shop_\n" >"$W/c/ccg-falkordb/acl"'
st_case "ACL system with a double separator" 'printf "default\nccg_build_a__b\n" >"$W/c/ccg-falkordb/acl"'
st_case "ACL system with a leading separator" 'printf "default\nccg_raw_-x\n" >"$W/c/ccg-falkordb/acl"'
st_case "ACL system in upper case" 'printf "default\nccg_raw_Shop\n" >"$W/c/ccg-falkordb/acl"'
st_case "ACL unknown role" 'printf "default\nccg_admin_shop\n" >"$W/c/ccg-falkordb/acl"'
st_case "ACL 49-character system name" 'printf "default\nccg_raw_%s\n" "$(printf "a%.0s" $(seq 1 49))" >"$W/c/ccg-falkordb/acl"'
st_case "no memory cap"      'echo 0 >"$W/c/ccg-falkordb/mem"'

# ------------------------------------------------------------------ reset-p0 post-check (m3)
new_world; mkc ccg-falkordb running 1 6390 3; touch "$W/c/ccg-falkordb/flushfail"; run_s -- reset-p0 --yes
check "R1 reset-p0 fails when keys remain after FLUSHALL" '[ $RC != 0 ]'
new_world; mkc ccg-falkordb running 1 6390 3; run_s CCG_CONFIG_DIR="$W/alt" -- reset-p0 --yes
check "R2 reset-p0 fails on a wrong password (NOAUTH is not 'done')" '[ $RC != 0 ]'

# ------------------------------------------------------------------ P1 instance flags (P1-48 (1), (2))
new_world; run_s -- up
check "U1 up sets TIMEOUT_MAX 10000 and MAX_INFO_QUERIES 0" '[ $RC = 0 ] && [ "$(cat "$W/c/ccg-falkordb/tmax")" = 10000 ] && [ "$(cat "$W/c/ccg-falkordb/miq")" = 0 ]'
check "U1b up reports both from GRAPH.CONFIG GET" 'echo "$OUT" | grep -q "GRAPH.CONFIG GET: TIMEOUT_MAX 10000, MAX_INFO_QUERIES 0" && grep -q "redis-cli -p 6390 GRAPH.CONFIG GET MAX_INFO_QUERIES" "$W/log"'
: >"$W/log"; run_s -- up
check "U2 up on a running instance sets nothing, still reports" '[ $RC = 0 ] && ! grep -q "GRAPH.CONFIG SET" "$W/log" && echo "$OUT" | grep -q "TIMEOUT_MAX 10000"'
# ACL users are per system, ccg_<role>_<system> (docs/plans/connected-code-graph-p1-graph.md §7).
new_world; mkc ccg-falkordb running 1 6390 0; printf 'default\nccg_build_shop\nccg_verdict_shop\nccg_engine_shop\nccg_raw_shop\nccg_build_fx-job\nccg_raw_1sys\nccg_engine_a_b-c\n' >"$W/c/ccg-falkordb/acl"; echo 20000 >"$W/c/ccg-falkordb/tmax"; run_s -- status
check "S-P1 status passes on the P1 steady state (per-system users of shop, fx-job, 1sys, a_b-c; re-derived nonzero TIMEOUT_MAX)" '[ $RC = 0 ]'
n48="$(printf 'a%.0s' $(seq 1 48))"
new_world; mkc ccg-falkordb running 1 6390 0; printf 'default\nccg_raw_%s\n' "$n48" >"$W/c/ccg-falkordb/acl"; run_s -- status
check "S-P1b a 48-character system name is accepted" '[ $RC = 0 ]'

# ------------------------------------------------------------------ --throwaway (P1-48 (3))
# every spelling: plain, and zero-padded to every length from 5 to 12 characters (P2-1)
PADDED=(); for n in 6379 6390; do for len in 5 6 7 8 9 10 11 12; do PADDED+=("$(printf "%0${len}d" "$n")"); done; done
for bad in 6379 6390 "${PADDED[@]}"; do
  for v in up status down; do
    new_world; run_s CCG_FALKORDB_PORT=$bad -- --throwaway $v
    check "T1 --throwaway $v refuses port $bad before any docker call" '[ $RC != 0 ] && ! grep -q "^docker" "$W/log" && echo "$OUT" | grep -q "refuses port $((10#$bad))"'
  done
done
for bad in 1023 65536 99999 123456 016390000 0000 0 6390x x6390 " 16395" 1e4; do
  new_world; run_s CCG_FALKORDB_PORT="$bad" -- --throwaway up
  check "T1c --throwaway refuses port value '$bad' before any docker call" '[ $RC != 0 ] && ! grep -q "^docker" "$W/log"'
done
new_world; run_s CCG_FALKORDB_PORT=0016395 -- --throwaway up
check "T1d a zero-padded free port is normalized and used (00016395 -> 16395)" '[ $RC = 0 ] && echo "$OUT" | grep -qx "CCG_FALKORDB_PORT=16395"'
new_world; run_s CCG_FALKORDB_PORT=16395 -- --throwaway reset-p0 --yes
check "T1b --throwaway reset-p0 is refused" '[ $RC != 0 ] && ! grep -q "FLUSHALL" "$W/log"'

new_world; mkc published-x running "" 16391 0; run_s CCG_FALKORDB_PORT=16391 -- --throwaway up
check "T2 explicit port already published is refused" '[ $RC != 0 ] && ! grep -q "^docker run" "$W/log"'
new_world; mkc published-x running "" 16391 0; rm -f "$W/p/16391"; run_s CCG_FALKORDB_PORT=16391 -- --throwaway up
check "T2a port published but not seen by ss (userland proxy off) is refused" '[ $RC != 0 ] && ! grep -q "^docker run" "$W/log"'
new_world; touch "$W/listen/16392"; run_s CCG_FALKORDB_PORT=16392 -- --throwaway up
check "T2b explicit port already listening (no container) is refused" '[ $RC != 0 ] && ! grep -q "^docker run" "$W/log"'
new_world; mkc ccg-falkordb-tw-16393 exited 1 16393 0; echo 1 >"$W/c/ccg-falkordb-tw-16393/twlabel"; rm -f "$W/p/16393"; run_s CCG_FALKORDB_PORT=16393 -- --throwaway up
check "T2c explicit port whose throwaway name already exists is refused" '[ $RC != 0 ] && ! grep -q "^docker run" "$W/log"'

new_world; touch "$W/listen/16390"; mkc published-x running "" 16391 0; run_s -- --throwaway up
check "T3 auto pick skips listening and published ports" '[ $RC = 0 ] && echo "$OUT" | grep -qx "CCG_FALKORDB_PORT=16392" && echo "$OUT" | grep -qx "CCG_FALKORDB_NAME=ccg-falkordb-tw-16392"'
check "T3b throwaway carries both labels, binds 127.0.0.1, gets the P1 module config" '[ "$(cat "$W/c/ccg-falkordb-tw-16392/label")" = 1 ] && [ "$(cat "$W/c/ccg-falkordb-tw-16392/twlabel")" = 1 ] && [ "$(cat "$W/c/ccg-falkordb-tw-16392/ports")" = "6379/tcp -> 127.0.0.1:16392" ] && [ "$(cat "$W/c/ccg-falkordb-tw-16392/tmax")" = 10000 ] && [ "$(cat "$W/c/ccg-falkordb-tw-16392/miq")" = 0 ]'
check "T3c stdout carries only the two variable lines" '[ "$(env HOME="$W/home" STUB_WORLD="$W" PATH="$PATH" bash "$SCRIPT" --throwaway up 2>/dev/null | grep -vc "^CCG_FALKORDB_\(NAME\|PORT\)=")" = 0 ]'
run_s CCG_FALKORDB_PORT=16392 -- --throwaway status
check "T4 --throwaway status passes on a fresh throwaway" '[ $RC = 0 ]'
echo 3 >"$W/c/ccg-falkordb-tw-16392/keys"; run_s CCG_FALKORDB_PORT=16392 -- --throwaway down
check "T5 --throwaway down removes it without --force (keys are the caller's) and its conf file" '[ $RC = 0 ] && [ ! -d "$W/c/ccg-falkordb-tw-16392" ] && [ ! -e "$W/home/.config/ccg/ccg-falkordb-tw-16392.redis.conf" ]'

new_world; touch "$W/race/16390"; run_s -- --throwaway up
check "T6 a lost port race leaves no created container and moves to the next port" '[ $RC = 0 ] && [ ! -d "$W/c/ccg-falkordb-tw-16390" ] && echo "$OUT" | grep -qx "CCG_FALKORDB_PORT=16391"'
new_world; touch "$W/race/16394"; run_s CCG_FALKORDB_PORT=16394 -- --throwaway up
check "T6b explicit port lost to a race fails and leaves nothing" '[ $RC != 0 ] && [ ! -d "$W/c/ccg-falkordb-tw-16394" ]'

new_world; mkc ccg-falkordb-tw-16400 running 1 16400 0; run_s CCG_FALKORDB_PORT=16400 -- --throwaway down
check "T7 --throwaway down refuses a container without the throwaway label" '[ $RC != 0 ] && [ -d "$W/c/ccg-falkordb-tw-16400" ]'
run_s CCG_FALKORDB_PORT=16400 -- --throwaway status
check "T7b --throwaway status flags a missing throwaway label" '[ $RC != 0 ]'
new_world; mkc ccg-falkordb running 1 6390 0; run_s CCG_FALKORDB_NAME=ccg-falkordb CCG_FALKORDB_PORT=16400 -- --throwaway down
check "T8 --throwaway with a NAME that is not tw-<port> is refused; ccg-falkordb untouched" '[ $RC != 0 ] && [ "$(cat "$W/c/ccg-falkordb/state")" = running ] && ! grep -q "^docker stop" "$W/log"'
new_world; mkc ccg-falkordb running 1 6390 0; run_s CCG_FALKORDB_NAME=ccg-falkordb -- --throwaway up
check "T8b --throwaway up with NAME but no PORT is refused" '[ $RC != 0 ] && ! grep -q "^docker run" "$W/log"'
new_world; run_s -- --throwaway down
check "T8c --throwaway down without a port is refused" '[ $RC != 0 ] && ! grep -q "^docker" "$W/log"'

new_world; run_s -- --throwaway up; run_s -- --throwaway list
check "T9 --throwaway list names the throwaway only" '[ $RC = 0 ] && echo "$OUT" | grep -q "^ccg-falkordb-tw-16390" && ! echo "$OUT" | grep -q "falkordb-dev"'

# ------------------------------------------------------------------ m3: lost module config, failed starts
new_world; mkc ccg-falkordb running 1 6390 0; echo 0 >"$W/c/ccg-falkordb/tmax"; run_s -- up
check "U3 up on a running instance whose module config was lost fails and names the remedy" '[ $RC != 0 ] && echo "$OUT" | grep -q "down --force" && [ "$(cat "$W/c/ccg-falkordb/state")" = running ]'
new_world; mkc ccg-falkordb running 1 6390 0; echo 1000 >"$W/c/ccg-falkordb/miq"; run_s -- up
check "U3b up fails on MAX_INFO_QUERIES lost too" '[ $RC != 0 ]'
new_world; touch "$W/noping"; run_s -- --throwaway up
check "T10 a throwaway that never answers PING is removed, no name printed" '[ $RC != 0 ] && [ -z "$(ls "$W/c" | grep tw)" ] && ! echo "$OUT" | grep -q "^CCG_FALKORDB_PORT="'
new_world; touch "$W/nocfg"; run_s -- --throwaway up
check "T10b a throwaway whose GRAPH.CONFIG SET fails is removed" '[ $RC != 0 ] && [ -z "$(ls "$W/c" | grep tw)" ]'
new_world; touch "$W/nocfg"; run_s -- up
check "T10c the shared instance is left in place on a failed config (status shows it)" '[ $RC != 0 ] && [ -d "$W/c/ccg-falkordb" ]'
new_world; run_s CCG_FALKORDB_PORT=16400 -- --throwaway status
check "N1 --throwaway status of an absent throwaway names the throwaway verb" '[ $RC != 0 ] && echo "$OUT" | grep -q -- "--throwaway up"'

# ------------------------------------------------------------------ report
echo "coverage probe runs: $N_PROBE"
echo "checks: $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
  printf 'FAIL: %s\n' "${FAILS[@]}"
  exit 1
fi
echo "ALL PASS"
