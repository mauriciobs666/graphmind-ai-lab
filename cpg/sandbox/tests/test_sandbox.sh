#!/usr/bin/env bash
# test_sandbox.sh — the S08 tests (egress spec docs/plans/connected-code-graph-egress-spec.md v1.3
# §10, tests 1–10) for cpg/sandbox/offline-run.sh, on both backends. On this WSL2 host the
# namespace backend refuses (N1 interop + N2 /dev/vsock); that refusal is asserted as the test,
# and per spec §11 R5 the namespace bwrap template is also run DIRECTLY (never through a runner
# bypass) for axes A1–A3, A5, A6 and the --help sweep.
#
# Tests 21-26 (egress-spec3 §10.1, P1-48b) cover the indexer entries `scip-typescript` and
# `scip-python`. They need the `indexer` image (cpg/infra/toolchain/build.sh --indexer); without it
# the tests that launch one print `NOT RUN` (never `ok`) and the suite exits 3, not 0. No test here
# runs npm, pip or uv: the fixtures are plain files, and a test that needs provisioned dependencies
# prints "NOT RUN: dependencies not provisioned".
#
# Usage: cpg/sandbox/tests/test_sandbox.sh [--runner PATH] [--only 1,2,...]
#   --runner  the runner under test (default ../offline-run.sh); used by tests/mutants.sh.
#   --only    run a subset of tests (test 4 always checks the records the run produced).
# Needs: Docker, the image ccg-toolchain:<hash> (cpg/infra/toolchain/build.sh), python3, bwrap.
# Docker side effects: only throwaway --rm containers named ccg-sbx-<auditId> and their anonymous
# local volumes, all labelled ccg.sandbox=s08; the test fails if any such volume or container
# outlives it. No network is created. Exit 0 iff every check passed.
#
# Exit 0 is never taken as proof that a tool ran: the probe must print PROBE-DONE with its full
# check count, parses must yield a CPG whose methods include the fixture's, the script run must
# print them, the sweep must see each tool's Usage line.
#
# Mutation testing (tests/mutants.sh, table in ../README.md): each mutant of the runner must
# turn the named test red.

set -uo pipefail
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$HERE/../offline-run.sh"
ONLY=""
while (($#)); do
  case "$1" in
    --runner) RUNNER=$2; shift 2 ;;
    --only) ONLY=",$2,"; shift 2 ;;
    -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
RUNNER="$(cd "$(dirname "$RUNNER")" && pwd)/$(basename "$RUNNER")"
want() { [ -z "$ONLY" ] || [[ $ONLY == *",$1,"* ]]; }

T=$(mktemp -d /tmp/ccg-s08-test.XXXXXX)
export CCG_AUDIT_DIR="$T/audit"           # production invocations (test 4 checks them all)
PROBE_AUDIT="$T/audit-probe"              # lib-mode launches (probe, sweep, test 9)
MARKER="ccg-s08-marker-$$-$RANDOM"
HOME_SENTINEL="$(getent passwd "$(id -u)" | cut -d: -f6)/.ccg-s08-sentinel-$$"
MARKER_PID=""
SUITE_T0=$(date +%s)
TMP_BEFORE=$(ls -d /tmp/ccg-sbx.* 2>/dev/null | sort)
DCFG="$(getent passwd "$(id -u)" | cut -d: -f6)/.local/state/ccg/docker-config"
DCFG_TOUCHED=0; SIG_CONTAINER=""; SIGNAL_AIDS=()
WW_BASE="$(getent passwd "$(id -u)" | cut -d: -f6)/.ccg-s08-ww-$$"   # test 14, outside /tmp
# own_aids: every auditId this run's runners used -- those in its records (production and
# library mode), plus the signalled runs of tests 12 and 13, which write none.
own_aids() {
  { cat "$CCG_AUDIT_DIR/sandbox.jsonl" "$PROBE_AUDIT/sandbox.jsonl" 2>/dev/null | sed -n 's/.*"auditId":"\([0-9a-f-]*\)".*/\1/p'
    printf '%s\n' "${SIGNAL_AIDS[@]}"; } | grep . | sort -u
}
# own_scratch: the /tmp/ccg-sbx.<auditId>.XXXXXX dirs whose auditId is one of ours (p4-m1).
own_scratch() {
  local d a
  for d in /tmp/ccg-sbx.*; do
    [ -d "$d" ] || continue
    a=$(basename "$d" | sed -n 's/^ccg-sbx\.\([0-9a-f-]\{36\}\)\.[A-Za-z0-9]*$/\1/p')
    [ -n "$a" ] && grep -qxF "$a" <<< "$1" && echo "$d"
  done
}
cleanup() {
  [ -n "$MARKER_PID" ] && kill "$MARKER_PID" 2>/dev/null
  # Restore the runner's Docker-config dir (only what this suite planted; it was verified empty).
  if [ "$DCFG_TOUCHED" = 1 ]; then
    [ -L "$DCFG" ] && rm -f "$DCFG"
    rm -f "$DCFG/config.json" 2>/dev/null; [ -d "$DCFG" ] || mkdir -m 0700 "$DCFG"; chmod 0700 "$DCFG"
  fi
  # Our own container from the signal test, if a mutant left it running.
  [ -n "$SIG_CONTAINER" ] && docker rm -f -v "$SIG_CONTAINER" >/dev/null 2>&1
  # Test 13's trials: whatever a defective runner leaked under one of their own auditIds.
  local aid x
  for aid in "${SIGNAL_AIDS[@]}"; do
    for x in $(docker ps -aq --filter "label=ccg.sandbox.auditId=$aid"); do docker rm -f -v "$x" >/dev/null 2>&1; done
    for x in $(docker volume ls -q --filter "label=ccg.sandbox.auditId=$aid"); do docker volume rm -f "$x" >/dev/null 2>&1; done
  done
  # Scratch a defective runner leaked: removed only when its name carries one of our auditIds.
  for x in $(own_scratch "$(own_aids)"); do chmod -R u+w "$x" 2>/dev/null; rm -rf "$x"; done
  rm -f "$HOME_SENTINEL"
  if [ -d "$WW_BASE" ]; then chmod -R u+w "$WW_BASE" 2>/dev/null; rm -rf "$WW_BASE"; fi
  chmod -R u+w "$T" 2>/dev/null; rm -rf "$T"
}
trap cleanup EXIT

PASS=0; FAIL=0; FAILED=(); NOTRUN=0; NOTRUN_LIST=()
ok()  { PASS=$((PASS + 1)); printf '  ok   %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); FAILED+=("$1"); printf '  FAIL %s\n' "$1"; [ -n "${2:-}" ] && printf '       %s\n' "$2"; }
notrun() { NOTRUN=$((NOTRUN + 1)); NOTRUN_LIST+=("$1"); printf '  NOT RUN %s\n' "$1"; }
chk() { local name=$1; shift; if "$@"; then ok "$name"; else bad "$name"; fi; }
section() { printf '\n== %s\n' "$1"; }

IMAGE="${CCG_TOOLCHAIN_IMAGE:-$(env -i PATH=/usr/bin:/bin "$HERE/../../infra/toolchain/build.sh" --print-tag)}"
export CCG_TOOLCHAIN_IMAGE="$IMAGE"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "image $IMAGE missing: build it with cpg/infra/toolchain/build.sh" >&2; exit 2
fi

# ------------------------------------------------------------------ record helpers
cat > "$T/rec.py" <<'PY'
import json, sys
path, expr = sys.argv[1], sys.argv[2]
recs = [json.loads(l) for l in open(path) if l.strip()]
r = recs[-1]
ok = eval(expr, {"r": r, "recs": recs})
if not ok:
    print("       record: " + json.dumps({k: r.get(k) for k in ("frontend", "refused", "refusalReason", "exitCode", "toolArgv")})[:600])
sys.exit(0 if ok else 1)
PY
NREC_EXPECTED=0
nrec() { [ -f "$CCG_AUDIT_DIR/sandbox.jsonl" ] && wc -l < "$CCG_AUDIT_DIR/sandbox.jsonl" || echo 0; }
# run_r <args...>: runs the runner, sets RC/OUT/ERR, checks exactly one new record.
run_r() {
  local before after
  before=$(nrec)
  # A hang is a red test, not a stuck suite (a mutant that admits `--cwd /` makes scip-python index the
  # whole image). timeout sends TERM; the runner's own trap then removes its container and volumes.
  timeout -k 15 "${RUN_TIMEOUT:-300}" "$RUNNER" "$@" > "$T/out.txt" 2> "$T/err.txt"; RC=$?
  OUT=$(cat "$T/out.txt"); ERR=$(cat "$T/err.txt")
  after=$(nrec)
  NREC_EXPECTED=$((NREC_EXPECTED + 1))
  NEW_RECORDS=$((after - before))
}
rec() { python3 -I "$T/rec.py" "$CCG_AUDIT_DIR/sandbox.jsonl" "$1"; }
# The argv after the runner's first `--`, NUL-separated, for the toolArgv check (m3).
expect_argv() { local a seen=0; : > "$T/expect-argv"; for a in "$@"; do ((seen)) && printf '%s\0' "$a" >> "$T/expect-argv"; [ "$a" = -- ] && seen=1; done; }
# A refused record's toolArgv is exactly the argv the caller passed after `--` (egress-spec2 §1.5).
ARGV_OK="r['toolArgv'] == open('$T/expect-argv').read().split(chr(0))[:-1]"
# expect_refusal <name> <exit> <reason> <runner args...>
expect_refusal() {
  local name=$1 code=$2 reason=$3; shift 3
  expect_argv "$@"
  run_r "$@"
  if [ "$RC" = "$code" ] && [ "$NEW_RECORDS" = 1 ] && rec "r['refused'] is True and r['refusalReason']=='$reason' and r['exitCode']==$code and r['launcherArgv']==[] and r['preconditions']['netnsDistinct'] is False and r['egressMode'] in ('offline','controlled') and $ARGV_OK" >/dev/null; then
    ok "$name"
  else
    bad "$name" "rc=$RC new_records=$NEW_RECORDS $(rec "r['refusalReason']=='$reason'" 2>&1 | head -1) stderr: ${ERR:0:300}"
  fi
}
SRC="$T/src"; OUTD="$T/out"; mkdir -p "$SRC" "$OUTD"; echo 'x = 1' > "$SRC/m.py"
C=(--backend container --mode offline)

# docker events window: no container may be created during a refusal block.
ev_start() { EV_T0=$(date +%s); sleep 1; }
ev_none() { # <name>
  sleep 1; local t1; t1=$(date +%s)
  local ev; ev=$(docker events --since "$EV_T0" --until "$t1" --filter type=container --filter event=create --format '{{.Actor.Attributes.name}}' 2>&1)
  # Only containers this run's runner created: ccg-sbx-<auditId> with an auditId from this run's
  # own records (other agents may be running containers of their own concurrently).
  local ids; ids=$(sed -n 's/.*"auditId":"\([0-9a-f-]*\)".*/\1/p' "$CCG_AUDIT_DIR/sandbox.jsonl" 2>/dev/null)
  ev=$(printf '%s\n' "$ev" | sed -n 's/^ccg-sbx-//p' | grep -xF -f <(printf '%s\n' $ids) || true)
  if [ -z "$ev" ]; then ok "$1: docker events shows no container created"; else bad "$1: containers were created" "$ev"; fi
}

# lib-mode launch (test-only entry point): lib_launch <backend> <src> <out> <sdir> <wd> -- tool...
lib_launch() {
  local be=$1 s=$2 o=$3 sd=$4 wd=$5; shift 6
  ( export CCG_AUDIT_DIR="$PROBE_AUDIT"
    . "$RUNNER"
    sbx_init
    [ -n "${LIB_INDEXER:-}" ] && SBX_INDEXER=1     # tests 23/24/26: the indexer pass-list and label check
    if [ "$be" = container ]; then sbx_resolve_backend container || { echo "LIB-REFUSED $SBX_REFUSAL_DETAIL"; sbx_cleanup; exit 99; }
    else SBX_BACKEND=namespace; SBX_HOST_JOERN=$(sbx_host_joern); SBX_HOST_JDK=$(sbx_host_jdk); fi
    SBX_RECORD_FRONTEND=lib-launch
    rc=0; sbx_launch "$s" "$o" "$sd" "$wd" -- "$@" || rc=$?
    printf '%s\0' "${SBX_LAUNCHER[@]}" > "$T/last-launcher"
    printf '%s\n' "${SBX_ENV_KEYS[@]}" > "$T/last-envkeys"
    sbx_write_record "$rc" ""; sbx_cleanup; exit "$rc" )
}

# ================================================================== test 8: refusal detection
if want 8; then
  section "test 8: refusal detection (interop_state, N1, N2, C1, C2 fixtures)"
  FX="$T/binfmt"
  wslbody=$'enabled\ninterpreter /init\nflags: P\noffset 0\nmagic 4d5a'
  mkfx() { rm -rf "$FX/$1"; mkdir -p "$FX/$1"; }
  put() { printf '%s\n' "$3" > "$FX/$1/$2"; }
  mkfx F1; put F1 status enabled; put F1 WSLInterop-late "$wslbody"; put F1 register ""
  mkfx F2; put F2 status enabled; put F2 WSLInterop "$wslbody"
  mkfx F3; put F3 status enabled; put F3 winexe $'enabled\ninterpreter /usr/bin/wine\nflags: \noffset 0\nmagic 4d5a'
  mkfx F4; put F4 status enabled; put F4 pe $'enabled\ninterpreter /opt/x/run\nflags: \nextension .exe'
  mkfx F5; put F5 status enabled; put F5 x $'enabled\ninterpreter /init\nflags: \noffset 0\nmagic 7f454c46'
  mkfx F6; put F6 status enabled; put F6 WSLInterop-late "${wslbody/flags: P/flags: PF}"
  mkfx F7; put F7 status enabled; put F7 WSLInterop-late "${wslbody/enabled/disabled}"
  mkfx F8; put F8 status disabled; put F8 WSLInterop-late "$wslbody"
  mkfx F9; put F9 status enabled; put F9 python3.12 $'enabled\ninterpreter /usr/bin/python3.12\nflags: \noffset 0\nmagic cb0d0d0a'
  mkfx F10
  # F13 (added beyond §7.1.1's table): rule (d) isolated — a WSLInterop* name, no PE magic, no /init.
  mkfx F13; put F13 status enabled; put F13 WSLInterop-late $'enabled\ninterpreter /usr/bin/other\nflags: P\noffset 0\nmagic 7f454c46'
  # F14: rule (b) with an upper-case extension.
  mkfx F14; put F14 status enabled; put F14 pe $'enabled\ninterpreter /opt/x/run\nflags: F\nextension .EXE'
  WSL=5.15.167.4-microsoft-standard-WSL2; GEN=6.8.0-generic
  # rows: id dir kernel want_state want_ns(start|refuse) want_c1(pass|refuse)
  rows=("F1 F1 $WSL enabled refuse pass" "F2 F2 $WSL enabled refuse pass" "F3 F3 $GEN enabled refuse pass"
        "F4 F4 $GEN enabled refuse pass" "F5 F5 $GEN enabled refuse pass" "F6 F6 $WSL enabled refuse refuse"
        "F7 F7 $WSL disabled start pass" "F8 F8 $WSL disabled start pass" "F9 F9 $GEN disabled start pass"
        "F10 F10 $WSL unreadable refuse pass" "F11 missing 4.4.0-19041-Microsoft unreadable refuse pass"
        "F12 missing $GEN unreadable start pass" "F13 F13 $GEN enabled refuse pass" "F14 F14 $GEN enabled refuse refuse")
  for row in "${rows[@]}"; do
    read -r id dir kr ws wn wc <<< "$row"
    got=$( . "$RUNNER"; sbx_interop_scan "$FX/$dir" "$kr"; st=$SBX_INTEROP_STATE
      ns=start; sbx_n1_refuses "$st" "$kr" && ns=refuse
      sbx_vsock_present "$T/no-vsock" && ns=refuse          # row isolates N1: vsock absent
      c1=pass; [ "$SBX_PE_PINNED" = 1 ] && c1=refuse
      echo "$st $ns $c1" )
    if [ "$got" = "$ws $wn $wc" ]; then ok "$id -> $got"; else bad "$id" "got '$got', want '$ws $wn $wc'"; fi
  done
  c2() { ( . "$RUNNER"; sbx_c2_check "$1" >/dev/null ) && echo pass || echo refuse; }
  for pair in 'pass|[name=seccomp,profile=builtin name=cgroupns]' 'pass|["name=seccomp,profile=builtin","name=cgroupns"]' \
              'refuse|[name=seccomp,profile=/etc/docker/seccomp.json name=cgroupns]' 'refuse|[name=seccomp,profile=unconfined]' \
              'refuse|[name=apparmor name=cgroupns]' 'refuse|'; do
    w=${pair%%|*}; in=${pair#*|}
    g=$(c2 "$in"); if [ "$g" = "$w" ]; then ok "C2 '$in' -> $g"; else bad "C2 '$in'" "got $g want $w"; fi
  done
  # N2 parameterised
  ( . "$RUNNER"; touch "$T/fake-vsock"; sbx_vsock_present "$T/fake-vsock" && ! sbx_vsock_present "$T/no-vsock" ) && ok "N2 vsock detection" || bad "N2 vsock detection"
  # Host row, for real, through the production runner.
  expect_refusal "Host row: namespace refuses (backend-precondition, 69)" 69 backend-precondition \
    --backend namespace --mode offline --frontend pysrc2cpg --source "$SRC" --output "$OUTD" -- /src
  chk "Host row: preconditions interopPE=enabled, vsock=present" rec "r['preconditions']['interopPE']=='enabled' and r['preconditions']['vsock']=='present' and r['backend']=='namespace'"
  # n4: this is a LAUNCH, not a refusal: `--fetch-dependencies-x` has no list entry for pysrc2cpg,
  # so the frontend runs (and rejects the option itself). It is here only for the record's
  # container preconditions, which are written after a real launch.
  run_r "${C[@]}" --frontend pysrc2cpg --source "$SRC" --output "$OUTD" -- --fetch-dependencies-x
  chk "Host row: container C1 passes (binfmtPinned false), C2 builtin, vsock blocked" rec "r['preconditions']['binfmtPinned'] is False and r['preconditions']['seccomp']=='builtin' and r['preconditions']['vsock']=='blocked' and r['preconditions']['interopPE']=='enabled'"
  # m1: auto resolves to container when container can start (list-independent refusal) ...
  expect_refusal "auto with a working container backend resolves to container" 64 unknown-frontend \
    --backend auto --mode offline --frontend sh --source "$SRC" --output "$OUTD" -- /src
  chk "  record backend = container" rec "r['backend']=='container'"
  # ... and falls back to namespace when container cannot (no such image), which refuses here.
  CCG_TOOLCHAIN_IMAGE=ccg-toolchain:nonexistent-s08 expect_refusal "auto falls back to namespace when the container image is missing (69)" 69 backend-precondition \
    --backend auto --mode offline --frontend pysrc2cpg --source "$SRC" --output "$OUTD" -- /src
  chk "  record backend = namespace, interopPE=enabled, vsock=present" rec "r['backend']=='namespace' and r['preconditions']['interopPE']=='enabled' and r['preconditions']['vsock']=='present'"
fi

# ================================================================== test 7: env pass-list text half
if want 7; then
  section "test 7: pass-list ∩ forbidden names = ∅ (text half)"
  offline_keys=$( . "$RUNNER"; SBX_IMG_PATH=x SBX_IMG_JAVA_HOME=x; sbx_env_pairs container; for kv in "${SBX_ENV[@]}"; do echo "${kv%%=*}"; done )
  ns_keys=$( . "$RUNNER"; SBX_HOST_JDK=x; sbx_env_pairs namespace; for kv in "${SBX_ENV[@]}"; do echo "${kv%%=*}"; done )
  ctl_keys=$(printf '%s\n' $offline_keys MAVEN_HOME GRADLE_HOME HTTPS_PROXY HTTP_PROXY)   # §7.3.4, controlled (P2)
  forb_common='ENABLE_DEPENDENCY_RESOLVER_V2 JAVASRC_FETCH_DEPENDENCIES JAVASRC_JDK_PATH JAVA_TOOL_OPTIONS _JAVA_OPTIONS JDK_JAVA_OPTIONS GOPROXY GOFLAGS GOPATH NUGET_* WSL_INTEROP WSLENV'
  hits() { local keys=$1 pats=$2 k p; for k in $keys; do for p in $pats; do [[ $k == $p ]] && echo "$k"; done; done; }
  chk "offline pass-list (container) is the 13 §7.3.4 keys" test "$(echo $offline_keys)" = "PATH JAVA_HOME HOME TMPDIR LANG LC_ALL JAVA_OPTS GOTOOLCHAIN GOROOT DOTNET_ROOT DOTNET_CLI_TELEMETRY_OPTOUT DOTNET_NOLOGO DOTNET_SKIP_FIRST_TIME_EXPERIENCE"
  chk "namespace pass-list has the same keys" test "$(echo $ns_keys)" = "$(echo $offline_keys)"
  chk "offline ∩ forbidden(offline) = ∅" test -z "$(hits "$offline_keys" "$forb_common *_PROXY NO_PROXY")"
  chk "controlled ∩ forbidden(controlled) = ∅" test -z "$(hits "$ctl_keys" "$forb_common")"
  chk "HTTPS_PROXY is forbidden offline (the pattern bites)" test -n "$(hits "HTTPS_PROXY" "*_PROXY")"
  vals=$( . "$RUNNER"; SBX_IMG_PATH=x SBX_IMG_JAVA_HOME=x; sbx_env_pairs container; printf '%s\n' "${SBX_ENV[@]}" | grep -vE '^(PATH|JAVA_HOME|GOROOT|DOTNET_ROOT)=' | sort | paste -sd' ' )
  chk "M3 (text): fixed §7.3.4 values (GOTOOLCHAIN=local, DOTNET_*=1, ...)" test "$vals" = "DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 GOTOOLCHAIN=local HOME=/home/ccg JAVA_OPTS=-Xmx2g LANG=C.UTF-8 LC_ALL=C.UTF-8 TMPDIR=/tmp"
  chk "ENABLE_DEPENDENCY_RESOLVER_V2 never on a pass-list" test -z "$(hits "$offline_keys $ctl_keys" ENABLE_DEPENDENCY_RESOLVER_V2)"
  # §7.6 indexer entries (P1-48b): the text half of A6 (b) for the indexer pass-list.
  idx_pairs=$( . "$RUNNER"; SBX_INDEXER=1; SBX_IMG_PATH=x; SBX_IMG_JAVA_HOME=x; sbx_env_pairs container; printf '%s\n' "${SBX_ENV[@]}" )
  idx_keys=$(sed 's/=.*//' <<< "$idx_pairs")
  forb_idx='NODE_OPTIONS NODE_PATH NODE_EXTRA_CA_CERTS NODE_PRESERVE_SYMLINKS npm_config_* NPM_CONFIG_* PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONUSERBASE PIP_* UV_* LD_PRELOAD BASH_ENV'
  chk "indexer pass-list is the 13 offline keys plus PYTHONSAFEPATH" test "$(echo $idx_keys)" = "$(echo $offline_keys) PYTHONSAFEPATH"
  chk "indexer PYTHONSAFEPATH has the value 1" grep -qx 'PYTHONSAFEPATH=1' <<< "$idx_pairs"
  chk "indexer pass-list ∩ forbidden(offline + indexer) = ∅" test -z "$(hits "$idx_keys" "$forb_common *_PROXY NO_PROXY $forb_idx")"
  chk "the indexer-only forbidden patterns bite (PIP_INDEX_URL, npm_config_registry)" test -n "$(hits "PIP_INDEX_URL npm_config_registry" "$forb_idx")"
  chk "a non-indexer pass-list never carries PYTHONSAFEPATH" test -z "$(hits "$offline_keys $ns_keys" PYTHONSAFEPATH)"
fi

# ================================================================== test 2: forbidden flags refused before launch
if want 2; then
  section "test 2: forbidden flags and launcher options refused before launch (exit 64, no container)"
  ev_start
  declare -A LANG_OF=([javasrc2cpg]=javasrc [kotlin2cpg]=kotlin [csharpsrc2cpg]=csharpsrc [gosrc2cpg]=golang)
  while IFS=$'\t' read -r -u 3 fe fl fin why; do
    [[ -z $fe || $fe == \#* ]] && continue
    if [ "$fe" = joern ]; then
      echo 'println("x")' > "$T/ok.sc"
      for form in "$fl" "$fl=x"; do
        expect_refusal "joern $form" 64 forbidden-flag "${C[@]}" --frontend joern --script-file "$T/ok.sc" --source "$SRC" --output "$OUTD" -- --script /scripts/ok.sc "$form" x
        chk "  hit recorded" rec "'$form' in r['flagCheck']['forbiddenHits']"
      done
      expect_refusal "joern --param v=$fl (value is audited)" 64 forbidden-flag "${C[@]}" --frontend joern --script-file "$T/ok.sc" --source "$SRC" --output "$OUTD" -- --script /scripts/ok.sc --param "v=$fl"
      continue
    fi
    for form in "$fl" "$fl=value" "${fl^^}"; do
      expect_refusal "$fe $form" 64 forbidden-flag "${C[@]}" --frontend "$fe" --source "$SRC" --output "$OUTD" -- /src "$form"
      chk "  forbiddenHits holds '$form'" rec "'$form' in r['flagCheck']['forbiddenHits']"
    done
    expect_refusal "$fe $fl after --" 64 forbidden-flag "${C[@]}" --frontend "$fe" --source "$SRC" --output "$OUTD" -- /src -- "$fl"
    if [ -n "${LANG_OF[$fe]:-}" ]; then
      expect_refusal "joern-parse --language ${LANG_OF[$fe]} --frontend-args $fl" 64 forbidden-flag "${C[@]}" --frontend joern-parse --source "$SRC" --output "$OUTD" -- /src --language "${LANG_OF[$fe]}" --frontend-args "$fl"
      expect_refusal "joern-parse --language=${LANG_OF[$fe]^^} --frontend-args $fl=1" 64 forbidden-flag "${C[@]}" --frontend joern-parse --source "$SRC" --output "$OUTD" -- /src "--language=${LANG_OF[$fe]^^}" --frontend-args "$fl=1"
    fi
  done 3< "$(dirname "$RUNNER")/forbidden-flags.txt"
  for opt in -Dhttps.proxyHost=x -J-Dfoo=1 -J-Xmx1g-x '-J-Xmx1g -Dx' -J-Xss1m-x -agentlib:jdwp=transport=dt_socket -agentpath:/x.so \
             -javaagent:/x.jar -XX:+UnlockDiagnosticVMOptions -jvm-debug -main -java-home -J-Djava.io.tmpdir=/tmp/n; do
    expect_refusal "launcher option '$opt'" 64 forbidden-flag "${C[@]}" --frontend pysrc2cpg --source "$SRC" --output "$OUTD" -- /src "$opt"
    expect_refusal "launcher option '$opt' via joern-parse --frontend-args" 64 forbidden-flag "${C[@]}" --frontend joern-parse --source "$SRC" --output "$OUTD" -- /src --language pythonsrc --frontend-args "$opt"
  done
  expect_refusal "caller-supplied -o is refused (runner owns the output stem)" 64 forbidden-flag "${C[@]}" --frontend pysrc2cpg --source "$SRC" --output "$OUTD" -- /src -o /out/x.bin
  expect_refusal "joern-parse without --language" 64 unknown-frontend "${C[@]}" --frontend joern-parse --source "$SRC" --output "$OUTD" -- /src
  expect_refusal "joern-parse with two --language" 64 unknown-frontend "${C[@]}" --frontend joern-parse --source "$SRC" --output "$OUTD" -- /src --language pythonsrc --language=jssrc
  for l in java python JAVA rubysrc php llvm fuzzy_test_lang; do
    expect_refusal "joern-parse --language $l" 64 unknown-frontend "${C[@]}" --frontend joern-parse --source "$SRC" --output "$OUTD" -- /src --language "$l"
  done
  for fe in rubysrc2cpg php2cpg joern-scan sh; do
    expect_refusal "frontend $fe outside the closed set" 64 unknown-frontend "${C[@]}" --frontend "$fe" --source "$SRC" --output "$OUTD" -- /src
  done
  expect_refusal "--mode controlled is not implemented (P2)" 64 mode-not-implemented --backend container --mode controlled --frontend pysrc2cpg --source "$SRC" --output "$OUTD" -- /src
  expect_refusal "unknown runner option" 64 usage "${C[@]}" --frontend pysrc2cpg --source "$SRC" --output "$OUTD" --env X=1 -- /src
  # M2: every invocation writes exactly one record, even when the runner's own argv is broken.
  expect_refusal "M2: no arguments at all" 64 usage
  expect_refusal "M2: missing --backend" 64 usage --mode offline --frontend pysrc2cpg --source "$SRC" --output "$OUTD" -- /src
  expect_refusal "M2: invalid --backend" 64 usage --backend xyz --mode offline --frontend pysrc2cpg --source "$SRC" --output "$OUTD" -- /src
  expect_refusal "M2: --backend without a value" 64 usage --backend
  expect_refusal "M2: missing --" 64 usage "${C[@]}" --frontend pysrc2cpg --source "$SRC" --output "$OUTD"
  # m4: an invalid --mode is refused and never lands in egressMode.
  expect_refusal "m4: invalid --mode" 64 usage --backend container --mode foo --frontend pysrc2cpg --source "$SRC" --output "$OUTD" -- /src
  chk "  egressMode stays in the enum (offline)" rec "r['egressMode']=='offline'"
  # n1: the runner's environment is not an input beyond CCG_AUDIT_DIR / CCG_TOOLCHAIN_IMAGE.
  DOCKER_HOST=unix:///nonexistent-s08.sock DOCKER_CONTEXT=nonexistent-s08 DOCKER_CONFIG="$T/no-docker-config" \
    DOCKER_TLS_VERIFY=1 DOCKER_CERT_PATH="$T/no-certs" expect_refusal "n1: DOCKER_HOST/DOCKER_CONTEXT in the caller's env are ignored" 64 forbidden-flag \
    "${C[@]}" --frontend javasrc2cpg --source "$SRC" --output "$OUTD" -- /src --fetch-dependencies
  chk "  docker info still answered (seccomp=builtin, backend container)" rec "r['preconditions']['seccomp']=='builtin' and r['backend']=='container'"
  printf 'touch %q\n' "$T/bash-env-ran" > "$T/bash-env.sh"
  BASH_ENV="$T/bash-env.sh" expect_refusal "n1: BASH_ENV runs nothing (direct exec)" 64 unknown-frontend "${C[@]}" --frontend sh --source "$SRC" --output "$OUTD" -- /src
  chk "  BASH_ENV file was not executed" test ! -e "$T/bash-env-ran"
  # egress-spec2 §1.5: a refused record carries the caller's argv verbatim, token included.
  expect_refusal "refused toolArgv is the caller's argv verbatim (@AUDIT_ID@ unsubstituted)" 64 forbidden-flag \
    "${C[@]}" --frontend pysrc2cpg --source "$SRC" --output "$OUTD" -- /src '-Dx=@AUDIT_ID@'
  # egress-spec2 §1.5: --help exits 0 and is the one invocation that writes no record.
  # p2-n2 / p3-m2: run under a private /tmp (bwrap; $T and the runner's dir bound back), so any
  # /tmp/ccg-sbx.* scratch this invocation leaks lands in $PTMP and is attributable to it alone.
  PTMP="$T/private-tmp"; mkdir -p "$PTMP"
  before=$(nrec)
  bwrap --dev-bind / / --bind "$PTMP" /tmp --bind "$T" "$T" --ro-bind "$(dirname "$RUNNER")" "$(dirname "$RUNNER")" -- "$RUNNER" --help > /dev/null 2>&1; hrc=$?
  after=$(nrec)
  chk "--help exits 0 and writes no record (rc $hrc, $((after - before)) new)" test "$hrc" = 0 -a "$after" = "$before"
  leaked=$(cd "$PTMP" && ls -d ccg-sbx.* 2>/dev/null)
  chk "--help leaves no /tmp/ccg-sbx.* scratch (private /tmp: ${leaked:-none})" test -z "$leaked"
  ev_none "test 2"
  # M4 (pinned by name): the three gate-reviewed allowed flags must stay accepted, whatever the files say.
  for pair in "c2cpg --compilation-database" "javasrc2cpg --delombok-mode" "jimple2cpg --full-resolver"; do
    set -- $pair
    got=$( . "$RUNNER"; sbx_load_lists; SBX_HITS=(); SBX_ALLOWED_HITS=(); sbx_audit_elements offline "$1" -- /src "$2"; echo "hits=${#SBX_HITS[@]} allowed=${SBX_ALLOWED_HITS[*]}" )
    chk "M4: reviewed allowed flag $1 $2 is accepted and recorded ($got)" test "$got" = "hits=0 allowed=$2"
  done
  set --
  # M4: every allowed entry is pinned — accepted with no hit, and recorded in allowedHits — in
  # three forms, for its frontend directly and (when one maps to it) through joern-parse.
  declare -A LANG_FOR=([c2cpg]=newc [javasrc2cpg]=javasrc [jimple2cpg]="" [kotlin2cpg]=kotlin [csharpsrc2cpg]=csharpsrc [gosrc2cpg]=golang [pysrc2cpg]=pythonsrc [jssrc2cpg]=jssrc)
  n_allowed=0
  while IFS=$'\t' read -r -u 3 fe fl why; do
    [[ -z $fe || $fe == \#* ]] && continue
    n_allowed=$((n_allowed + 1))
    for form in "$fl" "$fl=v" "${fl^^}"; do
      got=$( . "$RUNNER"; sbx_load_lists; SBX_HITS=(); SBX_ALLOWED_HITS=()
        sbx_audit_elements offline "$fe" -- /src "$form"; echo "hits=${#SBX_HITS[@]} allowed=${SBX_ALLOWED_HITS[*]}" )
      chk "M4: allowed $fe $form -> no hit, in allowedHits ($got)" test "$got" = "hits=0 allowed=$form"
    done
    if [ -n "${LANG_FOR[$fe]:-}" ]; then
      got=$( . "$RUNNER"; sbx_load_lists; SBX_HITS=(); SBX_ALLOWED_HITS=()
        sbx_audit_elements offline "$fe joern-parse" -- /src --language "${LANG_FOR[$fe]}" --frontend-args "$fl"; echo "hits=${#SBX_HITS[@]} allowed=${SBX_ALLOWED_HITS[*]}" )
      chk "M4: allowed $fe $fl via joern-parse --frontend-args ($got)" test "$got" = "hits=0 allowed=$fl"
    fi
  done 3< "$(dirname "$RUNNER")/allowed-flags.txt"
  chk "M4: allowed-flags.txt has its 3 reviewed entries" test "$n_allowed" = 3
  # Positive controls of the audit itself (no launch): the exceptions are full-match only.
  pos=$( . "$RUNNER"; sbx_load_lists; SBX_HITS=(); SBX_ALLOWED_HITS=()
    sbx_audit_elements offline "joern-parse c2cpg" -- /src --language newc -J-Xmx2g -J-Xss4m --frontend-args --compilation-database /src/cc.json
    echo "hits=${#SBX_HITS[@]} allowed=${SBX_ALLOWED_HITS[*]}" )
  chk "audit accepts -J-Xmx2g/-J-Xss4m and records the allowed c2cpg flag ($pos)" test "$pos" = "hits=0 allowed=--compilation-database"
  lang_ok=$( . "$RUNNER"; for l in pythonsrc jssrc javascript javasrc kotlin csharpsrc csharp golang newc c; do printf '%s ' "$(sbx_language_frontend "$l")"; done )
  chk "language table maps the FR-1 values" test "$lang_ok" = "pysrc2cpg jssrc2cpg jssrc2cpg javasrc2cpg kotlin2cpg csharpsrc2cpg csharpsrc2cpg gosrc2cpg c2cpg c2cpg "
fi

# ================================================================== test 10: joern script mode
if want 10; then
  section "test 10: joern script mode"
  echo 'println("hello")' > "$T/ok.sc"
  ev_start
  J=("${C[@]}" --frontend joern --source "$SRC" --output "$OUTD")
  for bad_args in "--import /scripts/ok.sc" "--runBefore x" "--classpathEntry /x" "-d a:b:1" "--dep a:b:1" "-r http://x" "--repo http://x" \
                  "--server" "--add-plugin /x.zip" "/out/cpg.bin" "-J-Xmx1g" "-Dx=y" "--param 1bad=v" "--param novalue" "--param" "--verbose" "--nocolors"; do
    # shellcheck disable=SC2086
    expect_refusal "10a: --script /scripts/ok.sc $bad_args" 64 forbidden-flag "${J[@]}" --script-file "$T/ok.sc" -- --script /scripts/ok.sc $bad_args
  done
  expect_refusal "10a: --script path other than /scripts/<basename>" 64 forbidden-flag "${J[@]}" --script-file "$T/ok.sc" -- --script /src/ok.sc
  expect_refusal "10a: --script /scripts/other.sc" 64 forbidden-flag "${J[@]}" --script-file "$T/ok.sc" -- --script /scripts/other.sc
  expect_refusal "10a: missing --script" 64 forbidden-flag "${J[@]}" --script-file "$T/ok.sc" -- --param a=b
  expect_refusal "10a: missing --script-file" 64 forbidden-flag "${J[@]}" -- --script /scripts/ok.sc
  expect_refusal "10a: --mode controlled" 64 unknown-frontend --backend container --mode controlled --frontend joern --source "$SRC" --output "$OUTD" --script-file "$T/ok.sc" -- --script /scripts/ok.sc
  # M1: replpp's UsingDirectives.scanFor is line.trim().startsWith("//> using"), and Java's
  # String.trim() strips EVERY char <= U+0020. The oracle is that rule, not a list of shapes:
  # every leading byte 0x00..0x20 (plus 0x7f, refused to fail closed) must be refused.
  nb=0; nbad=0
  for b in $(seq 0 32) 127; do
    hex=$(printf '%02x' "$b"); f="$T/lead-$hex.sc"
    printf "\\x$hex"'//> using dep com.michaelpollmeier:versionsort:1.0.7\nprintln("x")\n' > "$f"
    [ "$(head -c1 "$f" | od -An -tx1 | tr -d ' \n')" = "$hex" ] || { bad "10b: fixture lead-$hex not written"; continue; }
    expect_refusal "10b: leading byte 0x$hex + //> using dep" 64 forbidden-directive "${J[@]}" --script-file "$f" -- --script "/scripts/lead-$hex.sc"
    nb=$((nb + 1))
  done
  chk "10b: leading-byte probe covered 34 bytes (0x00-0x20, 0x7f)" test "$nb" = 34
  i=0
  for body in '//> using resolver https://x' '//> using file other.sc' '  //>   using dep a:b:1' '//> USING DEP a:b:1' \
              $'\x01\x1f \t\v\f//> using dep a:b:1' $'println(1)\r//> using dep a:b:1' $'/*\n//> using dep a:b:1\n*/' \
              $'x\r//> using dep com.michaelpollmeier:versionsort:1.0.7' $'val x = 1 \r//> using dep a:b:1'; do
    i=$((i + 1)); printf '%s\nprintln("x")\n' "$body" > "$T/d$i.sc"
    expect_refusal "10b: directive shape #$i $(printf '%q' "${body:0:28}")" 64 forbidden-directive "${J[@]}" --script-file "$T/d$i.sc" -- --script "/scripts/d$i.sc"
    chk "  forbiddenHits names the //> line" rec "any('//>' in h for h in r['flagCheck']['forbiddenHits'])"
  done
  ev_none "test 10(a,b)"
  # Inert forms are not refused at the directive step (they hit the forbidden list only if they contain a flag).
  # Negative controls (no launch): trim() leaves these alone, so they are not directive lines.
  for nc in 'println("a //> b")' $'\xef\xbb\xbf//> using dep a:b:1' '///> using dep a:b:1' 'x //> using dep a:b:1' 'val x = 1 //> using dep a:b:1'; do
    printf '%s\nprintln("x")\n' "$nc" > "$T/inert.sc"
    inert=$( . "$RUNNER"; sbx_directive_lines "$T/inert.sc" >/dev/null 2>&1 && echo clean || echo refused )
    chk "10b: not a directive line: $(printf '%q' "${nc:0:24}")" test "$inert" = clean
  done
fi

# ================================================================== test 3 (+10c): parse every FR-1 language, check content
if want 3 || want 10; then
  section "test 3: joern-parse of each FR-1 language through the runner; CPG content checked"
  FXD="$T/fx"; CPGS="$T/cpgs"; mkdir -p "$CPGS"
  mkfile() { mkdir -p "$FXD/$1"; printf "$3" > "$FXD/$1/$2"; }
  mkfile pythonsrc m.py 'def f(x):\n    return x + 1\n'
  mkfile jssrc m.js 'function f(x) { return x + 1; }\n'
  mkfile javasrc A.java 'class A { int f(int x) { return x + 1; } }\n'
  mkfile kotlin m.kt 'fun f(x: Int): Int = x + 1\n'
  mkfile csharpsrc A.cs 'class A { int F(int x) { return x + 1; } }\n'
  mkfile golang m.go 'package main\n\nfunc f(x int) int { return x + 1 }\n\nfunc main() { f(1) }\n'
  mkfile newc m.c 'int f(int x) { return x + 1; }\n'
  declare -A WANT=([pythonsrc]=f [jssrc]=f [javasrc]=f [kotlin]=f [csharpsrc]=F [golang]=f [newc]=f)
  declare -A ID_OF=()
  files=""
  for l in pythonsrc jssrc javasrc kotlin csharpsrc golang newc; do
    s=$(date +%s)
    run_r "${C[@]}" --frontend joern-parse --codebase "fixture-$l" --source "$FXD/$l" --output "$CPGS" -- /src --language "$l"
    dt=$(( $(date +%s) - s ))
    id=$(python3 -I -c 'import json,sys; print(json.loads(open(sys.argv[1]).readlines()[-1])["auditId"])' "$CCG_AUDIT_DIR/sandbox.jsonl")
    ID_OF[$l]=$id
    f="$CPGS/$id.cpg.bin"; sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$RC" = 0 ] && ((dt < 120)) && ! grep -q UnknownHostException <<< "$OUT$ERR" && ((sz > 20000)); then
      ok "$l: exit 0 in ${dt}s, no UnknownHostException, CPG /out/<auditId>.cpg.bin = $sz B"
    else bad "$l parse" "rc=$RC ${dt}s size=$sz UHE=$(grep -c UnknownHostException <<< "$OUT$ERR") ${ERR: -300}"; fi
    files+="${files:+;}/out/$id.cpg.bin"
  done
  # P3-7 / plan §2 D15: the durable content check, the one S11, S25, S28 and S39 reuse.
  CHECK_SC="$(dirname "$RUNNER")/scripts/cpg-content-check.sc"
  run_r "${C[@]}" --frontend joern --script-file "$CHECK_SC" --source "$SRC" --output "$CPGS" -- --script /scripts/cpg-content-check.sc --param "cpgFiles=$files"
  chk "10c: cpg-content-check.sc runs through the runner, all 7 CPGs ok (exit $RC)" test "$RC" = 0 -a "$(grep -c '^CONTENT .*=ok$' <<< "$OUT")" = 7
  CHECK_OUT=$OUT
  for l in pythonsrc jssrc javasrc kotlin csharpsrc golang newc; do
    n=$(sed -n "s#^METHOD_COUNT /out/${ID_OF[$l]}.cpg.bin=##p" <<< "$CHECK_OUT")
    fm=$(grep -c "^FILE_METHODS /out/${ID_OF[$l]}.cpg.bin"$'\t' <<< "$CHECK_OUT")
    chk "P3-7: $l METHOD_COUNT=$n (>=1) and per-file counts printed ($fm FILE lines)" test "${n:-0}" -ge 1 -a "$fm" -ge 1
  done
  # Negative: an exit-0 parse with no content (a bytecode frontend on Java SOURCE, §9 H13).
  run_r "${C[@]}" --frontend jimple2cpg --source "$FXD/javasrc" --output "$CPGS" -- /src
  eid=$(python3 -I -c 'import json,sys; print(json.loads(open(sys.argv[1]).readlines()[-1])["auditId"])' "$CCG_AUDIT_DIR/sandbox.jsonl")
  chk "the empty-CPG fixture is an exit-0 parse (rc $RC, $(stat -c %s "$CPGS/$eid.cpg.bin" 2>/dev/null) B)" test "$RC" = 0 -a -s "$CPGS/$eid.cpg.bin"
  run_r "${C[@]}" --frontend joern --script-file "$CHECK_SC" --source "$SRC" --output "$CPGS" -- --script /scripts/cpg-content-check.sc --param "cpgFiles=/out/${ID_OF[pythonsrc]}.cpg.bin;/out/$eid.cpg.bin"
  chk "P3-7: the empty CPG is CONTENT=EMPTY, the real one ok, and the check exits non-zero (rc $RC)" \
    test "$RC" != 0 -a -n "$(grep -x "CONTENT /out/$eid.cpg.bin=EMPTY" <<< "$OUT")" -a -n "$(grep -x "CONTENT /out/${ID_OF[pythonsrc]}.cpg.bin=ok" <<< "$OUT")"
  for l in pythonsrc jssrc javasrc kotlin csharpsrc golang newc; do
    line=$(grep -F "METHODS /out/${ID_OF[$l]}.cpg.bin=" <<< "$CHECK_OUT" || true)
    names=${line#*=}
    if [[ ",$names," == *",${WANT[$l]},"* ]]; then ok "$l CPG contains method ${WANT[$l]} ($names)"; else bad "$l CPG content" "line='$line'"; fi
  done
  sha=$(sha256sum < "$CHECK_SC" | cut -d' ' -f1)
  chk "10c: the check's record: script = /scripts/cpg-content-check.sc with the file's sha256" python3 -I -c '
import json, sys
rs = [json.loads(l) for l in open(sys.argv[1])]
sys.exit(0 if any(r["script"] == {"path": "/scripts/cpg-content-check.sc", "sha256": sys.argv[2]} and r["frontend"] == "joern" and r["exitCode"] == 0 for r in rs) else 1)' "$CCG_AUDIT_DIR/sandbox.jsonl" "$sha"
  chk "10c: launcherArgv has --workdir /tmp and the /scripts dir bind (ro,noexec)" rec "'--workdir' in r['launcherArgv'] and r['launcherArgv'][r['launcherArgv'].index('--workdir')+1]=='/tmp' and any(m['target']=='/scripts' and 'noexec' in m['mode'] and m['mode'].startswith('ro') for m in r['mounts'])"
  # 10(c) literally: --param cpgFile=/out/<x>.bin, one file.
  cat > "$T/one.sc" <<'SC'
@main def main(cpgFile: String) = { importCpg(cpgFile); println("ONE " + cpg.method.isExternal(false).name.l.sorted.mkString(",")) }
SC
  run_r "${C[@]}" --frontend joern --script-file "$T/one.sc" --source "$SRC" --output "$CPGS" -- --script /scripts/one.sc --param "cpgFile=/out/${ID_OF[pythonsrc]}.cpg.bin"
  chk "10c: --param cpgFile=/out/<x>.bin prints the fixture's methods" grep -q '^ONE .*\bf\b' <<< "$OUT"
  # m3: @AUDIT_ID@ is substituted, recorded substituted, and names the tool's own output file.
  cat > "$T/aid.sc" <<'SC'
@main def main(out: String) = { java.nio.file.Files.writeString(java.nio.file.Paths.get(out), "aid"); println("WROTE " + out) }
SC
  run_r "${C[@]}" --frontend joern --script-file "$T/aid.sc" --source "$SRC" --output "$CPGS" -- --script /scripts/aid.sc --param 'out=/out/@AUDIT_ID@.txt'
  aid=$(python3 -I -c 'import json,sys; print(json.loads(open(sys.argv[1]).readlines()[-1])["auditId"])' "$CCG_AUDIT_DIR/sandbox.jsonl")
  chk "m3: --param out=/out/@AUDIT_ID@.txt wrote /out/<auditId>.txt (rc $RC)" test "$RC" = 0 -a "$(cat "$CPGS/$aid.txt" 2>/dev/null)" = aid
  chk "m3: record toolArgv carries the substituted value" rec "('out=/out/%s.txt' % r['auditId']) in r['toolArgv'] and not any('@AUDIT_ID@' in x for x in r['toolArgv'])"
  # n3: a direct frontend (S25's route) produces content too.
  for pair in jssrc2cpg:jssrc pysrc2cpg:pythonsrc; do
    fe=${pair%%:*}; l=${pair#*:}
    run_r "${C[@]}" --frontend "$fe" --source "$FXD/$l" --output "$CPGS" -- /src
    did=$(python3 -I -c 'import json,sys; print(json.loads(open(sys.argv[1]).readlines()[-1])["auditId"])' "$CCG_AUDIT_DIR/sandbox.jsonl")
    run_r "${C[@]}" --frontend joern --script-file "$T/one.sc" --source "$SRC" --output "$CPGS" -- --script /scripts/one.sc --param "cpgFile=/out/$did.cpg.bin"
    chk "n3: direct $fe CPG contains f ($(grep '^ONE' <<< "$OUT"))" grep -q '^ONE .*\bf\b' <<< "$OUT"
  done
  # m2: a second run into an --output that is in use is refused (usage), never launched.
  LOCKD="$T/locked-out"; mkdir -p "$LOCKD"
  exec 7<"$LOCKD"; flock -n 7
  expect_refusal "m2: --output held by a concurrent run is refused" 64 usage "${C[@]}" --frontend pysrc2cpg --source "$FXD/pythonsrc" --output "$LOCKD" -- /src
  flock -u 7; exec 7<&-
  run_r "${C[@]}" --frontend pysrc2cpg --source "$FXD/pythonsrc" --output "$LOCKD" -- /src
  chk "m2: the same --output is usable once the lock is released (rc $RC)" test "$RC" = 0
  # egress-spec2 §1.3: two truly concurrent invocations into one --output -> exactly one refusal.
  CONC="$T/conc-out"; mkdir -p "$CONC"; before=$(nrec)
  "$RUNNER" "${C[@]}" --frontend joern-parse --source "$FXD/jssrc" --output "$CONC" -- /src --language jssrc > "$T/conc1" 2>&1 & p1=$!
  "$RUNNER" "${C[@]}" --frontend joern-parse --source "$FXD/jssrc" --output "$CONC" -- /src --language jssrc > "$T/conc2" 2>&1 & p2=$!
  wait "$p1"; r1=$?; wait "$p2"; r2=$?; after=$(nrec); NREC_EXPECTED=$((NREC_EXPECTED + 2))
  rcs=$(printf '%s\n' "$r1" "$r2" | sort | paste -sd' ')
  chk "ES2 §1.3: two concurrent runs into one --output: one runs, one is refused 64 (rcs: $rcs, records +$((after - before)))" test "$rcs" = "0 64" -a "$((after - before))" = 2
  chk "m2: the lock leaves no file in the output dir" test "$(ls -A "$LOCKD" | grep -v '\.cpg\.bin$' | wc -l)" = 0
  jparse_rec=$(python3 -I -c 'import json,sys; rs=[json.loads(l) for l in open(sys.argv[1])]; r=[x for x in rs if x["auditId"]==sys.argv[2]][0]; print(r["frontend"], r["refused"], r["codebases"])' "$CCG_AUDIT_DIR/sandbox.jsonl" "${ID_OF[golang]}")
  chk "joern-parse record names the derived frontend ($jparse_rec)" test "$jparse_rec" = "gosrc2cpg False ['fixture-golang']"
fi

# Python checkers shared by tests 4, 5 and 23 (written once, up front).
cat > "$T/clause.py" <<'PY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 else None
launched = [r for r in recs if not r["refused"] and r["backend"] == "container" and (only is None or r["frontend"] in only)]
errs = []
if not launched: errs.append("no launched container record to check")
for r in launched:
    a = r["launcherArgv"]; s = " ".join(a)
    def has(*seq):
        return any(a[i:i+len(seq)] == list(seq) for i in range(len(a)))
    for seq in (["--hostname", "localhost"], ["--read-only"], ["--cap-drop", "ALL"], ["--security-opt", "no-new-privileges"],
                ["--entrypoint", "/usr/bin/env"], ["--network", "none"], ["--tmpfs", "/tmp:rw,exec,nosuid,nodev"]):
        if not has(*seq): errs.append(f"{r['auditId']}: missing {seq}")
    ei = a.index("--entrypoint")
    if a[ei + 3] != "-i": errs.append(f"{r['auditId']}: first image argument is {a[ei+3]!r}, not -i")
    if a.count("--network") != 1: errs.append("more than one --network")
    for bad in ("--privileged", "--cap-add", "--device", "-e", "--env", "--env-file", "--add-host", "--dns", "--publish", "-p", "-v", "--volume"):
        if bad in a[:ei]: errs.append(f"{r['auditId']}: forbidden {bad}")
    for i, x in enumerate(a[:ei]):
        if x in ("--pid", "--ipc", "--uts", "--userns") and a[i+1] == "host": errs.append(f"{x}=host")
        if x == "--security-opt" and (a[i+1].startswith("seccomp") or "apparmor=unconfined" in a[i+1]): errs.append(f"security-opt {a[i+1]}")
        if x.startswith(("--privileged", "--cap-add", "--device", "--env", "--add-host", "--dns", "--publish", "--network=")): errs.append(f"joined form {x}")
    mounts = [a[i+1] for i, x in enumerate(a[:ei]) if x == "--mount"]
    dsts = sorted(m.split("dst=")[1].split(",")[0] for m in mounts)
    want = ["/out", "/scripts", "/src"] if r["frontend"] == "joern" else ["/out", "/src"]
    if dsts != want: errs.append(f"{r['auditId']} ({r['frontend']}): --mount targets {dsts}, want {want}")
    if r["frontend"] == "joern" and not has("--workdir", "/tmp"): errs.append("joern without --workdir /tmp")
    # p2-m1: the scoped hygiene check depends on these labels.
    if not has("--label", "ccg.sandbox.auditId=" + r["auditId"]): errs.append(f"{r['auditId']}: no --label ccg.sandbox.auditId=<auditId>")
    if not has("--name", "ccg-sbx-" + r["auditId"]): errs.append(f"{r['auditId']}: container not named ccg-sbx-<auditId>")
    if any(("volume-label=ccg.sandbox.auditId=" + r["auditId"]) not in m for m in mounts): errs.append(f"{r['auditId']}: a volume lacks the auditId label")
    keys = []
    for x in a[ei + 4:]:
        m = __import__("re").match(r"^([A-Za-z_][A-Za-z0-9_]*)=", x)
        if not m: break
        keys.append(m.group(1))
    if keys != r["envKeys"]: errs.append(f"envKeys {r['envKeys']} != env -i keys {keys}")
print(f"checked {len(launched)} launched records"); [print("  " + e) for e in errs]
sys.exit(1 if errs else 0)
PY
cat > "$T/equiv.py" <<'PY'
import json, re, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
# argv[3] "indexer": compare the last launched indexer record; otherwise the last parse-frontend one.
idx = ("scip-typescript", "scip-python")
if len(sys.argv) > 3 and sys.argv[3] == "indexer":
    prod = [r for r in recs if not r["refused"] and r["backend"] == "container" and r["frontend"] in idx][-1]
else:
    prod = [r for r in recs if not r["refused"] and r["backend"] == "container" and r["frontend"] != "joern" and r["frontend"] not in idx][-1]
lib = open(sys.argv[2], "rb").read().decode().split("\0")[:-1]
def norm(a):
    out, i = [], 0
    while i < len(a):
        x = a[i]
        if x == "--mount" and "dst=/scripts" in a[i+1]: i += 2; continue
        if x == "--mount": x2 = re.sub(r"(device|upperdir|lowerdir|workdir)=[^,\"]*", r"\1=P", a[i+1]); out += [x, x2]; i += 2; continue
        out.append(x); i += 1
    s = json.dumps(out)
    s = re.sub(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}", "AID", s)
    out = json.loads(s)
    k = out.index("-i") + 1
    while k < len(out) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", out[k]): k += 1
    return out[:k]
p, l = norm(prod["launcherArgv"]), norm(lib)
if p != l:
    import difflib; print("\n".join(difflib.unified_diff(p, l, lineterm="", n=0))); sys.exit(1)
print(f"{len(p)} elements equal")
PY

# ================================================================== test 1 / 5: escape-axis probe
host_targets() {
  HOSTNS=$(readlink /proc/self/ns/net)
  V4=$(ip -4 -o addr 2>/dev/null | awk '{split($4,a,"/"); if (a[1] !~ /^127\./) print a[1]}' | sort -u | paste -sd,)
  V6=$(ip -6 -o addr scope global 2>/dev/null | awk '{split($4,a,"/"); print a[1]}' | sort -u | paste -sd,)
  GW=$(docker network inspect bridge --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null)
  [ -n "$GW" ] && V4="${V4:+$V4,}$GW"
  NS53=$(awk '/^nameserver/{print $2}' /etc/resolv.conf | paste -sd,)
  PORTS=$( { echo 1234; echo 6379; echo 6390; ss -ltnH 2>/dev/null | awk '{n=split($4,a,":"); print a[n]}'; } | sort -un | paste -sd,)
  PEBIN=0; ( . "$RUNNER"; sbx_interop_scan /proc/sys/fs/binfmt_misc "$(uname -r)"; [ ${#SBX_PE_MATCHED[@]} -gt 0 ] ) && PEBIN=1
  PSRC="$T/probe-src"; POUT="$T/probe-out"; PSCR="$T/probe-scripts"; mkdir -p "$PSRC" "$POUT" "$PSCR" "$T/sentinel-dir"
  cp "$HERE/probe.pl" "$PSRC/"
  if [ -f /mnt/c/Windows/System32/whoami.exe ]; then cp /mnt/c/Windows/System32/whoami.exe "$PSRC/whoami.exe"; else printf 'MZ' > "$PSRC/whoami.exe"; fi
  chmod +x "$PSRC/whoami.exe"; echo 'println(1)' > "$PSCR/x.sc"
  echo s > "$T/sentinel-dir/sentinel"; echo s > "$HOME_SENTINEL"
  if [ -z "$MARKER_PID" ]; then bash -c "exec -a $MARKER sleep 900" & MARKER_PID=$!; fi
  PROBE_ARGS=(hostns="$HOSTNS" v4="$V4" v6="$V6" ns53="$NS53" ports="$PORTS" pe=/src/whoami.exe pebinfmt="$PEBIN"
    interp=/init uid="$(id -u)" sentinels="$T/sentinel-dir/sentinel:$HOME_SENTINEL" hostpid="$MARKER_PID" marker="$MARKER")
}
# check_probe <label> <file> <min checks>: PROBE-DONE present, check count, no FAIL.
check_probe() {
  local label=$1 f=$2 min=$3 done n fl
  done=$(grep '^PROBE-DONE' "$f" || true)
  n=$(sed -n 's/^PROBE-DONE checks=\([0-9]*\).*/\1/p' "$f"); fl=$(sed -n 's/.*fail=\([0-9]*\)$/\1/p' <<< "$done")
  if [ -n "$done" ] && [ "${n:-0}" -ge "$min" ] && [ "$fl" = 0 ]; then ok "$label: $done"; else
    bad "$label: ${done:-no PROBE-DONE (probe did not run to completion)}" "$(grep -E 'FAIL|LIB-REFUSED|rror' "$f" | head -8)"; fi
  local id
  for id in "${@:4}"; do grep -q "^CHECK $id PASS" "$f" && ok "$label $id" || bad "$label $id" "$(grep "^CHECK $id " "$f" | head -3)"; done
}
# check_env <label> <probe file> <envkeys file> <expected K=V file> [extra key]: A6 (a)(b)(c)(d)(e) + values (M3).
# The optional extra key is the indexer pass-list's PYTHONSAFEPATH (§7.6), which A6 (b) and (b') then pin.
check_env() {
  local label=$1 f=$2 ek=$3 ev=$4 xk=${5:-} nkeys=13
  local own; own=$(sed -n 's/^ENV //p' "$f" | sort | paste -sd' ')
  local rec_keys; rec_keys=$(sort "$ek" | paste -sd' ')
  local pass="DOTNET_CLI_TELEMETRY_OPTOUT DOTNET_NOLOGO DOTNET_ROOT DOTNET_SKIP_FIRST_TIME_EXPERIENCE GOROOT GOTOOLCHAIN HOME JAVA_HOME JAVA_OPTS LANG LC_ALL PATH TMPDIR"
  chk "$label A6a: probe's own environ keys == envKeys (from launcherArgv)" test -n "$own" -a "$own" = "$rec_keys"
  if [ -n "$xk" ]; then pass=$(printf '%s\n' $pass $xk | sort | paste -sd' '); nkeys=14; fi
  chk "$label A6b: envKeys == offline pass-list${xk:+ + $xk}" test "$rec_keys" = "$pass"
  chk "$label A6c: JAVA_OPTS matches ^-Xmx[0-9]+[mg]( -Xss[0-9]+[km])?\$" grep -qE '^JAVA_OPTS -Xmx[0-9]+[mg]( -Xss[0-9]+[km])?$' "$f"
  local leak; leak=$(grep -E '^ENV (JOERN_HOME|MAVEN_HOME|GRADLE_HOME|JAVA_VERSION|LANGUAGE|HOSTNAME|JAVA_TOOL_OPTIONS|_JAVA_OPTIONS|JDK_JAVA_OPTIONS|WSL_INTEROP|WSLENV|ENABLE_DEPENDENCY_RESOLVER_V2|.*_PROXY)$' "$f" || true)
  chk "$label A6d: no image-only key, HOSTNAME or forbidden name reaches the tool" test -z "$leak"
  # M3: §7.3.4 fixes the values, not just the names. JAVA_OPTS is A6c's pattern; every other key
  # must equal its expected value exactly.
  local vbad; vbad=$(grep -v '^JAVA_OPTS=' "$ev" | while IFS= read -r kv; do grep -qxF "ENVV $kv" "$f" || echo "${kv%%=*}"; done)
  chk "$label M3: every pass-list value equals §7.3.4 ($(grep -c . "$ev") keys)" test -z "$vbad" -a "$(grep -c '^ENVV ' "$f")" = "$nkeys"
  [ -n "$vbad" ] && printf '       wrong: %s\n' $vbad
  local sh extra
  for sh in bash sh; do
    extra=$(sed -n "s/^CHILDENV $sh //p" "$f" | grep -vxF -f "$ek" | grep -vxE 'PWD|SHLVL|_' || true)
    chk "$label A6e: /bin/$sh -c child keys ⊆ envKeys ∪ {PWD,SHLVL,_}" test -n "$(grep "^CHILDENV $sh " "$f")" -a -z "$extra"
  done
}
if want 1 || want 5; then
  section "test 1: escape-axis probe A1–A7 (container, through the runner's launch path)"
  host_targets
  lib_launch container "$PSRC" "$POUT" "$PSCR" "" -- /usr/bin/perl /src/probe.pl "${PROBE_ARGS[@]}" > "$T/probe-c.txt" 2>&1
  cp "$T/last-launcher" "$T/launcher-c"; cp "$T/last-envkeys" "$T/envkeys-c"
  check_probe "container probe" "$T/probe-c.txt" 60 A1a A1b A1c A2a A2b A3a A3b A3c A4a:/src A4a:/out A4a:/tmp A4a:/home/ccg A4a-memfd A4b A4c A5b A5c A5d A7
  for id in A1d-tcp A1d-udp A1e-tcp A1e-udp; do
    n=$(grep -c "^CHECK $id PASS" "$T/probe-c.txt"); chk "container probe $id ran $n targets, all pass" test "$n" -gt 0 -a -z "$(grep "^CHECK $id FAIL" "$T/probe-c.txt")"
  done
  grep -q 'CHECK A1d-tcp PASS 1.1.1.1:443' "$T/probe-c.txt" && grep -q 'CHECK A1e-tcp PASS 127.0.0.1:1234' "$T/probe-c.txt" && ok "probe covered 1.1.1.1:443 and LM Studio 127.0.0.1:1234" || bad "probe target coverage"
  img_env() { docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$IMAGE" | grep "^$1=" ; }
  { img_env PATH; img_env JAVA_HOME; img_env GOROOT; img_env DOTNET_ROOT
    printf '%s\n' HOME=/home/ccg TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8 GOTOOLCHAIN=local \
      DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 JAVA_OPTS=; } > "$T/expect-env-c"
  check_env "container" "$T/probe-c.txt" "$T/envkeys-c" "$T/expect-env-c"
  # A5(a)/test 5: mount points == the container's allowed set (with /scripts, since sdir was bound).
  mp_bad=$(sed -n 's/^MOUNT \([^ ]*\) .*/\1/p' "$T/probe-c.txt" | grep -vxE '/|/src|/out|/tmp|/home/ccg|/scripts|/proc(/.*)?|/dev(/.*)?|/sys(/.*)?|/etc/(hosts|hostname|resolv\.conf)' || true)
  chk "A5a/test 5: every container mount point is in the allowed set" test -z "$mp_bad"
  section "test 1: namespace backend"
  expect_refusal "namespace refuses through the runner on this host (N1 interop, N2 vsock)" 69 backend-precondition \
    --backend namespace --mode offline --frontend joern-parse --source "$SRC" --output "$OUTD" -- /src --language pythonsrc
  chk "  refusal record: interopPE=enabled, vsock=present, no launcher" rec "r['preconditions']['interopPE']=='enabled' and r['preconditions']['vsock']=='present' and r['envKeys']==[] and r['mounts']==[]"
  if command -v bwrap >/dev/null && bwrap --unshare-user --ro-bind / / /bin/true 2>/dev/null; then
    # §11 R5: the namespace template, run directly (not through the runner); A4/A7 excluded by design.
    POUT2="$T/probe-out-ns"; mkdir -p "$POUT2"
    lib_launch namespace "$PSRC" "$POUT2" "$PSCR" "" -- /usr/bin/perl /src/probe.pl axes=A1,A2,A3,A5,A6 "${PROBE_ARGS[@]}" > "$T/probe-n.txt" 2>&1
    cp "$T/last-envkeys" "$T/envkeys-n"
    check_probe "namespace template (direct, R5)" "$T/probe-n.txt" 50 A1a A1b A1c A2a A2b A3a A3b A3c A5b A5c A5d
    JDK=$( . "$RUNNER"; sbx_host_jdk )
    printf '%s\n' "PATH=/opt/joern/joern-cli:$JDK/bin:/usr/bin:/bin" "JAVA_HOME=$JDK" GOROOT=/opt/go DOTNET_ROOT=/opt/dotnet \
      HOME=/home/ccg TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8 GOTOOLCHAIN=local \
      DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 JAVA_OPTS= > "$T/expect-env-n"
    check_env "namespace" "$T/probe-n.txt" "$T/envkeys-n" "$T/expect-env-n"
    mp_bad=$(sed -n 's/^MOUNT \([^ ]*\) .*/\1/p' "$T/probe-n.txt" | grep -vxE '/|/usr|/usr/.*|/proc(/.*)?|/dev(/.*)?|/etc/(alternatives|java-21-openjdk|ld\.so\.cache|hosts|passwd|group|nsswitch\.conf)|/tmp|/home/ccg|/src|/out|/scripts|/opt/joern/joern-cli' | grep -v '^/usr/lib/wsl' || true)
    wsl=$(sed -n 's/^MOUNT \(\/usr\/lib\/wsl[^ ]*\) .*/\1/p' "$T/probe-n.txt")
    chk "test 5 (namespace): mount points are the §7.1 bind set, nothing at /usr/lib/wsl" test -z "$mp_bad$wsl"
    # m6: the namespace record lists every bind its launcher makes (§5.3 "every host path visible inside").
    m6=$(python3 -I -c '
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])][-1]
a = r["launcherArgv"]; tg = {m["target"] for m in r["mounts"]}
binds = [a[i+1] if a[i] == "--tmpfs" else a[i+2] for i in range(len(a)) if a[i] in ("--ro-bind", "--bind", "--tmpfs")]
miss = [b for b in binds if b not in tg]
print(f"{len(binds)} binds, {len(miss)} unlisted {miss[:5]}"); sys.exit(1 if miss or not binds else 0)' "$PROBE_AUDIT/sandbox.jsonl" 2>&1)
    chk "m6: namespace record mounts list every --ro-bind/--bind/--tmpfs target ($m6)" test $? = 0
  else
    bad "namespace template: bwrap unavailable, R5 partial probe not run"
  fi
fi

# ================================================================== test 5: launcher argv clause + container mount flags
if want 5; then
  section "test 5: mounts and the launcher argv clause"
  mopt() { sed -n "s#^MOUNT $1 \([^ ]*\) .*#\1#p" "$T/probe-c.txt"; }
  for m in /src /out /home/ccg /scripts; do
    o=$(mopt "$m"); chk "$m is noexec ($o)" grep -qw noexec <<< "$o"
  done
  chk "/src and /scripts are ro" bash -c "grep -q '^ro' <<< '$(mopt /src)' && grep -q '^ro' <<< '$(mopt /scripts)'"
  o=$(mopt /tmp); chk "/tmp is nosuid,nodev without noexec ($o)" bash -c "grep -qw nosuid <<< '$o' && grep -qw nodev <<< '$o' && ! grep -qw noexec <<< '$o'"
  run_r "${C[@]}" --frontend pysrc2cpg --source "$SRC" --output "$OUTD" -- /src   # a launched record of its own
  chk "a plain parse launches (exit $RC)" test "$RC" = 0
  if out=$(python3 -I "$T/clause.py" "$CCG_AUDIT_DIR/sandbox.jsonl" 2>&1); then ok "launcher clause holds on every launched record ($out)"; else bad "launcher clause" "$out"; fi
fi

# ================================================================== test 6: the --help sweep
write_sweep_py() {   # the classification checker, shared by test 6 and test 24
  cat > "$T/sweep.py" <<'PY'
import os, re, sys
d, fl, al = sys.argv[1:4]
TEN = "pysrc2cpg,jssrc2cpg,javasrc2cpg,jimple2cpg,kotlin2cpg,csharpsrc2cpg,gosrc2cpg,c2cpg,joern-parse,joern"
fes = (sys.argv[4] if len(sys.argv) > 4 else TEN).split(",")
BASE = r"fetch|download|url|https?:|dependenc|resolver|gradle|maven|nuget|proxy|registry|remote|jar4"
# test 24: for the indexers the regex also matches the spawn-shaped names (§10.1 (24)); an alarm only.
rx = re.compile(BASE + (r"|" + sys.argv[5] if len(sys.argv) > 5 else ""), re.I)
def entries(p, n):
    out = []
    for line in open(p, encoding="utf-8"):
        if line.startswith("#") or not line.strip(): continue
        f = line.rstrip("\n").split("\t")
        out.append((f[0], f[1]))
    return out
forb, allow = entries(fl, 4), entries(al, 3)
errs, hits = [], []
flags_of = {}
for fe in fes:
    blocks, cur = [], None
    for line in open(os.path.join(d, fe + ".txt"), encoding="utf-8", errors="replace"):
        line = line.rstrip("\n")
        if re.match(r"^\s+-", line): cur = [line]; blocks.append(cur)
        elif cur is not None and re.match(r"^\s{4,}\S", line): cur.append(line)
        else: cur = None
    flags_of[fe] = set()
    for b in blocks:
        head = re.split(r"\s{2,}", b[0].strip(), maxsplit=1)[0]
        names = re.findall(r"(?<![\w-])(--?[A-Za-z0-9][\w-]*)", head)
        flags_of[fe].update(names)
        if rx.search(" ".join(b)):
            listed_f = [n for n in names if (fe, n) in forb]; listed_a = [n for n in names if (fe, n) in allow]
            hits.append((fe, names, "forbidden" if listed_f else "allowed" if listed_a else "UNCLASSIFIED"))
            if not listed_f and not listed_a: errs.append(f"unclassified: {fe} {names}")
for fe, f in forb + allow:
    if fe in fes and f not in flags_of.get(fe, set()): errs.append(f"stale entry: {fe} {f} not in --help")
for e in set(forb) & set(allow): errs.append(f"in both files: {e}")
nf = sum(1 for h in hits if h[2] == "forbidden"); na = sum(1 for h in hits if h[2] == "allowed")
print(f"sweep hits={len(hits)} forbidden={nf} allowed={na} unclassified={len(hits)-nf-na}")
for h in hits: print(f"    {h[0]} {'/'.join(h[1])} -> {h[2]}")
[print("  " + e) for e in errs]
sys.exit(1 if errs else 0)
PY
}
sweep() { # <label> <backend>
  local label=$1 be=$2 fe d="$T/help-$2"; mkdir -p "$d" "$T/hsrc" "$T/hout-$be"
  for fe in pysrc2cpg jssrc2cpg javasrc2cpg jimple2cpg kotlin2cpg csharpsrc2cpg gosrc2cpg c2cpg joern-parse joern; do
    exe=$( . "$RUNNER"; sbx_frontend_exe "$fe" )
    lib_launch "$be" "$T/hsrc" "$T/hout-$be" "" /tmp -- "/opt/joern/joern-cli/$exe" --help > "$d/$fe.txt" 2>&1
    grep -qi '^Usage: ' "$d/$fe.txt" || bad "$label: $fe --help printed no Usage line" "$(head -3 "$d/$fe.txt")"
  done
  write_sweep_py
  local L; L=$(dirname "$RUNNER")
  if out=$(python3 -I "$T/sweep.py" "$d" "$L/forbidden-flags.txt" "$L/allowed-flags.txt" 2>&1); then ok "$label: $(head -1 <<< "$out")"; else bad "$label" "$out"; fi
  printf '%s\n' "$out" | tail -n +2 | head -20
}
if want 6; then
  section "test 6: --help sweep (§7.3.6)"
  sweep "container sweep" container
  if command -v bwrap >/dev/null && ( . "$RUNNER"; sbx_host_joern >/dev/null ); then
    sweep "namespace sweep (host Joern, template run directly, R5)" namespace
  else
    bad "namespace sweep: no bwrap or host Joern"
  fi
  # The sweep must fail on a stale, double or missing entry (self-test on doctored copies of the lists).
  L=$(dirname "$RUNNER"); mkdir -p "$T/lists"
  grep -v 'gosrc2cpg	--include-indirect' "$L/forbidden-flags.txt" > "$T/lists/f-missing"
  { cat "$L/forbidden-flags.txt"; printf 'pysrc2cpg\t--no-such-flag\toffline\tx\n'; } > "$T/lists/f-stale"
  { cat "$L/allowed-flags.txt"; printf 'kotlin2cpg\t--download-dependencies\tx\n'; } > "$T/lists/a-double"
  ! python3 -I "$T/sweep.py" "$T/help-container" "$T/lists/f-missing" "$L/allowed-flags.txt" >/dev/null && ok "sweep fails on an unclassified flag" || bad "sweep missed an unclassified flag"
  ! python3 -I "$T/sweep.py" "$T/help-container" "$T/lists/f-stale" "$L/allowed-flags.txt" >/dev/null && ok "sweep fails on a stale entry" || bad "sweep missed a stale entry"
  ! python3 -I "$T/sweep.py" "$T/help-container" "$L/forbidden-flags.txt" "$T/lists/a-double" >/dev/null && ok "sweep fails on a double entry" || bad "sweep missed a double entry"
fi

# ================================================================== test 9: frontend identity in the image
if want 9; then
  section "test 9: commercial frontend scripts absent from /opt/joern/joern-cli (inside the sandbox)"
  mkdir -p "$T/t9src" "$T/t9out"
  lib_launch container "$T/t9src" "$T/t9out" "" "" -- /bin/sh -c 'for s in java2cpg.sh csharp2cpg.sh go2cpg.sh py2cpg.sh js2cpg.sh; do if [ -e /opt/joern/joern-cli/$s ]; then echo "PRESENT $s"; else echo "ABSENT $s"; fi; done; test -x /opt/joern/joern-cli/joern-parse && echo "JOERN-CLI-OK"' > "$T/t9.txt" 2>&1
  n=$(grep -c '^ABSENT ' "$T/t9.txt"); chk "all 5 absent, and the directory is really there ($(paste -sd' ' "$T/t9.txt"))" test "$n" = 5 -a -z "$(grep PRESENT "$T/t9.txt")" -a -n "$(grep JOERN-CLI-OK "$T/t9.txt")"
fi

# ================================================================== test 11: closed runner environment (egress-spec2 §1.7)
if want 11; then
  section "test 11: polluted environment through the #! line (egress-spec2 §1.7)"
  PE="$T/pe"; mkdir -p "$PE/bin" "$PE/home/.docker" "$PE/dcfg"
  printf '#!/bin/sh\ntouch %q\nexit 1\n' "$PE/fake-docker-ran" > "$PE/bin/docker"; chmod +x "$PE/bin/docker"
  printf 'touch %q\n' "$PE/bash-env-ran" > "$PE/bash-env.sh"; printf 'touch %q\n' "$PE/env-ran" > "$PE/env.sh"
  EVILCFG='{"currentContext":"evil-s08","credsStore":"evil-s08","proxies":{"default":{"httpProxy":"http://evil-s08:1","httpsProxy":"http://evil-s08:1"}}}'
  printf '%s\n' "$EVILCFG" > "$PE/home/.docker/config.json"; printf '%s\n' "$EVILCFG" > "$PE/dcfg/config.json"
  POLLUTED=("PATH=$PE/bin:$PATH" "HOME=$PE/home" DOCKER_HOST=unix:///nonexistent-s08.sock DOCKER_CONTEXT=evil-s08
    "DOCKER_CONFIG=$PE/dcfg" SHELLOPTS=xtrace "BASH_ENV=$PE/bash-env.sh" "ENV=$PE/env.sh"
    "BASH_FUNC_docker%%=() { touch $PE/fn-ran; }" "BASH_FUNC_sbx_dcfg_check%%=() { return 0; }")
  # run_env <env assignments...> -- <runner args>: like run_r, under `env` (the runner's #! line).
  run_env() {
    local -a e=(); while [ "$1" != -- ]; do e+=("$1"); shift; done; shift
    local before; before=$(nrec)
    env "${e[@]}" "$RUNNER" "$@" > "$T/out.txt" 2> "$T/err.txt"; RC=$?
    OUT=$(cat "$T/out.txt"); ERR=$(cat "$T/err.txt"); NREC_EXPECTED=$((NREC_EXPECTED + 1)); NEW_RECORDS=$(( $(nrec) - before ))
  }
  lastrec() { python3 -I -c 'import json,sys; print(open(sys.argv[1]).readlines()[-1].strip())' "$CCG_AUDIT_DIR/sandbox.jsonl"; }
  run_r "${C[@]}" --frontend pysrc2cpg --codebase pe --source "$SRC" --output "$T/pe-clean" -- /src
  chk "clean reference run (rc $RC)" test "$RC" = 0; lastrec > "$T/pe-clean.json"
  run_env "${POLLUTED[@]}" -- "${C[@]}" --frontend pysrc2cpg --codebase pe --source "$SRC" --output "$T/pe-dirty" -- /src
  chk "polluted run launches like the clean one (rc $RC, $NEW_RECORDS record)" test "$RC" = 0 -a "$NEW_RECORDS" = 1; lastrec > "$T/pe-dirty.json"
  cat > "$T/pe-cmp.py" <<'PY'
import json, sys
a, b = (json.load(open(f)) for f in sys.argv[1:3])
def norm(r, out):
    s = json.dumps({k: v for k, v in r.items() if k not in ("auditId", "ts", "durationMs")}, sort_keys=True)
    s = s.replace(r["auditId"], "AID").replace(out, "OUT")
    s = __import__("re").sub(r"\.ccg-ovl-AID\.[A-Za-z0-9]{6}", ".ccg-ovl-AID.RND", s)  # mktemp suffix
    return json.loads(s)
na, nb = norm(a, sys.argv[3]), norm(b, sys.argv[4])
diff = [k for k in sorted(set(na) | set(nb)) if na.get(k) != nb.get(k)]
for k in diff:
    if isinstance(na.get(k), list): print("  %s: %s" % (k, [(x, y) for x, y in zip(na[k], nb[k]) if x != y][:3]))
print("fields differing: %s" % (diff or "none")); sys.exit(1 if diff else 0)
PY
  if out=$(python3 -I "$T/pe-cmp.py" "$T/pe-clean.json" "$T/pe-dirty.json" "$T/pe-clean" "$T/pe-dirty" 2>&1); then ok "record and launcherArgv equal the clean run's, modulo auditId/ts/paths ($out)"; else bad "polluted record differs from clean" "$out"; fi
  for m in fake-docker-ran bash-env-ran env-ran fn-ran; do chk "marker $m never written" test ! -e "$PE/$m"; done
  chk "SHELLOPTS=xtrace did not reach the runner (no trace lines on stderr)" test -z "$(grep -E '^\++ ' "$T/err.txt")"
  # The runner's own Docker-config dir (§1.7, E3-2): pre-populated, wrong mode, or a symlink -> 69.
  if [ -d "$DCFG" ] && [ ! -L "$DCFG" ] && [ -z "$(ls -A "$DCFG")" ]; then
    DCFG_TOUCHED=1
    printf '%s\n' "$EVILCFG" > "$DCFG/config.json"
    run_env "${POLLUTED[@]}" -- "${C[@]}" --frontend pysrc2cpg --source "$SRC" --output "$T/pe-x1" -- /src
    chk "pre-populated Docker-config dir (credsStore/proxies/currentContext) -> 69, one record (rc $RC)" test "$RC" = 69 -a "$NEW_RECORDS" = 1
    chk "  record: backend-precondition, nothing launched; stderr names it" bash -c "grep -q 'is not empty' '$T/err.txt'" 
    chk "  record fields" rec "r['refusalReason']=='backend-precondition' and r['launcherArgv']==[] and r['preconditions']['netnsDistinct'] is False"
    run_r "${C[@]}" --frontend pysrc2cpg --source "$SRC" --output "$T/pe-x2" -- /src
    chk "  same refusal with a clean caller environment (rc $RC)" test "$RC" = 69
    expect_refusal "  and 64 still takes precedence (forbidden flag + bad config dir)" 64 forbidden-flag "${C[@]}" --frontend javasrc2cpg --source "$SRC" --output "$T/pe-x3" -- /src --fetch-dependencies
    rm -f "$DCFG/config.json"
    chmod 0755 "$DCFG"
    run_r "${C[@]}" --frontend pysrc2cpg --source "$SRC" --output "$T/pe-x4" -- /src
    chk "Docker-config dir mode 0755 -> 69 (rc $RC)" test "$RC" = 69 -a -n "$(grep -o 'mode 0700' <<< "$ERR")"
    chmod 0700 "$DCFG"
    rmdir "$DCFG"; mkdir -m 0700 "$T/elsewhere-dcfg"; ln -s "$T/elsewhere-dcfg" "$DCFG"
    run_r "${C[@]}" --frontend pysrc2cpg --source "$SRC" --output "$T/pe-x5" -- /src
    chk "Docker-config dir replaced by a symlink -> 69 (rc $RC)" test "$RC" = 69 -a -n "$(grep -o 'not a plain directory' <<< "$ERR")"
    rm -f "$DCFG"; mkdir -m 0700 "$DCFG"
    rmdir "$DCFG"
    run_r "${C[@]}" --frontend sh --source "$SRC" --output "$T/pe-x6" -- /src
    chk "absent Docker-config dir is created 0700 and empty" test "$(stat -c '%a' "$DCFG" 2>/dev/null)" = 700 -a -z "$(ls -A "$DCFG" 2>/dev/null)"
    DCFG_TOUCHED=0
    run_r "${C[@]}" --frontend pysrc2cpg --source "$SRC" --output "$T/pe-x7" -- /src
    chk "after restore, runs launch again and leave the config dir empty (rc $RC)" test "$RC" = 0 -a -z "$(ls -A "$DCFG")"
  else
    bad "runner Docker-config dir $DCFG is not an empty plain directory; refusing to plant fixtures in it"
  fi
fi

# ================================================================== test 12: a signalled runner stops its own container (review S08 p2-n1)
if want 12; then
  section "test 12: SIGTERM while the sandbox runs"
  printf '@main def main() = { println("SLEEPING"); Thread.sleep(60000) }\n' > "$T/sleep.sc"
  SIGOUT="$T/sig-out"; mkdir -p "$SIGOUT" "$T/pe12/dcfg"; before=$(nrec)
  printf '%s\n' '{"currentContext":"evil-s08","proxies":{"default":{"httpProxy":"http://evil-s08:1","httpsProxy":"http://evil-s08:1","noProxy":"x"}}}' > "$T/pe12/dcfg/config.json"
  # Run it polluted, so the live runner and its container can be inspected for §1.7's closure.
  env FOO_S08=1 "DOCKER_CONFIG=$T/pe12/dcfg" DOCKER_HOST=unix:///var/run/docker.sock "TMPDIR=$T/pe12" \
  "$RUNNER" "${C[@]}" --frontend joern --script-file "$T/sleep.sc" --source "$SRC" --output "$SIGOUT" -- --script /scripts/sleep.sc > "$T/sig.txt" 2>&1 & sp=$!
  sid=""; for i in $(seq 1 80); do
    o=$(ls -d "$T"/.ccg-ovl-* 2>/dev/null | head -1); [ -n "$o" ] && sid=$(basename "$o" | sed -n 's/^\.ccg-ovl-\([0-9a-f-]\{36\}\)\..*/\1/p')
    [ -n "$sid" ] && [ -n "$(docker ps -q --filter "name=^/ccg-sbx-$sid\$")" ] && break; sleep 0.25; done
  SIG_CONTAINER="ccg-sbx-$sid"; [ -n "$sid" ] && SIGNAL_AIDS+=("$sid")
  chk "the sandbox container is running ($SIG_CONTAINER)" test -n "$sid" -a -n "$(docker ps -q --filter "name=^/$SIG_CONTAINER\$")"
  # §1.7: the live runner's own environment is exactly the closed set, with the pinned values.
  renv=$(tr '\0' '\n' < "/proc/$sp/environ" 2>/dev/null | sort)
  rkeys=$(sed 's/=.*//' <<< "$renv" | grep -vxE 'PWD|OLDPWD|SHLVL|_' | paste -sd' ')
  chk "§1.7: the running runner's environ keys are the closed set ($rkeys)" test "$rkeys" = "CCG_AUDIT_DIR CCG_TOOLCHAIN_IMAGE DOCKER_CONFIG DOCKER_HOST HOME PATH"
  chk "§1.7: pinned values (DOCKER_HOST, DOCKER_CONFIG, HOME, PATH)" bash -c "grep -qx 'DOCKER_HOST=unix:///var/run/docker.sock' <<< \"\$1\" && grep -qx \"DOCKER_CONFIG=$DCFG\" <<< \"\$1\" && grep -qx \"HOME=\$(getent passwd \$(id -u) | cut -d: -f6)\" <<< \"\$1\" && grep -qx 'PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin' <<< \"\$1\"" _ "$renv"
  # A caller's config.json `proxies` would be injected by the Docker client into the container.
  cenv=$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$SIG_CONTAINER" 2>/dev/null)
  chk "§1.7: no proxy variable injected into the container (caller config.json ignored)" test -n "$cenv" -a -z "$(grep -iE '^(https?|no|ftp|all)_proxy=' <<< "$cenv")"
  kill -TERM "$sp"
  for i in $(seq 1 40); do kill -0 "$sp" 2>/dev/null || break; sleep 0.25; done
  if kill -0 "$sp" 2>/dev/null; then bad "runner still alive 10 s after SIGTERM"; docker rm -f -v "$SIG_CONTAINER" >/dev/null 2>&1; wait "$sp"; src_rc=$?
  else wait "$sp"; src_rc=$?; ok "runner exited within 10 s of SIGTERM (rc $src_rc)"; fi
  chk "exit code is 143" test "$src_rc" = 143
  sleep 1
  chk "its container is gone" test -z "$(docker ps -aq --filter "name=^/$SIG_CONTAINER\$")"
  chk "its volumes are gone" test -z "$(docker volume ls -q --filter "label=ccg.sandbox.auditId=$sid")"
  chk "no overlay scratch left beside the output" test -z "$(ls -d "$T"/.ccg-ovl-"$sid".* 2>/dev/null)"
  chk "a killed runner writes no record (§5.3)" test "$(nrec)" = "$before"
  chk "the --output is free again (a new run takes the lock)" bash -c "exec 9<'$SIGOUT'; flock -n 9"
  SIG_CONTAINER=""
fi

# ================================================================== test 13: an early SIGTERM leaks nothing (review S08 p3-m1)
# Test 12's TERM lands seconds into the run. Here it lands 0-0.10 s after the overlay scratch
# appears, i.e. while `docker run` may be between its create and start calls. Every trial must
# leave no container and no volume carrying its auditId (checked after a settle pause).
if want 13; then
  ET_TRIALS=${S08_EARLY_TERM_TRIALS:-20}
  section "test 13: SIGTERM 0-0.10 s after the overlay scratch appears ($ET_TRIALS trials)"
  printf '@main def main() = { println("SLEEPING"); Thread.sleep(60000) }\n' > "$T/sleep.sc"
  before=$(nrec); et_fail=""; et_aids=()
  for i in $(seq 1 "$ET_TRIALS"); do
    d="$T/et$i"; mkdir -p "$d/out"; delay=$(printf '0.%02d' $(( (i - 1) % 11 )))
    "$RUNNER" "${C[@]}" --frontend joern --script-file "$T/sleep.sc" --source "$SRC" --output "$d/out" -- --script /scripts/sleep.sc > "$d/log" 2>&1 & p=$!
    o=(); for _ in $(seq 1 4000); do o=("$d"/.ccg-ovl-*); [ -e "${o[0]}" ] && break; kill -0 "$p" 2>/dev/null || break; sleep 0.005; done
    sleep "$delay"; kill -TERM "$p" 2>/dev/null; wait "$p"; rc=$?
    aid=$(basename "${o[0]}" | sed -n 's/^\.ccg-ovl-\([0-9a-f-]\{36\}\)\..*/\1/p')
    [ -n "$aid" ] && SIGNAL_AIDS+=("$aid") && et_aids+=("$aid")
    [ -n "$aid" ] && [ "$rc" = 143 ] || et_fail+=" trial $i (delay $delay): rc=$rc aid=${aid:-none};"
  done
  sleep 2
  for aid in "${et_aids[@]}"; do
    c=$(docker ps -a --filter "label=ccg.sandbox.auditId=$aid" --format '{{.Names}}:{{.State}}')
    v=$(docker volume ls -q --filter "label=ccg.sandbox.auditId=$aid" | grep -c .)
    [ -z "$c" ] && [ "$v" = 0 ] || et_fail+=" $aid left container [${c:-none}] and $v volumes;"
  done
  chk "every trial exited 143 and left no container or volume (${#et_aids[@]} auditIds)" test -z "$et_fail"
  [ -n "$et_fail" ] && printf '       %s\n' "${et_fail//;/$'\n'      }"
  chk "no early-signalled runner wrote a record (§5.3)" test "$(nrec)" = "$before"
  chk "no overlay scratch left by any trial" test -z "$(ls -d "$T"/et*/.ccg-ovl-* 2>/dev/null)"
fi

# ================================================================== test 14: sbx_dcfg_check's tree rules (review S08 p3-m3)
# Library mode: source the runner, redefine sbx_home to return a fixture home, call
# sbx_dcfg_check. The real config dir is never touched. The world-writable fixture lives under the
# passwd home, outside /tmp: under $T (itself under /tmp) the /tmp rule would fire first.
if want 14; then
  section "test 14: sbx_dcfg_check refuses a temporary tree and a world-writable ancestor (library mode)"
  # dcfg_lib <fixture home>: prints "rc=<n> fail=<SBX_DCFG_FAIL>"
  dcfg_lib() { ( FH=$1; . "$RUNNER"; sbx_home() { echo "$FH"; }; sbx_dcfg_check; echo "rc=$? fail=$SBX_DCFG_FAIL" ) 2>&1; }
  mkdir -m 0755 "$WW_BASE"
  mkdir -p "$T/h-tmp"; got=$(dcfg_lib "$T/h-tmp")
  chk "home under /tmp -> refused, 'under a temporary tree' ($got)" bash -c '[[ $1 == "rc=1 fail="*"/docker-config is under a temporary tree" ]]' _ "$got"
  mkdir -p "$WW_BASE/sym/home/.local/state"; mkdir -m 0700 "$T/sym-target"; ln -s "$T/sym-target" "$WW_BASE/sym/home/.local/state/ccg"
  got=$(dcfg_lib "$WW_BASE/sym/home")
  chk "ancestor symlink (.local/state/ccg -> a 0700 dir under /tmp) -> refused, temporary tree ($got)" bash -c '[[ $1 == "rc=1 fail=$2/docker-config is under a temporary tree" ]]' _ "$got" "$T/sym-target"
  mkdir -p "$WW_BASE/ww/home"; chmod 0777 "$WW_BASE/ww"; got=$(dcfg_lib "$WW_BASE/ww/home")
  chk "world-writable ancestor outside /tmp -> refused, names it ($got)" bash -c '[[ $1 == "rc=1 fail="*"/docker-config sits under world-writable $2" ]]' _ "$got" "$WW_BASE/ww"
  mkdir -p "$WW_BASE/clean/home"; chmod 0755 "$WW_BASE/clean" "$WW_BASE/clean/home"; got=$(dcfg_lib "$WW_BASE/clean/home")
  chk "clean control (same base, mode 0755) -> accepted ($got)" test "$got" = "rc=0 fail="
  chk "  and its config dir was created 0700 and empty" test "$(stat -c %a "$WW_BASE/clean/home/.local/state/ccg/docker-config")" = 700 -a -z "$(ls -A "$WW_BASE/clean/home/.local/state/ccg/docker-config")"
fi

# ================================================================== test 15: env keys read from /proc/$$/environ (review S08 p3-n1)
# `compgen -e` lists only identifier-named variables; an entry such as `A-B=1` is invisible to it
# yet reaches every child. Library mode: sbx_clean_env with SBX_SELF pointed at a stub that
# prints REEXEC, under an environment that is exactly the closed set, with and without `A-B=1`.
if want 15; then
  section "test 15: a non-identifier environment entry forces the env -i re-exec (library mode)"
  printf '#!/bin/bash -p\necho REEXEC\n' > "$T/reexec-stub"; chmod +x "$T/reexec-stub"
  PH=$(getent passwd "$(id -u)" | cut -d: -f6)
  CLOSED=("PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" "HOME=$PH" DOCKER_HOST=unix:///var/run/docker.sock
    "DOCKER_CONFIG=$PH/.local/state/ccg/docker-config" "CCG_AUDIT_DIR=$CCG_AUDIT_DIR")
  ce() { env -i "$@" /bin/bash -p -c '. "$1"; SBX_SELF=$2; sbx_clean_env && echo NO-REEXEC' _ "$RUNNER" "$T/reexec-stub" 2>&1; }
  got=$(ce "${CLOSED[@]}"); chk "control: exactly the closed set -> no re-exec ($got)" test "$got" = NO-REEXEC
  got=$(ce "${CLOSED[@]}" 'A-B=1'); chk "closed set plus 'A-B=1' -> re-exec ($got)" test "$got" = REEXEC
  got=$(ce "${CLOSED[@]}" 'BASH_FUNC_f%%=() { echo hi; }'); chk "closed set plus an exported function -> re-exec ($got)" test "$got" = REEXEC
  # The decision reads the block the runner was exec'd with, values included: a shell variable
  # changed after exec (as a host hook can append to PATH) is not caller input and must not
  # force a re-exec (else a hook that does it in every bash would loop the re-exec).
  got=$(env -i "${CLOSED[@]}" /bin/bash -p -c '. "$1"; SBX_SELF=$2; PATH=$PATH:/s08-post-exec; HOME=/s08-x; sbx_clean_env && echo NO-REEXEC' _ "$RUNNER" "$T/reexec-stub" 2>&1)
  chk "closed block, shell PATH/HOME changed after exec -> no re-exec ($got)" test "$got" = NO-REEXEC
  got=$(ce "${CLOSED[@]/#PATH=*/PATH=/usr/bin:/bin}"); chk "a different PATH value in the block -> re-exec ($got)" test "$got" = REEXEC
fi

# ================================================================== test 4: every invocation, one valid record

# ================================================================== tests 21-26: the indexer entries (P1-48b)
# egress-spec3 §7.4 / §10.1. Fixtures are plain files written here; nothing in this block runs npm, pip or uv.
if want 21 || want 22 || want 23 || want 24 || want 26 || want 27; then
  IDX_TAG=$(env -i PATH=/usr/bin:/bin "$HERE/../../infra/toolchain/build.sh" --indexer --print-tag 2>/dev/null)
  IDX_IMAGE="${CCG_TEST_INDEXER_IMAGE:-$IDX_TAG}"
  IDX_OK=0
  if [ -z "$IDX_IMAGE" ]; then bad "indexer tests: cannot resolve the indexer image tag (build.sh --indexer --print-tag printed nothing)"
  elif docker image inspect "$IDX_IMAGE" >/dev/null 2>&1; then IDX_OK=1; fi
  norun_idx() { notrun "$1: indexer image ${IDX_IMAGE:-?} not built (cpg/infra/toolchain/build.sh --indexer)"; }
  ISRC="$T/idx-src"; IOUT="$T/idx-out"; mkdir -p "$ISRC" "$IOUT"
  TSSRC="$T/idx-ts"; PYSRC="$T/idx-py"; mkdir -p "$TSSRC" "$PYSRC"
  echo '{"compilerOptions":{"target":"es2020","strict":true},"include":["*.ts"]}' > "$TSSRC/tsconfig.json"
  echo 'export function f(x: number): number { return x + 1; }' > "$TSSRC/a.ts"
  printf 'def f(x):\n    return x + 1\n' > "$PYSRC/m.py"; echo '[]' > "$PYSRC/env.json"
  echo 'x = 1' > "$ISRC/m.py"; echo '[]' > "$ISRC/env.json"
  # Every odd path used by the grammar cases exists on the host, so only the grammar (never the containment
  # walk's existence check) can be what refuses it.
  mkdir -p "$ISRC/a" "$ISRC/x" "$ISRC/srcfoo"
  for nm in 'a b' 'a:b' '$x' 'a;b' 'é' 'a=b' '*' $'a\nb'; do mkdir -p "$ISRC/$nm"; done
  idx_ref() { # <name> <exit> <reason> <frontend> -- tool args...
    local name=$1 code=$2 reason=$3 fe=$4; shift 5
    CCG_TOOLCHAIN_IMAGE=${IDX_IMAGE:-x} expect_refusal "$name" "$code" "$reason" "${C[@]}" --frontend "$fe" --source "${IDX_SRC:-$ISRC}" --output "$IOUT" -- "$@"
    case "$reason" in
      forbidden-flag) chk "  forbiddenHits not empty" rec "len(r['flagCheck']['forbiddenHits'])>0" ;;
      usage) chk "  no forbiddenHits (a usage refusal, not a forbidden element)" rec "r['flagCheck']['forbiddenHits']==[]" ;;
    esac
  }
  # Symlink fixtures (§7.4 containment). Each link is created here and checked below, so a refusal cannot
  # be the result of a link that was never made.
  SYM="$T/idx-sym"; SYMX="$T/idx-sym-evil"   # SYMX shares SYM's name as a prefix (a sibling-prefix probe)
  mkdir -p "$SYM/sub" "$SYMX"
  cp "$TSSRC/tsconfig.json" "$TSSRC/a.ts" "$SYM/sub/"; cp "$PYSRC/m.py" "$PYSRC/env.json" "$SYM/sub/"; cp "$PYSRC/env.json" "$SYM/env.json"
  ln -s / "$SYM/lnk_root"; ln -s ../.. "$SYM/lnk_up"; ln -s /src/sub "$SYM/lnk_abs"; ln -s ../idx-sym-evil "$SYM/lnk_sib"
  ln -s sub "$SYM/lnk_in"; ln -s sub/env.json "$SYM/envlink.json"
  ln -s "$SYM/sub" "$SYM/lnk_hostabs"          # absolute, and lies inside the tree by its HOST path
  ln -s nowhere "$SYM/lnk_dangle"; ln -s lnk_hostabs "$SYM/lnk_chain"   # dangling; a relative link to an absolute one
  ln -s /etc "$SYM/sub/inner_abs"
  mkfifo "$SYM/fifo"
  ln -s nope/../sub "$SYM/lnk_mid_dd"; ln -s nope/../env.json "$SYM/envlink_dd.json"   # a missing component, then `..` back to a real one
  # py_launch_check <name>: after an idx_run of scip-python (F5). The tool must have started (not 126/127),
  # printed the line it prints before it can fail, and now that --project-version is admitted, succeeded.
  py_launch_check() {   # <name> [cwd the tool prints; default /src]
    local cwd=${2:-/src} aid; aid=$(python3 -I -c 'import json,sys; print(json.loads(open(sys.argv[1]).readlines()[-1])["auditId"])' "$CCG_AUDIT_DIR/sandbox.jsonl")
    chk "  $1: the tool started (rc $RC is not 126/127)" test "$RC" != 126 -a "$RC" != 127
    chk "  $1: stdout/stderr carries 'Indexing $cwd with version'" grep -q "Indexing $cwd with version" <<< "$OUT$ERR"
    chk "  $1: exit 0 and a non-empty /out/<auditId>.scip" bash -c 'test "$1" = 0 && test -s "$2/$3.scip"' _ "$RC" "$IDX_OUTD" "$aid"
  }
  idx_run() { # <name> <frontend> <src> -- tool args...  -> sets RC, OUT; checks the record
    local name=$1 fe=$2 src=$3; shift 4
    mkdir -p "$T/idx-o-$NREC_EXPECTED"
    CCG_TOOLCHAIN_IMAGE=$IDX_IMAGE run_r "${C[@]}" --frontend "$fe" --source "$src" --output "$T/idx-o-$NREC_EXPECTED" -- "$@"
    IDX_OUTD="$T/idx-o-$((NREC_EXPECTED - 1))"
    expect_argv -- "$@"
    if [ "$NEW_RECORDS" = 1 ] && rec "r['refused'] is False and r['refusalReason'] is None and r['backend']=='container' and r['frontend']=='$fe' and r['job']=='parse' and r['launcherArgv']!=[] and r['toolchain']['image']=='$IDX_IMAGE' and r['toolArgv'][:-2]==open('$T/expect-argv').read().split(chr(0))[:-1] and r['toolArgv'][-2:]==['--output','/out/'+r['auditId']+'.scip'] and r['envKeys'][-1]=='PYTHONSAFEPATH' and r['flagCheck']['forbiddenHits']==[]" >/dev/null; then
      ok "$name: launched (record: container, indexer image, argv + runner-owned --output /out/<auditId>.scip, PYTHONSAFEPATH last in envKeys)"
    else bad "$name: record" "rc=$RC new=$NEW_RECORDS $(rec "False" 2>&1 | head -1)"; fi
  }
fi

if want 21; then
  section "test 21: SBX_FRONTENDS is the spec's twelve names, in order"
  want21='pysrc2cpg jssrc2cpg javasrc2cpg jimple2cpg kotlin2cpg csharpsrc2cpg gosrc2cpg c2cpg joern-parse joern scip-typescript scip-python'
  got21=$( . "$RUNNER"; echo "${SBX_FRONTENDS[*]}" )
  chk "(21) SBX_FRONTENDS == §7.4's twelve names in order (got: ${got21:-nothing sourced})" test "$got21" = "$want21"
  got=$( . "$RUNNER"; sbx_frontend_path scip-typescript; sbx_frontend_path scip-python; sbx_frontend_path c2cpg )
  chk "(21) indexer names map to fixed absolute paths under /opt/indexer; c2cpg stays a \$JOERN_HOME file" test "$got" = "/opt/indexer/bin/scip-typescript
/opt/indexer/bin/scip-python
/opt/joern/joern-cli/c2cpg.sh"
  CCG_TOOLCHAIN_IMAGE=${IDX_IMAGE:-x} expect_refusal "(21) a name outside the twelve (scip-java) is unknown-frontend" 64 unknown-frontend "${C[@]}" --frontend scip-java --source "$ISRC" --output "$IOUT" -- index --cwd /src
fi

if want 22; then
  section "test 22: the indexer argv grammar (launches and refusals)"
  ev_start
  # usage: a missing mandatory element
  idx_ref "22 usage: scip-typescript without --cwd" 64 usage scip-typescript -- index
  idx_ref "22 usage: scip-python without --cwd" 64 usage scip-python -- index --environment /src/env.json
  idx_ref "22 usage: scip-python without --environment" 64 usage scip-python -- index --cwd /src --project-version v1
  idx_ref "22 usage: scip-python with only --project-name" 64 usage scip-python -- index --project-name p
  # forbidden-flag: the first element must be `index`
  for fe in scip-typescript scip-python; do
    ENVA=(); [ "$fe" = scip-python ] && ENVA=(--environment /src/env.json --project-version v1)
    # the rest of the argv is valid, so only the first element can be what is refused
    for first in npm pip uv; do idx_ref "22 $fe first element '$first' (the rest is a valid grammar)" 64 forbidden-flag "$fe" -- "$first" --cwd /src "${ENVA[@]}"; done
    idx_ref "22 $fe bare --version" 64 forbidden-flag "$fe" -- --version
    idx_ref "22 $fe empty argv" 64 forbidden-flag "$fe" --
    idx_ref "22 $fe elements before index" 64 forbidden-flag "$fe" -- --cwd /src index
    idx_ref "22 $fe --cwd=/src form" 64 forbidden-flag "$fe" -- index --cwd=/src
    idx_ref "22 $fe repeated --cwd" 64 forbidden-flag "$fe" -- index --cwd /src --cwd /src
    idx_ref "22 $fe --cwd with no value" 64 forbidden-flag "$fe" -- index --cwd
    idx_ref "22 $fe --pnpm-workspaces" 64 forbidden-flag "$fe" -- index --cwd /src --pnpm-workspaces
    idx_ref "22 $fe caller --output" 64 forbidden-flag "$fe" -- index --cwd /src --output /out/x.scip
    idx_ref "22 $fe caller -o" 64 forbidden-flag "$fe" -- index --cwd /src -o /out/x.scip
    idx_ref "22 $fe caller --output= form" 64 forbidden-flag "$fe" -- index --cwd /src --output=/out/x.scip
    idx_ref "22 $fe launcher option -J-Dx=1" 64 forbidden-flag "$fe" -- index --cwd /src -J-Dx=1
  done
  for v in /srcfoo /src//a /src/./a /src/a/../a /src/../x /src/a/ / src /srcx/.. '/src/a b/../c'; do
    idx_ref "22 scip-typescript --cwd '$v'" 64 forbidden-flag scip-typescript -- index --cwd "$v"
    idx_ref "22 scip-python --cwd '$v'" 64 forbidden-flag scip-python -- index --cwd "$v" --environment /src/env.json --project-version v1
    idx_ref "22 scip-python --environment '$v'" 64 forbidden-flag scip-python -- index --cwd /src --environment "$v" --project-version v1
  done
  # options the tool does not take, and the scip-python-only ones
  idx_ref "22 scip-typescript --project-name" 64 forbidden-flag scip-typescript -- index --cwd /src --project-name p
  idx_ref "22 scip-typescript --environment" 64 forbidden-flag scip-typescript -- index --cwd /src --environment /src/env.json
  idx_ref "22 scip-python repeated --environment" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/env.json --environment /src/env.json --project-version v1
  idx_ref "22 scip-python repeated --project-name" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/env.json --project-name a --project-name b --project-version v1
  idx_ref "22 scip-python --environment=… form" 64 forbidden-flag scip-python -- index --cwd /src --environment=/src/env.json --project-version v1
  idx_ref "22 scip-python --project-name=x form" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/env.json --project-name=x --project-version v1
  idx_ref "22 scip-python abbreviation --proj" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/env.json --proj p --project-version v1
  idx_ref "22 scip-python abbreviation --env" 64 forbidden-flag scip-python -- index --cwd /src --env /src/env.json
  idx_ref "22 scip-typescript --project-version" 64 forbidden-flag scip-typescript -- index --cwd /src --project-version v1
  idx_ref "22 scip-python repeated --project-version" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/env.json --project-version a --project-version b
  idx_ref "22 scip-python --project-version=x form" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/env.json --project-version=x
  idx_ref "22 scip-python --project-version with no value" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/env.json --project-version
  for v in -x --x -- .a _a '' "$(printf 'a%.0s' $(seq 1 65))"; do
    idx_ref "22 scip-python --project-version '${v:0:12}'" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/env.json --project-version "$v"
  done
  idx_ref "22 usage: scip-python without --project-version" 64 usage scip-python -- index --cwd /src --environment /src/env.json
  idx_ref "22 usage: scip-python with only --project-version" 64 usage scip-python -- index --project-version v1
  # path components: a newline, a glob character, a space and other characters outside [A-Za-z0-9._@+-] are refused
  for v in $'/src/a\nb' '/src/*' '/src/a b' '/src/a:b' '/src/$x' '/src/a;b' '/src/é' '/src/a=b'; do
    idx_ref "22 scip-typescript --cwd component class: $(printf '%q' "$v")" 64 forbidden-flag scip-typescript -- index --cwd "$v"
    idx_ref "22 scip-python --environment component class: $(printf '%q' "$v")" 64 forbidden-flag scip-python -- index --cwd /src --environment "$v" --project-version v1
  done
  idx_ref "22 scip-python --target-only (not in the grammar)" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/env.json --target-only /src/a --project-version v1
  for v in -x --x -- .a _a '' "$(printf 'a%.0s' $(seq 1 65))"; do
    idx_ref "22 scip-python --project-name '${v:0:12}'" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/env.json --project-name "$v" --project-version v1
  done
  ev_none "test 22 refusals"
  # image and mode
  CCG_TOOLCHAIN_IMAGE=${IDX_IMAGE:-x} expect_refusal "22 controlled: scip-typescript is unknown-frontend" 64 unknown-frontend --backend container --mode controlled --frontend scip-typescript --source "$ISRC" --output "$IOUT" -- index --cwd /src
  CCG_TOOLCHAIN_IMAGE=${IDX_IMAGE:-x} expect_refusal "22 controlled: scip-python is unknown-frontend" 64 unknown-frontend --backend container --mode controlled --frontend scip-python --source "$ISRC" --output "$IOUT" -- index --cwd /src --environment /src/env.json
  CCG_TOOLCHAIN_IMAGE=${IDX_IMAGE:-x} expect_refusal "22 namespace backend: backend-unavailable (exit 69)" 69 backend-unavailable --backend namespace --mode offline --frontend scip-typescript --source "$ISRC" --output "$IOUT" -- index --cwd /src
  chk "  the namespace refusal names the container-only rule" grep -q 'container backend only' <<< "$ERR"
  CCG_TOOLCHAIN_IMAGE=${IDX_IMAGE:-x} expect_refusal "22 a grammar refusal outranks the namespace refusal (64, not 69)" 64 forbidden-flag --backend namespace --mode offline --frontend scip-python --source "$ISRC" --output "$IOUT" -- npm install
  # the default `toolchain` image has no label: a recorded refusal, not a tool error
  CCG_TOOLCHAIN_IMAGE=$IMAGE expect_refusal "22 indexer entry on the default toolchain image: backend-unavailable (69)" 69 backend-unavailable "${C[@]}" --frontend scip-typescript --source "$ISRC" --output "$IOUT" -- index --cwd /src
  chk "  the refusal names the missing label" grep -q 'ccg-toolchain-indexer=1' <<< "$ERR"
  CCG_TOOLCHAIN_IMAGE=$IMAGE expect_refusal "22 auto + toolchain image: still backend-unavailable (never the namespace precondition)" 69 backend-unavailable --backend auto --mode offline --frontend scip-python --source "$ISRC" --output "$IOUT" -- index --cwd /src --environment /src/env.json --project-version v1
  # launches
  if [ "$IDX_OK" = 1 ]; then
    idx_run "22 launch: scip-typescript index --cwd /src" scip-typescript "$TSSRC" -- index --cwd /src
    chk "  scip-typescript exit 0 and /out/<auditId>.scip is non-empty (rc $RC)" bash -c 'test "$1" = 0 && test -s "$2/$3.scip"' _ "$RC" "$IDX_OUTD" "$(python3 -I -c 'import json,sys; print(json.loads(open(sys.argv[1]).readlines()[-1])["auditId"])' "$CCG_AUDIT_DIR/sandbox.jsonl")"
    idx_run "22 launch: scip-python index --cwd /src --environment /src/env.json --project-name p --project-version v1" scip-python "$PYSRC" -- index --cwd /src --environment /src/env.json --project-name p --project-version v1
    py_launch_check "scip-python"
    idx_run "22 launch: scip-python, the same elements in another order" scip-python "$PYSRC" -- index --project-version v1 --project-name p --environment /src/env.json --cwd /src
    py_launch_check "scip-python reordered"
    idx_run "22 launch: scip-python without the optional --project-name (--cwd /src/)" scip-python "$PYSRC" -- index --environment /src/env.json --project-version v1.2-3 --cwd /src/
    py_launch_check "scip-python without --project-name" /src/
    CCG_TOOLCHAIN_IMAGE=$IDX_IMAGE run_r --backend auto --mode offline --frontend scip-typescript --source "$TSSRC" --output "$T/idx-o-auto" -- index --cwd /src
    chk "22 launch: --backend auto with the indexer image resolves to the container (rc $RC)" rec "r['backend']=='container' and r['refused'] is False and r['exitCode']==0"
  else
    norun_idx "22 launches (scip-typescript, scip-python, any order, auto)"
  fi

  # ---- symlink containment (§7.4): the host resolution of --cwd/--environment must stay inside the staged tree
  chk "22 symlink fixtures: lnk_root reaches / and lnk_up leaves the tree (the links are real)" bash -c '[ "$(realpath "$1/lnk_root")" = / ] && case "$(realpath "$1/lnk_up")" in "$1"|"$1"/*) exit 1;; esac; [ ! -e /src ] && [ -L "$1/lnk_abs" ] && case "$(realpath "$1/lnk_sib")" in "$1"/*) exit 1;; "$1"*) exit 0;; *) exit 1;; esac' _ "$SYM"
  chk "22 symlink fixtures: the in-tree links resolve inside the tree" bash -c 'case "$(realpath "$1/lnk_in")" in "$1"/sub) exit 0;; *) exit 1;; esac' _ "$SYM"
  chk "22 symlink fixtures: lnk_hostabs and lnk_chain resolve INTO the tree on the host (so only the absolute-target rule can refuse them); lnk_dangle dangles" bash -c '[ "$(realpath "$1/lnk_hostabs")" = "$1/sub" ] && [ "$(realpath "$1/lnk_chain")" = "$1/sub" ] && [ -L "$1/lnk_dangle" ] && [ ! -e "$1/lnk_dangle" ]' _ "$SYM"
  IDX_SRC=$SYM
  for lnk in lnk_root lnk_up lnk_abs lnk_sib lnk_root/etc lnk_up/x nope env.json lnk_hostabs lnk_hostabs/sub lnk_dangle lnk_chain sub/inner_abs sub/../lnk_root; do
    idx_ref "22 symlink: scip-typescript --cwd /src/$lnk is refused" 64 forbidden-flag scip-typescript -- index --cwd "/src/$lnk"
    idx_ref "22 symlink: scip-python --cwd /src/$lnk is refused" 64 forbidden-flag scip-python -- index --cwd "/src/$lnk" --environment /src/env.json --project-version v1
  done
  for lnk in lnk_root/etc/hostname lnk_up/x lnk_abs/env.json lnk_sib nope lnk_hostabs/env.json lnk_dangle lnk_chain/env.json sub/inner_abs/hostname; do
    idx_ref "22 symlink: scip-python --environment /src/$lnk is refused" 64 forbidden-flag scip-python -- index --cwd /src --environment "/src/$lnk" --project-version v1
  done
  chk "  the refusal names the containment rule" grep -q 'leaves the staged tree' <<< "$ERR"
  # --environment must be a regular file: a directory and a FIFO (a read of it would block) are refused, never a timeout (rc 124)
  chk "22 environment fixtures: /src/sub is a directory, /src/fifo a FIFO, /src/sub/env.json a regular file" bash -c '[ -d "$1/sub" ] && [ -p "$1/fifo" ] && [ -f "$1/sub/env.json" ]' _ "$SYM"
  idx_ref "22 --environment names a directory (/src/sub): refused" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/sub --project-version v1
  idx_ref "22 --environment names a FIFO (/src/fifo): refused" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/fifo --project-version v1
  chk "  the FIFO refusal is not a timeout (rc $RC is not 124)" test "$RC" != 124
  idx_ref "22 precedence: a FIFO --environment outranks a missing --project-version" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/fifo
  idx_ref "22 containment: a link through a missing component and back by .. (--cwd /src/lnk_mid_dd) is refused" 64 forbidden-flag scip-typescript -- index --cwd /src/lnk_mid_dd
  idx_ref "22 containment: a link through a missing component and back by .. (--environment /src/envlink_dd.json) is refused" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/envlink_dd.json --project-version v1
  idx_ref "22 --environment control: a regular file in a subdirectory passes containment (usage, no --project-version)" 64 usage scip-python -- index --cwd /src --environment /src/sub/env.json
  # precedence: a containment refusal is forbidden-class and outranks a missing mandatory element (usage)
  idx_ref "22 precedence: an escaping --cwd outranks a missing --project-version" 64 forbidden-flag scip-python -- index --cwd /src/lnk_root --environment /src/env.json
  idx_ref "22 precedence: an absolute-link --cwd outranks a missing --environment and --project-version" 64 forbidden-flag scip-python -- index --cwd /src/lnk_hostabs
  idx_ref "22 precedence: a missing --environment file outranks a missing --project-version" 64 forbidden-flag scip-python -- index --cwd /src --environment /src/nope.json
  idx_ref "22 precedence control: the same argv with in-tree values is usage" 64 usage scip-python -- index --cwd /src/lnk_in --environment /src/envlink.json
  unset IDX_SRC
  if [ "$IDX_OK" = 1 ]; then
    # positive controls: in-tree paths, including an in-tree symlink, are launched
    idx_run "22 symlink control: scip-typescript --cwd /src/sub (plain subdirectory)" scip-typescript "$SYM" -- index --cwd /src/sub
    chk "  exit 0 and a non-empty .scip (rc $RC)" bash -c 'test "$1" = 0 && test -s "$2/$3.scip"' _ "$RC" "$IDX_OUTD" "$(python3 -I -c 'import json,sys; print(json.loads(open(sys.argv[1]).readlines()[-1])["auditId"])' "$CCG_AUDIT_DIR/sandbox.jsonl")"
    idx_run "22 symlink control: scip-typescript --cwd /src/lnk_in (an in-tree symlink)" scip-typescript "$SYM" -- index --cwd /src/lnk_in
    chk "  exit 0 and a non-empty .scip (rc $RC)" bash -c 'test "$1" = 0 && test -s "$2/$3.scip"' _ "$RC" "$IDX_OUTD" "$(python3 -I -c 'import json,sys; print(json.loads(open(sys.argv[1]).readlines()[-1])["auditId"])' "$CCG_AUDIT_DIR/sandbox.jsonl")"
    idx_run "22 symlink control: scip-python --cwd /src/lnk_in --environment /src/envlink.json (in-tree symlinks)" scip-python "$SYM" -- index --cwd /src/lnk_in --environment /src/envlink.json --project-version v1
    py_launch_check "scip-python through in-tree symlinks" /src/lnk_in
  fi

  # ---- launch by image ID (§7.4): every container launch names the sha256 ID it inspected and recorded
  # (a name can be retagged between the inspect and the launch), for indexers and parse frontends alike.
  cat > "$T/imgid.py" <<'EOPY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
want = set(sys.argv[2].split(","))
launched = [r for r in recs if not r["refused"] and r["backend"] == "container" and r["frontend"] in want]
errs = []
if not launched: errs.append("no launched record of " + ",".join(sorted(want)))
for r in launched:
    a = r["launcherArgv"]; ei = a.index("--entrypoint")
    img = a[ei + 2]
    if img != r["toolchain"]["imageId"]: errs.append(f"{r['auditId']}: launcher image {img!r} != toolchain.imageId {r['toolchain']['imageId']!r}")
    if not img.startswith("sha256:"): errs.append(f"{r['auditId']}: launcher image {img!r} is not a sha256 ID")
print(f"checked {len(launched)} launched records"); [print("  " + e) for e in errs]
sys.exit(1 if errs else 0)
EOPY
  run_r "${C[@]}" --frontend pysrc2cpg --source "$SRC" --output "$T/imgid-parse-out" -- /src
  if out=$(python3 -I "$T/imgid.py" "$CCG_AUDIT_DIR/sandbox.jsonl" pysrc2cpg 2>&1); then ok "22 image by ID: a parse launch's launcherArgv image element == toolchain.imageId ($out)"; else bad "22 image by ID: parse launch" "$out"; fi
  if [ "$IDX_OK" = 1 ]; then
    if out=$(python3 -I "$T/imgid.py" "$CCG_AUDIT_DIR/sandbox.jsonl" scip-typescript,scip-python 2>&1); then ok "22 image by ID: every indexer launch's launcherArgv image element == toolchain.imageId ($out)"; else bad "22 image by ID: indexer launches" "$out"; fi
  fi
fi

if want 23; then
  section "test 23: the indexer boundary is the same launcher (clause, equivalence, escape-axis probe on the indexer image)"
  if [ "$IDX_OK" = 1 ]; then
    host_targets
    idx_run "23 launch: scip-typescript (a launched record for the clause and equivalence checks)" scip-typescript "$TSSRC" -- index --cwd /src
    if out=$(python3 -I "$T/clause.py" "$CCG_AUDIT_DIR/sandbox.jsonl" scip-typescript,scip-python 2>&1); then ok "(23) the §7.3 launcher clause holds on every launched indexer record ($out)"; else bad "(23) launcher clause on indexer records" "$out"; fi
    IPOUT="$T/probe-out-idx"; mkdir -p "$IPOUT"
    CCG_TOOLCHAIN_IMAGE=$IDX_IMAGE LIB_INDEXER=1 lib_launch container "$PSRC" "$IPOUT" "$PSCR" "" -- /usr/bin/perl /src/probe.pl "${PROBE_ARGS[@]}" > "$T/probe-i.txt" 2>&1
    cp "$T/last-launcher" "$T/launcher-idx"; cp "$T/last-envkeys" "$T/envkeys-idx"
    check_probe "indexer-image probe" "$T/probe-i.txt" 60 A1a A1b A1c A2a A2b A3a A3b A3c A4a:/src A4a:/out A4a:/tmp A4a:/home/ccg A4a-memfd A4b A4c A5b A5c A5d A7
    for id in A1d-tcp A1d-udp A1e-tcp A1e-udp; do
      n=$(grep -c "^CHECK $id PASS" "$T/probe-i.txt"); chk "indexer-image probe $id ran $n targets, all pass" test "$n" -gt 0 -a -z "$(grep "^CHECK $id FAIL" "$T/probe-i.txt")"
    done
    img_env_i() { docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$IDX_IMAGE" | grep "^$1=" ; }
    { img_env_i PATH; img_env_i JAVA_HOME; img_env_i GOROOT; img_env_i DOTNET_ROOT
      printf '%s\n' HOME=/home/ccg TMPDIR=/tmp LANG=C.UTF-8 LC_ALL=C.UTF-8 GOTOOLCHAIN=local PYTHONSAFEPATH=1 \
        DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 JAVA_OPTS=; } > "$T/expect-env-i"
    check_env "indexer image" "$T/probe-i.txt" "$T/envkeys-idx" "$T/expect-env-i" PYTHONSAFEPATH
    chk "indexer-image PATH carries /opt/indexer/bin (the image's PATH verbatim, A6 b')" grep -qx 'ENVV PATH=/opt/indexer/bin:.*' "$T/probe-i.txt"
    mp_bad=$(sed -n 's/^MOUNT \([^ ]*\) .*/\1/p' "$T/probe-i.txt" | grep -vxE '/|/src|/out|/tmp|/home/ccg|/scripts|/proc(/.*)?|/dev(/.*)?|/sys(/.*)?|/etc/(hosts|hostname|resolv\.conf)' || true)
    chk "(23) A5a: every mount point on the indexer image is in the allowed set" test -z "$mp_bad"
    if out=$(python3 -I "$T/equiv.py" "$CCG_AUDIT_DIR/sandbox.jsonl" "$T/launcher-idx" indexer 2>&1); then ok "(23) a launched indexer record's launcherArgv == the probe launch's modulo ids/paths/tool tail ($out)"; else bad "(23) indexer production and probe launchers differ" "$out"; fi
  else
    norun_idx "23 (launcher clause, equivalence, A1-A7 probe on the indexer image)"
  fi
fi

if want 24; then
  section "test 24: the --help sweep over both indexers (drift alarm; the grammar is the control)"
  if [ "$IDX_OK" = 1 ]; then
    d="$T/help-idx"; mkdir -p "$d" "$T/hsrc-i" "$T/hout-i"
    write_sweep_py
    [ -s "$T/sweep.py" ] && ok "24: the sweep checker is in place" || bad "24: the sweep checker was not written"
    for fe in scip-typescript scip-python; do
      # The §7.4 grammar refuses --help, so the help runs through the test-only launch path (lib-launch).
      CCG_TOOLCHAIN_IMAGE=$IDX_IMAGE LIB_INDEXER=1 lib_launch container "$T/hsrc-i" "$T/hout-i" "" /tmp -- "$(. "$RUNNER"; sbx_frontend_path "$fe")" index --help > "$d/$fe.txt" 2>&1
      grep -qi '^Usage: ' "$d/$fe.txt" && ok "24: $fe index --help printed its Usage line" || bad "24: $fe index --help printed no Usage line" "$(head -3 "$d/$fe.txt")"
    done
    L=$(dirname "$RUNNER"); mkdir -p "$T/lists-i"
    IDX_RX='pnpm|yarn|npm|pip|workspace|infer'
    if out=$(python3 -I "$T/sweep.py" "$d" "$L/forbidden-flags.txt" "$L/allowed-flags.txt" scip-typescript,scip-python "$IDX_RX" 2>&1); then ok "(24) indexer sweep: $(head -1 <<< "$out")"; else bad "(24) indexer sweep" "$out"; fi
    printf '%s\n' "$out" | tail -n +2 | head -12
    chk "(24) the widened regex finds the four spawn-shaped scip-typescript flags" test "$(grep -cE '^    scip-typescript .* -> (forbidden|allowed)$' <<< "$out")" -ge 4
    # Doctored copies: the alarm can fail. Each is a copy of the lists, never the real ones.
    grep -v 'scip-typescript	--pnpm-workspaces' "$L/forbidden-flags.txt" > "$T/lists-i/f-missing"
    { cat "$L/forbidden-flags.txt"; printf 'scip-python\t--no-such-flag\toffline\tx\n'; } > "$T/lists-i/f-stale"
    { cat "$L/allowed-flags.txt"; printf 'scip-typescript\t--infer-tsconfig\tx\n'; } > "$T/lists-i/a-double"
    # Each doctored run must fail for the stated reason (the output names it), so a missing checker
    # or a crash cannot pass as "the alarm fired".
    alarm() { # <name> <expected text in the output> <forb> <allow> [rx]
      local o rc=0
      o=$(python3 -I "$T/sweep.py" "$d" "$3" "$4" scip-typescript,scip-python "${5-$IDX_RX}" 2>&1) || rc=$?
      if [ "$rc" = 1 ] && grep -q "$2" <<< "$o"; then ok "$1"; else bad "$1" "rc=$rc; wanted '$2' in: ${o:0:300}"; fi
    }
    alarm "(24) the indexer sweep fails on an unclassified spawn-shaped flag" "unclassified: scip-typescript.*--pnpm-workspaces" "$T/lists-i/f-missing" "$L/allowed-flags.txt"
    alarm "(24) the indexer sweep fails on a stale entry" "stale entry: scip-python --no-such-flag" "$T/lists-i/f-stale" "$L/allowed-flags.txt"
    alarm "(24) the indexer sweep fails on a double entry" "in both files: .*--infer-tsconfig" "$L/forbidden-flags.txt" "$T/lists-i/a-double"
    # Control for the widening itself: with the base regex alone the missing entry goes unseen.
    o=$(python3 -I "$T/sweep.py" "$d" "$T/lists-i/f-missing" "$L/allowed-flags.txt" scip-typescript,scip-python 2>&1); rc=$?
    if [ "$rc" = 0 ] && grep -q '^sweep hits=0 ' <<< "$o"; then ok "(24) control: the unwidened regex sees no hit at all in either help (why the widening exists)"; else bad "(24) control: the unwidened regex" "rc=$rc $o"; fi
  else
    norun_idx "24 (index --help sweep of both indexers)"
  fi
fi

if want 25; then
  section "test 25: ccg-versions on the indexer image"
  printf '  info: asserted by cpg/infra/toolchain/tests/test_toolchain.sh section 5; not duplicated here (run it separately)\n'
fi

if want 26; then
  section "test 26: planted json.py under the indexer pass-list (PYTHONSAFEPATH, advisory M1)"
  if [ "$IDX_OK" = 1 ]; then
    PLANT="$T/plant-src"; mkdir -p "$PLANT"
    printf 'open("/out/json-planted-ran", "w").write("ran")\n' > "$PLANT/json.py"
    O26A="$T/plant-out-a"; O26B="$T/plant-out-b"; mkdir -p "$O26A" "$O26B"
    PYC='import json; print("JSON-FILE", json.__file__)'
    # (a) as specified: working directory /src, which holds the planted json.py.
    CCG_TOOLCHAIN_IMAGE=$IDX_IMAGE LIB_INDEXER=1 lib_launch container "$PLANT" "$O26A" "" /src -- python3 -c "$PYC" > "$T/plant-a.txt" 2>&1
    chk "(26) the launch ran: python3 imported json and printed its file ($(grep JSON-FILE "$T/plant-a.txt" | head -1))" grep -q '^JSON-FILE ' "$T/plant-a.txt"
    chk "(26) with PYTHONSAFEPATH the stdlib json is imported, not /src/json.py" grep -q '^JSON-FILE /usr/lib/python3' "$T/plant-a.txt"
    chk "(26) with PYTHONSAFEPATH the planted json.py did not run (marker absent)" test ! -e "$O26A/json-planted-ran"
    # (b) positive control, same launch path with the variable unset inside the sandbox: the plant is live.
    CCG_TOOLCHAIN_IMAGE=$IDX_IMAGE LIB_INDEXER=1 lib_launch container "$PLANT" "$O26B" "" /src -- /usr/bin/env -u PYTHONSAFEPATH python3 -c "$PYC" > "$T/plant-b.txt" 2>&1
    chk "(26) control: without PYTHONSAFEPATH the planted /src/json.py is the module imported" grep -q '^JSON-FILE /src/json.py' "$T/plant-b.txt"
    chk "(26) control: without PYTHONSAFEPATH the planted json.py ran (marker present)" test -e "$O26B/json-planted-ran"
    # (c) the launch's recorded envKeys carry the variable (A6 and test 7 pin it too).
    chk "(26) the lib-launch envKeys carry PYTHONSAFEPATH" grep -qx PYTHONSAFEPATH "$T/last-envkeys"
  else
    norun_idx "26 (planted json.py)"
  fi
fi

# ================================================================== test 27: the indexer wall-clock watchdog (egress-spec4 §1, §2)
if want 27; then
  section "test 27: indexer wall-clock watchdog (library mode; the constant is reassigned after sourcing)"
  # The constant is not an input: exported into the environment, it stays the literal 1800 after sourcing.
  got=$( export SBX_INDEXER_WALL_S=3; . "$RUNNER"; echo "$SBX_INDEXER_WALL_S" )
  chk "(27) library mode: an exported SBX_INDEXER_WALL_S=3 does not change the constant (got $got)" test "$got" = 1800
  chk "(27) the runner assigns the constant unconditionally, to the literal 1800" bash -c 'test "$(grep -c "^SBX_INDEXER_WALL_S=1800\$" "$1")" = 1 && ! grep -q "SBX_INDEXER_WALL_S:-" "$1"' _ "$RUNNER"
  # The timer's identity check, without a container: a timer armed for a live process whose recorded start
  # time does not match (the shape of a recycled PID) sends nothing; armed with the right start time, it does.
  got=$( . "$RUNNER"; SBX_INDEXER_WALL_S=2; sleep 30 & X=$!
         sbx_wd_arm "$X" 1; sleep 3.5; kill -0 "$X" 2>/dev/null && echo ALIVE || echo DEAD; kill -KILL "$X" 2>/dev/null; wait "$X" 2>/dev/null )
  chk "(27) timer identity: a recorded start time that does not match the process: no signal is sent (got $got)" test "$got" = ALIVE
  got=$( . "$RUNNER"; SBX_INDEXER_WALL_S=2; sleep 30 & X=$!
         sbx_wd_arm "$X"; sleep 3.5; wait "$X" 2>/dev/null; echo "rc=$?" )
  chk "(27) timer identity (control): with the process's own start time the cap's USR2 ends it (got $got)" test "$got" = rc=140
  # An unreadable start time: no timer is armed (the identity check would fail open), and the launch is refused.
  got=$( . "$RUNNER"; SBX_INDEXER_WALL_S=2; sbx_pid_start() { return 1; }; sbx_wd_arm >/dev/null 2>&1; echo "rc=$? pid=${SBX_WD_PID:-none}" )
  chk "(27) no readable start time: sbx_wd_arm arms no timer and fails (got $got)" test "$got" = "rc=1 pid=none"
  # A newline in the process name cannot hide field 22 from sbx_pid_start.
  NLN=$'a\n) 9 9 9'; cp /bin/sleep "$T/$NLN" 2>/dev/null; "$T/$NLN" 30 & NLP=$!
  want=$(python3 -I -c 'import sys; d=open("/proc/%s/stat" % sys.argv[1], "rb").read().decode(); print(d.rsplit(")", 1)[1].split()[19])' "$NLP" 2>&1)
  got=$( . "$RUNNER"; sbx_pid_start "$NLP" )
  kill -KILL "$NLP" 2>/dev/null; wait "$NLP" 2>/dev/null; rm -f "$T/$NLN"
  chk "(27) start time read for a process whose name contains a newline and ') ' (got '$got', want '$want')" bash -c 'test -n "$1" && test "$1" = "$2"' _ "$got" "$want"
  # wd_run <wall s> <indexer 0|1> <slow-removal 0|1> <src> <out> -- tool... : one library-mode launch (separate bash,
  # SIGINT restored to its default so the test can deliver one), its auditId in $T/wd-aid.
  cat > "$T/wd-run.sh" <<'EOW'
#!/bin/bash
wall=$1; ind=$2; slow=$3; src=$4; out=$5; shift 6
export CCG_AUDIT_DIR="$PROBE_AUDIT"
. "$RUNNER"
sbx_init
echo "$SBX_AUDIT_ID" > "$WD_AID_FILE"
[ "$ind" = 1 ] && SBX_INDEXER=1
sbx_resolve_backend container || { echo "LIB-REFUSED $SBX_REFUSAL_DETAIL"; sbx_cleanup; exit 99; }
SBX_RECORD_FRONTEND=lib-launch
SBX_INDEXER_WALL_S=$wall
if [ "${WD_LOCK:-0}" = 1 ]; then exec {SBX_LOCK_FD}<"$out"; flock -n "$SBX_LOCK_FD" || exit 98; fi   # the runner's --output lock
if [ "$slow" = 1 ]; then   # the signal handler's removal takes 10 s, so the watchdog can fire inside it
  eval "$(declare -f sbx_remove_own | sed '1s/sbx_remove_own/sbx_remove_own_real/')"
  sbx_remove_own() { sleep 10; sbx_remove_own_real; }
fi
rc=0; sbx_launch "$src" "$out" "" "" -- "$@" || rc=$?
sbx_write_record "$rc" ""; sbx_cleanup; exit "$rc"
EOW
  chmod +x "$T/wd-run.sh"
  wd_start() { # same args as wd_run; background, SIGINT default; sets WDPID
    WD_LOCK="${WD_LOCK:-0}" WD_AID_FILE="$T/wd-aid" RUNNER="$RUNNER" PROBE_AUDIT="$PROBE_AUDIT" \
      timeout -k 15 "${RUN_TIMEOUT:-300}" python3 -I -c 'import os,signal,sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execv(sys.argv[1], sys.argv[1:])' "$T/wd-run.sh" "$@" > "$T/wd-out.txt" 2>&1 &
    WDPID=$!
  }
  wd_nrec() { [ -f "$PROBE_AUDIT/sandbox.jsonl" ] && wc -l < "$PROBE_AUDIT/sandbox.jsonl" || echo 0; }
  wd_rec() { python3 -I -c 'import json,sys; r=json.loads(open(sys.argv[1]).readlines()[-1]); sys.exit(0 if eval(sys.argv[2], {"r": r}) else 1)' "$PROBE_AUDIT/sandbox.jsonl" "$1"; }
  if [ "$IDX_OK" = 1 ]; then
    W27=$T/wd-src; O27=$T/wd-out; mkdir -p "$W27" "$O27"; echo 'x' > "$W27/m.py"
    HANG=(/bin/sh -c 'trap "" TERM; while :; do sleep 1; done')
    # (a) the watchdog: an indexer-class launch whose tool ignores TERM is KILLed at the cap.
    n0=$(wd_nrec); t0=$(date +%s)
    CCG_TOOLCHAIN_IMAGE=$IDX_IMAGE wd_start 4 1 0 "$W27" "$O27" -- "${HANG[@]}"; wait "$WDPID"; WRC=$?; t1=$(date +%s)
    aid=$(cat "$T/wd-aid"); SIGNAL_AIDS+=("$aid")
    chk "(27) watchdog: the runner exits 124 (rc $WRC after $((t1 - t0)) s)" test "$WRC" = 124
    chk "(27) watchdog: exactly one record was written" test "$(( $(wd_nrec) - n0 ))" = 1
    chk "(27) watchdog: the record has exitCode 124 and refused false" wd_rec "r['exitCode']==124 and r['refused'] is False and r['refusalReason'] is None and r['auditId']=='$aid'"
    chk "(27) watchdog: no container labelled with its auditId remains" test -z "$(docker ps -aq --filter "label=ccg.sandbox.auditId=$aid")"
    chk "(27) watchdog: no volume labelled with its auditId remains" test -z "$(docker volume ls -q --filter "label=ccg.sandbox.auditId=$aid")"
    chk "(27) watchdog: no scratch of that run is left in /tmp" test -z "$(ls -d /tmp/ccg-sbx."$aid".* 2>/dev/null)"
    # (b) a normal exit before the cap: exit 0, a record with exitCode 0, and no stale timer signalling later.
    n0=$(wd_nrec)
    CCG_TOOLCHAIN_IMAGE=$IDX_IMAGE wd_start 4 1 0 "$W27" "$O27" -- /bin/true; wait "$WDPID"; WRC=$?
    SIGNAL_AIDS+=("$(cat "$T/wd-aid")")
    chk "(27) control: a tool that exits before the cap exits 0 (rc $WRC) with one record, exitCode 0" bash -c 'test "$1" = 0 && test "$2" = 1' _ "$WRC" "$(( $(wd_nrec) - n0 ))"
    chk "(27) control: the record has exitCode 0 and refused false" wd_rec "r['exitCode']==0 and r['refused'] is False"
    chk "(27) control: a disarmed timer leaves no 'Killed' job notice on stderr" test -z "$(grep -a Killed "$T/wd-out.txt")"
    # (b2) a normal exit leaves no orphan timer sleep (an unusual cap, 29 s, so the sleep is recognisable).
    CCG_TOOLCHAIN_IMAGE=$IDX_IMAGE wd_start 23 1 0 "$W27" "$O27" -- /bin/true; wait "$WDPID"; WRC=$?
    SIGNAL_AIDS+=("$(cat "$T/wd-aid")")
    sleep 1
    chk "(27) control: after a normal exit (rc $WRC) no 'sleep 23' timer process remains" bash -c '! pgrep -fx "sleep 23" >/dev/null'
    pkill -fx "sleep 23" 2>/dev/null || true
  else
    norun_idx "27 (watchdog on the indexer image)"
  fi
  # (c) a Joern frontend is not capped: a launch that outlives the cap still exits 0.
  n0=$(wd_nrec)
  wd_start 3 0 0 "$SRC" "$OUTD" -- /bin/sleep 7; wait "$WDPID"; WRC=$?
  SIGNAL_AIDS+=("$(cat "$T/wd-aid")")
  chk "(27) a non-indexer launch is not capped: it outlives the cap and exits 0 (rc $WRC)" test "$WRC" = 0
  chk "(27) a non-indexer launch: the record has exitCode 0" wd_rec "r['exitCode']==0 and r['refused'] is False"
  # (d) first wins: INT is delivered, then the watchdog's USR2 arrives while the INT handler is still removing
  # containers (10 s): exit 130, no record, no watchdog.
  if [ "$IDX_OK" = 1 ]; then
    n0=$(wd_nrec)
    rm -f "$T/wd-aid"
    CCG_TOOLCHAIN_IMAGE=$IDX_IMAGE wd_start 27 1 1 "$W27" "$O27" -- "${HANG[@]}"
    # the INT must land while the sandbox runs (its handler is installed): wait for the container, then signal
    wdc=""; for i in $(seq 1 60); do
      [ -s "$T/wd-aid" ] && wdc=$(docker ps -q --filter "label=ccg.sandbox.auditId=$(cat "$T/wd-aid")"); [ -n "$wdc" ] && break; sleep 0.25; done
    chk "(27) first wins: the sandbox container is running when the INT is sent" test -n "$wdc"
    wdr=$(pgrep -P "$WDPID" | head -1)      # the runner itself (timeout forwards INT, not USR2)
    kill -INT "$wdr" 2>/dev/null
    # The signal handler's own `jobs -p` KILL loop already ends the timer, so the timer cannot be what fires
    # inside the handler's window; deliver the timer's signal (USR2) by hand, while the handler is removing.
    sleep 2; kill -USR2 "$wdr" 2>/dev/null
    wait "$WDPID"; WRC=$?
    aid=$(cat "$T/wd-aid"); SIGNAL_AIDS+=("$aid")
    chk "(27) first wins: INT, then the timer's USR2 during the handler's removal: exit 130, not 124 (rc $WRC)" test "$WRC" = 130
    chk "(27) first wins: no record was written" test "$(( $(wd_nrec) - n0 ))" = 0
    sleep 1
    chk "(27) first wins: its container is gone" test -z "$(docker ps -aq --filter "label=ccg.sandbox.auditId=$aid")"
    chk "(27) after a signalled exit the timer is ended too: no 'sleep 27' remains" bash -c '! pgrep -fx "sleep 27" >/dev/null'
    pkill -fx "sleep 27" 2>/dev/null || true
    # (e) a second TERM while the INT handler is removing is ignored: the removal completes (exit 130, no
    # record, container gone), it does not die with 143 and leave the container running.
    n0=$(wd_nrec); rm -f "$T/wd-aid"
    CCG_TOOLCHAIN_IMAGE=$IDX_IMAGE wd_start 60 1 1 "$W27" "$O27" -- "${HANG[@]}"
    wdc=""; for i in $(seq 1 60); do
      [ -s "$T/wd-aid" ] && wdc=$(docker ps -q --filter "label=ccg.sandbox.auditId=$(cat "$T/wd-aid")"); [ -n "$wdc" ] && break; sleep 0.25; done
    wdr=$(pgrep -P "$WDPID" | head -1); kill -INT "$wdr" 2>/dev/null; sleep 2; kill -TERM "$wdr" 2>/dev/null
    wait "$WDPID"; WRC=$?; aid=$(cat "$T/wd-aid"); SIGNAL_AIDS+=("$aid"); sleep 1
    chk "(27) a second TERM during the INT handler's removal is ignored: exit 130 (rc $WRC), no record" bash -c 'test "$1" = 130 && test "$2" = 0' _ "$WRC" "$(( $(wd_nrec) - n0 ))"
    chk "(27) a second TERM during the INT handler's removal: the container is still removed" test -z "$(docker ps -aq --filter "label=ccg.sandbox.auditId=$aid")"
    docker rm -f -v $(docker ps -aq --filter "label=ccg.sandbox.auditId=$aid") >/dev/null 2>&1 || true
    # (e2) the same during the watchdog's own removal: exit 124 with its record, one expiry log line, and a
    # second USR2 and a TERM sent inside the removal window change nothing.
    n0=$(wd_nrec); rm -f "$T/wd-aid"; t0=$(date +%s)
    CCG_TOOLCHAIN_IMAGE=$IDX_IMAGE wd_start 5 1 1 "$W27" "$O27" -- "${HANG[@]}"
    wdr=""; for i in $(seq 1 60); do wdr=$(pgrep -P "$WDPID" | head -1); [ -n "$wdr" ] && [ -s "$T/wd-aid" ] && break; sleep 0.25; done
    # wait until the watchdog's handler has logged (its removal then takes 10 s), however slowly the container started
    for i in $(seq 1 120); do grep -aq 'wall-clock limit' "$T/wd-out.txt" 2>/dev/null && break; sleep 0.25; done
    sleep 0.5
    kill -USR2 "$wdr" 2>/dev/null; kill -TERM "$wdr" 2>/dev/null
    wait "$WDPID"; WRC=$?; aid=$(cat "$T/wd-aid"); SIGNAL_AIDS+=("$aid"); sleep 1
    chk "(27) a TERM and a second USR2 during the watchdog's removal change nothing: exit 124 (rc $WRC), one record" bash -c 'test "$1" = 124 && test "$2" = 1' _ "$WRC" "$(( $(wd_nrec) - n0 ))"
    chk "(27) the watchdog logged its expiry exactly once (a second USR2 did not re-enter the handler)" test "$(grep -ac 'wall-clock limit' "$T/wd-out.txt")" = 1
    chk "(27) during the watchdog's removal: the container is removed" test -z "$(docker ps -aq --filter "label=ccg.sandbox.auditId=$aid")"
    docker rm -f -v $(docker ps -aq --filter "label=ccg.sandbox.auditId=$aid") >/dev/null 2>&1 || true
    # (f) a runner killed with SIGKILL (no handler runs): its timer must not pin the --output lock and must
    # signal nothing at the cap (it checks the runner's start time). The Docker client the runner started is
    # ended by removing the container, so the only possible holder left is the timer or its sleep. Cap 47 s: the timer's first step is 30 s, and at its end the runner is gone, so it must exit then.
    rm -f "$T/wd-aid"; WD_LOCK=1 CCG_TOOLCHAIN_IMAGE=$IDX_IMAGE wd_start 47 1 0 "$W27" "$O27" -- "${HANG[@]}"
    wdc=""; for i in $(seq 1 60); do
      [ -s "$T/wd-aid" ] && wdc=$(docker ps -q --filter "label=ccg.sandbox.auditId=$(cat "$T/wd-aid")"); [ -n "$wdc" ] && break; sleep 0.25; done
    aid=$(cat "$T/wd-aid"); SIGNAL_AIDS+=("$aid")
    wdr=$(pgrep -P "$WDPID" | head -1); kill -KILL "$wdr" 2>/dev/null; wait "$WDPID" 2>/dev/null
    docker rm -f -v $(docker ps -aq --filter "label=ccg.sandbox.auditId=$aid") >/dev/null 2>&1 || true
    sleep 1.5
    chk "(27) SIGKILLed runner: the --output lock is free once its container is gone (the orphan timer does not pin it)" bash -c 'exec 9<"$1"; flock -n 9' _ "$O27"
    sleep 31
    chk "(27) SIGKILLed runner: one 30 s step after arming, the orphan timer has exited (no 'sleep 47' or 'sleep 17')" bash -c '! pgrep -fx "sleep (47|17)" >/dev/null'
    pkill -fx "sleep (47|17)" 2>/dev/null || true
    docker volume ls -q --filter "label=ccg.sandbox.auditId=$aid" | xargs -r docker volume rm >/dev/null 2>&1 || true
    # a SIGKILLed runner runs no cleanup: remove its scratch (the test's own, named by its auditId)
    for d in /tmp/ccg-sbx."$aid".* "$T"/.ccg-ovl-"$aid".*; do
      [ -d "$d" ] || continue; rmdir "$d/work/work" 2>/dev/null; chmod -R u+w "$d" 2>/dev/null; rm -rf "$d"; done
  else
    norun_idx "27 (first wins)"
  fi
fi

section "test 4: audit records (one per invocation, every §5.3 field, valid types)"
run_r "${C[@]}" --frontend pysrc2cpg --codebase t4 --source "$SRC" --output "$OUTD" -- /src
t4id=$(python3 -I -c 'import json,sys; print(json.loads(open(sys.argv[1]).readlines()[-1])["auditId"])' "$CCG_AUDIT_DIR/sandbox.jsonl")
chk "test 4: a launched parse writes /out/<auditId>.cpg.bin (exit $RC, $(stat -c %s "$OUTD/$t4id.cpg.bin" 2>/dev/null || echo 0) B)" test "$RC" = 0 -a -s "$OUTD/$t4id.cpg.bin"
cat > "$T/schema.py" <<'PY'
import json, re, sys, os
path, expected, cpgdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lib = len(sys.argv) > 4 and sys.argv[4] == "lib"
CPGS = {f for d in (cpgdir.split(":") if cpgdir else []) for _, _, fs in os.walk(d) for f in fs if f.endswith(".cpg.bin")}
lines = [l for l in open(path)] if os.path.exists(path) else []
errs = []
if len(lines) != expected: errs.append(f"{len(lines)} records for {expected} invocations")
U = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")
TS = re.compile(r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$")
S = lambda v: isinstance(v, str); L = lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v)
B = lambda v: isinstance(v, bool); I = lambda v: isinstance(v, int) and not isinstance(v, bool)
fields = {"schemaVersion": lambda v: v == 1, "recordType": lambda v: v == "sandbox", "auditId": lambda v: S(v) and U.match(v),
  "ts": lambda v: S(v) and TS.match(v), "backend": lambda v: v in ("container", "namespace"),
  "toolchain": lambda v: isinstance(v, dict) and (set(v) == {"image", "imageId"} or set(v) == {"joernVersion", "joernPath", "javaVersion"}),
  "job": lambda v: v in ("parse", "git-fetch"), "egressMode": lambda v: v in ("offline", "controlled"),
  "network": lambda v: v in ("none", "broker"), "networkName": lambda v: v is None or S(v),
  "launcherArgv": L, "toolArgv": L, "frontend": S,
  "script": lambda v: v is None or (isinstance(v, dict) and set(v) == {"path", "sha256"} and re.match(r"^[0-9a-f]{64}$", v["sha256"])),
  "flagCheck": lambda v: isinstance(v, dict) and set(v) == {"listDigest", "forbiddenHits", "allowedHits", "liftedHits"} and L(v["forbiddenHits"]) and L(v["allowedHits"]) and v["liftedHits"] == [],
  "envKeys": L, "mounts": lambda v: isinstance(v, list) and all(isinstance(m, dict) and set(m) == {"target", "source", "mode"} and all(S(x) for x in m.values()) for m in v),
  "preconditions": lambda v: isinstance(v, dict) and v.get("interopPE") in ("enabled", "disabled", "unreadable") and B(v.get("binfmtPinned")) and (v.get("seccomp") is None or S(v["seccomp"])) and v.get("vsock") in ("blocked", "present", "absent") and B(v.get("netnsDistinct")) and len(v) == 5,
  "refused": B, "refusalReason": lambda v: v in (None, "forbidden-flag", "forbidden-directive", "unknown-frontend", "mode-not-implemented", "backend-precondition", "backend-unavailable", "usage"),
  "exitCode": I, "codebases": L, "runId": lambda v: v is None, "durationMs": lambda v: I(v) and v >= 0}
ids = set()
for n, l in enumerate(lines, 1):
    try: r = json.loads(l)
    except Exception as e: errs.append(f"line {n}: not JSON ({e})"); continue
    if set(r) != set(fields): errs.append(f"line {n}: fields {sorted(set(r) ^ set(fields))}")
    for k, f in fields.items():
        if k in r and not f(r[k]): errs.append(f"line {n}: bad {k}={r[k]!r}")
    if r.get("auditId") in ids: errs.append(f"line {n}: duplicate auditId")
    ids.add(r.get("auditId"))
    if r.get("refused") != (r.get("refusalReason") is not None): errs.append(f"line {n}: refused/refusalReason mismatch")
    if r.get("refused") and r.get("exitCode") not in (64, 69): errs.append(f"line {n}: refusal exit {r.get('exitCode')}")
    if r.get("refused") and r.get("refusalReason") in ("forbidden-flag", "forbidden-directive") and not r["flagCheck"]["forbiddenHits"]: errs.append(f"line {n}: refusal without forbiddenHits")
    a = r.get("launcherArgv", [])
    if "/usr/bin/env" in a:
        i = a.index("-i", a.index("/usr/bin/env")) + 1
        ks = []
        for x in a[i:]:
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=", x)
            if not m: break
            ks.append(m.group(1))
        if ks != r["envKeys"]: errs.append(f"line {n}: envKeys != env -i keys")
    elif r.get("envKeys"): errs.append(f"line {n}: envKeys without env -i")
    if r.get("preconditions", {}).get("netnsDistinct") != bool(r.get("launcherArgv")): errs.append(f"line {n}: netnsDistinct={r['preconditions'].get('netnsDistinct')} but launcherArgv {'non-empty' if r.get('launcherArgv') else 'empty'}")
    if lib:
        if r.get("frontend") != "lib-launch" or r.get("refused"): errs.append(f"line {n}: library record {r.get('frontend')} refused={r.get('refused')}")
        continue
    if not r.get("refused") and r.get("frontend") in ("scip-typescript", "scip-python"):
        if f"/out/{r['auditId']}.scip" not in r["toolArgv"]: errs.append(f"line {n}: indexer output stem is not the auditId")
        continue
    if not r.get("refused") and r.get("frontend") not in ("joern",):
        if f"/out/{r['auditId']}.cpg.bin" not in r["toolArgv"]: errs.append(f"line {n}: tool output stem is not the auditId")
        elif r["exitCode"] == 0 and cpgdir and (r["auditId"] + ".cpg.bin") not in CPGS:
            errs.append(f"line {n}: /out/{r['auditId']}.cpg.bin missing on the host")
print(f"{len(lines)} records, {expected} invocations"); [print("  " + e) for e in errs[:20]]
sys.exit(1 if errs else 0)
PY
if out=$(python3 -I "$T/schema.py" "$CCG_AUDIT_DIR/sandbox.jsonl" "$NREC_EXPECTED" "$T" 2>&1); then ok "test 4: $out"; else bad "test 4" "$out"; fi
if [ -f "$PROBE_AUDIT/sandbox.jsonl" ]; then   # m5: library-mode records are validated too
  if out=$(python3 -I "$T/schema.py" "$PROBE_AUDIT/sandbox.jsonl" "$(wc -l < "$PROBE_AUDIT/sandbox.jsonl")" "" lib 2>&1); then ok "test 4 (library-mode records): $out"; else bad "test 4 (library-mode records)" "$out"; fi
fi
# m5: the probe's launch template IS the production template (modulo ids, paths, /scripts, tool tail).
if [ -f "$T/launcher-c" ]; then
  if out=$(python3 -I "$T/equiv.py" "$CCG_AUDIT_DIR/sandbox.jsonl" "$T/launcher-c" 2>&1); then ok "m5: probe launcherArgv == production launcherArgv modulo ids/paths/tool ($out)"; else bad "m5: probe and production templates differ" "$out"; fi
fi

# ------------------------------------------------------------------ Docker hygiene
section "hygiene"
sleep 1
# Scoped to this run: every container and volume the runner creates carries
# ccg.sandbox.auditId=<auditId>; only this run's ids count (other agents may hold ccg.sandbox
# containers of their own concurrently). A label-wide count is printed for information.
ids=$(own_aids)
mine() { grep -xF -f <(printf '%s\n' $ids) || true; }
left_v=$(docker volume ls --filter label=ccg.sandbox.auditId --format '{{.Label "ccg.sandbox.auditId"}}' 2>/dev/null | mine)
left_c=$(docker ps -a --filter label=ccg.sandbox.auditId --format '{{.Label "ccg.sandbox.auditId"}}' 2>/dev/null | mine)
chk "no volume of this run left behind ($(printf '%s\n' $ids | grep -c .) auditIds checked)" test -z "$left_v"
chk "no container of this run left behind" test -z "$left_c"
printf '  info any ccg.sandbox-labelled containers/volumes on the host now: %s / %s\n' \
  "$(docker ps -aq --filter label=ccg.sandbox | grep -c .)" "$(docker volume ls -q --filter label=ccg.sandbox | grep -c .)"
# p2-m1: anything named ccg-sbx-* or labelled ccg.sandbox that was CREATED during this run (docker
# events since the suite started) and still exists now -- no record needed to attribute it.
# p3-m2: attributed by path, not by time alone. Every launch this suite makes mounts paths under
# $T (source, overlay upper/work dirs), so a candidate counts only if its inspect JSON names $T/.
# Another session's sandbox work in the same window is reported, never counted.
NOW=$(date +%s)
names_t() { grep -qF "$T/" <<< "$1"; }
ev_c=$(docker events --since "$SUITE_T0" --until "$NOW" --filter type=container --filter event=create --format '{{.Actor.Attributes.name}}' 2>/dev/null | grep '^ccg-sbx-' | sort -u)
alive_c=""; foreign_c=0
for n in $ev_c; do
  j=$(docker inspect "$n" 2>/dev/null) || continue
  if names_t "$j"; then alive_c+=" $n"; else foreign_c=$((foreign_c + 1)); fi
done
ev_v=$(docker events --since "$SUITE_T0" --until "$NOW" --filter type=volume --filter event=create --format '{{.Actor.ID}}' 2>/dev/null | sort -u)
alive_v=""; foreign_v=0
for v in $ev_v; do
  j=$(docker volume inspect "$v" 2>/dev/null) || continue
  grep -q '"ccg.sandbox"' <<< "$j" || continue
  # A /scripts volume's device is the runner's own /tmp/ccg-sbx.* scratch, not a $T path; it is
  # ours if its auditId label is one of this run's.
  vaid=$(docker volume inspect --format '{{index .Labels "ccg.sandbox.auditId"}}' "$v" 2>/dev/null)
  if names_t "$j" || { [ -n "$vaid" ] && grep -qxF "$vaid" <<< "$ids"; }; then alive_v+=" $v"; else foreign_v=$((foreign_v + 1)); fi
done
chk "no ccg-sbx-* container this run created still exists ($(printf '%s\n' $ev_c | grep -c .) created; $foreign_c foreign ignored)" test -z "$alive_c"
chk "no ccg.sandbox volume this run created still exists ($(printf '%s\n' $ev_v | grep -c .) volumes created; $foreign_v foreign ignored)" test -z "$alive_v"
# p2-n2 / p4-m1: the runner names its scratch /tmp/ccg-sbx.<auditId>.XXXXXX, so a leaked one is
# attributed by auditId, like a container; any other new /tmp/ccg-sbx.* is foreign and ignored.
# (--help writes no record, so its leak path is checked in test 2, under a private /tmp.)
TMP_AFTER=$(ls -d /tmp/ccg-sbx.* 2>/dev/null | sort)
new_tmp=$(comm -13 <(printf '%s\n' $TMP_BEFORE) <(printf '%s\n' $TMP_AFTER) | grep -c . || true)
left_s=$(own_scratch "$ids")
chk "no /tmp/ccg-sbx.<auditId>.* scratch of this run left behind ($new_tmp new on the host; $(grep -c . <<< "$left_s") ours)" test -z "$left_s"
left_o=$(find "$T" -maxdepth 2 -name '.ccg-ovl-*' 2>/dev/null)
chk "no overlay scratch left beside an output dir" test -z "$left_o"

printf '\n%d passed, %d failed, %d not run\n' "$PASS" "$FAIL" "$NOTRUN"
((NOTRUN)) && printf 'NOT RUN: %s\n' "${NOTRUN_LIST[@]}"
((FAIL)) && { printf 'FAILED: %s\n' "${FAILED[@]}"; exit 1; }
((NOTRUN)) && exit 3
exit 0
