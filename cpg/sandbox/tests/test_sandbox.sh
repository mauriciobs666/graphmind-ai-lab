#!/usr/bin/env bash
# test_sandbox.sh — the S08 tests (egress spec docs/plans/connected-code-graph-egress-spec.md v1.3
# §10, tests 1–10) for cpg/sandbox/offline-run.sh, on both backends. On this WSL2 host the
# namespace backend refuses (N1 interop + N2 /dev/vsock); that refusal is asserted as the test,
# and per spec §11 R5 the namespace bwrap template is also run DIRECTLY (never through a runner
# bypass) for axes A1–A3, A5, A6 and the --help sweep.
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
cleanup() {
  [ -n "$MARKER_PID" ] && kill "$MARKER_PID" 2>/dev/null
  rm -f "$HOME_SENTINEL"
  chmod -R u+w "$T" 2>/dev/null; rm -rf "$T"
}
trap cleanup EXIT

PASS=0; FAIL=0; FAILED=()
ok()  { PASS=$((PASS + 1)); printf '  ok   %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); FAILED+=("$1"); printf '  FAIL %s\n' "$1"; [ -n "${2:-}" ] && printf '       %s\n' "$2"; }
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
  "$RUNNER" "$@" > "$T/out.txt" 2> "$T/err.txt"; RC=$?
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
  before=$(nrec); "$RUNNER" --help > /dev/null 2>&1; hrc=$?; after=$(nrec)
  chk "--help exits 0 and writes no record (rc $hrc, $((after - before)) new)" test "$hrc" = 0 -a "$after" = "$before"
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
  cat > "$T/methods.sc" <<'SC'
@main def main(cpgFiles: String) = {
  cpgFiles.split(";").foreach { f =>
    importCpg(f)
    println(s"METHODS $f=" + cpg.method.isExternal(false).name.l.distinct.sorted.mkString(","))
  }
}
SC
  run_r "${C[@]}" --frontend joern --script-file "$T/methods.sc" --source "$SRC" --output "$CPGS" -- --script /scripts/methods.sc --param "cpgFiles=$files"
  chk "10c: the directive-free script runs through the runner (exit $RC)" test "$RC" = 0
  for l in pythonsrc jssrc javasrc kotlin csharpsrc golang newc; do
    line=$(grep -F "METHODS /out/${ID_OF[$l]}.cpg.bin=" <<< "$OUT" || true)
    names=${line#*=}
    if [[ ",$names," == *",${WANT[$l]},"* ]]; then ok "$l CPG contains method ${WANT[$l]} ($names)"; else bad "$l CPG content" "line='$line'"; fi
  done
  sha=$(sha256sum < "$T/methods.sc" | cut -d' ' -f1)
  chk "10c: record script = /scripts/methods.sc with the file's sha256" rec "r['script']=={'path':'/scripts/methods.sc','sha256':'$sha'} and r['frontend']=='joern'"
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
# check_env <label> <probe file> <envkeys file> <expected K=V file>: A6 (a)(b)(c)(d)(e) + values (M3).
check_env() {
  local label=$1 f=$2 ek=$3 ev=$4
  local own; own=$(sed -n 's/^ENV //p' "$f" | sort | paste -sd' ')
  local rec_keys; rec_keys=$(sort "$ek" | paste -sd' ')
  local pass="DOTNET_CLI_TELEMETRY_OPTOUT DOTNET_NOLOGO DOTNET_ROOT DOTNET_SKIP_FIRST_TIME_EXPERIENCE GOROOT GOTOOLCHAIN HOME JAVA_HOME JAVA_OPTS LANG LC_ALL PATH TMPDIR"
  chk "$label A6a: probe's own environ keys == envKeys (from launcherArgv)" test -n "$own" -a "$own" = "$rec_keys"
  chk "$label A6b: envKeys == offline pass-list" test "$rec_keys" = "$pass"
  chk "$label A6c: JAVA_OPTS matches ^-Xmx[0-9]+[mg]( -Xss[0-9]+[km])?\$" grep -qE '^JAVA_OPTS -Xmx[0-9]+[mg]( -Xss[0-9]+[km])?$' "$f"
  local leak; leak=$(grep -E '^ENV (JOERN_HOME|MAVEN_HOME|GRADLE_HOME|JAVA_VERSION|LANGUAGE|HOSTNAME|JAVA_TOOL_OPTIONS|_JAVA_OPTIONS|JDK_JAVA_OPTIONS|WSL_INTEROP|WSLENV|ENABLE_DEPENDENCY_RESOLVER_V2|.*_PROXY)$' "$f" || true)
  chk "$label A6d: no image-only key, HOSTNAME or forbidden name reaches the tool" test -z "$leak"
  # M3: §7.3.4 fixes the values, not just the names. JAVA_OPTS is A6c's pattern; every other key
  # must equal its expected value exactly.
  local vbad; vbad=$(grep -v '^JAVA_OPTS=' "$ev" | while IFS= read -r kv; do grep -qxF "ENVV $kv" "$f" || echo "${kv%%=*}"; done)
  chk "$label M3: every pass-list value equals §7.3.4 ($(grep -c . "$ev") keys)" test -z "$vbad" -a "$(grep -c '^ENVV ' "$f")" = 13
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
  cat > "$T/clause.py" <<'PY'
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
launched = [r for r in recs if not r["refused"] and r["backend"] == "container"]
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
    keys = []
    for x in a[ei + 4:]:
        m = __import__("re").match(r"^([A-Za-z_][A-Za-z0-9_]*)=", x)
        if not m: break
        keys.append(m.group(1))
    if keys != r["envKeys"]: errs.append(f"envKeys {r['envKeys']} != env -i keys {keys}")
print(f"checked {len(launched)} launched records"); [print("  " + e) for e in errs]
sys.exit(1 if errs else 0)
PY
  if out=$(python3 -I "$T/clause.py" "$CCG_AUDIT_DIR/sandbox.jsonl" 2>&1); then ok "launcher clause holds on every launched record ($out)"; else bad "launcher clause" "$out"; fi
fi

# ================================================================== test 6: the --help sweep
sweep() { # <label> <backend>
  local label=$1 be=$2 fe d="$T/help-$2"; mkdir -p "$d" "$T/hsrc" "$T/hout-$be"
  for fe in pysrc2cpg jssrc2cpg javasrc2cpg jimple2cpg kotlin2cpg csharpsrc2cpg gosrc2cpg c2cpg joern-parse joern; do
    exe=$( . "$RUNNER"; sbx_frontend_exe "$fe" )
    lib_launch "$be" "$T/hsrc" "$T/hout-$be" "" /tmp -- "/opt/joern/joern-cli/$exe" --help > "$d/$fe.txt" 2>&1
    grep -qi '^Usage: ' "$d/$fe.txt" || bad "$label: $fe --help printed no Usage line" "$(head -3 "$d/$fe.txt")"
  done
  cat > "$T/sweep.py" <<'PY'
import os, re, sys
d, fl, al = sys.argv[1:4]
rx = re.compile(r"fetch|download|url|https?:|dependenc|resolver|gradle|maven|nuget|proxy|registry|remote|jar4", re.I)
def entries(p, n):
    out = []
    for line in open(p, encoding="utf-8"):
        if line.startswith("#") or not line.strip(): continue
        f = line.rstrip("\n").split("\t")
        out.append((f[0], f[1]))
    return out
forb, allow = entries(fl, 4), entries(al, 3)
fes = ["pysrc2cpg", "jssrc2cpg", "javasrc2cpg", "jimple2cpg", "kotlin2cpg", "csharpsrc2cpg", "gosrc2cpg", "c2cpg", "joern-parse", "joern"]
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
    if fe != "*" and f not in flags_of.get(fe, set()): errs.append(f"stale entry: {fe} {f} not in --help")
for e in set(forb) & set(allow): errs.append(f"in both files: {e}")
nf = sum(1 for h in hits if h[2] == "forbidden"); na = sum(1 for h in hits if h[2] == "allowed")
print(f"sweep hits={len(hits)} forbidden={nf} allowed={na} unclassified={len(hits)-nf-na}")
for h in hits: print(f"    {h[0]} {'/'.join(h[1])} -> {h[2]}")
[print("  " + e) for e in errs]
sys.exit(1 if errs else 0)
PY
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

# ================================================================== test 4: every invocation, one valid record
section "test 4: audit records (one per invocation, every §5.3 field, valid types)"
run_r "${C[@]}" --frontend pysrc2cpg --codebase t4 --source "$SRC" --output "$OUTD" -- /src
t4id=$(python3 -I -c 'import json,sys; print(json.loads(open(sys.argv[1]).readlines()[-1])["auditId"])' "$CCG_AUDIT_DIR/sandbox.jsonl")
chk "test 4: a launched parse writes /out/<auditId>.cpg.bin (exit $RC, $(stat -c %s "$OUTD/$t4id.cpg.bin" 2>/dev/null || echo 0) B)" test "$RC" = 0 -a -s "$OUTD/$t4id.cpg.bin"
cat > "$T/schema.py" <<'PY'
import json, re, sys, os
path, expected, cpgdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
lib = len(sys.argv) > 4 and sys.argv[4] == "lib"
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
    if not r.get("refused") and r.get("frontend") not in ("joern",):
        if f"/out/{r['auditId']}.cpg.bin" not in r["toolArgv"]: errs.append(f"line {n}: tool output stem is not the auditId")
        elif r["exitCode"] == 0 and cpgdir and not any(os.path.exists(os.path.join(d, r["auditId"] + ".cpg.bin")) for d in cpgdir.split(":")):
            errs.append(f"line {n}: /out/{r['auditId']}.cpg.bin missing on the host")
print(f"{len(lines)} records, {expected} invocations"); [print("  " + e) for e in errs[:20]]
sys.exit(1 if errs else 0)
PY
if out=$(python3 -I "$T/schema.py" "$CCG_AUDIT_DIR/sandbox.jsonl" "$NREC_EXPECTED" "$T/cpgs:$OUTD:$T/locked-out:$T/conc-out" 2>&1); then ok "test 4: $out"; else bad "test 4" "$out"; fi
if [ -f "$PROBE_AUDIT/sandbox.jsonl" ]; then   # m5: library-mode records are validated too
  if out=$(python3 -I "$T/schema.py" "$PROBE_AUDIT/sandbox.jsonl" "$(wc -l < "$PROBE_AUDIT/sandbox.jsonl")" "" lib 2>&1); then ok "test 4 (library-mode records): $out"; else bad "test 4 (library-mode records)" "$out"; fi
fi
# m5: the probe's launch template IS the production template (modulo ids, paths, /scripts, tool tail).
if [ -f "$T/launcher-c" ]; then
  cat > "$T/equiv.py" <<'PY'
import json, re, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
prod = [r for r in recs if not r["refused"] and r["backend"] == "container" and r["frontend"] != "joern"][-1]
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
  if out=$(python3 -I "$T/equiv.py" "$CCG_AUDIT_DIR/sandbox.jsonl" "$T/launcher-c" 2>&1); then ok "m5: probe launcherArgv == production launcherArgv modulo ids/paths/tool ($out)"; else bad "m5: probe and production templates differ" "$out"; fi
fi

# ------------------------------------------------------------------ Docker hygiene
section "hygiene"
sleep 1
# Scoped to this run: every container and volume the runner creates carries
# ccg.sandbox.auditId=<auditId>; only this run's ids count (other agents may hold ccg.sandbox
# containers of their own concurrently). A label-wide count is printed for information.
ids=$(cat "$CCG_AUDIT_DIR/sandbox.jsonl" "$PROBE_AUDIT/sandbox.jsonl" 2>/dev/null | sed -n 's/.*"auditId":"\([0-9a-f-]*\)".*/\1/p' | sort -u)
mine() { grep -xF -f <(printf '%s\n' $ids) || true; }
left_v=$(docker volume ls --filter label=ccg.sandbox.auditId --format '{{.Label "ccg.sandbox.auditId"}}' 2>/dev/null | mine)
left_c=$(docker ps -a --filter label=ccg.sandbox.auditId --format '{{.Label "ccg.sandbox.auditId"}}' 2>/dev/null | mine)
chk "no volume of this run left behind ($(printf '%s\n' $ids | grep -c .) auditIds checked)" test -z "$left_v"
chk "no container of this run left behind" test -z "$left_c"
printf '  info any ccg.sandbox-labelled containers/volumes on the host now: %s / %s\n' \
  "$(docker ps -aq --filter label=ccg.sandbox | grep -c .)" "$(docker volume ls -q --filter label=ccg.sandbox | grep -c .)"
left_o=$(find "$T" -maxdepth 2 -name '.ccg-ovl-*' 2>/dev/null)
chk "no overlay scratch left beside an output dir" test -z "$left_o"

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
((FAIL)) && { printf 'FAILED: %s\n' "${FAILED[@]}"; exit 1; }
exit 0
