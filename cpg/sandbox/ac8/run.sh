#!/usr/bin/env bash
# run.sh -- the AC-8 harness (C-1001 P1-31; docs/plans/connected-code-graph-egress-spec3.md §8, spec4).
#
#   run.sh detector-control [--id ID]
#       The detector control scenario (§8.1): run separately, before the main run. Starts a detector on a throwaway no-route
#       network, has a component container connect once, and checks that the verdict function FAILS on that `conn` and on a `conn`
#       planted before the start marker and after the flush marker (and passes a closed window with none).
#   run.sh run --manifest <system.yaml> [--repos-root DIR] [--id ID] [--components a,b] --store-standin [--witness-parse
#       [--lang repoId=lang]...] [--workload CMD] [--summary] [--stub-resolver] [--stub-model]
#       The main scenario: P-e first (before anything is created), then the network with NO route out, the detector, the broker
#       (production shape; under `strace` when installed), stand-in component containers, P-a..P-g, the in-run controls and the §7.3
#       escape probes, then the rows.
#       The store is the harness's own stand-in (`--store-standin`, a listener it creates on the scenario network); there is no option to
#       attach a caller's container (it would stay multi-homed with a route out, an unobserved way out): P1-40 adds a harness-created
#       real-store option when it needs one. --stub-model serves a stub local model on the manifest's loopback `local` port;
#       --stub-resolver answers P-e's resolver check with NXDOMAIN without any DNS query. --stub-model, --stub-resolver, --store-standin
#       and --witness-parse are SELF-TEST stand-ins: a run that uses any of them records `selfTest` and can end `incomplete` (exit 3) at
#       best, never `pass`. P1-40's acceptance is exit 0 with `selfTest == []`.
#       --workload CMD runs `bash -c CMD` as a HOST process with the host's environment and network, UNCONFINED: outside the detector, the
#       strace and the scenario network. It is the caller's own command (a test hook / the P1-40 driver), not something measured here.
#   run.sh verify <scenario dir> [--stub-resolver]       re-evaluate a scenario directory (assertions.py) without running it
#   run.sh pre-e <manifest> [--stub-resolver]            P-e alone (what `run` does first)
#
# Everything this script creates is named ccg-ac8-<id>-* and labelled ccg.ac8=<id>; it removes only those (checking the label first), by
# exact name, and verifies they are gone. It never touches another container or network (falkordb-dev, ccg-falkordb, ...), the host's
# network configuration, or ~/.config/ccg. What is MEASURED, and nothing more: (1) the scenario network is `--internal` with an isolated
# gateway and no address on its bridge, and the in-container probes (P-a: TCP connect and UDP sendto from each component to every
# host address x listening port, plus the public targets; P-b DNS; P-c sockets) saw no route; (2) the broker's own connect/sendto/
# sendmsg/sendmmsg calls under strace match its allow decisions (row 8.16); (3) the detector logged no TCP connection to ports
# 80/443/3128 of its own address, i.e. no component honoured the proxy variables. It does NOT show that no component attempted any other
# egress: with no route, a direct attempt fails locally and is invisible to the detector. The probes run as the components' own uid, so
# a hostile component could falsify them (theoretical for the stand-ins).
#
# Environment: CCG_AC8_BASE (scenario directories; default ~/.local/state/ccg-ac8: the broker's strict ancestor check refuses /tmp),
# CCG_AC8_PYTHON (default cpg/.venv/bin/python), CCG_TOOLCHAIN_IMAGE (default: infra/toolchain/build.sh --print-tag),
# CCG_AC8_NO_STRACE=1 (test hook: behave as if strace were not installed, so row 8.16 is reported unrun), CCG_AC8_HOOK=<stage>:die|sleep
# (test hook, stage `detector`: fail / pause once the scenario detector is up, to test cleanup).
# Exit: 0 pass, 1 a row/precondition/control failed, 3 incomplete (row 8.16 unrun: strace absent; or a self-test stand-in was used),
# 64 usage, 70 the harness itself could not establish a precondition it owns (e.g. the network mechanism): escalate, there is no
# fallback in P1 (spec3 R-E3). A run that leaves resources behind warns but keeps its verdict's exit code.
set -uo pipefail
umask 077
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
unset DOCKER_CONTEXT
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CPG="$(cd "$HERE/../.." && pwd)"
RUNNER="$HERE/../offline-run.sh"
PY="${CCG_AC8_PYTHON:-$CPG/.venv/bin/python}"
HOME_DIR="$(getent passwd "$(id -u)" | cut -d: -f6)"
BASE="${CCG_AC8_BASE:-$HOME_DIR/.local/state/ccg-ac8}"
UIDGID="$(id -u):$(id -g)"
# shellcheck disable=SC1091
. "$HERE/lib.sh"
AC8_PY="$PY"

ID=""; MANIFEST=""; REPOS_ROOT=""; COMPONENTS="answer,front"; STORE_MODE=""; STORE_IP=""; STORE_PORT=""
WITNESS=0; LANGS=(); WORKLOAD=""; SUMMARY=0; STUB_RES=0; STUB_MODEL=0; COMP_LIST=(); declare -A TOKS
OWN_CONTAINERS=(); OWN_NETS=(); BROKER_PID=""; STRACE_PID=""; MARK_PID=""; SENTINELS=()
SCEN=""; NET=""; DET=""; DET_IP=""; IMG=""

say() { printf 'ac8: %s\n' "$*" >&2; }
die() { say "$*"; exit "${2:-70}"; }
usage() { sed -n '2,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'; }

nm() { printf 'ccg-ac8-%s-%s' "$ID" "$1"; }

# ------------------------------------------------------------------ cleanup: exact names only, verified
cleanup() {
  local c n
  if [ -n "$BROKER_PID" ] && kill -0 "$BROKER_PID" 2>/dev/null; then
    kill -TERM "$BROKER_PID" 2>/dev/null
    for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$BROKER_PID" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$BROKER_PID" 2>/dev/null && kill -KILL "$BROKER_PID" 2>/dev/null
  fi
  if [ -n "$STRACE_PID" ] && kill -0 "$STRACE_PID" 2>/dev/null; then kill -TERM "$STRACE_PID" 2>/dev/null; sleep 0.5; kill -KILL "$STRACE_PID" 2>/dev/null; fi
  [ -n "$MARK_PID" ] && kill "$MARK_PID" 2>/dev/null
  # exact names this run recorded, and only when the object carries this scenario's label (a same-named foreign object is left alone)
  for c in "${OWN_CONTAINERS[@]}"; do
    case "$c" in ccg-ac8-"$ID"-*)
      [ "$(docker inspect -f '{{index .Config.Labels "ccg.ac8"}}' "$c" 2>/dev/null)" = "$ID" ] && docker rm -f "$c" >/dev/null 2>&1 ;; esac
  done
  for n in "${OWN_NETS[@]}"; do
    case "$n" in ccg-ac8-"$ID"-*)
      [ "$(docker network inspect -f '{{index .Labels "ccg.ac8"}}' "$n" 2>/dev/null)" = "$ID" ] && docker network rm "$n" >/dev/null 2>&1 ;; esac
  done
  for c in "${SENTINELS[@]}"; do [ -n "$c" ] && rm -rf -- "$c" 2>/dev/null; done
  local left
  left=$(docker ps -a --filter "label=ccg.ac8=$ID" --format '{{.Names}}' 2>/dev/null; docker network ls --filter "label=ccg.ac8=$ID" --format '{{.Name}}' 2>/dev/null)
  if [ -n "$left" ]; then say "WARNING: resources of scenario $ID still present: $(echo $left)"; else say "cleanup verified: no container or network labelled ccg.ac8=$ID remains"; fi
}

# ------------------------------------------------------------------ building blocks
toolchain_image() {
  IMG="${CCG_TOOLCHAIN_IMAGE:-$(env -i PATH=/usr/bin:/bin "$CPG/infra/toolchain/build.sh" --print-tag 2>/dev/null)}"
  [ -n "$IMG" ] && docker image inspect "$IMG" >/dev/null 2>&1 || die "the toolchain image '$IMG' is not built (cpg/infra/toolchain/build.sh)" 69
}

# The scenario's network: no route out, and no address on the bridge (P-a's mechanism candidate 1, spec3 §8.1).
net_up() { # <name>
  local net=$1 bridge
  OWN_NETS+=("$net")
  docker network create --internal -o com.docker.network.bridge.gateway_mode_ipv4=isolated --label "ccg.ac8=$ID" "$net" >/dev/null \
    || die "cannot create the network $net (isolated-gateway internal network)"
  [ "$(docker network inspect "$net" --format '{{.Internal}} {{.EnableIPv6}}')" = "true false" ] || die "network $net is not internal / has IPv6"
  bridge="br-$(docker network inspect "$net" --format '{{.Id}}' | cut -c1-12)"
  bridge_isolated_or_die "$bridge"
  say "network $net: internal, isolated gateway, bridge $bridge exists and has no address"
}

container_ip() { docker inspect --format "{{(index .NetworkSettings.Networks \"$2\").IPAddress}}" "$1"; }

wait_for() { # <seconds> <cmd...>
  local n=$(( $1 * 10 )); shift
  while ((n-- > 0)); do "$@" >/dev/null 2>&1 && return 0; sleep 0.1; done; return 1
}

detector_up() { # <net> <detector name> <detector host dir> <probe-tokens text: "<nonce> <label>" lines>
  local net=$1 name=$2 dir=$3 tokens=${4:-}
  mkdir -p "$dir"; chmod 700 "$dir"
  [ -n "$tokens" ] && { printf '%s\n' "$tokens" > "$dir/probe-tokens"; chmod 600 "$dir/probe-tokens"; }
  OWN_CONTAINERS+=("$name")
  docker run -d --name "$name" --label "ccg.ac8=$ID" --network "$net" --user "$UIDGID" --cap-drop ALL \
    --security-opt no-new-privileges --read-only --pids-limit 64 --memory 128m \
    -v "$dir:/detector:rw" -v "$HERE/detector.pl:/detector.pl:ro" \
    --entrypoint /usr/bin/perl "$IMG" /detector.pl /detector >/dev/null || die "cannot start the detector $name"
  wait_for 20 test -f "$dir/ready" || { docker logs "$name" >&2; die "the detector did not become ready"; }
}

signal_detector() { # <name> <USR1|USR2> <marker file>
  docker kill --signal="$2" "$1" >/dev/null || die "cannot signal the detector"
  wait_for 40 test -f "$3" || die "the detector did not write $3"   # a flush drains: bounded by the detector's own 30 s
}

stop_detector() { # <name>: stopped (it drains once more on TERM) and verified not running, before anything is judged
  docker stop -t 40 "$1" >/dev/null 2>&1   # > the detector's 30 s drain bound (detector.pl DRAIN_MAX), so the TERM drain is never SIGKILLed
  [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" = false ] || die "the detector $1 did not stop"
  [ "$(docker inspect -f '{{.State.ExitCode}}' "$1" 2>/dev/null)" = 0 ] || die "the detector $1 did not exit 0 (killed mid-drain?)"
}

rand_hex() { head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n'; }
now_ts() { date -u +%Y-%m-%dT%H:%M:%S.%3NZ; }

# ------------------------------------------------------------------ the detector control scenario
# Runs on its own throwaway network against the harness's OWN detector only. A component container makes: a token-less connection, one
# with a wrong nonce, one with the right nonce, and one replaying it (window.py CONTROL_CONNS/CONTROL_PROBE_LABEL: three are `conn`, one is
# a probe record); a second container opens CONTROL_FLOOD idle connections just before the flush (all must be logged before the flush
# marker, flush residue 0). window.py `control` then checks the detector's routing and that its verdict fails where it must.
detector_control() {
  local ctl="$SCEN/detector-control" cnet cdet cc cf rc nonce dip n
  cnet=$(nm ctl); cdet=$(nm ctl-detector); cc=$(nm ctl-comp); cf=$(nm ctl-flood)
  nonce=$(rand_hex)
  toolchain_image
  net_up "$cnet"
  detector_up "$cnet" "$cdet" "$ctl" "$nonce control/ok"
  dip=$(container_ip "$cdet" "$cnet")
  signal_detector "$cdet" USR1 "$ctl/start.marker"
  sleep 0.3
  OWN_CONTAINERS+=("$cc")
  printf '%s\n' "$nonce" | docker run --rm -i --name "$cc" --label "ccg.ac8=$ID" --network "$cnet" --user "$UIDGID" --cap-drop ALL --read-only \
    --security-opt no-new-privileges --pids-limit 64 --memory 128m \
    --entrypoint /usr/bin/perl "$IMG" -MIO::Socket::INET -e '
      chomp(my $n = <STDIN>);
      sub c { my $s = IO::Socket::INET->new(PeerAddr => $ARGV[0], PeerPort => 80, Proto => "tcp", Timeout => 3) or die "connect: $!\n";
              print $s $_[0]; close $s; select(undef, undef, undef, 0.4) }
      c("GET / HTTP/1.0\r\n\r\n"); c("AC8PROBE:" . ("0" x 32) . "\n"); c("AC8PROBE:$n\n"); c("AC8PROBE:$n\n"); print "four connections\n"' "$dip" \
    || die "the control's component could not connect"
  OWN_CONTAINERS+=("$cf")
  docker run -d --name "$cf" --label "ccg.ac8=$ID" --network "$cnet" --user "$UIDGID" --cap-drop ALL --read-only \
    --security-opt no-new-privileges --pids-limit 128 --memory 128m \
    --entrypoint /usr/bin/perl "$IMG" -MIO::Socket::INET -e '
      $| = 1; my @h; my @p = (80, 443, 3128);
      for my $i (0 .. '"$("$PY" -I -c 'import sys; sys.path.insert(0, sys.argv[1]); import window; print(window.CONTROL_FLOOD - 1)' "$HERE")"') {
        push @h, IO::Socket::INET->new(PeerAddr => $ARGV[0], PeerPort => $p[$i % 3], Proto => "tcp", Timeout => 3) or die "connect: $!\n" }
      print "ready\n"; sleep 15' "$dip" >/dev/null || die "cannot start the flood container"
  wait_for 10 bash -c "docker logs '$cf' 2>/dev/null | grep -q '^ready'" || { docker logs "$cf" >&2; die "the flood container did not open its connections"; }
  signal_detector "$cdet" USR2 "$ctl/flush.marker"
  docker rm -f "$cf" >/dev/null 2>&1
  stop_detector "$cdet"
  docker rm -f "$cdet" >/dev/null 2>&1
  docker network rm "$cnet" >/dev/null 2>&1
  "$PY" -I "$HERE/window.py" control "$ctl"; rc=$?
  return "$rc"
}

# ------------------------------------------------------------------ the §7.3 escape probes, inside the scenario
run_escape() {
  local t="$SCEN/escape" src out scr hostns v4 v6 gw ns53 ports pebin marker hs
  mkdir -p "$t/src" "$t/out" "$t/scripts" "$t/sentinel" "$SCEN/probe-audit" "$SCEN/probes"
  src="$t/src"; out="$t/out"; scr="$t/scripts"
  cp "$HERE/../tests/probe.pl" "$src/"
  printf 'MZ' > "$src/whoami.exe"; chmod +x "$src/whoami.exe"; echo 'println(1)' > "$scr/x.sc"
  hostns=$(readlink /proc/self/ns/net)
  v4=$(ip -4 -o addr 2>/dev/null | awk '{split($4,a,"/"); if (a[1] !~ /^127\./) print a[1]}' | sort -u | paste -sd,)
  v6=$(ip -6 -o addr scope global 2>/dev/null | awk '{split($4,a,"/"); print a[1]}' | sort -u | paste -sd,)
  gw=$(docker network inspect bridge --format '{{range .IPAM.Config}}{{.Gateway}}{{end}}' 2>/dev/null)
  [ -n "$gw" ] && v4="${v4:+$v4,}$gw"
  ns53=$(awk '/^nameserver/{print $2}' /etc/resolv.conf | paste -sd,)
  ports=$( { echo 1234; echo 6379; echo 6390; ss -ltnH 2>/dev/null | awk '{n=split($4,a,":"); print a[n]}'; } | sort -un | paste -sd,)
  pebin=0; ( . "$RUNNER"; sbx_interop_scan /proc/sys/fs/binfmt_misc "$(uname -r)"; [ ${#SBX_PE_MATCHED[@]} -gt 0 ] ) && pebin=1
  marker="ccg-ac8-marker-$$-$RANDOM"
  bash -c "exec -a $marker sleep 900" & MARK_PID=$!
  hs="$HOME_DIR/.ccg-ac8-sentinel-$ID"; echo s > "$t/sentinel/sentinel"; echo s > "$hs"; SENTINELS+=("$hs")
  ( export CCG_AUDIT_DIR="$SCEN/probe-audit"
    . "$RUNNER"
    sbx_init
    sbx_resolve_backend container || { echo "LIB-REFUSED $SBX_REFUSAL_DETAIL"; sbx_cleanup; exit 99; }
    SBX_RECORD_FRONTEND=lib-launch
    rc=0; sbx_launch "$src" "$out" "$scr" "" -- /usr/bin/perl /src/probe.pl hostns="$hostns" v4="$v4" v6="$v6" ns53="$ns53" \
      ports="$ports" pe=/src/whoami.exe pebinfmt="$pebin" interp=/init uid="$(id -u)" sentinels="$t/sentinel/sentinel:$hs" \
      hostpid="$MARK_PID" marker="$marker" || rc=$?
    sbx_write_record "$rc" ""; sbx_cleanup; exit "$rc" ) > "$SCEN/probes/escape-container.txt" 2>&1
  kill "$MARK_PID" 2>/dev/null; MARK_PID=""
  "$PY" -I "$HERE/facts.py" set "$SCEN/facts.json" escape '{"file":"probes/escape-container.txt","backend":"container"}'
  tail -1 "$SCEN/probes/escape-container.txt" | sed 's/^/ac8: escape probe: /' >&2
}

# ------------------------------------------------------------------ one component probe (P-a, P-b, P-c)
# The probe's inputs (the host's address/port inventory and its single-use nonces) go on STDIN (`docker exec -i`), never argv.
probe_components() { # <phase>; the host snapshot is read from facts.py
  local phase=$1 comp snap v4 v6 tcp udp pos det cip t0 t1
  snap=$("$PY" -I "$HERE/facts.py" snapshot) || die "cannot snapshot the host's addresses and ports"
  v4=$("$PY" -I -c 'import json,sys; print(",".join(json.loads(sys.argv[1])["v4"]))' "$snap")
  v6=$("$PY" -I -c 'import json,sys; print(",".join(json.loads(sys.argv[1])["v6"]))' "$snap")
  tcp=$("$PY" -I -c 'import json,sys; print(",".join(map(str,json.loads(sys.argv[1])["tcp"])))' "$snap")
  udp=$("$PY" -I -c 'import json,sys; print(",".join(map(str,json.loads(sys.argv[1])["udp"])))' "$snap")
  pos="$STORE_IP:$STORE_PORT"
  det="$DET_IP:80,$DET_IP:443,$DET_IP:3128"
  for comp in "${COMP_LIST[@]}"; do
    cip=$(container_ip "$(nm "$comp")" "$NET")
    t0=$(now_ts)
    printf 'v4=%s\nv6=%s\ntcp=%s\nudp=%s\npos=%s\ndet=%s\ntokens=%s\ndetname=%s\ndetip=%s\n' "$v4" "$v6" "$tcp" "$udp" "$pos" "$det" \
      "${TOKS["$comp/$phase"]}" "$DET" "$DET_IP" \
      | docker exec -i "$(nm "$comp")" /usr/bin/perl /ac8/netprobe.pl broker=/run/ccg-broker/broker.sock ifaces=eth0 > "$SCEN/probes/$phase-$comp.txt" 2>&1
    t1=$(now_ts)
    "$PY" -I "$HERE/facts.py" push "$SCEN/facts.json" probes "$("$PY" -I -c '
import json,sys
print(json.dumps({"phase":sys.argv[1],"component":sys.argv[2],"file":"probes/%s-%s.txt"%(sys.argv[1],sys.argv[2]),
  "snapshot":json.loads(sys.argv[3]),"pos":sys.argv[4].split(","),"det":sys.argv[5].split(","),"ip":sys.argv[6],"t0":sys.argv[7],"t1":sys.argv[8]}))' \
      "$phase" "$comp" "$snap" "$pos" "$det" "$cip" "$t0" "$t1")"
    say "P-a/b/c $phase probe from $comp: $(tail -1 "$SCEN/probes/$phase-$comp.txt")"
  done
}

# ------------------------------------------------------------------ the main run
main_run() {
  local t0 t1 comp STRACE=""
  [ -f "$MANIFEST" ] || die "--manifest is required and must exist" 64
  [ -x "$PY" ] || die "no Python at $PY (cpg/.venv)" 69
  [ "$STORE_MODE" = standin ] || die "give --store-standin (the harness-created store stand-in; attaching a caller's container is not supported)" 64
  # P-e first, before anything is created (spec3 §8.1 "before the run"): a manifest naming a real host, or a resolver that answers a
  # scenario name, refuses the run here. (Re-checked after the run over the scenario's copy of the manifest.)
  local pe=(pre-e "$MANIFEST"); [ "$STUB_RES" = 1 ] && pe+=(--stub-resolver)
  CCG_FORBID_REAL_CREDENTIALS=1 "$PY" -I "$HERE/assertions.py" "${pe[@]}" >&2 || { say "P-e failed before the run: no network or container was created"; return 1; }
  toolchain_image
  mkdir -p "$SCEN/audit/detector" "$SCEN/probes" "$SCEN/trace" "$SCEN/broker"; chmod 700 "$SCEN" "$SCEN/audit" "$SCEN/broker"
  cp "$MANIFEST" "$SCEN/manifest.yaml"
  echo '{}' > "$SCEN/facts.json"
  FACTS="$SCEN/facts.json"
  t0=$(date +%s)
  "$PY" -I "$HERE/facts.py" set "$FACTS" scenarioId "\"$ID\""
  # components are named by their short name in probe entries; the facts list the same names
  "$PY" -I "$HERE/facts.py" set "$FACTS" components "$("$PY" -I -c 'import json,sys; print(json.dumps(sys.argv[1].split(",")))' "$COMPONENTS")"
  "$PY" -I "$HERE/facts.py" set "$FACTS" summaryEnabled "$([ "$SUMMARY" = 1 ] && echo true || echo false)"
  # self-test stand-ins used by this run: any of them caps the verdict at `incomplete` (assertions.py)
  local st=() x
  [ "$STUB_RES" = 1 ] && st+=(stub-resolver); [ "$STUB_MODEL" = 1 ] && st+=(stub-model); st+=(store-standin); [ "$WITNESS" = 1 ] && st+=(witness-parse)
  "$PY" -I "$HERE/facts.py" set "$FACTS" selfTest "$("$PY" -I -c 'import json,sys; print(json.dumps(sys.argv[1:]))' "${st[@]}")"

  # 1. the detector control scenario, separately and first
  say "== detector control scenario"
  if ! detector_control; then say "the detector control scenario FAILED; the main run is not started"; return 1; fi

  # 2. the scenario network, the detector, the store
  say "== scenario network and detector"
  net_up "$NET"
  # one single-use nonce per (component, phase, detector target): the detector files a connection presenting one as a probe record
  # labelled <component>/<phase>; anything else it sees is a `conn`. (Nonces travel on the probe's stdin only.)
  local tokens="" ph n3 tl
  for comp in "${COMP_LIST[@]}"; do
    for ph in start end; do
      tl=""; for _ in 1 2 3; do n3=$(rand_hex); tl="${tl:+$tl,}$n3"; tokens="${tokens:+$tokens$'\n'}$n3 $comp/$ph"; done
      TOKS["$comp/$ph"]=$tl
    done
  done
  detector_up "$NET" "$DET" "$SCEN/audit/detector" "$tokens"
  DET_IP=$(container_ip "$DET" "$NET")
  case "${CCG_AC8_HOOK:-}" in detector:die) die "test hook: dying with the detector up" ;; detector:sleep) sleep 60 & wait $! ;; esac
  OWN_CONTAINERS+=("$(nm store)")
  docker run -d --name "$(nm store)" --label "ccg.ac8=$ID" --network "$NET" --user "$UIDGID" --cap-drop ALL --read-only \
    --security-opt no-new-privileges --pids-limit 64 --memory 128m \
    --entrypoint /usr/bin/perl "$IMG" -MIO::Socket::INET -e '
      my $l = IO::Socket::INET->new(Listen => 16, LocalPort => 6379, Proto => "tcp", ReuseAddr => 1) or die; $|=1;
      while (my $c = $l->accept) { close $c }' >/dev/null || die "cannot start the store stand-in"
  STORE_IP=$(container_ip "$(nm store)" "$NET"); STORE_PORT=6379
  sleep 0.5

  # 3. the broker, production shape, under strace when installed (row 8.16)
  say "== broker"
  local sock="$SCEN/broker/broker.sock" sarg=()
  if [ -z "${CCG_AC8_NO_STRACE:-}" ] && command -v strace >/dev/null 2>&1; then
    STRACE=strace
    sarg=(strace -f --seccomp-bpf -ttt -e trace=%network -o "$SCEN/trace/broker.strace")   # --seccomp-bpf: only the traced calls stop the broker (event set verified identical on a loopback R1 run)
    # the positive control of row 8.16: a small program dialling an unlisted loopback address, traced with the SAME flags as the broker
    local cport; cport=$("$PY" -I -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
    strace -f --seccomp-bpf -ttt -e trace=%network -o "$SCEN/trace/control.strace" \
      "$PY" -I -c "import socket; s=socket.socket(); s.settimeout(1); print(s.connect_ex(('127.0.0.1', $cport)))" >/dev/null 2>&1
    "$PY" -I "$HERE/facts.py" set "$FACTS" strace "{\"provisioned\":true,\"trace\":\"trace/broker.strace\",\"control\":\"trace/control.strace\",\"controlTarget\":\"127.0.0.1:$cport\"}"
  else
    "$PY" -I "$HERE/facts.py" set "$FACTS" strace '{"provisioned":false}'
    say "strace is not installed: row 8.16 will be reported unrun (sudo apt-get install -y --no-install-recommends strace)"
  fi
  ( cd "$CPG" && exec env -i PATH=/usr/bin:/bin HOME="$HOME_DIR" LANG=C.UTF-8 CCG_FORBID_REAL_CREDENTIALS=1 CCG_AUDIT_DIR="$SCEN/audit" \
      "${sarg[@]}" "$PY" -I -m ccg.broker.server --system "$SCEN/manifest.yaml" --socket "$sock" ) 2> "$SCEN/broker.log" &
  STRACE_PID=$!
  if [ -n "$STRACE" ]; then wait_for 10 pgrep -P "$STRACE_PID"; BROKER_PID=$(pgrep -P "$STRACE_PID" | head -1); else BROKER_PID=$STRACE_PID; STRACE_PID=""; fi
  wait_for 20 test -S "$sock" || { cat "$SCEN/broker.log" >&2; die "the broker did not come up"; }

  # 4. the component stand-ins: broker directory read-only, proxy variables naming the detector
  say "== component containers"
  for comp in "${COMP_LIST[@]}"; do
    OWN_CONTAINERS+=("$(nm "$comp")")
    docker run -d --name "$(nm "$comp")" --label "ccg.ac8=$ID" --network "$NET" --user "$UIDGID" --cap-drop ALL \
      --security-opt no-new-privileges --read-only --pids-limit 128 --memory 256m \
      -v "$SCEN/broker:/run/ccg-broker:ro" -v "$HERE/netprobe.pl:/ac8/netprobe.pl:ro" \
      -e CCG_BROKER_SOCKET=/run/ccg-broker/broker.sock -e "HTTP_PROXY=http://$DET_IP:3128" -e "HTTPS_PROXY=http://$DET_IP:3128" \
      -e "ALL_PROXY=http://$DET_IP:3128" --entrypoint /usr/bin/perl "$IMG" -e 'sleep 100000' >/dev/null || die "cannot start component $comp"
  done
  sleep 0.5
  say "== P-a, P-b, P-c (start)"
  probe_components start
  say "== §7.3 escape probes"
  run_escape
  "$PY" -I "$HERE/facts.py" set "$FACTS" resolver "\"$([ $STUB_RES = 1 ] && echo stub || echo real)\""

  # 5. the window: start marker, workload and controls, final probe, flush marker
  say "== window opens"
  signal_detector "$DET" USR1 "$SCEN/audit/detector/start.marker"
  sleep 0.2
  local crc=0
  local cargs=(--socket "$sock" --manifest "$SCEN/manifest.yaml" --audit-dir "$SCEN/audit")
  [ "$STUB_MODEL" = 1 ] && cargs+=(--stub-model)
  [ "$SUMMARY" = 1 ] && cargs+=(--summary)
  if [ "$WITNESS" = 1 ]; then cargs+=(--witness-parse --repos-root "${REPOS_ROOT:-$(dirname "$MANIFEST")}"); for l in "${LANGS[@]}"; do cargs+=(--lang "$l"); done; fi
  env CCG_FORBID_REAL_CREDENTIALS=1 CCG_AUDIT_DIR="$SCEN/audit" "$PY" -I "$HERE/controls.py" "${cargs[@]}" || crc=$?
  [ "$crc" = 0 ] || say "controls reported problems (exit $crc): the report will show which control is missing"
  if [ -n "$WORKLOAD" ]; then
    say "== workload: $WORKLOAD"
    env CCG_AUDIT_DIR="$SCEN/audit" CCG_BROKER_SOCKET="$sock" CCG_AC8_SCENARIO="$SCEN" CCG_AC8_NETWORK="$NET" bash -c "$WORKLOAD" || say "the workload exited non-zero ($?)"
  fi
  say "== P-a, P-b, P-c (end)"
  probe_components end
  signal_detector "$DET" USR2 "$SCEN/audit/detector/flush.marker"
  stop_detector "$DET"        # it drains once more on TERM; nothing is judged while it can still log
  say "== window closed"
  t1=$(date +%s)

  # 6. stop the broker (its strace ends with it), capture docker's own events for the scenario
  [ -n "$BROKER_PID" ] || BROKER_PID=$STRACE_PID    # (strace's child not found: stop whatever we started)
  kill -TERM "$BROKER_PID" 2>/dev/null
  wait_for 15 bash -c "! kill -0 $BROKER_PID 2>/dev/null" || { say "the broker ignored TERM: KILL"; kill -KILL "$BROKER_PID" 2>/dev/null; }
  if [ -n "$STRACE_PID" ]; then wait_for 10 bash -c "! kill -0 $STRACE_PID 2>/dev/null" || kill -KILL "$STRACE_PID" 2>/dev/null; fi
  docker events --since "$t0" --until "$((t1 + 2))" --filter type=container --format '{{json .}}' > "$SCEN/docker-events.jsonl" 2>/dev/null

  # 7. the rows
  say "== assertions"
  local aargs=("$SCEN" --json "$SCEN/report.json")
  [ "$STUB_RES" = 1 ] && aargs+=(--stub-resolver)
  "$PY" -I "$HERE/assertions.py" "${aargs[@]}"
}

# ------------------------------------------------------------------ main
CMD="${1:-}"; [ -n "$CMD" ] && shift
case "$CMD" in -h|--help|help|"") usage; exit 0 ;; esac
case "$CMD" in
  verify)
    [ -d "${1:-}" ] || die "usage: run.sh verify <scenario dir> [--stub-resolver]" 64
    d=$1; shift
    exec "$PY" -I "$HERE/assertions.py" "$d" "$@" ;;
  pre-e)
    [ -f "${1:-}" ] || die "usage: run.sh pre-e <manifest> [--stub-resolver]" 64
    exec "$PY" -I "$HERE/assertions.py" pre-e "$@" ;;
  run|detector-control) ;;
  *) usage; exit 64 ;;
esac
while (($#)); do
  case "$1" in
    --id) ID=$2; shift 2 ;;
    --manifest) MANIFEST=$2; shift 2 ;;
    --repos-root) REPOS_ROOT=$2; shift 2 ;;
    --components) COMPONENTS=$2; shift 2 ;;
    --store-standin) STORE_MODE=standin; shift ;;
    --witness-parse) WITNESS=1; shift ;;
    --lang) LANGS+=("$2"); shift 2 ;;
    --workload) WORKLOAD=$2; shift 2 ;;
    --store-container|--store-port) die "$1 is not supported: the store is the harness's own stand-in (--store-standin); see --help" 64 ;;
    --summary) SUMMARY=1; shift ;;
    --stub-resolver) STUB_RES=1; shift ;;
    --stub-model) STUB_MODEL=1; shift ;;
    *) die "unknown option $1" 64 ;;
  esac
done
[ -n "$ID" ] || ID="$(date +%s)$$"
[[ $ID =~ ^[a-z0-9]{1,24}$ ]] || die "--id must match ^[a-z0-9]{1,24}$" 64
[[ $COMPONENTS =~ ^[a-z0-9]+(,[a-z0-9]+)*$ ]] || die "--components must match ^[a-z0-9]+(,[a-z0-9]+)*$" 64
IFS=, read -ra COMP_LIST <<< "$COMPONENTS"
command -v docker >/dev/null && docker info >/dev/null 2>&1 || die "Docker is not reachable" 69
# the isolation checks below read the host's bridge devices: a Docker that is not this host's (a remote DOCKER_HOST or context) makes them meaningless
case "${DOCKER_HOST:-}" in ""|unix://*) ;; *) die "DOCKER_HOST=$DOCKER_HOST is not a local unix socket: the bridge check could not see its networks" 70 ;; esac
[ "$(docker context show 2>/dev/null)" = default ] || die "the docker context is not 'default': the bridge check could not see its networks" 70
[ -x "$PY" ] || die "no Python at $PY" 69
SCEN="$BASE/$ID"; NET=$(nm net); DET=$(nm detector)
[ ! -e "$SCEN" ] || die "scenario directory $SCEN already exists: scenarios are never reused" 64
mkdir -p "$BASE" && chmod 700 "$BASE" && [ -O "$BASE" ] && [ ! -L "$BASE" ] || die "$BASE is not a real directory owned by this user" 70
mkdir "$SCEN" && chmod 700 "$SCEN" || die "cannot create $SCEN"
docker network inspect "$NET" >/dev/null 2>&1 && die "network $NET already exists: scenarios never reuse a network" 64
ac8_install_traps cleanup
say "scenario $ID in $SCEN"
rc=0
case "$CMD" in
  detector-control) detector_control; rc=$? ;;
  run) main_run; rc=$? ;;
esac
exit "$rc"
