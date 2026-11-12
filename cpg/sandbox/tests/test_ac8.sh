#!/usr/bin/env bash
# test_ac8.sh -- live tests of the AC-8 harness's shell and Perl (cpg/sandbox/ac8/: run.sh, detector.pl, netprobe.pl). The Python
# (window.py, assertions.py) has its own pytest suites; this one runs what needs Docker.
#
# Usage: cpg/sandbox/tests/test_ac8.sh [--runner PATH] [--only 1,2,...] [--ac8-dir DIR]
#   --ac8-dir  the harness directory under test (default ../ac8); used by tests/mutants_ac8.sh on a mutated COPY.
#   --only     run a subset: 1 detector control scenario, 2 the probe and the network on a throwaway network, 3 the main run (baseline: a
#              self-test run, so exit 3 `incomplete`), 4 a component that connects to the detector (the run must fail), 5 no strace (row
#              8.16 unrun, exit 3), 6 a run that dies, is TERMinated, or is refused by P-e cleans up after itself / creates nothing
#              Tests 3-5 run P-a: the probe connects to / sends to every address and listening port of THIS host from inside the scenario
#              network (expected: every one fails unreachable). Tests 1, 2 and 6 only ever address the harness's own detector.
# Safety: every container and network is named ccg-ac8-<id>-* and labelled ccg.ac8=<id>, removed by exact name and verified gone;
# the networks are --internal (no route out) except ONE plain bridge network of test 2, on which a container only READS /proc/net/route
# (no connect, no packet). Test 2 aims the probe only at its own detector container. Nothing under
# ~/.config/ccg. Exit 0 iff every check passed; 3 if a test could not run (NOT RUN), 1 on a failure.
set -uo pipefail
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AC8="$HERE/../ac8"; CPG="$(cd "$HERE/../.." && pwd)"
ONLY=""
while (($#)); do
  case "$1" in
    --ac8-dir) AC8=$(cd "$2" && pwd); shift 2 ;;
    --only) ONLY=",$2,"; shift 2 ;;
    -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
want() { [ -z "$ONLY" ] || [[ $ONLY == *",$1,"* ]]; }
PASS=0; FAIL=0; FAILED=(); NOTRUN=0; NOTRUN_LIST=()
ok()  { PASS=$((PASS + 1)); printf '  ok   %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); FAILED+=("$1"); printf '  FAIL %s\n' "$1"; [ -n "${2:-}" ] && printf '       %s\n' "$2"; }
notrun() { NOTRUN=$((NOTRUN + 1)); NOTRUN_LIST+=("$1"); printf '  NOT RUN %s\n' "$1"; }
chk() { local name=$1; shift; if "$@"; then ok "$name"; else bad "$name"; fi; }
section() { printf '\n== %s\n' "$1"; }

PY="$CPG/.venv/bin/python"
[ -x "$PY" ] || { echo "no $PY"; exit 3; }
# shellcheck disable=SC1091
. "$AC8/lib.sh"; AC8_PY="$PY"
docker info >/dev/null 2>&1 || { echo "Docker is not reachable"; exit 3; }
IMG="$(env -i PATH=/usr/bin:/bin "$CPG/infra/toolchain/build.sh" --print-tag 2>/dev/null)"
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "toolchain image $IMG is not built"; exit 3; }
HOME_DIR="$(getent passwd "$(id -u)" | cut -d: -f6)"
export CCG_AC8_BASE="$HOME_DIR/.local/state/ccg-ac8-test-$$"
T=$(mktemp -d /tmp/ccg-ac8-test.XXXXXX)
UIDGID="$(id -u):$(id -g)"
TID="t$$"                      # test 2's own scenario id
OTHERS_BEFORE=$(docker ps --format '{{.Names}} {{.Status}}' | grep -v '^ccg-ac8-' | sed 's/ Up .*/ Up/' | sort)
NET2="ccg-ac8-$TID-net"; OPEN2="ccg-ac8-$TID-open"; DET2="ccg-ac8-$TID-detector"; C2="ccg-ac8-$TID-comp"; C2O="ccg-ac8-$TID-opencomp"; OPEN6="ccg-ac8-$TID-open6"; C2O6="ccg-ac8-$TID-open6comp"
cleanup() {
  local x
  for x in "$C2" "$C2O" "$C2O6" "$DET2"; do docker rm -f "$x" >/dev/null 2>&1; done
  for x in "$NET2" "$OPEN2" "$OPEN6"; do docker network rm "$x" >/dev/null 2>&1; done
  chmod -R u+w "$T" "$CCG_AC8_BASE" 2>/dev/null; rm -rf "$T" "$CCG_AC8_BASE"
}
trap cleanup EXIT
leftovers() { # <id>
  { docker ps -a --filter "label=ccg.ac8=$1" --format '{{.Names}}'; docker network ls --filter "label=ccg.ac8=$1" --format '{{.Name}}'; } | grep . || true
}

# ================================================================== 1: the detector control scenario
if want 1; then
  section "test 1: the detector control scenario"
  OUT=$(timeout -s KILL 150 "$AC8/run.sh" detector-control --id "t$$c" 2>&1); RC=$?
  chk "1a: detector-control exits 0 and prints PASS (exit $RC)" test "$RC" = 0 -a -n "$(grep -x 'detector control: PASS' <<< "$OUT")"
  chk "1b: it reports its own cleanup verified" grep -q 'cleanup verified' <<< "$OUT"
  chk "1c: no container or network of the scenario remains" test -z "$(leftovers "t$$c")"
  NCTL=$("$PY" -I -c 'import sys; sys.path.insert(0, sys.argv[1]); import window; print(window.CONTROL_CONNS)' "$AC8")
  chk "1d: the control recorded exactly $NCTL conns (token-less, wrong nonce, replay, and the flood) and one labelled probe record" \
    test "$(wc -l < "$CCG_AC8_BASE/t$$c/detector-control/conn.jsonl")" = "$NCTL" -a "$(wc -l < "$CCG_AC8_BASE/t$$c/detector-control/probe.jsonl")" = 1 -a -n "$(grep -c '"probe":"control/ok"' "$CCG_AC8_BASE/t$$c/detector-control/probe.jsonl")"
  chk "1g: the control's flush marker reports residue 0" grep -q '"residue":0' "$CCG_AC8_BASE/t$$c/detector-control/flush.marker"
  chk "1h: the control detector wrote a final marker (its TERM drain finished) reporting residue 0" grep -q '"residue":0' "$CCG_AC8_BASE/t$$c/detector-control/final.marker"
  chk "1e: a second run with the same id is refused (scenarios are never reused)" bash -c "! '$AC8/run.sh' detector-control --id 't$$c' >/dev/null 2>&1"
  chk "1f: a bad id is refused with exit 64" bash -c "'$AC8/run.sh' detector-control --id 'Bad_ID' >/dev/null 2>&1; test \$? = 64"
fi

# ================================================================== 2: the probe and the network
probe() { # <container> <key=value...>  -> output on stdout; every argument goes on the probe's STDIN (never argv), as run.sh does
  local c=$1; shift
  printf '%s\n' "$@" | docker exec -i "$c" /usr/bin/perl /ac8/netprobe.pl 2>&1
}
if want 2; then
  section "test 2: netprobe.pl and the scenario network (throwaway)"
  mkdir -p "$T/det2" "$T/brk"; chmod 700 "$T/det2" "$T/brk"
  N1=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n'); N2=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n'); N3=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')
  printf '%s comp/start\n%s comp/start\n%s comp/start\n' "$N1" "$N2" "$N3" > "$T/det2/probe-tokens"; chmod 600 "$T/det2/probe-tokens"
  docker network create --internal -o com.docker.network.bridge.gateway_mode_ipv4=isolated --label "ccg.ac8=$TID" "$NET2" >/dev/null
  docker network create --label "ccg.ac8=$TID" "$OPEN2" >/dev/null
  BR="br-$(docker network inspect "$NET2" --format '{{.Id}}' | cut -c1-12)"
  bridge_has_address "$BR"; RC2A=$?
  chk "2a: the isolated bridge exists and carries no address (bridge_has_address: 1; $RC2A)" test "$RC2A" = 1
  bridge_has_address "br-nonexistent0"; RC2A2=$?
  chk "2a2: a bridge that does not exist is UNKNOWN (2), not 'no address' ($RC2A2)" test "$RC2A2" = 2
  docker run -d --name "$DET2" --label "ccg.ac8=$TID" --network "$NET2" --user "$UIDGID" --cap-drop ALL --read-only \
    -v "$T/det2:/detector:rw" -v "$AC8/detector.pl:/detector.pl:ro" --entrypoint /usr/bin/perl "$IMG" /detector.pl /detector >/dev/null
  for _ in $(seq 50); do [ -f "$T/det2/ready" ] && break; sleep 0.1; done
  DIP=$(docker inspect --format "{{(index .NetworkSettings.Networks \"$NET2\").IPAddress}}" "$DET2")
  docker run -d --name "$C2" --label "ccg.ac8=$TID" --network "$NET2" --user "$UIDGID" --cap-drop ALL --read-only \
    -v "$T/brk:/run/ccg-broker:ro" -v "$AC8/netprobe.pl:/ac8/netprobe.pl:ro" --entrypoint /usr/bin/perl "$IMG" -e 'sleep 100000' >/dev/null
  # 2b/2c: no default route, no DNS, no AF_VSOCK. (The isolation against the host's real addresses and ports is P-a, run by tests 3-5 through
  # run.sh; this test aims the probe only at the detector, a container of its own network.)
  OUT=$(probe "$C2" ifaces=eth0 broker=/run/ccg-broker/none.sock "detname=$DET2" "detip=$DIP")
  echo "$OUT" > "$T/p2b.txt"
  chk "2b: no default route, IPv4 or IPv6 (A1b)" test -n "$(grep '^CHECK A1b-v4 PASS' <<< "$OUT")" -a -n "$(grep '^CHECK A1b-v6 PASS' <<< "$OUT")"
  chk "2c: no DNS to the outside (A2: getent ahosts example.com exits 2), AF_VSOCK refused (A7)" test -n "$(grep '^CHECK A2-ahosts PASS' <<< "$OUT")" -a -n "$(grep '^CHECK A7 PASS' <<< "$OUT")"
  chk "2c2: the resolver itself works: getent hosts localhost (A2-localhost) and Docker's embedded DNS resolving the detector's name (A2-embedded)" \
    test -n "$(grep '^CHECK A2-localhost PASS' <<< "$OUT")" -a -n "$(grep '^CHECK A2-embedded PASS' <<< "$OUT")"
  OUT2=$(probe "$C2" broker=/run/ccg-broker/none.sock "detname=ccg-ac8-$TID-no-such-container" "detip=$DIP")
  chk "2c3: A2-embedded is FAIL for a name the embedded DNS cannot resolve (the check can go red)" test -n "$(grep '^CHECK A2-embedded FAIL' <<< "$OUT2")"
  chk "2c4: a getent that is missing is not a pass: with /usr/bin/getent renamed in a copy of the probe, A2-ahosts reports FAIL (status 127, want 2)" \
    bash -c "docker exec -i '$C2' /usr/bin/perl -e 'local \$/; open(H, \"<\", \"/ac8/netprobe.pl\"); \$s = <H>; \$s =~ s{/usr/bin/getent}{/usr/bin/no-such-getent}g; eval \$s' </dev/null 2>&1 | grep -q '^CHECK A2-ahosts FAIL.*status=127'"
  # 2d: the probe's NEGATIVE control: aimed at the detector's own address (reachable by design) it must report FAIL for TCP and UDP
  OUT=$(probe "$C2" "v4=$DIP" tcp=80 udp=9999 broker=/run/ccg-broker/none.sock)
  chk "2d: a reachable target is reported FAIL for TCP and for UDP (the probe can see a leak)" \
    test -n "$(grep '^CHECK A1d-tcp FAIL '"$DIP"':80 CONNECTED' <<< "$OUT")" -a -n "$(grep '^CHECK A1d-udp FAIL '"$DIP"':9999 SENT' <<< "$OUT")"
  chk "2e: its PROBE-DONE counts the failures" test -n "$(grep -E '^PROBE-DONE checks=[0-9]+ fail=([2-9]|[1-9][0-9])' <<< "$OUT")"
  # 2f/2g: the detector's classes of connection (probe nonce / wrong / replay / none)
  sleep 0.6; N0=$(wc -l < "$T/det2/conn.jsonl"); P0=$(wc -l < "$T/det2/probe.jsonl")
  OUT=$(probe "$C2" "det=$DIP:80,$DIP:443,$DIP:3128" "tokens=$N1,$N2,$N3" broker=/run/ccg-broker/none.sock)
  sleep 0.6
  chk "2f: probe connections with their own single-use nonces are filed as probe records labelled comp/start, not conns (+3 probe, +0 conn)" \
    test "$(grep -c '^CHECK posctl-det PASS' <<< "$OUT")" = 3 -a "$(wc -l < "$T/det2/probe.jsonl")" = "$((P0 + 3))" -a "$(wc -l < "$T/det2/conn.jsonl")" = "$N0" -a -n "$(grep '^PROBE-SENT det=3' <<< "$OUT")" -a "$(grep -c '"probe":"comp/start"' "$T/det2/probe.jsonl")" = 3
  OUT=$(probe "$C2" "det=$DIP:80" "tokens=$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')" broker=/run/ccg-broker/none.sock)
  sleep 0.6
  chk "2g: the same connection with a wrong nonce is a conn record (+1 conn, +0 probe)" \
    test "$(wc -l < "$T/det2/conn.jsonl")" = "$((N0 + 1))" -a "$(wc -l < "$T/det2/probe.jsonl")" = "$((P0 + 3))"
  OUT=$(probe "$C2" "det=$DIP:80" "tokens=$N1" broker=/run/ccg-broker/none.sock)
  sleep 0.6
  chk "2g2: a REPLAYED nonce is a conn record (single-use: +1 conn, +0 probe)" \
    test "$(wc -l < "$T/det2/conn.jsonl")" = "$((N0 + 2))" -a "$(wc -l < "$T/det2/probe.jsonl")" = "$((P0 + 3))"
  OUT=$(probe "$C2" "det=$DIP:80" broker=/run/ccg-broker/none.sock)
  sleep 0.6
  chk "2g3: a token-less connection is a conn record (+1 conn, +0 probe)" \
    test "$(wc -l < "$T/det2/conn.jsonl")" = "$((N0 + 3))" -a "$(wc -l < "$T/det2/probe.jsonl")" = "$((P0 + 3))"
  docker exec "$C2" /usr/bin/perl -MIO::Socket::INET -e '$s=IO::Socket::INET->new(PeerAddr=>$ARGV[0],PeerPort=>3128,Timeout=>3) or die; print $s "CONNECT x:443 HTTP/1.1\r\n\r\n"; close $s' "$DIP"; sleep 0.6
  L=$(tail -1 "$T/det2/conn.jsonl")
  chk "2h: a proxy-style connection is a conn with the detector's address, port 3128 and its bytes (bytesIn>0, bytesOut=0)" \
    "$PY" -I -c 'import json,sys; r=json.loads(sys.argv[1]); assert r["dstPort"]==3128 and r["bytesIn"]>0 and r["bytesOut"]==0 and r["dstAddr"]==sys.argv[2] and set(r)=={"ts","srcAddr","srcPort","dstAddr","dstPort","bytesOut","bytesIn"}' "$L" "$DIP"
  # 2i/2j: the markers (written once, in order)
  docker kill --signal=USR1 "$DET2" >/dev/null; sleep 0.5; docker kill --signal=USR2 "$DET2" >/dev/null; sleep 0.5
  chk "2i: both markers exist and the verdict function sees a window (start <= flush)" "$PY" -I -c '
import sys; sys.path.insert(0, sys.argv[1]); import window as W
m, e = W.read_markers(__import__("pathlib").Path(sys.argv[2])); assert not e and m["start"] and m["flush"] and m["start"] <= m["flush"]' "$AC8" "$T/det2"
  S0=$(cat "$T/det2/start.marker"); docker kill --signal=USR1 "$DET2" >/dev/null; sleep 0.5
  chk "2j: a second signal does not rewrite a marker" test "$(cat "$T/det2/start.marker")" = "$S0"
  chk "2k: the verdict over this detector's log FAILS (its conn records predate the start marker: conn-outside-window:before-start)" bash -c "'$PY' -I '$AC8/window.py' verdict '$T/det2' | grep -q 'conn-outside-window:before-start'"
  # 2l: the default-route check sees a route: a container on a plain bridge network (reads /proc/net/route only; no packet is sent)
  docker run -d --name "$C2O" --label "ccg.ac8=$TID" --network "$OPEN2" --user "$UIDGID" --cap-drop ALL --read-only \
    -v "$AC8/netprobe.pl:/ac8/netprobe.pl:ro" --entrypoint /usr/bin/perl "$IMG" -e 'sleep 100000' >/dev/null
  OUT=$(probe "$C2O" broker=/run/ccg-broker/none.sock)
  chk "2l: a network WITH a default route fails A1b-v4 (the probe can see one)" test -n "$(grep '^CHECK A1b-v4 FAIL' <<< "$OUT")"
  # 2n: the same for IPv6: a ULA-subnet bridge (fd00:ac8:131::/64, no traffic) gives the container a ::/0 route, which A1b-v6 must see
  docker network create --ipv6 --subnet fd00:ac8:131::/64 --label "ccg.ac8=$TID" "$OPEN6" >/dev/null
  docker run -d --name "$C2O6" --label "ccg.ac8=$TID" --network "$OPEN6" --user "$UIDGID" --cap-drop ALL --read-only \
    -v "$AC8/netprobe.pl:/ac8/netprobe.pl:ro" --entrypoint /usr/bin/perl "$IMG" -e 'sleep 100000' >/dev/null
  OUT=$(probe "$C2O6" broker=/run/ccg-broker/none.sock)
  chk "2n: a network with an IPv6 default route fails A1b-v6" test -n "$(grep '^CHECK A1b-v6 FAIL' <<< "$OUT")"
  for x in "$C2" "$C2O" "$C2O6" "$DET2"; do docker rm -f "$x" >/dev/null 2>&1; done
  for x in "$NET2" "$OPEN2" "$OPEN6"; do docker network rm "$x" >/dev/null 2>&1; done
  chk "2m: this test's own containers and networks are gone" test -z "$(leftovers "$TID")"
fi

# ================================================================== 3-5: the main run, on a self-contained fixture
fixture() {
  mkdir -p "$T/fx/front" "$T/fx/docs"
  echo 'def f(): return 1' > "$T/fx/front/a.py"; echo 'def g(): return 2' > "$T/fx/docs/b.py"
  local port; port=$("$PY" -I -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1]); s.close()')
  cat > "$T/fx/system.yaml" <<EOT
schemaVersion: 1
system: ac8smoke
egress:
  schemaVersion: 1
  local:
    - {id: stubmodel, kind: model, endpoint: "http://127.0.0.1:$port", paths: ["/v1/chat/completions"]}
  outbound:
    - {id: idp, kind: idp, endpoint: "https://idp.ac8.invalid:443", paths: ["/who"]}
  hosted:
    - {id: hosted, kind: model, endpoint: "https://api.ac8.invalid:443", paths: ["/v1/chat/completions"]}
repos:
  - {repoId: front, path: front, policy: {egress: local-only}}
  - {repoId: docs, path: docs, policy: {egress: hosted}}
EOT
}
rowstatus() { "$PY" -I -c 'import json,sys; r=json.load(open(sys.argv[1])); print(next(x["status"] for x in r["rows"] if x["id"]==sys.argv[2]))' "$1" "$2"; }
overall() { "$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["overall"])' "$1"; }
HAVE_STRACE=0; command -v strace >/dev/null 2>&1 && HAVE_STRACE=1
if want 3 || want 4 || want 5 || want 6; then fixture; fi
if want 3; then
  section "test 3: the main run on a mixed fixture (stub local model, stand-in store)"
  OUT=$(timeout -s KILL 580 "$AC8/run.sh" run --id "t$$m" --manifest "$T/fx/system.yaml" --store-standin --witness-parse --stub-model --stub-resolver 2>&1); RC=$?
  R="$CCG_AC8_BASE/t$$m/report.json"
  # this run uses stand-ins (--store-standin --witness-parse --stub-model --stub-resolver): it can never be `pass`, only `incomplete` (exit 3)
  chk "3a: exit 3, overall incomplete, selfTest names the four stand-ins (exit $RC)" test "$RC" = 3 -a "$(overall "$R" 2>/dev/null)" = incomplete -a "$("$PY" -I -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["selfTest"]))' "$R" 2>/dev/null)" = "store-standin,stub-model,stub-resolver,witness-parse"
  for r in 8.1 8.3 8.5 8.6 8.7 8.8 8.9 8.10 8.11 8.13 8.14 8.15 close; do chk "3b: row $r pass" test "$(rowstatus "$R" "$r" 2>/dev/null)" = pass; done
  for r in 8.2 8.4 8.12; do chk "3c: row $r vacuous" test "$(rowstatus "$R" "$r" 2>/dev/null)" = vacuous; done
  chk "3d: row 8.16 is $([ $HAVE_STRACE = 1 ] && echo pass || echo unrun)" test "$(rowstatus "$R" 8.16 2>/dev/null)" = "$([ $HAVE_STRACE = 1 ] && echo pass || echo unrun)"
  chk "3e: zero conn records in the main run's detector log" test "$(wc -l < "$CCG_AC8_BASE/t$$m/audit/detector/conn.jsonl")" = 0
  chk "3f: the broker socket and its audit were the scenario's (no ~/.config/ccg path in the scenario)" test -z "$(grep -rl '\.config/ccg' "$CCG_AC8_BASE/t$$m/audit" "$CCG_AC8_BASE/t$$m/facts.json" "$CCG_AC8_BASE/t$$m/report.json" 2>/dev/null)"
  chk "3g: cleanup verified, nothing of the scenario remains" test -n "$(grep 'cleanup verified' <<< "$OUT")" -a -z "$(leftovers "t$$m")"
  chk "3h: no container of the scenario is left, and the broker process is gone" test -z "$(pgrep -f "ccg.broker.server --system $CCG_AC8_BASE/t$$m" || true)"
  chk "3i: 'verify' re-evaluates the directory to the same verdict (exit 3)" bash -c "'$AC8/run.sh' verify '$CCG_AC8_BASE/t$$m' --stub-resolver >/dev/null 2>&1; test \$? = 3"
  chk "3j: the detector was stopped before the verdict and the flush marker reports residue 0" test "$(docker inspect -f '{{.State.Running}}' "ccg-ac8-t$$m-detector" 2>/dev/null || echo gone)" != true -a -n "$(grep -c '"residue":0' "$CCG_AC8_BASE/t$$m/audit/detector/flush.marker")"
  chk "3k: every probe record is attributed (P-d ok in the report)" test "$("$PY" -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["preconditions"]["P-d"]["ok"])' "$R" 2>/dev/null)" = True
fi
if want 4; then
  section "test 4: a component that connects to the detector makes the run fail (and says why)"
  WL='ID=$(basename "$CCG_AC8_SCENARIO"); DIP=$(docker inspect --format "{{(index .NetworkSettings.Networks \"$CCG_AC8_NETWORK\").IPAddress}}" ccg-ac8-$ID-detector); docker exec ccg-ac8-$ID-answer /usr/bin/perl -MIO::Socket::INET -e '"'"'$s=IO::Socket::INET->new(PeerAddr=>$ARGV[0],PeerPort=>3128,Timeout=>3) or die "no connect"; print $s "CONNECT example.com:443 HTTP/1.1\r\n\r\n"; close $s'"'"' "$DIP"'
  OUT=$(timeout -s KILL 300 "$AC8/run.sh" run --id "t$$n" --manifest "$T/fx/system.yaml" --store-standin --stub-model --stub-resolver --workload "$WL" 2>&1); RC=$?
  R="$CCG_AC8_BASE/t$$n/report.json"
  chk "4a: exit 1, overall fail (exit $RC)" test "$RC" = 1 -a "$(overall "$R" 2>/dev/null)" = fail
  chk "4b: row 8.1 fails, naming the conn" test "$(rowstatus "$R" 8.1 2>/dev/null)" = fail -a -n "$(grep -c 'conn-in-window' "$R")"
  chk "4c: the conn log holds exactly the one connection" test "$(wc -l < "$CCG_AC8_BASE/t$$n/audit/detector/conn.jsonl")" = 1
  chk "4d: the proxy-style connection to port 3128 is what was recorded" grep -q '"dstPort":3128' "$CCG_AC8_BASE/t$$n/audit/detector/conn.jsonl"
  chk "4e: cleanup verified, nothing of the scenario remains" test -n "$(grep 'cleanup verified' <<< "$OUT")" -a -z "$(leftovers "t$$n")"
fi
if want 5; then
  section "test 5: strace not installed (CCG_AC8_NO_STRACE=1): row 8.16 unrun, never passed"
  OUT=$(CCG_AC8_NO_STRACE=1 timeout -s KILL 580 "$AC8/run.sh" run --id "t$$s" --manifest "$T/fx/system.yaml" --store-standin --witness-parse --stub-model --stub-resolver 2>&1); RC=$?
  R="$CCG_AC8_BASE/t$$s/report.json"
  chk "5a: exit 3, overall incomplete (exit $RC)" test "$RC" = 3 -a "$(overall "$R" 2>/dev/null)" = incomplete
  chk "5b: row 8.16 is unrun" test "$(rowstatus "$R" 8.16 2>/dev/null)" = unrun
  chk "5c: every other computed row still passes" test "$(rowstatus "$R" 8.1 2>/dev/null)" = pass -a "$(rowstatus "$R" 8.7 2>/dev/null)" = pass
  chk "5d: the run said how to provision strace" grep -q 'apt-get install -y --no-install-recommends strace' <<< "$OUT"
  chk "5e: cleanup verified, nothing of the scenario remains" test -z "$(leftovers "t$$s")"
fi

if want 6; then
  section "test 6: a run that dies, is terminated, or is refused by P-e: cleans up by exact name / creates nothing"
  # (the test hook stops the run once its detector is up: before any probe, so nothing here addresses a host address)
  OUT=$(CCG_AC8_HOOK=detector:die timeout -s KILL 150 "$AC8/run.sh" run --id "t$$d" --manifest "$T/fx/system.yaml" --store-standin --stub-resolver 2>&1); RC=$?
  chk "6a: the run dies with exit 70 (exit $RC)" test "$RC" = 70
  chk "6b: the detector and the networks it had created are gone (cleanup verified)" test -n "$(grep 'cleanup verified' <<< "$OUT")" -a -z "$(leftovers "t$$d")"
  # 6c: a TERM mid-run: cleanup runs, the run EXITS (143), and nothing of the scenario remains
  CCG_AC8_HOOK=detector:sleep "$AC8/run.sh" run --id "t$$g" --manifest "$T/fx/system.yaml" --store-standin --stub-resolver > "$T/term.out" 2>&1 &
  TP=$!
  for _ in $(seq 600); do docker ps --format '{{.Names}}' | grep -qx "ccg-ac8-t$$g-detector" && break; sleep 0.2; done
  sleep 0.5; kill -TERM "$TP"; wait "$TP"; RCT=$?
  chk "6c: SIGTERM mid-run: the run exits 143 (exit $RCT), not carrying on" test "$RCT" = 143
  chk "6d: and it cleaned up: cleanup verified, no container or network of the scenario remains" test -n "$(grep 'cleanup verified' "$T/term.out")" -a -z "$(leftovers "t$$g")" -a -z "$(grep 'P-a/b/c' "$T/term.out")"
  # 6f: a pre-existing UNLABELLED container with the name the run wants for its detector is left alone (cleanup checks the label first)
  FOREIGN="ccg-ac8-t$$f-detector"
  docker create --name "$FOREIGN" --entrypoint /usr/bin/perl "$IMG" -e 'sleep 1' >/dev/null 2>&1
  OUT=$(timeout -s KILL 140 "$AC8/run.sh" run --id "t$$f" --manifest "$T/fx/system.yaml" --store-standin --stub-resolver 2>&1); RC=$?
  chk "6f: the run dies (cannot start its detector: the name is taken) with exit 70 (exit $RC)" test "$RC" = 70 -a -n "$(grep 'cannot start the detector' <<< "$OUT")"
  chk "6g: the unlabelled foreign container of that name is still there (not removed by exact name alone)" test "$(docker inspect -f '{{.Name}}' "$FOREIGN" 2>/dev/null)" = "/$FOREIGN"
  docker rm -f "$FOREIGN" >/dev/null 2>&1
  # 6e: P-e fails BEFORE the run: a manifest naming a real host refuses the run before any network or container exists
  sed 's/idp.ac8.invalid/idp.example.com/' "$T/fx/system.yaml" > "$T/fx/real-host.yaml"
  OUT=$(timeout -s KILL 60 "$AC8/run.sh" run --id "t$$p" --manifest "$T/fx/real-host.yaml" --store-standin --stub-resolver 2>&1); RC=$?
  chk "6e: P-e refuses the run before it starts (exit 1, no network or container ever created)" test "$RC" = 1 -a -n "$(grep 'P-e failed before the run' <<< "$OUT")" -a -z "$(grep -E '== (detector control|scenario network)' <<< "$OUT")" -a -z "$(leftovers "t$$p")"
fi

# ================================================================== nothing outside the harness was touched
section "the rest of the host"
OTHERS_AFTER=$(docker ps --format '{{.Names}} {{.Status}}' | grep -v '^ccg-ac8-' | sed 's/ Up .*/ Up/' | sort)
chk "every container that was not ours is still up, unchanged" test "$OTHERS_BEFORE" = "$OTHERS_AFTER"
chk "no ccg-ac8-* container or network remains on the host" test -z "$(docker ps -a --format '{{.Names}}' | grep '^ccg-ac8-' || true; docker network ls --format '{{.Name}}' | grep '^ccg-ac8-' || true)"

printf '\n%d passed, %d failed, %d not run\n' "$PASS" "$FAIL" "$NOTRUN"
((NOTRUN)) && printf 'NOT RUN: %s\n' "${NOTRUN_LIST[@]}"
((FAIL)) && { printf 'FAILED: %s\n' "${FAILED[@]}"; exit 1; }
((NOTRUN)) && exit 3
exit 0
