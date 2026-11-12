#!/usr/bin/env bash
# mutants_ac8.sh -- mutation test of test_ac8.sh over the AC-8 harness's shell and Perl (run.sh, detector.pl, netprobe.pl; P1-31).
# Each mutant is a COPY of cpg/sandbox/ in a fresh scratch directory (cpg/.venv, ccg/ and infra/ symlinked in) with one exact text edit;
# the real files are never touched. A mutant is KILLED iff `test_ac8.sh --ac8-dir <copy> --only <tests>` exits non-zero. Every edit must
# apply exactly once. Containers and networks a failing mutant leaves behind are removed by exact name (only those created during the
# mutant's run whose names start ccg-ac8-t).
#
# Tests 3-5 (the main run, P-a against THIS host's addresses and ports) are never started by an author's mutation run; a mutant keyed to them
# is only checked for applying (MUTANT_DRY=1) and is run, if at all, by the human who owns the host. Mutants of detector.pl are also covered
# Docker-free by tests/test_ac8_detector.py (see mutants_ac8.py DT*).
# NOT RUN, on purpose: a mutant that leaves the scenario network reachable from the host (e.g. creating it without the isolated gateway AND
# without run.sh's bridge-address check). Its P-a probe would connect to the host's real listeners (falkordb-dev among them), which this
# harness must never do. The bridge-address check (mutant S01) is what the isolated-gateway option is verified by here.
# Usage: cpg/sandbox/tests/mutants_ac8.sh [id ...]   (MUTANT_DRY=1: only check the edits apply)
set -uo pipefail
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBX="$HERE/.."; CPG="$SBX/.."
W=$(mktemp -d "${TMPDIR:-/tmp}/p131_shmut.XXXXXX"); trap 'rm -rf "$W"' EXIT
SEL=" $* "
# id | file | tests | description  (old and new text are in the python table below)
python3 -I - "$W" "$SBX" "$CPG" "$SEL" "${MUTANT_DRY:-}" > "$W/manifest" <<'PY'
import os, shutil, sys
work, sbx, cpg, sel, dry = sys.argv[1:6]
R, D, N = "ac8/run.sh", "ac8/detector.pl", "ac8/netprobe.pl"
M = [
 ("S01", R, "1", "network created without the isolated gateway option",
  "docker network create --internal -o com.docker.network.bridge.gateway_mode_ipv4=isolated --label", "docker network create --internal --label"),
 ("S02", R, "1", "the control skips the start marker", "  signal_detector \"$cdet\" USR1 \"$ctl/start.marker\"\n", "  :\n"),
 ("S03", R, "1", "the control skips the flush marker", "  signal_detector \"$cdet\" USR2 \"$ctl/flush.marker\"\n", "  :\n"),
 ("S04", R, "6", "cleanup removes no container", "&& docker rm -f \"$c\" >/dev/null 2>&1 ;; esac", "&& : ;; esac"),
 ("S21", R, "6", "cleanup does not check the label before removing", "[ \"$(docker inspect -f '{{index .Config.Labels \"ccg.ac8\"}}' \"$c\" 2>/dev/null)\" = \"$ID\" ] && docker rm", "docker rm"),
 ("S22", R, "6", "a signal trap does not exit (the cleanup runs and the script resumes)", "ac8_install_traps cleanup", "trap cleanup EXIT; trap cleanup INT TERM HUP"),
 ("S23", R, "6", "P-e is not run before the run", "  \"$PY\" -I \"$HERE/assertions.py\" \"${pe[@]}\" >&2 || {", "  true || {"),
 ("S24", R, "1", "the control's flood is two connections, not CONTROL_FLOOD", "print(window.CONTROL_FLOOD - 1)", "print(1)"),
 ("S05", R, "1", "a scenario id may be reused", "[ ! -e \"$SCEN\" ] || die", "[ ! -e \"$SCEN.never\" ] || die"),
 ("S06", R, "1", "any id is accepted", "[[ $ID =~ ^[a-z0-9]{1,24}$ ]] || die", "true || die"),
 ("S07", R, "3", "the main run never signals the flush marker", "  signal_detector \"$DET\" USR2 \"$SCEN/audit/detector/flush.marker\"\n", "  :\n"),
 ("S08", R, "3", "the main run never signals the start marker", "  signal_detector \"$DET\" USR1 \"$SCEN/audit/detector/start.marker\"\n", "  :\n"),
 ("S09", R, "3", "the end-of-run probe is skipped", "  probe_components end\n", "  :\n"),
 ("S10", R, "3", "the start probe is skipped", "  probe_components start\n", "  :\n"),
 ("S11", R, "3", "no in-run controls", "  env CCG_AUDIT_DIR=\"$SCEN/audit\" \"$PY\" -I \"$HERE/controls.py\" \"${cargs[@]}\" || crc=$?\n", "  :\n"),
 ("S12", R, "3", "no docker events capture", "  docker events --since", "  : docker events --since"),
 ("S13", R, "3", "the escape probes do not run in the scenario", "  run_escape\n", "  :\n"),
 ("S14", R, "3", "the detector positive control targets are not passed", "  det=\"$DET_IP:80,$DET_IP:443,$DET_IP:3128\"", "  det=\"\""),
 ("S26", R, "3", "the detector is not stopped before the verdict", "  stop_detector \"$DET\"        #", "  : #"),
 ("S27", R, "3", "a store container could be attached: the self-test list omits store-standin", "st+=(store-standin); ", ""),
 ("S15", R, "3", "the detector control scenario is not run first", "  if ! detector_control; then", "  if ! true; then"),
 ("S16", R, "3", "the broker directory is mounted read-write", "-v \"$SCEN/broker:/run/ccg-broker:ro\"", "-v \"$SCEN/broker:/run/ccg-broker:rw\""),
 ("S17", R, "3", "the workload's exit is the run's (assertions never run)", "  \"$PY\" -I \"$HERE/assertions.py\" \"${aargs[@]}\"\n}", "  true\n}"),
 ("S18", R, "3", "strace's positive control trace is not made", "    strace -f -ttt -e trace=%network -o \"$SCEN/trace/control.strace\"", "    : strace -f -ttt -e trace=%network -o \"$SCEN/trace/control.strace\""),
 ("S19", R, "3", "the broker is not traced", "sarg=(strace -f --seccomp-bpf -ttt -e trace=%network -o \"$SCEN/trace/broker.strace\")", "sarg=()"),
 ("S20", R, "3", "the witness parse is not passed on", "--witness-parse --repos-root", "--repos-root"),
 ("D01", D, "2", "every connection is a conn (the nonces are never honoured)", "exists $label{$1} && !$used{$1}++", "0"),
 ("D02", D, "2", "any nonce is accepted", "exists $label{$1} && !$used{$1}++", "($label{$1} //= \"x\") && !$used{$1}++"),
 ("D07", D, "2", "a nonce may be replayed", "exists $label{$1} && !$used{$1}++", "exists $label{$1}"),
 ("D03", D, "2", "a marker is rewritten on every signal", "O_WRONLY|O_CREAT|O_EXCL", "O_WRONLY|O_CREAT|O_TRUNC"),
 ("D04", D, "2", "bytesIn is not counted", "$p->{in} += $n;", "$p->{in} += 0;"),
 ("D05", D, "2", "the detector does not listen on 3128", ": (80, 443, 3128);", ": (80, 443);"),
 ("D06", D, "2", "a conn is not logged", "  print $out '{\"ts\":\"' . ts($p->{t})", "  print { $out eq $log ? *STDERR : $out } '{\"ts\":\"' . ts($p->{t})"),
 ("N01", N, "2", "an unreachable-or-anything verdict (a leak passes)", "sub unreach { my $r = shift; return 1 if $UNREACH{$r};", "sub unreach { my $r = shift; return 1;"),
 ("N02", N, "2", "the IPv4 default-route check never fails", "check('A1b-v4', defined $r4 && !@def4,", "check('A1b-v4', 1,"),
 ("N03", N, "2", "the IPv6 default-route check never fails", "check('A1b-v6', !@def6,", "check('A1b-v6', 1,"),
 ("N04", N, "2", "a UDP sendto that succeeds is not a leak", "return 'SENT' if defined send($s, 'x', 0, $sa);", "return 'ENETUNREACH' if defined send($s, 'x', 0, $sa);"),
 ("N05", N, "2", "a TCP connect that completes is not a leak", "return $err ? ename($err) : 'CONNECTED'; }", "return ename($err ? $err : 101); }"),
 ("N06", N, "2", "the detector positive control does not send the token", "syswrite($s, \"AC8PROBE:$nonce\\n\");", "syswrite($s, \"hello\\n\");"),
 ("N07", N, "2", "PROBE-SENT undercounts", "$sent++;", "$sent += 0;"),
 ("N08", N, "2", "A2-ahosts accepts any non-zero status (a missing getent passes)", "($rc >> 8) == 2 && $dt < 5", "$rc != 0 && $dt < 5"),
 ("N09", N, "2", "the probe ignores its stdin", "unless (-t STDIN) {", "unless (1) {"),
 ("N10", N, "2", "A2-embedded always passes", "check('A2-embedded', $rc == 0 && $out", "check('A2-embedded', 1 || $rc == 0 && $out"),
]
os.makedirs(work, exist_ok=True)
for mid, rel, tests, desc, old, new in M:
    if sel.strip() and f" {mid} " not in sel:
        continue
    d = os.path.join(work, f"p131_{mid}", "cpg")
    os.makedirs(d)
    shutil.copytree(sbx, os.path.join(d, "sandbox"), ignore=shutil.ignore_patterns("__pycache__"))
    for name in (".venv", "ccg", "infra", "extractors"):
        if os.path.exists(os.path.join(cpg, name)):
            os.symlink(os.path.realpath(os.path.join(cpg, name)), os.path.join(d, name))
    f = os.path.join(d, "sandbox", rel)
    t = open(f).read()
    if t.count(old) != 1:
        print(f"ERROR\t{mid}\told text applies {t.count(old)} times"); continue
    open(f, "w").write(t.replace(old, new))
    print(f"MUT\t{mid}\t{tests}\t{desc}\t{os.path.join(d, 'sandbox', 'ac8')}")
PY
errs=$(grep -c '^ERROR' "$W/manifest" || true); grep '^ERROR' "$W/manifest" >&2
[ -n "${MUTANT_DRY:-}" ] && { echo "$(grep -c '^MUT' "$W/manifest") edits apply, $errs error(s)"; exit $((errs ? 1 : 0)); }
KILLED=0; SURV=(); n=0
while IFS=$'\t' read -r kind mid tests desc dir; do
  [ "$kind" = MUT ] || continue
  n=$((n + 1))
  before=$({ docker ps -aq --filter 'label=ccg.ac8'; docker network ls -q --filter 'label=ccg.ac8'; } | sort)
  out=$(timeout -k 30 "${MUTANT_TIMEOUT:-420}" "$HERE/test_ac8.sh" --ac8-dir "$dir" --only "$tests" 2>&1); rc=$?
  after_c=$(docker ps -aq --filter 'label=ccg.ac8' | sort); after_n=$(docker network ls -q --filter 'label=ccg.ac8' | sort)
  for c in $(comm -13 <(printf '%s\n' $before) <(printf '%s\n' $after_c)); do
    case "$(docker inspect --format '{{.Name}}' "$c" 2>/dev/null)" in /ccg-ac8-t*) docker rm -f "$c" >/dev/null 2>&1 ;; esac; done
  for x in $(comm -13 <(printf '%s\n' $before) <(printf '%s\n' $after_n)); do
    case "$(docker network inspect --format '{{.Name}}' "$x" 2>/dev/null)" in ccg-ac8-t*) docker network rm "$x" >/dev/null 2>&1 ;; esac; done
  first=$(grep -m1 '^  FAIL' <<< "$out" | cut -c1-110)
  if [ "$rc" != 0 ]; then KILLED=$((KILLED + 1)); printf '%s killed   %s -> %s (exit %s)\n' "$mid" "$desc" "${first:-no FAIL line}" "$rc"
  else SURV+=("$mid"); printf '%s SURVIVED %s\n' "$mid" "$desc"; fi
  rm -rf "$W/p131_$mid"
done < "$W/manifest"
echo "killed $KILLED of $n; survived: ${SURV[*]:-none}"
((${#SURV[@]})) && exit 1
exit 0
