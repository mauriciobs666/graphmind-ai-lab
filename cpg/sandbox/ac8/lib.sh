# lib.sh -- sourced by run.sh (and by the Docker-free tests): the two pieces of the AC-8 harness whose failure mode is a silent pass.
# No Docker, no network here.

# bridge_has_address <dev>   -> 0: the device exists and carries at least one address; 1: it exists and carries none;
#                               2: UNKNOWN (device missing, `ip` failed or absent, output not JSON, an empty interface list, an interface
#                                  without `addr_info`). Only 1 is "no address". A guard that decides from a command's output must read
#                                  "no answer" as "unknown" and fail; `[ -n "" ]` over a failed `ip` is the silent pass this prevents.
# Env: AC8_IP_BIN (default /usr/bin/ip) and AC8_PY (default: python3), both test hooks.
bridge_has_address() {
  local dev=$1 ip=${AC8_IP_BIN:-/usr/bin/ip} py=${AC8_PY:-python3} out
  [ -n "$dev" ] || return 2
  out=$("$ip" -j address show dev "$dev" 2>/dev/null) || return 2
  printf '%s' "$out" | "$py" -I -c '
import json, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit(2)
if not isinstance(d, list) or not d or not all(isinstance(i, dict) and isinstance(i.get("addr_info"), list) for i in d):
    sys.exit(2)
sys.exit(0 if any(i["addr_info"] for i in d) else 1)' 2>/dev/null
  case $? in 0) return 0 ;; 1) return 1 ;; *) return 2 ;; esac
}

# bridge_isolated_or_die <dev>: returns only when the bridge exists and carries NO address; otherwise calls the caller's `die` (exit 70: the
# harness could not establish the isolation mechanism it owns; there is no fallback in P1). Both "has an address" and "unknown" die.
bridge_isolated_or_die() {
  local dev=$1 rc
  bridge_has_address "$dev"; rc=$?
  case $rc in
    1) return 0 ;;
    0) die "the bridge $dev carries an address: the isolated gateway mode did not take (P-a's mechanism failed; escalate, no fallback in P1)" ;;
    *) die "cannot establish whether the bridge $dev carries an address (device missing, ip failed or unparsable: Docker not local, or another bridge name): the isolation mechanism is unverified; escalate, no fallback in P1" ;;
  esac
}

# ac8_install_traps <cleanup function>: the cleanup runs once, on EXIT, whatever the way out; INT/TERM/HUP EXIT (nonzero: 130/143/129).
# A trap that only ran the cleanup would let bash carry on with the interrupted script (verified: kill -TERM $$ resumed it).
# Further signals during the cleanup are ignored so it is not cut short.
ac8_install_traps() {
  AC8_CLEANUP_FN=$1; AC8_CLEANED=0
  trap ac8_cleanup_once EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'exit 129' HUP
}
ac8_cleanup_once() {
  trap '' INT TERM HUP
  [ "$AC8_CLEANED" = 1 ] && return 0
  AC8_CLEANED=1
  "$AC8_CLEANUP_FN"
}
