#!/bin/bash -p
# argv_audit.sh -- re-run egress-spec3 §7.5's element audit (the runner's OWN function, sourced, never reimplemented) over one
# recorded toolArgv, for AC-8 row 8.6.
#   argv_audit.sh <frontend> -- <toolArgv element>...
# <frontend> is the record's `frontend` field (for joern-parse, the frontend its --language derived). Output, one per line:
#   DIGEST <sha256 of forbidden-flags.txt + allowed-flags.txt>   (compare with the record's flagCheck.listDigest)
#   HIT <element>                                                (a §7.5 forbidden hit: the record must have none)
#   DONE                                                         (the audit ran to completion)
# Exit 0 whenever it ran; the caller reads HIT lines. No Docker, no network, nothing written.
set -uo pipefail
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fe=${1:?frontend}; shift
[ "${1:-}" = -- ] && shift
# shellcheck disable=SC1091
. "$HERE/../offline-run.sh"
sbx_load_lists || { echo "ERROR $SBX_LIST_ERR"; exit 2; }
echo "DIGEST $SBX_LIST_DIGEST"
SBX_HITS=(); SBX_ALLOWED_HITS=()
fes="$fe"
case "$fe" in joern|scip-typescript|scip-python) ;; *) fes="$fe joern-parse" ;; esac
sbx_audit_elements offline "$fes" -- "$@"
for h in "${SBX_HITS[@]}"; do echo "HIT $h"; done
echo DONE
