#!/usr/bin/env bash
# run_extract.sh -- S11 (P0-10): build fact bundles for the two slice units at a pinned commit.
#
#   run_extract.sh [--commit <sha>] [--tag <run tag>] [--units fcs,sp] [--work <dir>]
#
# Per unit (falkor-chat/server = python, salesperson = javascript):
#   1. stage   git archive <commit> <rootPath>           (host; no history is staged -- BF12)
#   2. parse   joern-parse in the sandbox                (offline-run.sh --frontend joern-parse)
#   3. check   cpg-content-check.sc in the sandbox       (plan §2 D15: a parse is certified by content)
#   4. astgen  (javascript only) Babel ASTs in the sandbox, for imports and type end lines
#   5. extract extract.sc in the sandbox                 (runner-staged script, @AUDIT_ID@ output stem)
#   6. normalize normalize.py on the host -> bundle (validated by ccg.bundle on write) + git blame
# Every stage is timed (wall seconds, docker start-up included) into <work>/<tag>/<unit>/timings.json.
# Nothing here touches any container that is not the runner's own ccg-sbx-<auditId> one.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
RUNNER="$REPO/cpg/sandbox/offline-run.sh"
CHECK_SC="$REPO/cpg/sandbox/scripts/cpg-content-check.sc"
COMMIT="defca60ff16534aa12a74e35856de9be98bfc710"   # the slice's pinned commit (repo HEAD at S11 time)
TAG="run1"
UNITS="fcs,sp"
WORK="$REPO/cpg/.ccg-work/vertical-slice"

while [ $# -gt 0 ]; do
  case "$1" in
    --commit) COMMIT="$2"; shift 2;;
    --tag) TAG="$2"; shift 2;;
    --units) UNITS="$2"; shift 2;;
    --work) WORK="$2"; shift 2;;
    *) echo "usage: run_extract.sh [--commit sha] [--tag t] [--units fcs,sp] [--work dir]" >&2; exit 64;;
  esac
done
COMMIT="$(git -C "$REPO" rev-parse --verify "$COMMIT^{commit}")"

unit_cfg() {  # tag -> root strip lang frontend-language
  case "$1" in
    fcs) echo "falkor-chat/server 2 python pythonsrc";;
    sp)  echo "salesperson 1 javascript jssrc";;
    *) echo "unknown unit $1" >&2; return 1;;
  esac
}
now() { date +%s.%N; }
fail() { echo "run_extract: FAIL: $*" >&2; exit 1; }

run_unit() {
  local u="$1" root strip lang fe
  read -r root strip lang fe <<<"$(unit_cfg "$u")"
  local D="$WORK/$TAG/$u" T0 T1 t_parse t_check t_astgen=0 t_extract t_norm
  rm -rf "$D"; mkdir -p "$D/stage" "$D/cpg"
  T0=$(now)

  # 1. stage
  git -C "$REPO" archive "$COMMIT" "$root" | tar -x -C "$D/stage" --strip-components="$strip"
  local nfiles
  case "$lang" in
    python) nfiles=$(find "$D/stage" -name '*.py' -size +0 | wc -l);;
    javascript) nfiles=$(find "$D/stage" \( -name '*.ts' -o -name '*.tsx' -o -name '*.js' -o -name '*.jsx' -o -name '*.mjs' -o -name '*.cjs' \) -size +0 | wc -l);;
  esac
  [ "$nfiles" -ge 1 ] || fail "unit $u: no non-empty $lang source file staged"

  # 2. parse
  local a b
  a=$(now)
  "$RUNNER" --backend container --mode offline --frontend joern-parse --source "$D/stage" --output "$D/cpg" \
      --codebase graphmind-ai-lab -- /src --language "$fe" >"$D/parse.log" 2>&1 || fail "unit $u: joern-parse failed (see $D/parse.log)"
  b=$(now); t_parse=$(echo "$b - $a" | bc)
  local cpgf; cpgf=$(ls "$D/cpg"/*.cpg.bin | head -1); local PARSE_ID; PARSE_ID=$(basename "$cpgf" .cpg.bin)

  # 3. content check (D15): the unit has >= 1 non-empty source file, so zero non-external METHODs fails
  a=$(now)
  local cc_rc=0
  "$RUNNER" --backend container --mode offline --frontend joern --script-file "$CHECK_SC" --source "$D/stage" \
      --output "$D/cpg" -- --script /scripts/cpg-content-check.sc --param "cpgFiles=/out/$PARSE_ID.cpg.bin" \
      >"$D/contentcheck.txt" 2>"$D/contentcheck.err" || cc_rc=$?
  if [ "$cc_rc" -ne 0 ]; then
    # only the check's own verdict line may claim an empty CPG; any other non-zero (docker down, timeout,
    # flag refusal, script error) is a runner failure and is reported as such
    if grep -q "^CONTENT /out/$PARSE_ID.cpg.bin=EMPTY" "$D/contentcheck.txt"; then
      fail "unit $u: content check failed -- the parse produced a CPG with no non-external METHOD although the unit has $nfiles non-empty $lang source files (egress R11/H12); see $D/contentcheck.txt"
    fi
    fail "unit $u: content-check run exited $cc_rc without a CONTENT=EMPTY verdict (runner or script failure, not an empty CPG); see $D/contentcheck.txt and $D/contentcheck.err"
  fi
  grep -q "^CONTENT /out/$PARSE_ID.cpg.bin=ok" "$D/contentcheck.txt" || fail "unit $u: no CONTENT ok line"
  b=$(now); t_check=$(echo "$b - $a" | bc)

  # 4. astgen (javascript)
  local ASTGEN_ID="" ASTGEN_ARG=()
  if [ "$lang" = javascript ]; then
    a=$(now)
    "$RUNNER" --backend container --mode offline --frontend joern --script-file "$HERE/astgen.sc" --source "$D/stage" \
        --output "$D/cpg" -- --script /scripts/astgen.sc --param cpgFile=unused --param 'outDir=/out/@AUDIT_ID@.astgen' \
        >"$D/astgen.log" 2>&1 || fail "unit $u: astgen failed (see $D/astgen.log)"
    b=$(now); t_astgen=$(echo "$b - $a" | bc)
    local ad; ad=$(ls -d "$D/cpg"/*.astgen | head -1); ASTGEN_ID=$(basename "$ad" .astgen)
    ASTGEN_ARG=(--astgen "$ad")
  fi

  # 5. extract
  a=$(now)
  "$RUNNER" --backend container --mode offline --frontend joern --script-file "$HERE/extract.sc" --source "$D/stage" \
      --output "$D/cpg" -- --script /scripts/extract.sc --param "cpgFile=/out/$PARSE_ID.cpg.bin" \
      --param 'out=/out/@AUDIT_ID@.facts.jsonl' --param "lang=$lang" >"$D/extract.log" 2>&1 || fail "unit $u: extract.sc failed (see $D/extract.log)"
  b=$(now); t_extract=$(echo "$b - $a" | bc)
  local factsf; factsf=$(ls "$D/cpg"/*.facts.jsonl | head -1); local EXTRACT_ID; EXTRACT_ID=$(basename "$factsf" .facts.jsonl)

  # 6. normalize (host). sandboxAuditIds in invocation order: parse, content check is a separate record
  # that is not an input of the bundle; astgen and extract are.
  local ids="$PARSE_ID"; [ -n "$ASTGEN_ID" ] && ids="$ids,$ASTGEN_ID"; ids="$ids,$EXTRACT_ID"
  local unitid out
  unitid=$(cd "$REPO/cpg" && python3 -c "from ccg import ids; print(ids.unit_id('graphmind-ai-lab','$root','$lang'))")
  out="$WORK/$TAG/$unitid.bundle.jsonl"
  a=$(now)
  python3 -I "$HERE/normalize.py" --facts "$factsf" --stage "$D/stage" --unit-root "$root" --lang "$lang" \
      --repo "$REPO" --commit "$COMMIT" --audit-ids "$ids" --extract-script "$HERE/extract.sc" \
      "${ASTGEN_ARG[@]}" --out "$out" --report "$D/report.json" >"$D/normalize.log" 2>&1 || { cat "$D/normalize.log" >&2; fail "unit $u: normalize.py failed"; }
  b=$(now); t_norm=$(echo "$b - $a" | bc)
  # 7. verify: schema re-read, qnames re-derived from source, per-file rule, audit records
  a=$(now)
  python3 -I -c "import sys; sys.path.insert(0, '$REPO/cpg'); from ccg import bundle; r = bundle.read_bundle('$out'); print('reread ok:', len(r), 'records')" >"$D/reread.log" 2>&1 || fail "unit $u: bundle failed to re-validate"
  local vq=(python3 -I "$HERE/verify_qnames.py" --lang "$lang" --stage "$D/stage" --root "$root" --bundle "$out" --json "$D/qnames.json")
  [ "$lang" = javascript ] && vq+=(--astgen "$D/cpg/$ASTGEN_ID.astgen")
  "${vq[@]}" | tee "$D/verify_qnames.txt" || echo "run_extract: $u: qname MISMATCH (SL-7 candidate) -- see $D/qnames.json" >&2
  python3 -I "$HERE/perfile_rule.py" --lang "$lang" --stage "$D/stage" --check "$D/contentcheck.txt" >"$D/perfile.json"
  python3 -I "$HERE/check_audit.py" "$out" "${CCG_AUDIT_DIR:-$REPO/cpg/.audit}" | tee "$D/audit_check.txt"
  b=$(now); local t_verify; t_verify=$(echo "$b - $a" | bc)
  T1=$(now)
  python3 - "$D/timings.json" "$u" "$root" "$lang" "$COMMIT" "$PARSE_ID" "$EXTRACT_ID" "$t_parse" "$t_check" "$t_astgen" "$t_extract" "$t_norm" "$t_verify" "$(echo "$T1 - $T0" | bc)" "$out" "$D/cpg/$EXTRACT_ID.facts.jsonl" <<'PY'
import json, sys
f, u, root, lang, commit, pid, eid, tp, tc, ta, te, tn, tv, tt, out, facts = sys.argv[1:]
stats = json.loads(open(facts).readlines()[-1])
json.dump({"unit": u, "rootPath": root, "lang": lang, "commit": commit, "parseAuditId": pid, "extractAuditId": eid,
           "bundle": out, "wallSeconds": {"parse": float(tp), "contentCheck": float(tc), "astgen": float(ta),
           "extract": float(te), "normalize": float(tn), "verify": float(tv), "total": float(tt)},
           "extractInternalMs": {k: int(stats[k]) for k in ("loadMs", "projectionMs", "totalMs")},
           "histogram": stats["histogram"]}, open(f, "w"), indent=2, sort_keys=True)
PY
  echo "run_extract: $u ok  parse=${t_parse}s check=${t_check}s astgen=${t_astgen}s extract=${t_extract}s normalize=${t_norm}s -> $out"
}

for u in ${UNITS//,/ }; do run_unit "$u"; done
