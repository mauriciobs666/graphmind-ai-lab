#!/usr/bin/env bash
# mutants.sh — mutation test of test_sandbox.sh (S08). Each mutant is a COPY of offline-run.sh
# and the two flag lists in a fresh temp dir, with one deliberate break (one or more exact text
# edits); the real files are never touched. A mutant is "killed" iff
# `test_sandbox.sh --runner <copy> --only <tests>` exits non-zero and prints a FAIL line. Every
# edit must apply (its old text must be present), so no mutant silently equals the original.
# Exit 0 iff every mutant is killed.
#
# Usage: cpg/sandbox/tests/mutants.sh [mutant-id ...]   (MUTANT_DRY=1: only check the edits apply)
#   MUTANT_TABLE_OUT=<file>: also write the markdown result table there.
set -uo pipefail
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBX="$HERE/.."
export CCG_TOOLCHAIN_IMAGE="${CCG_TOOLCHAIN_IMAGE:-$(env -i PATH=/usr/bin:/bin "$SBX/../infra/toolchain/build.sh" --print-tag)}"
W=$(mktemp -d /tmp/ccg-s08-mutants.XXXXXX); trap 'rm -rf "$W"' EXIT

# Build the mutant copies; print "id<TAB>tests<TAB>description" per mutant that applied.
python3 -I - "$SBX" "$W" "$@" > "$W/manifest" <<'PY'
import os, shutil, sys
sbx, work, sel = sys.argv[1], sys.argv[2], set(sys.argv[3:])
R, F, A = "offline-run.sh", "forbidden-flags.txt", "allowed-flags.txt"
def move(fe, fl, why):  # allowed -> forbidden
    line = next(l for l in open(os.path.join(sbx, A)) if l.startswith(f"{fe}\t{fl}\t"))
    return [(A, line, ""), (F, "joern\t--dep\t", f"{fe}\t{fl}\toffline\t{why}\njoern\t--dep\t")]
M = [
 # --- first round (S08 build) ---
 ("M01", "5", "drop --cap-drop ALL", [(R, "--read-only --cap-drop ALL", "--read-only")]),
 ("M02", "1,3,5", "mount /tmp noexec", [(R, 'SBX_TMP_OPTS="rw,exec,nosuid,nodev"', 'SBX_TMP_OPTS="rw,noexec,nosuid,nodev"')]),
 ("M03", "1,5", "remove env -i (container)", [(R, '--entrypoint /usr/bin/env "$SBX_IMAGE" -i "${SBX_ENV[@]}"', '--entrypoint /usr/bin/env "$SBX_IMAGE" "${SBX_ENV[@]}"')]),
 ("M04", "10", "accept a \\v-led //> (whitespace set without \\v)", [(R, "tr '\\r\\f\\v' '\\n\\n\\n'", "tr '\\r\\f' '\\n\\n'"), (R, "grep -a -E '^[[:cntrl:] ]*//>'", "grep -a -E '^[ \\t\\r\\f]*//>'")]),
 ("M05", "2", "allow a forbidden flag in --flag=value form", [(R, 'if [[ $el == "$f" || $el == "$f="* || $el == *"$f"* ]]', 'if [[ $el == "$f" ]]')]),
 ("M06", "1,5", "drop --network none", [(R, "--network none --hostname localhost", "--hostname localhost")]),
 ("M07", "3,5", "drop --hostname localhost", [(R, "--network none --hostname localhost", "--network none")]),
 ("M08", "1,5", "/src as a plain ro bind (no noexec)", [(R, 'volume-opt=o=bind,$SBX_SRC_OPTS\\",volume-opt=device=$src"', 'volume-opt=o=bind,ro\\",volume-opt=device=$src"')]),
 ("M09", "5", "/out as a plain rw bind instead of the noexec overlay", [(R, 'volume-opt=type=overlay,volume-opt=device=overlay,\\"volume-opt=o=lowerdir=$SBX_OVL/lower,upperdir=$out,workdir=$SBX_OVL/work,noexec,nosuid,nodev\\""', 'volume-opt=type=none,\\"volume-opt=o=bind\\",volume-opt=device=$out"')]),
 ("M10", "8", "N1 matches only the literal WSLInterop name (P0 plan D6 path)", [(R, '    case "$name" in WSLInterop*) hit=1 ;; esac', '    case "$name" in WSLInterop) hit=1 ;; esac')]),
 ("M11", "8", "N1 ignores the magic rule (a)", [(R, '        "magic 4d5a"*) hit=1 ;;', '        "magic 4d5a"*) : ;;')]),
 ("M12", "2", "-J-Xmx exception by prefix, not full match", [(R, "[[ $e =~ ^-J-Xmx[0-9]+[mg]$ || $e =~ ^-J-Xss[0-9]+[km]$ ]]", "[[ $e == -J-Xmx* || $e == -J-Xss* ]]")]),
 ("M13", "1", "namespace: the spec-literal /usr bind (recursive, wsl masked by tmpfs)", [(R, "  sbx_bind_tree /usr\n", "  SBX_LAUNCHER+=(--ro-bind /usr /usr --tmpfs /usr/lib/wsl)\n")]),
 ("M14", "1", "JAVA_OPTS carries a -D property", [(R, "SBX_JAVA_OPTS='-Xmx2g'", "SBX_JAVA_OPTS='-Xmx2g -Djava.io.tmpdir=/tmp'")]),
 ("M15", "1", "a forbidden name on the pass-list", [(R, '"DOTNET_CLI_TELEMETRY_OPTOUT=1" "DOTNET_NOLOGO=1"', '"DOTNET_CLI_TELEMETRY_OPTOUT=1" "JAVA_TOOL_OPTIONS=-Dx=1" "DOTNET_NOLOGO=1"')]),
 ("M16", "2", "map the refused java alias", [(R, "    javasrc) echo javasrc2cpg ;;", "    javasrc|java) echo javasrc2cpg ;;")]),
 ("M17", "4", "the tool output stem is not the auditId", [(R, '    SBX_TOOL_ARGV+=(--output "/out/$SBX_AUDIT_ID.cpg.bin")', '    SBX_TOOL_ARGV+=(--output "/out/cpg.bin")')]),
 ("M18", "5,10", "-v file bind for /scripts (no noexec)", [(R, 'SBX_LAUNCHER+=(--mount "type=volume,dst=/scripts,volume-driver=local,volume-nocopy=true,$(sbx_vol_labels),volume-opt=type=none,\\"volume-opt=o=bind,$SBX_SRC_OPTS\\",volume-opt=device=$sdir")', 'SBX_LAUNCHER+=(-v "$sdir:/scripts:ro")')]),
 ("M19", "8", "C2 accepts any seccomp profile", [(R, '    [ "$prof" = builtin ] || ok=0', "    :")]),
 ("M20", "2", "forbidden flags still launch (no pre-launch check)", [(R, '    ((${#SBX_HITS[@]})) && sbx_refuse 64 forbidden-flag "forbidden flag or launcher-level option"', "    ((${#SBX_HITS[@]})) && true")]),
 ("M21", "1,5", "add -e to docker run", [(R, "    --network none --hostname localhost", "    -e X=1 --network none --hostname localhost")]),
 # --- S08 gate fix round: one per finding, plus the reviewer's X mutants ---
 ("M22", "10", "M1: directive check strips only [[:space:]] (control bytes admitted)", [(R, "grep -a -E '^[[:cntrl:] ]*//>'", "grep -a -E '^[[:space:]]*//>'")]),
 ("M23", "2", "M2: SBX_BACKEND_REFUSAL not initialised (crash, no record)", [(R, '  SBX_BACKEND_REFUSAL=""; SBX_REFUSAL_DETAIL=""; SBX_LAUNCH_SETUP_FAIL=""\n', '  SBX_LAUNCH_SETUP_FAIL=""\n'), (R, "    *) sbx_try_container || true ;;", "    *) SBX_BACKEND=container ;;")]),
 ("X1", "2,6", "M4: jimple2cpg --full-resolver moved allowed -> forbidden", move("jimple2cpg", "--full-resolver", "x1")),
 ("X2", "2,6", "M4: javasrc2cpg --delombok-mode moved allowed -> forbidden", move("javasrc2cpg", "--delombok-mode", "x2")),
 ("X3", "2,6", "M4: c2cpg --compilation-database moved allowed -> forbidden", move("c2cpg", "--compilation-database", "x3")),
 ("X4", "1,7", "M3: GOTOOLCHAIN dropped from the pass-list", [(R, '"GOTOOLCHAIN=local" ', "")]),
 ("X5", "1,7", "M3: GOTOOLCHAIN=auto", [(R, '"GOTOOLCHAIN=local"', '"GOTOOLCHAIN=auto"')]),
 ("X6", "1,7", "M3: DOTNET_CLI_TELEMETRY_OPTOUT=0", [(R, '"DOTNET_CLI_TELEMETRY_OPTOUT=1"', '"DOTNET_CLI_TELEMETRY_OPTOUT=0"')]),
 ("X8", "3,10", "m3: @AUDIT_ID@ substitution removed", [(R, 'SBX_TOOL_ARGV+=("${_a//@AUDIT_ID@/$SBX_AUDIT_ID}")', 'SBX_TOOL_ARGV+=("$_a")'), (R, 'targv+=("${a//@AUDIT_ID@/$SBX_AUDIT_ID}")', 'targv+=("$a")')]),
 ("X9", "3,10", "m3: record the pre-substitution argv", [(R, '\\"toolArgv\\":$(sbx_jarr "${SBX_TOOL_ARGV[@]}")', '\\"toolArgv\\":$(sbx_jarr "${SBX_TOOL_ARGV[@]//$SBX_AUDIT_ID/@AUDIT_ID@}")')]),
 ("X10", "2,10", "m3: refusal records carry toolArgv []", [(R, '  SBX_TOOL_ARGV=("${SBX_CALLER_ARGV[@]}")\n', '  SBX_TOOL_ARGV=()\n')]),
 ("M24", "2", "m4: netnsDistinct claimed true before any launch", [(R, 'PRE_SECCOMP=""; PRE_NETNS=false', 'PRE_SECCOMP=""; PRE_NETNS=true')]),
 ("M25", "2", "m4: invalid --mode lands in egressMode", [(R, "  [[ $mode == offline || $mode == controlled ]] && SBX_MODE=$mode", '  [ -n "$mode" ] && SBX_MODE=$mode')]),
 ("M26", "3", "m2: no lock on --output", [(R, '  flock -n "$SBX_LOCK_FD" || sbx_refuse', '  true || sbx_refuse')]),
 ("M27", "1", "m6: namespace ro binds not recorded in mounts", [(R, '    ro) SBX_LAUNCHER+=(--ro-bind "$2" "$3"); SBX_MOUNTS+=("$3|$2|ro") ;;', '    ro) SBX_LAUNCHER+=(--ro-bind "$2" "$3") ;;')]),
 ("M28", "1,5", "m5: production launches with a flag the probe template lacks", [(R, '    sbx_container_argv "$src" "$out" "$sdir" "$wd" -- "$@"\n', '    sbx_container_argv "$src" "$out" "$sdir" "$wd" -- "$@"\n    [ "$SBX_RECORD_FRONTEND" != lib-launch ] && SBX_LAUNCHER=("${SBX_LAUNCHER[@]:0:3}" --memory 3g "${SBX_LAUNCHER[@]:3}")\n')]),
 ("M29", "2", "n1: no env -i re-exec (DOCKER_HOST reaches docker)", [(R, '  ((ok)) && return 0\n', '  return 0\n')]),
 ("M30", "2", "n1: shebang without -p (BASH_ENV runs)", [(R, "#!/bin/bash -p\n", "#!/bin/bash\n")]),
 ("M31", "8", "m1: auto never falls back to namespace", [(R, "    auto) sbx_try_container || sbx_try_namespace ;;", "    auto) sbx_try_container ;;")]),
 # --- egress-spec2 (ES2) additions ---
 ("M32", "10", "ES2 §1.1: no \\r line split (x\\r//> admitted)", [(R, "tr '\\r\\f\\v' '\\n\\n\\n'", "tr '\\f\\v' '\\n\\n'")]),
 ("M33", "2", "ES2 §1.5: refused toolArgv substituted, not the caller's argv", [(R, '  SBX_TOOL_ARGV=("${SBX_CALLER_ARGV[@]}")\n', "")]),
 ("M34", "2", "ES2 §1.5: --help writes a record", [(R, '  case "${1:-}" in -h|--help) sbx_usage; exit 0 ;; esac\n  sbx_init\n', '  sbx_init\n  case "${1:-}" in -h|--help) sbx_usage; sbx_write_record 0 ""; exit 0 ;; esac\n')]),
 ("M35", "2", "M4: the reviewed allowed list swapped (one out, one look-alike in)", [(A, "jimple2cpg\t--full-resolver\t", "jimple2cpg\t--full-resolverx\t")]),
]
for mid, tests, desc, edits in M:
    if sel and mid not in sel: continue
    d = os.path.join(work, mid); os.makedirs(d)
    for f in (R, F, A): shutil.copy2(os.path.join(sbx, f), d)
    ok = True
    for f, old, new in edits:
        p = os.path.join(d, f); s = open(p).read()
        if old not in s:
            print(f"{mid}\tNOT-APPLIED\t{desc}: {old[:60]!r} not in {f}", file=sys.stderr); ok = False; break
        open(p, "w").write(s.replace(old, new, 1))
    if ok: print(f"{mid}\t{tests}\t{desc}")
    else: print(f"{mid}\tNOT-APPLIED\t{desc}")
PY

killed=0; survived=0
printf '| Mutant | Break | Tests run | Result | First failing check |\n|---|---|---|---|---|\n' > "$W/table.md"
while IFS=$'\t' read -r -u 3 id tests desc; do
  d="$W/$id"
  if [ "$tests" = NOT-APPLIED ]; then
    survived=$((survived + 1)); echo "$id NOT APPLIED :: $desc"
    printf '| %s | %s | — | **NOT APPLIED** | |\n' "$id" "$desc" >> "$W/table.md"; continue
  fi
  if [ -n "${MUTANT_DRY:-}" ]; then echo "$id applies"; continue; fi
  "$HERE/test_sandbox.sh" --runner "$d/offline-run.sh" --only "$tests" > "$d/test.out" 2>&1; rc=$?
  first=$(grep -m1 '^  FAIL ' "$d/test.out" | sed 's/^  FAIL //; s/|/\\|/g' | cut -c1-110)
  if [ "$rc" != 0 ] && [ -n "$first" ]; then res=killed; killed=$((killed + 1)); else res=SURVIVED; survived=$((survived + 1)); fi
  printf '%s %-8s tests=%-6s %s :: %s\n' "$id" "$res" "$tests" "$desc" "$first"
  printf '| %s | %s | %s | %s | %s |\n' "$id" "${desc//|/\\|}" "$tests" "$res" "$first" >> "$W/table.md"
done 3< "$W/manifest"
echo; cat "$W/table.md"
echo; echo "killed=$killed survived=$survived"
[ -n "${MUTANT_TABLE_OUT:-}" ] && cp "$W/table.md" "$MUTANT_TABLE_OUT"
((survived == 0))
