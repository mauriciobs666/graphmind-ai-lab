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
#   MUTANT_TIMEOUT=<seconds>: bound on each mutant's test run (default 900; the whole suite takes
#   about 3 min). A mutant whose run hits it is "killed (timeout)": a hang is a red test, and it
#   must not block the harness (YC's runner re-execs forever).
set -uo pipefail
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBX="$HERE/.."
# A mutant that admits an escaping path makes scip-python index the whole image: bound each run (the suite default is 300 s).
export RUN_TIMEOUT="${RUN_TIMEOUT:-45}"
export CCG_TOOLCHAIN_IMAGE="${CCG_TOOLCHAIN_IMAGE:-$(env -i PATH=/usr/bin:/bin "$SBX/../infra/toolchain/build.sh" --print-tag)}"
W=$(mktemp -d /tmp/ccg-s08-mutants.XXXXXX); trap 'rm -rf "$W"' EXIT

# Build the mutant copies; print "id<TAB>tests<TAB>description" per mutant that applied.
python3 -I - "$SBX" "$W" "$@" > "$W/manifest" <<'PY'
import os, shutil, sys
sbx, work, sel = sys.argv[1], sys.argv[2], set(sys.argv[3:])
R, F, A, S = "offline-run.sh", "forbidden-flags.txt", "allowed-flags.txt", "scripts/cpg-content-check.sc"
def move(fe, fl, why):  # allowed -> forbidden
    line = next(l for l in open(os.path.join(sbx, A)) if l.startswith(f"{fe}\t{fl}\t"))
    return [(A, line, ""), (F, "joern\t--dep\t", f"{fe}\t{fl}\toffline\t{why}\njoern\t--dep\t")]
M = [
 # --- first round (S08 build) ---
 ("M01", "5", "drop --cap-drop ALL", [(R, "--read-only --cap-drop ALL", "--read-only")]),
 ("M02", "1,3,5", "mount /tmp noexec", [(R, 'SBX_TMP_OPTS="rw,exec,nosuid,nodev"', 'SBX_TMP_OPTS="rw,noexec,nosuid,nodev"')]),
 ("M03", "1,5", "remove env -i (container)", [(R, '--entrypoint /usr/bin/env "$SBX_IMAGE_ID" -i "${SBX_ENV[@]}"', '--entrypoint /usr/bin/env "$SBX_IMAGE_ID" "${SBX_ENV[@]}"')]),
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
 ("M23", "2", "M2: SBX_BACKEND_REFUSAL not initialised (crash, no record)", [(R, '  SBX_BACKEND_REFUSAL=""; SBX_REFUSAL_DETAIL=""; SBX_LAUNCH_SETUP_FAIL=""; SBX_INDEXER=0; SBX_IDX_MISSING=()\n', '  SBX_LAUNCH_SETUP_FAIL=""; SBX_INDEXER=0; SBX_IDX_MISSING=()\n'), (R, '    *) [ -n "${SBX_DCFG_FAIL:-}" ] || sbx_try_container || true ;;', "    *) SBX_BACKEND=container ;;")]),
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
 ("M29", "2,12", "n1: no env -i re-exec (DOCKER_HOST reaches docker)", [(R, '  ((ok)) && return 0\n', '  return 0\n')]),
 ("M30", "2,11", "n1 / §1.7: shebang without -p (BASH_ENV/ENV run)", [(R, "#!/bin/bash -p\n", "#!/bin/bash\n")]),
 ("M31", "8", "m1: auto never falls back to namespace", [(R, "    auto) sbx_try_container || sbx_try_namespace ;;", "    auto) sbx_try_container ;;")]),
 # --- egress-spec2 (ES2) additions ---
 ("M32", "10", "ES2 §1.1: no \\r line split (x\\r//> admitted)", [(R, "tr '\\r\\f\\v' '\\n\\n\\n'", "tr '\\f\\v' '\\n\\n'")]),
 ("M33", "2", "ES2 §1.5: refused toolArgv substituted, not the caller's argv", [(R, '  SBX_TOOL_ARGV=("${SBX_CALLER_ARGV[@]}")\n', "")]),
 ("M34", "2", "ES2 §1.5: --help writes a record", [(R, '  case "${1:-}" in -h|--help) sbx_usage; exit 0 ;; esac\n  sbx_init\n', '  sbx_init\n  case "${1:-}" in -h|--help) sbx_usage; sbx_write_record 0 ""; exit 0 ;; esac\n')]),
 ("M37", "11", "§1.7: skip the Docker-config dir check", [(R, "  sbx_dcfg_check || true\n", '  SBX_DCFG="$(sbx_home)/$SBX_DCFG_REL"; SBX_DCFG_FAIL=""\n')]),
 ("M38", "11", "§1.7: DOCKER_HOST left inherited (not pinned)", [
    (R, '  [ "${blk[DOCKER_HOST]:-}" = "$SBX_DOCKER_HOST" ] && [ "${blk[DOCKER_CONFIG]:-}"', '  [ "${blk[DOCKER_CONFIG]:-}"'),
    (R, 'exec /usr/bin/env -i "PATH=$SBX_PATH" "HOME=$home" "DOCKER_HOST=$SBX_DOCKER_HOST" \\', 'exec /usr/bin/env -i "PATH=$SBX_PATH" "HOME=$home" ${DOCKER_HOST:+"DOCKER_HOST=$DOCKER_HOST"} \\'),
    (R, "  export DOCKER_HOST=$SBX_DOCKER_HOST DOCKER_CONFIG=", "  export DOCKER_CONFIG=")]),
 ("M39", "11,12", "§1.7: DOCKER_CONFIG left inherited (caller config.json read)", [
    (R, '  [ "${blk[DOCKER_HOST]:-}" = "$SBX_DOCKER_HOST" ] && [ "${blk[DOCKER_CONFIG]:-}" = "$home/$SBX_DCFG_REL" ] || ok=0', '  [ "${blk[DOCKER_HOST]:-}" = "$SBX_DOCKER_HOST" ] || ok=0'),
    (R, '    "DOCKER_CONFIG=$home/$SBX_DCFG_REL" "${keep[@]}"', '    ${DOCKER_CONFIG:+"DOCKER_CONFIG=$DOCKER_CONFIG"} "${keep[@]}"'),
    (R, "  export DOCKER_HOST=$SBX_DOCKER_HOST DOCKER_CONFIG=${SBX_DCFG:-/nonexistent}", "  export DOCKER_HOST=$SBX_DOCKER_HOST")]),
 ("M40", "5", "p2-m1: container launched without the auditId label", [(R, ' --label "$SBX_LABEL.auditId=$SBX_AUDIT_ID"', "")]),
 ("M41", "12", "p2-n1: launcher in the foreground (signal trap deferred, container survives)", [(R, '  "${SBX_LAUNCHER[@]}" < /dev/null &\n  SBX_CHILD=$!\n  wait "$SBX_CHILD" || rc=$?\n', '  "${SBX_LAUNCHER[@]}" < /dev/null || rc=$?\n')]),
 ("M42", "2", "p2-n2: scratch made before --help, with no trap at the mktemp", [
    (R, "  trap 'sbx_cleanup' EXIT   # set with the mktemp", "  : # set with the mktemp"),
    (R, '  case "${1:-}" in -h|--help) sbx_usage; exit 0 ;; esac\n  sbx_init\n', '  sbx_init\n  case "${1:-}" in -h|--help) sbx_usage; exit 0 ;; esac\n')]),
 ("M43", "3", "P3-7: the content check accepts an empty CPG", [(S, "    val ok = ms.nonEmpty\n", "    val ok = true\n")]),
 ("M35", "2", "M4: the reviewed allowed list swapped (one out, one look-alike in)", [(A, "jimple2cpg\t--full-resolver\t", "jimple2cpg\t--full-resolverx\t")]),
 # --- S08 Pass 3 follow-up (p3-m1, p3-m3, p3-n1) ---
 ("YA", "14", "p3-m3: the /tmp|/var/tmp|/dev/shm tree rule deleted", [(R, '  case "$real/" in /tmp/*|/var/tmp/*|/dev/shm/*) SBX_DCFG_FAIL="$real is under a temporary tree"; return 1 ;; esac\n', "")]),
 ("YB", "14", "p3-m3: the world-writable ancestor test disabled", [(R, """    if (( 0$(stat -c '%a' "$p") & 02 )); then SBX_DCFG_FAIL="$real sits under world-writable $p"; return 1; fi\n""", "    :\n")]),
 ("YC", "15", "p3-n1: env keys from compgen -e (non-identifier entries unseen)", [(R, """  while IFS= read -r -d '' k; do\n    case "${k%%=*}" in""", """  for k in $(compgen -e); do\n    case "$k" in"""), (R, '  done < "/proc/$$/environ" 2>/dev/null\n', "  done\n")]),
 ("YH", "15", "p4: closed-env values read from live shell variables, not the exec'd block", [(R, '  [ "${blk[PATH]:-}" = "$SBX_PATH" ] && [ "${blk[HOME]:-}" = "$home" ] || ok=0\n', '  [ "${PATH:-}" = "$SBX_PATH" ] && [ "${HOME:-}" = "$home" ] || ok=0\n')]),
 ("YD", "13", "p3-m1: the rejected design -- rm the container by name, then kill the client", [(R, '  sbx_kill_jobs "$wd"\n  if [ "$SBX_BACKEND" = container ]; then sbx_remove_own; fi\n  sbx_cleanup\n  exit', '  if [ "$SBX_BACKEND" = container ]; then docker rm -f -v "ccg-sbx-$SBX_AUDIT_ID" >/dev/null 2>&1 || true; fi\n  if [ -n "${SBX_CHILD:-}" ]; then kill "$SBX_CHILD" 2>/dev/null; wait "$SBX_CHILD" 2>/dev/null; fi\n  sbx_cleanup\n  exit')]),
 ("YE", "13", "p3-m1: client killed first, but one removal pass (no settle loop)", [(R, "  while ((quiet < 3 && n < 50)); do", "  while ((quiet < 1 && n < 1)); do")]),
 ("YG", "2", "p4-m1: sbx_cleanup stops removing SBX_SCRATCH (every invocation leaks it)", [(R, '    [ -n "$d" ] && [ -d "$d" ] || continue\n', '    [ -n "$d" ] && [ -d "$d" ] || continue\n    [ "$d" = "$SBX_SCRATCH" ] && continue\n')]),
 ("YF", "12", "p3-m1: the client gets TERM, not KILL (docker run proxies it and stays up)", [(R, '  sbx_kill_jobs "$wd"\n  if [ "$SBX_BACKEND" = container ]; then sbx_remove_own; fi\n  sbx_cleanup\n  exit', '  local j\n  for j in $(jobs -p); do [ "$j" = "$wd" ] || { kill -TERM "$j" 2>/dev/null; wait "$j" 2>/dev/null; }; done\n  if [ "$SBX_BACKEND" = container ]; then sbx_remove_own; fi\n  sbx_cleanup\n  exit')]),
 # --- P1-48b: the indexer entries (egress-spec3 §7.4, §10.1 (21)-(26)) ---
 ("I01", "21,22", "(21) scip-python dropped from SBX_FRONTENDS", [(R, "joern scip-typescript scip-python)", "joern scip-typescript)")]),
 ("I02", "21", "(21) the two indexer names swapped (order is the contract)", [(R, "joern scip-typescript scip-python)", "joern scip-python scip-typescript)")]),
 ("I03", "22", "(22) --cwd is no longer mandatory", [(R, "  ((seen_cwd)) || SBX_IDX_MISSING+=(--cwd)\n", "  :\n")]),
 ("I04", "22", "(22) --environment is no longer mandatory for scip-python", [(R, '  if [ "$fe" = scip-python ] && ((!seen_env)); then SBX_IDX_MISSING+=(--environment); fi\n', "  :\n")]),
 ("I05", "22", "(22) --environment accepted for scip-typescript", [(R, '        [ "$fe" = scip-python ] && ((!seen_env && i + 1 < n))', '        ((!seen_env && i + 1 < n))')]),
 ("I06", "22", "(22) --project-name accepted for scip-typescript", [(R, '        [ "$fe" = scip-python ] && ((!seen_name && i + 1 < n))', '        ((!seen_name && i + 1 < n))')]),
 ("I07", "22", "(22) a repeated --cwd is accepted", [(R, "((!seen_cwd && i + 1 < n))", "((i + 1 < n))")]),
 ("I08", "22", "(22) a repeated --environment is accepted", [(R, "((!seen_env && i + 1 < n))", "((i + 1 < n))")]),
 ("I09", "22", "(22) a repeated --project-name is accepted", [(R, "((!seen_name && i + 1 < n))", "((i + 1 < n))")]),
 ("I10", "22", "(22) path check: `..` components allowed", [(R, "[[ $c =~ ^[A-Za-z0-9._@+-]+$ && $c != . && $c != .. ]] || return 1", "[[ $c =~ ^[A-Za-z0-9._@+-]+$ && $c != . ]] || return 1")]),
 ("I11", "22", "(22) path check: `.` components allowed", [(R, "[[ $c =~ ^[A-Za-z0-9._@+-]+$ && $c != . && $c != .. ]] || return 1", "[[ $c =~ ^[A-Za-z0-9._@+-]+$ && $c != .. ]] || return 1")]),
 ("I12", "22", "(22) path check: empty components allowed", [(R, "[[ $c =~ ^[A-Za-z0-9._@+-]+$ && $c != . && $c != .. ]] || return 1", "[[ $c =~ ^[A-Za-z0-9._@+-]*$ && $c != . && $c != .. ]] || return 1")]),
 ("I13", "22", "(22) path check: any /src prefix (/srcfoo, /src2) allowed", [(R, "  [[ $v == /src/* ]] || return 1\n", "  [[ $v == /src* ]] || return 0\n")]),
 ("I14", "22", "(22) --project-name pattern relaxed (a leading - or . allowed)", [(R, "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", "^[A-Za-z0-9._-]{0,64}$")]),
 ("I15", "22", "(22) --project-name length bound dropped", [(R, "^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", "^[A-Za-z0-9][A-Za-z0-9._-]*$")]),
 ("I16", "22", "(22) first element need not be `index`", [(R, '  [ "${a[0]}" = index ] || SBX_HITS+=("${a[0]}")\n', "  :\n")]),
 ("I17", "22", "(22) an unknown element is ignored (abbreviations, --opt=value pass)", [(R, '      *) SBX_HITS+=("$opt"); i=$((i + 1)); continue ;;', '      *) i=$((i + 1)); continue ;;')]),
 ("I18", "22", "(22) a caller --output/-o is accepted into the grammar and the output refusal removed", [
    (R, '      *) SBX_HITS+=("$opt"); i=$((i + 1)); continue ;;', '      --output|-o) ok=1 ;;\n      *) SBX_HITS+=("$opt"); i=$((i + 1)); continue ;;'),
    (R, "if [[ $a == -o || $a == --output || $a == --output=* ]]; then SBX_HITS+=(\"$a\"); fi", "if false; then :; fi")]),
 ("I19", "22", "(22) the runner appends the wrong output path", [(R, 'SBX_TOOL_ARGV+=(--output "/out/$SBX_AUDIT_ID.scip")', 'SBX_TOOL_ARGV+=(--output "/out/index.scip")')]),
 ("I20", "22", "(22) the indexer output keeps the parse-frontend suffix", [(R, 'SBX_TOOL_ARGV+=(--output "/out/$SBX_AUDIT_ID.scip")', 'SBX_TOOL_ARGV+=(--output "/out/$SBX_AUDIT_ID.cpg.bin")')]),
 ("I21", "22", "(22) a missing mandatory element outranks a forbidden one (usage before forbidden-flag)", [
    (R, '      if ((${#SBX_HITS[@]})); then sbx_refuse 64 forbidden-flag "indexer argv outside', '      if ((${#SBX_IDX_MISSING[@]})); then sbx_refuse 64 usage "mandatory element missing"; fi\n      if ((${#SBX_HITS[@]})); then sbx_refuse 64 forbidden-flag "indexer argv outside')]),
 ("I22", "22", "(22) the ccg-toolchain-indexer=1 label check removed", [(R, '"$SBX_IMAGE_ID" 2>/dev/null)" != 1 ]; then', '"$SBX_IMAGE_ID" 2>/dev/null)" = nope ]; then')]),
 ("I23", "22", "(22) controlled mode no longer unknown-frontend for an indexer", [(R, '  if ((SBX_INDEXER)) && [ "$mode" = controlled ]; then sbx_refuse 64 unknown-frontend "indexer entries are offline-only"; fi\n', "")]),
 ("I24", "22", "(22) the namespace backend is tried for an indexer (precondition, not unavailable)", [(R, '    namespace) if ((SBX_INDEXER)) && [ -z "${SBX_DCFG_FAIL:-}" ]; then', '    namespace) if false; then')]),
 ("I25", "22", "(22) auto may fall back to namespace for an indexer", [(R, '    auto) if ((SBX_INDEXER)); then sbx_resolve_backend container; else sbx_resolve_backend auto; fi ;;', '    auto) sbx_resolve_backend auto ;;')]),
 ("I26", "21,22", "(21)/(22) the indexer executable is resolved through PATH, not the fixed path", [(R, '  if sbx_is_indexer "$1"; then echo "$SBX_INDEXER_IMG_BIN/$1";', '  if sbx_is_indexer "$1"; then echo "$1";')]),
 ("I27", "22", "(22) the indexer entries point at a directory that has no tool", [(R, "SBX_INDEXER_IMG_BIN=/opt/indexer/bin", "SBX_INDEXER_IMG_BIN=/opt/indexer/nowhere")]),
 ("I28", "21,22", "(22) scip-python is not recognised as an indexer (parse path, no env, no label check)", [(R, "[[ $1 == scip-typescript || $1 == scip-python ]]", "[[ $1 == scip-typescript ]]")]),
 ("I29", "22,23", "(22)/(23) an indexer launch carries a flag the probe template lacks", [(R, '    sbx_container_argv "$src" "$out" "$sdir" "$wd" -- "$@"\n', '    sbx_container_argv "$src" "$out" "$sdir" "$wd" -- "$@"\n    [ "$SBX_RECORD_FRONTEND" != lib-launch ] && SBX_LAUNCHER=("${SBX_LAUNCHER[@]:0:3}" --memory 3g "${SBX_LAUNCHER[@]:3}")\n')]),
 ("I30", "23", "(23) an indexer launch without --cap-drop ALL (launcher clause)", [(R, "--read-only --cap-drop ALL", "--read-only")]),
 ("I31", "7,22,23,26", "(26) PYTHONSAFEPATH missing from the indexer pass-list", [(R, '  if [ "${SBX_INDEXER:-0}" = 1 ]; then SBX_ENV+=("PYTHONSAFEPATH=1"); fi', "  :")]),
 ("I32", "7,22,23,26", "(26) PYTHONSAFEPATH=0 on the indexer pass-list", [(R, 'SBX_ENV+=("PYTHONSAFEPATH=1")', 'SBX_ENV+=("PYTHONSAFEPATH=0")')]),
 ("I33", "7,1", "(26) PYTHONSAFEPATH on every entry, not only the indexers", [(R, '  if [ "${SBX_INDEXER:-0}" = 1 ]; then SBX_ENV+=("PYTHONSAFEPATH=1"); fi', '  SBX_ENV+=("PYTHONSAFEPATH=1")')]),
 ("I34", "7", "(26) a forbidden indexer name (PYTHONPATH) on the indexer pass-list", [(R, 'SBX_ENV+=("PYTHONSAFEPATH=1")', 'SBX_ENV+=("PYTHONSAFEPATH=1" "PYTHONPATH=/src")')]),
 ("I35", "24", "(24) scip-typescript --pnpm-workspaces dropped from the forbidden list", [(F, "scip-typescript\t--pnpm-workspaces\tall\t", "joern\t--dep\tall\t")]),
 ("I36", "24", "(24) scip-typescript --infer-tsconfig entry made stale", [(F, "scip-typescript\t--infer-tsconfig\t", "scip-typescript\t--infer-tsconfigx\t")]),
 ("I37", "24", "(24) scip-typescript --yarn-workspaces listed in both files", [(A, "c2cpg\t--compilation-database\t", "scip-typescript\t--yarn-workspaces\tx\nc2cpg\t--compilation-database\t")]),
 ("I38", "26", "(26) PYTHONSAFEPATH missing from the indexer pass-list (test 26 alone)", [(R, '  if [ "${SBX_INDEXER:-0}" = 1 ]; then SBX_ENV+=("PYTHONSAFEPATH=1"); fi', "  :")]),
 ("I39", "23", "(23) PYTHONSAFEPATH missing from the indexer pass-list (test 23 alone)", [(R, '  if [ "${SBX_INDEXER:-0}" = 1 ]; then SBX_ENV+=("PYTHONSAFEPATH=1"); fi', "  :")]),
 ("I40", "23", "(23) NODE_OPTIONS on the indexer pass-list (A6)", [(R, 'SBX_ENV+=("PYTHONSAFEPATH=1")', 'SBX_ENV+=("PYTHONSAFEPATH=1" "NODE_OPTIONS=--require=/src/x.js")')]),
 ("I41", "26", "(26) the indexer pass-list is never applied to a lib launch (SBX_INDEXER reset after use)", [(R, '  if [ "${SBX_INDEXER:-0}" = 1 ]; then SBX_ENV+=("PYTHONSAFEPATH=1"); fi', '  if [ "${SBX_INDEXER:-0}" = 1 ] && [ "$SBX_RECORD_FRONTEND" != lib-launch ]; then SBX_ENV+=("PYTHONSAFEPATH=1"); fi')]),
 # --- P1-48b fixes: --project-version, component class, symlink containment, launch by image ID ---
 ("J01", "22", "--project-version is no longer mandatory for scip-python", [(R, '  if [ "$fe" = scip-python ] && ((!seen_ver)); then SBX_IDX_MISSING+=(--project-version); fi\n', "")]),
 ("J02", "22", "--project-version accepted for scip-typescript", [(R, '        [ "$fe" = scip-python ] && ((!seen_ver && i + 1 < n))', '        ((!seen_ver && i + 1 < n))')]),
 ("J03", "22", "a repeated --project-version is accepted", [(R, "((!seen_ver && i + 1 < n)) && [[ $val", "((i + 1 < n)) && [[ $val")]),
 ("J04", "22", "--project-version pattern relaxed (a leading - or . allowed)", [(R, "((!seen_ver && i + 1 < n)) && [[ $val =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]]", "((!seen_ver && i + 1 < n)) && [[ $val =~ ^[A-Za-z0-9._-]{0,64}$ ]]")]),
 ("J05", "22", "path component class dropped (only non-empty, not . or ..)", [(R, "[[ $c =~ ^[A-Za-z0-9._@+-]+$ && $c != . && $c != .. ]] || return 1", "[[ -n $c && $c != . && $c != .. ]] || return 1")]),
 ("J06", "22", "path component class admits a space", [(R, "^[A-Za-z0-9._@+-]+$ && $c", "^[A-Za-z0-9._@+ -]+$ && $c")]),
 ("J07", "22", "symlink containment check removed", [(R, '      sbx_idx_contained "$src" -- "${SBX_TOOL_ARGV[@]}"\n', "")]),
 ("J08", "22", "containment checks --cwd only, not --environment", [(R, "    case \"${a[i]}\" in --cwd|--environment) ;; *) continue ;; esac", "    case \"${a[i]}\" in --cwd) ;; *) continue ;; esac")]),
 ("J09", "22", "containment: a `..` reached through a relative link may leave the tree", [(R, '      [[ $cur == "$src" || $cur == "$src"/* ]] || return 1\n', "")]),
 ("J10", "22", "containment does not require --cwd to be a directory", [(R, '{ [ "${a[i]}" = --cwd ] && [ -d "$SBX_IDX_REAL" ]; }', '[ "${a[i]}" = --cwd ]')]),
 ("J11", "22", "containment accepts a component that does not exist (dangling link, missing file)", [(R, '    [ -e "$next" ] || return 1\n', "")]),
 ("J12", "22", "the image is launched by name, not by the inspected ID", [(R, '--entrypoint /usr/bin/env "$SBX_IMAGE_ID" -i', '--entrypoint /usr/bin/env "$SBX_IMAGE" -i')]),
 ("J13", "22", "scip-python alone resolves to a wrong path (rc 127; scip-typescript unaffected)", [(R, 'echo "$SBX_INDEXER_IMG_BIN/$1";', 'echo "$SBX_INDEXER_IMG_BIN/${1/python/pythonx}";')]),
 ("J14", "22", "containment walks from the wrong base (the host root)", [(R, "local src=$1 cur=$1 rest=${2#/src}", "local src=$1 cur=/ rest=${2#/src}")]),
 ("J15", "22", "containment follows an absolute symlink by its host path (the realpath-only rule)", [(R, '      [[ $t == /* ]] && return 1\n', '      if [[ $t == /* ]]; then cur=$(realpath -e -- "$t") || return 1; [[ $cur == "$src" || $cur == "$src"/* ]] || return 1; continue; fi\n')]),
 ("J16", "22", "a containment refusal no longer outranks a missing mandatory element (usage first)", [(R, '      # Containment is forbidden-class: it outranks a missing mandatory element (usage).\n', '      if ((${#SBX_IDX_MISSING[@]})); then sbx_refuse 64 usage "mandatory element missing"; fi\n')]),
 ("J17", "22", "--environment no longer required to be a regular file (the -f check dropped: a directory or a FIFO passes)", [(R, '{ [ "${a[i]}" = --environment ] && [ -f "$SBX_IDX_REAL" ]; }', '[ "${a[i]}" = --environment ]')]),
 # --- indexer wall-clock watchdog (egress-spec4 §1) ---
 ("K01", "27", "(27) the wall-clock cap is applied to every frontend, not only the indexers", [(R, 'if [ "${SBX_INDEXER:-0}" = 1 ]; then\n    trap \'sbx_on_wall\' USR2', 'if true; then\n    trap \'sbx_on_wall\' USR2')]),
 ("K02", "27", "(27) the watchdog (and the signal handler) send TERM instead of KILL to the client", [(R, '    kill -KILL "$j" 2>/dev/null; wait "$j" 2>/dev/null\n', '    kill -TERM "$j" 2>/dev/null; wait "$j" 2>/dev/null\n')]),
 ("K03", "27", "(27) the watchdog skips the removal of the run's containers", [(R, '  if [ "$SBX_BACKEND" = container ]; then sbx_remove_own; fi\n  sbx_write_record 124 ""', '  sbx_write_record 124 ""')]),
 ("K04a", "27", "(27) first wins: sbx_on_signal no longer ignores USR2 (a late timer signal re-enters as a watchdog)", [(R, "  trap '' USR2\n  local wd=${SBX_WD_PID:-}\n  sbx_wd_disarm\n", "  local wd=${SBX_WD_PID:-}\n  sbx_wd_disarm\n")]),
 ("K04b", "27", "(27) first wins: sbx_on_wall no longer ignores USR2 (a second USR2 re-enters the watchdog handler)", [(R, "sbx_on_wall() {\n  trap '' USR2\n", "sbx_on_wall() {\n")]),
 ("K06", "27", "(27) the timer signals without checking that the runner is the same process (PID reuse)", [(R, '      [ "$(sbx_pid_start "$me")" = "$st" ] || exit 0\n', "")]),
 ("K07", "27", "(27) the timer keeps the runner's --output lock fd open", [(R, '    if [ -n "${SBX_LOCK_FD:-}" ]; then exec {SBX_LOCK_FD}<&-; fi\n', "")]),
 ("K08", "27", "(27) sbx_on_signal resets INT/TERM/HUP to the default during the removal (a second TERM aborts it)", [(R, "  sbx_wd_disarm\n  trap '' INT TERM HUP\n", "  sbx_wd_disarm\n  trap - INT TERM HUP\n")]),
 ("K09", "27", "(27) sbx_on_wall resets INT/TERM/HUP to the default during the removal", [(R, "  trap '' USR2\n  trap '' INT TERM HUP\n  sbx_log \"indexer", "  trap '' USR2\n  trap - INT TERM HUP\n  sbx_log \"indexer")]),
 ("K10", "27", "(27) the disarm does not stop the timer (its sleep outlives a normal exit)", [(R, '  kill -TERM "$SBX_WD_PID" 2>/dev/null\n  SBX_WD_PID=""', '  SBX_WD_PID=""')]),
 ("K05", "27", "(27) the wall-clock constant is read from the environment", [(R, '\nSBX_INDEXER_WALL_S=1800\n', '\nSBX_INDEXER_WALL_S=${SBX_INDEXER_WALL_S:-1800}\n')]),
 ("K11", "27", "(27) sbx_on_signal does not disarm the timer (its sleep outlives a signalled exit)", [(R, "  local wd=${SBX_WD_PID:-}\n  sbx_wd_disarm\n", "  local wd=${SBX_WD_PID:-}\n")]),
 ("K12", "27", "(27) no check that the runner's start time was read (the identity check fails open)", [(R, '  if [ -z "$st" ]; then sbx_log "watchdog: the runner\'s start time is unreadable, no timer armed"; return 1; fi\n', "")]),
 ("K13", "27", "(27) /proc stat read line by line (a newline in the process name hides field 22)", [(R, "IFS= read -r -d '' l <", "IFS= read -r l <")]),
 ("K14", "27", "(27) the timer's step is the whole remainder (an orphan outlives a dead runner up to the cap)", [(R, "((step > 30)) && step=30", ":")]),
]
for mid, tests, desc, edits in M:
    if sel and mid not in sel: continue
    d = os.path.join(work, mid); os.makedirs(d)
    for f in (R, F, A): shutil.copy2(os.path.join(sbx, f), d)
    shutil.copytree(os.path.join(sbx, "scripts"), os.path.join(d, "scripts"))
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
  tmp_before=$(ls -d /tmp/ccg-sbx.* 2>/dev/null | sort)
  timeout -k 10 "${MUTANT_TIMEOUT:-900}" "$HERE/test_sandbox.sh" --runner "$d/offline-run.sh" --only "$tests" > "$d/test.out" 2>&1; rc=$?
  # M42 leaks its scratch by design, into the suite's private /tmp under its own $T, which the
  # suite removes. A new /tmp/ccg-sbx.* here cannot be attributed (another session's runner may
  # own it), so it is reported and never removed (review S08 p3-m2).
  for t in $(comm -13 <(printf '%s\n' $tmp_before) <(ls -d /tmp/ccg-sbx.* 2>/dev/null | sort)); do
    echo "$id: note: new $t appeared during this mutant (not removed; may be another session's)" >&2
  done
  first=$(grep -m1 '^  FAIL ' "$d/test.out" | sed 's/^  FAIL //; s/|/\\|/g' | cut -c1-110)
  if [ "$rc" = 124 ] || [ "$rc" = 137 ]; then
    res="killed (timeout)"; killed=$((killed + 1)); first="test run exceeded ${MUTANT_TIMEOUT:-900} s"
  elif [ "$rc" != 0 ] && [ -n "$first" ]; then res=killed; killed=$((killed + 1)); else res=SURVIVED; survived=$((survived + 1)); fi
  printf '%s %-8s tests=%-6s %s :: %s\n' "$id" "$res" "$tests" "$desc" "$first"
  printf '| %s | %s | %s | %s | %s |\n' "$id" "${desc//|/\\|}" "$tests" "$res" "$first" >> "$W/table.md"
done 3< "$W/manifest"
echo; cat "$W/table.md"
echo; echo "killed=$killed survived=$survived"
[ -n "${MUTANT_TABLE_OUT:-}" ] && cp "$W/table.md" "$MUTANT_TABLE_OUT"
((survived == 0))
