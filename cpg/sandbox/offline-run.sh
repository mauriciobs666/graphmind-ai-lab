#!/bin/bash -p
# offline-run.sh — the C-1001 P0 offline build sandbox (S08, unit P0-07).
#
# Contract: docs/plans/connected-code-graph-egress-spec.md v1.3 §7 (interface, backends, escape
# axes, flags, argv audit), §5.3 (the `sandbox` audit record) and §10 (what S08 encodes). Where
# that spec and the P0 plan (docs/plans/connected-code-graph-p0.md §2 D6) disagree, the spec
# controls. Review items folded in: docs/reviews/connected-code-graph-p0-impl.md § S04 Pass 3
# P3-m1 (directive whitespace set) and P3-n1 (/scripts is a directory bind).
#
#   offline-run.sh --backend container|namespace|auto --mode offline|controlled \
#       --frontend <name> --source <dir> --output <dir> [--codebase <repoId>]... \
#       [--script-file <host path>] -- <tool args...>
#
# Exit codes: the tool's own exit code when it ran; 64 = refused before launch (argv, frontend,
# mode, directive, usage); 69 = backend refused or unavailable. Every invocation except --help
# appends exactly one JSON line to <auditDir>/sandbox.jsonl (auditDir = $CCG_AUDIT_DIR, default
# cpg/.audit/). The runner reads no other environment variable except CCG_TOOLCHAIN_IMAGE.
#
# Runner-owned argv conventions (README.md "Using the runner"):
#   * The literal token @AUDIT_ID@ in any tool argument is replaced by this invocation's auditId,
#     so a tool's own output files can carry the auditId stem (§5.3).
#   * For a parse frontend (everything but `joern`) the runner appends
#     `--output /out/<auditId>.cpg.bin` and refuses a caller-supplied -o/--output.
#
# Environment hygiene (review S08 n1): the shebang's -p makes bash ignore BASH_ENV/ENV and
# environment functions on a direct exec, and sbx_main re-execs itself under `env -i` (keeping
# only CCG_AUDIT_DIR, CCG_TOOLCHAIN_IMAGE, a fixed PATH and the passwd-db HOME) whenever the
# environment holds anything else, so DOCKER_HOST, DOCKER_CONTEXT and the like never reach
# `docker`. Invoking it as `bash offline-run.sh` bypasses the shebang: BASH_ENV would run before
# line 1, which no script can prevent. Run it directly.
#
# Sourced (`. offline-run.sh`) it only defines functions: the test-only entry points
# (fixture-injected precondition checks, the raw launch templates used by the escape-axis probe
# and the --help sweep). The production argv parser below never exposes them.

SBX_SELF="${BASH_SOURCE[0]}"
SBX_DIR="$(cd "$(dirname "$SBX_SELF")" && pwd)"
SBX_FORBIDDEN_LIST="$SBX_DIR/forbidden-flags.txt"
SBX_ALLOWED_LIST="$SBX_DIR/allowed-flags.txt"
SBX_JOERN_IMG_HOME=/opt/joern/joern-cli
SBX_JAVA_OPTS='-Xmx2g'
SBX_LABEL=ccg.sandbox
SBX_LABEL_VALUE=s08

# The closed frontend set (§7.3.1) and the name -> executable map.
SBX_FRONTENDS=(pysrc2cpg jssrc2cpg javasrc2cpg jimple2cpg kotlin2cpg csharpsrc2cpg gosrc2cpg c2cpg joern-parse joern)
sbx_frontend_exe() {
  case "$1" in
    c2cpg) echo c2cpg.sh ;;
    jssrc2cpg) echo jssrc2cpg.sh ;;
    *) echo "$1" ;;
  esac
}
# joern-parse --language value (lower-cased) -> frontend (§7.3.1 table). python/java: refused.
sbx_language_frontend() {
  case "$1" in
    pythonsrc) echo pysrc2cpg ;;
    jssrc|javascript) echo jssrc2cpg ;;
    javasrc) echo javasrc2cpg ;;
    kotlin) echo kotlin2cpg ;;
    csharpsrc|csharp) echo csharpsrc2cpg ;;
    golang) echo gosrc2cpg ;;
    newc|c) echo c2cpg ;;
    *) return 1 ;;
  esac
}

sbx_log() { printf 'offline-run: %s\n' "$*" >&2; }

# ------------------------------------------------------------------ JSON helpers
sbx_js() {
  local s=$1 out="" c i
  s=${s//\\/\\\\}; s=${s//\"/\\\"}
  s=${s//$'\n'/\\n}; s=${s//$'\r'/\\r}; s=${s//$'\t'/\\t}; s=${s//$'\f'/\\f}; s=${s//$'\b'/\\b}
  if [[ $s == *[$'\x01'-$'\x1f']* ]]; then
    for ((i = 0; i < ${#s}; i++)); do
      c=${s:i:1}
      if [[ $c == [$'\x01'-$'\x1f'] ]]; then printf -v c '\\u%04x' "'$c"; fi
      out+=$c
    done
    s=$out
  fi
  printf '"%s"' "$s"
}
sbx_jarr() { # items... -> JSON array of strings
  local out="[" first=1 x
  for x in "$@"; do
    ((first)) || out+=","
    first=0; out+=$(sbx_js "$x")
  done
  printf '%s]' "$out"
}
sbx_jnull() { if [ -n "$1" ]; then sbx_js "$1"; else printf null; fi; }

sbx_uuid() { tr '[:upper:]' '[:lower:]' < /proc/sys/kernel/random/uuid; }
sbx_now_ms() { date +%s%3N; }

# ------------------------------------------------------------------ preconditions (§7.1, §7.1.1)
# sbx_interop_state <binfmt_dir> <kernel_release>  ->  enabled | disabled | unreadable on stdout.
# Also sets SBX_PE_MATCHED (names of entries matched by rules a-d) and SBX_PE_PINNED (1 if any
# matched entry carries flag F, which is C1).
sbx_interop_state() { sbx_interop_scan "$@"; echo "$SBX_INTEROP_STATE"; }
sbx_interop_scan() { # same, without a subshell: sets SBX_INTEROP_STATE
  local dir=$1 kr=$2 f name first line state=disabled
  SBX_PE_MATCHED=(); SBX_PE_PINNED=0
  if [ ! -f "$dir/status" ]; then SBX_INTEROP_STATE=unreadable; return 0; fi
  if [ "$(head -n1 "$dir/status" 2>/dev/null)" = disabled ]; then SBX_INTEROP_STATE=disabled; return 0; fi
  for f in "$dir"/*; do
    [ -f "$f" ] || continue
    name=${f##*/}
    case "$name" in status|register) continue ;; esac
    first=$(head -n1 "$f" 2>/dev/null) || continue
    [ "$first" = enabled ] || continue
    local hit=0 flags="" etype=""
    while IFS= read -r line || [ -n "$line" ]; do
      case "$line" in
        "magic 4d5a"*) hit=1 ;;
        "interpreter /init") hit=1 ;;
        "flags:"*) flags=${line#flags:} ;;
      esac
      if [[ $line == extension\ * ]]; then
        local ext=${line#extension }; ext=${ext#.}
        [ "${ext,,}" = exe ] && hit=1
      fi
    done < "$f"
    case "$name" in WSLInterop*) hit=1 ;; esac
    if ((hit)); then
      state=enabled; SBX_PE_MATCHED+=("$name")
      [[ $flags == *F* ]] && SBX_PE_PINNED=1
    fi
  done
  SBX_INTEROP_STATE=$state
}
# Namespace N1 outcome: 0 = start, 1 = refuse.
sbx_n1_refuses() { # <interop_state> <kernel_release>
  [ "$1" = enabled ] && return 0
  [ "$1" = unreadable ] && [[ ${2,,} == *microsoft* ]] && return 0
  return 1
}
# C2: <docker info SecurityOptions text>. Prints the seccomp profile name (or nothing); returns 0
# iff there is a seccomp entry and every one has profile=builtin.
sbx_c2_check() {
  local text=$1 e prof found=0 ok=1
  while IFS= read -r e; do
    [ -n "$e" ] || continue
    found=1
    prof=$(printf '%s' "$e" | tr ',' '\n' | sed -n 's/^profile=//p' | head -n1)
    printf '%s' "$prof"
    [ "$prof" = builtin ] || ok=0
  done < <(printf '%s' "$text" | grep -oE 'name=seccomp(,[^]" ,]+)*' || true)
  ((found && ok))
}
# N2: /dev/vsock present?
sbx_vsock_present() { [ -e "${1:-/dev/vsock}" ]; }

# ------------------------------------------------------------------ flag lists (§7.3.3)
SBX_LIST_ERR=""
sbx_load_lists() {
  FORB_FE=(); FORB_FLAG=(); FORB_IN=(); ALLOW_FE=(); ALLOW_FLAG=(); SBX_LIST_ERR=""
  local fe fl fin why extra n=0
  [ -r "$SBX_FORBIDDEN_LIST" ] && [ -r "$SBX_ALLOWED_LIST" ] || { SBX_LIST_ERR="flag list missing"; return 1; }
  while IFS=$'\t' read -r fe fl fin why extra || [ -n "$fe" ]; do
    n=$((n + 1))
    [[ -z $fe || $fe == \#* ]] && continue
    if [ -z "$fl" ] || [ -z "$fin" ] || [ -z "$why" ] || [ -n "$extra" ] || [[ $fin != offline && $fin != all ]]; then
      SBX_LIST_ERR="forbidden-flags.txt line $n malformed"; return 1
    fi
    FORB_FE+=("$fe"); FORB_FLAG+=("${fl,,}"); FORB_IN+=("$fin")
  done < "$SBX_FORBIDDEN_LIST"
  n=0
  while IFS=$'\t' read -r fe fl why extra || [ -n "$fe" ]; do
    n=$((n + 1))
    [[ -z $fe || $fe == \#* ]] && continue
    if [ -z "$fl" ] || [ -z "$why" ] || [ -n "$extra" ]; then
      SBX_LIST_ERR="allowed-flags.txt line $n malformed"; return 1
    fi
    ALLOW_FE+=("$fe"); ALLOW_FLAG+=("${fl,,}")
  done < "$SBX_ALLOWED_LIST"
  SBX_LIST_DIGEST=$(cat "$SBX_FORBIDDEN_LIST" "$SBX_ALLOWED_LIST" | sha256sum | cut -d' ' -f1)
}

# §7.4 steps 2-3 over every element. Args: <mode> <frontends (space-separated)> -- elements...
# Appends to SBX_HITS / SBX_ALLOWED_HITS.
sbx_audit_elements() {
  local mode=$1 fes=" $2 " e el i f hit
  shift 3
  for e in "$@"; do
    el=${e,,}; hit=0
    for i in "${!FORB_FLAG[@]}"; do
      [[ ${FORB_FE[i]} == '*' || $fes == *" ${FORB_FE[i]} "* ]] || continue
      [[ ${FORB_IN[i]} == all || $mode == offline ]] || continue
      f=${FORB_FLAG[i]}
      if [[ $el == "$f" || $el == "$f="* || $el == *"$f"* ]]; then hit=1; break; fi
    done
    if ((!hit)) && [[ $e == -[!-]* ]]; then # launcher-level options, case-sensitive (§7.3.5)
      case "$e" in
        -D*|-J*|-agentlib*|-agentpath*|-javaagent*|-XX*|-jvm-debug|-main|-java-home) hit=1 ;;
      esac
      if ((hit)) && [[ $e =~ ^-J-Xmx[0-9]+[mg]$ || $e =~ ^-J-Xss[0-9]+[km]$ ]]; then hit=0; fi
    fi
    if ((hit)); then SBX_HITS+=("$e"); continue; fi
    for i in "${!ALLOW_FLAG[@]}"; do
      [[ $fes == *" ${ALLOW_FE[i]} "* ]] || continue
      if [[ $el == *"${ALLOW_FLAG[i]}"* ]]; then SBX_ALLOWED_HITS+=("$e"); break; fi
    done
  done
}

# §7.3.1 script-mode grammar. Args: <script basename> -- elements. Appends offenders to SBX_HITS.
sbx_joern_grammar() {
  local base=$1 n i ok=1
  shift 2
  local a=("$@")
  n=${#a[@]}
  if ((n < 2)) || [ "${a[0]}" != --script ] || [ "${a[1]}" != "/scripts/$base" ]; then
    ((n >= 1)) && SBX_HITS+=("${a[0]}") || SBX_HITS+=("(missing --script)")
    ((n >= 2)) && [ "${a[0]}" = --script ] && SBX_HITS+=("${a[1]}")
    ok=0; i=2
  else
    i=2
  fi
  while ((i < n)); do
    if [ "${a[i]}" = --param ] && ((i + 1 < n)) && [[ ${a[i+1]} =~ ^[A-Za-z][A-Za-z0-9_]*= ]]; then
      i=$((i + 2))
    else
      SBX_HITS+=("${a[i]}"); ok=0; i=$((i + 1))
    fi
  done
  ((ok))
}

# §7.3.1 directive refusal (P3-m1, review S08 M1, egress-spec2 §1.1): replpp's UsingDirectives.scanFor is
# `line.trim().startsWith("//> using")`, and Java's String.trim() strips EVERY char <= U+0020.
# So a line is refused when //> follows only bytes 0x00-0x20, plus 0x7f (fail closed): in the C
# locale that is [[:cntrl:] ]. \r splits lines too: verified, replpp honours `x\r//> using dep`
# (S08 fix round; it attempted the download). \f and \v also count as breaks (fail closed).
# Prints the offending lines; returns 0 iff there is none.
sbx_directive_lines() {
  local out
  out=$(LC_ALL=C tr '\r\f\v' '\n\n\n' < "$1" | LC_ALL=C grep -a -E '^[[:cntrl:] ]*//>' || true)
  [ -z "$out" ] && return 0
  printf '%s\n' "$out"
  return 1
}

# ------------------------------------------------------------------ environment (§7.3.4)
# sbx_env_pairs <backend>: sets SBX_ENV (K=V list, offline pass-list) from SBX_IMG_ENV_* (container)
# or the host JDK (namespace).
sbx_env_pairs() {
  if [ "$1" = container ]; then
    SBX_ENV=("PATH=$SBX_IMG_PATH" "JAVA_HOME=$SBX_IMG_JAVA_HOME")
  else
    SBX_ENV=("PATH=$SBX_JOERN_IMG_HOME:$SBX_HOST_JDK/bin:/usr/bin:/bin" "JAVA_HOME=$SBX_HOST_JDK")
  fi
  SBX_ENV+=("HOME=/home/ccg" "TMPDIR=/tmp" "LANG=C.UTF-8" "LC_ALL=C.UTF-8" "JAVA_OPTS=$SBX_JAVA_OPTS"
    "GOTOOLCHAIN=local" "GOROOT=${SBX_IMG_GOROOT:-/opt/go}" "DOTNET_ROOT=${SBX_IMG_DOTNET_ROOT:-/opt/dotnet}"
    "DOTNET_CLI_TELEMETRY_OPTOUT=1" "DOTNET_NOLOGO=1" "DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1")
}
sbx_read_image_env() { # <image>
  local kv
  SBX_IMG_PATH=""; SBX_IMG_JAVA_HOME=""; SBX_IMG_GOROOT=""; SBX_IMG_DOTNET_ROOT=""
  while IFS= read -r kv; do
    case "$kv" in
      PATH=*) SBX_IMG_PATH=${kv#PATH=} ;;
      JAVA_HOME=*) SBX_IMG_JAVA_HOME=${kv#JAVA_HOME=} ;;
      GOROOT=*) SBX_IMG_GOROOT=${kv#GOROOT=} ;;
      DOTNET_ROOT=*) SBX_IMG_DOTNET_ROOT=${kv#DOTNET_ROOT=} ;;
    esac
  done < <(docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$1" 2>/dev/null)
  [ -n "$SBX_IMG_PATH" ] && [ -n "$SBX_IMG_JAVA_HOME" ]
}

# ------------------------------------------------------------------ backend resolution
sbx_default_image() {
  local b="$SBX_DIR/../infra/toolchain/build.sh"
  [ -x "$b" ] && env -i PATH=/usr/bin:/bin "$b" --print-tag 2>/dev/null
}
sbx_host_joern() {
  local home c
  home=$(getent passwd "$(id -u)" | cut -d: -f6)
  for c in /opt/joern/joern-cli "$home/joern/joern-cli"; do
    [ -x "$c/joern-parse" ] && { echo "$c"; return 0; }
  done
  return 1
}
sbx_host_jdk() {
  local j
  j=$(readlink -f /usr/bin/java 2>/dev/null) || return 1
  j=${j%/bin/java}
  "$j/bin/java" -version 2>&1 | grep -q 'version "21' || return 1
  echo "$j"
}

# Sets SBX_BACKEND, SBX_BACKEND_REFUSAL ("" | backend-precondition | backend-unavailable) and the
# PRE_* precondition values. Read-only: creates no container, network, volume or namespace.
sbx_try_container() {
  local secopts
  SBX_BACKEND=container; SBX_BACKEND_REFUSAL=""; SBX_REFUSAL_DETAIL=""
  sbx_interop_scan /proc/sys/fs/binfmt_misc "$(uname -r)"; PRE_INTEROP=$SBX_INTEROP_STATE
  PRE_PINNED=$SBX_PE_PINNED; PRE_SECCOMP=""; PRE_VSOCK=absent
  sbx_vsock_present /dev/vsock && PRE_VSOCK=present
  if ! command -v docker >/dev/null || ! secopts=$(timeout 20 docker info --format '{{json .SecurityOptions}}' 2>/dev/null); then
    SBX_BACKEND_REFUSAL=backend-unavailable; SBX_REFUSAL_DETAIL="no Docker daemon"; return 1
  fi
  SBX_IMAGE=${CCG_TOOLCHAIN_IMAGE:-$(sbx_default_image)}
  if [ -z "$SBX_IMAGE" ] || ! SBX_IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$SBX_IMAGE" 2>/dev/null) \
     || ! sbx_read_image_env "$SBX_IMAGE"; then
    SBX_BACKEND_REFUSAL=backend-unavailable; SBX_REFUSAL_DETAIL="no image '${SBX_IMAGE:-?}'"; return 1
  fi
  if PRE_SECCOMP=$(sbx_c2_check "$secopts"); then PRE_VSOCK=blocked; else
    SBX_BACKEND_REFUSAL=backend-precondition; SBX_REFUSAL_DETAIL="C2: docker info has no name=seccomp,profile=builtin"
  fi
  if ((PRE_PINNED)); then
    SBX_BACKEND_REFUSAL=backend-precondition; SBX_REFUSAL_DETAIL="C1: a PE binfmt entry has flag F"
  fi
  [ -z "$SBX_BACKEND_REFUSAL" ]
}
sbx_try_namespace() {
  SBX_BACKEND=namespace; SBX_BACKEND_REFUSAL=""; SBX_REFUSAL_DETAIL=""
  local kr; kr=$(uname -r)
  sbx_interop_scan /proc/sys/fs/binfmt_misc "$kr"; PRE_INTEROP=$SBX_INTEROP_STATE
  PRE_PINNED=$SBX_PE_PINNED; PRE_SECCOMP=""; PRE_VSOCK=absent
  sbx_vsock_present /dev/vsock && PRE_VSOCK=present
  if sbx_n1_refuses "$PRE_INTEROP" "$kr"; then
    SBX_BACKEND_REFUSAL=backend-precondition; SBX_REFUSAL_DETAIL="N1: WSL/PE interop is $PRE_INTEROP"
  fi
  if [ "$PRE_VSOCK" = present ]; then
    SBX_BACKEND_REFUSAL=backend-precondition
    SBX_REFUSAL_DETAIL="${SBX_REFUSAL_DETAIL:+$SBX_REFUSAL_DETAIL; }N2: /dev/vsock exists"
  fi
  [ -n "$SBX_BACKEND_REFUSAL" ] && return 1
  if ! command -v bwrap >/dev/null || ! bwrap --unshare-user --ro-bind / / /bin/true 2>/dev/null; then
    SBX_BACKEND_REFUSAL=backend-unavailable; SBX_REFUSAL_DETAIL="no bwrap or no unprivileged user namespaces"; return 1
  fi
  if ! SBX_HOST_JOERN=$(sbx_host_joern) || ! SBX_HOST_JDK=$(sbx_host_jdk); then
    SBX_BACKEND_REFUSAL=backend-unavailable; SBX_REFUSAL_DETAIL="no host Joern or JDK 21"; return 1
  fi
  return 0
}
sbx_resolve_backend() { # <container|namespace|auto>
  case "$1" in
    container) sbx_try_container ;;
    namespace) sbx_try_namespace ;;
    auto) sbx_try_container || sbx_try_namespace ;;
  esac
}

# ------------------------------------------------------------------ launch templates
# Mount option strings (§7.1). Docker binds have no noexec, so /src and /scripts are local-driver
# volumes of type none (o=bind,ro,...: moby re-mounts a ro bind with every flag), and /out is a
# local-driver overlay volume whose upperdir is the output directory (a rw bind never gets noexec).
SBX_SRC_OPTS="ro,noexec,nosuid,nodev"
SBX_OUT_OPTS="rw,noexec,nosuid,nodev"
SBX_TMP_OPTS="rw,exec,nosuid,nodev"
SBX_HOME_OPTS="rw,noexec,nosuid,nodev"

sbx_vol_labels() { printf 'volume-label=%s=%s,volume-label=%s.auditId=%s' "$SBX_LABEL" "$SBX_LABEL_VALUE" "$SBX_LABEL" "$SBX_AUDIT_ID"; }

# Prepare the runner-owned scratch: overlay lower/work dirs (beside the output, same filesystem).
sbx_prepare_out() { # <output dir>
  local parent; parent=$(dirname "$1")
  SBX_OVL=$(mktemp -d "$parent/.ccg-ovl-$SBX_AUDIT_ID.XXXXXX" 2>/dev/null) || return 1
  mkdir -p "$SBX_OVL/lower" "$SBX_OVL/work"
  SBX_CLEANUP+=("$SBX_OVL")
}

# sbx_container_argv <src> <out> <scriptsdir|""> <workdir|""> -- <tool> <args...>  -> SBX_LAUNCHER
sbx_container_argv() {
  local src=$1 out=$2 sdir=$3 wd=$4
  shift 5
  local uid gid; uid=$(id -u); gid=$(id -g)
  SBX_LAUNCHER=(docker run --rm --pull=never --name "ccg-sbx-$SBX_AUDIT_ID"
    --label "$SBX_LABEL=$SBX_LABEL_VALUE" --label "$SBX_LABEL.auditId=$SBX_AUDIT_ID"
    --network none --hostname localhost --read-only --cap-drop ALL
    --security-opt no-new-privileges --user "$uid:$gid" --pids-limit 4096
    --tmpfs "/tmp:$SBX_TMP_OPTS" --tmpfs "/home/ccg:$SBX_HOME_OPTS,uid=$uid,gid=$gid"
    --mount "type=volume,dst=/src,volume-driver=local,volume-nocopy=true,$(sbx_vol_labels),volume-opt=type=none,\"volume-opt=o=bind,$SBX_SRC_OPTS\",volume-opt=device=$src"
    --mount "type=volume,dst=/out,volume-driver=local,volume-nocopy=true,$(sbx_vol_labels),volume-opt=type=overlay,volume-opt=device=overlay,\"volume-opt=o=lowerdir=$SBX_OVL/lower,upperdir=$out,workdir=$SBX_OVL/work,noexec,nosuid,nodev\"")
  SBX_MOUNTS=("/src|$src|$SBX_SRC_OPTS" "/out|$out|$SBX_OUT_OPTS" "/tmp|tmpfs|$SBX_TMP_OPTS" "/home/ccg|tmpfs|$SBX_HOME_OPTS")
  if [ -n "$sdir" ]; then
    SBX_LAUNCHER+=(--mount "type=volume,dst=/scripts,volume-driver=local,volume-nocopy=true,$(sbx_vol_labels),volume-opt=type=none,\"volume-opt=o=bind,$SBX_SRC_OPTS\",volume-opt=device=$sdir")
    SBX_MOUNTS+=("/scripts|$sdir|$SBX_SRC_OPTS")
  fi
  [ -n "$wd" ] && SBX_LAUNCHER+=(--workdir "$wd")
  sbx_env_pairs container
  SBX_LAUNCHER+=(--entrypoint /usr/bin/env "$SBX_IMAGE" -i "${SBX_ENV[@]}" "$@")
}

# sbx_bind_tree <dir>: read-only bind <dir>, or, when a host mount sits below it, its children
# one by one, skipping every mount point (needs the caller's `submounts` array).
# sbx_nsb <ro|rw|tmpfs> <src> <dst>: one namespace mount, recorded in SBX_MOUNTS (review S08 m6).
sbx_nsb() {
  case "$1" in
    ro) SBX_LAUNCHER+=(--ro-bind "$2" "$3"); SBX_MOUNTS+=("$3|$2|ro") ;;
    rw) SBX_LAUNCHER+=(--bind "$2" "$3"); SBX_MOUNTS+=("$3|$2|rw") ;;
    tmpfs) SBX_LAUNCHER+=(--tmpfs "$3"); SBX_MOUNTS+=("$3|tmpfs|rw") ;;
  esac
}
sbx_bind_tree() {
  local d=$1 m c has=0
  for m in "${submounts[@]}"; do [[ $m == "$d"/* ]] && has=1 && break; done
  if ((!has)); then sbx_nsb ro "$d" "$d"; return; fi
  SBX_LAUNCHER+=(--dir "$d")
  for c in "$d"/* "$d"/.[!.]*; do
    [ -e "$c" ] || [ -L "$c" ] || continue
    local skip=0
    [ "$c" = /usr/lib/wsl ] && continue   # never present (§7.1), not even as a mask
    for m in "${submounts[@]}"; do [ "$m" = "$c" ] && skip=1 && break; done
    ((skip)) && continue
    if [ -L "$c" ]; then SBX_LAUNCHER+=(--symlink "$(readlink "$c")" "$c")
    elif [ -d "$c" ]; then sbx_bind_tree "$c"
    else sbx_nsb ro "$c" "$c"; fi
  done
}

# sbx_namespace_argv <src> <out> <scriptsdir|""> <workdir|""> -- <tool> <args...>  -> SBX_LAUNCHER
sbx_namespace_argv() {
  local src=$1 out=$2 sdir=$3 wd=$4 kv d
  shift 5
  local etc="$SBX_SCRATCH/etc"
  mkdir -p "$etc"
  printf '127.0.0.1 localhost\n::1 localhost\n' > "$etc/hosts"
  printf 'ccg:x:%s:%s::/home/ccg:/bin/sh\n' "$(id -u)" "$(id -g)" > "$etc/passwd"
  printf 'ccg:x:%s:\n' "$(id -g)" > "$etc/group"
  printf 'passwd: files\ngroup: files\nhosts: files\n' > "$etc/nsswitch.conf"
  SBX_LAUNCHER=(bwrap --unshare-all --die-with-parent --new-session --clearenv --hostname localhost)
  SBX_MOUNTS=()
  # /usr without any host submount (spec deviation, README): bwrap's --ro-bind is always
  # recursive, so the spec's `--ro-bind /usr /usr --tmpfs /usr/lib/wsl` still lists the 9p
  # /usr/lib/wsl/drivers and overlay /usr/lib/wsl/lib in mountinfo (A5b). Bind around them.
  local -a submounts=()
  mapfile -t submounts < <(awk '{ m = $5; gsub(/\\040/, " ", m); if (m ~ "^/usr/") print m }' /proc/self/mountinfo | sort -u)
  sbx_bind_tree /usr
  for d in bin sbin lib lib64 lib32 libx32; do
    if [ -L "/$d" ]; then SBX_LAUNCHER+=(--symlink "$(readlink "/$d")" "/$d")
    elif [ -d "/$d" ]; then sbx_nsb ro "/$d" "/$d"; fi
  done
  SBX_LAUNCHER+=(--proc /proc --dev /dev --dir /etc)
  for d in alternatives java-21-openjdk ld.so.cache; do
    [ -e "/etc/$d" ] && sbx_nsb ro "/etc/$d" "/etc/$d"
  done
  for d in hosts passwd group nsswitch.conf; do sbx_nsb ro "$etc/$d" "/etc/$d"; done
  sbx_nsb tmpfs tmpfs /tmp; sbx_nsb tmpfs tmpfs /home/ccg
  sbx_nsb ro "$src" /src; sbx_nsb rw "$out" /out
  sbx_nsb ro "${SBX_HOST_JOERN:-/nonexistent}" "$SBX_JOERN_IMG_HOME"
  [ -n "$sdir" ] && sbx_nsb ro "$sdir" /scripts
  [ -n "$wd" ] && SBX_LAUNCHER+=(--chdir "$wd")
  SBX_LAUNCHER+=(--remount-ro /)
  sbx_env_pairs namespace
  for kv in "${SBX_ENV[@]}"; do SBX_LAUNCHER+=(--setenv "${kv%%=*}" "${kv#*=}"); done
  SBX_LAUNCHER+=(/usr/bin/env -i "${SBX_ENV[@]}" "$@")
}

# envKeys = the keys of the K=V arguments to `env -i` in launcherArgv (§5.3), derived from argv.
sbx_env_keys_from_argv() {
  local i=0 n=${#SBX_LAUNCHER[@]} a
  SBX_ENV_KEYS=()
  while ((i < n)); do
    if [ "${SBX_LAUNCHER[i]}" = /usr/bin/env ] && [ "${SBX_LAUNCHER[i+1]:-}" = -i ]; then
      i=$((i + 2))
      while ((i < n)); do
        a=${SBX_LAUNCHER[i]}
        [[ $a =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || break
        SBX_ENV_KEYS+=("${a%%=*}"); i=$((i + 1))
      done
      return 0
    fi
    # docker: the image argument follows --entrypoint /usr/bin/env <image>; skip the image.
    if [ "${SBX_LAUNCHER[i]}" = --entrypoint ] && [ "${SBX_LAUNCHER[i+1]:-}" = /usr/bin/env ] && [ "${SBX_LAUNCHER[i+3]:-}" = -i ]; then
      i=$((i + 4))
      while ((i < n)); do
        a=${SBX_LAUNCHER[i]}
        [[ $a =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || break
        SBX_ENV_KEYS+=("${a%%=*}"); i=$((i + 1))
      done
      return 0
    fi
    i=$((i + 1))
  done
}

# ------------------------------------------------------------------ the audit record (§5.3)
sbx_append_record() { # <file> <line>
  mkdir -p "$(dirname "$1")"
  if command -v python3 >/dev/null; then
    printf '%s\n' "$2" | python3 -I -c '
import fcntl, os, sys
d = sys.stdin.buffer.read()
fd = os.open(sys.argv[1], os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
fcntl.flock(fd, fcntl.LOCK_EX)
n = os.write(fd, d)
fcntl.flock(fd, fcntl.LOCK_UN)
sys.exit(0 if n == len(d) else 1)' "$1"
  else
    { flock -x 9; printf '%s\n' "$2" >&9; } 9>>"$1"
  fi
}

sbx_write_record() { # <exitCode> <refusalReason|"">
  local rc=$1 reason=$2 m t s o rest
  [ "${SBX_RECORD_WRITTEN:-0}" = 1 ] && return 0
  SBX_RECORD_WRITTEN=1
  local mounts="[" first=1
  for m in "${SBX_MOUNTS[@]}"; do
    t=${m%%|*}; rest=${m#*|}; s=${rest%%|*}; o=${rest#*|}
    ((first)) || mounts+=","
    first=0; mounts+="{\"target\":$(sbx_js "$t"),\"source\":$(sbx_js "$s"),\"mode\":$(sbx_js "$o")}"
  done
  mounts+="]"
  local toolchain
  if [ "$SBX_BACKEND" = container ]; then
    toolchain="{\"image\":$(sbx_jnull "${SBX_IMAGE:-}"),\"imageId\":$(sbx_jnull "${SBX_IMAGE_ID:-}")}"
  else
    local jv="" javav=""
    if [ -n "${SBX_HOST_JOERN:-}" ]; then
      jv=$(ls "$SBX_HOST_JOERN/lib" 2>/dev/null | sed -n 's/^io\.joern\.console_3-\(.*\)\.jar$/\1/p' | head -n1)
    fi
    [ -n "${SBX_HOST_JDK:-}" ] && javav=$("$SBX_HOST_JDK/bin/java" -version 2>&1 | head -n1)
    toolchain="{\"joernVersion\":$(sbx_jnull "$jv"),\"joernPath\":$(sbx_jnull "${SBX_HOST_JOERN:-}"),\"javaVersion\":$(sbx_jnull "$javav")}"
  fi
  local script=null
  [ -n "${SBX_SCRIPT_PATH:-}" ] && script="{\"path\":$(sbx_js "$SBX_SCRIPT_PATH"),\"sha256\":$(sbx_js "$SBX_SCRIPT_SHA")}"
  local refused=false; [ -n "$reason" ] && refused=true
  local pinned=false; [ "${PRE_PINNED:-0}" = 1 ] && pinned=true
  local dur=$(( $(sbx_now_ms) - SBX_T0 ))
  local line
  line="{\"schemaVersion\":1,\"recordType\":\"sandbox\",\"auditId\":$(sbx_js "$SBX_AUDIT_ID"),\"ts\":$(sbx_js "$SBX_TS")"
  line+=",\"backend\":$(sbx_js "${SBX_BACKEND:-container}"),\"toolchain\":$toolchain,\"job\":\"parse\""
  line+=",\"egressMode\":$(sbx_js "${SBX_MODE:-offline}"),\"network\":\"none\",\"networkName\":null"
  line+=",\"launcherArgv\":$(sbx_jarr "${SBX_LAUNCHER[@]}"),\"toolArgv\":$(sbx_jarr "${SBX_TOOL_ARGV[@]}")"
  line+=",\"frontend\":$(sbx_js "${SBX_RECORD_FRONTEND:-}"),\"script\":$script"
  line+=",\"flagCheck\":{\"listDigest\":$(sbx_jnull "${SBX_LIST_DIGEST:-}"),\"forbiddenHits\":$(sbx_jarr "${SBX_HITS[@]}"),\"allowedHits\":$(sbx_jarr "${SBX_ALLOWED_HITS[@]}"),\"liftedHits\":[]}"
  line+=",\"envKeys\":$(sbx_jarr "${SBX_ENV_KEYS[@]}"),\"mounts\":$mounts"
  line+=",\"preconditions\":{\"interopPE\":$(sbx_js "${PRE_INTEROP:-unreadable}"),\"binfmtPinned\":$pinned,\"seccomp\":$(sbx_jnull "${PRE_SECCOMP:-}"),\"vsock\":$(sbx_js "${PRE_VSOCK:-absent}"),\"netnsDistinct\":${PRE_NETNS:-false}}"
  line+=",\"refused\":$refused,\"refusalReason\":$(sbx_jnull "$reason"),\"exitCode\":$rc"
  line+=",\"codebases\":$(sbx_jarr "${SBX_CODEBASES[@]}"),\"runId\":null,\"durationMs\":$dur}"
  sbx_append_record "$SBX_AUDIT_DIR/sandbox.jsonl" "$line"
}

sbx_init() { # per-invocation state
  export LC_ALL=C
  SBX_T0=$(sbx_now_ms)
  SBX_TS=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
  SBX_AUDIT_ID=$(sbx_uuid)
  SBX_AUDIT_DIR=${CCG_AUDIT_DIR:-$SBX_DIR/../.audit}
  SBX_RECORD_WRITTEN=0
  SBX_LAUNCHER=(); SBX_TOOL_ARGV=(); SBX_HITS=(); SBX_ALLOWED_HITS=(); SBX_ENV_KEYS=(); SBX_MOUNTS=()
  SBX_CALLER_ARGV=(); SBX_CODEBASES=(); SBX_CLEANUP=(); SBX_SCRIPT_PATH=""; SBX_SCRIPT_SHA=""; SBX_RECORD_FRONTEND=""
  SBX_BACKEND=""; SBX_MODE=offline; SBX_LIST_DIGEST=""
  SBX_BACKEND_REFUSAL=""; SBX_REFUSAL_DETAIL=""; SBX_LAUNCH_SETUP_FAIL=""
  # Measured, read-only baseline, so even a usage refusal records real preconditions.
  sbx_interop_scan /proc/sys/fs/binfmt_misc "$(uname -r)"
  PRE_INTEROP=$SBX_INTEROP_STATE; PRE_PINNED=$SBX_PE_PINNED; PRE_SECCOMP=""; PRE_NETNS=false
  PRE_VSOCK=absent; sbx_vsock_present /dev/vsock && PRE_VSOCK=present
  SBX_SCRATCH=$(mktemp -d /tmp/ccg-sbx.XXXXXX)
  SBX_CLEANUP+=("$SBX_SCRATCH")
}
sbx_cleanup() {
  local d
  for d in "${SBX_CLEANUP[@]}"; do
    [ -n "$d" ] && [ -d "$d" ] || continue
    rmdir "$d/work/work" 2>/dev/null || true   # overlay leaves an empty root-owned work/work
    chmod -R u+w "$d" 2>/dev/null || true
    rm -rf "$d" 2>/dev/null || sbx_log "warning: could not remove $d"
  done
}

# sbx_launch <src> <out> <scriptsdir|""> <workdir|""> -- <tool> <args...>
# The one launch path. Production main() calls it after the argv audit; the escape-axis probe and
# the --help sweep (tests) call it directly, so they run exactly as a tool is launched.
sbx_launch() {
  local src=$1 out=$2 sdir=$3 wd=$4 rc
  shift 5
  if [ "$SBX_BACKEND" = container ]; then
    sbx_prepare_out "$out" || { SBX_LAUNCH_SETUP_FAIL="cannot create overlay scratch beside $out"; return 69; }
    sbx_container_argv "$src" "$out" "$sdir" "$wd" -- "$@"
  else
    sbx_namespace_argv "$src" "$out" "$sdir" "$wd" -- "$@"
  fi
  sbx_env_keys_from_argv
  # netnsDistinct holds by construction (--network none / --unshare-all); probe A1(c) verifies it.
  # It is claimed only once a launcher exists (review S08 m4).
  PRE_NETNS=true
  rc=0
  "${SBX_LAUNCHER[@]}" || rc=$?
  return "$rc"
}

# ------------------------------------------------------------------ main
sbx_usage() {
  sed -n '2,/^$/p' "$SBX_SELF" | sed 's/^# \{0,1\}//' >&2
}

sbx_refuse() { # <exitCode> <reason> <message>
  sbx_log "refused ($2): $3"
  SBX_TOOL_ARGV=("${SBX_CALLER_ARGV[@]}")
  sbx_write_record "$1" "$2"
  sbx_cleanup
  exit "$1"
}

SBX_PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
# Re-exec under env -i unless the environment is already exactly the allowed set (review S08 n1).
sbx_clean_env() {
  local home k ok=1
  home=$(getent passwd "$(id -u)" | cut -d: -f6)
  [ "${PATH:-}" = "$SBX_PATH" ] && [ "${HOME:-}" = "$home" ] || ok=0
  for k in $(compgen -e); do
    case "$k" in CCG_AUDIT_DIR|CCG_TOOLCHAIN_IMAGE|PATH|HOME|PWD|OLDPWD|SHLVL|_) ;; *) ok=0 ;; esac
  done
  ((ok)) && return 0
  local -a keep=()
  [ -n "${CCG_AUDIT_DIR+x}" ] && keep+=("CCG_AUDIT_DIR=$CCG_AUDIT_DIR")
  [ -n "${CCG_TOOLCHAIN_IMAGE+x}" ] && keep+=("CCG_TOOLCHAIN_IMAGE=$CCG_TOOLCHAIN_IMAGE")
  exec /usr/bin/env -i "PATH=$SBX_PATH" "HOME=$home" "${keep[@]}" /bin/bash -p "$SBX_SELF" "$@"
}

sbx_main() {
  set -uo pipefail
  sbx_clean_env "$@"
  PATH=$SBX_PATH
  case "${1:-}" in -h|--help) sbx_usage; exit 0 ;; esac
  sbx_init
  # toolArgv (review S08 m3; egress-spec2 §1.5): a launched record carries the argv as executed
  # (after @AUDIT_ID@ substitution and the runner's own --output); a refused record carries the
  # argv the caller passed after the first `--`, verbatim (sbx_refuse restores it).
  local _a _seen=0
  for _a in "$@"; do
    if ((_seen)); then SBX_CALLER_ARGV+=("$_a"); SBX_TOOL_ARGV+=("${_a//@AUDIT_ID@/$SBX_AUDIT_ID}"); elif [ "$_a" = -- ]; then _seen=1; fi
  done
  trap 'sbx_cleanup' EXIT
  local backend="" mode="" frontend="" src="" out="" script_file="" seen_dd=0 a
  local -a tool=()
  while (($#)); do
    a=$1
    case "$a" in
      --) shift; tool=("$@"); seen_dd=1; break ;;
      --backend|--mode|--frontend|--source|--output|--codebase|--script-file)
        if (($# < 2)); then sbx_refuse 64 usage "$a needs a value"; fi
        case "$a" in
          --backend) backend=$2 ;; --mode) mode=$2 ;; --frontend) frontend=$2 ;;
          --source) src=$2 ;; --output) out=$2 ;; --codebase) SBX_CODEBASES+=("$2") ;;
          --script-file) script_file=$2 ;;
        esac
        shift 2 ;;
      *) sbx_refuse 64 usage "unknown runner option '$a' (tool arguments go after --)" ;;
    esac
  done
  # @AUDIT_ID@ substitution, then the recorded toolArgv is what the tool receives.
  local -a targv=()
  for a in "${tool[@]}"; do targv+=("${a//@AUDIT_ID@/$SBX_AUDIT_ID}"); done
  SBX_TOOL_ARGV=("${targv[@]}")
  SBX_RECORD_FRONTEND=$frontend
  # egressMode takes the requested mode only once it is a valid one (review S08 m4).
  [[ $mode == offline || $mode == controlled ]] && SBX_MODE=$mode

  # Backend resolution first, read-only, so every record names a backend and its preconditions;
  # its refusal is applied only after the argv checks (64 takes precedence over 69).
  case "$backend" in
    container|namespace|auto) sbx_resolve_backend "$backend" ;;
    *) sbx_try_container || true ;;   # invalid/missing: refused below as usage; record measured values
  esac
  local backend_refusal=$SBX_BACKEND_REFUSAL backend_detail=$SBX_REFUSAL_DETAIL

  ((seen_dd)) || sbx_refuse 64 usage "missing '--' before the tool arguments"
  [[ $backend == container || $backend == namespace || $backend == auto ]] || sbx_refuse 64 usage "--backend must be container|namespace|auto"
  [[ $mode == offline || $mode == controlled ]] || sbx_refuse 64 usage "--mode must be offline|controlled"
  [ -n "$frontend" ] && [ -n "$src" ] && [ -n "$out" ] || sbx_refuse 64 usage "--frontend, --source and --output are required"
  sbx_load_lists || sbx_refuse 64 usage "$SBX_LIST_ERR"

  # §7.3.1 frontend set and mode.
  local f known=0
  for f in "${SBX_FRONTENDS[@]}"; do [ "$f" = "$frontend" ] && known=1; done
  ((known)) || sbx_refuse 64 unknown-frontend "frontend '$frontend' is not in the closed set"
  if [ "$frontend" = joern ] && [ "$mode" = controlled ]; then sbx_refuse 64 unknown-frontend "joern script mode is offline-only"; fi
  [ "$mode" = offline ] || sbx_refuse 64 mode-not-implemented "controlled mode is P2"
  if [ -n "$script_file" ] && [ "$frontend" != joern ]; then sbx_refuse 64 usage "--script-file is only for --frontend joern"; fi

  # Paths: must be expressible in Docker's --mount CSV and overlay's option string.
  [ -d "$src" ] || sbx_refuse 64 usage "--source '$src' is not a directory"
  mkdir -p "$out" 2>/dev/null || sbx_refuse 64 usage "cannot create --output '$out'"
  src=$(realpath -e "$src"); out=$(realpath -e "$out")
  for a in "$src" "$out" "$(dirname "$out")"; do
    [[ $a == *[,\":=$'\n']* ]] && sbx_refuse 64 usage "path '$a' contains one of , \" : = or a newline"
  done
  [ "$src" != "$out" ] || sbx_refuse 64 usage "--source and --output must differ"

  local -a fes=("$frontend")
  local sdir="" wd=""
  if [ "$frontend" = joern ]; then
    # Script staging: a runner-owned copy is checked, hashed and bound read-only at /scripts.
    if [ -z "$script_file" ]; then SBX_HITS+=("(missing --script-file)"); sbx_refuse 64 forbidden-flag "--frontend joern needs --script-file"; fi
    local base; base=$(basename "$script_file")
    [[ $base =~ ^[A-Za-z0-9][A-Za-z0-9._-]*\.sc$ ]] && [ -f "$script_file" ] || sbx_refuse 64 usage "--script-file must be an existing <name>.sc file"
    sdir="$SBX_SCRATCH/scripts"
    mkdir -p "$sdir"; cp -- "$script_file" "$sdir/$base"; chmod 0444 "$sdir/$base"; chmod 0555 "$sdir"
    SBX_SCRIPT_PATH="/scripts/$base"
    SBX_SCRIPT_SHA=$(sha256sum < "$sdir/$base" | cut -d' ' -f1)
    sbx_joern_grammar "$base" -- "${SBX_TOOL_ARGV[@]}"; local gram=$?
    local dlines directive=0
    if ! dlines=$(sbx_directive_lines "$sdir/$base"); then
      directive=1
      while IFS= read -r a; do SBX_HITS+=("$a"); done <<< "$dlines"
    fi
    sbx_audit_elements "$mode" "joern" -- "${SBX_TOOL_ARGV[@]}"
    wd=/tmp
    if ((gram != 0)) || { ((!directive)) && ((${#SBX_HITS[@]})); }; then
      sbx_refuse 64 forbidden-flag "joern argv outside the --script /scripts/$base [--param k=v]... grammar, or a forbidden element"
    fi
    ((directive)) && sbx_refuse 64 forbidden-directive "script carries a //> directive line"
  else
    # Runner-owned output flag.
    for a in "${SBX_TOOL_ARGV[@]}"; do
      if [[ $a == -o || $a == --output || $a == --output=* ]]; then SBX_HITS+=("$a"); fi
    done
    ((${#SBX_HITS[@]})) && sbx_refuse 64 forbidden-flag "the runner owns --output (/out/<auditId>.cpg.bin)"
    if [ "$frontend" = joern-parse ]; then
      local i lang="" nlang=0 n=${#SBX_TOOL_ARGV[@]}
      for ((i = 0; i < n; i++)); do
        a=${SBX_TOOL_ARGV[i]}
        if [ "$a" = --language ]; then nlang=$((nlang + 1)); lang=${SBX_TOOL_ARGV[i+1]:-}
        elif [[ $a == --language=* ]]; then nlang=$((nlang + 1)); lang=${a#--language=}; fi
      done
      ((nlang == 1)) || sbx_refuse 64 unknown-frontend "joern-parse needs exactly one --language (got $nlang)"
      local derived
      derived=$(sbx_language_frontend "${lang,,}") || sbx_refuse 64 unknown-frontend "joern-parse --language '$lang' is not an FR-1 language"
      fes=("$derived" joern-parse)
      SBX_RECORD_FRONTEND=$derived
    fi
    SBX_TOOL_ARGV+=(--output "/out/$SBX_AUDIT_ID.cpg.bin")
    sbx_audit_elements "$mode" "${fes[*]}" -- "${SBX_TOOL_ARGV[@]}"
    ((${#SBX_HITS[@]})) && sbx_refuse 64 forbidden-flag "forbidden flag or launcher-level option"
  fi

  [ -z "$backend_refusal" ] || { SBX_BACKEND_REFUSAL=$backend_refusal; sbx_refuse 69 "$backend_refusal" "$SBX_BACKEND backend: $backend_detail"; }

  # One run per --output (review S08 m2): overlayfs only warns about a shared upperdir, and the
  # result is undefined. The lock is flock(2) on the directory's own fd, so it leaves no file.
  exec {SBX_LOCK_FD}<"$out"
  flock -n "$SBX_LOCK_FD" || sbx_refuse 64 usage "--output '$out' is in use by another run"

  local rc=0
  sbx_launch "$src" "$out" "$sdir" "$wd" -- "$SBX_JOERN_IMG_HOME/$(sbx_frontend_exe "$frontend")" "${SBX_TOOL_ARGV[@]}" || rc=$?
  if [ -n "${SBX_LAUNCH_SETUP_FAIL:-}" ]; then
    SBX_LAUNCHER=(); SBX_ENV_KEYS=(); SBX_MOUNTS=()
    sbx_refuse 69 backend-unavailable "$SBX_LAUNCH_SETUP_FAIL"
  fi
  sbx_write_record "$rc" ""
  sbx_cleanup
  trap - EXIT
  exit "$rc"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  sbx_main "$@"
fi
