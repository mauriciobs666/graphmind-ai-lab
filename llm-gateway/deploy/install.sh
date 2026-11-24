#!/bin/bash
# install.sh — the human's root-run installer for the key holder (docs/plans/llm-gateway.md §3.4).
#
# Read this whole file before running it with sudo. It runs ONLY from the root-owned code root
# (/opt/llmgateway/deploy/install.sh): it refuses to run from anywhere else, and it refuses when
# `keyholder check-deployment --code-only` finds the code root or an ancestor writable by a
# non-root uid, or any entry that is not a regular file or directory. It runs no git, no ccg code
# and no repo Python as root; the code root arrives by the bootstrap pipeline in the plan.
#
#   install.sh [--dry-run] <subcommand> ...
#     holder-account [--key-from-stdin PROVIDER]  account llmholder, /var/lib/llmholder, the unit
#     owner-files --gateway-toml FILE --manifest FILE   byte copies into /etc/llmgateway (diff + confirm)
#     caller NAME CALLER_UID OUT [--rotate]        registry secret + caller-side credential file
#     service-unit CALLER_UID HOME                 the service's systemd --user unit
#     check [--caller-cred UID:PATH]...            keyholder check-deployment on the real layout
#
# Test mode (unprivileged only): `--test-root DIR [--test-home DIR]` runs the same steps under DIR
# as the invoking uid (no useradd, no chown, no systemctl). It is refused when run as root, so it
# cannot be used to weaken a real install.

set -euo pipefail
umask 077
PATH=/usr/sbin:/usr/bin:/sbin:/bin
export PATH LC_ALL=C
unset BASH_ENV ENV CDPATH IFS

DRY=0
TEST_ROOT=""
TEST_HOME=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1; shift ;;
    --test-root) TEST_ROOT="${2:?--test-root needs a directory}"; shift 2 ;;
    --test-home) TEST_HOME="${2:?--test-home needs a directory}"; shift 2 ;;
    *) break ;;
  esac
done

die() { echo "install.sh: $*" >&2; exit 1; }
usage() { sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
run() {
  if [ "$DRY" = 1 ]; then
    printf 'DRY-RUN:'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

[ $# -gt 0 ] || usage
SUB="$1"; shift
case "$SUB" in holder-account|owner-files|caller|service-unit|check) ;; *) usage ;; esac

if [ -n "$TEST_ROOT" ]; then
  [ "$(id -u)" -ne 0 ] || die "--test-root is refused when run as root"
  [ -d "$TEST_ROOT" ] || die "--test-root: no such directory"
  TRUSTED_UID="$(id -u)"
  HOLDER_UID="$TRUSTED_UID"
  OWN_ROOT=(); OWN_HOLDER=()
else
  [ -z "$TEST_HOME" ] || die "--test-home needs --test-root"
  [ "$(id -u)" -eq 0 ] || die "must be run as root (sudo)"
  TRUSTED_UID=0
  OWN_ROOT=(-o root -g root); OWN_HOLDER=(-o llmholder -g llmholder)
fi
P="$TEST_ROOT"
CODE_ROOT="$P/opt/llmgateway"
STATE_ROOT="$P/var/lib/llmholder"
ETC_DIR="$P/etc/llmgateway"
UNIT_PATH="$P/etc/systemd/system/llmholder.service"
SOCKET_PATH="$P/run/llmholder/holder.sock"
PY=/usr/bin/python3

kh() { "$PY" -I "$CODE_ROOT/bin/keyholder.py" "$@"; }

# --- the code-root guard: this exact file, installed, and nothing writable around it ---------------
# Pure shell, run BEFORE any Python from the code root: if the tree was extracted wrongly (owned by
# or writable for another uid) `keyholder check-deployment` would otherwise execute that code as root.
precheck_code_root() {
  local top="/" bad p stat_out
  [ -z "$TEST_ROOT" ] || top="$TEST_ROOT"
  [ -d "$CODE_ROOT" ] && [ ! -L "$CODE_ROOT" ] || die "$CODE_ROOT is missing or a symlink"
  bad="$(find "$CODE_ROOT" \( ! -uid "$TRUSTED_UID" -o -perm /022 -o \( ! -type f ! -type d \) \) -print -quit)"
  [ -z "$bad" ] || die "the code root is not protected: $(printf '%q' "$bad") is not a regular file or directory owned by uid $TRUSTED_UID and not group/other-writable"
  p="$CODE_ROOT"
  while :; do
    stat_out="$(stat -L -c '%u %a' -- "$p")" || die "cannot stat $p"
    # shellcheck disable=SC2086
    set -- $stat_out
    [ "$1" = "$TRUSTED_UID" ] && [ $((8#$2 & 8#022)) -eq 0 ] ||
      die "the code root is not protected: ancestor $(printf '%q' "$p") is not owned by uid $TRUSTED_UID or is group/other-writable"
    [ "$p" != "$top" ] && [ "$p" != "/" ] || break
    p="$(dirname -- "$p")"
  done
}

guard_code_root() {
  local here self
  [ ! -L "$0" ] || die "refusing to run through a symlink"
  here="$(cd -- "$(dirname -- "$0")" && pwd -P)"
  self="$here/$(basename -- "$0")"
  [ "$self" = "$CODE_ROOT/deploy/install.sh" ] ||
    die "not running from the installed code root ($CODE_ROOT/deploy/install.sh); got $self"
  [ -f "$CODE_ROOT/bin/keyholder.py" ] && [ ! -L "$CODE_ROOT/bin/keyholder.py" ] ||
    die "$CODE_ROOT/bin/keyholder.py is missing or a symlink"
  precheck_code_root
  local args=(check-deployment --code-only --root "$CODE_ROOT")
  [ -z "$TEST_ROOT" ] || args+=(--anchor "$TEST_ROOT" --trusted-uid "$TRUSTED_UID")
  local out
  if ! out="$(kh "${args[@]}" 2>&1)"; then
    echo "$out" >&2
    die "the code root is not protected (writable path, ancestor, or non-regular entry)"
  fi
}

valid_name() { [[ "$1" =~ ^[a-z0-9_-]{1,32}$ ]]; }
valid_uid() { [[ "$1" =~ ^[0-9]{1,10}$ ]]; }

# --- holder-account ----------------------------------------------------------------------------------
cmd_holder_account() {
  local key_provider=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --key-from-stdin) key_provider="${2:-}"; valid_name "$key_provider" || die "provider name must match [a-z0-9_-]{1,32}"; shift 2 ;;
      *) usage ;;
    esac
  done
  guard_code_root
  if [ -z "$TEST_ROOT" ]; then
    id llmholder >/dev/null 2>&1 ||
      run useradd --system --user-group --no-create-home --home-dir /var/lib/llmholder \
        --shell /usr/sbin/nologin llmholder
  fi
  run install -d -m 0700 "${OWN_HOLDER[@]}" "$STATE_ROOT" "$STATE_ROOT/keys" "$STATE_ROOT/registry"
  run install -d -m 0755 "${OWN_ROOT[@]}" "$(dirname "$UNIT_PATH")"
  write_unit
  if [ -n "$key_provider" ]; then
    local key dest="$STATE_ROOT/keys/$key_provider.key"
    if [ "$DRY" = 1 ]; then
      echo "DRY-RUN: would read the key from stdin into $dest (0600 llmholder)"
    else
      # Builtins only: the key is never an argument of any process, never echoed.
      IFS= read -r -N 4097 key || true
      key="${key%$'\n'}"
      [ -n "$key" ] && [ "${#key}" -le 4096 ] && [[ "$key" =~ ^[[:graph:]]+$ ]] ||
        die "the key on stdin must be one non-empty printable token of at most 4096 characters"
      printf '%s\n' "$key" | install -m 0600 "${OWN_HOLDER[@]}" /dev/stdin "$dest"
      key=""
      echo "installed key for $key_provider"
    fi
  fi
  [ -n "$TEST_ROOT" ] || run systemctl daemon-reload
  echo "next: install.sh owner-files ...; install.sh caller broker <uid> <path>; systemctl enable --now llmholder"
}

write_unit() {
  local tmp
  tmp="$(mktemp)"
  cat >"$tmp" <<'UNIT'
[Unit]
Description=LLM gateway key holder
After=network-online.target

[Service]
User=llmholder
RuntimeDirectory=llmholder
RuntimeDirectoryMode=0711
ExecStart=/usr/bin/python3 -I /opt/llmgateway/bin/keyholder.py
Restart=on-failure
NoNewPrivileges=yes
ProtectSystem=strict
ReadWritePaths=/var/lib/llmholder
PrivateTmp=yes
LimitCORE=0
ProtectProc=invisible
RestrictAddressFamilies=AF_UNIX AF_INET AF_INET6
CapabilityBoundingSet=

[Install]
WantedBy=multi-user.target
UNIT
  run install -m 0644 "${OWN_ROOT[@]}" "$tmp" "$UNIT_PATH"
  rm -f "$tmp"
}

# --- owner-files -------------------------------------------------------------------------------------
confirm() {
  local answer
  read -r -p "$1 [y/N] " answer || answer=""
  [ "$answer" = "y" ] || [ "$answer" = "yes" ]
}

cmd_owner_files() {
  local toml="" manifest=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --gateway-toml) toml="${2:?}"; shift 2 ;;
      --manifest) manifest="${2:?}"; shift 2 ;;
      *) usage ;;
    esac
  done
  [ -n "$toml" ] && [ -n "$manifest" ] || usage
  guard_code_root
  WORK="$(mktemp -d)"
  trap 'rm -rf "$WORK"' EXIT
  copy_confirm_install "$toml" "$WORK/gateway.toml" "$ETC_DIR/gateway.toml"
  copy_confirm_install "$manifest" "$WORK/system.yaml" "$ETC_DIR/ccg/system.yaml"
}

MAX_OWNER_FILE_BYTES=65536

# Read the source as the invoking (human) user: root never opens an agent-chosen path, so a swapped
# path or a symlink cannot make root read a file the human could not. At most MAX+1 bytes are read.
read_source() {
  if [ -n "$TEST_ROOT" ]; then
    head -c $((MAX_OWNER_FILE_BYTES + 1)) -- "$1"
  else
    [[ "${SUDO_USER:-}" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]] && [ "$SUDO_USER" != root ] ||
      die "run through sudo from the human's own account (SUDO_USER is unset or root)"
    runuser -u "$SUDO_USER" -- head -c $((MAX_OWNER_FILE_BYTES + 1)) -- "$1"
  fi
}

# Read the source ONCE into a root-owned scratch file; show the WHOLE file (control bytes made
# visible by `cat -v`) with its sha256, and the diff against what is installed; after the human
# confirms, install exactly those bytes (the hash is checked again). No ccg code, no validation as root.
copy_confirm_install() {
  local src="$1" scratch="$2" dest="$3" sha sha_now qsrc
  qsrc="$(printf '%q' "$src")"
  [ -f "$src" ] && [ ! -L "$src" ] || die "$qsrc must be a regular file, not a symlink"
  read_source "$src" >"$scratch" || die "cannot read $qsrc as the invoking user"
  [ "$(stat -c %s -- "$scratch")" -le "$MAX_OWNER_FILE_BYTES" ] ||
    die "$qsrc is larger than $MAX_OWNER_FILE_BYTES bytes; refusing"
  sha="$(sha256sum <"$scratch")"; sha="${sha%% *}"
  if [ -e "$dest" ]; then
    echo "--- diff of $dest (control bytes shown as ^X / M-x) ---"
    diff -u <(cat -v -- "$dest") <(cat -v -- "$scratch") || true
  else
    echo "(new file) $dest"
  fi
  echo "--- whole file as it will be installed (sha256 $sha) ---"
  cat -v -- "$scratch"
  echo "--- end (sha256 $sha) ---"
  if [ "$DRY" = 1 ]; then echo "DRY-RUN: would ask, then install $dest"; return 0; fi
  confirm "Install $dest from $qsrc?" || die "not confirmed; nothing installed for $dest"
  sha_now="$(sha256sum <"$scratch")"
  [ "${sha_now%% *}" = "$sha" ] || die "the bytes changed after they were shown; nothing installed"
  install -d -m 0755 "${OWN_ROOT[@]}" "$(dirname "$dest")"
  install -m 0644 "${OWN_ROOT[@]}" "$scratch" "$dest"
}

# --- caller / service-unit ---------------------------------------------------------------------------
cmd_caller() {
  [ $# -ge 3 ] || usage
  local name="$1" uid="$2" out="$3" rotate=()
  shift 3
  while [ $# -gt 0 ]; do
    case "$1" in --rotate) rotate=(--rotate); shift ;; *) usage ;; esac
  done
  valid_name "$name" || die "caller name must match [a-z0-9_-]{1,32}"
  valid_uid "$uid" || die "caller uid must be a number"
  [[ "$out" = /* ]] || die "OUT must be an absolute path"
  guard_code_root
  local extra=()
  if [ -n "$TEST_ROOT" ]; then
    extra=(--holder-uid "$HOLDER_UID" --state-root "$STATE_ROOT")
    [ -z "$TEST_HOME" ] || extra+=(--home "$TEST_HOME")
  fi
  run "$PY" -I "$CODE_ROOT/bin/keyholder.py" add-caller "$name" --caller-uid "$uid" --out "$out" \
    "${rotate[@]}" "${extra[@]}"
}

cmd_service_unit() {
  [ $# -eq 2 ] || usage
  local uid="$1" home="$2"
  valid_uid "$uid" || die "caller uid must be a number"
  [[ "$home" = /* ]] || die "HOME must be an absolute path"
  guard_code_root
  [ -f "$CODE_ROOT/bin/service.py" ] && [ ! -L "$CODE_ROOT/bin/service.py" ] ||
    die "$CODE_ROOT/bin/service.py does not exist yet (the service ships with a later release); refusing to write a unit that would crash-loop"
  local out="$home/.config/systemd/user/llmgateway.service" extra=()
  [ -z "$TEST_ROOT" ] || extra=(--home "${TEST_HOME:-$home}")
  if [ "$DRY" = 1 ]; then
    echo "DRY-RUN: would write $out as uid $uid"
    return 0
  fi
  kh install-user-file --caller-uid "$uid" --out "$out" "${extra[@]}" <<'UNIT'
[Unit]
Description=LLM gateway service

[Service]
ExecStart=/usr/bin/python3 -I /opt/llmgateway/bin/service.py
Restart=on-failure

[Install]
WantedBy=default.target
UNIT
  echo "unattended runs need: loginctl enable-linger <user>"
}

cmd_check() {
  guard_code_root
  local args=(check-deployment --root "$CODE_ROOT" --etc-dir "$ETC_DIR" --unit "$UNIT_PATH"
    --state-root "$STATE_ROOT" --socket "$SOCKET_PATH")
  [ -z "$TEST_ROOT" ] || args+=(--anchor "$TEST_ROOT" --trusted-uid "$TRUSTED_UID" --holder-uid "$HOLDER_UID")
  kh "${args[@]}" "$@"
}

case "$SUB" in
  holder-account) cmd_holder_account "$@" ;;
  owner-files) cmd_owner_files "$@" ;;
  caller) cmd_caller "$@" ;;
  service-unit) cmd_service_unit "$@" ;;
  check) cmd_check "$@" ;;
esac
