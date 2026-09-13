#!/usr/bin/env bash
# wrapper-scripts.sh — deterministic logic tests for compose-status.sh /
# compose-up.sh / compose-down.sh, no OpenCode/LM Studio/Docker daemon
# needed. Covers exactly plan §5's "Wrapper-script logic checks" bullet
# group (opencode/docs/plans/devops-opencode-headless.md):
#
#   - exactly-one-argument enforcement (0 args, 2+ args both refuse before
#     any docker call)
#   - exact-match lookup (known slug resolves; unknown/garbage/traversal
#     slugs all refuse identically, none reach `docker compose`)
#   - compose-down.sh never reads composeFile/projectDirectory from the
#     marker (grep-style regression check)
#   - resolve_slug is true key equality, never substring/prefix/glob
#     (security review Pass 2 nit, opencode/docs/reviews/
#     devops-opencode-headless.md, "New observations this pass")
#
# Runs the REAL, byte-identical shipped scripts, copied into an isolated
# fixture tree under a temp dir so nothing here ever touches this repo's own
# git-tracked opencode/agents/tank/{environments.json,state/} — a fake
# `docker` stub on PATH logs every invocation it receives instead of
# touching a real docker daemon, so "a known slug resolves" is verified by
# observing the exact resolved -f/--project-directory values reaching the
# logged docker-compose argv, not by actually running compose.
#
# Usage: ./wrapper-scripts.sh
set -uo pipefail  # deliberately not -e: assertions must keep running after a failure

REAL_SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../scripts" && pwd)"

PASS=0
FAIL=0
FAILED_NAMES=()

if [ -t 1 ]; then
  R=$'\033[31m'; G=$'\033[32m'; Z=$'\033[0m'
else
  R=; G=; Z=
fi

# ---------------------------------------------------------------------------
# Fixture repo: opencode/agents/tank/{environments.json,scripts/*,state/}
# four levels below a fake repo root, mirroring the real layout so
# lib.sh's REPO_ROOT resolution (four `../` hops) works unmodified.
# ---------------------------------------------------------------------------
FIXTURE_ROOT="$(mktemp -d)"
FIXTURE_TANK_DIR="$FIXTURE_ROOT/opencode/agents/tank"
FIXTURE_SCRIPTS_DIR="$FIXTURE_TANK_DIR/scripts"
mkdir -p "$FIXTURE_SCRIPTS_DIR" "$FIXTURE_TANK_DIR/state" "$FIXTURE_ROOT/falkor-chat"
cp "$REAL_SCRIPTS_DIR"/*.sh "$FIXTURE_SCRIPTS_DIR/"
touch "$FIXTURE_ROOT/falkor-chat/compose.yaml"

cat >"$FIXTURE_TANK_DIR/environments.json" <<'JSON'
{
  "falkor-chat": {
    "composeFile": "falkor-chat/compose.yaml",
    "projectDirectory": "falkor-chat",
    "label": "falkor-chat dev stack (FalkorDB + M1 server)"
  }
}
JSON

# Fake `docker` on PATH: logs its full argv, never touches a real daemon.
# `ps --status running -q` prints nothing (simulates "cold" so compose-up.sh
# proceeds to the up call under test).
FAKE_BIN_DIR="$(mktemp -d)"
DOCKER_LOG="$(mktemp)"
cat >"$FAKE_BIN_DIR/docker" <<EOF
#!/usr/bin/env bash
printf '%s\n' "\$*" >> "$DOCKER_LOG"
case "\$*" in
  *"ps --status running -q") exit 0 ;;  # empty stdout = nothing running
  *) exit 0 ;;
esac
EOF
chmod +x "$FAKE_BIN_DIR/docker"

cleanup() { rm -rf "$FIXTURE_ROOT" "$FAKE_BIN_DIR" "$DOCKER_LOG"; }
trap cleanup EXIT

run_script() {
  # run_script SCRIPT_NAME ARG...  -- runs the fixture copy with the fake
  # docker on PATH; sets STDOUT/STDERR/EXIT_CODE.
  local script="$1"
  shift
  local out err code
  out="$(mktemp)"; err="$(mktemp)"
  PATH="$FAKE_BIN_DIR:$PATH" "$FIXTURE_SCRIPTS_DIR/$script" "$@" >"$out" 2>"$err"
  code=$?
  STDOUT="$(cat "$out")"
  STDERR="$(cat "$err")"
  EXIT_CODE=$code
  rm -f "$out" "$err"
}

reset_state() {
  rm -f "$FIXTURE_TANK_DIR"/state/*.json
  : >"$DOCKER_LOG"
}

pass() { PASS=$((PASS + 1)); printf '%sPASS%s %s\n' "$G" "$Z" "$1"; }
fail() { FAIL=$((FAIL + 1)); FAILED_NAMES+=("$1"); printf '%sFAIL%s %s\n  %s\n' "$R" "$Z" "$1" "$2"; }

assert_refused_no_docker() {
  # assert_refused_no_docker NAME SCRIPT ARG...
  local name="$1" script="$2"
  shift 2
  reset_state
  run_script "$script" "$@"
  if [ "$EXIT_CODE" -eq 0 ]; then
    fail "$name" "expected non-zero exit, got 0 (stdout: $STDOUT)"
    return
  fi
  if [ -s "$DOCKER_LOG" ]; then
    fail "$name" "expected no docker invocation, but got: $(cat "$DOCKER_LOG")"
    return
  fi
  pass "$name"
}

assert_refused_no_docker_msg() {
  # Same as above, plus the refusal message must contain a substring
  # (pins WHICH gate refused it — argc vs lookup — not just that something did).
  local name="$1" script="$2" expect_substr="$3"
  shift 3
  reset_state
  run_script "$script" "$@"
  if [ "$EXIT_CODE" -eq 0 ]; then
    fail "$name" "expected non-zero exit, got 0 (stdout: $STDOUT)"
    return
  fi
  if [ -s "$DOCKER_LOG" ]; then
    fail "$name" "expected no docker invocation, but got: $(cat "$DOCKER_LOG")"
    return
  fi
  case "$STDERR" in
    *"$expect_substr"*) pass "$name" ;;
    *) fail "$name" "expected stderr to contain '$expect_substr', got: $STDERR" ;;
  esac
}

# ===========================================================================
# 1. Exactly-one-argument enforcement — 0 args and 2+ args (unquoted
#    flag-smuggling shape) both refuse before any docker call.
# ===========================================================================
for script in compose-status.sh compose-up.sh compose-down.sh; do
  assert_refused_no_docker_msg \
    "$script: 0 args refuses before any docker call" \
    "$script" "got 0"

  # The unquoted-flag-smuggling shape: a slug followed by a separately-
  # tokenized -f/--project-directory pair, exactly what the security review
  # (Pass 2, case 2) used to probe the wrapper-script boundary itself.
  assert_refused_no_docker_msg \
    "$script: 2+ unquoted args (flag-smuggling shape) refuses before any docker call" \
    "$script" "got 5" \
    falkor-chat -f /tmp/victim/compose.yaml --project-directory /tmp/victim

  # Exactly 2 args, first one a genuinely known slug — pins the boundary at
  # precisely 1, not "1 or a careless off-by-one like 2". A weakened argc
  # check (e.g. accepting 1-or-2 args) would let this reach a real docker
  # invocation while every other 2+-arg test above (5 tokens) still refuses,
  # so this case is the one that actually catches that regression.
  assert_refused_no_docker_msg \
    "$script: exactly 2 args (known slug + one extra token) refuses before any docker call" \
    "$script" "got 2" \
    falkor-chat extra-token
done

# ===========================================================================
# 2. Exact-match lookup — unknown slug, trailing-garbage-as-one-argument,
#    and a ../-shaped value are all refused identically; none reach docker.
# ===========================================================================
for script in compose-status.sh compose-up.sh compose-down.sh; do
  assert_refused_no_docker \
    "$script: unknown slug refuses, no docker call" \
    "$script" "not-a-real-environment"

  assert_refused_no_docker \
    "$script: trailing garbage as one quoted argument refuses, no docker call" \
    "$script" "falkor-chat -f /tmp/victim/compose.yaml --project-directory /tmp/victim"

  assert_refused_no_docker \
    "$script: ../-shaped slug refuses, no docker call" \
    "$script" "../../../etc/passwd"
done

# ===========================================================================
# 3. A known slug resolves — the correct, repo-root-joined composeFile/
#    projectDirectory values (and only those) reach the logged docker
#    invocation, for all three scripts.
# ===========================================================================
reset_state
run_script compose-status.sh falkor-chat
if [ "$EXIT_CODE" -ne 0 ]; then
  fail "compose-status.sh: known slug resolves and succeeds" "expected exit 0, got $EXIT_CODE (stderr: $STDERR)"
elif ! grep -qF -- "-f $FIXTURE_ROOT/falkor-chat/compose.yaml --project-directory $FIXTURE_ROOT/falkor-chat ps" "$DOCKER_LOG"; then
  fail "compose-status.sh: known slug resolves and succeeds" "docker log missing expected ps invocation: $(cat "$DOCKER_LOG")"
else
  pass "compose-status.sh: known slug resolves and succeeds"
fi

reset_state
run_script compose-up.sh falkor-chat
if [ "$EXIT_CODE" -ne 0 ]; then
  fail "compose-up.sh: known slug resolves, brings up, writes marker" "expected exit 0, got $EXIT_CODE (stderr: $STDERR)"
elif ! grep -qF -- "-f $FIXTURE_ROOT/falkor-chat/compose.yaml --project-directory $FIXTURE_ROOT/falkor-chat up -d --build" "$DOCKER_LOG"; then
  fail "compose-up.sh: known slug resolves, brings up, writes marker" "docker log missing expected up invocation: $(cat "$DOCKER_LOG")"
elif [ ! -f "$FIXTURE_TANK_DIR/state/falkor-chat.json" ]; then
  fail "compose-up.sh: known slug resolves, brings up, writes marker" "marker file was not written"
elif grep -q 'composeFile\|projectDirectory' "$FIXTURE_TANK_DIR/state/falkor-chat.json"; then
  fail "compose-up.sh: known slug resolves, brings up, writes marker" "marker unexpectedly contains a compose-path field: $(cat "$FIXTURE_TANK_DIR/state/falkor-chat.json")"
else
  pass "compose-up.sh: known slug resolves, brings up, writes marker"
fi

# compose-down.sh needs a fresh marker present first (its own ownership
# gate, orthogonal to the lookup this suite is about) — seed one directly,
# the same shape compose-up.sh itself would have written.
reset_state
FRESH_TIMESTAMP="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
python3 -c '
import json
json.dump(
    {"slug": "falkor-chat", "broughtUpAt": "'"$FRESH_TIMESTAMP"'", "sessionId": "test", "pid": 1, "host": "test"},
    open("'"$FIXTURE_TANK_DIR"'/state/falkor-chat.json", "w"),
)
'
run_script compose-down.sh falkor-chat
if [ "$EXIT_CODE" -ne 0 ]; then
  fail "compose-down.sh: known slug resolves and tears down" "expected exit 0, got $EXIT_CODE (stderr: $STDERR)"
elif ! grep -qF -- "-f $FIXTURE_ROOT/falkor-chat/compose.yaml --project-directory $FIXTURE_ROOT/falkor-chat down" "$DOCKER_LOG"; then
  fail "compose-down.sh: known slug resolves and tears down" "docker log missing expected down invocation: $(cat "$DOCKER_LOG")"
elif [ -f "$FIXTURE_TANK_DIR/state/falkor-chat.json" ]; then
  fail "compose-down.sh: known slug resolves and tears down" "marker was not removed after successful teardown"
else
  pass "compose-down.sh: known slug resolves and tears down"
fi

# ===========================================================================
# 4. compose-down.sh never reads composeFile/projectDirectory from the
#    marker — grep-style regression check on the shipped source itself.
# ===========================================================================
DOWN_SRC="$FIXTURE_SCRIPTS_DIR/compose-down.sh"
if grep -q 'composeFile\|projectDirectory' "$DOWN_SRC"; then
  fail "compose-down.sh: never reads composeFile/projectDirectory from the marker" \
    "found a forbidden key-name literal in compose-down.sh: $(grep -n 'composeFile\|projectDirectory' "$DOWN_SRC")"
else
  pass "compose-down.sh: never reads composeFile/projectDirectory from the marker"
fi

# ===========================================================================
# 5. resolve_slug is true key equality — never substring/prefix/glob.
#    (Pass-2 review nit — see reviews/devops-opencode-headless.md "New
#    observations this pass".) Exercises the real function directly, by
#    sourcing lib.sh, not through a whole script invocation.
# ===========================================================================
test_resolve_slug_is_exact_key_equality_not_substring_or_prefix() {
  local name="resolve_slug: exact key equality, not substring/prefix/glob"
  # shellcheck source=/dev/null
  ( set -uo pipefail
    source "$FIXTURE_SCRIPTS_DIR/lib.sh"

    # 1. The real key resolves.
    if ! resolve_slug "falkor-chat" "$FIXTURE_TANK_DIR/environments.json" >/dev/null; then
      echo "expected 'falkor-chat' (the real key) to resolve" >&2
      exit 1
    fi

    # 2. A PREFIX of the real key must NOT resolve — a substring/prefix
    #    matcher (e.g. [[ "$key" == "$slug"* ]] or a bash `case "$slug" in
    #    "$key"*)`) would wrongly accept this.
    if resolve_slug "falkor" "$FIXTURE_TANK_DIR/environments.json" >/dev/null 2>&1; then
      echo "'falkor' (a prefix of the real key) must NOT resolve" >&2
      exit 1
    fi

    # 3. The real key PLUS trailing garbage must NOT resolve — a substring
    #    matcher anchored only at the start would wrongly accept this too
    #    (this is exactly the Pass-2 "quoted single-argument smuggle" shape).
    if resolve_slug "falkor-chat-and-more" "$FIXTURE_TANK_DIR/environments.json" >/dev/null 2>&1; then
      echo "'falkor-chat-and-more' (real key + trailing garbage) must NOT resolve" >&2
      exit 1
    fi

    # 4. A grep-style substring match of the real key inside a longer
    #    unrelated string must NOT resolve.
    if resolve_slug "xxfalkor-chatxx" "$FIXTURE_TANK_DIR/environments.json" >/dev/null 2>&1; then
      echo "'xxfalkor-chatxx' (real key as a mid-string substring) must NOT resolve" >&2
      exit 1
    fi
  )
  if [ $? -eq 0 ]; then pass "$name"; else fail "$name" "see stderr above"; fi
}
test_resolve_slug_is_exact_key_equality_not_substring_or_prefix

# ===========================================================================
# Summary
# ===========================================================================
echo
echo "----------------------------------------"
printf 'PASS: %d  FAIL: %d\n' "$PASS" "$FAIL"
if [ "$FAIL" -gt 0 ]; then
  echo "Failed:"
  for n in "${FAILED_NAMES[@]}"; do echo "  - $n"; done
  exit 1
fi
exit 0
