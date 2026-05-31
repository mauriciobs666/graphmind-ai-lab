#!/usr/bin/env bash
#
# run.sh — agent eval harness (kaizen loop)
#
# Runs each test case under cases/ through `opencode run --agent <AGENT>`,
# captures the agent's answer as markdown in outputs/, and diffs it against a
# blessed reference in baseline/. Output is non-deterministic (local LLM), so
# this is a review aid, not a pass/fail gate: you read the diffs to spot
# regressions or improvements, then `--bless` to accept a new baseline.
#
# A case may also carry an optional expect.md with deterministic substring
# assertions, checked against the response body:
#   require: <text>    response MUST contain <text>   (case-insensitive, literal)
#   reject:  <text>    response must NOT contain <text>
# Blank lines and '#' comments are ignored. Diffs stay advisory, but a failed
# assertion makes the whole run exit non-zero (the deterministic gate).
#
# Usage:
#   ./run.sh                 run all cases, diff against baseline
#   ./run.sh 01-explain-python [02-...]   run only the named case(s)
#   ./run.sh --bless [case...]            promote outputs/ -> baseline/ (all, or named)
#   ./run.sh --list                       list discovered cases
#   ./run.sh -h | --help                  this help
#
# Env overrides:
#   AGENT     agent name           (default: name of the parent directory)
#   MODEL     provider/model       (default: agent's configured model)
#   ENDPOINT  LM Studio /v1 URL     (default: http://localhost:1234/v1)

set -euo pipefail

# --- locations -------------------------------------------------------------
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$TESTS_DIR")"   # the agent config dir (parent of tests/, has opencode.json)
CASES_DIR="$TESTS_DIR/cases"
OUT_DIR="$TESTS_DIR/outputs"
BASE_DIR="$TESTS_DIR/baseline"

# Agent name defaults to the parent directory's name (e.g. .../agents/severino/tests
# -> "severino"), so this same script drops into any agent's tests/ unchanged.
# Override with AGENT=<name> for non-standard layouts.
AGENT="${AGENT:-$(basename "$PROJECT_DIR")}"
ENDPOINT="${ENDPOINT:-http://localhost:1234/v1}"

# --- colors (no-op if not a tty) ------------------------------------------
if [ -t 1 ]; then
  R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; B=$'\033[34m'; DIM=$'\033[2m'; Z=$'\033[0m'
else
  R= ; G= ; Y= ; B= ; DIM= ; Z=
fi
info() { printf '%s\n' "${B}==>${Z} $*"; }
warn() { printf '%s\n' "${Y}warning:${Z} $*" >&2; }
die()  { printf '%s\n' "${R}error:${Z} $*" >&2; exit 1; }

# Suite-wide assertion tally (deterministic; gates the exit code).
ASSERT_PASS=0
ASSERT_FAIL=0

# --- helpers ---------------------------------------------------------------
list_cases() {
  find "$CASES_DIR" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/prompt.md' ';' -print \
    | sort | xargs -r -n1 basename
}

# Strip volatile header lines so baseline diffs only compare the answer body.
strip_volatile() { grep -v -E '^- \*\*(Generated|Duration|Model):\*\*' "$1" 2>/dev/null || true; }

# Run deterministic substring assertions from a case's expect.md against the
# response body. Format (one directive per line; blanks and '#' comments skipped):
#   require: <substring>   body MUST contain it  (case-insensitive, literal)
#   reject:  <substring>   body must NOT contain it
# Checks the response only (not the echoed prompt), so a prompt quoting the
# forbidden text won't trip a reject. Updates the global ASSERT_PASS/ASSERT_FAIL.
check_expectations() {
  local expect_file="$1" body_file="$2"
  [ -f "$expect_file" ] || return 0

  local raw line kind sub present ok n=0
  while IFS= read -r raw || [ -n "$raw" ]; do
    line="${raw%$'\r'}"                         # tolerate CRLF
    line="${line#"${line%%[![:space:]]*}"}"     # left-trim
    case "$line" in
      ''|'#'*)   continue ;;
      require:*) kind=require; sub="${line#require:}" ;;
      reject:*)  kind=reject;  sub="${line#reject:}"  ;;
      *) warn "expect.md: ignoring unrecognized line: $line"; continue ;;
    esac
    sub="${sub#"${sub%%[![:space:]]*}"}"        # left-trim the substring
    [ -n "$sub" ] || continue
    n=$((n + 1))

    if grep -iqF -- "$sub" "$body_file"; then present=yes; else present=no; fi
    if [ "$kind" = require ]; then [ "$present" = yes ] && ok=1 || ok=0
    else                           [ "$present" = no  ] && ok=1 || ok=0
    fi

    if [ "$ok" -eq 1 ]; then
      ASSERT_PASS=$((ASSERT_PASS + 1))
      printf '      %sPASS%s %-7s %s\n' "$G" "$Z" "$kind" "$sub"
    else
      ASSERT_FAIL=$((ASSERT_FAIL + 1))
      printf '      %sFAIL%s %-7s %s\n' "$R" "$Z" "$kind" "$sub"
    fi
  done < "$expect_file"
}

health_check() {
  info "Checking LM Studio at ${ENDPOINT} ..."
  local models
  if ! models="$(curl -s --max-time 5 "${ENDPOINT}/models" 2>/dev/null)" || [ -z "$models" ]; then
    die "LM Studio server not reachable at ${ENDPOINT}.
  Start it in LM Studio: Developer -> Local Server -> Start Server,
  and make sure a model is loaded with Context Length >= 16K.
  Override the URL with: ENDPOINT=http://host:port/v1 $0"
  fi
  printf '%s\n' "${G}ok${Z} server is up"
}

run_one() {
  local case_name="$1"
  local case_dir="$CASES_DIR/$case_name"
  local prompt_file="$case_dir/prompt.md"
  [ -f "$prompt_file" ] || die "no prompt.md in case '$case_name'"

  # Fixtures = every file in the case dir except the control files (prompt.md,
  # notes.md, expect.md), attached with -f. Paths are made relative to
  # PROJECT_DIR (we cd there before running) so the agent sees and echoes
  # relative paths instead of the absolute home dir / username.
  local -a attach=()
  local f base rel
  while IFS= read -r -d '' f; do
    base="$(basename "$f")"
    case "$base" in
      prompt.md|notes.md|expect.md) ;;
      *) rel="${f#"$PROJECT_DIR/"}"; attach+=( -f "$rel" ) ;;
    esac
  done < <(find "$case_dir" -mindepth 1 -maxdepth 1 -type f -print0 | sort -z)

  local -a model_arg=()
  [ -n "${MODEL:-}" ] && model_arg=( --model "$MODEL" )

  local out_file="$OUT_DIR/$case_name.md"
  local body_file; body_file="$(mktemp)"
  local err_file;  err_file="$(mktemp)"   # per-run, per-agent: no cross-run collisions
  local prompt; prompt="$(cat "$prompt_file")"

  info "Running ${case_name} ${DIM}(${#attach[@]} attach args)${Z}"
  local start end dur rc=0
  start="$(date +%s)"
  # Run from the project dir so the agent picks up opencode.json + its agent definition.
  # NOTE: `-f`/`--file` is a greedy array option — it swallows any following positional
  # args, so the prompt MUST come before the -f flags or it gets parsed as a filename.
  ( cd "$PROJECT_DIR" && opencode run --agent "$AGENT" "${model_arg[@]}" "$prompt" "${attach[@]}" ) \
    >"$body_file" 2>"$err_file" || rc=$?
  end="$(date +%s)"; dur=$(( end - start ))

  if [ "$rc" -ne 0 ]; then
    warn "opencode exited with code $rc for '$case_name' (stderr below)"
    sed 's/^/      /' "$err_file" >&2
  fi
  rm -f "$err_file"

  # Redact absolute paths the model may have echoed, so committed reports stay
  # relative and never leak the home dir / username. PROJECT_DIR first (more
  # specific) -> relative; any remaining $HOME -> ~.
  sed -i -e "s#${PROJECT_DIR}/##g" -e "s#${HOME}/#~/#g" "$body_file" 2>/dev/null || true

  # Assemble the human-readable output markdown.
  {
    printf '# %s\n\n' "$case_name"
    printf -- '- **Agent:** %s\n' "$AGENT"
    printf -- '- **Model:** %s\n' "${MODEL:-<agent default>}"
    printf -- '- **Generated:** %s\n' "$(date -Is)"
    printf -- '- **Duration:** %ss\n' "$dur"
    if [ "${#attach[@]}" -gt 0 ]; then
      printf -- '- **Attached:**'
      for ((i=1; i<${#attach[@]}; i+=2)); do printf ' `%s`' "$(basename "${attach[$i]}")"; done
      printf '\n'
    fi
    printf '\n## Prompt\n\n```\n%s\n```\n\n## Response\n\n' "$prompt"
    cat "$body_file"
    printf '\n'
  } > "$out_file"

  printf '    %swrote%s %s %s(%ss)%s\n' "$G" "$Z" "outputs/$case_name.md" "$DIM" "$dur" "$Z"

  # Deterministic assertions (optional per-case expect.md) — check against the
  # response body only, so run before body_file is removed.
  check_expectations "$case_dir/expect.md" "$body_file"
  rm -f "$body_file"

  # Compare against baseline.
  local base_file="$BASE_DIR/$case_name.md"
  if [ ! -f "$base_file" ]; then
    printf '    %sno baseline%s — run %s./run.sh --bless %s%s to seed\n' "$Y" "$Z" "$DIM" "$case_name" "$Z"
  elif diff -q <(strip_volatile "$base_file") <(strip_volatile "$out_file") >/dev/null; then
    printf '    %sunchanged%s vs baseline\n' "$G" "$Z"
  else
    printf '    %schanged%s vs baseline:\n' "$Y" "$Z"
    diff -u <(strip_volatile "$base_file") <(strip_volatile "$out_file") \
      | sed 's/^/      /' || true
  fi
}

bless() {
  mkdir -p "$BASE_DIR"
  local -a names
  if [ "$#" -gt 0 ]; then names=( "$@" ); else mapfile -t names < <(list_cases); fi
  local n missing=0
  for n in "${names[@]}"; do
    if [ -f "$OUT_DIR/$n.md" ]; then
      cp "$OUT_DIR/$n.md" "$BASE_DIR/$n.md"
      printf '%s blessed%s %s\n' "$G" "$Z" "$n"
    else
      warn "no output for '$n' — run it first"; missing=1
    fi
  done
  [ "$missing" -eq 0 ] || exit 1
}

# --- arg parsing -----------------------------------------------------------
case "${1:-}" in
  -h|--help)  awk 'NR>1 && /^#/{sub(/^# ?/,""); print; next} NR>1{exit}' "$0"; exit 0 ;;
  --list)     list_cases; exit 0 ;;
  --bless)    shift; bless "$@"; exit 0 ;;
esac

[ -d "$CASES_DIR" ] || die "no cases/ directory at $CASES_DIR"
command -v opencode >/dev/null || die "opencode not found on PATH"

mkdir -p "$OUT_DIR" "$BASE_DIR"

# Which cases to run.
declare -a TO_RUN
if [ "$#" -gt 0 ]; then
  TO_RUN=( "$@" )
  for c in "${TO_RUN[@]}"; do
    [ -f "$CASES_DIR/$c/prompt.md" ] || die "unknown case '$c' (see ./run.sh --list)"
  done
else
  mapfile -t TO_RUN < <(list_cases)
fi
[ "${#TO_RUN[@]}" -gt 0 ] || die "no cases found under $CASES_DIR"

health_check
for c in "${TO_RUN[@]}"; do run_one "$c"; done
info "done — ${#TO_RUN[@]} case(s). Review diffs above; ${DIM}./run.sh --bless${Z} to accept current outputs."

# Deterministic assertion summary. Diffs are advisory; assertions gate the exit.
if [ "$(( ASSERT_PASS + ASSERT_FAIL ))" -gt 0 ]; then
  if [ "$ASSERT_FAIL" -gt 0 ]; then
    printf '%s\n' "${R}assertions: ${ASSERT_PASS} passed, ${ASSERT_FAIL} failed${Z}"
    exit 1
  fi
  printf '%s\n' "${G}assertions: ${ASSERT_PASS} passed, 0 failed${Z}"
fi
