#!/usr/bin/env bash
#
# run.sh — Severino eval harness (kaizen loop)
#
# Runs each test case under cases/ through `opencode run --agent severino`,
# captures the agent's answer as markdown in outputs/, and diffs it against a
# blessed reference in baseline/. Output is non-deterministic (local LLM), so
# this is a review aid, not a pass/fail gate: you read the diffs to spot
# regressions or improvements, then `--bless` to accept a new baseline.
#
# Usage:
#   ./run.sh                 run all cases, diff against baseline
#   ./run.sh 01-explain-python [02-...]   run only the named case(s)
#   ./run.sh --bless [case...]            promote outputs/ -> baseline/ (all, or named)
#   ./run.sh --list                       list discovered cases
#   ./run.sh -h | --help                  this help
#
# Env overrides:
#   AGENT     agent name           (default: severino)
#   MODEL     provider/model       (default: agent's configured model)
#   ENDPOINT  LM Studio /v1 URL     (default: http://localhost:1234/v1)

set -euo pipefail

# --- locations -------------------------------------------------------------
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$TESTS_DIR")"   # the severino/ config dir (has opencode.json)
CASES_DIR="$TESTS_DIR/cases"
OUT_DIR="$TESTS_DIR/outputs"
BASE_DIR="$TESTS_DIR/baseline"

AGENT="${AGENT:-severino}"
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

# --- helpers ---------------------------------------------------------------
list_cases() {
  find "$CASES_DIR" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/prompt.md' ';' -print \
    | sort | xargs -r -n1 basename
}

# Strip volatile header lines so baseline diffs only compare the answer body.
strip_volatile() { grep -v -E '^- \*\*(Generated|Duration|Model):\*\*' "$1" 2>/dev/null || true; }

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

  # Fixtures = every file in the case dir except prompt.md / notes.md, attached with -f.
  local -a attach=()
  local f base
  while IFS= read -r -d '' f; do
    base="$(basename "$f")"
    case "$base" in
      prompt.md|notes.md) ;;
      *) attach+=( -f "$f" ) ;;
    esac
  done < <(find "$case_dir" -mindepth 1 -maxdepth 1 -type f -print0 | sort -z)

  local -a model_arg=()
  [ -n "${MODEL:-}" ] && model_arg=( --model "$MODEL" )

  local out_file="$OUT_DIR/$case_name.md"
  local body_file; body_file="$(mktemp)"
  local prompt; prompt="$(cat "$prompt_file")"

  info "Running ${case_name} ${DIM}(${#attach[@]} attach args)${Z}"
  local start end dur rc=0
  start="$(date +%s)"
  # Run from the project dir so the agent picks up opencode.json + the severino agent.
  # NOTE: `-f`/`--file` is a greedy array option — it swallows any following positional
  # args, so the prompt MUST come before the -f flags or it gets parsed as a filename.
  ( cd "$PROJECT_DIR" && opencode run --agent "$AGENT" "${model_arg[@]}" "$prompt" "${attach[@]}" ) \
    >"$body_file" 2>/tmp/severino-run.stderr || rc=$?
  end="$(date +%s)"; dur=$(( end - start ))

  if [ "$rc" -ne 0 ]; then
    warn "opencode exited with code $rc for '$case_name' (see /tmp/severino-run.stderr)"
  fi

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
  rm -f "$body_file"

  printf '    %swrote%s %s %s(%ss)%s\n' "$G" "$Z" "outputs/$case_name.md" "$DIM" "$dur" "$Z"

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
  -h|--help)  sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
