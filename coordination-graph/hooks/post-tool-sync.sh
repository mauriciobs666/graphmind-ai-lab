#!/usr/bin/env bash
# post-tool-sync.sh — Claude Code PostToolUse / PostToolUseFailure hook (matcher Write|Edit|Bash):
# mirror every coordination file the tool call changed into the graph before the agent reads the result.
# Behaviour: claude/docs/plans/coordination-graph.md section 3.2; contract: coordination-graph/README.md.
#
# Fast path (nothing changed): no Python, no fork beyond one pruned `find`. Python runs only when a mirrored
# file is newer than .state/stamp (`hook-sync`, foreground, `timeout 50`) or a retry is due (`retry-failed`,
# detached). Exit 2 tells the agent an edit is NOT mirrored (stderr names each file); exit 0 otherwise.
#
# Kill switch: COORDGRAPH_HOOK_DISABLED=1 (or true/yes/on) or a file .state/disabled -> exit 0.
# Root is $CLAUDE_PROJECT_DIR, never the payload's cwd. The shared-instance confirmation is this script's own
# decision: the LITERAL key below, passed as a flag, never read from the environment and never exported.
# COORDGRAPH_HOOK_CLI replaces the coordgraph entry point and COORDGRAPH_HOOK_TIMEOUT_S the 50 s foreground
# bound: both are test seams, never set in production (a non-numeric timeout falls back to 50).
# Needs bash 5 (EPOCHSECONDS, EPOCHREALTIME, ${v,,}).
set -u

HERE="${BASH_SOURCE[0]%/*}"
ROOT="${CLAUDE_PROJECT_DIR:-}"
[ -n "$ROOT" ] || ROOT="$(cd "$HERE/../.." && pwd)"
STATE="${COORDGRAPH_STATE_DIR:-$ROOT/coordination-graph/.state}"
STAMP="$STATE/stamp"
CLI="${COORDGRAPH_HOOK_CLI:-$HERE/../coordgraph.sh}"
CONFIRM_KEY=coordination

v="${COORDGRAPH_HOOK_DISABLED:-}"
v="${v//[[:space:]]/}"
case "${v,,}" in 1 | true | yes | on) exit 0 ;; esac
[ -e "$STATE/disabled" ] && exit 0

NOW="$EPOCHSECONDS"
START="${EPOCHREALTIME/,/.}"
export COORDGRAPH_REPO_ROOT="$ROOT" COORDGRAPH_STATE_DIR="$STATE"
unset COORDGRAPH_SHARED_CONFIRM  # the flag below is the only authority this hook acts on

payload=""
# `read -d ''` takes a pipe a byte at a time (~100 ms per 300 KB); `cat` does not
[ -t 0 ] || payload="$(cat)"

COVERED='^((.*/)?docs/(plans/[^/]*-coordination|reviews/[^/]*|requirements/[^/]*)\.md|(.*/)?docs/BACKLOG\.md|claude/[^/]+/kaizen/plan\.md)$'

# every mirrored file under $ROOT (newer than $1 when given), repo-relative, one per line
discover() {
  local newer=()
  [ -n "${1:-}" ] && newer=(-newer "$1")
  local f
  while IFS= read -r f; do
    f="${f#"$ROOT"/}"
    # find's `*` also crosses `/`; discover.py's globs do not: keep exactly the shapes it covers
    [[ $f =~ $COVERED ]] && printf '%s\n' "$f"
  done < <(
    find "$ROOT" -mindepth 1 \( -type d \( -name '.?*' -o -name node_modules -o -name archive \
      -o -path "$ROOT/deprecated" \) -prune \) -o -type f "${newer[@]}" \( \
      -path '*/docs/plans/*-coordination.md' -o -path '*/docs/reviews/*.md' -o -path '*/docs/requirements/*.md' \
      -o -path '*/docs/BACKLOG.md' -o -path '*/claude/*/kaizen/plan.md' \) -print 2>/dev/null)
}

# the KB registry's files (newer than $1 when given)
kb_registry() {
  local line reg="$ROOT/coordination-graph/kb-files.txt"
  [ -r "$reg" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    case "$line" in '' | '#'*) continue ;; esac
    if [ -z "${1:-}" ] || [ "$ROOT/$line" -nt "$1" ]; then [ -e "$ROOT/$line" ] && printf '%s\n' "$line"; fi
  done <"$reg"
}

FIRST=0
C=()
if [ ! -e "$STAMP" ]; then
  # no stamp (fresh clone, .state cleaned): one full hash-compare run, but only once retry-due has passed
  if [ -r "$STATE/retry-due" ]; then
    read -r due <"$STATE/retry-due"
    [[ ${due:-} =~ ^[0-9]+$ ]] && ((due > NOW)) && exit 0
  fi
  FIRST=1
else
  declare -A seen=()
  while IFS= read -r f; do
    [ -n "$f" ] && [ -z "${seen[$f]:-}" ] && seen[$f]=1 && C+=("$f")
  done < <(discover "$STAMP"; kb_registry "$STAMP")
  # a Write/Edit naming a file that is in the failure state is retried at once, even when the stamp is past it
  if [[ $payload =~ \"file_path\"[[:space:]]*:[[:space:]]*\"([^\"]*)\" ]] \
     && { [ -s "$STATE/failed.json" ] || [ -s "$STATE/failed.pending" ]; }; then
    fp="${BASH_REMATCH[1]//\\\//\/}"
    case "$fp" in "$ROOT"/*) fp="${fp#"$ROOT"/}" ;; /*) fp="" ;; *) fp="${fp#./}" ;; esac
    if [ -n "$fp" ] && [ -z "${seen[$fp]:-}" ] \
       && { grep -qF -- "\"$fp\"" "$STATE/failed.json" 2>/dev/null \
            || grep -qxF -- "$fp" "$STATE/failed.pending" 2>/dev/null; }; then
      C+=("$fp")
    fi
  fi
  if ((${#C[@]} == 0)); then
    # nothing to sync: launch the detached retry when one is due (retry-due is bumped first, so a crashing
    # retry costs one fork a minute, not one per call)
    if [ -r "$STATE/retry-due" ]; then
      read -r due <"$STATE/retry-due"
      if [[ ${due:-} =~ ^[0-9]+$ ]] && ((due <= NOW)); then
        echo $((NOW + 60)) >"$STATE/retry-due"
        setsid "$CLI" retry-failed --confirm "$CONFIRM_KEY" >>"$STATE/hook.log" 2>&1 </dev/null &
      fi
    fi
    exit 0
  fi
fi

[ -d "$STATE" ] || mkdir -p "$STATE"

args=(hook-sync --confirm "$CONFIRM_KEY")
if ((FIRST)); then args+=(--first-run --stamp-at "$START"); else args+=("${C[@]}"); fi
limit="${COORDGRAPH_HOOK_TIMEOUT_S:-}"
[[ $limit =~ ^[0-9]+$ ]] && ((limit > 0)) || limit=50
err="$(timeout "$limit" "$CLI" "${args[@]}" 2>&1 >>"$STATE/hook.log" </dev/null)"
rc=$?

fail=0
case $rc in
  0) ;;
  2) printf '%s\n' "$err" >&2; fail=1 ;;
  *)
    # the hook's own fault (venv missing, exception, timeout): never read as "nothing to sync"
    printf '%s: hook-sync exit %s\n%s\n' "$(date -u +%FT%TZ)" "$rc" "$err" >>"$STATE/hook.log"
    if ((FIRST)); then mapfile -t C < <(discover; kb_registry); fi
    ( flock 9; printf '%s\n' "${C[@]}" >>"$STATE/failed.pending" ) 9>>"$STATE/state.lock"
    echo $((NOW + 60)) >"$STATE/retry-due"
    {
      printf 'coordgraph failed: exit %s; see %s. Not synced (retried automatically):\n' "$rc" "$STATE/hook.log"
      printf '  %s\n' "${C[@]:0:20}"
      ((${#C[@]} > 20)) && printf '  ... and %s more (all queued in %s)\n' $((${#C[@]} - 20)) "$STATE/failed.pending"
      printf 'this update is not complete — say so\n'
    } >&2
    fail=1
    ;;
esac

# advance the stamp to max(stamp, start), never to "now", and never held back by a failure; a first run's
# stamp belongs to hook-sync (absent while FalkorDB is down), unless hook-sync itself failed
if ((!FIRST)) || { [ $rc -ne 0 ] && [ $rc -ne 2 ]; }; then
  tmp="$STATE/stamp.$$"
  touch -d "@$START" "$tmp"
  if [ ! -e "$STAMP" ] || [ "$tmp" -nt "$STAMP" ]; then mv -f "$tmp" "$STAMP"; else rm -f "$tmp"; fi
fi

exit $((fail ? 2 : 0))
