#!/usr/bin/env bash
# git-provenance.sh — capture the git provenance of the source a CPG is built
# from, and render it as the FalkorDB `CpgBuildInfo` stamp. SOURCE this (do not
# exec):
#   . "$(dirname "$0")/git-provenance.sh"
#
# WHY THIS IS A SEPARATE, CAPTURE-ONCE STEP.
# The commit and the dirty flag are properties of *the tree that was parsed*,
# not of the repository at the moment the load finished. Deriving them at stamp
# time — after a parse+export+load that runs for hours — gets both wrong:
#   1. It races HEAD. Observed 2026-09-07 on a ~3h `cpg_falkorchat` build: HEAD
#      moved four times under a concurrent session (a69422f → b795f4c →
#      439bcb8 → 2624425); the parse had read b795f4c, and the graph was
#      stamped 2624425 — a tree that was never parsed.
#   2. Without a pathspec, `git status --porcelain` reports dirt from anywhere
#      in the repository. The same build was stamped SOURCE_DIRTY=true because
#      of a modified file outside the parse root entirely, while the parsed
#      tree was verified byte-identical to its commit.
# Both directions are live failures for the consumer recipe
# (`skills/cpg-analysis/references/freshness.md`): a raced commit makes a stale
# graph look fresh, a repo-wide dirty flag makes a clean graph look untrustworthy.
# So: capture before the parse, scope to the source, carry the values through.
#
# Functions (all safe under `set -euo pipefail`):
#   cpg_provenance_capture <path>
#   cpg_provenance_stamp <built_at> <parsed_at> <source_path> <provenance>
#   cpg_provenance_stray_query          (call after the stamp; see its comment)

# cpg_provenance_capture <path>
#   Populate CPG_SOURCE_ORIGIN / CPG_SOURCE_COMMIT / CPG_SOURCE_TREE /
#   CPG_SOURCE_DIRTY from <path>'s git work tree, scoped to <path> itself.
#
#   CPG_SOURCE_ORIGIN  repo-relative path of <path> ("." at the repo root) —
#                      the path a consumer can hand to `git log`/`git rev-parse`
#   CPG_SOURCE_COMMIT  HEAD of that repo, at capture time
#   CPG_SOURCE_TREE    the tree object of ORIGIN at that commit (a BLOB when the
#                      source is a single file) — the exact identity of the
#                      source content, so a consumer can ask "is the committed
#                      source still what was parsed?" with one comparison
#                      instead of a heuristic commit count
#
#   Both object ids are FULL 40-char OIDs, never `--short`. Abbreviation width
#   comes from `core.abbrev=auto`, i.e. the repo's object count *at the time it
#   runs*: a value stamped at 7 chars and re-checked months later at 8 would
#   fail a string comparison on width alone, and CPG_SOURCE_TREE exists to be a
#   reliable equality test. Shorten only for human-readable log lines.
#   CPG_SOURCE_DIRTY   true/false from `git status --porcelain -- <path>`, i.e.
#                      scoped: modifications and untracked files *under the
#                      source*, ignoring the rest of the repository. Untracked
#                      files count — the parse sees them and the commit does not.
#
#   Returns 0 on success; 1 when <path> has no usable git provenance. That
#   failure deliberately includes a path that is INSIDE a work tree but not
#   tracked by it — a staged scratch copy under a gitignored directory, the
#   documented way to scope a parse (see SKILL.md). That repo's HEAD describes
#   some other tree, so reporting it is worse than reporting nothing: it feeds
#   the consumer a plausible commit that silently answers "unchanged". Pass the
#   real tracked directory the copy was staged from instead
#   (`pipeline.sh --source-origin`).
cpg_provenance_capture() {
  CPG_SOURCE_ORIGIN=""; CPG_SOURCE_COMMIT=""; CPG_SOURCE_TREE=""; CPG_SOURCE_DIRTY=""
  local target="${1:-}" dir name spec prefix tracked top
  [ -n "$target" ] && [ -e "$target" ] || return 1
  command -v git >/dev/null 2>&1 || return 1

  if [ -d "$target" ]; then dir="$target"; name="."; else dir="$(dirname "$target")"; name="$(basename "$target")"; fi
  git -C "$dir" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 1

  # `:(literal)` magic, because a bare pathspec is still wildmatched: a basename
  # beginning with `:` is pathspec magic and `[…]` is a character class. (The
  # global `--literal-pathspecs` flag would do the same, but it disables ALL
  # magic — including the `:(exclude…)` form a caller may want — so the
  # per-pathspec prefix is the composable one.)
  spec=":(literal)$name"

  # Tracked-ness gate. No `| head -1` on purpose: under `set -o pipefail` a
  # SIGPIPE'd `git` would make the substitution fail. Listing a source tree's
  # tracked files is trivial next to parsing it.
  tracked="$(git -C "$dir" ls-files -- "$spec" 2>/dev/null || true)"
  [ -n "$tracked" ] || return 1

  git -C "$dir" rev-parse --verify --quiet HEAD >/dev/null 2>&1 || return 1  # unborn branch

  prefix="$(git -C "$dir" rev-parse --show-prefix)"
  if [ "$name" = "." ]; then CPG_SOURCE_ORIGIN="${prefix%/}"; else CPG_SOURCE_ORIGIN="${prefix}${name}"; fi
  [ -n "$CPG_SOURCE_ORIGIN" ] || CPG_SOURCE_ORIGIN="."

  CPG_SOURCE_COMMIT="$(git -C "$dir" rev-parse HEAD)"
  # `HEAD:./<origin>` resolves both a subdirectory and the repo root itself,
  # where a bare `HEAD:.` is fatal — the same form the consumer recipe uses, so
  # producer and consumer cannot diverge. Two traps, both bought with blood:
  #   * the `./` form is CWD-relative, so this MUST run from the repo top level,
  #     not from `$dir` — `git -C src rev-parse HEAD:./src` looks for `src/src`;
  #   * without `--verify`, an unresolvable rev makes `git rev-parse` echo the
  #     ARGUMENT BACK on stdout (while exiting 128), so `|| true` would capture
  #     the literal string "HEAD:./x" as if it were an object id.
  # Stays empty when the source is tracked in the index but absent from HEAD
  # (added, never committed): see the SOURCE_TREE-null caveat in check 2 of the
  # freshness recipe.
  top="$(git -C "$dir" rev-parse --show-toplevel)"
  CPG_SOURCE_TREE="$(git -C "$top" rev-parse --verify --quiet "HEAD:./$CPG_SOURCE_ORIGIN" || true)"

  if [ -n "$(git -C "$dir" status --porcelain -- "$spec" 2>/dev/null || true)" ]; then
    CPG_SOURCE_DIRTY=true
  else
    CPG_SOURCE_DIRTY=false
  fi
  return 0
}

# cpg_provenance_stamp <built_at> <parsed_at> <source_path> <provenance>
#   Echo the Cypher that writes the singleton CpgBuildInfo marker, using
#   whatever cpg_provenance_capture left in CPG_SOURCE_*.
#
#   THE INVARIANT: THE MARKER DESCRIBES EXACTLY ONE BUILD AND NOTHING ELSE. A
#   human annotation on it is build-scoped and dies with the build; anything
#   durable about the graph or its component belongs in docs/, not here.
#
#   Every property this stamp names is written on every stamp — an absent one
#   explicitly to NULL, which REMOVES the property in FalkorDB rather than
#   storing one (verified 2026-09-08 by execution on a throwaway graph: a 13-key
#   marker put through a full `parse-root` stamp reported `Properties removed:
#   13` and `keys(b)` read back exactly the eight pipeline fields). Without
#   that, an `--append` re-stamp of a graph whose earlier build had a commit
#   would leave the old SOURCE_COMMIT in place, describing a build that no
#   longer exists.
#
#   BUT THE LIST BELOW IS A LIST, NOT THE INVARIANT — and a list cannot enforce
#   its own completeness. Naming a property here is what makes it *cleared
#   quietly*; it is no longer what makes the invariant hold. That distinction
#   was bought twice in two days: the five hand-authored keys graph-dba writes
#   on a hand-written or hand-backfilled marker (MARKER_ORIGIN,
#   MARKER_WRITTEN_AT, NOTE, STATUS, RENAMED_FROM) postdate the original eight
#   and were missing here until 2026-09-08, so an `--append` rebuild left them
#   standing over freshly captured pipeline fields — a marker announcing itself
#   as "NOT a pipeline stamp" while carrying one. Adding them closed the list
#   and did not close the hole: a SIXTH hand-authored key (MARKER_EVIDENCE),
#   invented the same day, survived a full stamp untouched, reproducing the
#   defect one key over.
#
#   The enforcement is cpg_provenance_stray_query below, which the pipeline runs
#   after every stamp: any property on the marker that this stamp did not write
#   FAILS THE BUILD, whatever it is called and whenever it was invented. So a
#   seventh hand-authored key does not silently corrupt a marker — it stops a
#   rebuild with its own name in the error, and the fix is to add it here.
#
#   <provenance> records HOW the values were obtained, because the consumer's
#   trust in them differs: `parse-root` (derived from a tracked parse root) ·
#   `source-origin` (derived from the tracked directory the caller named as the
#   parse root's real counterpart) · `none` (no git provenance — the three
#   SOURCE_COMMIT/TREE/DIRTY fields are absent, deliberately, not missing).

# _cpg_str <value> — a Cypher string literal, or NULL when empty. Backslashes
# first, then quotes. Defined at file scope on purpose: bash has no
# function-scoped functions, so nesting it inside cpg_provenance_stamp would
# still leak it into the sourcing shell, just less visibly.
_cpg_str() {
  if [ -n "${1:-}" ]; then
    printf '"%s"' "$(printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')"
  else
    printf 'NULL'
  fi
}

# _cpg_prop <NAME> <cypher-value> — append one `b.NAME = <value>` assignment to
# the SET clause under construction and, unless <value> is the literal NULL,
# record NAME in CPG_STAMPED_KEYS. That side effect is the point: it makes the
# stamp's own SET clause the single source of truth for which properties a
# pipeline-stamped marker may carry, so cpg_provenance_stray_query's allow-list
# cannot drift from it the way a hand-copied second list would.
#
# CALL IT AS A STATEMENT, NEVER INSIDE `$(…)`. A command substitution runs in a
# subshell, so both assignments would be discarded while the printed Cypher
# looked perfectly correct — leaving CPG_STAMPED_KEYS empty, which makes *every*
# property on the marker read as stray. (That is at least the safe direction:
# an empty allow-list fails the build loudly rather than passing everything.
# Verified 2026-09-08 against the live instance: `NOT k IN []` matched all 10
# keys of `cpg_falkorchat`'s marker.)
#
# At file scope for the same reason as _cpg_str.
_cpg_prop() {
  [ "$2" = NULL ] || CPG_STAMPED_KEYS="${CPG_STAMPED_KEYS}${CPG_STAMPED_KEYS:+ }$1"
  _CPG_SET_CLAUSE="${_CPG_SET_CLAUSE}${_CPG_SET_CLAUSE:+,
    }b.$1 = $2"
}

cpg_provenance_stamp() {
  local built_at="$1" parsed_at="$2" source_path="$3" provenance="$4"
  CPG_STAMPED_KEYS=""; _CPG_SET_CLAUSE=""
  _cpg_prop BUILT_AT          "$(_cpg_str "$built_at")"
  _cpg_prop PARSED_AT         "$(_cpg_str "$parsed_at")"
  _cpg_prop SOURCE_PATH       "$(_cpg_str "$source_path")"
  _cpg_prop PROVENANCE        "$(_cpg_str "$provenance")"
  _cpg_prop SOURCE_ORIGIN     "$(_cpg_str "${CPG_SOURCE_ORIGIN:-}")"
  _cpg_prop SOURCE_COMMIT     "$(_cpg_str "${CPG_SOURCE_COMMIT:-}")"
  _cpg_prop SOURCE_TREE       "$(_cpg_str "${CPG_SOURCE_TREE:-}")"
  _cpg_prop SOURCE_DIRTY      "${CPG_SOURCE_DIRTY:-NULL}"
  _cpg_prop MARKER_ORIGIN     NULL
  _cpg_prop MARKER_WRITTEN_AT NULL
  _cpg_prop NOTE              NULL
  _cpg_prop STATUS            NULL
  _cpg_prop RENAMED_FROM      NULL
  printf 'MERGE (b:CpgBuildInfo)\nSET %s' "$_CPG_SET_CLAUSE"
}

# cpg_provenance_stray_query
#   Echo the READ that lists every property on the marker which the preceding
#   cpg_provenance_stamp did NOT write. Call it after the stamp; it uses the
#   CPG_STAMPED_KEYS that call left behind.
#
#   WHY IT EXISTS. The NULLs above clear a CLOSED LIST of hand-authored keys,
#   and a closed list is only as good as whoever last edited it. On 2026-09-08 a
#   sixth hand-authored key (MARKER_EVIDENCE) was written onto a marker and
#   survived a full `parse-root` stamp untouched — reproducing the exact defect
#   the list had just been closed against, one key over. The list cannot enforce
#   its own completeness, and the discipline is invisible where it breaks: the
#   author of a hand-written marker is working in a different file entirely.
#
#   This query enforces the invariant instead of asking for it. The allow-list is
#   generated from the stamp's own assignments (see _cpg_prop), so it covers
#   every hand-authored key that will ever exist — including ones not yet
#   invented — by the simple fact that the pipeline did not write them.
#
#   Zero rows is the pass. Each failure renders as `STRAY_KEY=<NAME>` in
#   redis-cli's default output, greppable without parsing the reply, and names
#   the offending key so the fix is mechanical (verified 2026-09-08 against
#   `cpg_falkorchat`'s live 10-key marker: the 8-key allow-list returned exactly
#   STRAY_KEY=MARKER_ORIGIN / MARKER_WRITTEN_AT / NOTE, and the 11-key one
#   returned no rows).
#
#   It does NOT check that the stamped keys are PRESENT — absence is legitimate
#   (`provenance=none` deliberately omits SOURCE_COMMIT/TREE/DIRTY), and the
#   pipeline's PARSED_AT read-back already proves the write landed. Extra
#   properties are the defect; missing ones are not.
cpg_provenance_stray_query() {
  local k list=""
  for k in ${CPG_STAMPED_KEYS:-}; do list="${list}${list:+,}'$k'"; done
  printf '%s' "MATCH (b:CpgBuildInfo)
UNWIND keys(b) AS k
WITH k WHERE NOT k IN [$list]
RETURN 'STRAY_KEY=' + k AS stray"
}
