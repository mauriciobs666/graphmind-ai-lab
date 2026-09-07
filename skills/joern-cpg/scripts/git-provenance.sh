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
# Functions (both safe under `set -euo pipefail`):
#   cpg_provenance_capture <path>
#   cpg_provenance_stamp <built_at> <parsed_at> <source_path> <provenance>

# cpg_provenance_capture <path>
#   Populate CPG_SOURCE_ORIGIN / CPG_SOURCE_COMMIT / CPG_SOURCE_TREE /
#   CPG_SOURCE_DIRTY from <path>'s git work tree, scoped to <path> itself.
#
#   CPG_SOURCE_ORIGIN  repo-relative path of <path> ("." at the repo root) —
#                      the path a consumer can hand to `git log`/`git rev-parse`
#   CPG_SOURCE_COMMIT  short HEAD of that repo, at capture time
#   CPG_SOURCE_TREE    short tree object of ORIGIN at that commit — the exact
#                      identity of the source content, so a consumer can ask
#                      "is the committed source still what was parsed?" with one
#                      comparison instead of a heuristic commit count
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
  local target="${1:-}" dir name prefix tracked
  [ -n "$target" ] && [ -e "$target" ] || return 1
  command -v git >/dev/null 2>&1 || return 1

  if [ -d "$target" ]; then dir="$target"; name="."; else dir="$(dirname "$target")"; name="$(basename "$target")"; fi
  git -C "$dir" rev-parse --is-inside-work-tree >/dev/null 2>&1 || return 1

  # Tracked-ness gate. No `| head -1` on purpose: under `set -o pipefail` a
  # SIGPIPE'd `git` would make the substitution fail. Listing a source tree's
  # tracked files is trivial next to parsing it.
  tracked="$(git -C "$dir" ls-files -- "$name" 2>/dev/null || true)"
  [ -n "$tracked" ] || return 1

  git -C "$dir" rev-parse --verify --quiet HEAD >/dev/null 2>&1 || return 1  # unborn branch

  prefix="$(git -C "$dir" rev-parse --show-prefix)"
  if [ "$name" = "." ]; then CPG_SOURCE_ORIGIN="${prefix%/}"; else CPG_SOURCE_ORIGIN="${prefix}${name}"; fi
  [ -n "$CPG_SOURCE_ORIGIN" ] || CPG_SOURCE_ORIGIN="."

  CPG_SOURCE_COMMIT="$(git -C "$dir" rev-parse --short HEAD)"
  if [ "$CPG_SOURCE_ORIGIN" = "." ]; then
    CPG_SOURCE_TREE="$(git -C "$dir" rev-parse --short 'HEAD^{tree}' 2>/dev/null || true)"
  else
    CPG_SOURCE_TREE="$(git -C "$dir" rev-parse --short "HEAD:$CPG_SOURCE_ORIGIN" 2>/dev/null || true)"
  fi

  if [ -n "$(git -C "$dir" status --porcelain -- "$name" 2>/dev/null || true)" ]; then
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
#   EVERY property is written on EVERY stamp — an absent one explicitly to
#   NULL, which REMOVES the property in FalkorDB (verified 2026-09-07 against
#   the live instance). Without that, an `--append` re-stamp of a graph whose
#   earlier build had a commit would leave the old SOURCE_COMMIT in place,
#   now describing a build that no longer exists.
#
#   <provenance> records HOW the values were obtained, because the consumer's
#   trust in them differs: `parse-root` (derived from a tracked parse root) ·
#   `source-origin` (derived from the tracked directory the caller named as the
#   parse root's real counterpart) · `none` (no git provenance — the three
#   SOURCE_COMMIT/TREE/DIRTY fields are absent, deliberately, not missing).
cpg_provenance_stamp() {
  local built_at="$1" parsed_at="$2" source_path="$3" provenance="$4"
  # Cypher string literal, or NULL when empty. Backslashes first, then quotes.
  _cpg_str() {
    if [ -n "${1:-}" ]; then
      printf '"%s"' "$(printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g')"
    else
      printf 'NULL'
    fi
  }
  printf '%s' "MERGE (b:CpgBuildInfo)
SET b.BUILT_AT = $(_cpg_str "$built_at"),
    b.PARSED_AT = $(_cpg_str "$parsed_at"),
    b.SOURCE_PATH = $(_cpg_str "$source_path"),
    b.PROVENANCE = $(_cpg_str "$provenance"),
    b.SOURCE_ORIGIN = $(_cpg_str "${CPG_SOURCE_ORIGIN:-}"),
    b.SOURCE_COMMIT = $(_cpg_str "${CPG_SOURCE_COMMIT:-}"),
    b.SOURCE_TREE = $(_cpg_str "${CPG_SOURCE_TREE:-}"),
    b.SOURCE_DIRTY = ${CPG_SOURCE_DIRTY:-NULL}"
}
