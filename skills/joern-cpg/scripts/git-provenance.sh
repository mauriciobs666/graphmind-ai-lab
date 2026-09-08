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
#         -> assigns CPG_STAMP_CYPHER and CPG_STAMPED_KEYS. Does NOT echo, and
#            must NOT be called in `$(…)`: the subshell discards both.
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
#   Render the Cypher that writes the singleton CpgBuildInfo marker, using
#   whatever cpg_provenance_capture left in CPG_SOURCE_*, into CPG_STAMP_CYPHER.
#   It ASSIGNS rather than echoing, and the reason is a defect this shipped with
#   for two commits — see the note at the assignment itself.
#
#   THE INVARIANT: THE MARKER DESCRIBES EXACTLY ONE BUILD AND NOTHING ELSE. A
#   human annotation on it is build-scoped and dies with the build; anything
#   durable about the graph or its component belongs in docs/, not here.
#
#   HOW THE INVARIANT IS HELD: `SET b = {…}` — MAP ASSIGNMENT WITH `=`, WHICH
#   REPLACES THE NODE'S ENTIRE PROPERTY SET. Not `+=`, which merges, and not a
#   list of `b.X = NULL` clearing statements, which is what this used to be. A
#   property that is not in the map is gone afterwards whatever it is called,
#   whenever it was invented, and whether or not anyone remembered to name it
#   here. That is the whole point: the closure is by construction, so there is
#   nothing to keep in sync and nothing to remember.
#
#   EXECUTED, on throwaway graphs, 2026-09-08 — read back with `keys(b)` every
#   time, never from the reply's counters. What was OBSERVED about the counters,
#   which is all anyone here has checked: `Properties removed` does not track
#   actual removals — one probe reported 13 against 5 real ones, another 4
#   against none. Why it diverges is not known and is not asserted (see
#   graph-dba's U47a finding); that it cannot be cited as evidence is enough.
#     * A marker seeded with the 8 pipeline keys plus 3 hand-authored ones —
#       including MARKER_EVIDENCE, the out-of-list key that had just survived
#       the old clearing enumeration — put through `SET b = {…8 pipeline keys…}`
#       read back `keys(b)` of size 8: all three hand-authored keys gone,
#       MARKER_EVIDENCE included. `count(b) = 1` and `labels(b) =
#       [CpgBuildInfo]` both survived the replace, which was the one way this
#       could have been quietly catastrophic — `MATCH (b:CpgBuildInfo)` still
#       finds it.
#     * An explicit NULL *inside the map* OMITS that property from THE NODE; it
#       does not store a null. An 8-entry map with 4 NULL values read back
#       `keys(b)` of size 4, the four real keys. THIS IS WHY the eight lines
#       below are emitted unconditionally: an absent SOURCE_COMMIT can be
#       written as NULL and simply will not exist on the node.
#       Careful reading the map itself, though — the omission happens at
#       ASSIGNMENT, not in the literal. `WITH {…8 entries, 4 of them NULL…} AS m
#       RETURN keys(m)` returns all EIGHT (checked 2026-09-08 with the literal
#       this function actually emits, which also confirmed the quote/backslash
#       escaping survives: `p"q` and `a"b\c` round-tripped, SOURCE_DIRTY came
#       back as a real boolean). So `keys(b)` on the node is the discriminator
#       here, exactly as it is for the counters.
#     * `provenance=none` (SOURCE_ORIGIN/COMMIT/TREE/DIRTY all absent) read back
#       exactly the four unconditional keys.
#     * Against a graph with NO marker — the `--reset` path — the MERGE created
#       one: `count(b) = 1`, `labels(b) = [CpgBuildInfo]`, PARSED_AT correct. The
#       create path is unaffected, so pipeline.sh's PARSED_AT read-back still
#       discriminates.
#
#   THE RULE HERE HAS NOW OUTLIVED TWO WRONG MECHANISMS, which is why the
#   evidence above is written out rather than cited. (1) The stamp originally
#   wrote only its own eight fields, so an `--append` rebuild left a
#   hand-authored marker's keys standing over freshly captured pipeline values —
#   a marker announcing itself as "NOT a pipeline stamp" while carrying one.
#   (2) The fix enumerated the five hand-authored keys as `= NULL`, which closed
#   a list rather than a set: a sixth key invented the same day
#   (MARKER_EVIDENCE) survived a full stamp untouched, reproducing the defect
#   one key over, because a closed list cannot enforce its own completeness and
#   whoever hand-writes a marker is editing a different file from this one.
#   (3) The map form above, which was checked before it was written down. Don't
#   reintroduce a `b.X = NULL` line for a hand-authored key — it would be
#   harmless but it would also re-teach the next reader that the list is what
#   matters.
#
#   cpg_provenance_stray_query below is the standing check that (3) actually
#   happened. See its comment before deciding it is redundant.
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

# _cpg_prop <NAME> <cypher-value> — append one `NAME: <value>` entry to the map
# literal under construction and, unless <value> is the literal NULL, record
# NAME in CPG_STAMPED_KEYS. A NULL entry is emitted into the map anyway and
# omits the property (executed — see the stamp's comment above), so the map
# always carries all eight names and CPG_STAMPED_KEYS carries only the ones that
# will actually exist on the node.
#
# That side effect is the point: it makes the stamp's own map the single source
# of truth for which properties a pipeline-stamped marker may carry, so
# cpg_provenance_stray_query's allow-list cannot drift from it the way a
# hand-copied second list would.
#
# CALL IT AS A STATEMENT, NEVER INSIDE `$(…)` — and the same goes for
# cpg_provenance_stamp, which calls it. A command substitution runs in a
# subshell, so both assignments are discarded there while the rendered Cypher
# looks perfectly correct.
#
# THIS WARNING DID NOT WORK. It was written here, one level below the call site,
# while `pipeline.sh` did exactly the forbidden thing — `STAMP="$(cpg_provenance
# _stamp …)"` — and shipped that way for two commits: CPG_STAMPED_KEYS unset in
# the parent, the stray query rendered `NOT k IN []`, and every property on the
# marker read as a stray, so every `--load` build would have failed after a
# multi-hour parse with the annotation already replaced. An earlier version of
# this comment filed that outcome as "at least the safe direction"; it was not a
# direction, it was the shipped state.
#
# So the warning is no longer what protects it. cpg_provenance_stamp does not
# echo — it assigns CPG_STAMP_CYPHER — which makes the subshell misuse render an
# EMPTY stamp and fail at the stamp step instead of silently; and
# cpg_provenance_stray_query refuses to emit a query at all when the allow-list
# is empty. Two mechanisms, because a comment next to the trap did not reach a
# caller two lines away.
#
# At file scope for the same reason as _cpg_str.
_cpg_prop() {
  [ "$2" = NULL ] || CPG_STAMPED_KEYS="${CPG_STAMPED_KEYS}${CPG_STAMPED_KEYS:+ }$1"
  _CPG_MAP="${_CPG_MAP}${_CPG_MAP:+,
    }$1: $2"
}

cpg_provenance_stamp() {
  local built_at="$1" parsed_at="$2" source_path="$3" provenance="$4"
  CPG_STAMPED_KEYS=""; _CPG_MAP=""
  # These eight are the marker, in full. There is deliberately no entry here for
  # MARKER_ORIGIN / MARKER_WRITTEN_AT / NOTE / STATUS / RENAMED_FROM or any
  # other hand-authored key: `SET b = {…}` removes them by replacing the whole
  # property set, so naming them would add nothing but a list to maintain.
  _cpg_prop BUILT_AT      "$(_cpg_str "$built_at")"
  _cpg_prop PARSED_AT     "$(_cpg_str "$parsed_at")"
  _cpg_prop SOURCE_PATH   "$(_cpg_str "$source_path")"
  _cpg_prop PROVENANCE    "$(_cpg_str "$provenance")"
  _cpg_prop SOURCE_ORIGIN "$(_cpg_str "${CPG_SOURCE_ORIGIN:-}")"
  _cpg_prop SOURCE_COMMIT "$(_cpg_str "${CPG_SOURCE_COMMIT:-}")"
  _cpg_prop SOURCE_TREE   "$(_cpg_str "${CPG_SOURCE_TREE:-}")"
  _cpg_prop SOURCE_DIRTY  "${CPG_SOURCE_DIRTY:-NULL}"
  # ASSIGNS, does not echo. Deliberate: the rendered Cypher and the allow-list
  # are produced together and must be read from the same shell, so echoing would
  # invite `STAMP="$(cpg_provenance_stamp …)"` — which discards the allow-list in
  # a subshell and is precisely the defect that shipped in 0da3eb9/5417f0e. This
  # way that misuse yields an EMPTY stamp and dies at the stamp step, loudly,
  # instead of surfacing later as a false finding about the graph.
  CPG_STAMP_CYPHER="MERGE (b:CpgBuildInfo)
SET b = {
    $_CPG_MAP
}"
}

# cpg_provenance_stray_query
#   Echo the READ that lists every property on the marker which the preceding
#   cpg_provenance_stamp did NOT write. Call it after the stamp; it uses the
#   CPG_STAMPED_KEYS that call left behind. Zero rows is the pass; each failure
#   renders as `STRAY_KEY=<NAME>` in redis-cli's default output, greppable
#   without parsing the reply, and names the offending key.
#
#   WHAT IT ACTUALLY DETECTS, stated narrowly because the wider claim was wrong:
#   THE STAMP FAILED TO ERASE A FOREIGN KEY THAT WAS ALREADY ON THE MARKER. Not
#   "the replace semantics changed" — that was the previous framing here and it
#   overclaimed. A `+=`, a reversion to `b.X = …` assignments, or a FalkorDB
#   treating `=` as a merge all leave exactly the eight stamped keys on a marker
#   whose previous stamp was pipeline-clean, so none of them is caught on such a
#   graph. And an edit that drops a property out of the map cannot fire this at
#   all: _cpg_prop removes it from the map and the allow-list in the same call.
#
#   DO NOT DELETE IT, on the honest reason rather than the overclaimed one. Two
#   loaded graphs carry hand-authored markers today — cpg_falkorchat and
#   cpg_deprecated_salesperson — and on each one's next rebuild this is a real,
#   firing check on the exact defect this whole arc was about. After both have
#   rebuilt once it can no longer fire under any trigger named above; at that
#   point it costs one read per build and still catches a foreign key
#   reintroduced by any writer other than the stamp. That is a cheap standing
#   guard on a closed-by-construction invariant, not a regression test for the
#   replace semantics, and the distinction is checkable — which the previous
#   claim was not.
#
#   It does NOT check that the stamped keys are PRESENT — absence is legitimate
#   (`provenance=none` deliberately omits SOURCE_COMMIT/TREE/DIRTY, and an
#   allow-listed key that is not on the node matches nothing here anyway), and
#   the pipeline's PARSED_AT read-back already proves the write landed. Extra
#   properties are the defect; missing ones are not.
cpg_provenance_stray_query() {
  local k list=""
  # An empty allow-list would make every property a stray — a bug in the caller
  # (see _cpg_prop), never a finding about the graph. Refuse rather than emit it.
  if [ -z "${CPG_STAMPED_KEYS:-}" ]; then
    echo "cpg_provenance_stray_query: CPG_STAMPED_KEYS is empty — cpg_provenance_stamp was not run in this shell (a \$(…) subshell discards it). Refusing to emit a query that would report every property as stray." >&2
    return 1
  fi
  for k in $CPG_STAMPED_KEYS; do list="${list}${list:+,}'$k'"; done
  printf '%s' "MATCH (b:CpgBuildInfo)
UNWIND keys(b) AS k
WITH k WHERE NOT k IN [$list]
RETURN 'STRAY_KEY=' + k AS stray"
}
