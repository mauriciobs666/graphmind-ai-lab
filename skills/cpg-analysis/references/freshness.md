# Recipe: freshness check

> Back to [`../SKILL.md`](../SKILL.md) · schema in
> [`../../joern-cpg/references/cpg-model.md`](../../joern-cpg/references/cpg-model.md).
> **Consumer:** `teco`, at dispatch time, for a unit whose specialist will consult a CPG (2026-08-19: centralized — see `docs/plans/cpg-agent-adoption2.md`; a specialist invoked standalone no longer runs this check). **Covers:** FR-5, FR-6
> (`cpg-agent-adoption`, M4).

**Purpose.** Before trusting a loaded CPG's answers, find out how current it is
relative to the source it describes — and if it looks stale, say so rather than
silently treating it as ground truth. No parameter to change; run as-is against
whichever graph you're already querying.

```cypher
MATCH (b:CpgBuildInfo)
RETURN b.BUILT_AT AS builtAt, b.PARSED_AT AS parsedAt,
       b.PROVENANCE AS provenance, b.SOURCE_ORIGIN AS sourceOrigin,
       b.SOURCE_COMMIT AS sourceCommit, b.SOURCE_TREE AS sourceTree,
       b.SOURCE_DIRTY AS sourceDirty, b.SOURCE_PATH AS sourcePath
```

**Expected shape.** Zero or one row — this is a singleton marker node, not a
per-build history.

| Field | What it is |
|---|---|
| `builtAt` | When the **load finished** — "when this graph's content was last touched". |
| `parsedAt` | When the **source snapshot was taken**, i.e. what the graph actually describes. On a multi-hour build these differ by hours: anchor any `--since` on `parsedAt`, never `builtAt`. |
| `sourcePath` | The parse root handed to Joern. Often a pruned scratch copy, so **not** necessarily a path `git` understands. |
| `sourceOrigin` | The repo-relative directory the provenance below describes — **this** is the path to hand `git log` / `git rev-parse`. |
| `sourceCommit` | The repo's `HEAD` when the source was captured, *before* the parse. |
| `sourceTree` | The tree object of `sourceOrigin` at `sourceCommit` — the exact identity of the parsed content. |
| `sourceDirty` | `git status --porcelain -- <sourceOrigin>` was non-empty: modified **or untracked** files under the source. Scoped — it says nothing about the rest of the repo. |
| `provenance` | How the four `source*` values were obtained: `parse-root` (the parse root is itself tracked) · `source-origin` (the parse root is a staged copy; the builder named the real tracked directory) · `none` (no git identity — the three commit/tree/dirty fields are deliberately absent, not missing). |

- **One row with a `provenance` value** → a stamp from the current pipeline;
  read it with the table above.
- **One row, `provenance` null** → a **pre-2026-09-07 stamp**. Still usable, but
  its `sourceCommit`/`sourceDirty` were derived *after* the load, repo-wide —
  see Limits before acting on either.
- **Zero rows** → either the graph predates this feature (built before M4; no
  backfill was done — see the rollout note in the graph-dba design doc) or the
  pipeline run that built it failed its own verification and never reached the
  stamping step. Treat this the same as "stale": you have no freshness signal
  at all, which is itself a reason for caution, not an error to debug.
- **One row, but `builtAt` is not a parseable timestamp** → a **hand-written
  marker**, not a pipeline stamp. `graph-dba` writes one when a graph's
  provenance is genuinely unrecoverable but the graph is still worth keeping —
  a pre-M4 graph renamed rather than rebuilt, say. The tell is `BUILT_AT`
  holding the literal string `unknown`, chosen so it fails ISO parsing
  **loudly** rather than being coalesced into a plausible date. Such a marker
  explains itself in properties the query above doesn't return, so read the
  whole node (`MATCH (b:CpgBuildInfo) RETURN b`) — expect `STATUS`,
  `MARKER_ORIGIN`, `RENAMED_FROM` and a `NOTE`. Treat it as **"stale, and not
  rebuildable on demand"**: you have provenance but no date, so checks 0 and 1
  are unavailable and **check 2 must not be run** (see Limits). Live example:
  `cpg_deprecated_salesperson`, the CPG of the retired Streamlit `salesperson/`
  app whose source now sits at `deprecated/salesperson/`.

**Judging staleness (a suggestion, not a rule).** Three escalating checks,
strongest first — the threshold is yours to set given the task at hand:

0. **Exact content identity — a yes/no, when you can get it.** Requires
   `provenance` in (`parse-root`, `source-origin`), a `sourceTree`, and
   `sourceDirty = false`. Run from the repo root:

   ```bash
   git rev-parse --short HEAD:<sourceOrigin>     # compare to sourceTree
   ```

   **Equal** → the committed source is byte-identical to what was parsed. The
   graph is current no matter how old `builtAt` is, and you are done: no commit
   counting, and no false alarm from commits that touched the path and reverted.
   **Different** → the source moved; check 2 tells you by how much.
   *(When `sourceDirty = true` the parse also swallowed uncommitted work, so
   `sourceTree` describes only the committed part — equality then means "the
   commit matches", not "the graph matches what was parsed". Fall through to
   checks 1-2 and treat the graph as matching no commit exactly.)*
1. **Raw age.** `now − parsedAt` (fall back to `builtAt` on an older marker).
   There's no universal cutoff — a week-old CPG on a slow-moving component may
   be fine; an hour-old one on a component under active refactor might already
   be behind. Weigh it against how much the task leans on structural
   correctness.
2. **Actual source movement.** If `sourceCommit` is present, run
   `git log --oneline <sourceCommit>..HEAD -- <sourceOrigin>` from the repo
   root; if it's absent, `git log --oneline --since=<parsedAt> -- <the real
   source dir>`. A **nonzero** commit count is a much stronger staleness signal
   than raw age — it means the source moved since the graph was built,
   regardless of how long ago that was. Zero commits is the converse: the graph
   may look old but the code it describes hasn't changed under it.
   (This is exactly the check `docs/plans/cpg-query-access.md` §2.3 did by
   hand — `git log --oneline --since=2026-07-18 -- falkor-chat/server` → 8
   commits — to establish an M2-era CPG was stale before its M3 rebuild.)
   **Use `sourceOrigin`, not `sourcePath`** — see Limits. **Both forms need a
   real `parsedAt`/`sourceCommit`**; skip this check entirely for a
   hand-written marker.

**Surfacing the suggestion (FR-6).** When any check makes you doubt the
graph, say so in whatever you hand back — don't silently keep using it as if
current, and don't rebuild it yourself. Naming a concrete next step is enough:
*"this CPG was built at `<builtAt>` (or: has no freshness marker) and
`<sourceOrigin>` has moved since; consider asking `graph-dba` to rebuild
`<graph>` before trusting a broad structural claim from it."* Whether to pause
and ask, or flag it and proceed, is your call — this recipe hands you the
signal, not the threshold.

## Limits

- **One marker per graph, not a build history.** An `--append` load overwrites
  the existing marker (by design — freshness tracks "when was this graph's
  content last touched," not "when was it first created"). Every field is
  rewritten on each stamp, absent ones removed, so a marker never mixes two
  builds.
- **`sourcePath` is a parse root, not a git path.** It is what Joern was
  pointed at — frequently a pruned scratch copy staged to keep `.venv` and
  friends out of the parse. Running it straight through `git log` doesn't
  error; it silently returns zero commits, which reads as false "unchanged"
  confidence rather than "no signal available". `sourceOrigin` exists precisely
  so you never have to guess: use it, and if it is null, see the next bullet.
- **`provenance: none` means the builder had no git identity for the source,
  and the pipeline stamped nothing rather than something plausible.** A staged
  copy sitting *inside* a repo (under a gitignored path) is the common case:
  that repo's `HEAD` describes a different tree, so inheriting it would hand you
  a commit that answers "unchanged" about code it never saw. You then have
  `parsedAt` age as the direct signal — **but not necessarily the only signal
  you can act on.** If you can independently confirm, from task context (not
  from any field on the marker), the real repo-relative directory the copy was
  staged from, `git log --since=<parsedAt> -- <realSourceDir>` is still valid:
  one live dispatch correctly inferred `falkor-chat/server` as the real
  counterpart of a `.git`-less `/tmp/cpg-src/falkor-chat-server` scratch build
  and ran the stronger check on it, independently confirmed correct
  (`docs/test-reports/cpg-agent-adoption2-report.md` TP-002). Valid signal, but
  only once the real path is verified. **Prevention, for whoever rebuilds:**
  `pipeline.sh --source-origin <the tracked dir it was staged from>` records it
  so the next reader doesn't have to infer anything.
- **A marker with no `provenance` field predates the 2026-09-07 fix, and both
  its git fields are weaker than they look.** They were derived after the load
  finished, from the parse root's containing repo, with no pathspec — so
  `sourceCommit` is whatever `HEAD` happened to be *then*, not what was parsed
  (on the ~3h `cpg_falkorchat` build of 2026-09-07, `HEAD` moved four times and
  the graph was stamped with a commit that was never parsed), and
  `sourceDirty = true` may come from a file nowhere near the source (the same
  build stamped `true` because of a modified file under `claude/`, while the
  parsed tree was verified byte-identical to its commit). So: check 0 is
  unavailable (no `sourceTree`), check 2's commit is approximate — a *zero*
  result from it is the untrustworthy direction — and `sourceDirty` reads only
  as "the repo had changes somewhere". `cpg_falkorchat` carries such a marker,
  with both values hand-corrected afterwards to the parsed truth.
- **A hand-written marker has no date, and check 2 fails silently against it.**
  When `builtAt` is the literal `unknown`, `git log --oneline --since=unknown --
  <path>` **does not error**: git accepts the unparseable approxidate and
  returns **zero commits with exit 0** — verified 2026-09-02, and reproduced
  against a path with heavy recent history (`--since=unknown -- skills/` → no
  commits, while `--since=2026-08-01 -- skills/` → many). So don't run check 2
  for this shape at all. What you can still do is read the marker's
  `NOTE`/`STATUS`, and treat the graph as a frozen snapshot: for a retired
  component that is the correct reading, not a gap to close, and asking
  `graph-dba` for a rebuild is usually the wrong next step (the source may no
  longer be maintained, or may not be where the graph name suggests).
- **Opt-in per build.** Any CPG built before this feature shipped, or by a
  pipeline run whose own load verification failed, has no marker — that's the
  "zero rows" case above, not an error.
