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
       b.SOURCE_DIRTY AS sourceDirty, b.SOURCE_PATH AS sourcePath,
       b.MARKER_ORIGIN AS markerOrigin
```

**Expected shape.** Zero or one row — this is a singleton marker node, not a
per-build history.

| Field | What it is |
|---|---|
| `builtAt` | When the **load finished** — "when this graph's content was last touched". |
| `parsedAt` | When the **source snapshot was taken**, i.e. what the graph actually describes. On a multi-hour build these differ by hours: anchor any `--since` on `parsedAt`, never `builtAt`. |
| `sourcePath` | The parse root handed to Joern. Often a pruned scratch copy, so **not** necessarily a path `git` understands. |
| `sourceOrigin` | The repo-relative directory the provenance below describes — **this** is the path to hand `git log` / `git rev-parse`. |
| `sourceCommit` | The repo's `HEAD` when the source was captured, *before* the parse. A **full 40-char OID**. |
| `sourceTree` | The tree object of `sourceOrigin` at `sourceCommit` — a **blob** when the source is a single file — i.e. the exact identity of that content. Also a full OID. Can be absent while `sourceCommit` is present: see check 2. |
| `sourceDirty` | `git status --porcelain -- <sourceOrigin>` was non-empty: modified **or untracked** files under the source. Scoped — it says nothing about the rest of the repo. |
| `provenance` | How the four `source*` values were obtained. Three values are **pipeline stamps**: `parse-root` (the parse root is itself tracked) · `source-origin` (the parse root is a staged copy; the builder named the real tracked directory) · `none` (no git identity — the three commit/tree/dirty fields are deliberately absent, not missing). A fourth, **`hand-backfilled`**, is *not* a pipeline stamp: a human derived the `source*` values after the build and `graph-dba` wrote them in. Spelled as its own word, not a variant of the other three, so a skimmer — or a scripted `startswith("source-origin")` — cannot quietly treat it as a pre-parse capture. See the fifth bullet below. |
| `markerOrigin` | Non-null **exactly when a human wrote this marker** rather than the pipeline; the pipeline never sets it. This — not `builtAt`, not `provenance` — is the reliable hand-authored tell, because a hand-authored marker may carry a real timestamp *and* a real provenance value. Non-null → read the whole node (`MATCH (b:CpgBuildInfo) RETURN b`) for `NOTE`/`STATUS` before acting on any other field. |

- **One row, `provenance` a pipeline value (`parse-root`, `source-origin`,
  `none`) and `markerOrigin` null** → a stamp from the current pipeline; read it
  with the table above. A non-null `provenance` does **not** on its own mean
  "pipeline" — check `markerOrigin` too, and see the fifth bullet.
- **One row, `provenance` null *and* `builtAt` a real timestamp** → a
  **pre-2026-09-07 stamp**. Still usable, but its `sourceCommit`/`sourceDirty`
  were derived *after* the load, repo-wide — see Limits before acting on either.
  (`provenance` is null on a hand-written marker too, which is why this bullet
  is gated on the timestamp; that shape is the fourth bullet below, and it
  carries no `sourceCommit`/`sourceDirty` at all.)
- **Zero rows** → either the graph predates this feature (built before M4; no
  marker was ever back-filled *into a graph that had none* — see the rollout
  note in the graph-dba design doc; that is a different act from the
  `hand-backfilled` provenance in the fifth bullet, which fills in the *fields*
  of a marker that already existed) or the
  pipeline run that built it failed its own verification and never reached the
  stamping step. Treat this the same as "stale": you have no freshness signal
  at all, which is itself a reason for caution, not an error to debug.
- **One row, but `builtAt` is not a parseable timestamp** → a **hand-written
  marker**, not a pipeline stamp. `graph-dba` writes one when a graph's
  provenance is genuinely unrecoverable but the graph is still worth keeping —
  a pre-M4 graph renamed rather than rebuilt, say. The tell is `BUILT_AT`
  holding the literal string `unknown`, chosen so it fails ISO parsing
  **loudly** rather than being coalesced into a plausible date. `markerOrigin`
  is non-null, as it is on **every** hand-authored marker — there are two such
  shapes now and the unparseable `builtAt` distinguishes only this one, so read
  `markerOrigin` whenever it exists: a marker can be hand-authored and still
  carry a real `BUILT_AT`. Such a marker explains itself in properties the query
  above doesn't return, so read the whole node
  (`MATCH (b:CpgBuildInfo) RETURN b`) — expect `STATUS`, `RENAMED_FROM` and a
  `NOTE`. Treat it as **"stale, and not
  rebuildable on demand"**: you have provenance but no date, so checks 0 and 1
  are unavailable and **check 2 must not be run** (see Limits). Live example:
  `cpg_deprecated_salesperson`, the CPG of the retired Streamlit `salesperson/`
  app whose source now sits at `deprecated/salesperson/`.
- **One row, `provenance` = `hand-backfilled`** → a **hand-backfilled marker**:
  a real `builtAt` and real `source*` values, but derived *after* the build by a
  human and written in by `graph-dba` — not captured by the pipeline before the
  parse. `markerOrigin` is non-null; `parsedAt` is **absent**, because it is
  genuinely unrecoverable after the fact and was not invented. The literal says
  a human filled the fields in, not that they filled them in well, so trust it
  only as far as its own `NOTE` earns: read the whole node and check what the
  note says each value was derived from and how it was verified. When the note
  establishes that, checks 0 and 2 are both available (see check 0's gate);
  check 1 falls back to `builtAt`. **Live example:** `cpg_falkorchat`,
  backfilled 2026-09-08, whose `NOTE` cites `cpg/.cpg-artifacts/MANIFEST.txt`
  and a `diff -rq` of the staged parse root against the committed tree.

**Judging staleness (a suggestion, not a rule).** Three escalating checks,
strongest first — the threshold is yours to set given the task at hand:

0. **Content identity — a yes/no, when you can get it.** Requires a
   `sourceTree`, `sourceDirty = false`, a `sourceOrigin` that `git` understands,
   and a `provenance` you can stand behind: `parse-root` or `source-origin`
   unconditionally, or **`hand-backfilled` once you have read that marker's
   `NOTE`** and it records where `sourceTree` came from and how it was checked.
   That last clause is **per-marker, not per-literal** — `hand-backfilled` only
   asserts that a human filled the fields in, so a backfilled marker whose note
   doesn't establish the derivation stays out of this check. `cpg_falkorchat`'s
   note does establish it, and the evidence is *stronger* than a routine
   `source-origin` stamp's: its staged parse root was verified byte-identical to
   the committed tree (`diff -rq`, exit 0, recorded in
   `cpg/.cpg-artifacts/MANIFEST.txt`), where a `source-origin` stamp only implies
   as much. Run from the repo root:

   ```bash
   git rev-parse --verify "HEAD:./<sourceOrigin>"   # compare to sourceTree
   ```

   **Use exactly that form, from the repo root.** `HEAD:./<origin>` resolves
   both a subdirectory and the repo root itself, where a bare `HEAD:.` is fatal
   (`Needed a single revision`, exit 128) — and `sourceOrigin` really is `.` for
   a whole-repo build. The `./` makes it relative to your working directory, so
   run it at the top level. `--verify` matters too: without it an unresolvable
   path makes `git rev-parse` **echo your argument back on stdout**, which in a
   script compares unequal and reads as "the source moved" rather than "you
   typed the path wrong". Both sides are full 40-char OIDs; don't `--short`
   either, since
   abbreviation width follows the repo's object count at the moment it runs, so
   the same tree can render 7 chars today and 8 next month and fail a string
   comparison on width alone.

   **Equal** → **the source is unchanged since it was captured**, so the graph
   is as current as it was at build time, however old `builtAt` is — you are
   done: no commit counting, and no false alarm from commits that touched the
   path and reverted. **Different** → the source moved; check 2 tells you by how
   much. *(This answers staleness, not build fidelity: under
   `provenance: source-origin` the parse root was a pruned copy of
   `sourceOrigin`, and whether that copy was faithful when staged is a build
   question — `joern-cpg`'s `SKILL.md` owns it — not one this marker can settle.
   A `hand-backfilled` marker admitted by the clause above inverts that: it is
   admitted precisely because its note settles the staging question with
   evidence a stamp doesn't carry.
   And when `sourceDirty = true` the parse also swallowed uncommitted work, so
   `sourceTree` describes only the committed part: skip check 0 and treat the
   graph as matching no commit exactly.)*
1. **Raw age.** `now − parsedAt` (fall back to `builtAt` on an older marker, and
   on a hand-backfilled one, which has no `parsedAt`).
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
   **A zero result means nothing in two cases**: when `sourceDirty = true`, and
   when `sourceTree` is null while `sourceCommit` is present. The second is a
   source tracked in the index but never committed at capture — `git log
   <commit>..HEAD -- <origin>` then reports 0 commits about code that has no
   commit history at all, which reads as "unchanged" when the truth is "never
   recorded". Check `sourceDirty` before believing a zero.

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
  builds — and since 2026-09-07 the pipeline reads the marker back and fails the
  run unless this build's `parsedAt` is in it, so a marker you find is one that
  actually landed. (`redis-cli` exits 0 on an error reply, so before that a
  rejected stamp was silent and an `--append` build could leave the *previous*
  marker standing over new content. A marker whose `builtAt` predates content
  you can see in the graph is that shape.) **A hand-authored marker is subject
  to the same rule**, and nothing exempts it: the next successful `--load`
  overwrites it wholesale, `NOTE` and `MARKER_ORIGIN` included. A backfilled
  marker is therefore provisional — it stands exactly until the graph is
  rebuilt, and whoever rebuilds inherits none of its reasoning.
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
  as "the repo had changes somewhere".
  **`sourceOrigin` is absent on this shape too**, so check 2 needs the same
  independently-confirmed real directory as the `provenance: none` case above —
  never `sourcePath`, which on these markers is an absolute path into a
  gitignored staged copy and returns a silent zero. Anchor on `sourceCommit`
  (`<sourceCommit>..HEAD -- <the real dir>`), not on `parsedAt`, which these
  markers also lack. **No loaded graph carries this shape any more** (checked
  2026-09-08). `cpg_falkorchat` did — keys `BUILT_AT`, `SOURCE_PATH`,
  `SOURCE_COMMIT`, `SOURCE_DIRTY` and nothing else, with both git values
  hand-corrected afterwards to the parsed truth — until it was hand-backfilled
  on 2026-09-08; it now carries ten keys and `PROVENANCE = hand-backfilled`, so
  it is the fifth bullet's shape, not this one. The shape stays documented
  because reloading a pre-fix export re-creates it; when you meet one, the real
  directory is the one thing you must establish yourself (for `cpg_falkorchat`
  it was `falkor-chat/server`).
- **A hand-written marker — the `builtAt = unknown` shape, not the
  hand-backfilled one — has no date, and check 2 fails silently against it.**
  When `builtAt` is the literal `unknown`, `git log --oneline --since=unknown --
  <path>` **does not error**: git accepts the unparseable approxidate and
  returns **zero commits with exit 0** — verified 2026-09-02, and reproduced
  against a path with heavy recent history (`--since=unknown -- skills/` → no
  commits, while `--since=2026-08-01 -- skills/` → many). So don't run check 2
  for this shape at all — **and note the `provenance: none` escape hatch does
  not apply either**, because there is no `parsedAt` or `builtAt` to anchor a
  `--since` on even once you have confirmed the real directory. This shape also
  carries **no `sourceCommit`/`sourceDirty` whatsoever** — don't go hunting for
  them; `cpg_deprecated_salesperson`'s keys are `BUILT_AT`, `SOURCE_PATH`,
  `STATUS`, `MARKER_ORIGIN`, `MARKER_WRITTEN_AT`, `RENAMED_FROM`, `NOTE`.
  What you can still do is read the marker's
  `NOTE`/`STATUS`, and treat the graph as a frozen snapshot: for a retired
  component that is the correct reading, not a gap to close, and asking
  `graph-dba` for a rebuild is usually the wrong next step (the source may no
  longer be maintained, or may not be where the graph name suggests).
- **Opt-in per build.** Any CPG built before this feature shipped, or by a
  pipeline run whose own load verification failed, has no marker — that's the
  "zero rows" case above, not an error.
