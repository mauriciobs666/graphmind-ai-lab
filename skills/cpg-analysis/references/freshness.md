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
RETURN b.BUILT_AT AS builtAt, b.SOURCE_COMMIT AS sourceCommit,
       b.SOURCE_DIRTY AS sourceDirty, b.SOURCE_PATH AS sourcePath
```

**Expected shape.** Zero or one row — this is a singleton marker node, not a
per-build history.

- **One row** → the graph was built by a pipeline run that completed the
  freshness-stamping step (`graph-dba`'s `joern-cpg` pipeline, added M4).
  `builtAt` is an ISO-8601 UTC timestamp; `sourcePath` is the parse root passed
  to the build. `sourceCommit`/`sourceDirty` are present only when the parse
  root was inside a git working tree at build time — `sourceDirty = true` means
  the tree had uncommitted changes when the CPG was built, so even a fresh
  `builtAt` is describing an unrecorded snapshot.
- **Zero rows** → either the graph predates this feature (built before M4; no
  backfill was done — see the rollout note in the graph-dba design doc) or the
  pipeline run that built it failed its own verification and never reached the
  stamping step. Treat this the same as "stale": you have no freshness signal
  at all, which is itself a reason for caution, not an error to debug.

**Judging staleness (a suggestion, not a rule).** Two escalating checks,
cheapest first — the threshold is yours to set given the task at hand:

1. **Raw age.** `now − builtAt`. There's no universal cutoff — a week-old CPG
   on a slow-moving component may be fine; an hour-old one on a component
   under active refactor might already be behind. Weigh it against how much
   the task leans on structural correctness.
2. **Actual source movement (stronger, still cheap).** If `sourceCommit` is
   present, run `git log --oneline <sourceCommit>..HEAD -- <sourcePath>` from
   the repo root; if it's absent, `git log --oneline --since=<builtAt> --
   <sourcePath>`. A **nonzero** commit count is a much stronger staleness
   signal than raw age — it means the source moved since the graph was built,
   regardless of how long ago that was. Zero commits is the converse: the
   graph may look old but the code it describes hasn't changed under it.
   (This is exactly the check `docs/plans/cpg-query-access.md` §2.3 did by
   hand — `git log --oneline --since=2026-07-18 -- falkor-chat/server` → 8
   commits — to establish an M2-era CPG was stale before its M3 rebuild. This
   recipe just hands you the two inputs to run that check yourself instead of
   guessing.)

**Surfacing the suggestion (FR-6).** When either check makes you doubt the
graph, say so in whatever you hand back — don't silently keep using it as if
current, and don't rebuild it yourself. Naming a concrete next step is enough:
*"this CPG was built at `<builtAt>` (or: has no freshness marker) and
`<sourcePath>` has moved since; consider asking `graph-dba` to rebuild
`<graph>` before trusting a broad structural claim from it."* Whether to pause
and ask, or flag it and proceed, is your call — this recipe hands you the
signal, not the threshold.

## Limits

- **One marker per graph, not a build history.** An `--append` load overwrites
  the existing marker's timestamp (by design — freshness tracks "when was this
  graph's content last touched," not "when was it first created").
- **`sourceCommit`/`sourceDirty` need the parse root to be a real git working
  tree at build time.** A build staged from a pruned scratch copy (no `.git`
  copied along — the pattern `cpg-query-access.md` §S8 used for `cpg_falkorchat`
  itself, to keep `.venv` out of the parse) has no commit reference even though
  the *real* source is tracked. `sourcePath` and raw `builtAt` age are then the
  only signal — **but not necessarily the only signal you can act on.** If you
  can independently confirm, from task context (not from any field on the
  marker itself), the real repo-relative directory a scratch copy was staged
  from, the stronger `git log --since=<builtAt> -- <realSourcePath>` check is
  still valid: one live dispatch correctly inferred `falkor-chat/server` as the
  real counterpart of a `.git`-less `/tmp/cpg-src/falkor-chat-server` scratch
  build and ran the stronger check on it, independently confirmed correct
  (`docs/test-reports/cpg-agent-adoption2-report.md` TP-002). That's valid
  signal, not an over-reach past this limitation — but only once the real path
  is verified, never on the marker's literal `sourcePath` for this build
  pattern: running that literal straight through `git log` doesn't error, it
  silently returns zero commits, which reads as false "unchanged" confidence
  rather than "no signal available."
- **Opt-in per build.** Any CPG built before this feature shipped, or by a
  pipeline run whose own load verification failed, has no marker — that's the
  "zero rows" case above, not an error.
